//! Background process dispatch: select ready processes, decode task goals,
//! build questions, clone app context, and spawn each as a tokio task.
//!
//! Extracted from `driver/mod.rs` `run_loop` to establish a clear boundary
//! between background dispatch and the foreground interaction loop
//! (review Finding #1, Phase 2).

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ai::{
    agents::AgentManifest,
    mcp::SharedMcpClient,
    skills::SkillManifest,
    tools::task_tools::{capped_subagent_manifest, with_task_entry_by_pid},
    types::App,
};

use super::agent_routing::{activate_primary_agent, ensure_runtime_manifests_loaded};
use super::process_context::{
    build_background_process_question, finalize_turn_quota, process_history_path,
    resolve_background_subagent_context,
};
use super::runtime_ctx;
use super::scheduler::{
    DispatchOutcomeTag, classify_process_outcome, decode_background_process_task_goal,
    maybe_emit_scheduler_eval, publish_background_task_failure, record_scheduler_outcome,
    resolve_background_subagent_override, select_background_batch,
    should_publish_subagent_task_result,
};
use super::turn_runtime;
use super::{BgSubagentGuard, TASK_PID, terminate_and_cleanup};

const MAX_SUBAGENT_STATUS_DETAILS: usize = 3;
const MAX_SUBAGENT_DESCRIPTION_CHARS: usize = 32;
const MAX_SUBAGENT_PROGRESS_CHARS: usize = 140;
const MIN_STATUS_FIELD_WIDTH: usize = 4;

/// Background subagent history is kept only while the process can still be
/// scheduled. Normal termination, failure, panic, or a `task_cancel` that aborts
/// the future are all cleaned up via Drop. Private memory files and the exclusive
/// cwd scratch directory share history's lifetime and are reclaimed here too,
/// preventing buildup over long runs.
struct BackgroundSubagentHistoryGuard {
    path: Option<PathBuf>,
    memory_path: Option<PathBuf>,
    cwd_dir: Option<PathBuf>,
}

impl BackgroundSubagentHistoryGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path: Some(path),
            memory_path: None,
            cwd_dir: None,
        }
    }

    /// Register the private memory files and exclusive cwd scratch directory that
    /// are spawned with the task and must be reclaimed with history's lifetime.
    /// Paths are built deterministically, so they can be computed once at
    /// construction.
    fn with_scoped_artifacts(
        mut self,
        memory_path: Option<PathBuf>,
        cwd_dir: Option<PathBuf>,
    ) -> Self {
        self.memory_path = memory_path;
        self.cwd_dir = cwd_dir;
        self
    }

    fn preserve(&mut self) {
        self.path = None;
        self.release_memory_index();
        self.memory_path = None;
        self.cwd_dir = None;
    }

    fn preserve_memory(&mut self) {
        self.release_memory_index();
        self.memory_path = None;
    }

    /// Preserving the private memory file keeps it on disk for later evidence
    /// reads, but the derived SQLite index holds a database connection in this
    /// process. Drop the cached index so a finished task does not pin a handle
    /// for the rest of the parent's lifetime; a later access reopens it lazily.
    fn release_memory_index(&self) {
        if let Some(memory_path) = self.memory_path.as_deref() {
            crate::ai::tools::storage::memory_store::remove_memory_index_entry(memory_path);
        }
    }
}

impl Drop for BackgroundSubagentHistoryGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = crate::ai::history::delete_subagent_history(&path);
        }
        if let Some(memory_path) = self.memory_path.take() {
            let _ = crate::ai::history::delete_subagent_memory(&memory_path);
        }
        if let Some(cwd_dir) = self.cwd_dir.take() {
            let _ = std::fs::remove_dir_all(&cwd_dir);
        }
    }
}

/// The only foreground-facing subagent status display. Background tasks stay
/// silent, refreshing one compact status block only at scheduler safe points,
/// so it cannot interleave with foreground output.
pub(super) struct SubagentStatusLine {
    last_line: Option<String>,
    last_serve_key: Option<Vec<serde_json::Value>>,
    is_tty: bool,
}

/// Status bar fields must stay single-line and carry no terminal control
/// characters. The task description comes from model arguments, so it cannot be
/// written to the foreground TTY directly.
fn sanitize_status_field(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());
    let mut pending_space = false;
    for ch in value.chars() {
        if ch.is_control() || ch.is_whitespace() {
            pending_space = !sanitized.is_empty();
            continue;
        }
        if pending_space {
            sanitized.push(' ');
            pending_space = false;
        }
        sanitized.push(ch);
    }
    sanitized
}

impl SubagentStatusLine {
    pub(super) fn new() -> Self {
        Self {
            last_line: None,
            last_serve_key: None,
            is_tty: std::io::stdout().is_terminal(),
        }
    }

    pub(super) fn refresh(&mut self, app: &App) {
        let store = crate::ai::history::SessionStore::new(app.config.history_file.as_path());
        let current_pid = std::process::id() as i32;
        let mut session_ids =
            crate::ai::driver::session_pid::scan_all_session_pids(store.sessions_root())
                .unwrap_or_default()
                .into_iter()
                .filter(|(_, pid, alive)| *alive && *pid == current_pid)
                .map(|(session_id, _, _)| session_id)
                .collect::<Vec<_>>();
        if !session_ids
            .iter()
            .any(|session_id| session_id == &app.session_id)
        {
            session_ids.push(app.session_id.clone());
        }
        session_ids.sort();
        session_ids.dedup();

        // Snapshot the task registry before taking the kernel lock. The registry is never held
        // while the kernel is locked, so the status line cannot form the reverse of the
        // registry -> kernel order used by the task tools.
        let snapshots_by_session = session_ids
            .into_iter()
            .map(|session_id| {
                let snapshots =
                    crate::ai::tools::task_tools::subagent_status_snapshots(&session_id);
                (session_id, snapshots)
            })
            .collect::<Vec<_>>();
        let statuses_by_session = {
            let mut os = app.os.lock().unwrap_or_else(|err| err.into_inner());
            snapshots_by_session
                .into_iter()
                .map(|(session_id, snapshots)| {
                    let statuses = crate::ai::tools::task_tools::subagent_terminal_statuses(
                        os.as_mut(),
                        snapshots,
                    );
                    (session_id, statuses)
                })
                .collect::<Vec<_>>()
        };

        let mut statuses = Vec::new();
        for (session_id, session_statuses) in statuses_by_session {
            let snapshots = session_statuses
                .iter()
                .map(|status| crate::ai::driver::session_pid::AgentSnapshot {
                    agent_name: status.agent_name.clone(),
                    description: status.description.clone(),
                    state: status.state.clone(),
                    elapsed_secs: status.elapsed_secs,
                    progress: status.progress.clone(),
                })
                .collect::<Vec<_>>();
            let _ = crate::ai::driver::session_pid::write_agent_snapshots(
                store.sessions_root(),
                &session_id,
                &snapshots,
            );
            if session_id == app.session_id {
                statuses = session_statuses;
            }
        }

        if statuses.is_empty() {
            return;
        }

        let status_line = render_subagent_status_line(&statuses);
        if crate::ai::background::serve_live_streaming() {
            self.publish_status_if_changed(&statuses, &status_line, |status_line| {
                crate::ai::background::publish_serve_frame(
                    crate::ai::background::ServeLiveKind::SubagentProgress,
                    status_line,
                );
            });
        }

        // The side-note composer (Ctrl+G) owns the footer row / DECSTBM scroll
        // region while open: the in-place redraw is deferred (see
        // `redraw_status_line`) so its bare `\r\x1b[2K` cannot displace the
        // output cursor or erase the composer draft. Serve frames above still
        // publish. This deferral applies to every scheduler-tick refresh in the
        // run loop — both the task_wait PARKED wait window and the
        // background-subagent wait window.
        self.redraw_status_line(
            &status_line,
            crate::ai::stream::side_note_input::side_note_composer_active(),
        );
    }

    fn publish_status_if_changed(
        &mut self,
        statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
        status_line: &str,
        publish: impl FnOnce(&str),
    ) {
        let key = subagent_status_semantic_key(statuses);
        if self.last_serve_key.as_ref() == Some(&key) {
            return;
        }
        publish(status_line);
        self.last_serve_key = Some(key);
    }

    /// In-place TTY redraw of the status line, gated on the side-note composer.
    /// While `composer_active` the composer owns the footer row / DECSTBM scroll
    /// region, so the write is deferred (see `status_line_redraw_deferred`):
    /// its bare `\r\x1b[2K` would displace the output cursor and erase the
    /// composer draft, and the deferred line is not recorded.
    fn redraw_status_line(&mut self, status_line: &str, composer_active: bool) {
        let line = clamp_status_block(status_line);
        if status_line_redraw_deferred(
            composer_active,
            self.last_line.as_deref(),
            &line,
            self.is_tty,
        ) {
            return;
        }
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        if crate::ai::stream::side_note_input::side_note_composer_active() {
            return;
        }
        let _ = redraw_status_block(&mut out, self.last_line.as_deref(), &line);
        let _ = out.flush();
        self.last_line = Some(line);
    }

    /// Before the foreground resumes streaming output or the input box, finalize
    /// and end the dynamic line so it no longer occupies a cursor row.
    pub(super) fn finish(&mut self) {
        self.last_serve_key = None;
        self.finish_status_line(crate::ai::stream::side_note_input::side_note_composer_active());
    }

    /// Finalize the dynamic status line, deferred while the side-note composer
    /// owns the footer row / DECSTBM scroll region: never finalize a dynamic
    /// line into the transcript while the composer is open. The status itself
    /// is not lost (serve frames and tool results carry it); only the transient
    /// TTY redraw is deferred. Keep `last_line` so a later refresh/finish after
    /// the composer closes still sees and finalizes it.
    fn finish_status_line(&mut self, composer_active: bool) {
        if composer_active {
            return;
        }
        let Some(line) = self.last_line.take() else {
            return;
        };
        if self.is_tty {
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            if crate::ai::stream::side_note_input::side_note_composer_active() {
                self.last_line = Some(line);
                return;
            }
            let _ = writeln!(out);
            let _ = out.flush();
        }
    }
}

fn subagent_status_semantic_key(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> Vec<serde_json::Value> {
    statuses
        .iter()
        .map(|status| {
            let progress = status.progress.as_deref().unwrap_or_default();
            let plan = progress.split_once(" · plan: ").map(|(_, plan)| {
                let summary = plan
                    .rsplit_once(" · elapsed ")
                    .map_or(plan, |(summary, _)| summary);
                sanitize_status_field(summary)
            });
            let (phase, latest) = match parse_activity_progress(status) {
                Some(activity) => (activity.phase, Some(activity.latest)),
                None => (
                    sanitize_status_field(
                        &progress
                            .split(" · ")
                            .filter(|field| {
                                !field.starts_with("elapsed ")
                                    && !field.starts_with("last activity ")
                                    && !field.starts_with("updates ")
                                    && *field != "checkpoint due"
                            })
                            .collect::<Vec<_>>()
                            .join(" · "),
                    ),
                    None,
                ),
            };
            serde_json::json!({
                "task": [status.description, status.agent_name, status.backend,
                    status.model, status.reasoning_effort],
                "state": status.state,
                "phase": phase,
                "latest": latest,
                "plan": plan,
                "checkpoint_due": progress.ends_with(" · checkpoint due"),
            })
        })
        .collect()
}

fn render_subagent_status_line(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> String {
    render_subagent_status_line_at_width(statuses, crate::ai::stream::raw_terminal_cols())
}

fn render_subagent_status_line_at_width(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
    width: usize,
) -> String {
    if statuses
        .iter()
        .all(|status| status.backend.is_none() && parse_activity_progress(status).is_none())
    {
        return render_legacy_subagent_status_line(statuses);
    }

    let total = statuses.len();
    let active = statuses
        .iter()
        .filter(|status| !is_terminal_subagent_state(&status.state))
        .count();
    let count = if active == total {
        format!("subagents · {total} active")
    } else {
        format!("subagents · {active}/{total} active")
    };
    let Some((status, activity)) = statuses
        .iter()
        .find_map(|status| parse_activity_progress(status).map(|activity| (status, activity)))
    else {
        let line = render_compact_subagent_status_line(statuses);
        if display_width(&line) <= width {
            return line;
        }
        let status = statuses
            .iter()
            .find(|status| status.backend.is_some() && status.reasoning_effort.is_some())
            .unwrap_or(&statuses[0]);
        return render_narrow_subagent_status_header(status, &count, width);
    };

    let backend = status
        .backend
        .as_deref()
        .map(sanitize_status_field)
        .filter(|backend| !backend.is_empty());
    let line_one_suffix = format!(
        " · {} {}",
        sanitize_status_field(&status.state),
        format_subagent_elapsed(status.elapsed_secs)
    );
    let backend_prefix = format!("{count} · [backend=");
    let backend_suffix = "] ";
    // The model is what the user cannot otherwise see, so it gets a stable
    // budget before the backend tag and agent name compete for the remainder.
    let model_suffix = status
        .model
        .as_deref()
        .map(sanitize_status_field)
        .filter(|model| !model.is_empty())
        .map(|model| {
            let budget = (width / 5).clamp(MIN_STATUS_FIELD_WIDTH, 24);
            format!(" ({})", truncate_status_field_to_width(&model, budget))
        })
        .unwrap_or_default();
    let effort_suffix = status
        .reasoning_effort
        .as_deref()
        .map(sanitize_status_field)
        .filter(|effort| !effort.is_empty())
        .map(|effort| format!(" (effort: {effort})"))
        .unwrap_or_default();
    let backend = backend.map(|backend| {
        let budget = width
            .saturating_sub(display_width(&backend_prefix))
            .saturating_sub(display_width(backend_suffix))
            .saturating_sub(display_width(&line_one_suffix))
            .saturating_sub(display_width(&model_suffix))
            .saturating_sub(display_width(&effort_suffix))
            .saturating_sub(MIN_STATUS_FIELD_WIDTH)
            .max(MIN_STATUS_FIELD_WIDTH);
        truncate_status_field_to_width(&backend, budget)
    });
    let line_one_prefix = match backend {
        Some(backend) => format!("{backend_prefix}{backend}{backend_suffix}"),
        None => format!("{count} · "),
    };
    let line_one_suffix_with_model = format!("{model_suffix}{effort_suffix}{line_one_suffix}");
    let agent_width = available_field_width(width, &line_one_prefix, &line_one_suffix_with_model);
    let agent = truncate_status_field_to_width(&status.agent_name, agent_width);
    let line_one = format!("{line_one_prefix}{agent}{line_one_suffix_with_model}");
    let line_one = if display_width(&line_one) <= width {
        line_one
    } else {
        render_narrow_subagent_status_header(status, &count, width)
    };

    let latest_prefix = "↳ latest: ";
    // `age` already carries its unit phrase ("2s ago") from the progress slot.
    let latest_suffix = if activity.age.ends_with("ago") {
        format!(" · {}", activity.age)
    } else {
        format!(" · {} ago", activity.age)
    };
    let latest_width = available_field_width(width, latest_prefix, &latest_suffix);
    let latest = truncate_status_field_to_width(&activity.latest, latest_width);
    let line_two = format!("{latest_prefix}{latest}{latest_suffix}");

    let phase_prefix = "↳ phase: ";
    let updates = activity
        .updates
        .map(|updates| format!(" · updates {updates}"))
        .unwrap_or_default();
    let task_prefix = format!("{updates} · task: ");
    let content_width = width.saturating_sub(display_width(phase_prefix));
    let task_budget = content_width
        .saturating_sub(display_width(&task_prefix))
        .saturating_div(2);
    let phase_budget = content_width
        .saturating_sub(display_width(&task_prefix))
        .saturating_sub(task_budget);
    let phase = truncate_status_field_to_width(&activity.phase, phase_budget);
    let task = truncate_status_field_to_width(&status.description, task_budget);
    let line_three = format!("{phase_prefix}{phase}{task_prefix}{task}");

    [line_one, line_two, line_three]
        .map(|line| truncate_status_field_to_width(&line, width))
        .join("\n")
}

fn render_narrow_subagent_status_header(
    status: &crate::ai::tools::task_tools::SubagentTerminalStatus,
    count: &str,
    width: usize,
) -> String {
    let effort = status
        .reasoning_effort
        .as_deref()
        .map(sanitize_status_field)
        .filter(|effort| !effort.is_empty())
        .map(|effort| {
            // Compact `e=<tier>` keeps effort ahead of lower-priority fields.
            // When even that prefix cannot fit, reserve space for the value alone.
            let label = format!("e={effort}");
            if display_width(&label) <= width {
                label
            } else {
                effort
            }
        })
        .unwrap_or_default();
    let model = status
        .model
        .as_deref()
        .map(sanitize_status_field)
        .filter(|model| !model.is_empty())
        .map(|model| truncate_status_field_to_width(&model, (width / 5).min(24)))
        .unwrap_or_default();
    let fields = [
        (count.to_string(), "", ""),
        (
            status
                .backend
                .as_deref()
                .map(sanitize_status_field)
                .unwrap_or_default(),
            "[backend=",
            "]",
        ),
        (status.agent_name.clone(), "", ""),
        (model, "(", ")"),
        (effort, "", ""),
        (
            format!(
                "{} {}",
                sanitize_status_field(&status.state),
                format_subagent_elapsed(status.elapsed_secs)
            ),
            "",
            "",
        ),
    ];
    let mut parts: [String; 6] = std::array::from_fn(|_| String::new());
    let mut remaining = width;
    let mut kept = 0;
    // Allocate effort, model, activity, and count before agent/backend details,
    // then restore the usual display order without imposing a minimum width.
    for index in [4, 3, 5, 0, 2, 1] {
        let (value, prefix, suffix) = &fields[index];
        if value.is_empty() {
            continue;
        }
        let separator_width = if kept == 0 { 0 } else { display_width(" · ") };
        if remaining <= separator_width {
            break;
        }
        let value_width = remaining
            .saturating_sub(separator_width)
            .saturating_sub(display_width(prefix) + display_width(suffix));
        let value = truncate_status_field_to_width(value, value_width);
        if !value.is_empty() {
            let part = format!("{prefix}{value}{suffix}");
            remaining -= display_width(&part) + separator_width;
            parts[index] = part;
            kept += 1;
        }
    }
    let line = parts
        .iter()
        .filter(|part| !part.is_empty())
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" · ");
    truncate_status_field_to_width(&line, width)
}

fn render_legacy_subagent_status_line(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> String {
    let total = statuses.len();
    let active = statuses
        .iter()
        .filter(|status| !is_terminal_subagent_state(&status.state))
        .count();
    let mut parts = vec![if active == total {
        format!("subagents · {total} active")
    } else {
        format!("subagents · {active}/{total} active")
    }];
    parts.extend(
        statuses
            .iter()
            .take(MAX_SUBAGENT_STATUS_DETAILS)
            .map(|status| {
                let mut detail = format!(
                    "{} {} {} ({})",
                    sanitize_status_field(&status.description),
                    format_subagent_elapsed(status.elapsed_secs),
                    sanitize_status_field(&status.state),
                    sanitize_status_field(&status.agent_name)
                );
                if let Some(progress) = status.progress.as_deref() {
                    detail.push_str(": ");
                    detail.push_str(&sanitize_status_field(progress));
                }
                detail
            }),
    );
    if total > MAX_SUBAGENT_STATUS_DETAILS {
        parts.push(format!("+{} more", total - MAX_SUBAGENT_STATUS_DETAILS));
    }
    parts.join(" · ")
}

fn render_compact_subagent_status_line(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> String {
    let total = statuses.len();
    let active = statuses
        .iter()
        .filter(|status| !is_terminal_subagent_state(&status.state))
        .count();
    let mut parts = vec![if active == total {
        format!("subagents · {total} active")
    } else {
        format!("subagents · {active}/{total} active")
    }];
    parts.extend(
        statuses
            .iter()
            .take(MAX_SUBAGENT_STATUS_DETAILS)
            .map(|status| {
                let mut detail = String::new();
                if let Some(backend) = status.backend.as_deref() {
                    detail.push_str(&format!("[backend={}] ", sanitize_status_field(backend)));
                }
                detail.push_str(&sanitize_status_field(&status.agent_name));
                if let Some(model) = status.model.as_deref() {
                    detail.push_str(&format!(" ({})", sanitize_status_field(model)));
                }
                if let Some(effort) = status.reasoning_effort.as_deref() {
                    detail.push_str(&format!(" (effort: {})", sanitize_status_field(effort)));
                }
                detail.push(' ');
                detail.push_str(&format_subagent_elapsed(status.elapsed_secs));
                detail.push(' ');
                detail.push_str(&sanitize_status_field(&status.state));
                if let Some(progress) = status.progress.as_deref() {
                    detail.push_str(" · ");
                    detail.push_str(&compact_status_field(
                        &status_progress_text(progress),
                        MAX_SUBAGENT_PROGRESS_CHARS,
                    ));
                }
                detail.push_str(" · ");
                detail.push_str(&compact_status_field(
                    &status.description,
                    MAX_SUBAGENT_DESCRIPTION_CHARS,
                ));
                detail
            }),
    );
    if total > MAX_SUBAGENT_STATUS_DETAILS {
        parts.push(format!("+{} more", total - MAX_SUBAGENT_STATUS_DETAILS));
    }
    parts.join(" · ")
}

struct ActivityProgress {
    latest: String,
    age: String,
    phase: String,
    updates: Option<u64>,
}

fn parse_activity_progress(
    status: &crate::ai::tools::task_tools::SubagentTerminalStatus,
) -> Option<ActivityProgress> {
    let progress = status.progress.as_deref()?;
    let fields = progress.split(" · ").collect::<Vec<_>>();
    let last_index = fields
        .iter()
        .position(|field| field.starts_with("last activity "))?;
    let elapsed_index = fields
        .iter()
        .position(|field| field.starts_with("elapsed "))?;
    if last_index == 0 || last_index >= elapsed_index {
        return None;
    }
    let phase = fields.get(last_index + 1)?.trim();
    if phase.starts_with("updates ") || phase.starts_with("plan: ") || phase.starts_with("elapsed ")
    {
        return None;
    }
    Some(ActivityProgress {
        latest: fields[..last_index].join(" · "),
        age: fields[last_index]
            .trim_start_matches("last activity ")
            .to_string(),
        phase: phase.to_string(),
        updates: fields[last_index + 2..elapsed_index]
            .iter()
            .find_map(|field| field.strip_prefix("updates ")?.parse().ok()),
    })
}

fn display_width(value: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(value)
}

fn available_field_width(width: usize, prefix: &str, suffix: &str) -> usize {
    width
        .saturating_sub(display_width(prefix))
        .saturating_sub(display_width(suffix))
}

fn truncate_status_field_to_width(value: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    let value = sanitize_status_field(value);
    if display_width(&value) <= max_width {
        return value;
    }
    let budget = max_width.saturating_sub(1);
    let mut width = 0;
    let mut truncated = String::new();
    for ch in value.chars() {
        let char_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + char_width > budget {
            break;
        }
        truncated.push(ch);
        width += char_width;
    }
    truncated.push('…');
    // Presentation selectors can widen a character beyond its individual width.
    while display_width(&truncated) > max_width {
        truncated.pop();
        truncated.pop();
        truncated.push('…');
    }
    truncated
}

fn clamp_status_block(status: &str) -> String {
    status
        .lines()
        .take(3)
        .map(|line| crate::ai::stream::clamp_line_to_terminal_row_with_reserve(line, 0))
        .collect::<Vec<_>>()
        .join("\n")
}

fn redraw_status_block(
    out: &mut impl Write,
    previous: Option<&str>,
    status: &str,
) -> std::io::Result<()> {
    write!(out, "\r\x1b[2K")?;
    for _ in 1..previous.map_or(1, |line| line.lines().count()) {
        write!(out, "\x1b[1A\r\x1b[2K")?;
    }
    let mut lines = status.lines();
    if let Some(first) = lines.next() {
        write!(out, "{first}")?;
    }
    for line in lines {
        write!(out, "\r\n{line}")?;
    }
    Ok(())
}

fn is_terminal_subagent_state(state: &str) -> bool {
    matches!(
        state,
        "completed" | "failed" | "cancelled" | "canceled" | "timeout" | "terminated"
    )
}

/// Keep one status detail compact enough that several active subagents still
/// fit on a terminal row. Truncation is char-boundary safe and only applies
/// when the field exceeds the budget, so short kernel details are unchanged.
fn compact_status_field(value: &str, max_chars: usize) -> String {
    let value = sanitize_status_field(value);
    if value.chars().count() <= max_chars {
        return value;
    }
    let mut compact = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    compact.push('…');
    compact
}

/// The outer status line already prints the task's elapsed time, so drop the
/// duplicate `elapsed` segment from the progress snapshot while keeping the
/// more useful last-activity age.
fn status_progress_text(progress: &str) -> String {
    let Some((head, tail)) = progress.split_once(" · elapsed ") else {
        return progress.to_string();
    };
    match tail.split_once(" · last activity ") {
        Some((_, last)) => format!("{head} · last {last}"),
        None => head.to_string(),
    }
}

/// Coexistence rule with the side-note composer (Ctrl+G): while the composer
/// owns the footer row / DECSTBM scroll region, the in-place status-line
/// redraw must be deferred — its bare `\r\x1b[2K` would displace the output
/// cursor and erase the composer draft. The deferred line is also not
/// recorded, so a later `finish` has nothing stale to finalize; the next
/// refresh after the composer closes redraws it. Unchanged lines and non-TTY
/// streams defer for the same reason (nothing new to draw). This applies to
/// every scheduler-tick refresh in the run loop, i.e. both the task_wait
/// PARKED wait window and the background-subagent wait window.
fn status_line_redraw_deferred(
    composer_active: bool,
    last_line: Option<&str>,
    line: &str,
    is_tty: bool,
) -> bool {
    composer_active || last_line == Some(line) || !is_tty
}

fn format_subagent_elapsed(elapsed_secs: u64) -> String {
    if elapsed_secs < 60 {
        return format!("{elapsed_secs}s");
    }
    if elapsed_secs < 60 * 60 {
        return format!("{}m{}s", elapsed_secs / 60, elapsed_secs % 60);
    }
    format!("{}h{}m", elapsed_secs / 3600, (elapsed_secs % 3600) / 60)
}

impl Drop for SubagentStatusLine {
    fn drop(&mut self) {
        // Every early return and error path must first end the unterminated
        // dynamic status line, so the shell prompt or later error text does not
        // append to it.
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BackgroundSubagentHistoryGuard, SubagentStatusLine, format_subagent_elapsed,
        redraw_status_block, render_subagent_status_line, render_subagent_status_line_at_width,
        sanitize_status_field, status_line_redraw_deferred,
    };
    use crate::ai::tools::task_tools::SubagentTerminalStatus;

    #[test]
    fn subagent_status_field_is_safe_single_line_text() {
        assert_eq!(
            sanitize_status_field("review\ncode\r\x1b[31m now\tplease"),
            "review code [31m now please"
        );
    }

    #[test]
    fn live_status_dedup_ignores_elapsed_and_activity_age() {
        for (initial, later) in [
            (
                "command: rg -n foo src · last activity 1s ago · running command · updates 2 · elapsed 5s",
                "command: rg -n foo src · last activity 4s ago · running command · updates 9 · elapsed 8s",
            ),
            (
                "calling model · plan: inspect task path · elapsed 5s · last activity 1s ago",
                "calling model · plan: inspect task path · elapsed 8s · last activity 4s ago",
            ),
            (
                "command: rg -n foo src · last activity 1s ago · running command · updates 2 · plan: inspect task path · elapsed 5s",
                "command: rg -n foo src · last activity 4s ago · running command · updates 9 · plan: inspect  task\tpath · elapsed 8s",
            ),
            (
                "calling model · plan: inspect task path · elapsed 5s · last activity 1s ago",
                "calling model · plan: inspect  task\tpath · elapsed 8s · last activity 4s ago",
            ),
        ] {
            let mut status = SubagentTerminalStatus {
                description: "inspect".to_string(),
                agent_name: "explorer".to_string(),
                backend: Some("traecli".to_string()),
                model: Some("gpt-5.5".to_string()),
                reasoning_effort: None,
                state: "running".to_string(),
                elapsed_secs: 5,
                progress: Some(initial.to_string()),
            };
            let mut line = SubagentStatusLine::new();
            let mut published = Vec::new();
            let first = render_subagent_status_line_at_width(&[status.clone()], 120);
            line.publish_status_if_changed(&[status.clone()], &first, |payload| {
                published.push(payload.to_string());
            });
            line.publish_status_if_changed(&[status.clone()], &first, |payload| {
                published.push(payload.to_string());
            });
            assert_eq!(published, vec![first.clone()]);

            status.elapsed_secs = 8;
            status.progress = Some(later.to_string());
            let later = render_subagent_status_line_at_width(&[status.clone()], 120);
            assert_ne!(first, later);
            line.publish_status_if_changed(&[status], &later, |payload| {
                published.push(payload.to_string());
            });
            assert_eq!(published, vec![first]);
        }
    }

    #[test]
    fn live_status_dedup_publishes_semantic_changes_and_resets_on_finish() {
        let mut status = SubagentTerminalStatus {
            description: "inspect".to_string(),
            agent_name: "explorer".to_string(),
            backend: Some("traecli".to_string()),
            model: Some("gpt-5.5".to_string()),
            reasoning_effort: None,
            state: "running".to_string(),
            elapsed_secs: 5,
            progress: None,
        };
        let mut line = SubagentStatusLine::new();
        line.is_tty = false;
        let mut published = 0;
        for progress in [
            "command: rg -n foo src · last activity 1s ago · running command · elapsed 5s",
            "command: rg -n foo src · last activity 1s ago · processing output · elapsed 5s",
            "completed: rg -n foo src · exit 0 · last activity 1s ago · processing output · elapsed 5s",
            "completed: rg -n foo src · exit 0 · last activity 1s ago · processing output · plan: inspect task path · elapsed 5s",
            "completed: rg -n foo src · exit 0 · last activity 1s ago · processing output · plan: verify task path · elapsed 5s",
            "completed: rg -n foo src · exit 0 · last activity 1s ago · processing output · plan: verify task path · elapsed 5s · checkpoint due",
        ] {
            status.progress = Some(progress.to_string());
            let rendered = render_subagent_status_line_at_width(&[status.clone()], 120);
            line.publish_status_if_changed(&[status.clone()], &rendered, |_| published += 1);
        }
        assert_eq!(published, 6);

        status.description = "inspect another task".to_string();
        let rendered = render_subagent_status_line_at_width(&[status.clone()], 120);
        line.publish_status_if_changed(&[status.clone()], &rendered, |_| published += 1);
        assert_eq!(published, 7);
        line.finish();
        assert!(line.last_serve_key.is_none());
        line.publish_status_if_changed(&[status], &rendered, |_| published += 1);
        assert_eq!(published, 8);
    }

    #[test]
    fn subagent_status_line_shows_counts_elapsed_and_truncates_details() {
        let statuses = vec![
            SubagentTerminalStatus {
                description: "review stream".to_string(),
                agent_name: "explorer".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "running".to_string(),
                elapsed_secs: 38,
                progress: Some("using read_file · src/bin/ai/driver.rs".to_string()),
            },
            SubagentTerminalStatus {
                description: "review driver".to_string(),
                agent_name: "explorer".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "waiting".to_string(),
                elapsed_secs: 61,
                progress: None,
            },
            SubagentTerminalStatus {
                description: "compile check".to_string(),
                agent_name: "executor".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "completed".to_string(),
                elapsed_secs: 3661,
                progress: None,
            },
            SubagentTerminalStatus {
                description: "extra task".to_string(),
                agent_name: "executor".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "running".to_string(),
                elapsed_secs: 5,
                progress: None,
            },
        ];

        assert_eq!(
            render_subagent_status_line(&statuses),
            "subagents · 3/4 active · review stream 38s running (explorer): using read_file · src/bin/ai/driver.rs · review driver 1m1s waiting (explorer) · compile check 1h1m completed (executor) · +1 more"
        );
        assert_eq!(format_subagent_elapsed(59), "59s");
        assert_eq!(format_subagent_elapsed(60), "1m0s");
    }

    #[test]
    fn subagent_status_line_labels_external_backends_only() {
        let kernel = SubagentTerminalStatus {
            description: "inspect".to_string(),
            agent_name: "explorer".to_string(),
            backend: None,
            model: None,
            reasoning_effort: None,
            state: "running".to_string(),
            elapsed_secs: 5,
            progress: None,
        };
        assert_eq!(
            render_subagent_status_line(&[kernel]),
            "subagents · 1 active · inspect 5s running (explorer)"
        );

        let mut external = SubagentTerminalStatus {
            description: "inspect".to_string(),
            agent_name: "explorer".to_string(),
            backend: Some("traecli".to_string()),
            model: Some("gpt-5.5".to_string()),
            reasoning_effort: None,
            state: "running".to_string(),
            elapsed_secs: 5,
            progress: Some("thinking".to_string()),
        };
        // Full external details retain their exact format when the row can fit.
        assert_eq!(
            render_subagent_status_line_at_width(&[external.clone()], 120),
            "subagents · 1 active · [backend=traecli] explorer (gpt-5.5) 5s running · thinking · inspect"
        );
        external.reasoning_effort = Some("xhigh".to_string());
        assert_eq!(
            render_subagent_status_line_at_width(&[external], 120),
            "subagents · 1 active · [backend=traecli] explorer (gpt-5.5) (effort: xhigh) 5s running · thinking · inspect"
        );
    }

    #[test]
    fn subagent_status_block_is_three_width_bounded_lines_with_live_activity() {
        let status = SubagentTerminalStatus {
            description:
                "更新任务派发说明，同时保留每一个已有的内核兼容行为".to_string(),
            agent_name: "build-and-validation-worker".to_string(),
            backend: Some("traecli-with-a-very-long-backend-name".to_string()),
            model: Some("gpt-5.5".to_string()),
            reasoning_effort: Some("xhigh".to_string()),
            state: "running".to_string(),
            elapsed_secs: 598,
            progress: Some(
                "completed: rg -n \"backend\" src/bin/ai/driver/background_dispatch.rs · last activity 2s ago · processing output · updates 60 · plan: preserve compatibility · elapsed 9m58s"
                    .to_string(),
            ),
        };
        let block = render_subagent_status_line_at_width(&[status.clone()], 120);
        let lines = block.lines().collect::<Vec<_>>();

        assert_eq!(lines.len(), 3, "{block}");
        assert!(
            lines[0].starts_with("subagents · 1 active · [backend="),
            "{block}"
        );
        assert!(lines[0].contains("running 9m58s"), "{block}");
        assert!(lines[0].contains("(gpt-5.5)"), "{block}");
        assert!(lines[0].contains("(effort: xhigh)"), "{block}");
        assert!(lines[1].starts_with("↳ latest: completed:"), "{block}");
        assert!(lines[1].ends_with(" · 2s ago"), "{block}");
        assert!(
            lines[2].starts_with("↳ phase: processing output"),
            "{block}"
        );
        assert!(lines[2].contains("updates 60"), "{block}");
        assert!(lines[2].contains("task: "), "{block}");
        assert!(block.contains('…'), "{block}");
        for line in lines {
            assert!(
                unicode_width::UnicodeWidthStr::width(line) <= 120,
                "line exceeds width: {line:?}"
            );
        }

        let narrow = render_subagent_status_line_at_width(&[status.clone()], 72);
        assert_eq!(narrow.lines().count(), 3, "{narrow}");
        assert!(
            narrow.lines().next().unwrap().contains("e=xhigh"),
            "{narrow}"
        );
        assert!(
            narrow.lines().next().unwrap().contains("(gpt-5.5)"),
            "{narrow}"
        );
        for line in narrow.lines() {
            assert!(
                unicode_width::UnicodeWidthStr::width(line) <= 72,
                "line exceeds width: {line:?}"
            );
        }

        let mut long_fields = status.clone();
        long_fields.backend = Some("backend❤\u{fe0f}".repeat(24));
        long_fields.model = Some("model❤\u{fe0f}".repeat(24));
        long_fields.reasoning_effort = Some("effort❤\u{fe0f}".repeat(24));
        long_fields.agent_name = "worker❤\u{fe0f}".repeat(24);
        long_fields.state = "running❤\u{fe0f}".repeat(24);
        long_fields.description = "任务❤\u{fe0f}".repeat(24);
        long_fields.progress = Some(format!(
            "{} · last activity {} ago · {} · updates {} · elapsed 9m58s",
            "activity❤\u{fe0f}".repeat(24),
            "age❤\u{fe0f}".repeat(24),
            "phase❤\u{fe0f}".repeat(24),
            u64::MAX,
        ));
        let mut no_activity = long_fields.clone();
        no_activity.progress = None;
        let mut kernel_activity = status.clone();
        kernel_activity.backend = None;
        kernel_activity.model = None;
        for sample in [status, long_fields, no_activity, kernel_activity] {
            for width in 0..=128 {
                let block = render_subagent_status_line_at_width(&[sample.clone()], width);
                for line in block.split('\n') {
                    assert!(
                        unicode_width::UnicodeWidthStr::width(line) <= width,
                        "line exceeds width {width}: {line:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn kernel_status_without_activity_is_byte_identical_to_legacy_format() {
        let status = SubagentTerminalStatus {
            description: "inspect scheduler".to_string(),
            agent_name: "explorer".to_string(),
            backend: None,
            model: None,
            reasoning_effort: None,
            state: "running".to_string(),
            elapsed_secs: 65,
            progress: Some(
                "calling model · plan: inspect task path · elapsed 1m5s · last activity 3s ago"
                    .to_string(),
            ),
        };

        assert_eq!(
            render_subagent_status_line_at_width(&[status], 20),
            "subagents · 1 active · inspect scheduler 1m5s running (explorer): calling model · plan: inspect task path · elapsed 1m5s · last activity 3s ago"
        );
    }

    #[test]
    fn status_block_redraw_clears_every_previous_row() {
        let mut output = Vec::new();
        redraw_status_block(
            &mut output,
            Some("old one\nold two\nold three"),
            "new one\nnew two",
        )
        .unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "\r\x1b[2K\x1b[1A\r\x1b[2K\x1b[1A\r\x1b[2Knew one\r\nnew two"
        );
    }

    #[test]
    fn status_line_redraw_defers_while_side_note_composer_is_open() {
        // The coexistence rule: an OPEN Ctrl+G side-note composer owns the
        // footer row / DECSTBM scroll region, so the in-place redraw must be
        // deferred (and the line not recorded) on every scheduler tick — both
        // in the task_wait PARKED wait window and during a turn.
        assert!(status_line_redraw_deferred(
            true,
            None,
            "subagents · 1 active",
            true
        ));
        assert!(status_line_redraw_deferred(
            true,
            Some("subagents · 1 active"),
            "subagents · 1 active",
            true
        ));
        // Same-line dedup and non-TTY streams also defer the redraw.
        assert!(status_line_redraw_deferred(
            false,
            Some("line"),
            "line",
            true
        ));
        assert!(status_line_redraw_deferred(false, None, "line", false));
        // Live path: a new line on a TTY with no composer open is drawn.
        assert!(!status_line_redraw_deferred(
            false,
            None,
            "subagents · 1 active",
            true
        ));
    }

    #[test]
    fn status_line_redraw_and_finish_skip_writes_while_composer_is_open() {
        let mut line = SubagentStatusLine::new();
        line.is_tty = true; // Simulate a live TTY; cargo test captures stdout.

        // Refresh ticks while the composer is open: no record, no write.
        line.redraw_status_line("subagents · 1 active", true);
        assert_eq!(line.last_line, None);
        line.redraw_status_line("subagents · 2 active", true);
        assert_eq!(line.last_line, None);

        // A normal tick records the drawn line.
        line.redraw_status_line("subagents · 1 active", false);
        assert_eq!(line.last_line.as_deref(), Some("subagents · 1 active"));

        // finish while the composer is open defers finalizing the line...
        line.finish_status_line(true);
        assert_eq!(line.last_line.as_deref(), Some("subagents · 1 active"));
        // ...and finalizes it once the composer closes.
        line.finish_status_line(false);
        assert_eq!(line.last_line, None);
    }

    #[test]
    fn background_subagent_history_guard_cleans_or_preserves_history() {
        let root = std::env::temp_dir().join(format!(
            "background-subagent-history-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let cleaned = root.join("session.proc-1.sqlite");
        std::fs::write(&cleaned, b"db").unwrap();
        std::fs::write(format!("{}-wal", cleaned.display()), b"wal").unwrap();
        drop(BackgroundSubagentHistoryGuard::new(cleaned.clone()));
        assert!(!cleaned.exists());
        assert!(!std::path::Path::new(&format!("{}-wal", cleaned.display())).exists());

        let preserved = root.join("session.proc-2.sqlite");
        std::fs::write(&preserved, b"db").unwrap();
        let mut guard = BackgroundSubagentHistoryGuard::new(preserved.clone());
        guard.preserve();
        drop(guard);
        assert!(preserved.exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn background_subagent_guard_cleans_scoped_memory_and_cwd_on_drop() {
        let root = std::env::temp_dir().join(format!(
            "background-subagent-scoped-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let history = root.join("session.subagent-task_x.sqlite");
        std::fs::write(&history, b"db").unwrap();
        // Private memory: the jsonl body + derived .db and its WAL sidecar.
        let memory = root.join("agent_memory.subagent-task_x.jsonl");
        std::fs::write(&memory, b"[]").unwrap();
        let memory_db = root.join("agent_memory.subagent-task_x.db");
        std::fs::write(&memory_db, b"db").unwrap();
        std::fs::write(format!("{}-wal", memory_db.display()), b"wal").unwrap();
        // Exclusive cwd scratch directory (including contents; needs recursive removal).
        let cwd = root.join("subagent-cwd-task_x");
        std::fs::create_dir_all(cwd.join("nested")).unwrap();
        std::fs::write(cwd.join("scratch.txt"), b"tmp").unwrap();
        // Subagent-scoped assets (plan_state / side_note / working-checkpoint) live in
        // `<stem>.assets` next to the child history; guard Drop must reclaim them too.
        let assets = root.join("session.subagent-task_x.assets");
        std::fs::create_dir_all(assets.join("side_notes")).unwrap();
        std::fs::write(assets.join("plan-state.json"), b"{}").unwrap();

        drop(
            BackgroundSubagentHistoryGuard::new(history.clone())
                .with_scoped_artifacts(Some(memory.clone()), Some(cwd.clone())),
        );

        assert!(!history.exists(), "history db must be cleaned");
        assert!(!memory.exists(), "memory jsonl must be cleaned");
        assert!(!memory_db.exists(), "derived memory .db must be cleaned");
        assert!(
            !std::path::Path::new(&format!("{}-wal", memory_db.display())).exists(),
            "memory .db WAL sidecar must be cleaned"
        );
        assert!(!cwd.exists(), "cwd scratch dir must be recursively cleaned");
        assert!(!assets.exists(), "subagent assets dir must be cleaned");

        // preserve() must also keep memory/cwd/assets (normal-end / resume scenarios:
        // the resumed process still reads its plan-state / side-notes / checkpoint).
        let history2 = root.join("session.subagent-task_y.sqlite");
        let memory2 = root.join("agent_memory.subagent-task_y.jsonl");
        let cwd2 = root.join("subagent-cwd-task_y");
        let assets2 = root.join("session.subagent-task_y.assets");
        std::fs::write(&history2, b"db").unwrap();
        std::fs::write(&memory2, b"[]").unwrap();
        std::fs::create_dir_all(&cwd2).unwrap();
        std::fs::create_dir_all(&assets2).unwrap();
        std::fs::write(assets2.join("plan-state.json"), b"{}").unwrap();
        let mut guard = BackgroundSubagentHistoryGuard::new(history2.clone())
            .with_scoped_artifacts(Some(memory2.clone()), Some(cwd2.clone()));
        guard.preserve();
        drop(guard);
        assert!(history2.exists() && memory2.exists() && cwd2.exists());
        assert!(assets2.exists(), "preserve() must keep subagent assets dir");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn background_subagent_memory_is_merged_before_cleanup() {
        let root = std::env::temp_dir().join(format!(
            "background-subagent-merge-order-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let history = root.join("session.subagent-task_merge.sqlite");
        let private_memory = root.join("agent_memory.subagent-task_merge.jsonl");
        let main_memory = root.join("agent_memory.jsonl");
        std::fs::write(&history, b"db").unwrap();
        std::fs::write(
            &private_memory,
            serde_json::json!({
                "id": "mem-merge",
                "timestamp": "2026-07-31T00:00:00Z",
                "category": "project_memory",
                "note": "durable subagent conclusion",
                "tags": [],
                "source": "test",
                "priority": 180,
                "owner_pid": 42,
                "owner_pgid": 42,
                "image_path": null
            })
            .to_string()
                + "\n",
        )
        .unwrap();
        let guard = BackgroundSubagentHistoryGuard::new(history.clone())
            .with_scoped_artifacts(Some(private_memory.clone()), None);

        assert_eq!(
            crate::ai::tools::service::memory::merge_subagent_whitelist(
                &private_memory,
                &main_memory
            )
            .unwrap(),
            1
        );
        drop(guard);

        assert!(!private_memory.exists());
        assert!(
            std::fs::read_to_string(&main_memory)
                .unwrap()
                .contains("durable subagent conclusion")
        );
        let _ = std::fs::remove_dir_all(root);
    }
}

/// Dispatch a batch of background processes: select ready processes, decode
/// task goals, build questions, clone app context, and spawn each as a
/// tokio task with proper scope setup.
pub(super) fn dispatch_background_batch(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
    manifests_loaded: &mut bool,
    epoch: u64,
) {
    let background_procs: Vec<aios_kernel::kernel::Process> = {
        let mut os = app.os.lock().unwrap();
        select_background_batch(os.as_mut(), epoch, app.session_id.as_str())
    };
    maybe_emit_scheduler_eval(epoch, app.session_id.as_str());

    if background_procs.is_empty() {
        return;
    }

    ensure_runtime_manifests_loaded(app, skill_manifests, agent_manifests, manifests_loaded);

    let original_history_file = app.session_history_file.clone();

    let mut task_specs: Vec<(
        u64,
        String,
        PathBuf,
        Option<String>,
        Option<String>,
        Option<u64>,
        Option<aios_kernel::primitives::FutexAddr>,
        Option<String>,
        Option<serde_json::Value>,
        Option<crate::ai::models::AutoModelFallbackSpec>,
        usize,
        bool,
        bool,
        bool,
        bool,
        Option<String>,
        Option<String>,
        Vec<String>,
        Option<String>,
        Vec<String>,
    )> = Vec::new();
    let backend_reasoning_effort =
        crate::ai::request::resolve_reasoning_effort(app, &app.current_model)
            .and_then(|effort| {
                crate::ai::tools::task_tools::normalize_traecli_reasoning_effort(effort.as_str())
            })
            .map(str::to_string);
    for proc in &background_procs {
        let pid = proc.pid;
        let task_goal = match decode_background_process_task_goal(&proc.goal) {
            Ok(goal) => goal,
            Err(err) => {
                let (result_channel_id, completion_futex_addr) =
                    with_task_entry_by_pid(pid, |entry| {
                        (
                            Some(entry.result_channel_id),
                            Some(entry.completion_futex_addr),
                        )
                    })
                    .unwrap_or((None, None));
                let mut os = app.os.lock().unwrap();
                publish_background_task_failure(
                    os.as_mut(),
                    pid,
                    result_channel_id,
                    completion_futex_addr,
                    &format!("Corrupted subagent task goal for pid {}: {}", pid, err),
                );
                continue;
            }
        };
        let mailbox_messages: Vec<String> = proc.mailbox.iter().cloned().collect();
        // When the mailbox is non-empty, build_background_process_question goes
        // through format_wakeup_prompt, producing a system scheduling notice
        // (not user input), so it should be persisted marked as an internal_note.
        let is_resume_wakeup = !mailbox_messages.is_empty();
        if !mailbox_messages.is_empty() {
            let mut os = app.os.lock().unwrap();
            if let Some(actual) = os.get_process_mut(pid) {
                actual.mailbox.clear();
            }
        }
        let proc_question = build_background_process_question(
            pid,
            &proc.goal,
            task_goal.as_ref().map(|goal| goal.prompt.as_str()),
            &mailbox_messages,
        );

        let initialize_history = {
            let mut os = app.os.lock().unwrap();
            os.set_current_pid(Some(pid));
            let mut initialize_history = false;
            if let Some(p) = os.get_process_mut(pid) {
                initialize_history = p.history_file.is_none();
                if initialize_history {
                    p.history_file = Some(process_history_path(&original_history_file, pid));
                }
                let _ = os.process_pending_signals();
            }
            initialize_history
        };

        let history_path = process_history_path(&original_history_file, pid);
        task_specs.push((
            pid,
            proc_question,
            history_path,
            task_goal.as_ref().map(|goal| goal.agent_name.clone()),
            task_goal.as_ref().map(|goal| goal.model.clone()),
            task_goal.as_ref().map(|goal| goal.result_channel_id),
            task_goal
                .as_ref()
                .map(|goal| aios_kernel::primitives::FutexAddr(goal.completion_futex_addr)),
            task_goal.as_ref().map(|goal| goal.task_id.clone()),
            task_goal
                .as_ref()
                .and_then(|goal| goal.response_schema.clone()),
            task_goal.as_ref().and_then(|goal| goal.auto_model_fallback),
            task_goal.as_ref().map(|goal| goal.spawn_depth).unwrap_or(0),
            task_goal
                .as_ref()
                .is_some_and(|goal| goal.parent_agent_team_mode),
            is_resume_wakeup,
            initialize_history,
            task_goal.as_ref().is_some_and(|goal| {
                crate::ai::tools::task_tools::SubagentBackend::from_stored(goal.backend.as_deref())
                    .is_traecli()
            }),
            task_goal
                .as_ref()
                .and_then(|goal| goal.backend_model.clone()),
            task_goal
                .as_ref()
                .and_then(|goal| goal.backend_continue_from.clone()),
            task_goal
                .as_ref()
                .map(|goal| goal.active_skill_names.clone())
                .unwrap_or_default(),
            backend_reasoning_effort.clone(),
            task_goal
                .as_ref()
                .map(|goal| goal.image_files.clone())
                .unwrap_or_default(),
        ));
    }

    for (
        pid,
        proc_question,
        history_path,
        agent_override,
        model_override,
        result_channel_id,
        completion_futex_addr,
        task_id,
        response_schema,
        auto_model_fallback,
        spawn_depth,
        parent_agent_team_mode,
        is_resume_wakeup,
        initialize_history,
        is_traecli_backend,
        backend_model,
        backend_continue_from,
        active_skill_names,
        backend_reasoning_effort,
        image_files,
    ) in task_specs
    {
        if is_traecli_backend && let Some(task_id) = task_id.as_deref() {
            crate::ai::tools::task_tools::set_task_reasoning_effort(
                task_id,
                backend_reasoning_effort.clone(),
            );
        }
        let mut task_app = app.fork_for_subagent();
        // A resumed task keeps its original images in its first message; the
        // foreground may have moved on to a different image. Only the initial
        // dispatch attaches this task's own files, so a later wakeup cannot
        // silently swap in the live turn's attachments.
        task_app.attached_image_files = if initialize_history {
            image_files.clone()
        } else {
            Vec::new()
        };
        // Background tasks need their own streaming/cancel_stream flags:
        // `App::clone` by default shares the same Arc set with the parent App, so
        // concurrent background run_turns would overwrite each other's streaming
        // and cancel each other (clear_stream_cancel resets the shared
        // cancel_stream). A background task's cancel_stream must be shared with
        // its registry entry: the sync command polls it and kills the actual OS
        // process group on timeout/cancel, and Tokio abort alone cannot interrupt
        // an in-flight sync poll. shutdown still stays shared with the parent
        // App: a session-level exit must propagate to background tasks so they
        // can wind down gracefully.
        task_app.streaming = Arc::new(AtomicBool::new(false));
        task_app.cancel_stream = task_id
            .as_deref()
            .and_then(|task_id| {
                crate::ai::tools::task_tools::with_task_entry(task_id, |entry| {
                    entry.cancel_stream.clone()
                })
            })
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        crate::ai::types::clear_stream_cancel(&task_app);
        let task_mcp = mcp_client.clone();
        let task_os = app.os.clone();
        let task_agent = match resolve_background_subagent_override(
            agent_manifests.as_slice(),
            agent_override.as_deref(),
        ) {
            Ok(agent) => agent,
            Err(err) => {
                let mut os = app.os.lock().unwrap();
                publish_background_task_failure(
                    os.as_mut(),
                    pid,
                    result_channel_id,
                    completion_futex_addr,
                    &err,
                );
                continue;
            }
        };
        if let Some(agent) = task_agent {
            let capped_agent = capped_subagent_manifest(agent);
            activate_primary_agent(&mut task_app, &capped_agent);
        }
        let next_model = model_override.unwrap_or_else(|| app.current_model.clone());

        let inherit = task_id
            .as_deref()
            .and_then(|tid| crate::ai::tools::task_tools::with_task_entry(tid, |e| e.inherit))
            .unwrap_or_default();
        let (effective_history, task_skills) = match resolve_background_subagent_context(
            history_path,
            original_history_file.as_path(),
            skill_manifests,
            task_id.as_deref(),
            inherit,
            initialize_history,
        ) {
            Ok(context) => context,
            Err(err) => {
                let mut os = app.os.lock().unwrap();
                publish_background_task_failure(
                    os.as_mut(),
                    pid,
                    result_channel_id,
                    completion_futex_addr,
                    &err,
                );
                continue;
            }
        };
        task_app.session_history_file = effective_history;
        let _ =
            crate::ai::driver::commands::session::restore_prune_marks_for_history(&mut task_app);
        let task_driver_ctx = runtime_ctx::DriverContext::from_app_snapshot(
            &task_app,
            task_mcp.clone(),
            task_skills.clone(),
            agent_manifests.clone(),
        );
        let scope_task_id = task_id.clone().unwrap_or_else(|| format!("pid-{pid}"));
        let phase_slot = crate::ai::tools::task_tools::task_progress_slot(&scope_task_id)
            .unwrap_or_else(runtime_ctx::new_subagent_progress_slot);
        let phase_slot_for_payload = phase_slot.clone();
        let phase_slot_for_backend = phase_slot.clone();
        let parent_history_for_scopes = original_history_file.clone();

        // Private memory files and the exclusive cwd scratch directory are
        // spawned with the task and reclaimed with history's lifetime (preserve
        // keeps them on normal end; Drop cleans them up on error/abort/panic).
        // Both paths must reuse the same logic used at creation time, so a second
        // divergent definition cannot cause missed or wrong deletions.
        let scoped_memory_path = (!inherit.memory).then(|| {
            runtime_ctx::make_subagent_memory_path(&parent_history_for_scopes, &scope_task_id)
        });
        let scratch_base_for_cwd = parent_history_for_scopes
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let scoped_cwd_dir = if inherit.cwd {
            None
        } else {
            runtime_ctx::make_subagent_cwd(&scratch_base_for_cwd, &scope_task_id)
        };
        // Explicit inheritance: snapshot the parent's effective cwd now and
        // carry it into the child via `SUBAGENT_CWD`, instead of relying on
        // the ambient process cwd being right at every tool call inside the
        // subagent. Resolution-only: never registered with the history guard,
        // so the guard cannot delete the parent's working directory.
        let inherited_cwd_dir = if inherit.cwd {
            runtime_ctx::effective_cwd().ok()
        } else {
            None
        };
        let history_guard =
            BackgroundSubagentHistoryGuard::new(task_app.session_history_file.clone())
                .with_scoped_artifacts(scoped_memory_path.clone(), scoped_cwd_dir.clone());
        let preserve_scoped_artifacts = Arc::new(AtomicBool::new(false));
        let preserve_scoped_artifacts_for_inner = preserve_scoped_artifacts.clone();
        let memory_merge_failed = Arc::new(AtomicBool::new(false));
        let memory_merge_failed_for_inner = memory_merge_failed.clone();
        let private_memory_for_merge = scoped_memory_path.clone();
        let persona_memory_path = app.current_persona_memory_file();
        let main_memory_for_merge = persona_memory_path.clone();

        // Slot used by the sub-agent's `finalize_turn` to publish
        // its final assistant text. Cloned into the result-channel
        // payload below so `task_wait` can surface what the
        // sub-agent actually produced (instead of just "completed
        // with empty output").
        let result_slot_for_payload: runtime_ctx::SubagentResultSlot = std::sync::Arc::new(
            tokio::sync::Mutex::new(runtime_ctx::SubagentResult::default()),
        );
        let result_slot_for_scope = result_slot_for_payload.clone();

        // TraeCLI backend inputs, cloned before `inner_fut` moves the owners.
        // Kernel path ignores these; the branch below selects the runner.
        let traecli_cwd = if inherit.cwd {
            inherited_cwd_dir
                .clone()
                .or_else(|| runtime_ctx::effective_cwd().ok())
                .unwrap_or_else(|| PathBuf::from("."))
        } else {
            scoped_cwd_dir.clone().unwrap_or_else(std::env::temp_dir)
        };
        let traecli_cancel = task_app.cancel_stream.clone();
        let traecli_workspace_allowed = inherit.cwd;
        let traecli_parent_history = original_history_file.clone();

        let inner_fut = TASK_PID.scope(Some(pid), async move {
            crate::ai::tools::registry::common::clear_tool_cancel();
            // Resolve continuity and read deferred notes only for TraeCLI.
            // Unknown references fail closed before touching the queue.
            let traecli_continuation = if is_traecli_backend {
                crate::ai::tools::task_tools::prepare_traecli_continuation(
                    backend_continue_from.as_deref(),
                    &traecli_parent_history,
                    &proc_question,
                )
            } else {
                Ok((None, String::new(), Vec::new()))
            };
            // Subagent backend branch: TraeCLI runs a headless subprocess while
            // the kernel path below keeps the exact in-process `run_turn` flow.
            // Both branches publish through the same channel/futex/validation
            // logic after this block, so `task_wait` / `task_status` stay uniform.
            let (
                result,
                captured_output,
                response_for_validation,
                progress,
                memory_merge_error,
                backend_session,
            ): (
                Result<(), String>,
                String,
                String,
                String,
                Option<String>,
                Option<String>,
            ) = if is_traecli_backend {
                match traecli_continuation {
                    Err(err) => (
                        Err(err),
                        String::new(),
                        String::new(),
                        "traecli".to_string(),
                        None,
                        None,
                    ),
                    Ok((resume_session, prompt, side_note_files)) => {
                        let spec = crate::ai::tools::task_tools::ExternalSubagentSpec {
                            prompt,
                            cwd: traecli_cwd,
                            workspace_allowed: traecli_workspace_allowed,
                            model: backend_model,
                            skills: crate::ai::skills::render_external_skill_context(
                                task_skills.as_slice(),
                                &active_skill_names,
                            ),
                            resume_session,
                            side_note_files,
                            reasoning_effort: backend_reasoning_effort,
                            image_files: image_files.clone(),
                            phase_slot: Some(phase_slot_for_backend),
                        };
                        match crate::ai::tools::task_tools::run_external_subagent(
                            crate::ai::tools::task_tools::SubagentBackend::TraeCli,
                            &spec,
                            traecli_cancel,
                        )
                        .await
                        {
                            Ok(outcome) => (
                                Ok(()),
                                outcome.output.clone(),
                                outcome.output,
                                outcome.progress,
                                None,
                                outcome.session_id,
                            ),
                            Err(err) => (
                                Err(err),
                                String::new(),
                                String::new(),
                                "traecli".to_string(),
                                None,
                                None,
                            ),
                        }
                    }
                }
            } else {
                let run = runtime_ctx::IS_RESUME_TURN.scope(
                    is_resume_wakeup,
                    turn_runtime::run_turn(
                        &mut task_app,
                        &task_mcp,
                        &task_skills,
                        usize::MAX,
                        proc_question,
                        String::new(),
                        next_model,
                        None,
                        None,
                        false,
                        false,
                    ),
                );
                let run = turn_runtime::SUBAGENT_PARENT_AGENT_TEAM_MODE
                    .scope(parent_agent_team_mode, run);
                let result = if let Some(spec) = auto_model_fallback {
                    runtime_ctx::AUTO_MODEL_FALLBACK.scope(spec, run).await
                } else {
                    run.await
                }
                .map(|_| ())
                .map_err(|e| format!("{}", e));
                let captured_result = if result_channel_id.is_some() {
                    result_slot_for_payload.lock().await.clone()
                } else {
                    runtime_ctx::SubagentResult::default()
                };
                let captured_output = captured_result.parent_payload;
                let response_for_validation = captured_result.final_assistant_text;
                let progress = runtime_ctx::subagent_progress_snapshot(&phase_slot_for_payload)
                    .unwrap_or_else(|| "working".to_string());
                let memory_merge_error = private_memory_for_merge.and_then(|private_memory| {
                    crate::ai::tools::service::memory::merge_subagent_whitelist(
                        &private_memory,
                        &main_memory_for_merge,
                    )
                    .err()
                });
                if memory_merge_error.is_some() {
                    memory_merge_failed_for_inner.store(true, Ordering::Release);
                }
                (
                    result,
                    captured_output,
                    response_for_validation,
                    progress,
                    memory_merge_error,
                    None,
                )
            };
            let response_validation_error = if result.is_ok() && memory_merge_error.is_none() {
                crate::ai::tools::task_tools::validate_subagent_response(
                    response_schema.as_ref(),
                    &response_for_validation,
                )
                .err()
            } else {
                None
            };
            let mut os = task_os.lock().unwrap();
            os.set_current_pid(Some(pid));
            let task_succeeded = result.is_ok()
                && memory_merge_error.is_none()
                && response_validation_error.is_none();
            let publish_task_result = should_publish_subagent_task_result(
                task_succeeded,
                &captured_output,
                os.get_process(pid).map(|proc| &proc.state),
            );
            if publish_task_result && let Some(result_channel_id) = result_channel_id {
                let backend_session = match (is_traecli_backend, task_id.as_deref()) {
                    (true, Some(task_id)) => crate::ai::tools::task_tools::record_traecli_session(
                        task_id,
                        backend_session,
                    ),
                    _ => backend_session,
                };
                let payload = serde_json::json!({
                    "status": if task_succeeded { "completed" } else { "failed" },
                    "output": captured_output,
                    "error": result
                        .as_ref()
                        .err()
                        .cloned()
                        .or(memory_merge_error)
                        .or(response_validation_error),
                    "progress": progress,
                    "backend_session": backend_session,
                })
                .to_string();
                let _ = os.channel_send(
                    Some(pid),
                    aios_kernel::primitives::ChannelId(result_channel_id),
                    payload,
                );
                let _ = os.channel_close(
                    Some(pid),
                    aios_kernel::primitives::ChannelId(result_channel_id),
                );
                let _ = os.channel_release_named(
                    aios_kernel::primitives::ChannelId(result_channel_id),
                    "task_result.producer",
                );
            }
            if publish_task_result && let Some(addr) = completion_futex_addr {
                let _ = os.futex_store(addr, 1);
                super::notify_scheduler();
            }
            let preserve_history = match result {
                Ok(_outcome) => {
                    let outcome = classify_process_outcome(&**os, pid);
                    record_scheduler_outcome(os.as_mut(), pid, outcome);
                    os.increment_turns_used_for(pid);
                    let (should_terminate, termination_result) =
                        finalize_turn_quota(os.as_mut(), pid);
                    if should_terminate {
                        terminate_and_cleanup(os.as_mut(), pid, termination_result, true);
                        false
                    } else if os.is_round_robin() {
                        os.set_current_pid(Some(pid));
                        os.requeue_current();
                        true
                    } else {
                        true
                    }
                }
                Err(err) => {
                    record_scheduler_outcome(os.as_mut(), pid, DispatchOutcomeTag::Failed);
                    terminate_and_cleanup(os.as_mut(), pid, format!("Failed: {}", err), true);
                    false
                }
            };
            drop(os);
            preserve_scoped_artifacts_for_inner.store(preserve_history, Ordering::Release);
        });

        type BoxedTaskFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
        let mut wrapped: BoxedTaskFuture = Box::pin(inner_fut);
        wrapped =
            Box::pin(runtime_ctx::PERSONA_MEMORY_PATH.scope(persona_memory_path.clone(), wrapped));
        wrapped = Box::pin(runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot_for_scope, wrapped));
        wrapped = Box::pin(runtime_ctx::SUBAGENT_PHASE.scope(phase_slot, wrapped));
        wrapped = Box::pin(runtime_ctx::SUBAGENT_TASK_ID.scope(scope_task_id.clone(), wrapped));
        if let Some(mem_path) = scoped_memory_path {
            // Subagents have private memory by default: after finalize,
            // whitelisted entries (is_permanent_memory) are merged back into the
            // main memory file so long-term assets can be shared across tasks,
            // while ordinary task_events stay in the private file and never
            // pollute main memory.
            wrapped = Box::pin(runtime_ctx::SUBAGENT_MEMORY_PATH.scope(mem_path, wrapped));
        }
        if let Some(scratch) = scoped_cwd_dir {
            wrapped = Box::pin(runtime_ctx::SUBAGENT_CWD.scope(scratch, wrapped));
        }
        if let Some(dir) = inherited_cwd_dir {
            wrapped = Box::pin(runtime_ctx::SUBAGENT_CWD.scope(dir, wrapped));
        }
        // Set the subagent nesting depth, used by `task_spawn` / `task` to
        // detect recursive fan-out inside subagents.
        wrapped = Box::pin(runtime_ctx::SUBAGENT_DEPTH.scope(spawn_depth, wrapped));
        // Background tasks only hand final results to task_wait/task_status
        // aggregation; subagents are forbidden from contending for the foreground
        // terminal directly, avoiding concurrent streaming output and ANSI cursor
        // control corrupting each other.
        wrapped = Box::pin(runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.scope(true, wrapped));

        // Count the in-flight background subagent: the guard is moved into the
        // task with the spawned future, and Drop auto-decrements on task end
        // (normal/error/panic), so the input box is never gated permanently.
        let inflight_guard = BgSubagentGuard::new();
        let guarded_fut = async move {
            let _inflight_guard = inflight_guard;
            let mut history_guard = history_guard;
            wrapped.await;
            if memory_merge_failed.load(Ordering::Acquire) {
                history_guard.preserve_memory();
            }
            if preserve_scoped_artifacts.load(Ordering::Acquire) {
                history_guard.preserve();
            }
        };
        let handle = tokio::spawn(runtime_ctx::DRIVER_CTX.scope(task_driver_ctx, guarded_fut));
        crate::ai::tools::task_tools::set_task_abort_handle(&scope_task_id, handle.abort_handle());
    }
}
