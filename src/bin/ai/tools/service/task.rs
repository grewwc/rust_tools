// Host-side multi-worker task tool services: `task_start` / `task_assign` /
// `task_collect` / `task_close` / `task_room_status`.
//
// The host splits a goal into per-worker assignments over the talk mailbox;
// every function below touches only the host's own files (room state, room
// transcript, worker inboxes via `push_talk`). A worker's sqlite database is
// written solely by that worker's own process. All five stay on the
// fail-closed defaults: serial execution, no retry (`task_collect` drains
// the host inbox and must never be replayed).
use serde_json::Value;

use crate::ai::{
    driver::{task, talk},
    history::SessionStore,
};

fn current_ctx() -> Result<(SessionStore, std::path::PathBuf), String> {
    let ctx = crate::ai::driver::runtime_ctx::try_current().ok_or_else(|| {
        "task tools need an active driver context (not available in one-shot/test calls)"
            .to_string()
    })?;
    let store = SessionStore::new(&ctx.app_proto.config.history_file);
    let history_file = ctx.app_proto.session_history_file.clone();
    Ok((store, history_file))
}

fn arg_str<'a>(args: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .filter_map(|k| args.get(*k))
        .filter_map(Value::as_str)
        .map(str::trim)
        .find(|s| !s.is_empty())
}

fn arg_workers(args: &Value) -> Option<Vec<String>> {
    let arr = args.get("workers")?.as_array()?;
    let out: Vec<String> = arr
        .iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if out.is_empty() { None } else { Some(out) }
}

fn arg_scope_files(args: &Value) -> Vec<String> {
    for k in ["scope_files", "scope", "files"] {
        if let Some(arr) = args.get(k).and_then(Value::as_array) {
            return arr
                .iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    Vec::new()
}

fn arg_u64(args: &Value, keys: &[&str]) -> Option<u64> {
    for k in keys {
        if let Some(v) = args.get(*k) {
            if let Some(n) = v.as_u64() {
                return Some(n);
            }
            if let Some(s) = v.as_str()
                && let Ok(n) = s.trim().parse::<u64>()
            {
                return Some(n);
            }
        }
    }
    None
}

fn format_reports(replies: &[talk::TalkMessage]) -> String {
    if replies.is_empty() {
        return "(no reports)".to_string();
    }
    replies
        .iter()
        .enumerate()
        .map(|(i, m)| format!("── [report {}/{}] from={} ──\n{}", i + 1, replies.len(), m.from_session, m.body))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// ToolSpec adapter: `ToolSpec.execute` returns model-visible text directly;
/// `ToolResult` wrapping is done by the registry path.
pub(crate) fn execute_task_start(args: &Value) -> Result<String, String> {
    let goal = arg_str(args, &["goal", "task", "objective"])
        .ok_or_else(|| "goal must not be empty (the work to split across workers)".to_string())?;
    // Exactly one of `workers` (existing sessions) or `worker_count`
    // (host-owned auto-spawn); `start_task` enforces the either/or.
    let workers = arg_workers(args).unwrap_or_default();
    let worker_count =
        arg_u64(args, &["worker_count", "count", "workers_count", "n"]).map(|n| n as usize);
    let max_rounds = arg_u64(args, &["max_rounds", "rounds"]).map(|n| n as u32);
    let room = arg_str(args, &["room"]).map(str::to_string);

    let (store, history_file) = current_ctx()?;
    let req = task::StartRequest {
        goal: goal.to_string(),
        workers,
        worker_count,
        max_rounds,
        room,
    };
    let out = task::start_task(&store, &history_file, &req, None)?;
    let mut s = format!(
        "Task room '{}' opened (round 1 of max {}, goal: {}).\n",
        out.state.room, out.state.max_rounds, out.state.goal
    );
    for (w, live, note) in &out.deliveries {
        s.push_str(&format!("- {w} [{}]: {note}\n", if *live { "live" } else { "inactive" }));
    }
    if !out.state.spawned.is_empty() {
        s.push_str(&format!(
            "Auto-created {} worker session(s) owned by this room (their databases are created by their own processes, never by the host):\n",
            out.state.spawned.len()
        ));
        for w in &out.state.spawned {
            s.push_str(&format!("- {w}\n"));
        }
    }
    s.push_str("Next: run task_assign per worker with scope_files and acceptance, then task_collect to gather round-1 reports (it wakes inactive workers headlessly).");
    Ok(s)
}

pub(crate) fn execute_task_assign(args: &Value) -> Result<String, String> {
    let room = arg_str(args, &["room"])
        .ok_or_else(|| "room must not be empty (from task_start)".to_string())?;
    let worker = arg_str(args, &["worker", "member", "session"])
        .ok_or_else(|| "worker must not be empty (one worker session id from task_start)".to_string())?;
    let acceptance = arg_str(args, &["acceptance", "bar", "criteria"])
        .ok_or_else(|| "acceptance must not be empty (the bar this worker must clear)".to_string())?;
    let scope_files = arg_scope_files(args);
    if scope_files.is_empty() {
        return Err("scope_files must name at least one file boundary".to_string());
    }
    let (store, history_file) = current_ctx()?;
    let mut state = task::load_state(&history_file, room)?;
    let note = task::assign_task(&store, &history_file, &mut state, worker, scope_files, acceptance)?;
    Ok(format!(
        "Room '{}' round {}/{}: {note}. Next: run task_collect to gather reports.",
        state.room, state.round, state.max_rounds
    ))
}

pub(crate) fn execute_task_collect(args: &Value) -> Result<String, String> {
    let room = arg_str(args, &["room"])
        .ok_or_else(|| "room must not be empty (from task_start)".to_string())?;
    let (store, history_file) = current_ctx()?;
    let mut state = task::load_state(&history_file, room)?;
    if state.status != task::TaskStatus::Open {
        return Err(format!("room '{}' is closed", state.room));
    }
    let round = state.round;
    // Wake pass first: inactive workers that have not been woken this round
    // get a headless one-shot; their reports land in the host inbox below.
    let wake = args.get("wake").and_then(Value::as_bool).unwrap_or(true);
    let mut wake_notes = Vec::new();
    if wake {
        for w in state.workers.clone() {
            let note = task::wake_worker(&store, &history_file, &mut state, &w, None);
            if !note.contains("already woken") {
                wake_notes.push(note);
            }
        }
        // Persist the woken set even if the collect below finds nothing. A
        // failure here must not fail the collect: the woken set is an
        // idempotence optimization, not truth.
        let _ = task::save_state(&history_file, &state);
    }
    let timeout = arg_u64(args, &["timeout_secs", "timeout"]).unwrap_or(task::DEFAULT_COLLECT_TIMEOUT_SECS);
    // Reports are already in the transcript: collect_task appends every
    // newly drained report before the next poll, so a killed wait loses
    // nothing already drained. STATUS lines are folded into assignments.
    let out = task::collect_task(&history_file, &mut state, timeout);
    // A wake that produced no report must not block the next collect: release
    // dead wake marks so silent inactive workers are re-woken on retry.
    task::release_dead_wakes(&history_file, &mut state, &out.missing);
    let mut s = format!(
        "Room '{}' round {}/{} collect: {} report(s), {} missing.\n",
        state.room,
        round,
        state.max_rounds,
        out.replies.len(),
        out.missing.len()
    );
    for n in &wake_notes {
        s.push_str(&format!("(wake) {n}\n"));
    }
    s.push_str(&format_reports(&out.replies));
    if out.canceled {
        s.push_str("Collect canceled by user (Ctrl+C). Reports above are already in the room transcript; re-run task_collect to keep waiting for the missing workers.\n");
    }
    if !out.missing.is_empty() {
        s.push_str(&format!(
            "\n\nMissing: {}. They may still report next round; re-run task_collect to wait longer, or proceed without them.",
            out.missing.join(", ")
        ));
    }
    // Assignment verdicts after folding STATUS lines.
    s.push_str("\nAssignments:\n");
    for a in &state.assignments {
        let scope = if a.scope_files.is_empty() {
            "unassigned".to_string()
        } else {
            a.scope_files.join(", ")
        };
        s.push_str(&format!(
            "- {} [{} | scope: {scope}]\n",
            a.worker,
            match a.status {
                task::AssignmentStatus::Pending => "pending",
                task::AssignmentStatus::Working => "working",
                task::AssignmentStatus::Done => "done",
                task::AssignmentStatus::Failed => "failed",
                task::AssignmentStatus::Blocked => "blocked (waiting on host answer)",
            }
        ));
    }
    if round >= state.max_rounds {
        s.push_str(" This was the final round: run task_close with an acceptance summary.");
    } else {
        s.push_str(" Next: review reports against acceptance, then run task_assign for rework or task_close if accepted.");
    }
    Ok(s)
}

pub(crate) fn execute_task_close(args: &Value) -> Result<String, String> {
    let room = arg_str(args, &["room"])
        .ok_or_else(|| "room must not be empty (from task_start)".to_string())?;
    let summary = arg_str(args, &["summary", "conclusion", "text"])
        .ok_or_else(|| "summary must not be empty (the acceptance conclusion)".to_string())?;
    let (_, history_file) = current_ctx()?;
    let mut state = task::load_state(&history_file, room)?;
    let path = task::close_task(&history_file, &mut state, summary)?;
    let mut s = format!(
        "Room '{}' closed. Full transcript (all rounds plus this summary) at {}.",
        state.room,
        path.display()
    );
    if !state.spawned.is_empty() {
        s.push_str(&format!(
            " Auto-created worker sessions retained for inspection: {}. Delete them explicitly if unwanted.",
            state.spawned.join(", ")
        ));
    }
    Ok(s)
}

/// Read-only progress view: round state, per-worker flags, uncollected mail,
/// and the transcript tail. Never wakes workers or drains the inbox, so it is
/// safe to call any time — including right after resuming the host session.
pub(crate) fn execute_task_room_status(args: &Value) -> Result<String, String> {
    let room = arg_str(args, &["room"])
        .ok_or_else(|| "room must not be empty (from task_start)")?;
    let (_, history_file) = current_ctx()?;
    let st = task::room_status(&history_file, room)?;
    // Formatting is shared with the `/task status` local command so both
    // surfaces read identically; see `driver::task::format_room_status`.
    Ok(task::format_room_status(&st))
}
