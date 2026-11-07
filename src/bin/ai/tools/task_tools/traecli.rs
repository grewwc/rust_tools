//! TraeCLI subagent backend: headless execution via `traecli exec --json`.
//!
//! Session defaults prefer TraeCLI when its binary is available; explicit
//! `/use` selections or a `backend` argument on a `task` / `task_spawn` item
//! override that preference. All task control block
//! semantics (task registry, result channel, futex, evidence ledger,
//! cancellation) stay owned by the existing `task_spawn` / `task_wait` layer.
//! This runner retries backend failures and resumes any captured session.
//!
//! CLI surface tracked: traecli 0.213.x. The non-interactive entry is the
//! `exec` subcommand with `--json` (JSONL events on stdout); the pre-0.2xx
//! top-level `-p --output-format stream-json` flags no longer exist.
//!
//! Isolation contract (verified against the installed binary):
//! - Runs persist sessions (deliberately not `--ephemeral`) with `-C <cwd>` as
//!   working root, so follow-up tasks can resume them via
//!   `backend_continue_from`. Sessions live in TraeCLI's own store
//!   (`~/.trae/cli/` sqlite), never in the workspace.
//! - `--permission-mode` is deliberately omitted: `exec` rejects `default`
//!   (it cannot ask the user for approvals), and `auto`/`bypass_permissions`
//!   would weaken safety. The headless default resolves approvals itself.
//! - `-s/--sandbox` is the isolation control: workspace tasks (`inherit`
//!   contains `cwd`) get `workspace-write`, pure analysis tasks stay
//!   `read-only` (observed to block both writes and network).
//! - `workspace-write` is NOT a jail: the agent can write outside `-C` and
//!   reach the network, same as a shell owned by the user. Only `read-only`
//!   is strongly isolated.
//! - `exec` never prompts the user. Blocked actions surface as tool errors
//!   the agent sees and self-reports in its final message, which flows back
//!   as the subagent result. CLI-level failures print `Error: ...` to stderr
//!   with a non-zero exit, surfaced via the stderr-tail fallback below.
//! - `exec --help` and `exec resume --help` advertise only a single stdin
//!   prompt, with no live input queue. Main-agent side-notes are retained in
//!   the parent task queue after completion/cancellation and prepended when
//!   a continuation is dispatched. This does not use the TUI's Tab queue.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use rust_tools::commonw::{FastMap, FastSet};

use super::backend::{
    BackendId, ContinueContext, ExternalSubagentOutcome, ExternalSubagentRunner,
    ExternalSubagentSpec, SubagentBackend,
};
#[cfg(test)]
use super::backend_session::SessionState as TraeCliSessionState;
use super::backend_session::{self, SessionSlot as TraeCliSessionSlot};
use super::registry::lock_unpoisoned;

/// Parsed outcome of a TraeCLI `exec --json` run.
#[derive(Debug, Clone, Default)]
pub(crate) struct TraeCliOutcome {
    pub(crate) result: String,
    pub(crate) is_error: bool,
    /// `thread_id` from the `thread.started` event (ephemeral runs emit one).
    pub(crate) session_id: Option<String>,
    /// Count of `turn.completed` events.
    pub(crate) num_turns: Option<u64>,
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
}

pub(super) type TraeCliEvidenceSlot = Arc<Mutex<Option<TraeCliOutcome>>>;

pub(crate) fn traecli_binary() -> String {
    let raw = std::env::var("TRAECLI_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "traecli".to_string());
    let path = Path::new(&raw);
    if path.components().count() > 1 && path.is_relative() {
        if let Ok(cwd) = std::env::current_dir() {
            return cwd.join(path).to_string_lossy().into_owned();
        }
    }
    raw
}

/// Cheap preflight used when a task is admitted: an absent binary is a
/// definite startup failure, so `task_spawn` can report it in its immediate
/// per-task status instead of waiting for the runner's startup watchdog.
pub(crate) fn traecli_binary_available() -> bool {
    binary_available(&traecli_binary())
}

fn binary_available(binary: &str) -> bool {
    let path = Path::new(&binary);
    if path.components().count() > 1 {
        return path.is_file();
    }
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(&binary).is_file()))
        .unwrap_or(false)
}

/// Built-in default TraeCLI model, used when neither the call, the
/// environment, nor the config names one. Verified against `traecli models`.
pub(crate) const TRAECLI_BUILTIN_DEFAULT_MODEL: &str = "gpt-5.5";

/// Resolve the TraeCLI model: explicit per-call value first, then
/// `TRAECLI_MODEL`, then `ai.traecli.default_model`, otherwise the built-in
/// default. Never fed from the parent model name: the two model registries
/// use different naming.
pub(crate) fn traecli_model(spec: &ExternalSubagentSpec) -> String {
    traecli_model_for(spec.model.as_deref())
}

/// Resolve the effective TraeCLI model from a per-call or session pick without
/// requiring a full spec. Task preparation prefers the per-call pick over the
/// session pick before passing it here; environment/config defaults follow.
pub(crate) fn traecli_model_for(explicit: Option<&str>) -> String {
    let env = std::env::var("TRAECLI_MODEL").ok();
    let config = crate::commonw::configw::get_all_config()
        .get_opt(crate::ai::config_schema::AiConfig::TRAECLI_DEFAULT_MODEL);
    resolve_traecli_model(explicit, env.as_deref(), config.as_deref())
}

/// Pure precedence chain behind [`traecli_model`], kept separate so tests do
/// not depend on process environment or user config.
pub(crate) fn resolve_traecli_model(
    explicit: Option<&str>,
    env: Option<&str>,
    config: Option<&str>,
) -> String {
    for candidate in [explicit, env, config] {
        if let Some(name) = candidate.map(str::trim).filter(|s| !s.is_empty()) {
            return name.to_string();
        }
    }
    TRAECLI_BUILTIN_DEFAULT_MODEL.to_string()
}

/// Build the headless TraeCLI command for a subagent run.
///
/// Fresh runs are `exec --json` with `--skip-git-repo-check`, an explicit
/// `-C` working root, and a sandbox derived from workspace access. Runs are
/// deliberately not `--ephemeral` so follow-up tasks can resume them via
/// `backend_continue_from`. Resume runs are `exec resume --json
/// --skip-git-repo-check --all <session>`: resume inherits the recorded
/// working root and sandbox, so `-C`/`-s` are omitted, and skills are not
/// re-prepended (the session already has them). `--permission-mode` stays
/// omitted (see module docs). Never YOLO.
pub(crate) fn build_traecli_args(spec: &ExternalSubagentSpec) -> Vec<String> {
    if let Some(session) = spec
        .resume_session
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let mut args = vec![
            "exec".to_string(),
            "resume".to_string(),
            "--json".to_string(),
            "--skip-git-repo-check".to_string(),
            "--all".to_string(),
            session.to_string(),
            "-m".to_string(),
            traecli_model(spec),
        ];
        push_effort_arg(&mut args, spec);
        return args;
    }
    let cwd = normalize_cwd(&spec.cwd);
    let mut args = vec![
        "exec".to_string(),
        "--json".to_string(),
        "--skip-git-repo-check".to_string(),
        "-C".to_string(),
        cwd.to_string_lossy().into_owned(),
        "-s".to_string(),
        if spec.workspace_allowed {
            "workspace-write"
        } else {
            "read-only"
        }
        .to_string(),
    ];
    args.push("-m".to_string());
    args.push(traecli_model(spec));
    push_effort_arg(&mut args, spec);
    push_image_args(&mut args, spec);
    args
}

/// Prompt bytes written to the child's stdin. Fresh runs prepend the inherited
/// skill overlay; resume runs send only the follow-up because the persisted
/// session already holds the original prompt and skills.
fn prompt_payload(spec: &ExternalSubagentSpec) -> String {
    let resuming = spec
        .resume_session
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    let mut payload = if resuming {
        append_original_image_paths(spec.prompt.clone(), &spec.image_files)
    } else {
        full_prompt(spec)
    };
    if spec.phase_slot.is_some() {
        payload.push_str(
            "\n\nDuring long-running work, publish saved findings in a separate progress message \
             using this format:\n<context_checkpoint>\nsummary: Short finding summary\n\
             Checkpoint body with the evidence gathered so far.\n</context_checkpoint>\n\
             Keep each checkpoint concise. Send the final answer separately after finishing.",
        );
    }
    payload
}

/// Normalize effort for both TraeCLI argv and display. TraeCLI has no "off"
/// tier, so `none`/`minimal` clamp to `low`; other supported tiers pass through.
/// Unknown values are omitted so the backend default wins.
pub(crate) fn normalize_traecli_reasoning_effort(effort: &str) -> Option<&str> {
    match effort.trim() {
        "none" | "minimal" => Some("low"),
        effort @ ("low" | "medium" | "high" | "xhigh" | "max") => Some(effort),
        _ => None,
    }
}

fn push_effort_arg(args: &mut Vec<String>, spec: &ExternalSubagentSpec) {
    let Some(mapped) = spec
        .reasoning_effort
        .as_deref()
        .and_then(normalize_traecli_reasoning_effort)
    else {
        return;
    };
    args.push("-c".to_string());
    args.push(format!("model_reasoning_effort={mapped}"));
}

/// Attach the parent turn's original images to the child's first request.
/// Codex's `exec` accepts one `-i/--image <FILE>` per image; resume runs
/// already carry them in the persisted session, so only the path hint is
/// re-sent there.
fn push_image_args(args: &mut Vec<String>, spec: &ExternalSubagentSpec) {
    for path in &spec.image_files {
        args.push("-i".to_string());
        args.push(path.clone());
    }
}

/// Later turns can compact the original request down to a summary. Keep the
/// absolute paths in the follow-up prompt so the model can reopen the original
/// whenever the task still depends on its visual detail.
fn append_original_image_paths(mut prompt: String, image_files: &[String]) -> String {
    if image_files.is_empty() {
        return prompt;
    }
    prompt.push_str("\n\n## Original images\n");
    for path in image_files {
        prompt.push_str("- ");
        prompt.push_str(path);
        prompt.push('\n');
    }
    prompt.push_str("Use your image viewer on these paths when you need the original again.");
    prompt
}

/// Final prompt text: parent skill context first (mirroring the kernel path,
/// where skills are system-prompt overlays above the task), then the task.
fn full_prompt(spec: &ExternalSubagentSpec) -> String {
    match spec
        .skills
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(skills) => format!("{skills}\n\n{}", spec.prompt),
        None => spec.prompt.clone(),
    }
}

/// Scan full `exec --json` stdout: the last non-empty `agent_message` wins,
/// `thread_id` comes from `thread.started`, turns/usage accumulate from
/// `turn.completed`. `reasoning` and `command_execution` items are ignored.
/// Returns `None` when no turn, message, or error event exists.
pub(crate) fn parse_stream_json_output(output: &str) -> Option<TraeCliOutcome> {
    let mut outcome = TraeCliOutcome::default();
    let mut turns: u64 = 0;
    let mut input_tokens: u64 = 0;
    let mut output_tokens: u64 = 0;
    let mut saw_usage = false;
    let mut message: Option<String> = None;
    let mut error_text: Option<String> = None;

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.starts_with('{') {
            continue;
        }
        let value: Value = match serde_json::from_str(trimmed) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let event_type = value.get("type").and_then(Value::as_str).unwrap_or("");
        match event_type {
            "thread.started" => {
                if outcome.session_id.is_none() {
                    outcome.session_id = value
                        .get("thread_id")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
            }
            "item.completed" => {
                let item = value.get("item");
                let item_type = item.and_then(|i| i.get("type")).and_then(Value::as_str);
                if item_type == Some("agent_message") {
                    if let Some(text) = item.and_then(|i| i.get("text")).and_then(Value::as_str) {
                        if !text.trim().is_empty() {
                            let text = traecli_message_text(text, |_, _| {});
                            if !text.trim().is_empty() {
                                message = Some(text);
                            }
                        }
                    }
                }
            }
            "turn.completed" => {
                turns += 1;
                if let Some(usage) = value.get("usage") {
                    if let Some(n) = usage.get("input_tokens").and_then(Value::as_u64) {
                        input_tokens += n;
                        saw_usage = true;
                    }
                    if let Some(n) = usage.get("output_tokens").and_then(Value::as_u64) {
                        output_tokens += n;
                        saw_usage = true;
                    }
                }
            }
            "error" => {
                if error_text.is_none() {
                    error_text = value
                        .get("message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or_else(|| {
                            value
                                .get("error")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                        .or_else(|| Some(trimmed.to_string()));
                }
            }
            _ => {}
        }
    }

    if turns == 0 && message.is_none() && error_text.is_none() {
        return None;
    }
    if turns > 0 {
        outcome.num_turns = Some(turns);
    }
    if saw_usage {
        outcome.input_tokens = Some(input_tokens);
        outcome.output_tokens = Some(output_tokens);
    }
    match message {
        Some(text) => outcome.result = text,
        None => {
            outcome.result = error_text.unwrap_or_default();
            outcome.is_error = true;
        }
    }
    Some(outcome)
}

pub(crate) async fn run_traecli_subagent(
    spec: &ExternalSubagentSpec,
    cancel: Arc<AtomicBool>,
) -> Result<TraeCliOutcome, String> {
    run_traecli_subagent_with(spec, cancel, &traecli_binary(), TRAECLI_STARTUP_TIMEOUT).await
}

async fn run_traecli_subagent_with(
    spec: &ExternalSubagentSpec,
    cancel: Arc<AtomicBool>,
    binary: &str,
    startup_timeout: Duration,
) -> Result<TraeCliOutcome, String> {
    run_traecli_subagent_with_session(spec, cancel, binary, startup_timeout, None, None).await
}

/// The registered adapter supplies a session slot explicitly so failures retain
/// streamed discovery even when no task-local task ID is installed.
pub(super) async fn run_traecli_subagent_with_events(
    spec: &ExternalSubagentSpec,
    cancel: Arc<AtomicBool>,
    session_slot: TraeCliSessionSlot,
    evidence_slot: TraeCliEvidenceSlot,
) -> Result<TraeCliOutcome, String> {
    run_traecli_subagent_with_session(
        spec,
        cancel,
        &traecli_binary(),
        TRAECLI_STARTUP_TIMEOUT,
        Some(session_slot),
        Some(evidence_slot),
    )
    .await
}

async fn run_traecli_subagent_with_session(
    spec: &ExternalSubagentSpec,
    cancel: Arc<AtomicBool>,
    binary: &str,
    startup_timeout: Duration,
    supplied_session_slot: Option<TraeCliSessionSlot>,
    evidence_slot: Option<TraeCliEvidenceSlot>,
) -> Result<TraeCliOutcome, String> {
    if spec.prompt.trim().is_empty() {
        return Err("traecli subagent prompt is empty".to_string());
    }
    let original_resume_session = spec.resume_session.clone();
    let mut attempt_spec = spec.clone();
    let session_slot = match supplied_session_slot {
        Some(slot) => {
            if let Some(task_id) = crate::ai::driver::runtime_ctx::try_subagent_task_id() {
                backend_session::track_session_slot(BackendId::TRAECLI, &task_id, slot.clone());
            }
            slot
        }
        None => crate::ai::driver::runtime_ctx::try_subagent_task_id()
            .map(|task_id| track_traecli_session(&task_id))
            .unwrap_or_else(|| Arc::new(Mutex::new(None))),
    };
    let total_attempts = TRAECLI_MAX_RETRIES + 1;
    for attempt in 1..=total_attempts {
        if cancel.load(Ordering::Acquire) {
            return Err("traecli subagent cancelled".to_string());
        }
        // Each process stays on the blocking pool; retry waits use async sleep.
        // Retry routing prefers this attempt's session, then the latest session
        // captured by any attempt for the task, and finally the caller's
        // original session.
        let attempt_session_slot = Arc::new(Mutex::new(None));
        let owned = attempt_spec.clone();
        let worker_cancel = cancel.clone();
        let worker_slot = session_slot.clone();
        let worker_attempt_slot = attempt_session_slot.clone();
        let worker_evidence_slot = evidence_slot.clone();
        let worker_task_id = crate::ai::driver::runtime_ctx::try_subagent_task_id();
        let worker_context = crate::ai::driver::runtime_ctx::try_current();
        let run_id = worker_task_id
            .clone()
            .unwrap_or_else(|| format!("run-{}", uuid::Uuid::new_v4().simple()));
        let run_guard = backend_session::BackendRunGuard::start(
            BackendId::TRAECLI,
            &run_id,
            worker_slot.clone(),
            owned.resume_session.as_deref(),
        )?;
        let binary = binary.to_string();
        let result = tokio::task::spawn_blocking(move || {
            let _run_guard = run_guard;
            // Blocking workers do not inherit Tokio task locals. Carry the task and session
            // scopes so checkpoint events use the same persistence/notification path.
            let run = || {
                run_traecli_blocking_with_evidence(
                    &owned,
                    worker_cancel,
                    &binary,
                    startup_timeout,
                    vec![worker_slot, worker_attempt_slot],
                    worker_evidence_slot,
                )
            };
            let run = || match worker_context {
                Some(context) => {
                    crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(context, run)
                }
                None => run(),
            };
            match worker_task_id {
                Some(task_id) => {
                    crate::ai::driver::runtime_ctx::SUBAGENT_TASK_ID.sync_scope(task_id, run)
                }
                None => run(),
            }
        })
        .await
        .map_err(|err| format!("traecli join failed: {err}"))?;
        let error = match result {
            Ok(outcome) => return Ok(outcome),
            Err(error) => error,
        };
        if cancel.load(Ordering::Acquire) || error == "traecli subagent cancelled" {
            return Err("traecli subagent cancelled".to_string());
        }
        // Missing final output does not prove the run failed: tools may already
        // have mutated state. Return the incomplete result without replaying it.
        if error.starts_with("traecli produced no result event")
            || error == "traecli produced an empty result"
        {
            return Err(error);
        }
        if is_permanent_traecli_error(&error) {
            return Err(error);
        }
        if attempt == total_attempts {
            return Err(format!(
                "traecli subagent failed after {total_attempts} attempts: {error}"
            ));
        }
        let attempt_session = lock_unpoisoned(&attempt_session_slot).clone();
        let task_session = lock_unpoisoned(&session_slot).clone();
        attempt_spec.resume_session = attempt_session
            .or(task_session)
            .or_else(|| original_resume_session.clone());
        let delay = traecli_retry_backoff(attempt);
        let mode = if attempt_spec
            .resume_session
            .as_deref()
            .is_some_and(|session| !session.trim().is_empty())
        {
            "resuming session"
        } else {
            "fresh start"
        };
        publish_traecli_activity(
            spec,
            "waiting to retry traecli",
            &format!(
                "retrying traecli (attempt {}/{total_attempts}, {mode}); waiting {}s",
                attempt + 1,
                delay.as_secs()
            ),
        );
        wait_for_traecli_retry(delay, &cancel).await?;
    }
    unreachable!("the last failed attempt returns its error")
}

/// A backend that never emits its first JSON event has not started a turn.
/// This is deliberately much shorter than the parent wall-clock budget: the
/// parent should not wait ten minutes to learn that the CLI is wedged.
const TRAECLI_STARTUP_TIMEOUT: Duration = Duration::from_secs(45);
const TRAECLI_POLL_INTERVAL: Duration = Duration::from_millis(50);
const TRAECLI_MAX_RETRIES: usize = 2;
const TRAECLI_RETRY_BASE_DELAY: Duration = Duration::from_secs(1);
const TRAECLI_RETRY_CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Failures that cannot be fixed by replaying the same request.
///
/// The classifier stays conservative. Spawn errors mean the executable,
/// permissions, or working directory is wrong on the first attempt, and the
/// auth/config/argument markers below are CLI-level diagnostics that would
/// repeat identically. Startup timeouts, provider disconnects, rate limits,
/// and unclassified non-zero exits remain retryable; unknown errors are also
/// retried because a transient cause is more likely than a deterministic one.
fn is_permanent_traecli_error(error: &str) -> bool {
    if error.starts_with("failed to spawn traecli:") {
        return true;
    }
    let lower = error.to_ascii_lowercase();
    const PERMANENT_MARKERS: &[&str] = &[
        "unauthorized",
        "authentication failed",
        "authentication error",
        "not logged in",
        "login required",
        "please log in",
        "please login",
        "invalid api key",
        "missing api key",
        "invalid configuration",
        "configuration error",
        "config error",
        "unexpected argument",
        "invalid argument",
        "invalid value",
    ];
    PERMANENT_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// The two retries wait 1s then 2s; retry numbers are one-based.
fn traecli_retry_backoff(retry: usize) -> Duration {
    TRAECLI_RETRY_BASE_DELAY * 2_u32.pow((retry - 1) as u32)
}

async fn wait_for_traecli_retry(delay: Duration, cancel: &AtomicBool) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + delay;
    loop {
        if cancel.load(Ordering::Acquire) {
            return Err("traecli subagent cancelled".to_string());
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Ok(());
        }
        tokio::time::sleep(remaining.min(TRAECLI_RETRY_CANCEL_POLL_INTERVAL)).await;
    }
}

pub(super) fn run_traecli_blocking_with(
    spec: &ExternalSubagentSpec,
    cancel: Arc<AtomicBool>,
    binary: &str,
    startup_timeout: Duration,
    session_slots: Vec<TraeCliSessionSlot>,
) -> Result<TraeCliOutcome, String> {
    run_traecli_blocking_with_evidence(spec, cancel, binary, startup_timeout, session_slots, None)
}

/// Ensure early errors or panics cannot leave the owned CLI running after its run guard ends.
struct TraeCliChildGuard(std::process::Child);

impl std::ops::Deref for TraeCliChildGuard {
    type Target = std::process::Child;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for TraeCliChildGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for TraeCliChildGuard {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

fn run_traecli_blocking_with_evidence(
    spec: &ExternalSubagentSpec,
    cancel: Arc<AtomicBool>,
    binary: &str,
    startup_timeout: Duration,
    session_slots: Vec<TraeCliSessionSlot>,
    evidence_slot: Option<TraeCliEvidenceSlot>,
) -> Result<TraeCliOutcome, String> {
    let with_sandbox_hint = |mut error: String, stderr: &str| {
        if let Some(hint) = read_only_sandbox_hint(spec, true, &[error.as_str(), stderr]) {
            error.push_str(hint);
        }
        error
    };
    let cwd = normalize_cwd(&spec.cwd);
    publish_traecli_activity(spec, "starting traecli", "launching traecli");
    let mut child = TraeCliChildGuard(
        std::process::Command::new(binary)
            .args(build_traecli_args(spec))
            .current_dir(&cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| format!("failed to spawn traecli: {err}"))?,
    );

    backend_session::DeferredNotes::consume_after_spawn(&spec.side_note_files);

    publish_traecli_activity(
        spec,
        "waiting for traecli startup",
        "waiting for first backend event",
    );

    // Stream stdout line-by-line. The blocking runner needs to observe the
    // startup handshake and live progress while the child is still running;
    // reading to EOF would hide both until the final result arrives.
    let (stdout_tx, stdout_rx) = mpsc::channel::<String>();
    let stdout_handle = child.stdout.take().map(|stdout| {
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                // Capture before forwarding: the async waiter can be aborted
                // while the blocking runner is between cancellation polls.
                for slot in &session_slots {
                    capture_traecli_session(slot, &line);
                }
                if stdout_tx.send(line).is_err() {
                    break;
                }
            }
        })
    });
    let stderr_shared = Arc::new(Mutex::new(String::new()));
    let stderr_sink = stderr_shared.clone();
    let stderr_handle = child.stderr.take().map(|mut stderr| {
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            if let Ok(mut sink) = stderr_sink.lock() {
                *sink = text;
            }
        })
    });

    // The prompt travels over stdin: scoped instructions can exceed the OS
    // argv limit (E2BIG). Write it on its own thread: a child that is slow to
    // read stdin must not block the startup watchdog or cancellation checks.
    let stdin_handle = child.stdin.take().map(|mut stdin| {
        let payload = prompt_payload(spec);
        std::thread::spawn(move || {
            let _ = stdin.write_all(payload.as_bytes());
            // Dropping stdin closes the pipe so traecli starts the turn.
        })
    });

    let started_at = Instant::now();
    let mut stdout_text = String::new();
    let mut active_items = TraeCliActiveItems::default();
    let mut saw_event = false;
    let status = loop {
        while let Ok(line) = stdout_rx.try_recv() {
            if line.trim().is_empty() {
                continue;
            }
            saw_event = true;
            record_traecli_event(spec, &line, &mut stdout_text, &mut active_items);
        }
        if cancel.load(Ordering::Acquire) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("traecli subagent cancelled".to_string());
        }
        match child
            .try_wait()
            .map_err(|err| format!("failed to wait for traecli: {err}"))?
        {
            Some(exit) => break exit,
            None if !saw_event && started_at.elapsed() >= startup_timeout => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(handle) = stdout_handle {
                    if handle.is_finished() {
                        let _ = handle.join();
                    }
                }
                let stderr_text = take_stderr_snapshot(stderr_handle, &stderr_shared);
                let detail = truncate_head(stderr_text.trim(), 1_000);
                let error = if detail.is_empty() {
                    format!(
                        "traecli did not emit a startup event within {}s",
                        startup_timeout.as_secs()
                    )
                } else {
                    format!(
                        "traecli did not emit a startup event within {}s: {detail}",
                        startup_timeout.as_secs()
                    )
                };
                return Err(with_sandbox_hint(error, &stderr_text));
            }
            None => std::thread::sleep(TRAECLI_POLL_INTERVAL),
        }
    };

    drain_stdout_events(
        stdout_handle,
        &stdout_rx,
        spec,
        &mut stdout_text,
        &mut active_items,
    );
    // A descendant can inherit the child's stdio and keep a pipe open after
    // the direct child exits. Joining a reader/writer thread blocked on that
    // pipe would hang the task, so detached threads are dropped instead.
    drop(stdin_handle);
    let stderr_text = take_stderr_snapshot(stderr_handle, &stderr_shared);
    if let Some(outcome) = parse_stream_json_output(&stdout_text) {
        if let Some(slot) = evidence_slot {
            *lock_unpoisoned(&slot) = Some(outcome.clone());
        }
        if outcome.result.trim().is_empty() {
            return Err(with_sandbox_hint(
                "traecli produced an empty result".to_string(),
                &stderr_text,
            ));
        }
        if outcome.is_error {
            return Err(with_sandbox_hint(
                format!("traecli subagent failed: {}", outcome.result.trim()),
                &stderr_text,
            ));
        }
        return Ok(outcome);
    }

    // No parseable events: surface whatever the CLI printed. Errors normally
    // land on stderr (`Error: ...`, non-zero exit), but keep the first
    // non-JSON stdout line as a hint in case the shape changes.
    let stderr_tail = truncate_head(stderr_text.trim(), 2000);
    let hint = stdout_hint(&stdout_text);
    let result = if status.success() {
        match hint {
            Some(hint) => Err(format!("traecli produced no result event: {hint}")),
            None => Err("traecli produced no result event".to_string()),
        }
    } else if stderr_tail.is_empty() {
        match hint {
            Some(hint) => Err(format!("traecli exited with status {status}: {hint}")),
            None => Err(format!(
                "traecli exited with status {status} and no result event"
            )),
        }
    } else {
        Err(format!(
            "traecli exited with status {status}: {stderr_tail}"
        ))
    };
    result.map_err(|error| with_sandbox_hint(error, &stderr_text))
}

const TRAECLI_READ_ONLY_SANDBOX_HINT: &str = "\n\nSandbox hint: This task ran read-only (no network; no writes outside temp). \
     Respawn with inherit: \"cwd,skills\" to grant workspace-write.";

const TRAECLI_POSSIBLE_READ_ONLY_SANDBOX_HINT: &str = "\n\nSandbox hint: If this failure is due to the read-only sandbox \
     (no network; no writes outside temp), respawn with inherit: \"cwd,skills\" to grant workspace-write.";

/// Select advice according to the strength of the failure evidence.
/// Permission denials must identify an operation, rather than a word in a path.
/// Generic connectivity failures cannot establish that the sandbox caused them.
/// Resume inherits its recorded sandbox, so the spec cannot prove its mode.
fn read_only_sandbox_hint(
    spec: &ExternalSubagentSpec,
    failed: bool,
    evidence: &[&str],
) -> Option<&'static str> {
    if !failed
        || spec.workspace_allowed
        || spec
            .resume_session
            .as_deref()
            .map(str::trim)
            .is_some_and(|session| !session.is_empty())
    {
        return None;
    }

    static DENIED_OPERATION: LazyLock<regex::Regex> = LazyLock::new(|| {
        let operation = "(?:socket|connect|bind|send|sendto|write|writing|create|creating|mkdir|touch|save|saving|append|open|openat|rename|renameat|unlink|unlinkat)";
        regex::Regex::new(&format!(
            r#"(?x)
            (?:^|:\s*)\s*
            (?:
                (?:(?:error|failed|unable|cannot|could\s+not)(?:\s+to)?\s+)?
                {operation}
                (?:
                    \s*\([^\r\n)]*\)\s*=\s*-1\s+
                    | \s*\([^\r\n)]*\)\s+(?:creation\s+)?failed\s*:\s*
                    | \s*:\s*
                    | \s+(?:creation\s+)?failed\s*:\s*
                    | \s+(?:"[^"\r\n]+"|'[^'\r\n]+'|[^\s:]+)\s*:\s*
                )
                (?:eperm\b|eacces\b|operation\s+not\s+permitted\b|permission\s+denied\b)
                |
                (?:eperm|eacces)\s*:\s*(?:operation\s+not\s+permitted|permission\s+denied)\s*,\s*
                {operation}\s+(?:"[^"\r\n]+"|'[^'\r\n]+'|[^\s:]+)
            )"#
        ))
        .expect("valid sandbox operation diagnostic regex")
    });

    let mut connectivity_failure = false;
    for text in evidence {
        let text = text.to_ascii_lowercase();
        for line in text.lines() {
            if line.contains("read-only file system")
                || line.contains("read-only filesystem")
                || DENIED_OPERATION.is_match(line)
            {
                return Some(TRAECLI_READ_ONLY_SANDBOX_HINT);
            }
            connectivity_failure |= [
                "network is unreachable",
                "could not resolve host",
                "could not resolve proxy",
                "temporary failure in name resolution",
                "name or service not known",
                "dns lookup failed",
                "dns resolution failed",
                "connection refused",
                "failed to connect",
            ]
            .iter()
            .any(|signal| line.contains(*signal));
        }
    }
    connectivity_failure.then_some(TRAECLI_POSSIBLE_READ_ONLY_SANDBOX_HINT)
}

fn publish_traecli_activity(spec: &ExternalSubagentSpec, phase: &str, activity: &str) {
    if let Some(slot) = spec.phase_slot.as_ref() {
        crate::ai::driver::runtime_ctx::publish_subagent_activity_to_slot(slot, phase, activity);
    }
}

#[derive(Default)]
struct TraeCliActiveItems {
    identified: FastMap<String, &'static str>,
    unidentified: FastMap<&'static str, usize>,
    completed_ids: FastSet<String>,
}

impl TraeCliActiveItems {
    fn start(&mut self, value: &Value) {
        let phase = traecli_item_phase(value);
        if let Some(id) = traecli_item_id(value) {
            self.identified.insert(id.to_string(), phase);
        } else {
            // Some events omit ids, so count starts per phase to preserve
            // overlapping work without guessing which item completed.
            *self.unidentified.entry(phase).or_default() += 1;
        }
    }

    fn complete(&mut self, value: &Value) {
        if let Some(id) = traecli_item_id(value) {
            if !self.completed_ids.insert(id.to_string()) || self.identified.remove(id).is_some() {
                return;
            }
        }
        let phase = traecli_item_phase(value);
        if let Some(count) = self.unidentified.get_mut(phase)
            && *count > 0
        {
            *count -= 1;
        } else if let Some(id) = self
            .identified
            .iter()
            .find_map(|(id, active)| (*active == phase).then(|| id.clone()))
        {
            // Either event can omit the id. Consume one start in the same phase
            // while keeping unrelated and overlapping work active.
            self.identified.remove(&id);
        }
    }

    fn phase(&self) -> Option<&'static str> {
        // Executing work takes priority over drafting or reasoning so an
        // unrelated item cannot hide a command or tool that is still active.
        [
            "running command",
            "running tool",
            "writing response",
            "thinking",
            "working",
        ]
        .into_iter()
        .find(|phase| {
            self.identified.values().any(|active| active == phase)
                || self.unidentified.get(phase).is_some_and(|count| *count > 0)
        })
    }

    fn clear(&mut self) {
        self.identified.clear();
        self.unidentified.clear();
        self.completed_ids.clear();
    }
}

fn traecli_item_id(value: &Value) -> Option<&str> {
    value
        .get("item")
        .and_then(|item| item.get("id"))
        .or_else(|| value.get("item_id"))
        .and_then(Value::as_str)
}

/// Append one streamed event and refresh both the live phase and the
/// human-readable activity line from it.
fn record_traecli_event(
    spec: &ExternalSubagentSpec,
    line: &str,
    stdout: &mut String,
    active_items: &mut TraeCliActiveItems,
) {
    if let Ok(value) = serde_json::from_str::<Value>(line) {
        if let Some(kind) = value.get("type").and_then(Value::as_str) {
            let (phase, activity) = traecli_event_progress(kind, &value, active_items);
            publish_traecli_activity(spec, phase, &activity);
            if kind == "item.completed"
                && let Some(item) = value.get("item")
                && item.get("type").and_then(Value::as_str) == Some("agent_message")
                && let Some(text) = item.get("text").and_then(Value::as_str)
                && let Some(slot) = spec.phase_slot.as_ref()
            {
                traecli_message_text(text, |summary, body| {
                    crate::ai::driver::runtime_ctx::publish_subagent_checkpoint_to_slot(
                        slot, summary, body,
                    );
                });
            }
        }
    }
    if !stdout.is_empty() {
        stdout.push('\n');
    }
    stdout.push_str(line);
}

/// Only complete explicit checkpoint blocks become parent progress; ordinary messages,
/// reasoning, and tool output stay out of the checkpoint carrier.
fn traecli_message_text(text: &str, mut checkpoint: impl FnMut(&str, &str)) -> String {
    const OPEN: &str = "<context_checkpoint>";
    const CLOSE: &str = "</context_checkpoint>";
    let mut remaining = String::new();
    let mut cursor = text;
    while let Some(start) = cursor.find(OPEN) {
        remaining.push_str(&cursor[..start]);
        let after_open = &cursor[start + OPEN.len()..];
        let Some(end) = after_open.find(CLOSE) else {
            remaining.push_str(&cursor[start..]);
            return remaining;
        };
        let raw = after_open[..end].trim();
        let (summary, body) = raw.split_once('\n').unwrap_or((raw, ""));
        let summary = summary
            .trim()
            .strip_prefix("summary:")
            .unwrap_or(summary)
            .trim();
        let body = body.trim();
        if !summary.is_empty() && !body.is_empty() {
            checkpoint(summary, body);
        } else {
            remaining.push_str(&cursor[start..start + OPEN.len() + end + CLOSE.len()]);
        }
        cursor = &after_open[end + CLOSE.len()..];
    }
    remaining.push_str(cursor);
    remaining
}

/// Map a traecli stream event to a coarse phase and a live activity line.
/// Activity text must differ from the phase: `SubagentProgressSnapshot::display`
/// suppresses a phase-equal activity and falls back to the legacy one-line form.
fn traecli_event_progress(
    kind: &str,
    value: &Value,
    active_items: &mut TraeCliActiveItems,
) -> (&'static str, String) {
    let item = value.get("item");
    match kind {
        "thread.started" => {
            active_items.clear();
            ("backend session started", "session started".to_string())
        }
        "turn.started" => {
            active_items.clear();
            ("thinking", "turn started".to_string())
        }
        "item.started" => {
            active_items.start(value);
            (
                active_items.phase().unwrap_or("working"),
                item.map(traecli_item_activity)
                    .unwrap_or_else(|| "item started".to_string()),
            )
        }
        "item.completed" | "item.failed" => {
            active_items.complete(value);
            (
                active_items.phase().unwrap_or("processing output"),
                if kind == "item.failed" {
                    item.map(|item| format!("failed: {}", traecli_item_activity(item)))
                        .unwrap_or_else(|| "item failed".to_string())
                } else {
                    item.map(traecli_item_completion)
                        .unwrap_or_else(|| "processing output".to_string())
                },
            )
        }
        "turn.completed" => {
            active_items.clear();
            ("finishing", "turn complete".to_string())
        }
        "error" => (
            "backend error",
            value
                .get("message")
                .and_then(Value::as_str)
                .map(|message| format!("error: {}", truncate_head(message, 80)))
                .unwrap_or_else(|| "error event".to_string()),
        ),
        _ => (
            active_items.phase().unwrap_or("working"),
            format!("event: {kind}"),
        ),
    }
}

fn traecli_item_phase(value: &Value) -> &'static str {
    match value
        .get("item")
        .and_then(|item| item.get("type"))
        .and_then(Value::as_str)
    {
        Some("reasoning") => "thinking",
        Some("agent_message") => "writing response",
        Some("command_execution") => "running command",
        Some("tool_call" | "tool_use") => "running tool",
        _ => "working",
    }
}

/// Summarize a streamed item for the status line without echoing full tool
/// output. Commands keep the command; tool calls keep the tool name and a
/// compact argument preview.
fn traecli_item_activity(item: &Value) -> String {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
    match item_type {
        "command_execution" => item
            .get("command")
            .and_then(Value::as_str)
            .map(|command| format!("command: {}", truncate_head(command, 80)))
            .unwrap_or_else(|| "command started".to_string()),
        "tool_call" | "tool_use" => {
            let name = item
                .get("name")
                .or_else(|| item.get("tool_name"))
                .and_then(Value::as_str)
                .unwrap_or("tool");
            match traecli_item_input(item) {
                Some(input) if !input.is_empty() => {
                    format!("tool {name}: {}", truncate_head(&input, 70))
                }
                _ => format!("tool {name}"),
            }
        }
        "reasoning" => "reasoning".to_string(),
        "agent_message" => "drafting response".to_string(),
        _ if item_type.is_empty() => "item started".to_string(),
        _ => format!("item: {item_type}"),
    }
}

fn traecli_item_completion(item: &Value) -> String {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
    match item_type {
        "command_execution" => {
            let command = item
                .get("command")
                .and_then(Value::as_str)
                .map(|command| truncate_head(command, 70))
                .unwrap_or_else(|| "command".to_string());
            match item.get("exit_code").and_then(Value::as_i64) {
                Some(code) => format!("completed: {command} · exit {code}"),
                None => format!("completed: {command}"),
            }
        }
        "tool_call" | "tool_use" => {
            let name = item
                .get("name")
                .or_else(|| item.get("tool_name"))
                .and_then(Value::as_str)
                .unwrap_or("tool");
            format!("completed tool {name}")
        }
        "agent_message" => "response ready".to_string(),
        "reasoning" => "reasoning complete".to_string(),
        _ => "processing output".to_string(),
    }
}

fn traecli_item_input(item: &Value) -> Option<String> {
    let value = item
        .get("input")
        .or_else(|| item.get("arguments"))
        .or_else(|| item.get("parameters"))?;
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Null => None,
        other => serde_json::to_string(other).ok(),
    }
}

fn normalize_cwd(cwd: &Path) -> PathBuf {
    if cwd.as_os_str().is_empty() {
        std::env::temp_dir()
    } else {
        cwd.to_path_buf()
    }
}

/// First non-empty, non-JSON line of stdout, if any.
fn stdout_hint(output: &str) -> Option<String> {
    output
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('{'))
        .map(|line| truncate_head(line, 500))
}

/// Char-boundary-safe head truncation with an ellipsis marker.
fn truncate_head(text: &str, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        let mut out: String = text.chars().take(max_chars).collect();
        out.push_str("...");
        out
    } else {
        text.to_string()
    }
}

/// Read the stderr snapshot without waiting forever for a pipe held open by a
/// descendant process. Normal children close stderr on exit, so the thread
/// finishes within one poll; otherwise the partial snapshot is used.
fn take_stderr_snapshot(
    mut handle: Option<std::thread::JoinHandle<()>>,
    shared: &Arc<Mutex<String>>,
) -> String {
    let deadline = Instant::now() + Duration::from_millis(250);
    loop {
        if handle.as_ref().is_some_and(|h| h.is_finished()) {
            let _ = handle.take().expect("checked").join();
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    shared.lock().map(|text| text.clone()).unwrap_or_default()
}

/// Drain streamed lines after the child exits. The reader thread normally
/// finishes immediately, but a descendant can keep the pipe open; wait only a
/// bounded grace period so the task cannot hang on inherited stdio.
fn drain_stdout_events(
    mut handle: Option<std::thread::JoinHandle<()>>,
    receiver: &std::sync::mpsc::Receiver<String>,
    spec: &ExternalSubagentSpec,
    stdout_text: &mut String,
    active_items: &mut TraeCliActiveItems,
) {
    let deadline = Instant::now() + Duration::from_millis(250);
    loop {
        while let Ok(line) = receiver.try_recv() {
            if !line.trim().is_empty() {
                record_traecli_event(spec, &line, stdout_text, active_items);
            }
        }
        if handle.as_ref().is_some_and(|h| h.is_finished()) {
            let _ = handle.take().expect("checked").join();
            while let Ok(line) = receiver.try_recv() {
                if !line.trim().is_empty() {
                    record_traecli_event(spec, &line, stdout_text, active_items);
                }
            }
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TraeCliRunner;

impl ExternalSubagentRunner for TraeCliRunner {
    fn backend(&self) -> SubagentBackend {
        SubagentBackend::TraeCli
    }

    async fn run(
        &self,
        spec: &ExternalSubagentSpec,
        cancel: Arc<AtomicBool>,
    ) -> Result<ExternalSubagentOutcome, String> {
        let outcome = run_traecli_subagent(spec, cancel).await?;
        let progress = outcome
            .num_turns
            .map(|turns| format!("traecli turns {turns}"))
            .unwrap_or_else(|| "traecli".to_string());
        Ok(ExternalSubagentOutcome {
            output: outcome.result,
            progress,
            session_id: outcome.session_id,
        })
    }
}

/// Compatibility test view of the shared, backend-scoped session index.
#[cfg(test)]
static TRAECLI_SESSIONS: LazyLock<backend_session::SessionIndex> =
    LazyLock::new(|| backend_session::session_index(BackendId::TRAECLI));

/// Keep streamed session discovery alive even if the async runner is aborted.
pub(super) fn track_traecli_session(task_id: &str) -> TraeCliSessionSlot {
    backend_session::track_session(BackendId::TRAECLI, task_id)
}

fn capture_traecli_session(slot: &TraeCliSessionSlot, line: &str) {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return;
    };
    if value.get("type").and_then(Value::as_str) == Some("thread.started")
        && let Some(session) = value.get("thread_id").and_then(Value::as_str)
        && !session.trim().is_empty()
    {
        *lock_unpoisoned(slot) = Some(session.to_string());
    }
}

/// Promote a known session when publishing any terminal result. If the run
/// returned an error or its async wrapper was aborted, use the streamed slot.
/// Unknown sessions are discarded immediately; teardown never scans the store.
pub(crate) fn record_traecli_session(task_id: &str, session_id: Option<String>) -> Option<String> {
    backend_session::record_session(BackendId::TRAECLI, task_id, session_id)
}

/// Remove a discarded task's session without making it resumable.
pub(super) fn forget_traecli_session(task_id: &str) {
    backend_session::forget_session(BackendId::TRAECLI, task_id);
}

pub(super) const CONTINUE_TARGET_ERROR: &str = "backend_continue_from references '{value}', which is neither a terminal traecli \
     task_id nor a traecli session id; pass the task_id of a terminal backend 'traecli' run";

/// Resolve a `backend_continue_from` value: a recorded parent task id first,
/// else a raw TraeCLI session id passed through when it has the expected UUID
/// shape. Anything else fails closed with a chainable hint.
pub(crate) fn resolve_continue_target(value: &str) -> Result<String, String> {
    backend_session::resolve_continue_target(
        BackendId::TRAECLI,
        value,
        looks_like_session_id,
        CONTINUE_TARGET_ERROR,
    )
}

/// Resolve a continuation before reading its deferred notes. Both synchronous
/// and scheduled tasks use the parent's history, so child artifact cleanup
/// cannot discard notes when the original task finishes or is cancelled.
/// Raw session UUIDs also read queues of recorded terminal tasks for that
/// session, scoped by the supplied parent's history. The augmented prompt is
/// reused unchanged by backend retries; queue files are cleared only after a
/// successful process spawn, so failed resolution or launch leaves notes intact.
pub(crate) fn prepare_traecli_continuation(
    continue_from: Option<&str>,
    parent_history: &Path,
    prompt: &str,
) -> Result<(Option<String>, String, Vec<PathBuf>), String> {
    backend_session::prepare_continuation(
        BackendId::TRAECLI,
        &ContinueContext {
            continue_from,
            parent_history,
            prompt,
        },
        looks_like_session_id,
        CONTINUE_TARGET_ERROR,
    )
    .map(|continuation| continuation.into_parts())
}

/// TraeCLI reports `thread_id` as a lowercase UUID (8-4-4-4-12 hex).
pub(super) fn looks_like_session_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    for (idx, byte) in bytes.iter().enumerate() {
        let is_hyphen_slot = matches!(idx, 8 | 13 | 18 | 23);
        if is_hyphen_slot != (*byte == b'-') {
            return false;
        }
        if !is_hyphen_slot && !byte.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::driver::runtime_ctx::{DRIVER_CTX, DriverContext, TURN_IDENTITY};
    use crate::ai::mcp::McpClient;
    use crate::ai::middleware::test_util::test_app;
    use crate::ai::tools::task_tools::{
        AsyncTaskEntry, InheritOptions, backend_session_footer, discard_tasks_for_session,
        drain_task_result, execute_task_evidence_read, insert_task_entry_for_test,
        remove_task_entry,
    };

    #[cfg(unix)]
    static PROCESS_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn session_test_entry(
        os: &mut dyn aios_kernel::kernel::Kernel,
        owner_pid: u64,
        session_id: &str,
        backend: Option<SubagentBackend>,
    ) -> AsyncTaskEntry {
        let pid = os
            .spawn(
                Some(owner_pid),
                "child".to_string(),
                "goal".to_string(),
                20,
                8,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create_tagged_with_holders(
            Some(owner_pid),
            1,
            "task-result".to_string(),
            aios_kernel::primitives::ChannelOwnerTag::TaskResult,
            vec![
                "task_result.producer".to_string(),
                "task_result.consumer".to_string(),
            ],
        );
        let completion_futex = os.futex_create(0, "task-complete".to_string());
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: session_id.to_string(),
            owner_pid,
            pid,
            result_channel_id: channel.raw(),
            completion_futex_addr: completion_futex,
            description: "session promotion".to_string(),
            agent_name: "explore".to_string(),
            backend,
            model: "backend-test-model".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
        }
    }

    fn check_synthesized_terminal_collection(
        captured_session: Option<&str>,
        backend: Option<SubagentBackend>,
    ) {
        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
        if backend == Some(SubagentBackend::TraeCli) {
            let slot = track_traecli_session(&task_id);
            if let Some(session) = captured_session {
                capture_traecli_session(
                    &slot,
                    &serde_json::json!({"type": "thread.started", "thread_id": session})
                        .to_string(),
                );
            }
            assert!(resolve_continue_target(&task_id).is_err());
        }

        let root = std::env::temp_dir().join(format!(
            "traecli-session-promotion-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let mut app = test_app();
        app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
        app.config.history_file = root.join("history.sqlite");
        let shared_os = app.os.clone();
        let entry = {
            let mut os = shared_os.lock().unwrap();
            let owner_pid =
                os.begin_foreground("parent".to_string(), "goal".to_string(), 10, 8, None);
            let entry = session_test_entry(os.as_mut(), owner_pid, &app.session_id, backend);
            os.set_current_pid(Some(owner_pid));
            os.kill_process(
                entry.pid,
                "subagent terminated outside publisher".to_string(),
            )
            .unwrap();
            entry
        };
        insert_task_entry_for_test(task_id.clone(), entry.clone());
        let context = DriverContext::new(
            app,
            Arc::new(Mutex::new(McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        TURN_IDENTITY.sync_scope((entry.session_id.clone(), 0usize), || {
            DRIVER_CTX.sync_scope(context, || {
                let collected = {
                    let mut os = shared_os.lock().unwrap();
                    drain_task_result(os.as_mut(), &task_id, &entry, "test collection")
                        .unwrap()
                        .expect("collectable synthesized failure")
                };
                let error = format!(
                    "Subagent process pid={} terminated without publishing any output.",
                    entry.pid
                );
                assert_eq!(collected.result.status, "failed");
                assert_eq!(collected.result.error.as_deref(), Some(error.as_str()));
                assert_eq!(
                    collected.result.backend_session.as_deref(),
                    captured_session
                );
                remove_task_entry(&task_id).expect("collected task entry");

                let delivered = execute_task_evidence_read(&serde_json::json!({
                    "task_id": task_id,
                    "source": "delivered",
                }))
                .unwrap();
                let evidence: Value = serde_json::from_str(&delivered).unwrap();
                assert_eq!(evidence["task_id"], task_id);
                assert_eq!(evidence["source"], "delivered");
                assert_eq!(evidence["status"], "failed");
                let payload = evidence["payload"].as_str().unwrap();
                assert_eq!(payload, collected.rendered);
                assert!(payload.contains(&error), "{payload}");
                if let Some(session) = captured_session {
                    assert!(
                        payload.contains(&backend_session_footer(session)),
                        "{payload}"
                    );
                    assert_eq!(resolve_continue_target(&task_id).unwrap(), session);
                } else {
                    assert!(!payload.contains("Backend session:"), "{payload}");
                    assert!(!lock_unpoisoned(&TRAECLI_SESSIONS).contains_key(&task_id));
                    assert!(resolve_continue_target(&task_id).is_err());
                }
            })
        });

        let os = shared_os.lock().unwrap();
        assert!(
            os.channel_meta(aios_kernel::primitives::ChannelId(entry.result_channel_id))
                .is_none()
        );
        assert!(os.futex_event_id(entry.completion_futex_addr).is_none());
        assert!(os.get_process(entry.pid).is_none());
        assert!(os.get_process(entry.owner_pid).is_some());
        drop(os);
        forget_traecli_session(&task_id);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn traecli_session_promotion_synthesized_failure_with_session() {
        check_synthesized_terminal_collection(
            Some("dddddddd-1111-2222-3333-444444444444"),
            Some(SubagentBackend::TraeCli),
        );
    }

    #[test]
    fn traecli_session_promotion_synthesized_failure_without_session() {
        for backend in [
            Some(SubagentBackend::TraeCli),
            None,
            Some(SubagentBackend::Kernel),
        ] {
            check_synthesized_terminal_collection(None, backend);
        }
    }

    #[test]
    fn traecli_session_promotion_discard_removes_pending_and_terminal_entries() {
        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let session = "eeeeeeee-1111-2222-3333-444444444444";
        let mut app = test_app();
        app.session_id = format!("discard-session-{}", uuid::Uuid::new_v4().simple());
        let retained_session = format!("retained-session-{}", uuid::Uuid::new_v4().simple());
        let shared_os = app.os.clone();
        let owner_pid = shared_os.lock().unwrap().begin_foreground(
            "parent".to_string(),
            "goal".to_string(),
            10,
            8,
            None,
        );
        let mut discarded = Vec::new();
        for (known_session, terminal) in [(false, false), (true, false), (true, true)] {
            let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
            let slot = track_traecli_session(&task_id);
            if known_session {
                capture_traecli_session(
                    &slot,
                    &serde_json::json!({"type": "thread.started", "thread_id": session})
                        .to_string(),
                );
            }
            if terminal {
                assert_eq!(
                    record_traecli_session(&task_id, None).as_deref(),
                    Some(session)
                );
            }
            let entry = session_test_entry(
                shared_os.lock().unwrap().as_mut(),
                owner_pid,
                &app.session_id,
                Some(SubagentBackend::TraeCli),
            );
            insert_task_entry_for_test(task_id.clone(), entry.clone());
            discarded.push((task_id, entry));
        }
        let retained_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
        let _ = track_traecli_session(&retained_task_id);
        let retained_entry = session_test_entry(
            shared_os.lock().unwrap().as_mut(),
            owner_pid,
            &retained_session,
            Some(SubagentBackend::TraeCli),
        );
        insert_task_entry_for_test(retained_task_id.clone(), retained_entry.clone());

        let session_id = app.session_id.clone();
        let context = DriverContext::new(
            app,
            Arc::new(Mutex::new(McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        DRIVER_CTX.sync_scope(context, || {
            discard_tasks_for_session(&session_id);
            for (task_id, entry) in &discarded {
                assert!(entry.cancel_stream.load(Ordering::Acquire));
                assert!(remove_task_entry(task_id).is_none());
                assert!(!lock_unpoisoned(&TRAECLI_SESSIONS).contains_key(task_id));
                assert!(resolve_continue_target(task_id).is_err());
                let os = shared_os.lock().unwrap();
                assert!(os.get_process(entry.pid).is_none());
                assert!(
                    os.channel_meta(aios_kernel::primitives::ChannelId(entry.result_channel_id))
                        .is_none()
                );
                assert!(os.futex_event_id(entry.completion_futex_addr).is_none());
            }
            assert!(matches!(
                lock_unpoisoned(&TRAECLI_SESSIONS).get(&retained_task_id),
                Some(TraeCliSessionState::Pending(_))
            ));
            assert!(!retained_entry.cancel_stream.load(Ordering::Acquire));
            assert!(
                shared_os
                    .lock()
                    .unwrap()
                    .get_process(retained_entry.pid)
                    .is_some()
            );
            discard_tasks_for_session(&retained_session);
            assert!(!lock_unpoisoned(&TRAECLI_SESSIONS).contains_key(&retained_task_id));
            assert!(remove_task_entry(&retained_task_id).is_none());
        });
    }

    fn test_spec() -> ExternalSubagentSpec {
        ExternalSubagentSpec {
            prompt: "do work".to_string(),
            cwd: PathBuf::from("/tmp"),
            workspace_allowed: true,
            model: None,
            skills: None,
            resume_session: None,
            side_note_files: Vec::new(),
            reasoning_effort: None,
            image_files: Vec::new(),
            phase_slot: None,
        }
    }

    #[test]
    fn read_only_sandbox_hint_requires_failed_run_and_specific_evidence() {
        let mut spec = test_spec();
        spec.workspace_allowed = false;
        for evidence in [
            "socket: Operation not permitted (os error 1)",
            "connect: EACCES",
            "Network is unreachable",
            "curl: (6) Could not resolve host: api.example.test",
            "DNS lookup failed",
            "error creating /workspace/output.txt: Permission denied",
            "write /workspace/output.txt: EACCES",
            "mkdir /workspace/cache: EPERM",
            "Read-only file system (os error 30)",
        ] {
            assert!(read_only_sandbox_hint(&spec, true, &[evidence]).is_some());
            assert!(read_only_sandbox_hint(&spec, false, &[evidence]).is_none());
            assert!(read_only_sandbox_hint(&test_spec(), true, &[evidence]).is_none());
        }
        for evidence in [
            "",
            "traecli exited with status 1 and no result event",
            "compiler failed: unknown type",
            "permission denied reading /private/config",
            "read /private/config: EACCES",
            "socket command has invalid arguments",
            "invalid network configuration",
            "network\npermission denied",
        ] {
            assert!(read_only_sandbox_hint(&spec, true, &[evidence]).is_none());
        }
        assert!(
            read_only_sandbox_hint(
                &spec,
                true,
                &["generic failure", "write output.txt: Permission denied"]
            )
            .is_some()
        );
        assert!(read_only_sandbox_hint(&spec, true, &["network", "permission denied"]).is_none());
        spec.resume_session = Some("recorded-session".to_string());
        assert!(
            read_only_sandbox_hint(&spec, true, &["write output.txt: Permission denied"]).is_none()
        );
        for evidence in [
            "Read-only file system (os error 30)",
            "socket: Operation not permitted (os error 1)",
            "error sending request: connection refused",
        ] {
            assert!(read_only_sandbox_hint(&spec, true, &[evidence]).is_none());
        }
    }

    #[test]
    fn read_only_sandbox_hint_ignores_permission_denied_paths() {
        let mut spec = test_spec();
        spec.workspace_allowed = false;
        for evidence in [
            "permission denied reading /private/network.conf",
            "permission denied reading /private/socket.conf",
            "read /private/write: EACCES",
            "read /private/connect: EPERM",
            "cannot read socket: permission denied",
            "Error: EACCES: permission denied, read '/private/network.conf'",
        ] {
            assert_eq!(read_only_sandbox_hint(&spec, true, &[evidence]), None);
        }
    }

    #[test]
    fn read_only_sandbox_hint_makes_connectivity_advice_conditional() {
        let mut spec = test_spec();
        spec.workspace_allowed = false;
        for evidence in [
            "error sending request: connection refused",
            "failed to connect to backend endpoint",
            "Network is unreachable",
            "curl: (6) Could not resolve host: api.example.test",
            "Could not resolve proxy: proxy.example.test",
            "Temporary failure in name resolution",
            "Name or service not known",
            "DNS lookup failed",
            "DNS resolution failed",
        ] {
            let hint = read_only_sandbox_hint(&spec, true, &[evidence]).unwrap();
            assert_eq!(hint, TRAECLI_POSSIBLE_READ_ONLY_SANDBOX_HINT);
            assert!(hint.contains("If this failure is due to the read-only sandbox"));
            assert!(!hint.contains("Respawn with inherit:"));
        }
    }

    #[test]
    fn read_only_sandbox_hint_keeps_imperative_operation_advice() {
        let mut spec = test_spec();
        spec.workspace_allowed = false;
        for evidence in [
            "Read-only file system (os error 30)",
            "Read-only filesystem",
            "socket: Operation not permitted (os error 1)",
            "socket creation failed: EPERM",
            "socket(AF_INET, SOCK_STREAM, 0) = -1 EPERM (Operation not permitted)",
            "failed to create socket: EPERM",
            "traecli subagent failed: socket: Operation not permitted (os error 1)",
            "connect: EACCES",
            "error creating /workspace/output.txt: Permission denied",
            "write /workspace/output.txt: EACCES",
            "mkdir /workspace/cache: EPERM",
            "open '/workspace/output.txt': EACCES",
            "Error: EACCES: permission denied, open '/workspace/output.txt'",
            "rename /workspace/output.txt: EACCES",
            "unlink /workspace/output.txt: EPERM",
        ] {
            assert_eq!(
                read_only_sandbox_hint(&spec, true, &[evidence]),
                Some(TRAECLI_READ_ONLY_SANDBOX_HINT),
                "{evidence}"
            );
            for evidence in [
                ["error sending request: connection refused", evidence],
                [evidence, "error sending request: connection refused"],
            ] {
                assert_eq!(
                    read_only_sandbox_hint(&spec, true, &evidence),
                    Some(TRAECLI_READ_ONLY_SANDBOX_HINT)
                );
            }
        }
    }

    #[test]
    fn read_only_sandbox_hint_matches_parenthesized_failed_operations() {
        let mut spec = test_spec();
        spec.workspace_allowed = false;
        for evidence in [
            "socket() failed: EPERM",
            "socket() creation failed: EPERM",
            "sendto(fd, buf, len, 0) failed: EACCES",
            "socket(...) = -1 EPERM",
            "openat(AT_FDCWD, \"/x\", O_WRONLY) = -1 EACCES",
        ] {
            assert_eq!(
                read_only_sandbox_hint(&spec, true, &[evidence]),
                Some(TRAECLI_READ_ONLY_SANDBOX_HINT),
                "{evidence}"
            );
        }
        assert_eq!(
            read_only_sandbox_hint(&spec, true, &["socket() failed: ECONNRESET"]),
            None
        );
    }

    #[cfg(unix)]
    fn terminal_test_script(body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let script = std::env::temp_dir().join(format!(
            "traecli-terminal-{}.sh",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::write(&script, format!("#!/bin/sh\n{body}\n")).unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).unwrap();
        script
    }

    #[test]
    fn traecli_retry_backoff_waits_one_then_two_seconds() {
        assert_eq!(traecli_retry_backoff(1), Duration::from_secs(1));
        assert_eq!(traecli_retry_backoff(2), Duration::from_secs(2));
    }

    #[test]
    fn traecli_permanent_errors_are_classified_narrowly() {
        for error in [
            "traecli did not emit a startup event within 45s",
            "traecli exited with status 1 and no result event",
            "traecli subagent failed: connection reset by peer",
            "traecli subagent failed: rate limit exceeded",
        ] {
            assert!(!is_permanent_traecli_error(error), "{error}");
        }
        for error in [
            "failed to spawn traecli: No such file or directory (os error 2)",
            "traecli subagent failed: unauthorized",
            "traecli subagent failed: authentication failed: token expired",
            "traecli exited with status 2: error: unexpected argument '--json'",
        ] {
            assert!(is_permanent_traecli_error(error), "{error}");
        }
    }

    #[cfg(unix)]
    fn retry_test_script(body: &str) -> PathBuf {
        terminal_test_script(&format!(
            r#"attempt=1
if [ -f "${{0%.sh}}.count" ]; then
    read -r previous < "${{0%.sh}}.count"
    attempt=$((previous + 1))
fi
printf '%s\n' "$attempt" > "${{0%.sh}}.count"
printf 'attempt\n' >> "${{0%.sh}}.args"
printf '%s\n' "$@" >> "${{0%.sh}}.args"
/bin/cat > "${{0%.sh}}.prompt$attempt"
{body}"#
        ))
    }

    #[cfg(unix)]
    fn remove_retry_test_script(script: &Path) {
        std::fs::remove_file(script).unwrap();
        for suffix in ["args", "count", "prompt1", "prompt2", "prompt3"] {
            let _ = std::fs::remove_file(script.with_extension(suffix));
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn traecli_retry_resumes_latest_session_and_stops_after_three_attempts() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        let script = retry_test_script(
            r#"/bin/echo "{\"type\":\"thread.started\",\"thread_id\":\"session-$attempt\"}"
/bin/echo '{"type":"error","message":"boom"}'
exit 1"#,
        );
        let mut spec = test_spec();
        spec.skills = Some("original skill overlay".to_string());
        let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
        let error = crate::ai::driver::runtime_ctx::SUBAGENT_TASK_ID
            .scope(
                task_id.clone(),
                run_traecli_subagent_with(
                    &spec,
                    Arc::new(AtomicBool::new(false)),
                    script.to_str().unwrap(),
                    Duration::from_secs(2),
                ),
            )
            .await
            .unwrap_err();
        assert_eq!(
            error,
            "traecli subagent failed after 3 attempts: traecli subagent failed: boom"
        );
        assert_eq!(
            std::fs::read_to_string(script.with_extension("count")).unwrap(),
            "3\n"
        );
        let args = std::fs::read_to_string(script.with_extension("args")).unwrap();
        let attempts = args
            .split("attempt\n")
            .skip(1)
            .map(|args| args.lines().map(str::to_string).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(attempts.len(), 3, "{args}");
        assert_eq!(attempts[0], build_traecli_args(&spec));
        for attempt in 1..3 {
            let mut resume_spec = spec.clone();
            resume_spec.resume_session = Some(format!("session-{attempt}"));
            assert_eq!(attempts[attempt], build_traecli_args(&resume_spec));
            assert_eq!(
                std::fs::read_to_string(script.with_extension(format!("prompt{}", attempt + 1)))
                    .unwrap(),
                spec.prompt
            );
        }
        assert_eq!(
            std::fs::read_to_string(script.with_extension("prompt1")).unwrap(),
            full_prompt(&spec)
        );
        assert!(resolve_continue_target(&task_id).is_err());
        assert_eq!(
            record_traecli_session(&task_id, None).as_deref(),
            Some("session-3")
        );
        assert_eq!(resolve_continue_target(&task_id).unwrap(), "session-3");
        forget_traecli_session(&task_id);
        remove_retry_test_script(&script);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn traecli_missing_final_output_does_not_replay_tools() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        for (terminal_event, expected_error) in [
            (
                r#"{"type":"item.completed","item":{"id":"tool","type":"tool_call"}}"#,
                "traecli produced no result event",
            ),
            (
                r#"{"type":"turn.completed"}"#,
                "traecli produced an empty result",
            ),
        ] {
            let script = retry_test_script(&format!(
                "/bin/echo '{{\"type\":\"thread.started\",\"thread_id\":\"mutated-session\"}}'\n/bin/echo '{terminal_event}'"
            ));
            let error = run_traecli_subagent_with(
                &test_spec(),
                Arc::new(AtomicBool::new(false)),
                script.to_str().unwrap(),
                Duration::from_secs(2),
            )
            .await
            .unwrap_err();
            assert_eq!(error, expected_error);
            assert_eq!(
                std::fs::read_to_string(script.with_extension("count")).unwrap(),
                "1\n",
                "a completed tool may have already mutated state"
            );
            remove_retry_test_script(&script);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn traecli_retry_starts_fresh_without_session_and_returns_success() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        let script = retry_test_script(
            r#"if [ "$attempt" = 1 ]; then
    /bin/echo '{"type":"error","message":"boom"}'
    exit 1
fi
/bin/echo '{"type":"thread.started","thread_id":"successful-session"}'
/bin/echo '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}'"#,
        );
        let mut spec = test_spec();
        spec.skills = Some("original skill overlay".to_string());
        let outcome = run_traecli_subagent_with(
            &spec,
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        assert_eq!(outcome.result, "ok");
        assert_eq!(outcome.session_id.as_deref(), Some("successful-session"));
        assert_eq!(
            std::fs::read_to_string(script.with_extension("count")).unwrap(),
            "2\n"
        );
        let args = std::fs::read_to_string(script.with_extension("args")).unwrap();
        for args in args.split("attempt\n").skip(1) {
            assert_eq!(args.lines().collect::<Vec<_>>(), build_traecli_args(&spec));
        }
        for suffix in ["prompt1", "prompt2"] {
            assert_eq!(
                std::fs::read_to_string(script.with_extension(suffix)).unwrap(),
                full_prompt(&spec)
            );
        }
        remove_retry_test_script(&script);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn traecli_retry_keeps_explicit_resume_without_captured_session() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        let script = retry_test_script(
            r#"/bin/echo '{"type":"error","message":"boom"}'
exit 1"#,
        );
        let mut spec = test_spec();
        spec.resume_session = Some("original-session".to_string());
        spec.skills = Some("original skill overlay".to_string());
        let error = run_traecli_subagent_with(
            &spec,
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_secs(2),
        )
        .await
        .unwrap_err();
        assert_eq!(
            error,
            "traecli subagent failed after 3 attempts: traecli subagent failed: boom"
        );
        let args = std::fs::read_to_string(script.with_extension("args")).unwrap();
        let attempts = args.split("attempt\n").skip(1).collect::<Vec<_>>();
        assert_eq!(attempts.len(), 3, "{args}");
        for args in attempts {
            assert_eq!(args.lines().collect::<Vec<_>>(), build_traecli_args(&spec));
        }
        for suffix in ["prompt1", "prompt2", "prompt3"] {
            assert_eq!(
                std::fs::read_to_string(script.with_extension(suffix)).unwrap(),
                spec.prompt
            );
        }
        remove_retry_test_script(&script);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn traecli_retry_startup_failure_reuses_task_session() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        for original_resume in [None, Some("original-session".to_string())] {
            let script = retry_test_script(
                r#"if [ "$attempt" = 1 ]; then
    /bin/echo '{"type":"thread.started","thread_id":"captured-session"}'
fi
/bin/echo '{"type":"error","message":"boom"}'
exit 1"#,
            );
            let mut spec = test_spec();
            spec.resume_session = original_resume;
            spec.skills = Some("original skill overlay".to_string());
            let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
            let error = crate::ai::driver::runtime_ctx::SUBAGENT_TASK_ID
                .scope(
                    task_id.clone(),
                    run_traecli_subagent_with(
                        &spec,
                        Arc::new(AtomicBool::new(false)),
                        script.to_str().unwrap(),
                        Duration::from_secs(2),
                    ),
                )
                .await
                .unwrap_err();
            assert_eq!(
                error,
                "traecli subagent failed after 3 attempts: traecli subagent failed: boom"
            );
            let args = std::fs::read_to_string(script.with_extension("args")).unwrap();
            let attempts = args
                .split("attempt\n")
                .skip(1)
                .map(|args| args.lines().collect::<Vec<_>>())
                .collect::<Vec<_>>();
            assert_eq!(attempts.len(), 3, "{args}");
            assert_eq!(attempts[0], build_traecli_args(&spec));
            let mut resume_spec = spec.clone();
            resume_spec.resume_session = Some("captured-session".to_string());
            // The second attempt fails before emitting a session, so the third
            // must reuse the task-level session captured by the first attempt
            // instead of falling back to the caller's original session.
            assert_eq!(attempts[1], build_traecli_args(&resume_spec));
            assert_eq!(attempts[2], build_traecli_args(&resume_spec));
            for (attempt, attempt_spec) in [(1, &spec), (2, &resume_spec), (3, &resume_spec)] {
                assert_eq!(
                    std::fs::read_to_string(script.with_extension(format!("prompt{attempt}")))
                        .unwrap(),
                    prompt_payload(attempt_spec)
                );
            }
            assert_eq!(
                record_traecli_session(&task_id, None).as_deref(),
                Some("captured-session")
            );
            forget_traecli_session(&task_id);
            remove_retry_test_script(&script);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn traecli_retry_cancellation_interrupts_backoff() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        let script = retry_test_script(
            r#"/bin/echo '{"type":"error","message":"boom"}'
exit 1"#,
        );
        let mut spec = test_spec();
        let slot = crate::ai::driver::runtime_ctx::new_subagent_progress_slot();
        spec.phase_slot = Some(slot.clone());
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_wait = async {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let progress = crate::ai::driver::runtime_ctx::subagent_progress_snapshot(&slot)
                    .unwrap_or_default();
                if progress.contains("retrying traecli (attempt 2/3, fresh start)") {
                    assert!(progress.contains("waiting 1s"), "{progress}");
                    cancel.store(true, Ordering::Release);
                    return Instant::now();
                }
                assert!(
                    Instant::now() < deadline,
                    "retry progress missing: {progress}"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        };
        let (result, cancelled_at) = tokio::join!(
            run_traecli_subagent_with(
                &spec,
                cancel.clone(),
                script.to_str().unwrap(),
                Duration::from_secs(2),
            ),
            cancel_wait
        );
        assert_eq!(result.unwrap_err(), "traecli subagent cancelled");
        assert!(cancelled_at.elapsed() < Duration::from_secs(1));
        assert_eq!(
            std::fs::read_to_string(script.with_extension("count")).unwrap(),
            "1\n"
        );
        remove_retry_test_script(&script);
    }

    #[tokio::test]
    async fn traecli_retry_rejects_empty_prompt_and_pre_cancel_without_spawning() {
        let mut spec = test_spec();
        spec.prompt.clear();
        assert_eq!(
            run_traecli_subagent_with(
                &spec,
                Arc::new(AtomicBool::new(false)),
                "/nonexistent-traecli-test-binary",
                Duration::from_secs(2),
            )
            .await
            .unwrap_err(),
            "traecli subagent prompt is empty"
        );
        assert_eq!(
            run_traecli_subagent_with(
                &test_spec(),
                Arc::new(AtomicBool::new(true)),
                "/nonexistent-traecli-test-binary",
                Duration::from_secs(2),
            )
            .await
            .unwrap_err(),
            "traecli subagent cancelled"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn traecli_permanent_error_is_attempted_once() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        let script = retry_test_script(
            r#"/bin/echo '{"type":"error","message":"authentication failed: invalid api key"}'
exit 1"#,
        );
        let error = run_traecli_subagent_with(
            &test_spec(),
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_secs(2),
        )
        .await
        .unwrap_err();
        assert_eq!(
            error,
            "traecli subagent failed: authentication failed: invalid api key"
        );
        assert_eq!(
            std::fs::read_to_string(script.with_extension("count")).unwrap(),
            "1\n"
        );
        assert!(!script.with_extension("prompt2").exists());
        remove_retry_test_script(&script);

        let error = run_traecli_subagent_with(
            &test_spec(),
            Arc::new(AtomicBool::new(false)),
            "/nonexistent-traecli-test-binary",
            Duration::from_secs(2),
        )
        .await
        .unwrap_err();
        assert!(error.starts_with("failed to spawn traecli:"), "{error}");
        assert!(!error.contains("after 3 attempts"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn traecli_terminal_errors_keep_streamed_session() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        let session = "aaaaaaaa-1111-2222-3333-444444444444";
        for (body, expected_error) in [
            (
                "/bin/echo '{\"type\":\"error\",\"message\":\"boom\"}'",
                "traecli subagent failed: boom",
            ),
            (
                "/bin/echo '{\"type\":\"turn.completed\"}'",
                "traecli produced an empty result",
            ),
            ("exit 1", "traecli exited with status"),
        ] {
            let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
            let slot = track_traecli_session(&task_id);
            let script = terminal_test_script(&format!(
                "/bin/echo '{{\"type\":\"thread.started\",\"thread_id\":\"{session}\"}}'\n{body}"
            ));
            let err = run_traecli_blocking_with(
                &test_spec(),
                Arc::new(AtomicBool::new(false)),
                script.to_str().unwrap(),
                Duration::from_secs(2),
                vec![slot],
            )
            .unwrap_err();

            assert!(err.contains(expected_error), "{err}");
            assert!(resolve_continue_target(&task_id).is_err());
            assert_eq!(
                record_traecli_session(&task_id, None).as_deref(),
                Some(session)
            );
            assert_eq!(resolve_continue_target(&task_id).unwrap(), session);
            std::fs::remove_file(script).unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn traecli_terminal_cancel_keeps_streamed_session() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        let session = "bbbbbbbb-1111-2222-3333-444444444444";
        let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
        let slot = track_traecli_session(&task_id);
        let script = terminal_test_script(&format!(
            "/bin/echo '{{\"type\":\"thread.started\",\"thread_id\":\"{session}\"}}'\nexec sleep 30"
        ));
        let binary = script.to_str().unwrap().to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let worker_slot = slot.clone();
        let worker = std::thread::spawn(move || {
            run_traecli_blocking_with(
                &test_spec(),
                worker_cancel,
                &binary,
                Duration::from_secs(2),
                vec![worker_slot],
            )
        });

        let deadline = Instant::now() + Duration::from_secs(5);
        while lock_unpoisoned(&slot).is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(resolve_continue_target(&task_id).is_err());
        cancel.store(true, Ordering::Release);
        // Terminal publication can happen before the blocking worker returns,
        // because aborting its async wrapper does not stop spawn_blocking.
        let recorded = record_traecli_session(&task_id, None);
        let err = worker.join().unwrap().unwrap_err();
        std::fs::remove_file(script).unwrap();

        assert_eq!(err, "traecli subagent cancelled");
        assert_eq!(recorded.as_deref(), Some(session));
        assert_eq!(resolve_continue_target(&task_id).unwrap(), session);
    }

    #[test]
    fn streamed_checkpoint_reaches_parent_snapshot_once_per_revision() {
        use crate::ai::driver::runtime_ctx::{
            subagent_progress_state_snapshot, take_subagent_parent_progress_snapshot,
        };
        let mut spec = test_spec();
        let slot = crate::ai::driver::runtime_ctx::new_subagent_progress_slot();
        spec.phase_slot = Some(slot.clone());
        assert!(prompt_payload(&spec).contains("<context_checkpoint>"));
        let mut stdout = String::new();
        let mut active_items = TraeCliActiveItems::default();
        let body = format!(
            "Confirmed the external publication path.\n{}",
            "detail\n".repeat(200)
        );
        let checkpoint_text = format!(
            "<context_checkpoint>\nsummary: External finding\n{body}\n</context_checkpoint>"
        );
        let event = serde_json::json!({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": checkpoint_text}
        })
        .to_string();
        // No in-process child messaging runtime or task-local phase is installed.
        record_traecli_event(&spec, &event, &mut stdout, &mut active_items);
        let snapshot = subagent_progress_state_snapshot(&slot).unwrap();
        assert_eq!(snapshot.checkpoint_revision, 1);
        assert_eq!(
            snapshot.checkpoint_summary.as_deref(),
            Some("External finding")
        );
        assert!(!snapshot.checkpoint_due);
        let excerpt = snapshot.checkpoint_excerpt.as_deref().unwrap();
        assert!(excerpt.starts_with("Confirmed the external publication path."));
        assert!(excerpt.chars().count() <= 800);
        assert!(excerpt.lines().count() <= 8);
        assert!(excerpt.ends_with("… [truncated]"));
        let parent = take_subagent_parent_progress_snapshot(&slot, true).unwrap();
        assert_eq!(parent.checkpoint_excerpt.as_deref(), Some(excerpt));
        let rendered = parent.display();
        assert_eq!(
            rendered
                .matches("Confirmed the external publication path.")
                .count(),
            1
        );
        let rendered_excerpt = rendered.split_once("unverified):\n").unwrap().1;
        assert_eq!(
            rendered_excerpt
                .lines()
                .map(|line| line.strip_prefix("    ").unwrap())
                .collect::<Vec<_>>()
                .join("\n"),
            excerpt
        );
        assert!(
            parent
                .timeline
                .iter()
                .all(|event| !event.summary.contains(excerpt))
        );
        assert!(
            take_subagent_parent_progress_snapshot(&slot, true)
                .unwrap()
                .checkpoint_excerpt
                .is_none()
        );

        for (kind, item_type, text) in [
            ("item.started", "agent_message", checkpoint_text.as_str()),
            ("item.completed", "reasoning", checkpoint_text.as_str()),
            ("item.completed", "agent_message", "ordinary commentary"),
            (
                "item.completed",
                "agent_message",
                "<context_checkpoint>\nsummary: incomplete",
            ),
        ] {
            record_traecli_event(
                &spec,
                &serde_json::json!({
                    "type": kind, "item": {"type": item_type, "text": text}
                })
                .to_string(),
                &mut stdout,
                &mut active_items,
            );
        }
        assert_eq!(
            subagent_progress_state_snapshot(&slot)
                .unwrap()
                .checkpoint_revision,
            1
        );
        for body in ["superseded checkpoint", "latest checkpoint"] {
            record_traecli_event(&spec, &serde_json::json!({
                "type": "item.completed", "item": {"type": "agent_message", "text": format!(
                    "<context_checkpoint>\nsummary: Updated finding\n{body}\n</context_checkpoint>"
                )}
            }).to_string(), &mut stdout, &mut active_items);
        }
        let latest = take_subagent_parent_progress_snapshot(&slot, true).unwrap();
        assert_eq!(latest.checkpoint_revision, 3);
        assert_eq!(
            latest.checkpoint_excerpt.as_deref(),
            Some("latest checkpoint")
        );
        record_traecli_event(&spec, &event, &mut stdout, &mut active_items);
        assert!(
            take_subagent_parent_progress_snapshot(&slot, false)
                .unwrap()
                .checkpoint_excerpt
                .is_none()
        );
    }

    #[test]
    fn parse_exec_json_keeps_final_answer_separate_from_checkpoints() {
        let output = r#"{"type":"item.completed","item":{"type":"agent_message","text":"final answer"}}
{"type":"item.completed","item":{"type":"agent_message","text":"<context_checkpoint>\nsummary: Finding\ncheckpoint body\n</context_checkpoint>"}}"#;
        assert_eq!(
            parse_stream_json_output(output).unwrap().result,
            "final answer"
        );
        assert_eq!(
            traecli_message_text(
                "<context_checkpoint>\nsummary: Missing body\n</context_checkpoint>",
                |_, _| panic!("incomplete checkpoint")
            ),
            "<context_checkpoint>\nsummary: Missing body\n</context_checkpoint>"
        );
    }

    #[test]
    fn streamed_events_update_the_live_phase() {
        let mut spec = test_spec();
        let slot = crate::ai::driver::runtime_ctx::new_subagent_progress_slot();
        spec.phase_slot = Some(slot.clone());
        let mut stdout = String::new();
        let mut active_items = TraeCliActiveItems::default();

        record_traecli_event(
            &spec,
            r#"{"type":"turn.started"}"#,
            &mut stdout,
            &mut active_items,
        );
        assert!(
            crate::ai::driver::runtime_ctx::subagent_progress_snapshot(&slot)
                .unwrap_or_default()
                .contains("thinking")
        );

        record_traecli_event(
            &spec,
            r#"{"type":"item.started","item":{"type":"tool_call"}}"#,
            &mut stdout,
            &mut active_items,
        );
        assert!(
            crate::ai::driver::runtime_ctx::subagent_progress_snapshot(&slot)
                .unwrap_or_default()
                .contains("running tool")
        );
        record_traecli_event(
            &spec,
            r#"{"type":"item.started","item":{"type":"command_execution","command":"rg -n foo src"}}"#,
            &mut stdout,
            &mut active_items,
        );
        let progress =
            crate::ai::driver::runtime_ctx::subagent_progress_snapshot(&slot).unwrap_or_default();
        assert!(progress.contains("command: rg -n foo src"), "{progress}");
        assert!(progress.contains("running command"), "{progress}");
        assert!(progress.contains("updates "), "{progress}");
        assert!(stdout.contains("turn.started"));
        assert!(stdout.contains("item.started"));
    }

    #[test]
    fn streamed_command_completion_clears_the_running_phase() {
        let mut spec = test_spec();
        let slot = crate::ai::driver::runtime_ctx::new_subagent_progress_slot();
        spec.phase_slot = Some(slot.clone());
        let mut stdout = String::new();
        let mut active_items = TraeCliActiveItems::default();

        record_traecli_event(
            &spec,
            r#"{"type":"item.started","item":{"type":"command_execution","command":"rg -n foo src"}}"#,
            &mut stdout,
            &mut active_items,
        );
        assert_eq!(
            crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot)
                .unwrap()
                .phase,
            "running command"
        );
        for exit_code in [0, 1] {
            let event = serde_json::json!({
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "rg -n foo src",
                    "exit_code": exit_code,
                },
            });
            record_traecli_event(&spec, &event.to_string(), &mut stdout, &mut active_items);
            let snapshot =
                crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot).unwrap();
            assert_eq!(snapshot.phase, "processing output");
            let progress = snapshot.display();
            assert!(
                progress.starts_with(&format!("completed: rg -n foo src · exit {exit_code}")),
                "{progress}"
            );
            assert!(!progress.contains("running command"), "{progress}");
        }
        record_traecli_event(
            &spec,
            r#"{"type":"turn.completed"}"#,
            &mut stdout,
            &mut active_items,
        );
        assert_eq!(
            crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot)
                .unwrap()
                .phase,
            "finishing"
        );
    }

    #[test]
    fn streamed_mixed_item_ids_clear_completed_phases() {
        for start_identified in [false, true] {
            for id_location in ["item", "event"] {
                for terminal_kind in ["item.completed", "item.failed"] {
                    let mut spec = test_spec();
                    let slot = crate::ai::driver::runtime_ctx::new_subagent_progress_slot();
                    spec.phase_slot = Some(slot.clone());
                    let mut stdout = String::new();
                    let mut active_items = TraeCliActiveItems::default();
                    for (kind, identified, expected) in [
                        ("item.started", start_identified, "running command"),
                        (terminal_kind, !start_identified, "processing output"),
                    ] {
                        let mut event = serde_json::json!({
                            "type": kind,
                            "item": {"type": "command_execution", "command": "sleep 1"}
                        });
                        if identified {
                            match id_location {
                                "item" => event["item"]["id"] = Value::from("command"),
                                _ => event["item_id"] = Value::from("command"),
                            }
                        }
                        record_traecli_event(
                            &spec,
                            &event.to_string(),
                            &mut stdout,
                            &mut active_items,
                        );
                        assert_eq!(
                            crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot)
                                .unwrap()
                                .phase,
                            expected,
                            "start_identified={start_identified}, {id_location}, {kind}"
                        );
                    }
                    assert!(active_items.phase().is_none());
                }
            }
        }
    }

    #[test]
    fn streamed_overlapping_commands_keep_the_running_phase_until_all_finish() {
        for id_location in ["item", "event", "none"] {
            for terminal_kind in ["item.completed", "item.failed"] {
                let mut spec = test_spec();
                let slot = crate::ai::driver::runtime_ctx::new_subagent_progress_slot();
                spec.phase_slot = Some(slot.clone());
                let mut stdout = String::new();
                let mut active_items = TraeCliActiveItems::default();

                for (kind, id, command, expected_phase) in [
                    ("item.started", "A", "sleep 1", "running command"),
                    ("item.started", "B", "sleep 2", "running command"),
                    (terminal_kind, "A", "sleep 1", "running command"),
                    (terminal_kind, "B", "sleep 2", "processing output"),
                ] {
                    let mut event = serde_json::json!({
                        "type": kind,
                        "item": {
                            "type": "command_execution",
                            "command": command,
                        },
                    });
                    match id_location {
                        "item" => event["item"]["id"] = Value::String(id.to_string()),
                        "event" => event["item_id"] = Value::String(id.to_string()),
                        _ => {}
                    }
                    let event = event.to_string();
                    record_traecli_event(&spec, &event, &mut stdout, &mut active_items);
                    assert_eq!(
                        crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot)
                            .unwrap()
                            .phase,
                        expected_phase,
                        "{id_location}: {event}"
                    );
                    if id_location != "none" && id == "A" {
                        // Repeated events with an id must not double-count a
                        // start or clear another command on duplicate completion.
                        record_traecli_event(&spec, &event, &mut stdout, &mut active_items);
                        assert_eq!(
                            crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot)
                                .unwrap()
                                .phase,
                            expected_phase,
                            "duplicate {id_location}: {event}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn streamed_generic_events_keep_activity_distinct_from_phase() {
        let mut spec = test_spec();
        let slot = crate::ai::driver::runtime_ctx::new_subagent_progress_slot();
        spec.phase_slot = Some(slot.clone());
        let mut stdout = String::new();
        let mut active_items = TraeCliActiveItems::default();

        for (event, expected) in [
            (r#"{"type":"turn.started"}"#, "turn started"),
            (r#"{"type":"item.started"}"#, "item started"),
            (
                r#"{"type":"item.started","item":{"type":"future_item"}}"#,
                "item: future_item",
            ),
            (r#"{"type":"future.event"}"#, "event: future.event"),
            (r#"{"type":"error"}"#, "error event"),
        ] {
            record_traecli_event(&spec, event, &mut stdout, &mut active_items);
            let progress = crate::ai::driver::runtime_ctx::subagent_progress_snapshot(&slot)
                .unwrap_or_default();
            assert!(
                progress.starts_with(expected),
                "event {event} produced progress {progress:?}"
            );
            let last_activity = progress
                .find(" · last activity ")
                .unwrap_or_else(|| panic!("event {event} produced {progress:?}"));
            let elapsed = progress
                .find(" · elapsed ")
                .unwrap_or_else(|| panic!("event {event} produced {progress:?}"));
            assert!(
                last_activity < elapsed,
                "event {event} produced legacy ordering {progress:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn startup_watchdog_fails_a_silent_backend_quickly() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "traecli-startup-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("silent.sh");
        std::fs::write(&script, "#!/bin/sh\nexec sleep 30\n").unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).unwrap();

        let started = Instant::now();
        let err = run_traecli_blocking_with(
            &test_spec(),
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_millis(150),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.contains("did not emit a startup event"), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "startup watchdog took {:?}",
            started.elapsed()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn startup_watchdog_stops_after_the_first_event() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "traecli-started-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("started.sh");
        std::fs::write(
            &script,
            r#"#!/bin/sh
/bin/echo '{"type":"thread.started","thread_id":"t1"}'
sleep 2.5
/bin/echo '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}'
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).unwrap();

        let started = Instant::now();
        let outcome = run_traecli_blocking_with(
            &test_spec(),
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_secs(2),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(outcome.result, "ok");
        assert!(
            started.elapsed() >= Duration::from_millis(2_300),
            "the watchdog killed a backend that had already started"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn inherited_stdio_does_not_hang_after_child_exit() {
        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "traecli-inherited-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("inherited.sh");
        std::fs::write(
            &script,
            r#"#!/bin/sh
/bin/echo '{"type":"thread.started","thread_id":"t1"}'
(sleep 2) &
/bin/echo '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}'
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).unwrap();

        let started = Instant::now();
        let outcome = run_traecli_blocking_with(
            &test_spec(),
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_secs(1),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(outcome.result, "ok");
        assert!(
            started.elapsed() < Duration::from_millis(1_500),
            "runner waited for a descendant holding inherited stdio"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_args_use_exec_json_and_workspace_sandbox() {
        let args = build_traecli_args(&test_spec());
        assert_eq!(&args[0..3], &["exec", "--json", "--skip-git-repo-check"]);
        assert!(!args.iter().any(|a| a == "--ephemeral"));
        assert!(args.contains(&"-C".to_string()));
        assert!(args.contains(&"/tmp".to_string()));
        assert!(args.contains(&"workspace-write".to_string()));
        assert!(!args.iter().any(|a| a == "-y" || a == "--yolo"));
        assert!(!args.iter().any(|a| a == "--permission-mode"));
        assert!(!args.iter().any(|a| a == "-p" || a == "--output-format"));
        assert!(!args.iter().any(|a| a == "do work"));
    }

    #[test]
    fn build_args_stay_read_only_without_workspace() {
        let mut spec = test_spec();
        spec.workspace_allowed = false;
        let args = build_traecli_args(&spec);
        assert!(args.contains(&"read-only".to_string()));
        assert!(!args.contains(&"workspace-write".to_string()));
    }

    #[test]
    fn build_args_pass_explicit_model() {
        let mut spec = test_spec();
        spec.model = Some("Seed-Code".to_string());
        let args = build_traecli_args(&spec);
        let pos = args.iter().position(|a| a == "-m").expect("-m flag");
        assert_eq!(args[pos + 1], "Seed-Code");
    }

    #[test]
    fn build_args_always_pass_explicit_model() {
        let args = build_traecli_args(&test_spec());
        let pos = args.iter().position(|a| a == "-m").expect("-m flag");
        assert!(!args[pos + 1].trim().is_empty());
    }

    #[test]
    fn build_args_pass_reasoning_effort() {
        let mut spec = test_spec();
        for effort in ["low", "medium", "high", "xhigh", "max"] {
            spec.reasoning_effort = Some(effort.to_string());
            let args = build_traecli_args(&spec);
            let pos = args.iter().position(|a| a == "-c").expect("-c flag");
            assert_eq!(normalize_traecli_reasoning_effort(effort), Some(effort));
            assert_eq!(args[pos + 1], format!("model_reasoning_effort={effort}"));
        }
    }

    #[test]
    fn build_args_clamp_none_effort_to_low() {
        let mut spec = test_spec();
        for effort in ["none", "minimal", " minimal "] {
            spec.reasoning_effort = Some(effort.to_string());
            let args = build_traecli_args(&spec);
            assert_eq!(normalize_traecli_reasoning_effort(effort), Some("low"));
            assert!(args.iter().any(|a| a == "model_reasoning_effort=low"));
        }
        for effort in ["unknown", "off", "", " "] {
            spec.reasoning_effort = Some(effort.to_string());
            let args = build_traecli_args(&spec);
            assert_eq!(normalize_traecli_reasoning_effort(effort), None);
            assert!(
                !args
                    .iter()
                    .any(|arg| arg.starts_with("model_reasoning_effort="))
            );
        }
    }

    #[test]
    fn build_args_attach_original_images() {
        let mut spec = test_spec();
        spec.image_files = vec![
            "/tmp/original-a.png".to_string(),
            "/tmp/original-b.png".to_string(),
        ];
        let args = build_traecli_args(&spec);
        let positions = args
            .iter()
            .enumerate()
            .filter(|(_, arg)| arg.as_str() == "-i")
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(positions.len(), 2);
        assert_eq!(args[positions[0] + 1], "/tmp/original-a.png");
        assert_eq!(args[positions[1] + 1], "/tmp/original-b.png");
    }

    #[test]
    fn resume_prompt_keeps_original_image_paths() {
        let mut spec = test_spec();
        spec.resume_session = Some("session-id".to_string());
        spec.prompt = "follow up".to_string();
        spec.image_files = vec!["/tmp/original.png".to_string()];
        let payload = prompt_payload(&spec);
        assert!(payload.contains("follow up"));
        assert!(payload.contains("/tmp/original.png"));
    }

    #[test]
    fn resume_args_pass_reasoning_effort() {
        let mut spec = test_spec();
        spec.resume_session = Some("01a1a072-105d-7f60-b854-b7746a8ee3fb".to_string());
        spec.reasoning_effort = Some("xhigh".to_string());
        let args = build_traecli_args(&spec);
        assert_eq!(args[0], "exec");
        assert_eq!(args[1], "resume");
        assert!(args.iter().any(|a| a == "model_reasoning_effort=xhigh"));
    }

    #[test]
    fn resolve_model_prefers_explicit_then_env_then_config_then_builtin() {
        assert_eq!(resolve_traecli_model(Some("M"), Some("E"), Some("C")), "M");
        assert_eq!(resolve_traecli_model(None, Some("E"), Some("C")), "E");
        assert_eq!(resolve_traecli_model(None, None, Some("C")), "C");
        assert_eq!(
            resolve_traecli_model(Some("  "), Some(""), Some(" ")),
            TRAECLI_BUILTIN_DEFAULT_MODEL
        );
        assert_eq!(
            resolve_traecli_model(None, None, None),
            TRAECLI_BUILTIN_DEFAULT_MODEL
        );
    }

    #[test]
    fn binary_preflight_rejects_absent_paths() {
        assert!(binary_available("/bin/sh"));
        assert!(!binary_available("/definitely/not/a/traecli-binary"));
    }

    #[test]
    fn prompt_payload_prepends_skills_for_fresh_runs() {
        let mut spec = test_spec();
        spec.skills = Some("<parent_skills>follow me</parent_skills>".to_string());
        let prompt = prompt_payload(&spec);
        assert!(prompt.starts_with("<parent_skills>"));
        assert!(prompt.ends_with("do work"));
        assert!(!build_traecli_args(&spec).iter().any(|a| a == "do work"));
    }

    #[test]
    fn build_args_resume_omits_working_root_and_skills() {
        let mut spec = test_spec();
        spec.skills = Some("<parent_skills>follow me</parent_skills>".to_string());
        spec.resume_session = Some("01a1a036-538a-7003-8c48-7c0d1bef8fdd".to_string());
        let args = build_traecli_args(&spec);
        assert_eq!(
            &args[0..6],
            &[
                "exec",
                "resume",
                "--json",
                "--skip-git-repo-check",
                "--all",
                "01a1a036-538a-7003-8c48-7c0d1bef8fdd"
            ]
        );
        assert!(!args.iter().any(|a| a == "-C" || a == "-s"));
        assert!(!args.iter().any(|a| a == "do work"));
        assert_eq!(prompt_payload(&spec), "do work");
    }

    #[test]
    fn continue_target_resolves_recorded_task_or_uuid_shape() {
        let slot = track_traecli_session("test-task-continue");
        capture_traecli_session(
            &slot,
            r#"{"type":"thread.started","thread_id":"aaaaaaaa-1111-2222-3333-444444444444"}"#,
        );
        assert!(resolve_continue_target("test-task-continue").is_err());
        assert_eq!(
            record_traecli_session("test-task-continue", None).as_deref(),
            Some("aaaaaaaa-1111-2222-3333-444444444444")
        );
        assert_eq!(
            resolve_continue_target("test-task-continue").unwrap(),
            "aaaaaaaa-1111-2222-3333-444444444444"
        );
        let raw_session = "bbbbbbbb-1111-2222-3333-444444444444";
        let _ = track_traecli_session(raw_session);
        assert!(resolve_continue_target(raw_session).is_err());
        assert!(record_traecli_session(raw_session, None).is_none());
        assert_eq!(resolve_continue_target(raw_session).unwrap(), raw_session);
        assert!(resolve_continue_target("no-such-task").is_err());
        assert!(resolve_continue_target("").is_err());
        let error = resolve_continue_target("no-such-task").unwrap_err();
        assert!(error.contains("terminal traecli task_id"), "{error}");
        assert!(error.contains("terminal backend 'traecli' run"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn traecli_side_note_queue_then_resume_prepends_notes() {
        use crate::ai::driver::side_note::{
            drain_side_notes, pending_side_note_count, push_side_note,
        };

        let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
        let root = std::env::temp_dir().join(format!(
            "traecli-side-note-resume-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let history = root.join("lead.sqlite");
        let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
        let session = uuid::Uuid::new_v4().to_string();
        let _slot = track_traecli_session(&task_id);
        push_side_note(&history, "Keep parser behavior.", "lead", Some(&task_id)).unwrap();
        push_side_note(
            &history,
            "Use \"quoted\" names.\nPreserve ordering.",
            "lead",
            Some(&task_id),
        )
        .unwrap();
        push_side_note(&history, "Foreground only", "user", None).unwrap();
        push_side_note(&history, "Other task", "lead", Some("other-task")).unwrap();
        assert!(prepare_traecli_continuation(Some(&task_id), &history, "do work").is_err());
        assert_eq!(pending_side_note_count(&history, Some(&task_id)), 2);
        let fresh = prepare_traecli_continuation(None, &history, "do work").unwrap();
        assert_eq!(fresh, (None, "do work".to_string(), Vec::new()));
        assert_eq!(pending_side_note_count(&history, Some(&task_id)), 2);

        record_traecli_session(&task_id, Some(session.clone()));
        let (resume_session, prompt, side_note_files) =
            prepare_traecli_continuation(Some(&task_id), &history, "do work").unwrap();
        assert_eq!(resume_session.as_deref(), Some(session.as_str()));
        assert_eq!(
            prompt,
            "Main-agent follow-ups:\n- \"Keep parser behavior.\"\n- \"Use \\\"quoted\\\" names.\\nPreserve ordering.\"\n\ndo work"
        );
        assert_eq!(pending_side_note_count(&history, Some(&task_id)), 2);

        let script = terminal_test_script(
            r#"cat > "$0.prompt"
/bin/echo '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}'"#,
        );
        let mut spec = test_spec();
        spec.model = Some("test-model".into());
        spec.resume_session = resume_session;
        spec.side_note_files = side_note_files;
        spec.prompt = prompt.clone();
        spec.skills = Some("Already inherited skills".into());
        assert_eq!(&build_traecli_args(&spec)[..2], &["exec", "resume"]);
        let spawn_error = run_traecli_blocking_with(
            &spec,
            Arc::new(AtomicBool::new(false)),
            root.join("missing-traecli").to_str().unwrap(),
            Duration::from_secs(2),
            Vec::new(),
        )
        .unwrap_err();
        assert!(
            spawn_error.contains("failed to spawn traecli"),
            "{spawn_error}"
        );
        assert_eq!(pending_side_note_count(&history, Some(&task_id)), 2);
        let outcome = run_traecli_blocking_with(
            &spec,
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_secs(2),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(outcome.result, "ok");
        assert_eq!(pending_side_note_count(&history, Some(&task_id)), 0);
        let prompt_file = PathBuf::from(format!("{}.prompt", script.display()));
        assert_eq!(std::fs::read_to_string(&prompt_file).unwrap(), prompt);
        assert_eq!(
            prepare_traecli_continuation(Some(&task_id), &history, "next").unwrap(),
            (Some(session.clone()), "next".to_string(), Vec::new())
        );

        push_side_note(&history, "Resume by UUID", "lead", Some(&task_id)).unwrap();
        let (resumed, prompt, side_note_files) =
            prepare_traecli_continuation(Some(&session), &history, "next").unwrap();
        assert_eq!(resumed.as_deref(), Some(session.as_str()));
        assert_eq!(
            prompt,
            "Main-agent follow-ups:\n- \"Resume by UUID\"\n\nnext"
        );
        assert_eq!(pending_side_note_count(&history, Some(&task_id)), 1);
        spec.resume_session = resumed;
        spec.prompt = prompt.clone();
        spec.side_note_files = side_note_files;
        run_traecli_blocking_with(
            &spec,
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_secs(2),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(std::fs::read_to_string(&prompt_file).unwrap(), prompt);
        assert_eq!(pending_side_note_count(&history, Some(&task_id)), 0);
        assert_eq!(
            drain_side_notes(&history, None)[0].content,
            "Foreground only"
        );
        assert_eq!(
            drain_side_notes(&history, Some("other-task"))[0].content,
            "Other task"
        );
        forget_traecli_session(&task_id);
        std::fs::remove_file(script).unwrap();
        std::fs::remove_file(prompt_file).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parse_exec_json_keeps_last_agent_message() {
        let output = r#"{"type":"thread.started","thread_id":"t1"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"first"}}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"DONE"}}
{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20}}"#;
        let outcome = parse_stream_json_output(output).expect("outcome");
        assert_eq!(outcome.result, "DONE");
        assert!(!outcome.is_error);
        assert_eq!(outcome.session_id.as_deref(), Some("t1"));
        assert_eq!(outcome.num_turns, Some(1));
        assert_eq!(outcome.input_tokens, Some(100));
        assert_eq!(outcome.output_tokens, Some(20));
    }

    #[test]
    fn parse_exec_json_ignores_reasoning_and_command_items() {
        let output = r#"{"type":"item.completed","item":{"id":"item_0","type":"reasoning","text":"thinking"}}
{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"sed x","aggregated_output":"y","exit_code":0,"status":"completed"}}
{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"answer"}}
{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}"#;
        let outcome = parse_stream_json_output(output).expect("outcome");
        assert_eq!(outcome.result, "answer");
        assert!(!outcome.is_error);
    }

    #[test]
    fn parse_exec_json_returns_none_without_events() {
        let output = "not json\n{\"type\":\"turn.started\"}\n{\"type\":\"item.started\"}";
        assert!(parse_stream_json_output(output).is_none());
    }

    #[test]
    fn parse_exec_json_captures_error_event() {
        let output = r#"{"type":"error","message":"boom"}"#;
        let outcome = parse_stream_json_output(output).expect("outcome");
        assert!(outcome.is_error);
        assert_eq!(outcome.result, "boom");
    }
}
