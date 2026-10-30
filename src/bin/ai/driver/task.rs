// Host-side multi-worker task protocol.
//
// The host (coordinator) splits a goal into per-worker assignments, collects
// worker reports from its own talk inbox, and either re-assigns (rework) or
// closes the room with a summary. Workers participate with `talk_send` back
// to the host; absent (inactive) workers are woken by re-executing this
// binary headlessly (`a --session <worker> "<brief>"`), whose report comes
// back through the same inbox — the host never parses child-process output.
//
// Shared-file rules mirror the talk mailbox and the discuss protocol: the
// host appends to worker inboxes and its own room log/state only. A worker's
// sqlite database is written solely by that worker's own (possibly woken)
// process, and a wake spawn is refused while a live process owns the worker
// session, so two writers never interleave rows in one sqlite file.
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::ai::{
    driver::{
        discuss::{self, CollectOutcome},
        talk::{self, TalkMessage},
    },
    history::SessionStore,
};

/// Worker cap: more parallel workers add coordination cost faster than signal.
pub(crate) const MAX_TASK_WORKERS: usize = 8;
/// Round cap: bounds model cost when workers never converge on acceptance.
pub(crate) const MAX_TASK_ROUNDS: u32 = 10;
pub(crate) const DEFAULT_TASK_ROUNDS: u32 = 5;
/// `task_collect` wait window, clamped so one tool call stays bounded.
pub(crate) const DEFAULT_COLLECT_TIMEOUT_SECS: u64 = 60;
pub(crate) const MIN_COLLECT_TIMEOUT_SECS: u64 = 5;
pub(crate) const MAX_COLLECT_TIMEOUT_SECS: u64 = 600;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) enum TaskStatus {
    #[default]
    Open,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) enum AssignmentStatus {
    #[default]
    Pending,
    Working,
    Done,
    Failed,
    /// Worker is stuck on something only the host can decide and is waiting
    /// for an answer; set from a `STATUS: blocked` report line.
    Blocked,
}

impl AssignmentStatus {
    fn as_str(&self) -> &'static str {
        match self {
            AssignmentStatus::Pending => "pending",
            AssignmentStatus::Working => "working",
            AssignmentStatus::Done => "done",
            AssignmentStatus::Failed => "failed",
            AssignmentStatus::Blocked => "blocked",
        }
    }
}

/// One worker's slice of the goal: the files it may touch and the bar its
/// report must clear. Parallel to `TaskState.workers`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct Assignment {
    pub worker: String,
    #[serde(default)]
    pub scope_files: Vec<String>,
    #[serde(default)]
    pub acceptance: String,
    #[serde(default)]
    pub status: AssignmentStatus,
}

/// Host-side room state, persisted beside the room transcript at
/// `<host>.assets/talk/rooms/<room>.state.json` (shared with discuss rooms).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TaskState {
    pub room: String,
    pub goal: String,
    pub host: String,
    pub workers: Vec<String>,
    #[serde(default)]
    pub assignments: Vec<Assignment>,
    /// Workers auto-created by this room (subset of `workers`). Their sqlite
    /// databases are created by their own woken processes, never by the host;
    /// this list is the host's ownership record, alongside the room transcript.
    #[serde(default)]
    pub spawned: Vec<String>,
    /// Skills the `/task` command propagated to this room's workers. Rendered
    /// into every task card as `@skills:` mentions so each wake (headless
    /// re-exec or live inbox card) force-activates them. `#[serde(default)]`
    /// keeps rooms started before skill propagation loadable.
    #[serde(default)]
    pub skills: Vec<String>,
    pub max_rounds: u32,
    /// Current round: 1 right after start. Rework assignments bump it; the
    /// room refuses further rework once it reaches `max_rounds`.
    pub round: u32,
    pub status: TaskStatus,
    /// Workers already woken for `woken_round`; reset on each re-assignment.
    pub woken_round: u32,
    pub woken: Vec<String>,
}

/// `task_start` arguments: either explicit `workers` or `worker_count` for
/// host-owned auto-spawned workers — never both, never neither.
#[derive(Debug, Clone, Default)]
pub(crate) struct StartRequest {
    pub goal: String,
    pub workers: Vec<String>,
    pub worker_count: Option<usize>,
    pub max_rounds: Option<u32>,
    pub room: Option<String>,
}

#[derive(Debug)]
pub(crate) struct StartOutcome {
    pub state: TaskState,
    /// Per worker: (worker id, live?, delivery note).
    pub deliveries: Vec<(String, bool, String)>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn host_id(from_history_file: &Path) -> String {
    from_history_file
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("default")
        .to_string()
}

fn sessions_root_for(history_file: &Path) -> PathBuf {
    history_file
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub(crate) fn load_state(history_file: &Path, room: &str) -> Result<TaskState, String> {
    let file = talk::talk_state_file(history_file, room);
    let content =
        fs::read_to_string(&file).map_err(|_| format!("no task room '{room}' (state file missing)"))?;
    let state = serde_json::from_str::<TaskState>(&content)
        .map_err(|e| format!("task room '{room}' state is corrupt: {e}"))?;
    if state.goal.is_empty() {
        return Err(format!("room '{room}' is not a task room (no goal recorded)"));
    }
    Ok(state)
}

pub(crate) fn save_state(history_file: &Path, state: &TaskState) -> Result<(), String> {
    let file = talk::talk_state_file(history_file, &state.room);
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create task state dir: {e}"))?;
    }
    let content =
        serde_json::to_string_pretty(state).map_err(|e| format!("encode task state: {e}"))?;
    fs::write(&file, content).map_err(|e| format!("write task state: {e}"))?;
    Ok(())
}

/// Worker-facing task card. Carries everything a worker (headlessly woken
/// one-shot or live session) needs: what the goal is, which files it may
/// touch, what acceptance bar its report must clear, where to send the
/// report, and the `enable_tools` fallback in case `talk_send` is lazy.
pub(crate) fn task_brief_for(state: &TaskState, worker: &str) -> String {
    let assignment = state.assignments.iter().find(|a| a.worker == worker);
    let scope = assignment
        .map(|a| a.scope_files.as_slice())
        .unwrap_or(&[]);
    let scope_text = if scope.is_empty() {
        "(no file boundary assigned yet — confirm scope with the host before editing)".to_string()
    } else {
        scope.join(", ")
    };
    let acceptance = assignment
        .map(|a| a.acceptance.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("(no acceptance bar assigned yet — confirm with the host before reporting done)");
    let skills_lines = if state.skills.is_empty() {
        String::new()
    } else {
        let mentions = state
            .skills
            .iter()
            .map(|s| format!("@skills:{s}"))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{mentions}\nAssigned skills: {}\n", state.skills.join(", "))
    };
    format!(
        "[task room={} round={}/{} host={}]\n{skills_lines}Goal: {}\n\nYou are a worker in this task room. Do ONLY your own assignment; touch only these files: {scope_text}\nAcceptance bar: {acceptance}\n\nSend ONE report with the talk_send tool: to_session=\"{}\" room=\"{}\". The report must start with a STATUS line (`STATUS: done` when the acceptance bar is met, `STATUS: failed` otherwise, `STATUS: blocked` when you are stuck on something only the host can decide), then the list of files you changed, then your test output (summary only, not the full log). A blocked report must put its question on a `QUESTION:` line right after STATUS, followed by what you already tried; then stop and wait for the host's answer instead of guessing — do not ask other workers. If talk_send is not in your tool list, call enable_tools for it first. Do not create new sessions and do not spawn further workers.",
        state.room,
        state.round,
        state.max_rounds,
        state.host,
        state.goal,
        state.host,
        state.room,
    )
}

/// Room-name fragment safe for session ids (1-128 ASCII alnum/`-`/`_`):
/// anything else becomes `-`, repeats collapse, ends trim, 24 chars max.
fn sanitize_room_for_id(room: &str) -> String {
    let mut collapsed = String::with_capacity(room.len());
    for c in room.chars().map(|c| {
        if c.is_ascii_alphanumeric() {
            c
        } else {
            '-'
        }
    }) {
        if c != '-' || !collapsed.ends_with('-') {
            collapsed.push(c);
        }
    }
    let trimmed = collapsed
        .trim_matches('-')
        .chars()
        .take(24)
        .collect::<String>();
    let trimmed = trimmed.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "room".to_string()
    } else {
        trimmed
    }
}

/// Fresh worker session id embedding the room (visible ownership in
/// `session_list`, grouped by the `/ss workers` filter). Retried on
/// collision; the id is only reserved here — the worker's sqlite is created
/// later by its own woken process, never by us.
fn spawn_worker_id(store: &SessionStore, room_part: &str, index: usize) -> Result<String, String> {
    for _ in 0..100 {
        let rand = &uuid::Uuid::new_v4().simple().to_string()[..8];
        let id = format!("task-{room_part}-w{index}-{rand}");
        SessionStore::validate_session_id(&id).map_err(|e| e.to_string())?;
        if !store.session_history_file(&id).is_file() {
            return Ok(id);
        }
    }
    Err("failed to generate a fresh worker session id (too many collisions)".to_string())
}

/// Parse a worker report's STATUS line: the first line starting with
/// `STATUS:` (case-insensitive) decides (`done` / `failed` / `blocked`);
/// anything else leaves the assignment untouched so prose-heavy reports
/// never flip state by accident.
pub(crate) fn parse_worker_status(body: &str) -> Option<AssignmentStatus> {
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.len() < 7 {
            continue;
        }
        let (head, tail) = trimmed.split_at(7);
        if !head.eq_ignore_ascii_case("status:") {
            continue;
        }
        let verdict = tail.trim().to_ascii_lowercase();
        let verdict = verdict.split_whitespace().next().unwrap_or("");
        match verdict {
            "done" => return Some(AssignmentStatus::Done),
            "failed" => return Some(AssignmentStatus::Failed),
            "blocked" => return Some(AssignmentStatus::Blocked),
            _ => return None,
        }
    }
    None
}

fn host_marker(host: &str, room: &str, body: String) -> TalkMessage {
    TalkMessage {
        from_session: host.to_string(),
        body,
        ts: now_secs(),
        room: Some(room.to_string()),
    }
}

/// Record worker ownership beside the worker's assets (never its sqlite):
/// `<worker>.assets/managed.json` marks who spawned it and for which room.
fn write_managed_marker(
    store: &SessionStore,
    worker: &str,
    room: &str,
    host: &str,
) -> Result<(), String> {
    let dir = store.session_assets_dir(worker);
    fs::create_dir_all(&dir).map_err(|e| format!("create worker assets dir: {e}"))?;
    let marker = serde_json::json!({
        "kind": "task",
        "room": room,
        "host": host,
        "created": now_secs(),
    });
    let content =
        serde_json::to_string_pretty(&marker).map_err(|e| format!("encode worker marker: {e}"))?;
    fs::write(dir.join("managed.json"), content)
        .map_err(|e| format!("write worker marker: {e}"))?;
    Ok(())
}

/// Create a room, persist its state, and send the round-1 task card to every
/// worker. Returns per-worker liveness so the host knows who must be woken.
pub(crate) fn start_task(
    store: &SessionStore,
    from_history_file: &Path,
    req: &StartRequest,
    exe_override: Option<&Path>,
) -> Result<StartOutcome, String> {
    let goal = req.goal.trim();
    if goal.is_empty() {
        return Err("task goal must not be empty".to_string());
    }
    let use_explicit = !req.workers.is_empty();
    if use_explicit && req.worker_count.is_some() {
        return Err(format!(
            "pass either workers or worker_count, not both (received {} workers and worker_count={})",
            req.workers.len(),
            req.worker_count.unwrap_or(0)
        ));
    }
    if !use_explicit && req.worker_count.is_none() {
        return Err(format!(
            "task needs workers (an array of existing session ids) or worker_count (an integer 1-{MAX_TASK_WORKERS}); neither was provided"
        ));
    }
    let host = host_id(from_history_file);
    let max_rounds = req
        .max_rounds
        .unwrap_or(DEFAULT_TASK_ROUNDS)
        .clamp(1, MAX_TASK_ROUNDS);
    let room = match req.room.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(r) => r.to_string(),
        None => format!("task-{}", now_secs()),
    };
    if talk::talk_state_file(from_history_file, &room).is_file() {
        return Err(format!(
            "task room '{room}' already exists; pick another room name"
        ));
    }
    let (workers, spawned): (Vec<String>, Vec<String>) = if use_explicit {
        if req.workers.len() > MAX_TASK_WORKERS {
            return Err(format!(
                "task supports at most {MAX_TASK_WORKERS} workers (received {})",
                req.workers.len()
            ));
        }
        let mut seen = std::collections::HashSet::new();
        let mut clean = Vec::with_capacity(req.workers.len());
        for w in &req.workers {
            let w = w.trim();
            SessionStore::validate_session_id(w).map_err(|e| e.to_string())?;
            if w == host {
                return Err("the host session cannot be its own worker".to_string());
            }
            if !seen.insert(w.to_string()) {
                return Err(format!("duplicate worker session '{w}'"));
            }
            if !store.session_history_file(w).is_file() {
                return Err(format!("worker session '{w}' does not exist"));
            }
            clean.push(w.to_string());
        }
        (clean, Vec::new())
    } else {
        let n = req.worker_count.unwrap_or(0);
        if n == 0 || n > MAX_TASK_WORKERS {
            return Err(format!(
                "worker_count must be between 1 and {MAX_TASK_WORKERS} (received {n})"
            ));
        }
        let room_part = sanitize_room_for_id(&room);
        let mut fresh = Vec::with_capacity(n);
        for i in 1..=n {
            fresh.push(spawn_worker_id(store, &room_part, i)?);
        }
        let spawned = fresh.clone();
        (fresh, spawned)
    };
    let assignments = workers
        .iter()
        .map(|w| Assignment {
            worker: w.clone(),
            scope_files: Vec::new(),
            acceptance: String::new(),
            status: AssignmentStatus::Pending,
        })
        .collect();
    // Read the skills snapshot written by the `/task` command without
    // consuming it: any failure below must leave the payload for a retry, and
    // `clear_pending_room_skills` runs only on the success path.
    let room_skills = talk::peek_pending_room_skills(from_history_file);
    let mut state = TaskState {
        room: room.clone(),
        goal: goal.to_string(),
        host: host.clone(),
        workers,
        assignments,
        spawned,
        skills: room_skills,
        max_rounds,
        round: 1,
        status: TaskStatus::Open,
        woken_round: 1,
        woken: Vec::new(),
    };
    let sessions_root = sessions_root_for(from_history_file);
    let mut deliveries = Vec::new();
    for w in state.workers.clone() {
        let brief = task_brief_for(&state, &w);
        if state.spawned.iter().any(|s| s == &w) {
            // Fresh id with no sqlite yet: relaxed push queues the card, then
            // spawn immediately so round 1 is already being processed while the
            // host continues. The worker's sqlite is created by its own woken
            // process, never by us.
            match talk::push_talk(store, from_history_file, &w, Some(&room), &brief, true) {
                Ok(_) => {
                    let marker_note = match write_managed_marker(store, &w, &room, &host) {
                        Ok(()) => "ownership recorded".to_string(),
                        Err(e) => format!("ownership marker failed: {e}"),
                    };
                    match spawn_headless(from_history_file, &mut state, &w, &brief, exe_override)
                    {
                        Ok(note) => deliveries.push((
                            w.clone(),
                            false,
                            format!("auto-created; {marker_note}; {note}"),
                        )),
                        Err(e) => {
                            deliveries.push((w.clone(), false, format!("auto-created but {e}")))
                        }
                    }
                }
                Err(e) => deliveries.push((
                    w.clone(),
                    false,
                    format!("auto-created but task card queue failed: {e}"),
                )),
            }
        } else {
            let live = discuss::is_session_live(&sessions_root, &w);
            match talk::push_talk(store, from_history_file, &w, Some(&room), &brief, false) {
                Ok(_) => deliveries.push((
                    w.clone(),
                    live,
                    if live {
                        "live; task card delivered to inbox".to_string()
                    } else {
                        "inactive; task card queued — run task_collect (wakes it)".to_string()
                    },
                )),
                Err(e) => deliveries.push((w.clone(), live, format!("delivery failed: {e}"))),
            }
        }
    }
    let opening_brief = task_brief_for(&state, state.workers.first().map(String::as_str).unwrap_or(""));
    discuss::append_transcript(
        from_history_file,
        &room,
        &[host_marker(&host, &room, format!("[host opens round 1]\n{opening_brief}"))],
    )
    .map_err(|e| format!("write opening transcript: {e}"))?;
    save_state(from_history_file, &state)?;
    // Room started: the snapshot has done its job and must not leak into a
    // later room started without a fresh `/task` command.
    talk::clear_pending_room_skills(from_history_file);
    Ok(StartOutcome { state, deliveries })
}

/// Detached headless re-execution of this binary for one worker, with
/// stdout/stderr to a per-round wake log for audit. Records the woken
/// bookkeeping on success and returns the core note; callers add their own
/// prefix. `exe_override` is a test hook; production passes `None` to re-exec
/// itself.
///
/// The brief travels behind `--` as one verbatim element, so the woken
/// child's CLI parser never re-interprets brief bytes (leading dashes,
/// quotes, `--key=value`) as options: the turn sees exactly this text.
fn headless_cmd(exe: &Path, worker: &str, brief: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--session").arg(worker).arg("--").arg(brief);
    cmd
}
fn spawn_headless(
    from_history_file: &Path,
    state: &mut TaskState,
    worker: &str,
    brief: &str,
    exe_override: Option<&Path>,
) -> Result<String, String> {
    let log_file = talk::talk_room_file(from_history_file, &state.room).with_extension(format!(
        "wake-{}-r{}.log",
        worker, state.round
    ));
    if let Some(parent) = log_file.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file);
    let exe = exe_override
        .map(Path::to_path_buf)
        .or_else(|| std::env::current_exe().ok());
    match (exe, log) {
        (Some(exe), Ok(log)) => {
            let std_file = log.try_clone().ok();
            let mut cmd = headless_cmd(&exe, worker, brief);
            cmd
                .stdin(std::process::Stdio::null())
                .stdout(log);
            if let Some(f) = std_file {
                cmd.stderr(f);
            } else {
                cmd.stderr(std::process::Stdio::null());
            }
            // Detach the worker into its own process group. Without this it
            // shares the host terminal's foreground group, so a host-side
            // Ctrl+C (SIGINT to that group) kills the workers too. A detached
            // worker — and the command children it spawns itself — never sees
            // terminal SIGINT.
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                cmd.process_group(0);
            }
            match cmd.spawn() {
                Ok(child) => {
                    state.woken_round = state.round;
                    state.woken.push(worker.to_string());
                    Ok(format!(
                        "woken headlessly (pid {}, log {})",
                        child.id(),
                        log_file.display()
                    ))
                }
                Err(e) => Err(format!("headless wake failed to spawn: {e}")),
            }
        }
        _ => Err("headless wake unavailable (no binary or log file)".to_string()),
    }
}

/// Wake one worker for the state's current round: live workers just get the
/// task card pushed (they drain it on their next turn); inactive workers get
/// the card queued plus a detached headless re-execution of this binary whose
/// stdout/stderr goes to a per-round wake log for audit. `exe_override` is a
/// test hook; production passes `None` to re-exec itself.
pub(crate) fn wake_worker(
    store: &SessionStore,
    from_history_file: &Path,
    state: &mut TaskState,
    worker: &str,
    exe_override: Option<&Path>,
) -> String {
    if !state.workers.iter().any(|m| m == worker) {
        return format!("'{worker}' is not a worker of room '{}'", state.room);
    }
    if state.woken_round == state.round && state.woken.iter().any(|m| m == worker) {
        return format!("'{worker}' already woken for round {}", state.round);
    }
    let brief = task_brief_for(state, worker);
    // Spawned workers may have no sqlite yet (their own woken process creates
    // it); everyone else keeps the strict existence check.
    let allow_missing = state.spawned.iter().any(|m| m == worker);
    if let Err(e) = talk::push_talk(store, from_history_file, worker, Some(&state.room), &brief, allow_missing)
        .map(|_| ())
    {
        return format!("wake '{worker}' failed to queue task card: {e}");
    }
    let sessions_root = sessions_root_for(from_history_file);
    if discuss::is_session_live(&sessions_root, worker) {
        state.woken_round = state.round;
        state.woken.push(worker.to_string());
        return format!("'{worker}' is live; round-{} task card queued in its inbox", state.round);
    }
    match spawn_headless(from_history_file, state, worker, &brief, exe_override) {
        Ok(note) => format!("'{worker}' is inactive; {note}"),
        Err(e) => format!("'{worker}' task card queued but {e}"),
    }
}

/// Release the woken marks of workers that stayed silent and own no live
/// process, so the next `task_collect` retries their headless wake instead
/// of hitting the `already woken` guard forever. Live-but-slow workers keep
/// their marks (re-waking them would only queue duplicate cards), and a
/// failed liveness scan keeps every mark: without proof of death we assume
/// the woken process may still answer. State is saved only when a mark was
/// actually released; a save failure must not fail the collect, so it is
/// ignored like the wake-pass save.
pub(crate) fn release_dead_wakes(
    from_history_file: &Path,
    state: &mut TaskState,
    missing: &[String],
) {
    if missing.is_empty() {
        return;
    }
    let root = sessions_root_for(from_history_file);
    let before = state.woken.len();
    state.woken.retain(|m| {
        let stale =
            missing.iter().any(|x| x == m) && !discuss::is_session_live(&root, m);
        !stale
    });
    if state.woken.len() != before {
        let _ = save_state(from_history_file, state);
    }
}

/// First assignment or rework of one worker's slice. The first dispatch
/// (a `Pending` assignment) keeps the current round; every re-dispatch after
/// that is rework and bumps `round`, hard-refused once `round >= max_rounds`
/// so a failing worker terminates instead of burning model calls.
pub(crate) fn assign_task(
    store: &SessionStore,
    from_history_file: &Path,
    state: &mut TaskState,
    worker: &str,
    scope_files: Vec<String>,
    acceptance: &str,
) -> Result<String, String> {
    if state.status != TaskStatus::Open {
        return Err(format!(
            "room '{}' is closed; start a new room with task_start",
            state.room
        ));
    }
    let pos = state
        .workers
        .iter()
        .position(|w| w == worker)
        .ok_or_else(|| format!("'{worker}' is not a worker of room '{}'", state.room))?;
    let acceptance = acceptance.trim();
    if acceptance.is_empty() {
        return Err("acceptance must not be empty (the bar the worker must clear)".to_string());
    }
    let clean_scope: Vec<String> = scope_files
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if clean_scope.is_empty() {
        return Err("scope_files must name at least one file boundary".to_string());
    }
    let rework = state.assignments.get(pos).is_some_and(|a| a.status != AssignmentStatus::Pending);
    if rework {
        if state.round >= state.max_rounds {
            return Err(format!(
                "room '{}' already at round {}/{} (max); run task_close with a summary instead",
                state.room, state.round, state.max_rounds
            ));
        }
        state.round += 1;
        state.woken_round = state.round;
        state.woken.clear();
    }
    let slot = if let Some(slot) = state.assignments.get_mut(pos) {
        slot
    } else {
        state.assignments.push(Assignment {
            worker: worker.to_string(),
            ..Default::default()
        });
        state.assignments.last_mut().expect("just pushed")
    };
    slot.worker = worker.to_string();
    slot.scope_files = clean_scope;
    slot.acceptance = acceptance.to_string();
    slot.status = AssignmentStatus::Working;
    let brief = task_brief_for(state, worker);
    let allow_missing = state.spawned.iter().any(|m| m == worker);
    talk::push_talk(store, from_history_file, worker, Some(&state.room), &brief, allow_missing)
        .map(|_| ())
        .map_err(|e| format!("assign '{worker}' failed to queue task card: {e}"))?;
    discuss::append_transcript(
        from_history_file,
        &state.room,
        &[host_marker(
            &state.host.clone(),
            &state.room,
            format!("[host assigns {worker} round {}]\n{brief}", state.round),
        )],
    )
    .map_err(|e| format!("write assignment transcript: {e}"))?;
    save_state(from_history_file, state)?;
    if rework {
        Ok(format!(
            "'{worker}' reworked for round {}/{} (scope + acceptance updated)",
            state.round, state.max_rounds
        ))
    } else {
        Ok(format!("'{worker}' assigned (scope + acceptance set)"))
    }
}

/// Drain this room's worker reports from the host inbox (requeuing other-room
/// mail untouched), persist them to the transcript incrementally, and fold
/// each report's STATUS line into its assignment. Shares the drain/requeue/
/// incremental-persist semantics of the discuss collect path.
pub(crate) fn collect_task(
    history_file: &Path,
    state: &mut TaskState,
    timeout_secs: u64,
) -> CollectOutcome {
    let out = discuss::collect_round(history_file, &state.room, &state.workers, timeout_secs);
    let mut touched = false;
    for reply in &out.replies {
        if let Some(verdict) = parse_worker_status(&reply.body)
            && let Some(slot) = state.assignments.iter_mut().find(|a| a.worker == reply.from_session)
        {
            slot.status = verdict;
            touched = true;
        }
    }
    if touched {
        let _ = save_state(history_file, state);
    }
    out
}

/// Close the room and record the host's acceptance summary in the transcript.
/// Returns the transcript path (the work record).
pub(crate) fn close_task(
    from_history_file: &Path,
    state: &mut TaskState,
    summary: &str,
) -> Result<PathBuf, String> {
    let summary = summary.trim();
    if summary.is_empty() {
        return Err("closing summary must not be empty".to_string());
    }
    if state.status != TaskStatus::Open {
        return Err(format!("room '{}' is already closed", state.room));
    }
    state.status = TaskStatus::Closed;
    discuss::append_transcript(
        from_history_file,
        &state.room,
        &[host_marker(&state.host.clone(), &state.room, format!("[host closes room]\n{summary}"))],
    )
    .map_err(|e| format!("write closing transcript: {e}"))?;
    save_state(from_history_file, state)?;
    Ok(talk::talk_room_file(from_history_file, &state.room))
}

/// Read-only progress snapshot of one room: round state, per-worker
/// assignment/wake/liveness flags, uncollected room mail still in the host
/// inbox, and the transcript tail. Never drains, wakes, or writes — safe to
/// call any time, including right after a resume that left a `task_collect`
/// behind.
#[derive(Debug, Clone)]
pub(crate) struct WorkerProgress {
    pub worker: String,
    pub scope_files: Vec<String>,
    pub acceptance: String,
    pub assignment: String,
    pub live: bool,
    pub woken_this_round: bool,
    pub replied_this_round: bool,
    pub waiting_in_inbox: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct StatusEvent {
    pub from_session: String,
    pub host_line: bool,
    pub preview: String,
}

#[derive(Debug, Clone)]
pub(crate) struct RoomStatus {
    pub room: String,
    pub goal: String,
    pub status: TaskStatus,
    pub round: u32,
    pub max_rounds: u32,
    pub workers: Vec<WorkerProgress>,
    pub uncollected: usize,
    pub recent: Vec<StatusEvent>,
    pub transcript_path: PathBuf,
}

/// Rooms this host owns that hold task state (state files under `rooms/`
/// that decode as task rooms). Best-effort listing for error hints:
/// unreadable or non-task entries are skipped, never fatal.
pub(crate) fn list_rooms(history_file: &Path) -> Vec<String> {
    let mut rooms = Vec::new();
    if let Ok(entries) = fs::read_dir(talk::talk_rooms_dir(history_file)) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(room) = name.strip_suffix(".state.json") {
                let file = talk::talk_state_file(history_file, &room);
                if let Ok(content) = fs::read_to_string(&file)
                    && serde_json::from_str::<TaskState>(&content)
                        .is_ok_and(|st| !st.goal.is_empty())
                {
                    rooms.push(room.to_string());
                }
            }
        }
    }
    rooms.sort();
    rooms
}

/// One-line preview for status views: whitespace flattened and capped so a
/// long report cannot flood the host turn.
fn preview_body(body: &str) -> String {
    const CAP: usize = 160;
    let flat: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= CAP {
        flat
    } else {
        let cut: String = flat.chars().take(CAP).collect();
        format!("{cut}…")
    }
}

/// Build the progress snapshot for `room`. Reads the room state, the room
/// transcript, and the host inbox (peeked, never drained); the only files
/// touched are the host's own.
pub(crate) fn room_status(history_file: &Path, room: &str) -> Result<RoomStatus, String> {
    let state = load_state(history_file, room).map_err(|e| {
        let known = list_rooms(history_file);
        if known.is_empty() {
            e
        } else {
            format!("{e} (task rooms owned by this session: {})", known.join(", "))
        }
    })?;
    let mut transcript = Vec::new();
    if let Ok(content) = fs::read_to_string(talk::talk_room_file(history_file, &state.room)) {
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(msg) = serde_json::from_str::<TalkMessage>(trimmed) {
                transcript.push(msg);
            }
        }
    }
    // Current-round window: everything after the host's latest line. Reports
    // are attributed by arrival window, mirroring `collect_task`.
    let window_start = transcript
        .iter()
        .rposition(|m| m.from_session == state.host)
        .map(|i| i + 1)
        .unwrap_or(0);
    let mut replied = std::collections::HashSet::new();
    for m in &transcript[window_start..] {
        replied.insert(m.from_session.as_str());
    }
    let mut waiting = std::collections::HashSet::new();
    let mut uncollected = 0usize;
    for m in talk::peek_talk(history_file) {
        if m.room.as_deref() == Some(state.room.as_str()) {
            uncollected += 1;
            waiting.insert(m.from_session.clone());
        }
    }
    let root = sessions_root_for(history_file);
    let workers = state
        .workers
        .iter()
        .map(|w| {
            let slot = state.assignments.iter().find(|a| a.worker == *w);
            WorkerProgress {
                worker: w.clone(),
                scope_files: slot.map(|a| a.scope_files.clone()).unwrap_or_default(),
                acceptance: slot.map(|a| a.acceptance.clone()).unwrap_or_default(),
                assignment: slot
                    .map(|a| a.status.as_str().to_string())
                    .unwrap_or_else(|| "pending".to_string()),
                live: discuss::is_session_live(&root, w),
                woken_this_round: state.woken_round == state.round
                    && state.woken.iter().any(|x| x == w),
                replied_this_round: replied.contains(w.as_str()),
                waiting_in_inbox: waiting.contains(w),
            }
        })
        .collect();
    let recent = transcript
        .iter()
        .rev()
        .take(3)
        .rev()
        .map(|m| StatusEvent {
            from_session: m.from_session.clone(),
            host_line: m.from_session == state.host,
            preview: preview_body(&m.body),
        })
        .collect();
    Ok(RoomStatus {
        room: state.room.clone(),
        goal: state.goal.clone(),
        status: state.status,
        round: state.round,
        max_rounds: state.max_rounds,
        workers,
        uncollected,
        recent,
        transcript_path: talk::talk_room_file(history_file, &state.room),
    })
}

/// Render a [`RoomStatus`] as the host-visible progress report. Shared by the
/// `task_room_status` tool and the `/task status` local command so both surfaces
/// read identically. Pure formatting: reads nothing, writes nothing.
pub(crate) fn format_room_status(st: &RoomStatus) -> String {
    let mut s = format!(
        "Task room '{}' — {}\nStatus: {}, round {}/{}.\n",
        st.room,
        st.goal,
        if st.status == TaskStatus::Open {
            "open"
        } else {
            "closed"
        },
        st.round,
        st.max_rounds
    );
    for w in &st.workers {
        let mut flags = vec![if w.live { "live" } else { "inactive" }];
        if w.woken_this_round {
            flags.push("woken");
        }
        flags.push(if w.replied_this_round {
            "replied"
        } else {
            "waiting"
        });
        if w.waiting_in_inbox {
            flags.push("mail in inbox");
        }
        let scope = if w.scope_files.is_empty() {
            "unassigned".to_string()
        } else {
            w.scope_files.join(", ")
        };
        s.push_str(&format!(
            "- {} [{} | scope: {scope}] ({})\n",
            w.worker,
            w.assignment,
            flags.join(", ")
        ));
    }
    s.push_str(&format!(
        "Uncollected room mail in host inbox: {}.\n",
        st.uncollected
    ));
    if st.recent.is_empty() {
        s.push_str("Transcript is empty.\n");
    } else {
        s.push_str("Recent activity (oldest first):\n");
        for e in &st.recent {
            let kind = if e.host_line { "host" } else { "report" };
            s.push_str(&format!(
                "- [{kind} from={}] {}\n",
                e.from_session, e.preview
            ));
        }
    }
    s.push_str(&format!(
        "Full transcript at {}. ",
        st.transcript_path.display()
    ));
    if st.status != TaskStatus::Open {
        s.push_str("Room is closed; nothing further to run.");
    } else if st.uncollected > 0 {
        s.push_str("Next: run task_collect to gather the waiting reports.");
    } else if st.workers.iter().all(|w| w.replied_this_round) {
        if st.workers.iter().any(|w| w.assignment == "blocked") {
            if st.round >= st.max_rounds {
                s.push_str("Some workers are blocked with questions (see transcript), but this was the final round: task_assign refuses rework at the round cap. Record each open question and the decision you would have sent in your task_close summary instead.");
            } else {
                s.push_str("Some workers are blocked with questions (see transcript): answer by folding your decision into their scope/acceptance, then run task_assign for them.");
            }
        } else if st.round >= st.max_rounds {
            s.push_str(
                "All workers reported on the final round: run task_close with an acceptance summary.",
            );
        } else {
            s.push_str("All workers reported: review against acceptance, then run task_assign for rework or task_close if accepted.");
        }
    } else {
        s.push_str("Next: run task_collect to wait for and gather reports (it wakes inactive workers headlessly).");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The headless wake command carries the brief behind `--` as one
    /// verbatim element, and the child CLI parses it back byte-identical:
    /// hostile bytes (leading dashes, quotes, `--opt`) stay data, never
    /// become child options.
    #[test]
    fn headless_cmd_keeps_brief_verbatim_behind_terminator() {
        let brief = "--model=x \"quoted\" -m tail";
        let cmd = headless_cmd(Path::new("a"), "worker-1", brief);
        let args: Vec<String> = cmd
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args, vec!["--session", "worker-1", "--", brief]);
        let mut argv = vec!["a".to_string()];
        argv.extend(args);
        let cli = crate::ai::cli::parse_cli_args(argv.into_iter());
        assert_eq!(cli.session.as_deref(), Some("worker-1"));
        assert_eq!(cli.args, vec![brief.to_string()]);
    }

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("task_test_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Host plus two worker sessions, all backed by stub sqlite files.
    fn fixture() -> (PathBuf, SessionStore, PathBuf, PathBuf, PathBuf) {
        let dir = temp_dir();
        let store = SessionStore::new(&dir.join("hist.sqlite"));
        let host_hist = store.session_history_file("host-sess");
        let w1_hist = store.session_history_file("worker-1");
        let w2_hist = store.session_history_file("worker-2");
        for hist in [&host_hist, &w1_hist, &w2_hist] {
            std::fs::create_dir_all(hist.parent().unwrap()).unwrap();
            std::fs::write(hist, b"stub").unwrap();
        }
        (dir, store, host_hist, w1_hist, w2_hist)
    }

    fn workers() -> Vec<String> {
        vec!["worker-1".to_string(), "worker-2".to_string()]
    }

    /// Positional wrapper keeping the explicit-workers call shape.
    fn start(
        store: &SessionStore,
        host_hist: &Path,
        goal: &str,
        workers: &[String],
        max_rounds: Option<u32>,
        room: Option<&str>,
    ) -> Result<StartOutcome, String> {
        start_task(
            store,
            host_hist,
            &StartRequest {
                goal: goal.to_string(),
                workers: workers.to_vec(),
                worker_count: None,
                max_rounds,
                room: room.map(str::to_string),
            },
            None,
        )
    }

    #[test]
    fn start_validates_input_and_sends_task_cards() {
        let (dir, store, host_hist, w1_hist, _w2) = fixture();
        assert!(start(&store, &host_hist, "  ", &workers(), None, None).is_err());
        assert!(start(&store, &host_hist, "g", &[], None, None).is_err());
        assert!(
            start(
                &store,
                &host_hist,
                "g",
                &["host-sess".to_string()],
                None,
                None
            )
            .is_err()
        );
        assert!(
            start(&store, &host_hist, "g", &["ghost".to_string()], None, None).is_err()
        );
        assert!(
            start(
                &store,
                &host_hist,
                "g",
                &["worker-1".to_string(), "worker-1".to_string()],
                None,
                None
            )
            .is_err()
        );

        let out = start(&store, &host_hist, "fix login", &workers(), Some(2), Some("t1")).unwrap();
        assert_eq!(out.state.round, 1);
        assert_eq!(out.state.max_rounds, 2);
        assert_eq!(out.deliveries.len(), 2);
        assert!(out.deliveries.iter().all(|(_, live, _)| !live));
        // Round-1 task card reached the worker inbox with the room tag.
        let msgs = talk::drain_talk(&w1_hist);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].room.as_deref(), Some("t1"));
        assert!(msgs[0].body.contains("fix login"));
        // State persisted and reloads.
        let reloaded = load_state(&host_hist, "t1").unwrap();
        assert_eq!(reloaded.goal, "fix login");
        assert_eq!(reloaded.workers, workers());
        // Reusing an open room name is refused.
        assert!(start(&store, &host_hist, "g2", &workers(), None, Some("t1")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn task_brief_carries_file_boundary_acceptance_and_reply_shape() {
        let (dir, store, host_hist, _w1, _w2) = fixture();
        let mut state = start(&store, &host_hist, "fix login", &workers(), None, Some("t-brief"))
            .unwrap()
            .state;
        assign_task(
            &store,
            &host_hist,
            &mut state,
            "worker-1",
            vec!["src/login.rs".to_string()],
            "login test passes",
        )
        .unwrap();
        let brief = task_brief_for(&state, "worker-1");
        assert!(brief.contains("t-brief"), "{brief}");
        assert!(brief.contains("host-sess"), "{brief}");
        assert!(brief.contains("src/login.rs"), "{brief}");
        assert!(brief.contains("login test passes"), "{brief}");
        assert!(brief.contains("STATUS: done"), "{brief}");
        assert!(brief.contains("STATUS: blocked"), "{brief}");
        assert!(brief.contains("QUESTION:"), "{brief}");
        assert!(brief.contains("talk_send"), "{brief}");
        assert!(brief.contains("enable_tools"), "{brief}");
        assert!(brief.contains("Do not create new sessions"), "{brief}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn task_brief_renders_assigned_skills_mentions() {
        let (dir, store, host_hist, _w1, _w2) = fixture();
        let mut state = start(&store, &host_hist, "fix login", &workers(), None, Some("t-skills"))
            .unwrap()
            .state;
        let brief = task_brief_for(&state, "worker-1");
        assert!(!brief.contains("@skills:"), "no skills yet: {brief}");

        state.skills = vec!["s1".to_string(), "s2".to_string()];
        let brief = task_brief_for(&state, "worker-1");
        assert!(brief.contains("@skills:s1"), "{brief}");
        assert!(brief.contains("@skills:s2"), "{brief}");
        assert!(brief.contains("Assigned skills: s1, s2"), "{brief}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn task_state_round_trips_skills_with_old_state_compat() {
        // State files written before skill propagation lack the field;
        // `#[serde(default)]` must load them with an empty list.
        let old = r#"{"room":"r","goal":"g","host":"h","workers":["w"],"assignments":[],"spawned":[],"max_rounds":2,"round":1,"status":"Open","woken_round":1,"woken":[]}"#;
        let state: TaskState = serde_json::from_str(old).expect("old state must load");
        assert!(state.skills.is_empty());

        let with_skills = TaskState {
            skills: vec!["s1".to_string()],
            ..state
        };
        let encoded = serde_json::to_string(&with_skills).unwrap();
        let decoded: TaskState = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.skills, vec!["s1"]);
    }

    #[test]
    fn start_keeps_snapshot_on_failure_and_clears_it_on_success() {
        let (dir, store, host_hist, _w1_hist, _w2_hist) = fixture();
        // Failed start (invalid input) must leave the payload for a retry.
        assert!(talk::write_pending_room_skills(&host_hist, &["s1".to_string()]).is_ok());
        assert!(start(&store, &host_hist, "   ", &workers(), None, None).is_err());
        assert_eq!(talk::peek_pending_room_skills(&host_hist), vec!["s1"]);

        // Successful start consumes the snapshot: the skills land in state and
        // must not leak into a later room started without a fresh command.
        let out = start(&store, &host_hist, "g", &workers(), None, Some("t-skills-snap"))
            .unwrap();
        assert_eq!(out.state.skills, vec!["s1"]);
        assert!(talk::peek_pending_room_skills(&host_hist).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn worker_status_line_parses_verdicts() {
        assert_eq!(
            parse_worker_status("STATUS: done\nchanged a.rs\ntests ok"),
            Some(AssignmentStatus::Done)
        );
        assert_eq!(
            parse_worker_status("status: FAILED\nnothing worked"),
            Some(AssignmentStatus::Failed)
        );
        assert_eq!(
            parse_worker_status("STATUS: blocked\nQUESTION: which approach?\ntried X"),
            Some(AssignmentStatus::Blocked)
        );
        // Prose without a STATUS line leaves the assignment untouched.
        assert_eq!(parse_worker_status("all done, ship it"), None);
        assert_eq!(parse_worker_status("STATUS: almost done"), None);
        assert_eq!(parse_worker_status(""), None);
    }

    #[test]
    fn collect_folds_status_lines_into_assignments() {
        let (dir, store, host_hist, _w1, w2_hist) = fixture();
        let mut state = start(&store, &host_hist, "g", &workers(), None, Some("t-collect"))
            .unwrap()
            .state;
        assign_task(
            &store,
            &host_hist,
            &mut state,
            "worker-1",
            vec!["a.rs".to_string()],
            "bar one",
        )
        .unwrap();
        assign_task(
            &store,
            &host_hist,
            &mut state,
            "worker-2",
            vec!["b.rs".to_string()],
            "bar two",
        )
        .unwrap();
        talk::push_talk(
            &store,
            &w2_hist,
            "host-sess",
            Some("t-collect"),
            "STATUS: done\nchanged b.rs\ntests pass",
            false,
        )
        .unwrap();
        let out = collect_task(&host_hist, &mut state, 5);
        assert_eq!(out.replies.len(), 1);
        assert_eq!(out.missing, vec!["worker-1".to_string()]);
        let w2 = state.assignments.iter().find(|a| a.worker == "worker-2").unwrap();
        assert_eq!(w2.status, AssignmentStatus::Done);
        let w1 = state.assignments.iter().find(|a| a.worker == "worker-1").unwrap();
        assert_eq!(w1.status, AssignmentStatus::Working);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn collect_folds_blocked_question_into_assignment() {
        let (dir, store, host_hist, w1_hist, _w2) = fixture();
        let mut state = start(&store, &host_hist, "g", &workers(), None, Some("t-blocked"))
            .unwrap()
            .state;
        assign_task(
            &store,
            &host_hist,
            &mut state,
            "worker-1",
            vec!["a.rs".to_string()],
            "bar one",
        )
        .unwrap();
        talk::push_talk(
            &store,
            &w1_hist,
            "host-sess",
            Some("t-blocked"),
            "STATUS: blocked\nQUESTION: which approach?\ntried X and Y",
            false,
        )
        .unwrap();
        let out = collect_task(&host_hist, &mut state, 5);
        assert_eq!(out.replies.len(), 1);
        let w1 = state.assignments.iter().find(|a| a.worker == "worker-1").unwrap();
        assert_eq!(w1.status, AssignmentStatus::Blocked);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn assign_rework_counts_round_and_refuses_at_max() {
        let (dir, store, host_hist, _w1, _w2) = fixture();
        let mut state = start(&store, &host_hist, "g", &workers(), Some(2), Some("t-rework"))
            .unwrap()
            .state;
        // First dispatch keeps round 1.
        assign_task(
            &store,
            &host_hist,
            &mut state,
            "worker-1",
            vec!["a.rs".to_string()],
            "bar",
        )
        .unwrap();
        assert_eq!(state.round, 1);
        // Rework bumps to round 2 (max).
        assign_task(
            &store,
            &host_hist,
            &mut state,
            "worker-1",
            vec!["a.rs".to_string()],
            "bar, stricter",
        )
        .unwrap();
        assert_eq!(state.round, 2);
        // Past max the host must close instead.
        let err = assign_task(
            &store,
            &host_hist,
            &mut state,
            "worker-1",
            vec!["a.rs".to_string()],
            "bar again",
        )
        .unwrap_err();
        assert!(err.contains("task_close"), "{err}");
        // Empty acceptance and empty scope are fail-closed.
        assert!(
            assign_task(&store, &host_hist, &mut state, "worker-2", vec!["b.rs".to_string()], "  ")
                .is_err()
        );
        assert!(
            assign_task(&store, &host_hist, &mut state, "worker-2", Vec::new(), "bar").is_err()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn status_fixture(round: u32, max_rounds: u32, assignment: &str) -> RoomStatus {
        RoomStatus {
            room: "t-room".to_string(),
            goal: "g".to_string(),
            status: TaskStatus::Open,
            round,
            max_rounds,
            workers: vec![WorkerProgress {
                worker: "worker-1".to_string(),
                scope_files: vec!["a.rs".to_string()],
                acceptance: "bar".to_string(),
                assignment: assignment.to_string(),
                live: false,
                woken_this_round: true,
                replied_this_round: true,
                waiting_in_inbox: false,
            }],
            uncollected: 0,
            recent: Vec::new(),
            transcript_path: PathBuf::from("/tmp/t-room.jsonl"),
        }
    }

    #[test]
    fn final_round_blocked_points_at_close_not_assign() {
        // task_assign hard-refuses rework at the round cap, so the status
        // page must not send the host there for a blocked worker.
        let text = format_room_status(&status_fixture(3, 3, "blocked"));
        assert!(text.contains("task_close"), "{text}");
        assert!(!text.contains("run task_assign"), "{text}");
        // Below the cap the answer-through-assign path still applies.
        let text = format_room_status(&status_fixture(2, 3, "blocked"));
        assert!(text.contains("run task_assign"), "{text}");
    }

    #[test]
    fn worker_count_spawns_room_owned_ids_with_markers() {
        let (dir, store, host_hist, _w1, _w2) = fixture();
        let fake_bin = dir.join("fake-a");
        std::fs::write(&fake_bin, "#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(&fake_bin, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();
        let req = StartRequest {
            goal: "ship v2".to_string(),
            worker_count: Some(2),
            room: Some("fix-login".to_string()),
            ..Default::default()
        };
        let out = start_task(&store, &host_hist, &req, Some(&fake_bin)).unwrap();
        assert_eq!(out.state.workers.len(), 2);
        assert_eq!(out.state.spawned, out.state.workers);
        for (i, w) in out.state.workers.iter().enumerate() {
            // Room-owned id shape the `/ss workers` filter groups.
            assert!(w.starts_with("task-fix-login-w"), "{w}");
            assert!(w.contains(&format!("-w{}-", i + 1)), "{w}");
            let rand = w.rsplit('-').next().unwrap_or("");
            assert_eq!(rand.len(), 8, "{w}");
            assert!(rand.bytes().all(|b| b.is_ascii_hexdigit()), "{w}");
            SessionStore::validate_session_id(w).unwrap();
            // The host never creates the worker sqlite: only its own woken
            // process may. The file must still be absent right after start
            // (the fake binary exits without touching it).
            assert!(!store.session_history_file(w).is_file(), "{w}");
            // Ownership marker lives beside worker assets, never its sqlite.
            let marker = store.session_assets_dir(w).join("managed.json");
            assert!(marker.is_file(), "{w}");
            let content = std::fs::read_to_string(&marker).unwrap();
            assert!(content.contains("\"task\""), "{content}");
            assert!(content.contains("fix-login"), "{content}");
            let (_, _, note) = &out.deliveries[i];
            assert!(note.contains("auto-created"), "{note}");
            // Round-1 card (goal text) waits in the fresh inbox.
            let worker_hist = store.session_history_file(w);
            let msgs = talk::drain_talk(&worker_hist);
            assert_eq!(msgs.len(), 1);
            assert!(msgs[0].body.contains("ship v2"), "{}", msgs[0].body);
        }
        // Ownership record survives a state reload.
        let reloaded = load_state(&host_hist, "fix-login").unwrap();
        assert_eq!(reloaded.spawned, out.state.workers);
        // worker_count out of range and both/neither combinations refuse.
        let mut both = req.clone();
        both.workers = workers();
        assert!(start_task(&store, &host_hist, &both, None).is_err());
        let neither = StartRequest {
            goal: "g".to_string(),
            ..Default::default()
        };
        assert!(start_task(&store, &host_hist, &neither, None).is_err());
        let zero = StartRequest {
            goal: "g".to_string(),
            worker_count: Some(0),
            room: Some("r-zero".to_string()),
            ..Default::default()
        };
        assert!(start_task(&store, &host_hist, &zero, None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn close_marks_shut_and_returns_transcript() {
        let (dir, store, host_hist, _w1, _w2) = fixture();
        let mut state = start(&store, &host_hist, "g", &workers(), None, Some("t-close"))
            .unwrap()
            .state;
        assert!(close_task(&host_hist, &mut state, "  ").is_err());
        let path = close_task(&host_hist, &mut state, "accepted: all green").unwrap();
        assert!(path.is_file());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("accepted: all green"));
        assert!(content.contains("host opens round 1"));
        assert_eq!(load_state(&host_hist, "t-close").unwrap().status, TaskStatus::Closed);
        assert!(close_task(&host_hist, &mut state, "again").is_err());
        // Closed rooms refuse further assignments.
        assert!(
            assign_task(
                &store,
                &host_hist,
                &mut state,
                "worker-1",
                vec!["a.rs".to_string()],
                "bar"
            )
            .is_err()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
