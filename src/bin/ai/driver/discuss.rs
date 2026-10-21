// Host-side multi-session discuss/argue protocol.
//
// The host (coordinator) session runs rounds over the talk mailbox: it
// broadcasts a brief, collects member replies from its own inbox, and either
// broadcasts the next round or closes the room. Members participate with
// `talk_send` back to the host; absent (inactive) members are woken by
// re-executing this binary headlessly (`a --session <member> "<brief>"`),
// whose reply comes back through the same inbox — the host never parses
// child-process output.
//
// Shared-file rules mirror the talk mailbox: the host appends to member
// inboxes and its own room log/state only. A member's sqlite database is
// written solely by that member's own (possibly woken) process, and a wake
// spawn is refused while a live process owns the member session, so two
// writers never interleave rows in one sqlite file.
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::ai::{
    driver::talk::{self, TalkMessage},
    history::SessionStore,
};

/// Room member cap: more voices add noise faster than signal per round.
pub(crate) const MAX_DISCUSS_MEMBERS: usize = 8;
/// Round cap: bounds model cost when members never converge.
pub(crate) const MAX_DISCUSS_ROUNDS: u32 = 10;
pub(crate) const DEFAULT_DISCUSS_ROUNDS: u32 = 3;
/// `discuss_collect` wait window, clamped so one tool call stays bounded.
pub(crate) const DEFAULT_COLLECT_TIMEOUT_SECS: u64 = 60;
pub(crate) const MIN_COLLECT_TIMEOUT_SECS: u64 = 5;
pub(crate) const MAX_COLLECT_TIMEOUT_SECS: u64 = 600;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum DiscussStatus {
    Open,
    Closed,
}

/// Host-side room state, persisted beside the room transcript at
/// `<host>.assets/talk/rooms/<room>.state.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DiscussState {
    pub room: String,
    pub topic: String,
    pub host: String,
    pub members: Vec<String>,
    /// Per-member role line, parallel to `members`; empty means no assigned
    /// role. `#[serde(default)]` keeps pre-role room states loadable.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Members auto-created by this room (subset of `members`). Their sqlite
    /// databases are created by their own woken processes, never by the host;
    /// this list is the host's ownership record, alongside the room transcript.
    #[serde(default)]
    pub spawned: Vec<String>,
    pub max_rounds: u32,
    /// Current round: 1 right after start (round-1 brief sent).
    pub round: u32,
    pub status: DiscussStatus,
    /// Members already woken for `woken_round`; reset on each broadcast.
    pub woken_round: u32,
    pub woken: Vec<String>,
}

/// `discuss_start` arguments: either explicit `members` or `member_count` for
/// host-owned auto-spawned members — never both, never neither.
#[derive(Debug, Clone, Default)]
pub(crate) struct StartRequest {
    pub topic: String,
    pub members: Vec<String>,
    pub member_count: Option<usize>,
    pub roles: Vec<String>,
    pub max_rounds: Option<u32>,
    pub room: Option<String>,
}

#[derive(Debug)]
pub(crate) struct StartOutcome {
    pub state: DiscussState,
    /// Per member: (member id, live?, delivery note).
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

/// Whether a live process currently owns `session_id`. Best-effort: when the
/// PID scan fails we report live, so callers fall back to push-only delivery
/// instead of risking a second writer on the member's sqlite file.
pub(crate) fn is_session_live(sessions_root: &Path, session_id: &str) -> bool {
    match super::session_pid::scan_all_session_pids(sessions_root) {
        Ok(live) => live
            .into_iter()
            .any(|(sid, _, alive)| alive && sid == session_id),
        Err(_) => true,
    }
}

pub(crate) fn load_state(history_file: &Path, room: &str) -> Result<DiscussState, String> {
    let file = talk::talk_state_file(history_file, room);
    let content =
        fs::read_to_string(&file).map_err(|_| format!("no discuss room '{room}' (state file missing)"))?;
    Ok(serde_json::from_str::<DiscussState>(&content)
        .map_err(|e| format!("discuss room '{room}' state is corrupt: {e}"))?)
}

pub(crate) fn save_state(history_file: &Path, state: &DiscussState) -> Result<(), String> {
    let file = talk::talk_state_file(history_file, &state.room);
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create discuss state dir: {e}"))?;
    }
    let content =
        serde_json::to_string_pretty(state).map_err(|e| format!("encode discuss state: {e}"))?;
    fs::write(&file, content).map_err(|e| format!("write discuss state: {e}"))?;
    Ok(())
}

/// Member-facing round brief. Carries everything a member (human, auto-talk
/// bot, or headlessly woken one-shot) needs: what the topic is, where to send
/// the reply, and the `enable_tools` fallback in case `talk_send` is lazy in
/// its turn.
///
/// `role` appends an assigned stance. It is empty for explicit members without
/// one; spawned members always carry one (host-provided or the default argue
/// cycle).
pub(crate) fn round_brief_for(state: &DiscussState, role: &str) -> String {
    let base = format!(
        "[discuss room={} round={}/{} host={}]\nTopic: {}\n\nYou are a participant in this discussion. Send ONE reply with the talk_send tool: to_session=\"{}\" room=\"{}\". Keep it to your own viewpoint plus direct responses to others (no pleasantries). If talk_send is not in your tool list, call enable_tools for it first.",
        state.room,
        state.round,
        state.max_rounds,
        state.host,
        state.topic,
        state.host,
        state.room,
    );
    let role = role.trim();
    if role.is_empty() {
        base
    } else {
        format!("{base}\nYour role in this discussion: {role}")
    }
}

/// Role assigned to one member; empty when none was (explicit members, or
/// room states written before roles existed).
fn role_for<'a>(state: &'a DiscussState, member: &str) -> &'a str {
    state
        .members
        .iter()
        .position(|m| m == member)
        .and_then(|i| state.roles.get(i))
        .map(String::as_str)
        .unwrap_or("")
}

/// Default argue stances for auto-spawned members when the host passes no
/// roles: pro / contra / judge, then independent voices. A lone member gets a
/// generic participant brief instead of debating an empty room.
fn default_spawn_role(index: usize, count: usize) -> &'static str {
    if count == 1 {
        return "Participant: give your own in-depth take on the topic.";
    }
    match index {
        0 => "Proponent: argue FOR the topic with the strongest case you can make.",
        1 => "Opponent: argue AGAINST the topic; attack the proponent's points directly.",
        2 => "Judge: probe both sides' logic, then give a reasoned verdict.",
        _ => "Independent voice: bring an angle the others have missed.",
    }
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

/// Fresh member session id embedding the room (visible ownership in
/// `session_list`). Retried on collision; the id is only reserved here — the
/// member's sqlite is created later by its own woken process, never by us.
fn spawn_member_id(store: &SessionStore, room_part: &str, index: usize) -> Result<String, String> {
    for _ in 0..100 {
        let rand = &uuid::Uuid::new_v4().simple().to_string()[..8];
        let id = format!("dc-{room_part}-m{index}-{rand}");
        SessionStore::validate_session_id(&id).map_err(|e| e.to_string())?;
        if !store.session_history_file(&id).is_file() {
            return Ok(id);
        }
    }
    Err("failed to generate a fresh member session id (too many collisions)".to_string())
}

/// Append messages to the host-side room transcript (`rooms/<room>.jsonl`),
/// which is the meeting record. Broadcasts are mirrored there at send time by
/// the caller; collected replies and the closing summary land here.
pub(crate) fn append_transcript(
    history_file: &Path,
    room: &str,
    msgs: &[TalkMessage],
) -> Result<(), String> {
    if msgs.is_empty() {
        return Ok(());
    }
    let file = talk::talk_room_file(history_file, room);
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create talk room dir: {e}"))?;
    }
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file)
        .map_err(|e| format!("open room transcript: {e}"))?;
    for m in msgs {
        let line =
            serde_json::to_string(m).map_err(|e| format!("encode transcript line: {e}"))?;
        writeln!(f, "{line}").map_err(|e| format!("write room transcript: {e}"))?;
    }
    Ok(())
}

fn host_marker(host: &str, room: &str, body: String) -> TalkMessage {
    TalkMessage {
        from_session: host.to_string(),
        body,
        ts: now_secs(),
        room: Some(room.to_string()),
    }
}

/// Create a room, persist its state, and send the round-1 brief to every
/// member. Returns per-member liveness so the host knows who must be woken.
pub(crate) fn start_room(
    store: &SessionStore,
    from_history_file: &Path,
    req: &StartRequest,
    exe_override: Option<&Path>,
) -> Result<StartOutcome, String> {
    let topic = req.topic.trim();
    if topic.is_empty() {
        return Err("discuss topic must not be empty".to_string());
    }
    let use_explicit = !req.members.is_empty();
    if use_explicit && req.member_count.is_some() {
        return Err("pass either members or member_count, not both".to_string());
    }
    if !use_explicit && req.member_count.is_none() {
        return Err("discuss needs members (existing session ids) or member_count (auto-spawned members)"
            .to_string());
    }
    let host = host_id(from_history_file);
    let max_rounds = req
        .max_rounds
        .unwrap_or(DEFAULT_DISCUSS_ROUNDS)
        .clamp(1, MAX_DISCUSS_ROUNDS);
    let room = match req.room.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(r) => r.to_string(),
        None => format!("discuss-{}", now_secs()),
    };
    if talk::talk_state_file(from_history_file, &room).is_file() {
        return Err(format!(
            "discuss room '{room}' already exists; pick another room name"
        ));
    }
    // Resolve members, roles, and the spawned-ownership record.
    let (members, roles, spawned): (Vec<String>, Vec<String>, Vec<String>) = if use_explicit {
        if req.members.len() > MAX_DISCUSS_MEMBERS {
            return Err(format!(
                "discuss supports at most {MAX_DISCUSS_MEMBERS} members"
            ));
        }
        let mut seen = std::collections::HashSet::new();
        let mut clean = Vec::with_capacity(req.members.len());
        for m in &req.members {
            let m = m.trim();
            SessionStore::validate_session_id(m).map_err(|e| e.to_string())?;
            if m == host {
                return Err("the host session cannot be its own member".to_string());
            }
            if !seen.insert(m.to_string()) {
                return Err(format!("duplicate member session '{m}'"));
            }
            if !store.session_history_file(m).is_file() {
                return Err(format!("member session '{m}' does not exist"));
            }
            clean.push(m.to_string());
        }
        let roles = resolve_roles(&req.roles, clean.len())?;
        (clean, roles, Vec::new())
    } else {
        let n = req.member_count.unwrap_or(0);
        if n == 0 || n > MAX_DISCUSS_MEMBERS {
            return Err(format!(
                "member_count must be between 1 and {MAX_DISCUSS_MEMBERS}"
            ));
        }
        let room_part = sanitize_room_for_id(&room);
        let mut fresh = Vec::with_capacity(n);
        for i in 1..=n {
            fresh.push(spawn_member_id(store, &room_part, i)?);
        }
        let roles = if req.roles.is_empty() {
            (0..n)
                .map(|i| default_spawn_role(i, n).to_string())
                .collect()
        } else {
            resolve_roles(&req.roles, n)?
        };
        let spawned = fresh.clone();
        (fresh, roles, spawned)
    };
    let mut state = DiscussState {
        room: room.clone(),
        topic: topic.to_string(),
        host: host.clone(),
        members,
        roles,
        spawned,
        max_rounds,
        round: 1,
        status: DiscussStatus::Open,
        woken_round: 1,
        woken: Vec::new(),
    };
    let opening_brief = round_brief_for(&state, "");
    let sessions_root = sessions_root_for(from_history_file);
    let mut deliveries = Vec::new();
    for (m, role) in state.members.clone().into_iter().zip(state.roles.clone()) {
        let brief = round_brief_for(&state, &role);
        if state.spawned.iter().any(|s| s == &m) {
            // Fresh id with no sqlite yet: relaxed push queues the brief, then
            // spawn immediately so round 1 is already being processed while the
            // host continues. The member's sqlite is created by its own woken
            // process, never by us.
            match talk::push_talk(store, from_history_file, &m, Some(&room), &brief, true) {
                Ok(_) => {
                    match spawn_headless(from_history_file, &mut state, &m, &brief, exe_override)
                    {
                        Ok(note) => deliveries.push((
                            m.clone(),
                            false,
                            format!("auto-created (role: {}); {note}", display_role(&role)),
                        )),
                        Err(e) => {
                            deliveries.push((m.clone(), false, format!("auto-created but {e}")))
                        }
                    }
                }
                Err(e) => deliveries.push((
                    m.clone(),
                    false,
                    format!("auto-created but brief queue failed: {e}"),
                )),
            }
        } else {
            let live = is_session_live(&sessions_root, &m);
            match talk::push_talk(store, from_history_file, &m, Some(&room), &brief, false) {
                Ok(_) => deliveries.push((
                    m.clone(),
                    live,
                    if live {
                        "live; brief delivered to inbox".to_string()
                    } else {
                        "inactive; brief queued — run discuss_collect (wakes it) or discuss_wake"
                            .to_string()
                    },
                )),
                Err(e) => deliveries.push((m.clone(), live, format!("delivery failed: {e}"))),
            }
        }
    }
    // Host marker first so the transcript opens with the round-1 brief.
    append_transcript(
        from_history_file,
        &room,
        &[host_marker(&host, &room, format!("[host opens round 1]\n{opening_brief}") )],
    )
    .map_err(|e| format!("write opening transcript: {e}"))?;
    save_state(from_history_file, &state)?;
    Ok(StartOutcome { state, deliveries })
}

/// Host-provided roles must line up 1:1 with members; empty entries mean "no
/// assigned role".
fn resolve_roles(roles: &[String], count: usize) -> Result<Vec<String>, String> {
    if roles.is_empty() {
        return Ok(vec![String::new(); count]);
    }
    if roles.len() != count {
        return Err(format!(
            "roles has {} entries but there are {count} members; pass one role per member or none",
            roles.len()
        ));
    }
    Ok(roles.iter().map(|r| r.trim().to_string()).collect())
}

/// Display form of a role for host-visible notes.
pub(crate) fn display_role(role: &str) -> &str {
    let role = role.trim();
    if role.is_empty() {
        "default"
    } else {
        role
    }
}

/// Detached headless re-execution of this binary for one member, with
/// stdout/stderr to a per-round wake log for audit. Records the woken
/// bookkeeping on success and returns the core note; callers add their own
/// prefix. `exe_override` is a test hook; production passes `None` to re-exec
/// itself.
fn spawn_headless(
    from_history_file: &Path,
    state: &mut DiscussState,
    member: &str,
    brief: &str,
    exe_override: Option<&Path>,
) -> Result<String, String> {
    let log_file = talk::talk_room_file(from_history_file, &state.room).with_extension(format!(
        "wake-{}-r{}.log",
        member, state.round
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
            let mut cmd = std::process::Command::new(exe);
            cmd.arg("--session")
                .arg(member)
                .arg(brief)
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
                    state.woken.push(member.to_string());
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

/// Wake one member for the state's current round: live members just get the
/// brief pushed (they drain it on their next turn); inactive members get the
/// brief queued plus a detached headless re-execution of this binary whose
/// stdout/stderr goes to a per-round wake log for audit. `exe_override` is a
/// test hook; production passes `None` to re-exec itself.
pub(crate) fn wake_member(
    store: &SessionStore,
    from_history_file: &Path,
    state: &mut DiscussState,
    member: &str,
    exe_override: Option<&Path>,
) -> String {
    if !state.members.iter().any(|m| m == member) {
        return format!("'{member}' is not a member of room '{}'", state.room);
    }
    if state.woken_round == state.round && state.woken.iter().any(|m| m == member) {
        return format!("'{member}' already woken for round {}", state.round);
    }
    let brief = round_brief_for(state, role_for(state, member));
    // Spawned members may have no sqlite yet (their own woken process creates
    // it); everyone else keeps the strict existence check.
    let allow_missing = state.spawned.iter().any(|m| m == member);
    if let Err(e) = talk::push_talk(store, from_history_file, member, Some(&state.room), &brief, allow_missing)
        .map(|_| ())
    {
        return format!("wake '{member}' failed to queue brief: {e}");
    }
    let sessions_root = sessions_root_for(from_history_file);
    if is_session_live(&sessions_root, member) {
        state.woken_round = state.round;
        state.woken.push(member.to_string());
        return format!("'{member}' is live; round-{} brief queued in its inbox", state.round);
    }
    match spawn_headless(from_history_file, state, member, &brief, exe_override) {
        Ok(note) => format!("'{member}' is inactive; {note}"),
        Err(e) => format!("'{member}' brief queued but {e}"),
    }
}

/// Release the woken marks of members that stayed silent and own no live
/// process, so the next `discuss_collect` retries their headless wake instead
/// of hitting the `already woken` guard forever. Live-but-slow members keep
/// their marks (re-waking them would only queue duplicate briefs), and a
/// failed liveness scan keeps every mark: without proof of death we assume
/// the woken process may still answer. State is saved only when a mark was
/// actually released; a save failure must not fail the collect, so it is
/// ignored like the wake-pass save.
pub(crate) fn release_dead_wakes(
    from_history_file: &Path,
    state: &mut DiscussState,
    missing: &[String],
) {
    if missing.is_empty() {
        return;
    }
    let root = sessions_root_for(from_history_file);
    let before = state.woken.len();
    state.woken.retain(|m| {
        let stale = missing.iter().any(|x| x == m) && !is_session_live(&root, m);
        !stale
    });
    if state.woken.len() != before {
        let _ = save_state(from_history_file, state);
    }
}

pub(crate) struct CollectOutcome {
    pub replies: Vec<TalkMessage>,
    pub missing: Vec<String>,
    /// True when the host pressed Ctrl+C mid-wait: the wait ended early, but
    /// `replies` (already appended to the transcript) is still valid.
    pub canceled: bool,
}

/// Poll the host's own inbox until every member has replied for this room or
/// the timeout elapses, or the host presses Ctrl+C (reported as canceled with
/// whatever was already collected). Non-room mail is requeued untouched. Replies are
/// attributed by arrival window (drained since the previous collect).
///
/// Every newly drained reply is appended to the room transcript before the
/// next sleep, so killing the collect mid-wait loses at most in-flight
/// memory, never already-drained replies.
pub(crate) fn collect_round(
    history_file: &Path,
    room: &str,
    members: &[String],
    timeout_secs: u64,
) -> CollectOutcome {
    collect_round_inner(history_file, room, members, timeout_secs, &|| {
        crate::ai::tools::registry::common::is_tool_cancel_requested()
    })
}

/// [`collect_round`] with an injectable cancellation predicate, so tests can
/// script a Ctrl+C without a kernel signal harness. Production wires the
/// tool-cancel flag flipped by the SIGINT handler.
fn collect_round_inner(
    history_file: &Path,
    room: &str,
    members: &[String],
    timeout_secs: u64,
    cancelled: &dyn Fn() -> bool,
) -> CollectOutcome {
    let timeout = timeout_secs.clamp(MIN_COLLECT_TIMEOUT_SECS, MAX_COLLECT_TIMEOUT_SECS);
    let deadline = Instant::now() + Duration::from_secs(timeout);
    let mut replies = Vec::new();
    // Index of the first reply not yet persisted to the transcript.
    let mut persisted = 0usize;
    let mut canceled = false;
    loop {
        // A host-side Ctrl+C flips the tool-cancel flag via the SIGINT
        // handler; without this check the wait ignores it until the deadline
        // and the turn looks wedged with no way to inspect state.
        if cancelled() {
            canceled = true;
            break;
        }
        let drained = talk::drain_talk(history_file);
        let mut others = Vec::new();
        for m in drained {
            if m.room.as_deref() == Some(room) {
                replies.push(m);
            } else {
                others.push(m);
            }
        }
        if !others.is_empty() {
            talk::requeue_talk(history_file, &others);
        }
        if replies.len() > persisted {
            // Best-effort per batch: on failure the unpersisted tail stays in
            // memory and is retried on the next poll; nothing drained is ever
            // dropped here.
            if append_transcript(history_file, room, &replies[persisted..]).is_ok() {
                persisted = replies.len();
            }
        }
        let heard: std::collections::HashSet<&str> = replies
            .iter()
            .map(|m| m.from_session.as_str())
            .collect();
        if members.iter().all(|m| heard.contains(m.as_str())) || Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let heard: std::collections::HashSet<&str> = replies
        .iter()
        .map(|m| m.from_session.as_str())
        .collect();
    let missing: Vec<String> = members
        .iter()
        .filter(|m| !heard.contains(m.as_str()))
        .cloned()
        .collect();
    CollectOutcome {
        replies,
        missing,
        canceled,
    }
}

/// Broadcast the host's round summary plus next prompt. Refuses past
/// `max_rounds` so an inconclusive argue terminates instead of burning model
/// calls; the host then closes with `discuss_end`.
pub(crate) fn broadcast_round(
    store: &SessionStore,
    from_history_file: &Path,
    state: &mut DiscussState,
    text: &str,
) -> Result<Vec<(String, String)>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("broadcast text must not be empty".to_string());
    }
    if state.status != DiscussStatus::Open {
        return Err(format!("room '{}' is closed", state.room));
    }
    if state.round >= state.max_rounds {
        return Err(format!(
            "room '{}' already at round {}/{} (max); run discuss_end with a closing summary instead",
            state.room, state.round, state.max_rounds
        ));
    }
    state.round += 1;
    state.woken_round = state.round;
    state.woken.clear();
    let mut results = Vec::new();
    for m in &state.members.clone() {
        let brief = format!(
            "{}\n\n{}",
            text.trim(),
            round_brief_tail_for(state, role_for(state, m))
        );
        // Spawned members keep the relaxed push: if one never came alive its
        // sqlite may still be missing, and the brief simply waits in its inbox.
        let allow_missing = state.spawned.iter().any(|s| s == m);
        match talk::push_talk(store, from_history_file, m, Some(&state.room), &brief, allow_missing) {
            Ok(_) => results.push((m.clone(), "delivered".to_string())),
            Err(e) => results.push((m.clone(), format!("failed: {e}"))),
        }
    }
    append_transcript(
        from_history_file,
        &state.room,
        &[host_marker(&state.host.clone(), &state.room, format!("[host opens round {}]\n{text}", state.round))],
    )
    .map_err(|e| format!("write broadcast transcript: {e}"))?;
    save_state(from_history_file, state)?;
    Ok(results)
}

/// Short per-round tail appended after the host's own summary text, so the
/// member still gets the machine-readable reply instructions.
fn round_brief_tail_for(state: &DiscussState, role: &str) -> String {
    let tail = format!(
        "[discuss room={} round={}/{} host={} — reply once with talk_send to_session=\"{}\" room=\"{}\"]",
        state.room, state.round, state.max_rounds, state.host, state.host, state.room,
    );
    let role = role.trim();
    if role.is_empty() {
        tail
    } else {
        format!("{tail}\nYour role: {role}")
    }
}

/// Close the room and record the host's closing summary in the transcript.
/// Returns the transcript path (the meeting record).
pub(crate) fn close_room(
    from_history_file: &Path,
    state: &mut DiscussState,
    summary: &str,
) -> Result<PathBuf, String> {
    let summary = summary.trim();
    if summary.is_empty() {
        return Err("closing summary must not be empty".to_string());
    }
    if state.status != DiscussStatus::Open {
        return Err(format!("room '{}' is already closed", state.room));
    }
    state.status = DiscussStatus::Closed;
    append_transcript(
        from_history_file,
        &state.room,
        &[host_marker(&state.host.clone(), &state.room, format!("[host closes room]\n{summary}"))],
    )
    .map_err(|e| format!("write closing transcript: {e}"))?;
    save_state(from_history_file, state)?;
    Ok(talk::talk_room_file(from_history_file, &state.room))
}

/// Read-only progress snapshot of one room: round state, per-member
/// reply/wake/liveness flags, uncollected room mail still in the host inbox,
/// and the transcript tail. Never drains, wakes, or writes — safe to call any
/// time, including right after a resume that left a `discuss_collect` behind.
#[derive(Debug, Clone)]
pub(crate) struct MemberProgress {
    pub member: String,
    pub role: String,
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
    pub topic: String,
    pub status: DiscussStatus,
    pub round: u32,
    pub max_rounds: u32,
    pub members: Vec<MemberProgress>,
    pub uncollected: usize,
    pub recent: Vec<StatusEvent>,
    pub transcript_path: PathBuf,
}

/// Rooms this host owns (state files under `rooms/`). Best-effort listing for
/// error hints: unreadable entries are skipped, never fatal. Names are the
/// on-disk (sanitized) forms, which match room names for ordinary input.
pub(crate) fn list_rooms(history_file: &Path) -> Vec<String> {
    let mut rooms = Vec::new();
    if let Ok(entries) = fs::read_dir(talk::talk_rooms_dir(history_file)) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(room) = name.strip_suffix(".state.json") {
                rooms.push(room.to_string());
            }
        }
    }
    rooms.sort();
    rooms
}

/// One-line preview for status views: whitespace flattened and capped so a
/// long verdict cannot flood the host turn.
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
            format!("{e} (rooms owned by this session: {})", known.join(", "))
        }
    })?;
    // Transcript in file order; malformed lines skipped like the inbox path.
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
    // Current-round window: everything after the host's latest line. Replies
    // are attributed by arrival window, mirroring `collect_round`.
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
    let members = state
        .members
        .iter()
        .map(|m| MemberProgress {
            member: m.clone(),
            role: role_for(&state, m).to_string(),
            live: is_session_live(&root, m),
            woken_this_round: state.woken_round == state.round
                && state.woken.iter().any(|w| w == m),
            replied_this_round: replied.contains(m.as_str()),
            waiting_in_inbox: waiting.contains(m),
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
        topic: state.topic.clone(),
        status: state.status,
        round: state.round,
        max_rounds: state.max_rounds,
        members,
        uncollected,
        recent,
        transcript_path: talk::talk_room_file(history_file, &state.room),
    })
}

/// Render a [`RoomStatus`] as the host-visible progress report. Shared by the
/// `discuss_status` tool and the `/discuss status` local command so both
/// surfaces read identically. Pure formatting: reads nothing, writes nothing.
pub(crate) fn format_room_status(st: &RoomStatus) -> String {
    let mut s = format!(
        "Room '{}' — {}\nStatus: {}, round {}/{}.\n",
        st.room,
        st.topic,
        if st.status == DiscussStatus::Open {
            "open"
        } else {
            "closed"
        },
        st.round,
        st.max_rounds
    );
    for m in &st.members {
        let mut flags = vec![if m.live { "live" } else { "inactive" }];
        if m.woken_this_round {
            flags.push("woken");
        }
        flags.push(if m.replied_this_round {
            "replied"
        } else {
            "waiting"
        });
        if m.waiting_in_inbox {
            flags.push("mail in inbox");
        }
        s.push_str(&format!(
            "- {} [role: {}] ({})\n",
            m.member,
            display_role(&m.role),
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
            let kind = if e.host_line { "host" } else { "reply" };
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
    if st.status != DiscussStatus::Open {
        s.push_str("Room is closed; nothing further to run.");
    } else if st.uncollected > 0 {
        s.push_str("Next: run discuss_collect to gather the waiting replies.");
    } else if st.members.iter().all(|m| m.replied_this_round) {
        if st.round >= st.max_rounds {
            s.push_str(
                "All members replied on the final round: run discuss_end with a closing summary.",
            );
        } else {
            s.push_str("All members replied: summarize and run discuss_broadcast for the next round, or discuss_end if converged.");
        }
    } else {
        s.push_str("Next: run discuss_collect to wait for and gather replies (it wakes inactive members headlessly).");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("discuss_test_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Host plus two member sessions, all backed by stub sqlite files.
    fn fixture() -> (PathBuf, SessionStore, PathBuf, PathBuf, PathBuf) {
        let dir = temp_dir();
        let store = SessionStore::new(&dir.join("hist.sqlite"));
        let host_hist = store.session_history_file("host-sess");
        let m1_hist = store.session_history_file("member-1");
        let m2_hist = store.session_history_file("member-2");
        for hist in [&host_hist, &m1_hist, &m2_hist] {
            std::fs::create_dir_all(hist.parent().unwrap()).unwrap();
            std::fs::write(hist, b"stub").unwrap();
        }
        (dir, store, host_hist, m1_hist, m2_hist)
    }

    fn members() -> Vec<String> {
        vec!["member-1".to_string(), "member-2".to_string()]
    }

    /// Positional wrapper keeping the pre-`StartRequest` call shape for the
    /// explicit-members tests below.
    fn start(
        store: &SessionStore,
        host_hist: &Path,
        topic: &str,
        members: &[String],
        max_rounds: Option<u32>,
        room: Option<&str>,
    ) -> Result<StartOutcome, String> {
        start_room(
            store,
            host_hist,
            &StartRequest {
                topic: topic.to_string(),
                members: members.to_vec(),
                member_count: None,
                roles: Vec::new(),
                max_rounds,
                room: room.map(str::to_string),
            },
            None,
        )
    }

    #[test]
    fn start_validates_input_and_sends_round_one() {
        let (dir, store, host_hist, m1_hist, _m2) = fixture();
        assert!(start(&store, &host_hist, "  ", &members(), None, None).is_err());
        assert!(start(&store, &host_hist, "t", &[], None, None).is_err());
        assert!(
            start(
                &store,
                &host_hist,
                "t",
                &["host-sess".to_string()],
                None,
                None
            )
            .is_err()
        );
        assert!(
            start(&store, &host_hist, "t", &["ghost".to_string()], None, None).is_err()
        );
        assert!(
            start(
                &store,
                &host_hist,
                "t",
                &["member-1".to_string(), "member-1".to_string()],
                None,
                None
            )
            .is_err()
        );

        let out = start(&store, &host_hist, "tabs vs spaces", &members(), Some(2), Some("r1"))
            .unwrap();
        assert_eq!(out.state.round, 1);
        assert_eq!(out.state.max_rounds, 2);
        assert_eq!(out.deliveries.len(), 2);
        // Test sessions own no live process, so both report inactive.
        assert!(out.deliveries.iter().all(|(_, live, _)| !live));
        // Round-1 brief reached the member inbox with the room tag.
        let msgs = talk::drain_talk(&m1_hist);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].room.as_deref(), Some("r1"));
        assert!(msgs[0].body.contains("tabs vs spaces"));
        // State persisted and reloads.
        let reloaded = load_state(&host_hist, "r1").unwrap();
        assert_eq!(reloaded.topic, "tabs vs spaces");
        assert_eq!(reloaded.members, members());
        // Reusing an open room name is refused.
        assert!(start(&store, &host_hist, "t2", &members(), None, Some("r1")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn broadcast_increments_and_refuses_past_max() {
        let (dir, store, host_hist, _m1, _m2) = fixture();
        let mut state = start(&store, &host_hist, "t", &members(), Some(2), Some("r2"))
            .unwrap()
            .state;
        let res = broadcast_round(&store, &host_hist, &mut state, "round 2 prompt").unwrap();
        assert_eq!(state.round, 2);
        assert!(res.iter().all(|(_, s)| s == "delivered"));
        assert!(state.woken.is_empty());
        // At max rounds the next broadcast is refused; the host must close.
        let err = broadcast_round(&store, &host_hist, &mut state, "round 3").unwrap_err();
        assert!(err.contains("discuss_end"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn collect_filters_room_and_requeues_others() {
        let (dir, store, host_hist, _m1_hist, m2_hist) = fixture();
        let state =
            start(&store, &host_hist, "t", &members(), None, Some("r3")).unwrap().state;
        // A member reply plus unrelated mail land in the host inbox.
        talk::push_talk(&store, &m2_hist, "host-sess", Some("r3"), "member-2 view", false)
            .unwrap();
        talk::push_talk(
            &store,
            &m2_hist,
            "host-sess",
            Some("other-room"),
            "unrelated",
            false,
        )
        .unwrap();
        let out = collect_round(&host_hist, "r3", &state.members, 5);
        assert_eq!(out.replies.len(), 1);
        assert_eq!(out.replies[0].body, "member-2 view");
        assert_eq!(out.missing, vec!["member-1".to_string()]);
        // Unrelated mail survived the collect for a later drain.
        let rest = talk::drain_talk(&host_hist);
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].room.as_deref(), Some("other-room"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn collect_persists_replies_incrementally() {
        let (dir, store, host_hist, _m1_hist, m2_hist) = fixture();
        let state =
            start(&store, &host_hist, "t", &members(), None, Some("r4")).unwrap().state;
        talk::push_talk(&store, &m2_hist, "host-sess", Some("r4"), "member-2 view", false)
            .unwrap();
        let out = collect_round(&host_hist, "r4", &state.members, 5);
        assert_eq!(out.replies.len(), 1);
        // The reply reached the room transcript inside collect_round itself —
        // no service-layer follow-up needed, so a killed wait loses nothing
        // already drained, and nothing is recorded twice.
        let transcript =
            std::fs::read_to_string(talk::talk_room_file(&host_hist, "r4")).unwrap();
        assert_eq!(transcript.matches("member-2 view").count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn collect_releases_dead_wake_marks_for_retry() {
        let (dir, store, host_hist, _m1_hist, _m2_hist) = fixture();
        let mut state =
            start(&store, &host_hist, "t", &members(), None, Some("r5")).unwrap().state;
        // Both members woken, neither replied; test sessions own no live
        // process, so both marks are stale and must be released.
        state.woken_round = state.round;
        state.woken = members();
        save_state(&host_hist, &state).unwrap();
        release_dead_wakes(&host_hist, &mut state, &members());
        assert!(state.woken.is_empty());
        let reloaded = load_state(&host_hist, "r5").unwrap();
        assert!(reloaded.woken.is_empty());
        // A member that did reply keeps its mark (no pointless re-wake).
        state.woken = members();
        save_state(&host_hist, &state).unwrap();
        release_dead_wakes(&host_hist, &mut state, &["member-1".to_string()]);
        assert_eq!(state.woken, vec!["member-2".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn collect_inner_stops_promptly_on_cancel() {
        let (dir, store, host_hist, _m1_hist, m2_hist) = fixture();
        let state =
            start(&store, &host_hist, "t", &members(), None, Some("r-cancel")).unwrap().state;
        // One reply already waiting: a cancel must keep it, not drop it.
        talk::push_talk(&store, &m2_hist, "host-sess", Some("r-cancel"), "member-2 view", false)
            .unwrap();
        // Script a Ctrl+C on the third poll so at least one drain cycle runs.
        let polls = std::cell::Cell::new(0);
        let started = std::time::Instant::now();
        let out = collect_round_inner(&host_hist, "r-cancel", &state.members, 60, &|| {
            polls.set(polls.get() + 1);
            polls.get() >= 3
        });
        assert!(out.canceled);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(out.replies.len(), 1);
        assert_eq!(out.replies[0].body, "member-2 view");
        assert_eq!(out.missing, vec!["member-1".to_string()]);
        // The kept reply reached the transcript before the cancel broke out.
        let transcript =
            std::fs::read_to_string(talk::talk_room_file(&host_hist, "r-cancel")).unwrap();
        assert!(transcript.contains("member-2 view"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn wake_spawns_detached_process_group() {
        let (dir, store, host_hist, _m1, _m2) = fixture();
        let mut state =
            start(&store, &host_hist, "t", &members(), None, Some("r-detach")).unwrap().state;
        // The fake binary records its own pid/pgid, then sleeps so the test
        // can inspect it: a detached worker must lead its own group
        // (pgid == pid), so a host-terminal Ctrl+C never reaches it.
        let pid_file = dir.join("child.pid");
        let pgid_file = dir.join("child.pgid");
        let fake_bin = dir.join("fake-a-detached");
        std::fs::write(
            &fake_bin,
            format!(
                "#!/bin/sh\necho $$ > {}\nps -o pgid= -p $$ > {}\nsleep 30\n",
                pid_file.display(),
                pgid_file.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&fake_bin, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();
        let note = spawn_headless(&host_hist, &mut state, "member-1", "brief", Some(&fake_bin));
        assert!(note.is_ok(), "spawn failed: {note:?}");
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !pgid_file.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let pid: u32 = std::fs::read_to_string(&pid_file).unwrap().trim().parse().unwrap();
        let pgid: u32 = std::fs::read_to_string(&pgid_file).unwrap().trim().parse().unwrap();
        // Kill first so a failed assert cannot strand the sleeper.
        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        assert_eq!(pgid, pid, "worker must lead its own process group");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wake_spawns_headless_child_for_inactive_member() {
        let (dir, store, host_hist, _m1, _m2) = fixture();
        let mut state =
            start(&store, &host_hist, "t", &members(), None, Some("r4")).unwrap().state;
        // No live process owns the fixture sessions, so wake re-execs. A tiny
        // `exit 0` script stands in for the binary: spawn success plus the
        // woken bookkeeping is what this pins (the real child path needs a
        // live model and is verified manually).
        let fake_bin = dir.join("fake-a");
        std::fs::write(&fake_bin, "#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(&fake_bin, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();
        let note = wake_member(&store, &host_hist, &mut state, "member-1", Some(&fake_bin));
        assert!(note.contains("woken headlessly"), "{note}");
        assert_eq!(state.woken, vec!["member-1".to_string()]);
        // Second wake for the same round is a no-op (no duplicate children).
        let again = wake_member(&store, &host_hist, &mut state, "member-1", Some(&fake_bin));
        assert!(again.contains("already woken"), "{again}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn close_marks_shut_and_returns_transcript() {
        let (dir, store, host_hist, _m1, _m2) = fixture();
        let mut state =
            start(&store, &host_hist, "t", &members(), None, Some("r5")).unwrap().state;
        assert!(close_room(&host_hist, &mut state, "  ").is_err());
        let path = close_room(&host_hist, &mut state, "agreed on spaces").unwrap();
        assert!(path.is_file());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("agreed on spaces"));
        assert!(content.contains("host opens round 1"));
        assert_eq!(load_state(&host_hist, "r5").unwrap().status, DiscussStatus::Closed);
        assert!(close_room(&host_hist, &mut state, "again").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn spawn_req(topic: &str, count: usize, room: &str) -> StartRequest {
        StartRequest {
            topic: topic.to_string(),
            members: Vec::new(),
            member_count: Some(count),
            roles: Vec::new(),
            max_rounds: None,
            room: Some(room.to_string()),
        }
    }

    #[test]
    fn member_count_spawns_owned_members_with_default_roles() {
        let (dir, store, host_hist, _m1, _m2) = fixture();
        let fake_bin = dir.join("fake-a");
        std::fs::write(&fake_bin, "#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(&fake_bin, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();
        let out = start_room(&store, &host_hist, &spawn_req("tabs vs spaces", 3, "argue"), Some(&fake_bin))
            .unwrap();
        assert_eq!(out.state.members.len(), 3);
        assert_eq!(out.state.spawned, out.state.members);
        // Default argue cycle: pro / contra / judge.
        assert!(out.state.roles[0].contains("Proponent"), "{}", out.state.roles[0]);
        assert!(out.state.roles[1].contains("Opponent"), "{}", out.state.roles[1]);
        assert!(out.state.roles[2].contains("Judge"), "{}", out.state.roles[2]);
        for (i, m) in out.state.members.iter().enumerate() {
            // Room-owned id, valid, and fresh (no collision with fixtures).
            assert!(m.starts_with("dc-argue-m"), "{m}");
            SessionStore::validate_session_id(m).unwrap();
            // The host never creates the member sqlite: only its own woken
            // process may. The file must still be absent right after start
            // (the fake binary exits without touching it).
            assert!(!store.session_history_file(m).is_file(), "{m}");
            let (_, _, note) = &out.deliveries[i];
            assert!(note.contains("auto-created"), "{note}");
            assert!(note.contains("woken headlessly"), "{note}");
            // Round-1 brief (topic + assigned role) waits in the fresh inbox.
            let member_hist = store.session_history_file(m);
            let msgs = talk::drain_talk(&member_hist);
            assert_eq!(msgs.len(), 1);
            assert!(msgs[0].body.contains("tabs vs spaces"), "{}", msgs[0].body);
            assert!(msgs[0].body.contains(&out.state.roles[i]), "{}", msgs[0].body);
        }
        // Ownership record survives a state reload.
        let reloaded = load_state(&host_hist, "argue").unwrap();
        assert_eq!(reloaded.spawned, out.state.members);
        assert_eq!(reloaded.roles.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn member_count_rejects_bad_combinations() {
        let (dir, store, host_hist, _m1, _m2) = fixture();
        // Both members and member_count.
        let mut both = spawn_req("t", 2, "r-both");
        both.members = members();
        assert!(start_room(&store, &host_hist, &both, None).is_err());
        // Neither.
        let neither = StartRequest {
            topic: "t".to_string(),
            ..Default::default()
        };
        assert!(start_room(&store, &host_hist, &neither, None).is_err());
        // Out-of-range counts.
        assert!(start_room(&store, &host_hist, &spawn_req("t", 0, "r-zero"), None).is_err());
        assert!(start_room(&store, &host_hist, &spawn_req("t", 9, "r-many"), None).is_err());
        // Roles must line up 1:1 (both paths).
        let mut mismatch_count = spawn_req("t", 2, "r-mismatch");
        mismatch_count.roles = vec!["only-one".to_string()];
        assert!(start_room(&store, &host_hist, &mismatch_count, None).is_err());
        let mut mismatch_explicit = StartRequest {
            topic: "t".to_string(),
            members: members(),
            roles: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            room: Some("r-mismatch2".to_string()),
            ..Default::default()
        };
        assert!(start_room(&store, &host_hist, &mismatch_explicit, None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_state_without_roles_still_loads() {
        let (dir, store, host_hist, _m1, _m2) = fixture();
        // Room state written before roles/spawned existed: plain JSON without
        // the new fields must load with empty defaults, and per-member briefs
        // degrade to the role-free form.
        let file = talk::talk_state_file(&host_hist, "legacy");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(
            &file,
            r#"{"room":"legacy","topic":"t","host":"host-sess","members":["member-1"],"max_rounds":3,"round":1,"status":"Open","woken_round":1,"woken":[]}"#,
        )
        .unwrap();
        let mut state = load_state(&host_hist, "legacy").unwrap();
        assert!(state.roles.is_empty());
        assert!(state.spawned.is_empty());
        assert_eq!(role_for(&state, "member-1"), "");
        let results = broadcast_round(&store, &host_hist, &mut state, "host summary").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "delivered");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn room_status_reports_replies_pending_and_tail() {
        let (dir, store, host_hist, _m1, _m2) = fixture();
        let out = start(&store, &host_hist, "tea?", &members(), Some(3), Some("r-status")).unwrap();
        assert_eq!(out.state.room, "r-status");
        // One reply already in the transcript; one reply still waiting in the
        // host inbox (planted via requeue so the sender-side room log that
        // `push_talk` would mirror stays untouched).
        append_transcript(
            &host_hist,
            "r-status",
            &[TalkMessage {
                from_session: "member-1".to_string(),
                body: "green tea, brewed strong".to_string(),
                ts: 2,
                room: Some("r-status".to_string()),
            }],
        )
        .unwrap();
        talk::requeue_talk(
            &host_hist,
            &[TalkMessage {
                from_session: "member-2".to_string(),
                body: "oolong".to_string(),
                ts: 3,
                room: Some("r-status".to_string()),
            }],
        );
        let st = room_status(&host_hist, "r-status").unwrap();
        assert_eq!(st.round, 1);
        assert_eq!(st.max_rounds, 3);
        assert_eq!(st.topic, "tea?");
        assert_eq!(st.members.len(), 2);
        let m1 = st.members.iter().find(|m| m.member == "member-1").unwrap();
        assert!(m1.replied_this_round);
        assert!(!m1.waiting_in_inbox);
        let m2 = st.members.iter().find(|m| m.member == "member-2").unwrap();
        assert!(!m2.replied_this_round);
        assert!(m2.waiting_in_inbox);
        assert_eq!(st.uncollected, 1);
        // Tail: host lines first (opening marker plus send mirrors), then the
        // member-1 reply last.
        assert_eq!(st.recent.len(), 3);
        assert!(st.recent[0].host_line);
        assert!(st.recent[1].host_line);
        assert_eq!(st.recent[2].from_session, "member-1");
        assert!(!st.recent[2].host_line);
        assert!(st.recent[2].preview.contains("green tea"));
        assert!(st.transcript_path.is_file());
        // Unknown rooms name the rooms this session owns (resume aid).
        let err = room_status(&host_hist, "nope").unwrap_err();
        assert!(err.contains("r-status"), "unexpected error: {err}");
        // Status still works after close.
        let mut closed = load_state(&host_hist, "r-status").unwrap();
        close_room(&host_hist, &mut closed, "we drink tea").unwrap();
        let st = room_status(&host_hist, "r-status").unwrap();
        assert_eq!(st.status, DiscussStatus::Closed);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
