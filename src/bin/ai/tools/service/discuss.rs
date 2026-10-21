// Host-side discuss/argue tool services: `discuss_start` / `discuss_collect` /
// `discuss_broadcast` / `discuss_end` / `discuss_status`.
//
// The host orchestrates member sessions over the talk mailbox; every function
// below touches only the host's own files (room state, room transcript, member
// inboxes via `push_talk`). A member's sqlite database is written solely by
// that member's own process. All four stay on the fail-closed defaults:
// serial execution, no retry (`discuss_collect` drains the host inbox and must
// never be replayed).
use serde_json::Value;

use crate::ai::{
    driver::{discuss, talk},
    history::SessionStore,
};

fn current_ctx() -> Result<(SessionStore, std::path::PathBuf), String> {
    let ctx = crate::ai::driver::runtime_ctx::try_current().ok_or_else(|| {
        "discuss tools need an active driver context (not available in one-shot/test calls)"
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

fn arg_members(args: &Value) -> Option<Vec<String>> {
    let arr = args.get("members")?.as_array()?;
    let out: Vec<String> = arr
        .iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if out.is_empty() { None } else { Some(out) }
}

/// Optional per-member roles, positionally aligned with `members` (or with the
/// auto-spawned members for `member_count`). Empty entries are kept — an empty
/// role means "no assigned stance", so filtering would break alignment.
fn arg_roles(args: &Value) -> Vec<String> {
    for k in ["roles", "stances"] {
        if let Some(arr) = args.get(k).and_then(Value::as_array) {
            return arr
                .iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_string())
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

fn format_replies(replies: &[talk::TalkMessage]) -> String {
    if replies.is_empty() {
        return "(no replies)".to_string();
    }
    replies
        .iter()
        .enumerate()
        .map(|(i, m)| format!("── [reply {}/{}] from={} ──\n{}", i + 1, replies.len(), m.from_session, m.body))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// ToolSpec adapter: `ToolSpec.execute` returns model-visible text directly;
/// `ToolResult` wrapping is done by the registry path.
pub(crate) fn execute_discuss_start(args: &Value) -> Result<String, String> {
    let topic = arg_str(args, &["topic", "question", "subject"])
        .ok_or_else(|| "topic must not be empty (the question to argue)".to_string())?;
    // Exactly one of `members` (existing sessions) or `member_count`
    // (host-owned auto-spawn); `start_room` enforces the either/or.
    let members = arg_members(args).unwrap_or_default();
    let member_count =
        arg_u64(args, &["member_count", "count", "members_count", "n"]).map(|n| n as usize);
    let roles = arg_roles(args);
    let max_rounds = arg_u64(args, &["max_rounds", "rounds"]).map(|n| n as u32);
    let room = arg_str(args, &["room"]).map(str::to_string);

    let (store, history_file) = current_ctx()?;
    let req = discuss::StartRequest {
        topic: topic.to_string(),
        members,
        member_count,
        roles,
        max_rounds,
        room,
    };
    let out = discuss::start_room(&store, &history_file, &req, None)?;
    let mut s = format!(
        "Discuss room '{}' opened (round 1 of max {}, topic: {}).\n",
        out.state.room, out.state.max_rounds, out.state.topic
    );
    for (m, live, note) in &out.deliveries {
        s.push_str(&format!("- {m} [{}]: {note}\n", if *live { "live" } else { "inactive" }));
    }
    if !out.state.spawned.is_empty() {
        s.push_str(&format!(
            "Auto-created {} member session(s) owned by this room (their databases are created by their own processes, never by the host):\n",
            out.state.spawned.len()
        ));
        for (i, m) in out.state.spawned.iter().enumerate() {
            let role = out.state.roles.get(i).map(String::as_str).unwrap_or("");
            s.push_str(&format!("- {m} [role: {}]\n", discuss::display_role(role)));
        }
    }
    s.push_str("Next: run discuss_collect to gather round-1 replies (it wakes inactive members headlessly). Then summarize and run discuss_broadcast, or discuss_end when converged.");
    Ok(s)
}

pub(crate) fn execute_discuss_collect(args: &Value) -> Result<String, String> {
    let room = arg_str(args, &["room"])
        .ok_or_else(|| "room must not be empty (from discuss_start)".to_string())?;
    let (store, history_file) = current_ctx()?;
    let mut state = discuss::load_state(&history_file, room)?;
    if state.status != discuss::DiscussStatus::Open {
        return Err(format!("room '{}' is closed", state.room));
    }
    let round = state.round;
    // Wake pass first: inactive members that have not been woken this round
    // get a headless one-shot; their replies land in the host inbox below.
    let wake = args.get("wake").and_then(Value::as_bool).unwrap_or(true);
    let mut wake_notes = Vec::new();
    if wake {
        for m in state.members.clone() {
            let note = discuss::wake_member(&store, &history_file, &mut state, &m, None);
            if !note.contains("already woken") {
                wake_notes.push(note);
            }
        }
        // Persist the woken set even if the collect below finds nothing. A
        // failure here must not fail the collect: the woken set is an
        // idempotence optimization, not truth.
        let _ = discuss::save_state(&history_file, &state);
    }
    let timeout = arg_u64(args, &["timeout_secs", "timeout"]).unwrap_or(discuss::DEFAULT_COLLECT_TIMEOUT_SECS);
    // Replies are already in the transcript: collect_round appends every
    // newly drained reply before the next poll, so a killed wait loses
    // nothing already drained.
    let out = discuss::collect_round(&history_file, &state.room, &state.members, timeout);
    // A wake that produced no reply must not block the next collect: release
    // dead wake marks so silent inactive members are re-woken on retry.
    discuss::release_dead_wakes(&history_file, &mut state, &out.missing);
    let mut s = format!(
        "Room '{}' round {}/{} collect: {} replie(s), {} missing.\n",
        state.room,
        round,
        state.max_rounds,
        out.replies.len(),
        out.missing.len()
    );
    for n in &wake_notes {
        s.push_str(&format!("(wake) {n}\n"));
    }
    s.push_str(&format_replies(&out.replies));
    if out.canceled {
        s.push_str("Collect canceled by user (Ctrl+C). Replies above are already in the room transcript; re-run discuss_collect to keep waiting for the missing members.\n");
    }
    if !out.missing.is_empty() {
        s.push_str(&format!(
            "\n\nMissing: {}. They may still reply next round; re-run discuss_collect to wait longer, or proceed without them.",
            out.missing.join(", ")
        ));
    }
    if round >= state.max_rounds {
        s.push_str(" This was the final round: run discuss_end with a closing summary.");
    } else {
        s.push_str(" Next: summarize and run discuss_broadcast for the next round, or discuss_end if converged.");
    }
    Ok(s)
}

pub(crate) fn execute_discuss_broadcast(args: &Value) -> Result<String, String> {
    let room = arg_str(args, &["room"])
        .ok_or_else(|| "room must not be empty (from discuss_start)".to_string())?;
    let text = arg_str(args, &["text", "message", "prompt"])
        .ok_or_else(|| "text must not be empty (round summary plus next prompt)".to_string())?;
    let (store, history_file) = current_ctx()?;
    let mut state = discuss::load_state(&history_file, room)?;
    let results = discuss::broadcast_round(&store, &history_file, &mut state, text)?;
    let mut s = format!(
        "Room '{}' advanced to round {}/{}.\n",
        state.room, state.round, state.max_rounds
    );
    for (m, r) in &results {
        s.push_str(&format!("- {m}: {r}\n"));
    }
    s.push_str("Next: run discuss_collect to gather this round's replies.");
    Ok(s)
}

pub(crate) fn execute_discuss_end(args: &Value) -> Result<String, String> {
    let room = arg_str(args, &["room"])
        .ok_or_else(|| "room must not be empty (from discuss_start)".to_string())?;
    let summary = arg_str(args, &["summary", "conclusion", "text"])
        .ok_or_else(|| "summary must not be empty (the meeting conclusion)".to_string())?;
    let (_, history_file) = current_ctx()?;
    let mut state = discuss::load_state(&history_file, room)?;
    let path = discuss::close_room(&history_file, &mut state, summary)?;
    let mut s = format!(
        "Room '{}' closed. Full transcript (all rounds plus this summary) at {}.",
        state.room,
        path.display()
    );
    if !state.spawned.is_empty() {
        s.push_str(&format!(
            " Auto-created member sessions retained for inspection: {}. Delete them explicitly if unwanted.",
            state.spawned.join(", ")
        ));
    }
    Ok(s)
}

/// Read-only progress view: round state, per-member flags, uncollected mail,
/// and the transcript tail. Never wakes members or drains the inbox, so it is
/// safe to call any time — including right after resuming the host session.
pub(crate) fn execute_discuss_status(args: &Value) -> Result<String, String> {
    let room = arg_str(args, &["room"])
        .ok_or_else(|| "room must not be empty (from discuss_start)")?;
    let (_, history_file) = current_ctx()?;
    let st = discuss::room_status(&history_file, room)?;
    // Formatting is shared with the `/discuss status` local command so both
    // surfaces read identically; see `driver::discuss::format_room_status`.
    Ok(discuss::format_room_status(&st))
}
