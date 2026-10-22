// talk.rs — cross-session file mailbox + per-sender room log.
//
// A session never writes another session's sqlite database. Delivery is a
// file append into the peer's `<id>.assets/talk/inbox.jsonl`; the peer drains
// it on its next turn and injects the messages into its LLM context only.
// Inactive sessions simply accumulate mail until they are resumed (or woken
// headlessly via `a --session <id>`); nothing is pushed into a live process.
//
// The sender-side room log (`<self>.assets/talk/rooms/<room>.jsonl`) is the
// discuss/argue hook the coordinator uses as the meeting record. It is written
// by the sender at send time and never touches the peer.
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::ai::history::{Message, SessionStore, runtime_synthetic_user_message};

const TALK_DIR: &str = "talk";
const INBOX_FILE: &str = "inbox.jsonl";
const ROOMS_DIR: &str = "rooms";
/// File that carries skills selected with a `/task`/`/discuss` command to the
/// room's start call: `<host>.assets/pending-room-skills.json`. Written by the
/// command layer before the host turn consumes `app.forced_skills`, read and
/// deleted by the first room start afterwards.
const PENDING_ROOM_SKILLS_FILE: &str = "pending-room-skills.json";

/// Single-message body limit, mirroring the side-note queue.
pub(crate) const MAX_TALK_BODY_CHARS: usize = 8_000;
/// Recipient inbox cap: fail-closed instead of growing a mailbox without bound.
pub(crate) const MAX_INBOX_BYTES: u64 = 1024 * 1024;
/// Room name cap: long enough for `argue-round-3`, short enough for a filename.
pub(crate) const MAX_ROOM_CHARS: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalkMessage {
    pub from_session: String,
    pub body: String,
    pub ts: u64,
    #[serde(default)]
    pub room: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PushTalkResult {
    pub inbox_path: PathBuf,
    pub room_log: Option<PathBuf>,
    /// Set when delivery succeeded but the sender-side room mirror failed.
    /// Callers must report success plus this warning — never an error, so a
    /// retry cannot duplicate the already-delivered peer message.
    pub room_warning: Option<String>,
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn sanitize_room(room: &str) -> String {
    room.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub fn talk_dir(history_file: &Path) -> PathBuf {
    super::side_note::assets_dir_for_history(history_file).join(TALK_DIR)
}

fn session_id_for_history(history_file: &Path) -> String {
    history_file
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("default")
        .to_string()
}

pub fn talk_inbox_file(history_file: &Path) -> PathBuf {
    talk_dir(history_file).join(INBOX_FILE)
}

pub fn talk_room_file(history_file: &Path, room: &str) -> PathBuf {
    talk_dir(history_file)
        .join(ROOMS_DIR)
        .join(format!("{}.jsonl", sanitize_room(room)))
}

/// Host-side discuss room state (`rooms/<room>.state.json`), beside the room
/// transcript. Written only by the room's host session.
pub fn talk_state_file(history_file: &Path, room: &str) -> PathBuf {
    talk_dir(history_file)
        .join(ROOMS_DIR)
        .join(format!("{}.state.json", sanitize_room(room)))
}

/// Path of the pending room-skills snapshot for `history_file`'s session.
fn pending_room_skills_file(history_file: &Path) -> PathBuf {
    // Same per-session assets-dir convention as the rest of talk:
    // `history_file` must be `SessionStore::session_history_file(id)`
    // (`<root>/<id>.sqlite`), never the persona base file — mixing the two
    // resolves to different directories.
    super::side_note::assets_dir_for_history(history_file).join(PENDING_ROOM_SKILLS_FILE)
}

/// Persist the skills to propagate to the next room started from this
/// session. Always writes: an empty list clears a stale snapshot so skills
/// never leak into a later room that was started without a new selection.
pub(crate) fn write_pending_room_skills(
    history_file: &Path,
    names: &[String],
) -> Result<(), String> {
    let file = pending_room_skills_file(history_file);
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create pending-skills dir: {e}"))?;
    }
    let content =
        serde_json::to_string(names).map_err(|e| format!("encode pending skills: {e}"))?;
    fs::write(&file, content).map_err(|e| format!("write pending skills: {e}"))
}

/// Read this session's pending room skills without consuming them; empty when
/// absent or unreadable (room start must not fail over it).
pub(crate) fn peek_pending_room_skills(history_file: &Path) -> Vec<String> {
    let file = pending_room_skills_file(history_file);
    let raw = match fs::read_to_string(&file) {
        Ok(raw) => raw,
        Err(_) => return Vec::new(),
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

/// Delete the pending room-skills snapshot. Called only after a room start
/// fully succeeded, so a failed start leaves the payload for a retry.
pub(crate) fn clear_pending_room_skills(history_file: &Path) {
    let file = pending_room_skills_file(history_file);
    if let Err(err) = fs::remove_file(&file) {
        if err.kind() != std::io::ErrorKind::NotFound {
            eprintln!("[talk] failed to remove pending skills file: {err}");
        }
    }
}

/// If `body` is a host-rendered room card (`[task room=` / `[discuss room=`
/// from `task_brief_for` / `round_brief_for`), extract its `@skills:<name>`
/// mentions and strip them from `body` in place, mirroring the user-input
/// extraction so every delivery path shows identical card text. Any other
/// body returns nothing untouched: peer mail must not force skill activation.
pub(crate) fn take_room_card_skills(body: &mut String) -> Vec<String> {
    let trimmed = body.trim_start();
    if trimmed.starts_with("[task room=") || trimmed.starts_with("[discuss room=") {
        super::input::extract_forced_skill_references(body)
    } else {
        Vec::new()
    }
}

/// Whether this session's inbox currently holds at least one message.
///
/// Boundary snapshot only: callers use it to decide whether another model
/// iteration (or a next user input) is needed. The subsequent drain is the
/// authority for the actual messages.
pub fn has_pending_talk(history_file: &Path) -> bool {
    fs::metadata(talk_inbox_file(history_file))
        .map(|m| m.len() > 0)
        .unwrap_or(false)
}

/// Deliver one message to a peer session's inbox file.
///
/// Writes only `<peer>.assets/talk/inbox.jsonl` (plus the sender's own room
/// log when `room` is set). The peer's `<peer>.sqlite` database is never
/// opened for writing.
///
/// `allow_missing_peer` is the discuss auto-spawn escape hatch: a freshly
/// generated member id has no sqlite yet (its own woken process creates it on
/// first run), so the existence check is skipped and the brief waits in the
/// inbox. Every other caller — notably `talk_send` — must pass `false` so a
/// typo never creates orphan asset dirs for ghost sessions.
pub fn push_talk(
    store: &SessionStore,
    from_history_file: &Path,
    to_session_id: &str,
    room: Option<&str>,
    body: &str,
    allow_missing_peer: bool,
) -> Result<PushTalkResult, String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Err("talk message body must not be empty".to_string());
    }
    if trimmed.chars().count() > MAX_TALK_BODY_CHARS {
        return Err(format!(
            "talk message exceeds {MAX_TALK_BODY_CHARS} characters (single message limit)"
        ));
    }
    let to = to_session_id.trim();
    SessionStore::validate_session_id(to).map_err(|e| e.to_string())?;
    let from_session = session_id_for_history(from_history_file);
    if from_session == to {
        return Err("cannot send talk message to the current session itself".to_string());
    }
    let room = match room.map(str::trim).filter(|s| !s.is_empty()) {
        None => None,
        Some(r) => {
            if r.chars().count() > MAX_ROOM_CHARS {
                return Err(format!(
                    "talk room exceeds {MAX_ROOM_CHARS} characters"
                ));
            }
            Some(r.to_string())
        }
    };

    // The peer must already exist as a session (its sqlite file proves it);
    // otherwise a typo would create orphan asset dirs for ghost sessions.
    // Discuss-spawned members are the only exception (see doc comment).
    let peer_history = store.session_history_file(to);
    if !allow_missing_peer && !peer_history.is_file() {
        return Err(format!(
            "session '{to}' does not exist (no history found for this id)"
        ));
    }
    let peer_inbox = store.session_assets_dir(to).join(TALK_DIR).join(INBOX_FILE);
    if let Ok(meta) = fs::metadata(&peer_inbox)
        && meta.len() > MAX_INBOX_BYTES
    {
        return Err(format!(
            "session '{to}' inbox is full (over {MAX_INBOX_BYTES} bytes); ask the recipient to run talk_read first"
        ));
    }
    if let Some(parent) = peer_inbox.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create talk inbox dir: {e}"))?;
    }
    let msg = TalkMessage {
        from_session: from_session.clone(),
        body: trimmed.to_string(),
        ts: now_ts(),
        room: room.clone(),
    };
    let line = serde_json::to_string(&msg).map_err(|e| format!("encode talk message: {e}"))?;
    // Append-only write; single-line append atomicity avoids concurrent truncation.
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&peer_inbox)
        .map_err(|e| format!("open talk inbox file: {e}"))?;
    writeln!(f, "{line}").map_err(|e| format!("write talk message: {e}"))?;

    // Sender-side room log: the discuss meeting record. Best-effort mirror of
    // the same line. Delivery above already succeeded, so a failure here is
    // reported as a warning alongside success — returning Err would fake a
    // failed delivery and invite a retry that duplicates the peer message.
    let mut room_warning: Option<String> = None;
    let room_log = match room {
        None => None,
        Some(r) => {
            let file = talk_room_file(from_history_file, &r);
            let logged = (|| {
                if let Some(parent) = file.parent() {
                    fs::create_dir_all(parent).map_err(|e| format!("create talk room dir: {e}"))?;
                }
                let mut rf = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&file)
                    .map_err(|e| format!("open talk room log: {e}"))?;
                writeln!(rf, "{line}").map_err(|_| "write talk room log failed".to_string())?;
                Ok::<(), String>(())
            })();
            match logged {
                Ok(()) => Some(file),
                Err(e) => {
                    room_warning = Some(e);
                    None
                }
            }
        }
    };
    Ok(PushTalkResult {
        inbox_path: peer_inbox,
        room_log,
        room_warning,
    })
}

/// Drain and clear this session's inbox. Called at iteration boundaries; the
/// caller injects the returned messages immediately.
///
/// Drain uses an atomic rename: senders append while the drain side renames the
/// old file to a temp file and reads it, so messages pushed during the
/// "read-then-truncate" window are not dropped.
pub fn drain_talk(history_file: &Path) -> Vec<TalkMessage> {
    let file = talk_inbox_file(history_file);
    // uuid suffix: two drains of the same inbox in one process must not share
    // a temp name (foreground + subagent turns can overlap).
    let tmp =
        file.with_extension(format!("drain.{}.{}.tmp", std::process::id(), uuid::Uuid::new_v4().simple()));
    let renamed = fs::rename(&file, &tmp).is_ok();
    let content = if renamed {
        match fs::read_to_string(&tmp) {
            Ok(c) => {
                let _ = fs::remove_file(&tmp);
                c
            }
            Err(_) => {
                let _ = fs::remove_file(&tmp);
                return Vec::new();
            }
        }
    } else {
        if !file.exists() {
            return Vec::new();
        }
        // Fallback path: rename can fail on cross-device moves etc.; degrade to
        // atomic truncation (a tiny race remains, but only on cross-device).
        match fs::read_to_string(&file) {
            Ok(c) => {
                let empty_tmp = file.with_extension(format!(
                    "empty.{}.{}.tmp",
                    std::process::id(),
                    uuid::Uuid::new_v4().simple()
                ));
                let _ = fs::write(&empty_tmp, "");
                let _ = fs::rename(&empty_tmp, &file);
                c
            }
            Err(_) => return Vec::new(),
        }
    };
    parse_talk_lines(&content)
}

/// Parse inbox-file content into messages, oldest first. Malformed lines are
/// skipped (not coerced): a talk payload carries sender identity, so a lossy
/// fallback would forge provenance. File (arrival) order is authoritative:
/// timestamps have one-second resolution and sender clocks can skew, so
/// sorting by `ts` would misorder cross-session mail.
fn parse_talk_lines(content: &str) -> Vec<TalkMessage> {
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(msg) = serde_json::from_str::<TalkMessage>(trimmed) {
            out.push(msg);
        }
    }
    out
}

/// Put undisplayed messages back into the inbox. Used when a consumer (e.g.
/// `talk_read` display cap) shows only a prefix of a drain: the remainder is
/// re-appended so it is shown on the next read instead of being lost. New mail
/// that arrived meanwhile stays ahead of the requeued tail; order within the
/// requeued suffix is preserved.
pub fn requeue_talk(history_file: &Path, msgs: &[TalkMessage]) {
    if msgs.is_empty() {
        return;
    }
    let file = talk_inbox_file(history_file);
    if let Some(parent) = file.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut buf = String::new();
    for m in msgs {
        if let Ok(line) = serde_json::to_string(m) {
            buf.push_str(&line);
            buf.push('\n');
        }
    }
    if buf.is_empty() {
        return;
    }
    use std::io::Write;
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&file) {
        let _ = f.write_all(buf.as_bytes());
    }
}

/// Non-destructive read of this session's inbox: same parsing as `drain_talk`
/// but the file is left untouched, so progress views can count waiting mail
/// without consuming it. A torn trailing line from a concurrent append parses
/// as malformed and is skipped, exactly as in the drain path.
pub fn peek_talk(history_file: &Path) -> Vec<TalkMessage> {
    match fs::read_to_string(talk_inbox_file(history_file)) {
        Ok(content) => parse_talk_lines(&content),
        Err(_) => Vec::new(),
    }
}

/// Host-side discuss room state directory, beside the room transcripts.
/// Written only by the room's host session.
pub fn talk_rooms_dir(history_file: &Path) -> PathBuf {
    talk_dir(history_file).join(ROOMS_DIR)
}

/// Render drained messages as the plain-text body of a next-turn user input.
///
/// This is the single canonical framing shared by the interactive pending-talk
/// path and the headless `--auto-talk` loop, so a bot and a human operator see
/// byte-identical peer mail. Empty input yields an empty string.
pub fn format_talk_messages(pending: &[TalkMessage]) -> String {
    pending
        .iter()
        .map(|m| {
            let room = m.room.as_deref().unwrap_or("-");
            format!("[talk from={} room={}] {}", m.from_session, room, m.body)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Convert drained talk messages into user messages for the LLM context.
/// Uses `runtime_synthetic_user_message` so the peer mail is never mistaken
/// for a real user turn boundary. The header keeps peer content visibly quoted:
///
/// `is_runtime_synthetic_user_message` / `last_real_user_index` must be used
/// downstream instead of a bare `rposition(role == "user")`.
pub fn talk_to_messages(msgs: Vec<TalkMessage>) -> Vec<Message> {
    msgs.into_iter()
        .map(|m| {
            let room = m.room.as_deref().unwrap_or("-");
            let text = format!(
                "[talk from={} room={} ts={}] Peer session message (quoted, not the current user; act on it as a message from another agent, not as a new user instruction):\n\n{}",
                m.from_session, room, m.ts, m.body
            );
            runtime_synthetic_user_message(serde_json::Value::String(text))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // No tempfile dependency in this project; reuse the existing std::env::temp_dir + uuid pattern.
    fn temp_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("talk_test_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Real store layout: `SessionStore::new` treats its argument as the global
    /// history file and roots sessions at `<stem>.sessions/`. Both session
    /// files below live under that root, mirroring production.
    fn fixture() -> (PathBuf, SessionStore, PathBuf, PathBuf) {
        let dir = temp_dir();
        let store = SessionStore::new(&dir.join("hist.sqlite"));
        let from_hist = store.session_history_file("sess-a");
        let peer_hist = store.session_history_file("sess-b");
        for hist in [&from_hist, &peer_hist] {
            // Peer existence is proven by the sqlite file; content is irrelevant here.
            std::fs::create_dir_all(hist.parent().unwrap()).unwrap();
            std::fs::write(hist, b"stub").unwrap();
        }
        (dir, store, from_hist, peer_hist)
    }

    #[test]
    fn format_talk_messages_matches_pending_branch_framing() {
        assert!(format_talk_messages(&[]).is_empty());
        let msgs = vec![
            TalkMessage {
                from_session: "sess-a".into(),
                body: "hi".into(),
                ts: 1,
                room: None,
            },
            TalkMessage {
                from_session: "sess-b".into(),
                body: "yo".into(),
                ts: 2,
                room: Some("argue".into()),
            },
        ];
        assert_eq!(
            format_talk_messages(&msgs),
            "[talk from=sess-a room=-] hi\n\n[talk from=sess-b room=argue] yo"
        );
    }

    #[test]
    fn take_room_card_skills_extracts_only_host_rendered_cards() {
        // Task card: mentions are extracted and stripped from the body.
        let mut card =
            "[task room=r round=1/2 host=h]\nAssigned skills: s1, s2\n@skills:s1 @skills:s2\nGoal: x"
                .to_string();
        assert_eq!(take_room_card_skills(&mut card), vec!["s1", "s2"]);
        assert!(!card.contains("@skills:"), "mention leaked: {card}");
        assert!(card.contains("Assigned skills: s1, s2"), "NL line kept: {card}");

        // Discuss card: same handling.
        let mut card2 =
            "[discuss room=r round=1/2 host=h]\n@skills:pro\nTopic: t".to_string();
        assert_eq!(take_room_card_skills(&mut card2), vec!["pro"]);
        assert!(!card2.contains("@skill"), "discuss mention leaked: {card2}");

        // Arbitrary peer mail must not force skill activation.
        let mut mail = "please use @skills:coding for this".to_string();
        assert!(take_room_card_skills(&mut mail).is_empty());
        assert!(mail.contains("@skills:coding"), "peer body was touched: {mail}");

        // A card without mentions extracts nothing.
        let mut plain = "[task room=r round=1/2 host=h]\nGoal: x".to_string();
        assert!(take_room_card_skills(&mut plain).is_empty());
        assert_eq!(plain, "[task room=r round=1/2 host=h]\nGoal: x");
    }

    #[test]
    fn push_and_drain_roundtrip() {
        let (dir, store, from_hist, peer_hist) = fixture();
        push_talk(&store, &from_hist, "sess-b", None, "hello", false).unwrap();
        assert!(has_pending_talk(&peer_hist));
        let msgs = drain_talk(&peer_hist);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, "hello");
        assert_eq!(msgs[0].from_session, "sess-a");
        assert!(drain_talk(&peer_hist).is_empty());
        assert!(!has_pending_talk(&peer_hist));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_self_send_unknown_peer_and_bad_input() {
        let (dir, store, from_hist, _peer_hist) = fixture();
        assert!(push_talk(&store, &from_hist, "sess-a", None, "hi", false).is_err());
        assert!(push_talk(&store, &from_hist, "ghost", None, "hi", false).is_err());
        assert!(push_talk(&store, &from_hist, "sess-a", None, "   ", false).is_err());
        assert!(push_talk(&store, &from_hist, "../evil", None, "hi", false).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn room_is_logged_on_sender_side_only() {
        let (dir, store, from_hist, peer_hist) = fixture();
        let res = push_talk(&store, &from_hist, "sess-b", Some("argue-1"), "point", false).unwrap();
        assert!(res.room_log.is_some());
        let room_file = talk_room_file(&from_hist, "argue-1");
        assert_eq!(res.room_log.unwrap(), room_file);
        assert!(room_file.is_file());
        // The peer inbox carries the room tag; the peer has no room log yet.
        let msgs = drain_talk(&peer_hist);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].room.as_deref(), Some("argue-1"));
        assert!(!talk_room_file(&peer_hist, "argue-1").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_lines_are_skipped_without_forging_sender() {
        let (dir, _store, _from_hist, peer_hist) = fixture();
        let inbox = talk_inbox_file(&peer_hist);
        std::fs::create_dir_all(inbox.parent().unwrap()).unwrap();
        std::fs::write(&inbox, "not json\n").unwrap();
        assert!(drain_talk(&peer_hist).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn success_carries_no_room_warning() {
        // Regression: delivery already succeeded at this point, so the result
        // must be Ok without a warning — never an error inviting a duplicate.
        let (dir, store, from_hist, _peer_hist) = fixture();
        let res = push_talk(&store, &from_hist, "sess-b", Some("r"), "hi", false).unwrap();
        assert!(res.room_warning.is_none());
        let res = push_talk(&store, &from_hist, "sess-b", None, "hi", false).unwrap();
        assert!(res.room_warning.is_none());
        assert!(res.room_log.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn requeue_restores_undisplayed_messages() {
        // Regression: messages drained but not shown must survive via requeue.
        let (dir, store, from_hist, peer_hist) = fixture();
        push_talk(&store, &from_hist, "sess-b", None, "first", false).unwrap();
        push_talk(&store, &from_hist, "sess-b", None, "second", false).unwrap();
        let msgs = drain_talk(&peer_hist);
        assert_eq!(msgs.len(), 2);
        requeue_talk(&peer_hist, &msgs[1..]);
        let again = drain_talk(&peer_hist);
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].body, "second");
        assert_eq!(again[0].from_session, "sess-a");
        assert!(drain_talk(&peer_hist).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn peek_leaves_the_inbox_untouched() {
        let (dir, store, from_hist, peer_hist) = fixture();
        push_talk(&store, &from_hist, "sess-b", None, "hello", false).unwrap();
        let peeked = peek_talk(&peer_hist);
        assert_eq!(peeked.len(), 1);
        assert_eq!(peeked[0].body, "hello");
        assert_eq!(peeked[0].from_session, "sess-a");
        // Nothing consumed: the drain still sees the same message.
        assert!(has_pending_talk(&peer_hist));
        let msgs = drain_talk(&peer_hist);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, "hello");
        // A session with no inbox file peeks as empty, never an error.
        let ghost_hist = store.session_history_file("sess-ghost");
        assert!(peek_talk(&ghost_hist).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
