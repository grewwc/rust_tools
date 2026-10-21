// Cross-session talk mailbox: `talk_send` / `talk_read` tool services.
//
// `talk_send` appends to the peer's `<id>.assets/talk/inbox.jsonl` and the
// sender's own room log only. It never opens the peer's sqlite database for
// writing, so no canonical history row is created or mutated in another
// session. `talk_read` drains the caller's own inbox.
use serde_json::Value;

use crate::ai::{driver::talk, history::SessionStore};

/// Model-visible output cap for one `talk_read` call. Inbound mail is fresh
/// work the model must see, so this is roomier than the session-inspection
/// cap; pathological inboxes are truncated with an explicit marker.
const MAX_READ_OUTPUT_CHARS: usize = 64_000;

fn current_ctx() -> Result<(SessionStore, std::path::PathBuf), String> {
    let ctx = crate::ai::driver::runtime_ctx::try_current().ok_or_else(|| {
        "talk tools need an active driver context (not available in one-shot/test calls)"
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

/// ToolSpec adapter: `ToolSpec.execute` returns model-visible text directly;
/// `ToolResult` wrapping is done by the registry path.
pub(crate) fn execute_talk_send(args: &Value) -> Result<String, String> {
    let to = arg_str(args, &["to_session", "to", "session_id", "recipient"])
        .ok_or_else(|| "to_session must not be empty (target session id)".to_string())?;
    let message = arg_str(args, &["message", "body", "content", "note"])
        .ok_or_else(|| "message must not be empty".to_string())?;
    let room = arg_str(args, &["room"]).map(str::to_string);

    let (store, from_history) = current_ctx()?;
    let res = talk::push_talk(&store, &from_history, to, room.as_deref(), message, false)?;
    let room_note = match &res.room_log {
        Some(_) => " (also recorded in the sender-side room log)",
        None => "",
    };
    let warn_note = match &res.room_warning {
        Some(w) => format!(" Warning: sender-side room log failed ({w}); the message WAS delivered — do not retry."),
        None => String::new(),
    };
    Ok(format!(
        "Delivered talk message to session '{to}' ({} chars){room_note}.{warn_note} It appears in the recipient's next turn; an inactive session sees it on resume.",
        message.chars().count()
    ))
}

pub(crate) fn execute_talk_read(args: &Value) -> Result<String, String> {
    let _ = args;
    let (_, history_file) = current_ctx()?;
    let msgs = talk::drain_talk(&history_file);
    if msgs.is_empty() {
        return Ok("No pending talk messages.".to_string());
    }
    let total = msgs.len();
    let mut out = String::new();
    // Whole-message cap: show a prefix that fits, then requeue the rest back
    // into the inbox so nothing drained is ever lost to truncation. A single
    // message always fits (send-side body limit is far below this cap), so at
    // least the first message is shown.
    let mut shown = 0;
    for (i, m) in msgs.iter().enumerate() {
        let room = m.room.as_deref().unwrap_or("-");
        let block = format!(
            "── [talk {}/{}] from={} room={} ts={} ──\n{}\n\n",
            i + 1,
            total,
            m.from_session,
            room,
            m.ts,
            m.body
        );
        if i > 0 && out.len() + block.len() > MAX_READ_OUTPUT_CHARS {
            break;
        }
        out.push_str(&block);
        shown = i + 1;
    }
    let leftover = msgs.len() - shown;
    if leftover > 0 {
        talk::requeue_talk(&history_file, &msgs[shown..]);
        out.push_str(&format!(
            "({shown}/{total} shown; {leftover} more message(s) kept in the inbox — run talk_read again to see them.)"
        ));
    } else {
        out.push_str(&format!(
            "({total} talk message(s) drained from the inbox.)"
        ));
    }
    Ok(out)
}
