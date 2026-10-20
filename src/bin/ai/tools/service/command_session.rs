//! Control-plane service for persistent command sessions.
//!
//! `execute_command` starts a process and may return a session id; every later
//! interaction with that process goes through this tool. Keeping the operations
//! in one place with a required `action` avoids inferring intent from which
//! optional fields happen to be present.

use serde_json::Value;

use super::super::storage::command_session;

/// Default time a session call waits for new output before yielding. Each poll
/// is one full model request, so a longer default trades latency for fewer
/// rounds.
pub(crate) const DEFAULT_SESSION_YIELD_MS: u64 = 30_000;
/// Floor for one session yield: polling faster than this spends a model round
/// per call without giving the process time to produce new output.
pub(crate) const MIN_SESSION_YIELD_MS: u64 = 5_000;
/// Upper bound for one session yield; keeps one tool call from blocking too long.
pub(crate) const MAX_SESSION_YIELD_MS: u64 = 60_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandSessionAction {
    Poll,
    Write,
    Close,
}

impl CommandSessionAction {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "poll" => Ok(Self::Poll),
            "write" => Ok(Self::Write),
            "close" => Ok(Self::Close),
            other => Err(format!(
                "Invalid `action` {other:?}; expected one of: poll, write, close"
            )),
        }
    }
}

/// Treat a missing, null, or empty string argument as absent. A whitespace-only
/// value remains meaningful for `write_stdin`, where a space is valid input.
pub(crate) fn non_empty_str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

pub(crate) fn resolve_session_yield(value: u64) -> u64 {
    value.clamp(MIN_SESSION_YIELD_MS, MAX_SESSION_YIELD_MS)
}

pub(crate) fn execute_command_session(args: &Value) -> Result<String, String> {
    let raw_action = non_empty_str_arg(args, "action")
        .ok_or_else(|| "Missing `action`; expected one of: poll, write, close".to_string())?;
    let action = CommandSessionAction::parse(raw_action)?;
    let session_id = non_empty_str_arg(args, "session_id")
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| {
            "Missing non-empty `session_id`; use the id returned when `execute_command` \
             started a persistent session"
                .to_string()
        })?;
    let write_stdin = non_empty_str_arg(args, "write_stdin");
    let yield_ms = args
        .get("yield_time_ms")
        .and_then(Value::as_u64)
        .map(resolve_session_yield)
        .unwrap_or(DEFAULT_SESSION_YIELD_MS);

    match action {
        CommandSessionAction::Close => {
            if write_stdin.is_some() {
                return Err(
                    "`write_stdin` is only valid with action=write; omit it for action=close"
                        .to_string(),
                );
            }
            command_session::close(session_id)
        }
        CommandSessionAction::Write => {
            let bytes = write_stdin.ok_or_else(|| {
                "action=write requires non-empty `write_stdin`; use action=poll to only wait"
                    .to_string()
            })?;
            command_session::write_stdin(session_id, bytes, yield_ms)
        }
        CommandSessionAction::Poll => {
            if write_stdin.is_some() {
                return Err(
                    "`write_stdin` is only valid with action=write; use action=write to send input"
                        .to_string(),
                );
            }
            command_session::poll(session_id, yield_ms)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::tools::service::command::execute_command;
    use serde_json::json;

    fn test_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::ai::tools::storage::command_session::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn session_id_of(rendered: &str) -> &str {
        rendered
            .lines()
            .find_map(|line| line.strip_prefix("[session "))
            .and_then(|line| line.strip_suffix(']'))
            .expect("session id")
    }

    #[test]
    fn action_is_required_and_write_requires_non_empty_stdin() {
        let missing_action = execute_command_session(&json!({"session_id": "cmd-1"}))
            .expect_err("action is required");
        assert!(
            missing_action.contains("Missing `action`"),
            "err: {missing_action}"
        );

        let bad_action = execute_command_session(&json!({
            "action": "resume",
            "session_id": "cmd-1",
        }))
        .expect_err("unknown action must be rejected");
        assert!(bad_action.contains("Invalid `action`"), "err: {bad_action}");

        let empty_write = execute_command_session(&json!({
            "action": "write",
            "session_id": "cmd-1",
            "write_stdin": "",
        }))
        .expect_err("write with empty stdin must be rejected");
        assert!(
            empty_write.contains("requires non-empty"),
            "err: {empty_write}"
        );

        let poll_with_write = execute_command_session(&json!({
            "action": "poll",
            "session_id": "cmd-1",
            "write_stdin": "x\n",
        }))
        .expect_err("poll must reject stdin bytes");
        assert!(
            poll_with_write.contains("action=write"),
            "err: {poll_with_write}"
        );
    }

    #[test]
    fn session_yield_defaults_and_bounds_avoid_tight_polling() {
        assert_eq!(DEFAULT_SESSION_YIELD_MS, 30_000);
        assert_eq!(MIN_SESSION_YIELD_MS, 5_000);
        assert_eq!(MAX_SESSION_YIELD_MS, 60_000);
        assert_eq!(resolve_session_yield(1), MIN_SESSION_YIELD_MS);
        assert_eq!(resolve_session_yield(5_000), 5_000);
        assert_eq!(resolve_session_yield(30_000), DEFAULT_SESSION_YIELD_MS);
        assert_eq!(resolve_session_yield(u64::MAX), MAX_SESSION_YIELD_MS);
    }

    #[test]
    fn session_round_trip_uses_explicit_actions() {
        let _guard = test_guard();
        let started = execute_command(&json!({
            "command": "read line; echo GOT:$line; sleep 30",
            "pty": false,
            "yield_time_ms": 500,
        }))
        .expect("session should start");
        let session_id = session_id_of(&started).to_string();

        // Empty placeholders on a poll are treated as absent, not as a write.
        execute_command_session(&json!({
            "action": "poll",
            "session_id": session_id,
            "write_stdin": "",
            "yield_time_ms": 100,
        }))
        .expect("poll with an empty placeholder should be accepted");

        let mut combined = execute_command_session(&json!({
            "action": "write",
            "session_id": session_id,
            "write_stdin": "hello\n",
            "yield_time_ms": 5000,
        }))
        .expect("write should be accepted");
        if !combined.contains("GOT:hello") {
            combined.push_str(
                &execute_command_session(&json!({
                    "action": "poll",
                    "session_id": session_id,
                    "yield_time_ms": 5000,
                }))
                .expect("poll should be accepted"),
            );
        }
        assert!(combined.contains("GOT:hello"), "out: {combined}");

        let closed = execute_command_session(&json!({
            "action": "close",
            "session_id": session_id,
        }))
        .expect("close should be accepted");
        assert!(closed.contains("closed"), "out: {closed}");

        let err = execute_command_session(&json!({
            "action": "poll",
            "session_id": session_id,
        }))
        .expect_err("closed sessions must disappear");
        assert!(err.contains("Unknown command session"), "err: {err}");
    }
}