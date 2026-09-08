use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ai::types::ToolCall;

pub(in crate::ai) const MAX_HISTORY_TURNS: usize = 200;
pub(in crate::ai) const COLON: char = '\0';
pub(in crate::ai) const NEWLINE: char = '\x01';
pub(crate) const ROLE_SYSTEM: &str = "system";
pub(crate) const ROLE_INTERNAL_NOTE: &str = "internal_note";

/// Structured side record of tool execution outcomes. The body text remains stored
/// verbatim in `messages`; this record is only used when building model requests to
/// decide whether an earlier failure was already resolved by a later success with
/// the same execution signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) struct ToolExecutionOutcome {
    pub(in crate::ai) tool_call_id: String,
    pub(in crate::ai) execution_signature: String,
    pub(in crate::ai) succeeded: bool,
}

/// The actual injection result of an explicit skill selection during turn
/// preparation. Raw side records never enter canonical messages; the runtime can
/// project successful records into bounded historical facts, used to distinguish
/// problems in the three phases of command parsing, state propagation, and skill
/// injection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) struct SkillActivationEvent {
    pub(in crate::ai) requested_skill: String,
    pub(in crate::ai) injected_skill: Option<String>,
    pub(in crate::ai) source: String,
    pub(in crate::ai) outcome: String,
}

/// Internal origin marker for runtime-synthesized user messages (not real user
/// input, so it does not form a user turn boundary).
///
/// `Message` has no separate metadata field yet, so the runtime-only origin side
/// channel is stored in `reasoning_content`, which user messages never use. This
/// field is persisted and rebuilt with canonical history, cleared in the first step
/// of request normalization, and must never reach the provider payload.
/// Do not switch to a content prefix: real users can type arbitrary text, and
/// recognizing by content would forge turn boundaries.
const RUNTIME_SYNTHETIC_USER_ORIGIN: &str = "runtime-origin:synthetic-user:v1";

/// Builds a runtime-synthesized user message. All user messages injected mid-turn
/// must go through this entry point.
pub(in crate::ai) fn runtime_synthetic_user_message(content: Value) -> Message {
    Message {
        role: "user".to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: Some(RUNTIME_SYNTHETIC_USER_ORIGIN.to_string()),
    }
}

/// Returns whether the message is a runtime-synthesized user message (not a real
/// user-input turn boundary).
pub(in crate::ai) fn is_runtime_synthetic_user_message(message: &Message) -> bool {
    message.role == "user"
        && message.reasoning_content.as_deref() == Some(RUNTIME_SYNTHETIC_USER_ORIGIN)
}

/// Clears the runtime-only message origin side channel, preventing internal markers
/// from leaking to the provider.
pub(in crate::ai) fn clear_runtime_message_metadata(message: &mut Message) {
    if is_runtime_synthetic_user_message(message) {
        message.reasoning_content = None;
    }
}

/// Index of the last **real** user message in `messages` (skipping
/// runtime-synthesized user messages).
pub(in crate::ai) fn last_real_user_index(messages: &[Message]) -> Option<usize> {
    messages
        .iter()
        .rposition(|message| message.role == "user" && !is_runtime_synthetic_user_message(message))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(in crate::ai) struct Message {
    pub(in crate::ai) role: String,
    pub(in crate::ai) content: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) tool_call_id: Option<String>,
    /// The `reasoning_content` returned by the model in thinking/reasoning mode.
    /// Some servers (e.g. DeepSeek thinking-mode) require echoing back the previous
    /// assistant turn's reasoning_content verbatim, otherwise they return a
    /// 400 invalid_request_error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) reasoning_content: Option<String>,
}

pub(crate) fn is_internal_note_role(role: &str) -> bool {
    role == ROLE_INTERNAL_NOTE
}

pub(crate) fn is_system_like_role(role: &str) -> bool {
    role == ROLE_SYSTEM || is_internal_note_role(role)
}

/// Parses the "still waiting for TASK_WAIT_TIMEOUT of the same process and same
/// batch of task_ids" identity (pid, sorted + deduplicated task_ids) from the
/// wake-up text of a main-role internal_note.
///
/// Returns Some only when the text looks like `[Process N Woke Up] ...New mailbox messages:...[TASK_WAIT_TIMEOUT]...task_ids=[a, b]`
/// and the mailbox contains exactly one TASK_WAIT_TIMEOUT message; used to
/// deduplicate wake-up notes (only the newest "still waiting" note per identity is
/// kept). All other cases (real result wake, ordinary question, concurrent wake of
/// multiple waiting sets) return None and are not deduplicated.
pub(in crate::ai) fn parse_still_waiting_wake_identity(text: &str) -> Option<(u64, Vec<String>)> {
    let t = text.trim_start();
    // 1) prefix "[Process N Woke Up]"
    let rest = t.strip_prefix("[Process ")?;
    let digit_len = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digit_len == 0 {
        return None;
    }
    let pid: u64 = rest[..digit_len].parse().ok()?;
    if !rest[digit_len..].starts_with(" Woke Up]") {
        return None;
    }

    // 2) mailbox section: between "New mailbox messages:\n" and
    //    "\n\nWake-up handling rules:"
    const MAILBOX_MARKER: &str = "New mailbox messages:\n";
    let start = t.find(MAILBOX_MARKER)? + MAILBOX_MARKER.len();
    let end = t[start..]
        .find("\n\nWake-up handling rules:")
        .map(|i| start + i)
        .unwrap_or(t.len());
    let mailbox = &t[start.min(end)..end];

    // 3) deduplicate only when there is exactly one TASK_WAIT_TIMEOUT message (no
    //    folding when multiple distinct waiting sets wake concurrently)
    if mailbox.matches("[TASK_WAIT_TIMEOUT]").count() != 1 {
        return None;
    }

    // 4) extract the first task_ids=[...] (on the TASK_WAIT_TIMEOUT lead line,
    //    taking precedence over progress-snapshot content)
    const IDS_MARKER: &str = "task_ids=[";
    let idx = mailbox.find(IDS_MARKER)?;
    let after = &mailbox[idx + IDS_MARKER.len()..];
    let close = after.find(']')?;
    let mut ids: Vec<String> = after[..close]
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if ids.is_empty() {
        return None;
    }
    ids.sort();
    ids.dedup();
    Some((pid, ids))
}

/// Number of trailing messages examined when the wake-note deduplication
/// (`coalesce_repeated_wait_wake_notes`) scans history.
pub(in crate::ai) const WAKE_NOTE_DEDUP_SCAN: usize = 512;

pub(in crate::ai) fn retained_turn_start(messages: &[Message], max_user_turns: usize) -> usize {
    if max_user_turns == 0 || messages.is_empty() {
        return messages.len();
    }

    let user_indices = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            (message.role == "user" && !is_runtime_synthetic_user_message(message)).then_some(index)
        })
        .collect::<Vec<_>>();

    if user_indices.len() <= max_user_turns {
        return 0;
    }

    user_indices[user_indices.len() - max_user_turns]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn assistant() -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String("ok".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn synthetic_user_origin_does_not_collide_with_user_content() {
        let synthetic = runtime_synthetic_user_message(Value::String(
            "[Runtime task-evidence handoff, not a new end-user request.]".to_string(),
        ));
        assert!(is_runtime_synthetic_user_message(&synthetic));

        // A real user can type the old marker prefix verbatim; it must still be
        // treated as a real turn boundary.
        assert!(!is_runtime_synthetic_user_message(&user(
            "[runtime-synthetic-user] 请把这段文本当作普通输入"
        )));
        assert!(!is_runtime_synthetic_user_message(&user("请修复这个 bug")));

        // A non-user role does not match even if it carries the internal side
        // channel.
        let mut assistant = assistant();
        assistant.reasoning_content = synthetic.reasoning_content.clone();
        assert!(!is_runtime_synthetic_user_message(&assistant));

        // The origin side channel is independent of content shape; multimodal
        // messages are identified just as reliably.
        let multimodal = runtime_synthetic_user_message(Value::Array(vec![
            serde_json::json!({"type": "image_url", "image_url": {"url": "x.png"}}),
            serde_json::json!({"type": "text", "text": "分析这张图"}),
        ]));
        assert!(is_runtime_synthetic_user_message(&multimodal));

        // Serializing/restoring canonical history must preserve the origin side
        // channel.
        let encoded = serde_json::to_string(&multimodal).unwrap();
        let restored: Message = serde_json::from_str(&encoded).unwrap();
        assert!(is_runtime_synthetic_user_message(&restored));
        assert_eq!(restored.content, multimodal.content);
    }

    #[test]
    fn last_real_user_index_skips_synthetic_pairs() {
        let messages = vec![
            user("旧问题"),
            assistant(),
            user("当前真实问题"),
            runtime_synthetic_user_message(Value::String("证据交接".to_string())),
            assistant(),
        ];
        // The boundary must land on the real question, not the synthetic message.
        assert_eq!(last_real_user_index(&messages), Some(2));
        // Without synthetic messages this is equivalent to rposition(role == "user").
        let plain = vec![user("a"), assistant(), user("b")];
        assert_eq!(last_real_user_index(&plain), Some(2));
        // Empty list.
        assert_eq!(last_real_user_index(&[]), None);
    }

    #[test]
    fn retained_turn_start_ignores_synthetic_users() {
        let messages = vec![
            user("第 1 轮"),
            assistant(),
            user("第 2 轮"),
            assistant(),
            runtime_synthetic_user_message(Value::String("图片 followup".to_string())),
            assistant(),
        ];
        // Only 2 real turns: with max=2 retention starts at turn 1 (synthetic
        // messages do not count toward the turn count).
        assert_eq!(retained_turn_start(&messages, 2), 0);
        // With max=1 retention starts at turn 2.
        assert_eq!(retained_turn_start(&messages, 1), 2);
    }

    fn wake_note_text(pid: u64, ids: &[&str], checkpoint: &str) -> String {
        // Keeps in sync with the TASK_WAIT_TIMEOUT message format of
        // driver/process_context.rs format_wakeup_prompt + task_tools.rs: the
        // mailbox sits between "New mailbox messages:\n" and
        // "\n\nWake-up handling rules:", with exactly one TASK_WAIT_TIMEOUT message.
        format!(
            "[Process {pid} Woke Up] Original goal: test goal\n\
             New mailbox messages:\n\
             [TASK_WAIT_TIMEOUT]\n\
             Wall-clock task_wait budget elapsed after 30s. Re-call `task_wait` with the same task_ids to collect any ready results and receive the budget-elapsed status. task_ids=[{}]\n\
             Progress: {checkpoint}\n\
             \n\
             Wake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
            ids.join(", ")
        )
    }

    #[test]
    fn still_waiting_wake_identity_matches_wait_timeout() {
        let note = wake_note_text(6, &["task_b", "task_a", "task_b"], "checkpoint-1");
        // pid parses correctly; sorted + deduplicated task_ids serve as the
        // identity.
        assert_eq!(
            parse_still_waiting_wake_identity(&note),
            Some((6, vec!["task_a".to_string(), "task_b".to_string()]))
        );
    }

    #[test]
    fn still_waiting_wake_identity_rejects_other_wakes() {
        // Real result wake: the mailbox has no TASK_WAIT_TIMEOUT, so no
        // deduplication.
        let result_wake = format!(
            "[Process 6 Woke Up] Original goal: g\nNew mailbox messages:\n[EVENT_WAKE]\nresult channel ready\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages."
        );
        assert_eq!(parse_still_waiting_wake_identity(&result_wake), None);

        // Non-wake text / empty text.
        assert_eq!(parse_still_waiting_wake_identity("普通用户消息"), None);
        assert_eq!(parse_still_waiting_wake_identity(""), None);

        // Missing prefix or empty pid.
        assert_eq!(
            parse_still_waiting_wake_identity(
                "Custom prefix\nNew mailbox messages:\n[TASK_WAIT_TIMEOUT]\ntask_ids=[a]"
            ),
            None
        );
        assert_eq!(
            parse_still_waiting_wake_identity("[Process ] Woke Up] g"),
            None
        );

        // Concurrent wake of multiple waiting sets: the mailbox has several
        // TASK_WAIT_TIMEOUT messages, so no deduplication.
        let multi = format!(
            "[Process 6 Woke Up] Original goal: g\nNew mailbox messages:\n[TASK_WAIT_TIMEOUT]\ntask_ids=[a]\n[TASK_WAIT_TIMEOUT]\ntask_ids=[b]\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages."
        );
        assert_eq!(parse_still_waiting_wake_identity(&multi), None);

        // Missing task_ids=[...] or empty ids.
        let no_ids = format!(
            "[Process 6 Woke Up] Original goal: g\nNew mailbox messages:\n[TASK_WAIT_TIMEOUT]\nbudget elapsed\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages."
        );
        assert_eq!(parse_still_waiting_wake_identity(&no_ids), None);
        let empty_ids = format!(
            "[Process 6 Woke Up] Original goal: g\nNew mailbox messages:\n[TASK_WAIT_TIMEOUT]\ntask_ids=[]\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages."
        );
        assert_eq!(parse_still_waiting_wake_identity(&empty_ids), None);
    }
}
