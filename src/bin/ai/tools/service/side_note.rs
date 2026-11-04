// Lead-agent to subagent side-note tool service.
use serde_json::Value;

use crate::ai::{
    driver::{runtime_ctx, side_note},
    types::ToolResult,
};

/// ToolSpec adapter: `ToolSpec.execute` returns `Result<String, String>` with
/// model-visible text directly; the `ToolResult` wrapper is added by the
/// shared registry path.
pub(crate) fn execute_send_side_note(args: &Value) -> Result<String, String> {
    let note = args
        .get("note")
        .or_else(|| args.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if note.is_empty() {
        return Err("note/content must not be empty".into());
    }
    if note.chars().count() > 8_000 {
        return Err("side-note exceeds 8000 characters (single note limit)".into());
    }
    let target = args
        .get("task_id")
        .or_else(|| args.get("target_task_id"))
        .or_else(|| args.get("target"))
        .and_then(Value::as_str)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let foreground_history = runtime_ctx::try_current()
        .map(|ctx| ctx.app_proto.session_history_file.clone())
        .ok_or_else(|| {
            "cannot resolve current session history file (DRIVER_CTX missing)".to_string()
        })?;

    // Route the note to the queue the target actually drains. Unknown, foreign,
    // or already-finished targets fail fast so the caller does not mistake an
    // unconsumed queue file for a delivered note; foreground targets resolve to
    // the current history unchanged.
    let history_file = match target.as_deref() {
        Some(target_id) => crate::ai::tools::task_tools::resolve_side_note_history_file(
            target_id,
            &foreground_history,
        )?
        .unwrap_or(foreground_history),
        None => foreground_history,
    };
    let queued_for_continuation = target.as_deref().is_some_and(|target_id| {
        crate::ai::tools::task_tools::with_task_entry(target_id, |entry| {
            entry
                .backend
                .is_some_and(crate::ai::tools::task_tools::SubagentBackend::is_traecli)
        })
        .unwrap_or(false)
    });

    side_note::push_side_note(&history_file, &note, "lead", target.as_deref())
        .map_err(|e| e.to_string())?;

    let chars = note.chars().count();
    Ok(match target.as_deref() {
        Some(target_id) if queued_for_continuation => format!(
            "Queued side-note for traecli task `{target_id}` ({chars} chars); it will be prepended \
             to the next `task`/`task_spawn` prompt that resumes this session via \
             `backend_continue_from`. It does not interrupt the current run; queued notes are \
             retained after completion or cancellation until continuation."
        ),
        Some(target_id) => format!(
            "Queued side-note for task `{target_id}` ({chars} chars); it is injected into that \
             subagent's next model iteration, and is dropped if the task finishes first."
        ),
        None => format!(
            "Injected side-note into foreground ({chars} chars), visible in the next model iteration."
        ),
    })
}

pub fn handle_send_side_note(tool_call_id: &str, args: &Value) -> Result<ToolResult, String> {
    let content = execute_send_side_note(args)?;
    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content,
    })
}
