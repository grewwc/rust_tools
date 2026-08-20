// lead-agent -> subagent 实时 side-note 工具服务
use serde_json::Value;

use crate::ai::{
    driver::{runtime_ctx, side_note},
    types::ToolResult,
};

/// ToolSpec 适配：`ToolSpec.execute` 签名是 `fn(&Value) -> Result<String, String>`，
/// 直接返回模型可见的成功文案，`ToolResult` 的包装由 registry 通用路径完成。
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

    let history_file = runtime_ctx::try_current()
        .map(|ctx| ctx.app_proto.session_history_file.clone())
        .ok_or_else(|| "cannot resolve current session history file (DRIVER_CTX missing)".to_string())?;

    side_note::push_side_note(&history_file, &note, "lead", target.as_deref())
        .map_err(|e| e.to_string())?;

    let target_desc = target.as_deref().unwrap_or("foreground");
    Ok(format!(
        "Injected side-note into {target_desc} ({} chars), visible in the next model iteration.",
        note.chars().count()
    ))
}

pub fn handle_send_side_note(
    tool_call_id: &str,
    args: &Value,
) -> Result<ToolResult, String> {
    let content = execute_send_side_note(args)?;
    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content,
    })
}
