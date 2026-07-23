use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::ai::tools::storage::command_runner;
use crate::cmd::run::CommandRunResult;

const MAX_COMMAND_OUTPUT_CHARS: usize = 16_000;

/// 内置默认超时与上限（秒），可被 sandbox 配置覆盖。
const DEFAULT_COMMAND_TIMEOUT_SECS: u64 = 60;
const DEFAULT_COMMAND_TIMEOUT_MAX_SECS: u64 = 300;

/// 返回 `execute_command` 的 (默认超时, 超时上限)，由 sandbox 配置覆盖。
/// 非法/缺省值回退到内置常量；上限至少为 1 秒且不小于默认值。
fn config_command_timeout_bounds() -> (u64, u64) {
    let cfg = crate::commonw::configw::get_all_config();
    let default_timeout = cfg
        .get(AiConfig::SANDBOX_COMMAND_TIMEOUT_DEFAULT, "")
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|v| *v >= 1)
        .unwrap_or(DEFAULT_COMMAND_TIMEOUT_SECS);
    let max_timeout = cfg
        .get(AiConfig::SANDBOX_COMMAND_TIMEOUT_MAX, "")
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|v| *v >= 1)
        .unwrap_or(DEFAULT_COMMAND_TIMEOUT_MAX_SECS)
        .max(default_timeout);
    (default_timeout, max_timeout)
}

/// 纯函数：把请求的超时秒数夹在 `[1, max]` 范围内，缺省时用 `default`。
fn resolve_command_timeout(requested: Option<u64>, default: u64, max: u64) -> u64 {
    requested.unwrap_or(default).clamp(1, max)
}

/// 截断过长输出时同时保留头尾，并在中间附带**可操作的元信息**：总量、已显示量，
/// 以及一句明确警告——被省略的中段可能包含调用方要找的行，"没看到"不等于"不存在"。
///
/// 根因背景：`execute_command` 成功路径此前只裸截断加 `... (truncated)`，模型
/// 无法判断它要找的匹配是否被砍在了未显示部分，于是不断换姿势重试同一条
/// grep（history.json 的重复调用即源于此）。带上计数与分页提示后，重试动机
/// 从"信息不全的猜测"变成"有依据的收敛"。
fn truncate_chars(content: &str, max_chars: usize) -> String {
    let total_chars = content.chars().count();
    if total_chars <= max_chars {
        return content.to_string();
    }
    let total_lines = content.lines().count();
    let head_chars = (max_chars * 3 / 4).max(1);
    let tail_chars = max_chars.saturating_sub(head_chars);
    let head: String = content.chars().take(head_chars).collect();
    let tail: String = content
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let head_lines = head.lines().count();
    let tail_lines = tail.lines().count();
    let mut output = String::with_capacity(max_chars + 384);
    output.push_str(&head);
    output.push_str(&format!(
        "\n... [truncated: omitted middle; showing first {head_chars} and last {tail_chars} of {total_chars} chars \
(~{head_lines} + {tail_lines} of {total_lines} lines). The omitted middle is NOT shown — expected matches may be there, not absent. \
Do not re-run near-identical variants; narrow or page the result instead (e.g. `grep -c`, a more specific pattern, or `sed -n 'START,ENDp'`).]\n"
    ));
    output.push_str(&tail);
    output
}

// =========================================================================
// 执行逻辑（校验已移至 audit 模块）
// =========================================================================

fn format_command_result(output: CommandRunResult, timeout_secs: u64) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let stdout_trimmed = stdout.trim();
    let stderr_trimmed = stderr.trim();
    let combined = if stdout_trimmed.is_empty() {
        stderr_trimmed.to_string()
    } else if stderr_trimmed.is_empty() {
        stdout_trimmed.to_string()
    } else {
        format!("{stdout_trimmed}\n{stderr_trimmed}")
    };

    if output.timed_out || output.cancelled {
        let reason = if output.timed_out {
            format!("Command timed out after {timeout_secs}s and was terminated.")
        } else {
            "Command was cancelled and terminated.".to_string()
        };
        let partial = if combined.trim().is_empty() {
            "(no output was captured before termination)".to_string()
        } else {
            format!(
                "Partial output captured before termination:\n{}",
                combined.trim()
            )
        };
        return truncate_chars(&format!("{reason}\n{partial}"), MAX_COMMAND_OUTPUT_CHARS);
    }

    let status = output
        .status
        .expect("completed command must carry a status");
    if status.success() {
        let combined = combined.trim();
        // 空输出但成功退出：显式说明，避免模型把"命令成功、零匹配"误读为
        // "调用没生效"而反复重试同一条 grep。
        if combined.is_empty() {
            "(command succeeded with exit code 0 and produced no output)".to_string()
        } else {
            truncate_chars(combined, MAX_COMMAND_OUTPUT_CHARS)
        }
    } else {
        truncate_chars(
            &format!(
                "Exit code: {}\n{}\n{}",
                status.code().unwrap_or(-1),
                stdout_trimmed,
                stderr_trimmed
            ),
            MAX_COMMAND_OUTPUT_CHARS,
        )
    }
}

fn execute_command_inner<F>(args: &Value, on_chunk: F) -> Result<String, String>
where
    F: FnMut(&[u8]),
{
    let command = args["command"].as_str().ok_or("Missing command")?;
    let cwd = args["cwd"].as_str().filter(|dir| !dir.trim().is_empty());
    let pseudo_terminal = args["pty"].as_bool().unwrap_or(false);
    let (default_timeout, max_timeout) = config_command_timeout_bounds();
    let timeout = resolve_command_timeout(args["timeout"].as_u64(), default_timeout, max_timeout);

    // 命令安全校验委托给 audit 模块。
    super::audit::validate_execute_command(command)
        .map_err(|reason| format!("Command blocked: {reason}"))?;

    let output = command_runner::run_command_streaming(
        command,
        cwd,
        timeout,
        pseudo_terminal,
        on_chunk,
    )?;
    let interrupted = output.timed_out || output.cancelled;
    let formatted = format_command_result(output, timeout);
    if interrupted {
        Err(formatted)
    } else {
        Ok(formatted)
    }
}

pub(crate) fn execute_command(args: &Value) -> Result<String, String> {
    execute_command_inner(args, |_| {})
}

pub(crate) fn execute_command_streaming<F>(args: &Value, on_chunk: F) -> Result<String, String>
where
    F: FnMut(&[u8]),
{
    execute_command_inner(args, on_chunk)
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_COMMAND_OUTPUT_CHARS, format_command_result, resolve_command_timeout, truncate_chars,
    };
    use crate::cmd::run::CommandRunResult;

    // ---- truncate_chars ----

    #[test]
    fn truncate_passthrough_when_within_limit() {
        let s = "short output";
        assert_eq!(truncate_chars(s, MAX_COMMAND_OUTPUT_CHARS), s);
    }

    #[test]
    fn truncate_emits_actionable_metadata_when_over_limit() {
        // 1000 行，每行较短，整体远超小上限，触发截断。
        let content: String = (0..1000).map(|i| format!("line{i}\n")).collect();
        let out = truncate_chars(&content, 100);
        // 不再是无信息的 "... (truncated)"，而是带总量/已显示/分页提示。
        assert!(out.contains("truncated: omitted middle"), "out: {out}");
        assert!(out.contains("first 75 and last 25"), "out: {out}");
        assert!(out.contains("of 1000 lines"), "out: {out}");
        assert!(out.ends_with("line999\n"), "must preserve tail: {out}");
        assert!(
            out.contains("expected matches may be there, not absent"),
            "must warn that missing matches may be omitted, not absent"
        );
        assert!(
            out.contains("Do not re-run near-identical variants"),
            "must steer the model away from blind retries"
        );
    }

    #[test]
    fn timeout_result_keeps_partial_output_and_clear_reason() {
        let out = format_command_result(
            CommandRunResult {
                status: None,
                stdout: b"progress before timeout\n".to_vec(),
                stderr: b"last diagnostic\n".to_vec(),
                timed_out: true,
                cancelled: false,
            },
            30,
        );
        assert!(out.contains("timed out after 30s"), "out: {out}");
        assert!(out.contains("Partial output captured"), "out: {out}");
        assert!(out.contains("progress before timeout"), "out: {out}");
        assert!(out.contains("last diagnostic"), "out: {out}");
    }

    // ---- resolve_command_timeout ----

    #[test]
    fn timeout_uses_default_when_unset() {
        assert_eq!(resolve_command_timeout(None, 60, 300), 60);
    }

    #[test]
    fn timeout_clamps_to_max_and_floor() {
        assert_eq!(resolve_command_timeout(Some(10_000), 60, 300), 300);
        assert_eq!(resolve_command_timeout(Some(0), 60, 300), 1);
        assert_eq!(resolve_command_timeout(Some(120), 60, 300), 120);
    }
}
