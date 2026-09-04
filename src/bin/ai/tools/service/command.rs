use serde_json::Value;
use std::{
    fs::File,
    io::{IsTerminal, Read},
    path::Path,
};

use crate::ai::config_schema::AiConfig;
use crate::ai::tools::storage::command_runner;
use crate::cmd::run::CommandRunResult;

const MAX_COMMAND_OUTPUT_CHARS: usize = 16_000;
/// 将 `$(cat /absolute/literal/path)` 物化为一个普通 shell 参数时的读取上限。
/// 限制可防止工具在校验之前意外读取无限流或超大文件；常规 JSON / DSL 参数远小于此值。
const MAX_LITERAL_FILE_SUBSTITUTION_BYTES: usize = 64 * 1024;

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

/// 把 UTF-8 数据编码成一个完整的 POSIX shell 单词。单引号内不发生展开；遇到单引号时
/// 用 `'<backslash><quote>'` 过渡到下一段单引号字面量，文件内容不会成为 shell 代码。
fn shell_single_quote(value: &str) -> String {
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('\'');
    for ch in value.chars() {
        if ch == '\'' {
            quoted.push_str("'\\''");
        } else {
            quoted.push(ch);
        }
    }
    quoted.push('\'');
    quoted
}

fn read_literal_file_substitution(path: &str) -> Result<String, String> {
    let file = File::open(path)
        .map_err(|err| format!("cannot open literal file substitution '{path}': {err}"))?;
    let metadata = file
        .metadata()
        .map_err(|err| format!("cannot stat literal file substitution '{path}': {err}"))?;
    if !metadata.is_file() {
        return Err(format!(
            "literal file substitution '{path}' must name a regular file"
        ));
    }
    if metadata.len() > MAX_LITERAL_FILE_SUBSTITUTION_BYTES as u64 {
        return Err(format!(
            "literal file substitution '{path}' exceeds the {}-byte limit",
            MAX_LITERAL_FILE_SUBSTITUTION_BYTES
        ));
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    let mut limited = file.take((MAX_LITERAL_FILE_SUBSTITUTION_BYTES + 1) as u64);
    limited
        .read_to_end(&mut bytes)
        .map_err(|err| format!("cannot read literal file substitution '{path}': {err}"))?;
    if bytes.len() > MAX_LITERAL_FILE_SUBSTITUTION_BYTES {
        return Err(format!(
            "literal file substitution '{path}' exceeds the {}-byte limit",
            MAX_LITERAL_FILE_SUBSTITUTION_BYTES
        ));
    }
    let contents = String::from_utf8(bytes)
        .map_err(|_| format!("literal file substitution '{path}' must be valid UTF-8"))?;
    if contents.contains('\0') {
        return Err(format!(
            "literal file substitution '{path}' must not contain NUL bytes"
        ));
    }
    Ok(contents)
}

/// 执行无害命令替换的内部命令并捕获输出，用于物化 `"$(harmless_cmd)"`。
fn execute_inner_shell_command(inner: &str, cwd: Option<&str>) -> Result<String, String> {
    // 复用现有 runner，超时 10s 足以覆盖 date/echo/git 等短命令，又避免长阻塞
    let output = crate::ai::tools::storage::command_runner::run_command(inner, cwd, 10)
        .map_err(|err| format!("failed to execute substitution '{inner}': {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "substitution command '{inner}' failed with exit code {}",
            output.status.code().unwrap_or(-1)
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| format!("substitution '{inner}' produced non-UTF-8 output"))?;
    if stdout.len() > MAX_LITERAL_FILE_SUBSTITUTION_BYTES {
        return Err(format!(
            "substitution '{inner}' output exceeds the {}-byte limit",
            MAX_LITERAL_FILE_SUBSTITUTION_BYTES
        ));
    }
    if stdout.contains('\0') {
        return Err(format!("substitution '{inner}' output contains NUL bytes"));
    }
    // bash 的 $(...) 会剥离所有末尾换行，这里复刻该语义
    Ok(stdout
        .trim_end_matches(|c| c == '\n' || c == '\r')
        .to_string())
}

/// 对审计层已证明安全的无害 `"$(...)"` 做数据物化，支持 `cat` 字面量与通用无害命令。
/// 随后仍对替换后的命令做完整审计，因此替换结果若成为被禁程序名或危险参数仍会被拦截。
fn materialize_safe_shell_substitutions(
    command: &str,
    cwd: Option<&str>,
) -> Result<String, String> {
    let substitutions = super::audit::safe_shell_substitutions(command);
    if substitutions.is_empty() {
        // 回退到旧的 cat 物化以兼容仅含 cat 的历史路径（实际上 safe_shell 已覆盖 cat，此分支仅为无替换时的快速返回）
        return Ok(command.to_string());
    }
    let mut materialized = command.to_string();
    for substitution in substitutions.into_iter().rev() {
        let contents = match substitution.kind {
            super::audit::SafeShellSubstitutionKind::FileRead { path } => {
                read_literal_file_substitution(&path)?
            }
            super::audit::SafeShellSubstitutionKind::Command { inner } => {
                // 内层同样需经过完整的门禁：先重做 validate（防分类放宽），再做 git commit 确认（fail-closed）。
                // 审计算法的 Command 分类仅保证内层曾通过 validate，但不拦截 git commit；若此处
                // 不单独确认，则 `echo "$(git commit -am x)"` 会在外层确认前已提交，物化后外层
                // 仅剩 `echo '...'`，导致确认门被绕过。
                super::audit::validate_execute_command(&inner)
                    .map_err(|reason| format!("inner substitution blocked: {reason}"))?;
                confirm_git_commit_if_needed(&inner)?;
                execute_inner_shell_command(&inner, cwd)?
            }
        };
        materialized.replace_range(
            substitution.start..substitution.end,
            &shell_single_quote(&contents),
        );
    }
    Ok(materialized)
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
(~{head_lines} + {tail_lines} of {total_lines} lines); expected matches may be there, not absent. \
Do not re-run near-identical variants; narrow the query instead (e.g. `grep -c` or a more specific pattern). To page a local file, prefer `read_file` with offset/limit; `sed -n 'START,ENDp'` only for line paging, `head -c`/`tail -c +N` for byte windows or non-text files.]\n"
    ));
    output.push_str(&tail);
    output
}

// =========================================================================
// 执行逻辑（校验已移至 audit 模块）
// =========================================================================

/// 判断命令是否为 git 提交类命令（`git commit` / `git -C <dir> commit` 等），
/// 命中后需要先向用户确认再执行。复用审计层词法解析，保证物化后的带引号参数也不会
/// 绕过确认，并避免把 `echo 'git commit'` 之类的数据误判为真实命令。
fn is_git_commit_command(command: &str) -> bool {
    for segment in super::audit::split_unquoted_segments(command) {
        let tokens = super::audit::effective_command_tokens(&segment);
        let Some(program) = tokens
            .first()
            .and_then(|token| Path::new(token).file_name().and_then(|name| name.to_str()))
        else {
            continue;
        };
        if program != "git" {
            continue;
        }

        // 跳过 git 全局选项及其取值（-C <path>、-c <key>=<val>、--git-dir=... 等），
        // 它们可能出现在 `git` 与子命令之间。
        let mut j = 1usize;
        loop {
            match tokens.get(j).map(String::as_str) {
                Some(tok)
                    if tok == "-C"
                        || tok == "-c"
                        || tok == "--git-dir"
                        || tok == "--work-tree"
                        || tok == "--namespace" =>
                {
                    j += 2
                }
                Some(tok)
                    if tok.starts_with("--git-dir=")
                        || tok.starts_with("--work-tree=")
                        || tok.starts_with("--namespace=") =>
                {
                    j += 1
                }
                _ => break,
            }
        }
        if tokens.get(j).map(String::as_str) == Some("commit") {
            return true;
        }
    }
    false
}

/// git 提交前请求用户确认。
/// - 交互终端：红色高亮提示，y 放行，n / Ctrl+C / Esc 取消。
/// - 非交互环境：直接拒绝（fail-closed），既避免后台进程挂在读输入上，也避免静默提交。
fn confirm_git_commit_if_needed(command: &str) -> Result<(), String> {
    if !is_git_commit_command(command) {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        return Err(
            "Command blocked: git commit requires user confirmation, but stdin is not an \
             interactive terminal. Do not retry the commit; report to the user and wait for \
             explicit confirmation (or have them run it in an interactive session)."
                .to_string(),
        );
    }
    let confirmed = crate::commonw::prompt::prompt_yes_or_no_danger(&format!(
        "\nConfirm git commit:\n{command}\nProceed? (y/n): "
    ));
    match confirmed {
        Some(true) => Ok(()),
        Some(false) => Err("git commit canceled by user".to_string()),
        None => Err("git commit canceled by user (Ctrl+C)".to_string()),
    }
}

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

    if output.stalled {
        // PTY 交互命令停滞：命令存活但长时间没有输出，几乎可以断定它在等待人类输入
        // （扫码、密码、菜单），agent 无法提供输入；输出被管道（如 `| tail`）缓冲时
        // 更是从头到尾都看不到。给出明确诊断 + 已捕获的部分输出（典型：二维码），
        // 而不是让模型把结果误解为普通超时后盲目重试同一条命令。
        let partial = if combined.trim().is_empty() {
            "(no output was captured before termination)".to_string()
        } else {
            format!(
                "Partial output captured before termination:\n{}",
                combined.trim()
            )
        };
        let msg = "Command appears to be waiting for interactive input and was terminated: it kept running without producing output for a sustained period. This usually means the command is interactive — e.g. a QR-code login, a password prompt, or a menu — and cannot proceed without human input; or its output is buffered by a pipe (like `| tail`), which only flushes when the command exits, so nothing was visible. If the command is a long-running server or daemon, run it in the background instead (append `&` and redirect output to a log file). For login flows, prefer the CLI's non-blocking options (e.g. `--begin`/`--complete`, `--qr-image`, `--no-terminal-qr`, `-y`).";
        return truncate_chars(&format!("{msg}\n{partial}"), MAX_COMMAND_OUTPUT_CHARS);
    }

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
    let raw_command = args["command"].as_str().ok_or("Missing command")?;
    let cwd = args["cwd"].as_str().filter(|dir| !dir.trim().is_empty());
    // 优先物化无害的 "$(...)"（含 cat 字面量与通用无害命令），物化后仍做完整审计
    let command = materialize_safe_shell_substitutions(raw_command, cwd)
        .map_err(|reason| format!("Command blocked: {reason}"))?;
    let pseudo_terminal = args["pty"].as_bool().unwrap_or(false);
    let (default_timeout, max_timeout) = config_command_timeout_bounds();
    let timeout = resolve_command_timeout(args["timeout"].as_u64(), default_timeout, max_timeout);

    // 命令安全校验委托给 audit 模块。
    super::audit::validate_execute_command(&command)
        .map_err(|reason| format!("Command blocked: {reason}"))?;

    // git 提交类命令先向用户确认（非交互环境 fail-closed）。
    confirm_git_commit_if_needed(&command)?;

    let output =
        command_runner::run_command_streaming(&command, cwd, timeout, pseudo_terminal, on_chunk)?;
    let interrupted = output.timed_out || output.cancelled || output.stalled;
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
        MAX_COMMAND_OUTPUT_CHARS, confirm_git_commit_if_needed, execute_command,
        format_command_result, is_git_commit_command, resolve_command_timeout, truncate_chars,
    };
    use crate::cmd::run::CommandRunResult;
    use serde_json::json;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temporary_file_path(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rust_tools_{label}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    // ---- is_git_commit_command ----

    #[test]
    fn commit_detection_matches_plain_git_commit() {
        assert!(is_git_commit_command("git commit -m \"fix x\""));
        assert!(is_git_commit_command("git commit"));
        assert!(is_git_commit_command("git commit --amend"));
        assert!(is_git_commit_command("git 'commit' -m message"));
    }

    #[test]
    fn commit_detection_skips_global_options() {
        assert!(is_git_commit_command("git -C /repo commit -m x"));
        assert!(is_git_commit_command("git --git-dir=/repo/.git commit"));
        assert!(is_git_commit_command("git -c user.name=X commit"));
    }

    #[test]
    fn commit_detection_finds_commit_in_command_chains() {
        assert!(is_git_commit_command("git add -A && git commit -m x"));
        assert!(is_git_commit_command(
            "git -C /repo add . && git -C /repo commit"
        ));
    }

    #[test]
    fn commit_detection_ignores_non_commit_commands() {
        assert!(!is_git_commit_command("git status"));
        assert!(!is_git_commit_command("git log --oneline | grep commit"));
        assert!(!is_git_commit_command("git svn commit"));
        assert!(!is_git_commit_command("echo 'git commit' > note.txt"));
        assert!(!is_git_commit_command("git commitmessage"));
    }

    #[test]
    fn commit_confirmation_fails_closed_without_terminal() {
        // 测试环境 stdin 非终端：提交类命令必须被拒绝，且不挂起。
        let err = confirm_git_commit_if_needed("git commit -m x").unwrap_err();
        assert!(err.contains("blocked"), "err: {err}");
        assert!(err.contains("confirmation"), "err: {err}");
    }

    #[test]
    fn execute_command_blocks_git_commit_inside_substitution_without_terminal() {
        // P0 回归：非 TTY 下 `echo "$(git commit ...)"` 必须被 fail-closed 拦截，
        // 不能通过命令替换绕过确认门禁静默执行。
        let err = execute_command(&json!({
            "command": r#"echo "$(git commit -am x)""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap_err();
        assert!(err.contains("blocked"), "err: {err}");
        assert!(err.contains("confirmation"), "err: {err}");
    }

    #[test]
    fn commit_confirmation_passes_through_non_commit_commands() {
        assert!(confirm_git_commit_if_needed("git status").is_ok());
        assert!(confirm_git_commit_if_needed("echo hello").is_ok());
    }

    #[test]
    fn execute_command_materializes_file_data_for_any_simple_outer_command() {
        let path = temporary_file_path("safe_file_read_substitution");
        let contents = "literal $(whoami); 'quoted'";
        fs::write(&path, contents).expect("write test file");

        let result = execute_command(&json!({
            "command": format!(r#"printf '%s' "$(cat {})""#, path.display()),
            "pty": false,
            "timeout": 5,
        }));
        let _ = fs::remove_file(&path);

        assert_eq!(result.unwrap(), contents);
    }

    #[test]
    fn file_read_substitution_content_still_passes_command_audit() {
        let path = temporary_file_path("unsafe_file_read_substitution");
        fs::write(&path, "rm").expect("write test file");

        let result = execute_command(&json!({
            "command": format!(r#""$(cat {})" -rf /tmp/rust_tools_audit_test"#, path.display()),
            "pty": false,
            "timeout": 5,
        }));
        let _ = fs::remove_file(&path);

        let err = result.unwrap_err();
        assert!(err.contains("rm"), "err: {err}");
    }

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
        assert!(
            out.contains("`read_file` with offset/limit"),
            "must steer file paging toward read_file over sed: {out}"
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
                stalled: false,
            },
            30,
        );
        assert!(out.contains("timed out after 30s"), "out: {out}");
        assert!(out.contains("Partial output captured"), "out: {out}");
        assert!(out.contains("progress before timeout"), "out: {out}");
        assert!(out.contains("last diagnostic"), "out: {out}");
    }

    #[test]
    fn stalled_result_explains_interactive_wait_and_keeps_partial_output() {
        let out = format_command_result(
            CommandRunResult {
                status: None,
                stdout: b"scan this QR: QR-CONTENT\n".to_vec(),
                stderr: Vec::new(),
                timed_out: false,
                cancelled: false,
                stalled: true,
            },
            60,
        );
        assert!(out.contains("waiting for interactive input"), "out: {out}");
        assert!(out.contains("QR-CONTENT"), "out: {out}");
        assert!(out.contains("`| tail`"), "out: {out}");
        assert!(
            !out.contains("timed out"),
            "must not read as a timeout: {out}"
        );
    }

    #[test]
    fn stalled_result_without_output_stays_informative() {
        let out = format_command_result(
            CommandRunResult {
                status: None,
                stdout: Vec::new(),
                stderr: Vec::new(),
                timed_out: false,
                cancelled: false,
                stalled: true,
            },
            60,
        );
        assert!(out.contains("no output was captured"), "out: {out}");
        assert!(out.contains("waiting for interactive input"), "out: {out}");
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
