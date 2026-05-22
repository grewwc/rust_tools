use serde_json::Value;
use std::process::Output;

use crate::ai::tools::storage::command_runner;

const MAX_COMMAND_OUTPUT_CHARS: usize = 16_000;

fn truncate_chars(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let mut output = String::with_capacity(max_chars + 32);
    for (idx, ch) in content.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        output.push(ch);
    }
    output.push_str("\n... (truncated)");
    output
}

/// 把命令按 shell 控制操作符 (`;`, `&&`, `||`, `|`, `&`) 在引号外切分。
/// 引号识别仅做最小判断：单引号内的字符全部视为字面量；双引号同样视为字面量
/// （注意：bash 中双引号内 `$()`/`\`` 仍可生效，所以命令替换的检测需要在
/// **未拆分的整段命令**上独立完成，而不是依赖本函数）。
///
/// 这里的目的是把 `validate_execute_command` 已有的 program/参数黑名单
/// 套到链式命令的每一段上，例如 `echo ok && rm -rf /` 在原实现中只验
/// `echo`，分段后会再验 `rm`，从而拦下尾段。
fn split_unquoted_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let bytes = command.as_bytes();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            current.push(b as char);
            if b == b'\'' {
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            current.push(b as char);
            // 转义字符在双引号内仅对少数字符有效；这里粗粒度跳过下一个字节即可
            if b == b'\\' && i + 1 < bytes.len() {
                current.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if b == b'"' {
                in_double = false;
            }
            i += 1;
            continue;
        }
        match b {
            b'\'' => {
                in_single = true;
                current.push('\'');
                i += 1;
            }
            b'"' => {
                in_double = true;
                current.push('"');
                i += 1;
            }
            b'\\' if i + 1 < bytes.len() => {
                // 引号外的反斜杠转义：保留两个字节
                current.push(b as char);
                current.push(bytes[i + 1] as char);
                i += 2;
            }
            // 双字符操作符 `&&` / `||`
            b'&' if i + 1 < bytes.len() && bytes[i + 1] == b'&' => {
                segments.push(std::mem::take(&mut current));
                i += 2;
            }
            b'|' if i + 1 < bytes.len() && bytes[i + 1] == b'|' => {
                segments.push(std::mem::take(&mut current));
                i += 2;
            }
            // 单字符分隔符
            b';' | b'|' | b'&' | b'\n' => {
                segments.push(std::mem::take(&mut current));
                i += 1;
            }
            _ => {
                current.push(b as char);
                i += 1;
            }
        }
    }
    segments.push(current);
    segments
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// 对整个原始命令做"shell 注入面"层面的全局检查（独立于分段验证），
/// 拦截那些不被分段化策略覆盖的注入面：命令替换、heredoc/herestring、
/// 进程替换。它们在正当 dev 工作流里几乎不出现，但可以一举绕过任何
/// program/参数级黑名单（典型样例：`$(echo rm) -rf /tmp/foo`）。
fn validate_no_injection_surface(command: &str) -> Result<(), String> {
    let bytes = command.as_bytes();
    let mut i = 0;
    let mut in_single = false;
    while i < bytes.len() {
        let b = bytes[i];
        // 单引号内的所有内容都是字面量，shell 不会解析 $() / `；
        // 双引号内 $() 与 `` 仍会被 shell 解释，因此双引号不视为安全围栏。
        if in_single {
            if b == b'\'' {
                in_single = false;
            }
            i += 1;
            continue;
        }
        if b == b'\'' {
            in_single = true;
            i += 1;
            continue;
        }
        // 命令替换 `$(`
        if b == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            return Err(
                "command substitution `$(...)` is not allowed; pass a literal command instead"
                    .to_string(),
            );
        }
        // 进程替换 `<(...)` / `>(...)`
        if (b == b'<' || b == b'>') && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            return Err(
                "process substitution `<(...)` / `>(...)` is not allowed".to_string(),
            );
        }
        // heredoc / herestring `<<` / `<<<`
        if b == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'<' {
            return Err(
                "heredoc / herestring (`<<`, `<<<`) is not allowed; write input to a file instead"
                    .to_string(),
            );
        }
        // 反引号命令替换
        if b == b'`' {
            return Err(
                "backtick command substitution is not allowed; pass a literal command instead"
                    .to_string(),
            );
        }
        i += 1;
    }
    Ok(())
}

pub fn validate_execute_command(command: &str) -> Result<(), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("empty command".to_string());
    }

    // 第一道防线：阻断 shell 注入面（命令替换 / heredoc / 进程替换）。
    // 这些放过去，分段黑名单就是摆设。
    validate_no_injection_surface(command)?;

    // 第二道防线：把链式命令拆段，对每一段都跑一次 program/参数黑名单。
    // 这样 `echo ok && rm -rf /` 会在第二段被 `rm` 黑名单命中。
    let segments = split_unquoted_segments(command);
    if segments.is_empty() {
        return Err("empty command".to_string());
    }
    if segments.len() > 1 {
        for seg in &segments {
            validate_single_segment(seg)?;
        }
        return Ok(());
    }
    validate_single_segment(&segments[0])
}

fn validate_single_segment(command: &str) -> Result<(), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("empty command".to_string());
    }

    let tokens = command.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return Err("empty command".to_string());
    }

    let program = tokens[0].to_lowercase();
    let normalize_path = |path: &std::path::Path| {
        let mut normalized = std::path::PathBuf::new();
        for component in path.components() {
            match component {
                std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
                std::path::Component::RootDir => normalized.push(component.as_os_str()),
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    normalized.pop();
                }
                std::path::Component::Normal(part) => normalized.push(part),
            }
        }
        normalized
    };

    if program == "rm" || program == "mv" {
        let base_dir = crate::ai::driver::runtime_ctx::effective_cwd()
            .map_err(|err| format!("failed to resolve current directory: {err}"))?;
        let base_dir = normalize_path(&base_dir);
        let mut path_args: Vec<String> = Vec::new();
        let mut iter = tokens.iter().skip(1).peekable();
        let mut end_of_options = false;

        while let Some(token) = iter.next() {
            if !end_of_options {
                if *token == "--" {
                    end_of_options = true;
                    continue;
                }

                if token.starts_with('-') {
                    if program == "mv" {
                        let option = token.to_lowercase();
                        if option == "-t" || option == "--target-directory" {
                            let dir = iter
                                .next()
                                .ok_or_else(|| format!("missing target directory for '{token}'"))?;
                            path_args.push((*dir).to_string());
                            continue;
                        }

                        if let Some(dir) = option.strip_prefix("--target-directory=") {
                            if dir.is_empty() {
                                return Err(format!("missing target directory for '{token}'"));
                            }
                            path_args.push(dir.to_string());
                            continue;
                        }

                        if token.starts_with("-t") && token.len() > 2 {
                            path_args.push(token[2..].to_string());
                            continue;
                        }
                    }

                    continue;
                }
            }

            path_args.push((*token).to_string());
        }

        if path_args.is_empty() {
            return Err(format!("program '{program}' requires path arguments"));
        }

        for raw_path in path_args {
            let raw_path = raw_path.trim();
            if raw_path.is_empty() {
                return Err(format!("program '{program}' contains an empty path"));
            }

            let resolved = if std::path::Path::new(raw_path).is_absolute() {
                normalize_path(std::path::Path::new(raw_path))
            } else {
                normalize_path(&base_dir.join(raw_path))
            };

            if !resolved.starts_with(&base_dir) {
                return Err(format!(
                    "path '{raw_path}' is outside the current directory"
                ));
            }
        }

        return Ok(());
    }

    let denied_programs = [
        "fish",
        "jshell",
        "dd",
        "chmod",
        "chown",
        "chgrp",
        "kill",
        "pkill",
        "killall",
        "sudo",
        "su",
        "passwd",
        "shutdown",
        "reboot",
        "launchctl",
        "systemctl",
        "service",
        "diskutil",
        "mount",
        "umount",
        "ln",
        "truncate",
        "ssh",
        "scp",
        // 绕过手段：`eval` / `source` / `.` 会把后续字符串当 shell 代码再次
        // 解释，等于把验证完全 bypass 掉。
        "eval",
        "source",
        ".",
        // 反向 shell / 网络监听工具，正当 dev 流程几乎不会用，留着风险大于收益。
        "nc",
        "ncat",
        "netcat",
        "telnet",
        "socat",
    ];
    if denied_programs.contains(&program.as_str()) {
        return Err(format!("program '{program}' is blocked"));
    }

    // 拦下 `bash -c "..."` / `sh -c` / `zsh -c` 这种"二次解释"形式。
    // 直接执行脚本（`bash script.sh`）仍然允许，避免破坏正常工作流。
    if matches!(program.as_str(), "bash" | "sh" | "zsh" | "ksh" | "dash") {
        for tok in tokens.iter().skip(1) {
            let lower = tok.to_lowercase();
            if lower == "-c" || lower == "--command" {
                return Err(format!(
                    "shell `{program} -c ...` re-interprets a string as shell code; \
                     run the literal command directly instead"
                ));
            }
        }
    }

    let denied_tokens = [
        "-delete", "--remove", "rm", "mv", "chmod", "chown", "sudo", "ssh", "scp", "rsync",
    ];
    for token in tokens.iter().skip(1) {
        let token = token.to_lowercase();
        if denied_tokens.contains(&token.as_str()) {
            return Err(format!("argument '{token}' is blocked"));
        }
    }

    Ok(())
}

fn format_command_output(output: Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let stdout_trimmed = stdout.trim();
    let stderr_trimmed = stderr.trim();

    if output.status.success() {
        let combined = if stdout_trimmed.is_empty() {
            stderr_trimmed.to_string()
        } else if stderr_trimmed.is_empty() {
            stdout_trimmed.to_string()
        } else {
            format!("{stdout_trimmed}\n{stderr_trimmed}")
        };
        truncate_chars(combined.trim(), MAX_COMMAND_OUTPUT_CHARS)
    } else {
        truncate_chars(
            &format!(
                "Exit code: {}\n{}\n{}",
                output.status.code().unwrap_or(-1),
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
    let timeout = args["timeout"].as_u64().unwrap_or(60).clamp(1, 300);

    if let Err(reason) = validate_execute_command(command) {
        return Ok(format!("Command blocked: {reason}"));
    }

    let output = command_runner::run_command_streaming(command, cwd, timeout, on_chunk)?;
    Ok(format_command_output(output))
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
    use super::{split_unquoted_segments, validate_no_injection_surface};

    // ---- split_unquoted_segments ----

    #[test]
    fn split_handles_chained_operators() {
        let segs = split_unquoted_segments("echo ok && rm -rf /tmp/foo");
        assert_eq!(segs, vec!["echo ok".to_string(), "rm -rf /tmp/foo".to_string()]);
    }

    #[test]
    fn split_handles_pipe_and_semicolon() {
        let segs = split_unquoted_segments("a | b ; c || d");
        assert_eq!(
            segs,
            vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()]
        );
    }

    #[test]
    fn split_does_not_break_inside_single_quotes() {
        let segs = split_unquoted_segments("echo 'a && b' ; echo done");
        assert_eq!(
            segs,
            vec!["echo 'a && b'".to_string(), "echo done".to_string()]
        );
    }

    #[test]
    fn split_does_not_break_inside_double_quotes() {
        let segs = split_unquoted_segments("echo \"a | b\" && true");
        assert_eq!(
            segs,
            vec!["echo \"a | b\"".to_string(), "true".to_string()]
        );
    }

    // ---- injection surface ----

    #[test]
    fn injection_blocks_dollar_paren() {
        assert!(validate_no_injection_surface("echo $(whoami)").is_err());
    }

    #[test]
    fn injection_blocks_backtick() {
        assert!(validate_no_injection_surface("echo `whoami`").is_err());
    }

    #[test]
    fn injection_blocks_heredoc_and_herestring() {
        assert!(validate_no_injection_surface("cat <<EOF").is_err());
        assert!(validate_no_injection_surface("cat <<<\"hi\"").is_err());
    }

    #[test]
    fn injection_blocks_process_substitution() {
        assert!(validate_no_injection_surface("diff <(echo a) <(echo b)").is_err());
    }

    #[test]
    fn injection_allows_clean_command() {
        assert!(validate_no_injection_surface("cargo build --release").is_ok());
    }

    #[test]
    fn injection_treats_single_quoted_as_literal() {
        // 整段在单引号内的 `$()` 是字面量，bash 不会展开。
        assert!(validate_no_injection_surface("echo 'price: $(100)'").is_ok());
    }

    // ---- end-to-end validate_execute_command ----

    fn validate(cmd: &str) -> Result<(), String> {
        super::validate_execute_command(cmd)
    }

    #[test]
    fn blocks_chained_rm_after_safe_prefix() {
        // 修复前：仅验 `echo`，整体放行。
        // 修复后：第二段会被 `rm` 路径校验或黑名单拦下。
        let err = validate("echo ok && rm -rf /").unwrap_err();
        assert!(
            err.contains("rm") || err.contains("outside the current directory"),
            "expected rm/path block, got: {err}"
        );
    }

    #[test]
    fn blocks_sudo_anywhere_in_chain() {
        let err = validate("true ; sudo reboot").unwrap_err();
        assert!(
            err.contains("sudo") || err.contains("reboot"),
            "expected sudo/reboot to be blocked, got: {err}"
        );
    }

    #[test]
    fn blocks_eval_segment() {
        let err = validate("eval \"echo hi\"").unwrap_err();
        assert!(err.contains("eval"), "expected eval blocked, got: {err}");
    }

    #[test]
    fn blocks_bash_dash_c() {
        let err = validate("bash -c \"echo ok\"").unwrap_err();
        assert!(err.contains("-c"), "expected `bash -c` blocked, got: {err}");
    }

    #[test]
    fn allows_bash_running_a_script_file() {
        // `bash run.sh` 不是二次解释，正常工作流应继续允许。
        assert!(validate("bash run.sh").is_ok());
    }

    #[test]
    fn blocks_command_substitution() {
        let err = validate("echo $(whoami)").unwrap_err();
        assert!(
            err.contains("command substitution"),
            "expected $(...) blocked, got: {err}"
        );
    }

    #[test]
    fn allows_normal_dev_commands() {
        assert!(validate("cargo check --bin a").is_ok());
        assert!(validate("git status").is_ok());
        assert!(validate("ls -la").is_ok());
        // 使用单引号包裹的字面量 `$()` 应放行（不会被 shell 展开）
        assert!(validate("echo 'literal $(x)'").is_ok());
    }
}
