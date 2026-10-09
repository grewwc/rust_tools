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
/// Read cap when materializing `$(cat /absolute/literal/path)` into a plain
/// shell argument. The limit stops the tool from accidentally reading an
/// unbounded stream or an oversized file before validation; regular JSON/DSL
/// arguments are far smaller.
const MAX_LITERAL_FILE_SUBSTITUTION_BYTES: usize = 64 * 1024;

/// Built-in default timeout and ceiling (seconds), overridable via sandbox
/// config.
const DEFAULT_COMMAND_TIMEOUT_SECS: u64 = 60;
const DEFAULT_COMMAND_TIMEOUT_MAX_SECS: u64 = 300;

/// Returns `execute_command`'s (default timeout, timeout ceiling), overridden
/// by sandbox config. Invalid/missing values fall back to the built-in
/// constants; the ceiling is at least 1s and never below the default.
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

/// Pure function: clamp the requested timeout seconds into `[1, max]`, using
/// `default` when unset.
fn resolve_command_timeout(requested: Option<u64>, default: u64, max: u64) -> u64 {
    requested.unwrap_or(default).clamp(1, max)
}

/// Encode UTF-8 data as a complete POSIX shell word. Inside single quotes no
/// expansion happens; embedded single quotes are bridged with
/// `'<backslash><quote>'` into the next single-quoted literal, so file contents
/// can never become shell code.
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

/// Run the inner command of a harmless command substitution and capture its
/// output, used to materialize `"$(harmless_cmd)"`.
fn execute_inner_shell_command(inner: &str, cwd: Option<&str>) -> Result<String, String> {
    // Reuse the existing runner; a 10s timeout covers short commands like
    // date/echo/git while avoiding long blocks.
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
    // bash's $(...) strips all trailing newlines; replicate that behavior.
    Ok(stdout
        .trim_end_matches(|c| c == '\n' || c == '\r')
        .to_string())
}

/// Materialize harmless `"$(...)"` substitutions proven safe by the audit
/// layer as plain data, supporting `cat` literals and generic harmless
/// commands. The substituted command is still fully audited afterwards, so a
/// replacement that turns into a blocked program name or dangerous argument is
/// still rejected.
fn materialize_safe_shell_substitutions(
    command: &str,
    cwd: Option<&str>,
) -> Result<String, String> {
    let substitutions = super::audit::safe_shell_substitutions(command);
    if substitutions.is_empty() {
        // Fast path: nothing to materialize; return the command unchanged.
        return Ok(command.to_string());
    }
    let mut materialized = command.to_string();
    for substitution in substitutions.into_iter().rev() {
        let contents = match substitution.kind {
            super::audit::SafeShellSubstitutionKind::FileRead { path } => {
                read_literal_file_substitution(&path)?
            }
            super::audit::SafeShellSubstitutionKind::Command { inner } => {
                // The inner command must pass the full gate too: re-run
                // validate first (guards against relaxed classification), then
                // the git confirmation gate (commit/stash, fail-closed). The
                // audit classifier only proves the inner command passed
                // validate; it does not intercept confirm-gated git commands.
                // Without this, `echo "$(git commit -am x)"` would commit
                // before the outer confirmation, leaving only `echo '...'`
                // after materialization and bypassing the gate.
                super::audit::validate_execute_command(&inner)
                    .map_err(|reason| format!("inner substitution blocked: {reason}"))?;
                confirm_git_confirm_gated_if_needed(&inner)?;
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

/// Truncate over-long output keeping head and tail, with **actionable
/// metadata** in the middle: total count, shown count, and an explicit warning
/// that the omitted middle may contain the lines the caller is looking for —
/// "not seen" does not mean "not present".
///
/// Background: the `execute_command` success path used to truncate silently
/// with `... (truncated)`, so the model could not tell whether its target
/// match was cut off in the hidden part and kept retrying near-identical
/// greps (the repeated calls in history.json come from this). With counts and
/// a paging hint, the retry motive changes from guessing with incomplete
/// information to converging on evidence.
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
// Execution logic (validation moved to the audit module)
// =========================================================================

/// `git stash` subactions that are read-only — they touch neither the working
/// tree nor stash entries, so they need no confirmation. A word only counts as
/// the subaction when it directly follows `stash` (first positional): in
/// `git stash push list` the `list` is a path argument and still requires
/// confirmation.
const READ_ONLY_GIT_STASH_SUBACTIONS: &[&str] = &["list", "show"];

/// Git subcommands that must be confirmed by the user before running
/// (`git commit`; `git stash` and its subactions that change the working tree
/// or stash entries, e.g. pop/drop/clear — read-only `git stash list`/`show`
/// need no confirmation). Reuses the audit layer's lexical parsing so
/// materialized quoted arguments cannot bypass confirmation, and data like
/// `echo 'git commit'` is not mistaken for a real command. Returns the matched
/// subcommand name (`commit` / `stash`) for prompt and error wording.
fn confirm_gated_git_subcommand(command: &str) -> Option<&'static str> {
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

        // Skip git global options and their values (-C <path>,
        // -c <key>=<val>, --git-dir=..., etc.), which may sit between `git`
        // and the subcommand.
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
        if let Some(sub) = tokens.get(j).map(String::as_str) {
            // `sub` borrows the local `tokens`; return the matching static
            // instead so the signature (`Option<&'static str>`) holds.
            if sub == "commit" {
                return Some("commit");
            }
            if sub == "stash" {
                // Read-only subactions (list/show) directly after `stash`
                // need no confirmation; everything else (bare `git stash`=push,
                // `git stash -- <path>`, or stash options like `git stash -q
                // list`) still goes through the confirmation gate (fail-closed).
                if matches!(
                    tokens.get(j + 1).map(String::as_str),
                    Some("list") | Some("show")
                ) {
                    continue;
                }
                return Some("stash");
            }
        }
    }
    None
}

/// Ask the user to confirm confirm-gated git subcommands (`git commit` /
/// `git stash`) before running them.
/// - Interactive terminal: red-highlighted prompt; y proceeds, n / Ctrl+C /
///   Esc cancels.
/// - Non-interactive environment: rejected outright (fail-closed), both to
///   avoid background processes hanging on input and to avoid silent execution.
fn confirm_git_confirm_gated_if_needed(command: &str) -> Result<(), String> {
    let Some(subcommand) = confirm_gated_git_subcommand(command) else {
        return Ok(());
    };
    if !std::io::stdin().is_terminal() {
        return Err(format!(
            "Command blocked: git {subcommand} requires user confirmation, but stdin is not an \
             interactive terminal. Do not retry the command; report to the user and wait for \
             explicit confirmation (or have them run it in an interactive session)."
        ));
    }
    let confirmed = crate::commonw::prompt::prompt_yes_or_no_danger(&format!(
        "\nConfirm git {subcommand}:\n{command}\nProceed? (y/n): "
    ));
    match confirmed {
        Some(true) => Ok(()),
        Some(false) => Err(format!("git {subcommand} canceled by user")),
        None => Err(format!("git {subcommand} canceled by user (Ctrl+C)")),
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
        // PTY interactive command stalled: the process is alive but produced
        // no output for a long time, which almost certainly means it waits for
        // human input (QR scan, password, menu) the agent cannot provide; when
        // output is buffered by a pipe (e.g. `| tail`) nothing shows at all.
        // Give an explicit diagnosis plus any captured partial output (often
        // the QR code) instead of letting the model misread this as a plain
        // timeout and blindly retry the same command.
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
        // Empty output with successful exit: say so explicitly so the model
        // does not misread "command succeeded, zero matches" as "the call did
        // not take effect" and retry the same grep repeatedly.
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

/// True when the effective program of any command segment is `sleep`.
///
/// `sleep` is not blocked (it is occasionally necessary, e.g. waiting for a
/// server to accept connections), but it burns the whole turn on dead time: the
/// agent cannot do anything else while the command runs. Detection reuses the
/// audit segment splitter and wrapper unwrapping (`nohup sleep 30 &` still
/// hits, `command sleep 5` hits), so data arguments like `grep sleep file` or
/// `man sleep` do not.
fn contains_sleep(command: &str) -> bool {
    super::audit::split_unquoted_segments(command).iter().any(|segment| {
        let tokens = super::audit::effective_command_tokens(segment);
        let program = tokens
            .first()
            .and_then(|token| Path::new(token).file_name())
            .and_then(|name| name.to_str());
        if program == Some("sleep") {
            return true;
        }
        // Shells forwarding a script body via `-c` execute that body directly
        // (`bash -c "sleep 5"`, `sh -c 'sleep 3'`), so re-parse the body to
        // find `sleep` instead of treating it as data. `timeout 5 sleep 3`
        // is already unwrapped by the audit wrapper logic.
        if matches!(program, Some("bash" | "sh" | "zsh" | "ksh" | "dash")) {
            let Some(dash_c) = tokens.iter().position(|token| token == "-c") else {
                return false;
            };
            let body = tokens[dash_c + 1..].join(" ");
            return super::audit::effective_command_tokens(&body)
                .first()
                .and_then(|token| Path::new(token).file_name())
                .and_then(|name| name.to_str())
                == Some("sleep");
        }
        false
    })
}

/// Guidance prepended to a `sleep` command's result: the hint lives in the tool
/// description too, but showing it at execution time makes it hard to miss.
const SLEEP_WARNING: &str = "Note: this command runs `sleep`, which blocks the whole turn on \
dead time — it is expensive. Use it only when truly necessary (e.g. waiting for a server to \
accept connections); prefer backgrounding the work (`&` + log file) and polling a readiness \
condition, or `task_wait` for subagent tasks.";

fn execute_command_inner<F>(args: &Value, on_chunk: F) -> Result<String, String>
where
    F: FnMut(&[u8]),
{
    let raw_command = args["command"].as_str().ok_or("Missing command")?;
    let cwd = args["cwd"].as_str().filter(|dir| !dir.trim().is_empty());
    // Materialize harmless "$(...)" first (cat literals and generic harmless
    // commands); the result is still fully audited afterwards.
    let command = materialize_safe_shell_substitutions(raw_command, cwd)
        .map_err(|reason| format!("Command blocked: {reason}"))?;
    let pseudo_terminal = args["pty"].as_bool().unwrap_or(false);
    let (default_timeout, max_timeout) = config_command_timeout_bounds();
    let timeout = resolve_command_timeout(args["timeout"].as_u64(), default_timeout, max_timeout);

    // Command safety validation is delegated to the audit module.
    super::audit::validate_execute_command(&command)
        .map_err(|reason| format!("Command blocked: {reason}"))?;

    // Confirm-gated git subcommands (commit/stash) ask the user first
    // (fail-closed in non-interactive environments).
    confirm_git_confirm_gated_if_needed(&command)?;

    let output =
        command_runner::run_command_streaming(&command, cwd, timeout, pseudo_terminal, on_chunk)?;
    let interrupted = output.timed_out || output.cancelled || output.stalled;
    let mut formatted = format_command_result(output, timeout);
    if contains_sleep(&command) {
        formatted = format!("{SLEEP_WARNING}\n{formatted}");
    }
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
        MAX_COMMAND_OUTPUT_CHARS, SLEEP_WARNING, confirm_gated_git_subcommand,
        confirm_git_confirm_gated_if_needed, contains_sleep, execute_command, format_command_result,
        resolve_command_timeout, truncate_chars,
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

    // ---- confirm-gated git subcommands (commit / stash) ----

    #[test]
    fn commit_detection_matches_plain_git_commit() {
        assert_eq!(
            confirm_gated_git_subcommand("git commit -m \"fix x\""),
            Some("commit")
        );
        assert_eq!(confirm_gated_git_subcommand("git commit"), Some("commit"));
        assert_eq!(
            confirm_gated_git_subcommand("git commit --amend"),
            Some("commit")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git 'commit' -m message"),
            Some("commit")
        );
    }

    #[test]
    fn commit_detection_skips_global_options() {
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo commit -m x"),
            Some("commit")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git --git-dir=/repo/.git commit"),
            Some("commit")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -c user.name=X commit"),
            Some("commit")
        );
    }

    #[test]
    fn commit_detection_finds_commit_in_command_chains() {
        assert_eq!(
            confirm_gated_git_subcommand("git add -A && git commit -m x"),
            Some("commit")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo add . && git -C /repo commit"),
            Some("commit")
        );
    }

    #[test]
    fn stash_detection_matches_all_stash_forms() {
        assert_eq!(confirm_gated_git_subcommand("git stash"), Some("stash"));
        assert_eq!(confirm_gated_git_subcommand("git stash pop"), Some("stash"));
        assert_eq!(
            confirm_gated_git_subcommand("git stash drop stash@{0}"),
            Some("stash")
        );
        assert_eq!(confirm_gated_git_subcommand("git stash clear"), Some("stash"));
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo stash push -m wip"),
            Some("stash")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git stash && echo done"),
            Some("stash")
        );
        assert_eq!(confirm_gated_git_subcommand("env git stash"), Some("stash"));
        assert_eq!(confirm_gated_git_subcommand("xargs git stash"), Some("stash"));
        assert_eq!(confirm_gated_git_subcommand("nohup git stash"), Some("stash"));
        assert_eq!(confirm_gated_git_subcommand("command git stash"), Some("stash"));
        // Read-only subactions (list/show) directly after `stash` need no
        // confirmation.
        assert_eq!(confirm_gated_git_subcommand("git stash list"), None);
        assert_eq!(confirm_gated_git_subcommand("git stash show"), None);
        assert_eq!(confirm_gated_git_subcommand("git stash show stash@{0}"), None);
        assert_eq!(confirm_gated_git_subcommand("git stash list --format='%gd'"), None);
        // `list`/`show` outside the subaction position (`--` followed by a
        // path, `-q` a stash option, or coexisting with a confirm-gated
        // subaction) still require confirmation.
        assert_eq!(confirm_gated_git_subcommand("git stash -- list"), Some("stash"));
        assert_eq!(confirm_gated_git_subcommand("git stash -q list"), Some("stash"));
        assert_eq!(
            confirm_gated_git_subcommand("git stash list && git stash drop"),
            Some("stash")
        );
    }

    #[test]
    fn confirm_gated_detection_ignores_non_gated_commands() {
        assert_eq!(confirm_gated_git_subcommand("git status"), None);
        assert_eq!(confirm_gated_git_subcommand("git log --oneline | grep commit"), None);
        assert_eq!(confirm_gated_git_subcommand("git svn commit"), None);
        assert_eq!(confirm_gated_git_subcommand("echo 'git commit' > note.txt"), None);
        assert_eq!(confirm_gated_git_subcommand("git commitmessage"), None);
        assert_eq!(confirm_gated_git_subcommand("git stashlist"), None);
        assert_eq!(confirm_gated_git_subcommand("echo 'git stash' > note.txt"), None);
    }

    #[test]
    fn confirm_gated_fails_closed_without_terminal() {
        // Test stdin is not a terminal: confirm-gated git subcommands must be
        // rejected without hanging.
        for (command, sub) in [
            ("git commit -m x", "commit"),
            ("git stash", "stash"),
            ("git stash drop stash@{0}", "stash"),
            ("git -C /repo stash clear", "stash"),
        ] {
            let err = confirm_git_confirm_gated_if_needed(command).unwrap_err();
            assert!(err.contains("blocked"), "{command}: {err}");
            assert!(err.contains("confirmation"), "{command}: {err}");
            assert!(err.contains(sub), "{command}: {err}");
        }
    }

    #[test]
    fn execute_command_blocks_git_commit_inside_substitution_without_terminal() {
        // P0 regression: without a TTY, `echo "$(git commit ...)"` must be
        // rejected fail-closed; command substitution must not silently bypass
        // the confirmation gate.
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
    fn execute_command_blocks_git_stash_inside_substitution_without_terminal() {
        // Same as commit: without a TTY, `echo "$(git stash)"` must fail
        // closed; command substitution must not silently bypass the
        // confirmation gate.
        let err = execute_command(&json!({
            "command": r#"echo "$(git stash)""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap_err();
        assert!(err.contains("blocked"), "err: {err}");
        assert!(err.contains("confirmation"), "err: {err}");
    }

    #[test]
    fn confirm_gated_passes_through_non_gated_commands() {
        assert!(confirm_git_confirm_gated_if_needed("git status").is_ok());
        assert!(confirm_git_confirm_gated_if_needed("echo hello").is_ok());
        // Read-only stash subactions pass through even in non-interactive
        // environments without triggering confirmation.
        assert!(confirm_git_confirm_gated_if_needed("git stash list").is_ok());
        assert!(confirm_git_confirm_gated_if_needed("git stash show stash@{0}").is_ok());
    }

    // ---- contains_sleep ----

    #[test]
    fn sleep_detection_finds_effective_sleep_program() {
        assert!(contains_sleep("sleep 5"));
        assert!(contains_sleep("sleep 0.5"));
        assert!(contains_sleep("sleep 5 && cargo check"));
        assert!(contains_sleep("nohup sleep 30 &"));
        assert!(contains_sleep("env sleep 3"));
        assert!(contains_sleep("command sleep 1"));
        assert!(contains_sleep("/usr/bin/sleep 2"));
        // Programs forwarded through a shell `-c` body execute `sleep` directly.
        assert!(contains_sleep("bash -c \"sleep 5\""));
        assert!(contains_sleep("sh -c 'sleep 0.5'"));
        assert!(contains_sleep("timeout 5 sleep 3"));
    }

    #[test]
    fn sleep_detection_ignores_data_arguments() {
        assert!(!contains_sleep("grep sleep notes.txt"));
        assert!(!contains_sleep("man sleep"));
        assert!(!contains_sleep("echo sleep"));
        assert!(!contains_sleep("ls /usr/bin/sleep"));
        assert!(!contains_sleep("bash -c \"echo sleep\""));
        assert!(!contains_sleep("bash -c 'grep sleep notes.txt'"));
        assert!(!contains_sleep("timeout 5 echo 5"));
    }

    #[test]
    fn execute_command_prepends_sleep_warning() {
        let out = execute_command(&json!({
            "command": "sleep 0",
            "pty": false,
            "timeout": 5,
        }))
        .unwrap();
        assert!(
            out.starts_with(SLEEP_WARNING),
            "sleep result must carry the warning, got: {out}"
        );
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
        // 1000 short lines total far more than the small cap, triggering
        // truncation.
        let content: String = (0..1000).map(|i| format!("line{i}\n")).collect();
        let out = truncate_chars(&content, 100);
        // Not a bare "... (truncated)" anymore: the output carries
        // totals/shown counts and a paging hint.
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
