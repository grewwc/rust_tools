/// Audit module: command safety validation (injection-surface checks + segment
/// blacklists).
///
/// Separation of concerns:
/// - This module only "validates"; it never "executes".
/// - `execute_command` just calls the `validate_execute_command()` entry point.
/// - Easy to test and evolve the safety policy independently, decoupled from
///   execution logic.
use crate::ai::config_schema::AiConfig;

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

/// Read the user-configured list of denied programs.
fn config_blocked_commands() -> Vec<String> {
    let raw = crate::commonw::configw::get_all_config().get(AiConfig::SANDBOX_BLOCKED_COMMANDS, "");
    raw.split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}
mod lexer;
mod wrapper;
mod git;
mod substitute;
mod exec;

// Re-export names that sibling modules, the inline tests, and external callers
// resolve through this module root (`audit::X`). Items stay defined in their
// concern file; these `use` re-exports keep every `use super::*` / `audit::X`
// path working without renaming or moving the definitions.
pub(crate) use exec::validate_single_segment;
pub(crate) use git::command_subcommand_index;
pub(crate) use lexer::{
    effective_command_tokens, split_unquoted_command_segments, split_unquoted_segments, ShellJoin,
};
pub(crate) use substitute::{
    safe_shell_substitutions, SafeShellSubstitutionKind, validate_no_injection_surface,
};

// Test-only re-exports: the `#[cfg(test)]` modules below resolve these through
// `use super::*`, so they must not ship in non-test builds (keeps `cargo check`
// warning-free). Production code reaches the same items via the submodules.
#[cfg(test)]
pub(crate) use exec::{
    merge_heavy_search_dirs, process_group_of, summarize_dirs, validate_kill_pids,
    validate_pattern_kill, validate_rm_targets, PgrepMode, has_glob_metachar, strip_quotes,
};
#[cfg(test)]
pub(crate) use lexer::{effective_chain_uses_xargs, tokenize_shell_words};


// =========================================================================
// Public entry point
// =========================================================================

/// Validate the safety of one complete command (including chained `&&` / `||`).
///
/// This is the audit module's single public entry point; `execute_command` just
/// calls it.
pub(crate) fn validate_execute_command(command: &str) -> Result<(), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("empty command".to_string());
    }

    // First line of defense: block shell injection surfaces (unvalidated command
    // substitution, backticks, subshell grouping). `$(...)` / `<(...)` pass only
    // after their inner command passes the same validation; letting unvalidated
    // substitutions through would render the segment blacklist pointless.
    validate_no_injection_surface(command)?;

    // Second line of defense: split the chained command into segments and run the
    // program/argument blacklist on each one. That way `echo ok && rm -rf /` is
    // caught by the `rm` blacklist in the second segment.
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

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::{
        SafeShellSubstitutionKind, ShellJoin, command_subcommand_index, effective_command_tokens,
        safe_shell_substitutions, split_unquoted_command_segments, split_unquoted_segments,
        tokenize_shell_words, validate_no_injection_surface,
    };

    // ---- split_unquoted_segments ----

    #[test]
    fn split_handles_chained_operators() {
        let segs = split_unquoted_segments("echo ok && rm -rf /tmp/foo");
        assert_eq!(
            segs,
            vec!["echo ok".to_string(), "rm -rf /tmp/foo".to_string()]
        );
    }

    #[test]
    fn split_handles_pipe_and_semicolon() {
        let segs = split_unquoted_segments("a | b ; c || d");
        assert_eq!(
            segs,
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }

    #[test]
    fn split_preserves_fd_duplication_redirections() {
        let segments =
            split_unquoted_command_segments("cargo test --bin a 2>&1 | tail -6 && echo done");
        assert_eq!(
            segments
                .iter()
                .map(|segment| (segment.command.as_str(), segment.join))
                .collect::<Vec<_>>(),
            vec![
                ("cargo test --bin a 2>&1", ShellJoin::Start),
                ("tail -6", ShellJoin::Other),
                ("echo done", ShellJoin::And),
            ]
        );

        let segs = split_unquoted_segments("cmd &>out && cat out");
        assert_eq!(segs, vec!["cmd &>out".to_string(), "cat out".to_string()]);
    }

    #[test]
    fn split_preserves_success_chain_semantics() {
        let segments = split_unquoted_command_segments("a && b; c");
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.join)
                .collect::<Vec<_>>(),
            vec![ShellJoin::Start, ShellJoin::And, ShellJoin::Other]
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
        assert_eq!(segs, vec!["echo \"a | b\"".to_string(), "true".to_string()]);
    }

    #[test]
    fn split_ignores_quoted_heredoc_body_content() {
        let segs = split_unquoted_segments("cat <<'EOF'\nrm -rf /\nEOF\nls");
        assert_eq!(segs, vec!["cat <<'EOF'".to_string(), "ls".to_string()]);
    }

    // ---- tokenize_shell_words ----

    #[test]
    fn tokenize_shell_words_respects_single_and_double_quotes() {
        let tokens = tokenize_shell_words(r#"printf '%s\n' "a b" '\$(literal)'"#);
        assert_eq!(
            tokens,
            vec![
                "printf".to_string(),
                "%s\\n".to_string(),
                "a b".to_string(),
                "\\$(literal)".to_string()
            ]
        );
    }

    #[test]
    fn command_analysis_handles_wrappers_and_value_options() {
        let git = effective_command_tokens("env -C /tmp FOO=1 git -C /repo status");
        let git_index = command_subcommand_index(&git).unwrap();
        assert_eq!(git[git_index], "status");

        let cargo = effective_command_tokens("cargo --manifest-path Cargo.toml check");
        let cargo_index = command_subcommand_index(&cargo).unwrap();
        assert_eq!(cargo[cargo_index], "check");
    }

    // ---- injection surface ----

    #[test]
    fn injection_allows_validated_dollar_paren() {
        // `$()` passes when the inner command itself passes the safety check.
        assert!(validate_no_injection_surface("echo $(whoami)").is_ok());
        assert!(validate_no_injection_surface("for i in $(seq 1 40); do echo $i; done").is_ok());
        // Unsafe or unterminated inner commands stay blocked.
        let err = validate_no_injection_surface("echo $(rm -rf /)").unwrap_err();
        assert!(err.contains("command substitution"), "got: {err}");
        let err = validate_no_injection_surface("echo $(seq 1 40").unwrap_err();
        assert!(err.contains("command substitution"), "got: {err}");
    }

    #[test]
    fn injection_blocks_backtick_command_substitution() {
        assert!(validate_no_injection_surface("echo `whoami`").is_err());
    }

    #[test]
    fn injection_allows_heredoc_and_herestring() {
        assert!(validate_no_injection_surface("cat <<EOF").is_ok());
        assert!(validate_no_injection_surface("cat <<<\"hi\"").is_ok());
    }

    #[test]
    fn injection_allows_command_substitution_text_inside_quoted_heredoc() {
        assert!(validate_no_injection_surface("cat <<'EOF'\n$(whoami)\nEOF").is_ok());
        assert!(validate_no_injection_surface("cat <<'EOF'\n`whoami`\nEOF").is_ok());
    }

    #[test]
    fn injection_blocks_command_substitution_inside_unquoted_heredoc() {
        assert!(validate_no_injection_surface("cat <<EOF\n$(whoami)\nEOF").is_err());
        assert!(validate_no_injection_surface("cat <<EOF\n`whoami`\nEOF").is_err());
    }

    #[test]
    fn injection_allows_validated_process_substitution() {
        assert!(validate_no_injection_surface("diff <(echo a) <(echo b)").is_ok());
        assert!(validate_no_injection_surface("cat <(printf '%s' ok)").is_ok());
    }

    #[test]
    fn injection_blocks_unsafe_or_unterminated_process_substitution() {
        assert!(validate_no_injection_surface("cat <(rm -rf target)").is_err());
        assert!(validate_no_injection_surface("cat <(echo missing").is_err());
    }

    #[test]
    fn injection_allows_clean_command() {
        assert!(validate_no_injection_surface("cargo build --release").is_ok());
    }

    #[test]
    fn injection_treats_single_quoted_as_literal() {
        // A `$()` entirely inside single quotes is a literal; bash does not
        // expand it.
        assert!(validate_no_injection_surface("echo 'price: $(100)'").is_ok());
        assert!(validate_no_injection_surface("echo '`whoami`'").is_ok());
    }

    #[test]
    fn injection_treats_double_quoted_process_substitution_like_text_as_literal() {
        assert!(validate_no_injection_surface(r#"echo "<(literal)""#).is_ok());
        assert!(validate_no_injection_surface(r#"echo ">(literal)""#).is_ok());
    }

    #[test]
    fn injection_treats_escaped_substitution_markers_as_literal() {
        assert!(validate_no_injection_surface(r#"echo \$(whoami)"#).is_ok());
        assert!(validate_no_injection_surface(r#"echo "\$(whoami)""#).is_ok());
        assert!(validate_no_injection_surface(r#"echo "\`whoami\`""#).is_ok());
    }

    #[test]
    fn injection_validates_substitution_inside_double_quotes() {
        // `$()` inside double quotes is still a command substitution (not
        // literal text): harmless inners pass, unsafe inners stay blocked.
        assert!(validate_no_injection_surface(r#"echo "user=$(whoami)""#).is_ok());
        assert!(validate_no_injection_surface(r#"echo "user=$(rm -rf /)""#).is_err());
    }

    // ---- end-to-end validate_execute_command ----

    fn validate(cmd: &str) -> Result<(), String> {
        super::validate_execute_command(cmd)
    }

    #[test]
    fn blocks_chained_rm_after_safe_prefix() {
        let err = validate("echo ok && rm -rf /").unwrap_err();
        assert!(err.contains("rm"), "expected rm blocked, got: {err}");
    }

    #[test]
    fn blocks_rm_outside_session_temp_dir() {
        let err = validate("rm -rf ./target").unwrap_err();
        assert!(err.contains("rm"), "expected rm blocked, got: {err}");
    }

    #[test]
    fn blocks_shell_rm_with_glob_expansion() {
        let err = validate("rm -rf *.zcompdump").unwrap_err();
        assert!(err.contains("rm"), "expected rm blocked, got: {err}");
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
    fn allows_bash_script_arg_named_dash_c() {
        assert!(validate("bash script.sh -c literal").is_ok());
    }

    #[test]
    fn allows_bash_running_a_script_file() {
        assert!(validate("bash run.sh").is_ok());
    }

    #[test]
    fn command_substitution_validates_inner_and_blocks_program_name_generation() {
        // Quoted data substitutions and literal-seq loops whose inner command
        // passes the safety check are allowed end-to-end.
        assert!(validate("for i in $(seq 1 40); do echo $i; done").is_ok());
        assert!(validate(r#"echo "$(date)""#).is_ok());
        // Unquoted substitutions reach the shell raw and stay blocked even
        // with a harmless inner.
        let err = validate("echo $(date)").unwrap_err();
        assert!(err.contains("command substitution"), "got: {err}");
        // Unsafe inner commands stay blocked.
        let err = validate("echo $(rm -rf /)").unwrap_err();
        assert!(err.contains("command substitution"), "got: {err}");
        // Program-name generation stays banned even with a harmless inner:
        // `$(echo rm) -rf /` would execute `rm -rf /` at runtime.
        let err = validate("$(echo rm) -rf /").unwrap_err();
        assert!(err.contains("command substitution"), "got: {err}");
        let err = validate("env $(echo rm) -rf /").unwrap_err();
        assert!(err.contains("command substitution"), "got: {err}");
        // Backticks remain banned.
        let err = validate("echo `whoami`").unwrap_err();
        assert!(err.contains("backtick"), "got: {err}");
    }

    #[test]
    fn blocks_substitution_generated_sensitive_arguments() {
        // `$()` output is unverifiable, so it must not land in a position the
        // audit semantically checks: `git "pu$(printf sh)"` would run `git
        // push` at runtime, `bash "$(printf -- '-c')" ...` would re-interpret
        // code, and `rm -rf "$(printf '*')"` would glob-expand out of scope.
        let err = validate(r#"git "pu$(printf sh)""#).unwrap_err();
        assert!(err.contains("command substitution"), "got: {err}");
        for cmd in [
            r#"git "$(printf 'push')""#,
            r#"git "$(printf 'pu')sh""#,
            "git $(printf 'push')",
            r#"git status $(whoami)"#,
            r#"bash "$(printf -- '-c')" 'echo x'"#,
            r#"python3 "$(printf -- '-c')" 'print(1)'"#,
            r#"find "$(printf '.')" -name x"#,
            r#"rm -rf "$(printf '*')""#,
            "mv \"$(printf 'x')\" y",
            // Variable-channel escapes: the output would flow into `$cmd` /
            // `$i`, which no segment can audit.
            "for cmd in $(printf rm); do $cmd -rf target; done",
            r#"for f in "$(printf 'a b')"; do echo $f; done"#,
            "cmd=$(printf rm); $cmd -rf target",
            "cmd=$(printf rm)",
            "FOO=$(mktemp) git status",
            "FOO=bar$(seq 1 5) git status",
            // Unquoted inner with confirmation-gated side effects: runs the
            // commit without the confirmation gate.
            "echo $(git commit -am x)",
            // Partial-word quoted form is not materializable.
            r#"echo "user=$(whoami)""#,
            // Condition position would execute the output as a command.
            r#"while "$(printf true)"; do echo x; done"#,
        ] {
            assert!(validate(cmd).is_err(), "expected blocked: {cmd}");
        }
        // Safe positions stay allowed: literal-seq loop lists (bare or quoted
        // whole-word), materializable whole-word `"$(...)"` data arguments for
        // echo/printf; quoted-heredoc bodies hold literal text, and single
        // quotes keep `$(` literal.
        for cmd in [
            "for i in $(seq 1 40); do echo $i; done",
            r#"for i in "$(seq 1 5)"; do echo $i; done"#,
            r#"echo "$(date)""#,
            r#"printf "%s\n" "$(cat /tmp/x)""#,
            "cat <<'EOF'\n$(whoami)\nEOF",
            "echo 'price: $(100)'",
        ] {
            assert!(validate(cmd).is_ok(), "expected allowed: {cmd}");
        }
    }

    #[test]
    fn detects_literal_file_read_substitution_for_any_simple_outer_command() {
        // The FileRead branch of safe_shell_substitutions (replaced the removed
        // safe_file_read_substitutions)
        let substitutions = safe_shell_substitutions(
            r#"curl --data "$(cat /tmp/request.json)" https://example.test/api"#,
        );
        assert_eq!(substitutions.len(), 1);
        assert_eq!(
            substitutions[0].kind,
            SafeShellSubstitutionKind::FileRead {
                path: "/tmp/request.json".to_string()
            }
        );
    }

    #[test]
    fn literal_file_read_substitution_requires_a_complete_simple_shell_word() {
        let kinds: Vec<_> = safe_shell_substitutions(r#"echo "$(cat /tmp/dsl.json)""#)
            .into_iter()
            .map(|s| s.kind)
            .collect();
        assert_eq!(
            kinds,
            vec![SafeShellSubstitutionKind::FileRead {
                path: "/tmp/dsl.json".to_string()
            }]
        );
        assert_eq!(
            safe_shell_substitutions(r#"bytedcli --dsl "$(cat /tmp/dsl.json)""#).len(),
            1
        );
        assert_eq!(
            safe_shell_substitutions(r#"echo "$(cat /tmp/a)" "$(cat /tmp/b)""#).len(),
            2
        );
        // Outer pipes/connectors -> the whole command is not recognized as a safe
        // substitution
        assert!(safe_shell_substitutions(r#"echo "$(cat /tmp/dsl.json)" | jq ."#).is_empty());
        assert!(safe_shell_substitutions(r#"echo "$(cat /tmp/dsl.json)" && id"#).is_empty());
        // Complete-word or embedded `$()` are only allowed end-to-end for
        // pure-data programs (`echo`): the inner command is validated (`cat`
        // of a literal absolute path) and the program name is unaffected.
        // For any other program the raw audit is fail-closed without
        // materialization — production executes the safe FileRead whole-word
        // form via command.rs materialization before this audit runs.
        assert!(validate(r#"echo "$(cat /tmp/dsl.json)""#).is_ok());
        assert!(validate(r#"bytedcli --dsl "prefix$(cat /tmp/dsl.json)""#).is_err());
        assert!(validate(r#"bytedcli --dsl "$(cat /tmp/dsl.json)suffix""#).is_err());
        assert!(validate(r#"bytedcli --dsl "$(cat /tmp/dsl.json)""#).is_err());
    }

    #[test]
    fn file_read_substitution_requires_one_literal_absolute_path() {
        // Non-absolute, non-literal paths must not materialize as FileRead (may
        // fall back to a validate-passed harmless Command)
        for command in [
            r#"echo "$(cat /tmp/a /tmp/b)""#,
            r#"echo "$(cat $HOME/a)""#,
            r#"echo "$(cat /tmp/a; id)""#,
            r#"echo "$(cat /tmp/a$(id))""#,
            r#"echo "$(cat /tmp/../secret)""#,
            r#"echo "$(cat /tmp/*.json)""#,
        ] {
            let substitutions = safe_shell_substitutions(command);
            assert!(
                substitutions
                    .iter()
                    .all(|s| !matches!(s.kind, SafeShellSubstitutionKind::FileRead { .. })),
                "unsafe cat path must not be materialized as FileRead: {command}"
            );
        }
        // Nested $() rejected as a whole
        assert!(safe_shell_substitutions(r#"echo "$(cat /tmp/a$(id))""#).is_empty());
    }

    #[test]
    fn allows_arithmetic_expansion() {
        assert!(validate("echo $((RANDOM % 20 + 1))").is_ok());
        assert!(validate("echo $((1 + 2 * 3))").is_ok());
    }

    #[test]
    fn blocks_command_substitution_nested_in_arithmetic() {
        let err = validate("echo $(( $(whoami) + 1 ))").unwrap_err();
        assert!(
            err.contains("command substitution"),
            "expected nested $(...) blocked, got: {err}"
        );
    }

    #[test]
    fn allows_subcommand_patterns_that_resemble_blocked_programs() {
        // `git rm` is now blocked by BLOCKED_GIT_SUBCOMMANDS (unrecoverable
        // deletion).
        assert!(validate("git rm file.txt").is_err());
        assert!(validate("git mv old.txt new.txt").is_ok());
        assert!(validate("docker rm my_container").is_ok());
        assert!(validate("docker rmi my_image").is_ok());
        assert!(validate("npm rm some-package").is_ok());
        assert!(validate("pip install rsync").is_ok());
    }

    #[test]
    fn blocks_git_push_in_all_common_forms() {
        assert!(validate("git push").is_err());
        assert!(validate("git push origin main").is_err());
        assert!(validate("git push --force").is_err());
        assert!(validate("git push --force-with-lease origin").is_err());
        assert!(validate("git push -u origin main").is_err());
        assert!(validate("git push origin --tags").is_err());
        assert!(validate("git -C /repo push").is_err());
        assert!(validate("git -C /repo push origin main").is_err());
        assert!(validate("git -c user.email=a@b.c push").is_err());
        assert!(validate("git --git-dir=/repo push").is_err());
        assert!(validate("git --git-dir /repo push").is_err());
        assert!(validate("git --no-pager push").is_err());
        assert!(validate("git PUSH origin").is_err());
        assert!(validate("/usr/bin/git push").is_err());
        assert!(validate("git status && git push").is_err());
        assert!(validate("git push && echo done").is_err());
        assert!(validate("env git push").is_err());
        assert!(validate("env FOO=1 git push origin main").is_err());
        assert!(validate("xargs git push").is_err());
        assert!(validate("nohup git push").is_err());
        assert!(validate("command git push").is_err());
    }

    #[test]
    fn git_non_push_subcommands_remain_allowed() {
        assert!(validate("git status").is_ok());
        assert!(validate("git log --oneline -5").is_ok());
        assert!(validate("git diff").is_ok());
        assert!(validate("git diff --cached").is_ok());
        assert!(validate("git -C /repo status").is_ok());
        assert!(validate("git -C /repo log --oneline").is_ok());
        assert!(validate("git add -A").is_ok());
        assert!(validate("git commit -m msg").is_ok());
        assert!(validate("echo git push").is_ok());
        assert!(validate("printf '%s' push").is_ok());
    }

    #[test]
    fn allows_git_stash_forms_at_audit_level() {
        // `git stash` is no longer hard-blocked here: the whole stash family is
        // gated by user confirmation in `service/command.rs` (same as `git
        // commit`), which prompts on an interactive terminal and fails closed
        // otherwise. The audit layer must therefore let every form through so
        // the confirmation gate downstream is the single decision point.
        assert!(validate("git stash").is_ok());
        assert!(validate("git stash list").is_ok());
        assert!(validate("git stash pop").is_ok());
        assert!(validate("git stash drop").is_ok());
        assert!(validate("git stash clear").is_ok());
        assert!(validate("git stash push -m wip").is_ok());
        assert!(validate("git -C /repo stash").is_ok());
        assert!(validate("git -c user.email=a@b.c stash").is_ok());
        assert!(validate("git --git-dir=/repo stash").is_ok());
        assert!(validate("git --no-pager stash").is_ok());
        assert!(validate("git STASH").is_ok());
        assert!(validate("/usr/bin/git stash").is_ok());
        assert!(validate("git status && git stash").is_ok());
        assert!(validate("git stash && echo done").is_ok());
        assert!(validate("env git stash").is_ok());
        assert!(validate("xargs git stash").is_ok());
        assert!(validate("nohup git stash").is_ok());
        assert!(validate("command git stash").is_ok());
        assert!(validate("echo git stash").is_ok());
        assert!(validate("printf '%s' stash").is_ok());
    }

    #[test]
    fn shell_literal_rm_text_remains_allowed() {
        assert!(validate("echo 'rm -rf ~/.zcompdump*'").is_ok());
    }

    #[test]
    fn blocks_exec_flags_that_run_subsequent_args_as_commands() {
        assert!(validate("find . -exec rm {} +").is_err());
        assert!(validate("find . -execdir chmod 777 {} \\;").is_err());
        assert!(validate("find /tmp -ok rm {} \\;").is_err());
        assert!(validate("find . -okdir mv {} /tmp/ \\;").is_err());
        assert!(validate("find . -name '*.rs' -type f").is_ok());
        assert!(validate("find . -delete").is_err());
        assert!(validate("find . -empty -delete").is_err());
        assert!(validate(r#"find . "-exec" rm {} +"#).is_err());
        assert!(validate(r#"find . -name "-delete" -print"#).is_ok());
        assert!(validate(r#"find . -name "-exec" -print"#).is_ok());
        assert!(validate(r#"find . -printf "-delete\n""#).is_ok());
        // `git rm` is now blocked by BLOCKED_GIT_SUBCOMMANDS.
        assert!(validate("git rm file.txt").is_err());
        assert!(validate("docker rm container").is_ok());
        assert!(validate("npm rm pkg").is_ok());
        assert!(validate("pip install rsync").is_ok());
    }

    #[test]
    fn blocks_common_indirect_wrappers_but_allows_safe_payload_args() {
        assert!(validate("xargs rm").is_err());
        assert!(validate("env FOO=1 sudo whoami").is_err());
        assert!(validate("env FOO=1 rm -rf target").is_err());
        assert!(validate("nohup ssh user@host").is_err());
        assert!(validate("nice -n 5 chmod 777 file").is_err());
        assert!(validate("timeout --signal=KILL 10 dd if=/dev/zero of=foo").is_err());
        assert!(validate("command rm -rf *").is_err());
        assert!(validate("exec rm -rf *").is_err());

        assert!(validate(r#"xargs printf "%s\n" rm"#).is_ok());
        assert!(validate(r#"env FOO=1 cargo test"#).is_ok());
        assert!(validate(r#"nice -n 5 cargo check"#).is_ok());
        assert!(validate(r#"timeout 10 cargo test"#).is_ok());
    }

    #[test]
    fn leading_env_assignment_only_has_shell_meaning_when_shell_is_used() {
        assert!(validate("FOO=1 rm -rf target").is_ok());
        assert!(validate("FOO=1 rm -rf *.tmp").is_err());
    }

    #[test]
    fn allows_literal_dangerous_text_when_writing_files() {
        assert!(validate(r#"printf "%s\n" "-exec" "-delete" "rm -rf /""#).is_ok());
        assert!(validate("cat <<'EOF' > out.txt\n$(whoami)\n-exec\n-delete\nEOF").is_ok());
        assert!(validate("cat <<'EOF' > out.txt\n`whoami`\nEOF").is_ok());
        assert!(validate("printf '%s\n' '`whoami`'").is_ok());
    }

    #[test]
    fn allows_normal_dev_commands() {
        assert!(validate("cargo check --bin a").is_ok());
        assert!(validate("git status").is_ok());
        assert!(validate("ls -la").is_ok());
        assert!(validate("echo 'literal $(x)'").is_ok());
    }

    // ---- tilde / $HOME escape detection ----

    #[test]
    fn home_paths_are_allowed() {
        assert!(validate("ls ~").is_ok());
        assert!(validate("cat ~/.gitconfig").is_ok());
        assert!(validate("cat $HOME/.cargo/config.toml").is_ok());
    }

    #[test]
    fn tilde_escape_to_parent_dir_blocked() {
        // cwd=/Users/bytedance/rust_tools -> ~/.. walks up to /Users/bytedance ->
        // /Users -> /
        assert!(validate("cp foo.txt ~/../..").is_err());
        assert!(validate("cp foo.txt ~/..").is_err());
    }

    #[test]
    fn tilde_to_parent_blocked() {
        assert!(validate("ls ~/..").is_err());
    }

    #[test]
    fn home_env_var_escape_blocked() {
        assert!(validate("cp foo.txt $HOME/../../..").is_err());
    }

    // ---- python -c code-string audit ----

    #[test]
    fn python_dash_c_clean_code_allowed() {
        assert!(validate("python3 -c 'print(1 + 1)'").is_ok());
        assert!(validate(r#"python3 -c "import json; print(json.load(open('x.json')))""#).is_ok());
        assert!(validate("python -c 'print(sum(i*i for i in range(10)))'").is_ok());
        assert!(validate("python3 -u -c 'print(\"hi\")'").is_ok());
        assert!(validate("python3 -c'print(1)'").is_ok());
        assert!(validate("python3 -W ignore -c 'print(1)'").is_ok());
        assert!(validate("python3 -c 'print(len(\"abc\"))'").is_ok());
        assert!(validate("python3 -c 'import re; print(re.findall(r\"\\d+\", \"a1b2\"))'").is_ok());
        // Relaxed imports: plain `import os` / `import shutil` with read-only
        // attribute use stays allowed.
        assert!(validate("python3 -c 'import os; print(os.getcwd())'").is_ok());
        assert!(validate("python3 -c 'import os; print(os.path.exists(\"x\"))'").is_ok());
        assert!(validate("python3 -c 'import os; print(os.environ.get(\"HOME\"))'").is_ok());
        assert!(validate("python3 -c 'import shutil; print(shutil.which(\"git\"))'").is_ok());
        assert!(validate("python3 -c 'import os.path as p; print(p.exists(\"x\"))'").is_ok());
    }

    #[test]
    fn python_dash_c_dangerous_code_blocked() {
        let err = validate("python3 -c 'import os; os.system(\"rm -rf /\")'").unwrap_err();
        assert!(err.contains("blocked primitive"), "got: {err}");
        assert!(validate("python3 -c 'os.remove(\"x\")'").is_err());
        assert!(validate("python3 -c 'import subprocess; subprocess.run([\"ls\"])'").is_err());
        assert!(validate("python3 -c 'from subprocess import call; call(\"ls\")'").is_err());
        assert!(validate("python3 -c 'eval(\"1+1\")'").is_err());
        assert!(validate("python3 -c 'exec(\"x=1\")'").is_err());
        assert!(validate("python3 -c 'getattr(os, \"system\")(\"rm -rf /\")'").is_err());
        assert!(validate("python3 -c '__import__(\"os\").system(\"id\")'").is_err());
        assert!(validate("python3 -c 'shutil.rmtree(\"d\")'").is_err());
        assert!(validate("python3 -c 'import socket; socket.socket()'").is_err());
        assert!(validate("python3 -c 'ctypes.CDLL(None).system(\"id\")'").is_err());
        assert!(validate("python3 -c 'Path(\"x\").unlink()'").is_err());
        // Every import form of dangerous modules (from-import / aliasing /
        // sys.modules). Plain `import os` / `import shutil` are allowed for
        // read-only use, so the remaining blocked forms are exactly those that
        // expose bare or renamed dangerous calls.
        assert!(validate("python3 -c 'from os import system; system(\"rm -rf /\")'").is_err());
        assert!(validate("python3 -c 'import os as o; o.system(\"id\")'").is_err());
        assert!(validate("python3 -c 'from os import *'").is_err());
        assert!(validate("python3 -c 'import os as o; print(o.getcwd())'").is_err());
        assert!(validate("python3 -c 'import shutil as s; s.rmtree(\"d\")'").is_err());
        assert!(validate("python3 -c 'import os; os.__dict__[\"system\"](\"id\")'").is_err());
        assert!(validate("python3 -c 'import os; os.__getattribute__(\"system\")(\"id\")'").is_err());
        // `import os; x = os; x.system(...)` / `vars(os)["system"](...)`
        // (copied reference / vars indirection) are accepted residual blind
        // spots of the relaxed import policy; see validate_python_code.
        assert!(validate("python3 -c 'import sys; sys.modules[\"os\"].system(\"id\")'").is_err());
        assert!(validate("python3 -c 'import posix; posix.system(\"id\")'").is_err());
        assert!(validate("python3 -c 'import signal; signal.kill(1, 9)'").is_err());
        // Common obfuscations: still hit after stripping whitespace / changing
        // case.
        assert!(validate("python3 -c 'os . system(\"id\")'").is_err());
        assert!(validate("python3 -c 'OS.SYSTEM(\"id\")'").is_err());
        // Clustered short option `-uc` equals `-u -c`.
        assert!(validate("python3 -uc 'os.system(\"id\")'").is_err());
        // The `__subclasses__` sandbox escape chain.
        assert!(validate("python3 -c '().__class__.__bases__[0].__subclasses__()'").is_err());
    }

    #[test]
    fn python_dash_c_unverifiable_code_blocked() {
        // Code comes from shell variable expansion, statically unverifiable ->
        // fail-closed.
        assert!(validate("python3 -c $CODE").is_err());
        assert!(validate("CODE=x python3 -c $CODE").is_err());
        assert!(validate("python3 -c \"$CODE\"").is_err());
        // Missing / empty code.
        assert!(validate("python3 -c").is_err());
        assert!(validate("python3 -c ''").is_err());
        // Clustered short options carrying `-c` without code.
        assert!(validate("python3 -uc").is_err());
    }

    #[test]
    fn python_without_dash_c_unchanged() {
        assert!(validate("python3 script.py").is_ok());
        assert!(validate("python3 -m json.tool < data.json").is_ok());
        assert!(validate("python3 --version").is_ok());
    }

    #[test]
    fn grouped_short_options_caught() {
        // Clustered short options can smuggle `-c` / `-e` too: `bash -lc` /
        // `perl -le` / `node -pe` / `ruby -ne` are equivalent to `-c` / `-e` and
        // must not pass.
        assert!(validate("bash -lc 'rm -rf /'").is_err());
        assert!(validate("perl -le 'system(\"rm -rf /\")'").is_err());
        assert!(validate("node -pe 'require(\"child_process\").execSync(\"id\")'").is_err());
        assert!(validate("ruby -ne 'puts 1'").is_err());
        // Legitimate short options without `-c` / `-e` are unaffected (`-e` on a
        // shell is errexit, not code; `--norc` is a long option).
        assert!(validate("bash -e script.sh").is_ok());
        assert!(validate("bash --norc script.sh").is_ok());
        assert!(validate("perl -w script.pl").is_ok());
    }

    #[test]
    fn indirect_interpreter_dash_c_audited() {
        // Clean indirect python -c still passes.
        assert!(validate("env python3 -c 'print(1)'").is_ok());
        assert!(validate("nohup env python3 -c 'print(1)'").is_ok());
        assert!(validate("env env python3 -c 'print(1)'").is_ok());
        // Wrappers can no longer bypass validation via `-c`.
        let err = validate("env python3 -c 'os.system(\"id\")'").unwrap_err();
        assert!(err.contains("blocked primitive"), "got: {err}");
        assert!(validate("xargs python3 -c 'os.system(\"id\")'").is_err());
        assert!(validate("nohup python3 -c 'os.system(\"id\")'").is_err());
        // Layered wrappers (`nohup env python3 -c ...`) are likewise caught by
        // deep unwrapping.
        assert!(validate("nohup env python3 -c 'os.system(\"id\")'").is_err());
        assert!(validate("env bash -c 'echo ok && rm -rf /'").is_err());
        assert!(validate("timeout 10 bash -c 'rm -rf /'").is_err());
        assert!(validate("env perl -e 'system(\"rm -rf /\")'").is_err());
        assert!(validate("env env bash -c 'rm -rf /'").is_err());
    }
}

#[cfg(test)]
mod search_scope_tests {
    use super::*;

    fn blocked(command: &str) -> String {
        validate_execute_command(command).unwrap_err()
    }

    fn allowed(command: &str) {
        validate_execute_command(command).unwrap();
    }

    // ---- find ----

    #[test]
    fn find_absolute_root_outside_cwd_is_blocked() {
        let err = blocked("find / -maxdepth 2 -name request.txt");
        assert!(err.contains("outside the current directory"), "got: {err}");
    }

    #[test]
    fn find_parent_relative_root_is_blocked() {
        let err = blocked("find .. -name request.txt");
        assert!(err.contains("outside the current directory"), "got: {err}");
    }

    #[test]
    fn find_bare_name_not_under_cwd_is_blocked() {
        let err = blocked("find . -maxdepth 4 -name request.txt");
        assert!(err.contains("relative name 'request.txt'"), "got: {err}");
        assert!(err.contains("ask the user"), "got: {err}");
    }

    #[test]
    fn find_relative_target_inside_cwd_is_allowed() {
        // Cargo.toml is a direct child of the crate root (test cwd).
        allowed("find . -maxdepth 2 -name Cargo.toml");
    }

    #[test]
    fn find_glob_patterns_are_allowed() {
        allowed("find . -name '*.rs' -o -name '*.toml'");
        allowed("find src -maxdepth 3 -iname '*.json'");
    }

    #[test]
    fn incident_shape_global_name_hunt_is_blocked() {
        // The test_llm vs. AeolusLLM request.txt incident: a whole-disk
        // bare-name hunt (absolute root outside the cwd, piped to head) must
        // be rejected before it can pick the wrong duplicate.
        let err = blocked(
            "find /Users/bytedance -maxdepth 4 -name request.txt -o -maxdepth 4 \
             -name response.txt 2>/dev/null | head -20",
        );
        assert!(err.contains("outside the current directory"), "got: {err}");
    }

    // ---- glob ----

    #[test]
    fn absolute_glob_outside_cwd_is_blocked() {
        let err = blocked("cat /Users/*/request.txt");
        assert!(err.contains("glob pattern '/Users/*/request.txt'"), "got: {err}");
        let err = blocked("ls /usr/*");
        assert!(err.contains("outside the current directory"), "got: {err}");
    }

    #[test]
    fn parent_escaping_glob_is_blocked() {
        let err = blocked("cat ../*.rs");
        assert!(err.contains("outside the current directory"), "got: {err}");
    }

    #[test]
    fn relative_globs_are_allowed() {
        allowed("ls *.rs");
        allowed("echo src/*.rs");
        allowed("ls **/*.rs");
        allowed("cat '*.txt'");
    }

    // ---- grep / rg ----

    #[test]
    fn grep_recursive_search_outside_cwd_is_blocked() {
        let err = blocked("grep -rn pattern /etc");
        assert!(err.contains("search path '/etc'"), "got: {err}");
        let err = blocked("grep -r pattern ..");
        assert!(err.contains("outside the current directory"), "got: {err}");
        let err = blocked("rg pattern /Users/bytedance");
        assert!(err.contains("outside the current directory"), "got: {err}");
    }

    #[test]
    fn grep_within_cwd_is_allowed() {
        allowed("grep -rn pattern src");
        allowed("grep -rn --include='*.rs' pattern src");
        allowed("grep -e foo -e bar src");
        // Non-recursive single-file reads stay unrestricted.
        allowed("grep -n pattern /etc/hosts");
    }

    /// A throwaway project root under the session temp dir (always an allowed
    /// search root) holding a build directory and a source directory. Each test
    /// names its own fixture, since the tests run in parallel.
    fn with_temp_search_root<F: FnOnce(&std::path::Path)>(fixture: &str, f: F) {
        const TEST_SESSION: &str = "grep-build-dir-test-session";
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .sync_scope((TEST_SESSION.to_string(), 0), || {
                let tmp = crate::ai::driver::runtime_ctx::temp_dir()
                    .expect("session temp dir must resolve in tests");
                let root = tmp.join(fixture);
                // A previous run may have left the fixture (and other tests'
                // additions) behind: start from a known state.
                let _ = std::fs::remove_dir_all(&root);
                std::fs::create_dir_all(root.join("target")).unwrap();
                std::fs::create_dir_all(root.join("src")).unwrap();
                f(&root);
            });
    }

    #[test]
    fn recursive_grep_that_would_walk_build_output_is_blocked() {
        with_temp_search_root("grep-build-dir-fixture", |root| {
            // A build directory nested inside another one: rooting the search
            // above it (`target/debug`) must still count as deliberate.
            std::fs::create_dir_all(root.join("target/debug/build")).unwrap();
            let root = root.display().to_string();
            let err = blocked(&format!("grep -rn pattern {root}"));
            assert!(err.contains("'target/'"), "got: {err}");
            assert!(err.contains("on purpose"), "got: {err}");
            // Attached and spaced exclusions, a narrower root, a root inside the
            // build output, and ignore-aware tools all pass.
            allowed(&format!("grep -rn --exclude-dir=target pattern {root}"));
            allowed(&format!("grep -rn --exclude-dir target pattern {root}"));
            allowed(&format!("grep -rn pattern {root}/src"));
            allowed(&format!("grep -rn pattern {root}/target"));
            allowed(&format!("grep -rn pattern {root}/target/debug"));
            allowed(&format!("rg pattern {root}"));
        });
    }

    #[test]
    fn recursive_grep_needs_every_build_dir_excluded() {
        with_temp_search_root("grep-multi-build-dir-fixture", |root| {
            std::fs::create_dir_all(root.join("node_modules")).unwrap();
            let root = root.display().to_string();
            // Excluding only `target` still leaves `node_modules` in the walk.
            let err = blocked(&format!("grep -rn --exclude-dir=target pattern {root}"));
            assert!(err.contains("'node_modules/'"), "got: {err}");
            allowed(&format!(
                "grep -rn --exclude-dir=target --exclude-dir=node_modules pattern {root}"
            ));
        });
    }

    #[test]
    fn recursive_grep_finds_build_dirs_nested_in_a_workspace() {
        with_temp_search_root("grep-workspace-fixture", |root| {
            // Monorepo layout: the heavy directories sit three levels down.
            std::fs::create_dir_all(root.join("packages/web/node_modules")).unwrap();
            std::fs::create_dir_all(root.join("apps/api/.venv")).unwrap();
            // Beyond the scan's depth bound: the check stays cheap, and probing
            // this one stays the model's own decision.
            std::fs::create_dir_all(root.join("deep/one/two/venv")).unwrap();
            let root = root.display().to_string();
            let err = blocked(&format!("grep -rn pattern {root}"));
            assert!(err.contains("'node_modules/'"), "got: {err}");
            assert!(err.contains("'.venv/'"), "got: {err}");
            assert!(err.contains("'target/'"), "got: {err}");
            assert!(!err.contains("'venv/'"), "got: {err}");
            // Excluding every reported name lets the retry through.
            allowed(&format!(
                "grep -rn --exclude-dir=target --exclude-dir=node_modules \
                 --exclude-dir=.venv pattern {root}"
            ));
            // Rooting the search inside one of them is the deliberate path.
            allowed(&format!("grep -rn pattern {root}/packages/web/node_modules"));
        });
    }

    #[test]
    fn search_skip_dirs_config_extends_and_trims_the_builtin_list() {
        let defaults = merge_heavy_search_dirs("");
        assert!(defaults.iter().any(|name| name == "node_modules"));
        let merged = merge_heavy_search_dirs(" bazel-bin, -build, bazel-bin ");
        assert!(merged.iter().any(|name| name == "bazel-bin"));
        assert!(!merged.iter().any(|name| name == "build"));
        assert_eq!(merged.iter().filter(|name| *name == "bazel-bin").count(), 1);
    }

    #[test]
    fn long_directory_lists_are_summarized() {
        let names: Vec<String> = ["a", "b", "c", "d"].iter().map(|name| name.to_string()).collect();
        assert_eq!(summarize_dirs(&names), "'a/', 'b/', 'c/', and 1 more");
    }

    // ---- locate ----

    #[test]
    fn locate_global_search_is_blocked() {
        let err = blocked("locate request.txt");
        assert!(err.contains("ask the user"), "got: {err}");
        let err = blocked("locate '*.log'");
        assert!(err.contains("whole-disk"), "got: {err}");
    }

    // ---- helpers ----

    #[test]
    fn strip_quotes_handles_matching_pairs() {
        assert_eq!(strip_quotes("'abc'"), "abc");
        assert_eq!(strip_quotes("\"abc\""), "abc");
        assert_eq!(strip_quotes("abc"), "abc");
        assert_eq!(strip_quotes("'abc"), "'abc");
    }

    #[test]
    fn glob_metachar_detection() {
        assert!(has_glob_metachar("a*b"));
        assert!(has_glob_metachar("a?b"));
        assert!(has_glob_metachar("a[b]"));
        assert!(has_glob_metachar("{a,b}"));
        assert!(!has_glob_metachar("request.txt"));
    }

    #[test]
    fn find_quoted_parent_root_is_blocked() {
        // The shell strips quotes before `find` runs, so `find '..'` really
        // searches the parent directory; the confinement check must apply to
        // the unquoted path, not the quoted string.
        let err = blocked("find '..' -name request.txt");
        assert!(err.contains("outside the current directory"), "got: {err}");
        let err = blocked("find \"..\" -maxdepth 2 -name request.txt");
        assert!(err.contains("outside the current directory"), "got: {err}");
        let err = blocked("find './../..' -name request.txt");
        assert!(err.contains("outside the current directory"), "got: {err}");
    }

    #[test]
    fn rg_files_and_ag_glob_have_no_pattern_positional() {
        // `rg --files` / `ag -g GLOB` list file paths without a pattern, so
        // every positional is a search path and must be confined.
        let err = blocked("rg --files /Users/bytedance");
        assert!(err.contains("outside the current directory"), "got: {err}");
        let err = blocked("ag -g '*.rs' /Users/bytedance");
        assert!(err.contains("outside the current directory"), "got: {err}");
        allowed("rg --files src");
    }

    #[test]
    fn filter_glob_option_values_are_not_path_globs() {
        // `--glob` / `--include` values are filter patterns, not search
        // paths; they must not be rejected as escaping globs.
        allowed("rg --glob /etc/*.conf pattern src");
        allowed("grep -rn --include /usr/*.h pattern src");
    }
}

#[cfg(test)]
mod kill_target_tests {
    use super::*;
    use crate::ai::tools::storage::process_registry;

    fn tokens(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    // Kill validation is session-scoped: register the fake/real pgids under a
    // dedicated session id and validate against it, so tests never depend on
    // a live DRIVER_CTX and never touch other tests' entries.
    const TEST_SESSION: &str = "kill-target-test-session";

    #[test]
    fn kill_of_registered_pgid_is_allowed() {
        process_registry::register(TEST_SESSION, 4242);
        validate_kill_pids(&tokens(&["kill", "4242"]), TEST_SESSION).unwrap();
        validate_kill_pids(&tokens(&["kill", "-9", "4242"]), TEST_SESSION).unwrap();
        validate_kill_pids(&tokens(&["kill", "-s", "TERM", "4242"]), TEST_SESSION).unwrap();
        validate_kill_pids(&tokens(&["kill", "--signal=KILL", "4242"]), TEST_SESSION).unwrap();
        // Negative pid kills the whole registered group.
        validate_kill_pids(&tokens(&["kill", "-4242"]), TEST_SESSION).unwrap();
        // `--` ends option parsing; following tokens are still targets.
        validate_kill_pids(&tokens(&["kill", "--", "4242"]), TEST_SESSION).unwrap();
    }

    #[test]
    fn kill_of_unregistered_target_is_denied() {
        let err = validate_kill_pids(&tokens(&["kill", "4243"]), TEST_SESSION).unwrap_err();
        assert!(
            err.contains("not a process started by this agent session"),
            "got: {err}"
        );
        let err = validate_kill_pids(&tokens(&["kill", "-4243"]), TEST_SESSION).unwrap_err();
        assert!(
            err.contains("not a process started by this agent session"),
            "got: {err}"
        );
    }

    #[test]
    fn kill_of_descendant_in_registered_group_is_allowed() {
        // The test process itself: register its process group, then kill its
        // own pid. The pid is not the pgid, so this exercises the `ps`
        // pgid-resolution path end to end.
        let pid = std::process::id();
        let pgid = process_group_of(pid).expect("ps must resolve the test process pgid");
        process_registry::register(TEST_SESSION, pgid);
        validate_kill_pids(&tokens(&["kill", &pid.to_string()]), TEST_SESSION).unwrap();
    }

    #[test]
    fn kill_parse_failures_fail_closed() {
        let err = validate_kill_pids(&tokens(&["kill"]), TEST_SESSION).unwrap_err();
        assert!(err.contains("no verifiable target"), "got: {err}");
        let err = validate_kill_pids(&tokens(&["kill", "$$"]), TEST_SESSION).unwrap_err();
        assert!(err.contains("not a literal pid"), "got: {err}");
        let err = validate_kill_pids(&tokens(&["kill", "0"]), TEST_SESSION).unwrap_err();
        assert!(err.contains("cannot be verified"), "got: {err}");
        let err = validate_kill_pids(&tokens(&["kill", "abc"]), TEST_SESSION).unwrap_err();
        assert!(err.contains("not a literal pid"), "got: {err}");
        // No session at all fails closed.
        let err = validate_kill_pids(&tokens(&["kill", "4242"]), "").unwrap_err();
        assert!(
            err.contains("not a process started by this agent session"),
            "got: {err}"
        );
    }

    #[test]
    fn pkill_unsupported_options_are_denied() {
        let err =
            validate_pattern_kill(&tokens(&["pkill", "-v", "app"]), TEST_SESSION, "pkill", PgrepMode::Pkill)
                .unwrap_err();
        assert!(err.contains("cannot be verified"), "got: {err}");
        let err =
            validate_pattern_kill(&tokens(&["pkill", "-u", "root", "app"]), TEST_SESSION, "pkill", PgrepMode::Pkill)
                .unwrap_err();
        assert!(err.contains("cannot be verified"), "got: {err}");
        let err =
            validate_pattern_kill(&tokens(&["pkill"]), TEST_SESSION, "pkill", PgrepMode::Pkill)
                .unwrap_err();
        assert!(err.contains("no pattern given"), "got: {err}");
        let err = validate_pattern_kill(
            &tokens(&["killall", "-m", "py.*"]),
            TEST_SESSION,
            "killall",
            PgrepMode::Killall,
        )
        .unwrap_err();
        assert!(err.contains("cannot be verified"), "got: {err}");
    }

    #[test]
    fn pkill_matching_external_process_is_denied() {
        // `.` matches every process's command line (including pgrep's own
        // caller), and none of them is registered in this session.
        let err = validate_pattern_kill(
            &tokens(&["pkill", "-f", "."]),
            TEST_SESSION,
            "pkill",
            PgrepMode::Pkill,
        )
        .unwrap_err();
        assert!(err.contains("refusing to signal external processes"), "got: {err}");
    }

    #[test]
    fn pkill_matching_registered_process_is_allowed() {
        // Kill the test process itself by its exact executable path: register
        // its pgid, then `pkill -f <binary path>` must pass verification.
        let pgid = process_group_of(std::process::id()).expect("ps must resolve the test pgid");
        process_registry::register(TEST_SESSION, pgid);
        let binary = std::env::current_exe()
            .expect("test binary path")
            .to_string_lossy()
            .to_string();
        validate_pattern_kill(
            &tokens(&["pkill", "-f", &binary]),
            TEST_SESSION,
            "pkill",
            PgrepMode::Pkill,
        )
        .unwrap();
    }

    #[test]
    fn pkill_with_no_matches_is_allowed() {
        // A pattern nothing is running matches: the kill would signal zero
        // processes, so there is nothing to refuse.
        let name = format!("definitely-not-running-{}", std::process::id());
        validate_pattern_kill(
            &tokens(&["pkill", "-x", &name]),
            TEST_SESSION,
            "pkill",
            PgrepMode::Pkill,
        )
        .unwrap();
        validate_pattern_kill(
            &tokens(&["killall", &name]),
            TEST_SESSION,
            "killall",
            PgrepMode::Killall,
        )
        .unwrap();
    }

    #[test]
    fn kill_is_no_longer_blanket_blocked() {
        // Without a session context every kill fails closed, but with the
        // target-verification message, not the old blanket block.
        let err = validate_execute_command("kill 4242").unwrap_err();
        assert!(
            err.contains("not a process started by this agent session"),
            "got: {err}"
        );
        // Wrappers run the same verification instead of a blanket block.
        let err = validate_execute_command("env kill 4242").unwrap_err();
        assert!(
            err.contains("not a process started by this agent session"),
            "got: {err}"
        );
        // xargs-driven kills cannot be verified from the command line.
        let err = validate_execute_command("echo 4242 | xargs kill").unwrap_err();
        assert!(err.contains("behind 'xargs'"), "got: {err}");
    }

    #[test]
    fn kill_through_xargs_is_blocked() {
        // xargs appends stdin items as extra runtime arguments, so the kill
        // target set is never fully visible on the command line: the whole
        // invocation must be refused even when a literal pid is present.
        let err = validate_execute_command("xargs kill 123").unwrap_err();
        assert!(err.contains("behind 'xargs'"), "got: {err}");
        let err = validate_execute_command("timeout 5 xargs kill 123").unwrap_err();
        assert!(err.contains("behind 'xargs'"), "got: {err}");
        let err = validate_execute_command("xargs pkill -f app.py").unwrap_err();
        assert!(err.contains("behind 'xargs'"), "got: {err}");
        // A pattern argument named `xargs` is not a wrapper: `pkill -f xargs`
        // kills processes whose command line contains "xargs", so it must not
        // hit the xargs block (the pattern is still target-verified).
        assert!(!effective_chain_uses_xargs("pkill -f xargs"));
        assert!(!effective_chain_uses_xargs("kill 123"));
        assert!(effective_chain_uses_xargs("xargs kill 123"));
        assert!(effective_chain_uses_xargs("env xargs kill 123"));
        assert!(effective_chain_uses_xargs("timeout 5 xargs kill 123"));
    }
}

#[cfg(test)]
mod rm_target_tests {
    use super::*;

    fn tokens(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    // Temp-dir resolution is session-scoped (`.agent_tmp/<session>/` without a
    // live DRIVER_CTX); pin a dedicated session so tests never depend on a
    // live DRIVER_CTX and never touch other tests' entries.
    const TEST_SESSION: &str = "rm-target-test-session";

    fn with_temp_dir<F: FnOnce(&std::path::Path)>(f: F) {
        crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
            (TEST_SESSION.to_string(), 0),
            || {
                let tmp = crate::ai::driver::runtime_ctx::temp_dir()
                    .expect("session temp dir must resolve in tests");
                f(&tmp);
            },
        );
    }

    #[test]
    fn rm_inside_temp_dir_is_allowed() {
        with_temp_dir(|tmp| {
            let target = tmp.join("build-output.bin").display().to_string();
            validate_rm_targets(&tokens(&["rm", "-f", &target])).unwrap();
            validate_rm_targets(&tokens(&["rm", "--", &target])).unwrap();
            // Recursive removal under the temp dir.
            let dir = tmp.join("scratch").display().to_string();
            validate_rm_targets(&tokens(&["rm", "-rf", &dir])).unwrap();
            // Quoted targets are stripped like the shell would.
            let quoted = format!("'{target}'");
            validate_rm_targets(&tokens(&["rm", "-f", &quoted])).unwrap();
        });
    }

    #[test]
    fn rm_temp_globs_stay_confined() {
        with_temp_dir(|tmp| {
            let glob = tmp.join("*.log").display().to_string();
            validate_rm_targets(&tokens(&["rm", "-f", &glob])).unwrap();
            // `..` in the pattern collapses before the prefix check, so an
            // escape attempt is caught even when it is lexically "inside".
            let escape = tmp.join("..").join("x").display().to_string();
            let err = validate_rm_targets(&tokens(&["rm", "-rf", &escape])).unwrap_err();
            assert!(err.contains("outside the session temp dir"), "got: {err}");
        });
    }

    #[test]
    fn rm_outside_temp_dir_is_denied() {
        with_temp_dir(|tmp| {
            for cmd in [
                format!("rm -rf {}", tmp.join("..").join("target").display()),
                "rm -rf /etc/passwd".to_string(),
                "rm -f ./target".to_string(),
                "rm -rf *.zcompdump".to_string(),
            ] {
                let err = validate_execute_command(&cmd).unwrap_err();
                assert!(
                    err.contains("outside the session temp dir"),
                    "cmd `{cmd}`: got: {err}"
                );
            }
        });
    }

    #[test]
    fn rm_parse_failures_fail_closed() {
        with_temp_dir(|_| {
            let err = validate_rm_targets(&tokens(&["rm"])).unwrap_err();
            assert!(err.contains("no verifiable target"), "got: {err}");
            let err = validate_rm_targets(&tokens(&["rm", "-rf"])).unwrap_err();
            assert!(err.contains("no verifiable target"), "got: {err}");
            let err = validate_rm_targets(&tokens(&["rm", ""])).unwrap_err();
            assert!(err.contains("empty path"), "got: {err}");
        });
    }

    #[test]
    fn rm_through_wrappers_is_scoped() {
        with_temp_dir(|tmp| {
            let target = tmp.join("x").display().to_string();
            // Wrappers run the same verification.
            validate_execute_command(&format!("env rm -f {target}")).unwrap();
            validate_execute_command(&format!("timeout 5 env rm -f {target}")).unwrap();
            // Outside the temp dir stays blocked through wrappers.
            let err = validate_execute_command("env rm -f /etc/passwd").unwrap_err();
            assert!(err.contains("outside the session temp dir"), "got: {err}");
            let err = validate_execute_command("timeout 5 env rm -f /etc/passwd").unwrap_err();
            assert!(err.contains("outside the session temp dir"), "got: {err}");
        });
    }

    #[test]
    fn xargs_rm_is_blocked() {
        with_temp_dir(|tmp| {
            let target = tmp.join("x").display().to_string();
            let err = validate_execute_command(&format!("xargs rm -f {target}")).unwrap_err();
            assert!(err.contains("behind 'xargs'"), "got: {err}");
            let err =
                validate_execute_command(&format!("timeout 5 xargs rm {target}")).unwrap_err();
            assert!(err.contains("behind 'xargs'"), "got: {err}");
        });
    }

    #[test]
    fn rm_is_no_longer_blanket_blocked() {
        // Outside a session context every rm fails closed, but with the
        // temp-dir verification message rather than a blanket program block.
        let err = validate_execute_command("rm -rf /tmp/x").unwrap_err();
        assert!(err.contains("outside the session temp dir"), "got: {err}");
    }
}
