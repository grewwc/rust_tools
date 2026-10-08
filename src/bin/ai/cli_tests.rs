use std::fs;
use std::path::PathBuf;

#[test]
fn parse_cli_args_distill_session_accepts_zip_and_session_id() {
    for input in ["./archive.zip", "6e969353-01f4-40c2-bddb-59fae9970ad3"] {
        let cli = super::parse_cli_args(
            ["a", "--distill-session", input, "--distill-dry-run", "--distill-limit", "10"]
                .into_iter().map(str::to_owned),
        );
        assert_eq!(cli.distill_session.as_deref(), Some(input));
        assert!(cli.distill_dry_run);
        assert_eq!(cli.distill_limit, 10);
        assert!(cli.args.is_empty());
    }
}

#[test]
fn parse_cli_args_distill_session_preserves_missing_value_for_driver_error() {
    let cli = super::parse_cli_args(["a", "--distill-session"].into_iter().map(str::to_owned));
    assert_eq!(cli.distill_session.as_deref(), Some(""));
}

fn make_temp_file(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("ai-cli-{name}-{}.txt", uuid::Uuid::new_v4()));
    fs::write(&path, name).unwrap();
    path
}

#[test]
fn parse_cli_args_collects_space_separated_files_for_dash_f() {
    let first = make_temp_file("first");
    let second = make_temp_file("second");
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "-f".to_string(),
            first.to_string_lossy().to_string(),
            second.to_string_lossy().to_string(),
            "describe".to_string(),
        ]
        .into_iter(),
    );

    assert_eq!(
        cli.files,
        format!("{},{}", first.to_string_lossy(), second.to_string_lossy())
    );
    assert_eq!(cli.args, vec!["describe".to_string()]);

    let _ = fs::remove_file(first);
    let _ = fs::remove_file(second);
}

#[test]
fn parse_cli_args_merges_repeated_file_flags() {
    let first = make_temp_file("repeat-first");
    let second = make_temp_file("repeat-second");
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "-f".to_string(),
            first.to_string_lossy().to_string(),
            "--files".to_string(),
            second.to_string_lossy().to_string(),
            "summarize".to_string(),
        ]
        .into_iter(),
    );

    assert_eq!(
        cli.files,
        format!("{},{}", first.to_string_lossy(), second.to_string_lossy())
    );
    assert_eq!(cli.args, vec!["summarize".to_string()]);

    let _ = fs::remove_file(first);
    let _ = fs::remove_file(second);
}

#[test]
fn cli_parser_keeps_clear_and_completion_flags_visible() {
    let names = super::build_cli_parser()
        .collect_completion_info()
        .into_iter()
        .map(|(name, _, _, _)| name)
        .collect::<Vec<_>>();

    assert!(names.iter().any(|name| name == "clear"));
    assert!(names.iter().any(|name| name == "new-session"));
    assert!(names.iter().any(|name| name == "resume"));
    assert!(names.iter().any(|name| name == "generate-completions"));
}

#[test]
fn parse_cli_args_reads_new_session_flag() {
    let cli = super::parse_cli_args(["a".to_string(), "--new-session".to_string()].into_iter());

    assert!(cli.new_session);
}

#[test]
fn parse_cli_args_reads_resume_flag() {
    let cli = super::parse_cli_args(["a".to_string(), "--resume".to_string()].into_iter());

    assert!(cli.resume);
}

#[test]
fn parse_cli_args_reads_background_flag() {
    // Long form --background
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "fix the bug".to_string(),
            "--background".to_string(),
        ]
        .into_iter(),
    );
    assert!(cli.background);
    assert_eq!(cli.args, vec!["fix the bug".to_string()]);

    // Short alias -bg
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "fix the bug".to_string(),
            "-bg".to_string(),
        ]
        .into_iter(),
    );
    assert!(cli.background);
    assert_eq!(cli.args, vec!["fix the bug".to_string()]);

    // Defaults to false when -bg is not given
    let cli = super::parse_cli_args(["a".to_string(), "fix the bug".to_string()].into_iter());
    assert!(!cli.background);
}

#[test]
fn parse_cli_args_reads_stop_flag() {
    // --stop <sessionid>
    let cli = super::parse_cli_args(
        ["a".to_string(), "--stop".to_string(), "abc-123".to_string()].into_iter(),
    );
    assert_eq!(cli.stop_session, Some("abc-123".to_string()));

    // Defaults to None when --stop is not given
    let cli = super::parse_cli_args(["a".to_string()].into_iter());
    assert!(cli.stop_session.is_none());
}

#[test]
fn parse_cli_args_sharp_shortcut_selects_sharp_agent() {
    // `-s` selects the sharp agent and keeps the prompt.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "-s".to_string(),
            "quick question".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.agent.as_deref(), Some("sharp"));
    assert_eq!(cli.args, vec!["quick question".to_string()]);

    // Long form behaves the same.
    let cli = super::parse_cli_args(
        ["a".to_string(), "--sharp".to_string()].into_iter(),
    );
    assert_eq!(cli.agent.as_deref(), Some("sharp"));

    // `-s` selects sharp; the implied low reasoning effort is derived at
    // request time from the live agent, so parsing stores no override (this
    // keeps `/agent` switches and startup fallbacks from going stale).
    let cli = super::parse_cli_args(["a".to_string(), "-s".to_string()].into_iter());
    assert_eq!(cli.agent.as_deref(), Some("sharp"));
    assert!(cli.reasoning_effort_override.is_none());

    // An explicit `--reasoning-effort` always wins over the `-s` default.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "-s".to_string(),
            "--reasoning-effort".to_string(),
            "max".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.agent.as_deref(), Some("sharp"));
    assert_eq!(
        cli.reasoning_effort_override,
        Some(Some(super::ReasoningEffort::Max))
    );

    // An explicit `--agent` value wins over `-s`.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "-s".to_string(),
            "--agent".to_string(),
            "build".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.agent.as_deref(), Some("build"));
    // `-s` did not select sharp here, so it must not touch the effort default.
    assert!(cli.reasoning_effort_override.is_none());

    // `--agent sharp` without `-s` selects sharp the same way — still no
    // stored override; the low default is derived at resolve time.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "--agent".to_string(),
            "sharp".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.agent.as_deref(), Some("sharp"));
    assert!(cli.reasoning_effort_override.is_none());

    // A non-sharp agent never gets the low-effort default.
    let cli = super::parse_cli_args(
        ["a".to_string(), "--agent".to_string(), "build".to_string()].into_iter(),
    );
    assert_eq!(cli.agent.as_deref(), Some("build"));
    assert!(cli.reasoning_effort_override.is_none());

    // `-s` must not steal `-ss` (session).
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "-ss".to_string(),
            "my-session".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.session.as_deref(), Some("my-session"));
    assert!(cli.agent.is_none());

    // `-s` also works after a slash command (same bool-flag path as `-bg`).
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "/sessions".to_string(),
            "list".to_string(),
            "-s".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.agent.as_deref(), Some("sharp"));
    assert_eq!(cli.args, vec!["/sessions list".to_string()]);
}

#[test]
fn parse_cli_args_passes_slash_command_flags_through_verbatim() {
    // Unregistered command flags (e.g. `--prefix`) must survive argv parsing;
    // the command dispatcher owns the command grammar.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "/sessions".to_string(),
            "delete".to_string(),
            "prompt-eval-20260924".to_string(),
            "--prefix".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(
        cli.args,
        vec!["/sessions delete prompt-eval-20260924 --prefix".to_string()]
    );
}

#[test]
fn parse_cli_args_keeps_changes_short_flags_verbatim() {
    // `/changes -s` is `--stat`, not the global `--sharp` alias; `-h` is
    // command help, not global help. Neither may leak into `a`'s own options.
    for flag in ["-s", "-h"] {
        let cli = super::parse_cli_args(
            ["a".to_string(), "/changes".to_string(), flag.to_string()].into_iter(),
        );
        assert_eq!(cli.args, vec![format!("/changes {flag}")]);
        assert!(cli.agent.is_none());
        assert!(cli.reasoning_effort_override.is_none());
    }
    // `/diff` shares the changes grammar.
    let cli = super::parse_cli_args(
        ["a".to_string(), "/diff".to_string(), "-s".to_string()].into_iter(),
    );
    assert_eq!(cli.args, vec!["/diff -s".to_string()]);
    assert!(cli.agent.is_none());
    // A global flag before the command still applies.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "-s".to_string(),
            "/changes".to_string(),
            "-s".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.agent.as_deref(), Some("sharp"));
    assert_eq!(cli.args, vec!["/changes -s".to_string()]);
}

#[test]
fn parse_cli_args_keeps_audit_fast_flag_verbatim() {
    // `/audit -f` is fast mode, not the global `--files` alias (which would
    // additionally swallow the following token as a file path).
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "/audit".to_string(),
            "-f".to_string(),
            "review this".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.args, vec!["/audit -f review this".to_string()]);
    assert!(cli.files.is_empty());
}

#[test]
fn parse_cli_args_keeps_flags_before_slash_command() {
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "--model".to_string(),
            "gpt-test".to_string(),
            "/sessions".to_string(),
            "list".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.model.as_deref(), Some("gpt-test"));
    assert_eq!(cli.args, vec!["/sessions list".to_string()]);
}

#[test]
fn parse_cli_args_consumes_files_before_slash_command() {
    let file = make_temp_file("before-cmd");
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "-f".to_string(),
            file.to_string_lossy().to_string(),
            "/sessions".to_string(),
            "delete".to_string(),
            "x".to_string(),
            "--prefix".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.files, file.to_string_lossy());
    assert_eq!(cli.args, vec!["/sessions delete x --prefix".to_string()]);
    let _ = fs::remove_file(file);
}

#[test]
fn parse_cli_args_does_not_join_non_command_slash_tokens() {
    // A prompt that merely starts with a `/`-token (a path, not a command)
    // keeps its tokens separate.
    let cli = super::parse_cli_args(
        ["a".to_string(), "/etc/hosts".to_string(), "解释".to_string()].into_iter(),
    );
    assert_eq!(
        cli.args,
        vec!["/etc/hosts".to_string(), "解释".to_string()]
    );
}

#[test]
fn parse_cli_args_passes_colon_aliases_through() {
    let cli = super::parse_cli_args(
        ["a".to_string(), ":bg".to_string(), "now".to_string()].into_iter(),
    );
    assert_eq!(cli.args, vec![":bg now".to_string()]);
}

#[test]
fn parse_cli_args_consumes_registered_options_after_slash_command() {
    // Registered `a` options stay consumable after the command token while
    // the command text itself is preserved.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "/sessions".to_string(),
            "delete".to_string(),
            "x".to_string(),
            "--model".to_string(),
            "gpt-test".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.model.as_deref(), Some("gpt-test"));
    assert_eq!(cli.args, vec!["/sessions delete x".to_string()]);

    // Bool flag after the command.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "/sessions".to_string(),
            "list".to_string(),
            "-bg".to_string(),
        ]
        .into_iter(),
    );
    assert!(cli.background);
    assert_eq!(cli.args, vec!["/sessions list".to_string()]);

    // `--flag=value` form after the command.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "/sessions".to_string(),
            "list".to_string(),
            "--model=gpt-test".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.model.as_deref(), Some("gpt-test"));
    assert_eq!(cli.args, vec!["/sessions list".to_string()]);
}

#[test]
fn parse_cli_args_splits_known_and_unknown_flags_after_slash_command() {
    // `--prefix` is command grammar and stays verbatim; `--model` is an `a`
    // option and is consumed.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "/sessions".to_string(),
            "delete".to_string(),
            "prompt-eval-20260924".to_string(),
            "--prefix".to_string(),
            "--model".to_string(),
            "gpt-test".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.model.as_deref(), Some("gpt-test"));
    assert_eq!(
        cli.args,
        vec!["/sessions delete prompt-eval-20260924 --prefix".to_string()]
    );
}

#[test]
fn parse_cli_args_treats_option_value_as_value_not_command() {
    // `--model`'s value looks like a command: terminalw pairs the value with
    // the option, so `/sessions` is the model, not a command start.
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "--model".to_string(),
            "/sessions".to_string(),
            "list".to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(cli.model.as_deref(), Some("/sessions"));
    assert_eq!(cli.args, vec!["list".to_string()]);
}

#[test]
fn parse_cli_args_consumes_files_after_slash_command() {
    let file1 = make_temp_file("after-cmd-1");
    let file2 = make_temp_file("after-cmd-2");
    let cli = super::parse_cli_args(
        [
            "a".to_string(),
            "/sessions".to_string(),
            "list".to_string(),
            "-f".to_string(),
            file1.to_string_lossy().to_string(),
            file2.to_string_lossy().to_string(),
        ]
        .into_iter(),
    );
    assert_eq!(
        cli.files,
        format!("{},{}", file1.to_string_lossy(), file2.to_string_lossy())
    );
    assert_eq!(cli.args, vec!["/sessions list".to_string()]);
    let _ = fs::remove_file(file1);
    let _ = fs::remove_file(file2);
}

#[test]
fn model_selector_words_use_user_facing_selectors() {
    let selectors = super::model_selector_words();

    assert!(
        selectors.contains("-alibaba") || selectors.contains("-opencode"),
        "expected user-facing model selectors with a platform suffix, got: {selectors}"
    );
    for removed in [" use ", " select ", " switch "] {
        assert!(
            !format!(" {selectors} ").contains(removed),
            "model selector words should not include removed alias `{}`",
            removed.trim()
        );
    }
}
