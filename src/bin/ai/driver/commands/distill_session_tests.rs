use super::*;

#[test]
fn distill_session_requires_exact_command_name() {
    assert_eq!(
        command_args(" /distill-session\tsession-1 "),
        Some("session-1")
    );
    assert_eq!(command_args(":distill-session"), Some(""));
    for input in [
        "distill-session id",
        "/distill-session-extra id",
        "/distill-session/id",
        "hello",
    ] {
        assert_eq!(command_args(input), None, "{input}");
    }
    assert!(super::super::is_local_command_start("/distill-session"));
    assert!(super::super::is_local_command_start(":distill-session id"));
}

#[test]
fn distill_session_defaults_to_saving_and_preserves_requested_options() {
    assert_eq!(
        parse_options("session-1", 20).unwrap(),
        Some(Options {
            source: Some("session-1".into()),
            limit: 20,
            dry_run: false,
        })
    );
    assert_eq!(
        parse_options("--dry-run 'archives/full session.zip' --limit 7", 20).unwrap(),
        Some(Options {
            source: Some("archives/full session.zip".into()),
            limit: 7,
            dry_run: true,
        })
    );
    assert_eq!(
        parse_options("--limit=100 session-1", 20)
            .unwrap()
            .unwrap()
            .limit,
        100
    );
    assert_eq!(
        parse_options("session-1 --limit 1", 20)
            .unwrap()
            .unwrap()
            .limit,
        1
    );
    assert_eq!(
        parse_options("-- -archive.zip", 20)
            .unwrap()
            .unwrap()
            .source,
        Some("-archive.zip".into())
    );
}

#[test]
fn distill_session_omitted_source_selects_current_session() {
    for (input, limit, dry_run) in [
        ("", 20, false),
        (" \t ", 20, false),
        ("--dry-run", 20, true),
        ("--limit 7", 7, false),
        ("--dry-run --limit=9", 9, true),
        ("--", 20, false),
    ] {
        assert_eq!(
            parse_options(input, 20).unwrap(),
            Some(Options {
                source: None,
                limit,
                dry_run,
            }),
            "{input}"
        );
    }
}

#[test]
fn distill_session_paths_are_not_shell_expanded() {
    assert_eq!(
        words(r#""a b.zip" c\ d.zip 'e\f.zip' "g\h.zip""#).unwrap(),
        vec!["a b.zip", "c d.zip", "e\\f.zip", "g\\h.zip"]
    );
    assert_eq!(
        words(r#""a\"b.zip" '$HOME/$(touch bad).zip'"#).unwrap(),
        vec!["a\"b.zip", "$HOME/$(touch bad).zip"]
    );
}

#[test]
fn distill_session_rejects_ambiguous_or_invalid_arguments() {
    for input in [
        "''",
        "a b",
        "--unknown",
        "--limit",
        "--limit 0",
        "--limit 101",
        "--limit=",
        "id --unknown",
        "id --dry-run=false",
        "id --limit",
        "id --limit 0",
        "id --limit 101",
        "id --limit -1",
        "id --limit no",
        "id --limit=",
        "id --limit 1 --limit=2",
        "\"unclosed",
        "a\\",
    ] {
        assert!(parse_options(input, 20).is_err(), "{input}");
    }
}

#[tokio::test]
async fn distill_session_invalid_help_and_unmatched_inputs_never_execute() {
    for input in [
        "/distill-session ''",
        "/distill-session --limit",
        "/distill-session --limit 0",
        "/distill-session --unknown",
        "/distill-session id --limit 0",
        "/distill-session 'unclosed",
    ] {
        let result = dispatch(input, 20, |_| async {
            panic!("Invalid input must not execute")
        })
        .await;
        assert!(result.unwrap().is_err(), "{input}");
    }
    for input in ["/distill-session --help", ":distill-session -h"] {
        assert_eq!(
            dispatch(input, 20, |_| async { panic!("Help must not execute") }).await,
            Some(Ok(USAGE.into()))
        );
    }
    assert_eq!(
        dispatch("/distill-session-other id", 20, |_| async {
            panic!("Unmatched input must not execute")
        })
        .await,
        None
    );
}

#[tokio::test]
async fn distill_session_dispatch_passes_save_and_preview_to_executor_once() {
    for (input, source, limit, dry_run) in [
        ("/distill-session", None, 20, false),
        (":distill-session", None, 20, false),
        (" /distill-session \t", None, 20, false),
        ("/distill-session --dry-run", None, 20, true),
        ("/distill-session --limit 7", None, 7, false),
        (":distill-session --limit 7 --dry-run", None, 7, true),
        ("/distill-session id --limit 7", Some("id"), 7, false),
        (
            ":distill-session id --limit 7 --dry-run",
            Some("id"),
            7,
            true,
        ),
        (
            "/distill-session --dry-run 'archive name.zip'",
            Some("archive name.zip"),
            20,
            true,
        ),
    ] {
        let mut calls = 0;
        let calls_ref = &mut calls;
        let result = dispatch(input, 20, |options| async move {
            *calls_ref += 1;
            assert_eq!(
                options,
                Options {
                    source: source.map(str::to_owned),
                    limit,
                    dry_run
                }
            );
            Ok("complete".into())
        })
        .await;
        assert_eq!(result, Some(Ok("complete".into())));
        assert_eq!(calls, 1, "{input}");
    }
}

#[tokio::test]
async fn distill_session_execution_failure_remains_a_handled_command() {
    for input in ["/distill-session", "/distill-session id"] {
        assert_eq!(
            dispatch(input, 20, |_| async { Err("failed before commit".into()) }).await,
            Some(Err("failed before commit".into()))
        );
    }
}
