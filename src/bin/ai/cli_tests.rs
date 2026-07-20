    use std::fs;
    use std::path::PathBuf;

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
        // 长格式 --background
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

        // 短别名 -bg
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

        // 不带 -bg 时默认 false
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

        // 不带 --stop 时默认为 None
        let cli = super::parse_cli_args(["a".to_string()].into_iter());
        assert!(cli.stop_session.is_none());
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
