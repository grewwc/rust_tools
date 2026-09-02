//! CLI argument parsing tests.

use super::super::*;

#[test]
fn cli_parse_args_basic() {
    let cli = cli::parse_cli_args(
        ["a", "hello", "-m", "minimax"]
            .into_iter()
            .map(|s| s.to_string()),
    );
    assert_eq!(cli.model.as_deref(), Some("minimax"));
    assert_eq!(cli.args, vec!["hello".to_string()]);
}

#[test]
fn cli_parse_note_search_interactive_mode() {
    let cli = cli::parse_cli_args(
        ["a", "-ns", "-i", "帮我找之前记过的 trait object"]
            .into_iter()
            .map(|s| s.to_string()),
    );
    assert!(cli.note_search);
    assert!(cli.interactive);
    assert_eq!(cli.args, vec!["帮我找之前记过的 trait object".to_string()]);
}
