//! Prompt / multiline-editor state tests.

use super::super::prompt::MultilineHistoryState;

#[test]
fn multiline_history_navigation_restores_draft() {
    let mut history =
        MultilineHistoryState::new(vec!["first".to_string(), "second\nline".to_string()]);

    assert_eq!(history.previous("draft"), Some("second\nline".to_string()));
    assert_eq!(history.previous("ignored"), Some("first".to_string()));
    assert_eq!(history.previous("ignored"), None);
    assert_eq!(history.next(), Some("second\nline".to_string()));
    assert_eq!(history.next(), Some("draft".to_string()));
    assert_eq!(history.next(), None);
}
