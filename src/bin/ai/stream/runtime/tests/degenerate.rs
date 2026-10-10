use super::*;

#[test]
fn degenerate_reasoning_repetition_requires_three_long_contentful_copies() {
    let phrase = "需要先确认当前上下文是否仍然有效，然后再继续执行。";
    assert!(!has_degenerate_repetition(&phrase.repeat(2)));
    assert!(has_degenerate_repetition(&phrase.repeat(3)));
    assert!(!has_degenerate_repetition(&"----------------".repeat(3)));
}

#[test]
fn degenerate_reasoning_repetition_detects_suffix_after_normal_progress() {
    let phrase = "I need to inspect the existing implementation before changing it. ";
    let reasoning = format!(
        "First I will locate the relevant module. {}",
        phrase.repeat(3)
    );
    assert!(has_degenerate_repetition(&reasoning));
}

#[test]
fn degenerate_repetition_catches_visible_content_runaway() {
    // Reproduces an incident: the model repeated the same phrase verbatim in its **visible output**
    // until the budget was full, producing a giant junk message on disk and triggering a provider 400
    // on the next turn. The degenerate guard must also apply to visible assistant text (previously it only covered reasoning_content).
    let phrase = "我再重新读一遍修复区域，以确保我掌握的是当前状态。";
    assert!(has_degenerate_repetition(&phrase.repeat(3)));
    // Single-character repetition (e.g. the 80,000 repetitions of one character in the incident) must also match.
    assert!(has_degenerate_repetition(&"再".repeat(64)));
}

#[test]
fn degenerate_repetition_ignores_fold_placeholder_quotes() {
    // Terminal fold placeholders can legitimately appear repeated in a user-provided
    // screenshot transcript and then be quoted in model reasoning.
    let repeated_placeholder = "    … 13 earlier lines\n".repeat(3);
    let quoted_terminal_output = format!(
        "Looking at the displayed output:\n```text\n{}",
        repeated_placeholder
    );

    assert!(!has_degenerate_repetition(&quoted_terminal_output));
    assert!(!has_degenerate_repetition(
        &"    ... 13 earlier lines\n".repeat(3)
    ));
    assert!(!has_degenerate_repetition(&"    … more\n".repeat(3)));
}

#[test]
fn degenerate_repetition_strip_len_returns_tail_and_keeps_prefix() {
    let phrase = "需要先确认当前上下文是否仍然有效，然后再继续执行。";
    let prefix = "正常的前置说明文本。";
    let text = format!("{prefix}{}", phrase.repeat(3));
    // The detector must see through the prefix and return exactly the repeated tail length.
    let strip = degenerate_repetition_strip_len(&text).expect("degenerate tail must be detected");
    assert_eq!(strip, phrase.chars().count() * 3);
    // Stripping that tail must leave the clean prefix untouched.
    let keep = text.chars().count() - strip;
    let cleaned: String = text.chars().take(keep).collect();
    assert_eq!(cleaned, prefix);
    assert!(!has_degenerate_repetition(&cleaned));
    // Non-degenerate inputs have nothing to strip.
    assert_eq!(degenerate_repetition_strip_len(&phrase.repeat(2)), None);
    // Separator-only tails are not contentful, so nothing to strip.
    assert_eq!(degenerate_repetition_strip_len(&"----------------".repeat(3)), None);
}

#[test]
fn degenerate_repetition_strip_len_cleans_visible_content_runaway() {
    // The visible-content incident shape: real conclusion first, then the looped junk tail.
    let phrase = "我再重新读一遍修复区域，以确保我掌握的是当前状态。";
    let text = format!("先给出结论。{}", phrase.repeat(3));
    let strip = degenerate_repetition_strip_len(&text).expect("junk tail must be detected");
    let keep = text.chars().count() - strip;
    let cleaned: String = text.chars().take(keep).collect();
    assert_eq!(cleaned, "先给出结论。");
    assert!(!has_degenerate_repetition(&cleaned));
}

#[test]
fn thinking_fold_defaults_to_configured_lines_for_tty() {
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(true, None),
        DEFAULT_THINKING_MAX_VISIBLE_LINES
    );
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(true, Some("12")),
        12
    );
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(true, Some("0")),
        usize::MAX
    );
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(true, Some("oops")),
        DEFAULT_THINKING_MAX_VISIBLE_LINES
    );
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(false, Some("12")),
        usize::MAX
    );
}

#[test]
fn stream_text_event_to_content_ignores_thinking_events() {
    let mut markers = StreamMarkers::new();
    markers.enable_subagent_preview("build");

    assert_eq!(
        stream_text_event_to_content(
            &StreamTextEvent::OpenThinking,
            &markers,
            StreamEventMergeMode::Append,
            "",
        ),
        None
    );
    assert_eq!(
        stream_text_event_to_content(
            &StreamTextEvent::AppendThinking("step one".to_string()),
            &markers,
            StreamEventMergeMode::Append,
            "",
        ),
        None
    );
    assert_eq!(
        stream_text_event_to_content(
            &StreamTextEvent::AppendContent("final answer".to_string()),
            &markers,
            StreamEventMergeMode::Append,
            "",
        ),
        Some("final answer".to_string())
    );
    assert_eq!(
        stream_text_event_to_content(
            &StreamTextEvent::CloseThinking,
            &markers,
            StreamEventMergeMode::Append,
            "",
        ),
        None
    );
}
