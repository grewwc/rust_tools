use super::*;

#[test]
fn cancelled_stream_result_finalizes_active_thinking_fold() {
    // On cancel with an active fold window, it must be finalized (finalize→reset), preventing a partial
    // thinking remnant with a new header stacked under it on the next retry (cross-turn root cause of duplicated headers + large blank areas).
    let mut state = StreamProcessingState::new();
    {
        let fold = &mut state.render.thinking_fold;
        fold.active = true;
        fold.max_visible_lines = 2;
        fold.total_lines = 1;
        fold.recent_lines.push_back("partial".to_string());
        fold.window_rows = 2;
    }
    state.content.assistant_text = "partial body".to_string();
    state.content.reasoning_text = "partial reasoning".to_string();

    let result = cancelled_stream_result(&mut state);

    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert!(result.skip_response_drain);
    assert_eq!(result.assistant_text, "partial body");
    assert_eq!(result.reasoning_text, "partial reasoning");
    // After finalize the fold state is reset: no longer active, window rows zeroed, no orphan window left behind.
    assert!(!state.render.thinking_fold.active);
    assert_eq!(state.render.thinking_fold.window_rows, 0);
    assert!(state.render.thinking_fold.recent_lines.is_empty());
}

#[test]
fn cancelled_stream_result_flushes_unclassified_content_reasoner_tail() {
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let (reasoning, content) = state.content.content_think_demuxer.push("unfinished reply");
    assert!(reasoning.is_empty());
    assert!(content.is_empty());

    let result = cancelled_stream_result(&mut state);

    assert_eq!(result.assistant_text, "unfinished reply");
    assert!(result.reasoning_text.is_empty());
}

#[test]
fn cancelled_stream_result_finalizes_active_subagent_fold() {
    let mut state = StreamProcessingState::new();
    {
        let fold = &mut state.render.subagent_fold;
        fold.active = true;
        fold.max_visible_lines = 2;
        fold.total_lines = 1;
        fold.recent_lines.push_back("partial answer".to_string());
        fold.window_rows = 2;
    }

    let result = cancelled_stream_result(&mut state);

    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert!(!state.render.subagent_fold.active);
    assert_eq!(state.render.subagent_fold.window_rows, 0);
    assert!(state.render.subagent_fold.recent_lines.is_empty());
}
