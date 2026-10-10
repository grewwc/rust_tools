use super::*;

#[test]
fn app_registered_stream_filters_reach_runtime_state() {
    let mut app = crate::ai::middleware::test_util::test_app();
    app.hooks.register_stream_filter(PrefixStreamFilter);

    let state = initial_stream_processing_state(&app);

    assert_eq!(
        state.filters.apply("chunk".to_string()),
        Some("filtered:chunk".to_string())
    );
}

#[test]
fn forked_app_inherits_stream_filters_for_initial_state() {
    // Regression: forked turns (subagents / background) run their own
    // stream_response on an App fork; the fork must inherit the parent's
    // registered stream filter chain instead of silently using an empty one.
    let mut app = crate::ai::middleware::test_util::test_app();
    app.hooks.register_stream_filter(PrefixStreamFilter);

    let fork = app.fork_for_subagent();
    let state = initial_stream_processing_state(&fork);

    assert_eq!(state.filters.names(), vec!["prefix"]);
}

#[test]
fn prompt_cache_metrics_none_without_hit() {
    assert_eq!(format_prompt_cache_metrics(1000, 0), None);
    assert_eq!(format_prompt_cache_metrics(0, 0), None);
}

#[test]
fn prompt_cache_metrics_reports_hit_rate() {
    let line = format_prompt_cache_metrics(1000, 750).unwrap();
    assert_eq!(line, "↳ cache · 750/1.0k tokens · 75% hit");

    let large = format_prompt_cache_metrics(59_798, 59_648).unwrap();
    assert_eq!(large, "↳ cache · 59.6k/59.8k tokens · 100% hit");
}

#[test]
fn token_throughput_metrics_split_reasoning_from_output() {
    let line = format_token_throughput_metrics(
        800,
        400,
        Some(Duration::from_secs(2)),
        Some(Duration::from_secs(1)),
    )
    .unwrap();

    assert_eq!(
        line,
        "↳ speed · reasoning 800 tok @ 400 tok/s · output 400 tok @ 400 tok/s"
    );
}

#[test]
fn token_throughput_metrics_omits_unmeasurable_streams() {
    assert_eq!(format_token_throughput_metrics(800, 0, None, None), None);
    assert_eq!(
        format_token_throughput_metrics(0, 1_200, None, Some(Duration::from_secs(2))),
        Some("↳ speed · output 1.2k tok @ 600 tok/s".to_string())
    );
}

#[test]
fn fold_header_rate_tracks_live_reasoning_throughput() {
    let now = Instant::now();
    let app = crate::ai::middleware::test_util::test_app();
    let mut content = initial_stream_processing_state(&app).content;
    // No reasoning tokens yet: the in-progress header stays stable.
    assert_eq!(fold_header_rate(&content, now), None);
    content.reasoning_started_at = Some(now - Duration::from_secs(2));
    content.live_reasoning_tokens = 8;
    assert_eq!(
        fold_header_rate(&content, now).as_deref(),
        Some("~8 tok @ 4.00 tok/s")
    );
}

#[test]
fn token_rate_under_minimum_window_is_unmeasurable() {
    assert_eq!(format_token_rate(10_000, Some(Duration::from_millis(200))), "—");
    assert_eq!(format_token_rate(10_000, Some(Duration::from_secs(2))), "5.0k");
}

#[test]
fn live_rate_text_gates_pre_output_and_early_windows() {
    assert_eq!(format_live_rate_text(0, Some(Duration::from_secs(2))), None);
    assert_eq!(format_live_rate_text(1_000, None), None);
    assert_eq!(
        format_live_rate_text(1_000, Some(Duration::from_millis(100))),
        None
    );
}

#[test]
fn live_rate_text_formats_compact_count_and_rate() {
    assert_eq!(
        format_live_rate_text(1_234, Some(Duration::from_secs(2))).as_deref(),
        Some("~1.2k tok @ 617 tok/s")
    );
    assert_eq!(
        format_live_rate_text(57, Some(Duration::from_secs(3))).as_deref(),
        Some("~57 tok @ 19.0 tok/s")
    );
}

#[test]
fn deferred_rate_refresh_is_gated_and_throttled() {
    let mut state = StreamProcessingState::new();

    // No output window yet: hint stays untouched.
    refresh_deferred_body_rate_hint(&mut state).unwrap();
    assert!(state.render.waiting_hint_line.is_empty());

    // Output flowing (started 2 s ago, tokens present) but the throttle stamp is
    // fresh: the row must not be rewritten.
    state.render.waiting_hint_active = true;
    state.render.waiting_hint_buffering = true;
    state.render.waiting_hint_line = "  ⠋ generating…".to_string();
    state.content.output_started_at = Some(Instant::now() - Duration::from_secs(2));
    state.content.live_output_tokens = 1_000;
    state.render.waiting_hint_rate_refreshed_at = Some(Instant::now());
    refresh_deferred_body_rate_hint(&mut state).unwrap();
    assert_eq!(state.render.waiting_hint_line, "  ⠋ generating…");

    // A tool-call hint row must never be clobbered by the live-rate rewrite.
    state.render.waiting_hint_tool_call = true;
    state.render.waiting_hint_rate_refreshed_at = None;
    refresh_deferred_body_rate_hint(&mut state).unwrap();
    assert_eq!(state.render.waiting_hint_line, "  ⠋ generating…");
}

/// Tool-call arguments are never rendered, so the `receiving `X` arguments…` row is the only
/// place the arrival rate of a large payload (apply_patch / execute_command / task / …) shows.
#[test]
fn tool_call_rate_hint_reports_argument_throughput() {
    // Hermetic width: sibling fold tests narrow the process-global COLUMNS to 60
    // while holding ENV_LOCK. Without the lock this assertion can observe that
    // width mid-run and truncate the expected row, so pin a wide terminal here.
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _columns = SavedColumns(std::env::var_os("COLUMNS"));
    crate::ai::stream::side_note_input::set_scripted_true_width(None);
    unsafe {
        std::env::set_var("COLUMNS", "100");
    }
    let mut state = StreamProcessingState::new();
    state.render.waiting_hint_active = true;
    state.render.waiting_hint_tool_call = true;
    state.render.waiting_hint_line = tool_call_hint_label("apply_patch", None);
    // The counter is fed directly so the assertion does not depend on real stream timing,
    // while the window itself is deterministic.
    state.content.count_tool_arg_delta(600);
    state.content.count_tool_arg_delta(400);
    state.content.tool_args_started_at = Some(Instant::now() - Duration::from_secs(2));

    refresh_tool_call_rate_hint(&mut state, "apply_patch").unwrap();

    assert_eq!(
        state.render.waiting_hint_line,
        "  ⠋ receiving `apply_patch` arguments… · ~1.0k tok @ 500 tok/s"
    );
    assert!(state.render.waiting_hint_rate_refreshed_at.is_some());
}

#[test]
fn tool_call_rate_hint_is_gated_and_throttled() {
    let tool_row = || tool_call_hint_label("execute_command", None);
    let mut state = StreamProcessingState::new();

    // The "generating…" row is not the tool-call hint and must never be rewritten here.
    state.render.waiting_hint_active = true;
    state.render.waiting_hint_line = "  ⠋ generating…".to_string();
    state.content.tool_args_started_at = Some(Instant::now() - Duration::from_secs(2));
    state.content.live_tool_arg_tokens = 1_000;
    refresh_tool_call_rate_hint(&mut state, "execute_command").unwrap();
    assert_eq!(state.render.waiting_hint_line, "  ⠋ generating…");

    // Tool hint drawn and arguments flowing, but the throttle stamp is fresh: no repaint.
    state.render.waiting_hint_tool_call = true;
    state.render.waiting_hint_line = tool_row();
    state.render.waiting_hint_rate_refreshed_at = Some(Instant::now());
    refresh_tool_call_rate_hint(&mut state, "execute_command").unwrap();
    assert_eq!(state.render.waiting_hint_line, tool_row());

    // Throttle expired, but the window is still shorter than MIN_RATE_WINDOW.
    state.render.waiting_hint_rate_refreshed_at = None;
    state.content.tool_args_started_at = Some(Instant::now() - Duration::from_millis(100));
    refresh_tool_call_rate_hint(&mut state, "execute_command").unwrap();
    assert_eq!(state.render.waiting_hint_line, tool_row());

    // No argument received yet: there is nothing to report.
    state.content.tool_args_started_at = None;
    state.content.live_tool_arg_tokens = 0;
    refresh_tool_call_rate_hint(&mut state, "execute_command").unwrap();
    assert_eq!(state.render.waiting_hint_line, tool_row());
}

#[test]
fn tool_arg_metrics_reset_when_the_hint_switches_calls() {
    let mut state = StreamProcessingState::new();
    state.content.count_tool_arg_delta(120);
    assert_eq!(state.content.live_tool_arg_tokens, 120);
    assert!(state.content.tool_args_started_at.is_some());

    state.content.reset_tool_args_metrics();
    assert_eq!(state.content.live_tool_arg_tokens, 0);
    assert!(state.content.tool_args_started_at.is_none());
}

#[test]
fn live_token_estimate_is_zero_only_for_empty_deltas() {
    assert_eq!(estimate_stream_tokens(""), 0);
    assert_eq!(estimate_stream_tokens("a"), 1);
    assert_eq!(estimate_stream_tokens("中文"), 2);
}

#[test]
fn live_token_estimate_splits_cjk_and_ascii() {
    // CJK chars count ~1 token each instead of the old byte/4 underestimate.
    assert_eq!(estimate_stream_tokens("你好世界"), 4);
    assert_eq!(estimate_stream_tokens("Hello 世界"), 4); // 6 ascii -> 2 + 2 CJK
    assert_eq!(estimate_stream_tokens("Hello world"), 3); // 11 ascii -> ceil(11/4)
}

#[test]
fn format_compact_rate_escalates_units_across_rounding_boundaries() {
    // Values that round up across a unit threshold render in the next unit
    // instead of showing a misleading "1000" / "1000.0k".
    assert_eq!(format_compact_rate(999.4), "999");
    assert_eq!(format_compact_rate(999.6), "1.0k");
    assert_eq!(format_compact_rate(999_949.0), "999.9k");
    assert_eq!(format_compact_rate(999_950.0), "1.0m");
    // Sub-unit precision is unchanged.
    assert_eq!(format_compact_rate(9.94), "9.94");
    assert_eq!(format_compact_rate(99.4), "99.4");
}
