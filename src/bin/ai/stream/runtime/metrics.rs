use super::*;
/// Maximum number of decode errors before giving up and returning partial content
pub(super) const MAX_DECODE_ERRORS: usize = 3;
/// Delay in milliseconds between retry attempts on transient errors
pub(super) const DECODE_ERROR_RETRY_DELAY_MS: u64 = 100;
/// Grace window after an OpenAI-compatible `finish_reason` chunk. Some backends
/// do not emit `[DONE]` or close the HTTP body, while others can still send a
/// final snapshot immediately after the finish chunk.
pub(super) const FINISH_REASON_GRACE_MS: u64 = 750;

/// Silence allowance on a connection with no provider-declared work in progress.
///
/// This stays a *connection* bound: a stream that already delivered something is worth preserving through a quiet
/// stretch, but not indefinitely — after this much silence it is treated as stalled and the attempt replays
/// through the retry ladder. Healthy streams on this wire keep refreshing it (visible text, reasoning summaries,
/// tool-argument deltas, item open/close events), so a provider that really stopped sending is discovered
/// quickly. Silences that are *expected* because the provider declared work are covered by
/// `STREAM_DECLARED_ITEM_TIMEOUT_SECS` instead of by widening this bound.
///
/// A model whose gateway hides long stretches of work behind silence without opening an output item may widen
/// this default per model (`stream_silence_timeout_secs` in the model registry).
pub(super) const STREAM_IDLE_TIMEOUT_SECS: u64 = 45;
/// Silence allowance while the provider holds an open output item (`response.output_item.added` without its
/// matching `.done`).
///
/// An open item is the provider's own declaration that work is under way, and both shapes this wire produces can
/// then stay completely quiet: hidden thinking streams no content at all (measured 111s of total wire silence
/// inside one reasoning item, and ~285s before the first output item, on muse-spark-1.3 at xhigh effort), and a
/// large tool-argument payload is generated server-side before it is delivered (a 20KB `write_file` argument
/// arrives as a single late delta).
///
/// Only that declared state gets the allowance: with no item open, silence is still cut by the connection bound
/// above, so a dead connection is normally discovered in seconds rather than minutes. Inside an open item a dead
/// connection is indistinguishable from a slow provider — no bytes arrive either way — so this is the accepted
/// cost of not discarding healthy generations; `MAX_TOOL_ARG_BYTES` and the tool-argument stall bound below stay
/// as the runaway guards for tool calls.
pub(super) const STREAM_DECLARED_ITEM_TIMEOUT_SECS: u64 = 300;
/// Tool-call argument stall timeout: an absolute bound on how long one tool call may stay open, even if the provider
/// keeps trickling argument deltas (the silence allowance only catches total silence; a server sending one small
/// delta every few minutes refreshes the meaningful-progress timer forever, leaving the terminal stuck on
/// "receiving `tool` arguments…" — observed incident: an apply_patch call whose arguments never completed for 13+
/// minutes). Generating the arguments of a large artifact is legitimate work that can take minutes, so this is a
/// backstop against a stream that never finishes, not a normal-path timeout; `MAX_TOOL_ARG_BYTES` remains the primary
/// runaway guard. On expiry the open call is dropped and the attempt replays through the truncation path.
pub(super) const STREAM_TOOL_ARGS_STALL_TIMEOUT_SECS: u64 = 180;
/// First-chunk timeout: the request was sent but the server never sends the first byte (queued, stuck gateway, ...).
/// A stream that produced nothing yet gets this shorter window: unlike a stall after work has been delivered, there
/// is nothing to preserve by waiting longer, so the retry may start earlier.
pub(super) const STREAM_FIRST_CHUNK_TIMEOUT_SECS: u64 = 90;
/// Default visible-window height for `thinking` in the terminal. Only affects display, not reasoning accumulation.
/// Streaming shows the most recent N lines (default 2); when thinking ends, `finalize_fold` forces a pure-summary
/// fold (redraws with a 0-line window) so conclusions/questions restated at the tail of thinking are not shown twice
/// alongside the final answer in the terminal.
pub(super) const DEFAULT_THINKING_MAX_VISIBLE_LINES: usize = 2;
/// Physical rows the live fold window reserves outside its body budget: the anchored header (1), the fold-summary
/// marker (1), and one row of headroom so the cursor never sits on the very bottom row. Bounding the body to
/// `viewport_rows - this` keeps the whole window (header + body) inside the visible viewport, so the relative-cursor
/// erase (`\x1b[nA`) in `erase_fold_body` always reaches the top body row instead of being clamped at the viewport
/// top once the window scrolls into scrollback — the root cause of stacked `… more` / `… N earlier lines` markers.
pub(super) const FOLD_VIEWPORT_RESERVED_ROWS: usize = 3;
/// Indentation for folded thinking/subagent bodies: header/footer use 2 spaces, body is indented one more level.
pub(super) const THINKING_FOLD_BODY_INDENT: &str = "    ";
pub(super) const THINKING_FOLD_BODY_INDENT_WIDTH: usize = 4;
/// Terminals usually wrap at the right edge with delayed-wrap; folded redraws always leave two extra columns so that
/// a missing terminal flag or a one-column width/char-width drift cannot trigger an implicit wrap that was not counted
/// in cursor-up, leaving residue from the old window under the `✓`.
pub(super) const FOLD_REWRITE_RIGHT_MARGIN_COLS: usize = 2;
/// Shortest repeated fragment and decision count for reasoning-stream degeneration. Only reasoning is checked, so
/// legitimately repeated body the model was asked to produce (tables, code, test data) is not misjudged as degeneration.
pub(super) const MIN_REASONING_REPEAT_CHARS: usize = 16;
pub(super) const MAX_REASONING_REPEAT_CHARS: usize = 512;
pub(super) const REASONING_REPEAT_COUNT: usize = 3;
pub(super) const DEGENERATE_REPETITION_FINISH_REASON: &str = "degenerate_repetition";
/// Cap on accumulated streaming tool-call arguments (total across all tool calls in one turn).
/// Once the model opens a tool call it should close quickly; if arguments keep growing until the cap is hit (e.g.
/// an endless loop concatenating the same body in apply_patch), the output has degenerated. Existing degeneration
/// detection only covers reasoning/assistant text, not tool arguments; this total cap is a backstop against infinite waits and memory growth.
pub(super) const MAX_TOOL_ARG_BYTES: usize = 1 << 20; // 1 MiB

/// When `ai.prompt_cache.show_metrics` (default on) is set and this request hit the prompt
/// cache, print one line of cache-hit metrics. OpenAI / DashScope etc. cache server-side;
/// this just visualizes the `cached_tokens` they already reported.
pub(super) fn maybe_print_prompt_cache_metrics(usage: &crate::ai::request::StreamUsage) {
    if !runtime_ctx::terminal_output_enabled() {
        return;
    }
    let show = crate::commonw::configw::get_all_config()
        .get(
            crate::ai::config_schema::AiConfig::PROMPT_CACHE_SHOW_METRICS,
            "true",
        )
        .trim()
        .eq_ignore_ascii_case("true");
    if !show {
        return;
    }
    let cached = usage
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);
    if let Some(line) = format_prompt_cache_metrics(usage.prompt_tokens, cached) {
        println!("  {}{line}{RESET}", theme::current().accent_muted);
    }
}

/// Print a heuristic generation-throughput line after a stream finishes. The timing split is an
/// estimate, not server-reported: the reasoning window ends at the first visible output token, so
/// interleaved thinking after output starts (or idle gaps between phases) skews the reasoning
/// rate. `completion_tokens` includes reasoning tokens and tool-call argument tokens; the output
/// slice is the non-reasoning remainder after subtracting
/// `completion_tokens_details.reasoning_tokens`, so it can exceed what was rendered.
pub(super) fn maybe_print_token_throughput_metrics(
    usage: &crate::ai::request::StreamUsage,
    content: &StreamContentState,
    finished_at: Instant,
) {
    if !runtime_ctx::terminal_output_enabled() {
        return;
    }

    let reasoning_tokens = usage
        .completion_tokens_details
        .as_ref()
        .map(|details| details.reasoning_tokens)
        .unwrap_or(0)
        .min(usage.completion_tokens);
    let output_tokens = usage.completion_tokens.saturating_sub(reasoning_tokens);
    let reasoning_end = content.output_started_at.unwrap_or(finished_at);
    let reasoning_elapsed = content
        .reasoning_started_at
        .map(|started| reasoning_end.saturating_duration_since(started));
    let output_elapsed = content
        .output_started_at
        .map(|started| finished_at.saturating_duration_since(started));

    let Some(line) = format_token_throughput_metrics(
        reasoning_tokens,
        output_tokens,
        reasoning_elapsed,
        output_elapsed,
    ) else {
        return;
    };
    println!("  {}{line}{RESET}", theme::current().accent_muted);
}

/// Live reasoning-rate text for the in-progress thinking-fold header. It is
/// recomputed on every fold redraw (each thinking chunk) from the approximate
/// `~`-prefixed token estimate; the exact server-reported metrics are printed
/// separately at stream end. Returns `None` until the first reasoning token
/// arrives so the header stays stable during the pre-token window.
pub(super) fn fold_header_rate(content: &StreamContentState, now: Instant) -> Option<String> {
    if content.live_reasoning_tokens == 0 {
        return None;
    }
    let elapsed = content.reasoning_started_at.map(|started| {
        content
            .output_started_at
            .unwrap_or(now)
            .saturating_duration_since(started)
    });
    Some(format!(
        "~{} tok @ {} tok/s",
        format_compact_token_count(content.live_reasoning_tokens),
        format_token_rate(content.live_reasoning_tokens, elapsed),
    ))
}

/// Approximate token count for live throughput display (heuristic, prefixed with `~`):
/// ASCII text averages ~4 chars/token, CJK characters ~1 token each (3 UTF-8 bytes each).
/// A pure byte/4 estimate under-counts CJK by ~25%, so split by character class instead.
pub(super) fn estimate_stream_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let mut ascii = 0u64;
    let mut non_ascii = 0u64;
    for ch in text.chars() {
        if ch.is_ascii() {
            ascii += 1;
        } else {
            non_ascii += 1;
        }
    }
    non_ascii + ascii.div_ceil(4)
}

/// Pure formatter for the split reasoning/output throughput line.
/// A missing timing sample is rendered as an em dash instead of inventing a rate.
pub(super) fn format_token_throughput_metrics(
    reasoning_tokens: u64,
    output_tokens: u64,
    reasoning_elapsed: Option<Duration>,
    output_elapsed: Option<Duration>,
) -> Option<String> {
    if reasoning_tokens == 0 && output_tokens == 0 {
        return None;
    }
    if reasoning_elapsed.is_none() && output_elapsed.is_none() {
        return None;
    }

    let mut segments = Vec::with_capacity(2);
    if reasoning_tokens > 0 {
        segments.push(format!(
            "reasoning {} tok @ {} tok/s",
            format_compact_token_count(reasoning_tokens),
            format_token_rate(reasoning_tokens, reasoning_elapsed)
        ));
    }
    if output_tokens > 0 {
        segments.push(format!(
            "output {} tok @ {} tok/s",
            format_compact_token_count(output_tokens),
            format_token_rate(output_tokens, output_elapsed)
        ));
    }
    Some(format!("↳ speed · {}", segments.join(" · ")))
}

/// Minimum elapsed window before an instantaneous rate is reported; shorter windows
/// (e.g. the first chunk right after output starts) would print misleadingly large rates.
pub(super) const MIN_RATE_WINDOW: Duration = Duration::from_millis(500);

pub(super) fn format_token_rate(tokens: u64, elapsed: Option<Duration>) -> String {
    let Some(elapsed) = elapsed else {
        return "—".to_string();
    };
    if elapsed < MIN_RATE_WINDOW {
        return "—".to_string();
    }
    let seconds = elapsed.as_secs_f64();
    if seconds <= 0.0 {
        return "—".to_string();
    }
    format_compact_rate(tokens as f64 / seconds)
}

pub(super) fn format_compact_rate(rate: f64) -> String {
    if rate < 999.5 {
        // A value >= 999.5 would round to "1000" below; render it in the next unit instead.
        if rate >= 100.0 {
            format!("{rate:.0}")
        } else if rate >= 10.0 {
            format!("{rate:.1}")
        } else {
            format!("{rate:.2}")
        }
    } else if rate < 999_950.0 {
        // k-branch values >= 999_950 round to "1000.0k"; escalate to the m unit instead.
        format!("{:.1}k", rate / 1_000.0)
    } else {
        format!("{:.1}m", rate / 1_000_000.0)
    }
}

/// Pure function: build a readable cache-hit line from prompt_tokens / cached_tokens.
/// Returns Some only when there really was a hit (cached > 0), to avoid pointless noise.
pub(super) fn format_prompt_cache_metrics(prompt_tokens: u64, cached_tokens: u64) -> Option<String> {
    if cached_tokens == 0 || prompt_tokens == 0 {
        return None;
    }
    let pct = (cached_tokens as f64 / prompt_tokens as f64 * 100.0).min(100.0);
    Some(format!(
        "↳ cache · {}/{} tokens · {pct:.0}% hit",
        format_compact_token_count(cached_tokens),
        format_compact_token_count(prompt_tokens)
    ))
}

pub(super) fn format_compact_token_count(tokens: u64) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 1_000_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        format!("{:.1}m", tokens as f64 / 1_000_000.0)
    }
}
