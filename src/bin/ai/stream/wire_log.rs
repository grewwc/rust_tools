//! Opt-in raw-wire recorder for streamed model responses.
//!
//! Enabled by the `AIOS_STREAM_WIRE_LOG` environment switch (set it to any
//! non-empty value other than `0`). While enabled, every response stream
//! appends one JSONL line to `<session assets>/stream_wire.log` describing
//! what the provider actually sent over the wire: per-event-type counts, the
//! tail of the event sequence, the terminal event payload, usage numbers, the
//! output-item summary and the client-side outcome classification.
//!
//! This exists to diagnose responses that end "normally" while the model was
//! evidently cut off mid-work. In that case the client has no error, no retry
//! and no warning; a wire-level record is the evidence that separates "the
//! model chose to stop here" from "the gateway cut the generation after a few
//! tokens and still declared the response complete". A cut wire shows a
//! missing terminal event, an output item left open (`added` with no `done`),
//! or a short declared-complete response whose `output` array lacks the final
//! message item.
//!
//! Design constraints:
//! - Off by default: nothing is written unless the switch is set when the
//!   response stream starts.
//! - Best-effort: recording never influences streaming; failures are dropped.
//!   One line is written per streamed response by `Drop`, so every exit path
//!   (completion, error, cancellation, panic unwind) leaves the same evidence.
//! - Bounded: a line caps the raw terminal payload, the item list, the
//!   event-label tail and the distinct label count; the log file rotates to
//!   `stream_wire.log.1` past 8 MiB.
//! - Silent no-op without a driver context (unit tests, one-shot calls).

use std::{
    collections::{BTreeMap, VecDeque},
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use chrono::Utc;
use serde_json::{Value, json};

use crate::ai::{
    driver::runtime_ctx,
    tools::storage::file_store::current_session_assets_dir,
    types::StreamOutcome,
};

/// Environment switch that turns the recorder on.
const ENABLE_ENV: &str = "AIOS_STREAM_WIRE_LOG";

/// Log file name inside the session assets directory.
const WIRE_LOG_FILE: &str = "stream_wire.log";

/// Per-line cap for the raw terminal-event payload copy (the Responses wire
/// repeats the full output array — including encrypted reasoning blobs — inside
/// `response.completed`, so the raw copy must be bounded while the parsed
/// summary stays complete).
const MAX_TERMINAL_RAW_BYTES: usize = 16 * 1024;

/// Per-line cap for the serialized `incomplete_details` copy.
const MAX_DETAILS_BYTES: usize = 1024;

/// Log file size past which the file rotates to `stream_wire.log.1`.
const MAX_LOG_BYTES: u64 = 8 * 1024 * 1024;

/// Upper bound on output items retained per response.
const MAX_ITEMS: usize = 24;

/// Upper bound on the recent-event ring.
const TAIL_LEN: usize = 48;

/// Upper bound on distinct event labels tracked per response (defensive; a
/// healthy protocol sends a handful).
const MAX_EVENT_LABELS: usize = 96;

/// Process-local monotonic counter giving same-session lines a stable order.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// Serializes whole-line appends: parallel sub-agent streams share the same
/// session file within one process, and split writes would interleave lines.
static APPEND_LOCK: Mutex<()> = Mutex::new(());

/// Recorder state for one streamed response.
pub(super) struct StreamWireLog {
    enabled: bool,
    /// Target file, resolved when the stream starts; `None` disables writing.
    path: Option<PathBuf>,
    session: String,
    turn: usize,
    depth: usize,
    model: String,
    wire: &'static str,
    started: Instant,
    /// Milliseconds from `started` to the first and last observed events. Together with the
    /// `silence_ms` derived at record time they show whether a cut followed a long quiet
    /// stretch, and how long that stretch was.
    first_event_ms: Option<u64>,
    last_event_ms: Option<u64>,
    /// Largest gap between two consecutive observed events, plus the event that preceded it. This is the
    /// number the silence windows are calibrated against: it separates "the stream was quiet" from "the
    /// stream was quiet *and* then cut", which no single-event field can show.
    max_gap_ms: u64,
    max_gap_after: Option<String>,
    total_events: u64,
    payload_bytes: u64,
    counts: BTreeMap<String, u64>,
    counts_capped: bool,
    tail: VecDeque<String>,
    terminal: Option<RawNote>,
    terminal_status: Option<String>,
    terminal_details: Option<String>,
    done_marker: bool,
    usage: Option<Value>,
    output_text_chars: Option<usize>,
    items: Vec<ItemNote>,
    items_truncated: bool,
    client: Option<Value>,
}

impl StreamWireLog {
    /// Inert recorder for states that never observe a wire (tests, one-shots).
    pub(super) fn disabled() -> Self {
        Self {
            enabled: false,
            path: None,
            session: String::new(),
            turn: 0,
            depth: 0,
            model: String::new(),
            wire: "",
            started: Instant::now(),
            first_event_ms: None,
            last_event_ms: None,
            max_gap_ms: 0,
            max_gap_after: None,
            total_events: 0,
            payload_bytes: 0,
            counts: BTreeMap::new(),
            counts_capped: false,
            tail: VecDeque::new(),
            terminal: None,
            terminal_status: None,
            terminal_details: None,
            done_marker: false,
            usage: None,
            output_text_chars: None,
            items: Vec::new(),
            items_truncated: false,
            client: None,
        }
    }

    /// Arm the recorder for a new response stream. With the switch unset this
    /// resolves nothing and every later call is a no-op.
    pub(super) fn arm(&mut self, model: &str) {
        self.enabled = stream_wire_log_enabled();
        if !self.enabled {
            return;
        }
        self.model = model.to_string();
        self.session = runtime_ctx::current_session_id_or_empty();
        self.turn = runtime_ctx::current_turn_id_or_zero();
        self.depth = runtime_ctx::current_subagent_depth();
        self.started = Instant::now();
        self.first_event_ms = None;
        self.last_event_ms = None;
        self.max_gap_ms = 0;
        self.max_gap_after = None;
        self.path = current_session_assets_dir().map(|dir| dir.join(WIRE_LOG_FILE));
    }

    /// Provider adapter label (wire dialect) of the stream being recorded.
    pub(super) fn set_wire(&mut self, wire: &'static str) {
        self.wire = wire;
    }

    /// Record one raw SSE event. Called for every event before parsing, so a
    /// payload that fails to parse or is classified as ignorable is still
    /// visible in the record.
    pub(super) fn observe(&mut self, event_type: Option<&str>, payload: &str) {
        if !self.enabled {
            return;
        }
        self.total_events += 1;
        self.payload_bytes = self.payload_bytes.saturating_add(payload.len() as u64);
        let since_start = self.started.elapsed().as_millis() as u64;
        if let Some(previous) = self.last_event_ms {
            let gap = since_start.saturating_sub(previous);
            if gap > self.max_gap_ms {
                self.max_gap_ms = gap;
                // The event preceding the gap: the wait for its successor is what a silence window observes.
                self.max_gap_after = self.tail.back().cloned();
            }
        }
        self.first_event_ms.get_or_insert(since_start);
        self.last_event_ms = Some(since_start);

        let exact = event_type.map(str::trim).filter(|name| !name.is_empty());
        let label = match exact {
            Some(name) => name.to_string(),
            None => chunk_kind(payload).to_string(),
        };
        let normalized = label.to_ascii_lowercase();
        if normalized == "done" || normalized == "[done]" || normalized == "(done-marker)" {
            self.done_marker = true;
        }
        if self.counts.len() >= MAX_EVENT_LABELS && !self.counts.contains_key(&normalized) {
            self.counts_capped = true;
            *self.counts.entry("(other)".to_string()).or_insert(0) += 1;
        } else {
            *self.counts.entry(normalized.clone()).or_insert(0) += 1;
        }
        self.tail.push_back(label);
        if self.tail.len() > TAIL_LEN {
            self.tail.pop_front();
        }

        match normalized.as_str() {
            "response.output_item.added" => self.capture_item("added", payload),
            "response.output_item.done" => self.capture_item("done", payload),
            "response.output_text.done" => self.capture_output_text_done(payload),
            "response.completed" | "response.incomplete" | "response.failed" | "response.error"
            | "error" => {
                if self.terminal.is_none() {
                    self.terminal = Some(RawNote {
                        event: exact.unwrap_or_default().to_string(),
                        payload: cap_text(payload, MAX_TERMINAL_RAW_BYTES),
                    });
                }
                if let Ok(value) = serde_json::from_str::<Value>(payload) {
                    capture_terminal_fields(&value, self);
                }
            }
            _ => {}
        }

        // Usage arrives either inside a named terminal event or, on the
        // chat-completions wire, in a final unnamed chunk. Parsing is limited
        // to payloads that can actually carry it.
        if self.usage.is_none() && payload.contains("\"usage\"") {
            if let Ok(value) = serde_json::from_str::<Value>(payload) {
                capture_usage(&value, self);
            }
        }
    }

    /// Fill the client-side classification once the stream has been resolved
    /// (called from the finalize path, which knows the outcome and the
    /// collected content sizes).
    pub(super) fn note_outcome(&mut self, note: WireOutcomeNote<'_>) {
        if !self.enabled {
            return;
        }
        let usage = note.usage.map(|(prompt, cached, completion, reasoning)| {
            json!({
                "prompt_tokens": prompt,
                "cached_prompt_tokens": cached,
                "completion_tokens": completion,
                "reasoning_tokens": reasoning,
            })
        });
        self.client = Some(json!({
            "outcome": outcome_label(note.outcome),
            "truncated_by_length": note.truncated_by_length,
            "stream_error": note.stream_error,
            "finish_reason": note.finish_reason,
            "dropped_malformed_tool_call": note.dropped_malformed_tool_call,
            "tool_calls": note.tool_calls,
            "assistant_chars": note.assistant_chars,
            "reasoning_chars": note.reasoning_chars,
            "response_completed": note.response_completed,
            "response_incomplete": note.response_incomplete,
            "tool_args_cap_exceeded": note.tool_args_cap_exceeded,
            "decode_errors": note.decode_errors,
            "usage": usage,
        }));
    }

    /// Record that the stream was cut by a read error. Raw wire data alone
    /// cannot say the client treated the stream as failed, and this path does
    /// not go through `note_outcome`.
    pub(super) fn note_stream_cut(&mut self, decode_errors: usize) {
        if !self.enabled {
            return;
        }
        self.client = Some(json!({
            "outcome": "stream_error",
            "stream_error": true,
            "decode_errors": decode_errors,
        }));
    }

    /// Record a user cancellation (the raw record still shows what had arrived).
    pub(super) fn note_cancelled(&mut self) {
        if !self.enabled {
            return;
        }
        self.client = Some(json!({ "outcome": "cancelled" }));
    }

    fn capture_item(&mut self, phase: &'static str, payload: &str) {
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            return;
        };
        let index = value
            .get("output_index")
            .and_then(Value::as_u64)
            .unwrap_or(u64::MAX);
        if let Some(item) = value.get("item") {
            self.push_item(ItemNote::from_item(phase, index, item));
        }
    }

    fn capture_output_text_done(&mut self, payload: &str) {
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            return;
        };
        if let Some(text) = value.get("text").and_then(Value::as_str) {
            self.output_text_chars = Some(text.chars().count());
        }
    }

    fn push_item(&mut self, item: ItemNote) {
        if self.items.len() >= MAX_ITEMS {
            self.items_truncated = true;
            return;
        }
        self.items.push(item);
    }

    fn finish(&mut self) {
        if !self.enabled || self.total_events == 0 {
            return;
        }
        let Some(path) = self.path.clone() else {
            return;
        };
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let value = self.to_value(seq);
        let Ok(mut line) = serde_json::to_string(&value) else {
            return;
        };
        line.push('\n');
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        rotate_if_large(&path);
        let _guard = APPEND_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
            // One `write_all` per line, under O_APPEND, keeps records intact
            // when parallel sub-agent streams share the file.
            let _ = file.write_all(line.as_bytes());
        }
    }

    fn to_value(&self, seq: u64) -> Value {
        json!({
            "seq": seq,
            "ts": Utc::now().to_rfc3339(),
            "session": &self.session,
            "turn": self.turn,
            "depth": self.depth,
            "model": &self.model,
            "wire": self.wire,
            "elapsed_ms": self.started.elapsed().as_millis() as u64,
            "events_total": self.total_events,
            "payload_bytes": self.payload_bytes,
            "first_event_ms": self.first_event_ms,
            "last_event_ms": self.last_event_ms,
            "max_gap_ms": self.max_gap_ms,
            "max_gap_after": &self.max_gap_after,
            "silence_ms": self
                .last_event_ms
                .map(|last| (self.started.elapsed().as_millis() as u64).saturating_sub(last)),
            "event_counts": &self.counts,
            "event_counts_capped": self.counts_capped,
            "tail": &self.tail,
            "terminal": self.terminal.as_ref().map(RawNote::to_value),
            "terminal_status": &self.terminal_status,
            "terminal_details": &self.terminal_details,
            "done_marker": self.done_marker,
            "usage": &self.usage,
            "output_text_chars": self.output_text_chars,
            "items": self.items.iter().map(ItemNote::to_value).collect::<Vec<_>>(),
            "items_truncated": self.items_truncated,
            "client": &self.client,
        })
    }
}

impl Drop for StreamWireLog {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Client-side classification captured when the stream is finalized. Separate
/// struct instead of a long parameter list.
pub(super) struct WireOutcomeNote<'a> {
    pub(super) outcome: StreamOutcome,
    pub(super) truncated_by_length: bool,
    pub(super) stream_error: bool,
    pub(super) finish_reason: Option<&'a str>,
    pub(super) dropped_malformed_tool_call: bool,
    pub(super) tool_calls: usize,
    pub(super) assistant_chars: usize,
    pub(super) reasoning_chars: usize,
    pub(super) response_completed: bool,
    pub(super) response_incomplete: bool,
    pub(super) tool_args_cap_exceeded: bool,
    pub(super) decode_errors: usize,
    pub(super) usage: Option<(u64, u64, u64, u64)>,
}

/// One raw payload copy kept for later inspection.
struct RawNote {
    event: String,
    payload: String,
}

impl RawNote {
    fn to_value(&self) -> Value {
        json!({ "event": &self.event, "raw": &self.payload })
    }
}

/// One output item observed on the wire (`added`, `done`, or inside the
/// terminal payload's `output` array).
struct ItemNote {
    phase: &'static str,
    index: u64,
    item_type: String,
    status: Option<String>,
    name: Option<String>,
    chars: Option<usize>,
}

impl ItemNote {
    fn from_item(phase: &'static str, index: u64, item: &Value) -> Self {
        let item_type = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let status = item.get("status").and_then(Value::as_str).map(str::to_string);
        let name = item.get("name").and_then(Value::as_str).map(str::to_string);
        let chars = match item_type.as_str() {
            "message" => item.get("content").and_then(Value::as_array).map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .map(|text| text.chars().count())
                    .sum()
            }),
            "function_call" => item
                .get("arguments")
                .and_then(Value::as_str)
                .map(|args| args.chars().count()),
            _ => None,
        };
        Self {
            phase,
            index,
            item_type,
            status,
            name,
            chars,
        }
    }

    fn to_value(&self) -> Value {
        json!({
            "phase": self.phase,
            "index": self.index,
            "type": &self.item_type,
            "status": &self.status,
            "name": &self.name,
            "chars": self.chars,
        })
    }
}

fn stream_wire_log_enabled() -> bool {
    std::env::var(ENABLE_ENV)
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
        .unwrap_or(false)
}

fn outcome_label(outcome: StreamOutcome) -> &'static str {
    match outcome {
        StreamOutcome::Completed => "completed",
        StreamOutcome::EmptyResponse => "empty_response",
        StreamOutcome::Truncated => "truncated",
        StreamOutcome::Cancelled => "cancelled",
        StreamOutcome::ToolCall => "tool_call",
    }
}

/// Whether a named event declares how the response ended.
fn is_terminal_event(label: &str) -> bool {
    matches!(
        label,
        "response.completed" | "response.incomplete" | "response.failed" | "response.error" | "error"
    )
}

/// Coarse classification for events without an SSE event name (chat-completions
/// wire) so the recorded sequence stays readable. Order matters: every chat
/// chunk carries `"finish_reason":null`, so shape checks must come first.
fn chunk_kind(payload: &str) -> &'static str {
    if payload.trim().eq_ignore_ascii_case("[DONE]") {
        return "(done-marker)";
    }
    if payload.contains("\"usage\"") {
        return "(chunk:usage)";
    }
    if payload.contains("\"tool_calls\"") {
        return "(chunk:tool_calls)";
    }
    if payload.contains("\"reasoning_content\"") {
        return "(chunk:reasoning)";
    }
    if payload.contains("\"content\"") {
        return "(chunk:content)";
    }
    if payload.contains("\"finish_reason\"") {
        return "(chunk:finish)";
    }
    "(chunk:other)"
}

/// Extract the terminal event's essentials: declared status, incomplete
/// details and the provider's own final `output` array (the cross-check for
/// "did the wire consider the response complete, and with which items").
fn capture_terminal_fields(value: &Value, log: &mut StreamWireLog) {
    let body = value.get("response").unwrap_or(value);
    if log.terminal_status.is_none() {
        if let Some(status) = body.get("status").and_then(Value::as_str) {
            log.terminal_status = Some(status.to_string());
        }
    }
    if log.terminal_details.is_none() {
        if let Some(details) = body.get("incomplete_details").filter(|details| !details.is_null()) {
            log.terminal_details = Some(cap_text(&details.to_string(), MAX_DETAILS_BYTES));
        }
    }
    if let Some(items) = body.get("output").and_then(Value::as_array) {
        for (index, item) in items.iter().enumerate() {
            log.push_item(ItemNote::from_item("terminal_output", index as u64, item));
        }
    }
}

fn capture_usage(value: &Value, log: &mut StreamWireLog) {
    let usage = value
        .get("usage")
        .or_else(|| value.get("response").and_then(|response| response.get("usage")));
    if let Some(usage) = usage.filter(|usage| !usage.is_null()) {
        log.usage = Some(usage.clone());
    }
}

/// Cap a raw payload copy (char-boundary safe), marking the cut so a truncated
/// capture is never mistaken for the full wire payload.
fn cap_text(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}…[truncated {} bytes of {}]",
        &text[..end],
        text.len() - end,
        text.len()
    )
}

fn rotate_if_large(path: &Path) {
    let too_large = std::fs::metadata(path)
        .map(|meta| meta.len() > MAX_LOG_BYTES)
        .unwrap_or(false);
    if too_large {
        let rotated = path.with_file_name(format!("{WIRE_LOG_FILE}.1"));
        let _ = std::fs::rename(path, rotated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn test_log() -> StreamWireLog {
        let mut log = StreamWireLog::disabled();
        log.enabled = true;
        log.model = "test-model".to_string();
        log
    }

    #[test]
    fn disabled_recorder_ignores_events_and_writes_nothing() {
        let mut log = StreamWireLog::disabled();
        log.observe(Some("response.output_text.delta"), r#"{"delta":"hi"}"#);
        assert_eq!(log.total_events, 0);
        // `finish` must stay a no-op without the switch (and must not panic).
        log.finish();
    }

    #[test]
    fn event_timeline_records_first_last_and_silence() {
        let mut log = test_log();
        assert!(log.to_value(1)["silence_ms"].is_null());
        log.observe(Some("response.created"), r#"{"type":"response.created"}"#);
        log.observe(
            Some("response.in_progress"),
            r#"{"type":"response.in_progress"}"#,
        );
        let value = log.to_value(1);
        assert!(value["first_event_ms"].is_u64());
        assert!(value["last_event_ms"].is_u64());
        assert!(value["silence_ms"].is_u64());
        assert!(value["last_event_ms"].as_u64() >= value["first_event_ms"].as_u64());
    }

    #[test]
    fn timeline_reports_the_largest_gap_and_the_event_before_it() {
        let mut log = test_log();
        log.observe(Some("response.created"), r#"{"type":"response.created"}"#);
        log.observe(
            Some("response.in_progress"),
            r#"{"type":"response.in_progress"}"#,
        );
        // Wind the clock back instead of sleeping: the next observation then sees a long quiet stretch.
        log.started -= Duration::from_millis(70_000);
        log.observe(
            Some("response.output_item.added"),
            r#"{"output_index":0,"item":{"type":"reasoning","status":"in_progress"}}"#,
        );
        log.started -= Duration::from_millis(5_000);
        log.observe(
            Some("response.output_item.done"),
            r#"{"output_index":0,"item":{"type":"reasoning","status":"completed"}}"#,
        );

        let value = log.to_value(1);
        let gap = value["max_gap_ms"].as_u64().expect("gap is recorded");
        assert!(
            (70_000..75_000).contains(&gap),
            "expected the 70s stretch to dominate, got {gap}"
        );
        assert_eq!(value["max_gap_after"], "response.in_progress");
    }

    #[test]
    fn response_completed_captures_terminal_usage_and_items() {
        let mut log = test_log();
        log.observe(
            Some("response.output_item.added"),
            r#"{"output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"in_progress"}}"#,
        );
        log.observe(Some("response.reasoning_text.delta"), r#"{"delta":"think"}"#);
        log.observe(
            Some("response.output_item.done"),
            r#"{"output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"completed"}}"#,
        );
        log.observe(Some("response.output_text.delta"), r#"{"delta":"hello"}"#);
        log.observe(
            Some("response.output_text.done"),
            r#"{"text":"hello world"}"#,
        );
        log.observe(
            Some("response.completed"),
            r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","status":"completed","content":[{"type":"output_text","text":"hello world"}]}],"usage":{"input_tokens":100,"output_tokens":5,"total_tokens":105}}}"#,
        );

        let value = log.to_value(1);
        assert_eq!(value["terminal"]["event"], "response.completed");
        assert!(
            value["terminal"]["raw"].as_str().unwrap().contains("\"usage\""),
            "terminal raw must keep the usage block: {}",
            value["terminal"]["raw"]
        );
        assert_eq!(value["terminal_status"], "completed");
        assert_eq!(value["usage"]["input_tokens"], 100);
        assert_eq!(value["output_text_chars"], 11);
        assert_eq!(value["event_counts"]["response.output_text.delta"], 1);
        assert_eq!(value["events_total"], 6);
        let items = value["items"].as_array().unwrap();
        // added + done + terminal output item
        assert_eq!(items.len(), 3);
        assert_eq!(items[0]["phase"], "added");
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[1]["phase"], "done");
        assert_eq!(items[2]["phase"], "terminal_output");
        assert_eq!(items[2]["type"], "message");
        assert_eq!(items[2]["chars"], 11);
        assert!(!value["done_marker"].as_bool().unwrap());
    }

    #[test]
    fn unnamed_chat_chunks_are_classified_and_usage_captured() {
        let mut log = test_log();
        log.observe(None, r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#);
        log.observe(
            None,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0}]},"finish_reason":null}]}"#,
        );
        log.observe(
            None,
            r#"{"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#,
        );
        log.observe(None, r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#);
        log.observe(
            None,
            r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}"#,
        );
        log.observe(None, "[DONE]");

        let value = log.to_value(2);
        assert_eq!(value["event_counts"]["(chunk:content)"], 1);
        assert_eq!(value["event_counts"]["(chunk:tool_calls)"], 1);
        assert_eq!(value["event_counts"]["(chunk:reasoning)"], 1);
        assert_eq!(value["event_counts"]["(chunk:finish)"], 1);
        assert_eq!(value["event_counts"]["(chunk:usage)"], 1);
        assert_eq!(value["event_counts"]["(done-marker)"], 1);
        assert_eq!(value["usage"]["total_tokens"], 12);
        assert!(value["done_marker"].as_bool().unwrap());
        // No terminal event on this wire.
        assert!(value["terminal"].is_null());
    }

    #[test]
    fn chat_finish_reason_null_does_not_mask_content_chunks() {
        assert_eq!(
            chunk_kind(r#"{"choices":[{"delta":{"content":"a"},"finish_reason":null}]}"#),
            "(chunk:content)"
        );
        assert_eq!(
            chunk_kind(r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#),
            "(chunk:finish)"
        );
    }

    #[test]
    fn oversized_terminal_payload_is_capped() {
        let mut log = test_log();
        let big = format!(
            r#"{{"type":"response.completed","response":{{"status":"completed","output":[],"blob":"{}"}}}}"#,
            "x".repeat(64 * 1024)
        );
        log.observe(Some("response.completed"), &big);
        let value = log.to_value(3);
        let raw = value["terminal"]["raw"].as_str().unwrap();
        assert!(raw.len() < 2 * MAX_TERMINAL_RAW_BYTES, "raw capped: {}", raw.len());
        assert!(raw.contains("truncated"));
    }

    #[test]
    fn item_capture_is_bounded() {
        let mut log = test_log();
        for index in 0..(MAX_ITEMS + 5) {
            log.observe(
                Some("response.output_item.added"),
                &format!(r#"{{"output_index":{index},"item":{{"type":"function_call","name":"t"}}}}"#),
            );
        }
        assert_eq!(log.items.len(), MAX_ITEMS);
        assert!(log.items_truncated);
        let value = log.to_value(4);
        assert_eq!(value["items_truncated"], true);
        assert_eq!(value["items"].as_array().unwrap().len(), MAX_ITEMS);
    }

    #[test]
    fn outcome_note_is_recorded_when_enabled() {
        let mut log = test_log();
        log.observe(Some("response.completed"), r#"{"response":{"status":"completed"}}"#);
        log.note_outcome(WireOutcomeNote {
            outcome: StreamOutcome::Completed,
            truncated_by_length: false,
            stream_error: false,
            finish_reason: None,
            dropped_malformed_tool_call: false,
            tool_calls: 0,
            assistant_chars: 39,
            reasoning_chars: 0,
            response_completed: true,
            response_incomplete: false,
            tool_args_cap_exceeded: false,
            decode_errors: 0,
            usage: Some((53403, 0, 57, 27)),
        });
        let value = log.to_value(5);
        assert_eq!(value["client"]["outcome"], "completed");
        assert_eq!(value["client"]["assistant_chars"], 39);
        assert_eq!(value["client"]["usage"]["completion_tokens"], 57);
        assert_eq!(value["client"]["usage"]["reasoning_tokens"], 27);
    }

    #[test]
    fn switch_reads_env_variants() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        unsafe { std::env::remove_var(ENABLE_ENV) };
        assert!(!stream_wire_log_enabled(), "unset must stay off");
        unsafe { std::env::set_var(ENABLE_ENV, "0") };
        assert!(!stream_wire_log_enabled());
        unsafe { std::env::set_var(ENABLE_ENV, " ") };
        assert!(!stream_wire_log_enabled());
        unsafe { std::env::set_var(ENABLE_ENV, "1") };
        assert!(stream_wire_log_enabled());
        unsafe { std::env::remove_var(ENABLE_ENV) };
    }

    #[test]
    fn drop_appends_one_jsonl_line_to_the_target_file() {
        let dir = std::env::temp_dir().join(format!(
            ".agent_wire_log_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join(WIRE_LOG_FILE);
        {
            let mut log = test_log();
            log.path = Some(path.clone());
            log.observe(
                Some("response.completed"),
                r#"{"type":"response.completed","response":{"status":"completed"}}"#,
            );
            // The line is written by `Drop`.
        }
        let content = std::fs::read_to_string(&path).expect("line written on drop");
        let mut lines = content.lines();
        let line = lines.next().expect("one line per response");
        assert!(lines.next().is_none(), "exactly one line per response");
        let value: Value = serde_json::from_str(line).expect("line is JSON");
        assert_eq!(value["terminal"]["event"], "response.completed");
        assert_eq!(value["events_total"], 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
