//! Remote confirmation and text input for serve turn children.
//!
//! A serve turn child has no terminal: the user watches the turn from the web
//! client (`serve/app.html`) on another device. When a tool needs explicit
//! confirmation (today `git commit` / `git push` / `git stash` / `git checkout` / `git switch`) or a free-text answer,
//! the child publishes a request frame on the
//! serve live FIFO, which the daemon relays to the client and answers by
//! writing one line into the child's stdin. The child reads that line back
//! here, so the gate behaves like a local prompt whose keyboard is the phone.
//!
//! Frame payloads are JSON: `{"id":<u64>,"prompt":"..."}` for the request and
//! `{"id":<u64>}` for the resolution. A confirm answer is `yes` / `no`; an
//! input answer is the raw text, or the cancel marker `\u{1e}` (Record
//! Separator, untypeable from a web input) when the client cancels. `ai::serve`
//! owns the daemon half of this protocol; keep the two sides in sync.

use std::io::BufRead;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ai::background::{self, ServeLiveKind};

/// Confirmation requests issued by this process, monotonic per turn child.
static NEXT_CONFIRM_ID: AtomicU64 = AtomicU64::new(1);

/// Serializes questions raised by this process. The daemon holds one pending
/// question per session and answers with one line on stdin, so two concurrent
/// waiters would overwrite each other's question and then race for that line.
static CONFIRM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Starting budget for the prompt text inside one frame's payload.
const MAX_PROMPT_BYTES: usize = 1500;

/// Ceiling for the whole payload. The publisher splits an oversized payload
/// into fragments of the same kind, which a reader cannot reassemble into one
/// `{"id","prompt"}` object, so the question must fit in a single frame.
const MAX_PAYLOAD_BYTES: usize = background::MAX_SERVE_FRAME_BYTES - 64;

/// Marker that turns an ordinary input question into a phone-action request.
/// The client strips it, runs the action through the Android bridge, and
/// answers with one JSON line; the child reads that line as the tool result.
pub(crate) const NATIVE_PREFIX: &str = "@@aserve-native ";

/// Whether this turn child should route confirmation prompts to the serve
/// client instead of the local terminal. False in every non-serve process.
pub(crate) fn active() -> bool {
    background::serve_confirm_channel()
}

/// Ask the serve client to confirm `prompt`, blocking until it answers.
///
/// `Some(true)` approves, `Some(false)` declines, and `None` means the channel
/// closed (daemon gone or turn torn down); callers treat that as canceled.
pub(crate) fn confirm(prompt: &str) -> Option<bool> {
    if !active() {
        return None;
    }
    // Held across the whole exchange: one outstanding question per process.
    // A poisoned lock still serializes, so recover the guard.
    let _serialized = CONFIRM_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let id = NEXT_CONFIRM_ID.fetch_add(1, Ordering::Relaxed);
    let payload = request_payload(id, prompt);
    background::publish_serve_frame(ServeLiveKind::ConfirmRequest, &payload);
    let answer = read_answer();
    background::publish_serve_frame(
        ServeLiveKind::ConfirmDone,
        &serde_json::json!({ "id": id }).to_string(),
    );
    answer
}

/// Ask the serve client for a free-text answer to `prompt`, blocking until it
/// answers. `Some(text)` is the client's answer (non-empty); `None` means the
/// channel closed (daemon gone or turn torn down) or the client canceled.
pub(crate) fn input(prompt: &str) -> Option<String> {
    if !active() {
        return None;
    }
    // Same serialization as `confirm`: one outstanding question per process.
    let _serialized = CONFIRM_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let id = NEXT_CONFIRM_ID.fetch_add(1, Ordering::Relaxed);
    let payload = request_payload(id, prompt);
    background::publish_serve_frame(ServeLiveKind::InputRequest, &payload);
    let answer = read_input();
    background::publish_serve_frame(
        ServeLiveKind::InputDone,
        &serde_json::json!({ "id": id }).to_string(),
    );
    answer
}

/// Ask the Android client to perform one phone action, blocking until it
/// answers. The response is a single-line JSON object (`{"ok":bool,...}`), so
/// it can travel through the same stdin channel as a free-text answer.
pub(crate) fn native(request: &serde_json::Value) -> Option<String> {
    if !active() {
        return None;
    }
    let prompt = format!("{NATIVE_PREFIX}{request}");
    if prompt.len() > MAX_PAYLOAD_BYTES {
        return None;
    }
    let _serialized = CONFIRM_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let id = NEXT_CONFIRM_ID.fetch_add(1, Ordering::Relaxed);
    let payload = serde_json::json!({ "id": id, "prompt": prompt }).to_string();
    if payload.len() > MAX_PAYLOAD_BYTES {
        return None;
    }
    background::publish_serve_frame(ServeLiveKind::InputRequest, &payload);
    let answer = read_input();
    background::publish_serve_frame(
        ServeLiveKind::InputDone,
        &serde_json::json!({ "id": id }).to_string(),
    );
    answer
}

/// Build the request payload, shrinking the prompt until the whole JSON fits
/// one frame. JSON escaping can expand a byte sixfold (control characters), so
/// a byte budget on the raw text is only a starting estimate.
fn request_payload(id: u64, prompt: &str) -> String {
    let mut budget = MAX_PROMPT_BYTES;
    loop {
        let payload = serde_json::json!({
            "id": id,
            "prompt": clamp_prompt(prompt, budget),
        })
        .to_string();
        if payload.len() <= MAX_PAYLOAD_BYTES || budget == 0 {
            return payload;
        }
        budget /= 2;
    }
}

/// Read the daemon's one-line decision from stdin. EOF is `None`; an
/// unrecognized line keeps waiting, matching the local prompt's re-ask.
fn read_answer() -> Option<bool> {
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => return None,
            Ok(_) => match line.trim().to_ascii_lowercase().as_str() {
                "yes" | "y" => return Some(true),
                "no" | "n" => return Some(false),
                _ => {}
            },
            Err(_) => return None,
        }
    }
}

/// Read the daemon's one-line answer to an input question from stdin. EOF and
/// the cancel marker (`\u{1e}`, written by the daemon when the client cancels)
/// are `None`; an empty line keeps waiting, matching the confirm re-ask.
fn read_input() -> Option<String> {
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => return None,
            Ok(_) => {
                let text = line.trim_end_matches(['\r', '\n']).to_string();
                if text == "\u{1e}" {
                    return None;
                }
                if !text.is_empty() {
                    return Some(text);
                }
            }
            Err(_) => return None,
        }
    }
}

/// Truncate `text` to `budget` bytes on a char boundary.
fn clamp_prompt(text: &str, budget: usize) -> &str {
    if text.len() <= budget {
        return text;
    }
    let mut end = budget;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[cfg(test)]
mod tests {
    use super::{MAX_PAYLOAD_BYTES, request_payload};

    /// A prompt full of control characters still yields one frame's worth of
    /// JSON: a larger payload would be published as fragments the daemon
    /// cannot reassemble, leaving the question unanswerable while this process
    /// waits on stdin.
    #[test]
    fn request_payload_always_fits_one_frame() {
        let heavy = "\u{1}".repeat(20_000);
        let payload = request_payload(7, &heavy);
        assert!(
            payload.len() <= MAX_PAYLOAD_BYTES,
            "payload is {} bytes",
            payload.len()
        );
        let value: serde_json::Value = serde_json::from_str(&payload).expect("valid json");
        assert_eq!(value["id"], 7);
        // A normal prompt travels untruncated.
        let plain = request_payload(1, "Confirm git commit: git commit -m x");
        assert!(plain.contains("Confirm git commit"));
        assert!(!plain.contains("truncated"));
    }
}