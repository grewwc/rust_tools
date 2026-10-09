//! Remote confirmation for serve turn children.
//!
//! A serve turn child has no terminal: the user watches the turn from the web
//! client (`serve/app.html`) on another device. When a tool needs explicit
//! confirmation (today `git commit` / `git stash`), the child publishes a
//! `ConfirmRequest` frame on the serve live FIFO, which the daemon relays to
//! the client and answers by writing `yes` / `no` into the child's stdin. The
//! child reads that line back here, so the gate behaves like a local prompt
//! whose keyboard is the phone.
//!
//! Frame payloads are JSON: `{"id":<u64>,"prompt":"..."}` for the request and
//! `{"id":<u64>}` for the resolution. `ai::serve` owns the daemon half of this
//! protocol; keep the two sides in sync.

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