//! macOS backend: CoreGraphics event injection (`input.rs`), screen capture
//! through `/usr/sbin/screencapture` (`capture.rs`), and display / window /
//! application queries (`screen.rs`).
//!
//! Everything here follows the platform-neutral contract: query functions return a
//! model-facing report, action functions return `Ok(())` or a model-facing error
//! string. Adding another platform means another module with the same signatures,
//! never a change to the tools the model sees.

mod capture;
mod input;
mod screen;

pub(crate) use capture::screenshot;
pub(crate) use input::{click, drag, move_mouse, press_key, scroll, type_text};
pub(crate) use screen::{activate_app, list_windows, screen_info};

/// Phrases the host treats as transport failures (`src/bin/ai/mcp/client.rs`).
/// Error text that reached this module came from a helper binary, so it is scanned
/// before being passed back: forwarding one of these words would make the host kill
/// and restart this server in the middle of a conversation.
const TRANSPORT_TRIGGERS: [&str; 6] = [
    "mcp response timeout",
    "broken pipe",
    "closed the stream",
    "process exited",
    "failed to read response",
    "failed waiting for mcp response",
];

/// Replace any transport trigger phrase with a neutral marker, comparing
/// case-insensitively because the host's match is on the exact lowercase spelling
/// while a helper binary may capitalize it.
pub(super) fn sanitize(text: &str) -> String {
    let lowered = text.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < text.len() {
        match TRANSPORT_TRIGGERS
            .iter()
            .find(|trigger| lowered[index..].starts_with(**trigger))
        {
            Some(trigger) => {
                out.push_str("transport-error");
                index += trigger.len();
            }
            None => {
                let Some(character) = text[index..].chars().next() else {
                    break;
                };
                out.push(character);
                index += character.len_utf8();
            }
        }
    }
    out
}