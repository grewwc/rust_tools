mod blob;
mod compress;
mod markdown;
mod sessions;
mod sqlite;
mod types;

use std::path::PathBuf;

#[allow(unused_imports)]
pub(in crate::ai) use blob::{
    append_history, append_history_messages, build_message_arr, delete_history_artifacts,
};
#[allow(unused_imports)]
pub(in crate::ai) use compress::compress_messages_for_context;
#[allow(unused_imports)]
pub(in crate::ai) use compress::value_to_string;
#[allow(unused_imports)]
pub(in crate::ai) use markdown::messages_to_markdown;
#[allow(unused_imports)]
pub(in crate::ai) use sessions::{SessionInfo, SessionStore};
#[allow(unused_imports)]
pub(in crate::ai) use types::{COLON, MAX_HISTORY_TURNS, Message, NEWLINE};

pub(in crate::ai) const ROLE_SYSTEM: &str = types::ROLE_SYSTEM;
pub(in crate::ai) const ROLE_INTERNAL_NOTE: &str = types::ROLE_INTERNAL_NOTE;

pub(in crate::ai) fn is_internal_note_role(role: &str) -> bool {
    types::is_internal_note_role(role)
}

pub(in crate::ai) fn is_system_like_role(role: &str) -> bool {
    types::is_system_like_role(role)
}

pub(in crate::ai) fn build_context_history(
    history_count: usize,
    history_file: &PathBuf,
    history_max_chars: usize,
    history_keep_last: usize,
    history_summary_max_chars: usize,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    // Load full session history first, then apply context-size compression.
    // This avoids dropping old turns before compression has a chance to summarize them.
    let history = build_message_arr(usize::MAX, history_file)?;
    let out = if history_max_chars == 0 {
        if history_count >= history.len() {
            history
        } else {
            history[history.len() - history_count..].to_vec()
        }
    } else {
        let keep_last = if history_count == 0 {
            history_keep_last
        } else {
            history_count
        };
        compress_messages_for_context(
            history,
            history_max_chars,
            keep_last,
            history_summary_max_chars,
        )
    };
    Ok(out)
}
