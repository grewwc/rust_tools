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
pub(in crate::ai) use markdown::messages_to_markdown;
#[allow(unused_imports)]
pub(in crate::ai) use sessions::{SessionInfo, SessionStore};
#[allow(unused_imports)]
pub(in crate::ai) use types::{COLON, MAX_HISTORY_LINES, Message, NEWLINE};

pub(in crate::ai) fn build_context_history(
    history_count: usize,
    history_file: &PathBuf,
    history_max_chars: usize,
    history_keep_last: usize,
    history_summary_max_chars: usize,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let history = build_message_arr(history_count, history_file)?;
    let out = if history_max_chars == 0 {
        history
    } else {
        compress_messages_for_context(
            history,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
        )
    };
    Ok(out)
}
