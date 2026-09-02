// Names that only the sibling test module (`tests.rs`, `use super::*`) needs.
// Gated on `cfg(test)` so the production build sees no unused-import warnings.
#[cfg(test)]
use {
    std::path::{Path, PathBuf},
    std::time::{Duration, Instant},
    rustc_hash::{FxHashMap, FxHashSet},
    crate::ai::types::ToolCall,
    super::compress::value_to_string,
    super::types::{Message, ROLE_INTERNAL_NOTE, SkillActivationEvent, ToolExecutionOutcome},
    connection::{open_history_db, sqlite_error, sqlite_error_kind},
};

mod connection;
mod context;
mod lock;
mod metadata;
mod migrations;
mod outcomes;
mod revision;
mod rollback;
mod store;
mod trim;

const STALE_PATCH_TARGETS_META_KEY: &str = "stale_patch_targets_v1";
const LLM_PRUNE_MARKS_META_KEY: &str = "llm_prune_marks_v1";
const LAST_ACTIVITY_META_KEY: &str = "last_activity_unix_ms";
const SESSION_MARKED_META_KEY: &str = "session_marked";

pub(super) use lock::{
    delete_session_state_lock, remove_session_state_lock_entry, with_session_state_lock,
};

// External call surface (unchanged from the pre-split single module): these
// `pub(in crate::ai)` items are what history/mod.rs, sessions.rs, blob.rs and
// checkpoint.rs reach through `super::sqlite::…`.
#[cfg(test)]
pub(in crate::ai) use revision::history_revision_cache_contains;
#[cfg(test)]
pub(in crate::ai) use context::write_context_snapshot_sqlite;
pub(in crate::ai) use revision::{read_history_revision, remove_history_revision_cache_entry};
pub(in crate::ai) use rollback::{
    backup_sqlite, fork_history_for_subagent, live_rollback_transaction_is_published,
    reset_history_for_subagent, restore_sqlite_after_rollback,
    restore_sqlite_after_rollback_with_transaction,
};
pub(in crate::ai) use store::{
    ContextHistory, SessionListMetadata, append_history_sqlite,
    append_history_sqlite_for_model, coalesce_repeated_wait_wake_notes_sqlite,
    count_user_turns_sqlite, read_all_messages_sqlite, replace_all_messages_sqlite,
    reserve_turn_index_sqlite,
};
pub(in crate::ai) use outcomes::{
    append_interrupted_stream_diagnostic_sqlite, append_skill_activation_event_sqlite,
    append_tool_execution_outcomes_sqlite, read_image_digest_sqlite, read_llm_prune_marks_sqlite,
    read_skill_activation_events_sqlite, read_stale_patch_targets_sqlite,
    read_tool_execution_outcomes_sqlite, read_tool_message_ids_sqlite, upsert_image_digest_sqlite,
    write_llm_prune_marks_sqlite, write_stale_patch_targets_sqlite,
};
pub(in crate::ai) use context::{
    build_message_arr_sqlite, read_context_history_sqlite, read_recent_messages_sqlite,
    read_recent_turn_window_sqlite, write_context_snapshot_sqlite_with_busy_timeout,
};
pub(in crate::ai) use trim::{
    clear_session_history_sqlite, remap_context_checkpoint_paths_sqlite, truncate_messages_sqlite,
    truncate_messages_to_user_turns_sqlite,
};
pub(in crate::ai) use metadata::{
    read_first_user_prompt_sqlite, read_session_list_metadata_sqlite, read_session_marked_sqlite,
    read_session_title_origin_sqlite, read_session_title_sqlite, write_session_marked_sqlite,
    write_session_title_sqlite,
};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
