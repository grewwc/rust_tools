use crate::ai::{history::Message, types::App};

/// Dependency-inversion entry point: lets a `HistoryStore` be injected as the history
/// storage implementation (audit/encryption/mock/testing).
/// Keeps the same `ephemeral` and `coalesce` semantics as `persist_pending_turn_messages`
/// and only pushes the final `append` down to the port, so middleware can instrument
/// without changing any driver lines.
pub(in crate::ai::driver::turn_runtime) fn persist_pending_turn_messages_with_store(
    history_file: &std::path::Path,
    source_model: &str,
    one_shot_mode: bool,
    session_is_none: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
    store: &dyn crate::ai::ports::HistoryStore,
) -> bool {
    let ephemeral = one_shot_mode && session_is_none;
    if ephemeral || *persisted_turn_messages >= turn_messages.len() {
        return true;
    }
    if *persisted_turn_messages == 0 {
        if let Some(first) = turn_messages.first() {
            let _ = crate::ai::history::coalesce_repeated_wait_wake_notes(history_file, first);
        }
    }
    if let Err(err) = store.append_messages_for_model(
        history_file,
        &turn_messages[*persisted_turn_messages..],
        source_model,
    ) {
        // Keep the same warning wording as the old path when the append fails;
        // source_model is pushed down through the port to the sqlite provenance column
        // and is no longer dropped at this layer.
        eprintln!("[Warning] Failed to save history: {}", err);
        return false;
    }
    *persisted_turn_messages = turn_messages.len();
    true
}

pub(in crate::ai::driver::turn_runtime) fn persist_pending_turn_messages(
    app: &App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> bool {
    persist_pending_turn_messages_for_model(
        app,
        &app.current_model,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
    )
}

/// Writes the canonical source metadata using the model that actually produced these
/// messages.
///
/// Auto-fallback does not rewrite `app.current_model`, so the response path must pass the
/// actual model returned by the provider explicitly; other interrupt paths without a model
/// response can keep using the default entry above.
pub(in crate::ai::driver::turn_runtime) fn persist_pending_turn_messages_for_model(
    app: &App,
    source_model: &str,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> bool {
    // One-shot mode does not persist by default: an ordinary one-off session is deleted
    // by cleanup_one_shot right after it ends, so persisting is just wasted I/O. But
    // background mode (a -bg) and one-shots with an explicit --session (e.g. a -ss <id> "q")
    // keep the session, so they must persist for later flows like /sessions titles and
    // /history to read the content.
    // Step 6: uniformly delegate to the store port (`DefaultHistoryStore`); the default
    // implementation is 100% identical to the old path, and source_model is pushed down
    // to the sqlite provenance column via `append_messages_for_model`.
    persist_pending_turn_messages_with_store(
        &app.session_history_file,
        source_model,
        one_shot_mode,
        app.cli.session.is_none(),
        turn_messages,
        persisted_turn_messages,
        &crate::ai::ports::DefaultHistoryStore,
    )
}
