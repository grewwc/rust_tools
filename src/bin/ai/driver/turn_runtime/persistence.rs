use crate::ai::{
    history::{Message, append_history_messages_uncompacted},
    types::App,
};

pub(in crate::ai::driver::turn_runtime) fn persist_pending_turn_messages(
    app: &App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) {
    if one_shot_mode || *persisted_turn_messages >= turn_messages.len() {
        return;
    }

    if let Err(err) = append_history_messages_uncompacted(
        &app.session_history_file,
        &turn_messages[*persisted_turn_messages..],
    ) {
        eprintln!("[Warning] Failed to save history: {}", err);
        return;
    }

    *persisted_turn_messages = turn_messages.len();
}
