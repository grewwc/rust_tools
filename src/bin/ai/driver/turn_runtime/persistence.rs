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
    // one-shot 模式默认不落盘——普通一次性会话结束后会被 cleanup_one_shot
    // 立即删除，持久化只是无谓的 I/O。但后台模式（a -bg）以及显式指定
    // --session 的 one-shot（如 a -ss <id> "q"）会保留 session，必须落盘
    // 才能让后续 /sessions 的标题、/history 等查看流程读到内容。
    let ephemeral = one_shot_mode && app.cli.session.is_none();
    if ephemeral || *persisted_turn_messages >= turn_messages.len() {
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
