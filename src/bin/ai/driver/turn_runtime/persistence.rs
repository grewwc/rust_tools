use crate::ai::{
    history::{Message, append_history_messages_for_model},
    types::App,
};

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

/// 使用实际产出这批消息的模型写入 canonical 来源元数据。
///
/// 自动 fallback 不会改写 `app.current_model`，因此响应路径必须显式传入
/// provider 返回的实际模型；其余不含模型响应的中断路径可继续使用上面的默认入口。
pub(in crate::ai::driver::turn_runtime) fn persist_pending_turn_messages_for_model(
    app: &App,
    source_model: &str,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> bool {
    // one-shot 模式默认不落盘——普通一次性会话结束后会被 cleanup_one_shot
    // 立即删除，持久化只是无谓的 I/O。但后台模式（a -bg）以及显式指定
    // --session 的 one-shot（如 a -ss <id> "q"）会保留 session，必须落盘
    // 才能让后续 /sessions 的标题、/history 等查看流程读到内容。
    let ephemeral = one_shot_mode && app.cli.session.is_none();
    if ephemeral || *persisted_turn_messages >= turn_messages.len() {
        return true;
    }

    if let Err(err) = append_history_messages_for_model(
        &app.session_history_file,
        &turn_messages[*persisted_turn_messages..],
        source_model,
    ) {
        eprintln!("[Warning] Failed to save history: {}", err);
        return false;
    }

    *persisted_turn_messages = turn_messages.len();
    true
}
