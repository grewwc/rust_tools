use crate::ai::{
    history::Message,
    types::App,
};

/// 依赖倒置入口：允许通过 `HistoryStore` 注入历史存储实现（审计/加密/mock/测试）。
/// 保持与 `persist_pending_turn_messages` 相同的 `ephemeral` 与 `coalesce` 语义，
/// 仅将最终的 `append` 下沉到端口，便于中间件在不改 driver 行数的前提下插桩。
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
            let _ = crate::ai::history::coalesce_repeated_wait_wake_notes(
                history_file,
                first,
            );
        }
    }
    if let Err(err) = store.append_messages_for_model(
        history_file,
        &turn_messages[*persisted_turn_messages..],
        source_model,
    ) {
        // 追加失败仍保留与旧路径一致的 warning 文案；source_model 经端口下沉到
        // sqlite 溯源列，不再在此层丢弃。
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
    // Step 6：统一委托 store 端口（`DefaultHistoryStore`），默认实现与旧路径行为
    // 100% 一致；source_model 经 `append_messages_for_model` 下沉到 sqlite 溯源列。
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
