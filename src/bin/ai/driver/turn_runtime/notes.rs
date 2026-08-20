// =============================================================================
// Model-Visible Notes
// =============================================================================
// Extracted from orchestrator.rs during a logic-preserving split.
// Injectors for model-visible loop / progress / checkpoint notes and force-final reason recording.
// =============================================================================

use super::*;

pub(super) fn inject_task_anchor_note(
    messages: &mut Vec<crate::ai::history::Message>,
    question: &str,
    iteration: usize,
    reason: &str,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let goal = truncate_chars(question.trim(), TASK_ANCHOR_MAX_QUESTION_CHARS);
    let note = format!(
        "[task-anchor] reason={reason}, iteration={iteration}.\nPrimary task goal: {goal}\n\
Keep goal continuity in mind:\n- First summarize the facts confirmed so far\n- State the single next action\n- If information is insufficient, describe the blocker and stop repeating tool calls"
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 工具循环检测命中后，向 messages 注入一条 internal_note 让 agent 自我反思
/// （而非直接 force_final，给 agent 一个跳出循环的机会）。
pub(super) fn inject_loop_breaker_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[loop-detected] You have called the same tool with the same arguments for the last 4 rounds; the earlier tool results are still in context, so repeating the call would produce no new information.\n\
        Do not call that same argument set again. Decide the next step from the existing evidence:\n\
        (a) If you have enough information, perform a substantive action or answer the user directly;\n\
        (b) If information is insufficient, pick only one different and concrete action (e.g. read a previously uncovered line range, search a new symbol/target, or modify a file);\n\
        (c) If you truly cannot proceed, state the single missing key piece of information and why.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

pub(super) fn inject_hard_loop_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[loop-hard-stop] Despite the repeat-call notice, you called the same tool with the same arguments for 6 consecutive rounds; this is judged an ineffective loop.\n\
        From now on you are in no-tool wrap-up mode: do not issue any more tool calls;\n\
        give a phase summary and current conclusion based on existing information; if the task is not yet complete, clearly state the gap, remaining work, and suggested next steps.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

pub(super) const TOOL_STOP_REASON_PREFIX: &str = "[runtime-tool-stop]";

/// 将进入无工具收口模式的首个根因仅写入当前 request context。
pub(in crate::ai::driver::turn_runtime) fn record_force_final_reason(
    messages: &mut Vec<crate::ai::history::Message>,
    reason: &str,
    iteration: usize,
    target: Option<&str>,
) {
    use crate::ai::history::{Message, ROLE_INTERNAL_NOTE};
    use serde_json::Value;

    if messages.iter().any(|message| {
        message.role == ROLE_INTERNAL_NOTE
            && message
                .content
                .as_str()
                .is_some_and(|content| content.starts_with(TOOL_STOP_REASON_PREFIX))
    }) {
        return;
    }

    // 持久化到决策日志（磁盘 JSONL）：no-tool-handoff 根因的事后可观测通道。决策日志
    // 是会话旁路记录、不进入模型上下文，因此不存在 internal note 被提升为 system 的
    // 重放问题；canonical turn_messages 里的控制状态仍只保留在本次请求投影中。
    crate::ai::driver::decision_log::log_runtime_stop(
        crate::ai::driver::decision_log::get_decision_log_store(),
        &crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
        crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
        reason,
        target,
        iteration,
    );

    let target_suffix = target.map(|t| format!(", target={t}")).unwrap_or_default();
    let event = Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!(
            "{TOOL_STOP_REASON_PREFIX} reason={reason}, iteration={iteration}, action=no_tool_handoff{target_suffix}"
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    // 运行时停止原因只属于本次请求投影；若写入 canonical turn_messages，下一轮会把
    // 过期的 no-tool-handoff 控制状态重新提升为 system 并永久重放。
    messages.push(event);
}

/// 近似低收益重复命中：同一工具反复命中同一目标资源（仅翻页/检索参数在变）。
/// 提醒 agent 判断这些调用是否真的在推进问题；若只是碎片化翻页则收敛，
/// 若各轮服务于不同且明确的子问题则允许继续。软提示，不强制收敛。
pub(super) fn inject_coarse_loop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-yield-repetition] You have been calling the same tool on the same target for several rounds, with the main variation being only paging/search-window parameters.\n\
        This often means low-yield repetition, but it is not necessarily an error: if the calls serve distinct and well-defined sub-questions, you may continue;\n\
        otherwise prefer: (a) reading a larger line range at once (raise read_file's limit) or locating with a search tool in one shot;\n\
        (b) reusing content you already read instead of re-reading the same file/segment; (c) if you already have enough information, answer directly.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 混合工具轮的目标级重复提示：同一目标被穿插在不同工具批次里反复取证。
/// 与 coarse 提示同级（温和、不阻断），但措辞强调「换工具查同一个东西」这一
/// 特定反模式，引导模型复用已读结果而非再换个工具重查一遍。
pub(super) fn inject_target_repeat_loop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-yield-repetition] For several rounds you have kept re-gathering evidence on the same target (the same file / the same search target),\n\
        merely switching tools or padding each round with different side calls to dodge the repetition — but you gained no new information.\n\
        Stop and do one thing: reuse what you already read/searched about that target instead of checking the same thing with another tool.\n\
        Then choose one: (a) if you have enough information, immediately take the next substantive action or answer directly;\n\
        (b) if you really must continue, write down exactly which new piece of information about that target is still missing and why switching tools would obtain it.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

    /// 同目标「从头重读」重扫软提示：同一文件被反复从文件头开始读（页宽每轮都变、
    /// 或混入每轮不同的新归档路径），已累计多次从头重读。
pub(super) fn inject_target_rescan_note(
    messages: &mut Vec<crate::ai::history::Message>,
    target: &str,
    reads: u32,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
            "[target-rescan] File `{target}` has been re-read from the beginning {reads} times within the recent window.\n\
            If you have already covered the full content, converge now and answer based on the evidence you have; if you still need more of that file, delegate the remaining exploration to a subagent with the exact file and range to inspect.\n\
            Re-reading the same range injects byte-identical content: it is suppressed/deduped and does NOT count as new progress. What you already read is still in this turn's context (or archived - see its preserved stub's `file_path`); do not re-read it from the top. Continue from the exact offset you last reached, or use `search_overflow` to locate the archived content.\n\
            If you really must re-read it yourself, write down exactly which new piece of information you expect to gain and why."
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

    /// 同目标「从头重读」重扫硬停止：同一文件从头重读超过硬阈值，判定为翻页+混轮
/// 循环，强制无工具收口。
pub(super) fn inject_target_rescan_hard_stop_note(
    messages: &mut Vec<crate::ai::history::Message>,
    target: &str,
    reads: u32,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
            "[low-yield-hard-stop] File `{target}` has been re-read from the beginning {reads} times within recent rounds; this is judged a pagination loop.\n\
            The content you already read is still available in this turn's context or archived (see preserved stubs' `file_path`); base your conclusion on it instead of re-reading.\n\
        From now on you are in no-tool wrap-up mode: do not issue any more tool calls;\n\
        give a phase summary and current conclusion based on existing information; if the task is not yet complete, clearly state the current gap, remaining work, and suggested next steps."
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 低收益的 `execute_command` 粗粒度重复升级到 hard-stop：在同一 coarse 目标上
/// 连续多轮只改窗口/排序细节，基本可判定为无效探索。
pub(super) fn inject_coarse_hard_loop_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-yield-hard-stop] You have repeatedly called `execute_command` on the same target for several rounds, varying mainly window/sort details; this is judged ineffective exploration.\n\
        From now on you are in no-tool wrap-up mode: do not issue any more tool calls;\n\
        give a phase summary and current conclusion based on existing information; if the task is not yet complete, clearly state the current gap, remaining work, and suggested next steps.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget 第一级（软反思）：连续多轮无可测信息增益。
/// 收敛类提示在“工具仍可继续”的阶段（soft / breadth / ledger）要求模型写下的
/// 账本 / 归纳属于**内部自省**，必须落在隐藏的 `<meta:self_note>…</meta:self_note>`
/// 通道里：流层 [`push_text_with_hidden_meta`](crate::ai::stream) 会把它从可见输出
/// 剥离、持久化成 internal_note，下一轮模型仍能读到。若不约束落点，模型会把这段
/// 中途反思写进面向用户的正文，被立即流式呈现成「预结论」，并与真正的最终答复
/// 重复（本次事故的直接成因）。硬停 / 迭代上限等 force-final 提示不套用本约束——
/// 那时正文就是最终答复。
pub(super) const SELF_NOTE_REFLECTION_CHANNEL_HINT: &str = "\n\
    Important (placement constraint): the ledger / summary asked for above is internal self-reflection; write it in full \
    between `<meta:self_note>` and `</meta:self_note>`; it is not shown to the user but stays in your subsequent context.\n\
    Keep the user-facing text of this round empty or limited to the next step you are continuing with; write a real final conclusion only when you are genuinely wrapping up.";

/// 反思式提示，不阻断工具——给模型解释「为什么还要继续同方向」和继续探索的权利。
pub(super) fn inject_low_progress_soft_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[low-progress-review] The runtime recently observed no new target, success-state change, or new tool-result content.\n\
        This is a heuristic check; it does not mean the work on the same target is necessarily ineffective, and do not drop necessary steps just because of this note.\n\
        Before calling a tool, confirm which missing piece of evidence the next call would add, and what result would end this branch.\n\
        If existing evidence is enough, run the narrowest verification and answer; if not, you may continue along the clearly stated gap.\
        {SELF_NOTE_REFLECTION_CHANNEL_HINT}"
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// ReadOnly 广度检查：新目标仍算信息增益；这里只在目标面过宽时提醒先归纳，
/// 不阻断工具，避免把大型排查任务误判为低进展。
pub(super) fn inject_read_only_breadth_note(
    messages: &mut Vec<crate::ai::history::Message>,
    agent_team_active: bool,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let team_guidance = if agent_team_active {
        "\nBecause `agent-team` is active, do not continue a broad serial sweep: delegate any remaining branches now (serial ones one at a time via the synchronous `task`, passing prior results), or state the concrete dependency that makes delegation unsafe."
    } else {
        ""
    };
    let note = format!(
        "[read-only-breadth-check] You have already covered many different target resources in read-only analysis,\n\
        which may be a necessary broad sweep, or may have slid from filling key evidence into endlessly expanding branches.\n\
        Tools remain available; but before continuing, write down in at most 6 lines:\n\
        1) confirmed facts (at most 3); 2) current conclusion or most likely explanation;\n\
        3) the single still-missing key evidence; 4) the single next tool action.\n\
        If you can already answer, give the conclusion directly instead of expanding the search surface just to re-confirm.{team_guidance} {SELF_NOTE_REFLECTION_CHANNEL_HINT}"
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget 第二级（记账）：软提示后仍无进展，要求写下轻量决策账本，
/// 让模型显式说明继续探索的依据。仍不硬阻断工具。
pub(super) fn inject_progress_ledger_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[low-progress-ledger] Within the response window after the previous phase check, the runtime still observed no new target,\n\
        success-state change, or new tool-result content. To continue, first write a decision ledger in at most 6 lines:\n\
        1) confirmed facts (bullets, at most 3)\n\
        2) the single key question still to resolve\n\
        3) candidate branches A / B and which you pick now, and why\n\
        4) the single next action based on the chosen branch\n\
        If the gap is clear, you may execute that action; if you cannot articulate a gap, wrap up on existing evidence.\
        {SELF_NOTE_REFLECTION_CHANNEL_HINT}"
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget 第三级（硬停）：软提示 + 记账后仍连续无进展，切换无工具收口。
pub(super) fn inject_low_progress_hard_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-progress-hard-stop] After soft notices, response windows, and the ledger, the runtime still observed no measurable progress.\n\
        To avoid burning more budget, you are now in no-tool wrap-up mode: do not issue any more tool calls;\n\
        give a phase conclusion based on the information gathered: what has been confirmed, what is still missing,\n\
        and the suggested next step to finish the task (for change tasks, state directly which files to modify and how).";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 分级、阶段感知的工具轮次检查点；它调度下一步，但不把刚完成的工具标成失败。
pub(super) fn inject_tool_round_checkpoint_note(
    messages: &mut Vec<crate::ai::history::Message>,
    iteration: usize,
    checkpoint: ToolRoundCheckpoint,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[tool-round-checkpoint] level={} phase={} round={iteration} threshold={}.\n\
        {}\n\
        {}\n\
        Checkpoint does not change delegation rules: do not hand off the current branch due to context or iteration pressure; delegate bounded sub-steps (serial or parallel) and review their results.",
        checkpoint.level.label(),
        checkpoint.phase.recent_progress(),
        checkpoint.threshold,
        checkpoint.level.guidance(),
        checkpoint.phase.guidance(),
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// max_iterations 触发后的自反思 prompt（替代纯 force_final 举手投降）。
pub(super) fn inject_iteration_limit_reflect_note(
    messages: &mut Vec<crate::ai::history::Message>,
    max_iterations: usize,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[iteration-limit] You have iterated {max_iterations} rounds without converging.\n\
        Answer the user directly with the information you have. If information is insufficient, clearly tell the user where you are stuck,\
        what material is missing, and a suggested next step — do not issue any more tool calls."
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 同步等待快到硬超时时，请子代理停止扩展新分支，优先交付可验证结论。
pub(super) fn inject_subagent_pre_timeout_wrap_up_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;

    let note = "[subagent-pre-timeout-wrap-up] The foreground wait time for the current synchronous sub-task is about to run out.\n\
        You are now in no-tool wrap-up mode: do not issue new tool calls or expand into new audit branches.\n\
        Immediately produce a final answer based on the evidence gathered: first list the verified conclusions;\n\
        separately mark risks that are not yet verified — never guess.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}
