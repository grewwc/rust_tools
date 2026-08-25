// =============================================================================
// Turn Orchestrator - Main Turn Execution Coordinator
// =============================================================================
// This module contains run_turn(), the main entry point for executing a single turn.
//
// Flow:
//   1. prepare_turn(): Build initial messages
//   2. Loop (max_iterations):
//        - Call LLM with current messages
//        - Execute any tool calls
//        - Handle results and add back to messages
//   3. finalize_turn(): Build final response
//   4. Return TurnOutcome (Quit, Success, or Error)
// =============================================================================

use std::io::Write;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ai::{history, mcp::SharedMcpClient, types::App};

use super::{
    CompressionReport, MID_TURN_COMPRESS_COOLDOWN_ITERATIONS, MID_TURN_COMPRESS_DELTA_THRESHOLD,
    MID_TURN_COMPRESS_SOFT_FLOOR, MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
    MID_TURN_LLM_SUMMARY_MAX_CHARS,
    finalize::finalize_turn,
    iteration::{execute_turn_iteration, refresh_skill_turn_for_iteration},
    mid_turn_compress_hard_threshold, mid_turn_compress_soft_threshold,
    persistence::persist_pending_turn_messages,
    prepare::prepare_turn,
    record_llm_summary_attempt_chars, should_try_llm_summary,
    tool_result::{
        completion_evidence_state, completion_tool_result_succeeded,
        handle_iteration_execution_for_model, tool_call_is_successful_mutation_candidate,
    },
    types::{IterationExecution, TurnLoopStep, TurnOutcome, TurnPreparation},
};

/// 工具调用循环检测窗口：
/// - soft: 连续 4 轮调用 (tool_name, normalized_args) 完全一致，先注入反思提示
/// - hard: 收到 soft 提示后仍连续 6 轮完全一致，直接强制收敛，不再继续工具循环
const TOOL_LOOP_SOFT_WINDOW: usize = 4;
const TOOL_LOOP_HARD_WINDOW: usize = 6;
/// 近似低收益重复窗口：连续 N 轮对「同一目标资源」调用同一工具（忽略
/// offset/limit 等翻页参数）即命中。用于抓字节精确检测漏掉的「同文件反复
/// 翻页 / 仅微调分页参数的重复检索」这类真实膨胀。先注入一次温和提示；
/// 若继续长时间刷同一 coarse 目标（尤其 `execute_command` 的目录探测类命令），
/// 则升级为 hard-stop，避免白跑上百轮。
const TOOL_LOOP_COARSE_WINDOW: usize = 5;
const TOOL_LOOP_COARSE_HARD_WINDOW: usize = 8;
const TOOL_SIGNATURE_HISTORY_LIMIT: usize = TOOL_LOOP_COARSE_HARD_WINDOW + 2;
/// 同目标「从头重读」重扫检测阈值。翻页+换页宽+混轮循环（同一文件反复从文件头
/// 开始读，offset/limit/页宽每轮都变，整轮签名永不相等）绕过了 exact/coarse/
/// target 三道整轮检测；这里按目标累计「从文件头开始读」的次数——第 3 次注入
/// 软提示，第 4 次硬停止。write_file/apply_patch 修改该目标会清零（编辑后从头
/// 重读验证是合法行为）。计数不随 soft 清空、不被 made_progress 重置。
// P1-2：同一文件连续 2 次「从头重读」即发软提示，尽早引导模型停止翻页式重读。
const TARGET_RESCAN_SOFT_THRESHOLD: u32 = 2;
const TARGET_RESCAN_HARD_THRESHOLD: u32 = 4;
/// 从头重读计数窗口（轮）：目标距上次从头读取超过该轮数时，计数视为过期并从 1
/// 重新累计。真正的翻页循环每轮/隔轮都从头读同一文件 → 计数不会过期、4 轮内必然
/// 触发；上下文压缩后跨多轮的合法重读 → 计数过期，不会累积到硬阈值。
const TARGET_RESCAN_WINDOW_ROUNDS: usize = 8;
const TASK_ANCHOR_MAX_QUESTION_CHARS: usize = 220;

/// 计算 coarse 签名时需剥离的「易变翻页/窗口」参数键。剥离后同一文件的不同
/// 分页、同一检索的不同结果上限会折叠成同一 coarse 签名。
const VOLATILE_ARG_KEYS: &[&str] = &["offset", "limit", "page", "cursor", "max_results"];

/// 工具轮次检查点的首个固定阈值。默认 turn 硬预算是 4096；24 / 48 / 96
/// 三档检查点只调度收敛、不禁用工具，累计轮次也不会因 mutation 清零。
const TOOL_ROUND_CHECKPOINT: usize = 24;
const TOOL_ROUND_CHECKPOINT_MULTIPLIERS: [usize; 3] = [1, 2, 4];

/// 连续「流读取中断型」截断（stream_error）的重试上限。超过即放弃本 turn，
/// 避免服务端持续断流时无限重试（尤其后台任务的 max_iterations = usize::MAX）。
const MAX_STREAM_ERROR_RETRIES: usize = 16;
/// 连续「模型输出过长 / 工具调用 JSON 半截」截断的重试上限。
/// stream_error 使用独立上限，不参与该计数。
const MAX_MODEL_TRUNCATION_RETRIES: usize = 3;
/// reasoning 占 completion tokens 的比例阈值：达到即判定为「推理吃光预算」型
/// 截断，此时降 reasoning_effort 能直接缩短思考链、把预算让给正文；低于该阈值
/// 说明截断主因是正文过长，降档无收益（模型没在推理），只注入收缩提示即可。
const REASONING_BUDGET_DOMINANCE_RATIO: f64 = 0.5;

/// === 长循环感知的中段压缩 ===
/// 中段压缩的软阈值按模型 token 窗口换算（flagship 256K → ~135K 字符）。对
/// 「历史体积中等、但工具迭代轮次很多」的长循环 turn，历史峰值可能长期低于该
/// 阈值 → 压缩全程不触发 → 每轮把「截至当前的完整历史 + 全部 tool schema」重发
/// 一遍，累计发送量随迭代轮次 O(n²) 膨胀，几分钟内撞破 TPM 限流（真实案例：
/// 一个 provider 重构会话单 turn 56 轮迭代，历史峰值仅 ~120K < 135K 阈值，
/// turn 内累计发送 ~2.8M token 撞破 380K TPM 约 7 倍）。
///
/// 治理：一旦单 turn 工具迭代轮次达到该阈值，即认定进入「长循环」，把中段压缩的
/// 有效软阈值下调到 [`MID_TURN_COMPRESS_SOFT_FLOOR`]（36K），让内容级去重
/// （byte-identical 重读折叠）与旧结果裁剪尽早介入，遏制 O(n²) 累积。短 turn
/// （迭代轮次未达阈值）保持原窗口比例阈值，不影响正常单轮大任务的探索空间。
const LONG_LOOP_COMPRESS_ITERATION_THRESHOLD: usize = 12;

/// === Progress Budget（信息增益进展预算）===
/// 这是叠加在 exact / coarse 循环检测之上的第三层，用于治理「参数每轮都变、
/// 但整体不推进任务」的发散型 loop——前两层按「签名重复」判定，结构上抓不到
/// 每轮都在搜新符号 / 读新文件却始终零收敛的膨胀（真实案例：一个「删除方法」
/// 的变更请求连续 60+ 轮只读取/检索、零 apply_patch）。
///
/// 核心理念：不按「动作次数」计费，按「信息增益」这一**行为信号**计费——本轮
/// 触碰到新目标资源（成功读取 / 检索到新目标），或调用了变更类工具，即算推进；
/// 失败调用（无目标）与反复取证同一目标都不算。不再从用户问题文本去猜任务意图。
/// 早期探索几乎免费；越往后，「继续但没进展」越要付出显式理由。惩罚对象是
/// 「说不出理由的无进展重复」，而非探索本身。
///
/// 免费探索轮数：达到该轮次前，即使连续无进展也完全不打扰（删代码前先定位、
/// 陌生代码库先摸索都是正常的）。
const PROGRESS_FREE_EXPLORE_ROUNDS: usize = 20;
/// 已触碰大量不同目标仍未收口，说明可能从「补关键证据」滑向「不断扩分支」。
/// 此阈值只注入一次非阻断式广度检查，不把新目标判成无进展，避免压缩大型排查
/// 任务的正当探索空间。
const READ_ONLY_BREADTH_CHECK_TARGETS: usize = 32;
/// 纯只读发散硬停阈值：本 turn 从未发生任何 mutation，却已累计触碰这么多个
/// **不同**的只读目标仍未收口。`seen_targets` 是单调集合、全程不被任何 reset
/// 清空，因此「累计不同只读目标数」与内容新颖度完全解耦——不会被「每轮换新目标 /
/// 读到新字节复位常规无进展计数」绕过（这正是常规 soft→hard 阶梯对纯只读发散
/// 失效的根因）。仅在零 mutation（纯调查）时作为最后一道刹车强制收口；任何
/// 「读+改」任务都不受影响。取 `READ_ONLY_BREADTH_CHECK_TARGETS` 的 3 倍，给大型
/// 只读排查留足空间，又远早于迭代硬上限。
const READ_ONLY_BREADTH_HARD_STOP_TARGETS: usize = 96;
/// 宽限窗口：软提示后，若模型给出了「实质不同的理由」（新目标 / reasoning 指纹
/// 变化），则在该窗口内不升级，给它继续探索的空间。
const PROGRESS_GRACE_WINDOW: usize = 6;
/// 一次低进展 episode 被真实进展打断后，至少间隔这么多轮才允许再次注入 soft。
/// 避免复杂任务在「探索 → 小进展 → 再探索」的正常节奏中反复收到同一收敛提示。
const PROGRESS_EPISODE_COOLDOWN: usize = 16;
/// 从「软提示 / 记账」升级到「硬停收口」额外需要的连续无进展轮数。
const PROGRESS_NO_PROGRESS_HARD_MARGIN: usize = 16;
/// scoped instruction preflight 使用独立预算，不消耗正常工具迭代；上限防止模型
/// 通过不断切换新目录无限延长单 turn。
const MAX_SCOPED_PREFLIGHT_GRACE_ROUNDS: usize = 8;

#[derive(Default)]
struct ScopedPreflightTargets {
    required: Vec<std::path::PathBuf>,
}

impl ScopedPreflightTargets {
    fn required(&self) -> &[std::path::PathBuf] {
        &self.required
    }

    fn record_pause(&mut self, targets: Vec<std::path::PathBuf>) {
        self.required = targets;
    }
}


// 子模块与 orchestrator.rs 同级存放。注意：crate 内另有同名模块 driver::thinking::orchestrator，
// 同名时 rustc 会把本模块的子模块按“目录式”路径解析（会在 orchestrator/ 目录下查找），
// 因此这里必须用 #[path] 显式固定到同级文件，否则报 E0583。

#[path = "checkpoint.rs"]
pub(crate) mod checkpoint;
#[path = "loop_detection.rs"]
mod loop_detection;
#[path = "notes.rs"]
mod notes;
#[path = "progress.rs"]
pub(crate) mod progress;

use checkpoint::*;
use loop_detection::*;
use notes::*;
use progress::*;
pub(super) use notes::record_force_final_reason;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[derive(Default)]
struct TurnSupervisor {
    iteration: usize,
    skip_tool_signature_rounds: usize,
    loop_breaker_injected: bool,
    hard_loop_stop_injected: bool,
    coarse_loop_note_injected: bool,
    next_tool_round_checkpoint_level: usize,
    iteration_limit_note_injected: bool,
    scoped_preflight_grace_rounds: usize,
    task_anchor_injected: bool,
    last_compress_iteration: usize,
    last_compress_after_chars: usize,
    /// 等待与下次 pre-request LLM 压缩结果合并输出的 mid-turn 状态。
    pending_compression_report: CompressionReport,
    tool_signature_history: Vec<Vec<String>>,
    tool_signature_history_coarse: Vec<Vec<String>>,
    /// 每轮触碰的「coarse 目标资源」集合历史（同 read_file 文件 /
    /// 同 execute_command coarse 命令，忽略翻页参数）。用于抓「整轮签名不
    /// 相等、但同一目标被混在不同工具批次里反复取证」的循环——纯整轮签名比较对
    /// 这类混合批次无能为力（每轮多一个陪衬工具即逃逸）。
    tool_target_history: Vec<Vec<String>>,
    target_repeat_note_injected: bool,
    progress: ProgressLedger,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolLoopSignal {
    None,
    /// 近似低收益重复：同一工具反复命中同一目标资源（忽略翻页参数）。温和提示一次。
    Coarse,
    /// 混合工具轮里同一目标资源被反复取证：整轮签名各不相等（每轮穿插不同陪衬
    /// 工具）逃过了 exact/coarse 整轮比较，但某个 read_file 文件
    /// 在窗口每一轮都出现。温和提示一次。
    TargetRepeat,
    /// 同一目标被反复「从头重读」（翻页页宽不断变化、混入每轮新归档路径的循环）。
    /// 温和提示一次。
    TargetRescan(String, u32),
    /// 同一目标从头重读超过硬阈值：切换无工具收口模式强制作答。
    TargetRescanHard(String, u32),
    /// `execute_command` 在同一 coarse 目标上长时间空转，直接强制收敛。
    CoarseHard,
    Soft,
    Hard,
    /// Progress Budget 第一级：连续多轮无信息增益（既无新目标也无实质动作），
    /// 注入反思式软提示，不阻断工具。
    LowProgressSoft,
    /// Progress Budget 第二级：软提示后仍无进展，要求写下轻量决策账本
    /// （已确认事实 / 待解决问题 / 候选与已排除分支），仍不硬阻断。
    LowProgressLedger,
    /// Progress Budget 第三级：软提示 + 记账后仍连续无进展，切换无工具收口模式。
    LowProgressHard,
    /// 已覆盖大量不同目标，提醒先汇总当前证据和唯一关键缺口。
    ReadOnlyBreadth,
    /// 纯只读发散最后一道刹车：本 turn 零 mutation，却已累计触碰远超广度阈值的
    /// 不同只读目标仍未收口。与内容新颖度解耦，切换无工具收口模式强制作答。
    ReadOnlyBreadthHard,
}

/// Progress Budget 的运行时状态。挂在 `TurnSupervisor` 上，按「信息增益」而非
/// 动作次数计费；只惩罚「说不出理由的无进展重复」。进展是**行为信号**：本轮
/// 触碰到新目标资源，或调用了变更类工具（`round_has_mutation`），即算推进；不再
/// 从用户问题文本去猜任务意图。
#[derive(Default)]
struct ProgressLedger {
    /// 累计触碰过的目标资源（「新目标 = 信息增益」判定）。
    seen_targets: FxHashSet<String>,
    /// 累计见过的成功只读工具结果。新内容即使来自同一目标，也属于新证据。
    seen_evidence_fingerprints: FxHashSet<u64>,
    /// 连续无进展轮数。任意一轮判定为 Progress 即清零。
    consecutive_no_progress: usize,
    /// 上一轮 reasoning 指纹（软提示后指纹变化 → 视为给出新理由 → grace 宽限）。
    last_reasoning_fp: Option<u64>,
    /// grace 宽限截止迭代号：在此之前不升级，给模型继续探索的空间。
    grace_until_iteration: usize,
    /// reasoning 变化每个 turn 最多换取一次 grace，避免靠逐轮改写理由无限续期。
    grace_consumed: bool,
    soft_injected: bool,
    ledger_injected: bool,
    hard_injected: bool,
    read_only_breadth_injected: bool,
    /// 本 turn 是否发生过项目 mutation。一旦为真，纯只读发散硬停永久让路
    /// （「读+改」任务不受该刹车约束）。plan / task 生命周期只算进展，不置位。
    /// 单调置位，不被 reset 清除。
    observed_project_mutation_this_turn: bool,
    /// 纯只读发散硬停的一次性门（`ReadOnlyBreadthHard` 只触发一次）。
    read_only_breadth_hard_injected: bool,
    /// 新的 low-progress episode 最早允许注入 soft 的迭代号。实质进展会重置当前
    /// episode，但保留该 cooldown，防止复杂任务被同一提示反复打断。
    next_episode_iteration: usize,
    /// 每个目标「从文件头开始读」的重扫计数（重扫检测）。元组第一位是窗口内
    /// 从头读的次数，第二位是上次从头读的 iteration（超过 TARGET_RESCAN_WINDOW_ROUNDS
    /// 轮未从头读则清零重计）。与签名历史解耦：soft 清空历史、每轮新目标重置无进展
    /// 计数，都影响不到它；本轮有任何 write_file/apply_patch 实质改动时整表清零
    /// （编辑后从头重读是合法行为，不能误判为翻页循环）。
    from_top_reads: FxHashMap<String, (u32, usize)>,
    /// 已注入过 TargetRescan 软提示的目标（每目标本轮一次）。
    rescan_note_injected: FxHashSet<String>,
}

impl ProgressLedger {
    /// 重置升级阶梯：清空无进展计数与 soft/ledger/hard/grace 等一次性状态，
    /// 让计费从零重新开始。两类场景共用：
    /// 1. 截断重试（mark_truncation_skip）：截断清空历史后，重复读取是预期行为，
    ///    与 exact/coarse 检测的 mark_truncation_skip 语义保持一致，避免截断恢复后的
    ///    新循环跳过 soft 提示直接到 hard-stop。
    /// 2. 实质进展（assess_progress 的 made_progress 分支）：软提示后模型给出真正推进
    ///    任务的动作，应视为「这一轮提醒生效了」，给予完整的新预算而非继续累加，否则
    ///    模型在长任务中只要早期发散过一次，后续每次收敛提醒都会更快滑向硬停。
    fn reset_escalation(&mut self) {
        self.consecutive_no_progress = 0;
        self.soft_injected = false;
        self.ledger_injected = false;
        self.hard_injected = false;
        self.grace_until_iteration = 0;
        self.grace_consumed = false;
    }

    /// 截断是外部约束，不应继承上一 episode 的提示冷却。
    fn reset_after_truncation(&mut self) {
        self.reset_escalation();
        self.next_episode_iteration = 0;
        // 截断恢复后从头重读属于合法的上下文重建：re-scan 计数与提示标记必须随历史一并清空，
        // 否则恢复后的首次从头重读会继承旧计数直接触发 hard-stop。
        // 注意不能并入 reset_escalation：实质进展路径不得清空该计数（混轮新目标会不断重置
        // 无进展计数，re-scan 计数正是针对该逃逸的独立防线）。
        self.from_top_reads.clear();
        self.rescan_note_injected.clear();
    }
}

impl TurnSupervisor {
    fn next_iteration(&mut self) -> usize {
        self.iteration = self.iteration.saturating_add(1);
        self.iteration
    }

    fn grant_scoped_preflight_grace(&mut self) -> bool {
        if self.scoped_preflight_grace_rounds >= MAX_SCOPED_PREFLIGHT_GRACE_ROUNDS {
            return false;
        }
        self.scoped_preflight_grace_rounds += 1;
        true
    }

    fn effective_max_iterations(&self, max_iterations: usize) -> usize {
        max_iterations.saturating_add(self.scoped_preflight_grace_rounds)
    }

    fn should_try_mid_turn_compress(&self, total_chars: usize, soft_threshold: usize) -> bool {
        let cooldown_passed = self.iteration.saturating_sub(self.last_compress_iteration)
            >= MID_TURN_COMPRESS_COOLDOWN_ITERATIONS;
        let delta_significant = total_chars.saturating_sub(self.last_compress_after_chars)
            >= MID_TURN_COMPRESS_DELTA_THRESHOLD;
        total_chars > soft_threshold
            && cooldown_passed
            && (self.last_compress_after_chars == 0 || delta_significant)
    }

    /// 本轮实际生效的中段压缩软阈值。
    ///
    /// 长循环（工具迭代轮次 >= [`LONG_LOOP_COMPRESS_ITERATION_THRESHOLD`]）时把
    /// 阈值下调到 [`MID_TURN_COMPRESS_SOFT_FLOOR`]，让内容级去重与旧结果裁剪尽早
    /// 介入，遏制 O(n²) 累积重发；短 turn 保持按窗口换算的基准阈值，不影响正常
    /// 单轮大任务。门控与实际 [`mid_turn_compress`](crate::ai::history::mid_turn_compress)
    /// 调用必须共用本方法返回值——后者内部有 `before <= soft_threshold` 的 no-op
    /// 早退，若两处阈值不一致会「门开了却压不动」。
    fn effective_mid_turn_soft_threshold(&self, base_soft: usize) -> usize {
        if self.iteration >= LONG_LOOP_COMPRESS_ITERATION_THRESHOLD {
            base_soft.min(MID_TURN_COMPRESS_SOFT_FLOOR)
        } else {
            base_soft
        }
    }

    fn mark_compress(&mut self, after_chars: usize) {
        self.last_compress_iteration = self.iteration;
        self.last_compress_after_chars = after_chars;
    }

    /// 任务出现实质进展后，丢弃此前无效循环的样本并恢复 soft → hard 升级阶梯。
    /// 否则模型已经响应 soft 提示而改做有效动作时，后续一次新的重复会跳过 soft，
    /// 直接沿用旧标志进入 hard-stop。
    fn reset_tool_loop_escalation(&mut self) {
        self.tool_signature_history.clear();
        self.tool_signature_history_coarse.clear();
        self.tool_target_history.clear();
        self.hard_loop_stop_injected = false;
        self.loop_breaker_injected = false;
        self.coarse_loop_note_injected = false;
        self.target_repeat_note_injected = false;
    }

    /// 截断重试时重置工具循环检测状态：截断是外部约束（输出上限 / 模型可用性波动），
    /// 重试时重复调用相同工具属于预期行为，不应计入循环检测窗口。
    /// 截断本身已有独立的 `consecutive_truncations` 上限兜底，不需要循环检测再叠加。
    ///
    /// 清空历史 + 跳过当前迭代的签名记录，使截断重试不被误判为循环。
    /// 重置所有一次性标志：截断清空历史后，soft/coarse/hard 的完整升级阶梯
    /// 应从零重新开始，否则截断恢复后形成的新循环会跳过 soft 提示直接到 hard-stop。
    fn mark_truncation_skip(&mut self) {
        self.reset_tool_loop_escalation();
        self.skip_tool_signature_rounds += 1;
        self.progress.reset_after_truncation();
    }

    fn record_tool_signatures(
        &mut self,
        messages: &[crate::ai::history::Message],
        free_explore_rounds: usize,
    ) -> ToolLoopSignal {
        self.record_tool_signatures_for_progress(messages, messages, free_explore_rounds)
    }

    fn record_tool_signatures_for_progress(
        &mut self,
        messages: &[crate::ai::history::Message],
        progress_messages: &[crate::ai::history::Message],
        free_explore_rounds: usize,
    ) -> ToolLoopSignal {
        let signature_messages = if progress_messages.is_empty() {
            messages
        } else {
            progress_messages
        };
        // 截断重试跳过：清空历史后不记录本轮签名，避免截断重试
        // 被误判为工具循环。`skip_tool_signature_rounds` 由
        // `mark_truncation_skip()` 递增，每跳过一次递减。
        if self.skip_tool_signature_rounds > 0 {
            self.skip_tool_signature_rounds -= 1;
            return ToolLoopSignal::None;
        }
        let Some(sigs) = extract_round_tool_signatures(signature_messages) else {
            return ToolLoopSignal::None;
        };
        self.tool_signature_history.push(sigs);
        if self.tool_signature_history.len() > TOOL_SIGNATURE_HISTORY_LIMIT {
            let drop = self.tool_signature_history.len() - TOOL_SIGNATURE_HISTORY_LIMIT;
            self.tool_signature_history.drain(0..drop);
        }
        if let Some(coarse) = extract_round_tool_signatures_coarse(signature_messages) {
            self.tool_signature_history_coarse.push(coarse);
            if self.tool_signature_history_coarse.len() > TOOL_SIGNATURE_HISTORY_LIMIT {
                let drop = self.tool_signature_history_coarse.len() - TOOL_SIGNATURE_HISTORY_LIMIT;
                self.tool_signature_history_coarse.drain(0..drop);
            }
        }
        // 目标级历史：与 coarse 签名平行维护，供混合工具轮的目标交集检测使用。
        // 与 exact/coarse 一样受 TOOL_SIGNATURE_HISTORY_LIMIT 约束。
        self.tool_target_history
            .push(extract_round_probe_targets(signature_messages));
        if self.tool_target_history.len() > TOOL_SIGNATURE_HISTORY_LIMIT {
            let drop = self.tool_target_history.len() - TOOL_SIGNATURE_HISTORY_LIMIT;
            self.tool_target_history.drain(0..drop);
        }
        if !self.hard_loop_stop_injected
            && detect_tool_loop(&self.tool_signature_history, TOOL_LOOP_HARD_WINDOW)
        {
            self.hard_loop_stop_injected = true;
            return ToolLoopSignal::Hard;
        }
        if !self.loop_breaker_injected
            && detect_tool_loop(&self.tool_signature_history, TOOL_LOOP_SOFT_WINDOW)
        {
            self.loop_breaker_injected = true;
            // Soft 提示已明确要求停止重复调用。清空此前用于触发 soft 的样本，
            // 让模型有完整的 hard window 来响应提示，而不是只再重复两轮就被强制
            // 收口（旧逻辑中 soft=4、hard=6，实际恢复窗口只有两轮）。
            self.tool_signature_history.clear();
            // 设计洞补位：soft 只清了 exact 样本，但 coarse/target 历史未清，且它们
            // 的一次性门（coarse_loop_note_injected / target_repeat_note_injected）
            // 被下方 `!loop_breaker_injected` 永久挡死——模型 soft 后换一批参数继续
            // 翻同一目标时，coarse/target 再也无法触发，直到迭代上限才被收口。这里把
            // 两个门重新武装并清空对应历史，让「soft 后换姿势」的翻页 / 混合轮循环仍
            // 能被后续 coarse/target 捕获（提示仍是一次性、soft 优先级，不额外加压）。
            self.coarse_loop_note_injected = false;
            self.target_repeat_note_injected = false;
            self.tool_signature_history_coarse.clear();
            self.tool_target_history.clear();
            return ToolLoopSignal::Soft;
        }
        if !self.hard_loop_stop_injected
            && detect_execute_command_coarse_loop(
                &self.tool_signature_history_coarse,
                TOOL_LOOP_COARSE_HARD_WINDOW,
            )
        {
            self.hard_loop_stop_injected = true;
            return ToolLoopSignal::CoarseHard;
        }
        // 字节精确的 soft/hard 均未命中时，再看粗粒度：同一目标资源反复翻页/微调
        // 检索参数的膨胀会在这里被抓到。仅提示一次，且让位于精确检测。
        if !self.coarse_loop_note_injected
            && detect_tool_loop(&self.tool_signature_history_coarse, TOOL_LOOP_COARSE_WINDOW)
        {
            self.coarse_loop_note_injected = true;
            return ToolLoopSignal::Coarse;
        }
        // 整轮签名（exact/coarse）都要求整轮集合相等，抓不到「同一目标混在不同工具
        // 批次里反复取证」的混合轮循环。这里用目标交集补位：窗口内每轮都触碰同一
        // 目标即命中。让位于上面所有整轮检测，且与 coarse 一样只提示一次。
        if !self.target_repeat_note_injected
            && !self.coarse_loop_note_injected
            && detect_target_repeat_loop(&self.tool_target_history, TOOL_LOOP_COARSE_WINDOW)
        {
            self.target_repeat_note_injected = true;
            return ToolLoopSignal::TargetRepeat;
        }
        // 同目标「从头重读」重扫检测：翻页+换页宽+混轮循环的最终防线。上面三道
        // 整轮检测都要求「整轮签名集合相等/每轮共现同一目标」，而这类循环每轮
        // offset/limit/页宽都在变、还混着每轮不同的新归档路径与 execute_command，
        // 整轮集合永远不相等；soft 命中后历史还会被清空、每轮新目标又把无进展
        // 计数重置。这里按「目标文件」累计从文件头开始的读取次数（cat/head/
        // tail -c +1/read_file offset=1 …），不依赖整轮签名集合——页宽怎么换、
        // 混了多少新目标都影响不到计数。write_file/apply_patch 修改该文件时
        // 清零（编辑后从头重读验证是合法行为）。软提示一次后继续累计，第 4 次
        // 从头重读直接硬停止。
        // 本轮有实质文件改动（write_file/apply_patch，任意目标）→ 编辑后的验证性
        // 重读是合法行为，整表清空 rescan 计数，避免「读 A 改 B」的多文件工作流被
        // 误判为翻页循环。注意不能用 round_has_mutation：它把 sed/awk 等非只读
        // execute_command 也当 mutation，会把 sed 只读重读误判为「实质改动」而架空
        // rescan 检测（command_reads_from_top 反而把 sed -n 1 当作从头读）。
        if !extract_round_mutated_targets(signature_messages).is_empty() {
            self.progress.from_top_reads.clear();
            self.progress.rescan_note_injected.clear();
        }
        let from_top_targets = extract_round_from_top_read_targets(signature_messages);
        let mut rescan_signal: Option<ToolLoopSignal> = None;
        for target in from_top_targets {
            let entry = self
                .progress
                .from_top_reads
                .entry(target.clone())
                .or_insert((0, self.iteration));
            // 窗口过期：距上次从头读取超过 WINDOW 轮 → 旧计数作废，重新累计。
            if self.iteration.saturating_sub(entry.1) > TARGET_RESCAN_WINDOW_ROUNDS {
                *entry = (0, self.iteration);
                // 衰减 = 进入新的重读 episode：同步清除该目标的软提示标记，让每一段
                // 循环都能拿到自己的 soft 预警。否则 rescan_note_injected 保留上一段的
                // 标记，第二段累计到 soft 时 insert 返回 false → 跳过软提示直接硬停，
                // 与 soft→hard 升级不变量冲突（mutation/截断清空路径都成对清除）。
                self.progress.rescan_note_injected.remove(&target);
            }
            entry.0 += 1;
            entry.1 = self.iteration;
            let reads = entry.0;
            if reads >= TARGET_RESCAN_HARD_THRESHOLD {
                rescan_signal = Some(ToolLoopSignal::TargetRescanHard(target, reads));
                break;
            }
            if reads >= TARGET_RESCAN_SOFT_THRESHOLD
                && self.progress.rescan_note_injected.insert(target.clone())
            {
                rescan_signal = Some(ToolLoopSignal::TargetRescan(target, reads));
            }
        }
        if let Some(signal) = rescan_signal {
            return signal;
        }
        // exact/coarse 均未命中「签名重复」型循环时，交给 Progress Budget 补位：
        // 抓「参数每轮都变、但整体不推进任务」的发散型 loop。
        self.assess_progress(messages, progress_messages, free_explore_rounds)
    }

    /// Progress Budget 判定：按「信息增益」而非动作次数计费。只在 exact/coarse
    /// 签名检测未命中时补位调用。进展是纯**行为信号**，不再从问题文本猜意图：
    ///
    /// - 本轮触碰到新目标资源（`extract_round_targets` 首次出现）→ 信息增益，算进展；
    /// - 成功只读工具返回了此前未见的新内容 → 新证据，算进展；
    /// - 或本轮调用了变更类工具（`round_has_mutation`）→ 实质动作，算进展。
    ///
    /// 免费探索区（iteration <= free_explore_rounds）内完全不计费；退出后按稳定
    /// 阈值升级：软提示 → 固定响应窗口 → 记账 → 硬停。soft episode 之间还有 turn 内
    /// cooldown，避免复杂任务在正常的探索/推进节奏中反复收到同一提示。
    fn assess_progress(
        &mut self,
        messages: &[crate::ai::history::Message],
        progress_messages: &[crate::ai::history::Message],
        free_explore_rounds: usize,
    ) -> ToolLoopSignal {
        // 三类进展信号分开保留：ReadOnlyBreadth 只由新目标触发；调用模式历史只在
        // 新目标/变更动作后清空。同一目标返回新内容时只重置 Progress Budget，保留
        // exact/coarse detector 对重复翻页模式的一次性提醒能力。
        let round_had_mutation = round_has_mutation(progress_messages);
        let round_had_project_mutation = round_has_project_mutation(progress_messages);
        let mut added_new_target = false;
        for t in extract_round_targets(progress_messages) {
            if self.progress.seen_targets.insert(t) {
                added_new_target = true;
            }
        }
        let mut added_new_evidence = false;
        for fingerprint in extract_round_evidence_fingerprints(progress_messages) {
            if self.progress.seen_evidence_fingerprints.insert(fingerprint) {
                added_new_evidence = true;
            }
        }
        let made_progress = round_had_mutation || added_new_target || added_new_evidence;

        let reasoning_fp = extract_round_reasoning_fingerprint(progress_messages)
            .or_else(|| extract_round_reasoning_fingerprint(messages));
        if made_progress {
            // 任意真实进展都结束当前 low-progress episode；next_episode_iteration 保留，
            // 因而短暂推进后不会立刻再次注入 soft。seen targets/evidence 也跨轮累积。
            let recovered_from_low_progress_episode = self.progress.soft_injected
                || self.progress.ledger_injected
                || self.progress.hard_injected
                || self.progress.grace_consumed
                || self.progress.grace_until_iteration > 0;
            if recovered_from_low_progress_episode {
                self.progress.next_episode_iteration = self
                    .progress
                    .next_episode_iteration
                    .max(self.iteration + PROGRESS_EPISODE_COOLDOWN);
                self.progress.reset_escalation();
            }
            // 只有**结构性进展**（新目标 / 变更动作）且此前已提示过 loop 时，才清空
            // exact/coarse/target 重试窗口。新的只读结果内容已计入上面的 made_progress
            // （Progress Budget 不会因高效翻页误升级），但**绝不**清空签名/目标历史：
            // 否则同文件反复翻页会因每页内容不同而永远填不满 coarse 窗口，绕过循环刹车
            // ——这正是「进展哈希须忽略 offset/limit 以防预算逃逸」不变量要防的情况。
            if (round_had_mutation || added_new_target)
                && (self.hard_loop_stop_injected
                    || self.loop_breaker_injected
                    || self.coarse_loop_note_injected
                    || self.target_repeat_note_injected)
            {
                self.reset_tool_loop_escalation();
            }
            self.progress.consecutive_no_progress = 0;
            self.progress.last_reasoning_fp = reasoning_fp;
            // 只有真实项目变更才永久关闭纯只读发散硬停。plan / task 生命周期虽是任务
            // 进展，但不能让后续无限串行扫描绕过这个最终刹车。
            if round_had_project_mutation {
                self.progress.observed_project_mutation_this_turn = true;
            }
            // 纯只读发散最后一道刹车：本 turn 从未修改项目，却已累计触碰远超广度
            // 阈值的不同只读目标仍未收口。`seen_targets` 单调、不被任何 reset 清空，
            // 因此这条判据与「新目标/新字节复位常规无进展计数」完全解耦，是唯一能
            // 兜住「持续换目标的纯调查发散」的信号。让位于普通 breadth 提示之后，
            // 只触发一次并强制收口。
            if !self.progress.read_only_breadth_hard_injected
                && !self.progress.observed_project_mutation_this_turn
                && self.iteration > free_explore_rounds
                && self.progress.seen_targets.len() >= READ_ONLY_BREADTH_HARD_STOP_TARGETS
            {
                self.progress.read_only_breadth_hard_injected = true;
                return ToolLoopSignal::ReadOnlyBreadthHard;
            }
            if !self.progress.read_only_breadth_injected
                && !round_had_mutation
                && added_new_target
                && self.iteration > free_explore_rounds
                && self.progress.seen_targets.len() >= READ_ONLY_BREADTH_CHECK_TARGETS
            {
                self.progress.read_only_breadth_injected = true;
                return ToolLoopSignal::ReadOnlyBreadth;
            }
            return ToolLoopSignal::None;
        }

        // 免费探索区：探索完全免费，不计费也不升级（删代码前先定位、陌生代码库
        // 先摸索都属正常）。仅更新 reasoning 指纹基线。
        if self.iteration <= free_explore_rounds {
            self.progress.last_reasoning_fp = reasoning_fp;
            return ToolLoopSignal::None;
        }

        self.progress.consecutive_no_progress += 1;

        // 每次 soft 后都给固定响应窗口。旧逻辑只有暴露 reasoning_content 且指纹变化
        // 的模型才有 grace；不输出 reasoning 的模型会在下一轮立刻收到 ledger。基础
        // 窗口结束后，新理由仍可额外延长一次，但不得滚动续期。
        let reasoning_changed =
            reasoning_fp.is_some() && reasoning_fp != self.progress.last_reasoning_fp;
        self.progress.last_reasoning_fp = reasoning_fp;
        if self.iteration < self.progress.grace_until_iteration {
            return ToolLoopSignal::None;
        }
        if self.progress.soft_injected && reasoning_changed && !self.progress.grace_consumed {
            self.progress.grace_until_iteration = self.iteration + PROGRESS_GRACE_WINDOW;
            self.progress.grace_consumed = true;
            return ToolLoopSignal::None;
        }

        let soft_threshold = no_progress_soft_threshold(self.iteration, free_explore_rounds);
        if self.progress.consecutive_no_progress < soft_threshold {
            return ToolLoopSignal::None;
        }

        // 升级阶梯严格按 软提示 → 记账 → 硬停 推进，每级一次性。硬停额外要求
        // 连续无进展达到 soft_threshold + margin，避免越过软层直接收口。
        if !self.progress.soft_injected {
            if self.iteration < self.progress.next_episode_iteration {
                return ToolLoopSignal::None;
            }
            self.progress.soft_injected = true;
            self.progress.grace_until_iteration = self.iteration + PROGRESS_GRACE_WINDOW;
            self.progress.next_episode_iteration = self.iteration + PROGRESS_EPISODE_COOLDOWN;
            return ToolLoopSignal::LowProgressSoft;
        }
        if !self.progress.ledger_injected {
            self.progress.ledger_injected = true;
            return ToolLoopSignal::LowProgressLedger;
        }
        let hard_threshold = soft_threshold + PROGRESS_NO_PROGRESS_HARD_MARGIN;
        if self.progress.consecutive_no_progress >= hard_threshold && !self.progress.hard_injected {
            self.progress.hard_injected = true;
            return ToolLoopSignal::LowProgressHard;
        }
        ToolLoopSignal::None
    }

    /// 分级工具轮次检查点：保留累计轮次，在 24 / 48 / 96（小预算按首档缩放）
    /// 根据当前工作阶段注入不同调度提示。
    fn maybe_inject_tool_round_checkpoint(
        &mut self,
        messages: &mut Vec<crate::ai::history::Message>,
        max_iterations: usize,
        phase: ToolRoundCheckpointPhase,
    ) -> Option<ToolRoundCheckpoint> {
        let level_index = self.next_tool_round_checkpoint_level;
        let threshold = tool_round_checkpoint_threshold(max_iterations, level_index)?;
        if self.iteration < threshold {
            return None;
        }
        self.next_tool_round_checkpoint_level += 1;
        let checkpoint = ToolRoundCheckpoint {
            level: ToolRoundCheckpointLevel::from_index(level_index),
            phase,
            threshold,
        };
        inject_tool_round_checkpoint_note(messages, self.iteration, checkpoint);
        Some(checkpoint)
    }

    fn maybe_inject_iteration_limit_note(
        &mut self,
        messages: &mut Vec<crate::ai::history::Message>,
        max_iterations: usize,
        force_final_response: bool,
    ) {
        if force_final_response && !self.iteration_limit_note_injected {
            self.iteration_limit_note_injected = true;
            inject_iteration_limit_reflect_note(messages, max_iterations);
        }
    }

    fn maybe_inject_task_anchor(
        &mut self,
        messages: &mut Vec<crate::ai::history::Message>,
        question: &str,
        reason: &str,
    ) {
        if self.task_anchor_injected {
            return;
        }
        self.task_anchor_injected = true;
        inject_task_anchor_note(messages, question, self.iteration, reason);
    }
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "A",
    "turn_runtime::run_turn",
    "[DEBUG] run_turn started",
    "[DEBUG] run_turn finished",
    {
        "history_count": history_count,
        "question_len": question.chars().count(),
        "model": next_model.as_str(),
        "one_shot_mode": one_shot_mode,
        "should_quit": should_quit,
    },
    {
        "ok": __agent_hang_result.is_ok(),
        "outcome": __agent_hang_result
            .as_ref()
            .map(|v| format!("{:?}", v))
            .unwrap_or_else(|err| err.to_string()),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(in crate::ai::driver) async fn run_turn(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    question: String,
    attachments_text: String,
    next_model: String,
    precomputed_ocr: Option<crate::ai::driver::model::OcrExtraction>,
    one_shot_mode: bool,
    should_quit: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    // Step 3：turn 起点钩子（保留原配对语义，必须在 /audit 短路之前触发）
    // 先分配真实 turn 身份（session SQLite 原子递增）再触发起点钩子：
    // 让 on_turn_start / on_turn_end 读到真实 turn_index，而非伪造的 0。
    // 分配失败 = 本轮未开始，直接返回，不触发任何生命周期钩子（起点/终点配对保持）。
    let turn_index = history::reserve_turn_index(&app.session_history_file)?;
    app.fire_turn_start_hooks(turn_index);

    let result = (async {
        // `/audit` 是用户直接请求的同步子代理调用。必须在父 DRIVER_CTX 已建立、
        // 子 agent 尚未进入递归 turn 前处理，才能复用 task 的隔离与证据生命周期。
        if crate::ai::driver::runtime_ctx::current_subagent_depth() == 0 {
            if let Some(command) = crate::ai::driver::commands::audit::parse_audit_command(&question) {
                return Ok(execute_audit_command(app, command, should_quit));
            }
        }
        // 把 (session_id, turn_id) 注入 task_local，让下游工具调用与反馈
        // 写入路径能拿到正确身份。turn_id 由 session SQLite 原子分配，包含普通、
        // resume 和 internal turn，跨重启/多进程也不会重复。
        let session_id = app.session_id.clone();
        let turn_id = turn_index;
        // 仅前台主 turn 抬起「turn 活动」标志：子 agent（sync / background）持有私有
        // 信号标志，且都通过 SUBAGENT_RESULT_SLOT 作用域执行，据此排除。该标志让
        // prepare / 思考 / 阶段切换 / mid-turn 压缩等 streaming=false 的空窗里的
        // Ctrl+C 也走「取消本轮」而非「退出会话」。guard 随本 future drop 自动落下。
        let _foreground_turn_guard = (!crate::ai::driver::runtime_ctx::has_subagent_result_slot())
            .then(crate::ai::driver::signal::ForegroundTurnGuard::enter);
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .scope((session_id, turn_id), async {
                // enable_tools 的 per-turn 状态必须跟随整个 future，而不能只依赖
                // run_turn_body 的 happy-path 尾部清理；abort / early return 也会 Drop。
                let _enable_turn_guard = crate::ai::tools::enable_tools::EnableTurnStateGuard::enter();
                // 整个 turn（prepare / 流式 / thinking / 工具执行 / finalize）期间开启
                // 同终端 side-note 输入监听（Ctrl+G）。RAII 守卫随本 future 结束 drop，
                // 恢复 canonical 终端，保证 turn 之间 prompt_user 输入框不受残留 cbreak
                // 影响。子 agent（depth>0）/ 后台（终端输出抑制）/ 非 tty stdin 由
                // side_note_input_enabled() 排除；turn 内无其它 stdin 消费者（输入框只在
                // turn 之间打开，request_user_input 仅置标记不读 stdin），cbreak 全程接管
                // stdin 无冲突。
                let _side_note_input =
                    if crate::ai::stream::side_note_input::side_note_input_enabled() {
                        Some(crate::ai::stream::side_note_input::SideNoteInputGuard::spawn(
                            app.session_history_file.clone(),
                        ))
                    } else {
                        None
                    };
                run_turn_body(
                    app,
                    mcp_client,
                    skill_manifests,
                    history_count,
                    turn_index,
                    question,
                    attachments_text,
                    next_model,
                    precomputed_ocr,
                    one_shot_mode,
                    should_quit,
                )
                .await
            })
            .await
    })
    .await;

    // Step 3：turn 终点钩子。覆盖所有返回路径（含 /audit 短路），与起点配对。
    app.fire_turn_end_hooks(turn_index);

    result
}

fn execute_audit_command(
    app: &App,
    command: crate::ai::driver::commands::audit::AuditCommand,
    should_quit: bool,
) -> TurnOutcome {
    match command {
        crate::ai::driver::commands::audit::AuditCommand::Usage => {
            println!("Usage: /audit [--fast] <instruction>");
            println!("  --fast  快速审计：当前会话模型 + high 思考 + 更少步数 + 更短超时，适合轻量复查");
        }
        crate::ai::driver::commands::audit::AuditCommand::Run { instruction, fast } => {
            // 默认只继承 cwd/skills，避免把无关的父对话和 memory 带入审计任务。
            // 但子代理完全看不到父对话，必须显式告知 main agent 当前改了什么：
            // 经常多个需求并行改动，子代理只有看到当前工作区 diff 才能判断哪些属于本次审计。
            let prompt = crate::ai::driver::commands::audit::build_audit_prompt(&instruction);
            let description = if fast {
                format!("/audit --fast {instruction}")
            } else {
                format!("/audit {instruction}")
            };
            // fast 模式：audit-fast agent（更少步数）承载轻量审计契约，模型固定用
            // 当前会话模型（绕过 prompt 难度分类对 tier 的自动抬升），thinking 固定
            // high；超时同步缩短。
            let agent_name = if fast { "audit-fast" } else { "audit" };
            let (hard_timeout, wrap_up_lead_time) = if fast {
                (
                    crate::ai::driver::commands::audit::FAST_AUDIT_SUBAGENT_HARD_TIMEOUT,
                    Some(crate::ai::driver::commands::audit::FAST_AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME),
                )
            } else {
                (
                    crate::ai::driver::commands::audit::AUDIT_SUBAGENT_HARD_TIMEOUT,
                    Some(crate::ai::driver::commands::audit::AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME),
                )
            };
            let mut args = serde_json::json!({
                "description": description,
                "prompt": prompt,
                "agent": agent_name,
            });
            if fast {
                args["model"] = serde_json::Value::String(app.current_model.clone());
                args["reasoning_effort"] = serde_json::Value::String("high".to_string());
            }
            match crate::ai::driver::tools::execute_direct_subagent_task(
                "slash-audit",
                &args,
                hard_timeout,
                wrap_up_lead_time,
            ) {
                Ok(result) => println!(
                    "\n{}",
                    crate::ai::driver::commands::audit::terminal_audit_result(&result.content)
                ),
                Err(error) => println!("\n[audit] Unable to start audit subagent: {error}"),
            }
        }
    }

    if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_turn_body(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    turn_index: usize,
    question: String,
    attachments_text: String,
    next_model: String,
    precomputed_ocr: Option<crate::ai::driver::model::OcrExtraction>,
    one_shot_mode: bool,
    should_quit: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    // 每轮开始清除上一轮的打断标记，确保它只反映「本轮」是否被 Ctrl+C 打断。
    app.last_turn_interrupted = false;
    // 请求用户输入是工具层到 driver 的 turn 级侧信道；先清除遗留状态，避免异常退出
    // 的上一轮把续接误带入当前 turn。
    crate::ai::tools::skill_tools::clear_pending_user_input_request();
    // reasoning items 侧信道是 turn 级内存态：每轮开始清空，避免上一轮的
    // encrypted reasoning 泄漏到本轮请求（call_id 也不会再匹配）。
    app.turn_reasoning_items.clear();
    let TurnPreparation {
        mut skill_turn,
        mut messages,
        mut turn_messages,
        mut persisted_turn_messages,
        max_iterations,
    } = match prepare_turn(
        app,
        mcp_client,
        skill_manifests,
        history_count,
        turn_index,
        &question,
        &attachments_text,
        &next_model,
        precomputed_ocr,
    )
    .await
    {
        Ok(prep) => prep,
        Err(err) => return Err(err),
    };

    persist_pending_turn_messages(
        app,
        one_shot_mode,
        &turn_messages,
        &mut persisted_turn_messages,
    );

    // 历史预算压缩由 prepare_turn 统一负责（按 history_max_chars 构建带摘要/溢出指针的
    // 投影），mid-turn 预算路径负责长轮次兜底；此处不再挂第二遍 pipeline（无论真压还是
    // observe-only 都会引入每 turn 的 clone+推演开销，且真压第二遍会丢不可恢复的上下文）。
    // pipeline 模块仍保留为可选的测试/观测基建，需要时在调用方按 stage 显式挂载。

    let mut supervisor = TurnSupervisor::default();
    let mut force_final_response = false;
    let mut pre_timeout_wrap_up_requested = false;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut final_response_model = None::<String>;
    let mut terminal_dedupe_candidate = None;
    // 收集本 turn 实际调用过的 explicit-enabled tool 名字，turn 末用于老化未用项。
    let mut tools_used_this_turn: rust_tools::cw::SkipSet<String> =
        rust_tools::cw::SkipSet::default();
    let mut consecutive_empty_responses: usize = 0;
    let mut consecutive_truncations: usize = 0;
    // 独立统计"流读取中断型"截断（stream_error）。它与模型输出过长无关（网络抖动 /
    // 服务端断流），因此不参与 reasoning 降档，也不累加 consecutive_truncations；
    // 但持续断流仍需有上限兜底，否则 usize::MAX 迭代预算的后台任务会无限重试。
    let mut consecutive_stream_errors: usize = 0;
    let mut turn_had_tool_error = false;
    // 保存进入本 turn 时的 reasoning effort 覆盖值（可能是用户 `/model effort` 的
    // 显式选择，或 None=用模型默认）。截断重试时会临时把它降到 Low，把输出 token
    // 预算从 reasoning 让给实际内容；turn 结束后（含所有 break 出口）统一恢复，
    // 不污染用户的会话级设置。
    let saved_effort_override = app.cli.reasoning_effort_override;
    // 同理保存 thinking 兜底开关：截断重试可能置位它以强制关闭 always-thinking
    // 模型的思考链，turn 末统一恢复，不污染后续 turn。
    let saved_thinking_disabled = app.cli.thinking_disabled_override;
    // 同理保存 max_tokens 自适应覆盖：零输出截断时自动降 max_tokens 重试。
    // 降级是临时的：一旦有正常输出（正常完成或正常截断）就恢复原始值，
    // 因为原始值本身是合理的（首次请求能成功）。turn 末兜底恢复。
    let saved_max_tokens_override = app.cli.max_tokens_override;
    // 当前是否处于零输出降级状态。
    let mut mt_downgraded = false;
    // VL 图片摘要（429 TPM 缓解）状态：pending_digest_source 保存上一轮 tool-call
    // 响应的未截断原文（含 reasoning），作为“搭车”解析摘要的来源；
    // image_digest_resolved 一旦置位，表示图片已被摘要替换、或两条路径都拿不到
    // 摘要而按用户决定保留原图——两种情况都不再每轮重试，避免反复发兜底请求。
    let mut pending_digest_source: Option<String> = None;
    let mut image_digest_resolved = false;
    // 跨 turn 图片摘要：记录本轮解析出的摘要（内容指纹, 摘要, 原图路径），
    // turn 正常结束时持久化进历史元数据表，下个 turn 加载历史时用同一指纹
    // 取回摘要并替换旧图片，避免新一轮重复发送上一 turn 的图片。
    let mut turn_digest: Option<(String, String, Vec<String>)> = None;
    // preflight 拒绝的 mutation 目标必须在本 turn 后续每次 prompt 重建时优先获得
    // scoped 指令预算；不能只消费一次，因为中间读取和 mid-turn 压缩可能让同一
    // mutation 的目标从可观测历史消失，继而被重复暂停。
    let mut scoped_preflight_targets = ScopedPreflightTargets::default();
    let loop_result = 'turn: loop {
        let iteration = supervisor.next_iteration();
        let effective_max_iterations = supervisor.effective_max_iterations(max_iterations);
        // 从第二轮起，把请求投影里用户消息内联的图片换成上一轮产出的文字摘要，
        // 避免在工具循环里反复重放 base64 触发 Doubao/Ark 侧 429 TPM 限流。
        // 优先搭车解析上一轮响应；拿不到再发一次禁用工具的一次性 VL 请求兜底；
        // 两者都失败则保留原图。只改请求投影 messages，canonical turn_messages 不动。
        // 放在 mcp 锁块之前：兜底请求含 .await，绝不能跨 std::Mutex 持锁。
        if !image_digest_resolved && iteration >= 2 {
            // 只处理「当前 turn」的用户图片：用 rposition 取最后一条含图 user 消息。
            // messages 头部可能载入历史里更早 turn 的图片（只有无持久化摘要的旧图，
            // 如老会话；有摘要的已在 prepare 阶段被替换成文本），但我们采集/生成的
            // 摘要只描述当前 turn 的图片（指令注入在当前 user 消息、兜底用
            // app.attached_image_files=当前 turn），拿它去替换旧图会张冠李戴。这里要
            // 换掉的是当前 turn 这张原图——它在请求投影里每轮重放，正是 digest 的目标。
            if let Some(idx) = messages.iter().rposition(|m| {
                m.role == "user" && crate::ai::request::content_has_image(&m.content)
            }) {
                let image_paths = app.attached_image_files.clone();
                let digest = match pending_digest_source
                    .as_deref()
                    .and_then(crate::ai::request::parse_digest)
                {
                    Some(d) => Some(d),
                    None => {
                        crate::ai::request::describe_image_for_digest(
                            app,
                            &next_model,
                            &image_paths,
                        )
                        .await
                    }
                };
                if let Some(digest) = digest {
                    crate::ai::request::swap_images_with_digest(
                        &mut messages[idx].content,
                        &digest,
                        &image_paths,
                    );
                    // 跨 turn 摘要：把本轮摘要记入 turn_digest，等 turn 正常结束时
                    // 持久化进历史元数据（见下方 Ok(_) 分支），下个 turn 加载历史时
                    // 用指纹取回摘要替换旧图，避免跨 turn 重复发图。
                    // 指纹必须取自 turn_messages（canonical）而非刚替换的 messages[idx]：
                    // 请求投影里带着注入的图片处理协议指令 / reminder，那些 part 不落库，
                    // 加载侧（replace_old_images_with_persisted_digests）对库里持久化的
                    // 原样内容算指纹，只有 canonical 版本才对得上。turn_messages 无含图
                    // user 消息（resume turn 丢图等）时静默跳过，不持久化。
                    if let Some(fp) =
                        crate::ai::request::last_image_user_message_fingerprint(&turn_messages)
                    {
                        turn_digest = Some((fp, digest, image_paths));
                    }
                }
                image_digest_resolved = true;
            } else {
                image_digest_resolved = true;
            }
        }
        {
            let mc = mcp_client.lock().unwrap();
            let scoped_project_instructions_ready = refresh_skill_turn_for_iteration(
                app,
                &mc,
                skill_manifests,
                &question,
                iteration,
                &mut skill_turn,
                scoped_preflight_targets.required(),
                &mut messages,
            );
            if !scoped_project_instructions_ready {
                let targets = scoped_preflight_targets.required()
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                break 'turn Err(std::io::Error::other(format!(
                    "failed to load target-scoped project instructions before mutation: {targets}"
                ))
                .into());
            }
        }
        if crate::ai::driver::runtime_ctx::take_subagent_wrap_up_request() {
            pre_timeout_wrap_up_requested = true;
            record_force_final_reason(&mut messages, "subagent_pre_timeout_wrap_up", iteration, None);
            force_final_response = true;
            inject_subagent_pre_timeout_wrap_up_note(&mut messages);
        }
        // side-note 实时引导：每次迭代顶部把文件队列里的引导注入上下文，下一轮 LLM 立刻可见。
        // 对前景任务与 subagent 均生效；subagent 通过 task_local SUBAGENT_TASK_ID
        // （回退到 AIOS_SUBAGENT_TASK_ID / SUBAGENT_TASK_ID 环境变量）区分目标。
        // 统一走 side_note::poll_and_inject 避免两套注入路径分叉与原文回显泄露。
        {
            let before = messages.len();
            let cnt =
                crate::ai::driver::side_note::poll_and_inject(&app.session_history_file, &mut messages);
            if cnt > 0 {
                let injected = messages[before..].to_vec();
                turn_messages.extend(injected);
                crate::ai::driver::print::print_tool_note_line(
                    "side-note",
                    &format!("injected {cnt} note(s)"),
                );
            }
        }
        let active_skill_name = skill_turn.primary_skill_name().map(str::to_string);
        let compression_report = std::mem::take(&mut supervisor.pending_compression_report);
        let mut response_model = None;
        let execution = match execute_turn_iteration(
            app,
            &next_model,
            &mut response_model,
            &mut messages,
            &turn_messages,
            one_shot_mode,
            &mut persisted_turn_messages,
            should_quit,
            force_final_response,
            terminal_dedupe_candidate.as_deref(),
            active_skill_name.as_deref(),
            iteration,
            compression_report,
        )
        .await
        {
            Ok(e) => e,
            Err(err) => break 'turn Err(err),
        };
        // 预超时收口信号在模型请求中途触发：放弃当前请求，立即进入强制收口迭代，
        // 而不是等当前（可能很长的）迭代自然结束。消费信号，避免下一轮迭代顶部重复注入。
        if matches!(&execution, IterationExecution::WrapUpFinal) {
            let _ = crate::ai::driver::runtime_ctx::take_subagent_wrap_up_request();
            pre_timeout_wrap_up_requested = true;
            record_force_final_reason(&mut messages, "subagent_pre_timeout_wrap_up", iteration, None);
            force_final_response = true;
            inject_subagent_pre_timeout_wrap_up_note(&mut messages);
            continue 'turn;
        }
        let was_final_response = matches!(&execution, IterationExecution::FinalResponse(_));
        let had_tool_call_execution = matches!(&execution, IterationExecution::ToolCall(_));
        // 搭车采集：图片摘要尚未解决时，缓存本轮 tool-call 响应的未截断原文
        // （assistant_text + reasoning_text）。下一轮循环开头据此尝试 parse 出摘要，
        // 命中即可省掉一次兜底请求。必须用 stream_result 原文——写回 messages 的
        // assistant narration 会被截断到 800 字符，可能截掉摘要尾部哨兵。
        if !image_digest_resolved && let IterationExecution::ToolCall(tce) = &execution {
            let sr = &tce.stream_result;
            pending_digest_source = Some(format!("{}\n{}", sr.assistant_text, sr.reasoning_text));
        }
        {
            let mc = mcp_client.lock().unwrap().routing_snapshot();
            // 用服务端返回的实际 prompt_tokens 校正后续请求的 max_tokens clamp。
            // 字符估算偏保守（高估），服务端的实际值更准确，能减少不必要的钳小。
            let usage_prompt = match &execution {
                IterationExecution::Truncated(sr) | IterationExecution::FinalResponse(sr) => {
                    Some((sr.usage_prompt_tokens, sr.usage_cached_prompt_tokens))
                }
                IterationExecution::ToolCall(tce) => Some((
                    tce.stream_result.usage_prompt_tokens,
                    tce.stream_result.usage_cached_prompt_tokens,
                )),
                _ => None,
            };
            if let Some((pt, cached)) = usage_prompt.filter(|(pt, _)| *pt > 0) {
                app.last_known_prompt_tokens = Some(pt);
                app.last_known_cached_prompt_tokens = Some(cached.min(pt));
            }
            // 空响应重试计数：连续 >2 次空响应则放弃，避免浪费迭代预算
            if matches!(execution, IterationExecution::EmptyResponse) {
                consecutive_empty_responses += 1;
                if consecutive_empty_responses > 5 {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ✗ {} consecutive empty responses; giving up retry",
                        consecutive_empty_responses
                    );
                    final_assistant_text = "[Model returned empty responses repeatedly; please retry or switch models]".to_string();
                    break 'turn Ok(None);
                }
            } else {
                consecutive_empty_responses = 0;
            }
            // 截断重试计数：连续多次被截断（输出上限/工具 JSON 半截）仍无法收敛时
            // 放弃，避免无限重试烧预算。阈值取 3：给模型两次收缩重写的机会。
            if let IterationExecution::Truncated(stream_result) = &execution {
                consecutive_truncations += 1;
                // 重置工具循环检测：截断重试期间的重复调用是预期行为，
                // 不应被误判为工具死循环并触发 hard-stop 强制收敛。
                supervisor.mark_truncation_skip();

                if stream_result.stream_error {
                    // 流读取错误（网络抖动 / 服务端异常断流）导致的截断。
                    // 模型并没有输出太多，降 reasoning_effort 和注入收缩提示都无意义。
                    // 不累积 consecutive_truncations（这不是模型的错），但用独立计数
                    // consecutive_stream_errors 兜底，避免服务端持续断流时无限重试。
                    consecutive_truncations = 0;
                    consecutive_stream_errors += 1;
                    if consecutive_stream_errors > MAX_STREAM_ERROR_RETRIES {
                        let _ = writeln!(
                            std::io::stderr(),
                            "  ✗ {} consecutive response-stream read interruptions; giving up retry",
                            consecutive_stream_errors
                        );
                        final_assistant_text =
                            "[Response stream interrupted repeatedly; the server may be unstable. Please retry later or switch models]"
                                .to_string();
                        break 'turn Ok(None);
                    }
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ⚠ Response-stream read interrupted (consecutive #{consecutive_stream_errors}); auto-retrying, stopping after {MAX_STREAM_ERROR_RETRIES} consecutive failures…"
                    );
                } else {
                    // 真截断：模型撞输出上限或工具 JSON 半截。
                    consecutive_stream_errors = 0;

                    // 零输出截断检测：completion=0 + finish_reason=length 说明服务端
                    // 拒绝了 max_tokens 值（典型：relay/兼容层对超大 max_tokens 返回
                    // 空响应而非报错）。此时降 reasoning_effort / 禁 thinking 都无济于事
                    // ——问题不在模型输出太多，而在 max_tokens 本身被服务端拒绝。
                    // 策略：将 max_tokens 减半后重试，直到服务端接受。
                    let is_zero_completion = stream_result.usage_completion_tokens == 0
                        && stream_result
                            .finish_reason_value
                            .as_deref()
                            .is_some_and(|r| r == "length");
                    if is_zero_completion {
                        let current_max = app
                            .cli
                            .max_tokens_override
                            .or_else(|| crate::ai::models::max_output_tokens(&app.current_model))
                            .unwrap_or(32768);
                        let halved = (current_max / 2).max(4096);
                        let _ = writeln!(
                            std::io::stderr(),
                            "  ⚠ Zero-output truncation (completion=0); auto-downgrading max_tokens {} → {} and retrying",
                            current_max,
                            halved
                        );
                        app.cli.max_tokens_override = Some(halved);
                        mt_downgraded = true;
                    } else if mt_downgraded {
                        // 正常截断（有输出但被打断）：服务端接受了当前 max_tokens，
                        // 恢复原始值给后续迭代更大输出预算。
                        app.cli.max_tokens_override = saved_max_tokens_override;
                        mt_downgraded = false;
                    }

                    // 该模型降 reasoning_effort 是否真能缩短思考链。模型级 wire
                    // 声明优先于 provider 默认：例如 DashScope DeepSeek 虽使用
                    // enable_thinking 开关，但推理强度仍由顶层 reasoning_effort 控制。
                    // 未声明有效 effort wire 的布尔开关方言才需要直接关闭 thinking。
                    let effort_helps =
                        crate::ai::models::reasoning_effort_reduces_thinking(&next_model);

                    if effort_helps {
                        // 截断类型分流：区分「推理吃光预算」与「正文过长」两种截断。
                        // 用服务端上报的 reasoning_tokens 占比判断——推理占比高时降档
                        // 有效（直接缩短思考链、把预算让给正文）；正文过长时降档救不了
                        // 预算（模型没在推理），白损失质量，只靠注入的收缩提示即可。
                        // reasoning_tokens 未上报（0）或 completion 为 0 时按「未知」
                        // 保守处理，走降档阶梯保底，避免新逻辑对不报 usage 明细的
                        // provider 造成回归。
                        let reasoning_reported = stream_result.usage_reasoning_tokens > 0
                            && stream_result.usage_completion_tokens > 0;
                        let reasoning_dominant = reasoning_reported
                            && stream_result.usage_reasoning_tokens as f64
                                / stream_result.usage_completion_tokens as f64
                                >= REASONING_BUDGET_DOMINANCE_RATIO;
                        let text_too_long = reasoning_reported && !reasoning_dominant;

                        if !text_too_long {
                            // 渐进式 reasoning effort 降档，把输出预算从 reasoning 让给
                            // 实际内容。resolve_reasoning_effort 每次迭代实时读该字段，
                            // 改了立即对下一次生效。
                            //
                            // 1 次截断 → High（略降推理开销）
                            // 2 次截断 → Medium（进一步缩短思考链）
                            // 3 次以上 → Low（保留最小推理能力的下限）
                            //
                            // 相比旧版 2 次即归零（None/禁用）的重度阉割，这里保留推理
                            // 能力更温和：实测 effort 每降一档能省约 15-20% 预算，且模型
                            // 仍保留思考能力。thinking 关闭只留给第 3 次兜底（见下）。
                            // 本阶梯刻意不下发 `reasoning_effort: "none"`：显式 none 会
                            // 彻底阉割推理，而省略字段会让服务端回退到自身默认档
                            // （gpt-5.x 默认 medium）反而调高预算，两者都不适合作为
                            // 截断收敛手段。
                            app.cli.reasoning_effort_override =
                                Some(match consecutive_truncations {
                                    1 => Some(crate::ai::provider::ReasoningEffort::High),
                                    2 => Some(crate::ai::provider::ReasoningEffort::Medium),
                                    _ => Some(crate::ai::provider::ReasoningEffort::Low),
                                });
                            let note = if reasoning_dominant {
                                "reasoning ate the output budget; downgrading effort to free budget for visible text"
                            } else {
                                "reasoning usage unreported; conservative effort downgrade"
                            };
                            crate::ai::driver::decision_log::log_truncation_downgrade(
                                crate::ai::driver::decision_log::get_decision_log_store(),
                                &crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
                                crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
                                &next_model,
                                consecutive_truncations,
                                stream_result.usage_reasoning_tokens,
                                stream_result.usage_completion_tokens,
                                true,
                                note,
                            );
                        } else {
                            // 正文过长型截断：不降 effort（模型没在推理，降了也救不了
                            // 预算），仅靠已注入的 output_truncated 收缩提示让模型缩小
                            // 输出。记录决策供事后审计。
                            crate::ai::driver::decision_log::log_truncation_downgrade(
                                crate::ai::driver::decision_log::get_decision_log_store(),
                                &crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
                                crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
                                &next_model,
                                consecutive_truncations,
                                stream_result.usage_reasoning_tokens,
                                stream_result.usage_completion_tokens,
                                false,
                                "visible text dominated the output budget; keeping effort, prompting shrink only",
                            );
                        }
                        // effort 阶梯走到第 3 档仍截断，说明仅靠降 effort 已不足以收敛，
                        // 叠加强制关闭 thinking 作为兜底，把整个输出预算让给可见内容。
                        if consecutive_truncations >= MAX_MODEL_TRUNCATION_RETRIES {
                            app.cli.thinking_disabled_override = true;
                        }
                    } else {
                        // 降 effort 对该方言无效：不浪费重试轮次走无效阶梯，
                        // 首次真截断即强制关闭 thinking，把整个输出预算让给可见内容。
                        app.cli.thinking_disabled_override = true;
                    }
                }

                let partial_text = stream_result.assistant_text.trim();
                let has_visible_text = !partial_text.is_empty();

                // 模型已产出可见文本但仍连续撞长度上限（典型：推理模型 reasoning 占满
                // 预算）。继续重试通常无帮助——模型会反复产出同样长度的内容。
                // 给一次降档重试机会后即接受部分文本作为最终回答。
                // 但 stream_error 场景不计入 consecutive_truncations，不会触发此分支。
                if has_visible_text
                    && consecutive_truncations >= MAX_MODEL_TRUNCATION_RETRIES
                    && !stream_result.stream_error
                {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ▲ {} consecutive truncated outputs; keeping the partial text produced so far",
                        consecutive_truncations
                    );
                    final_assistant_text = partial_text.to_string();
                    break 'turn Ok(None);
                }

                // stream_error 已在上面重置 consecutive_truncations=0，不会进入此分支。
                if consecutive_truncations >= MAX_MODEL_TRUNCATION_RETRIES
                    && !stream_result.stream_error
                {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ✗ {} consecutive truncated responses; giving up retry",
                        consecutive_truncations
                    );
                    // 保留模型已产出的部分文本（若有），比直接丢弃更有价值。
                    final_assistant_text = if has_visible_text {
                        partial_text.to_string()
                    } else {
                        "[Model output truncated repeatedly; please shrink per-operation scope (e.g., write files in chunks) or switch models]"
                            .to_string()
                    };
                    break 'turn Ok(None);
                }
            } else {
                consecutive_truncations = 0;
                // 非截断：恢复因零输出降级的 max_tokens。
                if mt_downgraded {
                    app.cli.max_tokens_override = saved_max_tokens_override;
                    mt_downgraded = false;
                }
            }
            let step = match handle_iteration_execution_for_model(
                app,
                response_model.as_deref().unwrap_or(&next_model),
                &question,
                &mc,
                mcp_client,
                execution,
                &mut messages,
                &mut turn_messages,
                one_shot_mode,
                &mut persisted_turn_messages,
                &mut final_assistant_text,
                &mut final_assistant_recorded,
                &mut force_final_response,
                &mut terminal_dedupe_candidate,
                skill_turn.matched_skill_names().is_empty(),
                iteration,
                effective_max_iterations,
                consecutive_truncations,
                &mut turn_had_tool_error,
            ) {
                Ok(s) => s,
                Err(err) => break 'turn Err(err),
            };
            match step {
                TurnLoopStep::ScopedPreflightContinue(targets) => {
                    if supervisor.grant_scoped_preflight_grace() {
                        scoped_preflight_targets.record_pause(targets);
                        // 该轮没有执行 mutation，也不应计入 progress/loop 统计。
                        continue 'turn;
                    }
                    // 独立 preflight 预算耗尽后保持安全拒绝并收口，避免通过不断
                    // 切换目录无限扩张迭代预算。
                    record_force_final_reason(
                        &mut messages,
                        "scoped_preflight_budget_exhausted",
                        iteration,
                        None,
                    );
                    force_final_response = true;
                    continue 'turn;
                }
                TurnLoopStep::Continue => {
                    let mut new_tools = crate::ai::tools::enable_tools::drain_pending_enable();
                    let pending_mcp = crate::ai::tools::enable_tools::drain_pending_mcp_names();
                    if !pending_mcp.is_empty() {
                        let mcp_all = mc.get_all_tools();
                        for tool in mcp_all {
                            if pending_mcp.iter().any(|n| n == &tool.function.name) {
                                new_tools.push(tool);
                            }
                        }
                    }
                    if !new_tools.is_empty() {
                        if let Some(ctx) = app.agent_context.as_mut() {
                            for tool in new_tools {
                                if !ctx
                                    .tools
                                    .iter()
                                    .any(|t| t.function.name == tool.function.name)
                                {
                                    ctx.tools.push(tool);
                                }
                            }
                        }
                    }
                    // 记录本轮 assistant 实际调用过的 tool 名字（去重），
                    // 留给 turn 末用于老化未用 explicit tool。
                    if let Some(last_assistant) =
                        messages.iter().rev().find(|m| m.role == "assistant")
                        && let Some(tool_calls) = &last_assistant.tool_calls
                    {
                        for tc in tool_calls {
                            tools_used_this_turn.insert(tc.function.name.clone());
                        }
                    }
                }
                TurnLoopStep::Break => {
                    if was_final_response {
                        final_response_model.clone_from(&response_model);
                    }
                    break 'turn Ok(None);
                }
                TurnLoopStep::Return(outcome) => break 'turn Ok(Some(outcome)),
            }
        }
        // ↓↓↓ Continue 分支的后续处理（已离开 mc 锁，可以安全 await）↓↓↓
        let progress_messages = if had_tool_call_execution {
            current_tool_round_messages(&messages)
        } else {
            Vec::new()
        };

        // === Mid-turn 渐进式压缩 ===
        // 每轮 tool 执行完毕后检查 messages 总字符；超过软阈值时
        // 复用跨 turn 压缩管线，避免长链工具调用把上下文撑爆。
        // 节流：①冷却 N 轮 ②增量小于 DELTA 时跳过，避免 no-op 反复压缩。
        // 阈值按 history_max_chars 动态计算（floor 兜底），避免用户调整
        // history_max_chars 后 mid-turn 阈值依旧死锁在 36K/80K。
        let history_max_chars = app.config.history_max_chars;
        let mid_turn_soft_base = mid_turn_compress_soft_threshold(&next_model, history_max_chars);
        // 长循环时把软阈值下调到 SOFT_FLOOR，遏制 O(n²) 累积重发（详见
        // [`LONG_LOOP_COMPRESS_ITERATION_THRESHOLD`]）。门控与下面的实际
        // mid_turn_compress 调用共用同一 `mid_turn_soft`，避免「门开了却 no-op」。
        let mid_turn_soft = supervisor.effective_mid_turn_soft_threshold(mid_turn_soft_base);
        let mid_turn_hard = mid_turn_compress_hard_threshold(&next_model, history_max_chars);
        let total_chars = crate::ai::history::messages_total_chars_pub(&messages);
        if supervisor.should_try_mid_turn_compress(total_chars, mid_turn_soft) {
            // 与跨 turn 压缩（prepare.rs）一致地解析会话 overflow 目录：mid-turn
            // 压缩据此把 read_file/grep 等「不可压缩」工具的大输出零压缩外溢到
            // 文件 + 留预览 stub，既释放上下文又不丢信息（模型可重新 read_file）。
            let overflow_dir = {
                use crate::ai::history::SessionStore;
                let store = SessionStore::new(app.config.history_file.as_path());
                store.session_assets_dir(&app.session_id)
            };
            let drained: Vec<crate::ai::history::Message> = std::mem::take(&mut messages);
            let (compressed, before, after) = crate::ai::history::mid_turn_compress(
                drained,
                mid_turn_soft,
                Some(overflow_dir.as_path()),
                crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
            );
            messages = compressed;
            supervisor.mark_compress(after);
            let mut compression_report = CompressionReport::default();
            if after < before {
                compression_report.record("mid-turn", before, after);
            }
            // 硬阈值：无损 + 弱损管线之后仍超额，调用 LLM 摘要兜底，
            // 把早期对话压成单条 internal_note，并将各压缩阶段合并为一条 status line。
            if after > mid_turn_hard
                && should_try_llm_summary(&app.session_id, after, mid_turn_hard)
            {
                let drained: Vec<crate::ai::history::Message> = std::mem::take(&mut messages);
                let (after_msgs, llm_before, llm_after, was_effective, llm_summary_inserted) =
                    crate::ai::history::mid_turn_llm_summarize(
                        app,
                        drained,
                        MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
                        MID_TURN_LLM_SUMMARY_MAX_CHARS,
                        history_max_chars,
                    crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
                    )
                    .await;
                messages = after_msgs;
                record_llm_summary_attempt_chars(&app.session_id, llm_after);
                compression_report.record_llm_summary_attempt(
                    format!("mid-turn LLM (limit {mid_turn_hard})"),
                    llm_before,
                    llm_after,
                    was_effective,
                    llm_summary_inserted,
                );
                compression_report.emit();
            } else {
                supervisor.pending_compression_report = compression_report;
            }
        }

        // === 工具循环检测 ===
        // 若 execute 层已决定进入最终响应，就只保留 iteration-limit 调度，不再叠加
        // loop/progress/checkpoint prompt。反过来，health hard-stop 也不应冒充 iteration limit。
        let force_final_before_health = force_final_response;
        let tool_loop_signal = if force_final_before_health {
            ToolLoopSignal::None
        } else {
            supervisor.record_tool_signatures_for_progress(
                &messages,
                if progress_messages.is_empty() {
                    &messages
                } else {
                    &progress_messages
                },
                PROGRESS_FREE_EXPLORE_ROUNDS,
            )
        };
        let health_signal_injected = tool_loop_signal != ToolLoopSignal::None;
        match tool_loop_signal {
            ToolLoopSignal::None => {}
            ToolLoopSignal::Coarse => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "possible low-yield repetition detected (same target, paging only): injecting converge hint",
                );
                inject_coarse_loop_note(&mut messages);
            }
            ToolLoopSignal::TargetRepeat => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "possible low-yield repetition detected (same target across mixed tool rounds): injecting converge hint",
                );
                inject_target_repeat_loop_note(&mut messages);
            }
            ToolLoopSignal::TargetRescan(target, reads) => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    &format!(
                        "same file re-read from the beginning {reads} times (pagination loop): injecting converge hint (target: {target})"
                    ),
                );
                inject_target_rescan_note(&mut messages, &target, reads);
            }
            ToolLoopSignal::TargetRescanHard(target, reads) => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    &format!(
                        "same file re-read from the beginning past the hard threshold (pagination loop): switching to no-tool handoff (target: {target}, reads: {reads})"
                    ),
                );
                inject_target_rescan_hard_stop_note(&mut messages, &target, reads);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "target-rescan-hard-stop",
                );
                record_force_final_reason(&mut messages, "target_rescan", iteration, Some(&target));
                force_final_response = true;
            }
            ToolLoopSignal::CoarseHard => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "low-yield execute_command repetition hard-stop: switching to no-tool handoff",
                );
                inject_coarse_hard_loop_stop_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "low-yield-hard-stop",
                );
                record_force_final_reason(&mut messages, "low_yield_repetition", iteration, None);
                force_final_response = true;
            }
            ToolLoopSignal::Soft => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "tool-loop detected: injecting self-reflect prompt",
                );
                inject_loop_breaker_note(&mut messages);
                // 高风险异常才注入一次任务锚点，降低目标漂移概率。
                supervisor.maybe_inject_task_anchor(&mut messages, &question, "tool-loop");
            }
            ToolLoopSignal::Hard => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "tool-loop hard-stop: switching to no-tool handoff",
                );
                inject_hard_loop_stop_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "tool-loop-hard-stop",
                );
                record_force_final_reason(&mut messages, "tool_loop", iteration, None);
                force_final_response = true;
            }
            ToolLoopSignal::LowProgressSoft => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "progress-budget: ambiguous recent progress, injecting one review prompt",
                );
                inject_low_progress_soft_note(&mut messages);
            }
            ToolLoopSignal::LowProgressLedger => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "progress-budget: no new measurable evidence after response window, requesting decision ledger",
                );
                inject_progress_ledger_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "low-progress-ledger",
                );
            }
            ToolLoopSignal::ReadOnlyBreadth => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "read-only analysis breadth is high: requesting evidence summary before expanding further",
                );
                let agent_team_active = skill_turn
                    .matched_skill_names()
                    .iter()
                    .any(|name| name == "agent-team");
                inject_read_only_breadth_note(&mut messages, agent_team_active);
            }
            ToolLoopSignal::LowProgressHard => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "progress-budget hard-stop: switching to no-tool handoff",
                );
                inject_low_progress_hard_stop_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "low-progress-hard-stop",
                );
                record_force_final_reason(&mut messages, "progress_no_progress", iteration, None);
                force_final_response = true;
            }
            ToolLoopSignal::ReadOnlyBreadthHard => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "read-only breadth hard-stop: no mutation after wide investigation, switching to no-tool handoff",
                );
                inject_low_progress_hard_stop_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "read-only-breadth-hard-stop",
                );
                record_force_final_reason(&mut messages, "read_only_breadth", iteration, None);
                force_final_response = true;
            }
        }

        // === 分级、阶段感知的工具轮次检查点 ===
        // 使用 pre-compression 的当前工具轮判断阶段，累计轮次保持不变；检查点只调度
        // 下一步，不把刚完成的 mutation 误报为失败。同一轮已有更具体的 health signal
        // 时跳过 checkpoint，避免多个收敛 prompt 叠加。
        let effective_max_iterations = supervisor.effective_max_iterations(max_iterations);
        if !health_signal_injected && !force_final_before_health {
            let checkpoint_phase = tool_round_checkpoint_phase(&progress_messages, &turn_messages);
            if let Some(checkpoint) = supervisor.maybe_inject_tool_round_checkpoint(
                &mut messages,
                effective_max_iterations,
                checkpoint_phase,
            ) {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    &format!(
                        "tool-round checkpoint reached: round={} threshold={} level={} recent_progress={} action={}",
                        supervisor.iteration,
                        checkpoint.threshold,
                        checkpoint.level.label(),
                        checkpoint.phase.recent_progress(),
                        checkpoint.phase.action(),
                    ),
                );
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "tool-round-checkpoint",
                );
            }
        }

        // === Iteration limit 自反思 ===
        // execute.rs 在 iteration >= max_iterations 时会把
        // force_final_response 置 true。此时除原有的 "Tool limit reached"
        // system prompt 外，再额外补一条更具体的反思 prompt
        // （只注入一次，避免重复刷屏）。
        supervisor.maybe_inject_iteration_limit_note(
            &mut messages,
            effective_max_iterations,
            force_final_before_health && !pre_timeout_wrap_up_requested,
        );
        if force_final_before_health && !pre_timeout_wrap_up_requested {
            supervisor.maybe_inject_task_anchor(&mut messages, &question, "iteration-limit");
        }
    };

    // 恢复进入本 turn 前的 reasoning effort 覆盖值：截断重试可能把它临时降到了
    // Low，这里统一还原（覆盖所有 break 'turn 出口），避免把降档泄漏到后续 turn
    // 污染用户的会话级设置。
    app.cli.reasoning_effort_override = saved_effort_override;
    app.cli.thinking_disabled_override = saved_thinking_disabled;
    app.cli.max_tokens_override = saved_max_tokens_override;

    // 老化未在本 turn 使用的 explicit-enabled tool。
    // 连续 N 个 turn 闲置就 demote，避免"启用一次永久焊接"。
    crate::ai::tools::enable_tools::age_unused_explicit_tools(tools_used_this_turn.iter());

    let loop_result = loop_result.map_err(|e: Box<dyn std::error::Error>| e.to_string());

    // 只有 active skill 明确通过工具请求用户输入、且本轮正常结束时才建立一次性续接。
    // 这避免了按自然语言问号/关键词猜测，外部 skill 也无需修改自身 manifest。
    let requested_user_input = crate::ai::tools::skill_tools::take_pending_user_input_request();
    if requested_user_input && loop_result.is_ok() && !app.last_turn_interrupted {
        if !skill_turn.matched_skill_names().is_empty() {
            app.pending_skill_continuation =
                Some(crate::ai::types::PendingSkillContinuation {
                    skill_names: skill_turn.matched_skill_names().to_vec(),
                });
        }
    }

    let final_skill_name = skill_turn.primary_skill_name().map(str::to_owned);
    skill_turn.restore_agent_context(app);

    match loop_result {
        Ok(Some(outcome)) => {
            app.last_turn_had_tool_calls = false;
            Ok(outcome)
        }
        Ok(_) => {
            // 跨 turn 图片摘要持久化（仅 turn 正常完成路径；打断/退出/报错分支不执行，
            // 摘要丢弃，下轮会重发原图一次）。工具循环内解析到的摘要已记入 turn_digest；
            // 单轮响应（无工具循环）时摘要块不执行，退而解析最终回复文本里的摘要。
            let mut digest_to_persist = turn_digest.take();
            if digest_to_persist.is_none() {
                if let Some(digest) = crate::ai::request::parse_digest(&final_assistant_text) {
                    if let Some(fp) =
                        crate::ai::request::last_image_user_message_fingerprint(&turn_messages)
                    {
                        digest_to_persist = Some((fp, digest, app.attached_image_files.clone()));
                    }
                }
            }
            if let Some((fp, digest, paths)) = digest_to_persist {
                let _ = crate::ai::history::upsert_image_digest_sqlite(
                    &app.session_history_file,
                    &fp,
                    &digest,
                    &paths,
                );
            }
            finalize_turn(
                app,
                &next_model,
                final_response_model.as_deref().unwrap_or(&next_model),
                &question,
                &final_assistant_text,
                final_assistant_recorded,
                terminal_dedupe_candidate.as_deref(),
                final_skill_name.as_deref(),
                &mut turn_messages,
                one_shot_mode,
                &mut persisted_turn_messages,
                should_quit,
                turn_had_tool_error,
            )
            .await
        }
        Err(err_str) => Err(err_str.into()),
    }
}
