// =============================================================================
// AIOS Primitives - Futex & Trace
// =============================================================================
// 本文件为 AIOS 新增两类 primitive，使 agent 不再需要在用户态手搓同步/埋点：
//
//   1. Futex  —— 通用"条件变量 + 计数器"。用于取消信号、流式 IO 挂起唤醒、
//      任何"等某个条件成立"的场景。替代 agent 里散落的 AtomicBool。
//
//   2. Trace  —— 内核态 tracing ring buffer。所有 span / event 都经此落盘，
//      替代 agent_hang_span 宏。下游 driver 消费它做输出 / OTel / 挂起检测。
//
// 设计约束：
//   - 不破坏现有 Syscall / KernelInternal trait；作为独立 trait 加入 Kernel。
//   - LocalOS 同步实现即可，不触碰 tokio；async 等待由 agent 侧包一层（因为
//     当前 SharedKernel 是 std::sync::Mutex，await 持锁是反模式）。
//     Futex 的 wait 语义通过"获取一个 waker token，释放锁后再阻塞"来实现，
//     但 phase-0 提供同步接口 + 非阻塞 try_wait，agent 侧轮询即可验证设计。
// =============================================================================

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::FastMap;

// --------------------------------------------------------------------------
// Futex
// --------------------------------------------------------------------------

/// Futex 地址：不可伪造的 64-bit handle，由 kernel 分配。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FutexAddr(pub u64);

impl FutexAddr {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for FutexAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "futex_{}", self.0)
    }
}

/// Futex wait 的返回原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FutexWakeReason {
    /// 被 `futex_wake` 显式唤醒。
    Woken,
    /// `futex_wait` 时 value 已经不等于 expected（fast path，无需真正阻塞）。
    ValueChanged,
    /// 被 cancel signal / SIGCANCEL 打断。
    Cancelled,
    /// 不存在该 futex 地址。
    NotFound,
}

/// Futex 状态：当前值 + 总唤醒计数（供 wait 的 "expected" 语义）。
#[derive(Debug)]
pub(super) struct FutexState {
    pub(super) value: AtomicU64,
    /// 等待此 futex 的 PID 队列（FIFO）。
    pub(super) waiters: VecDeque<u64>,
    /// 每次 wake 自增，wait 可通过比对前后 seq 判断"是否被唤醒过"。
    pub(super) seq: u64,
    pub(super) event_id: crate::kernel::EventId,
}

impl FutexState {
    pub(super) fn new(initial: u64, event_id: crate::kernel::EventId) -> Self {
        Self {
            value: AtomicU64::new(initial),
            waiters: VecDeque::new(),
            seq: 0,
            event_id,
        }
    }
}

/// Futex 相关 syscall。
pub trait FutexOps {
    /// 创建一个 futex，返回 handle。label 仅用于诊断 / trace。
    fn futex_create(&mut self, initial: u64, label: String) -> FutexAddr;

    /// 读取当前值。
    fn futex_load(&self, addr: FutexAddr) -> Option<u64>;

    /// CAS 更新。返回 Ok(旧值) 成功，Err(当前值) 失败。
    fn futex_cas(&mut self, addr: FutexAddr, expected: u64, new_value: u64) -> Result<u64, u64>;

    /// 原子加 delta，返回旧值。
    fn futex_fetch_add(&mut self, addr: FutexAddr, delta: u64) -> Option<u64>;

    /// 存储新值，返回旧值。
    fn futex_store(&mut self, addr: FutexAddr, new_value: u64) -> Option<u64>;

    /// 非阻塞检测：若 value != expected 立即返回 ValueChanged；若相等返回 None 表示需要外部等待。
    /// 返回 Some(reason) 时表示无需再阻塞。
    fn futex_try_wait(&self, addr: FutexAddr, expected: u64) -> Option<FutexWakeReason>;

    /// 唤醒 n 个等待者。返回真正唤醒的数量。
    fn futex_wake(&mut self, addr: FutexAddr, n: usize) -> usize;

    /// 销毁 futex。等待者会收到 NotFound。
    fn futex_destroy(&mut self, addr: FutexAddr) -> bool;

    /// 向此 futex 的等待队列登记 pid（供 kernel 内部唤醒用）。
    /// 返回登记后的 seq 快照，供后续判断是否漏醒。
    fn futex_register_waiter(&mut self, addr: FutexAddr, pid: u64) -> Option<u64>;

    /// 取消等待（pid 不再等此 futex）。
    fn futex_cancel_waiter(&mut self, addr: FutexAddr, pid: u64) -> bool;

    /// 读当前 seq，用于 wait 的 "are we woken since seq0" 语义。
    fn futex_seq(&self, addr: FutexAddr) -> Option<u64>;

    /// 读当前 futex 关联的内核事件 ID，供多路复用等待统一降到 event 集合。
    fn futex_event_id(&self, addr: FutexAddr) -> Option<crate::kernel::EventId>;
}

// --------------------------------------------------------------------------
// Trace
// --------------------------------------------------------------------------

/// 内核 trace 事件级别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// 一条 trace 记录。结构化字段集中放在 fields 里，避免散字段。
#[derive(Debug, Clone)]
pub struct TraceRecord {
    pub seq: u64,
    pub tick: u64,
    pub pid: Option<u64>,
    pub level: TraceLevel,
    /// 稳定的 span/event 名字，形如 `turn_runtime::run_turn` 。
    pub name: String,
    /// 本条记录的 span_id（同一 span 的 enter/exit/event 共享）。
    pub span_id: Option<u64>,
    /// 父 span_id，用于拼父子关系。
    pub parent_span_id: Option<u64>,
    /// 事件种类：span_enter / span_exit / event。
    pub kind: TraceKind,
    /// 结构化字段（key -> json-like 字符串）。`None` 表示无字段，避免空 HashMap
    /// 在每条记录上多占一个 raw_table 头。访问时建议使用 [`TraceRecord::fields`]。
    pub fields: Option<FastMap<String, String>>,
    pub message: Option<String>,
}

impl TraceRecord {
    /// 取 fields 的引用；空字段统一返回 `None`，调用方可用 `.unwrap_or(&EMPTY)` 简化。
    pub fn fields(&self) -> Option<&FastMap<String, String>> {
        self.fields.as_ref()
    }

    /// 把可能为空的 fields HashMap 装箱为存储形态：空集合归 None。
    pub(super) fn pack_fields(fields: FastMap<String, String>) -> Option<FastMap<String, String>> {
        if fields.is_empty() { None } else { Some(fields) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceKind {
    SpanEnter,
    SpanExit,
    Event,
}

/// Trace 相关 syscall。
pub trait TraceOps {
    /// 创建一个 span，返回 span_id。parent=None 则为根 span。
    fn trace_span_enter(
        &mut self,
        name: String,
        parent: Option<u64>,
        fields: FastMap<String, String>,
    ) -> u64;

    /// 关闭 span（写入 SpanExit 记录）。
    fn trace_span_exit(&mut self, span_id: u64, fields: FastMap<String, String>);

    /// 发射一条独立 event。
    fn trace_event(
        &mut self,
        name: String,
        level: TraceLevel,
        span_id: Option<u64>,
        fields: FastMap<String, String>,
        message: Option<String>,
    );

    /// 读取最近 N 条 trace（从新到旧）。
    fn trace_recent(&self, n: usize) -> Vec<TraceRecord>;

    /// 自 `since_seq` 起的所有 trace（升序）。用于外部 drain。
    fn trace_drain_since(&self, since_seq: u64) -> Vec<TraceRecord>;

    /// 当前最新 seq（用于 drain 游标）。
    fn trace_head_seq(&self) -> u64;

    /// 设置 ring buffer 容量（超出会丢弃最旧记录）。
    fn trace_set_capacity(&mut self, cap: usize);
}

/// Trace ring buffer。
#[derive(Debug)]
pub(super) struct TraceRing {
    pub(super) buf: VecDeque<TraceRecord>,
    pub(super) capacity: usize,
    pub(super) next_seq: u64,
    pub(super) next_span_id: u64,
}

impl TraceRing {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity.min(4096)),
            capacity,
            next_seq: 1,
            next_span_id: 1,
        }
    }

    pub(super) fn push(&mut self, rec: TraceRecord) {
        if self.capacity == 0 {
            return;
        }
        while self.buf.len() >= self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(rec);
    }

    pub(super) fn alloc_seq(&mut self) -> u64 {
        let s = self.next_seq;
        self.next_seq += 1;
        s
    }

    pub(super) fn alloc_span(&mut self) -> u64 {
        let s = self.next_span_id;
        self.next_span_id += 1;
        s
    }
}

// --------------------------------------------------------------------------
// 小工具：便捷构造 fields map（agent 侧用）
// --------------------------------------------------------------------------
pub fn fields() -> FastMap<String, String> {
    FastMap::default()
}

#[doc(hidden)]
pub fn _field_insert<V: std::fmt::Display>(map: &mut FastMap<String, String>, key: &str, value: V) {
    map.insert(key.to_string(), value.to_string());
}

/// 便捷宏：`trace_fields!{"foo" => 1, "bar" => "baz"}` 返回 FastMap<String,String>。
#[macro_export]
macro_rules! trace_fields {
    () => {{ $crate::primitives::fields() }};
    ( $( $k:expr => $v:expr ),+ $(,)? ) => {{
        let mut __m = $crate::primitives::fields();
        $(
            $crate::primitives::_field_insert(&mut __m, $k, $v);
        )+
        __m
    }};
}

// 让 Ordering 在实现里可用（避免使用处忘记 use）
#[allow(dead_code)]
pub(super) const FUTEX_ORDER: Ordering = Ordering::SeqCst;

// --------------------------------------------------------------------------
// ResourceLimit / ResourceUsage —— cgroup-like 资源配额
// --------------------------------------------------------------------------

/// 进程资源上限。`u64::MAX` 表示无限。
///
/// 设计原则：所有配额集中在内核态；agent 不应该在用户态维护 max_iterations 常量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimit {
    /// LLM turn 数上限（替代 Process.quota_turns，为方便过渡保持同步）。
    pub max_turns: u64,
    /// Tool 调用次数上限。
    pub max_tool_calls: u64,
    /// 累计 prompt tokens 上限。
    pub max_tokens_in: u64,
    /// 累计 completion tokens 上限。
    pub max_tokens_out: u64,
    /// 累计成本（美分 / 微元，具体单位由 LLM 设备决定），约束不超过。
    pub max_cost_micros: u64,
    /// 墙钟 tick 上限：created_at_tick + max_wallclock_ticks 作为 deadline。
    pub max_wallclock_ticks: u64,
    /// 单次 tool 调用返回体字节上限（防止巨型输出炸上下文）。
    pub max_tool_call_bytes: u64,
    /// 通过 VfsOps 累计读写的字节数上限（/dev/llm 之外的磁盘 I/O 配额）。
    pub max_fs_bytes: u64,
}

impl ResourceLimit {
    /// 全部无限。用于兼容旧行为。
    pub const fn unlimited() -> Self {
        Self {
            max_turns: u64::MAX,
            max_tool_calls: u64::MAX,
            max_tokens_in: u64::MAX,
            max_tokens_out: u64::MAX,
            max_cost_micros: u64::MAX,
            max_wallclock_ticks: u64::MAX,
            max_tool_call_bytes: u64::MAX,
            max_fs_bytes: u64::MAX,
        }
    }

    /// 从旧的 `quota_turns: usize` 字段构造一个"只约束 turns 其他无限"的 limit。
    /// 0 按旧语义视作"不限制"。
    pub fn from_legacy(quota_turns: usize) -> Self {
        let mut l = Self::unlimited();
        if quota_turns > 0 {
            l.max_turns = quota_turns as u64;
        }
        l
    }
}

impl Default for ResourceLimit {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// 进程已用资源累计。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceUsage {
    pub turns: u64,
    pub tool_calls: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_micros: u64,
    /// 单调递增：last_tool_call_bytes 为最近一次 tool 返回体大小（供观察）。
    pub last_tool_call_bytes: u64,
    /// VfsOps 累计读写字节数。
    pub fs_bytes: u64,
}

/// 配额校验结果。kernel 在推进 usage 时返回此值，调用者据此决定是否终止进程。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RlimitVerdict {
    /// 未越界。
    Ok,
    /// 越界，并指出具体维度。
    Exceeded {
        dimension: RlimitDim,
        used: u64,
        limit: u64,
    },
    /// 进程不存在。
    NoSuchProcess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlimitDim {
    Turns,
    ToolCalls,
    TokensIn,
    TokensOut,
    CostMicros,
    WallclockTicks,
    ToolCallBytes,
    FsBytes,
}

/// 增量 usage 的补丁。字段为 0 表示不更新。
#[derive(Debug, Clone, Default)]
pub struct ResourceUsageDelta {
    pub turns: u64,
    pub tool_calls: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_micros: u64,
    /// 若 Some，则**覆盖** last_tool_call_bytes（而非累加）。
    pub last_tool_call_bytes: Option<u64>,
    /// VFS 读写字节增量（累加）。
    pub fs_bytes: u64,
}

/// ResourceLimit / Usage 相关 syscall。
pub trait RlimitOps {
    fn rlimit_set(&mut self, pid: u64, limits: ResourceLimit) -> Result<(), String>;
    fn rlimit_get(&self, pid: u64) -> Option<ResourceLimit>;
    fn rusage_get(&self, pid: u64) -> Option<ResourceUsage>;

    /// 原子地把 delta 累计到 pid 的 usage 上，并返回当前是否越界。
    /// 这是推进配额的唯一正确入口 —— 旧的 `increment_turns_used_for` /
    /// `increment_tool_calls_used_for` 在内部也应走到这里。
    fn rusage_charge(&mut self, pid: u64, delta: ResourceUsageDelta) -> RlimitVerdict;

    /// 纯查询：若 delta 会越界则返回 Exceeded；不修改 usage。
    /// 供调用方在执行昂贵操作前预检（例如发送大 prompt 前）。
    fn rlimit_check(&self, pid: u64, delta: &ResourceUsageDelta) -> RlimitVerdict;
}

// =============================================================================
// LLM Device (Phase 2)
// =============================================================================
// 设计目标：把 agent 里散落的"HTTP 请求完成后什么都没记"的问题，替换为
// 一个内核态 LLM 设备。任何 LLM 调用结束时（stream 或 non-stream），
// 解析出来的 usage 都通过 `sys_llm_account` 上报内核；内核负责：
//   1) 结合 `LlmPriceTable` 把 prompt/completion tokens 翻译成 cost_micros
//   2) 原子地把 token 和 cost 折算成 ResourceUsageDelta 并走 rusage_charge
//   3) 在 trace ring 记录一条 llm.account 事件
//
// 这样未来加 quota/caching/speculative decoding 都只改内核，不触碰 agent。

/// 每 1,000 token 的价格（微元 = 1e-6 USD）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmModelPrice {
    /// 输入 token 每千的价格（微元）。
    pub prompt_per_1k_micros: u64,
    /// 输出 token 每千的价格（微元）。
    pub completion_per_1k_micros: u64,
}

impl LlmModelPrice {
    pub const fn zero() -> Self {
        Self {
            prompt_per_1k_micros: 0,
            completion_per_1k_micros: 0,
        }
    }
}

/// 单次 LLM 调用返回的 usage 报告（从 provider 响应中解析得到）。
/// 字段语义与 OpenAI `chat.completions.usage` 对齐。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LlmUsageReport {
    /// provider 返回的模型名称，用于在价格表里查价。
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// cached prompt tokens（如果 provider 支持）。
    /// 目前仅做 trace，不折算 cost。
    pub cached_prompt_tokens: u64,
    /// 本次调用延迟（毫秒），0 表示未知。
    pub latency_ms: u64,
}

/// `sys_llm_account` 的返回值：既告诉调用者本次 cost，也透传 rlimit verdict。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmAccountOutcome {
    pub charged_cost_micros: u64,
    pub verdict: RlimitVerdict,
}

/// LLM 设备接口。`/dev/llm` 的内核态表达。
pub trait LlmOps {
    /// 设置/覆盖某个模型的价格。
    fn llm_set_price(&mut self, model: String, price: LlmModelPrice);

    /// 查询某个模型的价格（未知则返回 zero）。
    fn llm_price(&self, model: &str) -> LlmModelPrice;

    /// 将一次 LLM 调用的 usage 报告计入 pid 的账上：
    ///   1) 折算为 cost_micros（按 llm_price(model)）
    ///   2) 走 rusage_charge 推进 tokens_in/tokens_out/cost_micros
    ///   3) 在 trace ring 写一条 name="llm.account" 的事件
    fn llm_account(&mut self, pid: u64, report: LlmUsageReport) -> LlmAccountOutcome;
}

// --------------------------------------------------------------------------
// VFS —— /dev/vfs （Phase 3）
// --------------------------------------------------------------------------
// 设计说明：
//   - 路径式 API（非 fd 式）。对 agent 这种"每个 tool 调用一次性 I/O"的工作
//     负载，fd 式带来的是无谓的状态管理开销，而路径式天然幂等。
//   - 所有 I/O 通过 VfsOps 入口后：
//       1) 走敏感路径校验（拒绝 /.ssh/ 等）
//       2) 累计读写字节数到 ResourceUsage.fs_bytes（通过 rusage_charge）
//       3) 在 trace ring 写一条 name="vfs.{op}" 的事件
//     agent 侧 FileStore 只负责调参数语义，权限/配额/观测全落内核。

/// VFS 错误类型。从 std::io::Error 脱耦，方便在 trait 边界传递。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VfsError {
    /// 路径命中敏感路径黑名单。
    PermissionDenied(String),
    /// 文件/目录不存在。
    NotFound(String),
    /// 读写字节数超过 rlimit.max_fs_bytes（verdict 为 Exceeded）。
    QuotaExceeded {
        dimension: RlimitDim,
        used: u64,
        limit: u64,
    },
    /// 底层 I/O 失败（保留原 message）。
    Io(String),
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VfsError::PermissionDenied(p) => write!(f, "Access blocked: sensitive path ({})", p),
            VfsError::NotFound(p) => write!(f, "File not found: {}", p),
            VfsError::QuotaExceeded {
                dimension,
                used,
                limit,
            } => {
                write!(
                    f,
                    "VFS quota exceeded ({:?}): {}/{}",
                    dimension, used, limit
                )
            }
            VfsError::Io(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for VfsError {}

/// stat 信息，目前只暴露是否存在 + size，避免跨平台元数据歧义。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfsStat {
    pub size: u64,
    pub is_file: bool,
    pub is_dir: bool,
}

/// VFS 设备接口。`/dev/vfs` 的内核态表达。
pub trait VfsOps {
    /// 读整个文件到字符串。成功时会把字节数 charge 到 pid（若已知）。
    /// pid=None 表示无归属（通常是内核或测试代码）；这种情况下不走 rusage_charge。
    fn vfs_read_to_string(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
    ) -> Result<String, VfsError>;

    /// 写整个文件。会自动创建父目录。
    fn vfs_write_all(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
        content: &str,
    ) -> Result<(), VfsError>;

    /// 查询文件元信息。不计入 fs_bytes。
    fn vfs_stat(&mut self, path: &std::path::Path) -> Result<VfsStat, VfsError>;

    /// 删除文件。不计入 fs_bytes。
    fn vfs_remove_file(&mut self, path: &std::path::Path) -> Result<(), VfsError>;
}

// --------------------------------------------------------------------------
// Daemon —— 后台守护进程登记表（Phase 4）
// --------------------------------------------------------------------------
// 设计说明：
//   - 散落在 agent 代码里的 `tokio::spawn(async move { ... fire-and-forget ... })`
//     属于典型"后台守护进程"语义：无父 await、出错无人知、生命周期跨 turn。
//     这类 task 的可观测性/可控性必须由内核管起来。
//   - DaemonOps 不替代 tokio 执行器；它是一份"登记表 + cancel 协议 + trace 钩子"：
//       spawn_daemon(label, kind) → 分配 handle、记录条目、返回 CancelToken
//       daemon 内部 future 在 await 点轮询 CancelToken，退出时调用 daemon_exit(handle, result)
//     真正的 tokio::spawn 仍然在 agent 用户态发生，内核只做簿记。
//   - 这个边界和 Linux 的 `wait(2)` / `/proc/<pid>` 类似：内核不跑调度器，但它知道
//     每个进程的身份、状态、退出码。
//
//   为什么不是 spawn 完全托管？因为 SharedKernel 是 std::sync::Mutex + dyn Kernel + Send，
//   而 Future 往往需要持有 !Sync 状态（App 等）。让内核去 poll Future 会把锁的时效拉长
//   并引入 Send/Sync 限制污染。保留"agent 侧 tokio::spawn + 内核侧登记"的分工最务实。

/// Daemon 登记 ID（不可伪造）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DaemonHandle(pub u64);

impl DaemonHandle {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for DaemonHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "daemon_{}", self.0)
    }
}

/// Daemon 的语义分类，用于 trace/运维过滤。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonKind {
    /// 反思/critic/revise 等质量提升类。
    Reflection,
    /// 知识抽取 / 压缩 / 回填。
    KnowledgeBuild,
    /// MCP / 外部 I/O 预热。
    IoPreload,
    /// 其他。
    Other,
}

impl DaemonKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DaemonKind::Reflection => "reflection",
            DaemonKind::KnowledgeBuild => "knowledge_build",
            DaemonKind::IoPreload => "io_preload",
            DaemonKind::Other => "other",
        }
    }
}

/// Daemon 生命周期状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonState {
    /// 已登记，尚未退出。
    Running,
    /// 正常退出（Ok）。
    Exited,
    /// 失败退出（Err），错误 message 保存在 DaemonEntry.last_error。
    Failed,
    /// 被 cancel。
    Cancelled,
}

/// Daemon 登记条目（只读视图，供 list_daemons / daemon_status 返回）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonEntrySnapshot {
    pub handle: DaemonHandle,
    pub label: String,
    pub kind: DaemonKind,
    pub state: DaemonState,
    pub parent_pid: Option<u64>,
    pub spawn_tick: u64,
    pub exit_tick: Option<u64>,
    pub last_error: Option<String>,
}

/// 共享的协作式 cancel token：spawn_daemon 返回此 token 给调用方，daemon 内部在
/// await 点用 `load()` 检查是否需要提前退出。kernel 调用 `cancel_daemon` 时 `store(true)`。
#[derive(Debug, Clone)]
pub struct DaemonCancelToken(pub(crate) std::sync::Arc<std::sync::atomic::AtomicBool>);

impl DaemonCancelToken {
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }

    /// 供内核实现者使用；agent 侧不应直接调用。
    pub fn signal_cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Daemon 登记表的内核接口。
///
/// 注意：这与 `Syscall::spawn_daemon`（"产生后台 agent 进程"）是正交的两个概念：
/// - `Syscall::spawn_daemon` 产生真正的内核进程；
/// - `DaemonOps::daemon_register` 只登记一个 agent 用户态 tokio::spawn 出的 future，
///   负责其可观测性与 cancel 协议。
pub trait DaemonOps {
    /// 登记一个 daemon；返回内核分配的 handle 以及该 daemon 独有的 cancel token。
    /// 调用者随后应该 `tokio::spawn` 真正的 future，并在 future 退出时调用 `daemon_exit`。
    fn daemon_register(
        &mut self,
        label: String,
        kind: DaemonKind,
        parent_pid: Option<u64>,
    ) -> (DaemonHandle, DaemonCancelToken);

    /// Daemon 正常/异常退出时由 agent 侧调用，把结果写回登记表并 emit trace 事件。
    /// `err=None` → Exited，`err=Some(_)` → Failed。若此前已被 cancel 则状态保持 Cancelled。
    fn daemon_exit(&mut self, handle: DaemonHandle, err: Option<String>);

    /// 对指定 daemon 标记 cancel（设置其 token + 更新状态为 Cancelled）。
    /// 若 handle 未登记或已退出，返回 false。
    fn cancel_daemon(&mut self, handle: DaemonHandle) -> bool;

    /// 快照查询：某个 daemon 的当前状态。
    fn daemon_status(&self, handle: DaemonHandle) -> Option<DaemonEntrySnapshot>;

    /// 快照枚举：当前所有登记的 daemon（含已退出的，直到 GC）。
    fn list_daemons(&self) -> Vec<DaemonEntrySnapshot>;
}

// --------------------------------------------------------------------------
// IPC Channel / Pipe （Phase 5）
// --------------------------------------------------------------------------
// 设计说明：
//   - 现有 send_ipc/read_mailbox 是"进程邮箱"模型，适合短控制消息；
//     `task_tool` 需要的是"父进程创建结果通道，子进程写入一次完整结果，父进程稍后读取"。
//   - 因此新增点对点 channel primitive，而不是继续滥用 shm：
//       1) ownership 明确（owner_pid 创建并消费）
//       2) queue + capacity 天然支持背压
//       3) trace 事件稳定（ipc.channel_create / send / recv / close）
//   - Phase 5 先实现单接收端语义；发送端按"父子/同组/祖先"规则放行，
//     这已经足够覆盖 task_tool 的父子 agent 通信。

/// IPC channel ID（不可伪造）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(pub u64);

impl ChannelId {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "chan_{}", self.0)
    }
}

/// channel 的用途标签。普通通用 IPC 和 result pipe 显式区分。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOwnerTag {
    General,
    TaskResult,
    AsyncToolResult,
}

impl ChannelOwnerTag {
    pub fn as_str(self) -> &'static str {
        match self {
            ChannelOwnerTag::General => "general",
            ChannelOwnerTag::TaskResult => "task_result",
            ChannelOwnerTag::AsyncToolResult => "async_tool_result",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMetaSnapshot {
    pub channel: ChannelId,
    pub label: String,
    pub owner_pid: Option<u64>,
    pub owner_tag: ChannelOwnerTag,
    pub ref_count: u32,
    pub ref_holders: Vec<String>,
    pub queued_len: usize,
    pub closed: bool,
}

/// 非阻塞接收/窥视的返回结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcRecvResult {
    /// channel 中当前没有消息，但仍可继续接收。
    Empty,
    /// 成功拿到一条消息。
    Message(String),
    /// channel 已关闭，且队列已空。
    Closed,
}

/// 点对点 channel / pipe 原语。
pub trait IpcOps {
    /// 创建一个 channel。owner_pid 为消费端；capacity=0 时按 1 处理。
    fn channel_create(
        &mut self,
        owner_pid: Option<u64>,
        capacity: usize,
        label: String,
    ) -> ChannelId;

    /// 创建带 owner tag / 初始引用计数的 channel。
    /// result pipe 应该通过这个接口显式声明其生命周期模型。
    fn channel_create_tagged(
        &mut self,
        owner_pid: Option<u64>,
        capacity: usize,
        label: String,
        owner_tag: ChannelOwnerTag,
        initial_ref_count: u32,
    ) -> ChannelId;

    /// 创建带 owner tag / 命名 holder 的 channel。
    fn channel_create_tagged_with_holders(
        &mut self,
        owner_pid: Option<u64>,
        capacity: usize,
        label: String,
        owner_tag: ChannelOwnerTag,
        initial_ref_holders: Vec<String>,
    ) -> ChannelId;

    /// 查询 channel 对应的稳定 event id。
    /// 用途：wait_on_events 可以直接等待“该 channel 已进入可读/终态”。
    fn channel_event_id(&self, channel: ChannelId) -> Option<crate::kernel::EventId>;

    /// 查询 channel 元数据快照。
    fn channel_meta(&self, channel: ChannelId) -> Option<ChannelMetaSnapshot>;

    /// 列出当前所有 channel 的元数据快照。
    fn list_channels(&self) -> Vec<ChannelMetaSnapshot>;

    /// 发送一条消息。sender_pid=None 代表 runtime / test 环境。
    fn channel_send(
        &mut self,
        sender_pid: Option<u64>,
        channel: ChannelId,
        message: String,
    ) -> Result<(), String>;

    /// 非阻塞接收：有消息则弹出一条；空则 Empty；已关闭且空则 Closed。
    fn channel_try_recv(
        &mut self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<IpcRecvResult, String>;

    /// 非阻塞窥视：有消息则克隆队首；不消费。
    fn channel_peek(
        &self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<IpcRecvResult, String>;

    /// 非阻塞窥视全部消息：按队列顺序返回当前所有缓冲消息，不消费。
    fn channel_peek_all(
        &self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<Vec<String>, String>;

    /// 非阻塞批量接收：按队列顺序取走当前所有缓冲消息。
    fn channel_try_recv_all(
        &mut self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<Vec<String>, String>;

    /// 增加 channel 引用计数，返回新的引用数。
    fn channel_retain(&mut self, channel: ChannelId) -> Result<u32, String>;

    /// 增加带名字的引用，便于观测引用持有者。
    fn channel_retain_named(&mut self, channel: ChannelId, holder: String) -> Result<u32, String>;

    /// 释放 channel 引用计数，返回新的引用数。
    fn channel_release(&mut self, channel: ChannelId) -> Result<u32, String>;

    /// 释放指定 holder 的引用，返回新的引用数。
    fn channel_release_named(&mut self, channel: ChannelId, holder: &str) -> Result<u32, String>;

    /// 显式销毁一个 channel。
    /// 仅允许销毁已经 `closed` 且队列已空且 `ref_count==0` 的 channel，避免无声丢数据。
    fn channel_destroy(
        &mut self,
        caller_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<(), String>;

    /// GC: 扫描并回收所有 `closed && empty && ref_count==0` 的 channel，返回回收数量。
    fn channel_gc_closed_empty(&mut self) -> usize;

    /// 关闭 channel。关闭后不再允许 send，但剩余队列仍可被 recv/peek。
    fn channel_close(&mut self, closer_pid: Option<u64>, channel: ChannelId) -> Result<(), String>;
}

// --------------------------------------------------------------------------
// Epoll —— 内核态事件多路复用（Phase 6）
// --------------------------------------------------------------------------
// 设计说明：
//   - 现有 wait_on_events 适合一次性等待一组 EventId，但 agent runtime 缺少一个
//     可复用的“兴趣集”对象：注册多个源，反复 wait，像 epoll 一样只返回 ready 子集。
//   - 这里的 epoll 不绑定宿主 OS fd，而是绑定 AIOS 内部可轮询源：
//       1) 原始 EventId
//       2) Channel 可读/关闭状态
//       3) Futex 的“值不再等于 expected”或“seq 被 wake 推进”
//   - 实现仍是同步内核态对象；真正 async 阻塞继续通过 wait_on_events 完成，
//     因此不需要 tokio，也不依赖第三方库。

/// epoll 实例 ID（不可伪造）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpollId(pub u64);

impl EpollId {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for EpollId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "epoll_{}", self.0)
    }
}

/// epoll 关注的事件位。保持极小集合即可满足当前 agent 协调需求。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpollEventMask(pub u32);

impl EpollEventMask {
    pub const EMPTY: Self = Self(0);
    pub const IN: Self = Self(1 << 0);
    pub const HUP: Self = Self(1 << 1);
    pub const ERR: Self = Self(1 << 2);

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl std::ops::BitOr for EpollEventMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for EpollEventMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for EpollEventMask {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

/// epoll 可关注的源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpollSource {
    Event(crate::kernel::EventId),
    Channel(ChannelId),
    Futex { addr: FutexAddr, expected: u64 },
}

/// 一条 epoll 注册项的快照。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpollRegistrationSnapshot {
    pub source: EpollSource,
    pub events: EpollEventMask,
    pub user_data: u64,
}

/// 一条 ready 结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpollReadyEvent {
    pub source: EpollSource,
    pub events: EpollEventMask,
    pub user_data: u64,
}

/// epoll wait 的返回值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpollWaitResult {
    Ready(Vec<EpollReadyEvent>),
    Suspended { timeout_tick: Option<u64> },
}

/// epoll 元数据快照。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpollSnapshot {
    pub epoll: EpollId,
    pub label: String,
    pub registrations: Vec<EpollRegistrationSnapshot>,
}

/// epoll 相关 syscall。
pub trait EpollOps {
    fn epoll_create(&mut self, label: String) -> EpollId;
    fn epoll_ctl_add(
        &mut self,
        epoll: EpollId,
        source: EpollSource,
        events: EpollEventMask,
        user_data: u64,
    ) -> Result<(), String>;
    fn epoll_ctl_mod(
        &mut self,
        epoll: EpollId,
        source: EpollSource,
        events: EpollEventMask,
        user_data: u64,
    ) -> Result<(), String>;
    fn epoll_ctl_del(&mut self, epoll: EpollId, source: EpollSource) -> Result<(), String>;
    fn epoll_wait(
        &mut self,
        epoll: EpollId,
        max_events: usize,
        timeout_ticks: Option<u64>,
    ) -> Result<EpollWaitResult, String>;
    fn epoll_snapshot(&self, epoll: EpollId) -> Option<EpollSnapshot>;
    fn epoll_destroy(&mut self, epoll: EpollId) -> bool;
}
