// =============================================================================
// AIOS Primitives - Futex & Trace
// =============================================================================
// This file adds two new primitives to AIOS so agents no longer have to hand-roll
// synchronization / instrumentation in user space:
//
//   1. Futex — a generic "condition variable + counter". Used for cancel signals,
//      waking up suspended streaming I/O, and any "wait until a condition holds"
//      scenario. Replaces the scattered AtomicBools in agents.
//
//   2. Trace — an in-kernel tracing ring buffer. Every span / event is persisted
//      through it, replacing the agent_hang_span macro. Downstream drivers consume
//      it for output / OTel / hang detection.
//
// Design constraints:
//   - Don't break the existing Syscall / KernelInternal traits; add as an
//     independent trait on Kernel.
//   - A synchronous LocalOS implementation suffices; don't touch tokio. Async
//     waiting is wrapped agent-side (since SharedKernel is currently a
//     std::sync::Mutex, awaiting while holding the lock is an anti-pattern).
//     Futex wait semantics are implemented via "obtain a waker token, release
//     the lock, then block", but phase-0 provides a synchronous interface plus
//     a non-blocking try_wait, so the agent side can poll to validate the design.
// =============================================================================

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::FastMap;

// --------------------------------------------------------------------------
// Futex
// --------------------------------------------------------------------------

/// Futex address: an unforgeable 64-bit handle allocated by the kernel.
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

/// Reason a futex wait returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FutexWakeReason {
    /// Explicitly woken by `futex_wake`.
    Woken,
    /// At `futex_wait` time the value already differed from expected (fast path, no real blocking needed).
    ValueChanged,
    /// Interrupted by a cancel signal / SIGCANCEL.
    Cancelled,
    /// No such futex address.
    NotFound,
}

/// Futex state: current value + total wake count (for wait's "expected" semantics).
#[derive(Debug)]
pub(super) struct FutexState {
    pub(super) value: AtomicU64,
    /// FIFO queue of PIDs waiting on this futex.
    pub(super) waiters: VecDeque<u64>,
    /// Incremented on each wake; wait compares the seq before and after to tell whether it was woken.
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

/// Futex-related syscalls.
pub trait FutexOps {
    /// Create a futex and return its handle. The label is used only for diagnostics / trace.
    fn futex_create(&mut self, initial: u64, label: String) -> FutexAddr;

    /// Read the current value.
    fn futex_load(&self, addr: FutexAddr) -> Option<u64>;

    /// CAS update. Returns Ok(old value) on success, Err(current value) on failure.
    fn futex_cas(&mut self, addr: FutexAddr, expected: u64, new_value: u64) -> Result<u64, u64>;

    /// Atomically add delta and return the old value.
    fn futex_fetch_add(&mut self, addr: FutexAddr, delta: u64) -> Option<u64>;

    /// Store a new value and return the old one.
    fn futex_store(&mut self, addr: FutexAddr, new_value: u64) -> Option<u64>;

    /// Non-blocking check: if value != expected, immediately return ValueChanged; if equal,
    /// return None to indicate that external waiting is needed.
    /// Returning Some(reason) means no further blocking is required.
    fn futex_try_wait(&self, addr: FutexAddr, expected: u64) -> Option<FutexWakeReason>;

    /// Wake n waiters. Returns the number actually woken.
    fn futex_wake(&mut self, addr: FutexAddr, n: usize) -> usize;

    /// Destroy the futex. Waiters will observe NotFound.
    fn futex_destroy(&mut self, addr: FutexAddr) -> bool;

    /// Register a pid on this futex's wait queue (for kernel-internal wakeups).
    /// Returns a seq snapshot taken at registration, for detecting missed wakeups later.
    fn futex_register_waiter(&mut self, addr: FutexAddr, pid: u64) -> Option<u64>;

    /// Cancel the wait (pid no longer waits on this futex).
    fn futex_cancel_waiter(&mut self, addr: FutexAddr, pid: u64) -> bool;

    /// Read the current seq, for wait's "were we woken since seq0" semantics.
    fn futex_seq(&self, addr: FutexAddr) -> Option<u64>;

    /// Read the kernel event ID associated with this futex, so multiplexed waits can be
    /// reduced to a uniform event set.
    fn futex_event_id(&self, addr: FutexAddr) -> Option<crate::kernel::EventId>;
}

// --------------------------------------------------------------------------
// Trace
// --------------------------------------------------------------------------

/// Kernel trace event level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// A single trace record. Structured fields are kept together in `fields` rather than
/// scattered across many individual fields.
#[derive(Debug, Clone)]
pub struct TraceRecord {
    pub seq: u64,
    pub tick: u64,
    pub pid: Option<u64>,
    pub level: TraceLevel,
    /// Stable span/event name, e.g. `turn_runtime::run_turn`.
    pub name: String,
    /// span_id of this record (shared by a span's enter/exit/event).
    pub span_id: Option<u64>,
    /// Parent span_id, used to reconstruct parent-child relationships.
    pub parent_span_id: Option<u64>,
    /// Event kind: span_enter / span_exit / event.
    pub kind: TraceKind,
    /// Structured fields (key -> JSON-like string). `None` means no fields, avoiding an extra
    /// raw_table header for an empty HashMap on every record. Prefer [`TraceRecord::fields`] when reading.
    pub fields: Option<FastMap<String, String>>,
    pub message: Option<String>,
}

impl TraceRecord {
    /// Return a reference to fields; empty fields uniformly yield `None`, so callers can
    /// simplify with `.unwrap_or(&EMPTY)`.
    pub fn fields(&self) -> Option<&FastMap<String, String>> {
        self.fields.as_ref()
    }

    /// Box a possibly-empty fields HashMap into its storage form: empty maps become None.
    pub(super) fn pack_fields(fields: FastMap<String, String>) -> Option<FastMap<String, String>> {
        if fields.is_empty() {
            None
        } else {
            Some(fields)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceKind {
    SpanEnter,
    SpanExit,
    Event,
}

/// Trace-related syscalls.
pub trait TraceOps {
    /// Create a span and return its span_id. parent=None means a root span.
    fn trace_span_enter(
        &mut self,
        name: String,
        parent: Option<u64>,
        fields: FastMap<String, String>,
    ) -> u64;

    /// Close a span (writes a SpanExit record).
    fn trace_span_exit(&mut self, span_id: u64, fields: FastMap<String, String>);

    /// Emit a standalone event.
    fn trace_event(
        &mut self,
        name: String,
        level: TraceLevel,
        span_id: Option<u64>,
        fields: FastMap<String, String>,
        message: Option<String>,
    );

    /// Read the most recent N trace records (newest first).
    fn trace_recent(&self, n: usize) -> Vec<TraceRecord>;

    /// All trace records from `since_seq` onward (ascending). Used for external draining.
    fn trace_drain_since(&self, since_seq: u64) -> Vec<TraceRecord>;

    /// Current head seq (used as the drain cursor).
    fn trace_head_seq(&self) -> u64;

    /// Set the ring buffer capacity (excess drops the oldest records).
    fn trace_set_capacity(&mut self, cap: usize);
}

/// Trace ring buffer.
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
// Helpers: convenient construction of a fields map (for agent-side use)
// --------------------------------------------------------------------------
pub fn fields() -> FastMap<String, String> {
    FastMap::default()
}

#[doc(hidden)]
pub fn _field_insert<V: std::fmt::Display>(map: &mut FastMap<String, String>, key: &str, value: V) {
    map.insert(key.to_string(), value.to_string());
}

/// Convenience macro: `trace_fields!{"foo" => 1, "bar" => "baz"}` returns a FastMap<String,String>.
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

// Make Ordering available in implementations (in case a use site forgets the import)
#[allow(dead_code)]
pub(super) const FUTEX_ORDER: Ordering = Ordering::SeqCst;

// --------------------------------------------------------------------------
// ResourceLimit / ResourceUsage — cgroup-like resource quotas
// --------------------------------------------------------------------------

/// Per-process resource limits. `u64::MAX` means unlimited.
///
/// Design principle: all quotas live in the kernel; agents should not maintain
/// max_iterations constants in user space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimit {
    /// Cap on LLM turns (replaces Process.quota_turns; kept in sync for a smooth transition).
    pub max_turns: u64,
    /// Cap on the number of tool calls.
    pub max_tool_calls: u64,
    /// Cap on cumulative prompt tokens.
    pub max_tokens_in: u64,
    /// Cap on cumulative completion tokens.
    pub max_tokens_out: u64,
    /// Cap on cumulative cost (cents / micro-dollars; the exact unit is decided by the LLM device).
    pub max_cost_micros: u64,
    /// Cap on wall-clock ticks: created_at_tick + max_wallclock_ticks acts as the deadline.
    pub max_wallclock_ticks: u64,
    /// Cap on the byte size of a single tool-call return body (prevents huge outputs from
    /// blowing up the context).
    pub max_tool_call_bytes: u64,
    /// Cap on bytes read/written via VfsOps (disk I/O quota outside /dev/llm).
    pub max_fs_bytes: u64,
}

impl ResourceLimit {
    /// Everything unlimited. Used for backward compatibility with the old behavior.
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

    /// Build a limit from the legacy `quota_turns: usize` field: only turns are constrained,
    /// everything else is unlimited.
    /// 0 is treated as "unlimited" per the legacy semantics.
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

/// A process's cumulative resource usage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceUsage {
    pub turns: u64,
    pub tool_calls: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_micros: u64,
    /// Monotonic counter: last_tool_call_bytes is the size of the most recent tool return
    /// body (for observability).
    pub last_tool_call_bytes: u64,
    /// Cumulative bytes read/written via VfsOps.
    pub fs_bytes: u64,
}

/// Quota check result. The kernel returns this when advancing usage; callers use it to decide
/// whether to terminate the process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RlimitVerdict {
    /// Within limits.
    Ok,
    /// Exceeded, with the specific dimension.
    Exceeded {
        dimension: RlimitDim,
        used: u64,
        limit: u64,
    },
    /// No such process.
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

/// A delta patch to usage. A field of 0 means no update.
#[derive(Debug, Clone, Default)]
pub struct ResourceUsageDelta {
    pub turns: u64,
    pub tool_calls: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_micros: u64,
    /// If Some, **overwrites** last_tool_call_bytes (rather than accumulating).
    pub last_tool_call_bytes: Option<u64>,
    /// VFS read/write byte delta (accumulated).
    pub fs_bytes: u64,
}

/// ResourceLimit / Usage-related syscalls.
pub trait RlimitOps {
    fn rlimit_set(&mut self, pid: u64, limits: ResourceLimit) -> Result<(), String>;
    fn rlimit_get(&self, pid: u64) -> Option<ResourceLimit>;
    fn rusage_get(&self, pid: u64) -> Option<ResourceUsage>;

    /// Atomically accumulate delta onto pid's usage and report whether limits are now exceeded.
    /// This is the only correct entry point for advancing quotas — the legacy
    /// `increment_turns_used_for` / `increment_tool_calls_used_for` should also route through
    /// it internally.
    fn rusage_charge(&mut self, pid: u64, delta: ResourceUsageDelta) -> RlimitVerdict;

    /// Pure query: returns Exceeded if delta would cross a limit; does not modify usage.
    /// Lets callers pre-check before an expensive operation (e.g. before sending a large prompt).
    fn rlimit_check(&self, pid: u64, delta: &ResourceUsageDelta) -> RlimitVerdict;
}

// =============================================================================
// LLM Device (Phase 2)
// =============================================================================
// Design goal: replace the scattered agent-side problem of "nothing is recorded after an
// HTTP request completes" with an in-kernel LLM device. When any LLM call finishes
// (stream or non-stream), the parsed usage is reported to the kernel via `sys_llm_account`;
// the kernel is responsible for:
//   1) converting prompt/completion tokens into cost_micros using `LlmPriceTable`
//   2) atomically folding tokens and cost into a ResourceUsageDelta and charging via rusage_charge
//   3) recording an llm.account event in the trace ring
//
// This way, future quota/caching/speculative-decoding features only touch the kernel, not the agent.

/// Price per 1,000 tokens (micro-dollars = 1e-6 USD).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmModelPrice {
    /// Price per 1k input tokens (micro-dollars).
    pub prompt_per_1k_micros: u64,
    /// Price per 1k output tokens (micro-dollars).
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

/// Usage report returned by a single LLM call (parsed from the provider response).
/// Field semantics align with OpenAI `chat.completions.usage`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LlmUsageReport {
    /// Model name returned by the provider, used to look up the price table.
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Subset of `completion_tokens` that belongs to hidden reasoning/thinking.
    pub reasoning_tokens: u64,
    /// Cached prompt tokens (if supported by the provider).
    /// Currently only traced, not converted into cost.
    pub cached_prompt_tokens: u64,
    /// Latency of this call (milliseconds); 0 means unknown.
    pub latency_ms: u64,
}

/// Return value of `sys_llm_account`: reports this call's cost to the caller and passes
/// through the rlimit verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmAccountOutcome {
    pub charged_cost_micros: u64,
    pub verdict: RlimitVerdict,
}

/// An LLM usage audit record. On every `llm_account` settlement the kernel appends one to
/// the bounded ledger, which the agent drains into its own database (a separate SQLite table).
/// This embodies "auditing is provided by the OS".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmUsageRecord {
    /// Monotonic sequence number used as the drain cursor (independent of trace's seq).
    pub seq: u64,
    /// Kernel logical clock at settlement time (scheduler tick), not wall-clock time.
    pub tick: u64,
    /// Process that made the call.
    pub pid: u64,
    /// Model name returned by the provider.
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Subset of `completion_tokens` that belongs to hidden reasoning/thinking.
    pub reasoning_tokens: u64,
    /// Total token count (prompt + completion).
    pub total_tokens: u64,
    /// Cached prompt tokens (if supported by the provider).
    pub cached_prompt_tokens: u64,
    /// Latency of this call (milliseconds); 0 means unknown.
    pub latency_ms: u64,
    /// Cost converted for this call (micro-dollars).
    pub cost_micros: u64,
}

/// LLM device interface. The in-kernel representation of `/dev/llm`.
pub trait LlmOps {
    /// Set or override the price of a model.
    fn llm_set_price(&mut self, model: String, price: LlmModelPrice);

    /// Query a model's price (unknown models return zero).
    fn llm_price(&self, model: &str) -> LlmModelPrice;

    /// Charge one LLM call's usage report to pid's account:
    ///   1) convert it to cost_micros (via llm_price(model))
    ///   2) advance tokens_in/tokens_out/cost_micros through rusage_charge
    ///   3) write a name="llm.account" event into the trace ring
    ///   4) append a [`LlmUsageRecord`] to the bounded LLM usage ledger (for external drain / persistence)
    fn llm_account(&mut self, pid: u64, report: LlmUsageReport) -> LlmAccountOutcome;

    /// All LLM usage records from `since_seq` onward (ascending). Used for external drain /
    /// persistence.
    /// Returned records have seq strictly greater than `since_seq`.
    fn llm_usage_drain_since(&self, since_seq: u64) -> Vec<LlmUsageRecord>;

    /// Current head seq of the ledger (for initializing / aligning the drain cursor).
    fn llm_usage_head_seq(&self) -> u64;

    /// Set the LLM usage ledger's ring buffer capacity (excess drops the oldest records).
    fn llm_usage_set_capacity(&mut self, cap: usize);
}

/// LLM usage audit ledger: a bounded ring buffer following the same pattern as [`TraceRing`].
#[derive(Debug)]
pub(super) struct LlmUsageRing {
    pub(super) buf: VecDeque<LlmUsageRecord>,
    pub(super) capacity: usize,
    pub(super) next_seq: u64,
}

impl LlmUsageRing {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity.min(4096)),
            capacity,
            next_seq: 1,
        }
    }

    pub(super) fn alloc_seq(&mut self) -> u64 {
        let s = self.next_seq;
        self.next_seq += 1;
        s
    }

    pub(super) fn push(&mut self, rec: LlmUsageRecord) {
        if self.capacity == 0 {
            return;
        }
        while self.buf.len() >= self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(rec);
    }

    /// Records strictly after `since_seq`, in ascending order.
    pub(super) fn drain_since(&self, since_seq: u64) -> Vec<LlmUsageRecord> {
        self.buf
            .iter()
            .filter(|r| r.seq > since_seq)
            .cloned()
            .collect()
    }

    /// The latest allocated seq so far (0 when no records exist).
    pub(super) fn head_seq(&self) -> u64 {
        self.next_seq.saturating_sub(1)
    }

    pub(super) fn set_capacity(&mut self, cap: usize) {
        self.capacity = cap;
        while self.buf.len() > self.capacity {
            self.buf.pop_front();
        }
    }
}

// --------------------------------------------------------------------------
// VFS — /dev/vfs (Phase 3)
// --------------------------------------------------------------------------
// Design notes:
//   - Path-based API (not fd-based). For agent workloads where each tool call does one-shot
//     I/O, fd-based APIs impose needless state-management overhead, while path-based APIs
//     are naturally idempotent.
//   - All I/O enters through VfsOps, which:
//       1) runs sensitive-path validation (rejects /.ssh/ etc.)
//       2) accumulates read/write bytes into ResourceUsage.fs_bytes (via rusage_charge)
//       3) writes a name="vfs.{op}" event into the trace ring
//     The agent-side FileStore only handles call argument semantics; permissions, quotas,
//     and observability all live in the kernel.

/// VFS error type. Decoupled from std::io::Error so it can cross trait boundaries easily.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VfsError {
    /// The path hit the sensitive-path blocklist.
    PermissionDenied(String),
    /// File or directory does not exist.
    NotFound(String),
    /// Bytes read/written exceeded rlimit.max_fs_bytes (verdict was Exceeded).
    QuotaExceeded {
        dimension: RlimitDim,
        used: u64,
        limit: u64,
    },
    /// Underlying I/O failure (original message preserved).
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

/// stat info; currently only exposes size and existence, to avoid cross-platform metadata
/// ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfsStat {
    pub size: u64,
    pub is_file: bool,
    pub is_dir: bool,
}

/// VFS device interface. The in-kernel representation of `/dev/vfs`.
pub trait VfsOps {
    /// Read the entire file into a string. On success, the byte count is charged to pid
    /// (if known).
    /// pid=None means no owner (usually kernel or test code); in that case rusage_charge is skipped.
    fn vfs_read_to_string(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
    ) -> Result<String, VfsError>;

    /// Write the entire file. Parent directories are created automatically.
    fn vfs_write_all(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
        content: &str,
    ) -> Result<(), VfsError>;

    /// Query file metadata. Not counted toward fs_bytes.
    fn vfs_stat(&mut self, path: &std::path::Path) -> Result<VfsStat, VfsError>;

    /// Delete a file. Not counted toward fs_bytes.
    fn vfs_remove_file(&mut self, path: &std::path::Path) -> Result<(), VfsError>;
}

// --------------------------------------------------------------------------
// Daemon — background daemon registry (Phase 4)
// --------------------------------------------------------------------------
// Design notes:
//   - `tokio::spawn(async move { ... fire-and-forget ... })` scattered through agent code is
//     the classic "background daemon" pattern: no parent awaits it, nobody learns of
//     failures, and its lifetime spans turns. The observability / controllability of such
//     tasks must be managed by the kernel.
//   - DaemonOps does not replace the tokio executor; it is a "registry + cancel protocol +
//     trace hooks":
//       spawn_daemon(label, kind) → allocate a handle, record an entry, return a CancelToken
//       the daemon's inner future polls the CancelToken at await points and calls
//       daemon_exit(handle, result) on exit
//     The actual tokio::spawn still happens in agent user space; the kernel only does
//     bookkeeping.
//   - This boundary mirrors Linux `wait(2)` / `/proc/<pid>`: the kernel does not run the
//     scheduler, but it knows every process's identity, state, and exit code.
//
//   Why not fully managed spawn? Because SharedKernel is std::sync::Mutex + dyn Kernel + Send,
//   while futures often need to hold !Sync state (App, etc.). Having the kernel poll futures
//   would lengthen lock hold times and leak Send/Sync constraints. Keeping the split of
//   "agent-side tokio::spawn + kernel-side registration" is the most pragmatic choice.

/// Daemon registration ID (unforgeable).
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

/// Semantic classification of a daemon, used for trace / ops filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonKind {
    /// Quality-improvement tasks such as reflection / critic / revise.
    Reflection,
    /// Knowledge extraction / compression / backfill.
    KnowledgeBuild,
    /// MCP / external I/O preloading.
    IoPreload,
    /// Other.
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

/// Daemon lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonState {
    /// Registered, not yet exited.
    Running,
    /// Exited normally (Ok).
    Exited,
    /// Exited with failure (Err); the error message is kept in DaemonEntry.last_error.
    Failed,
    /// Was cancelled.
    Cancelled,
}

/// Daemon registration entry (read-only view, returned by list_daemons / daemon_status).
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

/// Shared cooperative cancel token: spawn_daemon returns this token to the caller, and the
/// daemon checks it with `load()` at await points to decide whether to exit early. The kernel
/// calls `store(true)` on it when `cancel_daemon` is invoked.
#[derive(Debug, Clone)]
pub struct DaemonCancelToken(pub(crate) std::sync::Arc<std::sync::atomic::AtomicBool>);

impl DaemonCancelToken {
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }

    /// For kernel implementors; agent-side code should not call it directly.
    pub fn signal_cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Kernel interface for the daemon registry.
///
/// Note: this is orthogonal to `Syscall::spawn_daemon` ("spawn a background agent process"):
/// - `Syscall::spawn_daemon` creates a real kernel process;
/// - `DaemonOps::daemon_register` merely registers a future spawned via tokio::spawn in agent
///   user space, taking care of its observability and cancel protocol.
pub trait DaemonOps {
    /// Register a daemon; returns a kernel-allocated handle plus a cancel token unique to it.
    /// The caller should then `tokio::spawn` the actual future and call `daemon_exit` when it
    /// finishes.
    fn daemon_register(
        &mut self,
        label: String,
        kind: DaemonKind,
        parent_pid: Option<u64>,
    ) -> (DaemonHandle, DaemonCancelToken);

    /// Called by the agent side when a daemon exits (normally or not) to write the result
    /// back to the registry and emit a trace event.
    /// `err=None` → Exited, `err=Some(_)` → Failed. If it was cancelled earlier, the state
    /// stays Cancelled.
    fn daemon_exit(&mut self, handle: DaemonHandle, err: Option<String>);

    /// Mark a daemon as cancelled (set its token + update state to Cancelled).
    /// Returns false if the handle was never registered or has already exited.
    fn cancel_daemon(&mut self, handle: DaemonHandle) -> bool;

    /// Snapshot query: the current state of a daemon.
    fn daemon_status(&self, handle: DaemonHandle) -> Option<DaemonEntrySnapshot>;

    /// Snapshot enumeration: all currently registered daemons (including exited ones, until GC).
    fn list_daemons(&self) -> Vec<DaemonEntrySnapshot>;
}

// --------------------------------------------------------------------------
// IPC Channel / Pipe (Phase 5)
// --------------------------------------------------------------------------
// Design notes:
//   - The existing send_ipc/read_mailbox use a "process mailbox" model suited to short
//     control messages; `task_tool` needs "the parent creates a result channel, the child
//     writes one complete result, and the parent reads it later".
//   - Hence a point-to-point channel primitive is added instead of continuing to abuse shm:
//       1) clear ownership (owner_pid creates and consumes)
//       2) queue + capacity give natural backpressure
//       3) stable trace events (ipc.channel_create / send / recv / close)
//   - Phase 5 implements single-receiver semantics; senders are admitted per
//     "parent-child / same-group / ancestor" rules, which already covers task_tool's
//     parent-child agent communication.

/// IPC channel ID (unforgeable).
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

/// Purpose tag of a channel, explicitly distinguishing generic IPC from result pipes.
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

/// Result of a non-blocking receive / peek.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcRecvResult {
    /// No messages in the channel right now, but receiving is still possible.
    Empty,
    /// Successfully retrieved one message.
    Message(String),
    /// The channel is closed and its queue is empty.
    Closed,
}

/// Point-to-point channel / pipe primitive.
pub trait IpcOps {
    /// Create a channel. owner_pid is the consumer; capacity=0 is treated as 1.
    fn channel_create(
        &mut self,
        owner_pid: Option<u64>,
        capacity: usize,
        label: String,
    ) -> ChannelId;

    /// Create a channel with an owner tag / initial reference count.
    /// Result pipes should use this interface to explicitly declare their lifecycle model.
    fn channel_create_tagged(
        &mut self,
        owner_pid: Option<u64>,
        capacity: usize,
        label: String,
        owner_tag: ChannelOwnerTag,
        initial_ref_count: u32,
    ) -> ChannelId;

    /// Create a channel with an owner tag / named holders.
    fn channel_create_tagged_with_holders(
        &mut self,
        owner_pid: Option<u64>,
        capacity: usize,
        label: String,
        owner_tag: ChannelOwnerTag,
        initial_ref_holders: Vec<String>,
    ) -> ChannelId;

    /// Query the stable event id associated with a channel.
    /// Use case: wait_on_events can directly wait for the channel to become readable /
    /// reach a terminal state.
    fn channel_event_id(&self, channel: ChannelId) -> Option<crate::kernel::EventId>;

    /// Query a channel's metadata snapshot.
    fn channel_meta(&self, channel: ChannelId) -> Option<ChannelMetaSnapshot>;

    /// List metadata snapshots for all current channels.
    fn list_channels(&self) -> Vec<ChannelMetaSnapshot>;

    /// Send a message. sender_pid=None indicates a runtime / test environment.
    fn channel_send(
        &mut self,
        sender_pid: Option<u64>,
        channel: ChannelId,
        message: String,
    ) -> Result<(), String>;

    /// Non-blocking receive: pops one message if present; Empty if none; Closed if closed and empty.
    fn channel_try_recv(
        &mut self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<IpcRecvResult, String>;

    /// Non-blocking peek: clones the head message if present; does not consume.
    fn channel_peek(
        &self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<IpcRecvResult, String>;

    /// Non-blocking peek of all messages: returns every currently buffered message in queue
    /// order without consuming.
    fn channel_peek_all(
        &self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<Vec<String>, String>;

    /// Non-blocking batch receive: takes all currently buffered messages in queue order.
    fn channel_try_recv_all(
        &mut self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<Vec<String>, String>;

    /// Increment a channel's reference count, returning the new count.
    fn channel_retain(&mut self, channel: ChannelId) -> Result<u32, String>;

    /// Increment a named reference, making reference holders observable.
    fn channel_retain_named(&mut self, channel: ChannelId, holder: String) -> Result<u32, String>;

    /// Decrement a channel's reference count, returning the new count.
    fn channel_release(&mut self, channel: ChannelId) -> Result<u32, String>;

    /// Release a named holder's reference, returning the new count.
    fn channel_release_named(&mut self, channel: ChannelId, holder: &str) -> Result<u32, String>;

    /// Explicitly destroy a channel.
    /// Only channels that are `closed`, have an empty queue, and have `ref_count==0` may be
    /// destroyed, to avoid silently dropping data.
    fn channel_destroy(
        &mut self,
        caller_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<(), String>;

    /// GC: scan and reclaim every channel with `closed && empty && ref_count==0`, returning
    /// the number reclaimed.
    fn channel_gc_closed_empty(&mut self) -> usize;

    /// Close a channel. After closing, sends are rejected but the remaining queue can still
    /// be recv'd / peeked.
    fn channel_close(&mut self, closer_pid: Option<u64>, channel: ChannelId) -> Result<(), String>;
}

// --------------------------------------------------------------------------
// Epoll — in-kernel event multiplexing (Phase 6)
// --------------------------------------------------------------------------
// Design notes:
//   - wait_on_events is fine for waiting on a one-off set of EventIds, but the agent runtime
//     lacks a reusable "interest set" object: register multiple sources, wait repeatedly, and
//     get only the ready subset back, like epoll.
//   - This epoll does not bind host OS fds; it binds AIOS-internal pollable sources:
//       1) raw EventIds
//       2) Channel readable / closed states
//       3) Futex "value no longer equals expected" or "seq advanced by a wake"
//   - The implementation remains a synchronous kernel-side object; real async blocking still
//     goes through wait_on_events, so no tokio or third-party dependencies are needed.

/// epoll instance ID (unforgeable).
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

/// Event bits an epoll can watch. A minimal set suffices for current agent coordination needs.
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

/// Sources an epoll can watch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpollSource {
    Event(crate::kernel::EventId),
    Channel(ChannelId),
    Futex { addr: FutexAddr, expected: u64 },
}

/// Snapshot of one epoll registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpollRegistrationSnapshot {
    pub source: EpollSource,
    pub events: EpollEventMask,
    pub user_data: u64,
}

/// One ready result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpollReadyEvent {
    pub source: EpollSource,
    pub events: EpollEventMask,
    pub user_data: u64,
}

/// Return value of an epoll wait.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpollWaitResult {
    Ready(Vec<EpollReadyEvent>),
    Suspended { timeout_tick: Option<u64> },
}

/// epoll metadata snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpollSnapshot {
    pub epoll: EpollId,
    pub label: String,
    pub registrations: Vec<EpollRegistrationSnapshot>,
}

/// epoll-related syscalls.
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
