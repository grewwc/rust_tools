// =============================================================================
// AIOS LocalOS - Local Process OS Implementation
// =============================================================================
// This module implements the Kernel trait for a single-machine process OS.
//
// Key features:
//   - Process table: HashMap of all processes (pid -> Process)
//   - Ready queue: FIFO/priority queue of ready processes
//   - Wait queue: HashMap of blocked processes waiting on each pid
//   - Tick counter: Scheduler time for sleeping processes
//   - Shared memory: Key-value store with ownership
//   - Process groups: Signal broadcasting
//
// Scheduling:
//   - pop_ready(): Get highest-priority ready process
//   - advance_tick(): Increment tick, wake sleeping processes
//   - Sleeping processes wake when tick >= their until_tick
//
// Process state transitions:
//   - spawn() -> ready queue (Ready)
//   - wait_on(pid) -> wait queue (Waiting)
//   - terminate() -> waiters become Ready
//   - sleep_current(N) -> sleeping until tick+N
//   - receive SIGSTOP -> Stopped
//   - receive SIGCONT -> Ready
// =============================================================================

use crate::types::{FastMap, FastSet};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use crate::kernel::{
    DEFAULT_MAILBOX_CAPACITY, EventId, Kernel, KernelInternal, Process, ProcessCapabilities,
    ProcessState, ShmReadError, Signal, Syscall, WaitPolicy, WaitReason,
};
use crate::primitives::{
    ChannelId, ChannelMetaSnapshot, ChannelOwnerTag, DaemonCancelToken, DaemonEntrySnapshot,
    DaemonHandle, DaemonKind, DaemonOps, DaemonState, EpollEventMask, EpollId, EpollOps,
    EpollReadyEvent, EpollRegistrationSnapshot, EpollSnapshot, EpollSource, EpollWaitResult,
    FutexAddr, FutexOps, FutexState, FutexWakeReason, IpcOps, IpcRecvResult, LlmAccountOutcome,
    LlmModelPrice, LlmOps, LlmUsageReport, ResourceLimit, ResourceUsage, ResourceUsageDelta,
    RlimitDim, RlimitOps, RlimitVerdict, TraceKind, TraceLevel, TraceOps, TraceRecord, TraceRing,
    VfsError, VfsOps, VfsStat,
};

const DEFAULT_COMPLETED_EVENT_RETENTION: usize = 8192;
const DEFAULT_TRACE_CAPACITY: usize = 4096;
const SHM_PERM_CACHE_SLOTS: usize = 32;

/// Cached SHM permission decision. Stored in a direct-mapped slot keyed by
/// `(current_pid ^ owner_pid)` modulo `SHM_PERM_CACHE_SLOTS`. The `version`
/// field is matched against `LocalOS::topology_version`; mismatches force a
/// fresh evaluation, which acts as the invalidation strategy.
#[derive(Clone, Copy)]
pub(super) struct ShmPermCacheEntry {
    pub(super) version: u64,
    pub(super) current_pid: u64,
    pub(super) owner_pid: u64,
    pub(super) accessible: bool,
    pub(super) readable: bool,
}

pub(super) struct ShmEntry {
    value: String,
    owner_pid: u64,
    checksum: u64,
    version: u64,
}

fn shm_checksum(value: &str, owner_pid: u64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    owner_pid.hash(&mut hasher);
    hasher.finish()
}

pub struct LocalOS {
    pub processes: FastMap<u64, Process>,
    /// 就绪队列，保存 `(pid, priority)`。优先级随条目内联存储，避免每次入队比较
    /// 都回表查 `Process.priority`。出队 / 调度过滤通过 `ready_set` 进行 O(1)
    /// 标记，因此终止 / 信号停止等路径不再需要线性 retain。
    pub(super) ready_queue: VecDeque<(u64, u8)>,
    /// `ready_queue` 的成员索引。一个 pid 是否真正“可被调度”以这个集合为准；
    /// 队列里的条目可能是过期 tombstone，由 [`pop_ready`] 丢弃。
    pub(super) ready_set: FastSet<u64>,
    pub(super) wait_queue: HashMap<u64, Vec<u64>>,
    /// Reverse index: parent_pid -> set of child pids. Maintained on spawn /
    /// remove so that descendant traversal and orphan reassignment do not
    /// require a full process-table scan.
    pub(super) children_by_parent: FastMap<u64, FastSet<u64>>,
    pub next_pid: u64,
    pub current_pid: Option<u64>,
    pub(super) yield_requested: bool,
    pub tick: u64,
    pub(super) round_robin: bool,
    pub(super) shared_memory: FastMap<String, ShmEntry>,
    pub next_pgid: u64,
    /// All event IDs that have ever been marked completed, used to detect
    /// already-satisfied wait conditions in wait_on_events.
    pub(super) completed_events: HashSet<EventId>,
    pub(super) completed_event_order: VecDeque<EventId>,
    pub(super) completed_event_retention: usize,
    /// Reverse index: event_id -> set of pids currently waiting on that event.
    /// Cleaned up lazily on notify (entries for completed events are removed)
    /// and verified against process state at wake-time, so stale pids are safe.
    pub(super) event_waiters: FastMap<EventId, FastSet<u64>>,
    /// Refcount of the kernel sources that reference each `event_id`：channel
    /// create / futex create / epoll registration of `EpollSource::Event`。
    /// 拿来给 [`Self::completed_event_is_live`] 走 O(1) 的判定，避免再去
    /// 扫 `channels` / `futexes` / `epolls` 三张表。重复 key 的复用与递减
    /// 必须严格成对，少一次会让 prune 提前误判事件已死。
    pub(super) event_source_refs: FastMap<EventId, u32>,
    /// Bumped whenever the process topology changes (spawn / terminate /
    /// set_process_group). Used as the version stamp for `shm_perm_cache`
    /// so we never need to walk the cache on invalidation.
    pub(super) topology_version: u64,
    /// Direct-mapped permission cache for SHM accesses. Keyed by
    /// (current_pid, owner_pid) hashed into a fixed-size slot array.
    /// Each entry carries the topology version it was computed under so
    /// stale entries get refreshed lazily on lookup. Wrapped in `Cell` so
    /// the cache can be updated through the `&self` `shm_read` syscall path.
    pub(super) shm_perm_cache: [std::cell::Cell<Option<ShmPermCacheEntry>>; SHM_PERM_CACHE_SLOTS],
    /// Futex table: FutexAddr -> state. Managed by FutexOps impl.
    pub(super) futexes: FastMap<u64, FutexState>,
    pub(super) next_futex_id: u64,
    /// Kernel trace ring buffer.
    pub(super) trace: TraceRing,
    /// LLM device: model name -> price table. See `LlmOps`.
    pub(super) llm_prices: FastMap<String, crate::primitives::LlmModelPrice>,
    /// Daemon registry: handle -> entry. See `DaemonOps`.
    pub(super) daemons: FastMap<u64, DaemonEntry>,
    pub(super) next_daemon_id: u64,
    /// IPC channel table: channel id -> queue entry.
    pub(super) channels: FastMap<u64, IpcChannelEntry>,
    pub(super) next_channel_id: u64,
    /// Internal event id allocator for kernel-owned primitives like channels.
    pub(super) next_internal_event_id: u64,
    /// epoll registry: epoll id -> registrations.
    pub(super) epolls: FastMap<u64, EpollEntry>,
    pub(super) next_epoll_id: u64,
}

/// 内核内部使用的 daemon 登记条目（含活引用 token）。对外只暴露 Snapshot。
pub(super) struct DaemonEntry {
    pub(super) label: String,
    pub(super) kind: crate::primitives::DaemonKind,
    pub(super) state: crate::primitives::DaemonState,
    pub(super) parent_pid: Option<u64>,
    pub(super) spawn_tick: u64,
    pub(super) exit_tick: Option<u64>,
    pub(super) last_error: Option<String>,
    pub(super) cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub(super) struct IpcChannelEntry {
    pub(super) owner_pid: Option<u64>,
    pub(super) label: String,
    pub(super) owner_tag: ChannelOwnerTag,
    pub(super) ref_count: u32,
    /// Insertion-ordered (holder, count) list. Repeated retain_named calls
    /// with the same name increment the count instead of pushing duplicates,
    /// which keeps release O(unique-holder count) instead of O(total retains).
    pub(super) ref_holders: Vec<(String, u32)>,
    pub(super) event_id: EventId,
    pub(super) capacity: usize,
    pub(super) queue: VecDeque<String>,
    pub(super) closed: bool,
}

pub(super) struct EpollEntry {
    pub(super) label: String,
    pub(super) registrations: FastMap<EpollSource, EpollRegistration>,
}

#[derive(Clone)]
pub(super) struct EpollRegistration {
    pub(super) snapshot: EpollRegistrationSnapshot,
    pub(super) futex_seq_cursor: Option<u64>,
}

impl Default for LocalOS {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalOS {
    pub fn new() -> Self {
        Self::with_trace_capacity(DEFAULT_TRACE_CAPACITY)
    }

    /// Construct a `LocalOS` with a caller-specified trace ring capacity.
    /// `0` disables trace recording entirely.
    pub fn with_trace_capacity(trace_capacity: usize) -> Self {
        Self {
            processes: FastMap::default(),
            ready_queue: VecDeque::new(),
            ready_set: FastSet::default(),
            wait_queue: HashMap::new(),
            children_by_parent: FastMap::default(),
            next_pid: 1,
            current_pid: None,
            yield_requested: false,
            tick: 0,
            round_robin: true,
            shared_memory: FastMap::default(),
            next_pgid: 1,
            completed_events: HashSet::new(),
            completed_event_order: VecDeque::new(),
            completed_event_retention: DEFAULT_COMPLETED_EVENT_RETENTION,
            event_waiters: FastMap::default(),
            event_source_refs: FastMap::default(),
            topology_version: 0,
            shm_perm_cache: [const { std::cell::Cell::new(None) }; SHM_PERM_CACHE_SLOTS],
            futexes: FastMap::default(),
            next_futex_id: 1,
            trace: TraceRing::new(trace_capacity),
            llm_prices: FastMap::default(),
            daemons: FastMap::default(),
            next_daemon_id: 1,
            channels: FastMap::default(),
            next_channel_id: 1,
            next_internal_event_id: 1_000_000,
            epolls: FastMap::default(),
            next_epoll_id: 1,
        }
    }

    fn remove_process_entry_raw(&mut self, pid: u64) -> bool {
        let parent_pid = match self.processes.remove(&pid) {
            Some(proc) => proc.parent_pid,
            None => return false,
        };
        // Topology changed (process gone): invalidate SHM perm cache lazily.
        self.bump_topology_version();
        if let Some(parent) = parent_pid {
            self.unregister_child(parent, pid);
        }
        // O(1) lazy removal: drop the membership marker; any tombstone left
        // in `ready_queue` is filtered out by `pop_ready`.
        self.ready_set.remove(&pid);
        self.wait_queue.remove(&pid);

        let orphaned_children: Vec<u64> = self
            .children_by_parent
            .remove(&pid)
            .map(|set| set.into_iter().collect())
            .unwrap_or_default();
        let mut terminated_children = Vec::new();
        for child_pid in orphaned_children {
            if let Some(child) = self.processes.get_mut(&child_pid) {
                if child.state == ProcessState::Terminated {
                    terminated_children.push(child_pid);
                } else {
                    child.parent_pid = None;
                    child.mailbox.push_back(format!(
                        "Parent process {} exited; this process is now orphaned.",
                        pid
                    ));
                }
            }
        }
        for child_pid in terminated_children {
            self.remove_process_entry_raw(child_pid);
        }
        true
    }

    fn cleanup_unreapable_zombies(&mut self) {
        loop {
            let zombies: Vec<u64> = self
                .processes
                .iter()
                .filter_map(|(pid, proc)| {
                    (proc.state == ProcessState::Terminated
                        && proc
                            .parent_pid
                            .is_some_and(|parent| !self.processes.contains_key(&parent)))
                    .then_some(*pid)
                })
                .collect();
            if zombies.is_empty() {
                break;
            }
            for pid in zombies {
                self.remove_process_entry_raw(pid);
            }
        }
    }

    fn remove_process_entry(&mut self, pid: u64) -> bool {
        let removed = self.remove_process_entry_raw(pid);
        if removed {
            self.cleanup_unreapable_zombies();
        }
        removed
    }

    fn process_priority(&self, pid: u64) -> u8 {
        self.processes
            .get(&pid)
            .map(|proc| proc.priority)
            .unwrap_or(u8::MAX)
    }

    fn enqueue_ready(&mut self, pid: u64) {
        // 不存在的 pid 或已在队列中都直接返回。`ready_set.insert` 返回值同时
        // 承担了去重检查，省一次额外查表。
        if !self.processes.contains_key(&pid) || !self.ready_set.insert(pid) {
            return;
        }

        let priority = self.process_priority(pid);
        // 优先级插入点查找现在只读 tuple 里的缓存优先级，不再进 processes
        // 表。这让批量唯醒 / spawn 从 O(n^2) 到 O(n*k)（k = 队列该优先级之后
        // 的长度）。序列 tombstone 在比较时也不会造成额外代价。
        let insert_at = self
            .ready_queue
            .iter()
            .position(|(_, queued_priority)| *queued_priority > priority);
        match insert_at {
            Some(index) => self.ready_queue.insert(index, (pid, priority)),
            None => self.ready_queue.push_back((pid, priority)),
        }
    }

    fn terminate_pid(&mut self, pid: u64, result: String) {
        // O(1) tombstone：仅移除集合成员身份，`pop_ready` 负责清理。
        self.ready_set.remove(&pid);
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Terminated;
            proc.result = Some(result.clone());
        }
        // 进程进入 Terminated 会影响 SHM 权限（owner 已不可达），才能及时
        // 失效缓存中“owner 还在”的 accessible=true 结果。严格讲另一条防线
        // 在 `shm_read` 里独立检查了 owner 状态，但在此处 bump 可以减少拍额
        // 外 syscall 返回错误不一致的概率。
        self.bump_topology_version();

        if let Some(waiting_pids) = self.wait_queue.remove(&pid) {
            for waiting_pid in waiting_pids {
                if let Some(waiting_proc) = self.processes.get_mut(&waiting_pid) {
                    waiting_proc.state = ProcessState::Ready;
                    waiting_proc.mailbox.push_back(format!(
                        "Process {} terminated with result: {}",
                        pid, result
                    ));
                    self.enqueue_ready(waiting_pid);
                }
            }
        }
    }

    fn require_capability<F>(&self, pid: u64, predicate: F, action: &str) -> Result<(), String>
    where
        F: FnOnce(&ProcessCapabilities) -> bool,
    {
        let proc = self
            .processes
            .get(&pid)
            .ok_or_else(|| format!("Current process {} does not exist.", pid))?;
        if predicate(&proc.capabilities) {
            Ok(())
        } else {
            Err(format!(
                "Process {} does not have capability to {}.",
                pid, action
            ))
        }
    }

    fn ensure_child_scope(&self, current: u64, target: u64) -> Result<(), String> {
        if current == target {
            return Ok(());
        }
        let mut cursor = target;
        while let Some(proc) = self.processes.get(&cursor) {
            if proc.parent_pid == Some(current) {
                return Ok(());
            }
            if let Some(parent) = proc.parent_pid {
                cursor = parent;
            } else {
                break;
            }
        }
        Err(format!(
            "Process {} can only manage its descendants, but target {} is outside its scope.",
            current, target
        ))
    }

    fn collect_descendants(&self, pid: u64) -> Vec<u64> {
        let mut result = Vec::new();
        let mut stack: Vec<u64> = self
            .children_by_parent
            .get(&pid)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default();
        while let Some(current) = stack.pop() {
            result.push(current);
            if let Some(children) = self.children_by_parent.get(&current) {
                stack.extend(children.iter().copied());
            }
        }
        result
    }

    fn register_child(&mut self, parent: u64, child: u64) {
        self.children_by_parent
            .entry(parent)
            .or_default()
            .insert(child);
    }

    fn unregister_child(&mut self, parent: u64, child: u64) {
        if let Some(set) = self.children_by_parent.get_mut(&parent) {
            set.remove(&child);
            if set.is_empty() {
                self.children_by_parent.remove(&parent);
            }
        }
    }

    /// Bumped on every change that can affect SHM permission decisions:
    /// process spawn / terminate / set_process_group. The
    /// `topology_version` doubles as a cache stamp, so old `shm_perm_cache`
    /// entries are simply ignored on lookup once the version moves.
    fn bump_topology_version(&mut self) {
        self.topology_version = self.topology_version.wrapping_add(1);
    }

    fn shm_perm_slot(current_pid: u64, owner_pid: u64) -> usize {
        ((current_pid ^ owner_pid).rotate_left(13) as usize) % SHM_PERM_CACHE_SLOTS
    }

    /// Returns `(accessible, readable)` for `(current_pid, entry.owner_pid)`,
    /// consulting the direct-mapped cache when possible. Cache misses (or
    /// stale entries from prior topology versions) fall back to the
    /// process-tree walk and refresh the slot. Uses interior mutability so
    /// the `&self` `shm_read` syscall can populate the cache too.
    fn shm_perm_lookup(&self, current_pid: u64, entry: &ShmEntry) -> (bool, bool) {
        let slot = Self::shm_perm_slot(current_pid, entry.owner_pid);
        if let Some(cached) = self.shm_perm_cache[slot].get()
            && cached.version == self.topology_version
            && cached.current_pid == current_pid
            && cached.owner_pid == entry.owner_pid
        {
            return (cached.accessible, cached.readable);
        }
        let accessible = self.shm_compute_accessible(current_pid, entry);
        let readable = accessible || self.is_sibling(current_pid, entry.owner_pid);
        self.shm_perm_cache[slot].set(Some(ShmPermCacheEntry {
            version: self.topology_version,
            current_pid,
            owner_pid: entry.owner_pid,
            accessible,
            readable,
        }));
        (accessible, readable)
    }

    fn shm_compute_accessible(&self, pid: u64, entry: &ShmEntry) -> bool {
        if pid == entry.owner_pid {
            return true;
        }
        // 这里走 live 的 owner pgid 查询（而不是 entry.owner_pgid 这一份创建
        // 时的快照），因为 owner 进程可能在创建之后才被 `set_process_group`
        // 加进新的 group，缓存里的 None 会让本来合法的写被拒。perm cache
        // 由 topology_version 负责失效，这里只关心一次决策的正确性。
        if self.is_same_process_group(pid, entry.owner_pid) {
            return true;
        }
        if self.is_ancestor_of(pid, entry.owner_pid) {
            return true;
        }
        false
    }

    fn register_event_waiter(&mut self, event: EventId, pid: u64) {
        self.event_waiters.entry(event).or_default().insert(pid);
    }

    /// 向 `event_source_refs` 登记一个新的内核 source 引用。channel / futex /
    /// epoll(EpollSource::Event) 任意一个新增都应调用一次，destroy 时
    /// [`dec_event_source_ref`] 配对调用。
    fn inc_event_source_ref(&mut self, event: EventId) {
        let entry = self.event_source_refs.entry(event).or_insert(0);
        *entry = entry.saturating_add(1);
    }

    /// 释放一次 source 引用，归零时清掉条目避免无界增长。
    fn dec_event_source_ref(&mut self, event: EventId) {
        if let Some(slot) = self.event_source_refs.get_mut(&event) {
            *slot = slot.saturating_sub(1);
            if *slot == 0 {
                self.event_source_refs.remove(&event);
            }
        }
    }

    fn is_same_process_group(&self, pid_a: u64, pid_b: u64) -> bool {
        let pgid_a = self.processes.get(&pid_a).and_then(|p| p.process_group);
        let pgid_b = self.processes.get(&pid_b).and_then(|p| p.process_group);
        match (pgid_a, pgid_b) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    fn is_shm_accessible_by(&self, pid: u64, entry: &ShmEntry) -> bool {
        self.shm_perm_lookup(pid, entry).0
    }

    fn is_shm_readable_by(&self, pid: u64, entry: &ShmEntry) -> bool {
        self.shm_perm_lookup(pid, entry).1
    }

    fn is_ancestor_of(&self, ancestor: u64, descendant: u64) -> bool {
        let mut cursor = descendant;
        while let Some(proc) = self.processes.get(&cursor) {
            if proc.parent_pid == Some(ancestor) {
                return true;
            }
            if let Some(parent) = proc.parent_pid {
                cursor = parent;
            } else {
                break;
            }
        }
        false
    }

    fn is_sibling(&self, pid_a: u64, pid_b: u64) -> bool {
        let parent_a = self.processes.get(&pid_a).and_then(|p| p.parent_pid);
        let parent_b = self.processes.get(&pid_b).and_then(|p| p.parent_pid);
        match (parent_a, parent_b) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    fn deliver_signal(&mut self, target_pid: u64, signal: Signal) -> Result<(), String> {
        match signal {
            Signal::SigCancel => {
                if let Some(proc) = self.processes.get_mut(&target_pid) {
                    if proc.state == ProcessState::Terminated {
                        return Ok(());
                    }
                    if !proc.pending_signals.contains(&Signal::SigCancel) {
                        proc.pending_signals.push_back(Signal::SigCancel);
                    }
                }
            }
            Signal::SigKill => {
                let descendants = self.collect_descendants(target_pid);
                for pid in descendants.iter().rev() {
                    if matches!(
                        self.processes.get(pid).map(|proc| &proc.state),
                        Some(ProcessState::Terminated)
                    ) {
                        continue;
                    }
                    self.terminate_pid(
                        *pid,
                        format!("Killed (cascade from SIGKILL to {})", target_pid),
                    );
                }
                self.terminate_pid(target_pid, "Killed by SIGKILL".to_string());
            }
            Signal::SigTerm => {
                if let Some(proc) = self.processes.get_mut(&target_pid) {
                    if proc.state == ProcessState::Terminated {
                        return Ok(());
                    }
                    proc.pending_signals.push_back(Signal::SigTerm);
                    if proc.state == ProcessState::Stopped {
                        proc.state = ProcessState::Ready;
                        self.enqueue_ready(target_pid);
                    }
                }
            }
            Signal::SigStop => {
                if let Some(proc) = self.processes.get_mut(&target_pid) {
                    if matches!(proc.state, ProcessState::Terminated | ProcessState::Stopped) {
                        return Ok(());
                    }
                    self.ready_set.remove(&target_pid);
                    proc.state = ProcessState::Stopped;
                    if self.current_pid == Some(target_pid) {
                        self.current_pid = None;
                        self.yield_requested = true;
                    }
                }
            }
            Signal::SigCont => {
                if let Some(proc) = self.processes.get_mut(&target_pid) {
                    if proc.state != ProcessState::Stopped {
                        return Ok(());
                    }
                    proc.state = ProcessState::Ready;
                    self.enqueue_ready(target_pid);
                }
            }
        }
        Ok(())
    }

    pub fn process_pending_signals(&mut self) -> bool {
        let current = match self.current_pid {
            Some(pid) => pid,
            None => return false,
        };

        let signals: Vec<Signal> = {
            if let Some(proc) = self.processes.get(&current) {
                proc.pending_signals.iter().copied().collect()
            } else {
                return false;
            }
        };

        let mut should_cancel = false;
        let mut should_terminate = false;
        let mut should_stop = false;

        for signal in signals {
            match signal {
                Signal::SigCancel => {
                    should_cancel = true;
                }
                Signal::SigKill => {
                    should_terminate = true;
                    break;
                }
                Signal::SigTerm => {
                    should_terminate = true;
                    break;
                }
                Signal::SigStop => {
                    should_stop = true;
                    break;
                }
                Signal::SigCont => {}
            }
        }

        if let Some(proc) = self.processes.get_mut(&current) {
            proc.pending_signals.clear();
        }

        if should_cancel {
            return true;
        }

        if should_terminate {
            self.terminate_current("Terminated by signal".to_string());
            return true;
        }

        if should_stop {
            if let Some(proc) = self.processes.get_mut(&current) {
                proc.state = ProcessState::Stopped;
            }
            self.current_pid = None;
            self.yield_requested = true;
            return true;
        }

        false
    }

    fn event_wait_is_satisfied(
        &self,
        event_ids: &[EventId],
        policy: &WaitPolicy,
        completed_event_ids: &HashSet<EventId>,
    ) -> bool {
        match policy {
            WaitPolicy::Any => event_ids
                .iter()
                .any(|event_id| completed_event_ids.contains(event_id)),
            WaitPolicy::All => event_ids
                .iter()
                .all(|event_id| completed_event_ids.contains(event_id)),
        }
    }

    fn remember_completed_event(&mut self, event_id: EventId) {
        if self.completed_events.insert(event_id) {
            self.completed_event_order.push_back(event_id);
        }
        self.prune_completed_events();
    }

    fn completed_event_is_live(&self, event_id: EventId) -> bool {
        // Process side uses the reverse waiter index (O(1)) instead of a full
        // process-table scan. Source side (channels / futexes / epoll
        // registrations) is consulted via `event_source_refs`，同样 O(1)。
        if self
            .event_waiters
            .get(&event_id)
            .is_some_and(|set| !set.is_empty())
        {
            return true;
        }
        self.event_source_refs
            .get(&event_id)
            .copied()
            .unwrap_or(0)
            > 0
    }

    fn prune_completed_events(&mut self) {
        let retention = self.completed_event_retention;
        if self.completed_event_order.len() <= retention {
            return;
        }

        // Examine only the OLDEST `excess` entries (matches the original
        // "always keep the newest `retention`" semantics). Among them, drop
        // dead entries from `completed_events` and re-insert live ones at
        // the front to preserve FIFO order.
        let excess = self.completed_event_order.len() - retention;
        let mut keep_live: Vec<EventId> = Vec::new();
        for _ in 0..excess {
            let Some(event_id) = self.completed_event_order.pop_front() else {
                break;
            };
            if self.completed_event_is_live(event_id) {
                keep_live.push(event_id);
            } else {
                self.completed_events.remove(&event_id);
            }
        }
        for event_id in keep_live.into_iter().rev() {
            self.completed_event_order.push_front(event_id);
        }
    }
}

impl Syscall for LocalOS {
    fn spawn(
        &mut self,
        parent_pid: Option<u64>,
        name: String,
        goal: String,
        priority: u8,
        quota_turns: usize,
        capabilities: Option<ProcessCapabilities>,
        allowed_tools: Option<FastSet<String>>,
    ) -> Result<u64, String> {
        if let Some(parent) = parent_pid {
            let parent_caps = self
                .processes
                .get(&parent)
                .map(|p| p.capabilities.clone())
                .ok_or_else(|| format!("Parent process {} does not exist.", parent))?;
            if !parent_caps.spawn {
                return Err("Current process is not allowed to spawn children.".to_string());
            }
        }

        let pid = self.next_pid;
        self.next_pid += 1;

        let mut env = FastMap::default();
        let requested_capabilities = capabilities.clone();
        let mut inherited_capabilities = requested_capabilities
            .clone()
            .unwrap_or_else(ProcessCapabilities::full);
        let inherited_allowed_tools = if let Some(parent) = parent_pid {
            if let Some(p_proc) = self.processes.get(&parent) {
                env = p_proc.env.clone();
                inherited_capabilities =
                    requested_capabilities.unwrap_or_else(|| p_proc.capabilities.clone());
                p_proc.allowed_tools.clone()
            } else {
                FastSet::default()
            }
        } else {
            FastSet::default()
        };

        let final_allowed_tools = allowed_tools.unwrap_or(inherited_allowed_tools);

        let inherited_working_dir = if let Some(parent) = parent_pid {
            self.processes
                .get(&parent)
                .and_then(|p| p.working_dir.clone())
        } else {
            None
        };

        let inherited_pgid = if let Some(parent) = parent_pid {
            self.processes.get(&parent).and_then(|p| p.process_group)
        } else {
            None
        };

        self.processes.insert(
            pid,
            Process {
                pid,
                parent_pid,
                name,
                goal,
                state: ProcessState::Ready,
                result: None,
                mailbox: VecDeque::new(),
                max_mailbox_capacity: DEFAULT_MAILBOX_CAPACITY,
                pending_signals: VecDeque::new(),
                priority,
                quota_turns,
                capabilities: inherited_capabilities,
                is_foreground: false,
                turns_used: 0,
                created_at_tick: self.tick,
                process_group: inherited_pgid,
                is_daemon: false,
                max_restarts: 0,
                restart_count: 0,
                env,
                history_file: None,
                allowed_tools: final_allowed_tools,
                tool_calls_used: 0,
                working_dir: inherited_working_dir,
                limits: ResourceLimit::from_legacy(quota_turns),
                usage: ResourceUsage::default(),
            },
        );
        if let Some(parent) = parent_pid {
            self.register_child(parent, pid);
        }
        // New process changed the topology: invalidate SHM perm cache lazily.
        self.bump_topology_version();
        self.enqueue_ready(pid);
        Ok(pid)
    }

    fn wait_on(&mut self, target_pid: u64) -> Result<(), String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.wait, "wait")?;
        if target_pid == current {
            return Err("Current process cannot wait on itself.".to_string());
        }
        self.ensure_child_scope(current, target_pid)?;

        if !self.processes.contains_key(&target_pid) {
            return Err(format!("Target process {} does not exist.", target_pid));
        }

        if let Some(target_proc) = self.processes.get(&target_pid) {
            if target_proc.state == ProcessState::Terminated {
                let result = target_proc.result.clone().unwrap_or_default();
                if let Some(current_proc) = self.processes.get_mut(&current) {
                    current_proc.mailbox.push_back(format!(
                        "Process {} already terminated with result: {}",
                        target_pid, result
                    ));
                }
                return Ok(());
            }
        }

        if let Some(current_proc) = self.processes.get_mut(&current) {
            current_proc.state = ProcessState::Waiting {
                reason: WaitReason::ProcessExit { on_pid: target_pid },
            };
        }

        self.wait_queue.entry(target_pid).or_default().push(current);
        self.current_pid = None;
        self.yield_requested = true;

        Ok(())
    }

    fn wait_on_events(
        &mut self,
        event_ids: Vec<EventId>,
        policy: WaitPolicy,
        timeout_ticks: Option<u64>,
    ) -> Result<Option<u64>, String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.wait, "wait on events")?;

        let mut deduped = Vec::new();
        for event_id in event_ids {
            if !deduped.iter().any(|existing| existing == &event_id) {
                deduped.push(event_id);
            }
        }
        if deduped.is_empty() {
            return Err("event_ids cannot be empty.".to_string());
        }

        // Check if the wait condition is already satisfied by previously completed events.
        // This avoids the TOCTOU race where events complete between the caller's snapshot
        // check and this wait_on_events call, causing lost notifications and a permanent stall.
        if self.event_wait_is_satisfied(&deduped, &policy, &self.completed_events) {
            let completed_ids_str = deduped
                .iter()
                .filter(|id| self.completed_events.contains(id))
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            if let Some(current_proc) = self.processes.get_mut(&current) {
                current_proc.mailbox.push_back(format!(
                    "[EVENT_WAKE]\nReason: event wait condition satisfied.\nCompleted event ids: {}\nRecommended next actions:\n1. Inspect the event-producing subsystem for fresh state.\n2. If these events came from async tool work, use tool_status or tool_wait to collect results.\n3. Cancel low-value still-running branches when appropriate.\n4. If enough results are already available, continue reasoning immediately.",
                    completed_ids_str
                ));
            }
            return Ok(None);
        }

        let timeout_tick = timeout_ticks.map(|ticks| self.tick.saturating_add(ticks.max(1)));
        if let Some(current_proc) = self.processes.get_mut(&current) {
            current_proc.state = ProcessState::Waiting {
                reason: WaitReason::Events {
                    event_ids: deduped.clone(),
                    policy,
                    timeout_tick,
                },
            };
        }
        for event_id in &deduped {
            self.register_event_waiter(*event_id, current);
        }
        self.current_pid = None;
        self.yield_requested = true;
        Ok(timeout_tick)
    }

    fn send_ipc(&mut self, target_pid: u64, message: String) -> Result<(), String> {
        let sender_pid = self.current_pid.ok_or("No process currently running.")?;
        self.require_capability(sender_pid, |caps| caps.ipc_send, "send ipc")?;

        if !self.processes.contains_key(&target_pid) {
            return Err(format!("Process {} does not exist.", target_pid));
        }

        // 一次走完所有亲缘关系判定，让两层权限检查复用同一组判定结果，
        // 避免 happy path 上把 process tree 走两遍。
        let same_pid = sender_pid == target_pid;
        let same_pgid = !same_pid && self.is_same_process_group(sender_pid, target_pid);
        let sender_is_ancestor =
            !same_pid && !same_pgid && self.is_ancestor_of(sender_pid, target_pid);
        let sender_is_descendant = !same_pid
            && !same_pgid
            && !sender_is_ancestor
            && self.is_ancestor_of(target_pid, sender_pid);
        let sender_is_sibling = !same_pid
            && !same_pgid
            && !sender_is_ancestor
            && !sender_is_descendant
            && self.is_sibling(sender_pid, target_pid);

        if !same_pid
            && !same_pgid
            && !sender_is_ancestor
            && !sender_is_descendant
            && !sender_is_sibling
        {
            return Err(format!(
                "Permission denied: process {} cannot send IPC to process {} (not in same process group or parent-child relationship).",
                sender_pid, target_pid
            ));
        }

        if !same_pid && !same_pgid && !sender_is_ancestor {
            let target_pgid = self
                .processes
                .get(&target_pid)
                .and_then(|p| p.process_group);
            if target_pgid.is_some() {
                return Err(format!(
                    "Permission denied: process {} is in a restricted process group, only group members or ancestors can send IPC.",
                    target_pid
                ));
            }
        }

        if let Some(target_proc) = self.processes.get_mut(&target_pid) {
            if target_proc.state == ProcessState::Terminated {
                return Err(format!("Process {} is already terminated.", target_pid));
            }
            if target_proc.mailbox.len() >= target_proc.max_mailbox_capacity {
                return Err(format!(
                    "Process {} mailbox is full (capacity: {}). Cannot send message.",
                    target_pid, target_proc.max_mailbox_capacity
                ));
            }
            target_proc
                .mailbox
                .push_back(format!("[IPC from {}] {}", sender_pid, message));
            Ok(())
        } else {
            Err(format!("Process {} does not exist.", target_pid))
        }
    }

    fn read_mailbox(&mut self) -> Result<Vec<String>, String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.ipc_receive, "read mailbox")?;
        if let Some(current_proc) = self.processes.get_mut(&current) {
            let messages: Vec<String> = current_proc.mailbox.drain(..).collect();
            Ok(messages)
        } else {
            Err("Current process not found in process table.".to_string())
        }
    }

    fn set_env(&mut self, key: String, value: String) -> Result<(), String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.env_write, "set env")?;
        if let Some(current_proc) = self.processes.get_mut(&current) {
            current_proc.env.insert(key, value);
            Ok(())
        } else {
            Err("Current process not found in process table.".to_string())
        }
    }

    fn get_env(&self, key: &str) -> Option<String> {
        let current = self.current_pid?;
        let current_proc = self.processes.get(&current)?;
        current_proc.env.get(key).cloned()
    }

    fn current_process_id(&self) -> Option<u64> {
        crate::kernel::current_task_pid().or(self.current_pid)
    }

    fn get_process(&self, pid: u64) -> Option<&Process> {
        self.processes.get(&pid)
    }

    fn list_processes(&self) -> Vec<Process> {
        self.processes
            .iter()
            .map(|(_, proc)| proc.clone())
            .collect()
    }

    fn sleep_current(&mut self, turns: u64) -> Result<u64, String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.sleep, "sleep")?;
        let until_tick = self.tick.saturating_add(turns.max(1));
        if let Some(proc) = self.processes.get_mut(&current) {
            proc.state = ProcessState::Sleeping { until_tick };
        }
        self.current_pid = None;
        self.yield_requested = true;
        Ok(until_tick)
    }

    fn kill_process(&mut self, target_pid: u64, reason: String) -> Result<(), String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.manage_children, "kill process")?;
        self.ensure_child_scope(current, target_pid)?;
        if target_pid == current {
            return Err("Current process cannot kill itself via `kill_process`.".to_string());
        }
        if matches!(
            self.processes.get(&target_pid).map(|proc| &proc.state),
            Some(ProcessState::Terminated)
        ) {
            return Ok(());
        }

        let descendants = self.collect_descendants(target_pid);
        for pid in descendants.iter().rev() {
            if matches!(
                self.processes.get(pid).map(|proc| &proc.state),
                Some(ProcessState::Terminated)
            ) {
                continue;
            }
            self.terminate_pid(
                *pid,
                format!("Killed (cascade from {}): {}", target_pid, reason),
            );
        }
        self.terminate_pid(target_pid, format!("Killed: {reason}"));
        Ok(())
    }

    fn reap_process(&mut self, target_pid: u64) -> Result<String, String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.reap, "reap process")?;
        self.ensure_child_scope(current, target_pid)?;
        let proc = self
            .processes
            .get(&target_pid)
            .ok_or_else(|| format!("Process {} does not exist.", target_pid))?;
        if proc.state != ProcessState::Terminated {
            return Err(format!("Process {} is not terminated yet.", target_pid));
        }
        let result = proc.result.clone().unwrap_or_default();
        self.remove_process_entry(target_pid);
        Ok(result)
    }

    fn signal_process(&mut self, target_pid: u64, signal: Signal) -> Result<(), String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.signal, "send signal")?;
        self.ensure_child_scope(current, target_pid)?;

        if !self.processes.contains_key(&target_pid) {
            return Err(format!("Target process {} does not exist.", target_pid));
        }
        if matches!(
            self.processes.get(&target_pid).map(|p| &p.state),
            Some(ProcessState::Terminated)
        ) {
            return Err(format!("Cannot signal terminated process {}.", target_pid));
        }

        self.deliver_signal(target_pid, signal)
    }

    fn set_process_group(&mut self, pid: u64, pgid: u64) -> Result<(), String> {
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.process_group = Some(pgid);
            // Process group membership feeds into SHM perm decisions.
            self.bump_topology_version();
            Ok(())
        } else {
            Err(format!("Process {} does not exist.", pid))
        }
    }

    fn signal_process_group(&mut self, pgid: u64, signal: Signal) -> Result<usize, String> {
        let target_pids: Vec<u64> = self
            .processes
            .iter()
            .filter(|(_, proc)| {
                proc.process_group == Some(pgid) && proc.state != ProcessState::Terminated
            })
            .map(|(pid, _)| *pid)
            .collect();

        if target_pids.is_empty() {
            return Err(format!("No active processes in group {}.", pgid));
        }

        let count = target_pids.len();
        for pid in target_pids {
            let _ = self.deliver_signal(pid, signal);
        }
        Ok(count)
    }

    fn shm_create(&mut self, key: String, value: String) -> Result<(), String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        if self.shared_memory.contains_key(&key) {
            return Err(format!("Shared memory key '{}' already exists.", key));
        }
        let owner_pgid = self.processes.get(&current).and_then(|p| p.process_group);
        // owner_pgid 仅越过 trace 下发作为创建时快照；SHM 权限走实时
        // 查表以应对 set_process_group 后的变更。
        let _ = owner_pgid;
        let checksum = shm_checksum(&value, current);
        self.shared_memory.insert(
            key,
            ShmEntry {
                value,
                owner_pid: current,
                checksum,
                version: 1,
            },
        );
        Ok(())
    }

    fn shm_read(&self, key: &str) -> Result<String, ShmReadError> {
        let entry = self.shared_memory.get(key).ok_or(ShmReadError::NotFound)?;

        let current = match self.current_pid {
            Some(pid) => pid,
            None => return Ok(entry.value.clone()),
        };

        if !self.is_shm_readable_by(current, entry) {
            return Err(ShmReadError::PermissionDenied {
                owner_pid: entry.owner_pid,
            });
        }

        let actual = shm_checksum(&entry.value, entry.owner_pid);
        if actual != entry.checksum {
            return Err(ShmReadError::Corrupted {
                expected_checksum: entry.checksum,
                actual_checksum: actual,
            });
        }

        let owner_terminated = self
            .processes
            .get(&entry.owner_pid)
            .map(|p| p.state == ProcessState::Terminated)
            .unwrap_or(true);
        if owner_terminated {
            return Err(ShmReadError::OwnerTerminated {
                owner_pid: entry.owner_pid,
            });
        }

        Ok(entry.value.clone())
    }

    fn shm_read_degraded(&self, key: &str) -> Option<String> {
        match self.shm_read(key) {
            Ok(value) => Some(value),
            Err(ShmReadError::OwnerTerminated { .. }) => self
                .shared_memory
                .get(key)
                .map(|e| format!("[DEGRADED: owner terminated] {}", e.value)),
            Err(ShmReadError::Corrupted { .. }) => self
                .shared_memory
                .get(key)
                .map(|e| format!("[DEGRADED: checksum mismatch] {}", e.value)),
            Err(_) => None,
        }
    }

    fn shm_write(&mut self, key: String, value: String) -> Result<(), String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        let entry = self
            .shared_memory
            .get(&key)
            .ok_or(format!("Shared memory key '{}' does not exist.", key))?;
        if !self.is_shm_accessible_by(current, entry) {
            return Err(format!(
                "Permission denied: process {} cannot write shared memory key '{}' owned by process {}.",
                current, key, entry.owner_pid
            ));
        }
        let owner_pid = entry.owner_pid;
        let _ = entry;
        if let Some(e) = self.shared_memory.get_mut(&key) {
            e.value = value;
            e.checksum = shm_checksum(&e.value, owner_pid);
            e.version += 1;
        }
        Ok(())
    }

    fn shm_delete(&mut self, key: &str) -> Result<(), String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        let entry = self
            .shared_memory
            .get(key)
            .ok_or(format!("Shared memory key '{}' does not exist.", key))?;
        if !self.is_shm_accessible_by(current, entry) {
            return Err(format!(
                "Permission denied: process {} cannot delete shared memory key '{}' owned by process {}.",
                current, key, entry.owner_pid
            ));
        }
        let _ = entry;
        self.shared_memory.remove(key);
        Ok(())
    }

    fn set_working_dir(&mut self, dir: PathBuf) -> Result<(), String> {
        let current = self.current_pid.ok_or("No process currently running.")?;
        if let Some(proc) = self.processes.get_mut(&current) {
            proc.working_dir = Some(dir);
            Ok(())
        } else {
            Err("Current process not found in process table.".to_string())
        }
    }

    fn get_working_dir(&self) -> Option<PathBuf> {
        let current = self.current_pid?;
        self.processes
            .get(&current)
            .and_then(|p| p.working_dir.clone())
    }

    fn spawn_daemon(
        &mut self,
        parent_pid: Option<u64>,
        name: String,
        goal: String,
        priority: u8,
        quota_turns: usize,
        max_restarts: usize,
    ) -> Result<u64, String> {
        let pid = self.spawn(parent_pid, name, goal, priority, quota_turns, None, None)?;
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.is_daemon = true;
            proc.max_restarts = max_restarts;
        }
        Ok(pid)
    }

    fn shm_health_check(&self) -> Vec<(String, ShmReadError)> {
        let mut issues = Vec::new();
        for (key, entry) in &self.shared_memory {
            let actual = shm_checksum(&entry.value, entry.owner_pid);
            if actual != entry.checksum {
                issues.push((
                    key.clone(),
                    ShmReadError::Corrupted {
                        expected_checksum: entry.checksum,
                        actual_checksum: actual,
                    },
                ));
                continue;
            }
            let owner_alive = self
                .processes
                .get(&entry.owner_pid)
                .map(|p| p.state != ProcessState::Terminated)
                .unwrap_or(false);
            if !owner_alive {
                issues.push((
                    key.clone(),
                    ShmReadError::OwnerTerminated {
                        owner_pid: entry.owner_pid,
                    },
                ));
            }
        }
        issues
    }

    fn shm_cleanup_orphans(&mut self) -> usize {
        let orphan_keys: Vec<String> = self
            .shared_memory
            .iter()
            .filter(|(_, entry)| {
                let owner_alive = self
                    .processes
                    .get(&entry.owner_pid)
                    .map(|p| p.state != ProcessState::Terminated)
                    .unwrap_or(false);
                !owner_alive
            })
            .map(|(key, _)| key.clone())
            .collect();
        let count = orphan_keys.len();
        for key in orphan_keys {
            self.shared_memory.remove(&key);
        }
        count
    }
}

impl KernelInternal for LocalOS {
    fn begin_foreground(
        &mut self,
        name: String,
        goal: String,
        priority: u8,
        quota_turns: usize,
        allowed_tools: Option<FastSet<String>>,
    ) -> u64 {
        let pid = self.next_pid;
        self.next_pid += 1;
        self.processes.insert(
            pid,
            Process {
                pid,
                parent_pid: None,
                name,
                goal,
                state: ProcessState::Running,
                result: None,
                mailbox: VecDeque::new(),
                max_mailbox_capacity: DEFAULT_MAILBOX_CAPACITY,
                pending_signals: VecDeque::new(),
                priority,
                quota_turns,
                capabilities: ProcessCapabilities::full(),
                is_foreground: true,
                turns_used: 0,
                created_at_tick: self.tick,
                process_group: None,
                is_daemon: false,
                max_restarts: 0,
                restart_count: 0,
                env: FastMap::default(),
                history_file: None,
                allowed_tools: allowed_tools.unwrap_or_default(),
                tool_calls_used: 0,
                working_dir: None,
                limits: ResourceLimit::from_legacy(quota_turns),
                usage: ResourceUsage::default(),
            },
        );
        self.current_pid = Some(pid);
        pid
    }

    fn pop_ready(&mut self) -> Option<Process> {
        // 跳过被惰性删除的 tombstone：ready_set 中不再存在的 pid 丢弃。
        while let Some((pid, _priority)) = self.ready_queue.pop_front() {
            if !self.ready_set.remove(&pid) {
                continue;
            }
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Running;
                self.current_pid = Some(pid);
                self.yield_requested = false;
                return Some(proc.clone());
            }
        }
        None
    }

    fn pop_all_ready(&mut self, max: usize) -> Vec<Process> {
        let mut result = Vec::new();
        // ready_set.len() 是实际可调度进程数，与可能含 tombstone 的
        // ready_queue.len() 不同。
        let cap = max.min(self.ready_set.len());
        while result.len() < cap {
            let Some((pid, _priority)) = self.ready_queue.pop_front() else {
                break;
            };
            if !self.ready_set.remove(&pid) {
                continue;
            }
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Running;
                result.push(proc.clone());
            }
        }
        if let Some(first) = result.first() {
            self.current_pid = Some(first.pid);
            self.yield_requested = false;
        }
        result
    }

    fn set_current_pid(&mut self, pid: Option<u64>) {
        self.current_pid = pid;
    }

    fn terminate_current(&mut self, result: String) {
        if let Some(pid) = self.current_pid.take() {
            self.terminate_pid(pid, result);
        }
    }

    fn get_process_mut(&mut self, pid: u64) -> Option<&mut Process> {
        self.processes.get_mut(&pid)
    }

    fn consume_yield_requested(&mut self) -> bool {
        let yielded = self.yield_requested;
        self.yield_requested = false;
        yielded
    }

    fn event_is_completed(&self, event_id: EventId) -> bool {
        self.completed_events.contains(&event_id)
    }

    fn drop_terminated(&mut self, target_pid: u64) -> bool {
        if !matches!(
            self.processes.get(&target_pid).map(|proc| &proc.state),
            Some(ProcessState::Terminated)
        ) {
            return false;
        }
        self.remove_process_entry(target_pid)
    }

    fn advance_tick(&mut self) {
        self.tick = self.tick.saturating_add(1);
        let mut wake_sleeping_pids = Vec::new();
        let mut wake_async_timeout_pids = Vec::new();
        for (pid, proc) in self.processes.iter() {
            if let ProcessState::Sleeping { until_tick } = proc.state
                && until_tick <= self.tick
            {
                wake_sleeping_pids.push(*pid);
                continue;
            }
            if let ProcessState::Waiting {
                reason:
                    WaitReason::Events {
                        timeout_tick: Some(until_tick),
                        ..
                    },
            } = &proc.state
                && *until_tick <= self.tick
            {
                wake_async_timeout_pids.push(*pid);
            }
        }
        for pid in wake_sleeping_pids {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Ready;
                proc.mailbox
                    .push_back(format!("Sleep finished at scheduler tick {}.", self.tick));
            }
            self.enqueue_ready(pid);
        }
        for pid in wake_async_timeout_pids {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Ready;
                proc.mailbox.push_back(format!(
                    "Event wait timeout reached at scheduler tick {}.",
                    self.tick
                ));
            }
            self.enqueue_ready(pid);
        }
    }

    fn has_ready(&self) -> bool {
        !self.ready_set.is_empty()
    }

    fn ready_count(&self) -> usize {
        self.ready_set.len()
    }

    fn set_round_robin(&mut self, enabled: bool) {
        self.round_robin = enabled;
    }

    fn is_round_robin(&self) -> bool {
        self.round_robin
    }

    fn requeue_current(&mut self) -> bool {
        let pid = match self.current_pid {
            Some(pid) => pid,
            None => return false,
        };
        if let Some(proc) = self.processes.get_mut(&pid) {
            if proc.state != ProcessState::Running || proc.is_foreground {
                return false;
            }
            proc.state = ProcessState::Ready;
        }
        self.current_pid = None;
        self.enqueue_ready(pid);
        true
    }

    fn process_pending_signals(&mut self) -> bool {
        let current = match self.current_pid {
            Some(pid) => pid,
            None => return false,
        };

        let signals: Vec<Signal> = {
            if let Some(proc) = self.processes.get(&current) {
                proc.pending_signals.iter().copied().collect()
            } else {
                return false;
            }
        };

        let mut should_cancel = false;
        let mut should_terminate = false;
        let mut should_stop = false;

        for signal in signals {
            match signal {
                Signal::SigCancel => {
                    should_cancel = true;
                }
                Signal::SigKill => {
                    should_terminate = true;
                    break;
                }
                Signal::SigTerm => {
                    should_terminate = true;
                    break;
                }
                Signal::SigStop => {
                    should_stop = true;
                    break;
                }
                Signal::SigCont => {}
            }
        }

        if let Some(proc) = self.processes.get_mut(&current) {
            proc.pending_signals.clear();
        }

        if should_cancel {
            return true;
        }

        if should_terminate {
            self.terminate_current("Terminated by signal".to_string());
            return true;
        }

        if should_stop {
            if let Some(proc) = self.processes.get_mut(&current) {
                proc.state = ProcessState::Stopped;
            }
            self.current_pid = None;
            self.yield_requested = true;
            return true;
        }

        false
    }

    fn notify_events_completed(&mut self, completed_event_ids: &[EventId]) -> Vec<u64> {
        if completed_event_ids.is_empty() {
            return Vec::new();
        }

        for &eid in completed_event_ids {
            self.remember_completed_event(eid);
        }

        // Drain candidate pids from the reverse waiter index. Any remaining
        // entries for these event ids are obsolete because future
        // wait_on_events calls will short-circuit via completed_events.
        let mut candidates: FastSet<u64> = FastSet::default();
        for &eid in completed_event_ids {
            if let Some(set) = self.event_waiters.remove(&eid) {
                for pid in set {
                    candidates.insert(pid);
                }
            }
        }

        let mut wake_pids = Vec::new();
        for pid in candidates {
            // Lazy verification: pid may have left Waiting via terminate /
            // sigkill / timeout, in which case the stale entry is silently
            // dropped by *not* re-inserting it.
            if let Some(proc) = self.processes.get(&pid)
                && let ProcessState::Waiting {
                    reason:
                        WaitReason::Events {
                            event_ids, policy, ..
                        },
                } = &proc.state
                && self.event_wait_is_satisfied(event_ids, policy, &self.completed_events)
            {
                wake_pids.push(pid);
            }
        }

        for pid in &wake_pids {
            if let Some(proc) = self.processes.get_mut(pid) {
                proc.state = ProcessState::Ready;
                proc.mailbox.push_back(format!(
                    "[EVENT_WAKE]\nReason: event wait condition satisfied.\nCompleted event ids: {}\nRecommended next actions:\n1. Inspect the event-producing subsystem for fresh state.\n2. If these events came from async tool work, use tool_status or tool_wait to collect results.\n3. Cancel low-value still-running branches when appropriate.\n4. If enough results are already available, continue reasoning immediately.",
                    completed_event_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ")
                ));
                self.enqueue_ready(*pid);
            }
        }
        wake_pids
    }

    fn increment_turns_used_for(&mut self, pid: u64) {
        let _ = <Self as RlimitOps>::rusage_charge(
            self,
            pid,
            ResourceUsageDelta {
                turns: 1,
                ..Default::default()
            },
        );
    }

    fn increment_tool_calls_used_for(&mut self, pid: u64) {
        let _ = <Self as RlimitOps>::rusage_charge(
            self,
            pid,
            ResourceUsageDelta {
                tool_calls: 1,
                ..Default::default()
            },
        );
    }

    fn check_daemon_restart(&mut self) -> Vec<u64> {
        let mut restarted = Vec::new();
        let terminated_daemons: Vec<(
            u64,
            String,
            u8,
            usize,
            usize,
            Option<u64>,
            FastMap<String, String>,
            FastSet<String>,
            Option<PathBuf>,
        )> = self
            .processes
            .iter()
            .filter(|(_, proc)| {
                proc.is_daemon
                    && proc.state == ProcessState::Terminated
                    && proc.restart_count < proc.max_restarts
            })
            .map(|(pid, proc)| {
                (
                    *pid,
                    proc.name.clone(),
                    proc.priority,
                    proc.quota_turns,
                    proc.restart_count,
                    proc.parent_pid,
                    proc.env.clone(),
                    proc.allowed_tools.clone(),
                    proc.working_dir.clone(),
                )
            })
            .collect();

        for (
            old_pid,
            name,
            priority,
            quota_turns,
            restart_count,
            parent_pid,
            env,
            allowed_tools,
            working_dir,
        ) in terminated_daemons
        {
            self.processes.remove(&old_pid);
            if let Some(parent) = parent_pid {
                self.unregister_child(parent, old_pid);
            }
            self.children_by_parent.remove(&old_pid);
            self.ready_set.remove(&old_pid);
            self.wait_queue.remove(&old_pid);

            let new_pid = self.next_pid;
            self.next_pid += 1;

            self.processes.insert(
                new_pid,
                Process {
                    pid: new_pid,
                    parent_pid,
                    name: name.clone(),
                    goal: format!("{} (daemon restart #{})", name, restart_count + 1),
                    state: ProcessState::Ready,
                    result: None,
                    mailbox: VecDeque::new(),
                    max_mailbox_capacity: DEFAULT_MAILBOX_CAPACITY,
                    pending_signals: VecDeque::new(),
                    priority,
                    quota_turns,
                    capabilities: ProcessCapabilities::full(),
                    is_foreground: false,
                    turns_used: 0,
                    created_at_tick: self.tick,
                    process_group: None,
                    is_daemon: true,
                    max_restarts: 0,
                    restart_count: restart_count + 1,
                    env,
                    history_file: None,
                    allowed_tools,
                    tool_calls_used: 0,
                    working_dir,
                    limits: ResourceLimit::from_legacy(quota_turns),
                    usage: ResourceUsage::default(),
                },
            );
            if let Some(parent) = parent_pid {
                self.register_child(parent, new_pid);
            }
            // Daemon restart replaces the pid: invalidate SHM perm cache.
            self.bump_topology_version();
            self.enqueue_ready(new_pid);
            restarted.push(new_pid);
        }
        restarted
    }

    fn cleanup_process_resources(&mut self, pid: u64) {
        if let Some(proc) = self.processes.get(&pid) {
            if let Some(ref history_path) = proc.history_file {
                let _ = std::fs::remove_file(history_path);
            }
        }
    }
}

impl FutexOps for LocalOS {
    fn futex_create(&mut self, initial: u64, label: String) -> FutexAddr {
        let id = self.next_futex_id;
        self.next_futex_id += 1;
        let event_id = self.alloc_internal_event_id();
        let owner = self.current_pid;
        self.futexes
            .insert(id, FutexState::new(initial, event_id));
        // 与 futex 生命周期绑定的 event_id 入资源计数。
        self.inc_event_source_ref(event_id);
        // Label and owner used to live on FutexState; emitting a trace event
        // at create time keeps them visible for diagnostics without bloating
        // the per-futex struct.
        let mut fields = FastMap::default();
        if let Some(pid) = owner {
            fields.insert("owner_pid".to_string(), pid.to_string());
        }
        fields.insert("label".to_string(), label);
        fields.insert("addr".to_string(), id.to_string());
        <Self as TraceOps>::trace_event(
            self,
            "futex.create".to_string(),
            TraceLevel::Debug,
            None,
            fields,
            None,
        );
        FutexAddr(id)
    }

    fn futex_load(&self, addr: FutexAddr) -> Option<u64> {
        self.futexes
            .get(&addr.0)
            .map(|s| s.value.load(std::sync::atomic::Ordering::SeqCst))
    }

    fn futex_cas(&mut self, addr: FutexAddr, expected: u64, new_value: u64) -> Result<u64, u64> {
        use std::sync::atomic::Ordering::SeqCst;
        let state = match self.futexes.get(&addr.0) {
            Some(s) => s,
            None => return Err(u64::MAX),
        };
        match state
            .value
            .compare_exchange(expected, new_value, SeqCst, SeqCst)
        {
            Ok(prev) => {
                if prev != new_value {
                    self.complete_futex_event(addr, true);
                }
                Ok(prev)
            }
            Err(cur) => Err(cur),
        }
    }

    fn futex_fetch_add(&mut self, addr: FutexAddr, delta: u64) -> Option<u64> {
        let state = self.futexes.get(&addr.0)?;
        let prev = state
            .value
            .fetch_add(delta, std::sync::atomic::Ordering::SeqCst);
        if delta != 0 {
            self.complete_futex_event(addr, true);
        }
        Some(prev)
    }

    fn futex_store(&mut self, addr: FutexAddr, new_value: u64) -> Option<u64> {
        let state = self.futexes.get(&addr.0)?;
        let prev = state
            .value
            .swap(new_value, std::sync::atomic::Ordering::SeqCst);
        if prev != new_value {
            self.complete_futex_event(addr, true);
        }
        Some(prev)
    }

    fn futex_try_wait(&self, addr: FutexAddr, expected: u64) -> Option<FutexWakeReason> {
        let state = self.futexes.get(&addr.0)?;
        let cur = state.value.load(std::sync::atomic::Ordering::SeqCst);
        if cur != expected {
            Some(FutexWakeReason::ValueChanged)
        } else {
            None
        }
    }

    fn futex_wake(&mut self, addr: FutexAddr, n: usize) -> usize {
        let state = match self.futexes.get_mut(&addr.0) {
            Some(s) => s,
            None => return 0,
        };
        state.seq = state.seq.wrapping_add(1);
        let mut woken = 0usize;
        let mut to_ready: Vec<u64> = Vec::new();
        while woken < n {
            match state.waiters.pop_front() {
                Some(pid) => {
                    to_ready.push(pid);
                    woken += 1;
                }
                None => break,
            }
        }
        for pid in to_ready {
            if let Some(proc) = self.processes.get_mut(&pid) {
                if !matches!(proc.state, ProcessState::Terminated | ProcessState::Stopped) {
                    proc.state = ProcessState::Ready;
                    self.enqueue_ready(pid);
                }
            }
        }
        self.complete_futex_event(addr, true);
        woken
    }

    fn futex_destroy(&mut self, addr: FutexAddr) -> bool {
        let event_id = self.futexes.get(&addr.0).map(|state| state.event_id);
        let removed = self.futexes.remove(&addr.0).is_some();
        if removed && let Some(event_id) = event_id {
            self.dec_event_source_ref(event_id);
            self.notify_events_completed(&[event_id]);
        }
        removed
    }

    fn futex_register_waiter(&mut self, addr: FutexAddr, pid: u64) -> Option<u64> {
        let state = self.futexes.get_mut(&addr.0)?;
        if !state.waiters.iter().any(|p| *p == pid) {
            state.waiters.push_back(pid);
        }
        Some(state.seq)
    }

    fn futex_cancel_waiter(&mut self, addr: FutexAddr, pid: u64) -> bool {
        let state = match self.futexes.get_mut(&addr.0) {
            Some(s) => s,
            None => return false,
        };
        let before = state.waiters.len();
        state.waiters.retain(|p| *p != pid);
        state.waiters.len() != before
    }

    fn futex_seq(&self, addr: FutexAddr) -> Option<u64> {
        self.futexes.get(&addr.0).map(|s| s.seq)
    }

    fn futex_event_id(&self, addr: FutexAddr) -> Option<EventId> {
        self.futexes.get(&addr.0).map(|s| s.event_id)
    }
}

impl TraceOps for LocalOS {
    fn trace_span_enter(
        &mut self,
        name: String,
        parent: Option<u64>,
        fields: FastMap<String, String>,
    ) -> u64 {
        let span_id = self.trace.alloc_span();
        let seq = self.trace.alloc_seq();
        let rec = TraceRecord {
            seq,
            tick: self.tick,
            pid: self.current_pid,
            level: TraceLevel::Debug,
            name,
            span_id: Some(span_id),
            parent_span_id: parent,
            kind: TraceKind::SpanEnter,
            fields: TraceRecord::pack_fields(fields),
            message: None,
        };
        self.trace.push(rec);
        span_id
    }

    fn trace_span_exit(&mut self, span_id: u64, fields: FastMap<String, String>) {
        let seq = self.trace.alloc_seq();
        let rec = TraceRecord {
            seq,
            tick: self.tick,
            pid: self.current_pid,
            level: TraceLevel::Debug,
            name: String::new(),
            span_id: Some(span_id),
            parent_span_id: None,
            kind: TraceKind::SpanExit,
            fields: TraceRecord::pack_fields(fields),
            message: None,
        };
        self.trace.push(rec);
    }

    fn trace_event(
        &mut self,
        name: String,
        level: TraceLevel,
        span_id: Option<u64>,
        fields: FastMap<String, String>,
        message: Option<String>,
    ) {
        let seq = self.trace.alloc_seq();
        let rec = TraceRecord {
            seq,
            tick: self.tick,
            pid: self.current_pid,
            level,
            name,
            span_id,
            parent_span_id: None,
            kind: TraceKind::Event,
            fields: TraceRecord::pack_fields(fields),
            message,
        };
        self.trace.push(rec);
    }

    fn trace_recent(&self, n: usize) -> Vec<TraceRecord> {
        self.trace.buf.iter().rev().take(n).cloned().collect()
    }

    fn trace_drain_since(&self, since_seq: u64) -> Vec<TraceRecord> {
        self.trace
            .buf
            .iter()
            .filter(|r| r.seq > since_seq)
            .cloned()
            .collect()
    }

    fn trace_head_seq(&self) -> u64 {
        self.trace.buf.back().map(|r| r.seq).unwrap_or(0)
    }

    fn trace_set_capacity(&mut self, cap: usize) {
        self.trace.capacity = cap;
        while self.trace.buf.len() > cap {
            self.trace.buf.pop_front();
        }
    }
}

impl RlimitOps for LocalOS {
    fn rlimit_set(&mut self, pid: u64, limits: ResourceLimit) -> Result<(), String> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or_else(|| format!("rlimit_set: no such pid {}", pid))?;
        if limits.max_turns == u64::MAX {
            proc.quota_turns = 0;
        } else {
            proc.quota_turns = limits.max_turns as usize;
        }
        proc.limits = limits;
        Ok(())
    }

    fn rlimit_get(&self, pid: u64) -> Option<ResourceLimit> {
        self.processes.get(&pid).map(|p| p.limits.clone())
    }

    fn rusage_get(&self, pid: u64) -> Option<ResourceUsage> {
        self.processes.get(&pid).map(|p| p.usage.clone())
    }

    fn rusage_charge(&mut self, pid: u64, delta: ResourceUsageDelta) -> RlimitVerdict {
        let Some(proc) = self.processes.get_mut(&pid) else {
            return RlimitVerdict::NoSuchProcess;
        };
        proc.usage.turns = proc.usage.turns.saturating_add(delta.turns);
        proc.usage.tool_calls = proc.usage.tool_calls.saturating_add(delta.tool_calls);
        proc.usage.tokens_in = proc.usage.tokens_in.saturating_add(delta.tokens_in);
        proc.usage.tokens_out = proc.usage.tokens_out.saturating_add(delta.tokens_out);
        proc.usage.cost_micros = proc.usage.cost_micros.saturating_add(delta.cost_micros);
        proc.usage.fs_bytes = proc.usage.fs_bytes.saturating_add(delta.fs_bytes);
        if let Some(b) = delta.last_tool_call_bytes {
            proc.usage.last_tool_call_bytes = b;
        }
        // keep legacy per-process counters in sync
        proc.turns_used = proc.usage.turns as usize;
        proc.tool_calls_used = proc.usage.tool_calls as usize;

        let lim = &proc.limits;
        if proc.usage.turns > lim.max_turns {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::Turns,
                used: proc.usage.turns,
                limit: lim.max_turns,
            };
        }
        if proc.usage.tool_calls > lim.max_tool_calls {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::ToolCalls,
                used: proc.usage.tool_calls,
                limit: lim.max_tool_calls,
            };
        }
        if proc.usage.tokens_in > lim.max_tokens_in {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::TokensIn,
                used: proc.usage.tokens_in,
                limit: lim.max_tokens_in,
            };
        }
        if proc.usage.tokens_out > lim.max_tokens_out {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::TokensOut,
                used: proc.usage.tokens_out,
                limit: lim.max_tokens_out,
            };
        }
        if proc.usage.cost_micros > lim.max_cost_micros {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::CostMicros,
                used: proc.usage.cost_micros,
                limit: lim.max_cost_micros,
            };
        }
        if let Some(b) = delta.last_tool_call_bytes {
            if b > lim.max_tool_call_bytes {
                return RlimitVerdict::Exceeded {
                    dimension: RlimitDim::ToolCallBytes,
                    used: b,
                    limit: lim.max_tool_call_bytes,
                };
            }
        }
        if proc.usage.fs_bytes > lim.max_fs_bytes {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::FsBytes,
                used: proc.usage.fs_bytes,
                limit: lim.max_fs_bytes,
            };
        }
        // wallclock: elapsed = self.tick - created_at_tick
        let elapsed = self.tick.saturating_sub(proc.created_at_tick);
        if elapsed > lim.max_wallclock_ticks {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::WallclockTicks,
                used: elapsed,
                limit: lim.max_wallclock_ticks,
            };
        }
        RlimitVerdict::Ok
    }

    fn rlimit_check(&self, pid: u64, delta: &ResourceUsageDelta) -> RlimitVerdict {
        let Some(proc) = self.processes.get(&pid) else {
            return RlimitVerdict::NoSuchProcess;
        };
        let lim = &proc.limits;
        let new_turns = proc.usage.turns.saturating_add(delta.turns);
        if new_turns > lim.max_turns {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::Turns,
                used: new_turns,
                limit: lim.max_turns,
            };
        }
        let new_calls = proc.usage.tool_calls.saturating_add(delta.tool_calls);
        if new_calls > lim.max_tool_calls {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::ToolCalls,
                used: new_calls,
                limit: lim.max_tool_calls,
            };
        }
        let new_in = proc.usage.tokens_in.saturating_add(delta.tokens_in);
        if new_in > lim.max_tokens_in {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::TokensIn,
                used: new_in,
                limit: lim.max_tokens_in,
            };
        }
        let new_out = proc.usage.tokens_out.saturating_add(delta.tokens_out);
        if new_out > lim.max_tokens_out {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::TokensOut,
                used: new_out,
                limit: lim.max_tokens_out,
            };
        }
        let new_cost = proc.usage.cost_micros.saturating_add(delta.cost_micros);
        if new_cost > lim.max_cost_micros {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::CostMicros,
                used: new_cost,
                limit: lim.max_cost_micros,
            };
        }
        if let Some(b) = delta.last_tool_call_bytes {
            if b > lim.max_tool_call_bytes {
                return RlimitVerdict::Exceeded {
                    dimension: RlimitDim::ToolCallBytes,
                    used: b,
                    limit: lim.max_tool_call_bytes,
                };
            }
        }
        let new_fs = proc.usage.fs_bytes.saturating_add(delta.fs_bytes);
        if new_fs > lim.max_fs_bytes {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::FsBytes,
                used: new_fs,
                limit: lim.max_fs_bytes,
            };
        }
        let elapsed = self.tick.saturating_sub(proc.created_at_tick);
        if elapsed > lim.max_wallclock_ticks {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::WallclockTicks,
                used: elapsed,
                limit: lim.max_wallclock_ticks,
            };
        }
        RlimitVerdict::Ok
    }
}

impl LlmOps for LocalOS {
    fn llm_set_price(&mut self, model: String, price: LlmModelPrice) {
        self.llm_prices.insert(model, price);
    }

    fn llm_price(&self, model: &str) -> LlmModelPrice {
        self.llm_prices
            .get(model)
            .copied()
            .unwrap_or_else(LlmModelPrice::zero)
    }

    fn llm_account(&mut self, pid: u64, report: LlmUsageReport) -> LlmAccountOutcome {
        // 1) price table lookup
        let price = self.llm_price(&report.model);
        let cost_in = (report.prompt_tokens as u128 * price.prompt_per_1k_micros as u128) / 1_000;
        let cost_out =
            (report.completion_tokens as u128 * price.completion_per_1k_micros as u128) / 1_000;
        // saturate to u64 on overflow (defensive; real usage fits easily)
        let charged_cost_micros: u64 = cost_in
            .saturating_add(cost_out)
            .try_into()
            .unwrap_or(u64::MAX);

        // 2) trace: record the accounting event (best-effort; never fails)
        {
            use crate::types::FastMap;
            let mut fields: FastMap<String, String> = FastMap::with_capacity(6);
            fields.insert("model".to_string(), report.model.clone());
            fields.insert(
                "prompt_tokens".to_string(),
                report.prompt_tokens.to_string(),
            );
            fields.insert(
                "completion_tokens".to_string(),
                report.completion_tokens.to_string(),
            );
            fields.insert(
                "cached_prompt_tokens".to_string(),
                report.cached_prompt_tokens.to_string(),
            );
            fields.insert("latency_ms".to_string(), report.latency_ms.to_string());
            fields.insert("cost_micros".to_string(), charged_cost_micros.to_string());
            <Self as TraceOps>::trace_event(
                self,
                "llm.account".to_string(),
                TraceLevel::Info,
                Some(pid),
                fields,
                None,
            );
        }

        // 3) charge rusage (atomic via rusage_charge, which saturates + enforces limits)
        let delta = ResourceUsageDelta {
            tokens_in: report.prompt_tokens,
            tokens_out: report.completion_tokens,
            cost_micros: charged_cost_micros,
            ..Default::default()
        };
        let verdict = <Self as RlimitOps>::rusage_charge(self, pid, delta);

        LlmAccountOutcome {
            charged_cost_micros,
            verdict,
        }
    }
}

/// 敏感路径黑名单（与 agent 侧 FileStore 行为等价，是"根权限不可穿透"的安全边界）。
fn is_sensitive_fs_path(path: &std::path::Path) -> bool {
    let rendered = path.to_string_lossy();
    let rendered = rendered.as_ref();
    if rendered.contains("/.ssh/")
        || rendered.ends_with("/.ssh")
        || rendered.contains("/.gnupg/")
        || rendered.ends_with("/.gnupg")
        || rendered.contains("/.aws/")
        || rendered.ends_with("/.aws")
        || rendered.contains("/.kube/")
        || rendered.ends_with("/.kube")
        || rendered.contains("/.configW")
        || rendered.ends_with("/.configW")
    {
        return true;
    }
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    matches!(
        name,
        "id_rsa"
            | "id_rsa.pub"
            | "id_ed25519"
            | "id_ed25519.pub"
            | "authorized_keys"
            | "known_hosts"
            | ".netrc"
            | ".npmrc"
            | ".pypirc"
            | ".git-credentials"
            | "credentials"
            | "config.json"
    )
}

impl LocalOS {
    /// 统一的 VFS trace 事件工具，给 vfs_* 方法共用。
    fn vfs_emit_trace(
        &mut self,
        op: &'static str,
        pid: Option<u64>,
        path: &std::path::Path,
        bytes: u64,
        verdict: Option<&RlimitVerdict>,
    ) {
        use crate::types::FastMap;
        let mut fields: FastMap<String, String> = FastMap::with_capacity(3);
        fields.insert("path".to_string(), path.display().to_string());
        fields.insert("bytes".to_string(), bytes.to_string());
        if let Some(v) = verdict {
            fields.insert("verdict".to_string(), format!("{:?}", v));
        }
        <Self as TraceOps>::trace_event(
            self,
            format!("vfs.{}", op),
            TraceLevel::Info,
            pid,
            fields,
            None,
        );
    }
}

impl VfsOps for LocalOS {
    // SAFETY/PERF NOTE: the methods on this impl block call into blocking
    // `std::fs` while the global `SharedKernel` mutex is held. They must
    // therefore not be invoked from latency-sensitive async code paths that
    // expect non-blocking semantics — large reads/writes will stall every
    // other tenant of the kernel for the duration of the syscall. Use
    // out-of-band tooling (e.g. dedicated worker threads) for big files.
    fn vfs_read_to_string(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
    ) -> Result<String, VfsError> {
        if is_sensitive_fs_path(path) {
            self.vfs_emit_trace("read.denied", pid, path, 0, None);
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        if !path.exists() {
            self.vfs_emit_trace("read.notfound", pid, path, 0, None);
            return Err(VfsError::NotFound(path.display().to_string()));
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| VfsError::Io(format!("Failed to read file: {}", e)))?;
        let bytes = content.len() as u64;

        // charge fs_bytes（pid 缺失或不受约束时 skip；rlimit 超出则返回 QuotaExceeded）
        let verdict = if let Some(pid) = pid {
            let delta = ResourceUsageDelta {
                fs_bytes: bytes,
                ..Default::default()
            };
            Some(<Self as RlimitOps>::rusage_charge(self, pid, delta))
        } else {
            None
        };

        self.vfs_emit_trace("read", pid, path, bytes, verdict.as_ref());

        if let Some(RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        }) = verdict
        {
            return Err(VfsError::QuotaExceeded {
                dimension,
                used,
                limit,
            });
        }
        Ok(content)
    }

    fn vfs_write_all(
        &mut self,
        pid: Option<u64>,
        path: &std::path::Path,
        content: &str,
    ) -> Result<(), VfsError> {
        if is_sensitive_fs_path(path) {
            self.vfs_emit_trace("write.denied", pid, path, 0, None);
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| VfsError::Io(format!("Failed to create directory: {}", e)))?;
            }
        }
        std::fs::write(path, content)
            .map_err(|e| VfsError::Io(format!("Failed to write file: {}", e)))?;
        let bytes = content.len() as u64;

        let verdict = if let Some(pid) = pid {
            let delta = ResourceUsageDelta {
                fs_bytes: bytes,
                ..Default::default()
            };
            Some(<Self as RlimitOps>::rusage_charge(self, pid, delta))
        } else {
            None
        };

        self.vfs_emit_trace("write", pid, path, bytes, verdict.as_ref());

        if let Some(RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        }) = verdict
        {
            return Err(VfsError::QuotaExceeded {
                dimension,
                used,
                limit,
            });
        }
        Ok(())
    }

    fn vfs_stat(&mut self, path: &std::path::Path) -> Result<VfsStat, VfsError> {
        if is_sensitive_fs_path(path) {
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        let meta = std::fs::metadata(path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => VfsError::NotFound(path.display().to_string()),
            _ => VfsError::Io(e.to_string()),
        })?;
        Ok(VfsStat {
            size: meta.len(),
            is_file: meta.is_file(),
            is_dir: meta.is_dir(),
        })
    }

    fn vfs_remove_file(&mut self, path: &std::path::Path) -> Result<(), VfsError> {
        if is_sensitive_fs_path(path) {
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        std::fs::remove_file(path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => VfsError::NotFound(path.display().to_string()),
            _ => VfsError::Io(e.to_string()),
        })?;
        self.vfs_emit_trace("remove", None, path, 0, None);
        Ok(())
    }
}

impl LocalOS {
    /// DaemonOps 的共用 trace emitter。
    fn daemon_emit_trace(
        &mut self,
        op: &'static str,
        handle: DaemonHandle,
        label: &str,
        kind: DaemonKind,
        parent_pid: Option<u64>,
        err: Option<&str>,
    ) {
        use crate::types::FastMap;
        let mut fields: FastMap<String, String> = FastMap::with_capacity(4);
        fields.insert("handle".to_string(), handle.raw().to_string());
        fields.insert("label".to_string(), label.to_string());
        fields.insert("kind".to_string(), kind.as_str().to_string());
        if let Some(err) = err {
            fields.insert("error".to_string(), err.to_string());
        }
        <Self as TraceOps>::trace_event(
            self,
            format!("daemon.{}", op),
            match op {
                "spawn" => TraceLevel::Info,
                "exit" => TraceLevel::Info,
                "cancel" => TraceLevel::Warn,
                "failed" => TraceLevel::Error,
                _ => TraceLevel::Info,
            },
            parent_pid,
            fields,
            None,
        );
    }

    fn daemon_snapshot(&self, handle: DaemonHandle, entry: &DaemonEntry) -> DaemonEntrySnapshot {
        DaemonEntrySnapshot {
            handle,
            label: entry.label.clone(),
            kind: entry.kind,
            state: entry.state,
            parent_pid: entry.parent_pid,
            spawn_tick: entry.spawn_tick,
            exit_tick: entry.exit_tick,
            last_error: entry.last_error.clone(),
        }
    }

    fn channel_allows_sender(&self, owner_pid: Option<u64>, sender_pid: Option<u64>) -> bool {
        match (owner_pid, sender_pid) {
            (_, None) => true,
            (None, _) => true,
            (Some(owner), Some(sender)) if owner == sender => true,
            (Some(owner), Some(sender)) => {
                self.is_same_process_group(sender, owner)
                    || self.is_ancestor_of(sender, owner)
                    || self.is_ancestor_of(owner, sender)
                    || self.is_sibling(sender, owner)
            }
        }
    }

    fn channel_allows_receiver(&self, owner_pid: Option<u64>, receiver_pid: Option<u64>) -> bool {
        match (owner_pid, receiver_pid) {
            (_, None) => true,
            (None, _) => true,
            (Some(owner), Some(receiver)) => owner == receiver,
        }
    }

    fn channel_emit_trace(
        &mut self,
        op: &'static str,
        channel: ChannelId,
        pid: Option<u64>,
        label: &str,
        depth: usize,
    ) {
        use crate::types::FastMap;
        let mut fields: FastMap<String, String> = FastMap::with_capacity(3);
        fields.insert("channel".to_string(), channel.raw().to_string());
        fields.insert("label".to_string(), label.to_string());
        fields.insert("depth".to_string(), depth.to_string());
        <Self as TraceOps>::trace_event(
            self,
            format!("ipc.{}", op),
            TraceLevel::Info,
            pid,
            fields,
            None,
        );
    }

    fn channel_can_manage(&self, owner_pid: Option<u64>, caller_pid: Option<u64>) -> bool {
        match (owner_pid, caller_pid) {
            (_, None) => true,
            (None, _) => true,
            (Some(owner), Some(caller)) => owner == caller,
        }
    }

    fn channel_is_gc_eligible(entry: &IpcChannelEntry) -> bool {
        entry.closed && entry.queue.is_empty() && entry.ref_count == 0
    }

    fn flatten_ref_holders(holders: &[(String, u32)]) -> Vec<String> {
        let total: usize = holders.iter().map(|(_, c)| *c as usize).sum();
        let mut out = Vec::with_capacity(total);
        for (name, count) in holders {
            for _ in 0..*count {
                out.push(name.clone());
            }
        }
        out
    }

    fn epoll_actual_events_for_source(&self, source: EpollSource) -> EpollEventMask {
        match source {
            EpollSource::Event(event_id) => {
                if self.completed_events.contains(&event_id) {
                    EpollEventMask::IN
                } else {
                    EpollEventMask::EMPTY
                }
            }
            EpollSource::Channel(channel) => match self.channels.get(&channel.0) {
                Some(entry) => {
                    let mut mask = EpollEventMask::EMPTY;
                    if !entry.queue.is_empty() {
                        mask |= EpollEventMask::IN;
                    }
                    if entry.closed {
                        mask |= EpollEventMask::HUP;
                    }
                    mask
                }
                None => EpollEventMask::ERR,
            },
            EpollSource::Futex { addr, expected } => match self.futexes.get(&addr.0) {
                Some(state) => {
                    let current = state.value.load(std::sync::atomic::Ordering::SeqCst);
                    if current != expected {
                        EpollEventMask::IN
                    } else {
                        EpollEventMask::EMPTY
                    }
                }
                None => EpollEventMask::ERR,
            },
        }
    }

    fn epoll_wait_event_id_for_registration(
        &self,
        registration: &EpollRegistration,
    ) -> Option<EventId> {
        match registration.snapshot.source {
            EpollSource::Event(event_id) => {
                (!self.completed_events.contains(&event_id)).then_some(event_id)
            }
            EpollSource::Channel(channel) => {
                // 单次 channels 查表得到 entry，然后直接判定 mask，避免再走
                // `epoll_actual_events_for_source` 二次查表。
                let entry = self.channels.get(&channel.0)?;
                let has_data = !entry.queue.is_empty();
                if has_data || entry.closed {
                    None
                } else {
                    Some(entry.event_id)
                }
            }
            EpollSource::Futex { addr, expected } => self.futexes.get(&addr.0).and_then(|state| {
                let current = state.value.load(std::sync::atomic::Ordering::SeqCst);
                let seq_cursor = registration.futex_seq_cursor.unwrap_or(state.seq);
                if current != expected || state.seq != seq_cursor {
                    None
                } else {
                    Some(state.event_id)
                }
            }),
        }
    }

    fn epoll_collect_ready_for_registration(
        &self,
        registration: &EpollRegistration,
    ) -> EpollEventMask {
        match registration.snapshot.source {
            EpollSource::Futex { addr, expected } => match self.futexes.get(&addr.0) {
                Some(state) => {
                    let current = state.value.load(std::sync::atomic::Ordering::SeqCst);
                    let seq_cursor = registration.futex_seq_cursor.unwrap_or(state.seq);
                    let mut actual = EpollEventMask::EMPTY;
                    if current != expected || state.seq != seq_cursor {
                        actual |= EpollEventMask::IN;
                    }
                    actual & registration.snapshot.events
                }
                None => EpollEventMask::ERR & registration.snapshot.events,
            },
            _ => {
                let actual = self.epoll_actual_events_for_source(registration.snapshot.source);
                actual & registration.snapshot.events
            }
        }
    }

    fn epoll_collect_ready(
        &self,
        registrations: &[EpollRegistration],
        max_events: usize,
    ) -> Vec<EpollReadyEvent> {
        let mut ready = Vec::new();
        let limit = max_events.max(1);
        for registration in registrations {
            let matched = self.epoll_collect_ready_for_registration(registration);
            if matched.is_empty() {
                continue;
            }
            ready.push(EpollReadyEvent {
                source: registration.snapshot.source,
                events: matched,
                user_data: registration.snapshot.user_data,
            });
            if ready.len() >= limit {
                break;
            }
        }
        ready
    }

    fn epoll_collect_wait_ids(&self, registrations: &[EpollRegistration]) -> Vec<EventId> {
        let mut wait_ids = Vec::new();
        // 用临时集合做去重，避免每次插入都对 wait_ids 做线性扫描。
        // 在 registration 数大的 epoll 上，O(n^2) 退化为 O(n)。
        let mut seen: FastSet<EventId> = FastSet::default();
        for registration in registrations {
            let actual = self.epoll_collect_ready_for_registration(registration);
            if !actual.is_empty() {
                continue;
            }
            if let Some(event_id) = self.epoll_wait_event_id_for_registration(registration)
                && seen.insert(event_id)
            {
                wait_ids.push(event_id);
            }
        }
        wait_ids
    }

    fn epoll_snapshot_from_entry(&self, epoll: EpollId, entry: &EpollEntry) -> EpollSnapshot {
        let mut registrations = entry
            .registrations
            .values()
            .map(|registration| registration.snapshot.clone())
            .collect::<Vec<_>>();
        registrations.sort_by_key(|item| item.user_data);
        EpollSnapshot {
            epoll,
            label: entry.label.clone(),
            registrations,
        }
    }

    fn complete_futex_event(&mut self, addr: FutexAddr, rotate: bool) {
        let Some(event_id) = self.futexes.get(&addr.0).map(|state| state.event_id) else {
            return;
        };
        self.notify_events_completed(&[event_id]);
        if rotate {
            // 轮转时老 event_id 不再被任何 source 索引，换上新的 event_id
            // 同步调整 event_source_refs 以保持严格配对。
            self.dec_event_source_ref(event_id);
            let next_event_id = self.alloc_internal_event_id();
            if let Some(state) = self.futexes.get_mut(&addr.0) {
                state.event_id = next_event_id;
            }
            self.inc_event_source_ref(next_event_id);
        }
    }

    fn epoll_refresh_futex_cursors(&mut self, epoll: EpollId, ready: &[EpollReadyEvent]) {
        let mut futex_updates = Vec::new();
        for event in ready {
            let EpollSource::Futex { addr, .. } = event.source else {
                continue;
            };
            futex_updates.push((event.source, self.futex_seq(addr)));
        }
        let Some(entry) = self.epolls.get_mut(&epoll.0) else {
            return;
        };
        for (source, current_seq) in futex_updates {
            if let Some(registration) = entry.registrations.get_mut(&source) {
                registration.futex_seq_cursor = current_seq;
            }
        }
    }

    fn alloc_internal_event_id(&mut self) -> EventId {
        let event_id = EventId::new(self.next_internal_event_id);
        self.next_internal_event_id += 1;
        event_id
    }
}

impl DaemonOps for LocalOS {
    fn daemon_register(
        &mut self,
        label: String,
        kind: DaemonKind,
        parent_pid: Option<u64>,
    ) -> (DaemonHandle, DaemonCancelToken) {
        let id = self.next_daemon_id;
        self.next_daemon_id += 1;
        let handle = DaemonHandle(id);
        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        let entry = DaemonEntry {
            label: label.clone(),
            kind,
            state: DaemonState::Running,
            parent_pid,
            spawn_tick: self.tick,
            exit_tick: None,
            last_error: None,
            cancel_flag: cancel_flag.clone(),
        };
        self.daemons.insert(id, entry);

        self.daemon_emit_trace("spawn", handle, &label, kind, parent_pid, None);
        (handle, DaemonCancelToken(cancel_flag))
    }

    fn daemon_exit(&mut self, handle: DaemonHandle, err: Option<String>) {
        let (label, kind, parent_pid, new_state, err_clone) = {
            let Some(entry) = self.daemons.get_mut(&handle.0) else {
                return;
            };
            // 若之前已经 cancel，退出 state 保持 Cancelled（cancel 是更显著的语义）。
            let state = match (entry.state, &err) {
                (DaemonState::Cancelled, _) => DaemonState::Cancelled,
                (_, None) => DaemonState::Exited,
                (_, Some(_)) => DaemonState::Failed,
            };
            entry.state = state;
            entry.exit_tick = Some(self.tick);
            entry.last_error = err.clone();
            (
                entry.label.clone(),
                entry.kind,
                entry.parent_pid,
                state,
                err.clone(),
            )
        };
        let op = match new_state {
            DaemonState::Failed => "failed",
            _ => "exit",
        };
        self.daemon_emit_trace(op, handle, &label, kind, parent_pid, err_clone.as_deref());
    }

    fn cancel_daemon(&mut self, handle: DaemonHandle) -> bool {
        let (label, kind, parent_pid) = {
            let Some(entry) = self.daemons.get_mut(&handle.0) else {
                return false;
            };
            if !matches!(entry.state, DaemonState::Running) {
                return false;
            }
            entry.state = DaemonState::Cancelled;
            entry
                .cancel_flag
                .store(true, std::sync::atomic::Ordering::Release);
            (entry.label.clone(), entry.kind, entry.parent_pid)
        };
        self.daemon_emit_trace("cancel", handle, &label, kind, parent_pid, None);
        true
    }

    fn daemon_status(&self, handle: DaemonHandle) -> Option<DaemonEntrySnapshot> {
        self.daemons
            .get(&handle.0)
            .map(|e| self.daemon_snapshot(handle, e))
    }

    fn list_daemons(&self) -> Vec<DaemonEntrySnapshot> {
        self.daemons
            .iter()
            .map(|(id, e)| self.daemon_snapshot(DaemonHandle(*id), e))
            .collect()
    }
}

impl IpcOps for LocalOS {
    fn channel_create(
        &mut self,
        owner_pid: Option<u64>,
        capacity: usize,
        label: String,
    ) -> ChannelId {
        self.channel_create_tagged(owner_pid, capacity, label, ChannelOwnerTag::General, 0)
    }

    fn channel_create_tagged(
        &mut self,
        owner_pid: Option<u64>,
        capacity: usize,
        label: String,
        owner_tag: ChannelOwnerTag,
        initial_ref_count: u32,
    ) -> ChannelId {
        let initial_ref_holders = (0..initial_ref_count)
            .map(|i| format!("{}#{}", owner_tag.as_str(), i))
            .collect::<Vec<_>>();
        self.channel_create_tagged_with_holders(
            owner_pid,
            capacity,
            label,
            owner_tag,
            initial_ref_holders,
        )
    }

    fn channel_create_tagged_with_holders(
        &mut self,
        owner_pid: Option<u64>,
        capacity: usize,
        label: String,
        owner_tag: ChannelOwnerTag,
        initial_ref_holders: Vec<String>,
    ) -> ChannelId {
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        let event_id = self.alloc_internal_event_id();
        let cap = capacity.max(1);
        let initial_count = initial_ref_holders.len() as u32;
        let mut holder_counts: Vec<(String, u32)> = Vec::with_capacity(initial_ref_holders.len());
        for name in initial_ref_holders {
            if let Some(slot) = holder_counts.iter_mut().find(|(n, _)| *n == name) {
                slot.1 = slot.1.saturating_add(1);
            } else {
                holder_counts.push((name, 1));
            }
        }
        self.channels.insert(
            id,
            IpcChannelEntry {
                owner_pid,
                label: label.clone(),
                owner_tag,
                ref_count: initial_count,
                ref_holders: holder_counts,
                event_id,
                capacity: cap,
                queue: VecDeque::new(),
                closed: false,
            },
        );
        // 与 channel 生命周期绑定的 event_id 在表中记一次引用。
        self.inc_event_source_ref(event_id);
        let channel = ChannelId(id);
        self.channel_emit_trace("channel_create", channel, owner_pid, &label, 0);
        channel
    }

    fn channel_event_id(&self, channel: ChannelId) -> Option<EventId> {
        self.channels.get(&channel.0).map(|entry| entry.event_id)
    }

    fn channel_meta(&self, channel: ChannelId) -> Option<ChannelMetaSnapshot> {
        self.channels
            .get(&channel.0)
            .map(|entry| ChannelMetaSnapshot {
                channel,
                label: entry.label.clone(),
                owner_pid: entry.owner_pid,
                owner_tag: entry.owner_tag,
                ref_count: entry.ref_count,
                ref_holders: Self::flatten_ref_holders(&entry.ref_holders),
                queued_len: entry.queue.len(),
                closed: entry.closed,
            })
    }

    fn list_channels(&self) -> Vec<ChannelMetaSnapshot> {
        let mut items = self
            .channels
            .iter()
            .map(|(id, entry)| ChannelMetaSnapshot {
                channel: ChannelId(*id),
                label: entry.label.clone(),
                owner_pid: entry.owner_pid,
                owner_tag: entry.owner_tag,
                ref_count: entry.ref_count,
                ref_holders: Self::flatten_ref_holders(&entry.ref_holders),
                queued_len: entry.queue.len(),
                closed: entry.closed,
            })
            .collect::<Vec<_>>();
        items.sort_by_key(|m| m.channel.raw());
        items
    }

    fn channel_send(
        &mut self,
        sender_pid: Option<u64>,
        channel: ChannelId,
        message: String,
    ) -> Result<(), String> {
        let owner_pid = self.channels.get(&channel.0).map(|e| e.owner_pid);
        if !self.channel_allows_sender(owner_pid.flatten(), sender_pid) {
            return Err(format!(
                "Permission denied: pid {:?} cannot send to channel {} owned by {:?}.",
                sender_pid,
                channel,
                owner_pid.flatten()
            ));
        }
        let (label, depth, should_notify, event_id) = {
            let entry = self
                .channels
                .get_mut(&channel.0)
                .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
            if entry.closed {
                return Err(format!("Channel {} is closed.", channel));
            }
            if entry.queue.len() >= entry.capacity {
                return Err(format!(
                    "Channel {} is full (capacity: {}).",
                    channel, entry.capacity
                ));
            }
            let should_notify = entry.queue.is_empty();
            entry.queue.push_back(message);
            (
                entry.label.clone(),
                entry.queue.len(),
                should_notify,
                entry.event_id,
            )
        };
        if should_notify {
            self.notify_events_completed(&[event_id]);
            // 旧 event 不再是 channel 的 source，转换后由新 event 负责后续 waiter
            // 的 live 判定。
            self.dec_event_source_ref(event_id);
            let next_event_id = self.alloc_internal_event_id();
            if let Some(entry) = self.channels.get_mut(&channel.0) {
                entry.event_id = next_event_id;
            }
            self.inc_event_source_ref(next_event_id);
        }
        self.channel_emit_trace("send", channel, sender_pid, &label, depth);
        Ok(())
    }

    fn channel_try_recv(
        &mut self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<IpcRecvResult, String> {
        let owner_pid = self.channels.get(&channel.0).map(|e| e.owner_pid);
        if !self.channel_allows_receiver(owner_pid.flatten(), receiver_pid) {
            return Err(format!(
                "Permission denied: pid {:?} cannot receive from channel {} owned by {:?}.",
                receiver_pid,
                channel,
                owner_pid.flatten()
            ));
        }
        let (result, label, depth) = {
            let entry = self
                .channels
                .get_mut(&channel.0)
                .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
            let res = if let Some(msg) = entry.queue.pop_front() {
                IpcRecvResult::Message(msg)
            } else if entry.closed {
                IpcRecvResult::Closed
            } else {
                IpcRecvResult::Empty
            };
            (res, entry.label.clone(), entry.queue.len())
        };
        self.channel_emit_trace("recv", channel, receiver_pid, &label, depth);
        Ok(result)
    }

    fn channel_peek(
        &self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<IpcRecvResult, String> {
        let entry = self
            .channels
            .get(&channel.0)
            .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
        if !self.channel_allows_receiver(entry.owner_pid, receiver_pid) {
            return Err(format!(
                "Permission denied: pid {:?} cannot receive from channel {} owned by {:?}.",
                receiver_pid, channel, entry.owner_pid
            ));
        }
        if let Some(msg) = entry.queue.front() {
            Ok(IpcRecvResult::Message(msg.clone()))
        } else if entry.closed {
            Ok(IpcRecvResult::Closed)
        } else {
            Ok(IpcRecvResult::Empty)
        }
    }

    fn channel_peek_all(
        &self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<Vec<String>, String> {
        let entry = self
            .channels
            .get(&channel.0)
            .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
        if !self.channel_allows_receiver(entry.owner_pid, receiver_pid) {
            return Err(format!(
                "Permission denied: pid {:?} cannot receive from channel {} owned by {:?}.",
                receiver_pid, channel, entry.owner_pid
            ));
        }
        Ok(entry.queue.iter().cloned().collect())
    }

    fn channel_try_recv_all(
        &mut self,
        receiver_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<Vec<String>, String> {
        let (label, messages) = {
            let owner_pid = self.channels.get(&channel.0).map(|e| e.owner_pid);
            if !self.channel_allows_receiver(owner_pid.flatten(), receiver_pid) {
                return Err(format!(
                    "Permission denied: pid {:?} cannot receive from channel {} owned by {:?}.",
                    receiver_pid,
                    channel,
                    owner_pid.flatten()
                ));
            }
            let entry = self
                .channels
                .get_mut(&channel.0)
                .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
            let drained = entry.queue.drain(..).collect::<Vec<_>>();
            (entry.label.clone(), drained)
        };
        self.channel_emit_trace("recv", channel, receiver_pid, &label, 0);
        Ok(messages)
    }

    fn channel_retain(&mut self, channel: ChannelId) -> Result<u32, String> {
        let next_idx = self
            .channels
            .get(&channel.0)
            .ok_or_else(|| format!("Channel {} does not exist.", channel))?
            .ref_count;
        self.channel_retain_named(channel, format!("retain#{}", next_idx))
    }

    fn channel_retain_named(&mut self, channel: ChannelId, holder: String) -> Result<u32, String> {
        let entry = self
            .channels
            .get_mut(&channel.0)
            .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
        if let Some(slot) = entry
            .ref_holders
            .iter_mut()
            .find(|(name, _)| *name == holder)
        {
            slot.1 = slot.1.saturating_add(1);
        } else {
            entry.ref_holders.push((holder, 1));
        }
        entry.ref_count = entry.ref_count.saturating_add(1);
        Ok(entry.ref_count)
    }

    fn channel_release(&mut self, channel: ChannelId) -> Result<u32, String> {
        let holder = self
            .channels
            .get(&channel.0)
            .and_then(|entry| entry.ref_holders.last().map(|(name, _)| name.clone()))
            .ok_or_else(|| format!("Channel {} ref_count is already zero.", channel))?;
        self.channel_release_named(channel, &holder)
    }

    fn channel_release_named(&mut self, channel: ChannelId, holder: &str) -> Result<u32, String> {
        let entry = self
            .channels
            .get_mut(&channel.0)
            .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
        if entry.ref_count == 0 {
            return Err(format!("Channel {} ref_count is already zero.", channel));
        }
        let Some(idx) = entry
            .ref_holders
            .iter()
            .position(|(name, _)| name == holder)
        else {
            return Err(format!(
                "Channel {} does not have ref holder {:?}.",
                channel, holder
            ));
        };
        entry.ref_holders[idx].1 -= 1;
        if entry.ref_holders[idx].1 == 0 {
            entry.ref_holders.remove(idx);
        }
        entry.ref_count -= 1;
        Ok(entry.ref_count)
    }

    fn channel_destroy(
        &mut self,
        caller_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<(), String> {
        let (owner_pid, label, eligible) = {
            let entry = self
                .channels
                .get(&channel.0)
                .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
            (
                entry.owner_pid,
                entry.label.clone(),
                Self::channel_is_gc_eligible(entry),
            )
        };
        if !self.channel_can_manage(owner_pid, caller_pid) {
            return Err(format!(
                "Permission denied: pid {:?} cannot destroy channel {} owned by {:?}.",
                caller_pid, channel, owner_pid
            ));
        }
        if !eligible {
            return Err(format!(
                "Channel {} is not destroyable yet; it must be closed, empty, and have ref_count=0.",
                channel
            ));
        }
        let removed_event_id = self.channels.remove(&channel.0).map(|e| e.event_id);
        if let Some(event_id) = removed_event_id {
            self.dec_event_source_ref(event_id);
        }
        self.channel_emit_trace("destroy", channel, caller_pid, &label, 0);
        Ok(())
    }

    fn channel_gc_closed_empty(&mut self) -> usize {
        let doomed = self
            .channels
            .iter()
            .filter_map(|(id, entry)| {
                if Self::channel_is_gc_eligible(entry) {
                    Some((*id, entry.label.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for (id, label) in &doomed {
            if let Some(removed) = self.channels.remove(id) {
                self.dec_event_source_ref(removed.event_id);
            }
            self.channel_emit_trace("gc", ChannelId(*id), None, label, 0);
        }
        doomed.len()
    }

    fn channel_close(&mut self, closer_pid: Option<u64>, channel: ChannelId) -> Result<(), String> {
        let owner_pid = self.channels.get(&channel.0).map(|e| e.owner_pid);
        if !self.channel_allows_receiver(owner_pid.flatten(), closer_pid) {
            return Err(format!(
                "Permission denied: pid {:?} cannot close channel {} owned by {:?}.",
                closer_pid,
                channel,
                owner_pid.flatten()
            ));
        }
        let (label, depth, should_notify, event_id) = {
            let entry = self
                .channels
                .get_mut(&channel.0)
                .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
            let should_notify = entry.queue.is_empty() && !entry.closed;
            entry.closed = true;
            (
                entry.label.clone(),
                entry.queue.len(),
                should_notify,
                entry.event_id,
            )
        };
        if should_notify {
            self.notify_events_completed(&[event_id]);
            // 旧 event 不再是 channel 的 source，转换后由新 event 负责后续 waiter
            // 的 live 判定。
            self.dec_event_source_ref(event_id);
            let next_event_id = self.alloc_internal_event_id();
            if let Some(entry) = self.channels.get_mut(&channel.0) {
                entry.event_id = next_event_id;
            }
            self.inc_event_source_ref(next_event_id);
        }
        self.channel_emit_trace("close", channel, closer_pid, &label, depth);
        Ok(())
    }
}

impl EpollOps for LocalOS {
    fn epoll_create(&mut self, label: String) -> EpollId {
        let id = self.next_epoll_id;
        self.next_epoll_id += 1;
        self.epolls.insert(
            id,
            EpollEntry {
                label,
                registrations: FastMap::default(),
            },
        );
        EpollId(id)
    }

    fn epoll_ctl_add(
        &mut self,
        epoll: EpollId,
        source: EpollSource,
        events: EpollEventMask,
        user_data: u64,
    ) -> Result<(), String> {
        let futex_seq_cursor = match source {
            EpollSource::Futex { addr, .. } => self.futex_seq(addr),
            _ => None,
        };
        let entry = self
            .epolls
            .get_mut(&epoll.0)
            .ok_or_else(|| format!("Epoll {} does not exist.", epoll))?;
        if events.is_empty() {
            return Err("epoll interest mask cannot be empty.".to_string());
        }
        if entry.registrations.contains_key(&source) {
            return Err(format!("Epoll {} already watches {:?}.", epoll, source));
        }
        entry.registrations.insert(
            source,
            EpollRegistration {
                snapshot: EpollRegistrationSnapshot {
                    source,
                    events,
                    user_data,
                },
                futex_seq_cursor,
            },
        );
        if let EpollSource::Event(event_id) = source {
            self.inc_event_source_ref(event_id);
        }
        Ok(())
    }

    fn epoll_ctl_mod(
        &mut self,
        epoll: EpollId,
        source: EpollSource,
        events: EpollEventMask,
        user_data: u64,
    ) -> Result<(), String> {
        let futex_seq_cursor = match source {
            EpollSource::Futex { addr, .. } => self.futex_seq(addr),
            _ => None,
        };
        let entry = self
            .epolls
            .get_mut(&epoll.0)
            .ok_or_else(|| format!("Epoll {} does not exist.", epoll))?;
        if events.is_empty() {
            return Err("epoll interest mask cannot be empty.".to_string());
        }
        let registration = entry
            .registrations
            .get_mut(&source)
            .ok_or_else(|| format!("Epoll {} does not watch {:?}.", epoll, source))?;
        registration.snapshot.events = events;
        registration.snapshot.user_data = user_data;
        registration.futex_seq_cursor = futex_seq_cursor;
        Ok(())
    }

    fn epoll_ctl_del(&mut self, epoll: EpollId, source: EpollSource) -> Result<(), String> {
        let entry = self
            .epolls
            .get_mut(&epoll.0)
            .ok_or_else(|| format!("Epoll {} does not exist.", epoll))?;
        if entry.registrations.remove(&source).is_none() {
            return Err(format!("Epoll {} does not watch {:?}.", epoll, source));
        }
        if let EpollSource::Event(event_id) = source {
            self.dec_event_source_ref(event_id);
        }
        Ok(())
    }

    fn epoll_wait(
        &mut self,
        epoll: EpollId,
        max_events: usize,
        timeout_ticks: Option<u64>,
    ) -> Result<EpollWaitResult, String> {
        let registrations = self
            .epolls
            .get(&epoll.0)
            .ok_or_else(|| format!("Epoll {} does not exist.", epoll))?
            .registrations
            .values()
            .cloned()
            .collect::<Vec<_>>();
        if registrations.is_empty() {
            return Ok(EpollWaitResult::Ready(Vec::new()));
        }

        let ready = self.epoll_collect_ready(&registrations, max_events);
        if !ready.is_empty() {
            self.epoll_refresh_futex_cursors(epoll, &ready);
            return Ok(EpollWaitResult::Ready(ready));
        }

        let wait_ids = self.epoll_collect_wait_ids(&registrations);
        if wait_ids.is_empty() {
            return Ok(EpollWaitResult::Ready(Vec::new()));
        }

        let timeout_tick = self.wait_on_events(wait_ids, WaitPolicy::Any, timeout_ticks)?;
        if self.consume_yield_requested() || timeout_tick.is_some() {
            return Ok(EpollWaitResult::Suspended { timeout_tick });
        }

        let ready = self.epoll_collect_ready(&registrations, max_events);
        self.epoll_refresh_futex_cursors(epoll, &ready);
        Ok(EpollWaitResult::Ready(ready))
    }

    fn epoll_snapshot(&self, epoll: EpollId) -> Option<EpollSnapshot> {
        self.epolls
            .get(&epoll.0)
            .map(|entry| self.epoll_snapshot_from_entry(epoll, entry))
    }

    fn epoll_destroy(&mut self, epoll: EpollId) -> bool {
        let Some(entry) = self.epolls.remove(&epoll.0) else {
            return false;
        };
        // 释放该 epoll 中所有 EpollSource::Event 的引用计数。
        for registration in entry.registrations.values() {
            if let EpollSource::Event(event_id) = registration.snapshot.source {
                self.dec_event_source_ref(event_id);
            }
        }
        true
    }
}

impl Kernel for LocalOS {}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{LocalOS, ShmReadError};
    use crate::kernel::{
        EventId, KernelInternal, ProcessCapabilities, ProcessState, Signal, Syscall, WaitPolicy,
        WaitReason,
    };

    #[test]
    fn foreground_process_can_wait_and_resume_on_child_exit() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );
        let child = os
            .spawn(
                Some(root),
                "child".to_string(),
                "child goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.wait_on(child).unwrap();
        assert!(os.consume_yield_requested());
        assert!(os.current_process_id().is_none());
        assert!(matches!(
            os.get_process(root).map(|p| &p.state),
            Some(ProcessState::Waiting { reason: WaitReason::ProcessExit { on_pid } }) if *on_pid == child
        ));

        let resumed = os.pop_ready().unwrap();
        assert_eq!(resumed.pid, child);
        os.terminate_current("child done".to_string());

        let root_proc = os.get_process(root).unwrap();
        assert_eq!(root_proc.state, ProcessState::Ready);
        assert_eq!(root_proc.mailbox.len(), 1);
    }

    #[test]
    fn foreground_process_can_wait_on_events() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );

        let timeout_tick = os
            .wait_on_events(
                vec![EventId::new(1), EventId::new(2)],
                WaitPolicy::Any,
                Some(3),
            )
            .unwrap();

        assert_eq!(timeout_tick, Some(3));
        assert!(os.consume_yield_requested());
        assert!(os.current_process_id().is_none());
        assert!(matches!(
            os.get_process(root).map(|p| &p.state),
            Some(ProcessState::Waiting {
                reason: WaitReason::Events {
                    event_ids,
                    policy: WaitPolicy::Any,
                    timeout_tick: Some(3),
                }
            }) if event_ids == &vec![EventId::new(1), EventId::new(2)]
        ));
    }

    #[test]
    fn event_wait_timeout_wakes_process() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );

        os.wait_on_events(vec![EventId::new(1)], WaitPolicy::All, Some(2))
            .unwrap();
        os.advance_tick();
        assert!(matches!(
            os.get_process(root).map(|p| &p.state),
            Some(ProcessState::Waiting {
                reason: WaitReason::Events { .. }
            })
        ));

        os.advance_tick();
        let root_proc = os.get_process(root).unwrap();
        assert_eq!(root_proc.state, ProcessState::Ready);
        assert_eq!(
            root_proc.mailbox.back().map(|s| s.as_str()),
            Some("Event wait timeout reached at scheduler tick 2.")
        );
    }

    #[test]
    fn event_completion_wakes_waiting_process() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );

        os.wait_on_events(
            vec![EventId::new(1), EventId::new(2)],
            WaitPolicy::Any,
            None,
        )
        .unwrap();

        let woke = os.notify_events_completed(&[EventId::new(2)]);
        assert_eq!(woke, vec![root]);

        let root_proc = os.get_process(root).unwrap();
        assert_eq!(root_proc.state, ProcessState::Ready);
        assert_eq!(
            root_proc.mailbox.back().map(|s| s.as_str()),
            Some(
                "[EVENT_WAKE]\nReason: event wait condition satisfied.\nCompleted event ids: evt_2\nRecommended next actions:\n1. Inspect the event-producing subsystem for fresh state.\n2. If these events came from async tool work, use tool_status or tool_wait to collect results.\n3. Cancel low-value still-running branches when appropriate.\n4. If enough results are already available, continue reasoning immediately."
            )
        );
    }

    #[test]
    fn ready_queue_insertion_preserves_priority_order() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );
        let low_priority_pid = os
            .spawn(
                Some(root),
                "low".to_string(),
                "low priority".to_string(),
                30,
                4,
                None,
                None,
            )
            .unwrap();
        let high_priority_pid = os
            .spawn(
                Some(root),
                "high".to_string(),
                "high priority".to_string(),
                5,
                4,
                None,
                None,
            )
            .unwrap();
        let mid_priority_pid = os
            .spawn(
                Some(root),
                "mid".to_string(),
                "mid priority".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        assert_eq!(os.pop_ready().map(|proc| proc.pid), Some(high_priority_pid));
        assert_eq!(os.pop_ready().map(|proc| proc.pid), Some(mid_priority_pid));
        assert_eq!(os.pop_ready().map(|proc| proc.pid), Some(low_priority_pid));
    }

    #[test]
    fn completed_events_are_retention_bounded_for_inactive_events() {
        let mut os = LocalOS::new();
        os.completed_event_retention = 2;

        os.notify_events_completed(&[EventId::new(1)]);
        os.notify_events_completed(&[EventId::new(2)]);
        os.notify_events_completed(&[EventId::new(3)]);

        assert!(!os.event_is_completed(EventId::new(1)));
        assert!(os.event_is_completed(EventId::new(2)));
        assert!(os.event_is_completed(EventId::new(3)));
        assert_eq!(os.completed_events.len(), 2);
    }

    #[test]
    fn completed_event_retention_preserves_live_epoll_event_sources() {
        use crate::primitives::{EpollEventMask, EpollOps, EpollSource};

        let mut os = LocalOS::new();
        os.completed_event_retention = 1;
        let watched_event = EventId::new(10);
        let epoll = os.epoll_create("live-event".to_string());
        os.epoll_ctl_add(
            epoll,
            EpollSource::Event(watched_event),
            EpollEventMask::IN,
            10,
        )
        .unwrap();

        os.notify_events_completed(&[watched_event]);
        os.notify_events_completed(&[EventId::new(11)]);
        os.notify_events_completed(&[EventId::new(12)]);

        assert!(os.event_is_completed(watched_event));
        assert!(!os.event_is_completed(EventId::new(11)));
        assert!(os.event_is_completed(EventId::new(12)));
    }

    #[test]
    fn notify_events_completed_uses_waiter_index() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
        let child = os
            .spawn(
                Some(root),
                "c".to_string(),
                "g".to_string(),
                10,
                8,
                None,
                None,
            )
            .unwrap();
        // Run the child so its current_pid context is set, then have it wait.
        let popped = os.pop_ready().unwrap();
        assert_eq!(popped.pid, child);
        let event = EventId::new(42);
        os.wait_on_events(vec![event], WaitPolicy::Any, None)
            .unwrap();
        // Index should now contain the child pid.
        assert!(
            os.event_waiters
                .get(&event)
                .is_some_and(|set| set.contains(&child))
        );
        // Notify wakes the child.
        let woken = os.notify_events_completed(&[event]);
        assert_eq!(woken, vec![child]);
        // The waiter index entry for the completed event should be drained.
        assert!(os.event_waiters.get(&event).is_none());
    }

    #[test]
    fn notify_events_completed_skips_stale_waiters() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
        let child = os
            .spawn(
                Some(root),
                "c".to_string(),
                "g".to_string(),
                10,
                8,
                None,
                None,
            )
            .unwrap();
        let popped = os.pop_ready().unwrap();
        assert_eq!(popped.pid, child);
        let event = EventId::new(99);
        os.wait_on_events(vec![event], WaitPolicy::Any, None)
            .unwrap();
        // Forcibly terminate the child while it is in the waiter index.
        os.terminate_pid(child, "killed".to_string());
        // Stale entry must not be woken (and notify must not panic).
        let woken = os.notify_events_completed(&[event]);
        assert!(woken.is_empty());
    }

    #[test]
    fn descendants_use_persistent_index() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
        let a = os
            .spawn(
                Some(root),
                "a".to_string(),
                "g".to_string(),
                10,
                8,
                None,
                None,
            )
            .unwrap();
        let b = os
            .spawn(Some(a), "b".to_string(), "g".to_string(), 10, 8, None, None)
            .unwrap();
        let c = os
            .spawn(Some(a), "c".to_string(), "g".to_string(), 10, 8, None, None)
            .unwrap();
        let mut descendants = os.collect_descendants(root);
        descendants.sort();
        let mut expected = vec![a, b, c];
        expected.sort();
        assert_eq!(descendants, expected);
        // Index must be properly maintained: removing a node updates parent's set.
        assert!(
            os.children_by_parent
                .get(&a)
                .is_some_and(|s| s.contains(&b) && s.contains(&c))
        );
        os.terminate_pid(b, "done".to_string());
        os.remove_process_entry(b);
        assert!(
            os.children_by_parent
                .get(&a)
                .is_some_and(|s| !s.contains(&b) && s.contains(&c))
        );
    }

    /// 终止进程 + 重新登入应当让 SHM 权限缓存里的旧答复失效，否则
    /// 会出现“owner 已死但 cache 仍允许写”的尴尬窗口。
    #[test]
    fn shm_perm_cache_invalidates_on_topology_change() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
        let owner = os
            .spawn(
                Some(root),
                "owner".to_string(),
                "g".to_string(),
                10,
                8,
                None,
                None,
            )
            .unwrap();
        let stranger = os
            .spawn(
                Some(root),
                "stranger".to_string(),
                "g".to_string(),
                10,
                8,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(owner));
        os.shm_create("k".to_string(), "v".to_string()).unwrap();

        // stranger 当前是 sibling，可读但不可写。
        os.set_current_pid(Some(stranger));
        let entry = os.shared_memory.get("k").unwrap();
        assert!(!os.is_shm_accessible_by(stranger, entry));
        assert!(os.is_shm_readable_by(stranger, entry));

        // 把 stranger 拉进 owner 的 process group：拓扑变更应让缓存里
        // “stranger 不可写”的旧答案失效，重新计算后变为可写。
        let pgid = os.next_pgid;
        os.next_pgid += 1;
        os.set_process_group(owner, pgid).unwrap();
        os.set_process_group(stranger, pgid).unwrap();
        let entry = os.shared_memory.get("k").unwrap();
        assert!(
            os.is_shm_accessible_by(stranger, entry),
            "set_process_group must invalidate stale shm perm cache",
        );

        // 反向：owner terminate 后，cache 也要让 accessible 重新计算。
        os.terminate_pid(owner, "done".to_string());
        os.remove_process_entry(owner);
        let entry = os.shared_memory.get("k").unwrap();
        // owner pid 不再存在；non-owner 走 ancestor / sibling 链路，
        // accessible 取决于 stranger 与 owner 的 pgid，但 owner 已被擦除，
        // 这里只断言查询不会 panic 并返回 bool。
        let _ = os.is_shm_accessible_by(stranger, entry);
    }

    /// `event_source_refs` 与 channel/futex/epoll(EpollSource::Event) 的
    /// 生命周期严格一一对应，destroy 后引用必须归零。
    #[test]
    fn event_source_refs_track_channel_and_futex_lifetimes() {
        use crate::primitives::{FutexOps, IpcOps};
        let mut os = LocalOS::new();
        let _root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);

        let ch = os.channel_create(None, 4, "test".to_string());
        let ch_event = os.channels.get(&ch.0).unwrap().event_id;
        assert_eq!(os.event_source_refs.get(&ch_event).copied(), Some(1));
        assert!(os.completed_event_is_live(ch_event));

        let addr = os.futex_create(0, "f".to_string());
        let fx_event = os.futex_event_id(addr).unwrap();
        assert_eq!(os.event_source_refs.get(&fx_event).copied(), Some(1));

        // futex_destroy 后引用归零、条目必须被清掉以免无界增长。
        assert!(os.futex_destroy(addr));
        assert!(os.event_source_refs.get(&fx_event).is_none());

        // channel destroy 同上。
        os.channel_close(None, ch).unwrap();
        os.channel_destroy(None, ch).unwrap();
        assert!(os.event_source_refs.get(&ch_event).is_none());
    }

    /// epoll 注册 EpollSource::Event 时也应让事件保持 live；del / destroy
    /// 后归零，确保 prune_completed_events 不会过早回收。
    #[test]
    fn event_source_refs_track_epoll_event_registration() {
        use crate::primitives::{EpollEventMask, EpollOps, EpollSource};
        let mut os = LocalOS::new();
        let _root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
        let ep = os.epoll_create("ep".to_string());

        // 用一个内部 event_id；直接构造一个 EpollSource::Event。
        let watched = os.alloc_internal_event_id();
        os.epoll_ctl_add(ep, EpollSource::Event(watched), EpollEventMask::IN, 1)
            .unwrap();
        assert_eq!(os.event_source_refs.get(&watched).copied(), Some(1));
        assert!(os.completed_event_is_live(watched));

        // del 一次：归零并删除条目。
        os.epoll_ctl_del(ep, EpollSource::Event(watched)).unwrap();
        assert!(os.event_source_refs.get(&watched).is_none());

        // 重新注册再走 destroy 路径，destroy 必须把所有 event 引用清掉。
        os.epoll_ctl_add(ep, EpollSource::Event(watched), EpollEventMask::IN, 2)
            .unwrap();
        assert_eq!(os.event_source_refs.get(&watched).copied(), Some(1));
        assert!(os.epoll_destroy(ep));
        assert!(os.event_source_refs.get(&watched).is_none());
    }

    /// 终止大量进程时，ready_queue 不再被线性 retain；tombstone 由
    /// pop_ready 在出队时清掉。
    #[test]
    fn ready_queue_uses_lazy_tombstones_on_termination() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
        let mut spawned = Vec::new();
        for i in 0..32 {
            let pid = os
                .spawn(
                    Some(root),
                    format!("c{i}"),
                    "g".to_string(),
                    50,
                    8,
                    None,
                    None,
                )
                .unwrap();
            spawned.push(pid);
        }
        // begin_foreground 后 root 已是当前运行进程，不在 ready_set
        // 里；spawn 出的子进程全部入 ready。
        assert_eq!(os.ready_count(), spawned.len());
        // 批量终止：ready_set 立即收缩，但 ready_queue 留有 tombstone。
        for &pid in &spawned {
            os.terminate_pid(pid, "x".to_string());
        }
        assert_eq!(os.ready_count(), 0);
        assert!(os.ready_queue.len() >= spawned.len());
        // pop_ready 必须丢弃全部 tombstone 并返回 None。
        assert!(os.pop_ready().is_none());
        // 队列也最终被排空。
        assert!(os.ready_queue.is_empty());
    }

    /// 优先级插入不再为每个比较都查 processes 表；新增高优先级进程
    /// 时应该插到队首。
    #[test]
    fn ready_queue_priority_insertion_uses_cached_priority() {
        let mut os = LocalOS::new();
        // root priority = 10
        let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
        // 后入低优先级（数值更大 -> 更低）
        let low = os
            .spawn(
                Some(root),
                "low".to_string(),
                "g".to_string(),
                100,
                8,
                None,
                None,
            )
            .unwrap();
        // 再入高优先级（数值最小 -> 最高）
        let high = os
            .spawn(
                Some(root),
                "high".to_string(),
                "g".to_string(),
                1,
                8,
                None,
                None,
            )
            .unwrap();
        // 队列中 high 必须排在 low 之前；不必以 root 为参考（begin_foreground
        // 帮 root 设为 Running，不在 ready_set中）。
        let pids: Vec<u64> = os.ready_queue.iter().map(|(pid, _)| *pid).collect();
        let pos_high = pids.iter().position(|p| *p == high).expect("high in queue");
        let pos_low = pids.iter().position(|p| *p == low).expect("low in queue");
        assert!(pos_high < pos_low, "high priority must come before low");
        // 优先级缓存不再需要 processes 查表：pop_ready 顶部必是 high。
        let next = os.pop_ready().unwrap();
        assert_eq!(next.pid, high);
    }

    #[test]
    fn channel_ref_holders_dedupe_by_name() {
        use crate::primitives::IpcOps;
        let mut os = LocalOS::new();
        let ch = os.channel_create(None, 4, "test".to_string());
        os.channel_retain_named(ch, "alpha".to_string()).unwrap();
        os.channel_retain_named(ch, "alpha".to_string()).unwrap();
        os.channel_retain_named(ch, "beta".to_string()).unwrap();
        let meta = os.channel_meta(ch).unwrap();
        assert_eq!(meta.ref_count, 3);
        // Snapshot flattens (alpha, 2) and (beta, 1) -> 3 entries.
        assert_eq!(meta.ref_holders.iter().filter(|h| *h == "alpha").count(), 2);
        assert_eq!(meta.ref_holders.iter().filter(|h| *h == "beta").count(), 1);
        // Internal storage groups duplicates: only 2 unique slots.
        let entry = os.channels.get(&ch.0).unwrap();
        assert_eq!(entry.ref_holders.len(), 2);
        // Releasing one alpha leaves one alpha + one beta.
        os.channel_release_named(ch, "alpha").unwrap();
        let entry = os.channels.get(&ch.0).unwrap();
        assert_eq!(entry.ref_count, 2);
        assert_eq!(entry.ref_holders.len(), 2);
        // Releasing the last alpha drops the slot entirely.
        os.channel_release_named(ch, "alpha").unwrap();
        let entry = os.channels.get(&ch.0).unwrap();
        assert_eq!(entry.ref_holders.len(), 1);
        assert_eq!(entry.ref_holders[0].0, "beta");
    }

    #[test]
    fn foreground_process_enables_env_access() {
        let mut os = LocalOS::new();
        os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );
        os.set_env("scope".to_string(), "root".to_string()).unwrap();
        assert_eq!(os.get_env("scope").as_deref(), Some("root"));
    }

    #[test]
    fn sleeping_process_wakes_after_tick_advance() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );
        let wake_tick = os.sleep_current(2).unwrap();
        assert_eq!(wake_tick, 2);
        assert!(os.consume_yield_requested());
        assert!(matches!(
            os.get_process(root).map(|p| &p.state),
            Some(ProcessState::Sleeping { until_tick }) if *until_tick == 2
        ));

        os.advance_tick();
        assert!(os.pop_ready().is_none());
        os.advance_tick();
        let resumed = os.pop_ready().unwrap();
        assert_eq!(resumed.pid, root);
    }

    #[test]
    fn child_can_be_spawned_with_reduced_capabilities() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );
        let child = os
            .spawn(
                Some(root),
                "restricted".to_string(),
                "restricted goal".to_string(),
                20,
                4,
                Some(ProcessCapabilities {
                    spawn: false,
                    wait: true,
                    ipc_send: false,
                    ipc_receive: true,
                    env_write: false,
                    manage_children: false,
                    sleep: true,
                    reap: false,
                    signal: false,
                }),
                None,
            )
            .unwrap();
        let restricted = os.get_process(child).unwrap();
        assert!(!restricted.capabilities.spawn);
        assert!(!restricted.capabilities.manage_children);
        assert!(restricted.capabilities.sleep);
    }

    #[test]
    fn parent_can_kill_and_reap_descendant() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );
        let child = os
            .spawn(
                Some(root),
                "child".to_string(),
                "child goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        os.kill_process(child, "no longer needed".to_string())
            .unwrap();
        assert!(matches!(
            os.get_process(child).map(|proc| &proc.state),
            Some(ProcessState::Terminated)
        ));
        let result = os.reap_process(child).unwrap();
        assert!(result.contains("no longer needed"));
        assert!(os.get_process(child).is_none());
    }

    #[test]
    fn removing_parent_reparents_live_children_and_collects_unreapable_zombies() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            8,
            None,
        );
        let live_child = os
            .spawn(
                Some(root),
                "live".to_string(),
                "live goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let dead_child = os
            .spawn(
                Some(root),
                "dead".to_string(),
                "dead goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(root));
        os.kill_process(dead_child, "done".to_string()).unwrap();
        os.terminate_pid(root, "root exited".to_string());
        assert!(os.drop_terminated(root));

        let live_proc = os.get_process(live_child).unwrap();
        assert_eq!(live_proc.parent_pid, None);
        assert!(
            live_proc
                .mailbox
                .iter()
                .any(|msg| msg.contains("now orphaned"))
        );
        assert!(os.get_process(dead_child).is_none());
    }

    #[test]
    fn kill_cascades_to_grandchildren() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground(
            "foreground".to_string(),
            "root goal".to_string(),
            10,
            usize::MAX,
            None,
        );
        let child = os
            .spawn(
                Some(root),
                "child".to_string(),
                "child goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let grandchild = os
            .spawn(
                Some(child),
                "grandchild".to_string(),
                "gc goal".to_string(),
                30,
                2,
                None,
                None,
            )
            .unwrap();

        os.kill_process(child, "cascade test".to_string()).unwrap();

        assert!(matches!(
            os.get_process(child).map(|p| &p.state),
            Some(ProcessState::Terminated)
        ));
        assert!(matches!(
            os.get_process(grandchild).map(|p| &p.state),
            Some(ProcessState::Terminated)
        ));
        assert!(
            os.get_process(grandchild)
                .unwrap()
                .result
                .as_ref()
                .unwrap()
                .contains("cascade")
        );
    }

    #[test]
    fn foreground_process_has_is_foreground_flag() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        assert!(os.get_process(root).unwrap().is_foreground);

        let child = os
            .spawn(
                Some(root),
                "bg".to_string(),
                "bg goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        assert!(!os.get_process(child).unwrap().is_foreground);
    }

    #[test]
    fn sigstop_stops_and_sigcont_resumes_process() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "worker".to_string(),
                "work".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.signal_process(child, Signal::SigStop).unwrap();
        assert!(matches!(
            os.get_process(child).map(|p| &p.state),
            Some(ProcessState::Stopped)
        ));

        os.signal_process(child, Signal::SigCont).unwrap();
        assert!(matches!(
            os.get_process(child).map(|p| &p.state),
            Some(ProcessState::Ready)
        ));
        assert!(os.has_ready());
    }

    #[test]
    fn sigkill_immediately_terminates_with_cascade() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "worker".to_string(),
                "work".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let grandchild = os
            .spawn(
                Some(child),
                "gc".to_string(),
                "gc work".to_string(),
                30,
                2,
                None,
                None,
            )
            .unwrap();

        os.signal_process(child, Signal::SigKill).unwrap();
        assert!(matches!(
            os.get_process(child).map(|p| &p.state),
            Some(ProcessState::Terminated)
        ));
        assert!(matches!(
            os.get_process(grandchild).map(|p| &p.state),
            Some(ProcessState::Terminated)
        ));
    }

    #[test]
    fn sigterm_queues_signal_for_graceful_termination() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "worker".to_string(),
                "work".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.signal_process(child, Signal::SigTerm).unwrap();
        let child_proc = os.get_process(child).unwrap();
        assert!(child_proc.pending_signals.contains(&Signal::SigTerm));
    }

    #[test]
    fn sigcancel_is_consumed_without_terminating_process() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);

        os.signal_process(root, Signal::SigCancel).unwrap();
        assert!(
            os.get_process(root)
                .unwrap()
                .pending_signals
                .contains(&Signal::SigCancel)
        );

        os.set_current_pid(Some(root));
        assert!(os.process_pending_signals());
        assert!(matches!(
            os.get_process(root).map(|p| &p.state),
            Some(ProcessState::Ready | ProcessState::Running)
        ));
        assert!(os.get_process(root).unwrap().pending_signals.is_empty());
    }

    #[test]
    fn mailbox_capacity_limits_ipc() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "worker".to_string(),
                "work".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        if let Some(proc) = os.get_process_mut(child) {
            proc.max_mailbox_capacity = 2;
        }

        os.send_ipc(child, "msg1".to_string()).unwrap();
        os.send_ipc(child, "msg2".to_string()).unwrap();
        let result = os.send_ipc(child, "msg3".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("mailbox is full"));
    }

    #[test]
    fn round_robin_can_be_toggled() {
        let mut os = LocalOS::new();
        assert!(os.is_round_robin());
        os.set_round_robin(false);
        assert!(!os.is_round_robin());
    }

    #[test]
    fn resource_accounting_tracks_turns_and_tool_calls() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        assert_eq!(os.get_process(root).unwrap().turns_used, 0);
        assert_eq!(os.get_process(root).unwrap().tool_calls_used, 0);
        assert_eq!(os.get_process(root).unwrap().created_at_tick, 0);

        os.increment_turns_used_for(root);
        assert_eq!(os.get_process(root).unwrap().turns_used, 1);

        os.increment_tool_calls_used_for(root);
        os.increment_tool_calls_used_for(root);
        assert_eq!(os.get_process(root).unwrap().tool_calls_used, 2);
    }

    #[test]
    fn process_group_signal_affects_all_members() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child1 = os
            .spawn(
                Some(root),
                "c1".to_string(),
                "g1".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let child2 = os
            .spawn(
                Some(root),
                "c2".to_string(),
                "g2".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_process_group(child1, 100).unwrap();
        os.set_process_group(child2, 100).unwrap();

        assert_eq!(os.get_process(child1).unwrap().process_group, Some(100));
        assert_eq!(os.get_process(child2).unwrap().process_group, Some(100));

        let count = os.signal_process_group(100, Signal::SigStop).unwrap();
        assert_eq!(count, 2);
        assert!(matches!(
            os.get_process(child1).unwrap().state,
            ProcessState::Stopped
        ));
        assert!(matches!(
            os.get_process(child2).unwrap().state,
            ProcessState::Stopped
        ));
    }

    #[test]
    fn shared_memory_crud_operations() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);

        os.shm_create("config".to_string(), "value1".to_string())
            .unwrap();
        assert_eq!(os.shm_read("config"), Ok("value1".to_string()));
        assert_eq!(
            os.shared_memory.get("config").map(|e| e.owner_pid),
            Some(root)
        );

        os.shm_write("config".to_string(), "value2".to_string())
            .unwrap();
        assert_eq!(os.shm_read("config"), Ok("value2".to_string()));

        os.shm_delete("config").unwrap();
        assert_eq!(os.shm_read("config"), Err(ShmReadError::NotFound));
        assert!(os.shared_memory.get("config").is_none());

        assert!(
            os.shm_create("config".to_string(), "value3".to_string())
                .is_ok()
        );
        assert!(
            os.shm_create("config".to_string(), "value4".to_string())
                .is_err()
        );
    }

    #[test]
    fn working_dir_inherits_to_children() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        os.set_working_dir(std::path::PathBuf::from("/tmp/work"))
            .unwrap();

        let child = os
            .spawn(
                Some(root),
                "child".to_string(),
                "goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            os.get_process(child).unwrap().working_dir,
            Some(std::path::PathBuf::from("/tmp/work"))
        );
    }

    #[test]
    fn daemon_auto_restarts_on_termination() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let daemon_pid = os
            .spawn_daemon(
                Some(root),
                "watcher".to_string(),
                "watch files".to_string(),
                20,
                4,
                2,
            )
            .unwrap();

        assert!(os.get_process(daemon_pid).unwrap().is_daemon);
        assert_eq!(os.get_process(daemon_pid).unwrap().max_restarts, 2);
        assert_eq!(os.get_process(daemon_pid).unwrap().restart_count, 0);

        os.terminate_pid(daemon_pid, "crashed".to_string());
        let restarted = os.check_daemon_restart();
        assert_eq!(restarted.len(), 1);

        let new_pid = restarted[0];
        assert_ne!(new_pid, daemon_pid);
        assert!(os.get_process(new_pid).unwrap().is_daemon);
        assert_eq!(os.get_process(new_pid).unwrap().restart_count, 1);
        assert!(
            os.get_process(new_pid)
                .unwrap()
                .goal
                .contains("daemon restart #1")
        );
    }

    #[test]
    fn daemon_respects_max_restarts() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let daemon_pid = os
            .spawn_daemon(
                Some(root),
                "watcher".to_string(),
                "watch".to_string(),
                20,
                4,
                1,
            )
            .unwrap();

        os.terminate_pid(daemon_pid, "crashed".to_string());
        let restarted1 = os.check_daemon_restart();
        assert_eq!(restarted1.len(), 1);

        os.terminate_pid(restarted1[0], "crashed again".to_string());
        let restarted2 = os.check_daemon_restart();
        assert!(restarted2.is_empty());
    }

    #[test]
    fn ipc_is_rejected_between_unrelated_processes() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os
            .spawn(
                Some(root),
                "a".to_string(),
                "goal a".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let child_b = os
            .spawn(
                Some(root),
                "b".to_string(),
                "goal b".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(child_a));
        let result = os.send_ipc(child_b, "hello".to_string());
        assert!(result.is_ok());

        let unrelated_root =
            os.begin_foreground("fg2".to_string(), "goal2".to_string(), 10, usize::MAX, None);
        let orphan = os
            .spawn(
                Some(unrelated_root),
                "orphan".to_string(),
                "goal orphan".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(orphan));
        let result = os.send_ipc(child_a, "intrusion".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Permission denied"));
    }

    #[test]
    fn ipc_allowed_within_same_process_group() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os
            .spawn(
                Some(root),
                "a".to_string(),
                "goal a".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let child_b = os
            .spawn(
                Some(root),
                "b".to_string(),
                "goal b".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_process_group(child_a, 42).unwrap();
        os.set_process_group(child_b, 42).unwrap();

        os.set_current_pid(Some(child_a));
        let result = os.send_ipc(child_b, "hello group".to_string());
        assert!(result.is_ok());

        let outsider = os
            .spawn(
                Some(root),
                "outsider".to_string(),
                "goal out".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        os.set_current_pid(Some(outsider));
        let result = os.send_ipc(child_a, "intrusion".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn kill_process_rejected_for_non_descendant() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os
            .spawn(
                Some(root),
                "a".to_string(),
                "goal a".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let child_b = os
            .spawn(
                Some(root),
                "b".to_string(),
                "goal b".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(child_a));
        let result = os.kill_process(child_b, "sibling kill".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("outside its scope"));
    }

    #[test]
    fn shm_write_rejected_for_non_owner_outside_group() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os
            .spawn(
                Some(root),
                "a".to_string(),
                "goal a".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(child_a));
        os.shm_create("secret".to_string(), "value_a".to_string())
            .unwrap();

        let child_b = os
            .spawn(
                Some(root),
                "b".to_string(),
                "goal b".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        os.set_current_pid(Some(child_b));
        let result = os.shm_write("secret".to_string(), "tampered".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Permission denied"));
    }

    #[test]
    fn shm_write_allowed_for_same_process_group() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os
            .spawn(
                Some(root),
                "a".to_string(),
                "goal a".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let child_b = os
            .spawn(
                Some(root),
                "b".to_string(),
                "goal b".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_process_group(child_a, 100).unwrap();
        os.set_process_group(child_b, 100).unwrap();

        os.set_current_pid(Some(child_a));
        os.shm_create("shared_config".to_string(), "v1".to_string())
            .unwrap();

        os.set_current_pid(Some(child_b));
        let result = os.shm_write("shared_config".to_string(), "v2".to_string());
        assert!(result.is_ok());
        assert_eq!(os.shm_read("shared_config"), Ok("v2".to_string()));
    }

    #[test]
    fn shm_write_allowed_for_ancestor_of_owner() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "child".to_string(),
                "goal child".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(child));
        os.shm_create("child_data".to_string(), "original".to_string())
            .unwrap();

        os.set_current_pid(Some(root));
        let result = os.shm_write("child_data".to_string(), "parent_override".to_string());
        assert!(result.is_ok());
        assert_eq!(os.shm_read("child_data"), Ok("parent_override".to_string()));
    }

    #[test]
    fn shm_delete_rejected_for_non_owner_outside_group() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os
            .spawn(
                Some(root),
                "a".to_string(),
                "goal a".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let child_b = os
            .spawn(
                Some(root),
                "b".to_string(),
                "goal b".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(child_a));
        os.shm_create("owned_by_a".to_string(), "data".to_string())
            .unwrap();

        os.set_current_pid(Some(child_b));
        let result = os.shm_delete("owned_by_a");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Permission denied"));

        assert_eq!(os.shm_read("owned_by_a"), Ok("data".to_string()));
    }

    #[test]
    fn shm_owner_pgid_tracked_on_create() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "child".to_string(),
                "goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_process_group(child, 77).unwrap();

        os.set_current_pid(Some(child));
        os.shm_create("group_key".to_string(), "val".to_string())
            .unwrap();

        let entry = os.shared_memory.get("group_key").unwrap();
        assert_eq!(entry.owner_pid, child);
    }

    #[test]
    fn shm_read_detects_checksum_corruption() {
        let mut os = LocalOS::new();
        let _root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        os.shm_create("data".to_string(), "original".to_string())
            .unwrap();

        if let Some(entry) = os.shared_memory.get_mut("data") {
            entry.value = "tampered".to_string();
        }

        let result = os.shm_read("data");
        assert!(matches!(result, Err(ShmReadError::Corrupted { .. })));
    }

    #[test]
    fn shm_read_degraded_returns_data_on_corruption() {
        let mut os = LocalOS::new();
        let _root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        os.shm_create("data".to_string(), "original".to_string())
            .unwrap();

        if let Some(entry) = os.shared_memory.get_mut("data") {
            entry.value = "tampered".to_string();
        }

        let degraded = os.shm_read_degraded("data");
        assert!(degraded.is_some());
        let val = degraded.unwrap();
        assert!(val.contains("DEGRADED"));
        assert!(val.contains("tampered"));
    }

    #[test]
    fn shm_read_detects_owner_terminated() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "worker".to_string(),
                "work".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        os.set_current_pid(Some(child));
        os.shm_create("child_data".to_string(), "value".to_string())
            .unwrap();
        os.set_current_pid(Some(root));

        os.terminate_pid(child, "done".to_string());

        let result = os.shm_read("child_data");
        assert!(matches!(result, Err(ShmReadError::OwnerTerminated { .. })));
    }

    #[test]
    fn shm_read_degraded_returns_data_on_owner_terminated() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "worker".to_string(),
                "work".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        os.set_current_pid(Some(child));
        os.shm_create("child_data".to_string(), "important_value".to_string())
            .unwrap();
        os.set_current_pid(Some(root));

        os.terminate_pid(child, "done".to_string());

        let degraded = os.shm_read_degraded("child_data");
        assert!(degraded.is_some());
        let val = degraded.unwrap();
        assert!(val.contains("DEGRADED"));
        assert!(val.contains("important_value"));
    }

    #[test]
    fn shm_read_permission_denied_for_unrelated_process() {
        let mut os = LocalOS::new();
        let root1 =
            os.begin_foreground("fg1".to_string(), "goal1".to_string(), 10, usize::MAX, None);
        let child_a = os
            .spawn(
                Some(root1),
                "a".to_string(),
                "ga".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        let root2 =
            os.begin_foreground("fg2".to_string(), "goal2".to_string(), 10, usize::MAX, None);
        let child_b = os
            .spawn(
                Some(root2),
                "b".to_string(),
                "gb".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(child_a));
        os.shm_create("secret".to_string(), "private_data".to_string())
            .unwrap();

        os.set_current_pid(Some(child_b));
        let result = os.shm_read("secret");
        assert!(matches!(result, Err(ShmReadError::PermissionDenied { .. })));
    }

    #[test]
    fn shm_health_check_detects_corrupted_and_orphaned() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "worker".to_string(),
                "work".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(child));
        os.shm_create("orphan_data".to_string(), "val1".to_string())
            .unwrap();
        os.shm_create("good_data".to_string(), "val2".to_string())
            .unwrap();
        os.set_current_pid(Some(root));

        os.terminate_pid(child, "done".to_string());

        if let Some(entry) = os.shared_memory.get_mut("good_data") {
            entry.value = "corrupted".to_string();
        }

        let issues = os.shm_health_check();
        assert_eq!(issues.len(), 2);

        let has_orphan = issues
            .iter()
            .any(|(k, e)| k == "orphan_data" && matches!(e, ShmReadError::OwnerTerminated { .. }));
        let has_corrupt = issues
            .iter()
            .any(|(k, e)| k == "good_data" && matches!(e, ShmReadError::Corrupted { .. }));
        assert!(has_orphan);
        assert!(has_corrupt);
    }

    #[test]
    fn shm_cleanup_orphans_removes_dead_owner_entries() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(
                Some(root),
                "worker".to_string(),
                "work".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        os.set_current_pid(Some(child));
        os.shm_create("orphan".to_string(), "will_be_removed".to_string())
            .unwrap();
        os.set_current_pid(Some(root));
        os.shm_create("root_data".to_string(), "stays".to_string())
            .unwrap();

        os.terminate_pid(child, "done".to_string());

        let removed = os.shm_cleanup_orphans();
        assert_eq!(removed, 1);
        assert!(os.shared_memory.get("orphan").is_none());
        assert!(os.shared_memory.get("root_data").is_some());
    }

    #[test]
    fn shm_write_updates_checksum_and_version() {
        let mut os = LocalOS::new();
        let _root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        os.shm_create("data".to_string(), "v1".to_string()).unwrap();

        let v1_checksum = os.shared_memory.get("data").unwrap().checksum;
        let v1_version = os.shared_memory.get("data").unwrap().version;
        assert_eq!(v1_version, 1);

        os.shm_write("data".to_string(), "v2".to_string()).unwrap();

        let v2_checksum = os.shared_memory.get("data").unwrap().checksum;
        let v2_version = os.shared_memory.get("data").unwrap().version;
        assert_eq!(v2_version, 2);
        assert_ne!(v1_checksum, v2_checksum);

        assert_eq!(os.shm_read("data"), Ok("v2".to_string()));
    }

    // ------------------------------------------------------------------
    // Phase 0 primitives: futex + trace
    // ------------------------------------------------------------------

    #[test]
    fn futex_basic_create_load_store_cas() {
        use crate::primitives::FutexOps;
        let mut os = LocalOS::new();
        let addr = os.futex_create(0, "stream_cancel".to_string());
        assert_eq!(os.futex_load(addr), Some(0));
        assert_eq!(os.futex_store(addr, 1), Some(0));
        assert_eq!(os.futex_load(addr), Some(1));
        assert!(os.futex_cas(addr, 1, 2).is_ok());
        assert!(os.futex_cas(addr, 1, 3).is_err());
        assert_eq!(os.futex_load(addr), Some(2));
    }

    #[test]
    fn futex_wake_moves_waiter_to_ready() {
        use crate::primitives::FutexOps;
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let worker = os
            .spawn(
                Some(root),
                "w".to_string(),
                "do".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        // Simulate worker going to sleep on a futex
        if let Some(p) = os.processes.get_mut(&worker) {
            p.state = ProcessState::Waiting {
                reason: WaitReason::ProcessExit { on_pid: root },
            };
        }
        os.ready_set.remove(&worker);

        let addr = os.futex_create(0, "ready_bell".to_string());
        let seq_before = os.futex_register_waiter(addr, worker).unwrap();
        let woken = os.futex_wake(addr, 1);
        assert_eq!(woken, 1);
        let seq_after = os.futex_seq(addr).unwrap();
        assert!(seq_after > seq_before);
        assert_eq!(
            os.processes.get(&worker).unwrap().state,
            ProcessState::Ready
        );
        assert!(os.ready_set.contains(&worker));
    }

    #[test]
    fn futex_try_wait_reports_value_changed() {
        use crate::primitives::{FutexOps, FutexWakeReason};
        let mut os = LocalOS::new();
        let addr = os.futex_create(0, "t".to_string());
        assert!(
            os.futex_try_wait(addr, 0).is_none(),
            "should block when equal"
        );
        os.futex_store(addr, 7);
        assert_eq!(
            os.futex_try_wait(addr, 0),
            Some(FutexWakeReason::ValueChanged)
        );
    }

    #[test]
    fn trace_records_spans_and_events_in_order() {
        use crate::primitives::{TraceKind, TraceLevel, TraceOps};
        use crate::types::FastMap;
        let mut os = LocalOS::new();
        let _fg = os.begin_foreground("fg".to_string(), "g".to_string(), 10, usize::MAX, None);

        let span = os.trace_span_enter("turn.run".to_string(), None, FastMap::default());
        let mut fields: FastMap<String, String> = FastMap::default();
        fields.insert("model".to_string(), "gpt".to_string());
        os.trace_event(
            "llm.submit".to_string(),
            TraceLevel::Info,
            Some(span),
            fields,
            Some("sent".to_string()),
        );
        os.trace_span_exit(span, FastMap::default());

        let recs = os.trace_drain_since(0);
        assert_eq!(recs.len(), 3);
        assert!(matches!(recs[0].kind, TraceKind::SpanEnter));
        assert!(matches!(recs[1].kind, TraceKind::Event));
        assert!(matches!(recs[2].kind, TraceKind::SpanExit));
        assert_eq!(recs[1].name, "llm.submit");
        assert_eq!(
            recs[1].fields().and_then(|f| f.get("model")).map(String::as_str),
            Some("gpt")
        );
    }

    #[test]
    fn trace_ring_respects_capacity() {
        use crate::primitives::{TraceLevel, TraceOps};
        use crate::types::FastMap;
        let mut os = LocalOS::new();
        os.trace_set_capacity(4);
        for i in 0..10 {
            os.trace_event(
                format!("evt.{}", i),
                TraceLevel::Debug,
                None,
                FastMap::default(),
                None,
            );
        }
        let recs = os.trace_drain_since(0);
        assert_eq!(recs.len(), 4);
        // oldest kept should be evt.6 (after dropping 0..=5)
        assert_eq!(recs[0].name, "evt.6");
        assert_eq!(recs[3].name, "evt.9");
    }

    #[test]
    fn rlimit_set_and_get_roundtrips() {
        use crate::primitives::{ResourceLimit, RlimitOps};
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        let mut lim = ResourceLimit::unlimited();
        lim.max_turns = 7;
        lim.max_tool_calls = 3;
        lim.max_tokens_in = 1000;
        os.rlimit_set(pid, lim.clone()).unwrap();
        let got = os.rlimit_get(pid).unwrap();
        assert_eq!(got, lim);
        // quota_turns mirror must be synced too
        assert_eq!(os.get_process(pid).unwrap().quota_turns, 7);
    }

    #[test]
    fn rusage_charge_enforces_turns_limit() {
        use crate::primitives::{
            ResourceLimit, ResourceUsageDelta, RlimitDim, RlimitOps, RlimitVerdict,
        };
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        let mut lim = ResourceLimit::unlimited();
        lim.max_turns = 2;
        os.rlimit_set(pid, lim).unwrap();

        assert_eq!(
            os.rusage_charge(
                pid,
                ResourceUsageDelta {
                    turns: 1,
                    ..Default::default()
                }
            ),
            RlimitVerdict::Ok
        );
        assert_eq!(
            os.rusage_charge(
                pid,
                ResourceUsageDelta {
                    turns: 1,
                    ..Default::default()
                }
            ),
            RlimitVerdict::Ok
        );
        match os.rusage_charge(
            pid,
            ResourceUsageDelta {
                turns: 1,
                ..Default::default()
            },
        ) {
            RlimitVerdict::Exceeded {
                dimension,
                used,
                limit,
            } => {
                assert_eq!(dimension, RlimitDim::Turns);
                assert_eq!(used, 3);
                assert_eq!(limit, 2);
            }
            v => panic!("expected Exceeded Turns, got {:?}", v),
        }
        // legacy mirror stays in sync
        assert_eq!(os.get_process(pid).unwrap().turns_used, 3);
        assert_eq!(os.rusage_get(pid).unwrap().turns, 3);
    }

    #[test]
    fn rlimit_check_is_pure() {
        use crate::primitives::{
            ResourceLimit, ResourceUsageDelta, RlimitDim, RlimitOps, RlimitVerdict,
        };
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        let mut lim = ResourceLimit::unlimited();
        lim.max_tokens_in = 100;
        os.rlimit_set(pid, lim).unwrap();
        // pre-check a big prompt
        let probe = ResourceUsageDelta {
            tokens_in: 200,
            ..Default::default()
        };
        match os.rlimit_check(pid, &probe) {
            RlimitVerdict::Exceeded { dimension, .. } => {
                assert_eq!(dimension, RlimitDim::TokensIn);
            }
            v => panic!("expected Exceeded TokensIn, got {:?}", v),
        }
        // usage must NOT have moved
        assert_eq!(os.rusage_get(pid).unwrap().tokens_in, 0);
    }

    #[test]
    fn increment_helpers_route_through_rusage_charge() {
        use crate::primitives::RlimitOps;
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        os.increment_turns_used_for(pid);
        os.increment_turns_used_for(pid);
        os.increment_tool_calls_used_for(pid);
        let u = os.rusage_get(pid).unwrap();
        assert_eq!(u.turns, 2);
        assert_eq!(u.tool_calls, 1);
        // legacy mirrors stay in sync
        let p = os.get_process(pid).unwrap();
        assert_eq!(p.turns_used, 2);
        assert_eq!(p.tool_calls_used, 1);
    }

    #[test]
    fn llm_account_charges_cost_and_updates_rusage() {
        use crate::primitives::{LlmModelPrice, LlmOps, LlmUsageReport, RlimitOps, RlimitVerdict};
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        // 1000 prompt tok => 2500 micros; 500 completion tok => 3000 micros.
        os.llm_set_price(
            "gpt-test".into(),
            LlmModelPrice {
                prompt_per_1k_micros: 2_500,
                completion_per_1k_micros: 6_000,
            },
        );
        let out = os.llm_account(
            pid,
            LlmUsageReport {
                model: "gpt-test".into(),
                prompt_tokens: 1_000,
                completion_tokens: 500,
                cached_prompt_tokens: 100,
                latency_ms: 42,
            },
        );
        assert_eq!(out.charged_cost_micros, 2_500 + 3_000);
        assert_eq!(out.verdict, RlimitVerdict::Ok);
        let u = os.rusage_get(pid).unwrap();
        assert_eq!(u.tokens_in, 1_000);
        assert_eq!(u.tokens_out, 500);
        assert_eq!(u.cost_micros, 5_500);
    }

    #[test]
    fn llm_account_with_unknown_model_is_free_but_still_charges_tokens() {
        use crate::primitives::RlimitOps;
        use crate::primitives::{LlmOps, LlmUsageReport};
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        // No price registered for "mystery-model"
        let out = os.llm_account(
            pid,
            LlmUsageReport {
                model: "mystery-model".into(),
                prompt_tokens: 123,
                completion_tokens: 45,
                cached_prompt_tokens: 0,
                latency_ms: 0,
            },
        );
        assert_eq!(out.charged_cost_micros, 0);
        let u = os.rusage_get(pid).unwrap();
        assert_eq!(u.tokens_in, 123);
        assert_eq!(u.tokens_out, 45);
        assert_eq!(u.cost_micros, 0);
    }

    #[test]
    fn llm_account_respects_cost_rlimit() {
        use crate::primitives::{
            LlmModelPrice, LlmOps, LlmUsageReport, ResourceLimit, RlimitDim, RlimitOps,
            RlimitVerdict,
        };
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        os.llm_set_price(
            "g".into(),
            LlmModelPrice {
                prompt_per_1k_micros: 1_000,
                completion_per_1k_micros: 0,
            },
        );
        // cost budget = 500 micros. 1000 prompt tokens -> 1000 micros -> Exceeded.
        let mut lim = ResourceLimit::unlimited();
        lim.max_cost_micros = 500;
        os.rlimit_set(pid, lim).unwrap();
        let out = os.llm_account(
            pid,
            LlmUsageReport {
                model: "g".into(),
                prompt_tokens: 1_000,
                completion_tokens: 0,
                cached_prompt_tokens: 0,
                latency_ms: 0,
            },
        );
        match out.verdict {
            RlimitVerdict::Exceeded {
                dimension,
                used,
                limit,
            } => {
                assert_eq!(dimension, RlimitDim::CostMicros);
                assert_eq!(used, 1_000);
                assert_eq!(limit, 500);
            }
            v => panic!("expected Exceeded CostMicros, got {:?}", v),
        }
    }

    #[test]
    fn llm_account_emits_trace_event() {
        use crate::primitives::{LlmOps, LlmUsageReport, TraceOps};
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        os.llm_account(
            pid,
            LlmUsageReport {
                model: "m".into(),
                prompt_tokens: 10,
                completion_tokens: 5,
                cached_prompt_tokens: 0,
                latency_ms: 77,
            },
        );
        let recs = os.trace_drain_since(0);
        let found = recs.iter().any(|r| r.name == "llm.account");
        assert!(found, "expected a trace event named llm.account");
    }

    // ---- VfsOps (Phase 3) ----

    fn tmp_path(name: &str) -> std::path::PathBuf {
        static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let seq = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "aios_vfs_{}_{}_{}_{}",
            name,
            std::process::id(),
            nanos,
            seq
        ));
        p
    }

    #[test]
    fn vfs_read_write_roundtrip_and_charges_fs_bytes() {
        use crate::primitives::{RlimitOps, VfsOps};
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        let p = tmp_path("rw");
        os.vfs_write_all(Some(pid), &p, "hello world").unwrap();
        let got = os.vfs_read_to_string(Some(pid), &p).unwrap();
        assert_eq!(got, "hello world");

        let usage = os.rusage_get(pid).unwrap();
        // 写入 11 字节 + 读出 11 字节 = 22
        assert_eq!(usage.fs_bytes, 22);

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn vfs_sensitive_path_is_denied() {
        use crate::primitives::{VfsError, VfsOps};
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        let bad = std::path::PathBuf::from("/tmp/.ssh/id_rsa");
        match os.vfs_read_to_string(Some(pid), &bad).unwrap_err() {
            VfsError::PermissionDenied(_) => {}
            other => panic!("expected PermissionDenied, got {:?}", other),
        }
    }

    #[test]
    fn vfs_read_missing_returns_not_found() {
        use crate::primitives::{VfsError, VfsOps};
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        let p = tmp_path("missing");
        match os.vfs_read_to_string(Some(pid), &p).unwrap_err() {
            VfsError::NotFound(_) => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn vfs_respects_fs_bytes_rlimit() {
        use crate::primitives::{ResourceLimit, RlimitDim, RlimitOps, VfsError, VfsOps};
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        let mut limits = ResourceLimit::unlimited();
        limits.max_fs_bytes = 5;
        os.rlimit_set(pid, limits).unwrap();

        let p = tmp_path("quota");
        // 写 10 字节——超过 5 字节上限
        match os.vfs_write_all(Some(pid), &p, "0123456789").unwrap_err() {
            VfsError::QuotaExceeded {
                dimension: RlimitDim::FsBytes,
                ..
            } => {}
            other => panic!("expected QuotaExceeded(FsBytes), got {:?}", other),
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn vfs_emits_trace_event() {
        use crate::primitives::{TraceOps, VfsOps};
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        let p = tmp_path("trace");
        os.vfs_write_all(Some(pid), &p, "x").unwrap();
        let _ = os.vfs_read_to_string(Some(pid), &p).unwrap();
        let recs = os.trace_drain_since(0);
        assert!(recs.iter().any(|r| r.name == "vfs.write"));
        assert!(recs.iter().any(|r| r.name == "vfs.read"));
        let _ = std::fs::remove_file(&p);
    }

    // ---- IpcOps (Phase 5) ----

    #[test]
    fn channel_send_recv_roundtrip() {
        use crate::primitives::{IpcOps, IpcRecvResult};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let child = os
            .spawn(Some(root), "child".into(), "goal".into(), 20, 4, None, None)
            .unwrap();
        let ch = os.channel_create(Some(root), 2, "task-result".into());

        os.channel_send(Some(child), ch, "hello".into()).unwrap();
        match os.channel_try_recv(Some(root), ch).unwrap() {
            IpcRecvResult::Message(msg) => assert_eq!(msg, "hello"),
            other => panic!("expected message, got {:?}", other),
        }
    }

    #[test]
    fn channel_peek_is_non_destructive() {
        use crate::primitives::{IpcOps, IpcRecvResult};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 1, "peek".into());
        os.channel_send(Some(root), ch, "payload".into()).unwrap();

        assert_eq!(
            os.channel_peek(Some(root), ch).unwrap(),
            IpcRecvResult::Message("payload".into())
        );
        assert_eq!(
            os.channel_try_recv(Some(root), ch).unwrap(),
            IpcRecvResult::Message("payload".into())
        );
    }

    #[test]
    fn channel_respects_capacity_backpressure() {
        use crate::primitives::IpcOps;
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 1, "cap".into());
        os.channel_send(Some(root), ch, "one".into()).unwrap();
        let err = os.channel_send(Some(root), ch, "two".into()).unwrap_err();
        assert!(err.contains("full"));
    }

    #[test]
    fn channel_permissions_follow_parent_child_rules() {
        use crate::primitives::IpcOps;
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let child = os
            .spawn(Some(root), "child".into(), "goal".into(), 20, 4, None, None)
            .unwrap();
        let outsider_root = os.begin_foreground("other".into(), "goal".into(), 10, 0, None);
        let outsider = os
            .spawn(
                Some(outsider_root),
                "outsider".into(),
                "goal".into(),
                20,
                4,
                None,
                None,
            )
            .unwrap();

        let ch = os.channel_create(Some(root), 1, "perm".into());
        assert!(os.channel_send(Some(child), ch, "ok".into()).is_ok());
        assert!(os.channel_send(Some(outsider), ch, "bad".into()).is_err());
        assert!(os.channel_peek(Some(root), ch).is_ok());
        assert!(os.channel_peek(Some(child), ch).is_err());
    }

    #[test]
    fn channel_close_yields_closed_after_drain() {
        use crate::primitives::{IpcOps, IpcRecvResult};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 1, "close".into());
        os.channel_send(Some(root), ch, "done".into()).unwrap();
        os.channel_close(Some(root), ch).unwrap();
        assert_eq!(
            os.channel_try_recv(Some(root), ch).unwrap(),
            IpcRecvResult::Message("done".into())
        );
        assert_eq!(
            os.channel_try_recv(Some(root), ch).unwrap(),
            IpcRecvResult::Closed
        );
    }

    #[test]
    fn channel_emits_trace_events() {
        use crate::primitives::{IpcOps, TraceOps};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 1, "trace".into());
        os.channel_send(Some(root), ch, "x".into()).unwrap();
        let _ = os.channel_try_recv(Some(root), ch).unwrap();
        os.channel_close(Some(root), ch).unwrap();
        let recs = os.trace_drain_since(0);
        assert!(recs.iter().any(|r| r.name == "ipc.channel_create"));
        assert!(recs.iter().any(|r| r.name == "ipc.send"));
        assert!(recs.iter().any(|r| r.name == "ipc.recv"));
        assert!(recs.iter().any(|r| r.name == "ipc.close"));
    }

    #[test]
    fn channel_send_completes_event_and_wakes_waiter() {
        use crate::primitives::IpcOps;
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let child = os
            .spawn(Some(root), "child".into(), "goal".into(), 20, 4, None, None)
            .unwrap();
        let ch = os.channel_create(Some(root), 1, "wake".into());
        let evt = os.channel_event_id(ch).unwrap();

        os.wait_on_events(vec![evt], WaitPolicy::All, None).unwrap();
        assert!(os.consume_yield_requested());
        os.set_current_pid(Some(child));
        os.channel_send(Some(child), ch, "done".into()).unwrap();

        let root_proc = os.get_process(root).unwrap();
        assert_eq!(root_proc.state, ProcessState::Ready);
        assert!(
            root_proc
                .mailbox
                .back()
                .map(|s| s.contains("[EVENT_WAKE]"))
                .unwrap_or(false)
        );
    }

    #[test]
    fn channel_close_without_message_still_completes_event() {
        use crate::primitives::IpcOps;
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 1, "close-event".into());
        let evt = os.channel_event_id(ch).unwrap();

        os.wait_on_events(vec![evt], WaitPolicy::All, None).unwrap();
        assert!(os.consume_yield_requested());
        os.set_current_pid(Some(root));
        os.channel_close(Some(root), ch).unwrap();

        let root_proc = os.get_process(root).unwrap();
        assert_eq!(root_proc.state, ProcessState::Ready);
    }

    #[test]
    fn channel_peek_all_and_recv_all_preserve_pipe_order() {
        use crate::primitives::IpcOps;
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 4, "pipe".into());
        os.channel_send(Some(root), ch, "a".into()).unwrap();
        os.channel_send(Some(root), ch, "b".into()).unwrap();
        os.channel_send(Some(root), ch, "c".into()).unwrap();

        assert_eq!(
            os.channel_peek_all(Some(root), ch).unwrap(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(
            os.channel_try_recv_all(Some(root), ch).unwrap(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert!(os.channel_peek_all(Some(root), ch).unwrap().is_empty());
    }

    #[test]
    fn channel_event_id_rotates_after_each_ready_edge() {
        use crate::primitives::{IpcOps, IpcRecvResult};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 2, "edge".into());
        let evt1 = os.channel_event_id(ch).unwrap();

        os.channel_send(Some(root), ch, "first".into()).unwrap();
        let evt2 = os.channel_event_id(ch).unwrap();
        assert_ne!(evt1, evt2);
        assert_eq!(
            os.channel_try_recv(Some(root), ch).unwrap(),
            IpcRecvResult::Message("first".into())
        );

        os.wait_on_events(vec![evt2], WaitPolicy::All, None)
            .unwrap();
        assert!(os.consume_yield_requested());
        os.channel_send(Some(root), ch, "second".into()).unwrap();
        let root_proc = os.get_process(root).unwrap();
        assert_eq!(root_proc.state, ProcessState::Ready);
    }

    #[test]
    fn channel_destroy_requires_closed_and_empty() {
        use crate::primitives::IpcOps;
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 1, "destroy".into());

        let err = os.channel_destroy(Some(root), ch).unwrap_err();
        assert!(err.contains("ref_count=0"));

        os.channel_send(Some(root), ch, "payload".into()).unwrap();
        os.channel_close(Some(root), ch).unwrap();
        let err = os.channel_destroy(Some(root), ch).unwrap_err();
        assert!(err.contains("ref_count=0"));

        let _ = os.channel_try_recv_all(Some(root), ch).unwrap();
        os.channel_destroy(Some(root), ch).unwrap();
        assert!(os.channel_event_id(ch).is_none());
    }

    #[test]
    fn tagged_result_pipe_exposes_owner_tag_and_refcount() {
        use crate::primitives::{ChannelOwnerTag, IpcOps};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create_tagged(
            Some(root),
            1,
            "task-result".into(),
            ChannelOwnerTag::TaskResult,
            2,
        );
        let meta = os.channel_meta(ch).unwrap();
        assert_eq!(meta.owner_tag, ChannelOwnerTag::TaskResult);
        assert_eq!(meta.ref_count, 2);
        assert!(!meta.closed);
    }

    #[test]
    fn result_pipe_requires_ref_release_before_destroy() {
        use crate::primitives::{ChannelOwnerTag, IpcOps};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create_tagged(
            Some(root),
            1,
            "async-result".into(),
            ChannelOwnerTag::AsyncToolResult,
            2,
        );
        os.channel_close(Some(root), ch).unwrap();
        let _ = os.channel_release(ch).unwrap();
        let err = os.channel_destroy(Some(root), ch).unwrap_err();
        assert!(err.contains("ref_count=0"));
        let _ = os.channel_release(ch).unwrap();
        os.channel_destroy(Some(root), ch).unwrap();
    }

    #[test]
    fn channel_gc_collects_closed_empty_channels_only() {
        use crate::primitives::IpcOps;
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let keep_open = os.channel_create(Some(root), 1, "open".into());
        let keep_buffered = os.channel_create(Some(root), 1, "buffered".into());
        let gc_me = os.channel_create(Some(root), 1, "gc".into());

        os.channel_send(Some(root), keep_buffered, "x".into())
            .unwrap();
        os.channel_close(Some(root), keep_buffered).unwrap();
        os.channel_close(Some(root), gc_me).unwrap();

        assert_eq!(os.channel_gc_closed_empty(), 1);
        assert!(os.channel_event_id(gc_me).is_none());
        assert!(os.channel_event_id(keep_open).is_some());
        assert!(os.channel_event_id(keep_buffered).is_some());
    }

    #[test]
    fn channel_gc_skips_tagged_result_pipe_with_live_refs() {
        use crate::primitives::{ChannelOwnerTag, IpcOps};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create_tagged(
            Some(root),
            1,
            "gc-result".into(),
            ChannelOwnerTag::TaskResult,
            1,
        );
        os.channel_close(Some(root), ch).unwrap();
        assert_eq!(os.channel_gc_closed_empty(), 0);
        assert!(os.channel_event_id(ch).is_some());
        let _ = os.channel_release(ch).unwrap();
        assert_eq!(os.channel_gc_closed_empty(), 1);
        assert!(os.channel_event_id(ch).is_none());
    }

    #[test]
    fn channel_destroy_and_gc_emit_trace_events() {
        use crate::primitives::{IpcOps, TraceOps};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let destroy_ch = os.channel_create(Some(root), 1, "destroy-trace".into());
        os.channel_close(Some(root), destroy_ch).unwrap();
        os.channel_destroy(Some(root), destroy_ch).unwrap();

        let gc_ch = os.channel_create(Some(root), 1, "gc-trace".into());
        os.channel_close(Some(root), gc_ch).unwrap();
        assert_eq!(os.channel_gc_closed_empty(), 1);

        let recs = os.trace_drain_since(0);
        assert!(recs.iter().any(|r| r.name == "ipc.destroy"));
        assert!(recs.iter().any(|r| r.name == "ipc.gc"));
    }

    // ---- EpollOps (Phase 6) ----

    #[test]
    fn epoll_wait_returns_ready_channel_without_suspending() {
        use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult, IpcOps};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 2, "epoll-ready".into());
        let ep = os.epoll_create("main".into());
        os.epoll_ctl_add(ep, EpollSource::Channel(ch), EpollEventMask::IN, 7)
            .unwrap();

        os.channel_send(Some(root), ch, "payload".into()).unwrap();
        match os.epoll_wait(ep, 8, None).unwrap() {
            EpollWaitResult::Ready(events) => {
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].source, EpollSource::Channel(ch));
                assert_eq!(events[0].events, EpollEventMask::IN);
                assert_eq!(events[0].user_data, 7);
            }
            other => panic!("expected ready, got {:?}", other),
        }
    }

    #[test]
    fn epoll_wait_suspends_and_then_observes_event_source() {
        use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ep = os.epoll_create("main".into());
        let watched = EventId::new(42);
        os.epoll_ctl_add(ep, EpollSource::Event(watched), EpollEventMask::IN, 99)
            .unwrap();

        match os.epoll_wait(ep, 8, Some(5)).unwrap() {
            EpollWaitResult::Suspended { timeout_tick } => assert_eq!(timeout_tick, Some(5)),
            other => panic!("expected suspended, got {:?}", other),
        }
        assert!(os.current_process_id().is_none());

        let woke = os.notify_events_completed(&[watched]);
        assert_eq!(woke, vec![root]);
        let resumed = os.pop_ready().unwrap();
        assert_eq!(resumed.pid, root);

        match os.epoll_wait(ep, 8, None).unwrap() {
            EpollWaitResult::Ready(events) => {
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].source, EpollSource::Event(watched));
                assert_eq!(events[0].events, EpollEventMask::IN);
                assert_eq!(events[0].user_data, 99);
            }
            other => panic!("expected ready, got {:?}", other),
        }
    }

    #[test]
    fn epoll_ctl_mod_del_and_snapshot_work() {
        use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult, IpcOps};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let ch = os.channel_create(Some(root), 1, "epoll-ctl".into());
        let ep = os.epoll_create("ctl".into());
        os.epoll_ctl_add(ep, EpollSource::Channel(ch), EpollEventMask::HUP, 1)
            .unwrap();
        os.epoll_ctl_mod(
            ep,
            EpollSource::Channel(ch),
            EpollEventMask::IN | EpollEventMask::HUP,
            2,
        )
        .unwrap();

        let snapshot = os.epoll_snapshot(ep).unwrap();
        assert_eq!(snapshot.label, "ctl");
        assert_eq!(snapshot.registrations.len(), 1);
        assert_eq!(snapshot.registrations[0].source, EpollSource::Channel(ch));
        assert_eq!(
            snapshot.registrations[0].events,
            EpollEventMask::IN | EpollEventMask::HUP
        );
        assert_eq!(snapshot.registrations[0].user_data, 2);

        os.channel_close(Some(root), ch).unwrap();
        match os.epoll_wait(ep, 8, None).unwrap() {
            EpollWaitResult::Ready(events) => {
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].events, EpollEventMask::HUP);
                assert_eq!(events[0].user_data, 2);
            }
            other => panic!("expected ready, got {:?}", other),
        }

        os.epoll_ctl_del(ep, EpollSource::Channel(ch)).unwrap();
        match os.epoll_wait(ep, 8, None).unwrap() {
            EpollWaitResult::Ready(events) => assert!(events.is_empty()),
            other => panic!("expected empty ready set, got {:?}", other),
        }
        assert!(os.epoll_destroy(ep));
        assert!(os.epoll_snapshot(ep).is_none());
    }

    #[test]
    fn epoll_wait_returns_ready_for_futex_value_change() {
        use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult, FutexOps};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let addr = os.futex_create(0, "epoll-futex-value".into());
        let ep = os.epoll_create("futex".into());
        os.epoll_ctl_add(
            ep,
            EpollSource::Futex { addr, expected: 0 },
            EpollEventMask::IN,
            11,
        )
        .unwrap();

        assert!(matches!(
            os.epoll_wait(ep, 8, None).unwrap(),
            EpollWaitResult::Suspended { timeout_tick: None }
        ));
        os.set_current_pid(Some(root));
        let _ = os.futex_store(addr, 9);

        match os.epoll_wait(ep, 8, None).unwrap() {
            EpollWaitResult::Ready(events) => {
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].source, EpollSource::Futex { addr, expected: 0 });
                assert_eq!(events[0].events, EpollEventMask::IN);
                assert_eq!(events[0].user_data, 11);
            }
            other => panic!("expected ready, got {:?}", other),
        }
    }

    #[test]
    fn epoll_wait_observes_futex_wake_even_when_value_is_unchanged() {
        use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult, FutexOps};
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
        let addr = os.futex_create(0, "epoll-futex-seq".into());
        let ep = os.epoll_create("futex-seq".into());
        os.epoll_ctl_add(
            ep,
            EpollSource::Futex { addr, expected: 0 },
            EpollEventMask::IN,
            22,
        )
        .unwrap();

        match os.epoll_wait(ep, 8, Some(4)).unwrap() {
            EpollWaitResult::Suspended { timeout_tick } => assert_eq!(timeout_tick, Some(4)),
            other => panic!("expected suspended, got {:?}", other),
        }
        assert!(os.current_process_id().is_none());

        os.futex_wake(addr, 1);
        let resumed = os.pop_ready().unwrap();
        assert_eq!(resumed.pid, root);

        match os.epoll_wait(ep, 8, None).unwrap() {
            EpollWaitResult::Ready(events) => {
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].source, EpollSource::Futex { addr, expected: 0 });
                assert_eq!(events[0].events, EpollEventMask::IN);
                assert_eq!(events[0].user_data, 22);
            }
            other => panic!("expected ready, got {:?}", other),
        }

        match os.epoll_wait(ep, 8, None).unwrap() {
            EpollWaitResult::Suspended { timeout_tick } => assert_eq!(timeout_tick, None),
            other => panic!("expected suspended after cursor refresh, got {:?}", other),
        }
    }

    // ---- DaemonOps (Phase 4) ----

    #[test]
    fn daemon_register_and_exit_marks_state() {
        use crate::primitives::{DaemonKind, DaemonOps, DaemonState};
        let mut os = LocalOS::new();
        let (h, _tok) = os.daemon_register("r1".into(), DaemonKind::Reflection, None);
        let snap = os.daemon_status(h).unwrap();
        assert_eq!(snap.state, DaemonState::Running);
        assert_eq!(snap.label, "r1");

        os.daemon_exit(h, None);
        let snap = os.daemon_status(h).unwrap();
        assert_eq!(snap.state, DaemonState::Exited);
        assert!(snap.exit_tick.is_some());
    }

    #[test]
    fn daemon_exit_with_error_becomes_failed_and_preserves_message() {
        use crate::primitives::{DaemonKind, DaemonOps, DaemonState};
        let mut os = LocalOS::new();
        let (h, _) = os.daemon_register("r2".into(), DaemonKind::KnowledgeBuild, None);
        os.daemon_exit(h, Some("boom".to_string()));
        let snap = os.daemon_status(h).unwrap();
        assert_eq!(snap.state, DaemonState::Failed);
        assert_eq!(snap.last_error.as_deref(), Some("boom"));
    }

    #[test]
    fn cancel_daemon_sets_token_and_state_and_wins_over_exit() {
        use crate::primitives::{DaemonKind, DaemonOps, DaemonState};
        let mut os = LocalOS::new();
        let (h, tok) = os.daemon_register("r3".into(), DaemonKind::Other, None);
        assert!(!tok.is_cancelled());
        assert!(os.cancel_daemon(h));
        assert!(tok.is_cancelled(), "cancel token should flip to true");
        assert_eq!(os.daemon_status(h).unwrap().state, DaemonState::Cancelled);

        // Subsequent daemon_exit must not override Cancelled.
        os.daemon_exit(h, None);
        assert_eq!(os.daemon_status(h).unwrap().state, DaemonState::Cancelled);
    }

    #[test]
    fn cancel_unknown_or_exited_daemon_returns_false() {
        use crate::primitives::{DaemonHandle, DaemonKind, DaemonOps};
        let mut os = LocalOS::new();
        assert!(!os.cancel_daemon(DaemonHandle(9999)));

        let (h, _) = os.daemon_register("r4".into(), DaemonKind::Other, None);
        os.daemon_exit(h, None);
        assert!(!os.cancel_daemon(h));
    }

    #[test]
    fn list_daemons_returns_all_entries() {
        use crate::primitives::{DaemonKind, DaemonOps};
        let mut os = LocalOS::new();
        let (h1, _) = os.daemon_register("a".into(), DaemonKind::Reflection, None);
        let (h2, _) = os.daemon_register("b".into(), DaemonKind::IoPreload, None);
        let snap = os.list_daemons();
        assert_eq!(snap.len(), 2);
        let handles: std::collections::HashSet<u64> = snap.iter().map(|e| e.handle.raw()).collect();
        assert!(handles.contains(&h1.raw()));
        assert!(handles.contains(&h2.raw()));
    }

    #[test]
    fn daemon_spawn_and_exit_emit_trace_events() {
        use crate::primitives::{DaemonKind, DaemonOps, TraceOps};
        let mut os = LocalOS::new();
        let (h, _) = os.daemon_register("traceme".into(), DaemonKind::Reflection, None);
        os.daemon_exit(h, None);
        let recs = os.trace_drain_since(0);
        assert!(recs.iter().any(|r| r.name == "daemon.spawn"));
        assert!(recs.iter().any(|r| r.name == "daemon.exit"));
    }
}
