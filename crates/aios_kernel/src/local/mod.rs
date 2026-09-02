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
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
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
/// Default capacity of the LLM usage audit ledger. The agent side drains and persists records
/// periodically after account entries, so this only needs to buffer the records between two drains.
const DEFAULT_LLM_USAGE_CAPACITY: usize = 4096;
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
    /// Ready queue holding `(pid, priority)`. Priority is stored inline with each entry so enqueue
    /// comparisons never look up `Process.priority` in the table. Dequeue / scheduling filtering uses
    /// `ready_set` for O(1) marking, so terminate / signal-stop paths no longer need a linear retain.
    pub(super) ready_queue: VecDeque<(u64, u8)>,
    /// Membership index of `ready_queue`. Whether a pid is actually "schedulable" is decided by this set;
    /// queue entries can be stale tombstones, discarded by [`pop_ready`].
    pub(super) ready_set: FastSet<u64>,
    pub(super) wait_queue: FastMap<u64, Vec<u64>>,
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
    pub(super) completed_events: FastSet<EventId>,
    pub(super) completed_event_order: VecDeque<EventId>,
    pub(super) completed_event_retention: usize,
    /// Reverse index: event_id -> set of pids currently waiting on that event.
    /// Cleaned up lazily on notify (entries for completed events are removed)
    /// and verified against process state at wake-time, so stale pids are safe.
    pub(super) event_waiters: FastMap<EventId, FastSet<u64>>,
    /// Refcount of the kernel sources that reference each `event_id`: channel
    /// create / futex create / epoll registration of `EpollSource::Event`.
    /// Used by [`Self::completed_event_is_live`] for an O(1) check instead of scanning the
    /// `channels` / `futexes` / `epolls` tables. Reuse and decrement of a duplicate key
    /// must be strictly paired; missing one makes prune wrongly declare the event dead early.
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
    /// LLM usage audit ledger (bounded ring). Each `llm_account` appends one entry for external drain-and-persist.
    pub(super) llm_usage: crate::primitives::LlmUsageRing,
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
    /// Expiry wakeup heap: (until_tick, pid). One-to-one with until_tick/timeout_tick in process state;
    /// after a process is woken/terminated early its entry goes stale and is validated then dropped on pop.
    /// Saves advance_ticks / next_wakeup_tick from a full scan of sleeping processes every time.
    wakeup_heap: BinaryHeap<Reverse<(u64, u64)>>,
}

/// Internal daemon registration entry (with a live-reference token). Only Snapshot is exposed externally.
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
            wakeup_heap: BinaryHeap::new(),
            wait_queue: FastMap::default(),
            children_by_parent: FastMap::default(),
            next_pid: 1,
            current_pid: None,
            yield_requested: false,
            tick: 0,
            round_robin: true,
            shared_memory: FastMap::default(),
            next_pgid: 1,
            completed_events: FastSet::default(),
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
            llm_usage: crate::primitives::LlmUsageRing::new(DEFAULT_LLM_USAGE_CAPACITY),
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
        // Re-attach surviving children to the root (foreground) process instead of
        // orphaning them. `ensure_child_scope` walks parent chains, so a `None`
        // parent makes a live process unmanageable by every process (including the
        // foreground model) even though it keeps running and is listed by
        // `list_processes`. The root outlives its subtree in a session, so
        // re-parenting keeps the whole tree inside the foreground's management
        // scope. Fall back to orphaning only when no other foreground process
        // exists (e.g. the root itself is being removed). Only a live foreground
        // qualifies as target: `terminate_pid` keeps terminated processes as
        // tombstones in the table, and re-parenting to one would silently
        // reproduce the unmanageable-orphan bug (the parent chain still ends in
        // `None` for a zombie).
        // The foreground with the lowest pid is the session root:
        // `begin_foreground` creates it first and pid allocation is monotonic.
        // `min_by_key` makes the choice independent of the `processes` HashMap
        // iteration order, so the re-attachment target is deterministic even if
        // several foregrounds exist.
        let root_pid = self
            .processes
            .values()
            .filter(|proc| {
                proc.is_foreground && proc.pid != pid && proc.state != ProcessState::Terminated
            })
            .min_by_key(|proc| proc.pid)
            .map(|proc| proc.pid);
        // Snapshot the live orphans first: `register_child` needs `&mut self`, so it
        // cannot be called while a `child` borrow from the process table is alive.
        let live_orphans: Vec<u64> = orphaned_children
            .iter()
            .copied()
            .filter(|child_pid| {
                self.processes
                    .get(child_pid)
                    .is_some_and(|proc| proc.state != ProcessState::Terminated)
            })
            .collect();
        for child_pid in &live_orphans {
            if let Some(root) = root_pid {
                self.register_child(root, *child_pid);
            }
        }
        for child_pid in orphaned_children {
            if let Some(child) = self.processes.get_mut(&child_pid) {
                if child.state == ProcessState::Terminated {
                    terminated_children.push(child_pid);
                } else if let Some(root) = root_pid {
                    child.parent_pid = Some(root);
                    child.mailbox.push_back(format!(
                        "Parent process {} exited; this process was re-attached to the root process {}.",
                        pid, root
                    ));
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

    /// Resolve the pid of the "currently calling process".
    ///
    /// Background sub-agents each run in their own `tokio::spawn` task yet share one
    /// `Arc<Mutex<Kernel>>`. `self.current_pid` is a single scalar: any concurrent task finishing /
    /// yielding rewrites it to `None` or another pid, so a blocking syscall executed by another task
    /// (e.g. `wait_on_events`) may read a concurrently-cleared `self.current_pid`,
    /// falsely reporting "No process currently running.".
    ///
    /// The `TASK_PID` task-local is each task's authoritative caller identity (`current_process_id`
    /// resolves through it too). Read the task-local first here and fall back to `self.current_pid` only
    /// when missing, so identity syscalls always see the true caller under concurrent sub-agents.
    fn effective_current_pid(&self) -> Option<u64> {
        crate::kernel::current_task_pid().or(self.current_pid)
    }

    fn enqueue_ready(&mut self, pid: u64) {
        // Return directly for a nonexistent pid or one already queued. The `ready_set.insert` return
        // value doubles as the dedup check, saving an extra table lookup.
        if !self.processes.contains_key(&pid) || !self.ready_set.insert(pid) {
            return;
        }

        let priority = self.process_priority(pid);
        // The priority insertion-point search now reads the cached priority from the tuple instead of
        // querying the processes table. This takes batch wakeup / spawn from O(n^2) to O(n*k) (k =
        // queue length after that priority). Sequence tombstones add no extra cost during comparison.
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
        // O(1) tombstone: only removes set membership; `pop_ready` does the cleanup.
        self.ready_set.remove(&pid);
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Terminated;
            proc.result = Some(result.clone());
        }
        // A process entering Terminated affects SHM permissions (the owner is unreachable), which is what
        // invalidates the cached "owner still alive" accessible=true results in time. Strictly speaking, a second
        // defense line in `shm_read` independently checks owner state, but bumping here reduces the chance of
        // inconsistent syscall error returns.
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
        // Report a missing target as "does not exist" instead of the generic scope
        // error: without this check a nonexistent pid falls through the parent-chain
        // walk below and is misreported as "outside its scope", which conflates "not
        // allowed" with "no such process" for all callers (kill/reap/signal/wait).
        if !self.processes.contains_key(&target) {
            return Err(format!("Process {} does not exist.", target));
        }
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
        // Consult the live owner pgid lookup here (not the creation-time snapshot in entry.owner_pgid),
        // because the owner process may only be added to its new group by `set_process_group` after
        // creation, and a cached None would reject an otherwise legitimate write. The perm cache is
        // invalidated via topology_version; this spot only cares about the correctness of one decision.
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

    /// Register a new kernel source reference in `event_source_refs`. Call once for every new
    /// channel / futex / epoll(EpollSource::Event), with [`dec_event_source_ref`]
    /// as the paired call on destroy.
    fn inc_event_source_ref(&mut self, event: EventId) {
        let entry = self.event_source_refs.entry(event).or_insert(0);
        *entry = entry.saturating_add(1);
    }

    /// Release one source reference; clear the entry when it reaches zero to avoid unbounded growth.
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
        completed_event_ids: &FastSet<EventId>,
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
        // registrations) is consulted via `event_source_refs`, also O(1).
        if self
            .event_waiters
            .get(&event_id)
            .is_some_and(|set| !set.is_empty())
        {
            return true;
        }
        self.event_source_refs.get(&event_id).copied().unwrap_or(0) > 0
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

/// Sensitive-path blocklist (equivalent to the agent-side FileStore behavior; the "root permission cannot be pierced" security boundary).
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
    /// Unified VFS trace event helper shared by the vfs_* methods.
    fn vfs_emit_trace(
        &mut self,
        op: &'static str,
        pid: Option<u64>,
        path: &std::path::Path,
        bytes: u64,
        verdict: Option<&RlimitVerdict>,
    ) {
        use crate::types::FastMap;
        let mut fields: FastMap<String, String> = FastMap::default();
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

impl LocalOS {
    /// Shared trace emitter for DaemonOps.
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
        let mut fields: FastMap<String, String> = FastMap::default();
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
        let mut fields: FastMap<String, String> = FastMap::default();
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
                // One channels lookup gets the entry, then the mask is judged directly, avoiding a second
                // lookup via `epoll_actual_events_for_source`.
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
        // Dedup with a temporary set instead of linearly scanning wait_ids on every insert.
        // On epolls with many registrations this takes O(n^2) down to O(n).
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
            // After rotation the old event_id is no longer indexed by any source; swap in the new event_id
            // and adjust event_source_refs in lockstep to keep the strict pairing.
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

impl Kernel for LocalOS {}

mod syscall;
mod kernel_internal;
mod futex;
mod trace;
mod rlimit;
mod llm;
mod vfs;
mod daemon;
mod ipc;
mod epoll;

#[cfg(test)]
mod tests;
