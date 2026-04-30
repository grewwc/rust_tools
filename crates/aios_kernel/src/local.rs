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

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use crate::types::{FastMap, FastSet};

use crate::kernel::{
    EventId, Kernel, KernelInternal, Process, ProcessCapabilities, ProcessState, ShmReadError,
    Signal, Syscall, WaitPolicy, WaitReason, DEFAULT_MAILBOX_CAPACITY,
};
use crate::primitives::{
    ChannelId, ChannelMetaSnapshot, ChannelOwnerTag, DaemonCancelToken, DaemonEntrySnapshot,
    DaemonHandle, DaemonKind, DaemonOps, DaemonState, IpcOps, IpcRecvResult,
    FutexAddr, FutexOps, FutexState, FutexWakeReason, LlmAccountOutcome, LlmModelPrice, LlmOps,
    LlmUsageReport, ResourceLimit, ResourceUsage, ResourceUsageDelta, RlimitDim, RlimitOps,
    RlimitVerdict, TraceKind, TraceLevel, TraceOps, TraceRecord, TraceRing, VfsError, VfsOps,
    VfsStat,
};

pub(super) struct ShmEntry {
    value: String,
    owner_pid: u64,
    owner_pgid: Option<u64>,
    checksum: u64,
    version: u64,
    created_at_tick: u64,
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
    pub(super) ready_queue: VecDeque<u64>,
    pub(super) wait_queue: HashMap<u64, Vec<u64>>,
    pub next_pid: u64,
    pub current_pid: Option<u64>,
    pub(super) yield_requested: bool,
    pub tick: u64,
    pub(super) round_robin: bool,
    pub(super) shared_memory: FastMap<String, ShmEntry>,
    pub next_pgid: u64,
    /// All event IDs that have ever been marked completed, used to detect
    /// already-satisfied wait conditions in wait_on_events.
    pub(super) completed_events: std::collections::HashSet<EventId>,
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
    pub(super) ref_holders: Vec<String>,
    pub(super) event_id: EventId,
    pub(super) capacity: usize,
    pub(super) queue: VecDeque<String>,
    pub(super) closed: bool,
}

impl Default for LocalOS {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalOS {
    pub fn new() -> Self {
        Self {
            processes: FastMap::default(),
            ready_queue: VecDeque::new(),
            wait_queue: HashMap::new(),
            next_pid: 1,
            current_pid: None,
            yield_requested: false,
            tick: 0,
            round_robin: true,
            shared_memory: FastMap::default(),
            next_pgid: 1,
            completed_events: std::collections::HashSet::new(),
            futexes: FastMap::default(),
            next_futex_id: 1,
            trace: TraceRing::new(4096),
            llm_prices: FastMap::default(),
            daemons: FastMap::default(),
            next_daemon_id: 1,
            channels: FastMap::default(),
            next_channel_id: 1,
            next_internal_event_id: 1_000_000,
        }
    }

    fn sort_ready_queue(&mut self) {
        let mut pids: Vec<u64> = self.ready_queue.drain(..).collect();
        pids.sort_by_key(|&pid| self.processes.get(&pid).map(|p| p.priority).unwrap_or(255));
        self.ready_queue = pids.into_iter().collect();
    }

    fn terminate_pid(&mut self, pid: u64, result: String) {
        self.ready_queue.retain(|queued_pid| *queued_pid != pid);
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Terminated;
            proc.result = Some(result.clone());
        }

        if let Some(waiting_pids) = self.wait_queue.remove(&pid) {
            for waiting_pid in waiting_pids {
                if let Some(waiting_proc) = self.processes.get_mut(&waiting_pid) {
                    waiting_proc.state = ProcessState::Ready;
                    waiting_proc.mailbox.push_back(format!(
                        "Process {} terminated with result: {}",
                        pid, result
                    ));
                    self.ready_queue.push_back(waiting_pid);
                }
            }
        }
        if !self.ready_queue.is_empty() {
            self.sort_ready_queue();
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
        let mut stack = vec![pid];
        while let Some(current) = stack.pop() {
            for (child_pid, proc) in self.processes.iter() {
                if proc.parent_pid == Some(current) && !result.contains(child_pid) {
                    result.push(*child_pid);
                    stack.push(*child_pid);
                }
            }
        }
        result
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
        if pid == entry.owner_pid {
            return true;
        }
        if self.is_same_process_group(pid, entry.owner_pid) {
            return true;
        }
        if self.is_ancestor_of(pid, entry.owner_pid) {
            return true;
        }
        false
    }

    fn is_shm_readable_by(&self, pid: u64, entry: &ShmEntry) -> bool {
        if pid == entry.owner_pid {
            return true;
        }
        if self.is_same_process_group(pid, entry.owner_pid) {
            return true;
        }
        if self.is_ancestor_of(pid, entry.owner_pid) {
            return true;
        }
        if self.is_sibling(pid, entry.owner_pid) {
            return true;
        }
        false
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
                    self.terminate_pid(*pid, format!("Killed (cascade from SIGKILL to {})", target_pid));
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
                        self.ready_queue.push_back(target_pid);
                        self.sort_ready_queue();
                    }
                }
            }
            Signal::SigStop => {
                if let Some(proc) = self.processes.get_mut(&target_pid) {
                    if matches!(proc.state, ProcessState::Terminated | ProcessState::Stopped) {
                        return Ok(());
                    }
                    self.ready_queue.retain(|p| *p != target_pid);
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
                    self.ready_queue.push_back(target_pid);
                    self.sort_ready_queue();
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
        completed_event_ids: &std::collections::HashSet<EventId>,
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
        let mut inherited_capabilities =
            requested_capabilities.clone().unwrap_or_else(ProcessCapabilities::full);
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
            self.processes.get(&parent).and_then(|p| p.working_dir.clone())
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
        self.ready_queue.push_back(pid);
        self.sort_ready_queue();
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
                    event_ids: deduped,
                    policy,
                    timeout_tick,
                },
            };
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

        if sender_pid != target_pid
            && !self.is_same_process_group(sender_pid, target_pid)
            && !self.is_ancestor_of(sender_pid, target_pid)
            && !self.is_ancestor_of(target_pid, sender_pid)
            && !self.is_sibling(sender_pid, target_pid)
        {
            return Err(format!(
                "Permission denied: process {} cannot send IPC to process {} (not in same process group or parent-child relationship).",
                sender_pid, target_pid
            ));
        }

        if sender_pid != target_pid
            && !self.is_same_process_group(sender_pid, target_pid)
            && !self.is_ancestor_of(sender_pid, target_pid)
        {
            let target_pgid = self.processes.get(&target_pid).and_then(|p| p.process_group);
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
        self.processes.iter().map(|(_, proc)| proc.clone()).collect()
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
        self.processes.remove(&target_pid);
        self.ready_queue.retain(|pid| *pid != target_pid);
        self.wait_queue.remove(&target_pid);
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
            Ok(())
        } else {
            Err(format!("Process {} does not exist.", pid))
        }
    }

    fn signal_process_group(&mut self, pgid: u64, signal: Signal) -> Result<usize, String> {
        let target_pids: Vec<u64> = self.processes.iter()
            .filter(|(_, proc)| proc.process_group == Some(pgid) && proc.state != ProcessState::Terminated)
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
        let owner_pgid = self
            .processes
            .get(&current)
            .and_then(|p| p.process_group);
        let checksum = shm_checksum(&value, current);
        self.shared_memory.insert(
            key,
            ShmEntry {
                value,
                owner_pid: current,
                owner_pgid,
                checksum,
                version: 1,
                created_at_tick: self.tick,
            },
        );
        Ok(())
    }

    fn shm_read(&self, key: &str) -> Result<String, ShmReadError> {
        let entry = self
            .shared_memory
            .get(key)
            .ok_or(ShmReadError::NotFound)?;

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
            Err(ShmReadError::OwnerTerminated { .. }) => {
                self.shared_memory.get(key).map(|e| {
                    format!(
                        "[DEGRADED: owner terminated] {}",
                        e.value
                    )
                })
            }
            Err(ShmReadError::Corrupted { .. }) => {
                self.shared_memory.get(key).map(|e| {
                    format!(
                        "[DEGRADED: checksum mismatch] {}",
                        e.value
                    )
                })
            }
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
        self.processes.get(&current).and_then(|p| p.working_dir.clone())
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
        let pid = self.spawn(
            parent_pid,
            name,
            goal,
            priority,
            quota_turns,
            None,
            None,
        )?;
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
        let pid = self.ready_queue.pop_front()?;
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Running;
            self.current_pid = Some(pid);
            self.yield_requested = false;
            Some(proc.clone())
        } else {
            None
        }
    }

    fn pop_all_ready(&mut self, max: usize) -> Vec<Process> {
        let mut result = Vec::new();
        let count = max.min(self.ready_queue.len());
        for _ in 0..count {
            if let Some(pid) = self.ready_queue.pop_front() {
                if let Some(proc) = self.processes.get_mut(&pid) {
                    proc.state = ProcessState::Running;
                    result.push(proc.clone());
                }
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

    fn drop_terminated(&mut self, target_pid: u64) -> bool {
        if !matches!(
            self.processes.get(&target_pid).map(|proc| &proc.state),
            Some(ProcessState::Terminated)
        ) {
            return false;
        }
        self.processes.remove(&target_pid);
        self.ready_queue.retain(|pid| *pid != target_pid);
        self.wait_queue.remove(&target_pid);
        true
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
                proc.mailbox.push_back(format!(
                    "Sleep finished at scheduler tick {}.",
                    self.tick
                ));
                self.ready_queue.push_back(pid);
            }
        }
        for pid in wake_async_timeout_pids {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Ready;
                proc.mailbox.push_back(format!(
                        "Event wait timeout reached at scheduler tick {}.",
                    self.tick
                ));
                self.ready_queue.push_back(pid);
            }
        }
        if !self.ready_queue.is_empty() {
            self.sort_ready_queue();
        }
    }

    fn has_ready(&self) -> bool {
        !self.ready_queue.is_empty()
    }

    fn ready_count(&self) -> usize {
        self.ready_queue.len()
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
        self.ready_queue.push_back(pid);
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

        // Accumulate into the persistent completed-events set so that wait_on_events
        // called after this notification can still detect the already-satisfied condition.
        for &eid in completed_event_ids {
            self.completed_events.insert(eid);
        }

        let completed_set: std::collections::HashSet<EventId> =
            self.completed_events.iter().copied().collect();
        let mut wake_pids = Vec::new();
        for (pid, proc) in self.processes.iter() {
            if let ProcessState::Waiting {
                reason:
                    WaitReason::Events {
                        event_ids,
                        policy,
                        ..
                    },
            } = &proc.state
                && self.event_wait_is_satisfied(event_ids, policy, &completed_set)
            {
                wake_pids.push(*pid);
            }
        }

        for pid in &wake_pids {
            if let Some(proc) = self.processes.get_mut(pid) {
                proc.state = ProcessState::Ready;
                proc.mailbox.push_back(format!(
                    "[EVENT_WAKE]\nReason: event wait condition satisfied.\nCompleted event ids: {}\nRecommended next actions:\n1. Inspect the event-producing subsystem for fresh state.\n2. If these events came from async tool work, use tool_status or tool_wait to collect results.\n3. Cancel low-value still-running branches when appropriate.\n4. If enough results are already available, continue reasoning immediately.",
                    completed_event_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ")
                ));
                self.ready_queue.push_back(*pid);
            }
        }
        if !wake_pids.is_empty() {
            self.sort_ready_queue();
        }
        wake_pids
    }

    fn increment_turns_used_for(&mut self, pid: u64) {
        let _ = <Self as RlimitOps>::rusage_charge(
            self,
            pid,
            ResourceUsageDelta { turns: 1, ..Default::default() },
        );
    }

    fn increment_tool_calls_used_for(&mut self, pid: u64) {
        let _ = <Self as RlimitOps>::rusage_charge(
            self,
            pid,
            ResourceUsageDelta { tool_calls: 1, ..Default::default() },
        );
    }

    fn check_daemon_restart(&mut self) -> Vec<u64> {
        let mut restarted = Vec::new();
        let terminated_daemons: Vec<(u64, String, u8, usize, usize, Option<u64>, FastMap<String, String>, FastSet<String>, Option<PathBuf>)> = self.processes.iter()
            .filter(|(_, proc)| {
                proc.is_daemon
                    && proc.state == ProcessState::Terminated
                    && proc.restart_count < proc.max_restarts
            })
            .map(|(pid, proc)| {
                (*pid, proc.name.clone(), proc.priority, proc.quota_turns, proc.restart_count, proc.parent_pid, proc.env.clone(), proc.allowed_tools.clone(), proc.working_dir.clone())
            })
            .collect();

        for (old_pid, name, priority, quota_turns, restart_count, parent_pid, env, allowed_tools, working_dir) in terminated_daemons {
            self.processes.remove(&old_pid);
            self.ready_queue.retain(|p| *p != old_pid);
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
            self.ready_queue.push_back(new_pid);
            restarted.push(new_pid);
        }

        if !self.ready_queue.is_empty() {
            self.sort_ready_queue();
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
        let owner = self.current_pid;
        self.futexes
            .insert(id, FutexState::new(initial, owner, label));
        FutexAddr(id)
    }

    fn futex_load(&self, addr: FutexAddr) -> Option<u64> {
        self.futexes
            .get(&addr.0)
            .map(|s| s.value.load(std::sync::atomic::Ordering::SeqCst))
    }

    fn futex_cas(&self, addr: FutexAddr, expected: u64, new_value: u64) -> Result<u64, u64> {
        use std::sync::atomic::Ordering::SeqCst;
        let state = match self.futexes.get(&addr.0) {
            Some(s) => s,
            None => return Err(u64::MAX),
        };
        match state
            .value
            .compare_exchange(expected, new_value, SeqCst, SeqCst)
        {
            Ok(prev) => Ok(prev),
            Err(cur) => Err(cur),
        }
    }

    fn futex_fetch_add(&self, addr: FutexAddr, delta: u64) -> Option<u64> {
        let state = self.futexes.get(&addr.0)?;
        Some(
            state
                .value
                .fetch_add(delta, std::sync::atomic::Ordering::SeqCst),
        )
    }

    fn futex_store(&self, addr: FutexAddr, new_value: u64) -> Option<u64> {
        let state = self.futexes.get(&addr.0)?;
        Some(state.value.swap(new_value, std::sync::atomic::Ordering::SeqCst))
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
                if !matches!(
                    proc.state,
                    ProcessState::Terminated | ProcessState::Stopped
                ) {
                    proc.state = ProcessState::Ready;
                    if !self.ready_queue.contains(&pid) {
                        self.ready_queue.push_back(pid);
                    }
                }
            }
        }
        if woken > 0 && !self.ready_queue.is_empty() {
            self.sort_ready_queue();
        }
        woken
    }

    fn futex_destroy(&mut self, addr: FutexAddr) -> bool {
        self.futexes.remove(&addr.0).is_some()
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
            fields,
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
            fields,
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
            fields,
            message,
        };
        self.trace.push(rec);
    }

    fn trace_recent(&self, n: usize) -> Vec<TraceRecord> {
        self.trace
            .buf
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect()
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
        self.trace
            .buf
            .back()
            .map(|r| r.seq)
            .unwrap_or(0)
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
        self.llm_prices.get(model).copied().unwrap_or_else(LlmModelPrice::zero)
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
            fields.insert("prompt_tokens".to_string(), report.prompt_tokens.to_string());
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

        LlmAccountOutcome { charged_cost_micros, verdict }
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
            let delta = ResourceUsageDelta { fs_bytes: bytes, ..Default::default() };
            Some(<Self as RlimitOps>::rusage_charge(self, pid, delta))
        } else {
            None
        };

        self.vfs_emit_trace("read", pid, path, bytes, verdict.as_ref());

        if let Some(RlimitVerdict::Exceeded { dimension, used, limit }) = verdict {
            return Err(VfsError::QuotaExceeded { dimension, used, limit });
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
                std::fs::create_dir_all(parent).map_err(|e| {
                    VfsError::Io(format!("Failed to create directory: {}", e))
                })?;
            }
        }
        std::fs::write(path, content)
            .map_err(|e| VfsError::Io(format!("Failed to write file: {}", e)))?;
        let bytes = content.len() as u64;

        let verdict = if let Some(pid) = pid {
            let delta = ResourceUsageDelta { fs_bytes: bytes, ..Default::default() };
            Some(<Self as RlimitOps>::rusage_charge(self, pid, delta))
        } else {
            None
        };

        self.vfs_emit_trace("write", pid, path, bytes, verdict.as_ref());

        if let Some(RlimitVerdict::Exceeded { dimension, used, limit }) = verdict {
            return Err(VfsError::QuotaExceeded { dimension, used, limit });
        }
        Ok(())
    }

    fn vfs_stat(&mut self, path: &std::path::Path) -> Result<VfsStat, VfsError> {
        if is_sensitive_fs_path(path) {
            return Err(VfsError::PermissionDenied(path.display().to_string()));
        }
        let meta = std::fs::metadata(path)
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => VfsError::NotFound(path.display().to_string()),
                _ => VfsError::Io(e.to_string()),
            })?;
        Ok(VfsStat { size: meta.len(), is_file: meta.is_file(), is_dir: meta.is_dir() })
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
            entry.cancel_flag.store(true, std::sync::atomic::Ordering::Release);
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
        self.channels.insert(
            id,
            IpcChannelEntry {
                owner_pid,
                label: label.clone(),
                owner_tag,
                ref_count: initial_ref_holders.len() as u32,
                ref_holders: initial_ref_holders,
                event_id,
                capacity: cap,
                queue: VecDeque::new(),
                closed: false,
            },
        );
        let channel = ChannelId(id);
        self.channel_emit_trace("channel_create", channel, owner_pid, &label, 0);
        channel
    }

    fn channel_event_id(&self, channel: ChannelId) -> Option<EventId> {
        self.channels.get(&channel.0).map(|entry| entry.event_id)
    }

    fn channel_meta(&self, channel: ChannelId) -> Option<ChannelMetaSnapshot> {
        self.channels.get(&channel.0).map(|entry| ChannelMetaSnapshot {
            channel,
            label: entry.label.clone(),
            owner_pid: entry.owner_pid,
            owner_tag: entry.owner_tag,
            ref_count: entry.ref_count,
            ref_holders: entry.ref_holders.clone(),
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
                ref_holders: entry.ref_holders.clone(),
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
            (entry.label.clone(), entry.queue.len(), should_notify, entry.event_id)
        };
        if should_notify {
            self.notify_events_completed(&[event_id]);
            let next_event_id = self.alloc_internal_event_id();
            if let Some(entry) = self.channels.get_mut(&channel.0) {
                entry.event_id = next_event_id;
            }
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
        self.channel_retain_named(
            channel,
            format!("retain#{}", next_idx),
        )
    }

    fn channel_retain_named(&mut self, channel: ChannelId, holder: String) -> Result<u32, String> {
        let entry = self
            .channels
            .get_mut(&channel.0)
            .ok_or_else(|| format!("Channel {} does not exist.", channel))?;
        entry.ref_holders.push(holder);
        entry.ref_count = entry.ref_count.saturating_add(1);
        Ok(entry.ref_count)
    }

    fn channel_release(&mut self, channel: ChannelId) -> Result<u32, String> {
        let holder = self
            .channels
            .get(&channel.0)
            .and_then(|entry| entry.ref_holders.last().cloned())
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
        let Some(idx) = entry.ref_holders.iter().position(|h| h == holder) else {
            return Err(format!(
                "Channel {} does not have ref holder {:?}.",
                channel, holder
            ));
        };
        entry.ref_holders.remove(idx);
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
            (entry.owner_pid, entry.label.clone(), Self::channel_is_gc_eligible(entry))
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
        self.channels.remove(&channel.0);
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
            self.channels.remove(id);
            self.channel_emit_trace("gc", ChannelId(*id), None, label, 0);
        }
        doomed.len()
    }

    fn channel_close(
        &mut self,
        closer_pid: Option<u64>,
        channel: ChannelId,
    ) -> Result<(), String> {
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
            (entry.label.clone(), entry.queue.len(), should_notify, entry.event_id)
        };
        if should_notify {
            self.notify_events_completed(&[event_id]);
            let next_event_id = self.alloc_internal_event_id();
            if let Some(entry) = self.channels.get_mut(&channel.0) {
                entry.event_id = next_event_id;
            }
        }
        self.channel_emit_trace("close", channel, closer_pid, &label, depth);
        Ok(())
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
        let root = os.begin_foreground("foreground".to_string(), "root goal".to_string(), 10, 8, None);
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
        let root = os.begin_foreground("foreground".to_string(), "root goal".to_string(), 10, 8, None);

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
        let root = os.begin_foreground("foreground".to_string(), "root goal".to_string(), 10, 8, None);

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
        let root = os.begin_foreground("foreground".to_string(), "root goal".to_string(), 10, 8, None);

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
            Some("[EVENT_WAKE]\nReason: event wait condition satisfied.\nCompleted event ids: evt_2\nRecommended next actions:\n1. Inspect the event-producing subsystem for fresh state.\n2. If these events came from async tool work, use tool_status or tool_wait to collect results.\n3. Cancel low-value still-running branches when appropriate.\n4. If enough results are already available, continue reasoning immediately.")
        );
    }

    #[test]
    fn foreground_process_enables_env_access() {
        let mut os = LocalOS::new();
        os.begin_foreground("foreground".to_string(), "root goal".to_string(), 10, 8, None);
        os.set_env("scope".to_string(), "root".to_string()).unwrap();
        assert_eq!(os.get_env("scope").as_deref(), Some("root"));
    }

    #[test]
    fn sleeping_process_wakes_after_tick_advance() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("foreground".to_string(), "root goal".to_string(), 10, 8, None);
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
        let root = os.begin_foreground("foreground".to_string(), "root goal".to_string(), 10, 8, None);
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
        let root = os.begin_foreground("foreground".to_string(), "root goal".to_string(), 10, 8, None);
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
    fn kill_cascades_to_grandchildren() {
        let mut os = LocalOS::new();
        let root =
            os.begin_foreground("foreground".to_string(), "root goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(Some(root), "child".to_string(), "child goal".to_string(), 20, 4, None, None)
            .unwrap();
        let grandchild = os
            .spawn(Some(child), "grandchild".to_string(), "gc goal".to_string(), 30, 2, None, None)
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
        assert!(os.get_process(grandchild).unwrap().result.as_ref().unwrap().contains("cascade"));
    }

    #[test]
    fn foreground_process_has_is_foreground_flag() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        assert!(os.get_process(root).unwrap().is_foreground);

        let child = os
            .spawn(Some(root), "bg".to_string(), "bg goal".to_string(), 20, 4, None, None)
            .unwrap();
        assert!(!os.get_process(child).unwrap().is_foreground);
    }

    #[test]
    fn sigstop_stops_and_sigcont_resumes_process() {
        let mut os = LocalOS::new();
        let root =
            os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(Some(root), "worker".to_string(), "work".to_string(), 20, 4, None, None)
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
        let root =
            os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(Some(root), "worker".to_string(), "work".to_string(), 20, 4, None, None)
            .unwrap();
        let grandchild = os
            .spawn(Some(child), "gc".to_string(), "gc work".to_string(), 30, 2, None, None)
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
        let root =
            os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(Some(root), "worker".to_string(), "work".to_string(), 20, 4, None, None)
            .unwrap();

        os.signal_process(child, Signal::SigTerm).unwrap();
        let child_proc = os.get_process(child).unwrap();
        assert!(child_proc.pending_signals.contains(&Signal::SigTerm));
    }

    #[test]
    fn sigcancel_is_consumed_without_terminating_process() {
        let mut os = LocalOS::new();
        let root =
            os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);

        os.signal_process(root, Signal::SigCancel).unwrap();
        assert!(os.get_process(root).unwrap().pending_signals.contains(&Signal::SigCancel));

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
        let root =
            os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os
            .spawn(Some(root), "worker".to_string(), "work".to_string(), 20, 4, None, None)
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
        let child1 = os.spawn(Some(root), "c1".to_string(), "g1".to_string(), 20, 4, None, None).unwrap();
        let child2 = os.spawn(Some(root), "c2".to_string(), "g2".to_string(), 20, 4, None, None).unwrap();

        os.set_process_group(child1, 100).unwrap();
        os.set_process_group(child2, 100).unwrap();

        assert_eq!(os.get_process(child1).unwrap().process_group, Some(100));
        assert_eq!(os.get_process(child2).unwrap().process_group, Some(100));

        let count = os.signal_process_group(100, Signal::SigStop).unwrap();
        assert_eq!(count, 2);
        assert!(matches!(os.get_process(child1).unwrap().state, ProcessState::Stopped));
        assert!(matches!(os.get_process(child2).unwrap().state, ProcessState::Stopped));
    }

    #[test]
    fn shared_memory_crud_operations() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);

        os.shm_create("config".to_string(), "value1".to_string()).unwrap();
        assert_eq!(os.shm_read("config"), Ok("value1".to_string()));
        assert_eq!(os.shared_memory.get("config").map(|e| e.owner_pid), Some(root));
        assert_eq!(os.shared_memory.get("config").and_then(|e| e.owner_pgid), None);

        os.shm_write("config".to_string(), "value2".to_string()).unwrap();
        assert_eq!(os.shm_read("config"), Ok("value2".to_string()));

        os.shm_delete("config").unwrap();
        assert_eq!(os.shm_read("config"), Err(ShmReadError::NotFound));
        assert!(os.shared_memory.get("config").is_none());

        assert!(os.shm_create("config".to_string(), "value3".to_string()).is_ok());
        assert!(os.shm_create("config".to_string(), "value4".to_string()).is_err());
    }

    #[test]
    fn working_dir_inherits_to_children() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        os.set_working_dir(std::path::PathBuf::from("/tmp/work")).unwrap();

        let child = os.spawn(Some(root), "child".to_string(), "goal".to_string(), 20, 4, None, None).unwrap();
        assert_eq!(os.get_process(child).unwrap().working_dir, Some(std::path::PathBuf::from("/tmp/work")));
    }

    #[test]
    fn daemon_auto_restarts_on_termination() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let daemon_pid = os.spawn_daemon(
            Some(root),
            "watcher".to_string(),
            "watch files".to_string(),
            20,
            4,
            2,
        ).unwrap();

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
        assert!(os.get_process(new_pid).unwrap().goal.contains("daemon restart #1"));
    }

    #[test]
    fn daemon_respects_max_restarts() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let daemon_pid = os.spawn_daemon(
            Some(root),
            "watcher".to_string(),
            "watch".to_string(),
            20,
            4,
            1,
        ).unwrap();

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
        let child_a = os.spawn(Some(root), "a".to_string(), "goal a".to_string(), 20, 4, None, None).unwrap();
        let child_b = os.spawn(Some(root), "b".to_string(), "goal b".to_string(), 20, 4, None, None).unwrap();

        os.set_current_pid(Some(child_a));
        let result = os.send_ipc(child_b, "hello".to_string());
        assert!(result.is_ok());

        let unrelated_root = os.begin_foreground("fg2".to_string(), "goal2".to_string(), 10, usize::MAX, None);
        let orphan = os.spawn(Some(unrelated_root), "orphan".to_string(), "goal orphan".to_string(), 20, 4, None, None).unwrap();

        os.set_current_pid(Some(orphan));
        let result = os.send_ipc(child_a, "intrusion".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Permission denied"));
    }

    #[test]
    fn ipc_allowed_within_same_process_group() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os.spawn(Some(root), "a".to_string(), "goal a".to_string(), 20, 4, None, None).unwrap();
        let child_b = os.spawn(Some(root), "b".to_string(), "goal b".to_string(), 20, 4, None, None).unwrap();

        os.set_process_group(child_a, 42).unwrap();
        os.set_process_group(child_b, 42).unwrap();

        os.set_current_pid(Some(child_a));
        let result = os.send_ipc(child_b, "hello group".to_string());
        assert!(result.is_ok());

        let outsider = os.spawn(Some(root), "outsider".to_string(), "goal out".to_string(), 20, 4, None, None).unwrap();
        os.set_current_pid(Some(outsider));
        let result = os.send_ipc(child_a, "intrusion".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn kill_process_rejected_for_non_descendant() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os.spawn(Some(root), "a".to_string(), "goal a".to_string(), 20, 4, None, None).unwrap();
        let child_b = os.spawn(Some(root), "b".to_string(), "goal b".to_string(), 20, 4, None, None).unwrap();

        os.set_current_pid(Some(child_a));
        let result = os.kill_process(child_b, "sibling kill".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("outside its scope"));
    }

    #[test]
    fn shm_write_rejected_for_non_owner_outside_group() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os.spawn(Some(root), "a".to_string(), "goal a".to_string(), 20, 4, None, None).unwrap();

        os.set_current_pid(Some(child_a));
        os.shm_create("secret".to_string(), "value_a".to_string()).unwrap();

        let child_b = os.spawn(Some(root), "b".to_string(), "goal b".to_string(), 20, 4, None, None).unwrap();
        os.set_current_pid(Some(child_b));
        let result = os.shm_write("secret".to_string(), "tampered".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Permission denied"));
    }

    #[test]
    fn shm_write_allowed_for_same_process_group() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os.spawn(Some(root), "a".to_string(), "goal a".to_string(), 20, 4, None, None).unwrap();
        let child_b = os.spawn(Some(root), "b".to_string(), "goal b".to_string(), 20, 4, None, None).unwrap();

        os.set_process_group(child_a, 100).unwrap();
        os.set_process_group(child_b, 100).unwrap();

        os.set_current_pid(Some(child_a));
        os.shm_create("shared_config".to_string(), "v1".to_string()).unwrap();

        os.set_current_pid(Some(child_b));
        let result = os.shm_write("shared_config".to_string(), "v2".to_string());
        assert!(result.is_ok());
        assert_eq!(os.shm_read("shared_config"), Ok("v2".to_string()));
    }

    #[test]
    fn shm_write_allowed_for_ancestor_of_owner() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os.spawn(Some(root), "child".to_string(), "goal child".to_string(), 20, 4, None, None).unwrap();

        os.set_current_pid(Some(child));
        os.shm_create("child_data".to_string(), "original".to_string()).unwrap();

        os.set_current_pid(Some(root));
        let result = os.shm_write("child_data".to_string(), "parent_override".to_string());
        assert!(result.is_ok());
        assert_eq!(os.shm_read("child_data"), Ok("parent_override".to_string()));
    }

    #[test]
    fn shm_delete_rejected_for_non_owner_outside_group() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child_a = os.spawn(Some(root), "a".to_string(), "goal a".to_string(), 20, 4, None, None).unwrap();
        let child_b = os.spawn(Some(root), "b".to_string(), "goal b".to_string(), 20, 4, None, None).unwrap();

        os.set_current_pid(Some(child_a));
        os.shm_create("owned_by_a".to_string(), "data".to_string()).unwrap();

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
        let child = os.spawn(Some(root), "child".to_string(), "goal".to_string(), 20, 4, None, None).unwrap();

        os.set_process_group(child, 77).unwrap();

        os.set_current_pid(Some(child));
        os.shm_create("group_key".to_string(), "val".to_string()).unwrap();

        let entry = os.shared_memory.get("group_key").unwrap();
        assert_eq!(entry.owner_pid, child);
        assert_eq!(entry.owner_pgid, Some(77));
    }

    #[test]
    fn shm_read_detects_checksum_corruption() {
        let mut os = LocalOS::new();
        let _root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        os.shm_create("data".to_string(), "original".to_string()).unwrap();

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
        os.shm_create("data".to_string(), "original".to_string()).unwrap();

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
        let child = os.spawn(Some(root), "worker".to_string(), "work".to_string(), 20, 4, None, None).unwrap();
        os.set_current_pid(Some(child));
        os.shm_create("child_data".to_string(), "value".to_string()).unwrap();
        os.set_current_pid(Some(root));

        os.terminate_pid(child, "done".to_string());

        let result = os.shm_read("child_data");
        assert!(matches!(result, Err(ShmReadError::OwnerTerminated { .. })));
    }

    #[test]
    fn shm_read_degraded_returns_data_on_owner_terminated() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os.spawn(Some(root), "worker".to_string(), "work".to_string(), 20, 4, None, None).unwrap();
        os.set_current_pid(Some(child));
        os.shm_create("child_data".to_string(), "important_value".to_string()).unwrap();
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
        let root1 = os.begin_foreground("fg1".to_string(), "goal1".to_string(), 10, usize::MAX, None);
        let child_a = os.spawn(Some(root1), "a".to_string(), "ga".to_string(), 20, 4, None, None).unwrap();

        let root2 = os.begin_foreground("fg2".to_string(), "goal2".to_string(), 10, usize::MAX, None);
        let child_b = os.spawn(Some(root2), "b".to_string(), "gb".to_string(), 20, 4, None, None).unwrap();

        os.set_current_pid(Some(child_a));
        os.shm_create("secret".to_string(), "private_data".to_string()).unwrap();

        os.set_current_pid(Some(child_b));
        let result = os.shm_read("secret");
        assert!(matches!(result, Err(ShmReadError::PermissionDenied { .. })));
    }

    #[test]
    fn shm_health_check_detects_corrupted_and_orphaned() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os.spawn(Some(root), "worker".to_string(), "work".to_string(), 20, 4, None, None).unwrap();

        os.set_current_pid(Some(child));
        os.shm_create("orphan_data".to_string(), "val1".to_string()).unwrap();
        os.shm_create("good_data".to_string(), "val2".to_string()).unwrap();
        os.set_current_pid(Some(root));

        os.terminate_pid(child, "done".to_string());

        if let Some(entry) = os.shared_memory.get_mut("good_data") {
            entry.value = "corrupted".to_string();
        }

        let issues = os.shm_health_check();
        assert_eq!(issues.len(), 2);

        let has_orphan = issues.iter().any(|(k, e)| {
            k == "orphan_data" && matches!(e, ShmReadError::OwnerTerminated { .. })
        });
        let has_corrupt = issues.iter().any(|(k, e)| {
            k == "good_data" && matches!(e, ShmReadError::Corrupted { .. })
        });
        assert!(has_orphan);
        assert!(has_corrupt);
    }

    #[test]
    fn shm_cleanup_orphans_removes_dead_owner_entries() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let child = os.spawn(Some(root), "worker".to_string(), "work".to_string(), 20, 4, None, None).unwrap();

        os.set_current_pid(Some(child));
        os.shm_create("orphan".to_string(), "will_be_removed".to_string()).unwrap();
        os.set_current_pid(Some(root));
        os.shm_create("root_data".to_string(), "stays".to_string()).unwrap();

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
            .spawn(Some(root), "w".to_string(), "do".to_string(), 20, 4, None, None)
            .unwrap();
        // Simulate worker going to sleep on a futex
        if let Some(p) = os.processes.get_mut(&worker) {
            p.state = ProcessState::Waiting {
                reason: WaitReason::ProcessExit { on_pid: root },
            };
        }
        os.ready_queue.retain(|&p| p != worker);

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
        assert!(os.ready_queue.contains(&worker));
    }

    #[test]
    fn futex_try_wait_reports_value_changed() {
        use crate::primitives::{FutexOps, FutexWakeReason};
        let mut os = LocalOS::new();
        let addr = os.futex_create(0, "t".to_string());
        assert!(os.futex_try_wait(addr, 0).is_none(), "should block when equal");
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
        assert_eq!(recs[1].fields.get("model").unwrap(), "gpt");
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
            os.rusage_charge(pid, ResourceUsageDelta { turns: 1, ..Default::default() }),
            RlimitVerdict::Ok
        );
        assert_eq!(
            os.rusage_charge(pid, ResourceUsageDelta { turns: 1, ..Default::default() }),
            RlimitVerdict::Ok
        );
        match os.rusage_charge(pid, ResourceUsageDelta { turns: 1, ..Default::default() }) {
            RlimitVerdict::Exceeded { dimension, used, limit } => {
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
        let probe = ResourceUsageDelta { tokens_in: 200, ..Default::default() };
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
        use crate::primitives::{
            LlmModelPrice, LlmOps, LlmUsageReport, RlimitOps, RlimitVerdict,
        };
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        // 1000 prompt tok => 2500 micros; 500 completion tok => 3000 micros.
        os.llm_set_price(
            "gpt-test".into(),
            LlmModelPrice { prompt_per_1k_micros: 2_500, completion_per_1k_micros: 6_000 },
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
        use crate::primitives::{LlmOps, LlmUsageReport};
        use crate::primitives::RlimitOps;
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
            LlmModelPrice { prompt_per_1k_micros: 1_000, completion_per_1k_micros: 0 },
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
            RlimitVerdict::Exceeded { dimension, used, limit } => {
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
        p.push(format!("aios_vfs_{}_{}_{}_{}", name, std::process::id(), nanos, seq));
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
        use crate::primitives::{
            ResourceLimit, RlimitDim, RlimitOps, VfsError, VfsOps,
        };
        let mut os = LocalOS::new();
        let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
        let mut limits = ResourceLimit::unlimited();
        limits.max_fs_bytes = 5;
        os.rlimit_set(pid, limits).unwrap();

        let p = tmp_path("quota");
        // 写 10 字节——超过 5 字节上限
        match os.vfs_write_all(Some(pid), &p, "0123456789").unwrap_err() {
            VfsError::QuotaExceeded { dimension: RlimitDim::FsBytes, .. } => {}
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
            .spawn(Some(outsider_root), "outsider".into(), "goal".into(), 20, 4, None, None)
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
        assert_eq!(os.channel_try_recv(Some(root), ch).unwrap(), IpcRecvResult::Closed);
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
        assert!(root_proc
            .mailbox
            .back()
            .map(|s| s.contains("[EVENT_WAKE]"))
            .unwrap_or(false));
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

        os.wait_on_events(vec![evt2], WaitPolicy::All, None).unwrap();
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

        os.channel_send(Some(root), keep_buffered, "x".into()).unwrap();
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
        let handles: std::collections::HashSet<u64> =
            snap.iter().map(|e| e.handle.raw()).collect();
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
