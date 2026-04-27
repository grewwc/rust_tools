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
use rust_tools::commonw::{FastMap, FastSet};

use super::kernel::{
    EventId, Kernel, KernelInternal, Process, ProcessCapabilities, ProcessState, ShmReadError,
    Signal, Syscall, WaitPolicy, WaitReason, DEFAULT_MAILBOX_CAPACITY,
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
        super::kernel::TASK_PID
            .try_with(|v| *v)
            .unwrap_or(None)
            .or(self.current_pid)
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
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.turns_used += 1;
        }
    }

    fn increment_tool_calls_used_for(&mut self, pid: u64) {
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.tool_calls_used += 1;
        }
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

impl Kernel for LocalOS {}

#[cfg(test)]
mod tests {
    use super::{LocalOS, ShmReadError};
    use crate::ai::os::kernel::{
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
}
