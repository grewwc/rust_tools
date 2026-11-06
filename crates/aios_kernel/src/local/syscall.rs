use super::*;

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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
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
        if let Some(tt) = timeout_tick {
            self.wakeup_heap.push(Reverse((tt, current)));
        }
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
        let sender_pid = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
        self.require_capability(sender_pid, |caps| caps.ipc_send, "send ipc")?;

        if !self.processes.contains_key(&target_pid) {
            return Err(format!("Process {} does not exist.", target_pid));
        }

        // Walk all kinship determinations once so both permission-check layers reuse the same verdicts,
        // avoiding two passes over the process tree on the happy path.
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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.ipc_receive, "read mailbox")?;
        if let Some(current_proc) = self.processes.get_mut(&current) {
            let messages: Vec<String> = current_proc.mailbox.drain(..).collect();
            Ok(messages)
        } else {
            Err("Current process not found in process table.".to_string())
        }
    }

    fn set_env(&mut self, key: String, value: String) -> Result<(), String> {
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.env_write, "set env")?;
        if let Some(current_proc) = self.processes.get_mut(&current) {
            current_proc.env.insert(key, value);
            Ok(())
        } else {
            Err("Current process not found in process table.".to_string())
        }
    }

    fn get_env(&self, key: &str) -> Option<String> {
        let current = self.effective_current_pid()?;
        let current_proc = self.processes.get(&current)?;
        current_proc.env.get(key).cloned()
    }

    fn current_process_id(&self) -> Option<u64> {
        self.effective_current_pid()
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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
        self.require_capability(current, |caps| caps.sleep, "sleep")?;
        let until_tick = self.tick.saturating_add(turns.max(1));
        if let Some(proc) = self.processes.get_mut(&current) {
            proc.state = ProcessState::Sleeping { until_tick };
        }
        self.wakeup_heap.push(Reverse((until_tick, current)));
        self.current_pid = None;
        self.yield_requested = true;
        Ok(until_tick)
    }

    fn kill_process(&mut self, target_pid: u64, reason: String) -> Result<(), String> {
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
        if self.shared_memory.contains_key(&key) {
            return Err(format!("Shared memory key '{}' already exists.", key));
        }
        let owner_pgid = self.processes.get(&current).and_then(|p| p.process_group);
        // owner_pgid only flows through trace as a creation-time snapshot; SHM permissions consult
        // the table live to handle changes made by set_process_group afterwards.
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

        let current = match self.effective_current_pid() {
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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
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
        let current = self
            .effective_current_pid()
            .ok_or("No process currently running.")?;
        if let Some(proc) = self.processes.get_mut(&current) {
            proc.working_dir = Some(dir);
            Ok(())
        } else {
            Err("Current process not found in process table.".to_string())
        }
    }

    fn get_working_dir(&self) -> Option<PathBuf> {
        let current = self.effective_current_pid()?;
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
