use super::*;

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
        // Skip lazily deleted tombstones: drop pids no longer present in ready_set.
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
        let cap = max.min(self.ready_set.len());
        while result.len() < cap {
            let Some((pid, _priority)) = self.ready_queue.pop_front() else {
                break;
            };
            if !self.ready_set.remove(&pid) {
                continue;
            }
            if let Some(proc) = self.processes.get_mut(&pid) {
                if proc.is_foreground {
                    self.ready_set.insert(pid);
                    continue;
                }
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

    fn request_yield(&mut self) {
        self.yield_requested = true;
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
        self.advance_ticks(1);
    }

    fn advance_ticks(&mut self, ticks: u64) {
        if ticks == 0 {
            return;
        }
        self.tick = self.tick.saturating_add(ticks);
        // Pop all due entries from the wakeup heap; stale ones (process already woken/terminated/re-stated early) are dropped outright.
        while let Some(Reverse((until_tick, pid))) = self.wakeup_heap.peek() {
            let (until_tick, pid) = (*until_tick, *pid);
            if until_tick > self.tick {
                break;
            }
            self.wakeup_heap.pop();
            let is_sleeping = matches!(
                self.processes.get(&pid).map(|p| &p.state),
                Some(ProcessState::Sleeping { until_tick: t }) if *t == until_tick
            );
            if is_sleeping {
                if let Some(proc) = self.processes.get_mut(&pid) {
                    proc.state = ProcessState::Ready;
                    proc.mailbox.push_back(format!(
                        "Sleep finished at scheduler tick {}.",
                        self.tick
                    ));
                }
                self.enqueue_ready(pid);
                continue;
            }
            let is_timeout = matches!(
                self.processes.get(&pid).map(|p| &p.state),
                Some(ProcessState::Waiting {
                    reason: WaitReason::Events {
                        timeout_tick: Some(t),
                        ..
                    }
                }) if *t == until_tick
            );
            if is_timeout {
                if let Some(proc) = self.processes.get_mut(&pid) {
                    proc.state = ProcessState::Ready;
                    proc.mailbox.push_back(format!(
                        "Event wait timeout reached at scheduler tick {}.",
                        self.tick
                    ));
                }
                self.enqueue_ready(pid);
            }
            // All other cases are stale entries, already popped and dropped.
        }
    }

    fn current_tick(&self) -> u64 {
        self.tick
    }

    fn next_wakeup_tick(&self) -> Option<u64> {
        // The heap top is the earliest deadline; stale tops are only cleaned up during advance_ticks,
        // so returning that tick merely wakes the caller early and never misses an earlier wakeup.
        self.wakeup_heap.peek().map(|Reverse((t, _))| *t)
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

    fn pop_foreground_ready(&mut self) -> Option<Process> {
        let fg_pid = self
            .processes
            .iter()
            .find(|(pid, proc)| proc.is_foreground && self.ready_set.contains(pid))
            .map(|(pid, _)| *pid)?;
        self.ready_set.remove(&fg_pid);
        if let Some(proc) = self.processes.get_mut(&fg_pid) {
            proc.state = ProcessState::Running;
            self.current_pid = Some(fg_pid);
            self.yield_requested = false;
            Some(proc.clone())
        } else {
            None
        }
    }

    fn wake_process(&mut self, pid: u64, message: String) -> bool {
        let should_enqueue = if let Some(proc) = self.processes.get_mut(&pid) {
            if matches!(proc.state, ProcessState::Terminated) {
                return false;
            }
            proc.mailbox.push_back(message);
            if matches!(
                proc.state,
                ProcessState::Waiting { .. } | ProcessState::Sleeping { .. }
            ) {
                proc.state = ProcessState::Ready;
                true
            } else {
                false
            }
        } else {
            return false;
        };
        if should_enqueue {
            self.enqueue_ready(pid);
        }
        should_enqueue
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
                    "[EVENT_WAKE]\nReason: event wait condition satisfied.\nCompleted event ids: {}\nRecommended next actions:\n1. If you were parked by task_wait, re-call task_wait with the same task_ids and wait_policy to collect subagent results.\n2. If these events came from async tool work, use tool_status or tool_wait to collect results.\n3. Inspect the event-producing subsystem for fresh state when unsure.\n4. Cancel low-value still-running tool branches when appropriate.\n5. If enough results are already available, continue reasoning immediately.",
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
                    proc.max_restarts,
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
            max_restarts,
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
                    max_restarts,
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
