use super::*;

impl FutexOps for LocalOS {
    fn futex_create(&mut self, initial: u64, label: String) -> FutexAddr {
        let id = self.next_futex_id;
        self.next_futex_id += 1;
        let event_id = self.alloc_internal_event_id();
        let owner = self.effective_current_pid();
        self.futexes.insert(id, FutexState::new(initial, event_id));
        // Charge the event_id bound to the futex lifecycle into the resource count.
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
