use super::*;

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
            // If already cancelled, keep the exit state Cancelled (cancel is the more salient semantics).
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
