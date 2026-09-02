use super::*;

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
        // Release the reference counts of all EpollSource::Event entries in this epoll.
        for registration in entry.registrations.values() {
            if let EpollSource::Event(event_id) = registration.snapshot.source {
                self.dec_event_source_ref(event_id);
            }
        }
        true
    }
}
