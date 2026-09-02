use super::*;

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
        // Record one reference in the table for the event_id bound to the channel lifecycle.
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
            // The old event is no longer the channel's source; after the conversion the new event owns
            // the liveness checks for subsequent waiters.
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
            // The old event is no longer the channel's source; after the conversion the new event owns
            // the liveness checks for subsequent waiters.
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
