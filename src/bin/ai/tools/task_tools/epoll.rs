use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EpollWaitManyOutcome {
    pub(crate) ready_sources: Vec<WaitManySource>,
    pub(crate) pending_sources: Vec<WaitManySource>,
    pub(crate) event_ids: Vec<EventId>,
    pub(crate) suspended: bool,
    pub(crate) timeout_tick: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WaitManySource {
    Channel(u64),
    Event(EventId),
    Futex { addr: FutexAddr, expected: u64 },
}

pub(crate) fn wait_sources_for_channel_and_futex(
    os: &mut dyn Kernel,
    channel_id: u64,
    completion_futex_addr: Option<FutexAddr>,
) -> Result<Vec<WaitManySource>, String> {
    let mut sources = vec![WaitManySource::Channel(channel_id)];
    let channel_event = os
        .channel_event_id(ChannelId(channel_id))
        .ok_or_else(|| format!("Channel {} has no waitable event id.", channel_id))?;
    sources.push(WaitManySource::Event(channel_event));
    if let Some(addr) = completion_futex_addr {
        sources.push(WaitManySource::Futex { addr, expected: 0 });
    }
    Ok(sources)
}

pub(crate) fn append_current_process_cancel_source(
    os: &mut dyn Kernel,
    sources: &mut Vec<WaitManySource>,
) -> Result<(), String> {
    if let Some(addr) = current_process_tool_cancel_futex(os)? {
        sources.push(WaitManySource::Futex { addr, expected: 0 });
    }
    Ok(())
}

impl WaitManySource {
    fn epoll_source(self) -> EpollSource {
        match self {
            Self::Channel(channel_id) => EpollSource::Channel(ChannelId(channel_id)),
            Self::Event(event_id) => EpollSource::Event(event_id),
            Self::Futex { addr, expected } => EpollSource::Futex { addr, expected },
        }
    }

    fn epoll_mask(self) -> EpollEventMask {
        match self {
            Self::Channel(_) => EpollEventMask::IN | EpollEventMask::HUP | EpollEventMask::ERR,
            Self::Event(_) | Self::Futex { .. } => EpollEventMask::IN | EpollEventMask::ERR,
        }
    }
}

pub(super) fn wait_many_snapshot(
    os: &mut dyn Kernel,
    sources: &[WaitManySource],
) -> Result<(Vec<WaitManySource>, Vec<WaitManySource>, Vec<EventId>), String> {
    let mut ready = Vec::new();
    let mut pending = Vec::new();
    let mut event_ids = Vec::new();
    for source in sources {
        let event_id = match *source {
            WaitManySource::Channel(channel_id) => {
                let channel = ChannelId(channel_id);
                let meta = os
                    .channel_meta(channel)
                    .ok_or_else(|| format!("Channel {} no longer exists.", channel_id))?;
                if meta.queued_len > 0 || meta.closed {
                    ready.push(*source);
                    continue;
                }
                os.channel_event_id(channel)
                    .ok_or_else(|| format!("Channel {} has no waitable event id.", channel_id))?
            }
            WaitManySource::Event(event_id) => {
                if os.event_is_completed(event_id) {
                    ready.push(*source);
                    continue;
                }
                event_id
            }
            WaitManySource::Futex { addr, expected } => {
                if os.futex_try_wait(addr, expected).is_some() {
                    ready.push(*source);
                    continue;
                }
                os.futex_event_id(addr)
                    .ok_or_else(|| format!("Futex {} has no waitable event id.", addr.raw()))?
            }
        };
        pending.push(*source);
        event_ids.push(event_id);
    }
    Ok((ready, pending, event_ids))
}

/// Combines the kernel's epoll / channel / futex / event primitives at the agent layer to
/// implement a "wait for any of several sources to complete" semantic across **multiple wait
/// source kinds**, primarily serving the `task_wait` tool.
///
/// **Design positioning**: this function does *not* re-implement the kernel's wait primitives; it
/// assembles several low-level APIs (`epoll_create` / `epoll_ctl` / `epoll_wait` /
/// `wait_on_events`) to the agent's business semantics:
/// 1. Build a short-lived epoll set for channel/futex-style wait sources, then `epoll_wait` for
///    the ready set;
/// 2. For event-style wait sources, call `wait_on_events` directly;
/// 3. Normalize both kinds of results into `EpollWaitManyOutcome`.
///
/// **Future lowering suggestion**: once the kernel gains native syscall support for
/// `Vec<WaitManySource>` (similar to a hybrid of epoll_pwait2 + EVENTFD), this function can become
/// a thin wrapper around a single syscall. Until that migration, this function keeps the current
/// multi-step composite implementation; any behavior change **must keep task_wait regression-free
/// in the following scenarios**:
/// - all sources ready: return immediately (epoll_wait is not called);
/// - all sources pending: decide whether to actually suspend according to `wait_policy`;
/// - mixed ready + pending: return only the ready set, without adding extra blocking.
pub(crate) fn epoll_wait_many(
    os: &mut dyn Kernel,
    label: &str,
    sources: &[WaitManySource],
    wait_policy: WaitPolicy,
    timeout_ticks: Option<u64>,
) -> Result<EpollWaitManyOutcome, String> {
    if sources.is_empty() {
        return Ok(EpollWaitManyOutcome {
            ready_sources: Vec::new(),
            pending_sources: Vec::new(),
            event_ids: Vec::new(),
            suspended: false,
            timeout_tick: None,
        });
    }

    let epoll = os.epoll_create(label.to_string());
    let result = (|| {
        for (index, source) in sources.iter().enumerate() {
            os.epoll_ctl_add(
                epoll,
                source.epoll_source(),
                source.epoll_mask(),
                index as u64,
            )?;
        }

        let (ready_sources, pending_sources, event_ids) = wait_many_snapshot(os, sources)?;
        let satisfied = match wait_policy {
            WaitPolicy::Any => !ready_sources.is_empty(),
            WaitPolicy::All => pending_sources.is_empty(),
        };
        if satisfied {
            return Ok(EpollWaitManyOutcome {
                ready_sources,
                pending_sources,
                event_ids,
                suspended: false,
                timeout_tick: None,
            });
        }

        match wait_policy {
            WaitPolicy::Any => match os.epoll_wait(epoll, sources.len(), timeout_ticks)? {
                EpollWaitResult::Ready(_) => {
                    let (ready_sources, pending_sources, event_ids) =
                        wait_many_snapshot(os, sources)?;
                    Ok(EpollWaitManyOutcome {
                        ready_sources,
                        pending_sources,
                        event_ids,
                        suspended: false,
                        timeout_tick: None,
                    })
                }
                EpollWaitResult::Suspended { timeout_tick } => {
                    // epoll_wait internally consumed the yield_requested flag to decide whether it
                    // suspended; it must be re-set here, otherwise the turn-loop's
                    // consume_yield_requested() reads false, control is never returned to the
                    // scheduler, and a ready subagent is never dispatched.
                    os.request_yield();
                    Ok(EpollWaitManyOutcome {
                        ready_sources,
                        pending_sources,
                        event_ids,
                        suspended: true,
                        timeout_tick,
                    })
                }
            },
            WaitPolicy::All => {
                let wake_tick =
                    os.wait_on_events(event_ids.clone(), WaitPolicy::All, timeout_ticks)?;
                let suspended = os.consume_yield_requested() || wake_tick.is_some();
                if suspended {
                    // Same as above: this branch probes suspension via consume_yield_requested(),
                    // which clears the yield intent. Once suspension is confirmed, re-set the flag
                    // so the turn-loop can notice it and hand control back to the scheduler.
                    os.request_yield();
                }
                let (ready_sources, pending_sources, refreshed_event_ids) =
                    wait_many_snapshot(os, sources)?;
                Ok(EpollWaitManyOutcome {
                    ready_sources,
                    pending_sources,
                    event_ids: if suspended {
                        event_ids
                    } else {
                        refreshed_event_ids
                    },
                    suspended,
                    timeout_tick: wake_tick,
                })
            }
        }
    })();
    let _ = os.epoll_destroy(epoll);
    result
}

pub(crate) fn epoll_wait_many_channels(
    os: &mut dyn Kernel,
    label: &str,
    channel_ids: &[u64],
    wait_policy: WaitPolicy,
    timeout_ticks: Option<u64>,
) -> Result<EpollWaitManyOutcome, String> {
    let sources = channel_ids
        .iter()
        .copied()
        .map(WaitManySource::Channel)
        .collect::<Vec<_>>();
    epoll_wait_many(os, label, &sources, wait_policy, timeout_ticks)
}
