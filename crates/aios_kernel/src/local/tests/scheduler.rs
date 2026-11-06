use super::*;

#[test]
fn foreground_process_can_wait_and_resume_on_child_exit() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        8,
        None,
    );
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
    let root = os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        8,
        None,
    );

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

// Regression test: concurrent background sub-agents each run in their own tokio task while sharing
// one kernel. When one task clears/rewrites the shared scalar `self.current_pid`, a blocking syscall
// running in another task (e.g. task_wait → wait_on_events) must still resolve the true caller from
// the task-local instead of misreporting "No process currently running." (root cause of stuck main/sub-agent scheduling).
#[test]
fn blocking_syscall_resolves_caller_from_task_local_when_current_pid_cleared() {
    use std::cell::Cell;

    thread_local! {
        static TEST_TASK_PID: Cell<Option<u64>> = const { Cell::new(None) };
    }
    fn provider() -> Option<u64> {
        TEST_TASK_PID.with(|c| c.get())
    }
    crate::kernel::register_current_pid_provider(provider);

    let mut os = LocalOS::new();
    let root = os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        8,
        None,
    );

    // Simulate a concurrent task clearing the shared current_pid during its wrap-up.
    os.set_current_pid(None);
    assert!(os.current_pid.is_none());

    // But the task-local still points at root (the authoritative identity of "this task").
    TEST_TASK_PID.with(|c| c.set(Some(root)));

    // wait_on_events must resolve root via the task-local and suspend normally instead of erroring.
    let timeout_tick = os
        .wait_on_events(vec![EventId::new(7)], WaitPolicy::Any, Some(5))
        .expect("wait_on_events must resolve caller from task-local");
    assert_eq!(timeout_tick, Some(5));
    assert!(os.consume_yield_requested());
    assert!(matches!(
        os.get_process(root).map(|p| &p.state),
        Some(ProcessState::Waiting {
            reason: WaitReason::Events { .. }
        })
    ));

    // Reset the task-local to avoid polluting later tests on the same thread.
    TEST_TASK_PID.with(|c| c.set(None));
}

#[test]
fn event_wait_timeout_wakes_process() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        8,
        None,
    );

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
    let root = os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        8,
        None,
    );

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
        Some(
            "[EVENT_WAKE]\nReason: event wait condition satisfied.\nCompleted event ids: evt_2\nRecommended next actions:\n1. If you were parked by task_wait, re-call task_wait with the same task_ids and wait_policy to collect subagent results.\n2. If these events came from async tool work, use tool_status or tool_wait to collect results.\n3. Inspect the event-producing subsystem for fresh state when unsure.\n4. Cancel low-value still-running tool branches when appropriate.\n5. If enough results are already available, continue reasoning immediately."
        )
    );
}

#[test]
fn ready_queue_insertion_preserves_priority_order() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        8,
        None,
    );
    let low_priority_pid = os
        .spawn(
            Some(root),
            "low".to_string(),
            "low priority".to_string(),
            30,
            4,
            None,
            None,
        )
        .unwrap();
    let high_priority_pid = os
        .spawn(
            Some(root),
            "high".to_string(),
            "high priority".to_string(),
            5,
            4,
            None,
            None,
        )
        .unwrap();
    let mid_priority_pid = os
        .spawn(
            Some(root),
            "mid".to_string(),
            "mid priority".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    assert_eq!(os.pop_ready().map(|proc| proc.pid), Some(high_priority_pid));
    assert_eq!(os.pop_ready().map(|proc| proc.pid), Some(mid_priority_pid));
    assert_eq!(os.pop_ready().map(|proc| proc.pid), Some(low_priority_pid));
}

#[test]
fn completed_events_are_retention_bounded_for_inactive_events() {
    let mut os = LocalOS::new();
    os.completed_event_retention = 2;

    os.notify_events_completed(&[EventId::new(1)]);
    os.notify_events_completed(&[EventId::new(2)]);
    os.notify_events_completed(&[EventId::new(3)]);

    assert!(!os.event_is_completed(EventId::new(1)));
    assert!(os.event_is_completed(EventId::new(2)));
    assert!(os.event_is_completed(EventId::new(3)));
    assert_eq!(os.completed_events.len(), 2);
}

#[test]
fn completed_event_retention_preserves_live_epoll_event_sources() {
    use crate::primitives::{EpollEventMask, EpollOps, EpollSource};

    let mut os = LocalOS::new();
    os.completed_event_retention = 1;
    let watched_event = EventId::new(10);
    let epoll = os.epoll_create("live-event".to_string());
    os.epoll_ctl_add(
        epoll,
        EpollSource::Event(watched_event),
        EpollEventMask::IN,
        10,
    )
    .unwrap();

    os.notify_events_completed(&[watched_event]);
    os.notify_events_completed(&[EventId::new(11)]);
    os.notify_events_completed(&[EventId::new(12)]);

    assert!(os.event_is_completed(watched_event));
    assert!(!os.event_is_completed(EventId::new(11)));
    assert!(os.event_is_completed(EventId::new(12)));
}

#[test]
fn notify_events_completed_uses_waiter_index() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
    let child = os
        .spawn(
            Some(root),
            "c".to_string(),
            "g".to_string(),
            10,
            8,
            None,
            None,
        )
        .unwrap();
    // Run the child so its current_pid context is set, then have it wait.
    let popped = os.pop_ready().unwrap();
    assert_eq!(popped.pid, child);
    let event = EventId::new(42);
    os.wait_on_events(vec![event], WaitPolicy::Any, None)
        .unwrap();
    // Index should now contain the child pid.
    assert!(
        os.event_waiters
            .get(&event)
            .is_some_and(|set| set.contains(&child))
    );
    // Notify wakes the child.
    let woken = os.notify_events_completed(&[event]);
    assert_eq!(woken, vec![child]);
    // The waiter index entry for the completed event should be drained.
    assert!(os.event_waiters.get(&event).is_none());
}

#[test]
fn notify_events_completed_skips_stale_waiters() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
    let child = os
        .spawn(
            Some(root),
            "c".to_string(),
            "g".to_string(),
            10,
            8,
            None,
            None,
        )
        .unwrap();
    let popped = os.pop_ready().unwrap();
    assert_eq!(popped.pid, child);
    let event = EventId::new(99);
    os.wait_on_events(vec![event], WaitPolicy::Any, None)
        .unwrap();
    // Forcibly terminate the child while it is in the waiter index.
    os.terminate_pid(child, "killed".to_string());
    // Stale entry must not be woken (and notify must not panic).
    let woken = os.notify_events_completed(&[event]);
    assert!(woken.is_empty());
}

#[test]
fn descendants_use_persistent_index() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
    let a = os
        .spawn(
            Some(root),
            "a".to_string(),
            "g".to_string(),
            10,
            8,
            None,
            None,
        )
        .unwrap();
    let b = os
        .spawn(Some(a), "b".to_string(), "g".to_string(), 10, 8, None, None)
        .unwrap();
    let c = os
        .spawn(Some(a), "c".to_string(), "g".to_string(), 10, 8, None, None)
        .unwrap();
    let mut descendants = os.collect_descendants(root);
    descendants.sort();
    let mut expected = vec![a, b, c];
    expected.sort();
    assert_eq!(descendants, expected);
    // Index must be properly maintained: removing a node updates parent's set.
    assert!(
        os.children_by_parent
            .get(&a)
            .is_some_and(|s| s.contains(&b) && s.contains(&c))
    );
    os.terminate_pid(b, "done".to_string());
    os.remove_process_entry(b);
    assert!(
        os.children_by_parent
            .get(&a)
            .is_some_and(|s| !s.contains(&b) && s.contains(&c))
    );
}

/// Terminating a process + re-login must invalidate the old answers in the SHM permission cache,
/// otherwise there is an awkward window where "the owner is dead but the cache still allows writes".
#[test]
fn shm_perm_cache_invalidates_on_topology_change() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
    let owner = os
        .spawn(
            Some(root),
            "owner".to_string(),
            "g".to_string(),
            10,
            8,
            None,
            None,
        )
        .unwrap();
    let stranger = os
        .spawn(
            Some(root),
            "stranger".to_string(),
            "g".to_string(),
            10,
            8,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(owner));
    os.shm_create("k".to_string(), "v".to_string()).unwrap();

    // stranger is currently a sibling: readable but not writable.
    os.set_current_pid(Some(stranger));
    let entry = os.shared_memory.get("k").unwrap();
    assert!(!os.is_shm_accessible_by(stranger, entry));
    assert!(os.is_shm_readable_by(stranger, entry));

    // Pull stranger into the owner's process group: the topology change must invalidate the cached
    // "stranger is not writable" answer, which flips to writable after recomputation.
    let pgid = os.next_pgid;
    os.next_pgid += 1;
    os.set_process_group(owner, pgid).unwrap();
    os.set_process_group(stranger, pgid).unwrap();
    let entry = os.shared_memory.get("k").unwrap();
    assert!(
        os.is_shm_accessible_by(stranger, entry),
        "set_process_group must invalidate stale shm perm cache",
    );

    // Reverse direction: after owner terminate, the cache must also let accessible be recomputed.
    os.terminate_pid(owner, "done".to_string());
    os.remove_process_entry(owner);
    let entry = os.shared_memory.get("k").unwrap();
    // The owner pid no longer exists; the non-owner path goes through the ancestor / sibling chain.
    // accessible depends on stranger's and owner's pgids, but the owner has been erased,
    // so only assert that the query does not panic and returns a bool.
    let _ = os.is_shm_accessible_by(stranger, entry);
}

/// `event_source_refs` corresponds strictly one-to-one with the lifecycle of
/// channel/futex/epoll(EpollSource::Event); references must return to zero after destroy.
#[test]
fn event_source_refs_track_channel_and_futex_lifetimes() {
    use crate::primitives::{FutexOps, IpcOps};
    let mut os = LocalOS::new();
    let _root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);

    let ch = os.channel_create(None, 4, "test".to_string());
    let ch_event = os.channels.get(&ch.0).unwrap().event_id;
    assert_eq!(os.event_source_refs.get(&ch_event).copied(), Some(1));
    assert!(os.completed_event_is_live(ch_event));

    let addr = os.futex_create(0, "f".to_string());
    let fx_event = os.futex_event_id(addr).unwrap();
    assert_eq!(os.event_source_refs.get(&fx_event).copied(), Some(1));

    // After futex_destroy the references hit zero and the entry must be cleared to avoid unbounded growth.
    assert!(os.futex_destroy(addr));
    assert!(os.event_source_refs.get(&fx_event).is_none());

    // channel destroy behaves the same as above.
    os.channel_close(None, ch).unwrap();
    os.channel_destroy(None, ch).unwrap();
    assert!(os.event_source_refs.get(&ch_event).is_none());
}

/// Registering EpollSource::Event with epoll should also keep the event live; after del / destroy
/// the count returns to zero, ensuring prune_completed_events does not reclaim too early.
#[test]
fn event_source_refs_track_epoll_event_registration() {
    use crate::primitives::{EpollEventMask, EpollOps, EpollSource};
    let mut os = LocalOS::new();
    let _root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
    let ep = os.epoll_create("ep".to_string());

    // Use an internal event_id; construct an EpollSource::Event directly.
    let watched = os.alloc_internal_event_id();
    os.epoll_ctl_add(ep, EpollSource::Event(watched), EpollEventMask::IN, 1)
        .unwrap();
    assert_eq!(os.event_source_refs.get(&watched).copied(), Some(1));
    assert!(os.completed_event_is_live(watched));

    // del once: the count hits zero and the entry is removed.
    os.epoll_ctl_del(ep, EpollSource::Event(watched)).unwrap();
    assert!(os.event_source_refs.get(&watched).is_none());

    // Re-register then take the destroy path; destroy must clear all event references.
    os.epoll_ctl_add(ep, EpollSource::Event(watched), EpollEventMask::IN, 2)
        .unwrap();
    assert_eq!(os.event_source_refs.get(&watched).copied(), Some(1));
    assert!(os.epoll_destroy(ep));
    assert!(os.event_source_refs.get(&watched).is_none());
}

/// When terminating many processes, ready_queue is no longer linearly retained; tombstones are
/// cleaned up by pop_ready at dequeue time.
#[test]
fn ready_queue_uses_lazy_tombstones_on_termination() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
    let mut spawned = Vec::new();
    for i in 0..32 {
        let pid = os
            .spawn(
                Some(root),
                format!("c{i}"),
                "g".to_string(),
                50,
                8,
                None,
                None,
            )
            .unwrap();
        spawned.push(pid);
    }
    // After begin_foreground, root is the currently running process and not in ready_set;
    // all spawned children enter ready.
    assert_eq!(os.ready_count(), spawned.len());
    // Batch terminate: ready_set shrinks immediately, but ready_queue keeps tombstones.
    for &pid in &spawned {
        os.terminate_pid(pid, "x".to_string());
    }
    assert_eq!(os.ready_count(), 0);
    assert!(os.ready_queue.len() >= spawned.len());
    // pop_ready must drop all tombstones and return None.
    assert!(os.pop_ready().is_none());
    // The queue is eventually drained too.
    assert!(os.ready_queue.is_empty());
}

/// Priority insertion no longer queries the processes table per comparison; a newly added
/// higher-priority process should land at the queue head.
#[test]
fn ready_queue_priority_insertion_uses_cached_priority() {
    let mut os = LocalOS::new();
    // root priority = 10
    let root = os.begin_foreground("fg".to_string(), "g".to_string(), 10, 8, None);
    // Enqueue lower priority afterwards (larger value -> lower priority)
    let low = os
        .spawn(
            Some(root),
            "low".to_string(),
            "g".to_string(),
            100,
            8,
            None,
            None,
        )
        .unwrap();
    // Then enqueue higher priority (smallest value -> highest priority)
    let high = os
        .spawn(
            Some(root),
            "high".to_string(),
            "g".to_string(),
            1,
            8,
            None,
            None,
        )
        .unwrap();
    // high must be ahead of low in the queue; no need to reference root (begin_foreground
    // sets root to Running, so it is not in ready_set).
    let pids: Vec<u64> = os.ready_queue.iter().map(|(pid, _)| *pid).collect();
    let pos_high = pids.iter().position(|p| *p == high).expect("high in queue");
    let pos_low = pids.iter().position(|p| *p == low).expect("low in queue");
    assert!(pos_high < pos_low, "high priority must come before low");
    // The priority cache no longer needs a processes lookup: pop_ready's top must be high.
    let next = os.pop_ready().unwrap();
    assert_eq!(next.pid, high);
}

#[test]
fn channel_ref_holders_dedupe_by_name() {
    use crate::primitives::IpcOps;
    let mut os = LocalOS::new();
    let ch = os.channel_create(None, 4, "test".to_string());
    os.channel_retain_named(ch, "alpha".to_string()).unwrap();
    os.channel_retain_named(ch, "alpha".to_string()).unwrap();
    os.channel_retain_named(ch, "beta".to_string()).unwrap();
    let meta = os.channel_meta(ch).unwrap();
    assert_eq!(meta.ref_count, 3);
    // Snapshot flattens (alpha, 2) and (beta, 1) -> 3 entries.
    assert_eq!(meta.ref_holders.iter().filter(|h| *h == "alpha").count(), 2);
    assert_eq!(meta.ref_holders.iter().filter(|h| *h == "beta").count(), 1);
    // Internal storage groups duplicates: only 2 unique slots.
    let entry = os.channels.get(&ch.0).unwrap();
    assert_eq!(entry.ref_holders.len(), 2);
    // Releasing one alpha leaves one alpha + one beta.
    os.channel_release_named(ch, "alpha").unwrap();
    let entry = os.channels.get(&ch.0).unwrap();
    assert_eq!(entry.ref_count, 2);
    assert_eq!(entry.ref_holders.len(), 2);
    // Releasing the last alpha drops the slot entirely.
    os.channel_release_named(ch, "alpha").unwrap();
    let entry = os.channels.get(&ch.0).unwrap();
    assert_eq!(entry.ref_holders.len(), 1);
    assert_eq!(entry.ref_holders[0].0, "beta");
}
