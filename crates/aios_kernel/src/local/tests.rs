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

#[test]
fn foreground_process_enables_env_access() {
    let mut os = LocalOS::new();
    os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        8,
        None,
    );
    os.set_env("scope".to_string(), "root".to_string()).unwrap();
    assert_eq!(os.get_env("scope").as_deref(), Some("root"));
}

#[test]
fn sleeping_process_wakes_after_tick_advance() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        8,
        None,
    );
    let wake_tick = os.sleep_current(2).unwrap();
    assert_eq!(wake_tick, 2);
    assert_eq!(os.current_tick(), 0);
    assert_eq!(os.next_wakeup_tick(), Some(2));
    assert!(os.consume_yield_requested());
    assert!(matches!(
        os.get_process(root).map(|p| &p.state),
        Some(ProcessState::Sleeping { until_tick }) if *until_tick == 2
    ));

    os.advance_ticks(2);
    let resumed = os.pop_ready().unwrap();
    assert_eq!(resumed.pid, root);
    assert_eq!(os.current_tick(), 2);
    assert_eq!(os.next_wakeup_tick(), None);
}

#[test]
fn child_can_be_spawned_with_reduced_capabilities() {
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
fn removing_parent_reparents_live_children_and_collects_unreapable_zombies() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        8,
        None,
    );
    let live_child = os
        .spawn(
            Some(root),
            "live".to_string(),
            "live goal".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let dead_child = os
        .spawn(
            Some(root),
            "dead".to_string(),
            "dead goal".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(root));
    os.kill_process(dead_child, "done".to_string()).unwrap();
    os.terminate_pid(root, "root exited".to_string());
    assert!(os.drop_terminated(root));

    let live_proc = os.get_process(live_child).unwrap();
    assert_eq!(live_proc.parent_pid, None);
    assert!(
        live_proc
            .mailbox
            .iter()
            .any(|msg| msg.contains("now orphaned"))
    );
    assert!(os.get_process(dead_child).is_none());
}

#[test]
fn kill_cascades_to_grandchildren() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground(
        "foreground".to_string(),
        "root goal".to_string(),
        10,
        usize::MAX,
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
    let grandchild = os
        .spawn(
            Some(child),
            "grandchild".to_string(),
            "gc goal".to_string(),
            30,
            2,
            None,
            None,
        )
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
    assert!(
        os.get_process(grandchild)
            .unwrap()
            .result
            .as_ref()
            .unwrap()
            .contains("cascade")
    );
}

#[test]
fn foreground_process_has_is_foreground_flag() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    assert!(os.get_process(root).unwrap().is_foreground);

    let child = os
        .spawn(
            Some(root),
            "bg".to_string(),
            "bg goal".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    assert!(!os.get_process(child).unwrap().is_foreground);
}

#[test]
fn sigstop_stops_and_sigcont_resumes_process() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child = os
        .spawn(
            Some(root),
            "worker".to_string(),
            "work".to_string(),
            20,
            4,
            None,
            None,
        )
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
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child = os
        .spawn(
            Some(root),
            "worker".to_string(),
            "work".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let grandchild = os
        .spawn(
            Some(child),
            "gc".to_string(),
            "gc work".to_string(),
            30,
            2,
            None,
            None,
        )
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
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child = os
        .spawn(
            Some(root),
            "worker".to_string(),
            "work".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.signal_process(child, Signal::SigTerm).unwrap();
    let child_proc = os.get_process(child).unwrap();
    assert!(child_proc.pending_signals.contains(&Signal::SigTerm));
}

#[test]
fn sigcancel_is_consumed_without_terminating_process() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);

    os.signal_process(root, Signal::SigCancel).unwrap();
    assert!(
        os.get_process(root)
            .unwrap()
            .pending_signals
            .contains(&Signal::SigCancel)
    );

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
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child = os
        .spawn(
            Some(root),
            "worker".to_string(),
            "work".to_string(),
            20,
            4,
            None,
            None,
        )
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
    let child1 = os
        .spawn(
            Some(root),
            "c1".to_string(),
            "g1".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let child2 = os
        .spawn(
            Some(root),
            "c2".to_string(),
            "g2".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_process_group(child1, 100).unwrap();
    os.set_process_group(child2, 100).unwrap();

    assert_eq!(os.get_process(child1).unwrap().process_group, Some(100));
    assert_eq!(os.get_process(child2).unwrap().process_group, Some(100));

    let count = os.signal_process_group(100, Signal::SigStop).unwrap();
    assert_eq!(count, 2);
    assert!(matches!(
        os.get_process(child1).unwrap().state,
        ProcessState::Stopped
    ));
    assert!(matches!(
        os.get_process(child2).unwrap().state,
        ProcessState::Stopped
    ));
}

#[test]
fn shared_memory_crud_operations() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);

    os.shm_create("config".to_string(), "value1".to_string())
        .unwrap();
    assert_eq!(os.shm_read("config"), Ok("value1".to_string()));
    assert_eq!(
        os.shared_memory.get("config").map(|e| e.owner_pid),
        Some(root)
    );

    os.shm_write("config".to_string(), "value2".to_string())
        .unwrap();
    assert_eq!(os.shm_read("config"), Ok("value2".to_string()));

    os.shm_delete("config").unwrap();
    assert_eq!(os.shm_read("config"), Err(ShmReadError::NotFound));
    assert!(os.shared_memory.get("config").is_none());

    assert!(
        os.shm_create("config".to_string(), "value3".to_string())
            .is_ok()
    );
    assert!(
        os.shm_create("config".to_string(), "value4".to_string())
            .is_err()
    );
}

#[test]
fn working_dir_inherits_to_children() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    os.set_working_dir(std::path::PathBuf::from("/tmp/work"))
        .unwrap();

    let child = os
        .spawn(
            Some(root),
            "child".to_string(),
            "goal".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    assert_eq!(
        os.get_process(child).unwrap().working_dir,
        Some(std::path::PathBuf::from("/tmp/work"))
    );
}

#[test]
fn daemon_auto_restarts_on_termination() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let daemon_pid = os
        .spawn_daemon(
            Some(root),
            "watcher".to_string(),
            "watch files".to_string(),
            20,
            4,
            2,
        )
        .unwrap();

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
    assert!(
        os.get_process(new_pid)
            .unwrap()
            .goal
            .contains("daemon restart #1")
    );
}

#[test]
fn daemon_respects_max_restarts() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let daemon_pid = os
        .spawn_daemon(
            Some(root),
            "watcher".to_string(),
            "watch".to_string(),
            20,
            4,
            1,
        )
        .unwrap();

    os.terminate_pid(daemon_pid, "crashed".to_string());
    let restarted1 = os.check_daemon_restart();
    assert_eq!(restarted1.len(), 1);

    os.terminate_pid(restarted1[0], "crashed again".to_string());
    let restarted2 = os.check_daemon_restart();
    assert!(restarted2.is_empty());
}

#[test]
fn daemon_restarts_up_to_max_restarts_then_stops() {
    // Regression: a restarted process must keep its original max_restarts, otherwise it only restarts once.
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let daemon_pid = os
        .spawn_daemon(
            Some(root),
            "watcher".to_string(),
            "watch".to_string(),
            20,
            4,
            3,
        )
        .unwrap();

    let mut current = daemon_pid;
    for expected_count in 1..=3 {
        os.terminate_pid(current, "crashed".to_string());
        let restarted = os.check_daemon_restart();
        assert_eq!(restarted.len(), 1, "restart #{expected_count} should occur");
        current = restarted[0];
        let proc = os.get_process(current).unwrap();
        assert_eq!(proc.restart_count, expected_count);
        assert_eq!(proc.max_restarts, 3);
    }

    // After the 4th termination the limit is reached; no more restarts.
    os.terminate_pid(current, "crashed".to_string());
    assert!(os.check_daemon_restart().is_empty());
}

#[test]
fn ipc_is_rejected_between_unrelated_processes() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child_a = os
        .spawn(
            Some(root),
            "a".to_string(),
            "goal a".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let child_b = os
        .spawn(
            Some(root),
            "b".to_string(),
            "goal b".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(child_a));
    let result = os.send_ipc(child_b, "hello".to_string());
    assert!(result.is_ok());

    let unrelated_root =
        os.begin_foreground("fg2".to_string(), "goal2".to_string(), 10, usize::MAX, None);
    let orphan = os
        .spawn(
            Some(unrelated_root),
            "orphan".to_string(),
            "goal orphan".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(orphan));
    let result = os.send_ipc(child_a, "intrusion".to_string());
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Permission denied"));
}

#[test]
fn ipc_allowed_within_same_process_group() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child_a = os
        .spawn(
            Some(root),
            "a".to_string(),
            "goal a".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let child_b = os
        .spawn(
            Some(root),
            "b".to_string(),
            "goal b".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_process_group(child_a, 42).unwrap();
    os.set_process_group(child_b, 42).unwrap();

    os.set_current_pid(Some(child_a));
    let result = os.send_ipc(child_b, "hello group".to_string());
    assert!(result.is_ok());

    let outsider = os
        .spawn(
            Some(root),
            "outsider".to_string(),
            "goal out".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    os.set_current_pid(Some(outsider));
    let result = os.send_ipc(child_a, "intrusion".to_string());
    assert!(result.is_err());
}

#[test]
fn kill_process_rejected_for_non_descendant() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child_a = os
        .spawn(
            Some(root),
            "a".to_string(),
            "goal a".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let child_b = os
        .spawn(
            Some(root),
            "b".to_string(),
            "goal b".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(child_a));
    let result = os.kill_process(child_b, "sibling kill".to_string());
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("outside its scope"));
}

#[test]
fn orphaned_grandchild_reattached_to_root_and_killable() {
    // Regression for the `kill_process` "outside its scope" failure: when a
    // subagent (child) exits and its kernel entry is dropped, its live
    // descendants used to be orphaned (`parent_pid = None`), which made them
    // unmanageable by the foreground model even though they kept running and
    // were listed by `list_processes`. They must be re-attached to the root
    // process so the whole tree stays inside the foreground's scope.
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child = os
        .spawn(
            Some(root),
            "subagent".to_string(),
            "goal sub".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let grandchild = os
        .spawn(
            Some(child),
            "grandchild".to_string(),
            "goal gc".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    // Simulate subagent completion: terminate + drop its kernel entry, the same
    // path `terminate_and_cleanup` in the `a` binary takes after a turn ends.
    os.terminate_pid(child, "done".to_string());
    assert!(os.drop_terminated(child), "drop_terminated should succeed");

    // The grandchild must now be a descendant of the root again, so the
    // foreground model (full capabilities) can kill it.
    os.set_current_pid(Some(root));
    let result = os.kill_process(grandchild, "cleanup".to_string());
    assert!(
        result.is_ok(),
        "root should be able to kill the re-attached grandchild, got: {:?}",
        result.err()
    );
}

#[test]
fn kill_process_nonexistent_reports_missing_process() {
    // A nonexistent pid must be reported as "does not exist", not as
    // "outside its scope": the two errors mean different things to the caller.
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    os.set_current_pid(Some(root));
    let result = os.kill_process(9999, "no such process".to_string());
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("does not exist"));
}

#[test]
fn orphan_reattaches_to_oldest_live_foreground() {
    // Determinism guard for the re-attachment target: with several live
    // foregrounds, the choice must be the lowest foreground pid (the session
    // root, created first by monotonic pid allocation), independent of the
    // process-table (HashMap) iteration order.
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg1".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let second = os.begin_foreground("fg2".to_string(), "goal".to_string(), 10, usize::MAX, None);
    assert!(second > root, "foreground pids must be monotonic");

    // A background process under `root` holds the surviving grandchild.
    let child = os
        .spawn(
            Some(root),
            "subagent".to_string(),
            "goal sub".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let grandchild = os
        .spawn(
            Some(child),
            "grandchild".to_string(),
            "goal gc".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    // Simulate the subagent finishing: terminate + drop its kernel entry.
    os.terminate_pid(child, "done".to_string());
    assert!(os.drop_terminated(child), "drop_terminated should succeed");

    // Both foregrounds are live candidates; the grandchild must re-attach to
    // the oldest one (`root`), not `second`.
    let proc = os.get_process(grandchild).expect("grandchild still alive");
    assert_eq!(proc.parent_pid, Some(root));

    // The session root (full capabilities) can manage it again.
    os.set_current_pid(Some(root));
    assert!(os.kill_process(grandchild, "cleanup".to_string()).is_ok());
}

#[test]
fn shm_write_rejected_for_non_owner_outside_group() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child_a = os
        .spawn(
            Some(root),
            "a".to_string(),
            "goal a".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(child_a));
    os.shm_create("secret".to_string(), "value_a".to_string())
        .unwrap();

    let child_b = os
        .spawn(
            Some(root),
            "b".to_string(),
            "goal b".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    os.set_current_pid(Some(child_b));
    let result = os.shm_write("secret".to_string(), "tampered".to_string());
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Permission denied"));
}

#[test]
fn shm_write_allowed_for_same_process_group() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child_a = os
        .spawn(
            Some(root),
            "a".to_string(),
            "goal a".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let child_b = os
        .spawn(
            Some(root),
            "b".to_string(),
            "goal b".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_process_group(child_a, 100).unwrap();
    os.set_process_group(child_b, 100).unwrap();

    os.set_current_pid(Some(child_a));
    os.shm_create("shared_config".to_string(), "v1".to_string())
        .unwrap();

    os.set_current_pid(Some(child_b));
    let result = os.shm_write("shared_config".to_string(), "v2".to_string());
    assert!(result.is_ok());
    assert_eq!(os.shm_read("shared_config"), Ok("v2".to_string()));
}

#[test]
fn shm_write_allowed_for_ancestor_of_owner() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child = os
        .spawn(
            Some(root),
            "child".to_string(),
            "goal child".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(child));
    os.shm_create("child_data".to_string(), "original".to_string())
        .unwrap();

    os.set_current_pid(Some(root));
    let result = os.shm_write("child_data".to_string(), "parent_override".to_string());
    assert!(result.is_ok());
    assert_eq!(os.shm_read("child_data"), Ok("parent_override".to_string()));
}

#[test]
fn shm_delete_rejected_for_non_owner_outside_group() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child_a = os
        .spawn(
            Some(root),
            "a".to_string(),
            "goal a".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    let child_b = os
        .spawn(
            Some(root),
            "b".to_string(),
            "goal b".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(child_a));
    os.shm_create("owned_by_a".to_string(), "data".to_string())
        .unwrap();

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
    let child = os
        .spawn(
            Some(root),
            "child".to_string(),
            "goal".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_process_group(child, 77).unwrap();

    os.set_current_pid(Some(child));
    os.shm_create("group_key".to_string(), "val".to_string())
        .unwrap();

    let entry = os.shared_memory.get("group_key").unwrap();
    assert_eq!(entry.owner_pid, child);
}

#[test]
fn shm_read_detects_checksum_corruption() {
    let mut os = LocalOS::new();
    let _root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    os.shm_create("data".to_string(), "original".to_string())
        .unwrap();

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
    os.shm_create("data".to_string(), "original".to_string())
        .unwrap();

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
    let child = os
        .spawn(
            Some(root),
            "worker".to_string(),
            "work".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    os.set_current_pid(Some(child));
    os.shm_create("child_data".to_string(), "value".to_string())
        .unwrap();
    os.set_current_pid(Some(root));

    os.terminate_pid(child, "done".to_string());

    let result = os.shm_read("child_data");
    assert!(matches!(result, Err(ShmReadError::OwnerTerminated { .. })));
}

#[test]
fn shm_read_degraded_returns_data_on_owner_terminated() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child = os
        .spawn(
            Some(root),
            "worker".to_string(),
            "work".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    os.set_current_pid(Some(child));
    os.shm_create("child_data".to_string(), "important_value".to_string())
        .unwrap();
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
    let child_a = os
        .spawn(
            Some(root1),
            "a".to_string(),
            "ga".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    let root2 = os.begin_foreground("fg2".to_string(), "goal2".to_string(), 10, usize::MAX, None);
    let child_b = os
        .spawn(
            Some(root2),
            "b".to_string(),
            "gb".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(child_a));
    os.shm_create("secret".to_string(), "private_data".to_string())
        .unwrap();

    os.set_current_pid(Some(child_b));
    let result = os.shm_read("secret");
    assert!(matches!(result, Err(ShmReadError::PermissionDenied { .. })));
}

#[test]
fn shm_health_check_detects_corrupted_and_orphaned() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child = os
        .spawn(
            Some(root),
            "worker".to_string(),
            "work".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(child));
    os.shm_create("orphan_data".to_string(), "val1".to_string())
        .unwrap();
    os.shm_create("good_data".to_string(), "val2".to_string())
        .unwrap();
    os.set_current_pid(Some(root));

    os.terminate_pid(child, "done".to_string());

    if let Some(entry) = os.shared_memory.get_mut("good_data") {
        entry.value = "corrupted".to_string();
    }

    let issues = os.shm_health_check();
    assert_eq!(issues.len(), 2);

    let has_orphan = issues
        .iter()
        .any(|(k, e)| k == "orphan_data" && matches!(e, ShmReadError::OwnerTerminated { .. }));
    let has_corrupt = issues
        .iter()
        .any(|(k, e)| k == "good_data" && matches!(e, ShmReadError::Corrupted { .. }));
    assert!(has_orphan);
    assert!(has_corrupt);
}

#[test]
fn shm_cleanup_orphans_removes_dead_owner_entries() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let child = os
        .spawn(
            Some(root),
            "worker".to_string(),
            "work".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();

    os.set_current_pid(Some(child));
    os.shm_create("orphan".to_string(), "will_be_removed".to_string())
        .unwrap();
    os.set_current_pid(Some(root));
    os.shm_create("root_data".to_string(), "stays".to_string())
        .unwrap();

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
        .spawn(
            Some(root),
            "w".to_string(),
            "do".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    // Simulate worker going to sleep on a futex
    if let Some(p) = os.processes.get_mut(&worker) {
        p.state = ProcessState::Waiting {
            reason: WaitReason::ProcessExit { on_pid: root },
        };
    }
    os.ready_set.remove(&worker);

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
    assert!(os.ready_set.contains(&worker));
}

#[test]
fn futex_try_wait_reports_value_changed() {
    use crate::primitives::{FutexOps, FutexWakeReason};
    let mut os = LocalOS::new();
    let addr = os.futex_create(0, "t".to_string());
    assert!(
        os.futex_try_wait(addr, 0).is_none(),
        "should block when equal"
    );
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
    assert_eq!(
        recs[1]
            .fields()
            .and_then(|f| f.get("model"))
            .map(String::as_str),
        Some("gpt")
    );
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
        os.rusage_charge(
            pid,
            ResourceUsageDelta {
                turns: 1,
                ..Default::default()
            }
        ),
        RlimitVerdict::Ok
    );
    assert_eq!(
        os.rusage_charge(
            pid,
            ResourceUsageDelta {
                turns: 1,
                ..Default::default()
            }
        ),
        RlimitVerdict::Ok
    );
    match os.rusage_charge(
        pid,
        ResourceUsageDelta {
            turns: 1,
            ..Default::default()
        },
    ) {
        RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        } => {
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
    let probe = ResourceUsageDelta {
        tokens_in: 200,
        ..Default::default()
    };
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
    use crate::primitives::{LlmModelPrice, LlmOps, LlmUsageReport, RlimitOps, RlimitVerdict};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    // 1000 prompt tok => 2500 micros; 500 completion tok => 3000 micros.
    os.llm_set_price(
        "gpt-test".into(),
        LlmModelPrice {
            prompt_per_1k_micros: 2_500,
            completion_per_1k_micros: 6_000,
        },
    );
    let out = os.llm_account(
        pid,
        LlmUsageReport {
            model: "gpt-test".into(),
            prompt_tokens: 1_000,
            completion_tokens: 500,
            reasoning_tokens: 0,
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
    use crate::primitives::RlimitOps;
    use crate::primitives::{LlmOps, LlmUsageReport};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    // No price registered for "mystery-model"
    let out = os.llm_account(
        pid,
        LlmUsageReport {
            model: "mystery-model".into(),
            prompt_tokens: 123,
            completion_tokens: 45,
            reasoning_tokens: 0,
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
        LlmModelPrice, LlmOps, LlmUsageReport, ResourceLimit, RlimitDim, RlimitOps, RlimitVerdict,
    };
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    os.llm_set_price(
        "g".into(),
        LlmModelPrice {
            prompt_per_1k_micros: 1_000,
            completion_per_1k_micros: 0,
        },
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
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            latency_ms: 0,
        },
    );
    match out.verdict {
        RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        } => {
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
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            latency_ms: 77,
        },
    );
    let recs = os.trace_drain_since(0);
    let found = recs.iter().any(|r| r.name == "llm.account");
    assert!(found, "expected a trace event named llm.account");
}

#[test]
fn llm_usage_ledger_records_and_drains() {
    use crate::primitives::{LlmModelPrice, LlmOps, LlmUsageReport};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    os.llm_set_price(
        "m".into(),
        LlmModelPrice {
            prompt_per_1k_micros: 1_000,
            completion_per_1k_micros: 2_000,
        },
    );
    // The initial ledger is empty.
    assert_eq!(os.llm_usage_head_seq(), 0);
    assert!(os.llm_usage_drain_since(0).is_empty());

    os.llm_account(
        pid,
        LlmUsageReport {
            model: "m".into(),
            prompt_tokens: 100,
            completion_tokens: 50,
            reasoning_tokens: 30,
            cached_prompt_tokens: 10,
            latency_ms: 7,
        },
    );
    os.llm_account(
        pid,
        LlmUsageReport {
            model: "m".into(),
            prompt_tokens: 200,
            completion_tokens: 80,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            latency_ms: 0,
        },
    );

    // Full drain: two records, ascending, fields correct.
    let all = os.llm_usage_drain_since(0);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].seq, 1);
    assert_eq!(all[0].pid, pid);
    assert_eq!(all[0].model, "m");
    assert_eq!(all[0].prompt_tokens, 100);
    assert_eq!(all[0].completion_tokens, 50);
    assert_eq!(all[0].reasoning_tokens, 30);
    assert_eq!(all[0].total_tokens, 150);
    assert_eq!(all[0].cached_prompt_tokens, 10);
    assert_eq!(all[0].latency_ms, 7);
    // 100 prompt -> 100 micros; 50 completion -> 100 micros.
    assert_eq!(all[0].cost_micros, 200);
    assert_eq!(all[1].seq, 2);
    assert_eq!(all[1].total_tokens, 280);

    // Cursor drain: fetch only records with seq>1.
    assert_eq!(os.llm_usage_head_seq(), 2);
    let tail = os.llm_usage_drain_since(1);
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].seq, 2);
    // Draining does not consume; repeated drains return the same result.
    assert_eq!(os.llm_usage_drain_since(0).len(), 2);
}

#[test]
fn llm_usage_ledger_capacity_evicts_oldest() {
    use crate::primitives::{LlmOps, LlmUsageReport};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    os.llm_usage_set_capacity(2);
    for i in 0..5u64 {
        os.llm_account(
            pid,
            LlmUsageReport {
                model: "m".into(),
                prompt_tokens: i,
                completion_tokens: 0,
                reasoning_tokens: 0,
                cached_prompt_tokens: 0,
                latency_ms: 0,
            },
        );
    }
    // seq still monotonically reaches 5, but only the last two records are kept.
    assert_eq!(os.llm_usage_head_seq(), 5);
    let recs = os.llm_usage_drain_since(0);
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].seq, 4);
    assert_eq!(recs[1].seq, 5);
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
    p.push(format!(
        "aios_vfs_{}_{}_{}_{}",
        name,
        std::process::id(),
        nanos,
        seq
    ));
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
    // Write 11 bytes + read back 11 bytes = 22
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
    use crate::primitives::{ResourceLimit, RlimitDim, RlimitOps, VfsError, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let mut limits = ResourceLimit::unlimited();
    limits.max_fs_bytes = 5;
    os.rlimit_set(pid, limits).unwrap();

    let p = tmp_path("quota");
    // Write 10 bytes — over the 5-byte cap
    match os.vfs_write_all(Some(pid), &p, "0123456789").unwrap_err() {
        VfsError::QuotaExceeded {
            dimension: RlimitDim::FsBytes,
            ..
        } => {}
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
        .spawn(
            Some(outsider_root),
            "outsider".into(),
            "goal".into(),
            20,
            4,
            None,
            None,
        )
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
    assert_eq!(
        os.channel_try_recv(Some(root), ch).unwrap(),
        IpcRecvResult::Closed
    );
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
    assert!(
        root_proc
            .mailbox
            .back()
            .map(|s| s.contains("[EVENT_WAKE]"))
            .unwrap_or(false)
    );
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

    os.wait_on_events(vec![evt2], WaitPolicy::All, None)
        .unwrap();
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

    os.channel_send(Some(root), keep_buffered, "x".into())
        .unwrap();
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

// ---- EpollOps (Phase 6) ----

#[test]
fn epoll_wait_returns_ready_channel_without_suspending() {
    use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult, IpcOps};
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
    let ch = os.channel_create(Some(root), 2, "epoll-ready".into());
    let ep = os.epoll_create("main".into());
    os.epoll_ctl_add(ep, EpollSource::Channel(ch), EpollEventMask::IN, 7)
        .unwrap();

    os.channel_send(Some(root), ch, "payload".into()).unwrap();
    match os.epoll_wait(ep, 8, None).unwrap() {
        EpollWaitResult::Ready(events) => {
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].source, EpollSource::Channel(ch));
            assert_eq!(events[0].events, EpollEventMask::IN);
            assert_eq!(events[0].user_data, 7);
        }
        other => panic!("expected ready, got {:?}", other),
    }
}

#[test]
fn epoll_wait_suspends_and_then_observes_event_source() {
    use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult};
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
    let ep = os.epoll_create("main".into());
    let watched = EventId::new(42);
    os.epoll_ctl_add(ep, EpollSource::Event(watched), EpollEventMask::IN, 99)
        .unwrap();

    match os.epoll_wait(ep, 8, Some(5)).unwrap() {
        EpollWaitResult::Suspended { timeout_tick } => assert_eq!(timeout_tick, Some(5)),
        other => panic!("expected suspended, got {:?}", other),
    }
    assert!(os.current_process_id().is_none());

    let woke = os.notify_events_completed(&[watched]);
    assert_eq!(woke, vec![root]);
    let resumed = os.pop_ready().unwrap();
    assert_eq!(resumed.pid, root);

    match os.epoll_wait(ep, 8, None).unwrap() {
        EpollWaitResult::Ready(events) => {
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].source, EpollSource::Event(watched));
            assert_eq!(events[0].events, EpollEventMask::IN);
            assert_eq!(events[0].user_data, 99);
        }
        other => panic!("expected ready, got {:?}", other),
    }
}

#[test]
fn epoll_ctl_mod_del_and_snapshot_work() {
    use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult, IpcOps};
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
    let ch = os.channel_create(Some(root), 1, "epoll-ctl".into());
    let ep = os.epoll_create("ctl".into());
    os.epoll_ctl_add(ep, EpollSource::Channel(ch), EpollEventMask::HUP, 1)
        .unwrap();
    os.epoll_ctl_mod(
        ep,
        EpollSource::Channel(ch),
        EpollEventMask::IN | EpollEventMask::HUP,
        2,
    )
    .unwrap();

    let snapshot = os.epoll_snapshot(ep).unwrap();
    assert_eq!(snapshot.label, "ctl");
    assert_eq!(snapshot.registrations.len(), 1);
    assert_eq!(snapshot.registrations[0].source, EpollSource::Channel(ch));
    assert_eq!(
        snapshot.registrations[0].events,
        EpollEventMask::IN | EpollEventMask::HUP
    );
    assert_eq!(snapshot.registrations[0].user_data, 2);

    os.channel_close(Some(root), ch).unwrap();
    match os.epoll_wait(ep, 8, None).unwrap() {
        EpollWaitResult::Ready(events) => {
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].events, EpollEventMask::HUP);
            assert_eq!(events[0].user_data, 2);
        }
        other => panic!("expected ready, got {:?}", other),
    }

    os.epoll_ctl_del(ep, EpollSource::Channel(ch)).unwrap();
    match os.epoll_wait(ep, 8, None).unwrap() {
        EpollWaitResult::Ready(events) => assert!(events.is_empty()),
        other => panic!("expected empty ready set, got {:?}", other),
    }
    assert!(os.epoll_destroy(ep));
    assert!(os.epoll_snapshot(ep).is_none());
}

#[test]
fn epoll_wait_returns_ready_for_futex_value_change() {
    use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult, FutexOps};
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
    let addr = os.futex_create(0, "epoll-futex-value".into());
    let ep = os.epoll_create("futex".into());
    os.epoll_ctl_add(
        ep,
        EpollSource::Futex { addr, expected: 0 },
        EpollEventMask::IN,
        11,
    )
    .unwrap();

    assert!(matches!(
        os.epoll_wait(ep, 8, None).unwrap(),
        EpollWaitResult::Suspended { timeout_tick: None }
    ));
    os.set_current_pid(Some(root));
    let _ = os.futex_store(addr, 9);

    match os.epoll_wait(ep, 8, None).unwrap() {
        EpollWaitResult::Ready(events) => {
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].source, EpollSource::Futex { addr, expected: 0 });
            assert_eq!(events[0].events, EpollEventMask::IN);
            assert_eq!(events[0].user_data, 11);
        }
        other => panic!("expected ready, got {:?}", other),
    }
}

#[test]
fn epoll_wait_observes_futex_wake_even_when_value_is_unchanged() {
    use crate::primitives::{EpollEventMask, EpollOps, EpollSource, EpollWaitResult, FutexOps};
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".into(), "goal".into(), 10, 0, None);
    let addr = os.futex_create(0, "epoll-futex-seq".into());
    let ep = os.epoll_create("futex-seq".into());
    os.epoll_ctl_add(
        ep,
        EpollSource::Futex { addr, expected: 0 },
        EpollEventMask::IN,
        22,
    )
    .unwrap();

    match os.epoll_wait(ep, 8, Some(4)).unwrap() {
        EpollWaitResult::Suspended { timeout_tick } => assert_eq!(timeout_tick, Some(4)),
        other => panic!("expected suspended, got {:?}", other),
    }
    assert!(os.current_process_id().is_none());

    os.futex_wake(addr, 1);
    let resumed = os.pop_ready().unwrap();
    assert_eq!(resumed.pid, root);

    match os.epoll_wait(ep, 8, None).unwrap() {
        EpollWaitResult::Ready(events) => {
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].source, EpollSource::Futex { addr, expected: 0 });
            assert_eq!(events[0].events, EpollEventMask::IN);
            assert_eq!(events[0].user_data, 22);
        }
        other => panic!("expected ready, got {:?}", other),
    }

    match os.epoll_wait(ep, 8, None).unwrap() {
        EpollWaitResult::Suspended { timeout_tick } => assert_eq!(timeout_tick, None),
        other => panic!("expected suspended after cursor refresh, got {:?}", other),
    }
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
    let handles: std::collections::HashSet<u64> = snap.iter().map(|e| e.handle.raw()).collect();
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
