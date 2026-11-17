use super::*;

fn foreground() -> (LocalOS, u64) {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("foreground".into(), "goal".into(), 10, 8, None);
    (os, root)
}

fn spawn_at_priority(os: &mut LocalOS, root: u64, name: &str, priority: u8) -> u64 {
    os.spawn(
        Some(root),
        name.into(),
        "goal".into(),
        priority,
        8,
        None,
        None,
    )
    .unwrap()
}

fn event_input() -> (Vec<EventId>, Vec<EventId>) {
    let unique = vec![EventId::new(3), EventId::new(1), EventId::new(2)];
    let input = vec![
        unique[0], unique[1], unique[0], unique[2], unique[1], unique[2], unique[0],
    ];
    (input, unique)
}

fn assert_waiting(
    os: &LocalOS,
    root: u64,
    expected: &[EventId],
    policy: WaitPolicy,
    timeout: Option<u64>,
) {
    assert_eq!(
        os.get_process(root).unwrap().state,
        ProcessState::Waiting {
            reason: WaitReason::Events {
                event_ids: expected.to_vec(),
                policy,
                timeout_tick: timeout,
            },
        }
    );
}

fn assert_single_wake(os: &mut LocalOS, root: u64, repeated_completion: &[EventId]) {
    assert_eq!(os.get_process(root).unwrap().state, ProcessState::Ready);
    assert_eq!(os.get_process(root).unwrap().mailbox.len(), 1);
    assert_eq!(os.ready_count(), 1);
    assert!(os.notify_events_completed(repeated_completion).is_empty());
    assert_eq!(os.get_process(root).unwrap().mailbox.len(), 1);
    assert_eq!(os.ready_count(), 1);
    assert_eq!(os.pop_ready().unwrap().pid, root);
    assert!(os.pop_ready().is_none());
}

#[test]
fn optimized_scheduler_equal_priority_fifo() {
    let (mut os, root) = foreground();
    let expected: Vec<_> = (0..8)
        .map(|index| spawn_at_priority(&mut os, root, &format!("child_{index}"), 50))
        .collect();

    assert_eq!(os.ready_count(), expected.len());
    for pid in expected {
        assert_eq!(os.pop_ready().unwrap().pid, pid);
    }
    assert_eq!(os.ready_count(), 0);
    assert!(os.pop_ready().is_none());
}

#[test]
fn optimized_scheduler_equal_priority_fifo_with_tombstones() {
    let (mut os, root) = foreground();
    let dead_head = spawn_at_priority(&mut os, root, "dead_head", 50);
    os.terminate_pid(dead_head, "done".into());
    let first = spawn_at_priority(&mut os, root, "first", 50);
    let dead_middle = spawn_at_priority(&mut os, root, "dead_middle", 50);
    let second = spawn_at_priority(&mut os, root, "second", 50);
    os.terminate_pid(dead_middle, "done".into());
    let dead_tail = spawn_at_priority(&mut os, root, "dead_tail", 50);
    os.terminate_pid(dead_tail, "done".into());
    // The append fast path must preserve FIFO when the equal-priority tail is a tombstone.
    let third = spawn_at_priority(&mut os, root, "third", 50);
    let dead_lower_priority = spawn_at_priority(&mut os, root, "dead_lower_priority", 100);
    os.terminate_pid(dead_lower_priority, "done".into());
    // The insertion search must also preserve FIFO with a lower-priority tail tombstone.
    let fourth = spawn_at_priority(&mut os, root, "fourth", 50);

    assert_eq!(os.ready_queue.len(), 8);
    assert_eq!(os.ready_count(), 4);
    for pid in [first, second, third, fourth] {
        assert_eq!(os.pop_ready().unwrap().pid, pid);
    }
    assert!(os.pop_ready().is_none());
    assert_eq!(os.ready_count(), 0);
    assert!(os.ready_queue.is_empty());
}

#[test]
fn optimized_scheduler_duplicate_events_preserve_first_order_any() {
    let (mut os, root) = foreground();
    let (input, unique) = event_input();
    assert_eq!(
        os.wait_on_events(input, WaitPolicy::Any, None).unwrap(),
        None
    );
    assert_waiting(&os, root, &unique, WaitPolicy::Any, None);
    assert_eq!(os.event_waiters.len(), unique.len());
    for event in &unique {
        let waiters = os.event_waiters.get(event).unwrap();
        assert_eq!(waiters.len(), 1);
        assert!(waiters.contains(&root));
    }
    let repeated = [unique[1], unique[1], unique[1]];
    assert_eq!(os.notify_events_completed(&repeated), vec![root]);
    assert_single_wake(&mut os, root, &repeated);
}

#[test]
fn optimized_scheduler_duplicate_events_preserve_first_order_all() {
    let (mut os, root) = foreground();
    let (input, unique) = event_input();
    os.wait_on_events(input, WaitPolicy::All, None).unwrap();
    assert_waiting(&os, root, &unique, WaitPolicy::All, None);

    for event in [unique[1], unique[0]] {
        assert!(os.notify_events_completed(&[event, event]).is_empty());
        assert_waiting(&os, root, &unique, WaitPolicy::All, None);
        assert!(os.get_process(root).unwrap().mailbox.is_empty());
        assert_eq!(os.ready_count(), 0);
    }
    let repeated = [unique[2], unique[2]];
    assert_eq!(os.notify_events_completed(&repeated), vec![root]);
    assert_single_wake(&mut os, root, &repeated);
}

#[test]
fn optimized_scheduler_duplicate_events_already_completed_keep_message_order() {
    for policy in [WaitPolicy::Any, WaitPolicy::All] {
        let (mut os, root) = foreground();
        let (input, unique) = event_input();
        assert!(
            os.notify_events_completed(&[unique[2], unique[0], unique[1]])
                .is_empty()
        );
        assert_eq!(os.wait_on_events(input, policy, Some(2)).unwrap(), None);
        assert_eq!(os.current_process_id(), Some(root));
        assert_eq!(os.get_process(root).unwrap().state, ProcessState::Running);
        assert!(!os.consume_yield_requested());
        assert_eq!(os.ready_count(), 0);
        assert!(os.event_waiters.is_empty());
        let mailbox = &os.get_process(root).unwrap().mailbox;
        assert_eq!(mailbox.len(), 1);
        assert_eq!(
            mailbox.back().unwrap().lines().nth(2),
            Some("Completed event ids: evt_3, evt_1, evt_2")
        );
    }
}

#[test]
fn optimized_scheduler_empty_events_leave_running_process_unchanged() {
    for policy in [WaitPolicy::Any, WaitPolicy::All] {
        let (mut os, root) = foreground();
        assert_eq!(
            os.wait_on_events(Vec::new(), policy, Some(2)),
            Err("event_ids cannot be empty.".to_string())
        );
        assert_eq!(os.current_process_id(), Some(root));
        assert_eq!(os.get_process(root).unwrap().state, ProcessState::Running);
        assert!(os.get_process(root).unwrap().mailbox.is_empty());
        assert!(!os.consume_yield_requested());
        assert!(os.event_waiters.is_empty());
        assert_eq!(os.next_wakeup_tick(), None);
        assert_eq!(os.ready_count(), 0);
    }
}

#[test]
fn optimized_scheduler_duplicate_events_timeout_wakes_once() {
    for policy in [WaitPolicy::Any, WaitPolicy::All] {
        for (timeout_ticks, deadline) in [(2, 2), (0, 1)] {
            let (mut os, root) = foreground();
            let (input, unique) = event_input();
            assert_eq!(
                os.wait_on_events(input, policy.clone(), Some(timeout_ticks))
                    .unwrap(),
                Some(deadline)
            );
            assert_waiting(&os, root, &unique, policy.clone(), Some(deadline));
            if deadline > 1 {
                os.advance_tick();
                assert_waiting(&os, root, &unique, policy.clone(), Some(deadline));
            }
            os.advance_tick();
            assert_eq!(os.get_process(root).unwrap().state, ProcessState::Ready);
            let mailbox = &os.get_process(root).unwrap().mailbox;
            assert_eq!(mailbox.len(), 1);
            assert_eq!(
                mailbox.back().unwrap(),
                &format!("Event wait timeout reached at scheduler tick {deadline}.")
            );
            os.advance_tick();
            assert_single_wake(&mut os, root, &[unique[1], unique[1]]);
        }
    }
}
