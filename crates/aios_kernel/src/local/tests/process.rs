use super::*;

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
