use super::*;

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
