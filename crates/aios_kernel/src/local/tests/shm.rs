use super::*;

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
