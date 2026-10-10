use super::*;

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
