use super::*;

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
