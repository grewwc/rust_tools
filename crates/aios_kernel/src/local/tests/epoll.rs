use super::*;

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
