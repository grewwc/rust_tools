use super::*;

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

#[test]
fn vfs_read_range_returns_bounded_chunks_and_charges_actual_bytes() {
    use crate::primitives::{RlimitOps, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let p = tmp_path("range_basic");
    os.vfs_write_all(Some(pid), &p, "0123456789").unwrap();

    // First chunk covers exactly the first 5 bytes; next_offset resumes at byte 5.
    let r1 = os.vfs_read_range(Some(pid), &p, 0, 5).unwrap();
    assert_eq!(r1.content, "01234");
    assert_eq!(r1.next_offset, 5);
    assert!(!r1.hit_eof);

    let r2 = os.vfs_read_range(Some(pid), &p, r1.next_offset, 100).unwrap();
    assert_eq!(r2.content, "56789");
    assert_eq!(r2.next_offset, 10);
    assert!(r2.hit_eof);

    // Reading at/after EOF returns empty content with hit_eof.
    let r3 = os.vfs_read_range(Some(pid), &p, 10, 100).unwrap();
    assert!(r3.content.is_empty());
    assert!(r3.hit_eof);

    // Charged fs_bytes: write 10 + chunk1 5 + chunk2 5 = 20.
    let usage = os.rusage_get(pid).unwrap();
    assert_eq!(usage.fs_bytes, 20);
    let _ = std::fs::remove_file(&p);
}

#[test]
fn vfs_read_range_trims_tail_at_char_boundary() {
    use crate::primitives::{RlimitOps, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let p = tmp_path("range_utf8");
    // "a" (1 byte) + "中" (3 bytes) + "b" (1 byte) = 5 bytes total.
    os.vfs_write_all(Some(pid), &p, "a中b").unwrap();

    // Requesting 3 bytes cuts inside the "中" char: the tail partial char must be trimmed and
    // next_offset must resume at the char's start (byte 1) so the next read re-reads the partial
    // char together with its remaining bytes and decodes it correctly.
    let r = os.vfs_read_range(Some(pid), &p, 0, 3).unwrap();
    assert_eq!(r.content, "a");
    assert_eq!(r.next_offset, 1);
    assert!(!r.hit_eof);

    let r2 = os.vfs_read_range(Some(pid), &p, r.next_offset, 100).unwrap();
    assert_eq!(r2.content, "中b");
    assert_eq!(r2.next_offset, 5);
    assert!(r2.hit_eof);

    // Charged: write 5 + "a" (1 byte) + "中b" (4 bytes) = 10.
    let usage = os.rusage_get(pid).unwrap();
    assert_eq!(usage.fs_bytes, 5 + 1 + 4);
    let _ = std::fs::remove_file(&p);
}

#[test]
fn vfs_read_range_invalid_utf8_mid_stream_errors_like_whole_read() {
    use crate::primitives::{VfsError, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let p = tmp_path("range_invalid");
    // 0xFF is not valid UTF-8 anywhere in the range: hard error, matching whole-file semantics.
    std::fs::write(&p, b"ok\xffbad").unwrap();
    match os.vfs_read_range(Some(pid), &p, 0, 100).unwrap_err() {
        VfsError::Io(msg) => assert!(msg.contains("valid UTF-8"), "unexpected: {msg}"),
        other => panic!("expected Io(valid UTF-8), got {other:?}"),
    }
    let _ = std::fs::remove_file(&p);
}

#[test]
fn vfs_read_range_tiny_chunk_at_leading_multibyte_char_errors_instead_of_stalling() {
    use crate::primitives::{VfsError, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let p = tmp_path("range_tiny");
    // "中" (3 bytes) + "文" (3 bytes) = 6 bytes total.
    os.vfs_write_all(Some(pid), &p, "中文").unwrap();

    // A chunk entirely inside the leading 3-byte char must error (actionably) instead of returning
    // ("", 0, hit_eof=false), which would make a paged caller retry offset 0 forever.
    match os.vfs_read_range(Some(pid), &p, 0, 1).unwrap_err() {
        VfsError::Io(msg) => {
            assert!(msg.contains("too small") && msg.contains("max_bytes"), "unexpected: {msg}")
        }
        other => panic!("expected Io(too small), got {other:?}"),
    }
    assert!(os.vfs_read_range(Some(pid), &p, 0, 2).is_err());

    // max_bytes == 0 can never make progress either.
    match os.vfs_read_range(Some(pid), &p, 0, 0).unwrap_err() {
        VfsError::Io(msg) => assert!(msg.contains("max_bytes"), "unexpected: {msg}"),
        other => panic!("expected Io(max_bytes), got {other:?}"),
    }

    // A chunk that contains the whole leading char still reads fine and resumes past it.
    let r = os.vfs_read_range(Some(pid), &p, 0, 3).unwrap();
    assert_eq!(r.content, "中");
    assert_eq!(r.next_offset, 3);
    assert!(!r.hit_eof);
    let r2 = os.vfs_read_range(Some(pid), &p, r.next_offset, 100).unwrap();
    assert_eq!(r2.content, "文");
    assert_eq!(r2.next_offset, 6);
    assert!(r2.hit_eof);
    let _ = std::fs::remove_file(&p);
}

#[test]
fn vfs_read_range_sensitive_and_missing_are_denied() {
    use crate::primitives::{VfsError, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let bad = std::path::PathBuf::from("/tmp/.ssh/id_rsa");
    match os.vfs_read_range(Some(pid), &bad, 0, 100).unwrap_err() {
        VfsError::PermissionDenied(_) => {}
        other => panic!("expected PermissionDenied, got {other:?}"),
    }
    let p = tmp_path("range_missing");
    match os.vfs_read_range(Some(pid), &p, 0, 100).unwrap_err() {
        VfsError::NotFound(_) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn vfs_read_range_exact_fill_reports_hit_eof() {
    use crate::primitives::{RlimitOps, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let p = tmp_path("range_exact");
    os.vfs_write_all(Some(pid), &p, "0123456789").unwrap();

    // max_bytes == remaining bytes: the chunk fills exactly and lands on EOF. hit_eof must be
    // true (contract: "a subsequent read at next_offset would return empty content") so a paged
    // caller does not issue a redundant final empty read.
    let r = os.vfs_read_range(Some(pid), &p, 0, 10).unwrap();
    assert_eq!(r.content, "0123456789");
    assert_eq!(r.next_offset, 10);
    assert!(r.hit_eof, "exact fill at EOF must report hit_eof");

    // A shorter chunk leaves bytes behind: hit_eof stays false, then true once resumed.
    let r2 = os.vfs_read_range(Some(pid), &p, 0, 5).unwrap();
    assert!(!r2.hit_eof);
    let r3 = os.vfs_read_range(Some(pid), &p, r2.next_offset, 100).unwrap();
    assert_eq!(r3.content, "56789");
    assert!(r3.hit_eof);

    // Multibyte char ending exactly at the range end (6 bytes, max_bytes == 6) is also EOF.
    let p2 = tmp_path("range_exact_utf8");
    os.vfs_write_all(Some(pid), &p2, "中文").unwrap();
    let ru = os.vfs_read_range(Some(pid), &p2, 0, 6).unwrap();
    assert_eq!(ru.content, "中文");
    assert_eq!(ru.next_offset, 6);
    assert!(ru.hit_eof);

    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(&p2);
}

#[test]
fn vfs_write_if_unchanged_is_compare_and_swap() {
    use crate::primitives::{VfsError, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let p = tmp_path("cas_write");
    let q = tmp_path("cas_create");
    let r = tmp_path("cas_missing");
    os.vfs_write_all(Some(pid), &p, "original").unwrap();

    // Stale snapshot: aborts with [FILE_CHANGED] and leaves the file untouched.
    match os
        .vfs_write_if_unchanged(Some(pid), &p, Some("stale"), "new")
        .unwrap_err()
    {
        VfsError::Io(msg) if msg.contains("[FILE_CHANGED]") => {}
        other => panic!("expected Io([FILE_CHANGED]), got {other:?}"),
    }
    assert_eq!(
        os.vfs_read_to_string(Some(pid), &p).unwrap(),
        "original",
        "a failed CAS must not modify the file"
    );

    // Matching snapshot: writes.
    os.vfs_write_if_unchanged(Some(pid), &p, Some("original"), "new")
        .unwrap();
    assert_eq!(os.vfs_read_to_string(Some(pid), &p).unwrap(), "new");

    // expected=None on an existing file is a mismatch (create-only).
    match os
        .vfs_write_if_unchanged(Some(pid), &p, None, "x")
        .unwrap_err()
    {
        VfsError::Io(msg) if msg.contains("[FILE_CHANGED]") => {}
        other => panic!("expected Io([FILE_CHANGED]), got {other:?}"),
    }
    assert_eq!(os.vfs_read_to_string(Some(pid), &p).unwrap(), "new");

    // expected=None on a missing file creates it.
    os.vfs_write_if_unchanged(Some(pid), &q, None, "fresh")
        .unwrap();
    assert_eq!(
        os.vfs_read_to_string(Some(pid), &q).unwrap(),
        "fresh"
    );

    // expected=Some on a missing file is a mismatch (file must exist).
    match os
        .vfs_write_if_unchanged(Some(pid), &r, Some("x"), "y")
        .unwrap_err()
    {
        VfsError::Io(msg) if msg.contains("[FILE_CHANGED]") => {}
        other => panic!("expected Io([FILE_CHANGED]), got {other:?}"),
    }
    assert!(!r.exists(), "a failed CAS must not create the file");

    // Sensitive paths are denied like any other VFS op.
    let bad = std::path::PathBuf::from("/tmp/.ssh/id_rsa");
    match os
        .vfs_write_if_unchanged(Some(pid), &bad, Some("x"), "y")
        .unwrap_err()
    {
        VfsError::PermissionDenied(_) => {}
        other => panic!("expected PermissionDenied, got {other:?}"),
    }

    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(&q);
}

#[test]
fn vfs_remove_if_unchanged_is_compare_and_swap() {
    use crate::primitives::{VfsError, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let p = tmp_path("cas_remove");
    os.vfs_write_all(Some(pid), &p, "original").unwrap();

    // Stale snapshot: aborts with [FILE_CHANGED] and keeps the file.
    match os
        .vfs_remove_if_unchanged(Some(pid), &p, Some("stale"))
        .unwrap_err()
    {
        VfsError::Io(msg) if msg.contains("[FILE_CHANGED]") => {}
        other => panic!("expected Io([FILE_CHANGED]), got {other:?}"),
    }
    assert!(p.exists(), "a failed CAS delete must keep the file");
    assert_eq!(
        os.vfs_read_to_string(Some(pid), &p).unwrap(),
        "original"
    );

    // Matching snapshot: removes.
    os.vfs_remove_if_unchanged(Some(pid), &p, Some("original"))
        .unwrap();
    assert!(!p.exists());
}

#[test]
fn vfs_create_only_preserves_quota_and_trace_contract() {
    use crate::primitives::{ResourceLimit, RlimitDim, RlimitOps, TraceOps, VfsError, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let mut limits = ResourceLimit::unlimited();
    limits.max_fs_bytes = 5;
    os.rlimit_set(pid, limits).unwrap();
    let path = tmp_path("create_quota");

    let err = os.vfs_write_if_unchanged(Some(pid), &path, None, "0123456789").unwrap_err();
    assert!(matches!(err, VfsError::QuotaExceeded { dimension: RlimitDim::FsBytes, .. }));
    // VFS quota is charged after successful I/O; a quota error does not undo the write.
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "0123456789");
    assert_eq!(os.rusage_get(pid).unwrap().fs_bytes, 10);
    let err = os.vfs_write_if_unchanged(Some(pid), &path, None, "overwrite").unwrap_err();
    assert!(matches!(err, VfsError::Io(ref msg) if msg.contains("[FILE_CHANGED]")));
    assert_eq!(os.rusage_get(pid).unwrap().fs_bytes, 10, "rejected creation must not charge bytes");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "0123456789");
    let traces = os.trace_drain_since(0);
    assert_eq!(traces.iter().filter(|r| r.name == "vfs.write").count(), 1);
    assert_eq!(traces.iter().filter(|r| r.name == "vfs.write.stale").count(), 1);
    let _ = std::fs::remove_file(&path);
}

#[cfg(unix)]
#[test]
fn vfs_create_only_rejects_dangling_symlink_without_touching_target() {
    use crate::primitives::{RlimitOps, TraceOps, VfsError, VfsOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let link = tmp_path("create_link");
    let target = tmp_path("create_link_target");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let err = os.vfs_write_if_unchanged(Some(pid), &link, None, "new").unwrap_err();
    assert!(matches!(err, VfsError::Io(ref msg) if msg.contains("[FILE_CHANGED]")));
    assert_eq!(std::fs::read_link(&link).unwrap(), target);
    assert!(!target.exists());
    assert_eq!(os.rusage_get(pid).unwrap().fs_bytes, 0);
    assert!(os.trace_drain_since(0).iter().any(|r| r.name == "vfs.write.stale"));
    let _ = std::fs::remove_file(&link);
}

#[test]
fn vfs_create_only_is_exclusive_across_independent_kernels() {
    use crate::primitives::{VfsError, VfsOps};
    let path = tmp_path("independent_create_race");
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    let workers: Vec<_> = (0..8).map(|id| {
        let path = path.clone();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            // Independent kernels share no mutex; only the OS create-new operation serializes them.
            let mut os = LocalOS::new();
            let content = format!("creator-{id}");
            barrier.wait();
            let result = os.vfs_write_if_unchanged(None, &path, None, &content);
            (content, result)
        })
    }).collect();
    let results: Vec<_> = workers.into_iter().map(|worker| worker.join().unwrap()).collect();
    let winners: Vec<_> = results.iter().filter(|(_, result)| result.is_ok()).collect();
    assert_eq!(winners.len(), 1);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), winners[0].0);
    for (_, result) in &results {
        if let Err(err) = result {
            assert!(matches!(err, VfsError::Io(msg) if msg.contains("[FILE_CHANGED]")), "{err:?}");
        }
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn remove_partial_create_file_cleans_only_own_partial_write() {
    use crate::local::vfs::remove_partial_create_file;

    // A write that failed partway leaves a strict prefix of the intended content;
    // that file is ours and must be removed to restore the "absent" precondition.
    let partial = tmp_path("partial_create_cleanup");
    std::fs::write(&partial, "hello world, prefix").unwrap();
    remove_partial_create_file(&partial, "hello world, prefix and more");
    assert!(!partial.exists(), "own partial write must be cleaned up");

    // Content that matches the intended content in full is NOT removed: the write
    // completed, so the caller's CAS rollback must decide instead of us removing it.
    let full = tmp_path("partial_create_full");
    std::fs::write(&full, "exact match").unwrap();
    remove_partial_create_file(&full, "exact match");
    assert!(full.exists(), "completed write must be left for CAS rollback");

    // Unrelated content (an external writer replaced the file) must be preserved.
    let foreign = tmp_path("partial_create_foreign");
    std::fs::write(&foreign, "somebody else's bytes").unwrap();
    remove_partial_create_file(&foreign, "our intended content");
    assert!(foreign.exists(), "foreign file must not be deleted");

    // Non-prefix but shorter content (external partial overwrite) is preserved.
    let shorter = tmp_path("partial_create_shorter");
    std::fs::write(&shorter, "short").unwrap();
    remove_partial_create_file(&shorter, "a much longer intended content");
    assert!(shorter.exists(), "non-prefix content must not be deleted");

    let _ = std::fs::remove_file(&full);
    let _ = std::fs::remove_file(&foreign);
    let _ = std::fs::remove_file(&shorter);
}
