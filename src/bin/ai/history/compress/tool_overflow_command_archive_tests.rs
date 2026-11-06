use std::cell::Cell;

use super::*;
use crate::ai::types::{FunctionCall, ToolCall};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ScanCounts {
    directory_enumerations: usize,
    candidate_checks: usize,
}

thread_local! {
    static SCAN_COUNTS: Cell<ScanCounts> = Cell::new(ScanCounts::default());
}

pub(super) fn record_directory_enumeration() {
    SCAN_COUNTS.with(|counts| {
        let mut next = counts.get();
        next.directory_enumerations += 1;
        counts.set(next);
    });
}

pub(super) fn record_candidate_check() {
    SCAN_COUNTS.with(|counts| {
        let mut next = counts.get();
        next.candidate_checks += 1;
        counts.set(next);
    });
}

fn reset_scan_counts() {
    SCAN_COUNTS.with(|counts| counts.set(ScanCounts::default()));
}

fn scan_counts() -> ScanCounts {
    SCAN_COUNTS.with(Cell::get)
}

struct Archives {
    root: PathBuf,
    overflow_dir: PathBuf,
    archive_dir: PathBuf,
}

impl Archives {
    fn new() -> Self {
        let root =
            std::env::temp_dir().join(format!("ai-command-archive-batch-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let overflow_dir = root.join("session-assets");
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        std::fs::create_dir_all(&archive_dir).unwrap();
        Self {
            root,
            overflow_dir,
            archive_dir,
        }
    }

    fn write(&self, name: &str, content: impl AsRef<[u8]>) -> PathBuf {
        let path = self.archive_dir.join(name);
        std::fs::write(&path, content).unwrap();
        path.canonicalize().unwrap()
    }

    fn snapshot(&self) -> Vec<(String, Vec<u8>)> {
        let mut files = std::fs::read_dir(&self.archive_dir)
            .unwrap()
            .map(|entry| {
                let path = entry.unwrap().path();
                (
                    path.file_name().unwrap().to_str().unwrap().to_string(),
                    std::fs::read(path).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        files.sort();
        files
    }
}

impl Drop for Archives {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn command_messages(queries: &[(&str, String, String)]) -> Vec<Message> {
    let mut messages = Vec::new();
    for (id, command, output) in queries {
        messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "execute_command".to_string(),
                    arguments: serde_json::json!({"command": command}).to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        messages.push(Message {
            role: "tool".to_string(),
            content: Value::String(output.clone()),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            reasoning_content: None,
        });
    }
    messages
}

fn protected_ids(messages: &[Message]) -> FxHashSet<String> {
    messages
        .iter()
        .filter_map(|message| message.tool_call_id.clone())
        .collect()
}

fn expected_stubs(messages: &[Message], paths: &[PathBuf]) -> Vec<Message> {
    let mut expected = messages.to_vec();
    for (query, path) in paths.iter().enumerate() {
        let call = &messages[query * 2].tool_calls.as_ref().unwrap()[0];
        let output = value_to_string(&messages[query * 2 + 1].content);
        let recall = build_tool_overflow_recall_lines("execute_command", &call.function.arguments);
        expected[query * 2 + 1].content = Value::String(build_preserved_tool_overflow_stub(
            path,
            "execute_command",
            &output,
            &recall,
        ));
    }
    expected
}

#[test]
fn command_archive_batch_reuses_shared_archive_with_exact_stubs() {
    let archives = Archives::new();
    let body = "\n  archived evidence line\n".repeat(300);
    let shared = archives.write("shared.txt", &body);
    for index in 0..3 {
        archives.write(&format!("unreferenced-{index}.txt"), "other evidence\n");
    }
    let queries = [
        ("same-a", format!("cat {}", shared.display()), body.clone()),
        (
            "same-b",
            format!("cat {}", shared.display()),
            body.trim().to_string(),
        ),
        (
            "same-c",
            "cat shared.txt".to_string(),
            format!("\t{body}\n"),
        ),
    ];
    for (_, command, output) in &queries {
        assert_eq!(
            archived_asset_path_in_command(command, &archives.overflow_dir, None, output),
            Some(shared.clone()),
        );
    }
    let mut messages = command_messages(&queries);
    let expected = expected_stubs(&messages, &[shared.clone(), shared.clone(), shared]);
    let protected = protected_ids(&messages);
    let before = archives.snapshot();
    reset_scan_counts();

    let spilled = spill_protected_precision_to_fit(
        &mut messages,
        0,
        Some(&archives.overflow_dir),
        None,
        &protected,
    );

    assert_eq!(spilled, 3);
    assert_eq!(messages, expected);
    assert_eq!(archives.snapshot(), before);
    assert_eq!(
        scan_counts(),
        ScanCounts {
            directory_enumerations: 1,
            candidate_checks: 4,
        },
    );
}

#[test]
fn command_archive_batch_preserves_absolute_relative_and_bare_matches() {
    let archives = Archives::new();
    let path = archives.write("shared.txt", "  archive body\n");
    let relative = path.strip_prefix(&archives.root).unwrap();
    let mut cache = CommandArchiveCandidates::new(&archives.overflow_dir);
    for command in [
        format!("cat {}", path.display()),
        format!("cat {}", relative.display()),
        "cat shared.txt".to_string(),
        "cat session-asset:tool-overflow-compressed/shared.txt".to_string(),
    ] {
        let expected = Some(path.clone());
        assert_eq!(
            archived_asset_path_in_command(
                &command,
                &archives.overflow_dir,
                Some(&archives.root),
                "\narchive body\t",
            ),
            expected,
        );
        assert_eq!(
            cache.path_in_command(&command, Some(&archives.root), "\narchive body\t"),
            expected,
        );
    }
    assert_eq!(
        cache.path_in_command("cat unrelated.txt", Some(&archives.root), "archive body"),
        None,
    );
}

#[test]
fn command_archive_batch_preserves_collision_content_fallback_and_order() {
    let archives = Archives::new();
    let first = archives.write("shared.txt", "wrong content\n");
    let second = archives.write("copy-shared.txt", "right content\n");
    std::fs::write(archives.root.join("shared.txt"), "cwd-only content\n").unwrap();
    let mut cache = CommandArchiveCandidates::new(&archives.overflow_dir);
    for (command, output, expected) in [
        ("cat copy-shared.txt", "right content", Some(second.clone())),
        ("cat copy-shared.txt", "wrong content", Some(first.clone())),
        ("cat shared.txt", "cwd-only content", None),
        (
            "cat copy-shared.txt | transform",
            "transformed content",
            None,
        ),
    ] {
        assert_eq!(
            archived_asset_path_in_command(
                command,
                &archives.overflow_dir,
                Some(&archives.root),
                output,
            ),
            expected,
        );
        assert_eq!(
            cache.path_in_command(command, Some(&archives.root), output),
            expected,
        );
    }

    // A basename can mention multiple candidates. Keep the full ordered list
    // rather than replacing it with a single filename-to-path entry.
    cache.candidates = Some(vec![first.clone(), second.clone()]);
    assert_eq!(
        cache.path_in_command("cat copy-shared.txt", None, "right content"),
        Some(second),
    );
    std::fs::write(&first, "right content\n").unwrap();
    assert_eq!(
        cache.path_in_command("cat copy-shared.txt", None, "right content"),
        Some(first),
    );
}

#[test]
fn command_archive_batch_preserves_missing_directory_write_fallback() {
    let archives = Archives::new();
    std::fs::remove_dir(&archives.archive_dir).unwrap();
    let queries = [
        (
            "missing-a",
            "cat lost.txt".to_string(),
            "first missing archive result\n".repeat(250),
        ),
        (
            "missing-b",
            "cat lost.txt".to_string(),
            "second missing archive result\n".repeat(250),
        ),
    ];
    let mut paths = Vec::new();
    let mut files = Vec::new();
    for (id, command, output) in &queries {
        assert_eq!(
            archived_asset_path_in_command(command, &archives.overflow_dir, None, output),
            None,
        );
        let digest = content_sha256_hex(format!("{id}\0{output}").as_bytes());
        let name = format!("spilled-execute_command-{}.txt", &digest[..24]);
        paths.push(archives.archive_dir.join(&name));
        files.push((name, output.as_bytes().to_vec()));
    }
    files.sort();
    let mut messages = command_messages(&queries);
    let expected = expected_stubs(&messages, &paths);
    let protected = protected_ids(&messages);
    reset_scan_counts();

    assert_eq!(
        spill_protected_precision_to_fit(
            &mut messages,
            0,
            Some(&archives.overflow_dir),
            None,
            &protected,
        ),
        2,
    );
    assert_eq!(messages, expected);
    assert_eq!(archives.snapshot(), files);
    assert_eq!(scan_counts(), ScanCounts::default());
}

#[cfg(unix)]
struct PermissionBlock {
    path: PathBuf,
    original: std::fs::Permissions,
}

#[cfg(unix)]
impl PermissionBlock {
    fn new(path: &Path) -> Self {
        use std::os::unix::fs::PermissionsExt;

        let original = std::fs::metadata(path).unwrap().permissions();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0)).unwrap();
        Self {
            path: path.to_path_buf(),
            original,
        }
    }
}

#[cfg(unix)]
impl Drop for PermissionBlock {
    fn drop(&mut self) {
        let _ = std::fs::set_permissions(&self.path, self.original.clone());
    }
}

#[cfg(unix)]
#[test]
fn command_archive_batch_preserves_unreadable_directory_fallback() {
    let archives = Archives::new();
    // Privileged runners bypass mode bits; an archive path that is a regular
    // file exercises the same read_dir failure there.
    let privileged = unsafe { libc::geteuid() } == 0;
    let permission_block = if privileged {
        std::fs::remove_dir(&archives.archive_dir).unwrap();
        std::fs::write(&archives.archive_dir, "not a directory").unwrap();
        None
    } else {
        Some(PermissionBlock::new(&archives.archive_dir))
    };
    assert!(std::fs::read_dir(&archives.archive_dir).is_err());
    let queries = [
        (
            "blocked-a",
            "cat shared.txt".to_string(),
            "first inaccessible archive result\n".repeat(250),
        ),
        (
            "blocked-b",
            "cat shared.txt".to_string(),
            "second inaccessible archive result\n".repeat(250),
        ),
    ];
    for (_, command, output) in &queries {
        assert_eq!(
            archived_asset_path_in_command(command, &archives.overflow_dir, None, output),
            None,
        );
    }
    let mut messages = command_messages(&queries);
    let original = messages.clone();
    let protected = protected_ids(&messages);
    reset_scan_counts();

    assert_eq!(
        spill_protected_precision_to_fit(
            &mut messages,
            0,
            Some(&archives.overflow_dir),
            None,
            &protected,
        ),
        0,
    );
    assert_eq!(messages, original);
    assert_eq!(
        scan_counts(),
        ScanCounts {
            directory_enumerations: 2,
            candidate_checks: 0,
        },
    );
    drop(permission_block);
    if privileged {
        assert_eq!(
            std::fs::read(&archives.archive_dir).unwrap(),
            b"not a directory"
        );
    } else {
        assert_eq!(archives.snapshot(), Vec::new());
    }
}

#[test]
fn command_archive_batch_rereads_contents_after_read_failure() {
    let archives = Archives::new();
    let unreadable = archives.write("shared.txt", [0xff]);
    let readable = archives.write("copy-shared.txt", "intact evidence\n");
    let mut cache = CommandArchiveCandidates::new(&archives.overflow_dir);
    assert_eq!(
        archived_asset_path_in_command(
            "cat copy-shared.txt",
            &archives.overflow_dir,
            None,
            "intact evidence",
        ),
        Some(readable.clone()),
    );
    reset_scan_counts();
    assert_eq!(
        cache.path_in_command("cat copy-shared.txt", None, "intact evidence"),
        Some(readable),
    );
    assert_eq!(
        cache.path_in_command("cat shared.txt", None, "new evidence"),
        None
    );
    std::fs::write(&unreadable, "new evidence\n").unwrap();
    assert_eq!(
        cache.path_in_command("cat shared.txt", None, "new evidence"),
        Some(unreadable.clone()),
    );
    assert_eq!(
        scan_counts(),
        ScanCounts {
            directory_enumerations: 1,
            candidate_checks: 2,
        },
    );
    assert_eq!(
        archived_asset_path_in_command(
            "cat shared.txt",
            &archives.overflow_dir,
            None,
            "new evidence",
        ),
        Some(unreadable),
    );
}

#[cfg(unix)]
#[test]
fn command_archive_batch_retries_failed_entry_checks_and_rejects_symlink_escape() {
    let archives = Archives::new();
    archives.write("inside.txt", "inside evidence\n");
    let outside = archives.root.join("outside.txt");
    std::fs::write(&outside, "outside evidence\n").unwrap();
    std::os::unix::fs::symlink(&outside, archives.archive_dir.join("escape.txt")).unwrap();
    let broken = archives.archive_dir.join("broken.txt");
    std::os::unix::fs::symlink(archives.root.join("missing.txt"), &broken).unwrap();
    let mut cache = CommandArchiveCandidates::new(&archives.overflow_dir);
    assert_eq!(
        archived_asset_path_in_command(
            "cat escape.txt",
            &archives.overflow_dir,
            None,
            "outside evidence",
        ),
        None,
    );
    reset_scan_counts();
    assert_eq!(
        cache.path_in_command("cat escape.txt", None, "outside evidence"),
        None
    );
    assert!(cache.candidates.is_none());
    std::fs::remove_file(&broken).unwrap();
    let restored = archives.write("broken.txt", "restored evidence\n");
    assert_eq!(
        cache.path_in_command("cat broken.txt", None, "restored evidence"),
        Some(restored.clone()),
    );
    assert_eq!(
        scan_counts(),
        ScanCounts {
            directory_enumerations: 2,
            candidate_checks: 6,
        },
    );
    assert_eq!(
        archived_asset_path_in_command(
            "cat broken.txt",
            &archives.overflow_dir,
            None,
            "restored evidence",
        ),
        Some(restored),
    );
}

#[test]
fn command_archive_batch_enumerates_once_instead_of_per_command() {
    let archives = Archives::new();
    for index in 0..6 {
        archives.write(&format!("archive-{index}.txt"), "archived evidence\n");
    }
    let before = archives.snapshot();
    reset_scan_counts();
    let legacy_results = (0..12)
        .map(|_| {
            archived_asset_path_in_command(
                "printf unrelated-output",
                &archives.overflow_dir,
                None,
                "unrelated-output",
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(legacy_results, vec![None; 12]);
    assert_eq!(
        scan_counts(),
        ScanCounts {
            directory_enumerations: 12,
            candidate_checks: 72,
        },
    );
    reset_scan_counts();
    let mut cache = CommandArchiveCandidates::new(&archives.overflow_dir);
    let cached_results = (0..12)
        .map(|_| cache.path_in_command("printf unrelated-output", None, "unrelated-output"))
        .collect::<Vec<_>>();
    assert_eq!(cached_results, vec![None; 12]);
    assert_eq!(
        scan_counts(),
        ScanCounts {
            directory_enumerations: 1,
            candidate_checks: 6,
        },
    );
    assert_eq!(archives.snapshot(), before);
}
