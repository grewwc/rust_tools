use super::*;

#[test]
fn preserved_read_file_hints_separate_history_from_current_content() {
    let path = Path::new("/tmp/historical-read-result.txt");
    let content = "historical read result: original text\n".repeat(200);
    for has_original in [true, false] {
        let recall = if has_original {
            build_tool_overflow_recall_lines(
                "read_file",
                r#"{"file_path":"src/lib.rs","offset":120,"limit":40}"#,
            )
        } else {
            Vec::new()
        };
        let stub = build_preserved_tool_overflow_stub_with_archive(
            Some(path),
            "read_file",
            &content,
            &recall,
        );
        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("preview can be collapsed");
        for (view, preview_inline) in [(&stub, true), (&anchor, false)] {
            let hint = preserved_tool_overflow_hint("read_file", &recall, preview_inline);
            assert!(
                view.contains(hint),
                "generated view must use its matching hint: {view}"
            );
            assert!(is_preserved_tool_overflow_stub(view));
            assert!(view.contains("- file_path: /tmp/historical-read-result.txt"));
            assert!(hint.contains("For exact historical output, including following, citing, or transforming what was read, read `file_path`"));
            assert!(hint.contains("do not substitute a new read of the original file"));
            assert!(hint.contains("Only for current content"));
            assert!(!hint.contains("re-read `original_file_path` before proceeding"));
            assert!(!hint.contains("re-read the original path before proceeding"));
            assert_eq!(
                hint.contains("The preview is an anchor, not the content"),
                preview_inline
            );
            if has_original {
                assert!(hint.contains("read a smaller range of `original_file_path`"));
                assert!(view.contains("- original_file_path: src/lib.rs"));
                assert!(view.contains("- original_range: lines=120..159"));
            } else {
                assert!(
                    hint.contains("locate the original source and read a small relevant range")
                );
                assert!(!view.contains("- original_file_path:"));
            }
        }
    }
}

#[test]
fn compaction_state_separates_historical_recovery_from_current_reads() {
    let state = super::super::mid_turn::CONTEXT_COMPACTION_STATE;
    assert!(state.contains("For exact historical output, read the stub's saved archive"));
    assert!(
        state.contains(
            "only for current content, read a small relevant range of original_file_path"
        )
    );
    assert!(!state.contains("only when the original source is unavailable"));
}

#[test]
fn compression_contract_keeps_original_pointer_through_mechanical_recompression() {
    let path = Path::new("/tmp/historical-command-result.txt");
    let content = format!("stdout:\n{}", "historical row: alpha βeta\n".repeat(200));
    let recall = vec!["- original_command: git commit".to_string()];
    let stub = build_preserved_tool_overflow_stub_with_archive(
        Some(path),
        "execute_command",
        &content,
        &recall,
    );
    assert!(is_preserved_tool_overflow_stub(&stub));
    assert!(stub.contains("[COMPRESSED: excerpt]"));
    assert!(stub.contains("coverage: partial preview"));
    assert!(stub.contains("Read `file_path` for exact historical output"));
    assert!(stub.contains("not an instruction to execute it again"));
    assert!(!stub.contains("re-run `original_command` before proceeding"));

    let anchor = collapse_overflow_stub_to_anchor(&stub).expect("preview can be collapsed");
    assert!(is_preserved_tool_overflow_stub(&anchor));
    assert!(anchor.contains("[COMPRESSED: pointer]"));
    assert!(anchor.contains("- file_path: /tmp/historical-command-result.txt"));
    assert!(anchor.contains("- original_command: git commit"));
    assert!(collapse_overflow_stub_to_anchor(&anchor).is_none());

    let pointer = minimize_overflow_stub_to_pointer(&anchor).expect("recognized archive pointer");
    assert!(is_preserved_tool_overflow_stub(&pointer));
    assert!(pointer.contains("[COMPRESSED: pointer]"));
    assert!(pointer.contains("- file_path: /tmp/historical-command-result.txt"));
    assert!(!pointer.contains("re-run"));
}

#[test]
fn compression_contract_without_archive_reports_canonical_recovery_not_an_original_file() {
    let stub = build_preserved_tool_overflow_stub_with_archive(
        None,
        "execute_command",
        "stdout: historical output",
        &[],
    );
    assert!(is_preserved_tool_overflow_stub(&stub));
    assert!(stub.contains("[COMPRESSED: pointer]"));
    assert!(stub.contains("recover from canonical history"));
    assert!(!stub.contains("[COMPRESSED: excerpt]"));
    assert!(!stub.contains("- file_path:"));
}
