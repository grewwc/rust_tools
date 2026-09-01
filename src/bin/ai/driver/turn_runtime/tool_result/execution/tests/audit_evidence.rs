//! Tests for the audit-only structured-evidence final-response gate.

use super::super::*;
use super::common::*;

fn temporary_root(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "audit-evidence-{label}-{}",
        uuid::Uuid::new_v4().simple()
    ))
}

fn write_fixture(root: &PathBuf) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), "first\nsecond\nthird\nfourth\n").unwrap();
}

fn complete_report(path: &str) -> String {
    format!(
        "<audit_report>{}</audit_report>",
        serde_json::json!({
            "findings": [{
                "severity": "P1",
                "title": "A verified defect",
                "claim": "The branch returns the wrong value.",
                "trigger": "The caller reaches the affected branch.",
                "impact": "The caller observes an incorrect result.",
                "source_evidence": [{
                    "path": path,
                    "start_line": 1,
                    "explanation": "The local branch contains the returned value."
                }],
                "semantic_evidence": [{
                    "path": path,
                    "start_line": 2,
                    "explanation": "The caller consumes that branch result."
                }],
                "falsification_checks": [{
                    "path": path,
                    "start_line": 3,
                    "explanation": "The nearby guard does not exclude the trigger."
                }]
            }],
            "open_questions": [],
            "coverage_gaps": []
        })
    )
}

fn successful_reads(path: &str) -> Vec<Message> {
    let mut messages = Vec::new();
    for (id, offset) in [
        ("read-source", 1_u64),
        ("read-semantic", 2_u64),
        ("read-falsify", 3_u64),
    ] {
        messages.push(assistant_tool_call_message(test_tool_call(
            id,
            "read_file",
            serde_json::json!({"file_path": path, "offset": offset, "limit": 1}),
        )));
        messages.push(tool_result_message(
            id,
            &format!("{offset:>6}\tline {offset}"),
        ));
    }
    messages
}

#[test]
fn non_audit_agents_keep_existing_final_response_behavior() {
    let mut messages = Vec::new();
    let mut final_text = "ordinary final response".to_string();
    assert_eq!(
        audit_evidence_gate_action(
            "build",
            &mut messages,
            &[],
            &mut final_text,
            None,
            false,
            1,
            16,
        ),
        AuditEvidenceGateAction::Allow
    );
    assert!(messages.is_empty());
    assert_eq!(final_text, "ordinary final response");
}

#[test]
fn audit_requires_the_structured_report_protocol_and_reopens_once() {
    let mut messages = Vec::new();
    let mut first_draft = "P1: an unsupported review claim".to_string();
    assert_eq!(
        audit_evidence_gate_action(
            "audit",
            &mut messages,
            &[],
            &mut first_draft,
            None,
            false,
            1,
            16,
        ),
        AuditEvidenceGateAction::Reopen
    );
    assert!(current_turn_has_internal_marker(
        &messages,
        AUDIT_EVIDENCE_RETRY_MARKER
    ));

    let mut second_draft = "P1: the same unsupported review claim".to_string();
    assert_eq!(
        audit_evidence_gate_action(
            "audit",
            &mut messages,
            &[],
            &mut second_draft,
            None,
            false,
            2,
            16,
        ),
        AuditEvidenceGateAction::Warn
    );
    assert!(second_draft.contains("No verified findings could be published"));
    assert!(!second_draft.contains("same unsupported review claim"));
}

#[test]
fn audit_renders_only_a_finding_with_three_grounded_evidence_kinds() {
    let root = temporary_root("complete-chain");
    write_fixture(&root);
    let mut messages = Vec::new();
    let turn_messages = successful_reads("src/lib.rs");
    let mut final_text = complete_report("src/lib.rs");

    assert_eq!(
        audit_evidence_gate_action(
            "audit",
            &mut messages,
            &turn_messages,
            &mut final_text,
            Some(&root),
            false,
            1,
            16,
        ),
        AuditEvidenceGateAction::Allow
    );
    assert!(final_text.contains("## Verified findings"));
    assert!(final_text.contains("### P1 — A verified defect"));
    assert!(final_text.contains("**Source evidence.**"));
    assert!(final_text.contains("`src/lib.rs:1`"));
    assert!(final_text.contains("**Semantic evidence.**"));
    assert!(final_text.contains("**Falsification checks.**"));
    assert!(messages.is_empty());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn audit_accepts_the_read_file_path_alias() {
    let root = temporary_root("path-alias");
    write_fixture(&root);
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    for (id, offset) in [
        ("alias-source", 1_u64),
        ("alias-semantic", 2_u64),
        ("alias-falsify", 3_u64),
    ] {
        turn_messages.push(assistant_tool_call_message(test_tool_call(
            id,
            "read_file",
            serde_json::json!({"path": "src/lib.rs", "offset": offset, "limit": 1}),
        )));
        turn_messages.push(tool_result_message(
            id,
            &format!("{offset:>6}\tline {offset}"),
        ));
    }
    let mut final_text = complete_report("src/lib.rs");

    assert_eq!(
        audit_evidence_gate_action(
            "audit",
            &mut messages,
            &turn_messages,
            &mut final_text,
            Some(&root),
            false,
            1,
            16,
        ),
        AuditEvidenceGateAction::Allow
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn audit_rejects_read_file_snapshot_artifacts_as_evidence() {
    let root = temporary_root("snapshot");
    let relative_path =
        "session.assets/tool-overflow-compressed/20260101T000000Z-read_file-deadbeef.txt";
    let snapshot = root.join(relative_path);
    fs::create_dir_all(snapshot.parent().expect("snapshot parent")).expect("create snapshot dir");
    fs::write(&snapshot, "old line 1\nold line 2\nold line 3\n").expect("write snapshot");
    let mut messages = Vec::new();
    let turn_messages = successful_reads(relative_path);
    let mut final_text = complete_report(relative_path);

    assert_eq!(
        audit_evidence_gate_action(
            "audit",
            &mut messages,
            &turn_messages,
            &mut final_text,
            Some(&root),
            false,
            1,
            16,
        ),
        AuditEvidenceGateAction::Reopen
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn audit_rejects_evidence_not_read_in_the_current_turn() {
    let root = temporary_root("unread");
    write_fixture(&root);
    let mut messages = Vec::new();
    let mut final_text = complete_report("src/lib.rs");

    assert_eq!(
        audit_evidence_gate_action(
            "audit-fast",
            &mut messages,
            &[],
            &mut final_text,
            Some(&root),
            false,
            1,
            16,
        ),
        AuditEvidenceGateAction::Reopen
    );
    assert!(current_turn_has_internal_marker(
        &messages,
        AUDIT_EVIDENCE_RETRY_MARKER
    ));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn audit_rejects_requested_lines_that_read_file_did_not_render() {
    let root = temporary_root("truncated-read");
    write_fixture(&root);
    let mut messages = Vec::new();
    let turn_messages = vec![
        assistant_tool_call_message(test_tool_call(
            "truncated-read",
            "read_file",
            serde_json::json!({"file_path": "src/lib.rs", "offset": 1, "limit": 3}),
        )),
        tool_result_message(
            "truncated-read",
            "     1\tline 1\n... [truncated: output capped at 64000 chars; showing lines 1-1 of 3; 2 more line(s) not shown. Continue with offset=2 to read the rest.]",
        ),
    ];
    let mut final_text = complete_report("src/lib.rs");

    assert_eq!(
        audit_evidence_gate_action(
            "audit",
            &mut messages,
            &turn_messages,
            &mut final_text,
            Some(&root),
            false,
            1,
            16,
        ),
        AuditEvidenceGateAction::Reopen
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn audit_rejects_a_zero_limit_read_file_result() {
    let root = temporary_root("zero-limit");
    write_fixture(&root);
    let mut messages = Vec::new();
    let turn_messages = vec![
        assistant_tool_call_message(test_tool_call(
            "zero-limit-read",
            "read_file",
            serde_json::json!({"file_path": "src/lib.rs", "offset": 1, "limit": 0}),
        )),
        tool_result_message(
            "zero-limit-read",
            "... [truncated: showing lines 1-0 of 3; 3 more line(s) not shown. Continue with offset=1 to read the rest.]",
        ),
    ];
    let mut final_text = complete_report("src/lib.rs");

    assert_eq!(
        audit_evidence_gate_action(
            "audit",
            &mut messages,
            &turn_messages,
            &mut final_text,
            Some(&root),
            false,
            1,
            16,
        ),
        AuditEvidenceGateAction::Reopen
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn audit_rejects_reads_stale_after_a_direct_mutation() {
    let root = temporary_root("stale-read");
    write_fixture(&root);
    let mut messages = Vec::new();
    let mut turn_messages = successful_reads("src/lib.rs");
    let mut final_text = complete_report("src/lib.rs");
    turn_messages.push(assistant_tool_call_message(test_tool_call(
        "write-after-read",
        "write_file",
        serde_json::json!({"file_path": "src/lib.rs", "content": "updated\n"}),
    )));
    turn_messages.push(tool_result_message("write-after-read", "written"));

    assert_eq!(
        audit_evidence_gate_action(
            "audit",
            &mut messages,
            &turn_messages,
            &mut final_text,
            Some(&root),
            false,
            1,
            16,
        ),
        AuditEvidenceGateAction::Reopen
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn malformed_or_incomplete_reports_are_withheld_at_the_finalization_cap() {
    let root = temporary_root("withhold");
    write_fixture(&root);
    let mut messages = Vec::new();
    let mut final_text = complete_report("src/lib.rs");

    assert_eq!(
        audit_evidence_gate_action(
            "audit",
            &mut messages,
            &[],
            &mut final_text,
            Some(&root),
            true,
            16,
            16,
        ),
        AuditEvidenceGateAction::Warn
    );
    assert!(final_text.contains("No verified findings."));
    assert!(final_text.contains(AUDIT_EVIDENCE_WARNING));
    assert!(!final_text.contains("A verified defect"));

    let _ = fs::remove_dir_all(root);
}
