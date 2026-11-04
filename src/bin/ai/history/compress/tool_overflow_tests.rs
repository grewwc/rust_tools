use super::tool_overflow::{
    TOOL_RESULT_FOLD_MIN_CHARS, build_preserved_tool_overflow_stub_with_archive,
    is_preserved_tool_overflow_stub, task_evidence_review_lines,
};
use super::*;
use crate::ai::types::{FunctionCall, ToolCall};
use rustc_hash::FxHashSet;

/// Fallback wrapper: a zero caller threshold still observes the shared floor.
fn prepare_tool_messages_structured(
    messages: &mut [Message],
    max_chars_per_msg: usize,
    keep_recent_groups: usize,
    overflow_dir: Option<&std::path::Path>,
    cwd: Option<&std::path::Path>,
    protected_tool_call_ids: &FxHashSet<String>,
) {
    super::prepare_tool_messages_structured(
        messages,
        max_chars_per_msg,
        0,
        keep_recent_groups,
        overflow_dir,
        cwd,
        protected_tool_call_ids,
    );
}

#[test]
fn mid_turn_spill_floor_keeps_small_results_inline() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-tool-overflow-floor-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&overflow_dir).unwrap();
    // ~2.6 KB of plain document text: over the per-message cap, under the
    // mid-turn floor, and shaped like nothing the key-line index knows.
    let doc = (1..=42)
        .map(|i| format!("{i}: reference step body text without any code markers"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(doc.chars().count() > 480);
    assert!(doc.chars().count() < MID_TURN_SPILL_MIN_CHARS);
    assert!(doc.chars().count() < TOOL_RESULT_FOLD_MIN_CHARS);
    let mut messages = vec![
        assistant_call("old", "read_file"),
        tool_result("old", &doc),
        assistant_call("recent", "read_file"),
        tool_result("recent", "recent result"),
    ];

    super::prepare_tool_messages_structured(
        &mut messages,
        480,
        MID_TURN_SPILL_MIN_CHARS,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );

    assert_eq!(value_to_string(&messages[1].content), doc);
    // Exercise each fallback, including whole-group folding with no retention
    // window. A small but spillable preview-shaped read must stay verbatim.
    super::prepare_tool_messages_structured(
        &mut messages,
        480,
        0,
        0,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let protected = ["old".to_string()].into_iter().collect();
    enforce_protected_precision_group_budget(
        &mut messages,
        2,
        0,
        Some(&overflow_dir),
        None,
        &protected,
        true,
    );
    assert_eq!(
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected,),
        0
    );
    let (folded, count) =
        fold_early_tool_groups(&messages, 0, Some(&overflow_dir), &FxHashSet::default());
    assert_eq!(count, 0);
    assert_eq!(value_to_string(&folded[1].content), doc);
    let shrunk = shrink_messages_to_fit(messages, 1, Some(&overflow_dir), None, &protected);
    assert!(
        shrunk
            .iter()
            .any(|message| message.tool_call_id.as_deref() == Some("old")
                && value_to_string(&message.content) == doc)
    );
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    assert!(
        !archive_dir.exists() || std::fs::read_dir(&archive_dir).unwrap().count() == 0,
        "a below-floor result must not be archived"
    );
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn small_tool_results_survive_soft_compaction_and_automatic_caps() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-generic-truncation-floor-{}",
        uuid::Uuid::new_v4()
    ));
    for chars in [
        3_000,
        TOOL_RESULT_FOLD_MIN_CHARS - 1,
        TOOL_RESULT_FOLD_MIN_CHARS,
    ] {
        for name in ["read_file", "text_grep", "task_wait"] {
            let content = "文".repeat(chars);
            let original = vec![
                assistant_call("small", name),
                tool_result("small", &content),
            ];
            let protected = ["small".to_string()].into_iter().collect();
            for ids in [&FxHashSet::default(), &protected] {
                let mut direct = original[1].clone();
                assert!(!truncate_mutable_field(
                    &mut direct,
                    MutableMessageField::Content,
                    chars,
                    Some(&overflow_dir),
                    FieldArchivePolicy::BestEffort
                ));
                assert_eq!(value_to_string(&direct.content), content);
                let mut soft = original.clone();
                assert!(!truncate_unprotected_messages_to_fit(
                    &mut soft,
                    128,
                    Some(&overflow_dir),
                    ids
                ));
                let shrunk =
                    shrink_messages_to_fit(original.clone(), 128, Some(&overflow_dir), None, ids);
                for candidate in [soft, shrunk] {
                    assert!(
                        candidate
                            .iter()
                            .any(|message| message.tool_call_id.as_deref() == Some("small")
                                && value_to_string(&message.content) == content),
                        "{name}, {chars}: a generic backstop changed the result"
                    );
                }
            }
            let mut capped = original;
            assert_eq!(
                cap_oversized_tool_results_for_context(
                    &mut capped,
                    160,
                    0,
                    Some(&overflow_dir),
                    None
                ),
                0
            );
            assert_eq!(value_to_string(&capped[1].content), content);
        }
    }
    assert!(
        !overflow_dir.exists(),
        "small raw results must not create archives"
    );

    let content = "large result\n".repeat(500);
    for emergency in [false, true] {
        let mut candidate = vec![
            assistant_call("large", "text_grep"),
            tool_result("large", &content),
        ];
        if emergency {
            emergency_cap_messages_to_fit(
                &mut candidate,
                700,
                600,
                Some(&overflow_dir),
                &FxHashSet::default(),
            );
        } else {
            truncate_mutable_messages_to_fit(
                &mut candidate,
                700,
                Some(&overflow_dir),
                &FxHashSet::default(),
            );
        }
        let result = candidate
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("large"))
            .unwrap();
        assert!(value_to_string(&result.content).chars().count() < content.chars().count());
        assert!(value_to_string(&result.content).contains("archived at:"));
    }
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn accumulated_small_current_turn_results_fit_the_hard_budget() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-small-results-hard-budget-{}",
        uuid::Uuid::new_v4()
    ));
    let mut original = vec![Message {
        role: "user".to_string(),
        content: Value::from("Inspect these results."),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mut protected = FxHashSet::default();
    for index in 0..8 {
        let id = format!("small-{index}");
        protected.insert(id.clone());
        original.push(assistant_call_args(
            &id,
            "text_grep",
            &serde_json::json!({"pattern": format!("result-{index}")}).to_string(),
        ));
        original.push(tool_result(
            &id,
            &format!("result {index}: {}", "文".repeat(3_000)),
        ));
    }
    let budget = 12_000;
    assert!(messages_total_chars(&original) > budget);
    for ids in [&FxHashSet::default(), &protected] {
        let mut soft = original.clone();
        assert!(!truncate_unprotected_messages_to_fit(
            &mut soft,
            budget,
            Some(&overflow_dir),
            ids
        ));
        assert_eq!(
            soft, original,
            "soft compaction must still preserve the floor"
        );
        for path in ["hard", "emergency", "final"] {
            let mut candidate = original.clone();
            match path {
                "hard" => assert!(truncate_mutable_messages_to_fit(
                    &mut candidate,
                    budget,
                    Some(&overflow_dir),
                    ids
                )),
                "emergency" => {
                    emergency_cap_messages_to_fit(
                        &mut candidate,
                        budget,
                        TOOL_RESULT_FOLD_MIN_CHARS,
                        Some(&overflow_dir),
                        ids,
                    );
                }
                _ => {
                    candidate = shrink_messages_to_fit_with_summary(
                        candidate,
                        budget,
                        0,
                        Some(&overflow_dir),
                        None,
                        ids,
                    )
                }
            }
            assert!(messages_total_chars(&candidate) <= budget, "{path}");
            assert!(candidate.iter().any(|message| message.role == "user"
                && value_to_string(&message.content) == "Inspect these results."));
            for index in 0..8 {
                let id = format!("small-{index}");
                assert!(
                    candidate
                        .iter()
                        .any(|message| message.tool_call_id.as_deref() == Some(id.as_str())),
                    "{path}: {id}"
                );
                assert!(
                    candidate.iter().any(|message| message
                        .tool_calls
                        .as_ref()
                        .is_some_and(|calls| calls.iter().any(|call| call.id == id))),
                    "{path}: {id}"
                );
            }
            let oldest = candidate
                .iter()
                .find(|message| message.tool_call_id.as_deref() == Some("small-0"))
                .unwrap();
            assert!(
                value_to_string(&oldest.content).contains("archived at:"),
                "{path}"
            );
            let newest = candidate
                .iter()
                .find(|message| message.tool_call_id.as_deref() == Some("small-7"))
                .unwrap();
            assert_eq!(newest.content, original.last().unwrap().content, "{path}");
        }
    }
    let archive = std::fs::read_to_string(overflow_dir.join(OVERFLOW_HISTORY_FILENAME)).unwrap();
    assert!(archive.contains(&value_to_string(&original[2].content)));
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn synchronous_task_results_keep_uuid_review_and_recovery() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-sync-task-review-{}", uuid::Uuid::new_v4()));
    for task_id in [
        "0123456789abcdef0123456789abcdef",
        "01234567-89ab-cdef-0123-456789abcdef",
        "task_async",
    ] {
        let content = format!(
            "[task_id={task_id}]\n[Task: inspect via worker @ model] COMPLETED after 1.0s\nConclusion: verified synchronous task.\n{}\nFinal conclusion: retain recovery.",
            "supporting evidence\n".repeat(500)
        );
        let mut candidate = vec![
            assistant_call("sync", "task"),
            tool_result("sync", &content),
        ];
        prepare_tool_messages_structured(
            &mut candidate,
            480,
            0,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let stub = value_to_string(&candidate[1].content);
        assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
        assert!(stub.chars().count() < content.chars().count());
        for retained in [
            stub.clone(),
            collapse_overflow_stub_to_anchor(&stub).unwrap(),
            minimize_overflow_stub_to_pointer(&stub).unwrap(),
        ] {
            assert!(
                retained.contains(&format!("task_id={task_id}; status=completed")),
                "{retained}"
            );
            assert!(
                retained.contains("Conclusion: verified synchronous task."),
                "{retained}"
            );
            assert!(
                retained.contains("Final conclusion: retain recovery."),
                "{retained}"
            );
            assert!(
                retained.contains(&format!(
                    "task_evidence_read({})",
                    serde_json::json!({"task_id": task_id, "source": "delivered"})
                )),
                "{retained}"
            );
            assert_eq!(
                task_evidence_review_lines(&retained),
                task_evidence_review_lines(&content)
            );
        }
    }
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn large_progress_evidence_keeps_progress_recovery_through_compression() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-progress-evidence-review-{}",
        uuid::Uuid::new_v4()
    ));
    for explicit_source in [false, true] {
        let mut evidence = serde_json::json!({
            "version": 1, "task_id": "task_progress", "status": "running", "phase": "working",
            "checkpoint_excerpt": "Confirmed the running task path.", "checkpoint_revision": 2,
            "timeline": [{"summary": "supporting progress detail".repeat(300)}]
        });
        if explicit_source {
            evidence["source"] = Value::from("progress");
        }
        let content = evidence.to_string();
        assert!(content.chars().count() > TOOL_RESULT_FOLD_MIN_CHARS);
        let mut candidate = vec![
            assistant_call_args(
                "progress",
                "task_evidence_read",
                r#"{"task_id":"task_progress","source":"progress"}"#,
            ),
            tool_result("progress", &content),
        ];
        prepare_tool_messages_structured(
            &mut candidate,
            480,
            0,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let stub = value_to_string(&candidate[1].content);
        assert!(stub.chars().count() < content.chars().count(), "{stub}");
        assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
        let pointer = format!(
            "task_evidence_read({})",
            serde_json::json!({
                "task_id": "task_progress", "source": "progress"
            })
        );
        for retained in [
            stub.clone(),
            collapse_overflow_stub_to_anchor(&stub).unwrap(),
            minimize_overflow_stub_to_pointer(&stub).unwrap(),
        ] {
            assert!(retained.contains(&pointer), "{retained}");
            assert!(retained.contains("task_id=task_progress"), "{retained}");
            assert!(retained.contains("status=running"), "{retained}");
            assert!(
                retained.contains("Confirmed the running task path."),
                "{retained}"
            );
            assert!(!retained.contains(r#""source":"delivered""#), "{retained}");
        }
        let (folded, count) =
            fold_early_tool_groups(&candidate, 0, Some(&overflow_dir), &FxHashSet::default());
        assert_eq!(count, 1);
        assert!(value_to_string(&folded[0].content).contains(&pointer));
    }
    std::fs::remove_dir_all(overflow_dir).unwrap();
}

#[test]
fn large_task_evidence_compresses_with_review_and_delivery_recovery() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-task-evidence-review-{}", uuid::Uuid::new_v4()));
    for name in ["task_wait", "task_evidence_read"] {
        for status in ["completed", "failed", "timeout", "cancelled"] {
            let conclusion = format!(
                "Conclusion: {status} findings verified.\n{}\nFinal conclusion: retain the review evidence.\n{}",
                "supporting detail\n".repeat(700),
                crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER
            );
            let content = if name == "task_wait" {
                format!(
                    "[task_id=task_review]\n[Task: review via worker @ model] {} after 1.0s\n{conclusion}",
                    status.to_uppercase()
                )
            } else {
                serde_json::json!({"task_id": "task_review", "source": "delivered", "status": "delivered", "terminal_status": status, "payload": conclusion}).to_string()
            };
            assert!(content.chars().count() > TOOL_RESULT_FOLD_MIN_CHARS);
            assert!(content.chars().count() < super::persisted::TOOL_RESULT_RAW_HARD_CAP_CHARS);
            let original = vec![
                assistant_call_args(
                    "review",
                    name,
                    r#"{"task_id":"task_review","source":"delivered"}"#,
                ),
                tool_result("review", &content),
            ];
            for generic in [false, true] {
                let mut candidate = original.clone();
                if generic {
                    assert!(truncate_mutable_field(
                        &mut candidate[1],
                        MutableMessageField::Content,
                        content.chars().count(),
                        Some(&overflow_dir),
                        FieldArchivePolicy::BestEffort
                    ));
                } else {
                    prepare_tool_messages_structured(
                        &mut candidate,
                        480,
                        0,
                        Some(&overflow_dir),
                        None,
                        &FxHashSet::default(),
                    );
                }
                let stub = value_to_string(&candidate[1].content);
                assert!(stub.chars().count() < content.chars().count(), "{stub}");
                assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
                for retained in [
                    stub.clone(),
                    collapse_overflow_stub_to_anchor(&stub).unwrap(),
                    minimize_overflow_stub_to_pointer(&stub).unwrap(),
                ] {
                    assert!(retained.contains("task_id=task_review"), "{retained}");
                    assert!(retained.contains(&format!("status={status}")), "{retained}");
                    assert!(retained.contains("Conclusion:"), "{retained}");
                    assert!(retained.contains("Final conclusion:"), "{retained}");
                    assert!(
                        retained.contains(&format!(
                            "task_evidence_read({})",
                            serde_json::json!({"task_id": "task_review", "source": "delivered"})
                        )),
                        "{retained}"
                    );
                }
                let (folded, count) = fold_early_tool_groups(
                    &candidate,
                    0,
                    Some(&overflow_dir),
                    &FxHashSet::default(),
                );
                assert_eq!(count, 1);
                let recall = value_to_string(&folded[0].content);
                assert!(
                    recall.contains(&format!("status={status}"))
                        && recall.contains("task_evidence_read(")
                        && recall.contains("Conclusion:"),
                    "{recall}"
                );
                let archived = std::fs::read_to_string(resolve_stub_archive_path(
                    stub.lines()
                        .find_map(|line| line.strip_prefix("- file_path: "))
                        .unwrap(),
                    &overflow_dir,
                ))
                .unwrap();
                assert!(
                    archived.contains(&content),
                    "archive must retain the full result"
                );
            }
        }
    }
    let aggregate = serde_json::json!({"tasks": [
        {"runtime_task_id": "task_first", "state": "completed", "output": "first conclusion"},
        {"pool_task_id": "task_second", "status": "failed", "output": "second conclusion"}
    ]})
    .to_string();
    let stub = build_preserved_tool_overflow_stub(
        &overflow_dir.join("aggregate.txt"),
        "task_wait",
        &aggregate,
        &[],
    );
    let minimal = minimize_overflow_stub_to_pointer(&stub).unwrap();
    for expected in [
        "- task_status: task_id=task_first; status=completed",
        "- task_status: task_id=task_second; status=failed",
        "- task_conclusion: task_id=task_first; first conclusion",
        "- task_conclusion: task_id=task_second; second conclusion",
    ] {
        assert!(minimal.contains(expected), "{minimal}");
    }
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn archive_unavailable_task_evidence_stub_preserves_each_review_across_passes() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-task-evidence-unavailable-{}",
        uuid::Uuid::new_v4()
    ));
    // Enough bounded task reviews remain above the floor to exercise the
    // terminal-stub guard independently of the raw-result size guard.
    let content = (0..12)
        .map(|index| {
            let status = if index % 2 == 0 { "completed" } else { "failed" };
            format!(
                "[task_id=task_review_{index}]\n[Task: review via worker @ model] {} after 1.0s\nConclusion: {status} findings for task_review_{index}.\n{}\nFinal conclusion: retain task_review_{index} evidence.\n{}",
                status.to_uppercase(),
                "supporting detail\n".repeat(400),
                crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let expected_reviews = task_evidence_review_lines(&content);
    let legacy_fallback = format!(
        "Task evidence folded for context budget.\n{}",
        expected_reviews.join("\n")
    );
    assert!(legacy_fallback.chars().count() > TOOL_RESULT_FOLD_MIN_CHARS);
    assert_eq!(
        task_evidence_review_lines(&legacy_fallback),
        expected_reviews
    );

    // A file in place of the sink directory forces an actual archive failure.
    std::fs::write(&overflow_dir, "unavailable archive directory").unwrap();
    for sink in [None, Some(overflow_dir.as_path())] {
        let mut messages = vec![
            assistant_call("review", "task_wait"),
            tool_result("review", &content),
        ];
        assert!(truncate_mutable_field(
            &mut messages[1],
            MutableMessageField::Content,
            content.chars().count(),
            sink,
            FieldArchivePolicy::BestEffort,
        ));
        let first_stub = value_to_string(&messages[1].content);
        assert!(first_stub.chars().count() > TOOL_RESULT_FOLD_MIN_CHARS);
        assert!(first_stub.chars().count() < content.chars().count());
        assert!(is_preserved_tool_overflow_stub(&first_stub), "{first_stub}");
        assert!(first_stub.contains("- recovery: canonical history;"));
        assert!(!first_stub.contains("- file_path: "));
        assert!(!first_stub.contains("Full historical result saved to session asset:"));
        assert_eq!(task_evidence_review_lines(&first_stub), expected_reviews);
        for index in 0..12 {
            let status = if index % 2 == 0 {
                "completed"
            } else {
                "failed"
            };
            assert!(first_stub.contains(&format!(
                "- task_status: task_id=task_review_{index}; status={status}"
            )));
            assert!(first_stub.contains(&format!(
                "Conclusion: {status} findings for task_review_{index}."
            )));
            assert!(first_stub.contains(&format!(
                "Final conclusion: retain task_review_{index} evidence."
            )));
        }
        let minimal = minimize_overflow_stub_to_pointer(&first_stub).unwrap();
        assert!(is_preserved_tool_overflow_stub(&minimal));
        assert!(minimal.contains("- recovery: canonical history;"));
        assert!(!minimal.contains("- file_path: "));
        assert_eq!(task_evidence_review_lines(&minimal), expected_reviews);

        prepare_tool_messages_structured(&mut messages, 480, 0, sink, None, &FxHashSet::default());
        assert!(!truncate_mutable_field(
            &mut messages[1],
            MutableMessageField::Content,
            first_stub.chars().count(),
            sink,
            FieldArchivePolicy::BestEffort,
        ));
        let second_stub = value_to_string(&messages[1].content);
        assert_eq!(second_stub, first_stub);
        assert_eq!(task_evidence_review_lines(&second_stub), expected_reviews);
    }
    assert!(overflow_dir.is_file());
    std::fs::remove_file(overflow_dir).unwrap();
}

/// A result may name arbitrarily many tasks; the review block appended to a
/// preserved stub must stay bounded and say how much it dropped.
#[test]
fn task_evidence_review_lines_bound_many_task_ids() {
    let task_count = 200;
    let content = (0..task_count)
        .map(|index| {
            format!(
                "[task_id=task_many_{index}]\n[Task: bulk review @ model] COMPLETED after 1.0s\nConclusion: detail for task_many_{index}.\nFinal conclusion: retain bulk evidence."
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let lines = task_evidence_review_lines(&content);
    assert_eq!(lines, task_evidence_review_lines(&content), "deterministic");
    assert!(lines.len() <= 49, "line cap: {}", lines.len());
    let emitted_chars: usize = lines.iter().map(|line| line.chars().count() + 1).sum();
    assert!(emitted_chars <= 8 * 1024 + 64, "char cap: {emitted_chars}");

    let marker = lines.last().expect("non-empty review block");
    let remaining: usize = marker
        .strip_prefix("- review lines truncated: ")
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|count| count.parse().ok())
        .unwrap_or_else(|| panic!("expected truncation marker, got {marker}"));
    let emitted_tasks = lines
        .iter()
        .filter(|line| line.starts_with("- task_evidence: "))
        .count();
    assert_eq!(emitted_tasks + remaining, task_count);

    // The same bound keeps a freshly built stub recognizable and well under the
    // 64 KiB physical cap.
    let stub = build_preserved_tool_overflow_stub(
        Path::new("/tmp/many-tasks.txt"),
        "task_wait",
        &content,
        &[],
    );
    assert!(is_preserved_tool_overflow_stub(&stub));
    assert!(
        stub.chars().count() < 24_000,
        "stub size: {}",
        stub.chars().count()
    );
}

#[test]
fn authentic_preserved_stub_round_trips_recognition() {
    let full = "line\n".repeat(200);
    for path in [Some(Path::new("/tmp/session/round-trip.txt")), None] {
        let stub = build_preserved_tool_overflow_stub_with_archive(path, "read_file", &full, &[]);
        assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
        let annotated = format!("Recovered tool output:\n```text\n  {stub}\n```");
        assert!(is_preserved_tool_overflow_stub(&annotated), "{annotated}");
        if path.is_some() {
            let anchor = collapse_overflow_stub_to_anchor(&stub).expect("collapse");
            assert!(is_preserved_tool_overflow_stub(&anchor), "{anchor}");
            let pointer = minimize_overflow_stub_to_pointer(&stub).expect("minimize");
            assert!(is_preserved_tool_overflow_stub(&pointer), "{pointer}");
        }
    }
}

#[test]
fn archived_task_evidence_above_floor_keeps_each_review_field() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-task-evidence-archive-review-{}",
        uuid::Uuid::new_v4()
    ));
    let content = serde_json::json!({"tasks": [
        {"runtime_task_id": "task_first", "state": "completed", "output": format!(
            "Conclusion: first task completed.\n{}\nFinal conclusion: keep first findings.",
            "first supporting detail\n".repeat(400)
        )},
        {"pool_task_id": "task_second", "status": "failed", "output": format!(
            "Conclusion: second task failed.\n{}\nFinal conclusion: keep second failure evidence.",
            "second supporting detail\n".repeat(400)
        )}
    ]})
    .to_string();
    assert!(content.chars().count() > TOOL_RESULT_FOLD_MIN_CHARS);
    let expected_reviews = task_evidence_review_lines(&content);
    let mut messages = vec![
        assistant_call("review", "task_wait"),
        tool_result("review", &content),
    ];
    prepare_tool_messages_structured(
        &mut messages,
        480,
        0,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let stub = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
    assert!(stub.chars().count() < content.chars().count());
    let file_path = stub
        .lines()
        .find_map(|line| line.strip_prefix("- file_path: "))
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(resolve_stub_archive_path(file_path, &overflow_dir)).unwrap(),
        content
    );
    for retained in [
        stub.clone(),
        collapse_overflow_stub_to_anchor(&stub).unwrap(),
        minimize_overflow_stub_to_pointer(&stub).unwrap(),
    ] {
        assert_eq!(task_evidence_review_lines(&retained), expected_reviews);
        for expected in [
            "- task_status: task_id=task_first; status=completed",
            "- task_status: task_id=task_second; status=failed",
            "- task_conclusion: task_id=task_first; Conclusion: first task completed.",
            "- task_conclusion: task_id=task_second; Conclusion: second task failed.",
            "Final conclusion: keep first findings.",
            "Final conclusion: keep second failure evidence.",
        ] {
            assert!(retained.contains(expected), "{retained}");
        }
        for task_id in ["task_first", "task_second"] {
            assert!(retained.contains(&format!(
                "task_evidence_read({})",
                serde_json::json!({"task_id": task_id, "source": "delivered"})
            )));
        }
    }
    std::fs::remove_dir_all(overflow_dir).unwrap();
}

#[test]
fn tool_result_quoting_an_old_stub_is_archived_as_new_evidence() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-quoted-overflow-stub-{}", uuid::Uuid::new_v4()));
    let old_path = write_preserved_tool_overflow_file_stable(
        &overflow_dir,
        "old",
        "task_wait",
        "old evidence",
    )
    .unwrap();
    let old_stub = build_preserved_tool_overflow_stub(&old_path, "task_wait", "old evidence", &[]);
    let content = format!(
        "[task_id=task_new]\nNew findings that need review:\n{}\nQuoted prior stub:\n{old_stub}\nNew conclusion after the quote.",
        "new evidence detail\n".repeat(500)
    );
    assert!(!is_preserved_tool_overflow_stub(&content));
    assert!(collapse_overflow_stub_to_anchor(&content).is_none());
    assert!(minimize_overflow_stub_to_pointer(&content).is_none());
    let annotation_with_findings =
        format!("Recovered tool output:\n```text\n{old_stub}\n```\nNew findings follow.");
    assert!(!is_preserved_tool_overflow_stub(&annotation_with_findings));
    let original = vec![
        assistant_call("new", "task_wait"),
        tool_result("new", &content),
    ];
    let mut prepared = original.clone();
    prepare_tool_messages_structured(
        &mut prepared,
        480,
        0,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let stub = value_to_string(&prepared[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub));
    let path = resolve_stub_archive_path(
        stub.lines()
            .find_map(|line| line.strip_prefix("- file_path: "))
            .unwrap(),
        &overflow_dir,
    );
    assert_ne!(path, old_path);
    assert_eq!(std::fs::read_to_string(path).unwrap(), content);
    let (folded, count) =
        fold_early_tool_groups(&original, 0, Some(&overflow_dir), &FxHashSet::default());
    assert_eq!(count, 1);
    assert!(value_to_string(&folded[0].content).contains("task_id=task_new"));
    assert!(
        std::fs::read_dir(overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR))
            .unwrap()
            .any(|entry| std::fs::read_to_string(entry.unwrap().path()).unwrap() == content)
    );
    assert_eq!(std::fs::read_to_string(old_path).unwrap(), "old evidence");
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn mid_turn_spill_floor_still_folds_results_above_it() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-tool-overflow-floor-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&overflow_dir).unwrap();
    let big = "z".repeat(MID_TURN_SPILL_MIN_CHARS + 1);
    let mut messages = vec![
        assistant_call("old", "read_file"),
        tool_result("old", &big),
        assistant_call("recent", "read_file"),
        tool_result("recent", "recent result"),
    ];

    super::prepare_tool_messages_structured(
        &mut messages,
        480,
        MID_TURN_SPILL_MIN_CHARS,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );

    let stub = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
}

#[test]
fn overflow_preview_probes_the_omitted_middle_without_format_assumptions() {
    // Plain document lines: none of them match the code-shaped key-line index,
    // so the probes are the only anchors showing that the body continues.
    let doc = (1..=42)
        .map(|i| format!("{i}. document section {i} body text"))
        .collect::<Vec<_>>()
        .join("\n");
    let preview = build_overflow_content_preview(&doc);

    assert!(preview.contains("- middle_samples ("), "{preview}");
    // The first and last omitted lines are always probed. Content without read_file
    // line numbers is labeled by 0-based index (`#<idx>`), never as `L<idx>`, which
    // reads like a line address the preview cannot resolve.
    assert!(preview.contains("  #4: "), "{preview}");
    assert!(preview.contains("  #39: "), "{preview}");
    assert!(!preview.contains("  L"), "{preview}");
    // Deterministic: identical content yields a byte-identical preview.
    assert_eq!(preview, build_overflow_content_preview(&doc));

    // Short content keeps the legacy head+tail-only shape.
    let short_preview = build_overflow_content_preview("a\nb\nc\nd\ne\nf");
    assert!(!short_preview.contains("middle_samples"), "{short_preview}");
    assert_eq!(short_preview.lines().count(), 7);
}

fn assistant_call(id: &str, name: &str) -> Message {
    assistant_call_args(id, name, "{}")
}

fn assistant_call_args(id: &str, name: &str, arguments: &str) -> Message {
    Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn tool_result(id: &str, content: &str) -> Message {
    Message {
        role: "tool".to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    }
}

#[test]
fn preserved_tool_overflow_stub_is_not_spilled_again() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-tool-overflow-stub-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![
        assistant_call("old", "read_file"),
        tool_result("old", &"x".repeat(8_000)),
        assistant_call("recent", "read_file"),
        tool_result("recent", "recent result"),
    ];

    prepare_tool_messages_structured(
        &mut messages,
        80,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let first_stub = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&first_stub));
    let overflow_path = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

    prepare_tool_messages_structured(
        &mut messages,
        80,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    assert_eq!(value_to_string(&messages[1].content), first_stub);
    assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

    let annotated = format!("Recovered tool output:\n```text\n  {first_stub}\n```");
    assert!(is_preserved_tool_overflow_stub(&annotated));
    messages[1].content = Value::String(annotated.clone());
    prepare_tool_messages_structured(
        &mut messages,
        80,
        0,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let protected = ["old".to_string()].into_iter().collect();
    assert_eq!(
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected,),
        0
    );
    assert_eq!(
        cap_oversized_tool_results_for_context(&mut messages, 80, 0, Some(&overflow_dir), None,),
        0
    );
    assert_eq!(value_to_string(&messages[1].content), annotated);
    assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);
    let wrapped = serde_json::json!([{"type": "text", "text": annotated}]);
    assert!(super::tool_overflow::is_preserved_tool_overflow_content(
        &wrapped
    ));
    messages[1].content = wrapped.clone();
    prepare_tool_messages_structured(
        &mut messages,
        80,
        0,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    assert_eq!(messages[1].content, wrapped);
    assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn pruned_stable_archive_is_content_addressed() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-pruned-stable-content-addressed-{}",
        uuid::Uuid::new_v4()
    ));

    // Same (tool, id, content) must map to one idempotent archive file.
    let first = write_preserved_tool_overflow_file_stable(
        &overflow_dir,
        "call-1",
        "read_file",
        "result body",
    )
    .unwrap();
    let second = write_preserved_tool_overflow_file_stable(
        &overflow_dir,
        "call-1",
        "read_file",
        "result body",
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(std::fs::read_to_string(&first).unwrap(), "result body");

    // Same call id with different content must NOT reuse the old file:
    // a replayed tool call would otherwise read back stale bytes.
    let replayed = write_preserved_tool_overflow_file_stable(
        &overflow_dir,
        "call-1",
        "read_file",
        "replayed body",
    )
    .unwrap();
    assert_ne!(first, replayed);
    assert_eq!(std::fs::read_to_string(&replayed).unwrap(), "replayed body");
    let overflow_path = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 2);

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn preserved_read_file_overflow_stub_keeps_original_target_anchor() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-tool-overflow-read-anchor-{}",
        uuid::Uuid::new_v4()
    ));
    let mut messages = vec![
        assistant_call_args(
            "old",
            "read_file",
            r#"{"file_path":"src/lib.rs","offset":120,"limit":40}"#,
        ),
        tool_result("old", &"x".repeat(8_000)),
        assistant_call("recent", "read_file"),
        tool_result("recent", "recent result"),
    ];

    prepare_tool_messages_structured(
        &mut messages,
        80,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let stub = value_to_string(&messages[1].content);
    assert!(
        stub.contains("- original_file_path: src/lib.rs"),
        "stub: {stub}"
    );
    assert!(
        stub.contains("- original_range: lines=120..159"),
        "stub: {stub}"
    );
    assert!(
        stub.contains("Archived snapshot of an earlier read"),
        "stub: {stub}"
    );
    assert!(
        stub.contains("folded out of inline context for budget reasons"),
        "stub should explain the context-budget cause: {stub}"
    );
    assert!(
        stub.contains("read a smaller range of `original_file_path`"),
        "stub should steer recovery toward bounded reads: {stub}"
    );

    let anchor = collapse_overflow_stub_to_anchor(&stub).expect("stub should collapse");
    assert!(
        anchor.contains("- original_file_path: src/lib.rs"),
        "anchor: {anchor}"
    );
    assert!(
        anchor.contains("Archived snapshot of an earlier read"),
        "anchor: {anchor}"
    );
    assert!(
        anchor.contains("split large patches instead of re-reading the archive"),
        "anchor should keep actionable budget guidance: {anchor}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn preserved_stub_preview_includes_line_numbered_key_lines() {
    // read_file output carries a `{:>6}\t` line-number prefix: key_lines
    // should parse the prefix and use real line numbers for L tags (rather
    // than failing every match because of the prefix), so long files remain
    // locatable by line number after being spilled.
    let content = "\
 1\tuse std::fmt;\n\
 2\t\n\
 3\tpub fn main() {\n\
 4\t    let x = 1;\n\
 5\t}\n\
 6\tfn helper() {}\n\
 7\t//! crate docs\n\
 8\tstruct Foo;\n";
    let preview = build_overflow_content_preview(content);
    assert!(preview.contains("- key_lines (5):"), "preview: {preview}");
    assert!(preview.contains("L1: use std::fmt;"), "preview: {preview}");
    assert!(preview.contains("L3: pub fn main()"), "preview: {preview}");
    assert!(preview.contains("L6: fn helper()"), "preview: {preview}");
    assert!(preview.contains("L8: struct Foo;"), "preview: {preview}");
}

#[test]
fn preserved_stub_preview_tags_diff_context_as_unchanged() {
    // A preview index that drops the marker column renders a pre-existing context
    // line exactly like part of the change: the context marker is a single space and
    // survives no trim, while only `+`/`-` lines were ever matched as key lines.
    // Joined explicitly: a `\`-continued string literal strips the next line's
    // indentation, so the fixture would hold markerless lines instead of the real
    // leading-space context form.
    let diff = [
        "diff --git a/src/x.rs b/src/x.rs",
        "index 1111111..2222222 100644",
        "--- a/src/x.rs",
        "+++ b/src/x.rs",
        "@@ -63,6 +64,10 @@ pub(crate) struct Body {",
        "     /// Pre-existing doc comment.",
        "     #[serde(skip)]",
        "     pub(crate) estimated: usize,",
        "+    /// Added field doc comment.",
        "+    pub(crate) added: usize,",
        "-    /// Removed field doc comment.",
        "}",
    ]
    .join("\n");
    let preview = build_overflow_content_preview(&diff);
    assert!(
        preview.contains("#5: [ctx] /// Pre-existing doc comment."),
        "preview: {preview}"
    );
    assert!(
        preview.contains("#8: [+] /// Added field doc comment."),
        "added lines must still be indexed: {preview}"
    );
    assert!(
        preview.contains("#10: [-] /// Removed field doc comment."),
        "preview: {preview}"
    );
    // File and hunk headers name files; they must not read as added/removed lines.
    assert!(
        !preview.contains("[+] +++ b/src/x.rs"),
        "preview: {preview}"
    );
    assert!(!preview.contains("[+] @@"), "preview: {preview}");

    // Combined diffs (`git diff --cc`, merge `git show`) open a hunk with `@@@`.
    let combined = [
        "diff --cc src/x.rs",
        "@@@ -1,3 -1,3 +1,3 @@@",
        "   /// Pre-existing doc comment.",
        "+  /// Added doc comment.",
    ]
    .join("\n");
    let combined_preview = build_overflow_content_preview(&combined);
    assert!(
        combined_preview.contains("[ctx] /// Pre-existing doc comment."),
        "combined diffs still tag their context lines: {combined_preview}"
    );
    assert!(
        combined_preview.contains("[+] /// Added doc comment."),
        "combined diffs still tag their added lines: {combined_preview}"
    );

    // The first and last omitted lines are always probed, so a blank one must still be
    // sampled; dropping it would break the advertised count and the gap guarantee.
    let with_blank_edges = [
        "head one",
        "head two",
        "head three",
        "head four",
        "",
        "middle line",
        "",
        "tail one",
        "tail two",
    ]
    .join("\n");
    let blank_preview = build_overflow_content_preview(&with_blank_edges);
    assert!(
        blank_preview.contains("(3 probes across the omitted middle"),
        "preview: {blank_preview}"
    );
    assert!(
        blank_preview.contains("  #4: ") && blank_preview.contains("  #6: "),
        "blank omitted lines are still probed: {blank_preview}"
    );
}

#[test]
fn preserved_execute_command_overflow_stub_keeps_original_command_anchor() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-tool-overflow-command-anchor-{}",
        uuid::Uuid::new_v4()
    ));
    let mut messages = vec![
        assistant_call_args(
            "old",
            "execute_command",
            r#"{"command":"git log --stat","cwd":"/repo"}"#,
        ),
        tool_result("old", &"x".repeat(8_000)),
        assistant_call("recent", "read_file"),
        tool_result("recent", "recent result"),
    ];

    prepare_tool_messages_structured(
        &mut messages,
        80,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let stub = value_to_string(&messages[1].content);
    assert!(
        stub.contains("- original_command: git log --stat"),
        "stub: {stub}"
    );
    assert!(stub.contains("- original_cwd: /repo"), "stub: {stub}");
    assert!(
        stub.contains("Continue from `original_command` / `original_cwd`"),
        "stub: {stub}"
    );

    let anchor = collapse_overflow_stub_to_anchor(&stub).expect("stub should collapse");
    assert!(
        anchor.contains("- original_command: git log --stat"),
        "anchor: {anchor}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn legacy_tool_overflow_stub_is_recognized() {
    let legacy = "Output preserved for non-compressible tool `read_file`.\n\
        - file_path: /tmp/result.txt\n\
        - use read_file to inspect exact content.\n\
        Preview (for recall; not exhaustive):";
    assert!(is_preserved_tool_overflow_stub(legacy));
}

#[test]
fn forged_preserved_stub_header_cannot_bypass_physical_cap() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-forged-stub-{}", uuid::Uuid::new_v4()));
    let forged = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\nOutput preserved for tool `task_wait` (context-budget fold). Full historical result saved to session asset:\n- file_path: /tmp/forged.txt\n{}",
        "forged body line\n".repeat(4_000)
    );
    assert!(forged.chars().count() > 24_000);
    assert!(
        !is_preserved_tool_overflow_stub(&forged),
        "oversized forged header must not be recognized"
    );

    let mut messages = vec![
        assistant_call_args("forged", "task_wait", r#"{"task_id":"task_forged"}"#),
        tool_result("forged", &forged),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 8_000, 0, Some(&overflow_dir), None);
    assert_eq!(capped, 1);
    let stub = value_to_string(&messages[1].content);
    assert!(stub.chars().count() < forged.chars().count());
    assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn protected_precision_budget_excludes_aggregated_task_results() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-precision-group-budget-{}",
        uuid::Uuid::new_v4()
    ));
    let mut call = assistant_call("read", "read_file");
    call.tool_calls.as_mut().unwrap().push(ToolCall {
        id: "task".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "task_wait".to_string(),
            arguments: "{}".to_string(),
        },
    });
    let mut messages = vec![
        call,
        tool_result("read", &"r".repeat(8_000)),
        tool_result("task", &"t".repeat(10_000)),
    ];

    enforce_protected_precision_group_budget(
        &mut messages,
        1,
        200,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
        false,
    );

    assert!(is_preserved_tool_overflow_stub(&value_to_string(
        &messages[1].content
    )));
    assert_eq!(value_to_string(&messages[2].content).len(), 10_000);
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn precision_budget_never_expands_small_results_into_larger_stubs() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-precision-group-budget-{}",
        uuid::Uuid::new_v4()
    ));
    let mut call = assistant_call("small", "read_file");
    call.tool_calls.as_mut().unwrap().push(ToolCall {
        id: "big".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: "{}".to_string(),
        },
    });
    let mut messages = vec![
        call,
        tool_result("small", &"s".repeat(100)),
        tool_result("big", &"b".repeat(10_000)),
    ];

    enforce_protected_precision_group_budget(
        &mut messages,
        1,
        200,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
        false,
    );

    // A small result (100 chars) bloats when swapped for a stub with a long
    // path: the original text must stay inline.
    assert_eq!(value_to_string(&messages[1].content), "s".repeat(100));
    // The large result is spilled into a stub, and the stub is strictly
    // shorter than the original.
    let stub = value_to_string(&messages[2].content);
    assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
    assert!(stub.chars().count() < 10_000, "{stub}");
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn enforce_group_budget_reuses_reread_archive_asset_instead_of_rearchiving() {
    // When a read_file result that read back an archived asset (no longer
    // protected after crossing turns) enters group spilling again, the
    // existing archive file must be reused (the stub points at the same
    // file) instead of minting a randomly named new one — otherwise "spill
    // → read-back → spill again" generates a new archive on every
    // read-back, forming an unbounded chain where the model never gets
    // stable content.
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-precision-group-reuse-{}", uuid::Uuid::new_v4()));
    // 1) Generate an archive asset via Path C
    let mut messages = vec![
        assistant_call("spill", "read_file"),
        // Leave ample headroom in the payload: the fingerprint adds a fixed
        // ~16-byte overhead to the stub, and if the original and stub sizes
        // were extremely close the anti-bloat guard (stub>=original) could
        // flip, making this test about reuse semantics rather than byte
        // coincidence.
        tool_result("spill", &"x".repeat(8_000)),
        assistant_call("recent", "read_file"),
        tool_result("recent", "recent result"),
    ];
    let mut protected = FxHashSet::default();
    protected.insert("spill".to_string());
    let stub1 =
        spill_protected_precision_to_fit(&mut messages, 80, Some(&overflow_dir), None, &protected);
    assert!(stub1 > 0);
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archive_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let raw = std::fs::read_to_string(&archive_path).unwrap();

    // 2) The model reads the archive back: the result exceeds the spill floor and
    // the group inline budget → triggers the enforce spill
    let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
    let mut messages = vec![
        assistant_call_args("re-read", "read_file", &read_args.to_string()),
        tool_result("re-read", &raw),
    ];
    enforce_protected_precision_group_budget(
        &mut messages,
        1,
        120,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
        false,
    );

    // 3) The existing asset is reused: the directory still has exactly 1
    // file, and the stub points at the same archive_path
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    let stub_text = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
    assert!(
        stub_text.contains(archive_path.to_str().unwrap()),
        "{stub_text}"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn cap_oversized_reuses_reread_archive_asset_instead_of_rearchiving() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-cap-reread-reuse-{}", uuid::Uuid::new_v4()));
    // 1) First let the cap itself write an archive asset
    let mut messages = vec![
        assistant_call("first", "read_file"),
        tool_result("first", &"y".repeat(70_000)),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, 0, Some(&overflow_dir), None);
    assert!(capped > 0);
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archive_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let raw = std::fs::read_to_string(&archive_path).unwrap();

    // 2) The model reads the archive back (body 70k > the 64k hard cap) →
    // the existing file is reused, no new file written
    let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
    let mut messages = vec![
        assistant_call_args("re-read", "read_file", &read_args.to_string()),
        tool_result("re-read", &raw),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, 0, Some(&overflow_dir), None);
    assert_eq!(capped, 1);
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    let stub_text = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
    assert!(
        stub_text.contains(archive_path.to_str().unwrap()),
        "{stub_text}"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn prepare_structured_reuses_reread_archive_asset_instead_of_rearchiving() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-prepare-reread-reuse-{}", uuid::Uuid::new_v4()));
    // 1) First let prepare write an archive asset (an old read_file result,
    // above the shared spill floor and outside the tail window)
    let mut messages = vec![
        assistant_call("first", "read_file"),
        tool_result("first", &"z".repeat(8_000)),
    ];
    prepare_tool_messages_structured(
        &mut messages,
        480,
        0,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archive_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let raw = std::fs::read_to_string(&archive_path).unwrap();

    // 2) The model reads the archive back and the result enters prepare
    // again (unprotected, outside the tail window) → the existing file is
    // reused
    let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
    let mut messages = vec![
        assistant_call_args("re-read", "read_file", &read_args.to_string()),
        tool_result("re-read", &raw),
    ];
    prepare_tool_messages_structured(
        &mut messages,
        480,
        0,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    let stub_text = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
    assert!(
        stub_text.contains(archive_path.to_str().unwrap()),
        "{stub_text}"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn prepare_structured_spill_is_deterministic_across_reprojections() {
    // P1 regression: when the same canonical tool result is spilled in two
    // independent projections, it must map to the same deterministic
    // archive file rather than minting a randomly named new copy every
    // round (the old behavior caused unbounded bloat within one session:
    // 368 files for only 211 unique contents). Using **different**
    // overflow_dirs for the two projections would hide idempotence, so the
    // same dir is reused to simulate round-by-round compaction of one
    // session.
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-prepare-deterministic-spill-{}",
        uuid::Uuid::new_v4()
    ));
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    // A single very long line: after line truncation the preview makes the
    // stub significantly smaller than the original → it does spill (the
    // anti-bloat guard does not trigger).
    let big = "b".repeat(8_000);
    let build = || {
        vec![
            assistant_call("spill", "read_file"),
            tool_result("spill", &big),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ]
    };

    let mut first = build();
    prepare_tool_messages_structured(
        &mut first,
        480,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let first_stub = value_to_string(&first[1].content);
    assert!(is_preserved_tool_overflow_stub(&first_stub), "{first_stub}");
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);

    // Second projection: the same canonical result (same tool_call_id +
    // same body) is compacted again. Deterministic naming → the existing
    // file is hit, no new copy is added, and the stub text stays stable
    // across rounds.
    let mut second = build();
    prepare_tool_messages_structured(
        &mut second,
        480,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let second_stub = value_to_string(&second[1].content);
    assert_eq!(
        second_stub, first_stub,
        "重投影后 stub 文本必须稳定（prompt cache 不断裂）"
    );
    assert_eq!(
        std::fs::read_dir(&archive_dir).unwrap().count(),
        1,
        "同一结果重复外溢不得铸造新归档文件"
    );

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn prepare_structured_keeps_small_multiline_result_inline() {
    // P2 regression: a few-hundred-byte multi-line grep result (over
    // max_chars_per_msg but still small) becomes larger when swapped for a
    // stub with a full head/tail preview. The anti-bloat guard must keep
    // the original inline and write no archive file — otherwise the model
    // sees "evicted, please re-read" and reads back repeatedly (session
    // 9f4d0fae's "read results kept being archived as stubs" was exactly
    // this path: a 673-char grep swapped for a larger stub).
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-prepare-small-inline-{}", uuid::Uuid::new_v4()));
    // 20 lines × ~30 chars ≈ 600 chars: over max_chars_per_msg=480, but the
    // preview contains the whole body verbatim, so the stub cannot be
    // smaller than the original.
    let grep_like = (0..20)
        .map(|i| format!("src/bin/ai/mod.rs:{i}: use crate::ai::x;"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(grep_like.chars().count() > 480);
    let mut messages = vec![
        assistant_call("grep", "execute_command"),
        tool_result("grep", &grep_like),
        assistant_call("recent", "read_file"),
        tool_result("recent", "recent result"),
    ];

    prepare_tool_messages_structured(
        &mut messages,
        480,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );

    let content = value_to_string(&messages[1].content);
    assert_eq!(content, grep_like, "小的多行精确结果必须保留原文内联");
    assert!(
        !is_preserved_tool_overflow_stub(&content),
        "不应被换成 stub"
    );
    // No archive file is written (on bloat the new file is deleted; here it
    // should never have been written at all).
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archived = std::fs::read_dir(&archive_dir)
        .map(|d| d.count())
        .unwrap_or(0);
    assert_eq!(archived, 0, "膨胀结果不得留下归档文件");

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn cap_reuses_execute_command_cat_archive_instead_of_rearchiving() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-cap-cat-reuse-{}", uuid::Uuid::new_v4()));
    // 1) First let the cap write an execute_command archive asset
    let mut messages = vec![
        assistant_call("run", "execute_command"),
        tool_result("run", &"log line\n".repeat(30_000)),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, 0, Some(&overflow_dir), None);
    assert!(capped > 0);
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archive_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let raw = std::fs::read_to_string(&archive_path).unwrap();

    // 2) The model reads the archive body back with `cat <archive>` (over
    // the hard cap) → the existing file is reused
    let run_args = serde_json::json!({
        "command": format!("cat {}", archive_path.to_string_lossy()),
        "pty": false,
    });
    let mut messages = vec![
        assistant_call_args("re-cat", "execute_command", &run_args.to_string()),
        tool_result("re-cat", &raw),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, 0, Some(&overflow_dir), None);
    assert_eq!(capped, 1);
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    let stub_text = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
    assert!(
        stub_text.contains(archive_path.to_str().unwrap()),
        "{stub_text}"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn cap_does_not_reuse_input_archive_for_transformed_command_output() {
    // A command that merely *mentions* an archive (awk/diff/pipes/multi-input
    // cat) must not reuse that input file for its transformed output: the
    // stub would otherwise mislabel input bytes as the full historical
    // result. Only byte-identical read-backs (plain `cat`, see the test
    // above) may reuse.
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-cap-transform-no-reuse-{}",
        uuid::Uuid::new_v4()
    ));
    let mut messages = vec![
        assistant_call("run", "execute_command"),
        tool_result("run", &"log line\n".repeat(30_000)),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, 0, Some(&overflow_dir), None);
    assert!(capped > 0);
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let input_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let input_name = input_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();

    // `awk '{print "DIFF:" $0}' <archive>`: every output line differs from the
    // input, so no existing archive may be reused.
    let transformed = "log line\n"
        .repeat(30_000)
        .lines()
        .map(|line| format!("DIFF:{line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let run_args = serde_json::json!({
        "command": format!("awk '{{print \"DIFF:\" $0}}' {}", input_path.to_string_lossy()),
        "pty": false,
    });
    let mut messages = vec![
        assistant_call_args("awk-run", "execute_command", &run_args.to_string()),
        tool_result("awk-run", &transformed),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, 0, Some(&overflow_dir), None);
    assert_eq!(capped, 1);
    assert_eq!(
        std::fs::read_dir(&archive_dir).unwrap().count(),
        2,
        "transformed output must be archived as its own bytes"
    );
    let stub_text = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
    // The stub must point its `file_path` at a fresh archive holding the
    // transformed bytes. (The input path still appears in the recall lines
    // via `original_command`, which is correct provenance, not reuse.)
    let file_line = stub_text
        .lines()
        .find(|line| line.trim_start().starts_with("- file_path: "))
        .expect("stub must carry a file_path line");
    assert!(
        !file_line.contains(&input_name),
        "stub file_path must not point at the transform input: {stub_text}"
    );
    // The fresh archive holds exactly the transformed output.
    let fresh_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.file_name().unwrap().to_string_lossy() != input_name)
        .expect("a second archive file must exist");
    assert_eq!(std::fs::read_to_string(&fresh_path).unwrap(), transformed);
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn path_c_spills_all_protected_precision_groups_without_recent_group_cap() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-global-precision-budget-{}",
        uuid::Uuid::new_v4()
    ));
    let mut messages = Vec::new();
    let mut protected = FxHashSet::default();
    for index in 0..8 {
        let id = format!("read-{index}");
        protected.insert(id.clone());
        messages.push(assistant_call(&id, "read_file"));
        messages.push(tool_result(&id, &"line of exact evidence\n".repeat(600)));
    }

    let spilled =
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected);

    // Covers the second half of Path C: when still over budget after the
    // spill, the emergency cap kicks in. Every preserved stub must first
    // shrink into a non-truncatable minimal pointer and must not be run
    // through generic head/tail truncation again.
    assert!(super::messages_total_chars(&messages) > 4_000);
    super::emergency_cap_messages_to_fit(
        &mut messages,
        4_000,
        160,
        Some(&overflow_dir),
        &protected,
    );

    assert_eq!(spilled, 8);
    let stubs = messages
        .iter()
        .filter_map(|message| {
            let content = value_to_string(&message.content);
            is_preserved_tool_overflow_stub(&content).then_some(content)
        })
        .collect::<Vec<_>>();
    assert_eq!(stubs.len(), 8);
    for stub in stubs {
        let file_path = stub
            .lines()
            .find_map(|line| line.strip_prefix("- file_path: "))
            .expect("minimal overflow stub must retain file_path");
        assert!(resolve_stub_archive_path(file_path, &overflow_dir).is_file());
        assert!(!stub.contains("Preview ("));
    }
    assert!(super::messages_total_chars(&messages) <= 4_000);
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn path_c_reuses_reread_session_asset_instead_of_rearchiving_it() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let test_root =
        std::env::temp_dir().join(format!("ai-reread-session-asset-{}", uuid::Uuid::new_v4()));
    let effective_cwd = test_root.join("workspace");
    let overflow_dir = effective_cwd.join("session-assets");
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&archive_dir).unwrap();
    let archive_path = archive_dir.join("prior-read.txt");
    let content = "previously preserved evidence\n".repeat(800);
    std::fs::write(&archive_path, &content).unwrap();
    let relative_archive_path = archive_path
        .strip_prefix(&effective_cwd)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let arguments = serde_json::json!({
        "file_path": relative_archive_path,
        "offset": 1,
        "limit": 10_000,
    })
    .to_string();
    let mut protected = FxHashSet::default();
    protected.insert("reread".to_string());
    let mut messages = vec![
        assistant_call_args("reread", "read_file", &arguments),
        tool_result("reread", &content),
    ];

    let spilled = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(effective_cwd, || {
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected)
    });

    assert_eq!(spilled, 1);
    let stub = value_to_string(&messages[1].content);
    let file_path = stub
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .expect("reused stub must retain the existing archive pointer");
    assert_eq!(
        resolve_stub_archive_path(file_path, &overflow_dir)
            .canonicalize()
            .unwrap(),
        archive_path.canonicalize().unwrap()
    );
    assert!(stub.contains("- original_range: lines=1..10000"));
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    assert_eq!(std::fs::read_to_string(&archive_path).unwrap(), content);
    let _ = std::fs::remove_dir_all(test_root);
}

#[test]
fn path_c_snapshots_mutable_session_temp_asset_instead_of_reusing_it() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let test_root =
        std::env::temp_dir().join(format!("ai-reread-session-temp-{}", uuid::Uuid::new_v4()));
    let effective_cwd = test_root.join("workspace");
    let overflow_dir = effective_cwd.join("session-assets");
    let temp_dir = overflow_dir.join("tmp");
    std::fs::create_dir_all(&temp_dir).unwrap();
    let temp_path = temp_dir.join("mutable.txt");
    let content = "temporary evidence before mutation\n".repeat(800);
    std::fs::write(&temp_path, &content).unwrap();
    let relative_temp_path = temp_path
        .strip_prefix(&effective_cwd)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let arguments = serde_json::json!({
        "file_path": relative_temp_path,
        "offset": 1,
        "limit": 10_000,
    })
    .to_string();
    let mut protected = FxHashSet::default();
    protected.insert("reread".to_string());
    let mut messages = vec![
        assistant_call_args("reread", "read_file", &arguments),
        tool_result("reread", &content),
    ];

    let spilled = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(effective_cwd, || {
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected)
    });

    assert_eq!(spilled, 1);
    let stub = value_to_string(&messages[1].content);
    let snapshot_path = stub
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .map(|raw| resolve_stub_archive_path(raw, &overflow_dir))
        .expect("mutable session file must be snapshotted into an overflow archive");
    assert_ne!(snapshot_path, temp_path.canonicalize().unwrap());
    assert!(snapshot_path.starts_with(overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR)));
    assert_eq!(std::fs::read_to_string(&snapshot_path).unwrap(), content);

    std::fs::write(&temp_path, "temporary evidence after mutation\n").unwrap();
    assert_eq!(std::fs::read_to_string(&snapshot_path).unwrap(), content);
    let _ = std::fs::remove_dir_all(test_root);
}

#[test]
fn path_c_spills_aggregated_task_wait_result_losslessly() {
    // task_wait forbids lossy compression but occupies no inline budget;
    // Path C's global fallback must spill it losslessly with a file pointer
    // left behind, rather than excluding it from candidates and letting
    // later lossy truncation lose the aggregate truth.
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-global-precision-taskwait-{}",
        uuid::Uuid::new_v4()
    ));
    let mut protected = FxHashSet::default();
    protected.insert("wait".to_string());
    let content = format!(
        "[task_id=task_one]\n[task_id=task_two]\n{}",
        "aggregated subagent conclusion\n".repeat(3_000)
    );
    assert!(content.chars().count() > super::persisted::TOOL_RESULT_RAW_HARD_CAP_CHARS);
    let mut messages = vec![
        assistant_call("wait", "task_wait"),
        tool_result("wait", &content),
    ];

    let spilled =
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected);

    assert_eq!(
        spilled, 1,
        "a large task_wait result must spill losslessly in Path C"
    );
    let stub = value_to_string(&messages[1].content);
    assert!(
        is_preserved_tool_overflow_stub(&stub),
        "a spilled result must become an overflow stub"
    );
    let file_path = stub
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .expect("an overflow stub must retain a recoverable file_path");
    assert!(
        resolve_stub_archive_path(file_path, &overflow_dir).is_file(),
        "the original spilled text must be archived"
    );
    assert_eq!(
        std::fs::read_to_string(resolve_stub_archive_path(file_path, &overflow_dir)).unwrap(),
        content
    );
    for retained in [
        stub.clone(),
        collapse_overflow_stub_to_anchor(&stub).unwrap(),
        minimize_overflow_stub_to_pointer(&stub).unwrap(),
    ] {
        for task_id in ["task_one", "task_two"] {
            assert!(
                retained.contains(&format!("task_id={task_id}")),
                "{retained}"
            );
            assert!(
                retained.contains(&format!(
                    "task_evidence_read({})",
                    serde_json::json!({"task_id": task_id, "source": "delivered"})
                )),
                "{retained}"
            );
        }
    }
    let (folded, count) =
        fold_early_tool_groups(&messages, 0, Some(&overflow_dir), &FxHashSet::default());
    assert_eq!(count, 1);
    let recall = value_to_string(&folded[0].content);
    assert!(recall.contains("task_evidence_read("), "{recall}");
    assert!(
        recall.contains("task_id=task_one") && recall.contains("task_id=task_two"),
        "{recall}"
    );
    assert_eq!(
        std::fs::read_dir(overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR))
            .unwrap()
            .count(),
        1
    );
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn task_evidence_read_recovery_stays_inline_through_floor() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-task-recovery-floor-{}", uuid::Uuid::new_v4()));
    let cap = TOOL_RESULT_FOLD_MIN_CHARS;
    for name in ["task_evidence_read", "task_wait"] {
        let prefix = "[task_id=task_recover]\n";
        let content = if name == "task_evidence_read" {
            let envelope = serde_json::json!({"task_id": "task_recover", "source": "delivered", "status": "delivered", "payload": ""});
            let overhead = envelope.to_string().chars().count();
            serde_json::json!({"task_id": "task_recover", "source": "delivered", "status": "delivered", "payload": "e".repeat(cap - overhead)}).to_string()
        } else {
            format!("{prefix}{}", "e".repeat(cap - prefix.chars().count()))
        };
        assert_eq!(content.chars().count(), cap);
        let mut messages = vec![
            assistant_call_args(
                "evidence",
                name,
                r#"{"task_id":"task_recover","source":"delivered"}"#,
            ),
            tool_result("evidence", &content),
        ];
        let protected = ["evidence".to_string()].into_iter().collect();
        super::prepare_tool_messages_structured(
            &mut messages,
            480,
            0,
            0,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        enforce_protected_precision_group_budget(
            &mut messages,
            1,
            0,
            Some(&overflow_dir),
            None,
            &protected,
            true,
        );
        assert_eq!(
            spill_protected_precision_to_fit(
                &mut messages,
                0,
                Some(&overflow_dir),
                None,
                &protected,
            ),
            0
        );
        assert_eq!(
            cap_oversized_tool_results_for_context(
                &mut messages,
                cap,
                0,
                Some(&overflow_dir),
                None,
            ),
            0
        );
        let (folded, count) =
            fold_early_tool_groups(&messages, 0, Some(&overflow_dir), &FxHashSet::default());
        assert_eq!(count, 0);
        assert_eq!(value_to_string(&folded[1].content), content);
        assert!(!overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR).exists());

        let oversized = if name == "task_evidence_read" {
            let mut envelope: Value = serde_json::from_str(&content).unwrap();
            envelope["payload"] =
                Value::String(format!("{}e", envelope["payload"].as_str().unwrap()));
            envelope.to_string()
        } else {
            format!("{content}e")
        };
        messages[1].content = Value::String(oversized.clone());
        let (folded, count) =
            fold_early_tool_groups(&messages, 0, Some(&overflow_dir), &FxHashSet::default());
        assert_eq!(count, 1);
        let recall = value_to_string(&folded[0].content);
        assert!(
            recall.contains("task_id=task_recover") && recall.contains("task_evidence_read("),
            "{recall}"
        );
        for use_projection_cap in [false, true] {
            let mut candidate = messages.clone();
            let spilled = if use_projection_cap {
                cap_oversized_tool_results_for_context(
                    &mut candidate,
                    cap,
                    0,
                    Some(&overflow_dir),
                    None,
                )
            } else {
                spill_protected_precision_to_fit(
                    &mut candidate,
                    0,
                    Some(&overflow_dir),
                    None,
                    &protected,
                )
            };
            assert_eq!(spilled, 1);
            let stub = value_to_string(&candidate[1].content);
            assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
            let path = stub
                .lines()
                .find_map(|line| line.strip_prefix("- file_path: "))
                .unwrap();
            assert_eq!(
                std::fs::read_to_string(resolve_stub_archive_path(path, &overflow_dir)).unwrap(),
                oversized
            );
            assert!(
                stub.contains("task_id=task_recover") && stub.contains("task_evidence_read("),
                "{stub}"
            );
            assert_eq!(
                spill_protected_precision_to_fit(
                    &mut candidate,
                    0,
                    Some(&overflow_dir),
                    None,
                    &protected
                ),
                0
            );
        }
        std::fs::remove_dir_all(&overflow_dir).unwrap();
    }
}

#[test]
fn shared_tool_result_fold_floor_counts_characters_at_boundary() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-fold-boundary-{}", uuid::Uuid::new_v4()));
    for chars in [
        TOOL_RESULT_FOLD_MIN_CHARS - 1,
        TOOL_RESULT_FOLD_MIN_CHARS,
        TOOL_RESULT_FOLD_MIN_CHARS + 1,
    ] {
        let content = "é".repeat(chars);
        let mut messages = vec![
            assistant_call("read", "read_file"),
            tool_result("read", &content),
        ];
        prepare_tool_messages_structured(
            &mut messages,
            480,
            0,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let inline = value_to_string(&messages[1].content);
        if chars <= TOOL_RESULT_FOLD_MIN_CHARS {
            assert_eq!(inline, content);
        } else {
            assert!(is_preserved_tool_overflow_stub(&inline), "{inline}");
            let path = inline
                .lines()
                .find_map(|line| line.strip_prefix("- file_path: "))
                .unwrap();
            assert_eq!(
                std::fs::read_to_string(resolve_stub_archive_path(path, &overflow_dir)).unwrap(),
                content
            );
        }
    }
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn path_c_does_not_expand_short_protected_results_into_stubs() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-global-precision-short-{}",
        uuid::Uuid::new_v4()
    ));
    let mut protected = FxHashSet::default();
    protected.insert("read-short".to_string());
    let mut messages = vec![
        assistant_call("read-short", "read_file"),
        tool_result("read-short", "ok"),
    ];
    let before = super::messages_total_chars(&messages);

    let spilled =
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected);

    assert_eq!(spilled, 0);
    assert_eq!(value_to_string(&messages[1].content), "ok");
    assert_eq!(super::messages_total_chars(&messages), before);
    let _ = std::fs::remove_dir_all(overflow_dir);
}

fn user_msg(text: &str) -> Message {
    Message {
        role: "user".to_string(),
        content: Value::String(text.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

/// Builds a stub in its "first spill" shape (with a multi-line Preview
/// body), for fold testing.
/// Resolves a stub `- file_path:` value to an on-disk archive path: absolute
/// (legacy) stubs directly, typed aliases and bare names against the temp overflow dir
/// (see `stub_archive_display_path`).
fn resolve_stub_archive_path(raw: &str, overflow_dir: &std::path::Path) -> std::path::PathBuf {
    let raw = raw.trim();
    if let Some(path) =
        crate::ai::tools::storage::file_store::session_asset_for_alias(raw, overflow_dir)
    {
        return path;
    }
    let path = std::path::Path::new(raw);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR).join(raw)
}

fn overflow_stub_with_preview(file_path: &str, tool_name: &str) -> String {
    let full = (0..40)
        .map(|i| format!("line {i}: some content"))
        .collect::<Vec<_>>()
        .join("\n");
    build_preserved_tool_overflow_stub(Path::new(file_path), tool_name, &full, &[])
}

#[test]
fn collapse_overflow_stub_to_anchor_drops_preview_keeps_file_path() {
    let stub = overflow_stub_with_preview("/tmp/session/read-abc.txt", "read_file");
    // Precondition: the first-spill stub really does carry a Preview body.
    assert!(stub.contains("Preview ("));

    let anchor = collapse_overflow_stub_to_anchor(&stub).expect("should collapse");
    // The preview body is discarded.
    assert!(!anchor.contains("Preview ("));
    // The file_path is kept.
    assert!(anchor.contains("- file_path: /tmp/session/read-abc.txt"));
    // The tool name is kept (the new format uses "Output preserved for
    // tool").
    assert!(anchor.contains("Output preserved for tool `read_file`"));
    // read_file-type archives carry the context-budget notice (instead of the
    // old leading "use read_file").
    assert!(anchor.contains("Archived snapshot of an earlier read"));
    assert!(anchor.contains("folded out of inline context for budget reasons"));
    assert!(
        !anchor.contains("Use preview"),
        "collapsed anchors must not point to a preview that was removed: {anchor}"
    );
    // Still a valid stub (prefix unchanged); the downstream compaction
    // chain keeps recognizing it via the stub exemption.
    assert!(is_preserved_tool_overflow_stub(&anchor));
    // The size drops sharply.
    assert!(anchor.len() < stub.len());
}

#[test]
fn preserved_stub_carries_fingerprint_line() {
    let full = "Compiling rust_tools v0.1.0 (/repo)\n\
                warning: unused variable `root_idx`\n\
                error[E0308]: mismatched types in sched_ctx\n";
    let stub =
        build_preserved_tool_overflow_stub(Path::new("/tmp/fp.txt"), "execute_command", full, &[]);
    assert!(is_preserved_tool_overflow_stub(&stub));

    // Deterministic in content: same bytes -> byte-identical stub text.
    let stub_again =
        build_preserved_tool_overflow_stub(Path::new("/tmp/fp.txt"), "execute_command", full, &[]);
    assert_eq!(stub, stub_again);

    let fp_line = stub
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("- fingerprint: "))
        .expect("fingerprint line present on fresh stub");
    // sha= segment: exactly 12 hex chars.
    let sha = fp_line
        .split("sha=")
        .nth(1)
        .unwrap()
        .split(' ')
        .next()
        .unwrap();
    assert_eq!(sha.len(), 12);
    assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
    // Keyword casing is preserved verbatim so tokens stay greppable in the archive.
    assert!(
        fp_line.contains("keys="),
        "keys= segment present: {fp_line}"
    );
    assert!(
        fp_line.contains("rust_tools") || fp_line.contains("root_idx"),
        "keywords: {fp_line}"
    );
}

#[test]
fn collapse_and_minimize_carry_fingerprint_through() {
    let full = "alpha beta gamma\nE0308 mismatched_types hit\n".repeat(30);
    let stub = build_preserved_tool_overflow_stub(
        Path::new("/tmp/carry.txt"),
        "execute_command",
        full.as_str(),
        &[],
    );

    let anchor = collapse_overflow_stub_to_anchor(&stub).expect("collapse");
    assert!(!anchor.contains("Preview ("));
    assert!(
        anchor.contains("- fingerprint: "),
        "anchor carries fingerprint: {anchor}"
    );

    let pointer = minimize_overflow_stub_to_pointer(&stub).expect("minimize");
    assert!(pointer.contains("- file_path: /tmp/carry.txt"));
    assert!(
        pointer.contains("- fingerprint: "),
        "pointer keeps retrieval signal"
    );
    assert!(is_preserved_tool_overflow_stub(&pointer));

    // Legacy stubs (pre-fingerprint) minimize cleanly without fabricated fields.
    let legacy = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\nOutput preserved for tool `x`.\n- file_path: /tmp/legacy.txt"
    );
    let minimized = minimize_overflow_stub_to_pointer(&legacy).unwrap();
    assert!(!minimized.contains("fingerprint"));
}

#[test]
fn fingerprint_skips_degenerate_gist() {
    // A single repeated character carries no recall signal; the gist segment is
    // omitted entirely so aged stubs of degenerate outputs stay minimal.
    let fp = stub_fingerprint_line(&"x".repeat(1_000)).expect("non-empty content has fingerprint");
    assert!(fp.contains("sha="), "{fp}");
    assert!(!fp.contains("gist="), "no gist for degenerate body: {fp}");

    // Real signal still gets a gist.
    let fp2 = stub_fingerprint_line("warning: sched_ctx drifted from kernel root\n");
    assert!(fp2.unwrap().contains("gist=\""), "informative line kept");
}

#[test]
fn fingerprint_keywords_dedup_case_insensitively_and_stay_deterministic() {
    // Mixed-case repeats of the same token must collapse to one keyword,
    // keeping the first-seen casing; the set-based dedup must not perturb
    // ordering vs. the previous linear scan.
    let content = "sched_ctx SCHED_CTX Sched_Ctx root_idx ROOT_IDX payloadxyz\n";
    let keys = extract_fingerprint_keywords(content);
    let sched = keys
        .iter()
        .filter(|k| k.eq_ignore_ascii_case("sched_ctx"))
        .count();
    assert_eq!(
        sched, 1,
        "case-insensitive dedup collapses repeats: {keys:?}"
    );
    assert!(
        keys.contains(&"sched_ctx".to_string()),
        "first casing kept: {keys:?}"
    );
    assert!(keys.len() <= FINGERPRINT_KEY_COUNT);

    // Fully deterministic across calls (no RNG / hash-order leakage into output).
    assert_eq!(keys, extract_fingerprint_keywords(content));
}

#[test]
fn age_out_overflow_stub_previews_is_idempotent() {
    let stub = overflow_stub_with_preview("/tmp/session/read-xyz.txt", "read_file");
    // Two user turns place the stub outside the protected tail window
    // (before retained_turn_start).
    let mut messages = vec![
        user_msg("q1"),
        assistant_call("s", "read_file"),
        tool_result("s", "placeholder"),
        user_msg("q2"),
        user_msg("q3"),
    ];
    messages[2].content = Value::String(stub);

    age_out_overflow_stub_previews(&mut messages, 1);
    let after_first = value_to_string(&messages[2].content);
    assert!(!after_first.contains("Preview ("));

    // Run again: it is already in anchor shape and the content must not
    // change (prevents stub->stub churn).
    age_out_overflow_stub_previews(&mut messages, 1);
    assert_eq!(value_to_string(&messages[2].content), after_first);
}

#[test]
fn age_out_overflow_stub_previews_respects_protected_tail() {
    // One early stub (outside the tail window) and one recent stub (inside
    // the tail window).
    let early = overflow_stub_with_preview("/tmp/session/early.txt", "read_file");
    let recent = overflow_stub_with_preview("/tmp/session/recent.txt", "read_file");
    let mut messages = vec![
        user_msg("q1"),
        assistant_call("early", "read_file"),
        tool_result("early", "placeholder"),
        user_msg("q2"),
        assistant_call("recent", "read_file"),
        tool_result("recent", "placeholder"),
    ];
    messages[2].content = Value::String(early);
    messages[5].content = Value::String(recent.clone());

    // Protect the most recent user turn (from q2 on): the early stub folds,
    // while the recent one inside the tail window keeps its full preview.
    age_out_overflow_stub_previews(&mut messages, 1);
    assert!(!value_to_string(&messages[2].content).contains("Preview ("));
    assert_eq!(value_to_string(&messages[5].content), recent);
    assert!(value_to_string(&messages[5].content).contains("Preview ("));
}

#[test]
fn cap_oversized_spills_only_results_outside_recent_group_window() {
    // The unconditional size gate must leave the most recent complete tool
    // group raw while spilling an equally oversized older result: recent
    // evidence stays directly usable and only aged output gets archived.
    let overflow_dir = std::env::temp_dir().join(format!("ai-cap-recent-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![
        assistant_call("old", "read_file"),
        tool_result("old", &"y".repeat(70_000)),
        assistant_call("new", "read_file"),
        tool_result("new", &"z".repeat(70_000)),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 32_000, 1, Some(&overflow_dir), None);
    assert_eq!(capped, 1);
    assert!(
        is_preserved_tool_overflow_stub(&value_to_string(&messages[1].content)),
        "old oversized result must spill"
    );
    assert!(
        !is_preserved_tool_overflow_stub(&value_to_string(&messages[3].content)),
        "newest group inside the protection window must stay raw"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn cap_no_oversized_returns_zero_and_keeps_messages_unchanged() {
    // Fast path: when every result is below the cap, the pre-scan must
    // short-circuit before index building / content cloning and touch nothing.
    let overflow_dir = std::env::temp_dir().join(format!("ai-cap-fast-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![
        user_msg("q1"),
        assistant_call("a", "read_file"),
        tool_result("a", "small output"),
        user_msg("q2"),
        assistant_call("b", "read_file"),
        tool_result("b", "also small"),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 32_000, 0, Some(&overflow_dir), None);
    assert_eq!(capped, 0);
    assert!(!is_preserved_tool_overflow_stub(&value_to_string(
        &messages[2].content
    )));
    assert!(!is_preserved_tool_overflow_stub(&value_to_string(
        &messages[5].content
    )));
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn shortened_stub_carries_typed_archive_alias_for_session_archives() {
    // Archive children must drop the long repeating session directory prefix;
    // any other layout keeps the full path so read-back never breaks.
    let archive_child = std::path::Path::new("/sessions/abc.assets")
        .join(PRESERVED_TOOL_OVERFLOW_DIR)
        .join("spilled-read_file-deadbeef12345678.txt");
    assert_eq!(
        super::tool_overflow::stub_archive_display_path(&archive_child),
        "session-asset:tool-overflow-compressed/spilled-read_file-deadbeef12345678.txt"
    );
    let plain = std::path::PathBuf::from("/tmp/result.txt");
    assert_eq!(
        super::tool_overflow::stub_archive_display_path(&plain),
        "/tmp/result.txt"
    );
    // A directory merely named like the archive dir elsewhere still shortens
    // (names are content-addressed); resolution safety lives in the reader.
    let stub = super::tool_overflow::build_preserved_tool_overflow_stub(
        &archive_child,
        "read_file",
        "line\n".repeat(50).as_str(),
        &[],
    );
    assert!(stub.contains(
        "- file_path: session-asset:tool-overflow-compressed/spilled-read_file-deadbeef12345678.txt"
    ));
    assert!(!stub.contains("/sessions/abc.assets"));
}

#[test]
fn shortened_stub_alias_covers_all_archive_dirs_and_round_trips() {
    // Every allowlisted archive dir shortens; the mapping is idempotent so
    // stub normalization can rebuild old absolute pointers without a session
    // directory at hand.
    for (dir, name) in [
        (
            PRESERVED_TOOL_OVERFLOW_DIR,
            "spilled-read_file-aaaaaaaaaaaaaaaaaaaaaaaa.txt",
        ),
        (
            PRESERVED_USER_OVERFLOW_DIR,
            "user-bbbbbbbbbbbbbbbbbbbbbbbb.json",
        ),
        (
            PRESERVED_IMAGE_OVERFLOW_DIR,
            "image-cccccccccccccccccccccccc.json",
        ),
        ("folded-tool-groups", "dddddddddddddddddddddddd.md"),
        ("internal-note-overflow", "eeeeeeeeeeeeeeeeeeeeeeee.md"),
        ("summary-sources", "ffffffffffffffffffffffff.jsonl"),
    ] {
        let absolute = std::path::Path::new("/sessions/abc.assets")
            .join(dir)
            .join(name);
        let alias = super::tool_overflow::stub_archive_display_path(&absolute);
        assert_eq!(alias, format!("session-asset:{dir}/{name}"));
        assert_eq!(
            super::tool_overflow::stub_archive_display_path(std::path::Path::new(&alias)),
            alias,
            "alias must round-trip for {dir}"
        );
    }
    // Unknown dirs and non-archive layouts keep the full path.
    for plain in [
        "/sessions/abc.assets/pruned-folds/note.txt",
        "/sessions/abc.assets/overflow-history.md",
    ] {
        assert_eq!(
            super::tool_overflow::stub_archive_display_path(std::path::Path::new(plain)),
            plain
        );
    }
}

#[test]
fn archived_asset_reuse_accepts_typed_aliases_and_legacy_paths_safely() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-bare-name-reuse-{}", uuid::Uuid::new_v4()));
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&archive_dir).unwrap();
    let archived = archive_dir.join("spilled-read_file-aaaaaaaaaaaaaaaa.txt");
    std::fs::write(&archived, "evidence\n").unwrap();
    let canonical = archived.canonicalize().unwrap();

    // Legacy absolute stubs keep working.
    assert_eq!(
        super::tool_overflow::archived_asset_path_from_raw(
            &canonical.to_string_lossy(),
            &overflow_dir
        ),
        Some(canonical.clone())
    );
    // Shortened bare names resolve under this session's overflow dir.
    assert_eq!(
        super::tool_overflow::archived_asset_path_from_raw(
            "spilled-read_file-aaaaaaaaaaaaaaaa.txt",
            &overflow_dir
        ),
        Some(canonical.clone())
    );
    // The generated alias must resolve to the original asset, not a new archive.
    let alias = super::tool_overflow::stub_archive_display_path(&archived);
    assert_eq!(
        super::tool_overflow::archived_asset_path_from_raw(&alias, &overflow_dir),
        Some(canonical.clone())
    );
    for invalid in [
        "session-asset:tool-overflow-compressed/../evil.txt",
        "session-asset:tool-overflow-compressed/sub/dir.txt",
        "session-asset:tool-overflow-compressed/spilled-missing.txt",
        "session-asset:tool-overflow/spilled-read_file-aaaaaaaaaaaaaaaa.txt",
        "session-asset:unknown/spilled-read_file-aaaaaaaaaaaaaaaa.txt",
    ] {
        assert_eq!(
            super::tool_overflow::archived_asset_path_from_raw(invalid, &overflow_dir),
            None,
            "accepted {invalid}",
        );
    }
    #[cfg(unix)]
    {
        let outside = overflow_dir.join("outside.txt");
        std::fs::write(&outside, "not an immutable compressed archive").unwrap();
        std::os::unix::fs::symlink(&outside, archive_dir.join("escape.txt")).unwrap();
        assert_eq!(
            super::tool_overflow::archived_asset_path_from_raw(
                "session-asset:tool-overflow-compressed/escape.txt",
                &overflow_dir,
            ),
            None,
        );
    }
    // Traversal, absolute outsiders, and missing files stay rejected.
    assert_eq!(
        super::tool_overflow::archived_asset_path_from_raw("../evil.txt", &overflow_dir),
        None
    );
    assert_eq!(
        super::tool_overflow::archived_asset_path_from_raw("sub/dir.txt", &overflow_dir),
        None
    );
    assert_eq!(
        super::tool_overflow::archived_asset_path_from_raw("/tmp/elsewhere.txt", &overflow_dir),
        None
    );
    assert_eq!(
        super::tool_overflow::archived_asset_path_from_raw(
            "spilled-missing-bbbbbbbbbbbbbbbb.txt",
            &overflow_dir
        ),
        None
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

fn plain_message(role: &str, content: String) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(content),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

#[test]
fn persisted_summary_reserves_budget_for_latest_correction_and_pending() {
    // Twenty fat old turns must not crowd the fragment's latest correction
    // and unanswered task out of the draft: newest intent claims budget
    // before older history.
    let mut messages: Vec<Message> = Vec::new();
    for turn in 0..20 {
        messages.push(plain_message(
            "user",
            format!("work on topic_{turn}.rs {}", "u".repeat(300)),
        ));
        messages.push(plain_message(
            "assistant",
            format!("old_answer_{turn} {}", "a".repeat(400)),
        ));
    }
    messages.push(plain_message(
        "user",
        "LATEST_CORRECTION switch the plan to tabs".to_string(),
    ));
    let draft = build_persisted_summary_text(&messages, 2_800);
    assert!(
        draft.contains("LATEST_CORRECTION"),
        "newest correction must survive old-turn pressure: {draft}"
    );
    assert!(
        draft.contains("Pending tasks:"),
        "unanswered latest turn must stay listed: {draft}"
    );
    assert!(
        draft.chars().count() <= 2_800,
        "draft exceeds budget: {}",
        draft.chars().count()
    );
}
