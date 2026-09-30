use super::super::{
    COMPRESSED_TOOL_EVIDENCE_MARKER, KEEP_RECENT_TOOL_GROUPS, PruneAuthorization, llm_prune,
    tool_groups,
};
use super::{
    MIN_FOLDED_CHARS, attach_provenance, candidates, canonical_fold_ids, provenance,
    recent_group_indices, tool_allows_explicit_prune,
};
use crate::ai::history::types::{FOLDED_TOOL_ORIGIN, Message, ROLE_INTERNAL_NOTE};
use crate::ai::request::normalize_messages_for_request_for_test;
use crate::ai::types::{FunctionCall, ToolCall};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("ai-folded-prune-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        // This directory is uniquely owned by the fixture, never session storage.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn message(role: &str, content: impl Into<String>) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(content.into()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn raw_group(prefix: &str, names: &[&str]) -> Vec<Message> {
    let calls = names
        .iter()
        .enumerate()
        .map(|(index, name)| ToolCall {
            id: format!("{prefix}-{index}"),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: (*name).to_string(),
                arguments: match *name {
                    "read_file" => json!({
                        "file_path": "src/lib.rs", "offset": 1, "limit": 80
                    })
                    .to_string(),
                    "execute_command" => json!({"command": "printf fixture"}).to_string(),
                    _ => "{}".to_string(),
                },
            },
        })
        .collect::<Vec<_>>();
    let mut assistant = message(
        "assistant",
        "The inspected evidence remains recoverable. ".repeat(24),
    );
    assistant.tool_calls = Some(calls.clone());
    let mut messages = vec![assistant];
    for call in calls {
        let mut result = message(
            "tool",
            format!(
                "Evidence for {}\n{}",
                call.id,
                "A complete fixture line with a recoverable result and stable context.\n"
                    .repeat(96)
            ),
        );
        result.tool_call_id = Some(call.id);
        messages.push(result);
    }
    messages
}

fn append_recent_raw(messages: &mut Vec<Message>, count: usize) {
    for index in 0..count {
        let mut group = raw_group(&format!("recent-{index}"), &["read_file"]);
        group[1].content = Value::String("Recent result kept verbatim.".to_string());
        messages.extend(group);
    }
}

/// Exercise the real fold producer, including its archive commit boundary.
fn fold_group(raw: &[Message], dir: &Path) -> Message {
    let before = raw.to_vec();
    let plan = tool_groups::plan_early_tool_groups(raw, 0, Some(dir), &FxHashSet::default());
    assert_eq!(plan.folded_groups(), 1);
    assert_eq!(plan.messages().len(), 1);
    assert!(plan.commit());
    let (folded, count) = plan.into_result();
    assert_eq!(count, 1);
    assert_eq!(raw, before);
    let note = folded.into_iter().next().unwrap();
    assert!(provenance(&note).is_some());
    assert!(note.content.as_str().unwrap().chars().count() >= MIN_FOLDED_CHARS);
    note
}

fn old_fold_projection(note: Message) -> Vec<Message> {
    let mut messages = vec![
        message("system", "System fixture."),
        message("user", "Inspect the evidence."),
        note,
    ];
    append_recent_raw(&mut messages, KEEP_RECENT_TOOL_GROUPS);
    messages
}

fn without_protocol(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .filter(|message| !llm_prune::is_prune_protocol_message(message))
        .cloned()
        .collect()
}

fn fold_note(messages: &[Message]) -> &Message {
    messages
        .iter()
        .find(|message| provenance(message).is_some())
        .expect("the projection must retain its folded note")
}

fn archive_path(note: &Message) -> PathBuf {
    note.content
        .as_str()
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("- archive_file_path: "))
        .map(PathBuf::from)
        .expect("the note must carry an archive locator")
}

fn assert_complete_pairs(messages: &[Message]) {
    let mut pending = FxHashSet::default();
    for message in messages {
        if message.role == "assistant" {
            assert!(
                pending.is_empty(),
                "a later assistant must not split a group"
            );
            if let Some(calls) = &message.tool_calls {
                for call in calls {
                    assert!(pending.insert(call.id.clone()), "duplicate call id");
                }
            }
        } else if message.role == "tool" {
            assert!(
                pending.remove(message.tool_call_id.as_deref().unwrap()),
                "every result must match exactly one outstanding call"
            );
        } else {
            assert!(
                pending.is_empty(),
                "a non-tool message must not split a group"
            );
        }
    }
    assert!(pending.is_empty(), "all calls must have results");
}

#[test]
fn folded_prune_planning_binds_complete_sources_without_writing_archives() {
    let dir = TestDir::new();
    let overflow = dir.path().join("not-yet-committed");
    let raw = raw_group("planned", &["read_file", "execute_command"]);
    let canonical = raw.clone();
    let plan = tool_groups::plan_early_tool_groups(&raw, 0, Some(&overflow), &FxHashSet::default());

    assert_eq!(plan.folded_groups(), 1);
    assert!(!overflow.exists(), "speculative plans must not write files");
    let note = &plan.messages()[0];
    let meta = provenance(note).unwrap();
    assert_eq!(
        meta.source_ids().collect::<Vec<_>>(),
        ["planned-0", "planned-1"]
    );
    assert!(meta.allows_prune());
    assert!(!meta.is_offloaded());
    assert!(canonical_fold_ids(&canonical).contains(&meta.id()));
    let path = archive_path(note);
    assert!(!path.exists());

    assert!(plan.commit());
    let archived = std::fs::read_to_string(&path).unwrap();
    let json = archived.split_once("```json\n").unwrap().1;
    let json = json.rsplit_once("\n```").unwrap().0;
    let archived_messages: Vec<Message> = serde_json::from_str(json).unwrap();
    assert_eq!(archived_messages, canonical);
    assert_eq!(raw, canonical);
    assert_complete_pairs(&archived_messages);
}

#[test]
fn folded_prune_two_independent_marks_archive_exact_note_and_leave_canonical_unchanged() {
    let dir = TestDir::new();
    let mut canonical = raw_group("offload", &["read_file", "execute_command"]);
    let note = fold_group(&canonical, dir.path());
    let id = provenance(&note).unwrap().id();
    let original_text = note.content.as_str().unwrap().to_string();
    let source_archive = archive_path(&note);
    let checkpoint = original_text
        .lines()
        .find(|line| line.starts_with("assistant_checkpoint:"))
        .unwrap()
        .to_string();
    append_recent_raw(&mut canonical, KEEP_RECENT_TOOL_GROUPS);
    let canonical_before = canonical.clone();
    let mut request = old_fold_projection(note.clone());
    let projection_before = request.clone();
    let mut marks = FxHashMap::default();

    assert!(llm_prune::active_prunable_tool_ids(&request).contains(&id));
    assert!(llm_prune::update_prune_marks_for_messages(
        &mut marks,
        &[id.clone(), id.clone(), id.clone()],
        &request,
    ));
    assert_eq!(
        marks.get(&id),
        Some(&1),
        "duplicates in one response count once"
    );
    let first = llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path()));
    assert_eq!(first.pruned_count, 0);
    assert_eq!(first.freed_chars, 0);
    assert_eq!(without_protocol(&request), projection_before);
    assert!(!dir.path().join("pruned-folds").exists());
    let protocol = request
        .iter()
        .find(|message| llm_prune::is_prune_protocol_message(message))
        .unwrap()
        .content
        .as_str()
        .unwrap();
    assert!(protocol.contains(&id));
    assert!(protocol.contains("marks 1/2"));

    assert!(!llm_prune::update_prune_marks_for_messages(
        &mut marks,
        &[],
        &request
    ));
    assert_eq!(marks.get(&id), Some(&1));
    assert!(llm_prune::update_prune_marks_for_messages(
        &mut marks,
        std::slice::from_ref(&id),
        &request,
    ));
    assert_eq!(marks.get(&id), Some(&llm_prune::PRUNE_THRESHOLD));
    let report = llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path()));
    assert_eq!(report.pruned_count, 1);
    let stub = fold_note(&request);
    let stub_text = stub.content.as_str().unwrap();
    let exact_archive = archive_path(stub);
    assert_eq!(
        std::fs::read(&exact_archive).unwrap(),
        original_text.as_bytes()
    );
    assert_ne!(exact_archive, source_archive);
    assert!(source_archive.is_file());
    assert!(stub_text.contains("archive_scope: exact_folded_note_before_explicit_prune"));
    assert!(stub_text.contains(&format!(
        "source_projection_archive: {}",
        source_archive.display()
    )));
    assert!(stub_text.lines().any(|line| line == checkpoint));
    assert!(stub_text.contains("offload-0"));
    assert!(stub_text.contains("offload-1"));
    assert_eq!(
        report.freed_chars,
        original_text.chars().count() - stub_text.chars().count()
    );
    let offloaded = provenance(stub).unwrap();
    assert!(offloaded.is_offloaded());
    assert_eq!(offloaded.id(), id);
    assert_eq!(canonical, canonical_before);
    assert_eq!(request.len(), projection_before.len());
    assert_eq!(request[0..2], projection_before[0..2]);
    assert_eq!(request[3..], projection_before[3..]);
    assert_complete_pairs(&request);
    assert_complete_pairs(&canonical);
    assert!(llm_prune::active_prunable_tool_ids(&request).is_empty());
    assert!(llm_prune::retained_prune_ids(&request).contains(&id));

    let completed_request = request.clone();
    assert!(!llm_prune::update_prune_marks_for_messages(
        &mut marks,
        &[],
        &request
    ));
    assert!(!llm_prune::update_prune_marks_for_messages(
        &mut marks,
        std::slice::from_ref(&id),
        &request,
    ));
    assert_eq!(marks.get(&id), Some(&llm_prune::PRUNE_THRESHOLD));
    let repeat = llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path()));
    assert_eq!(repeat.pruned_count, 0);
    assert_eq!(repeat.freed_chars, 0);
    assert_eq!(request, completed_request);
    assert_eq!(
        std::fs::read(exact_archive).unwrap(),
        original_text.as_bytes()
    );
}

#[test]
fn folded_prune_large_note_does_not_inherit_raw_single_mark_exception() {
    let dir = TestDir::new();
    let raw = raw_group("large-fold", &["read_file"]);
    let mut note = fold_group(&raw, dir.path());
    let source_archive = archive_path(&note);
    let mut content = note.content.as_str().unwrap().to_string();
    content.push_str(&"\nAdditional preserved preview evidence.".repeat(1_000));
    note.content = Value::String(content);
    attach_provenance(
        &mut note,
        &raw,
        &(0..raw.len()).collect::<Vec<_>>(),
        source_archive.to_str(),
    );
    let id = provenance(&note).unwrap().id();
    assert!(note.content.as_str().unwrap().chars().count() > 16_384);
    let mut request = old_fold_projection(note.clone());
    let mut marks = FxHashMap::default();
    llm_prune::update_prune_marks_for_messages(&mut marks, &[id.clone(), id.clone()], &request);
    assert_eq!(marks.get(&id), Some(&1));
    assert_eq!(llm_prune::needed_marks_for(&request, &id), 2);
    assert_eq!(
        llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path())).pruned_count,
        0
    );
    assert_eq!(fold_note(&request), &note);
    assert!(!dir.path().join("pruned-folds").exists());
}

#[test]
fn folded_prune_protects_plan_task_unknown_and_mixed_groups_past_preview_cutoff() {
    let dir = TestDir::new();
    for (index, protected) in [
        "plan",
        "plan_update",
        "task_spawn",
        "task_spawn_batch",
        "task_retry",
        "unregistered_future_tool",
        "multi_tool_use.parallel",
    ]
    .into_iter()
    .enumerate()
    {
        assert!(!tool_allows_explicit_prune(protected), "{protected}");
        for mixed in [false, true] {
            let mut names = if mixed {
                vec!["read_file"; 8]
            } else {
                Vec::new()
            };
            names.push(protected);
            let raw = raw_group(&format!("protected-{index}-{mixed}"), &names);
            let note = fold_group(&raw, dir.path());
            let meta = provenance(&note).unwrap();
            let id = meta.id();
            assert_eq!(meta.source_ids().count(), names.len());
            assert!(!meta.allows_prune());
            assert!(!canonical_fold_ids(&raw).contains(&id));
            if mixed {
                let text = note.content.as_str().unwrap();
                assert!(text.contains("1 more tools omitted"));
                assert!(
                    !text
                        .lines()
                        .any(|line| line.starts_with(&format!("- {protected}")))
                );
            }
            let mut request = old_fold_projection(note);
            let before = request.clone();
            assert!(candidates(&request).is_empty());
            assert!(llm_prune::active_prunable_tool_ids(&request).is_empty());
            assert!(!llm_prune::retained_prune_ids(&request).contains(&id));
            let marks = FxHashMap::from_iter([(id, u8::MAX)]);
            let report =
                llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path()));
            assert_eq!(report.pruned_count, 0, "{protected}, mixed={mixed}");
            assert_eq!(request, before, "{protected}, mixed={mixed}");
        }
    }
    assert!(!dir.path().join("pruned-folds").exists());
}

#[test]
fn folded_prune_rejects_legacy_folds_ordinary_notes_checkpoints_and_role_spoofing() {
    let dir = TestDir::new();
    let raw = raw_group("legacy", &["read_file"]);
    let valid = fold_group(&raw, dir.path());
    let id = provenance(&valid).unwrap().id();
    let mut legacy = valid.clone();
    legacy.reasoning_content = None;
    let mut notes = vec![
        legacy,
        message(ROLE_INTERNAL_NOTE, "Ordinary internal memory. ".repeat(200)),
        message(
            ROLE_INTERNAL_NOTE,
            "self_note: Preserve the decision. ".repeat(200),
        ),
        message(
            ROLE_INTERNAL_NOTE,
            format!(
                "[context_checkpoint path=fixture.md]\n{}",
                "Evidence. ".repeat(400)
            ),
        ),
        message(
            ROLE_INTERNAL_NOTE,
            "Plan: keep all remaining steps. ".repeat(200),
        ),
        message(
            ROLE_INTERNAL_NOTE,
            "[TASK_RESULT] delegated evidence. ".repeat(200),
        ),
    ];
    for role in ["system", "user", "assistant", "tool"] {
        let mut spoof = valid.clone();
        spoof.role = role.to_string();
        notes.push(spoof);
    }
    for note in notes {
        assert!(provenance(&note).is_none());
        let mut request = old_fold_projection(note);
        let before = request.clone();
        assert!(candidates(&request).is_empty());
        assert!(llm_prune::active_prunable_tool_ids(&request).is_empty());
        let marks = FxHashMap::from_iter([(id.clone(), u8::MAX)]);
        assert_eq!(
            llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path()))
                .pruned_count,
            0
        );
        assert_eq!(request, before);
    }
    assert!(!dir.path().join("pruned-folds").exists());
}

#[test]
fn folded_prune_provenance_requires_complete_unique_call_result_pairing() {
    let dir = TestDir::new();
    let raw = raw_group("pairing", &["read_file", "execute_command"]);
    let template = fold_group(&raw, dir.path());
    let path = archive_path(&template);
    let mut malformed = Vec::new();

    let mut missing = raw.clone();
    missing.pop();
    malformed.push(("missing result", missing));
    let mut duplicate_result = raw.clone();
    duplicate_result[2] = duplicate_result[1].clone();
    malformed.push(("duplicate result", duplicate_result));
    let mut extra = raw.clone();
    extra.push(raw[1].clone());
    malformed.push(("extra result", extra));
    let mut orphan = raw.clone();
    orphan[2].tool_call_id = Some("unrelated".to_string());
    malformed.push(("orphan result", orphan));
    let mut missing_id = raw.clone();
    missing_id[1].tool_call_id = None;
    malformed.push(("result without id", missing_id));
    let mut non_text = raw.clone();
    non_text[1].content = json!({"text": "not a textual tool result"});
    malformed.push(("non-text result", non_text));
    let mut wrong_role = raw.clone();
    wrong_role[1].role = ROLE_INTERNAL_NOTE.to_string();
    malformed.push(("non-tool result", wrong_role));
    let mut wrong_assistant = raw.clone();
    wrong_assistant[0].role = "user".to_string();
    malformed.push(("non-assistant caller", wrong_assistant));
    let mut duplicate_call = raw.clone();
    let calls = duplicate_call[0].tool_calls.as_mut().unwrap();
    calls[1].id = calls[0].id.clone();
    malformed.push(("duplicate call id", duplicate_call));
    let mut empty_id = raw.clone();
    empty_id[0].tool_calls.as_mut().unwrap()[0].id.clear();
    malformed.push(("empty call id", empty_id));
    for arguments in ["not json", "[]", "null", "42"] {
        let mut invalid_arguments = raw.clone();
        invalid_arguments[0].tool_calls.as_mut().unwrap()[0]
            .function
            .arguments = arguments.to_string();
        malformed.push(("non-object arguments", invalid_arguments));
    }
    let mut empty_calls = raw.clone();
    empty_calls[0].tool_calls = Some(Vec::new());
    malformed.push(("empty calls", empty_calls));

    for (case, group) in malformed {
        let mut note = template.clone();
        note.reasoning_content = None;
        attach_provenance(
            &mut note,
            &group,
            &(0..group.len()).collect::<Vec<_>>(),
            path.to_str(),
        );
        assert!(note.reasoning_content.is_none(), "{case}");
        assert!(provenance(&note).is_none(), "{case}");
        assert!(candidates(&old_fold_projection(note)).is_empty(), "{case}");
    }

    let mut reordered = raw.clone();
    reordered.swap(1, 2);
    let mut note = template.clone();
    note.reasoning_content = None;
    attach_provenance(&mut note, &reordered, &[0, 1, 2], path.to_str());
    assert_eq!(
        provenance(&note).unwrap().id(),
        provenance(&template).unwrap().id()
    );
    assert_eq!(canonical_fold_ids(&reordered), canonical_fold_ids(&raw));
}

#[test]
fn folded_prune_rejects_changed_content_and_malformed_persisted_metadata() {
    let dir = TestDir::new();
    let raw = raw_group("tamper", &["read_file", "execute_command"]);
    let valid = fold_group(&raw, dir.path());
    let id = provenance(&valid).unwrap().id();
    let encoded = valid.reasoning_content.as_deref().unwrap();
    let meta: Value =
        serde_json::from_str(encoded.strip_prefix(FOLDED_TOOL_ORIGIN).unwrap()).unwrap();
    let mut variants = Vec::new();
    let mut content_changed = valid.clone();
    content_changed.content = Value::String(format!(
        "{}\nChanged preview",
        valid.content.as_str().unwrap()
    ));
    variants.push(content_changed);
    let mut non_text = valid.clone();
    non_text.content = json!({"text": valid.content});
    variants.push(non_text);
    let mut with_calls = valid.clone();
    with_calls.tool_calls = Some(Vec::new());
    variants.push(with_calls);
    let mut with_result_id = valid.clone();
    with_result_id.tool_call_id = Some("tamper-0".to_string());
    variants.push(with_result_id);
    for encoding in ["not provenance", "runtime-origin:folded-tool-evidence:v1:{"] {
        let mut note = valid.clone();
        note.reasoning_content = Some(encoding.to_string());
        variants.push(note);
    }
    let mut altered_metadata = Vec::new();
    let mut checksum = meta.clone();
    checksum["content_sha256"] = json!("0".repeat(64));
    altered_metadata.push(checksum);
    let mut empty_path = meta.clone();
    empty_path["archive_file_path"] = json!("");
    altered_metadata.push(empty_path);
    let mut empty_sources = meta.clone();
    empty_sources["sources"] = json!([]);
    altered_metadata.push(empty_sources);
    let mut duplicate_source = meta.clone();
    duplicate_source["sources"][1]["id"] = duplicate_source["sources"][0]["id"].clone();
    altered_metadata.push(duplicate_source);
    for field in ["id", "name", "arguments_sha256"] {
        let mut missing = meta.clone();
        missing["sources"][0][field] = json!("");
        altered_metadata.push(missing);
    }
    let mut bad_digest = meta.clone();
    bad_digest["sources"][0]["arguments_sha256"] = json!("z".repeat(64));
    altered_metadata.push(bad_digest);
    let mut unknown_field = meta.clone();
    unknown_field["unrecognized_authority"] = json!(true);
    altered_metadata.push(unknown_field);
    let mut unknown_source_field = meta.clone();
    unknown_source_field["sources"][0]["prune_allowed"] = json!(true);
    altered_metadata.push(unknown_source_field);
    for altered in altered_metadata {
        let mut note = valid.clone();
        note.reasoning_content = Some(format!("{FOLDED_TOOL_ORIGIN}{altered}"));
        variants.push(note);
    }
    for note in variants {
        assert!(provenance(&note).is_none());
        let mut request = old_fold_projection(note);
        let before = request.clone();
        assert!(llm_prune::active_prunable_tool_ids(&request).is_empty());
        let marks = FxHashMap::from_iter([(id.clone(), u8::MAX)]);
        assert_eq!(
            llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path()))
                .pruned_count,
            0
        );
        assert_eq!(request, before);
    }
    assert!(!dir.path().join("pruned-folds").exists());
}

#[test]
fn folded_prune_requires_archive_provenance_and_minimum_character_count() {
    let dir = TestDir::new();
    let raw = raw_group("size", &["read_file"]);
    let template = fold_group(&raw, dir.path());
    let path = archive_path(&template);
    let mut unbound = template.clone();
    unbound.reasoning_content = None;
    attach_provenance(&mut unbound, &raw, &[0, 1], None);
    assert!(provenance(&unbound).is_none());

    let prefix = format!(
        "compressed_tool_round: 1 tool calls\n{COMPRESSED_TOOL_EVIDENCE_MARKER}\nassistant_checkpoint: Preserved evidence.\n"
    );
    let id_suffix_chars = format!("\nprune_id: {}", provenance(&template).unwrap().id())
        .chars()
        .count();
    for (chars, expected) in [(MIN_FOLDED_CHARS - 1, false), (MIN_FOLDED_CHARS, true)] {
        let mut note = template.clone();
        note.reasoning_content = None;
        note.content = Value::String(format!(
            "{prefix}{}",
            "界".repeat(chars - prefix.chars().count() - id_suffix_chars)
        ));
        attach_provenance(&mut note, &raw, &[0, 1], path.to_str());
        assert_eq!(note.content.as_str().unwrap().chars().count(), chars);
        assert!(provenance(&note).is_some());
        assert_eq!(!candidates(&old_fold_projection(note)).is_empty(), expected);
    }
}

#[test]
fn folded_prune_recent_window_counts_raw_folded_and_legacy_groups() {
    let dir = TestDir::new();
    let old = fold_group(&raw_group("old-mixed", &["read_file"]), dir.path());
    let old_id = provenance(&old).unwrap().id();
    let mut request = vec![old];
    let mut recent_anchors = FxHashSet::default();
    let mut protected_fold_ids = Vec::new();
    for index in 0..KEEP_RECENT_TOOL_GROUPS {
        recent_anchors.insert(request.len());
        if index % 2 == 0 {
            request.extend(raw_group(&format!("mixed-raw-{index}"), &["read_file"]));
        } else {
            let mut note = fold_group(
                &raw_group(&format!("mixed-fold-{index}"), &["read_file"]),
                dir.path(),
            );
            protected_fold_ids.push(provenance(&note).unwrap().id());
            if index == KEEP_RECENT_TOOL_GROUPS - 1 {
                note.reasoning_content = None;
            }
            request.push(note);
        }
        request.push(message(
            ROLE_INTERNAL_NOTE,
            "An ordinary note is not a tool group.",
        ));
    }
    assert_eq!(recent_group_indices(&request), recent_anchors);
    let active = llm_prune::active_prunable_tool_ids(&request);
    assert_eq!(active, FxHashSet::from_iter([old_id.clone()]));
    let mut marks = FxHashMap::from_iter([(old_id, llm_prune::PRUNE_THRESHOLD)]);
    for id in protected_fold_ids {
        marks.insert(id, u8::MAX);
    }
    let protected_tail = request[1..].to_vec();
    let report = llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path()));
    assert_eq!(report.pruned_count, 1);
    assert_eq!(without_protocol(&request)[1..], protected_tail);
    assert_complete_pairs(&request);
}

#[test]
fn folded_prune_newer_folds_age_raw_results_out_of_the_recent_window() {
    let dir = TestDir::new();
    let mut request = raw_group("aged-raw", &["read_file"]);
    for index in 0..KEEP_RECENT_TOOL_GROUPS {
        request.push(fold_group(
            &raw_group(&format!("newer-fold-{index}"), &["read_file"]),
            dir.path(),
        ));
    }
    assert!(!recent_group_indices(&request).contains(&0));
    assert!(
        llm_prune::active_prunable_tool_ids(&request).contains("aged-raw-0"),
        "raw and folded groups must share the same chronological protection window"
    );
}

#[test]
fn folded_prune_missing_or_failed_archive_keeps_exact_note_and_metadata() {
    let dir = TestDir::new();
    let raw = raw_group("archive-failure", &["read_file", "execute_command"]);
    let note = fold_group(&raw, dir.path());
    let id = provenance(&note).unwrap().id();
    let before = old_fold_projection(note);
    let marks = FxHashMap::from_iter([(id, llm_prune::PRUNE_THRESHOLD)]);

    let mut without_dir = before.clone();
    let report = llm_prune::prepare_request_projection(&mut without_dir, &marks, None);
    assert_eq!(report.pruned_count, 0);
    assert_eq!(report.freed_chars, 0);
    assert_eq!(without_protocol(&without_dir), before);
    assert!(!dir.path().join("pruned-folds").exists());

    let blocker = dir.path().join("pruned-folds");
    std::fs::write(&blocker, "This fixture blocks the archive directory.").unwrap();
    let mut failed = before.clone();
    let report = llm_prune::prepare_request_projection(&mut failed, &marks, Some(dir.path()));
    assert_eq!(report.pruned_count, 0);
    assert_eq!(report.freed_chars, 0);
    assert_eq!(without_protocol(&failed), before);
    assert!(!provenance(fold_note(&failed)).unwrap().is_offloaded());
    assert_eq!(
        std::fs::read_to_string(blocker).unwrap(),
        "This fixture blocks the archive directory."
    );
}

#[test]
fn folded_prune_indistinguishable_fold_identity_fails_closed_for_every_note() {
    let dir = TestDir::new();
    // Both raw groups reuse one call ID with the same tool name, arguments and
    // result, so the computed candidate identity cannot tell the notes apart.
    // Only the assistant checkpoint differs, and it is not part of that identity.
    let raw = raw_group("ambiguous", &["read_file"]);
    let mut changed_checkpoint = raw.clone();
    changed_checkpoint[0].content =
        Value::String("A different assistant checkpoint over the same call. ".repeat(24));
    let note = fold_group(&raw, dir.path());
    let changed_note = fold_group(&changed_checkpoint, dir.path());
    let id = provenance(&note).unwrap().id();
    assert_eq!(provenance(&changed_note).unwrap().id(), id);
    assert_ne!(note.content, changed_note.content);

    let mut ambiguous = vec![
        message("system", "System fixture."),
        message("user", "Inspect the evidence."),
        note.clone(),
        changed_note.clone(),
    ];
    append_recent_raw(&mut ambiguous, KEEP_RECENT_TOOL_GROUPS);
    let ids = candidates(&ambiguous)
        .into_iter()
        .map(|(_, meta)| meta.id())
        .collect::<Vec<_>>();
    assert!(
        ids.is_empty(),
        "an identity carried by two notes must offer no candidate: {ids:?}"
    );
    assert!(!llm_prune::active_prunable_tool_ids(&ambiguous).contains(&id));
    let marks = FxHashMap::from_iter([(id.clone(), u8::MAX)]);
    let before = ambiguous.clone();
    let report = llm_prune::prepare_request_projection(&mut ambiguous, &marks, Some(dir.path()));
    assert_eq!(report.pruned_count, 0);
    assert_eq!(report.freed_chars, 0);
    assert_eq!(without_protocol(&ambiguous), before);
    assert!(ambiguous
        .iter()
        .filter_map(provenance)
        .all(|meta| !meta.is_offloaded()));
    assert!(!dir.path().join("pruned-folds").exists());

    // Control: a single note with this identity stays an ordinary candidate.
    for control in [note, changed_note] {
        let mut request = old_fold_projection(control);
        let found = candidates(&request);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, 2);
        assert_eq!(found[0].1.id(), id);
        let control_marks = FxHashMap::from_iter([(id.clone(), llm_prune::PRUNE_THRESHOLD)]);
        let report =
            llm_prune::prepare_request_projection(&mut request, &control_marks, Some(dir.path()));
        assert_eq!(report.pruned_count, 1);
        assert!(provenance(fold_note(&request)).unwrap().is_offloaded());
    }
}

#[test]
fn folded_prune_ambiguous_marks_do_not_carry_into_a_later_resolved_state() {
    let dir = TestDir::new();
    let raw = raw_group("carryover", &["read_file"]);
    let mut changed_checkpoint = raw.clone();
    changed_checkpoint[0].content =
        Value::String("A different assistant checkpoint over the same call. ".repeat(24));
    let note = fold_group(&raw, dir.path());
    let twin = fold_group(&changed_checkpoint, dir.path());
    let id = provenance(&note).unwrap().id();
    assert_eq!(provenance(&twin).unwrap().id(), id);

    // Two independent rounds mark the note while its identity is unique.
    let unique = old_fold_projection(note.clone());
    let mut marks = FxHashMap::default();
    for _ in 0..llm_prune::PRUNE_THRESHOLD {
        assert!(llm_prune::update_prune_marks_for_messages(
            &mut marks,
            std::slice::from_ref(&id),
            &unique,
        ));
    }
    assert_eq!(marks.get(&id), Some(&llm_prune::PRUNE_THRESHOLD));

    // A twin appears. The id no longer resolves to one item, so the accumulated
    // consent is withdrawn instead of sleeping until the id resolves again.
    let mut ambiguous = vec![
        message("system", "System fixture."),
        message("user", "Inspect the evidence."),
        note.clone(),
        twin,
    ];
    append_recent_raw(&mut ambiguous, KEEP_RECENT_TOOL_GROUPS);
    assert_eq!(
        llm_prune::explain_rejected_prune_mark_authorized(
            &ambiguous,
            &id,
            &PruneAuthorization::default(),
        ),
        Some("fold id is shared by more than one evidence item")
    );
    assert!(llm_prune::update_prune_marks_for_messages(
        &mut marks,
        &[],
        &ambiguous
    ));
    assert!(
        marks.is_empty(),
        "an unresolvable id must lose its accumulated consent: {marks:?}"
    );

    // Even after the twin is gone, the withdrawn consent stays gone: the note
    // needs two fresh decisions before it can be offloaded again.
    let mut resolved = old_fold_projection(note.clone());
    assert_eq!(
        llm_prune::prepare_request_projection(&mut resolved, &marks, Some(dir.path())).pruned_count,
        0
    );
    assert!(!provenance(fold_note(&resolved)).unwrap().is_offloaded());

    // Control: the state is otherwise prunable, so stale consent would have been
    // enough to offload the note, while fresh marks still are.
    let mut stale = old_fold_projection(note.clone());
    let stale_marks = FxHashMap::from_iter([(id.clone(), llm_prune::PRUNE_THRESHOLD)]);
    assert_eq!(
        llm_prune::prepare_request_projection(&mut stale, &stale_marks, Some(dir.path()))
            .pruned_count,
        1
    );
    let mut fresh = old_fold_projection(note);
    let mut fresh_marks = FxHashMap::default();
    for _ in 0..llm_prune::PRUNE_THRESHOLD {
        assert!(llm_prune::update_prune_marks_for_messages(
            &mut fresh_marks,
            std::slice::from_ref(&id),
            &fresh,
        ));
    }
    assert_eq!(
        llm_prune::prepare_request_projection(&mut fresh, &fresh_marks, Some(dir.path()))
            .pruned_count,
        1
    );
}

#[test]
fn folded_prune_incomplete_raw_anchor_holds_no_recent_window_slot() {
    let mut messages = vec![message("user", "Inspect the evidence.")];
    let mut anchors = Vec::new();
    for index in 0..=KEEP_RECENT_TOOL_GROUPS {
        anchors.push(messages.len());
        messages.extend(raw_group(&format!("complete-{index}"), &["read_file"]));
    }
    // An interrupted call that never produced a response holds no evidence, so it
    // must not push the oldest complete group out of the protection window.
    let mut interrupted = message("assistant", "Interrupted request without results.");
    interrupted.tool_calls = raw_group("interrupted", &["read_file"])
        .swap_remove(0)
        .tool_calls;
    let anchor = messages.len();
    messages.push(interrupted);

    let recent = recent_group_indices(&messages);
    assert!(!recent.contains(&anchor));
    assert_eq!(
        recent,
        anchors[1..].iter().copied().collect::<FxHashSet<_>>(),
        "the anchor must claim no slot"
    );
    let active = llm_prune::active_prunable_tool_ids(&messages);
    assert!(
        !active.contains("complete-1-0"),
        "the oldest protected group must keep its protection"
    );
    assert!(
        active.contains("complete-0-0"),
        "the fixture results must stay prunable once the window passes them"
    );
}

#[test]
fn folded_prune_partially_answered_anchor_keeps_its_recent_window_slot() {
    let mut messages = vec![message("user", "Inspect the evidence.")];
    let mut anchors = Vec::new();
    for index in 0..=KEEP_RECENT_TOOL_GROUPS {
        anchors.push(messages.len());
        messages.extend(raw_group(&format!("complete-{index}"), &["read_file"]));
    }
    // The newest anchor produced one response and never received the other. It did
    // hold evidence, so it keeps its slot: dropping the slot would evict the oldest
    // protected group in favour of an anchor whose surviving result stays prunable.
    let mut partial = raw_group("partial", &["read_file", "read_file"]);
    let unanswered = partial.pop().expect("second response");
    assert_eq!(unanswered.tool_call_id.as_deref(), Some("partial-1"));
    let anchor = messages.len();
    messages.extend(partial);

    let recent = recent_group_indices(&messages);
    assert!(recent.contains(&anchor), "an answered anchor keeps its slot");
    let expected = std::iter::once(anchor)
        .chain(
            anchors
                .iter()
                .rev()
                .take(KEEP_RECENT_TOOL_GROUPS - 1)
                .copied(),
        )
        .collect::<FxHashSet<_>>();
    assert_eq!(
        recent, expected,
        "the answered anchor consumes one of the window slots"
    );
    let active = llm_prune::active_prunable_tool_ids(&messages);
    assert!(
        !active.contains("partial-0"),
        "the surviving response of a partially answered anchor stays protected"
    );
    assert!(
        !active.contains("complete-2-0"),
        "a group inside the window stays protected"
    );
    assert!(
        active.contains("complete-1-0"),
        "a group pushed out of the window becomes prunable"
    );
}

#[test]
fn folded_prune_restore_retains_only_complete_surviving_canonical_keys() {
    let dir = TestDir::new();
    let raw = raw_group("restore", &["read_file", "execute_command"]);
    let note = fold_group(&raw, dir.path());
    let id = provenance(&note).unwrap().id();
    let mut marks = FxHashMap::from_iter([
        (id.clone(), llm_prune::PRUNE_THRESHOLD),
        ("fold_no_longer_present".to_string(), 1),
    ]);

    assert!(llm_prune::retained_prune_ids(&raw).contains(&id));
    assert!(!llm_prune::active_prunable_tool_ids(&raw).contains(&id));
    assert!(llm_prune::update_prune_marks_for_messages(
        &mut marks,
        &[],
        &raw
    ));
    assert_eq!(
        marks,
        FxHashMap::from_iter([(id.clone(), llm_prune::PRUNE_THRESHOLD)])
    );
    assert!(!llm_prune::update_prune_marks_for_messages(
        &mut marks,
        std::slice::from_ref(&id),
        &raw,
    ));

    let mut partial = raw.clone();
    partial.pop();
    assert!(!canonical_fold_ids(&partial).contains(&id));
    assert!(!llm_prune::retained_prune_ids(&partial).contains(&id));
    assert!(llm_prune::update_prune_marks_for_messages(
        &mut marks,
        &[],
        &partial
    ));
    assert!(marks.is_empty());

    let mut changed_arguments = raw.clone();
    changed_arguments[0].tool_calls.as_mut().unwrap()[0]
        .function
        .arguments = json!({"file_path": "src/other.rs"}).to_string();
    assert!(!canonical_fold_ids(&changed_arguments).contains(&id));
    let mut duplicate_result = raw.clone();
    duplicate_result[2] = duplicate_result[1].clone();
    assert!(!canonical_fold_ids(&duplicate_result).contains(&id));
    let mut protected = raw.clone();
    protected[0].tool_calls.as_mut().unwrap()[1].function.name = "plan".to_string();
    assert!(canonical_fold_ids(&protected).is_empty());
}

#[test]
fn folded_prune_consent_is_bound_to_the_observed_result_not_reused_call_ids() {
    let dir = TestDir::new();
    let raw = raw_group("reused", &["read_file"]);
    let original = fold_group(&raw, dir.path());
    let id = provenance(&original).unwrap().id();
    let mut changed = raw.clone();
    changed[1].content = Value::String("Different evidence under the same call ID.\n".repeat(160));
    let replacement = fold_group(&changed, dir.path());
    assert_ne!(provenance(&replacement).unwrap().id(), id);
    assert!(!canonical_fold_ids(&changed).contains(&id));

    let marks = FxHashMap::from_iter([(id.clone(), llm_prune::PRUNE_THRESHOLD)]);
    let mut request = old_fold_projection(replacement);
    let before = request.clone();
    assert_eq!(
        llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path())).pruned_count,
        0
    );
    assert_eq!(without_protocol(&request), before);
    let mut restored_marks = marks;
    assert!(llm_prune::update_prune_marks_for_messages(
        &mut restored_marks,
        &[],
        &changed
    ));
    assert!(restored_marks.is_empty());
}

#[test]
fn folded_prune_id_collisions_never_authorize_raw_or_ambiguous_evidence() {
    let dir = TestDir::new();
    let note = fold_group(&raw_group("fold-source", &["read_file"]), dir.path());
    let id = provenance(&note).unwrap().id();
    let marks = FxHashMap::from_iter([(id.clone(), u8::MAX)]);
    for name in ["read_file", "plan", "task"] {
        let mut raw = raw_group("colliding", &[name]);
        raw[0].tool_calls.as_mut().unwrap()[0].id = id.clone();
        raw[1].tool_call_id = Some(id.clone());
        for include_fold in [false, true] {
            let mut request = if include_fold {
                vec![note.clone()]
            } else {
                Vec::new()
            };
            request.extend(raw.clone());
            append_recent_raw(&mut request, KEEP_RECENT_TOOL_GROUPS);
            let before = request.clone();
            assert!(!llm_prune::active_prunable_tool_ids(&request).contains(&id));
            assert_eq!(
                llm_prune::prepare_request_projection(&mut request, &marks, Some(dir.path()))
                    .pruned_count,
                0,
                "fold consent must not apply to {name} (include_fold={include_fold})"
            );
            assert_eq!(without_protocol(&request), before);
            assert_complete_pairs(&request);
        }
    }
}

#[test]
fn folded_prune_provenance_survives_serialization_but_not_provider_normalization() {
    let dir = TestDir::new();
    let raw = raw_group("serialized", &["read_file", "execute_command"]);
    let note = fold_group(&raw, dir.path());
    let id = provenance(&note).unwrap().id();
    let encoded = serde_json::to_string(&note).unwrap();
    assert!(encoded.contains(FOLDED_TOOL_ORIGIN));
    let restored: Message = serde_json::from_str(&encoded).unwrap();
    assert_eq!(restored, note);
    assert_eq!(provenance(&restored).unwrap().id(), id);
    let mut projection = old_fold_projection(restored);
    let marks = FxHashMap::from_iter([(id.clone(), llm_prune::PRUNE_THRESHOLD)]);
    assert_eq!(
        llm_prune::prepare_request_projection(&mut projection, &marks, Some(dir.path()))
            .pruned_count,
        1
    );
    let encoded_stub = serde_json::to_string(fold_note(&projection)).unwrap();
    let restored_stub: Message = serde_json::from_str(&encoded_stub).unwrap();
    assert!(provenance(&restored_stub).unwrap().is_offloaded());
    assert_eq!(provenance(&restored_stub).unwrap().id(), id);

    for fold in [note, restored_stub] {
        let mut assistant = message("assistant", "Ordinary answer.");
        assistant.reasoning_content = Some("Ordinary provider reasoning.".to_string());
        let messages = vec![message("user", "Continue."), fold, assistant.clone()];
        let before = messages.clone();
        let normalized = normalize_messages_for_request_for_test(&messages);
        let wire = serde_json::to_string(&normalized).unwrap();
        assert!(!wire.contains(FOLDED_TOOL_ORIGIN));
        assert!(wire.contains("compressed_tool_round:"));
        assert!(
            normalized
                .iter()
                .all(|message| message.role != ROLE_INTERNAL_NOTE)
        );
        assert!(normalized.iter().any(|message| message == &assistant));
        assert_eq!(
            messages, before,
            "provider normalization must not mutate persistence"
        );
        assert_eq!(provenance(&messages[1]).unwrap().id(), id);
    }
}

#[test]
fn folded_source_ids_stay_nameable_for_marks() {
    let dir = TestDir::new();
    // Ids whose spelling sits outside the parser's fallback shape: they can only
    // resolve through the ids the projection carries.
    let raw = raw_group("call:opaque/id", &["read_file", "execute_command"]);
    let note = fold_group(&raw, dir.path());
    let meta = provenance(&note).expect("fixture note carries provenance");
    let source = raw[1].tool_call_id.clone().expect("raw result id");

    // Only the folded projection is left: the raw results are gone, while the
    // offloaded stub renders each constituent id, so a mark naming one must still
    // resolve instead of being read as prose.
    let nameable = llm_prune::nameable_prune_ids(std::slice::from_ref(&note));
    let (ids, remaining) =
        llm_prune::parse_prune_from_hidden_meta(&format!("prune:{source}"), &nameable);

    assert!(nameable.contains(&meta.id()));
    assert_eq!(ids, vec![source]);
    assert!(remaining.is_empty());
}
