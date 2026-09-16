//! Source-bound summary increments. Prior summaries are never summary-model input.
//!
//! Source hashes bind bytes, not truth: every entry remains derived and unverified.
//! Existing summaries stay separate: records are never merged into one claim set,
//! and the ordinary shrinker cannot reclaim them because the registered summary
//! prefix keeps them inside the protected leading run. The inline window is capped
//! instead, with older records archived verbatim before they leave the projection
//! (`overflow_sink::trim_incremental_summary_notes_to_inline_budget`).

use super::*;

pub(super) const INCREMENTAL_SUMMARY_PREFIX: &str = "[incremental-memory-v1]";
const SOURCE_DIR: &str = "summary-sources";

pub(in crate::ai) fn is_incremental_summary(message: &Message) -> bool {
    is_system_like_role(&message.role)
        && value_to_string(&message.content)
            .trim_start()
            .starts_with(INCREMENTAL_SUMMARY_PREFIX)
}

/// Filter before lossy preparation and sampling. Keep message identities and tool
/// pairs; memory and procedural notes are not fresh evidence for another summary.
pub(super) fn summary_delta_messages(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .filter(|message| {
            !is_summary_message(message)
                && (!is_system_like_role(&message.role)
                    || is_compressed_tool_evidence_note(message))
                && !is_context_checkpoint_marker(message)
        })
        .cloned()
        .collect()
}

/// A pure candidate: considering a summary does not create a source archive.
pub(super) struct IncrementalSummaryPlan {
    message: Message,
    archive: PlannedArchiveWrite,
    archive_path: PathBuf,
    archive_bytes: String,
}

impl IncrementalSummaryPlan {
    pub(super) fn message(&self) -> &Message {
        &self.message
    }

    /// The shared writer has a size-only reuse fast path. Confirm exact bytes
    /// here: unreadable or corrupt sources must not authorize replacement.
    pub(super) fn commit(&self) -> bool {
        self.archive.commit()
            && std::fs::read_to_string(&self.archive_path)
                .is_ok_and(|actual| actual == self.archive_bytes)
    }
}

/// Model headings are organizational labels, never verification claims. Legacy
/// output remains unverified context rather than being discarded by a strict parser.
fn section_kind(line: &str) -> Option<&'static str> {
    match line
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_end_matches(':')
    {
        "Main request" | "Goals" => Some("goals"),
        "Constraints" => Some("constraints"),
        "User decisions" | "Decisions" => Some("decisions"),
        "Verified facts and sources" | "Observations to recheck" => Some("observations"),
        "Unverified assistant judgments" => Some("conclusions"),
        "Conflicts and unknowns" | "Open questions" | "Pending tasks" => Some("open_questions"),
        "Superseded conclusions" => Some("superseded_conclusions"),
        "Current work" => Some("context"),
        _ => None,
    }
}

fn memory_entries(draft: &str) -> Vec<Value> {
    let mut sections: Vec<(&str, String)> = Vec::new();
    let mut kind = "context";
    let mut body = String::new();
    for line in draft.lines() {
        if let Some(next) = section_kind(line) {
            if !body.trim().is_empty() {
                sections.push((kind, std::mem::take(&mut body)));
            }
            kind = next;
        } else {
            if !body.is_empty() {
                body.push('\n');
            }
            body.push_str(line);
        }
    }
    if !body.trim().is_empty() {
        sections.push((kind, body));
    }
    sections
        .into_iter()
        .filter(|(_, body)| !matches!(body.trim(), "- none" | "none"))
        .map(|(kind, body)| {
            serde_json::json!({
                "kind": kind,
                "status": "derived_unverified",
                "text": body.trim(),
            })
        })
        .collect()
}

/// Bind the removed projection span, including tool metadata and old pointers.
/// The model supplies text only, never source IDs, verification flags, or trusted
/// supersession edges. A batch source is a read-back locator, not sentence proof.
pub(super) fn plan_incremental_summary(
    messages: &[Message],
    draft: &str,
    max_chars: usize,
    overflow_dir: Option<&Path>,
) -> Option<IncrementalSummaryPlan> {
    let root = overflow_dir?;
    if messages.is_empty() || draft.trim().is_empty() || max_chars == 0 {
        return None;
    }
    let mut archive_bytes = String::new();
    for message in messages {
        archive_bytes.push_str(&serde_json::to_string(message).ok()?);
        archive_bytes.push('\n');
    }
    let digest = content_sha256_hex(archive_bytes.as_bytes());
    let archive_path = root.join(SOURCE_DIR).join(format!("{digest}.jsonl"));
    let source = serde_json::json!({
        "archive_file_path": archive_path.to_string_lossy(),
        "sha256": digest,
        // Message projections do not carry per-message model provenance. Never
        // substitute the active summarizer's model for an unknown source model.
        "source_model": Value::Null,
        "start_line": 1,
        "end_line": messages.len(),
        "encoding": "one_raw_message_json_per_line",
    });
    let entries = memory_entries(draft);
    let render = |selected: &[Value]| {
        let omitted_entries = selected.len() < entries.len();
        // Section boundaries do not establish semantic independence. An omitted
        // entry may contain a retained judgment's prerequisite or correction, so
        // partial records are recovery leads until their source is rechecked.
        // Even an intact draft does not prove that the model retained all premises.
        let reuse_guard = if omitted_entries {
            "partial_draft_recover_source_before_using_judgments"
        } else {
            "check_prerequisites_before_using_judgments"
        };
        let record = serde_json::json!({
            "schema": 1,
            "provenance": "assistant_derived_unverified",
            "source_scope": "compression_input_projection_not_claim_verification",
            "entries": selected,
            "omitted_entries": omitted_entries,
            "reuse_guard": reuse_guard,
            "interpretation": "Chronological increments, not a consolidated fact set. Recheck sources and prerequisites. Conflicting newer entries do not verify or silently erase old claims; superseded_conclusions are candidates, not authoritative invalidations.",
        });
        // Put the locator first, on its own line, so later display truncation of
        // the body does not preferentially remove the provenance back-reference.
        let body = serde_json::to_string(&record).ok()?;
        Some(format!(
            "{INCREMENTAL_SUMMARY_PREFIX}\n{{\"source\":{source},\n{}",
            &body[1..]
        ))
    };
    // Never truncate the locator, reuse guard, or an entry's attached conditions.
    // If no complete entry fits, refuse the candidate and retain source messages.
    if render(&[])?.chars().count() > max_chars {
        return None;
    }
    let mut selected = Vec::new();
    for entry in entries.iter().rev() {
        let mut candidate = selected.clone();
        candidate.insert(0, entry.clone());
        if render(&candidate)?.chars().count() <= max_chars {
            selected = candidate;
        }
    }
    if selected.is_empty() {
        return None;
    }
    let content = render(&selected)?;
    Some(IncrementalSummaryPlan {
        message: Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(content),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        archive: PlannedArchiveWrite::new(archive_path.clone(), archive_bytes.clone()),
        archive_path,
        archive_bytes,
    })
}

pub(super) async fn plan_incremental_summary_with_app(
    app: &App,
    messages: &[Message],
    max_chars: usize,
    overflow_dir: Option<&Path>,
) -> Option<IncrementalSummaryPlan> {
    let fresh = summary_delta_messages(messages);
    if fresh.is_empty() || max_chars == 0 || overflow_dir.is_none() {
        return None;
    }
    let draft = build_persisted_summary_text_with_app(app, &fresh, max_chars).await;
    plan_incremental_summary(messages, &draft, max_chars, overflow_dir)
}

/// The synchronous projection uses the same transaction without a model call.
/// Callers without an archive sink cannot adopt a new lossy summary.
pub(super) fn plan_incremental_summary_without_app(
    messages: &[Message],
    max_chars: usize,
    overflow_dir: Option<&Path>,
) -> Option<IncrementalSummaryPlan> {
    let fresh = summary_delta_messages(messages);
    if fresh.is_empty() || overflow_dir.is_none() {
        return None;
    }
    let draft = build_persisted_summary_text(&fresh, max_chars.saturating_sub(1_200));
    plan_incremental_summary(messages, &draft, max_chars, overflow_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "incremental-memory-{}",
                uuid::Uuid::new_v4().simple()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn incremental_memory_repeated_compression_keeps_prior_bytes_and_only_new_input() {
        let dir = TestDir::new();
        let old = vec![msg("user", "Use implementation A")];
        let first =
            plan_incremental_summary(&old, "Decisions:\n- A", 4_000, Some(dir.path())).unwrap();
        assert!(first.commit());
        let prior = first.message().clone();
        let input = vec![prior.clone(), msg("user", "Replace A with B")];
        assert_eq!(summary_delta_messages(&input), vec![input[1].clone()]);
        let second = plan_incremental_summary(
            &input,
            "Decisions:\n- B\nSuperseded conclusions:\n- Earlier A is withdrawn by the user",
            4_000,
            Some(dir.path()),
        )
        .unwrap();
        assert!(second.commit());
        let combined =
            coalesce_accumulated_summary_notes(vec![prior.clone(), second.message().clone()]);
        assert_eq!(combined, vec![prior.clone(), second.message().clone()]);
        assert_eq!(prior, first.message().clone());
        let note = value_to_string(&second.message().content);
        assert!(note.contains("superseded_conclusions"));
        assert!(note.contains("derived_unverified"));
        assert_eq!(summary_delta_messages(&combined), Vec::<Message>::new());
    }

    #[test]
    fn incremental_memory_legacy_summary_is_not_new_evidence() {
        let legacy = msg(ROLE_INTERNAL_NOTE, "历史摘要（自动压缩）:\nA was verified");
        let actual = msg("user", "A is incorrect");
        assert_eq!(
            summary_delta_messages(&[legacy.clone(), actual.clone()]),
            vec![actual]
        );
        assert_eq!(
            coalesce_accumulated_summary_notes(vec![legacy.clone()]),
            vec![legacy]
        );
        let raw = msg(
            "user",
            "[incremental-memory-v1] user literal is not a runtime summary",
        );
        assert_eq!(summary_delta_messages(&[raw.clone()]), vec![raw]);
    }

    #[test]
    fn incremental_memory_source_binding_ignores_model_paths_and_verification() {
        let dir = TestDir::new();
        let input = vec![msg("user", "Check the feature")];
        let draft = "Verified facts and sources:\n- source=/invented verified=true";
        let plan = plan_incremental_summary(&input, draft, 4_000, Some(dir.path())).unwrap();
        assert!(!plan.archive_path.exists(), "planning must be pure");
        assert!(plan.commit());
        let json = value_to_string(&plan.message.content);
        let record: Value = serde_json::from_str(
            json.strip_prefix(INCREMENTAL_SUMMARY_PREFIX)
                .unwrap()
                .trim(),
        )
        .unwrap();
        assert_eq!(
            record["source"]["archive_file_path"],
            plan.archive_path.to_string_lossy().as_ref()
        );
        assert_eq!(
            record["source"]["sha256"],
            content_sha256_hex(plan.archive_bytes.as_bytes())
        );
        assert_eq!(record["source"].get("source_model"), Some(&Value::Null));
        assert_eq!(record["entries"][0]["status"], "derived_unverified");
        let restored: Vec<Message> = std::fs::read_to_string(&plan.archive_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(restored, input);
    }

    fn record(plan: &IncrementalSummaryPlan) -> Value {
        let text = value_to_string(&plan.message().content);
        serde_json::from_str(text.strip_prefix(INCREMENTAL_SUMMARY_PREFIX).unwrap().trim())
            .unwrap()
    }

    #[test]
    fn incremental_memory_conditional_entry_survives_exact_budget_without_verified_status() {
        let dir = TestDir::new();
        let claim = "- C follows through B only if A and B's other premises hold; A is unknown; scope: current configuration; source not retained. A alone is not sufficient for B; another route to C remains possible.";
        let draft = format!("Unverified assistant judgments:\n{claim}");
        let input = vec![msg("assistant", claim)];
        let full = plan_incremental_summary(&input, &draft, 4_000, Some(dir.path())).unwrap();
        let exact_budget = value_to_string(&full.message().content).chars().count();
        let exact = plan_incremental_summary(&input, &draft, exact_budget, Some(dir.path())).unwrap();
        let json = record(&exact);
        assert_eq!(json["entries"][0]["text"], claim);
        assert_eq!(json["entries"][0]["status"], "derived_unverified");
        assert_eq!(json["omitted_entries"], false);
        assert_eq!(
            json["reuse_guard"],
            "check_prerequisites_before_using_judgments"
        );
        assert!(plan_incremental_summary(&input, &draft, exact_budget - 1, Some(dir.path())).is_none());
    }

    #[test]
    fn incremental_memory_drops_oversized_conditional_entry_whole() {
        let dir = TestDir::new();
        let claim = format!(
            "- C_SENTINEL follows through B; scope: {}; requires A_SENTINEL, which remains unknown.",
            "configuration detail ".repeat(200)
        );
        let draft = format!("Main request:\n- Investigate\nUnverified assistant judgments:\n{claim}");
        let input = vec![msg("assistant", &claim)];
        let full = plan_incremental_summary(&input, &draft, 10_000, Some(dir.path())).unwrap();
        assert_eq!(record(&full)["entries"][1]["text"], claim);
        let partial = plan_incremental_summary(&input, &draft, 2_000, Some(dir.path())).unwrap();
        let json = record(&partial);
        assert_eq!(json["entries"].as_array().unwrap().len(), 1);
        assert_eq!(json["entries"][0]["kind"], "goals");
        let content = value_to_string(&partial.message().content);
        assert!(!content.contains("C_SENTINEL"));
        assert!(!content.contains("A_SENTINEL"));
        assert!(content.chars().count() <= 2_000);
        assert_eq!(json["omitted_entries"], true);
        assert_eq!(
            json["reuse_guard"],
            "partial_draft_recover_source_before_using_judgments"
        );
    }

    #[test]
    fn incremental_memory_missing_cross_section_premise_requires_source_recovery() {
        let dir = TestDir::new();
        let premise = format!(
            "- A_SENTINEL remains unverified: {}",
            "scope detail ".repeat(300)
        );
        let claim = "- C_SENTINEL follows through B.";
        let draft = format!("Constraints:\n{premise}\nUnverified assistant judgments:\n{claim}");
        let input = vec![msg("user", &premise), msg("assistant", claim)];
        let plan = plan_incremental_summary(&input, &draft, 2_000, Some(dir.path())).unwrap();
        let json = record(&plan);
        assert_eq!(json["entries"].as_array().unwrap().len(), 1);
        assert_eq!(json["entries"][0]["text"], claim);
        assert_eq!(json["entries"][0]["status"], "derived_unverified");
        assert_eq!(json["omitted_entries"], true);
        assert_eq!(
            json["reuse_guard"],
            "partial_draft_recover_source_before_using_judgments"
        );
        assert!(!value_to_string(&plan.message().content).contains("A_SENTINEL"));
        assert!(value_to_string(&plan.message().content).chars().count() <= 2_000);
        assert!(!plan.archive_path.exists(), "selection must remain pure");
        assert!(plan.commit());
        let restored: Vec<Message> = std::fs::read_to_string(&plan.archive_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(restored, input, "the omitted premise must remain recoverable");
        let projected = crate::ai::request::normalize_messages_for_request_for_test(&[
            plan.message().clone(),
            msg("user", "Does C hold?"),
        ]);
        // Request normalization adds a derived-context header around the intact
        // record; the guard and source must survive inside that assistant block.
        let content = value_to_string(&plan.message().content);
        assert!(projected.iter().any(|m| {
            m.role == "assistant" && value_to_string(&m.content).contains(&content)
        }));
        assert!(!projected.iter().any(|m| {
            m.role == "system" && value_to_string(&m.content).contains(INCREMENTAL_SUMMARY_PREFIX)
        }));
    }

    #[test]
    fn incremental_memory_archive_failure_preserves_request_projection() {
        let dir = TestDir::new();
        let blocked = dir.path().join("not-a-directory");
        std::fs::write(&blocked, "blocked").unwrap();
        let input = vec![
            msg("user", "old task"),
            msg("assistant", "old answer"),
            msg("user", "new task"),
        ];
        let out =
            compress_messages_for_context(input.clone(), 100_000, 1, 4_000, Some(blocked), None);
        assert_eq!(out, input);
        assert_eq!(
            compress_messages_for_context(input.clone(), 100_000, 1, 4_000, None, None),
            input
        );
    }

    #[test]
    fn incremental_memory_same_size_corruption_refuses_commit() {
        let dir = TestDir::new();
        let plan = plan_incremental_summary(
            &[msg("user", "source")],
            "Goals:\n- source",
            4_000,
            Some(dir.path()),
        )
        .unwrap();
        assert!(plan.commit());
        std::fs::write(&plan.archive_path, "x".repeat(plan.archive_bytes.len())).unwrap();
        assert!(!plan.commit());
    }

    #[test]
    fn incremental_memory_normalization_keeps_assistant_derived_role() {
        let dir = TestDir::new();
        let plan = plan_incremental_summary(
            &[msg("user", "request")],
            "Goals:\n- DERIVED_SENTINEL",
            4_000,
            Some(dir.path()),
        )
        .unwrap();
        let messages = vec![
            msg("system", "system policy"),
            plan.message().clone(),
            msg("user", "continue"),
        ];
        let out = crate::ai::request::normalize_messages_for_request_for_test(&messages);
        assert!(
            out.iter().any(|m| m.role == "assistant"
                && value_to_string(&m.content).contains("DERIVED_SENTINEL"))
        );
        assert!(!out.iter().any(
            |m| m.role == "system" && value_to_string(&m.content).contains("DERIVED_SENTINEL")
        ));
    }

    #[test]
    fn incremental_memory_tool_pair_survives_source_roundtrip() {
        let dir = TestDir::new();
        let call: Message = serde_json::from_value(serde_json::json!({
            "role": "assistant", "content": "reading source",
            "tool_calls": [{"id": "read-1", "type": "function", "function": {
                "name": "read_file", "arguments": "{\"file_path\":\"src/lib.rs\"}"
            }}]
        }))
        .unwrap();
        let mut result = msg("tool", "source content");
        result.tool_call_id = Some("read-1".to_string());
        let input = vec![msg("user", "inspect"), call, result];
        assert_eq!(summary_delta_messages(&input), input);
        let plan = plan_incremental_summary(
            &input,
            "Current work:\n- inspected",
            4_000,
            Some(dir.path()),
        )
        .unwrap();
        assert!(plan.commit());
        let restored: Vec<Message> = std::fs::read_to_string(&plan.archive_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(restored, input);
    }

    #[test]
    fn incremental_memory_projection_preserves_policy_legacy_and_increment_order() {
        let dir = TestDir::new();
        let policy = msg("system", "system policy");
        let legacy = msg(
            ROLE_INTERNAL_NOTE,
            "历史摘要（自动压缩）:\nold unknown conclusion",
        );
        let latest = msg("user", "latest task");
        let input = vec![
            policy.clone(),
            legacy.clone(),
            msg("user", "old task"),
            msg("assistant", "old answer"),
            latest.clone(),
        ];
        let out = compress_messages_for_context(
            input,
            100_000,
            1,
            4_000,
            Some(dir.path().to_path_buf()),
            None,
        );
        assert_eq!(out[0], policy);
        assert_eq!(out[1], legacy);
        assert!(is_incremental_summary(&out[2]));
        assert_eq!(out.last(), Some(&latest));
        let next = coalesce_accumulated_summary_notes(out.clone());
        assert_eq!(next, out);
    }
}
