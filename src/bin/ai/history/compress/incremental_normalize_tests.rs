use super::*;

struct SummaryShrinkDir(PathBuf);

impl SummaryShrinkDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "summary-shrink-chronology-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for SummaryShrinkDir {
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

fn shrink_with_increment(messages: Vec<Message>, dir: Option<&Path>) -> Vec<Message> {
    shrink_messages_to_fit_with_summary(
        messages,
        12_000,
        4_000,
        dir,
        None,
        &rustc_hash::FxHashSet::default(),
    )
}

fn shrink_dialogue(label: &str) -> (Vec<Message>, Vec<Message>) {
    // A short user body avoids proactive spilling. The large plain assistant
    // forces both messages out while the three small recent turns stay intact.
    let fresh = vec![
        msg("user", &format!("{label} fresh request")),
        msg(
            "assistant",
            &format!("{label} answer {}", "x".repeat(24_000)),
        ),
    ];
    let tail = (0..3)
        .flat_map(|index| {
            [
                msg("user", &format!("{label} recent request {index}")),
                msg("assistant", &format!("{label} recent answer {index}")),
            ]
        })
        .collect();
    (fresh, tail)
}

fn prior_increment(dir: &Path) -> Message {
    let plan = incremental::plan_incremental_summary(
        &[msg("user", "PRIOR_ONLY_SENTINEL")],
        "Decisions:\n- PRIOR_ONLY_SENTINEL",
        4_000,
        Some(dir),
    )
    .unwrap();
    assert!(plan.commit());
    plan.message().clone()
}

fn assert_increment_source(message: &Message, expected: &[Message]) {
    let body = value_to_string(&message.content);
    let record: Value = serde_json::from_str(
        body.strip_prefix(INCREMENTAL_SUMMARY_PREFIX)
            .unwrap()
            .trim(),
    )
    .unwrap();
    let bytes =
        std::fs::read_to_string(record["source"]["archive_file_path"].as_str().unwrap()).unwrap();
    let restored: Vec<Message> = bytes
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(restored, expected);
    assert_eq!(
        record["source"]["sha256"],
        content_sha256_hex(bytes.as_bytes())
    );
    assert_eq!(record["source"]["end_line"], expected.len());
    assert_eq!(summary_delta_messages(&restored), restored);
    assert!(
        !record["entries"]
            .to_string()
            .contains("PRIOR_ONLY_SENTINEL")
    );
}

#[test]
fn summary_shrink_appends_chronological_increments_after_system_prefix() {
    // No policy prefix also matters: an increment at index zero must not block
    // summarization of a later, independent span.
    for policy_count in 0..=2 {
        let dir = SummaryShrinkDir::new();
        let policies: Vec<Message> = (0..policy_count)
            .map(|index| msg("system", &format!("runtime policy {index}")))
            .collect();
        let prior = prior_increment(&dir.0);
        let (fresh, tail) = shrink_dialogue("first");
        let mut input = policies.clone();
        input.push(prior.clone());
        input.extend(fresh.clone());
        input.extend(tail.clone());
        let first = shrink_with_increment(input, Some(&dir.0));
        let first_increments: Vec<Message> = first
            .iter()
            .filter(|message| is_incremental_summary(message))
            .cloned()
            .collect();
        assert_eq!(first_increments.len(), 2);
        assert_eq!(first_increments[0], prior);
        assert_increment_source(&first_increments[1], &fresh);
        assert_eq!(summary_delta_messages(&first), tail);

        let (next_fresh, next_tail) = shrink_dialogue("second");
        let mut next_input = first.clone();
        next_input.extend(next_fresh.clone());
        next_input.extend(next_tail.clone());
        let second = shrink_with_increment(next_input, Some(&dir.0));
        let increments: Vec<Message> = second
            .iter()
            .filter(|message| is_incremental_summary(message))
            .cloned()
            .collect();
        assert_eq!(increments.len(), 3);
        assert_eq!(increments[..2], first_increments);
        let mut next_source = tail;
        next_source.extend(next_fresh);
        assert_increment_source(&increments[2], &next_source);
        assert_eq!(summary_delta_messages(&second), next_tail);
        assert!(messages_total_chars(&second) <= 12_000);
        assert_eq!(coalesce_accumulated_summary_notes(second.clone()), second);

        let first_increment_index = second.iter().position(is_incremental_summary).unwrap();
        for policy in &policies {
            assert!(
                second.iter().position(|message| message == policy).unwrap()
                    < first_increment_index
            );
        }
        let normalized = crate::ai::request::normalize_messages_for_request_for_test(&second);
        // Request normalization may combine increments into one assistant message.
        // Compare both message order and text offsets to preserve that valid shape.
        let positions: Vec<(usize, usize)> = increments
            .iter()
            .map(|increment| {
                let body = value_to_string(&increment.content);
                normalized
                    .iter()
                    .enumerate()
                    .find_map(|(index, message)| {
                        if message.role != "assistant" {
                            return None;
                        }
                        value_to_string(&message.content)
                            .find(&body)
                            .map(|offset| (index, offset))
                    })
                    .unwrap()
            })
            .collect();
        assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(!normalized.iter().any(|message| {
            message.role == "system"
                && value_to_string(&message.content).contains(INCREMENTAL_SUMMARY_PREFIX)
        }));

        // A pass with no newly dropped dialogue must neither add an increment
        // nor append the previously summarized bytes to the main archive again.
        let archive = dir.0.join(OVERFLOW_HISTORY_FILENAME);
        let archive_before = std::fs::read(&archive).unwrap();
        assert_eq!(shrink_with_increment(second.clone(), Some(&dir.0)), second);
        assert_eq!(std::fs::read(&archive).unwrap(), archive_before);
        assert_eq!(
            std::fs::read_dir(dir.0.join("summary-sources"))
                .unwrap()
                .count(),
            3
        );
    }
}

#[test]
fn summary_shrink_source_commit_failure_keeps_archive_without_new_increment() {
    let dir = SummaryShrinkDir::new();
    let prior = prior_increment(&dir.0.join("prior"));
    let blocked = dir.0.join("summary-sources");
    std::fs::write(&blocked, "not a directory").unwrap();
    let (fresh, tail) = shrink_dialogue("commit failure");
    assert!(plan_incremental_summary_without_app(&fresh, 4_000, Some(&dir.0)).is_some());
    let policy = msg("system", "runtime policy");
    let mut input = vec![policy.clone(), prior.clone()];
    input.extend(fresh.clone());
    input.extend(tail.clone());
    let output = shrink_with_increment(input, Some(&dir.0));
    assert_eq!(output.first(), Some(&policy));
    assert_eq!(
        output
            .iter()
            .filter(|message| is_incremental_summary(message))
            .cloned()
            .collect::<Vec<_>>(),
        vec![prior]
    );
    assert_eq!(summary_delta_messages(&output), tail);
    let archive_path = dir.0.join(OVERFLOW_HISTORY_FILENAME);
    let archive = std::fs::read_to_string(&archive_path).unwrap();
    for message in fresh {
        assert!(archive.contains(&value_to_string(&message.content)));
    }
    assert!(output.iter().any(|message| {
        is_archive_note_message(message)
            && value_to_string(&message.content).contains(archive_path.to_str().unwrap())
    }));
    assert_eq!(
        std::fs::read_to_string(&blocked).unwrap(),
        "not a directory"
    );
}

#[test]
fn summary_shrink_archive_failure_preserves_prefix_and_fresh_dialogue() {
    let dir = SummaryShrinkDir::new();
    let blocked = dir.0.join("not-a-directory");
    std::fs::write(&blocked, "blocked").unwrap();
    let prior = prior_increment(&dir.0);
    let (fresh, tail) = shrink_dialogue("archive failure");
    let mut input = vec![msg("system", "runtime policy"), prior];
    input.extend(fresh);
    input.extend(tail);
    for sink in [None, Some(blocked.as_path())] {
        assert_eq!(shrink_with_increment(input.clone(), sink), input);
    }
}

#[test]
fn incremental_memory_normalization_preserves_full_record_and_derived_role() {
    // Deliberately exceed the independent request-note cap; source metadata at
    // the end must survive both prefix and mid-stream projection paths.
    let body = format!(
        "{INCREMENTAL_SUMMARY_PREFIX}\n{}",
        serde_json::json!({
            "entries": [{"status": "derived_unverified", "text": "x".repeat(24_000)}],
            "source": {"archive_file_path": "summary-sources/example.jsonl", "sha256": "example"}
        })
    );
    for role in [ROLE_INTERNAL_NOTE, "system"] {
        for mid_stream in [false, true] {
            for with_policy in [false, true] {
                let mut input = Vec::new();
                if with_policy {
                    input.push(msg("system", "runtime policy"));
                }
                if mid_stream {
                    input.push(msg("user", "earlier request"));
                    input.push(msg("assistant", "earlier answer"));
                }
                input.push(msg(role, &body));
                input.push(msg("user", "latest request"));
                let normalized =
                    crate::ai::request::normalize_messages_for_request_for_test(&input);
                assert!(normalized.iter().any(|message| {
                    message.role == "assistant" && value_to_string(&message.content).contains(&body)
                }));
                assert!(!normalized.iter().any(|message| {
                    message.role == "system"
                        && value_to_string(&message.content).contains(INCREMENTAL_SUMMARY_PREFIX)
                }));
                assert_eq!(
                    value_to_string(&normalized.last().unwrap().content),
                    "latest request"
                );
            }
        }
    }
}

/// Plan one source-bound record whose rendered note is roughly `body_chars` long.
fn large_increment(dir: &Path, label: &str, body_chars: usize) -> Message {
    let draft = format!("Current work:\n- {label} {}", "x".repeat(body_chars));
    let plan = incremental::plan_incremental_summary(
        &[msg("user", &format!("{label} source"))],
        &draft,
        8_000,
        Some(dir),
    )
    .unwrap();
    assert!(plan.commit());
    plan.message().clone()
}

fn inline_increments(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .filter(|message| is_incremental_summary(message))
        .cloned()
        .collect()
}

fn inline_increment_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|message| is_incremental_summary(message))
        .map(message_billable_chars)
        .sum()
}

#[test]
fn incremental_window_demotes_oldest_records_without_losing_them() {
    let dir = SummaryShrinkDir::new();
    let records: Vec<Message> = (0..6)
        .map(|index| large_increment(&dir.0, &format!("ROUND{index}"), 3_000))
        .collect();
    assert!(
        records.iter().map(message_billable_chars).sum::<usize>()
            > MAX_INCREMENTAL_SUMMARY_INLINE_CHARS
    );
    let policy = msg("system", "runtime policy");
    let latest = msg("user", "latest request");
    let mut input = vec![policy.clone()];
    input.extend(records.clone());
    input.push(latest.clone());

    let trimmed = trim_incremental_summary_notes_to_inline_budget(
        input.clone(),
        Some(&dir.0),
        MAX_INCREMENTAL_SUMMARY_INLINE_CHARS,
    );
    let kept = inline_increments(&trimmed);
    assert!(!kept.is_empty() && kept.len() < records.len());
    // The newest window stays byte-identical; only the oldest records are demoted.
    assert_eq!(kept, records[records.len() - kept.len()..].to_vec());
    assert!(inline_increment_chars(&trimmed) <= MAX_INCREMENTAL_SUMMARY_INLINE_CHARS);
    assert_eq!(trimmed.first(), Some(&policy));
    assert_eq!(trimmed.last(), Some(&latest));

    // Demoted records are archived verbatim behind exactly one back-reference.
    let archive_path = dir.0.join(OVERFLOW_HISTORY_FILENAME);
    let archive = std::fs::read_to_string(&archive_path).unwrap();
    for demoted in &records[..records.len() - kept.len()] {
        assert!(archive.contains(&value_to_string(&demoted.content)));
    }
    let pointers: Vec<&Message> = trimmed
        .iter()
        .filter(|message| is_archive_note_message(message))
        .collect();
    assert_eq!(pointers.len(), 1);
    assert!(value_to_string(&pointers[0].content).contains(archive_path.to_str().unwrap()));

    // A window already inside the cap is returned untouched: a second pass neither
    // re-archives nor re-orders anything.
    assert_eq!(
        trim_incremental_summary_notes_to_inline_budget(
            trimmed.clone(),
            Some(&dir.0),
            MAX_INCREMENTAL_SUMMARY_INLINE_CHARS
        ),
        trimmed
    );

    // A missing or unwritable archive sink must keep every record inline: dropping
    // memory without a locator would be unrecoverable.
    let blocked = dir.0.join("blocked-sink");
    std::fs::write(&blocked, "blocked").unwrap();
    for sink in [None, Some(blocked.as_path())] {
        assert_eq!(
            trim_incremental_summary_notes_to_inline_budget(
                input.clone(),
                sink,
                MAX_INCREMENTAL_SUMMARY_INLINE_CHARS
            ),
            input
        );
    }
}

#[test]
fn capped_note_prepass_reports_archive_failure_independently_of_global_budget() {
    let dir = SummaryShrinkDir::new();
    let increments: Vec<Message> = (0..6)
        .map(|index| large_increment(&dir.0, &format!("ROUND{index}"), 3_000))
        .collect();
    let evidence: Vec<Message> = (0..6)
        .map(|index| {
            msg(
                ROLE_INTERNAL_NOTE,
                &format!("{COMPRESSED_TOOL_EVIDENCE_MARKER}\nround={index}\n{}", "x".repeat(3_000)),
            )
        })
        .collect();
    assert!(inline_increment_chars(&increments) > MAX_INCREMENTAL_SUMMARY_INLINE_CHARS);
    assert!(compressed_tool_evidence_exceeds_inline_budget(&evidence));
    let blocked = dir.0.join("blocked-prepass-sink");
    std::fs::write(&blocked, "blocked").unwrap();
    for notes in [increments, evidence] {
        let mut messages = vec![msg("system", "runtime policy")];
        messages.extend(notes);
        messages.push(msg("user", "latest request"));
        for (sink, status) in [
            (None, ContextCompressionStatus::MissingArchiveSink),
            (Some(blocked.clone()), ContextCompressionStatus::ArchiveCommitFailed),
        ] {
            for budget in [4_000, 100_000] {
                let outcome = compress_messages_for_context_with_outcome(
                    messages.clone(), budget, usize::MAX, 2_400, sink.clone(), None,
                );
                assert_eq!(outcome.status, status);
                assert_eq!(outcome.messages, messages);
                assert_eq!(outcome.budget_met(), budget == 100_000);
                assert!(outcome.diagnostic().is_some());
            }
        }
    }
}

#[test]
fn incremental_window_stays_bounded_across_compression_rounds() {
    // Regression: one record is appended per compression round and the registered
    // summary prefix protects every older one, so repeated rounds used to grow the
    // projection head linearly. Request-time projection builds must instead stay
    // inside the inline cap plus the record appended by the round itself.
    const ROUNDS: usize = 12;
    let dir = SummaryShrinkDir::new();
    let mut history = vec![msg("system", "runtime policy"), prior_increment(&dir.0)];
    let mut counts = Vec::new();
    let mut totals = Vec::new();
    for round in 0..ROUNDS {
        let (fresh, tail) = shrink_dialogue(&format!("round{round}"));
        let mut input = history.clone();
        input.extend(fresh);
        input.extend(tail);
        history =
            compress_messages_for_context(input, 12_000, 1, 8_000, Some(dir.0.clone()), None);
        counts.push(inline_increments(&history).len());
        totals.push(inline_increment_chars(&history));
    }
    // Premise: these records are large enough that the cap has to bind, otherwise
    // the bound below would hold trivially and prove nothing.
    assert!(
        totals[0] * ROUNDS > MAX_INCREMENTAL_SUMMARY_INLINE_CHARS,
        "test premise: {totals:?}"
    );
    assert!(counts[0] >= 1, "the memory window must stay populated: {counts:?}");
    assert!(
        totals
            .iter()
            .all(|total| *total <= MAX_INCREMENTAL_SUMMARY_INLINE_CHARS + 8_000),
        "inline increments must stay bounded: {totals:?}"
    );
    // Demotion happened: the window stopped taking one more record every round, and
    // the demoted records are archived rather than lost.
    assert!(
        counts.last().unwrap() < &(ROUNDS + 1),
        "records were never demoted: {counts:?}"
    );
    let archive = std::fs::read_to_string(dir.0.join(OVERFLOW_HISTORY_FILENAME)).unwrap();
    assert!(archive.contains(INCREMENTAL_SUMMARY_PREFIX));
}
