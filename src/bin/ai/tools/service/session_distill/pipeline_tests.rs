use super::*;
use std::future::ready;

const QUOTE: &str = "Keep configured topics independent and give the final topic highest priority.";

fn source(id: usize) -> SourceSegment {
    SourceSegment { id: format!("m{id}"), role: "user".into(), text: QUOTE.into() }
}

fn proposal(id: usize, message_id: &str) -> Value {
    json!({"topic_key":format!("topic-{id}"), "category":"decision_log",
        "note":format!("Apply the configured policy independently to topic {id}."),
        "evidence":[{"message_id":message_id,"quote":QUOTE}], "replaces":null})
}

fn items<'a>(payload: &'a Value, key: &str) -> &'a [Value] {
    payload[key].as_array().unwrap().as_slice()
}

fn phase(payload: &Value) -> &str {
    payload["phase"].as_str().unwrap()
}

fn id(value: &Value) -> usize {
    value["id"].as_u64().unwrap() as usize
}

fn source_index(value: &Value) -> usize {
    value["id"].as_str().unwrap().strip_prefix('m').unwrap().parse().unwrap()
}

fn checked(budget: &RequestBudget, payload: &Value) {
    budget.check(payload).unwrap_or_else(|error| panic!("{} request: {error}", phase(payload)));
}

fn answer(payload: &Value, extracted: &mut Option<Vec<Value>>, related: bool) -> Value {
    match phase(payload) {
        "extract" => json!({"complete":true,"conclusions":extracted.take().unwrap_or_default()}),
        "rank" => json!({"scores":items(payload,"proposals").iter()
            .map(|item| json!({"id":item["id"],"utility":50})).collect::<Vec<_>>()}),
        "group" => json!({"decisions":items(payload,"proposals").iter()
            .map(|item| json!({"id":item["id"],"related":related})).collect::<Vec<_>>()}),
        "merge" => {
            let carry = if payload["carry"].is_null() { &payload["proposals"][0] }
                else { &payload["carry"] };
            json!({"complete":true,"conclusions":[carry]})
        }
        "catalog" => json!({"decisions":items(payload,"existing").iter()
            .map(|item| json!({"id":item["id"],"status":"unrelated"})).collect::<Vec<_>>()}),
        "verify" => json!({"verdicts":[{"id":0,"status":"supported"}]}),
        other => panic!("Unexpected phase {other}"),
    }
}

fn existing(id: usize) -> AgentMemoryEntry {
    let mut entry = AgentMemoryEntry {
        id: Some(format!("existing-{id}")), category: "decision_log".into(),
        note: format!("Preserve the established independent policy for existing topic {id}."),
        ..Default::default()
    };
    entry.distilled = Some(DistilledMetadata {
        schema: DISTILLED_SCHEMA, scope: "test-project".into(), revision: 1,
        topic_key: format!("existing-topic-{id}"), verified: true,
        content_digest: entry_content_digest(&entry),
        evidence: vec![DistilledEvidence {
            source_digest: "test-source".into(), message_id: "m0".into(), role: "user".into(),
            quote: QUOTE.into(), text_digest: digest(QUOTE),
        }],
        source_digests: vec!["test-source".into()], previous_revisions: vec![],
    });
    assert!(active_distilled_metadata(&entry).is_some());
    entry
}

#[tokio::test]
async fn session_distill_over_100k_proposals_rank_before_limit_and_keep_late_winner() {
    const COUNT: usize = 100_001;
    const PER_SOURCE: usize = 64;
    let sources: Vec<_> = (0..COUNT.div_ceil(PER_SOURCE)).map(source).collect();
    let budget = RequestBudget::for_test(24_000);
    let mut extracted = vec![false; COUNT];
    let mut ranked = vec![false; COUNT];
    let mut grouped = vec![false; COUNT];
    let (mut extract_calls, mut rank_calls, mut group_calls) = (0, 0, 0);
    let mut emitted = 0;
    let mut ranked_count = 0;
    let mut empty = None;
    let (accepted, rejected) = run(&sources, &[], 1, &budget, |payload| {
        checked(&budget, &payload);
        let response = match phase(&payload) {
            "extract" => {
                assert_eq!(rank_calls, 0, "Extraction must finish before ranking starts");
                extract_calls += 1;
                let mut proposals = Vec::new();
                for segment in items(&payload, "source") {
                    let start = source_index(segment) * PER_SOURCE;
                    for index in start..(start + PER_SOURCE).min(COUNT) {
                        assert!(!std::mem::replace(&mut extracted[index], true));
                        proposals.push(proposal(index, segment["id"].as_str().unwrap()));
                        emitted += 1;
                    }
                }
                json!({"complete":true,"conclusions":proposals})
            }
            "rank" => {
                assert_eq!(emitted, COUNT);
                rank_calls += 1;
                let scores: Vec<_> = items(&payload, "proposals").iter().map(|item| {
                    let index = id(item);
                    assert!(!std::mem::replace(&mut ranked[index], true));
                    ranked_count += 1;
                    json!({"id":index,"utility":if index == COUNT - 1 {100} else {1}})
                }).collect();
                json!({"scores":scores})
            }
            "group" => {
                assert_eq!(ranked_count, COUNT, "Final selection cannot truncate ranking");
                assert_eq!(id(&payload["seed"]), COUNT - 1);
                group_calls += 1;
                for item in items(&payload, "proposals") {
                    assert!(!std::mem::replace(&mut grouped[id(item)], true));
                }
                answer(&payload, &mut empty, false)
            }
            _ => answer(&payload, &mut empty, false),
        };
        ready(Ok(response.to_string()))
    }).await.unwrap();
    assert!(extract_calls > 1 && rank_calls > 1 && group_calls > 1);
    assert_eq!(emitted, COUNT);
    assert!(extracted.into_iter().all(|seen| seen));
    assert!(ranked.into_iter().all(|seen| seen));
    assert_eq!(grouped.into_iter().filter(|seen| *seen).count(), COUNT - 1);
    assert_eq!(accepted.len(), 1);
    assert_eq!(accepted[0].topic_key, format!("topic-{}", COUNT - 1));
    assert_eq!(rejected, 0);
}

#[tokio::test]
async fn session_distill_more_than_twelve_nonshrinking_unique_candidates_finish() {
    const COUNT: usize = 17;
    let sources = [source(0)];
    let budget = RequestBudget::for_test(24_000);
    let mut extracted = Some((0..COUNT).map(|index| proposal(index, "m0")).collect());
    let mut ranked = BTreeSet::new();
    let (mut merges, mut verifies) = (0, 0);
    let (accepted, rejected) = run(&sources, &[], COUNT, &budget, |payload| {
        checked(&budget, &payload);
        match phase(&payload) {
            "rank" => for item in items(&payload, "proposals") { assert!(ranked.insert(id(item))); },
            "merge" => {
                merges += 1;
                assert_eq!(items(&payload, "proposals").len(), 1);
                assert!(payload["carry"].is_null());
            }
            "verify" => verifies += 1,
            _ => {}
        }
        ready(Ok(answer(&payload, &mut extracted, false).to_string()))
    }).await.unwrap();
    assert_eq!(ranked.len(), COUNT);
    assert_eq!((merges, verifies, accepted.len(), rejected), (COUNT, COUNT, COUNT, 0));
    assert_eq!(accepted.iter().map(|item| &item.topic_key).collect::<BTreeSet<_>>().len(), COUNT);
}

#[tokio::test]
async fn session_distill_related_candidates_merge_every_page_with_nonshrinking_carry() {
    const COUNT: usize = 257;
    let sources = [source(0)];
    let budget = RequestBudget::for_test(16_000);
    let mut extracted = Some((0..COUNT).map(|index| proposal(index, "m0")).collect());
    let mut merged = BTreeSet::new();
    let mut previous_carry = Value::Null;
    let (mut merge_calls, mut grouped) = (0, BTreeSet::new());
    let (accepted, rejected) = run(&sources, &[], 1, &budget, |payload| {
        checked(&budget, &payload);
        if phase(&payload) == "group" {
            for item in items(&payload, "proposals") { assert!(grouped.insert(id(item))); }
        }
        if phase(&payload) == "merge" {
            merge_calls += 1;
            assert_eq!(payload["carry"], previous_carry);
            assert!(!items(&payload, "proposals").is_empty());
            for item in items(&payload, "proposals") {
                assert!(merged.insert(item["topic_key"].as_str().unwrap().to_owned()));
            }
        }
        let response = answer(&payload, &mut extracted, true);
        if phase(&payload) == "merge" { previous_carry = response["conclusions"][0].clone(); }
        ready(Ok(response.to_string()))
    }).await.unwrap();
    assert!(merge_calls >= COUNT.div_ceil(64));
    assert_eq!((grouped.len(), merged.len(), accepted.len(), rejected), (COUNT - 1, COUNT, 1, 0));
    assert_eq!(serde_json::to_value(&accepted[0]).unwrap(), previous_carry);
}

#[tokio::test]
async fn session_distill_source_over_four_million_characters_has_full_two_phase_coverage() {
    let sources: Vec<_> = (0..701).map(|index| {
        let mut segment = source(index);
        segment.text.push_str(&"x".repeat(6_000 - QUOTE.len()));
        segment
    }).collect();
    assert!(sources.iter().map(|s| s.text.chars().count()).sum::<usize>() > 4_000_000);
    let budget = RequestBudget::for_test(20_000);
    let mut extracted = Some(vec![proposal(0, "m0")]);
    let mut extraction_text = vec![String::new(); sources.len()];
    let mut verification_text = vec![String::new(); sources.len()];
    let (mut extracts, mut verifies) = (0, 0);
    let (accepted, rejected) = run(&sources, &[], 1, &budget, |payload| {
        checked(&budget, &payload);
        let coverage = match phase(&payload) {
            "extract" => { extracts += 1; Some(&mut extraction_text) }
            "verify" => { verifies += 1; Some(&mut verification_text) }
            _ => None,
        };
        if let Some(coverage) = coverage {
            for segment in items(&payload, "source") {
                coverage[source_index(segment)].push_str(segment["text"].as_str().unwrap());
            }
        }
        ready(Ok(answer(&payload, &mut extracted, false).to_string()))
    }).await.unwrap();
    assert!(extracts > 1 && verifies > 1);
    for (index, segment) in sources.iter().enumerate() {
        assert_eq!(extraction_text[index], segment.text, "Extraction source {index}");
        assert_eq!(verification_text[index], segment.text, "Verification source {index}");
    }
    assert_eq!((accepted.len(), rejected), (1, 0));
}

#[tokio::test]
async fn session_distill_incomplete_extraction_bisects_utf8_without_retaining_partial_results() {
    let text = "甲乙🦀e\u{301}丙丁戊".repeat(31);
    let sources = [SourceSegment { id: "m0".into(), role: "user".into(), text: text.clone() }];
    let budget = RequestBudget::for_test(24_000);
    let mut completed = String::new();
    let mut attempts = 0;
    let (accepted, rejected) = run(&sources, &[], 1, &budget, |payload| {
        checked(&budget, &payload);
        assert_eq!(phase(&payload), "extract");
        attempts += 1;
        assert!(attempts <= 2 * text.chars().count() - 1, "Bisection must make finite progress");
        let page = items(&payload, "source");
        assert_eq!(page.len(), 1);
        assert_eq!(page[0]["id"], "m0");
        let piece = page[0]["text"].as_str().unwrap();
        assert!(!piece.is_empty());
        let response = if piece.chars().count() > 7 {
            let mut partial = proposal(0, "m0");
            partial["evidence"][0]["quote"] = json!(piece.chars().take(8).collect::<String>());
            json!({"complete":false,"conclusions":[partial]})
        } else {
            completed.push_str(piece);
            json!({"complete":true,"conclusions":[]})
        };
        ready(Ok(response.to_string()))
    }).await.unwrap();
    assert!(attempts > 1);
    assert_eq!(completed, text);
    assert!(accepted.is_empty());
    assert_eq!(rejected, 0, "Incomplete attempt proposals are discarded, not retained");
}

#[tokio::test]
async fn session_distill_incomplete_single_character_extraction_fails_explicitly() {
    let sources = [SourceSegment { id: "m0".into(), role: "user".into(), text: "界".into() }];
    let budget = RequestBudget::for_test(24_000);
    let mut attempts = 0;
    let result = run(&sources, &[], 1, &budget, |payload| {
        checked(&budget, &payload);
        attempts += 1;
        assert_eq!(phase(&payload), "extract");
        assert_eq!(payload["source"][0]["text"], "界");
        ready(Ok(json!({"complete":false,"conclusions":[]}).to_string()))
    }).await;
    assert!(result.is_err());
    assert!(!result.unwrap_err().is_empty());
    assert_eq!(attempts, 1);
}

#[derive(Clone, Copy, Debug)]
enum BadIds { Missing, Duplicate, Unknown }

fn corrupt_ids(response: &mut Value, phase: &str, fault: BadIds) {
    let key = match phase { "rank" => "scores", "verify" => "verdicts", _ => "decisions" };
    let rows = response[key].as_array_mut().unwrap();
    assert!(!rows.is_empty());
    match fault {
        BadIds::Missing => { rows.pop(); }
        BadIds::Duplicate => rows.push(rows[0].clone()),
        BadIds::Unknown => rows[0]["id"] = if phase == "catalog" { json!("not-in-this-catalog") }
            else { json!(u64::MAX) },
    }
}

#[tokio::test]
async fn session_distill_rank_group_catalog_verify_require_exact_decision_ids() {
    let sources = [source(0)];
    let catalog = [existing(0), existing(1)];
    let budget = RequestBudget::for_test(24_000);
    for target in ["rank", "group", "catalog", "verify"] {
        for fault in [BadIds::Missing, BadIds::Duplicate, BadIds::Unknown] {
            let mut extracted = Some(vec![proposal(0, "m0"), proposal(1, "m0")]);
            let mut corrupted = false;
            let result = run(&sources, &catalog, 1, &budget, |payload| {
                checked(&budget, &payload);
                assert!(!corrupted, "Pipeline continued after invalid {target} IDs: {fault:?}");
                let mut response = answer(&payload, &mut extracted, false);
                if phase(&payload) == target {
                    corrupt_ids(&mut response, target, fault);
                    corrupted = true;
                }
                ready(Ok(response.to_string()))
            }).await;
            assert!(corrupted, "Did not reach {target}");
            assert!(result.is_err(), "Accepted invalid {target} IDs: {fault:?}");
        }
    }
}

#[tokio::test]
async fn session_distill_large_catalog_pages_every_entry_and_matches_last_page() {
    const COUNT: usize = 24_001;
    let sources = [source(0)];
    let catalog: Vec<_> = (0..COUNT).map(existing).collect();
    let budget = RequestBudget::for_test(18_000);
    let mut extracted = Some(vec![proposal(0, "m0")]);
    let mut seen = BTreeSet::new();
    let mut pages = 0;
    let mut matched_page = None;
    let (accepted, rejected) = run(&sources, &catalog, 1, &budget, |payload| {
        checked(&budget, &payload);
        let response = if phase(&payload) == "catalog" {
            pages += 1;
            let decisions: Vec<_> = items(&payload, "existing").iter().map(|entry| {
                let entry_id = entry["id"].as_str().unwrap();
                assert!(seen.insert(entry_id.to_owned()));
                let status = if entry_id == format!("existing-{}", COUNT - 1) {
                    matched_page = Some(pages);
                    "replace"
                } else { "unrelated" };
                json!({"id":entry_id,"status":status})
            }).collect();
            json!({"decisions":decisions})
        } else {
            if phase(&payload) == "verify" {
                assert_eq!(seen.len(), COUNT);
                assert_eq!(items(&payload, "existing").len(), 1);
                assert_eq!(payload["existing"][0]["id"], format!("existing-{}", COUNT - 1));
            }
            answer(&payload, &mut extracted, false)
        };
        ready(Ok(response.to_string()))
    }).await.unwrap();
    assert!(pages >= COUNT.div_ceil(64));
    assert_eq!(matched_page, Some(pages));
    assert_eq!(seen.len(), COUNT);
    assert_eq!((accepted.len(), rejected), (1, 0));
    assert_eq!(accepted[0].topic_key, format!("existing-topic-{}", COUNT - 1));
    assert_eq!(accepted[0].replaces.as_deref(), catalog.last().unwrap().id.as_deref());
}

#[tokio::test]
async fn session_distill_late_correction_vetoes_earlier_supported_conclusion() {
    let mut sources: Vec<_> = (0..129).map(source).collect();
    sources.last_mut().unwrap().text = "Correction: the earlier independent policy is withdrawn.".into();
    let budget = RequestBudget::for_test(24_000);
    let mut extracted = Some(vec![proposal(0, "m0")]);
    let mut verified = BTreeSet::new();
    let (mut supported, mut vetoed) = (false, false);
    let (accepted, rejected) = run(&sources, &[], 1, &budget, |payload| {
        checked(&budget, &payload);
        let response = if phase(&payload) == "verify" {
            let page = items(&payload, "source");
            for item in page { assert!(verified.insert(source_index(item))); }
            let status = if page.iter().any(|item| source_index(item) == 128) {
                assert!(supported, "Correction must arrive after an earlier supported page");
                vetoed = true;
                "contradiction"
            } else if page.iter().any(|item| source_index(item) == 0) {
                supported = true;
                "supported"
            } else { "irrelevant" };
            json!({"verdicts":[{"id":0,"status":status}]})
        } else { answer(&payload, &mut extracted, false) };
        ready(Ok(response.to_string()))
    }).await.unwrap();
    assert!(supported && vetoed);
    assert_eq!(verified.len(), sources.len());
    assert!(accepted.is_empty());
    assert_eq!(rejected, 1);
}

#[tokio::test]
async fn session_distill_intermediate_merge_and_verification_errors_abort_pipeline() {
    for target in ["merge", "verify"] {
        let sources: Vec<_> = (0..129).map(source).collect();
        let budget = RequestBudget::for_test(16_000);
        let count = if target == "merge" { 150 } else { 1 };
        let mut extracted = Some((0..count).map(|index| proposal(index, "m0")).collect());
        let mut target_calls = 0;
        let mut completed_items = 0;
        let result = run(&sources, &[], 1, &budget, |payload| {
            checked(&budget, &payload);
            if phase(&payload) == target {
                target_calls += 1;
                if target_calls == 2 {
                    assert!(completed_items > 0);
                    return ready(Err(format!("injected intermediate {target} failure")));
                }
                completed_items += items(&payload, if target == "merge" { "proposals" } else { "source" }).len();
            }
            ready(Ok(answer(&payload, &mut extracted, true).to_string()))
        }).await;
        assert_eq!(target_calls, 2);
        assert!(result.unwrap_err().contains(&format!("injected intermediate {target} failure")));
        assert!(completed_items < if target == "merge" { count } else { sources.len() });
    }
}
