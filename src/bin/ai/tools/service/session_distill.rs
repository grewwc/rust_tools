//! Complete-archive semantic distillation with independent source verification.
//! Source messages are evidence, never entries. Oversized inputs fail closed
//! rather than silently distilling a prefix of the conversation.

mod input;
pub(in crate::ai) use input::DistillInput;

use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::Read;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use crate::ai::history::{Message, is_runtime_synthetic_user_message, read_all_messages_sqlite, value_to_string};
use crate::ai::knowledge::distilled::{DISTILLED_SCHEMA, DistilledEvidence, DistilledMetadata,
    active_distilled_metadata, current_project_scope, digest, entry_content_digest};
use crate::ai::tools::service::memory::{MemoryOwnerScope, prepare_memory_save_entry};
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::types::App;

pub(crate) const DEFAULT_DISTILL_LIMIT: usize = 20;
pub(crate) const DISTILL_TAG: &str = "session-distill";
const SEGMENT_CHARS: usize = 6000;
const CHUNK_CHARS: usize = 18000;
const MAX_SOURCE_CHARS: usize = 4_000_000;
const MAX_REDUCE_CHARS: usize = 100_000;
const MAX_CATALOG_CHARS: usize = 24_000;
const MODEL_RULES: &str = include_str!("prompts/session_distill_rules.md");
const EXTRACTION_SCHEMA: &str = r#"{"conclusions":[{"topic_key":"stable-lowercase-topic-key","category":"architecture|decision_log|user_preference|project_info|user_memory","note":"one self-contained durable conclusion","evidence":[{"message_id":"m1p1","quote":"exact source quote"}],"replaces":null}]}"#;

#[derive(Debug, Clone, Serialize)]
struct SourceSegment { id: String, role: String, text: String }
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EvidenceRef { message_id: String, quote: String }
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Conclusion {
    topic_key: String, category: String, note: String,
    evidence: Vec<EvidenceRef>, replaces: Option<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Extraction { conclusions: Vec<Conclusion> }
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Verification { verdicts: Vec<Verdict> }
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Verdict { id: usize, status: String }

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DistillReport {
    pub(crate) session_id: String,
    pub(crate) extracted: usize,
    pub(crate) saved: usize,
    pub(crate) updated: usize,
    pub(crate) duplicates: usize,
    pub(crate) rejected: usize,
    pub(crate) chunks: usize,
    pub(crate) dry_run: bool,
}

fn parse_json<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, String> {
    let raw = raw.trim();
    let raw = raw.strip_prefix("```json").or_else(|| raw.strip_prefix("```"))
        .and_then(|text| text.strip_suffix("```"))
        .unwrap_or(raw).trim();
    serde_json::from_str(raw).map_err(|err| format!("Invalid semantic response; no knowledge saved: {err}"))
}

fn source_segments(messages: &[Message]) -> Result<Vec<SourceSegment>, String> {
    let mut segments = Vec::new();
    let mut count = 0usize;
    for (index, message) in messages.iter().enumerate() {
        if !matches!(message.role.as_str(), "user" | "assistant" | "tool")
            || is_runtime_synthetic_user_message(message) { continue; }
        let text = value_to_string(&message.content);
        let chars: Vec<char> = text.chars().collect();
        count = count.saturating_add(chars.len());
        if count > MAX_SOURCE_CHARS {
            return Err("Archive exceeds the full-coverage source budget; nothing saved. Split the archive explicitly.".into());
        }
        for (part, chars) in chars.chunks(SEGMENT_CHARS).enumerate() {
            segments.push(SourceSegment { id: format!("m{}p{}", index + 1, part + 1),
                role: message.role.clone(), text: chars.iter().collect() });
        }
    }
    Ok(segments)
}

fn source_chunks(segments: &[SourceSegment]) -> Vec<&[SourceSegment]> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut size = 0;
    for (index, segment) in segments.iter().enumerate() {
        let len = segment.text.chars().count();
        if index > start && size + len > CHUNK_CHARS {
            chunks.push(&segments[start..index]); start = index; size = 0;
        }
        size += len;
    }
    if start < segments.len() { chunks.push(&segments[start..]); }
    chunks
}

fn resolve_evidence(conclusion: &Conclusion, segments: &[SourceSegment], source_digest: &str)
    -> Result<Vec<DistilledEvidence>, String>
{
    if conclusion.topic_key.is_empty() || conclusion.topic_key.len() > 96
        || !conclusion.topic_key.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
        || !matches!(conclusion.category.as_str(), "architecture" | "decision_log" | "user_preference" | "project_info" | "user_memory")
        || !(20..=2000).contains(&conclusion.note.chars().count())
        || conclusion.evidence.is_empty() || conclusion.evidence.len() > 8
    { return Err("Invalid or unsupported conclusion shape".into()); }
    conclusion.evidence.iter().map(|reference| {
        let segment = segments.iter().find(|segment| segment.id == reference.message_id)
            .ok_or("Unknown evidence segment")?;
        if !matches!(segment.role.as_str(), "user" | "tool")
            || !(8..=1200).contains(&reference.quote.chars().count())
            || !segment.text.contains(&reference.quote)
        { return Err("Evidence must be an exact user/tool quote, not assistant self-attestation".into()); }
        Ok(DistilledEvidence { source_digest: source_digest.to_string(), message_id: segment.id.clone(),
            role: segment.role.clone(), quote: reference.quote.clone(), text_digest: digest(&segment.text) })
    }).collect()
}

/// The callback seam tests coverage and rejection without a live provider.
async fn semantic_conclusions<F, Fut>(segments: &[SourceSegment], existing: &[AgentMemoryEntry],
    limit: usize, mut ask: F) -> Result<(Vec<Conclusion>, usize), String>
where F: FnMut(Value) -> Fut, Fut: Future<Output = Result<String, String>>,
{
    let chunks = source_chunks(segments);
    let mut proposals = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let response = ask(json!({"phase":"extract", "segment_batch":index+1, "total_batches":chunks.len(),
            "instruction":"Extract at most 12 meaningful durable conclusions from this batch. Include explicit corrections, not tentative plans. This is partial context; merge and verification will resolve the full conversation.",
            "schema":EXTRACTION_SCHEMA, "source":chunk})).await?;
        let extraction: Extraction = parse_json(&response)?;
        if extraction.conclusions.len() > 12 {
            return Err("Extraction exceeded the candidate budget; nothing saved".into());
        }
        proposals.extend(extraction.conclusions);
        if serde_json::to_string(&proposals).map_err(|e| e.to_string())?.chars().count() > MAX_REDUCE_CHARS {
            return Err("Complete merge exceeds the context budget; nothing saved. Split the archive explicitly.".into());
        }
    }
    if proposals.is_empty() { return Ok((Vec::new(), 0)); }
    let catalog: Vec<_> = existing.iter().filter_map(|entry| {
        let metadata = active_distilled_metadata(entry)?;
        Some(json!({"id":entry.id,"topic_key":metadata.topic_key,"revision":metadata.revision,"note":entry.note}))
    }).collect();
    if serde_json::to_string(&catalog).map_err(|e| e.to_string())?.chars().count() > MAX_CATALOG_CHARS {
        return Err("Existing-topic catalog exceeds the safe merge budget; nothing saved".into());
    }
    let response = ask(json!({"phase":"merge", "schema":EXTRACTION_SCHEMA, "limit":limit,
        "instruction":include_str!("prompts/session_distill_merge.md"),
        "proposals":proposals, "existing":catalog})).await?;
    let merged: Extraction = parse_json(&response)?;
    if merged.conclusions.len() > limit { return Err("Merge exceeded the requested limit; nothing saved".into()); }
    let proposed_count = merged.conclusions.len();
    let source_digest = digest(serde_json::to_vec(segments).map_err(|e| e.to_string())?);
    let mut conclusions = Vec::new();
    let mut topic_keys = std::collections::BTreeSet::new();
    for conclusion in merged.conclusions {
        if resolve_evidence(&conclusion, segments, &source_digest).is_err() { continue; }
        if !topic_keys.insert(conclusion.topic_key.clone()) {
            return Err("Merge returned conflicting duplicate topics; nothing saved".into());
        }
        if let Some(id) = &conclusion.replaces {
            if !existing.iter().any(|entry| entry.id.as_ref() == Some(id)
                && active_distilled_metadata(entry).is_some_and(|metadata| metadata.topic_key == conclusion.topic_key))
            { return Err("Merge targeted an unknown topic or revision; nothing saved".into()); }
        }
        conclusions.push(conclusion);
    }
    let mut accepted = Vec::new();
    // Re-check every original segment, not merely extractor summaries.
    for batch in conclusions.chunks(6) {
        let mut supported = vec![false; batch.len()];
        let mut vetoed = vec![false; batch.len()];
        for (index, chunk) in chunks.iter().enumerate() {
            let response = ask(json!({"phase":"verify", "segment_batch":index+1, "total_batches":chunks.len(),
                "instruction":include_str!("prompts/session_distill_verify.md"),
                "schema":{"verdicts":[{"id":0,"status":"supported|irrelevant|contradiction|uncertain"}]},
                "conclusions":batch.iter().enumerate().map(|(id,item)|json!({"id":id,"conclusion":item})).collect::<Vec<_>>(),
                "existing":catalog, "source":chunk})).await?;
            let verification: Verification = parse_json(&response)?;
            let mut seen = vec![false; batch.len()];
            for verdict in verification.verdicts {
                if verdict.id >= batch.len() || seen[verdict.id] {
                    return Err("Incomplete or duplicate verification coverage; nothing saved".into());
                }
                seen[verdict.id] = true;
                match verdict.status.as_str() {
                    "supported" => supported[verdict.id] = true,
                    "irrelevant" => {},
                    "contradiction" | "uncertain" => vetoed[verdict.id] = true,
                    _ => return Err("Unknown verification verdict; nothing saved".into()),
                }
            }
            if seen.contains(&false) { return Err("Incomplete verification coverage; nothing saved".into()); }
        }
        for (index, conclusion) in batch.iter().enumerate() {
            if supported[index] && !vetoed[index] { accepted.push(conclusion.clone()); }
        }
    }
    let rejected = proposed_count - accepted.len();
    Ok((accepted, rejected))
}

pub(in crate::ai) async fn run_distill_command(app: &App, zip_path: &Path, limit: usize, dry_run: bool)
    -> Result<DistillReport, String>
{
    run_distill_source_command(app, &DistillInput::Archive(zip_path.to_path_buf()), limit, dry_run).await
}

pub(in crate::ai) async fn run_distill_source_command(app: &App, input: &DistillInput, limit: usize, dry_run: bool)
    -> Result<DistillReport, String>
{
    if !(1..=100).contains(&limit) { return Err("Distill limit must be 1..=100".into()); }
    let scope = current_project_scope();
    if scope.is_empty() { return Err("Cannot determine the ingestion project scope".into()); }
    let (session_id, messages) = input.read_messages()?;
    let segments = source_segments(&messages)?;
    let source_digest = digest(serde_json::to_vec(&segments).map_err(|e| e.to_string())?);
    let store = MemoryStore::from_env_or_config();
    let existing = store.active_distilled_entries(&scope)?;
    let model = crate::ai::models::initial_model(&app.cli);
    let (conclusions, rejected) = semantic_conclusions(&segments, &existing, limit, |payload| {
        let model = model.clone();
        async move {
            let messages = vec![json!({"role":"system","content":MODEL_RULES}), json!({"role":"user","content":payload.to_string()})];
            crate::ai::request::do_request_text_streaming(app, &model, &messages).await
                .map_err(|error|format!("Semantic distillation failed before commit: {error}"))
        }
    }).await?;
    let mut report = DistillReport { session_id: session_id.clone(), extracted: conclusions.len(), saved:0,
        updated:0, duplicates:0, rejected, chunks:source_chunks(&segments).len(), dry_run };
    let source = format!("session-distill:{session_id}");
    let mut pending = Vec::new();
    for conclusion in conclusions {
        let evidence = resolve_evidence(&conclusion, &segments, &source_digest)?;
        let args = json!({"content":conclusion.note,"category":conclusion.category,"tags":[DISTILL_TAG],"source":source});
        let prepared = prepare_memory_save_entry(&args, &conclusion.category, &[DISTILL_TAG], &source,
            MemoryOwnerScope::Global, "session_distill_rejected")?;
        if prepared.downgraded { report.rejected += 1; continue; }
        let mut entry = prepared.entry;
        let old = existing.iter().find(|entry|active_distilled_metadata(entry)
            .is_some_and(|metadata|metadata.topic_key == conclusion.topic_key));
        let expected = old.and_then(|old| {
            let metadata = active_distilled_metadata(old)?;
            if old.note == entry.note || conclusion.replaces.as_ref() == old.id.as_ref() {
                Some((old.id.clone()?, metadata.revision))
            } else { None }
        });
        if old.is_some() && expected.is_none() { report.rejected += 1; continue; }
        if let Some((id,_)) = &expected { entry.id = Some(id.clone()); }
        entry.distilled = Some(DistilledMetadata { schema:DISTILLED_SCHEMA, scope:scope.clone(), revision:1,
            topic_key:conclusion.topic_key, verified:true, content_digest:entry_content_digest(&entry),
            evidence, source_digests:vec![source_digest.clone()], previous_revisions:Vec::new() });
        pending.push((entry, expected));
    }
    for outcome in store.upsert_distilled_batch(pending, dry_run)? {
        report.saved += usize::from(outcome.inserted);
        report.updated += usize::from(outcome.updated);
        report.duplicates += usize::from(outcome.duplicate);
    }
    Ok(report)
}

fn text_message(role: &str, text: impl Into<String>) -> Message {
    Message { role: role.into(), content: Value::String(text.into()), tool_calls: None,
        tool_call_id: None, reasoning_content: None }
}

pub(crate) fn format_report(report: &DistillReport, path: &Path) -> String {
    format!("Session distill {}: {}\n  session: {}\n  fully scanned batches: {}\n  verified conclusions: {}\n  {}: {}\n  updated: {}\n  duplicates: {}\n  rejected: {}\nVerified current-project conclusions are eligible for bounded automatic task recall; uncertain or unrelated results are not loaded.",
        if report.dry_run {"preview"} else {"complete"}, path.display(), report.session_id, report.chunks,
        report.extracted, if report.dry_run {"would save"} else {"saved"}, report.saved,
        report.updated, report.duplicates, report.rejected)
}

struct StagedArchive(PathBuf);
impl Drop for StagedArchive { fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); } }

fn read_archive_messages(zip_path: &Path) -> Result<(String, Vec<Message>), String> {
    let file = File::open(zip_path).map_err(|e|format!("Failed to open archive: {e}"))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e|format!("Invalid archive: {e}"))?;
    let mut sqlite_count = 0;
    for index in 0..archive.len() {
        let entry = archive.by_index(index).map_err(|e|e.to_string())?;
        if entry.enclosed_name().is_none() { return Err("Unsafe archive entry rejected".into()); }
        if entry.name() == "session.sqlite" {
            if entry.is_dir() || entry.size() > 512*1024*1024 { return Err("Invalid or oversized session.sqlite".into()); }
            sqlite_count += 1;
        }
    }
    if sqlite_count != 1 { return Err("Archive must contain exactly one session.sqlite".into()); }
    let session_id = archive.by_name("manifest.json").ok().and_then(|entry| {
        let mut text = String::new();
        entry.take(65537).read_to_string(&mut text).ok()?;
        if text.len() > 65536 { return None; }
        serde_json::from_str::<Value>(&text).ok()?.get("session_id")?.as_str()
            .filter(|id|!id.trim().is_empty()).map(str::to_string)
    }).unwrap_or_else(||zip_path.file_stem().unwrap_or_default().to_string_lossy().into_owned());
    let path = std::env::temp_dir().join(format!("session-distill-{}.sqlite",uuid::Uuid::new_v4()));
    let mut output = OpenOptions::new().write(true).create_new(true).open(&path).map_err(|e|e.to_string())?;
    let staged = StagedArchive(path);
    let mut entry = archive.by_name("session.sqlite").map_err(|e|e.to_string())?;
    std::io::copy(&mut entry, &mut output).map_err(|e|e.to_string())?;
    drop(output);
    let messages = read_all_messages_sqlite(&staged.0).map_err(|e|format!("Failed to read archived messages: {e}"))?;
    Ok((session_id, messages))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn conclusion() -> Value {
        json!({"topic_key":"storage-choice","category":"decision_log",
            "note":"The project uses SQLite as its canonical store, not Redis.",
            "evidence":[{"message_id":"m2p1","quote":"Use SQLite as the canonical store instead of Redis."}],"replaces":null})
    }
    #[test]
    fn session_distill_full_coverage_keeps_long_message_tail() {
        let text = format!("{}TAIL-CORRECTION","a".repeat(SEGMENT_CHARS*3));
        let segments = source_segments(&[text_message("user",text.clone())]).unwrap();
        assert_eq!(segments.iter().map(|s|s.text.as_str()).collect::<String>(),text);
        assert_eq!(segments.len(),4);
        assert!(segments.last().unwrap().text.ends_with("TAIL-CORRECTION"));
    }
    #[test]
    fn session_distill_rejects_assistant_and_fabricated_evidence() {
        let item: Conclusion = serde_json::from_value(conclusion()).unwrap();
        let segments = vec![SourceSegment{id:"m2p1".into(),role:"assistant".into(),text:item.evidence[0].quote.clone()}];
        assert!(resolve_evidence(&item,&segments,"source").is_err());
        let segments = vec![SourceSegment{id:"m2p1".into(),role:"user".into(),text:"Unrelated text".into()}];
        assert!(resolve_evidence(&item,&segments,"source").is_err());
    }
    #[tokio::test]
    async fn session_distill_late_correction_vetoes_stale_conclusion() {
        let segments = source_segments(&[text_message("user","x".repeat(CHUNK_CHARS)),
            text_message("user","Use SQLite as the canonical store instead of Redis."),
            text_message("user","Correction: the final decision is PostgreSQL, not SQLite.")]).unwrap();
        let total = source_chunks(&segments).len();
        let mut extract_calls = 0;
        let mut verify_calls = 0;
        let (accepted,rejected) = semantic_conclusions(&segments,&[],1,|payload| {
            let response = match payload["phase"].as_str().unwrap() {
                "extract" => { extract_calls += 1; json!({"conclusions":[conclusion()]}) },
                "merge" => json!({"conclusions":[conclusion()]}),
                "verify" => { verify_calls += 1; json!({"verdicts":[{"id":0,
                    "status":if verify_calls == total {"contradiction"} else {"supported"}}]}) },
                _ => unreachable!(),
            };
            std::future::ready(Ok(response.to_string()))
        }).await.unwrap();
        assert_eq!(extract_calls,total);
        assert_eq!(verify_calls,total);
        assert!(accepted.is_empty());
        assert_eq!(rejected,1);
    }
    #[tokio::test]
    async fn session_distill_incomplete_verifier_fails_closed() {
        let segments = source_segments(&[text_message("user","Other discussion"),
            text_message("user","Use SQLite as the canonical store instead of Redis.")]).unwrap();
        let result = semantic_conclusions(&segments,&[],1,|payload| {
            std::future::ready(Ok(if payload["phase"] == "verify" { json!({"verdicts":[]}) }
                else { json!({"conclusions":[conclusion()]}) }.to_string()))
        }).await;
        assert!(result.unwrap_err().contains("Incomplete verification"));
    }
}
