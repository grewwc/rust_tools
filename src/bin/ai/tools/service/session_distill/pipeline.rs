//! A lossless proposal ledger feeds bounded final selection, not recursive top-k
//! summaries. Unique proposals need not shrink: every loop advances a source,
//! proposal, or catalog cursor. Only accepted final conclusions consume `limit`.

use std::collections::{BTreeSet, VecDeque};
use serde::de::DeserializeOwned;
use super::*;
use super::budget::RequestBudget;

#[derive(Clone, Copy, Serialize)]
struct SourceSlice<'a> {
    id: &'a str,
    role: &'a str,
    text: &'a str,
}

fn bisect<'a>(page: &[SourceSlice<'a>]) -> Result<[Vec<SourceSlice<'a>>; 2], String> {
    if page.len() > 1 {
        let middle = page.len() / 2;
        return Ok([page[..middle].to_vec(), page[middle..].to_vec()]);
    }
    let Some(source) = page.first() else { return Err("Empty source page".into()); };
    let middle = source.text.char_indices().map(|(i, _)| i)
        .find(|&i| i > 0 && i >= source.text.len() / 2)
        .ok_or("A single source character cannot be processed completely; nothing saved")?;
    Ok([
        vec![SourceSlice { text: &source.text[..middle], ..*source }],
        vec![SourceSlice { text: &source.text[middle..], ..*source }],
    ])
}

fn source_pages<'a>(segments: &'a [SourceSegment], budget: &RequestBudget,
    payload: impl Fn(&[SourceSlice<'a>]) -> Value) -> Result<Vec<Vec<SourceSlice<'a>>>, String>
{
    let mut pending: Vec<_> = segments.iter().rev().map(|s| SourceSlice {
        id: &s.id, role: &s.role, text: &s.text,
    }).collect();
    let mut slices = Vec::new();
    while let Some(source) = pending.pop() {
        if budget.fits(&payload(&[source])) {
            slices.push(source);
        } else {
            let [left, right] = bisect(&[source])?;
            pending.extend(right.into_iter().chain(left));
        }
    }
    let pages = budget.pages(&slices, payload)?;
    Ok(pages.into_iter().map(|range| slices[range].to_vec()).collect())
}

struct Pipeline<'a, F> { budget: &'a RequestBudget, ask: F }

impl<F, Fut> Pipeline<'_, F>
where F: FnMut(Value) -> Fut, Fut: Future<Output = Result<String, String>>,
{
    async fn request<T: DeserializeOwned>(&mut self, payload: Value) -> Result<T, String> {
        let phase = payload["phase"].as_str().unwrap_or("unknown").to_owned();
        self.budget.check(&payload).map_err(|e| format!("{phase}: {e}"))?;
        let raw = (self.ask)(payload).await.map_err(|e| format!("{phase}: {e}"))?;
        parse_json(&raw).map_err(|e| format!("{phase}: {e}"))
    }

    async fn extract(&mut self, segments: &[SourceSegment]) -> Result<Vec<Conclusion>, String> {
        let payload = |source: &[SourceSlice<'_>]| json!({"phase":"extract",
            "instruction":include_str!("../prompts/session_distill_extract.md"),
            "schema":EXTRACTION_SCHEMA, "source":source});
        let mut pending: VecDeque<_> = source_pages(segments, self.budget, payload)?.into();
        let mut ledger = Vec::new();
        while let Some(page) = pending.pop_front() {
            let extracted: Extraction = self.request(payload(&page)).await?;
            if !extracted.complete {
                // Discard the incomplete attempt, not source coverage. Both halves
                // must succeed, and single-character failure is terminal.
                let [left, right] = bisect(&page)?;
                pending.push_front(right);
                pending.push_front(left);
                continue;
            }
            ledger.extend(extracted.conclusions);
        }
        Ok(ledger)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ranking { scores: Vec<Score> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Score { id: usize, utility: u8 }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Grouping { decisions: Vec<Related> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Related { id: usize, related: bool }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogReview { decisions: Vec<CatalogDecision> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogDecision { id: String, status: String }

fn require_coverage<T: Ord>(expected: impl IntoIterator<Item = T>, actual: impl IntoIterator<Item = T>)
    -> Result<(), String>
{
    let mut remaining: BTreeSet<_> = expected.into_iter().collect();
    for id in actual {
        if !remaining.remove(&id) {
            return Err("Unknown or duplicate decision ID; nothing saved".into());
        }
    }
    if !remaining.is_empty() { return Err("Incomplete decision coverage; nothing saved".into()); }
    Ok(())
}

fn card(id: usize, conclusion: &Conclusion) -> Value {
    json!({"id":id,"topic_key":conclusion.topic_key,"category":conclusion.category,"note":conclusion.note})
}

impl<F, Fut> Pipeline<'_, F>
where F: FnMut(Value) -> Fut, Fut: Future<Output = Result<String, String>>,
{
    async fn rank(&mut self, cards: &[Value]) -> Result<Vec<usize>, String> {
        let payload = |page: &[Value]| json!({"phase":"rank",
            "instruction":include_str!("../prompts/session_distill_rank.md"),
            "schema":{"scores":[{"id":0,"utility":80}]}, "proposals":page});
        let mut scores = vec![0; cards.len()];
        for range in self.budget.pages(cards, payload)? {
            let ranked: Ranking = self.request(payload(&cards[range.clone()])).await?;
            require_coverage(range, ranked.scores.iter().map(|s| s.id))?;
            for score in ranked.scores {
                if score.utility > 100 { return Err("Utility must be in 0..=100; nothing saved".into()); }
                scores[score.id] = score.utility;
            }
        }
        let mut ids: Vec<_> = (0..cards.len()).filter(|&id| scores[id] > 0).collect();
        ids.sort_by_key(|&id| (std::cmp::Reverse(scores[id]), id));
        Ok(ids)
    }

    async fn group(&mut self, seed: usize, cards: &[Value], consumed: &BTreeSet<usize>)
        -> Result<Vec<usize>, String>
    {
        let payload = |page: &[Value]| json!({"phase":"group", "seed":cards[seed],
            "instruction":include_str!("../prompts/session_distill_group.md"),
            "schema":{"decisions":[{"id":0,"related":true}]}, "proposals":page});
        let mut members = vec![seed];
        for range in self.budget.pages(cards, payload)? {
            let page: Vec<_> = range.filter(|id| *id != seed && !consumed.contains(id))
                .map(|id| cards[id].clone()).collect();
            if page.is_empty() { continue; }
            let grouped: Grouping = self.request(payload(&page)).await?;
            require_coverage(page.iter().map(|v| v["id"].as_u64().unwrap() as usize),
                grouped.decisions.iter().map(|d| d.id))?;
            members.extend(grouped.decisions.into_iter().filter(|d| d.related).map(|d| d.id));
        }
        members.sort_unstable();
        Ok(members)
    }

    async fn reconcile(&mut self, mut conclusion: Conclusion, catalog: &[Value])
        -> Result<Option<Conclusion>, String>
    {
        let payload = |page: &[Value]| json!({"phase":"catalog", "conclusion":conclusion,
            "instruction":include_str!("../prompts/session_distill_catalog.md"),
            "schema":{"decisions":[{"id":"existing-id","status":"unrelated|same|replace|conflict"}]},
            "existing":page});
        let mut matched: Option<(&Value, String)> = None;
        let mut conflict = false;
        for range in self.budget.pages(catalog, payload)? {
            let page = &catalog[range];
            let review: CatalogReview = self.request(payload(page)).await?;
            require_coverage(page.iter().map(|v| v["id"].as_str().unwrap()),
                review.decisions.iter().map(|d| d.id.as_str()))?;
            for decision in review.decisions {
                let entry = page.iter().find(|e| e["id"] == decision.id).unwrap();
                match decision.status.as_str() {
                    "unrelated" => {
                        // A semantic classifier cannot bypass a canonical topic
                        // collision merely by declaring that entry unrelated.
                        if entry["topic_key"] == conclusion.topic_key { conflict = true; }
                    }
                    "conflict" => conflict = true,
                    "same" | "replace" => {
                        if matched.is_some() { conflict = true; }
                        matched = Some((entry, decision.status));
                    }
                    _ => return Err("Unknown catalog verdict; nothing saved".into()),
                }
            }
        }
        if conflict { return Ok(None); }
        if let Some((entry, status)) = matched {
            conclusion.topic_key = entry["topic_key"].as_str().unwrap().to_owned();
            if status == "same" {
                conclusion.note = entry["note"].as_str().unwrap().to_owned();
            } else {
                conclusion.replaces = Some(entry["id"].as_str().unwrap().to_owned());
            }
        }
        Ok(Some(conclusion))
    }

    async fn verify(&mut self, conclusion: &Conclusion, segments: &[SourceSegment], catalog: &[Value])
        -> Result<bool, String>
    {
        // Catalog reconciliation already visited every current entry. Only the
        // matched old revision is relevant to this conclusion's source audit.
        let existing: Vec<_> = catalog.iter().filter(|e| e["topic_key"] == conclusion.topic_key).collect();
        let payload = |source: &[SourceSlice<'_>]| json!({"phase":"verify",
            "instruction":include_str!("../prompts/session_distill_verify.md"),
            "schema":{"verdicts":[{"id":0,"status":"supported|irrelevant|contradiction|uncertain"}]},
            "conclusions":[{"id":0,"conclusion":conclusion}], "existing":existing, "source":source});
        let mut supported = false;
        let mut vetoed = false;
        for page in source_pages(segments, self.budget, payload)? {
            let verification: Verification = self.request(payload(&page)).await?;
            require_coverage([0], verification.verdicts.iter().map(|v| v.id))?;
            match verification.verdicts[0].status.as_str() {
                "supported" => supported = true,
                "irrelevant" => {},
                "contradiction" | "uncertain" => vetoed = true,
                _ => return Err("Unknown verification verdict; nothing saved".into()),
            }
        }
        Ok(supported && !vetoed)
    }

    async fn merge(&mut self, ledger: &[Conclusion], members: &[usize]) -> Result<Option<Conclusion>, String> {
        let proposals: Vec<_> = members.iter().map(|&id| &ledger[id]).collect();
        let mut carry: Option<Conclusion> = None;
        let mut cursor = 0;
        while cursor < proposals.len() {
            let payload = |page: &[&Conclusion]| json!({"phase":"merge", "carry":carry,
                "instruction":include_str!("../prompts/session_distill_merge.md"),
                "schema":EXTRACTION_SCHEMA, "proposals":page});
            // Only probe the next bounded window, not the entire remaining ledger.
            // A non-shrinking carry cannot stall this cursor.
            let end = (cursor + 64).min(proposals.len());
            let range = self.budget.pages(&proposals[cursor..end], payload)?.remove(0);
            let response: Extraction = self.request(payload(&proposals[cursor..cursor + range.end])).await?;
            if !response.complete || response.conclusions.len() > 1 {
                return Err("Incomplete or multi-topic group merge; nothing saved".into());
            }
            carry = response.conclusions.into_iter().next();
            cursor += range.end;
        }
        Ok(carry)
    }
}

pub(super) async fn run<F, Fut>(segments: &[SourceSegment], existing: &[AgentMemoryEntry],
    limit: usize, budget: &RequestBudget, ask: F) -> Result<(Vec<Conclusion>, usize), String>
where F: FnMut(Value) -> Fut, Fut: Future<Output = Result<String, String>>,
{
    if limit == 0 { return Err("Distill limit must be greater than zero".into()); }
    let mut pipeline = Pipeline { budget, ask };
    let extracted = pipeline.extract(segments).await?;
    let extracted_count = extracted.len();
    let source_digest = digest(serde_json::to_vec(segments).map_err(|e| e.to_string())?);
    let ledger: Vec<_> = extracted.into_iter().filter(|item| item.replaces.is_none()
        && resolve_evidence(item, segments, &source_digest).is_ok()).collect();
    let mut rejected = extracted_count - ledger.len();
    let cards: Vec<_> = ledger.iter().enumerate().map(|(id, item)| card(id, item)).collect();
    // Every candidate receives a globally comparable score before final selection.
    // The original ledger, including low-score corrections, is never truncated.
    let ranked = pipeline.rank(&cards).await?;
    let catalog: Vec<_> = existing.iter().filter_map(|entry|
        active_distilled_metadata(entry).map(|metadata| (entry, metadata)))
        .map(|(entry, metadata)| {
            let id = entry.id.as_ref().ok_or("Current catalog entry has no ID; nothing saved")?;
            Ok(json!({"id":id,"topic_key":metadata.topic_key,"revision":metadata.revision,"note":entry.note}))
        }).collect::<Result<_, String>>()?;
    let mut consumed = BTreeSet::new();
    let mut topic_keys = BTreeSet::new();
    let mut accepted = Vec::new();
    for seed in ranked {
        if accepted.len() == limit { break; }
        if consumed.contains(&seed) { continue; }
        let members = pipeline.group(seed, &cards, &consumed).await?;
        consumed.extend(members.iter().copied());
        let Some(conclusion) = pipeline.merge(&ledger, &members).await? else {
            rejected += 1;
            continue;
        };
        let grounded = conclusion.evidence.iter().all(|reference| members.iter().any(|&id|
            ledger[id].evidence.iter().any(|original| original.message_id == reference.message_id
                && original.quote.contains(&reference.quote))));
        if conclusion.replaces.is_some() || !grounded
            || resolve_evidence(&conclusion, segments, &source_digest).is_err()
        {
            rejected += 1;
            continue;
        }
        let Some(conclusion) = pipeline.reconcile(conclusion, &catalog).await? else {
            rejected += 1;
            continue;
        };
        if !pipeline.verify(&conclusion, segments, &catalog).await? {
            rejected += 1;
            continue;
        }
        if !topic_keys.insert(conclusion.topic_key.clone()) {
            return Err("Final selection returned conflicting duplicate topics; nothing saved".into());
        }
        accepted.push(conclusion);
    }
    Ok((accepted, rejected))
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;