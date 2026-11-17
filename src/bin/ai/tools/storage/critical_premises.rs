use rust_tools::commonw::{FastMap, FastSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(in crate::ai) const MAX_PREMISES: usize = 8;
pub(in crate::ai) const MAX_ID_BYTES: usize = 64;
const MAX_STRING_BYTES: usize = 2_048;
const MAX_OBJECT_BYTES: usize = 1_024;
const MAX_CALL_ID_BYTES: usize = 256;
const MAX_REFS: usize = 16;
const MAX_UPDATE_STRING_BYTES: usize = 8_192;
const MAX_STATE_STRING_BYTES: usize = 65_536;
const MAX_ORIGINAL_CONTENT_BYTES: usize = 4_096;
const MAX_RECEIPT_BYTES: usize = 32_768;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::ai) enum Impact {
    MainAnswer,
    Choice,
    Mutation,
    ExternalAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::ai) enum Status {
    Pending,
    Supported,
    Refuted,
    Limited,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::ai) enum Coverage {
    Full,
    Partial,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::ai) struct EvidenceRef {
    pub(in crate::ai) tool_call_id: String,
    /// Exact runtime catalogue object token, not a free-form provenance claim.
    pub(in crate::ai) object: String,
    pub(in crate::ai) coverage: Coverage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::ai) struct Update {
    pub(in crate::ai) id: String,
    pub(in crate::ai) proposition: String,
    pub(in crate::ai) impact: Impact,
    pub(in crate::ai) status: Status,
    #[serde(default)]
    pub(in crate::ai) dependencies: Vec<String>,
    #[serde(default)]
    pub(in crate::ai) evidence: Vec<EvidenceRef>,
    #[serde(default)]
    pub(in crate::ai) gap: String,
    #[serde(default)]
    pub(in crate::ai) conflict: String,
    #[serde(default)]
    pub(in crate::ai) request_review: bool,
}

/// Validated syntax only. This type never certifies the proposition's truth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub(in crate::ai) struct ValidatedUpdate(Update);

impl ValidatedUpdate {
    pub(in crate::ai) fn from_value(value: &Value) -> Result<Self, String> {
        argument_budget(value)?;
        Self::new(Update::deserialize(value).map_err(|e| e.to_string())?)
    }

    pub(in crate::ai) fn new(update: Update) -> Result<Self, String> {
        validate_id(&update.id)?;
        bounded_text("proposition", &update.proposition, MAX_STRING_BYTES, true)?;
        bounded_text("gap", &update.gap, MAX_STRING_BYTES, false)?;
        bounded_text("conflict", &update.conflict, MAX_STRING_BYTES, false)?;
        if update.dependencies.len() >= MAX_PREMISES || update.evidence.len() > MAX_REFS {
            return Err("At most 7 dependencies and 16 evidence references are allowed.".into());
        }
        let mut dependencies = FastSet::default();
        for dependency in &update.dependencies {
            validate_id(dependency)?;
            if dependency == &update.id || !dependencies.insert(dependency) {
                return Err("Self or duplicate dependencies are not allowed.".into());
            }
        }
        let mut references = FastSet::default();
        for reference in &update.evidence {
            bounded_text(
                "tool_call_id",
                &reference.tool_call_id,
                MAX_CALL_ID_BYTES,
                true,
            )?;
            bounded_text("object", &reference.object, MAX_OBJECT_BYTES, true)?;
            if !references.insert((&reference.tool_call_id, &reference.object)) {
                return Err("Duplicate evidence references are not allowed.".into());
            }
        }
        if update.status == Status::Supported && update.evidence.is_empty() {
            return Err("Supported requires nonempty runtime-bindable evidence references.".into());
        }
        if update.status == Status::Supported
            && (!update.gap.trim().is_empty() || !update.conflict.trim().is_empty())
        {
            return Err(
                "Supported cannot retain an open gap/conflict; use pending or limited.".into(),
            );
        }
        if update.status == Status::Limited && update.gap.trim().is_empty() {
            return Err("Limited requires an explicit remaining gap or scope limitation.".into());
        }
        if update_string_bytes(&update) > MAX_UPDATE_STRING_BYTES {
            return Err("Update strings exceed the 8192-byte total limit.".into());
        }
        Ok(Self(update))
    }

    pub(in crate::ai) fn update(&self) -> &Update {
        &self.0
    }
}

// Reject oversized/unexpectedly nested input before deserialization copies any
// strings. The wire budget includes keys; the tighter update budget excludes them.
fn argument_budget(value: &Value) -> Result<(), String> {
    let mut stack = vec![(value, 0)];
    let mut nodes = 0;
    let mut bytes = 0;
    while let Some((value, depth)) = stack.pop() {
        nodes += 1;
        if nodes > 256 || depth > 8 {
            return Err("Premise arguments exceed the bounded shape/depth limit.".into());
        }
        match value {
            Value::String(text) => bytes += text.len(),
            Value::Array(items) => {
                if items.len() > 256 {
                    return Err("Premise argument array is too large.".into());
                }
                stack.extend(items.iter().map(|item| (item, depth + 1)));
            }
            Value::Object(fields) => {
                if fields.len() > 256 {
                    return Err("Premise argument object is too large.".into());
                }
                for (key, item) in fields {
                    bytes += key.len();
                    stack.push((item, depth + 1));
                }
            }
            _ => {}
        }
        if bytes > MAX_UPDATE_STRING_BYTES + 4_096 {
            return Err("Premise arguments exceed the bounded wire-string limit.".into());
        }
    }
    Ok(())
}

fn bounded_text(label: &str, text: &str, max: usize, nonempty: bool) -> Result<(), String> {
    if text.len() > max || (nonempty && text.trim().is_empty()) || text.contains('\0') {
        return Err(format!(
            "{label} must be {}at most {max} bytes without NUL.",
            if nonempty { "nonempty and " } else { "" }
        ));
    }
    Ok(())
}

fn validate_id(id: &str) -> Result<(), String> {
    if id.is_empty()
        || id.len() > MAX_ID_BYTES
        || !id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
    {
        return Err("id must contain 1..=64 ASCII letters, digits, '_', '.' or '-'.".into());
    }
    Ok(())
}

fn update_string_bytes(update: &Update) -> usize {
    update.id.len()
        + update.proposition.len()
        + update.gap.len()
        + update.conflict.len()
        + update.dependencies.iter().map(String::len).sum::<usize>()
        + update
            .evidence
            .iter()
            .map(|r| r.tool_call_id.len() + r.object.len())
            .sum::<usize>()
}

/// Driver-established object and coverage. Unknown provenance is an empty list
/// or Partial; neither model labels nor arbitrary command parsing establish it.
/// Exact tokens may describe file paths/ranges or command subjects only when the
/// driver has established that scope. This layer deliberately does no guessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) struct ObservedObject {
    pub(in crate::ai) object: String,
    pub(in crate::ai) coverage: Coverage,
}

/// One actual canonical tool result from the current turn, supplied by the driver.
/// `success == None` means the outcome is unknown, not successful. Keep only a
/// bounded original excerpt here; `full_result_known` describes the actual result,
/// not whether this excerpt fits. No original content is copied into receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) struct EvidenceRecord {
    pub(in crate::ai) tool_call_id: String,
    pub(in crate::ai) tool_name: String,
    pub(in crate::ai) canonical_index: usize,
    pub(in crate::ai) success: Option<bool>,
    pub(in crate::ai) full_result_known: bool,
    pub(in crate::ai) is_verification: bool,
    pub(in crate::ai) observed_objects: Vec<ObservedObject>,
    pub(in crate::ai) original_content: String,
}

/// Borrow a current-turn catalogue with an exclusive canonical-position cutoff.
/// The driver must not supply session-global records or invented outcome/scope
/// metadata. Records at/after `before` cannot support a preceding update.
#[derive(Debug, Clone, Copy)]
pub(in crate::ai) struct EvidenceCatalog<'a> {
    pub(in crate::ai) records: &'a [EvidenceRecord],
    pub(in crate::ai) before: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::ai) struct BoundEvidenceRef {
    pub(in crate::ai) reference: EvidenceRef,
    pub(in crate::ai) tool_name: String,
    pub(in crate::ai) success: Option<bool>,
    pub(in crate::ai) full_result_known: bool,
    pub(in crate::ai) observed_coverage: Option<Coverage>,
    pub(in crate::ai) is_verification: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ReceiptKind {
    #[serde(rename = "critical_premises.runtime_bound.v1")]
    RuntimeBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum SemanticOrigin {
    #[serde(rename = "model_derived")]
    ModelDerived,
}

/// Runtime binding verifies observation identity/outcome/completeness, not the
/// model's semantic inference. Only the driver may append this receipt as the
/// successful tool result, after `apply` succeeds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::ai) struct Receipt {
    kind: ReceiptKind,
    semantic_origin: SemanticOrigin,
    pub(in crate::ai) sequence: u64,
    pub(in crate::ai) update: Update,
    pub(in crate::ai) invalidated_ids: Vec<String>,
    pub(in crate::ai) evidence: Vec<BoundEvidenceRef>,
}

impl Receipt {
    pub(in crate::ai) fn to_tool_result(&self) -> Result<String, String> {
        let result = serde_json::to_string(self).map_err(|e| e.to_string())?;
        if result.len() > MAX_RECEIPT_BYTES {
            return Err("Runtime receipt exceeds the bounded result size.".into());
        }
        Ok(result)
    }
}

/// Driver-selected actual current-turn tool-result envelope. A marker in user
/// text, a model message, a failed result, or provisional output is not authority.
#[derive(Debug, Clone)]
pub(in crate::ai) struct CanonicalReceipt {
    pub(in crate::ai) tool_call_id: String,
    pub(in crate::ai) tool_name: String,
    pub(in crate::ai) canonical_index: usize,
    pub(in crate::ai) success: bool,
    pub(in crate::ai) content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    update: Update,
    evidence: Vec<BoundEvidenceRef>,
}

/// Owned by one driver turn. No global state, session reads or persistence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(in crate::ai) struct PremiseState {
    entries: FastMap<String, Entry>,
    sequence: u64,
}

impl PremiseState {
    pub(in crate::ai) fn has_records(&self) -> bool {
        !self.entries.is_empty()
    }

    pub(in crate::ai) fn pending(&self) -> Vec<&Update> {
        self.selected(|entry| entry.update.status == Status::Pending)
    }

    /// Review cannot create evidence or promote a status. The driver owns its
    /// one-shot consumed flag. Only open material gaps with an actual verification
    /// attempt are eligible, including failed/partial/unknown-outcome attempts.
    pub(in crate::ai) fn review_candidates(&self) -> Vec<&Update> {
        self.selected(|entry| {
            let update = &entry.update;
            update.status == Status::Pending
                && update.request_review
                && (!update.gap.trim().is_empty() || !update.conflict.trim().is_empty())
                && entry.evidence.iter().any(|r| r.is_verification)
        })
    }

    fn selected(&self, predicate: impl Fn(&Entry) -> bool) -> Vec<&Update> {
        let mut result: Vec<_> = self
            .entries
            .values()
            .filter(|e| predicate(e))
            .map(|e| &e.update)
            .collect();
        result.sort_by(|a, b| a.id.cmp(&b.id));
        result
    }

    /// Atomic update: all binding, graph, dependency and size checks finish on a
    /// candidate before commit. Supported is a model-derived disposition whose
    /// references must each bind to complete successful compatible observations.
    pub(in crate::ai) fn apply(
        &mut self,
        validated: ValidatedUpdate,
        catalog: &EvidenceCatalog<'_>,
    ) -> Result<Receipt, String> {
        let update = validated.0;
        if !self.entries.contains_key(&update.id) && self.entries.len() >= MAX_PREMISES {
            return Err("At most 8 premise records are allowed in a turn.".into());
        }
        for dependency in &update.dependencies {
            if !self.entries.contains_key(dependency) {
                return Err(format!("Unknown dependency: {dependency}"));
            }
        }
        let evidence = bind_evidence(&update, catalog)?;
        let old = self.entries.get(&update.id);
        let changed = old.is_some_and(|e| substantive_change(&e.update, &update));
        if let Some(old) = old {
            let identity_changed = old.update.proposition != update.proposition
                || old.update.impact != update.impact
                || (old.update.status == Status::Supported
                    && object_scope(&old.update) != object_scope(&update));
            if identity_changed && update.status == Status::Supported {
                return Err("Changed proposition/impact/object scope must first be pending or limited, then reevaluated.".into());
            }
        }
        let mut candidate = self.clone();
        candidate.entries.insert(
            update.id.clone(),
            Entry {
                update: update.clone(),
                evidence: evidence.clone(),
            },
        );
        validate_graph(&candidate.entries)?;
        let invalidated_ids = if changed {
            candidate.invalidate_dependents(&update.id)
        } else {
            Vec::new()
        };
        if update.status == Status::Supported {
            for dependency in &update.dependencies {
                if candidate.entries[dependency].update.status != Status::Supported {
                    return Err(format!("Dependency {dependency} is not supported."));
                }
            }
        }
        for entry in candidate.entries.values() {
            ValidatedUpdate::new(entry.update.clone())?;
        }
        let total: usize = candidate
            .entries
            .values()
            .map(|e| {
                update_string_bytes(&e.update)
                    + e.evidence
                        .iter()
                        .map(|r| {
                            r.reference.tool_call_id.len()
                                + r.reference.object.len()
                                + r.tool_name.len()
                        })
                        .sum::<usize>()
            })
            .sum();
        if total > MAX_STATE_STRING_BYTES {
            return Err("Premise state strings exceed the 65536-byte total limit.".into());
        }
        candidate.sequence = candidate
            .sequence
            .checked_add(1)
            .ok_or("Premise sequence overflow.")?;
        let receipt = Receipt {
            kind: ReceiptKind::RuntimeBound,
            semantic_origin: SemanticOrigin::ModelDerived,
            sequence: candidate.sequence,
            update,
            invalidated_ids,
            evidence,
        };
        receipt.to_tool_result()?;
        *self = candidate;
        Ok(receipt)
    }

    fn invalidate_dependents(&mut self, id: &str) -> Vec<String> {
        let mut affected = FastSet::default();
        affected.insert(id.to_string());
        // Each pass either adds an unseen record or stops; at most eight records.
        loop {
            let next: Vec<_> = self
                .entries
                .iter()
                .filter(|(key, entry)| {
                    !affected.contains(*key)
                        && entry
                            .update
                            .dependencies
                            .iter()
                            .any(|d| affected.contains(d))
                })
                .map(|(key, _)| key.clone())
                .collect();
            if next.is_empty() {
                break;
            }
            affected.extend(next);
        }
        affected.remove(id);
        let mut ids: Vec<_> = affected.into_iter().collect();
        ids.sort();
        for id in &ids {
            let entry = self.entries.get_mut(id).expect("existing dependent");
            entry.update.status = Status::Pending;
            if entry.update.gap.trim().is_empty() {
                entry.update.gap =
                    "A dependency changed; reevaluate this premise before relying on it.".into();
            }
        }
        ids
    }

    /// Replay actual successful receipts in canonical order with each receipt's
    /// own evidence cutoff. Rebind and compare the whole receipt, rather than
    /// trusting a serialized supported label. Invalid bound receipts fail closed;
    /// non-receipts and unsuccessful results do not introduce state.
    pub(in crate::ai) fn rebuild(
        receipts: &[CanonicalReceipt],
        evidence: &[EvidenceRecord],
    ) -> Result<Self, String> {
        let mut state = Self::default();
        let mut previous = None;
        let mut call_ids = FastSet::default();
        for canonical in receipts {
            if canonical.tool_name != "critical_premises" || !canonical.success {
                continue;
            }
            bounded_text(
                "receipt tool_call_id",
                &canonical.tool_call_id,
                MAX_CALL_ID_BYTES,
                true,
            )?;
            if previous.is_some_and(|index| canonical.canonical_index <= index)
                || !call_ids.insert(&canonical.tool_call_id)
            {
                return Err(
                    "Canonical receipts must have unique IDs and increasing positions.".into(),
                );
            }
            previous = Some(canonical.canonical_index);
            if canonical.content.len() > MAX_RECEIPT_BYTES {
                return Err("Canonical premise result exceeds the bounded size.".into());
            }
            let Ok(value) = serde_json::from_str::<Value>(&canonical.content) else {
                continue;
            };
            if value["kind"] != "critical_premises.runtime_bound.v1" {
                continue;
            }
            let receipt: Receipt = serde_json::from_value(value).map_err(|e| e.to_string())?;
            let rebuilt = state.apply(
                ValidatedUpdate::new(receipt.update.clone())?,
                &EvidenceCatalog {
                    records: evidence,
                    before: canonical.canonical_index,
                },
            )?;
            if rebuilt != receipt {
                return Err(
                    "Canonical receipt disagrees with replayed runtime binding/state.".into(),
                );
            }
        }
        Ok(state)
    }
}

fn substantive_change(old: &Update, new: &Update) -> bool {
    old.proposition != new.proposition
        || old.impact != new.impact
        || old.status != new.status
        || old.dependencies != new.dependencies
        || old.evidence != new.evidence
        || old.gap != new.gap
        || old.conflict != new.conflict
}

fn object_scope(update: &Update) -> Vec<&str> {
    let mut objects: Vec<_> = update.evidence.iter().map(|r| r.object.as_str()).collect();
    objects.sort_unstable();
    objects.dedup();
    objects
}

fn bind_evidence(
    update: &Update,
    catalog: &EvidenceCatalog<'_>,
) -> Result<Vec<BoundEvidenceRef>, String> {
    update.evidence.iter().map(|reference| {
        let mut matching = catalog.records.iter().filter(|r| r.tool_call_id == reference.tool_call_id);
        let record = matching.next().ok_or_else(|| format!("Unknown evidence: {}", reference.tool_call_id))?;
        if matching.next().is_some() {
            return Err(format!("Ambiguous canonical evidence ID: {}", reference.tool_call_id));
        }
        if record.canonical_index >= catalog.before || record.tool_name == "critical_premises" {
            return Err("Future or premise/self-update results cannot be evidence.".into());
        }
        bounded_text("canonical tool_call_id", &record.tool_call_id, MAX_CALL_ID_BYTES, true)?;
        bounded_text("canonical tool_name", &record.tool_name, 128, true)?;
        if record.original_content.len() > MAX_ORIGINAL_CONTENT_BYTES {
            return Err("Canonical original-content excerpt exceeds 4096 bytes.".into());
        }
        if record.observed_objects.len() > MAX_REFS {
            return Err("Canonical record has too many observed objects.".into());
        }
        let mut tokens = FastSet::default();
        for object in &record.observed_objects {
            bounded_text("observed object", &object.object, MAX_OBJECT_BYTES, true)?;
            if !tokens.insert(&object.object) {
                return Err("Canonical record has ambiguous observed object tokens.".into());
            }
        }
        let observed_coverage = record.observed_objects.iter().find(|o| o.object == reference.object).map(|o| o.coverage);
        if update.status == Status::Supported && (record.success != Some(true)
            || !record.full_result_known || reference.coverage != Coverage::Full
            || observed_coverage != Some(Coverage::Full))
        {
            return Err(format!("Evidence {} is not a known successful complete observation of the requested object; use pending or explicit limited disposition.", reference.tool_call_id));
        }
        Ok(BoundEvidenceRef {
            reference: reference.clone(),
            tool_name: record.tool_name.clone(),
            success: record.success,
            full_result_known: record.full_result_known,
            observed_coverage,
            is_verification: record.is_verification,
        })
    }).collect()
}

fn validate_graph(entries: &FastMap<String, Entry>) -> Result<(), String> {
    fn visit<'a>(
        id: &'a str,
        entries: &'a FastMap<String, Entry>,
        path: &mut Vec<&'a str>,
        done: &mut FastSet<&'a str>,
    ) -> Result<(), String> {
        if path.contains(&id) {
            return Err("Premise dependencies contain a cycle.".into());
        }
        if done.contains(id) {
            return Ok(());
        }
        let entry = entries
            .get(id)
            .ok_or_else(|| format!("Unknown dependency: {id}"))?;
        path.push(id);
        for dependency in &entry.update.dependencies {
            visit(dependency, entries, path, done)?;
        }
        path.pop();
        done.insert(id);
        Ok(())
    }
    let mut done = FastSet::default();
    for id in entries.keys() {
        visit(id, entries, &mut Vec::new(), &mut done)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBJECT: &str = "/repo/key.rs:1-20";

    fn reference(call_id: &str) -> EvidenceRef {
        EvidenceRef {
            tool_call_id: call_id.into(),
            object: OBJECT.into(),
            coverage: Coverage::Full,
        }
    }

    fn update(id: &str, status: Status) -> Update {
        Update {
            id: id.into(),
            proposition: format!("{id}: target property holds"),
            impact: Impact::MainAnswer,
            status,
            dependencies: Vec::new(),
            evidence: if status == Status::Supported {
                vec![reference("read-0")]
            } else {
                Vec::new()
            },
            gap: if status == Status::Limited {
                "Only a limited scope was observed.".into()
            } else {
                String::new()
            },
            conflict: String::new(),
            request_review: false,
        }
    }

    fn observation(call_id: &str, canonical_index: usize) -> EvidenceRecord {
        EvidenceRecord {
            tool_call_id: call_id.into(),
            tool_name: "read_file".into(),
            canonical_index,
            success: Some(true),
            full_result_known: true,
            is_verification: true,
            observed_objects: vec![ObservedObject {
                object: OBJECT.into(),
                coverage: Coverage::Full,
            }],
            original_content: "The bounded original observation.".into(),
        }
    }

    fn apply(
        state: &mut PremiseState,
        update: Update,
        records: &[EvidenceRecord],
    ) -> Result<Receipt, String> {
        state.apply(
            ValidatedUpdate::new(update)?,
            &EvidenceCatalog {
                records,
                before: 20,
            },
        )
    }

    fn canonical(receipt: &Receipt, canonical_index: usize) -> CanonicalReceipt {
        CanonicalReceipt {
            tool_call_id: format!("premise-{}", receipt.sequence),
            tool_name: "critical_premises".into(),
            canonical_index,
            success: true,
            content: receipt.to_tool_result().unwrap(),
        }
    }

    #[test]
    fn critical_premises_argument_length_and_id_limits() {
        let mut valid = update(&"a".repeat(MAX_ID_BYTES), Status::Pending);
        valid.proposition = "x".repeat(MAX_STRING_BYTES);
        assert!(ValidatedUpdate::new(valid.clone()).is_ok());
        for id in [
            String::new(),
            "a".repeat(65),
            "bad id".into(),
            "前提".into(),
        ] {
            let mut invalid = valid.clone();
            invalid.id = id;
            assert!(ValidatedUpdate::new(invalid).is_err());
        }
        for field in ["proposition", "gap", "conflict"] {
            let mut wire = serde_json::to_value(&valid).unwrap();
            wire[field] = "x".repeat(MAX_STRING_BYTES + 1).into();
            assert!(ValidatedUpdate::from_value(&wire).is_err());
        }
        valid.proposition = "界".repeat(683);
        assert!(ValidatedUpdate::new(valid).is_err());
        let mut invalid = update("x", Status::Pending);
        invalid.evidence = (0..17).map(|i| reference(&format!("call-{i}"))).collect();
        assert!(ValidatedUpdate::new(invalid).is_err());
        let mut invalid = update("x", Status::Pending);
        invalid.dependencies = (0..8).map(|i| format!("d{i}")).collect();
        assert!(ValidatedUpdate::new(invalid).is_err());
        let mut invalid = update("x", Status::Pending);
        invalid.evidence = vec![reference("call"), reference("call")];
        assert!(ValidatedUpdate::new(invalid).is_err());
        let mut invalid = update("x", Status::Supported);
        invalid.evidence.clear();
        assert!(ValidatedUpdate::new(invalid).is_err());
        let mut invalid = update("x", Status::Supported);
        invalid.conflict = "An unresolved contradiction remains.".into();
        assert!(ValidatedUpdate::new(invalid).is_err());
        let mut invalid = update("x", Status::Pending);
        invalid.evidence = (0..10)
            .map(|i| EvidenceRef {
                tool_call_id: format!("call-{i}"),
                object: "o".repeat(900),
                coverage: Coverage::Partial,
            })
            .collect();
        assert!(ValidatedUpdate::new(invalid).is_err());
        let mut wire = serde_json::to_value(update("x", Status::Pending)).unwrap();
        wire["reviewer_says_supported"] = true.into();
        assert!(ValidatedUpdate::from_value(&wire).is_err());
        wire["reviewer_says_supported"] = "x".repeat(20_000).into();
        assert!(ValidatedUpdate::from_value(&wire).is_err());
    }

    #[test]
    fn critical_premises_record_and_state_total_limits_are_atomic() {
        let mut state = PremiseState::default();
        for i in 0..8 {
            apply(&mut state, update(&format!("p{i}"), Status::Pending), &[]).unwrap();
        }
        let before = state.clone();
        assert!(apply(&mut state, update("ninth", Status::Pending), &[]).is_err());
        assert_eq!(state, before);
        apply(&mut state, update("p0", Status::Limited), &[]).unwrap();

        let mut records = vec![
            observation(&"a".repeat(256), 0),
            observation(&"b".repeat(256), 1),
        ];
        for (i, record) in records.iter_mut().enumerate() {
            record.observed_objects[0].object = format!("{i}{}", "o".repeat(699));
        }
        let mut state = PremiseState::default();
        for i in 0..6 {
            let mut next = update(&format!("p{i}"), Status::Pending);
            next.proposition = "p".repeat(2_048);
            next.gap = "g".repeat(2_048);
            next.conflict = "c".repeat(2_048);
            next.evidence = records
                .iter()
                .map(|r| EvidenceRef {
                    tool_call_id: r.tool_call_id.clone(),
                    object: r.observed_objects[0].object.clone(),
                    coverage: Coverage::Full,
                })
                .collect();
            apply(&mut state, next, &records).unwrap();
        }
        let before = state.clone();
        let mut seventh = state.entries["p0"].update.clone();
        seventh.id = "p6".into();
        assert!(apply(&mut state, seventh, &records).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn critical_premises_unknown_self_future_evidence_rejected() {
        let mut state = PremiseState::default();
        for status in [
            Status::Pending,
            Status::Supported,
            Status::Refuted,
            Status::Limited,
        ] {
            let mut next = update("key", status);
            next.evidence = vec![reference("read-0")];
            assert!(apply(&mut state, next, &[]).is_err());
        }
        let mut record = observation("read-0", 0);
        record.tool_name = "critical_premises".into();
        assert!(apply(&mut state, update("key", Status::Supported), &[record]).is_err());
        let record = observation("read-0", 20);
        assert!(apply(&mut state, update("key", Status::Supported), &[record]).is_err());
        let records = vec![observation("read-0", 0), observation("read-0", 1)];
        assert!(apply(&mut state, update("key", Status::Supported), &records).is_err());
        assert!(!state.has_records());
    }

    #[test]
    fn critical_premises_failed_truncated_partial_unknown_not_supported() {
        for case in 0..6 {
            let mut record = observation("read-0", 0);
            let mut next = update("key", Status::Supported);
            match case {
                0 => record.success = Some(false),
                1 => record.success = None,
                2 => record.full_result_known = false,
                3 => record.observed_objects[0].coverage = Coverage::Partial,
                4 => next.evidence[0].coverage = Coverage::Partial,
                _ => record.observed_objects.clear(),
            }
            let mut state = PremiseState::default();
            assert!(apply(&mut state, next.clone(), &[record.clone()]).is_err());
            assert!(!state.has_records());
            next.status = Status::Pending;
            next.gap = "Verification did not cover the claim.".into();
            let receipt = apply(&mut state, next, &[record]).unwrap();
            assert_eq!(receipt.update.status, Status::Pending);
            assert_eq!(state.pending().len(), 1);
        }
    }

    #[test]
    fn critical_premises_object_mismatch_and_unknown_provenance() {
        let mut next = update("key", Status::Supported);
        next.evidence[0].object = "/repo/other.rs:1-20".into();
        let mut state = PremiseState::default();
        assert!(apply(&mut state, next.clone(), &[observation("read-0", 0)]).is_err());
        next.status = Status::Limited;
        next.gap = "The other target was not observed.".into();
        let receipt = apply(&mut state, next, &[observation("read-0", 0)]).unwrap();
        assert_eq!(receipt.evidence[0].observed_coverage, None);
        assert_eq!(receipt.update.status, Status::Limited);
        let mut record = observation("read-0", 0);
        record.tool_name = "execute_command".into();
        record.original_content = format!("The model labels the command as checking {OBJECT}");
        record.observed_objects.clear();
        let mut state = PremiseState::default();
        assert!(apply(&mut state, update("key", Status::Supported), &[record]).is_err());
    }

    #[test]
    fn critical_premises_supported_dependencies_and_transitive_invalidation() {
        let records = [observation("read-0", 0)];
        let mut state = PremiseState::default();
        apply(&mut state, update("a", Status::Supported), &records).unwrap();
        let mut b = update("b", Status::Supported);
        b.dependencies = vec!["a".into()];
        apply(&mut state, b.clone(), &records).unwrap();
        let mut c = update("c", Status::Supported);
        c.dependencies = vec!["b".into()];
        apply(&mut state, c.clone(), &records).unwrap();
        let receipt = apply(&mut state, update("a", Status::Refuted), &records).unwrap();
        assert_eq!(receipt.invalidated_ids, vec!["b", "c"]);
        assert_eq!(
            state
                .pending()
                .iter()
                .map(|p| p.id.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "c"]
        );
        let before = state.clone();
        assert!(apply(&mut state, b.clone(), &records).is_err());
        assert_eq!(state, before);
        apply(&mut state, update("a", Status::Supported), &records).unwrap();
        assert_eq!(state.entries["b"].update.status, Status::Pending);
        apply(&mut state, b, &records).unwrap();
        assert_eq!(state.entries["c"].update.status, Status::Pending);
        apply(&mut state, c, &records).unwrap();
        assert!(state.pending().is_empty());
    }

    #[test]
    fn critical_premises_changed_proposition_scope_cannot_retain_support() {
        let mut records = vec![observation("read-0", 0), observation("read-1", 1)];
        records[1].observed_objects[0].object = "/repo/other.rs:1-20".into();
        for case in 0..3 {
            let mut state = PremiseState::default();
            apply(&mut state, update("a", Status::Supported), &records).unwrap();
            let mut b = update("b", Status::Supported);
            b.dependencies = vec!["a".into()];
            apply(&mut state, b, &records).unwrap();
            let mut next = update("a", Status::Supported);
            match case {
                0 => next.proposition = "A different proposition".into(),
                1 => next.impact = Impact::Mutation,
                _ => {
                    next.evidence = vec![EvidenceRef {
                        tool_call_id: "read-1".into(),
                        object: "/repo/other.rs:1-20".into(),
                        coverage: Coverage::Full,
                    }]
                }
            }
            let before = state.clone();
            assert!(apply(&mut state, next.clone(), &records).is_err());
            assert_eq!(state, before);
            next.status = Status::Pending;
            let receipt = apply(&mut state, next.clone(), &records).unwrap();
            assert_eq!(receipt.invalidated_ids, vec!["b"]);
            next.status = Status::Supported;
            apply(&mut state, next, &records).unwrap();
            assert_eq!(state.entries["b"].update.status, Status::Pending);
        }
    }

    #[test]
    fn critical_premises_cycles_unknown_duplicate_dependencies_rollback() {
        let mut state = PremiseState::default();
        apply(&mut state, update("a", Status::Pending), &[]).unwrap();
        let mut b = update("b", Status::Pending);
        b.dependencies = vec!["a".into()];
        apply(&mut state, b, &[]).unwrap();
        let mut c = update("c", Status::Pending);
        c.dependencies = vec!["b".into()];
        apply(&mut state, c, &[]).unwrap();
        let before = state.clone();
        for dependencies in [
            vec!["c".into()],
            vec!["unknown".into()],
            vec!["a".into()],
            vec!["b".into(), "b".into()],
        ] {
            let mut a = update("a", Status::Pending);
            a.dependencies = dependencies;
            assert!(apply(&mut state, a, &[]).is_err());
            assert_eq!(state, before);
        }
    }

    #[test]
    fn critical_premises_explicit_limited_closure_and_review_eligibility() {
        let mut record = observation("read-0", 0);
        record.success = Some(false);
        record.full_result_known = false;
        record.observed_objects[0].coverage = Coverage::Partial;
        let mut next = update("key", Status::Limited);
        next.evidence = vec![reference("read-0")];
        next.request_review = true;
        let mut state = PremiseState::default();
        let receipt = apply(&mut state, next.clone(), &[record.clone()]).unwrap();
        assert_eq!(receipt.update.status, Status::Limited);
        assert!(state.has_records());
        assert!(state.pending().is_empty());
        assert!(state.review_candidates().is_empty());
        let before = state.clone();
        next.status = Status::Supported;
        assert!(apply(&mut state, next.clone(), &[record.clone()]).is_err());
        assert_eq!(state, before);
        next.status = Status::Pending;
        apply(&mut state, next.clone(), &[record.clone()]).unwrap();
        assert_eq!(state.review_candidates().len(), 1);
        next.request_review = false;
        apply(&mut state, next.clone(), &[record.clone()]).unwrap();
        assert!(state.review_candidates().is_empty());
        next.request_review = true;
        next.gap.clear();
        apply(&mut state, next.clone(), &[record.clone()]).unwrap();
        assert!(state.review_candidates().is_empty());
        next.conflict = "Observed outcome conflicts with the proposition.".into();
        record.is_verification = false;
        record.tool_name = "plan".into();
        apply(&mut state, next.clone(), &[record.clone()]).unwrap();
        assert!(state.review_candidates().is_empty());
        next.evidence.clear();
        apply(&mut state, next, &[]).unwrap();
        assert!(state.review_candidates().is_empty());
        let mut invalid = update("bad", Status::Limited);
        invalid.gap.clear();
        assert!(ValidatedUpdate::new(invalid).is_err());
    }

    #[test]
    fn critical_premises_reconstruct_and_turn_isolation() {
        let records = [observation("read-0", 0)];
        let mut state = PremiseState::default();
        let a = apply(&mut state, update("a", Status::Supported), &records).unwrap();
        let mut b = update("b", Status::Supported);
        b.dependencies = vec!["a".into()];
        let b = apply(&mut state, b, &records).unwrap();
        let changed = apply(&mut state, update("a", Status::Refuted), &records).unwrap();
        let mut receipts = vec![canonical(&a, 1), canonical(&b, 2), canonical(&changed, 3)];
        assert_eq!(PremiseState::rebuild(&receipts, &records).unwrap(), state);
        let mut provisional = canonical(&changed, 4);
        provisional.tool_call_id = "provisional".into();
        provisional.content = super::super::service::execute(
            &serde_json::to_value(update("phantom", Status::Pending)).unwrap(),
        )
        .unwrap();
        receipts.push(provisional);
        let mut failed = canonical(&a, 5);
        failed.success = false;
        receipts.push(failed);
        assert_eq!(PremiseState::rebuild(&receipts, &records).unwrap(), state);
        assert!(!PremiseState::rebuild(&[], &records).unwrap().has_records());
        assert!(PremiseState::rebuild(&receipts, &[]).is_err());
        let mut future = records.to_vec();
        future[0].canonical_index = 1;
        assert!(PremiseState::rebuild(&receipts, &future).is_err());
        let mut forged = canonical(&a, 1);
        let mut value: Value = serde_json::from_str(&forged.content).unwrap();
        value["sequence"] = 99.into();
        forged.content = value.to_string();
        assert!(PremiseState::rebuild(&[forged], &records).is_err());
        let mut mismatch = records.to_vec();
        mismatch[0].observed_objects[0].object = "/other".into();
        assert!(PremiseState::rebuild(&receipts, &mismatch).is_err());
        assert!(PremiseState::rebuild(&[canonical(&b, 2), canonical(&a, 1)], &records).is_err());
        let value: Value = serde_json::from_str(&a.to_tool_result().unwrap()).unwrap();
        assert_eq!(value["kind"], "critical_premises.runtime_bound.v1");
        assert_eq!(value["semantic_origin"], "model_derived");
        assert!(
            !a.to_tool_result()
                .unwrap()
                .contains(&records[0].original_content)
        );
    }
}
