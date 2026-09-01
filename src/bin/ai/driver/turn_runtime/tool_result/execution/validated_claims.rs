//! Manifest-driven final-response protocol for evidence-heavy investigations.
//!
//! The model supplies artifact receipts and exact field facts. The runtime resolves
//! those receipts against canonical current-turn tool messages, validates each fact,
//! derives comparisons and identity relationships with closed rules, and renders the
//! user-visible response. Free-form model text is confined to explicitly unverified
//! questions and coverage gaps.

use super::*;
use serde::Deserialize;

pub(in crate::ai::driver::turn_runtime) const VALIDATED_CLAIMS_RETRY_MARKER: &str =
    "[validated-claims-retry]";
pub(in crate::ai::driver::turn_runtime) const VALIDATED_CLAIMS_UNVERIFIED_NOTE: &str = "runtime:validated_claims_withheld\nA multi-artifact investigation result was withheld because its structured claims could not be verified from canonical current-turn tool evidence.";
pub(in crate::ai::driver::turn_runtime) const VALIDATED_CLAIMS_WARNING: &str = "[Runtime warning] Cross-artifact conclusions were withheld because the validated-claims evidence protocol did not verify them.";

const REPORT_OPEN: &str = "<validated_claims>";
const REPORT_CLOSE: &str = "</validated_claims>";
const PROTOCOL_VERSION: &str = "validated_claims/v1";
const MAX_REPORT_BYTES: usize = 64 * 1024;
const MAX_ARTIFACTS: usize = 24;
const MAX_FACTS: usize = 96;
const MAX_COMPARISONS: usize = 64;
const MAX_RELATIONS: usize = 32;
const MAX_LIST_ITEMS: usize = 64;
const MAX_ID_BYTES: usize = 64;
const MAX_FIELD_BYTES: usize = 128;
const MAX_VALUE_BYTES: usize = 512;
const MAX_EVIDENCE_BYTES: usize = 2 * 1024;
const MAX_UNVERIFIED_TEXT_BYTES: usize = 2 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum ValidatedClaimsGateAction {
    Allow,
    Reopen,
    Warn,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClaimsReport {
    protocol: String,
    artifacts: Vec<ArtifactReceipt>,
    facts: Vec<FieldFact>,
    #[serde(default)]
    comparisons: Vec<FieldComparison>,
    relations: Vec<IdentityRelation>,
    #[serde(default)]
    open_questions: Vec<String>,
    #[serde(default)]
    coverage_gaps: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactReceipt {
    id: String,
    tool_call_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldFact {
    id: String,
    artifact: String,
    field: String,
    value: String,
    evidence: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldComparison {
    left_fact: String,
    right_fact: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityRelation {
    left_artifact: String,
    right_artifact: String,
    scope: IdentityScope,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
enum IdentityScope {
    Request,
    Session,
    Trace,
}

impl IdentityScope {
    fn label(self) -> &'static str {
        match self {
            Self::Request => "Request",
            Self::Session => "Session",
            Self::Trace => "Trace",
        }
    }
}

#[derive(Debug, Clone)]
struct ObservedArtifact {
    call: ToolCall,
    content: String,
}

#[derive(Debug, Clone)]
struct ValidatedArtifact {
    id: String,
    ordinal: usize,
    observed: ObservedArtifact,
}

#[derive(Debug, Clone)]
struct ValidatedFact {
    spec: FieldFact,
    artifact_ordinal: usize,
    source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdentityVerdict {
    Same,
    Different,
    Unknown,
}

pub(in crate::ai::driver::turn_runtime) fn validated_claims_required(
    enabled: bool,
    turn_messages: &[Message],
) -> bool {
    if !enabled || current_turn_has_successful_mutation(turn_messages) {
        return false;
    }
    successful_non_mutation_call_ids(turn_messages).len() >= 2
}

#[allow(clippy::too_many_arguments)]
pub(in crate::ai::driver::turn_runtime) fn validated_claims_gate_action(
    enabled: bool,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    final_text: &mut String,
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> ValidatedClaimsGateAction {
    if !validated_claims_required(enabled, turn_messages) {
        return ValidatedClaimsGateAction::Allow;
    }

    let observed = observed_read_only_artifacts(turn_messages);
    let report = parse_report(final_text).and_then(|report| validate_report(report, &observed));
    match report {
        Ok(report) => {
            *final_text = render_report(&report);
            ValidatedClaimsGateAction::Allow
        }
        Err(reason) => retry_or_withhold(
            messages,
            final_text,
            reason,
            force_final_response,
            iteration,
            max_iterations,
        ),
    }
}

fn retry_or_withhold(
    messages: &mut Vec<Message>,
    final_text: &mut String,
    reason: &'static str,
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> ValidatedClaimsGateAction {
    if current_turn_has_internal_marker(messages, VALIDATED_CLAIMS_RETRY_MARKER)
        || force_final_response
        || iteration >= max_iterations
    {
        *final_text = format!(
            "## Verified cross-artifact conclusions\n\nNo cross-artifact conclusion was published because the evidence protocol could not be validated.\n\n{VALIDATED_CLAIMS_WARNING}"
        );
        return ValidatedClaimsGateAction::Warn;
    }

    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(format!(
            "{VALIDATED_CLAIMS_RETRY_MARKER}\n\
             This is not a final answer. The multi-artifact investigation draft is not publishable because {reason}.\n\
             Return exactly one `<validated_claims>{{...}}</validated_claims>` JSON payload and no surrounding text.\n\
             Use protocol `validated_claims/v1`. Declare stable current-turn read-only artifacts by `tool_call_id`; exact field facts with `id`, `artifact`, `field`, `value`, and a single-line verbatim `evidence` fragment; optional same-field `comparisons`; and identity `relations` with `left_artifact`, `right_artifact`, and `scope` (`request`, `session`, or `trace`). If no tool result is admissible as evidence, return empty artifact/fact/comparison/relation arrays and explain the limitation in `coverage_gaps`.\n\
             Do not author verified conclusions. The runtime derives them. Put unsupported interpretation only in `open_questions` or `coverage_gaps`."
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    ValidatedClaimsGateAction::Reopen
}

fn parse_report(text: &str) -> Result<ClaimsReport, &'static str> {
    if text.len() > MAX_REPORT_BYTES {
        return Err("the validated-claims report exceeds the size limit");
    }
    let trimmed = text.trim();
    let Some(body) = trimmed
        .strip_prefix(REPORT_OPEN)
        .and_then(|text| text.strip_suffix(REPORT_CLOSE))
    else {
        return Err("the validated-claims envelope is missing or has surrounding text");
    };
    let report: ClaimsReport = serde_json::from_str(body.trim())
        .map_err(|_| "the validated-claims payload is not valid protocol JSON")?;
    if report.protocol != PROTOCOL_VERSION {
        return Err("the validated-claims protocol version is unsupported");
    }
    if report.artifacts.len() > MAX_ARTIFACTS
        || report.facts.len() > MAX_FACTS
        || report.comparisons.len() > MAX_COMPARISONS
        || report.relations.len() > MAX_RELATIONS
        || (report.facts.is_empty() && report.coverage_gaps.is_empty())
        || (!report.facts.is_empty() && report.artifacts.is_empty())
        || (!report.relations.is_empty() && report.artifacts.len() < 2)
        || report.open_questions.len() > MAX_LIST_ITEMS
        || report.coverage_gaps.len() > MAX_LIST_ITEMS
        || report
            .open_questions
            .iter()
            .chain(report.coverage_gaps.iter())
            .any(|text| !bounded_unverified_text(text))
    {
        return Err("the validated-claims payload violates a cardinality or text limit");
    }
    Ok(report)
}

fn validate_report(
    report: ClaimsReport,
    observed: &[ObservedArtifact],
) -> Result<ValidatedReport, &'static str> {
    let observed_by_call = observed
        .iter()
        .map(|artifact| (artifact.call.id.as_str(), artifact))
        .collect::<HashMap<_, _>>();
    let mut all_ids = HashSet::new();
    let mut call_ids = HashSet::new();
    let mut artifacts = Vec::with_capacity(report.artifacts.len());
    for (index, receipt) in report.artifacts.iter().enumerate() {
        if !valid_id(&receipt.id)
            || !all_ids.insert(receipt.id.clone())
            || !call_ids.insert(receipt.tool_call_id.as_str())
        {
            return Err("artifact ids and tool-call receipts must be unique and well formed");
        }
        let Some(observed) = observed_by_call.get(receipt.tool_call_id.as_str()) else {
            return Err(
                "an artifact receipt does not resolve to successful current-turn read-only evidence",
            );
        };
        artifacts.push(ValidatedArtifact {
            id: receipt.id.clone(),
            ordinal: index + 1,
            observed: (*observed).clone(),
        });
    }
    let artifact_by_id = artifacts
        .iter()
        .map(|artifact| (artifact.id.as_str(), artifact))
        .collect::<HashMap<_, _>>();

    let mut facts = Vec::with_capacity(report.facts.len());
    for fact in &report.facts {
        if !valid_id(&fact.id) || !all_ids.insert(fact.id.clone()) {
            return Err("fact ids must be globally unique and well formed");
        }
        let Some(artifact) = artifact_by_id.get(fact.artifact.as_str()) else {
            return Err("a fact references an unknown artifact");
        };
        if !valid_fact_text(fact)
            || !artifact.observed.content.contains(&fact.evidence)
            || !evidence_binds_field_value(&fact.evidence, &fact.field, &fact.value)
        {
            return Err("a field fact is not exactly grounded in its artifact result");
        }
        facts.push(ValidatedFact {
            spec: fact.clone(),
            artifact_ordinal: artifact.ordinal,
            source: render_fact_source(artifact, &fact.evidence),
        });
    }
    let fact_by_id = facts
        .iter()
        .map(|fact| (fact.spec.id.as_str(), fact))
        .collect::<HashMap<_, _>>();

    let mut comparisons = Vec::with_capacity(report.comparisons.len());
    let mut comparison_keys = HashSet::new();
    for comparison in &report.comparisons {
        let Some(left) = fact_by_id.get(comparison.left_fact.as_str()) else {
            return Err("a comparison references an unknown fact");
        };
        let Some(right) = fact_by_id.get(comparison.right_fact.as_str()) else {
            return Err("a comparison references an unknown fact");
        };
        if left.artifact_ordinal == right.artifact_ordinal
            || normalize_field(&left.spec.field) != normalize_field(&right.spec.field)
        {
            return Err("comparisons must connect the same field across distinct artifacts");
        }
        let key = ordered_pair(&left.spec.id, &right.spec.id);
        if !comparison_keys.insert(key) {
            return Err("duplicate field comparison");
        }
        comparisons.push(((*left).clone(), (*right).clone()));
    }

    let mut relations = Vec::with_capacity(report.relations.len());
    let mut relation_keys = HashSet::new();
    for relation in &report.relations {
        let Some(left) = artifact_by_id.get(relation.left_artifact.as_str()) else {
            return Err("a relation references an unknown artifact");
        };
        let Some(right) = artifact_by_id.get(relation.right_artifact.as_str()) else {
            return Err("a relation references an unknown artifact");
        };
        if left.ordinal == right.ordinal {
            return Err("an identity relation must connect distinct artifacts");
        }
        let (first, second) = if left.ordinal < right.ordinal {
            (left.ordinal, right.ordinal)
        } else {
            (right.ordinal, left.ordinal)
        };
        if !relation_keys.insert((first, second, relation.scope)) {
            return Err("duplicate identity relation");
        }
        let (verdict, keys) = derive_identity_relation(relation.scope, left, right, &facts);
        relations.push((left.ordinal, right.ordinal, relation.scope, verdict, keys));
    }

    Ok(ValidatedReport {
        facts,
        comparisons,
        relations,
        open_questions: report.open_questions,
        coverage_gaps: report.coverage_gaps,
    })
}

#[derive(Debug)]
struct ValidatedReport {
    facts: Vec<ValidatedFact>,
    comparisons: Vec<(ValidatedFact, ValidatedFact)>,
    relations: Vec<(
        usize,
        usize,
        IdentityScope,
        IdentityVerdict,
        Vec<(String, String, String)>,
    )>,
    open_questions: Vec<String>,
    coverage_gaps: Vec<String>,
}

fn observed_read_only_artifacts(turn_messages: &[Message]) -> Vec<ObservedArtifact> {
    let turn_start = crate::ai::history::last_real_user_index(turn_messages).unwrap_or(0);
    let mut calls_by_id: HashMap<String, ToolCall> = HashMap::new();
    let mut artifacts = Vec::new();
    for message in turn_messages.iter().skip(turn_start) {
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                calls_by_id.insert(call.id.clone(), call.clone());
            }
        }
        if message.role != "tool" || !completion_tool_result_succeeded(&message.content) {
            continue;
        }
        let Some(call) = message
            .tool_call_id
            .as_deref()
            .and_then(|id| calls_by_id.get(id))
        else {
            continue;
        };
        if read_only_tool_signature(call).is_none() {
            continue;
        }
        let Some(content) = message.content.as_str() else {
            continue;
        };
        artifacts.push(ObservedArtifact {
            call: call.clone(),
            content: content.to_string(),
        });
    }
    artifacts
}

fn successful_non_mutation_call_ids(turn_messages: &[Message]) -> HashSet<String> {
    let turn_start = crate::ai::history::last_real_user_index(turn_messages).unwrap_or(0);
    let mut calls_by_id: HashMap<String, ToolCall> = HashMap::new();
    let mut call_ids = HashSet::new();
    for message in turn_messages.iter().skip(turn_start) {
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                calls_by_id.insert(call.id.clone(), call.clone());
            }
        }
        if message.role != "tool" || !completion_tool_result_succeeded(&message.content) {
            continue;
        }
        let Some(call) = message
            .tool_call_id
            .as_deref()
            .and_then(|id| calls_by_id.get(id))
        else {
            continue;
        };
        if !tool_call_is_successful_mutation_candidate(call) {
            call_ids.insert(call.id.clone());
        }
    }
    call_ids
}

fn current_turn_has_successful_mutation(turn_messages: &[Message]) -> bool {
    let turn_start = crate::ai::history::last_real_user_index(turn_messages).unwrap_or(0);
    let mut calls_by_id: HashMap<String, ToolCall> = HashMap::new();
    for message in turn_messages.iter().skip(turn_start) {
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                calls_by_id.insert(call.id.clone(), call.clone());
            }
        }
        if message.role != "tool" || !completion_tool_result_succeeded(&message.content) {
            continue;
        }
        if message
            .tool_call_id
            .as_deref()
            .and_then(|id| calls_by_id.get(id))
            .is_some_and(tool_call_is_successful_mutation_candidate)
        {
            return true;
        }
    }
    false
}

fn valid_id(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= MAX_ID_BYTES
        && text
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn valid_fact_text(fact: &FieldFact) -> bool {
    bounded_single_line(&fact.field, MAX_FIELD_BYTES)
        && bounded_single_line(&fact.value, MAX_VALUE_BYTES)
        && bounded_single_line(&fact.evidence, MAX_EVIDENCE_BYTES)
        && !normalize_field(&fact.field).is_empty()
}

fn bounded_unverified_text(text: &str) -> bool {
    bounded_single_line(text, MAX_UNVERIFIED_TEXT_BYTES)
}

fn bounded_single_line(text: &str, max_bytes: usize) -> bool {
    !text.trim().is_empty()
        && text.len() <= max_bytes
        && !text.contains('\n')
        && !text.contains('\r')
}

fn normalize_field(field: &str) -> String {
    field
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn evidence_binds_field_value(evidence: &str, field: &str, value: &str) -> bool {
    let mut search_from = 0;
    while let Some(relative) = evidence[search_from..].find(field) {
        let field_start = search_from + relative;
        let before_is_identifier =
            evidence[..field_start]
                .chars()
                .next_back()
                .is_some_and(|character| {
                    character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
                });
        let after_field = field_start + field.len();
        if before_is_identifier {
            search_from = after_field;
            continue;
        }
        let mut tail = &evidence[after_field..];
        tail = tail.trim_start_matches(|character: char| {
            character.is_whitespace() || matches!(character, '\\' | '"' | '\'' | '`')
        });
        let Some(delimiter) = tail.chars().next() else {
            return false;
        };
        if !matches!(delimiter, ':' | '=') {
            search_from = after_field;
            continue;
        }
        tail = &tail[delimiter.len_utf8()..];
        tail = tail.trim_start_matches(char::is_whitespace);
        if value_matches_complete_field_value(tail, value) {
            return true;
        }
        search_from = after_field;
    }
    false
}

fn value_matches_complete_field_value(tail: &str, value: &str) -> bool {
    for (opening, closing) in [
        ("\\\"", "\\\""),
        ("\\'", "\\'"),
        ("\"", "\""),
        ("'", "'"),
        ("`", "`"),
    ] {
        if let Some(quoted) = tail.strip_prefix(opening) {
            return quoted
                .strip_prefix(value)
                .is_some_and(|remainder| remainder.starts_with(closing));
        }
    }
    tail.strip_prefix(value).is_some_and(|remainder| {
        remainder.chars().next().is_none_or(|character| {
            character.is_whitespace() || matches!(character, ',' | '}' | ']' | '&' | ';')
        })
    })
}

fn identity_scope_for_field(field: &str) -> Option<IdentityScope> {
    match normalize_field(field).as_str() {
        "requestid" | "reqid" | "xrequestid" => Some(IdentityScope::Request),
        "sessionid" => Some(IdentityScope::Session),
        "traceid" | "correlationid" => Some(IdentityScope::Trace),
        _ => None,
    }
}

fn derive_identity_relation(
    scope: IdentityScope,
    left: &ValidatedArtifact,
    right: &ValidatedArtifact,
    facts: &[ValidatedFact],
) -> (IdentityVerdict, Vec<(String, String, String)>) {
    let values_for = |ordinal: usize| {
        let mut values = HashSet::new();
        let mut aliases = HashSet::new();
        for fact in facts.iter().filter(|fact| fact.artifact_ordinal == ordinal) {
            if identity_scope_for_field(&fact.spec.field) == Some(scope) {
                aliases.insert(normalize_field(&fact.spec.field));
                values.insert(fact.spec.value.clone());
            }
        }
        (values, aliases)
    };
    let (left_values, left_aliases) = values_for(left.ordinal);
    let (right_values, right_aliases) = values_for(right.ordinal);
    if left_values.len() != 1 || right_values.len() != 1 {
        return (IdentityVerdict::Unknown, Vec::new());
    }
    let left_value = left_values.into_iter().next().unwrap_or_default();
    let right_value = right_values.into_iter().next().unwrap_or_default();
    let mut aliases = left_aliases
        .union(&right_aliases)
        .cloned()
        .collect::<Vec<_>>();
    aliases.sort();
    let comparisons = vec![(aliases.join("/"), left_value.clone(), right_value.clone())];
    let verdict = if left_value == right_value {
        IdentityVerdict::Same
    } else {
        IdentityVerdict::Different
    };
    (verdict, comparisons)
}

fn ordered_pair(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
}

fn render_fact_source(artifact: &ValidatedArtifact, evidence: &str) -> String {
    if artifact.observed.call.function.name == "read_file"
        && let Ok(arguments) =
            serde_json::from_str::<serde_json::Value>(&artifact.observed.call.function.arguments)
        && let Some(path) = arguments
            .get("file_path")
            .or_else(|| arguments.get("path"))
            .and_then(serde_json::Value::as_str)
    {
        let line = artifact
            .observed
            .content
            .lines()
            .find(|line| line.contains(evidence))
            .and_then(|line| line.split_once('\t'))
            .and_then(|(prefix, _)| prefix.trim().parse::<u64>().ok());
        return line
            .map(|line| format!("{path}:{line}"))
            .unwrap_or_else(|| path.to_string());
    }
    format!(
        "{} tool call {}",
        artifact.observed.call.function.name, artifact.observed.call.id
    )
}

fn render_report(report: &ValidatedReport) -> String {
    let mut output = String::from("## Verified facts\n\n");
    if report.facts.is_empty() {
        output.push_str("No admissible field facts were available.\n");
    } else {
        for fact in &report.facts {
            output.push_str(&format!(
                "- Artifact {}: `{}` = `{}` — `{}`\n",
                fact.artifact_ordinal, fact.spec.field, fact.spec.value, fact.source
            ));
        }
    }

    if !report.comparisons.is_empty() {
        output.push_str("\n## Verified field comparisons\n\n");
        for (left, right) in &report.comparisons {
            let relation = if left.spec.value == right.spec.value {
                "has the same value"
            } else {
                "differs"
            };
            output.push_str(&format!(
                "- `{}` {relation} between Artifact {} (`{}`) and Artifact {} (`{}`).\n",
                left.spec.field,
                left.artifact_ordinal,
                left.spec.value,
                right.artifact_ordinal,
                right.spec.value
            ));
        }
    }

    if !report.relations.is_empty() {
        output.push_str("\n## Verified artifact identity\n\n");
        for (left, right, scope, verdict, keys) in &report.relations {
            match verdict {
                IdentityVerdict::Same => output.push_str(&format!(
                    "- {} identity for Artifact {} and Artifact {} is **the same**, proven by {}.\n",
                    scope.label(), left, right, render_identity_keys(keys)
                )),
                IdentityVerdict::Different => output.push_str(&format!(
                    "- {} identity for Artifact {} and Artifact {} is **different**, proven by {}.\n",
                    scope.label(), left, right, render_identity_keys(keys)
                )),
                IdentityVerdict::Unknown => output.push_str(&format!(
                    "- {} identity for Artifact {} and Artifact {} is **not established**; no single comparable registered identity key proves the relationship.\n",
                    scope.label(), left, right
                )),
            }
        }
    }

    if !report.open_questions.is_empty() {
        output.push_str("\n## Unverified questions or hypotheses\n\n");
        for item in &report.open_questions {
            output.push_str(&format!("- **Unverified:** {item}\n"));
        }
    }
    if !report.coverage_gaps.is_empty() {
        output.push_str("\n## Coverage gaps\n\n");
        for item in &report.coverage_gaps {
            output.push_str(&format!("- **Unverified gap:** {item}\n"));
        }
    }
    output
}

fn render_identity_keys(keys: &[(String, String, String)]) -> String {
    keys.iter()
        .map(|(key, left, right)| {
            if left == right {
                format!("`{key}` = `{left}`")
            } else {
                format!("`{key}`: `{left}` vs `{right}`")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}
