//! Worker-reported verification is optional evidence, never terminal or integration authority.

use rust_tools::commonw::FastSet;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

pub(super) const TASK_VERIFICATION_SCHEMA: &str = "ai.task-verification";
pub(super) const TASK_VERIFICATION_MAX_BYTES: usize = 8 * 1024;
pub(super) const TASK_VERIFICATION_MAX_CHECKS: usize = 16;
const MAX_ASSESSMENT_GAPS: usize = 16;
const MAX_GAP_BYTES: usize = 192;
const INLINE_ASSESSMENT_BYTES: usize = 1024;
const OPEN_TAG: &str = "<task_verification>";
const CLOSE_TAG: &str = "</task_verification>";
const UNAVAILABLE_REASON: &str = "_runtime_unavailable_reason";

/// An opaque, bounded nested value keeps malformed/future shapes from invalidating task truth.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(transparent)]
pub(crate) struct TaskVerificationMetadata(Value);

impl<'de> Deserialize<'de> for TaskVerificationMetadata {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Value::deserialize(deserializer).map(Self::from_value)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VerificationSchemaStatus {
    Valid,
    Invalid,
    Unsupported,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct TaskVerificationAssessment {
    pub(crate) provenance: &'static str,
    pub(crate) schema_status: VerificationSchemaStatus,
    pub(crate) acceptance: &'static str,
    pub(crate) gaps: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TaskKind {
    Fix,
    Implementation,
    Research,
    Review,
    Other,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CheckPhase {
    Before,
    After,
    Other,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CheckOutcome {
    Pass,
    Fail,
    Blocked,
    NotRun,
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
struct TaskVerificationArtifact {
    #[serde(default)]
    paths: Vec<String>,
    #[serde(default)]
    revision: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TaskVerificationCheck {
    id: String,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    cwd: Option<String>,
    subject: String,
    #[serde(default)]
    configuration: Option<Value>,
    #[serde(default)]
    scenario: Option<String>,
    phase: CheckPhase,
    outcome: CheckOutcome,
    /// A before-check must identify the old artifact; the result artifact can identify after.
    #[serde(default)]
    artifact: Option<TaskVerificationArtifact>,
    #[serde(default)]
    evidence_locations: Vec<String>,
    #[serde(default)]
    limitations: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TaskVerificationRegression {
    #[serde(default)]
    before_check_id: Option<String>,
    #[serde(default)]
    after_check_id: Option<String>,
    #[serde(default)]
    unavailable_reason: Option<String>,
    #[serde(default)]
    alternative_evidence: Vec<String>,
    #[serde(default)]
    limitations: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TaskVerification {
    schema: String,
    version: u32,
    task_kind: TaskKind,
    #[serde(default)]
    artifact: Option<TaskVerificationArtifact>,
    checks: Vec<TaskVerificationCheck>,
    #[serde(default)]
    regression: Option<TaskVerificationRegression>,
    #[serde(default)]
    assumptions: Vec<String>,
    #[serde(default)]
    remaining_uncertainty: Vec<String>,
}

fn bounded_text(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes.saturating_sub(3);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

impl TaskVerificationAssessment {
    fn new(schema_status: VerificationSchemaStatus, mut gaps: Vec<String>) -> Self {
        if gaps.len() > MAX_ASSESSMENT_GAPS {
            let omitted = gaps.len() - MAX_ASSESSMENT_GAPS + 1;
            gaps.truncate(MAX_ASSESSMENT_GAPS - 1);
            gaps.push(format!(
                "{omitted} additional gaps omitted; review the full verification metadata."
            ));
        }
        Self {
            provenance: "worker_reported",
            schema_status,
            acceptance: "needs_parent_review",
            gaps: gaps
                .iter()
                .map(|gap| bounded_text(gap, MAX_GAP_BYTES))
                .collect(),
        }
    }
}

impl TaskVerificationMetadata {
    fn unavailable(reason: &str) -> Self {
        Self(json!({
            "schema": TASK_VERIFICATION_SCHEMA,
            "version": 1,
            (UNAVAILABLE_REASON): bounded_text(reason, MAX_GAP_BYTES),
        }))
    }

    pub(super) fn from_value(value: Value) -> Self {
        if value.is_null() {
            return Self::unavailable("Verification block is null rather than a schema object.");
        }
        if value
            .get("checks")
            .and_then(Value::as_array)
            .is_some_and(|checks| checks.len() > TASK_VERIFICATION_MAX_CHECKS)
        {
            return Self::unavailable(
                "Verification exceeds the 16-check limit; no checks were accepted.",
            );
        }
        if serde_json::to_vec(&value)
            .map_or(true, |bytes| bytes.len() > TASK_VERIFICATION_MAX_BYTES)
        {
            return Self::unavailable(
                "Verification exceeds the 8 KiB metadata limit; review the original output.",
            );
        }
        Self(value)
    }

    fn decode(&self) -> Result<TaskVerification, (VerificationSchemaStatus, String)> {
        let invalid = |reason: String| (VerificationSchemaStatus::Invalid, reason);
        if let Some(reason) = self.0.get(UNAVAILABLE_REASON).and_then(Value::as_str) {
            return Err(invalid(reason.to_string()));
        }
        let schema = self
            .0
            .get("schema")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("Verification is missing its schema identifier.".to_string()))?;
        let version = self
            .0
            .get("version")
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid("Verification is missing an integer version.".to_string()))?;
        if schema != TASK_VERIFICATION_SCHEMA || version != 1 {
            return Err((
                VerificationSchemaStatus::Unsupported,
                "Unsupported verification schema/version; review the original worker evidence."
                    .to_string(),
            ));
        }
        let verification: TaskVerification = serde_json::from_value(self.0.clone())
            .map_err(|error| invalid(format!("Invalid verification schema: {error}")))?;
        debug_assert_eq!(verification.schema, TASK_VERIFICATION_SCHEMA);
        debug_assert_eq!(verification.version, 1);
        let mut ids = FastSet::default();
        for check in &verification.checks {
            if check.id.trim().is_empty() || check.subject.trim().is_empty() {
                return Err(invalid(
                    "Check IDs and subjects must be non-empty.".to_string(),
                ));
            }
            if !ids.insert(check.id.as_str()) {
                return Err(invalid(
                    "Verification has duplicate check IDs; regression references are ambiguous."
                        .to_string(),
                ));
            }
        }
        Ok(verification)
    }

    pub(crate) fn assessment(&self) -> TaskVerificationAssessment {
        let verification = match self.decode() {
            Ok(verification) => verification,
            Err((status, reason)) => return TaskVerificationAssessment::new(status, vec![reason]),
        };
        let mut gaps = Vec::new();
        for check in &verification.checks {
            let expected_before_failure = verification.task_kind == TaskKind::Fix
                && check.phase == CheckPhase::Before
                && check.outcome == CheckOutcome::Fail;
            if check.outcome != CheckOutcome::Pass && !expected_before_failure {
                gaps.push(format!(
                    "Check {:?} reported {:?}; it is not passing evidence.",
                    check.id, check.outcome
                ));
            }
            if !check
                .evidence_locations
                .iter()
                .any(|location| !location.trim().is_empty())
            {
                gaps.push(format!("Check {:?} has no evidence locations.", check.id));
            }
            if check
                .command
                .as_deref()
                .is_some_and(|command| !command.trim().is_empty())
                && check.cwd.as_deref().is_none_or(|cwd| cwd.trim().is_empty())
            {
                gaps.push(format!(
                    "Check {:?} does not identify the command's working directory.",
                    check.id
                ));
            }
            for limitation in &check.limitations {
                gaps.push(format!("Check {:?} limitation: {limitation}", check.id));
            }
        }
        if verification.task_kind == TaskKind::Fix {
            assess_fix_regression(&verification, &mut gaps);
        } else if verification.task_kind == TaskKind::Implementation
            && !verification.checks.iter().any(|check| {
                check.phase == CheckPhase::After && check.outcome == CheckOutcome::Pass
            })
        {
            gaps.push("Implementation has no passing after-change check.".to_string());
        }
        for assumption in &verification.assumptions {
            gaps.push(format!("Assumption to review: {assumption}"));
        }
        for uncertainty in &verification.remaining_uncertainty {
            gaps.push(format!("Remaining uncertainty: {uncertainty}"));
        }
        TaskVerificationAssessment::new(VerificationSchemaStatus::Valid, gaps)
    }

    pub(crate) fn inline_assessment(&self) -> String {
        let assessment = self.assessment();
        let status = match assessment.schema_status {
            VerificationSchemaStatus::Valid => "valid",
            VerificationSchemaStatus::Invalid => "invalid",
            VerificationSchemaStatus::Unsupported => "unsupported",
        };
        let mut text = format!(
            "[Task verification] provenance=worker_reported schema_status={status} acceptance=needs_parent_review"
        );
        if assessment.gaps.is_empty() {
            text.push_str(
                "\nNo schema gaps detected; the reported checks still require parent review.",
            );
        } else {
            for gap in assessment.gaps.iter().take(3) {
                text.push_str("\n- ");
                text.push_str(gap);
            }
            if assessment.gaps.len() > 3 {
                text.push_str(&format!(
                    "\n- {} more gap(s); read the full assessment.",
                    assessment.gaps.len() - 3
                ));
            }
        }
        text.push_str(&format!(
            "\nDetails: {}. This assessment is advisory.",
            super::full_delivered_evidence_hint(super::EvidenceRecoveryFormat::Compact)
        ));
        bounded_text(&text, INLINE_ASSESSMENT_BYTES)
    }
}

fn assess_fix_regression(verification: &TaskVerification, gaps: &mut Vec<String>) {
    let Some(regression) = &verification.regression else {
        gaps.push("Fix result has no regression before/after references.".to_string());
        return;
    };
    if let Some(reason) = &regression.unavailable_reason
        && !reason.trim().is_empty()
    {
        gaps.push(format!(
            "Reproduction unavailable (scoped limitation): {reason}"
        ));
        if !regression
            .alternative_evidence
            .iter()
            .any(|evidence| !evidence.trim().is_empty())
        {
            gaps.push("Unavailable reproduction has no alternative evidence.".to_string());
        }
    }
    for limitation in &regression.limitations {
        gaps.push(format!("Regression limitation: {limitation}"));
    }
    let find = |id: Option<&str>| {
        verification
            .checks
            .iter()
            .find(|check| Some(check.id.as_str()) == id)
    };
    let before = find(regression.before_check_id.as_deref());
    let after = find(regression.after_check_id.as_deref());
    if before.is_none_or(|check| {
        check.phase != CheckPhase::Before || check.outcome != CheckOutcome::Fail
    }) {
        gaps.push(
            "Fix needs a referenced before-check showing fail against the old artifact."
                .to_string(),
        );
    }
    if after
        .is_none_or(|check| check.phase != CheckPhase::After || check.outcome != CheckOutcome::Pass)
    {
        gaps.push(
            "Fix needs a referenced after-check showing pass against the changed artifact."
                .to_string(),
        );
    }
    let (Some(before), Some(after)) = (before, after) else {
        return;
    };
    for (matches, label) in [
        (before.command == after.command, "command"),
        (before.subject == after.subject, "subject"),
        (before.configuration == after.configuration, "configuration"),
        (before.scenario == after.scenario, "scenario"),
    ] {
        if !matches {
            gaps.push(format!(
                "Before/after check {label} differs; the reproduction scopes do not match."
            ));
        }
    }
    if before
        .command
        .as_deref()
        .is_none_or(|command| command.trim().is_empty())
        && before
            .scenario
            .as_deref()
            .is_none_or(|scenario| scenario.trim().is_empty())
    {
        gaps.push("Before/after checks do not identify a command or scenario.".to_string());
    }
    let before_artifact = before.artifact.as_ref();
    let after_artifact = after.artifact.as_ref().or(verification.artifact.as_ref());
    match (
        artifact_revision(before_artifact),
        artifact_revision(after_artifact),
    ) {
        (Some(before), Some(after)) if before != after => {}
        (Some(_), Some(_)) => {
            gaps.push("Before/after checks identify the same artifact revision.".to_string())
        }
        _ => gaps.push("Before/after checks must identify both artifact revisions.".to_string()),
    }
    let before_paths = artifact_paths(before_artifact);
    let after_paths = artifact_paths(after_artifact);
    if before_paths.is_empty() || after_paths.is_empty() || before_paths != after_paths {
        gaps.push("Before/after checks must identify matching artifact paths.".to_string());
    }
    if let Some(result_artifact) = &verification.artifact
        && (artifact_revision(Some(result_artifact)) != artifact_revision(after_artifact)
            || artifact_paths(Some(result_artifact)) != after_paths)
    {
        gaps.push("After-check artifact does not match the reported result artifact.".to_string());
    }
}

fn artifact_revision(artifact: Option<&TaskVerificationArtifact>) -> Option<&str> {
    artifact
        .and_then(|artifact| artifact.revision.as_deref())
        .filter(|revision| !revision.trim().is_empty())
}

fn artifact_paths(artifact: Option<&TaskVerificationArtifact>) -> FastSet<&str> {
    artifact
        .map(|artifact| {
            artifact
                .paths
                .iter()
                .map(String::as_str)
                .filter(|path| !path.trim().is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Extract only the optional contract; the caller validates the original final body first.
pub(crate) fn extract_task_verification(
    output: &str,
    response_schema: Option<&Value>,
) -> Option<TaskVerificationMetadata> {
    let trimmed = output.trim();
    if trimmed.starts_with("<subagent_handoff>") {
        return None;
    }
    if response_schema.is_some() {
        let value: Value = serde_json::from_str(trimmed).ok()?;
        let verification = value.as_object()?.get("verification")?;
        return (verification.get("schema").and_then(Value::as_str)
            == Some(TASK_VERIFICATION_SCHEMA))
        .then(|| TaskVerificationMetadata::from_value(verification.clone()));
    }
    let mut opening = None;
    let mut offset = 0;
    let mut fence = None;
    for line in output.split_inclusive('\n') {
        let text = line.trim_start();
        let fence_marker = if text.starts_with("```") {
            Some('`')
        } else if text.starts_with("~~~") {
            Some('~')
        } else {
            None
        };
        if let Some(marker) = fence_marker {
            if fence == Some(marker) {
                fence = None;
            } else if fence.is_none() {
                fence = Some(marker);
            }
        } else if fence.is_none() && text.starts_with("<audit_report>") {
            return None;
        } else if fence.is_none() && text.starts_with(OPEN_TAG) {
            if opening.is_some() {
                return Some(TaskVerificationMetadata::unavailable(
                    "Multiple standalone verification blocks are ambiguous.",
                ));
            }
            opening = Some(offset + line.len() - text.len());
        }
        offset += line.len();
    }
    let start = opening? + OPEN_TAG.len();
    let Some(relative_end) = output[start..].rfind(CLOSE_TAG) else {
        return Some(TaskVerificationMetadata::unavailable(
            "Verification block is missing its closing tag.",
        ));
    };
    let end = start + relative_end;
    if !output[end + CLOSE_TAG.len()..]
        .split('\n')
        .next()
        .unwrap_or_default()
        .trim()
        .is_empty()
    {
        return Some(TaskVerificationMetadata::unavailable(
            "Verification block must be standalone.",
        ));
    }
    let body = output[start..end].trim();
    if body.len() > TASK_VERIFICATION_MAX_BYTES {
        return Some(TaskVerificationMetadata::unavailable(
            "Verification exceeds the 8 KiB metadata limit; review the original output.",
        ));
    }
    Some(match serde_json::from_str(body) {
        Ok(value) => TaskVerificationMetadata::from_value(value),
        Err(error) => {
            TaskVerificationMetadata::unavailable(&format!("Malformed verification JSON: {error}"))
        }
    })
}

#[cfg(test)]
pub(super) fn fix_verification_fixture() -> Value {
    json!({
        "schema": TASK_VERIFICATION_SCHEMA, "version": 1, "task_kind": "fix",
        "artifact": {"paths": ["src/parser.rs"], "revision": "fixed-revision"},
        "checks": [
            {
                "id": "reproduction", "command": "cargo test --bin a new_regression",
                "cwd": "/work/old-revision", "subject": "parser empty-input regression",
                "configuration": {"features": []}, "scenario": "empty input",
                "phase": "before", "outcome": "fail",
                "artifact": {"paths": ["src/parser.rs"], "revision": "old-revision"},
                "evidence_locations": ["artifacts/before.log"], "limitations": []
            },
            {
                "id": "regression", "command": "cargo test --bin a new_regression",
                "cwd": "/work/fixed-revision", "subject": "parser empty-input regression",
                "configuration": {"features": []}, "scenario": "empty input",
                "phase": "after", "outcome": "pass",
                "evidence_locations": ["artifacts/after.log"], "limitations": []
            }
        ],
        "regression": {
            "before_check_id": "reproduction", "after_check_id": "regression",
            "alternative_evidence": [], "limitations": []
        },
        "assumptions": [], "remaining_uncertainty": []
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::tools::task_tools::spawn::wrap_subagent_prompt;
    use crate::ai::tools::task_tools::{
        StoredTaskResult, TaskReport, TaskReportTask, parse_subagent_handoff,
        validate_subagent_response,
    };

    #[test]
    fn task_verification_accepts_fix_reproduction_and_research_without_regression() {
        let fix = fix_verification_fixture();
        let assessment = TaskVerificationMetadata::from_value(fix.clone()).assessment();
        assert_eq!(assessment.schema_status, VerificationSchemaStatus::Valid);
        assert_eq!(assessment.provenance, "worker_reported");
        assert_eq!(assessment.acceptance, "needs_parent_review");
        assert!(assessment.gaps.is_empty(), "{:?}", assessment.gaps);
        let research = TaskVerificationMetadata::from_value(json!({
            "schema": TASK_VERIFICATION_SCHEMA, "version": 1, "task_kind": "research",
            "checks": [{"id": "source", "subject": "owning source",
                "phase": "other", "outcome": "pass", "evidence_locations": ["src/source.rs:10"]}],
        }))
        .assessment();
        assert_eq!(research.schema_status, VerificationSchemaStatus::Valid);
        assert!(research.gaps.is_empty());

        for (pointer, replacement, expected) in [
            (
                "/checks/0/outcome",
                json!("pass"),
                "before-check showing fail",
            ),
            (
                "/checks/1/outcome",
                json!("blocked"),
                "after-check showing pass",
            ),
            (
                "/checks/1/command",
                json!("cargo check --bin a"),
                "command differs",
            ),
            (
                "/checks/1/subject",
                json!("a different function"),
                "subject differs",
            ),
            (
                "/checks/1/configuration",
                json!({"features": ["serve"]}),
                "configuration differs",
            ),
            (
                "/checks/1/scenario",
                json!("non-empty input"),
                "scenario differs",
            ),
            (
                "/checks/0/artifact/revision",
                json!("fixed-revision"),
                "same artifact revision",
            ),
            (
                "/checks/0/artifact/revision",
                Value::Null,
                "both artifact revisions",
            ),
            (
                "/checks/0/artifact/paths",
                json!(["src/other.rs"]),
                "matching artifact paths",
            ),
        ] {
            let mut mismatch = fix.clone();
            *mismatch.pointer_mut(pointer).unwrap() = replacement;
            let assessment = TaskVerificationMetadata::from_value(mismatch).assessment();
            assert_eq!(assessment.schema_status, VerificationSchemaStatus::Valid);
            assert!(
                assessment.gaps.iter().any(|gap| gap.contains(expected)),
                "{expected}: {:?}",
                assessment.gaps
            );
        }
        let unavailable = TaskVerificationMetadata::from_value(json!({
            "schema": TASK_VERIFICATION_SCHEMA, "version": 1, "task_kind": "fix", "checks": [],
            "regression": {"unavailable_reason": "production-only trigger",
                "alternative_evidence": ["artifacts/source-trace.md"], "limitations": ["not reproduced"]},
        })).assessment();
        assert_eq!(unavailable.schema_status, VerificationSchemaStatus::Valid);
        assert!(
            unavailable
                .gaps
                .iter()
                .any(|gap| gap.contains("scoped limitation"))
        );
        assert!(
            unavailable
                .gaps
                .iter()
                .any(|gap| gap.contains("before-check showing fail"))
        );
    }

    #[test]
    fn task_verification_preserves_explicit_schema_and_control_envelopes() {
        let schema = json!({"type": "object", "properties": {
            "summary": {"type": "string"}, "verification": {"type": "object"}
        }, "required": ["summary"], "additionalProperties": false});
        let prompt = wrap_subagent_prompt("fix", "fix the parser", Some(&schema));
        assert!(prompt.contains("Return exactly one JSON value"));
        assert!(!prompt.contains(OPEN_TAG));
        let output =
            json!({"summary": "done", "verification": fix_verification_fixture()}).to_string();
        validate_subagent_response(Some(&schema), &output).unwrap();
        let verification = extract_task_verification(&output, Some(&schema)).unwrap();
        assert_eq!(
            verification.assessment().schema_status,
            VerificationSchemaStatus::Valid
        );
        validate_subagent_response(Some(&schema), &output).unwrap();
        let quoted_tag = json!({
            "summary": "Fixed the <audit_report> parser",
            "verification": fix_verification_fixture(),
        })
        .to_string();
        validate_subagent_response(Some(&schema), &quoted_tag).unwrap();
        assert!(extract_task_verification(&quoted_tag, Some(&schema)).is_some());
        assert!(
            validate_subagent_response(
                Some(&schema),
                &format!("{output}\n{OPEN_TAG}{{}}{CLOSE_TAG}")
            )
            .is_err()
        );
        for nested in [
            json!("ordinary output"),
            json!({"schema": "other.schema", "version": 1}),
        ] {
            let output = json!({"summary": "done", "verification": nested}).to_string();
            assert!(extract_task_verification(&output, Some(&schema)).is_none());
        }
        let handoff = r#"<subagent_handoff>{"kind":"user_input_required","reason":"need a fixture"}</subagent_handoff>"#;
        assert!(parse_subagent_handoff(handoff).unwrap().is_some());
        assert!(extract_task_verification(handoff, Some(&schema)).is_none());
        assert!(
            extract_task_verification(&format!("{handoff}\n{OPEN_TAG}{{}}{CLOSE_TAG}"), None)
                .is_none()
        );
        let audit = format!("<audit_report>{output}</audit_report>\n{OPEN_TAG}{{}}{CLOSE_TAG}");
        assert!(extract_task_verification(&audit, None).is_none());
        assert!(
            extract_task_verification(
                &format!("Example:\n```\n{OPEN_TAG}{{}}{CLOSE_TAG}\n```"),
                None
            )
            .is_none()
        );
        let quoted_audit = format!(
            "Example:\n```\n<audit_report>{{}}</audit_report>\n```\n{OPEN_TAG}{}{CLOSE_TAG}",
            fix_verification_fixture(),
        );
        assert!(extract_task_verification(&quoted_audit, None).is_some());
    }

    #[test]
    fn task_verification_malformed_or_oversized_block_preserves_completed_output() {
        let mut oversized = fix_verification_fixture();
        oversized["remaining_uncertainty"] = json!(["界".repeat(TASK_VERIFICATION_MAX_BYTES)]);
        let mut too_many = fix_verification_fixture();
        too_many["checks"] = json!(vec![
            too_many["checks"][0].clone();
            TASK_VERIFICATION_MAX_CHECKS + 1
        ]);
        let mut unsupported = fix_verification_fixture();
        unsupported["version"] = json!(2);
        unsupported["checks"] = json!({"future": "shape"});
        let mut malformed = fix_verification_fixture();
        malformed["checks"] = json!(false);
        for (block, expected) in [
            (
                format!("{OPEN_TAG}{{bad JSON}}{CLOSE_TAG}"),
                VerificationSchemaStatus::Invalid,
            ),
            (format!("{OPEN_TAG}{{}}"), VerificationSchemaStatus::Invalid),
            (
                format!("{OPEN_TAG}null{CLOSE_TAG}"),
                VerificationSchemaStatus::Invalid,
            ),
            (
                format!("{OPEN_TAG}{malformed}{CLOSE_TAG}"),
                VerificationSchemaStatus::Invalid,
            ),
            (
                format!("{OPEN_TAG}{oversized}{CLOSE_TAG}"),
                VerificationSchemaStatus::Invalid,
            ),
            (
                format!("{OPEN_TAG}{too_many}{CLOSE_TAG}"),
                VerificationSchemaStatus::Invalid,
            ),
            (
                format!("{OPEN_TAG}{unsupported}{CLOSE_TAG}"),
                VerificationSchemaStatus::Unsupported,
            ),
            (
                format!("{OPEN_TAG}{{}}{CLOSE_TAG}\n{OPEN_TAG}{{}}{CLOSE_TAG}"),
                VerificationSchemaStatus::Invalid,
            ),
        ] {
            let output = format!("  completed body  \n{block}\n  ");
            let metadata = extract_task_verification(&output, None).unwrap();
            let result: StoredTaskResult = serde_json::from_value(json!({
                "status": "completed", "output": output, "error": null, "verification": metadata,
            }))
            .unwrap();
            assert_eq!(result.status, "completed");
            assert_eq!(result.output, output);
            assert_eq!(
                result
                    .verification
                    .as_ref()
                    .unwrap()
                    .assessment()
                    .schema_status,
                expected
            );
            assert!(
                result
                    .verification
                    .as_ref()
                    .unwrap()
                    .inline_assessment()
                    .len()
                    <= INLINE_ASSESSMENT_BYTES
            );
            let report = TaskReport::for_task(TaskReportTask {
                task_id: Some("task-completed".to_string()),
                status: result.status,
                output: Some(result.output),
                verification: result.verification,
                ..TaskReportTask::default()
            });
            let restored: TaskReport =
                serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
            assert!(restored.supported());
            assert_eq!(restored, report);
            assert_eq!(restored.tasks[0].output.as_deref(), Some(output.as_str()));
            assert_eq!(
                restored.tasks[0]
                    .verification
                    .as_ref()
                    .unwrap()
                    .assessment()
                    .schema_status,
                expected
            );
            assert!(
                serde_json::to_vec(&restored.tasks[0].verification)
                    .unwrap()
                    .len()
                    <= TASK_VERIFICATION_MAX_BYTES
            );
        }
        let legacy: StoredTaskResult =
            serde_json::from_value(json!({"status": "completed", "output": "legacy"})).unwrap();
        assert!(legacy.verification.is_none());
    }
}
