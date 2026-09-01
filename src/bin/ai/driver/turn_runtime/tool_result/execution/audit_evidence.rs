//! Audit-only final-response gate: require structured, locally grounded evidence
//! before a review observation may be published as a verified finding.

use super::*;
use serde::Deserialize;

pub(in crate::ai::driver::turn_runtime) const AUDIT_EVIDENCE_RETRY_MARKER: &str =
    "[audit-evidence-retry]";
pub(in crate::ai::driver::turn_runtime) const AUDIT_EVIDENCE_UNVERIFIED_NOTE: &str = "runtime:audit_evidence_withheld\nOne or more audit findings were withheld because their structured evidence could not be verified from current-turn reads.";
pub(in crate::ai::driver::turn_runtime) const AUDIT_EVIDENCE_WARNING: &str = "[Runtime warning] Audit findings without a complete, current-turn evidence chain were withheld rather than published as verified.";

const AUDIT_REPORT_OPEN: &str = "<audit_report>";
const AUDIT_REPORT_CLOSE: &str = "</audit_report>";
const MAX_AUDIT_REPORT_BYTES: usize = 64 * 1024;
const MAX_AUDIT_FINDINGS: usize = 32;
const MAX_AUDIT_EVIDENCE_PER_KIND: usize = 16;
const MAX_AUDIT_EVIDENCE_LINES: u64 = 256;
const MAX_AUDIT_TEXT_BYTES: usize = 8 * 1024;
const MAX_AUDIT_LIST_ITEMS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum AuditEvidenceGateAction {
    Allow,
    Reopen,
    Warn,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditReport {
    #[serde(default)]
    findings: Vec<AuditFinding>,
    #[serde(default)]
    open_questions: Vec<String>,
    #[serde(default)]
    coverage_gaps: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditFinding {
    severity: String,
    title: String,
    claim: String,
    trigger: String,
    impact: String,
    source_evidence: Vec<AuditEvidence>,
    semantic_evidence: Vec<AuditEvidence>,
    falsification_checks: Vec<AuditEvidence>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditEvidence {
    path: String,
    start_line: u64,
    #[serde(default)]
    end_line: Option<u64>,
    explanation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EvidenceLocation {
    path: PathBuf,
    start_line: u64,
    end_line: u64,
}

#[derive(Debug, Clone)]
struct ObservedRead {
    path: PathBuf,
    shown_lines: HashSet<u64>,
    message_index: usize,
}

/// Only the dedicated review agents are constrained by this protocol. Normal
/// conversations retain their existing final-response behavior.
pub(in crate::ai::driver::turn_runtime) fn is_evidence_gated_audit_agent(agent_name: &str) -> bool {
    matches!(agent_name, "audit" | "audit-fast")
}

#[allow(clippy::too_many_arguments)]
pub(in crate::ai::driver::turn_runtime) fn audit_evidence_gate_action(
    agent_name: &str,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    final_text: &mut String,
    effective_cwd: Option<&Path>,
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> AuditEvidenceGateAction {
    if !is_evidence_gated_audit_agent(agent_name) {
        return AuditEvidenceGateAction::Allow;
    }

    let report = match parse_audit_report(final_text) {
        Ok(report) => report,
        Err(reason) => {
            return audit_evidence_retry_or_withhold(
                messages,
                final_text,
                reason,
                force_final_response,
                iteration,
                max_iterations,
                None,
            );
        }
    };
    let observed_reads = observed_successful_reads(turn_messages, effective_cwd);
    let last_mutation = last_successful_direct_mutation(turn_messages);
    let valid_findings = report
        .findings
        .iter()
        .filter(|finding| {
            finding_has_complete_evidence(finding, &observed_reads, last_mutation, effective_cwd)
        })
        .cloned()
        .collect::<Vec<_>>();
    if valid_findings.len() == report.findings.len() {
        *final_text = render_audit_report(&report, &valid_findings, false);
        return AuditEvidenceGateAction::Allow;
    }

    audit_evidence_retry_or_withhold(
        messages,
        final_text,
        "one or more findings lack a complete evidence chain",
        force_final_response,
        iteration,
        max_iterations,
        Some(render_audit_report(&report, &valid_findings, true)),
    )
}

#[allow(clippy::too_many_arguments)]
fn audit_evidence_retry_or_withhold(
    messages: &mut Vec<Message>,
    final_text: &mut String,
    reason: &str,
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
    withheld_report: Option<String>,
) -> AuditEvidenceGateAction {
    let already_retried = current_turn_has_internal_marker(messages, AUDIT_EVIDENCE_RETRY_MARKER);
    if already_retried || force_final_response || iteration >= max_iterations {
        *final_text = withheld_report.unwrap_or_else(|| {
            format!(
                "## Verified findings\n\nNo verified findings could be published because the audit response did not provide a valid structured evidence report.\n\n{AUDIT_EVIDENCE_WARNING}"
            )
        });
        return AuditEvidenceGateAction::Warn;
    }

    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(format!(
            "{AUDIT_EVIDENCE_RETRY_MARKER}\n\
             This is not a final answer. The audit draft is not publishable because {reason}.\n\
             Return exactly one `<audit_report>{{...}}</audit_report>` JSON payload and no text outside it.\n\
             Every verified finding needs non-empty `source_evidence`, `semantic_evidence`, and `falsification_checks` arrays. Each evidence item needs `path`, `start_line`, optional `end_line`, and `explanation`; every cited range must have been successfully read in this user turn after the last direct mutation.\n\
             Findings without that complete chain must be omitted or moved to `open_questions` / `coverage_gaps`."
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    AuditEvidenceGateAction::Reopen
}

fn parse_audit_report(text: &str) -> Result<AuditReport, &'static str> {
    if text.len() > MAX_AUDIT_REPORT_BYTES {
        return Err("the audit report exceeds the protocol size limit");
    }
    let trimmed = text.trim();
    let Some(body) = trimmed
        .strip_prefix(AUDIT_REPORT_OPEN)
        .and_then(|text| text.strip_suffix(AUDIT_REPORT_CLOSE))
    else {
        return Err("the audit report protocol is missing or has text outside the report tag");
    };
    let report: AuditReport = serde_json::from_str(body.trim())
        .map_err(|_| "the audit report is not valid protocol JSON")?;
    if report.findings.len() > MAX_AUDIT_FINDINGS
        || report.open_questions.len() > MAX_AUDIT_LIST_ITEMS
        || report.coverage_gaps.len() > MAX_AUDIT_LIST_ITEMS
        || report
            .open_questions
            .iter()
            .chain(report.coverage_gaps.iter())
            .any(|item| !bounded_text(item))
    {
        return Err("the audit report exceeds a protocol size limit");
    }
    Ok(report)
}

fn finding_has_complete_evidence(
    finding: &AuditFinding,
    observed_reads: &[ObservedRead],
    last_mutation: Option<usize>,
    effective_cwd: Option<&Path>,
) -> bool {
    if !matches!(finding.severity.as_str(), "P0" | "P1" | "P2" | "P3")
        || !bounded_text(&finding.title)
        || !bounded_text(&finding.claim)
        || !bounded_text(&finding.trigger)
        || !bounded_text(&finding.impact)
        || finding.source_evidence.is_empty()
        || finding.semantic_evidence.is_empty()
        || finding.falsification_checks.is_empty()
        || finding.source_evidence.len() > MAX_AUDIT_EVIDENCE_PER_KIND
        || finding.semantic_evidence.len() > MAX_AUDIT_EVIDENCE_PER_KIND
        || finding.falsification_checks.len() > MAX_AUDIT_EVIDENCE_PER_KIND
    {
        return false;
    }

    let mut locations = HashSet::new();
    finding
        .source_evidence
        .iter()
        .chain(finding.semantic_evidence.iter())
        .chain(finding.falsification_checks.iter())
        .all(|evidence| {
            grounded_evidence_location(evidence, observed_reads, last_mutation, effective_cwd)
                .is_some_and(|location| locations.insert(location))
        })
}

fn grounded_evidence_location(
    evidence: &AuditEvidence,
    observed_reads: &[ObservedRead],
    last_mutation: Option<usize>,
    effective_cwd: Option<&Path>,
) -> Option<EvidenceLocation> {
    if !bounded_text(&evidence.explanation)
        || evidence.path.trim().is_empty()
        || evidence.start_line == 0
    {
        return None;
    }
    let end_line = evidence.end_line.unwrap_or(evidence.start_line);
    if end_line < evidence.start_line
        || end_line.saturating_sub(evidence.start_line) >= MAX_AUDIT_EVIDENCE_LINES
    {
        return None;
    }
    let path = resolve_audit_path(&evidence.path, effective_cwd)?;
    let path = std::fs::canonicalize(path).ok()?;
    if citation_file_contains_line(&path, end_line) != Some(true) {
        return None;
    }
    let location = EvidenceLocation {
        path,
        start_line: evidence.start_line,
        end_line,
    };
    observed_reads
        .iter()
        .any(|read| {
            last_mutation.is_none_or(|mutation| read.message_index > mutation)
                && read.path == location.path
                && (location.start_line..=location.end_line)
                    .all(|line| read.shown_lines.contains(&line))
        })
        .then_some(location)
}

fn observed_successful_reads(
    turn_messages: &[Message],
    effective_cwd: Option<&Path>,
) -> Vec<ObservedRead> {
    let mut calls_by_id = HashMap::new();
    let mut reads = Vec::new();
    for (message_index, message) in turn_messages.iter().enumerate() {
        if let Some(tool_calls) = &message.tool_calls {
            for call in tool_calls {
                calls_by_id.insert(call.id.clone(), call);
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
        if call.function.name != "read_file" {
            continue;
        }
        let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments) else {
            continue;
        };
        let Some(path) = args
            .get("file_path")
            .or_else(|| args.get("path"))
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        if args
            .get("use_line_numbers")
            .and_then(serde_json::Value::as_bool)
            == Some(false)
        {
            continue;
        }
        let start_line = args
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1);
        let limit = args
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1_000);
        let Some(end_line) = start_line.checked_add(limit.saturating_sub(1)) else {
            continue;
        };
        let Some(path) = resolve_audit_path(path, effective_cwd)
            .and_then(|path| std::fs::canonicalize(path).ok())
        else {
            continue;
        };
        if is_evidence_snapshot_path(&path) {
            continue;
        }
        let Some(content) = message.content.as_str() else {
            continue;
        };
        let shown_lines = observed_rendered_line_numbers(content, start_line, end_line);
        if shown_lines.is_empty() {
            continue;
        }
        reads.push(ObservedRead {
            path,
            shown_lines,
            message_index,
        });
    }
    reads
}

fn observed_rendered_line_numbers(
    content: &str,
    requested_start: u64,
    requested_end: u64,
) -> HashSet<u64> {
    content
        .lines()
        .filter_map(|line| {
            let (prefix, _) = line.split_once('\t')?;
            let line_number = prefix.trim().parse::<u64>().ok()?;
            (requested_start <= line_number && line_number <= requested_end).then_some(line_number)
        })
        .collect()
}

fn is_evidence_snapshot_path(path: &Path) -> bool {
    crate::ai::tools::storage::file_store::is_read_file_overflow_artifact(path)
        || (path.file_name().and_then(|name| name.to_str()) == Some("overflow-history.md")
            && path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".assets")))
}

fn last_successful_direct_mutation(turn_messages: &[Message]) -> Option<usize> {
    let mut calls_by_id = HashMap::new();
    let mut last_mutation = None;
    for (message_index, message) in turn_messages.iter().enumerate() {
        if let Some(tool_calls) = &message.tool_calls {
            for call in tool_calls {
                calls_by_id.insert(call.id.clone(), call);
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
        if tool_call_is_successful_mutation_candidate(call) {
            last_mutation = Some(message_index);
        }
    }
    last_mutation
}

fn resolve_audit_path(path: &str, effective_cwd: Option<&Path>) -> Option<PathBuf> {
    let path = Path::new(path);
    if path.is_absolute() {
        Some(path.to_path_buf())
    } else {
        effective_cwd.map(|cwd| cwd.join(path))
    }
}

fn bounded_text(text: &str) -> bool {
    !text.trim().is_empty()
        && text.len() <= MAX_AUDIT_TEXT_BYTES
        && !PATH_LINE_CITATION_RE.is_match(text)
}

fn render_audit_report(
    report: &AuditReport,
    valid_findings: &[AuditFinding],
    withheld_findings: bool,
) -> String {
    let mut out = String::from("## Verified findings\n\n");
    if valid_findings.is_empty() {
        out.push_str("No verified findings.\n");
    } else {
        for (index, finding) in valid_findings.iter().enumerate() {
            out.push_str(&format!(
                "### {} — {}\n\n**Claim.** {}\n\n**Trigger.** {}\n\n**Impact.** {}\n\n",
                finding.severity, finding.title, finding.claim, finding.trigger, finding.impact
            ));
            render_evidence_list(&mut out, "Source evidence", &finding.source_evidence);
            render_evidence_list(&mut out, "Semantic evidence", &finding.semantic_evidence);
            render_evidence_list(
                &mut out,
                "Falsification checks",
                &finding.falsification_checks,
            );
            if index + 1 < valid_findings.len() {
                out.push('\n');
            }
        }
    }
    if !report.open_questions.is_empty() {
        out.push_str("\n## Open questions\n\n");
        render_text_list(&mut out, &report.open_questions);
    }
    if !report.coverage_gaps.is_empty() {
        out.push_str("\n## Coverage gaps\n\n");
        render_text_list(&mut out, &report.coverage_gaps);
    }
    if withheld_findings {
        out.push_str(&format!("\n{AUDIT_EVIDENCE_WARNING}\n"));
    }
    out
}

fn render_evidence_list(out: &mut String, heading: &str, evidence: &[AuditEvidence]) {
    out.push_str(&format!("**{heading}.**\n"));
    for item in evidence {
        let end_line = item.end_line.unwrap_or(item.start_line);
        let range = (end_line != item.start_line)
            .then(|| format!("-{end_line}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "- `{}:{}{}` — {}\n",
            item.path, item.start_line, range, item.explanation
        ));
    }
    out.push('\n');
}

fn render_text_list(out: &mut String, items: &[String]) {
    for item in items {
        out.push_str("- ");
        out.push_str(item);
        out.push('\n');
    }
}
