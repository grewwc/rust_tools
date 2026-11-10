use std::path::Path;
use std::time::Duration;

/// `/audit` is an explicitly requested deep audit; it allows a longer foreground wait than an
/// ordinary delegated subagent.
pub(crate) const AUDIT_SUBAGENT_HARD_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// Reserve wrap-up time so the audit does not lose collected conclusions at the hard timeout.
/// Two minutes was too short: a long model request could not be interrupted before the hard
/// timeout aborted it, losing up to fifteen minutes of work. Five minutes lets the wrap-up signal
/// interrupt a request mid-flight and force a final conclusion.
pub(crate) const AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME: Duration = Duration::from_secs(5 * 60);

/// Hard timeout for `/audit --fast`: the current session model with high reasoning and a bounded
/// step count normally finishes a quick audit in a few minutes; eight minutes caps foreground use.
pub(crate) const FAST_AUDIT_SUBAGENT_HARD_TIMEOUT: Duration = Duration::from_secs(8 * 60);
/// Wrap-up reserve for a fast audit: three minutes is enough to interrupt a request and produce a
/// final conclusion while preserving the foreground runner's wrap-up guarantee.
pub(crate) const FAST_AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME: Duration = Duration::from_secs(3 * 60);

const SUBAGENT_FINAL_ANSWER_MARKER: &str = "[Subagent final answer]\n";
pub(crate) const AUDIT_PROGRESS_PROTOCOL: &str = include_str!("prompts/audit_progress_protocol.md");
/// Builtin skill whose procedure the slash-command lead guidance follows.
const AUDIT_PROCEDURE_SKILL: &str = "audit_own_changes";

/// Private opt-in lead procedure and closure rules; never injected into default audits.
const AUDIT_LEAD_CONTINUATION: &str = include_str!("prompts/audit_deep_procedure.md");
const AUDIT_DEEP_CHILD: &str = include_str!("prompts/audit_deep_child.md");

/// Parse `/audit` before the model loop; enhanced lead orchestration requires `--deep`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AuditCommand {
    Run {
        instruction: String,
        fast: bool,
        deep: bool,
    },
    /// Bare `/audit` / `/audit -f` (no instruction): do not start a subagent yet. Let the current
    /// turn continue so the lead agent can build the audit scope before starting the audit subagent.
    SelfGenerated { fast: bool, deep: bool },
}

pub(crate) fn parse_audit_command(input: &str) -> Option<AuditCommand> {
    let trimmed = input.trim();
    let normalized = trimmed
        .strip_prefix('/')
        .or_else(|| trimmed.strip_prefix(':'))?;
    let remainder = normalized.strip_prefix("audit")?;

    if remainder
        .chars()
        .next()
        .is_some_and(|first| !first.is_whitespace())
    {
        return None;
    }

    let instruction = remainder.trim();
    let (deep, instruction) = instruction
        .strip_prefix("--deep")
        .filter(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace))
        .map_or((false, instruction), |rest| (true, rest.trim_start()));
    if instruction.is_empty() {
        Some(AuditCommand::SelfGenerated { fast: false, deep })
    } else if let Some(rest) = strip_fast_flag(instruction) {
        if rest.is_empty() {
            // Bare `/audit --fast` lets the lead generate the fast-audit scope.
            Some(AuditCommand::SelfGenerated { fast: true, deep })
        } else {
            Some(AuditCommand::Run {
                instruction: rest.to_string(),
                fast: true,
                deep,
            })
        }
    } else {
        Some(AuditCommand::Run {
            instruction: instruction.to_string(),
            fast: false,
            deep,
        })
    }
}

/// Recognize a leading `--fast` / `-f` flag, returning None for other input.
fn strip_fast_flag(instruction: &str) -> Option<&str> {
    let rest = instruction
        .strip_prefix("--fast")
        .or_else(|| instruction.strip_prefix("-f"))?;
    // Require a complete flag token; prefixes such as `--fastly` remain intact.
    if instruction.len() != rest.len() && rest.chars().next().is_some_and(|c| !c.is_whitespace()) {
        return None;
    }
    Some(rest.trim_start())
}

/// Guidance injected for `/audit` without an instruction: the audit scope is left to the lead
/// agent (only it knows what changed in this session), which builds the self-contained audit
/// prompt and starts the `audit` / `audit-fast` subagent through `task_spawn`.
///
/// The procedure is not restated here: it is quoted verbatim from the builtin
/// `audit_own_changes` skill, so the command path and the skill text cannot drift apart.
/// Written to the `messages` projection only (like loop notes), never to persisted turn messages.
pub(crate) fn inject_self_generated_audit_note(
    messages: &mut Vec<crate::ai::history::Message>,
    fast: bool,
) {
    let mode = if fast { "/audit -f (fast)" } else { "/audit" };
    let agent = if fast { "audit-fast" } else { "audit" };
    let fast_flag = if fast { " --fast" } else { "" };
    let procedure = match crate::ai::skills::builtin_skill_body(AUDIT_PROCEDURE_SKILL) {
        Some(body) => format!(
            "Follow the `{AUDIT_PROCEDURE_SKILL}` skill quoted below verbatim:\n\
             ----- begin `{AUDIT_PROCEDURE_SKILL}` -----\n\
             {body}\n\
             ----- end `{AUDIT_PROCEDURE_SKILL}` -----"
        ),
        None => format!(
            "The builtin `{AUDIT_PROCEDURE_SKILL}` skill text is unavailable; activate that skill \
             (`enable_tools` -> `list_skills` -> `activate_skill`) and follow it, or ask the user \
             to re-run `/audit <audit_prompt>` with the audit scope spelled out."
        ),
    };
    let note = format!(
        "[audit-self-generated] The user invoked {mode} without an audit instruction: the audit \
         scope is left to you, the lead agent, because only you know the details of what you \
         changed in this session. This explicit invocation, not any skill's activation criteria, \
         decides that an audit runs now, and it starts agent=\"{agent}\" instead of letting the \
         skill pick the tier.\n\
         \n\
         If you end this turn without having started the `{agent}` subagent through `task_spawn`, \
         the runtime forces a foreground `/audit{fast_flag}` with this turn's output as the audit \
         instruction, so the audit always runs in a subagent.\n\
         \n\
         {procedure}"
    );
    messages.push(crate::ai::history::Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Legacy bare-command detection deliberately matches the recorded audit spawn call.
pub(crate) fn turn_spawned_audit_subagent(turn_messages: &[crate::ai::history::Message]) -> bool {
    turn_messages.iter().any(|message| {
        message.tool_calls.as_ref().is_some_and(|tool_calls| {
            tool_calls.iter().any(|tool_call| {
                let name = &tool_call.function.name;
                let is_audit = |value: &serde_json::Value| {
                    value
                        .get("agent")
                        .and_then(|agent| agent.as_str())
                        .is_some_and(|agent| agent == "audit" || agent == "audit-fast")
                };
                serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
                    .ok()
                    .is_some_and(|args| match name.as_str() {
                        "task_spawn" => args
                            .get("tasks")
                            .and_then(|tasks| tasks.as_array())
                            .is_some_and(|tasks| tasks.iter().any(|task| is_audit(task))),
                        _ => false,
                    })
            })
        })
    })
}

/// Only a matched successful `task_spawn` result proves an audit child started.
/// Preflight errors, missing results, and failed children do not suppress the bare-command fallback.
pub(crate) fn turn_successfully_spawned_audit_subagent(
    turn_messages: &[crate::ai::history::Message],
) -> bool {
    turn_messages.iter().any(|message| {
        message.tool_calls.as_ref().is_some_and(|tool_calls| {
            tool_calls.iter().any(|tool_call| {
                tool_call.function.name == "task_spawn"
                    && turn_messages.iter().any(|result| {
                        result.role == "tool"
                            && result.tool_call_id.as_deref() == Some(tool_call.id.as_str())
                            && result
                                .content
                                .as_str()
                                .and_then(|text| {
                                    serde_json::from_str::<serde_json::Value>(text).ok()
                                })
                                .and_then(|value| {
                                    value.get("tasks").and_then(|v| v.as_array()).cloned()
                                })
                                .is_some_and(|tasks| {
                                    tasks.iter().any(|task| {
                                        task.get("status").and_then(|v| v.as_str())
                                            == Some("spawned")
                                            && task
                                                .get("agent")
                                                .and_then(|v| v.as_str())
                                                .is_some_and(|agent| {
                                                    matches!(agent, "audit" | "audit-fast")
                                                })
                                            && task
                                                .get("task_id")
                                                .and_then(|v| v.as_str())
                                                .is_some_and(|id| !id.is_empty())
                                            && task
                                                .get("pid")
                                                .and_then(|v| v.as_u64())
                                                .is_some_and(|pid| pid > 0)
                                    })
                                })
                    })
            })
        })
    })
}

/// The full `/audit` payload is persisted as the main agent's evidence; the terminal shows only
/// the subagent's final conclusion so tool-call records and parent-facing control notes never leak
/// to the user.
pub(crate) fn terminal_audit_result(payload: &str) -> String {
    // Preferred path: when the subagent produced tool evidence, finalize separates
    // the evidence from the final answer with a marker.
    if let Some((_, answer)) = payload.split_once(SUBAGENT_FINAL_ANSWER_MARKER) {
        let answer = strip_parent_reminder(answer);
        if !answer.is_empty() {
            return answer.to_string();
        }
    }

    // Timeout/failure path: the recovery evidence block collected before interruption
    // (`SUBAGENT_TIMEOUT_RECOVERY_V1` from the foreground runner) goes to the lead agent only as
    // part of the full payload. The terminal shows no excerpt — the user only needs to know the
    // audit did not finish, with the reason in the compact `Error:` line.
    if let Some(error) = payload
        .lines()
        .find_map(|line| line.trim().strip_prefix("Error: "))
    {
        return format!("[audit] {error}");
    }

    // Fallback: a subagent that answers inline with zero tool calls produces no marker (see
    // finalize.rs `format_subagent_result_for_parent`, which returns the final text unchanged when
    // tool evidence is empty). Inline answers are common because the audit prompt already injects
    // the mutation log / git diff. Strip the foreground-subagent payload header and parent reminder
    // tail, then show the actual conclusion instead of reporting "no final answer".
    let stripped = strip_subagent_payload_header(payload);
    let body = strip_parent_reminder(&stripped);
    if !body.is_empty() && body != SUBAGENT_NO_FINAL_TEXT_SENTINEL {
        return body.to_string();
    }

    "[audit] Audit subagent finished without a final answer. Its full result was delivered to the main agent."
        .to_string()
}

/// Placeholder written by a foreground subagent when it produced no final text
/// (see `sync_task::format_subagent_output`). A match means there really is no
/// conclusion, so fall back to the generic notice instead of showing the placeholder.
const SUBAGENT_NO_FINAL_TEXT_SENTINEL: &str = "(subagent did not produce any final assistant text)";

/// Full audit payload for the lead-agent fix loop: unlike the terminal view
/// (which hides timeout-recovery evidence), the fixer needs every collected
/// fact. Only the parent-reminder suffix is stripped.
pub(crate) fn audit_report_for_fix(payload: &str) -> String {
    strip_parent_reminder(payload).trim().to_string()
}

/// Strip the trailing parent reminder (`finalize` and the foreground runner both append
/// `SUBAGENT_PARENT_SUMMARY_REMINDER`).
fn strip_parent_reminder(text: &str) -> &str {
    text.trim()
        .strip_suffix(crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER)
        .unwrap_or_else(|| text.trim())
        .trim()
}

/// Strip the foreground-subagent payload header: the `[task_id=…]` line, the
/// `[Task: …] STATUS after …s` line, and the selection explanation's
/// `agent_reason=` / `model_reason=` lines (see `sync_task::format_subagent_output`
/// and `task_tools::build_selection_explanation`). Only consecutive known metadata
/// lines at the start are skipped; audit conclusions (prose / markdown) do not
/// start with those prefixes, so they are never swallowed. If the header format
/// changes later, the worst case is a few extra displayed header lines.
fn strip_subagent_payload_header(payload: &str) -> String {
    const HEADER_PREFIXES: &[&str] = &["[task_id=", "[Task: ", "agent_reason=", "model_reason="];
    let mut lines = payload.lines().peekable();
    while let Some(line) = lines.peek() {
        if HEADER_PREFIXES
            .iter()
            .any(|prefix| line.trim_start().starts_with(prefix))
        {
            lines.next();
        } else {
            break;
        }
    }
    lines.collect::<Vec<_>>().join("\n")
}

/// Build the legacy session-first foreground audit prompt.
pub(crate) fn build_audit_prompt(instruction: &str) -> String {
    let changes = capture_current_changes_context();
    compose_audit_prompt(instruction, &changes)
}

/// Build the opt-in child prompt, including union inventory even after planning failure.
fn build_audit_deep_prompt(instruction: &str) -> String {
    compose_audit_deep_prompt(instruction, &capture_deep_changes_context())
}

fn compose_audit_deep_prompt(instruction: &str, changes_context: &str) -> String {
    format!(
        "{instruction}\n\n{AUDIT_DEEP_CHILD}\n\n{AUDIT_PROGRESS_PROTOCOL}\n\n\
         ===== Observed changes: session mutations and Git inventory =====\n\
         {changes_context}\n\
         ===== End of file changes ====="
    )
}

fn capture_deep_changes_context() -> String {
    let inventory = crate::ai::tools::storage::changes::collect_current_changes(true);
    format!(
        "{}\n{}",
        inventory.format_inventory(),
        inventory.format_snippets()
    )
}

/// Let the lead partition the snapshot; the driver retains foreground timeout/recovery authority.
pub(crate) fn build_audit_lead_request(
    instruction: &str,
    fast: bool,
    current_model: &str,
) -> String {
    let changes = capture_deep_changes_context();
    compose_audit_lead_request(instruction, fast, current_model, &changes)
}

/// Keep lead guidance independent of the workspace content it quotes.
fn compose_audit_lead_request(
    instruction: &str,
    fast: bool,
    current_model: &str,
    changes_context: &str,
) -> String {
    let single_scope_task = if fast {
        serde_json::json!({
            "agent": "audit-fast",
            "model": current_model,
            "reasoning_effort": "high",
        })
    } else {
        serde_json::json!({"agent": "audit"})
    };
    format!(
        "The user explicitly requested `/audit --deep`. You are the lead: follow the \
         private deep-audit procedure and prepare the requested first round for subagents. \
         Before building their prompts, resolve request ownership against R0 (the union \
         inventory below); keep each item `covered` with a cited read range, `gap`, or `N.A.` \
         with reason. Unknown ownership stays a gap. Each child needs a self-contained scope, \
         changed contracts, verification already run, its own read list and named gaps, plus \
         the supplied audit instruction and audit progress protocol.\n\n\
         Prepare a single scope for a small change; its child settings are {single_scope_task}. \
         For a large change set (roughly \
         150+ changed lines) or independent subsystems, partition by invariant/consumer \
         boundary into 2-4 scoped audits, each with `agent: \"audit\"`. \
         All children for the same snapshot form one round. \
         Keep default `inherit`, never `\"all\"`, and do not pass `response_schema`. Tell \
         children to stay on scope, check relevant baselines/consumers, preserve the \
         `<audit_report>` protocol, avoid repeating checks already run, and report \
         `coverage_gaps` instead of widening scope. Return exactly one `<audit_plan>` envelope \
         containing JSON with a `scopes` array of 1-4 nonempty instruction strings, each \
         self-contained with its scope and read list. Do not start `task_spawn` children or \
         modify source while planning. The driver launches all scopes in parallel through \
         the foreground runner, preserving audit-specific hard timeout, wrap-up, durable \
         delivery, and `SUBAGENT_TIMEOUT_RECOVERY_V1` evidence. An invalid/missing plan or \
         failed lead preparation falls back to the original foreground audit request. \
         After the driver returns the reports, merge and deduplicate findings, integrate \
         every delivered task with `task_integrate` using its full `task_id`, verify evidence, and apply authorized \
         fixes. Any second-round audit must use the same foreground runner; provide a narrow \
         `<audit_plan>` only with `triggers` (1-3 objects containing `trigger: \"T1\"|\"T2\"|\"T3\"` \
         and concrete `evidence`) and a nonempty `changed_axis`; combine applicable triggers \
         without repeating one.\n\n\
         {AUDIT_LEAD_CONTINUATION}\n\n\
         Original audit request: {instruction}\n\n\
         ===== Observed changes: session mutations and Git inventory =====\n\
         {changes}\n\
         ===== End of file changes =====\n\n\
         Audit progress protocol for each child:\n{AUDIT_PROGRESS_PROTOCOL}",
        instruction = instruction.trim(),
        changes = changes_context,
    )
}

/// The lead's scope plan is separate from the auditor's unchanged report protocol.
pub(crate) fn parse_audit_scope_plan(text: &str) -> Result<Vec<String>, String> {
    let value = parse_audit_plan_value(text)?;
    audit_plan_scopes(&value)
}

pub(crate) fn parse_audit_followup_scope_plan(text: &str) -> Result<Vec<String>, String> {
    let value = parse_audit_plan_value(text)?;
    let triggers = value
        .get("triggers")
        .and_then(|v| v.as_array())
        .filter(|triggers| (1..=3).contains(&triggers.len()))
        .ok_or("round two requires 1-3 recorded triggers")?;
    let mut seen = Vec::new();
    for trigger in triggers {
        let name = trigger
            .get("trigger")
            .and_then(|v| v.as_str())
            .filter(|name| matches!(*name, "T1" | "T2" | "T3"))
            .ok_or("round two trigger must be T1, T2, or T3")?;
        if seen.contains(&name) {
            return Err("round two cannot repeat a trigger".into());
        }
        seen.push(name);
        if !trigger
            .get("evidence")
            .and_then(|v| v.as_str())
            .is_some_and(|evidence| !evidence.trim().is_empty())
        {
            return Err("round two requires concrete trigger evidence".into());
        }
    }
    if !value
        .get("changed_axis")
        .and_then(|v| v.as_str())
        .is_some_and(|axis| !axis.trim().is_empty())
    {
        return Err("round two requires a changed axis".into());
    }
    audit_plan_scopes(&value)
}

fn parse_audit_plan_value(text: &str) -> Result<serde_json::Value, String> {
    const OPEN: &str = "<audit_plan>";
    const CLOSE: &str = "</audit_plan>";
    if text.matches(OPEN).count() != 1 || text.matches(CLOSE).count() != 1 {
        return Err("expected one <audit_plan> envelope".into());
    }
    let body = text.split_once(OPEN).unwrap().1;
    let body = body.split_once(CLOSE).ok_or("unclosed audit plan")?.0;
    serde_json::from_str(body).map_err(|error| format!("invalid audit plan JSON: {error}"))
}

fn audit_plan_scopes(value: &serde_json::Value) -> Result<Vec<String>, String> {
    let scopes = value
        .get("scopes")
        .and_then(|value| value.as_array())
        .ok_or("audit plan requires a scopes array")?;
    if !(1..=4).contains(&scopes.len()) {
        return Err("audit plan requires 1-4 scopes".into());
    }
    scopes
        .iter()
        .map(|scope| {
            scope
                .as_str()
                .filter(|scope| !scope.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| "audit scope must be a nonempty instruction string".into())
        })
        .collect()
}

pub(crate) struct ForegroundAuditInvocation {
    pub(crate) tool_call_id: String,
    pub(crate) args: serde_json::Value,
    pub(crate) hard_timeout: Duration,
    pub(crate) wrap_up_lead_time: Option<Duration>,
}

/// Build every scope against the same inventory snapshot, retaining the existing timeout policy.
pub(crate) fn foreground_audit_invocations(
    instruction: &str,
    scopes: &[String],
    fast: bool,
    current_model: &str,
) -> Vec<ForegroundAuditInvocation> {
    let changes = capture_deep_changes_context();
    scopes
        .iter()
        .enumerate()
        .map(|(index, scope)| {
            let scoped_instruction = format!(
                "Original audit request: {instruction}\n\nAudit scope {} of {}: {scope}\n\n\
             Stay on this invariant/consumer scope, including its baseline and consumers; \
             report coverage_gaps instead of widening it. Do not repeat supplied verification.",
                index + 1,
                scopes.len()
            );
            foreground_audit_invocation(
                &scoped_instruction,
                fast && scopes.len() == 1,
                current_model,
                compose_audit_deep_prompt(&scoped_instruction, &changes),
                format!("slash-audit-scope-{}", index + 1),
            )
        })
        .collect()
}

pub(crate) fn foreground_audit_fallback(
    instruction: &str,
    fast: bool,
    current_model: &str,
) -> ForegroundAuditInvocation {
    foreground_audit_invocation(
        instruction,
        fast,
        current_model,
        build_audit_deep_prompt(instruction),
        "slash-audit".into(),
    )
}

fn foreground_audit_invocation(
    instruction: &str,
    fast: bool,
    current_model: &str,
    prompt: String,
    tool_call_id: String,
) -> ForegroundAuditInvocation {
    let mut args = serde_json::json!({
        "description": format!("/audit --deep{} {instruction}", if fast { " --fast" } else { "" }),
        "prompt": prompt,
        "agent": if fast { "audit-fast" } else { "audit" },
    });
    let (hard_timeout, wrap_up_lead_time) = if fast {
        args["model"] = serde_json::json!(current_model);
        args["reasoning_effort"] = serde_json::json!("high");
        (
            FAST_AUDIT_SUBAGENT_HARD_TIMEOUT,
            FAST_AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME,
        )
    } else {
        (
            AUDIT_SUBAGENT_HARD_TIMEOUT,
            AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME,
        )
    };
    ForegroundAuditInvocation {
        tool_call_id,
        args,
        hard_timeout,
        wrap_up_lead_time: Some(wrap_up_lead_time),
    }
}

/// Combine the request with collected changes without side effects.
fn compose_audit_prompt(instruction: &str, changes_context: &str) -> String {
    let instruction = format!("{instruction}\n\n{AUDIT_PROGRESS_PROTOCOL}");
    if changes_context.is_empty() {
        return instruction;
    }
    format!(
        "{instruction}\n\n\
         ===== Files changed by the main agent in this session =====\n\
         {changes_context}\n\
         ===== End of file changes ====="
    )
}

/// Build the legacy follow-up that verifies and fixes every confirmed issue.
pub(crate) fn build_audit_fix_followup(original_instruction: &str, audit_report: &str) -> String {
    format!(
        "The `/audit` subagent has finished. Verify each finding below against the \
         current workspace (read the cited files/lines; do not trust the report blindly), \
         fix every confirmed issue, and explain why any false positive is not actionable. \
         Run the narrowest focused checks the fix requires.\n\n\
         Original audit request: {instruction}\n\n\
         ===== Audit report =====\n\
         {report}\n\
         ===== End of audit report =====",
        instruction = original_instruction.trim(),
        report = audit_report.trim()
    )
}

pub(crate) fn build_audit_round_followup(
    original_instruction: &str,
    audit_report: &str,
    round: u8,
) -> String {
    let round_note = if round == 1 {
        "The initial `/audit` subagent attempt is round one; its returned report is \
         first-round evidence, partial on timeout/failure. Do not restart the main round. \
         If T1/T2/T3 requires round two, return a narrow `<audit_plan>` with `scopes`, \
         `triggers` (each with `trigger` and concrete `evidence`), and `changed_axis`; \
         the driver retains foreground timeout and V1 recovery authority."
    } else {
        "Round two has returned its bounded evidence, partial on timeout/failure. \
         Stop auditing; no further audit round is allowed. Apply the disposition/closure \
         gate and report unresolved items without claiming closed task acceptance."
    };
    format!(
        "{round_note} \
         Read the cited files/lines, verify findings, and explain refuted false positives.\n\n\
         {AUDIT_LEAD_CONTINUATION}\n\n\
         Original audit request: {instruction}\n\n\
         ===== Audit report =====\n\
         {report}\n\
         ===== End of audit report =====",
        instruction = original_instruction.trim(),
        report = audit_report.trim()
    )
}

/// Default audits prefer recorded mutations and use Git only when the log is empty.
fn capture_current_changes_context() -> String {
    let entries = crate::ai::tools::storage::mutation_log::read_all();
    if !entries.is_empty() {
        return format_mutation_log(&entries);
    }
    capture_git_diff_fallback()
}

fn format_mutation_log(
    entries: &[crate::ai::tools::storage::mutation_log::MutationEntry],
) -> String {
    let summary = crate::ai::tools::storage::changes::grouped_changes(entries);
    let log_path = crate::ai::tools::storage::mutation_log::log_path();
    let mut out = String::from(
        "Below are the files the main agent changed in this session via write_file / apply_patch (in order of first change).\n\
         This covers only this session's tool-level changes, not uncommitted changes left by other concurrent tasks.\n\n",
    );
    const CAP: usize = 14_000;
    let mut truncated = false;
    for (i, g) in summary.iter().enumerate() {
        if out.len() >= CAP {
            truncated = true;
            break;
        }
        let mut counts = String::new();
        if g.write_count > 0 {
            counts.push_str(&format!("{} write", g.write_count));
        }
        if g.delete_count > 0 {
            if !counts.is_empty() {
                counts.push_str(", ");
            }
            counts.push_str(&format!("{} delete", g.delete_count));
        }
        out.push_str(&format!("{}. {}  [{}]  ({counts})\n", i + 1, g.rel, g.net));
        if let Some(diff) = crate::ai::tools::storage::changes::file_snippet(g, 30) {
            out.push_str(&diff);
            out.push_str("\n\n");
        }
    }
    if truncated {
        out.push_str("…(more changes omitted)\n\n");
    }
    if let Some(lp) = &log_path {
        out.push_str(&format!(
            "Recorded snapshots and per-write diffs: {}\n\
             Snapshots may be truncated or unavailable; read the log for recorded evidence.\n",
            lp.display()
        ));
    }
    out
}

fn capture_git_diff_fallback() -> String {
    let cwd = match crate::ai::driver::runtime_ctx::effective_cwd() {
        Ok(p) => p,
        Err(_) => return String::new(),
    };
    if !is_inside_git_work_tree(&cwd) {
        return String::new();
    }
    let status = git_output(&cwd, &["status", "--porcelain=v1"]);
    if status.trim().is_empty() {
        return String::new();
    }
    const MAX_DIFF_BYTES: usize = 32_768;
    let diff = git_output(&cwd, &["diff", "HEAD", "--no-color"]);
    let mut sections = vec![format!("## git status --porcelain\n{status}")];
    if diff.len() <= MAX_DIFF_BYTES {
        if !diff.trim().is_empty() {
            sections.push(format!("## git diff HEAD\n{diff}"));
        }
    } else {
        let stat = git_output(&cwd, &["diff", "HEAD", "--stat", "--no-color"]);
        sections.push(format!(
            "## git diff HEAD --stat\n\
             (the complete diff exceeds {MAX_DIFF_BYTES} bytes, so only the stat is shown; run `git diff HEAD` yourself if you need the full content)\n\
             {stat}"
        ));
    }
    let mut out = String::from(
        "(no tool-level mutation log for this session; below are the workspace's uncommitted changes, which may include changes from concurrent tasks)\n\n",
    );
    out.push_str(&sections.join("\n\n"));
    out
}

fn is_inside_git_work_tree(cwd: &Path) -> bool {
    git_output(cwd, &["rev-parse", "--is-inside-work-tree"])
        .trim()
        .eq_ignore_ascii_case("true")
}

fn git_output(cwd: &Path, args: &[&str]) -> String {
    crate::fork_guard::output(
        std::process::Command::new("git")
            .args(args)
            .current_dir(cwd),
    )
    .ok()
    .filter(|o| o.status.success())
    .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        AUDIT_PROCEDURE_SKILL, AUDIT_SUBAGENT_HARD_TIMEOUT, AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME,
        AuditCommand, FAST_AUDIT_SUBAGENT_HARD_TIMEOUT, FAST_AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME,
        build_audit_fix_followup, compose_audit_lead_request, compose_audit_prompt,
        format_mutation_log, inject_self_generated_audit_note, parse_audit_command,
        parse_audit_followup_scope_plan, parse_audit_scope_plan, terminal_audit_result,
        turn_spawned_audit_subagent,
    };
    use crate::ai::history::Message;
    use crate::ai::tools::storage::changes::diff_snippet;
    use crate::ai::types::{FunctionCall, ToolCall};

    #[test]
    fn audit_subagent_hard_timeout_is_fifteen_minutes() {
        assert_eq!(AUDIT_SUBAGENT_HARD_TIMEOUT, Duration::from_secs(15 * 60));
    }

    #[test]
    fn audit_fix_followup_carries_instruction_and_report() {
        let prompt = build_audit_fix_followup("review my diff", "P1: foo is broken");
        assert!(prompt.contains("review my diff"), "{prompt}");
        assert!(prompt.contains("P1: foo is broken"), "{prompt}");
        // Must not re-trigger `/audit` parsing when fed back as the turn question.
        assert!(parse_audit_command(&prompt).is_none(), "{prompt}");
    }

    #[test]
    fn audit_deep_fix_followup_requires_disposition_and_authorization() {
        let prompt = super::build_audit_round_followup("review my diff", "P1: foo is broken", 1);
        assert!(prompt.contains("review my diff"), "{prompt}");
        assert!(prompt.contains("P1: foo is broken"), "{prompt}");
        assert!(
            prompt.contains("already authorized by the task"),
            "{prompt}"
        );
        assert!(
            prompt.contains("fix / narrow / accept / rollback"),
            "{prompt}"
        );
        assert!(
            prompt.contains("reversibility, and estimated cost"),
            "{prompt}"
        );
        assert!(!prompt.contains("fix every confirmed issue"), "{prompt}");
        for required in [
            "attempt is round one",
            "T1",
            "T2",
            "T3",
            "at most two rounds total",
            "Round two must change axis",
            "every inventory item",
            "no unreviewed source mutation",
            "empty or explicitly accepted",
        ] {
            assert!(prompt.contains(required), "missing {required}: {prompt}");
        }
        // Must not re-trigger `/audit` parsing when fed back as the turn question.
        assert!(parse_audit_command(&prompt).is_none(), "{prompt}");
    }

    #[test]
    fn audit_lead_request_routes_scoped_parallel_tasks_and_preserves_fast_settings() {
        for fast in [false, true] {
            let prompt = compose_audit_lead_request(
                "inspect the consumer contract",
                fast,
                "test-model",
                "Complete inventory fixture",
            );
            for required in [
                "inspect the consumer contract",
                "2-4 scoped audits",
                "invariant/consumer boundary",
                "same snapshot form one round",
                "task_integrate",
                "do not pass `response_schema`",
                "at most two rounds total",
                "Round two must change axis",
                "<audit_plan>",
                "Do not start `task_spawn` children",
                "SUBAGENT_TIMEOUT_RECOVERY_V1",
            ] {
                assert!(prompt.contains(required), "missing {required}: {prompt}");
            }
            assert_eq!(
                prompt.contains("\"agent\":\"audit-fast\""),
                fast,
                "{prompt}"
            );
            assert_eq!(prompt.contains("test-model"), fast, "{prompt}");
            assert_eq!(prompt.contains("reasoning_effort"), fast, "{prompt}");
            assert!(parse_audit_command(&prompt).is_none(), "{prompt}");
        }
    }

    #[test]
    fn audit_subagent_reserves_five_minutes_for_wrap_up() {
        assert_eq!(
            AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME,
            Duration::from_secs(5 * 60)
        );
    }

    #[test]
    fn fast_audit_shorter_timeouts_than_full_audit() {
        assert!(FAST_AUDIT_SUBAGENT_HARD_TIMEOUT < AUDIT_SUBAGENT_HARD_TIMEOUT);
        assert!(FAST_AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME < AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME);
    }

    #[test]
    fn turn_spawned_audit_subagent_detects_task_spawn_calls() {
        let call = |name: &str, args: &str| ToolCall {
            id: "1".into(),
            tool_type: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: args.into(),
            },
        };
        let msg = |calls: Vec<ToolCall>| Message {
            role: "assistant".into(),
            content: serde_json::Value::Null,
            tool_calls: Some(calls),
            tool_call_id: None,
            reasoning_content: None,
        };
        // Either audit agent spawned through task_spawn is detected.
        assert!(turn_spawned_audit_subagent(&[msg(vec![call(
            "task_spawn",
            r#"{"tasks":[{"agent":"audit-fast"},{"agent":"build"}]}"#
        )])]));
        // Other agents, other tools, malformed args, and empty history are not.
        assert!(!turn_spawned_audit_subagent(&[msg(vec![call(
            "task_spawn",
            r#"{"tasks":[{"agent":"build"}]}"#
        )])]));
        assert!(!turn_spawned_audit_subagent(&[msg(vec![call(
            "read_file",
            r#"{"agent":"audit"}"#
        )])]));
        assert!(!turn_spawned_audit_subagent(&[msg(vec![call(
            "task_spawn",
            "not json"
        )])]));
        assert!(!turn_spawned_audit_subagent(&[]));
    }

    #[test]
    fn audit_orchestration_startup_requires_a_matched_success_result() {
        let call = |name: &str, args: &str| ToolCall {
            id: "1".into(),
            tool_type: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: args.into(),
            },
        };
        let msg = |calls: Vec<ToolCall>| Message {
            role: "assistant".into(),
            content: serde_json::Value::Null,
            tool_calls: Some(calls),
            tool_call_id: None,
            reasoning_content: None,
        };
        let result = |id: &str, agent: &str, status: &str| Message {
            role: "tool".into(),
            content: serde_json::Value::String(
                serde_json::json!({
                    "spawned": 1,
                    "tasks": [{"status": status, "task_id": "child", "pid": 42, "agent": agent}]
                })
                .to_string(),
            ),
            tool_calls: None,
            tool_call_id: Some(id.into()),
            reasoning_content: None,
        };
        for agent in ["audit", "audit-fast"] {
            let attempted = msg(vec![call(
                "task_spawn",
                &serde_json::json!({"tasks": [{"agent": agent}]}).to_string(),
            )]);
            assert!(!super::turn_successfully_spawned_audit_subagent(&[
                attempted.clone()
            ]));
            assert!(!super::turn_successfully_spawned_audit_subagent(&[
                attempted.clone(),
                result("1", agent, "failed")
            ]));
            assert!(!super::turn_successfully_spawned_audit_subagent(&[
                attempted.clone(),
                result("other-call", agent, "spawned")
            ]));
            assert!(!super::turn_successfully_spawned_audit_subagent(&[
                attempted.clone(),
                result("1", "build", "spawned")
            ]));
            assert!(super::turn_successfully_spawned_audit_subagent(&[
                attempted,
                result("1", agent, "spawned")
            ]));
        }
        // Other agents, other tools, malformed args, and empty history are not.
        assert!(!super::turn_successfully_spawned_audit_subagent(&[msg(
            vec![call("task_spawn", r#"{"tasks":[{"agent":"build"}]}"#)]
        )]));
        assert!(!super::turn_successfully_spawned_audit_subagent(&[msg(
            vec![call("read_file", r#"{"agent":"audit"}"#)]
        )]));
        assert!(!super::turn_successfully_spawned_audit_subagent(&[msg(
            vec![call("task_spawn", "not json")]
        )]));
        assert!(!super::turn_successfully_spawned_audit_subagent(&[]));
    }

    #[test]
    fn audit_orchestration_scope_plan_and_second_round_are_bounded() {
        assert_eq!(
            parse_audit_scope_plan(
                r#"<audit_plan>{"scopes":["producer invariant","consumer invariant"]}</audit_plan>"#
            )
            .unwrap()
            .len(),
            2
        );
        for bad in [
            "no scope plan",
            r#"<audit_plan>{"scopes":[]}</audit_plan>"#,
            r#"<audit_plan>{"scopes":[""]}</audit_plan>"#,
            r#"<audit_plan>{"scopes":["a","b","c","d","e"]}</audit_plan>"#,
        ] {
            assert!(parse_audit_scope_plan(bad).is_err());
        }
        let valid = r#"<audit_plan>{"scopes":["check the missing consumer"],"triggers":[{"trigger":"T1","evidence":"consumer read gap remains"}],"changed_axis":"previously unread consumer"}</audit_plan>"#;
        assert!(parse_audit_followup_scope_plan(valid).is_ok());
        assert!(
            parse_audit_followup_scope_plan(r#"<audit_plan>{"scopes":["repeat"]}</audit_plan>"#)
                .is_err()
        );
        assert!(parse_audit_followup_scope_plan(&valid.replace("\"T1\"", "\"T4\"")).is_err());
        assert!(
            parse_audit_followup_scope_plan(&valid.replace("previously unread consumer", ""))
                .is_err()
        );
    }

    #[test]
    fn parse_audit_command_recognizes_instruction_and_alias() {
        assert_eq!(
            parse_audit_command("/audit review the current diff"),
            Some(AuditCommand::Run {
                instruction: "review the current diff".to_string(),
                fast: false,
                deep: false,
            })
        );
        assert_eq!(
            parse_audit_command(":audit inspect src/bin/a.rs"),
            Some(AuditCommand::Run {
                instruction: "inspect src/bin/a.rs".to_string(),
                fast: false,
                deep: false,
            })
        );
    }

    #[test]
    fn parse_audit_command_recognizes_fast_flag() {
        assert_eq!(
            parse_audit_command("/audit --fast review the diff"),
            Some(AuditCommand::Run {
                instruction: "review the diff".to_string(),
                fast: true,
                deep: false,
            })
        );
        assert_eq!(
            parse_audit_command("/audit -f check src/lib.rs"),
            Some(AuditCommand::Run {
                instruction: "check src/lib.rs".to_string(),
                fast: true,
                deep: false,
            })
        );
    }

    #[test]
    fn parse_audit_command_fast_flag_without_instruction_is_self_generated() {
        assert_eq!(
            parse_audit_command("/audit --fast"),
            Some(AuditCommand::SelfGenerated {
                fast: true,
                deep: false
            })
        );
        assert_eq!(
            parse_audit_command("/audit -f  "),
            Some(AuditCommand::SelfGenerated {
                fast: true,
                deep: false
            })
        );
    }

    #[test]
    fn parse_audit_command_does_not_confuse_fast_prefix_with_word() {
        // `--fastly` is not a flag, so preserve the instruction.
        assert_eq!(
            parse_audit_command("/audit --fastly failing path"),
            Some(AuditCommand::Run {
                instruction: "--fastly failing path".to_string(),
                fast: false,
                deep: false,
            })
        );
    }

    #[test]
    fn parse_audit_command_without_instruction_is_self_generated() {
        assert_eq!(
            parse_audit_command("/audit"),
            Some(AuditCommand::SelfGenerated {
                fast: false,
                deep: false
            })
        );
        assert_eq!(
            parse_audit_command(" :audit   "),
            Some(AuditCommand::SelfGenerated {
                fast: false,
                deep: false
            })
        );
    }

    #[test]
    fn parse_audit_command_deep_requires_a_complete_leading_token() {
        for (input, expected) in [
            (
                "/audit --deep inspect",
                AuditCommand::Run {
                    instruction: "inspect".into(),
                    fast: false,
                    deep: true,
                },
            ),
            (
                ":audit --deep --fast inspect",
                AuditCommand::Run {
                    instruction: "inspect".into(),
                    fast: true,
                    deep: true,
                },
            ),
            (
                "/audit --deep -f inspect",
                AuditCommand::Run {
                    instruction: "inspect".into(),
                    fast: true,
                    deep: true,
                },
            ),
            (
                "/audit --deep",
                AuditCommand::SelfGenerated {
                    fast: false,
                    deep: true,
                },
            ),
            (
                "/audit --deep --fast",
                AuditCommand::SelfGenerated {
                    fast: true,
                    deep: true,
                },
            ),
            (
                "/audit --deeply inspect",
                AuditCommand::Run {
                    instruction: "--deeply inspect".into(),
                    fast: false,
                    deep: false,
                },
            ),
            (
                "/audit --deep=1 inspect",
                AuditCommand::Run {
                    instruction: "--deep=1 inspect".into(),
                    fast: false,
                    deep: false,
                },
            ),
            (
                "/audit -f --deep inspect",
                AuditCommand::Run {
                    instruction: "--deep inspect".into(),
                    fast: true,
                    deep: false,
                },
            ),
            (
                "/audit --deep --fastly inspect",
                AuditCommand::Run {
                    instruction: "--fastly inspect".into(),
                    fast: false,
                    deep: true,
                },
            ),
            (
                "/audit --unknown inspect",
                AuditCommand::Run {
                    instruction: "--unknown inspect".into(),
                    fast: false,
                    deep: false,
                },
            ),
        ] {
            assert_eq!(parse_audit_command(input), Some(expected), "{input}");
        }
    }

    #[test]
    fn audit_default_prompt_bytes_match_legacy_composition() {
        let progress = super::AUDIT_PROGRESS_PROTOCOL;
        assert_eq!(
            compose_audit_prompt("inspect", ""),
            format!("inspect\n\n{progress}")
        );
        assert_eq!(
            compose_audit_prompt("inspect", "recorded changes"),
            format!(
                "inspect\n\n{progress}\n\n===== Files changed by the main agent in this session =====\nrecorded changes\n===== End of file changes ====="
            )
        );
        assert_eq!(
            build_audit_fix_followup(" inspect ", " report "),
            "The `/audit` subagent has finished. Verify each finding below against the current workspace (read the cited files/lines; do not trust the report blindly), fix every confirmed issue, and explain why any false positive is not actionable. Run the narrowest focused checks the fix requires.\n\nOriginal audit request: inspect\n\n===== Audit report =====\nreport\n===== End of audit report ====="
        );
    }

    #[test]
    fn audit_deep_fix_followup_stops_after_the_second_round() {
        let prompt = super::build_audit_round_followup("inspect", "partial report", 2);
        assert!(prompt.contains("no further audit round is allowed"));
        assert!(prompt.contains("partial report"));
        assert!(prompt.contains("report unresolved items"));
    }

    #[test]
    fn inject_self_generated_audit_note_quotes_the_builtin_skill() {
        use crate::ai::history::ROLE_INTERNAL_NOTE;
        use crate::ai::skills::builtin_skill_body;

        let body = builtin_skill_body(AUDIT_PROCEDURE_SKILL)
            .expect("the builtin audit_own_changes skill is compiled in");
        // Pin the source independently of the note, so quoting the wrong builtin cannot pass.
        assert!(body.starts_with("# Self-audit"), "{body}");

        let mut messages = Vec::new();
        inject_self_generated_audit_note(&mut messages, false);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, ROLE_INTERNAL_NOTE);
        let note = messages[0].content.as_str().unwrap();
        assert!(note.contains("[audit-self-generated]"), "{note}");
        assert!(note.contains("agent=\"audit\""), "{note}");
        // The note quotes the shipped skill text verbatim instead of restating the procedure.
        assert_eq!(note.matches(body.as_str()).count(), 1, "{note}");

        // Fast mode names the audit-fast subagent.
        let mut fast_messages = Vec::new();
        inject_self_generated_audit_note(&mut fast_messages, true);
        let fast_note = fast_messages[0].content.as_str().unwrap();
        assert!(fast_note.contains("/audit -f (fast)"), "{fast_note}");
        assert!(fast_note.contains("agent=\"audit-fast\""), "{fast_note}");
        assert!(fast_messages[0].tool_calls.is_none());
        assert!(fast_messages[0].tool_call_id.is_none());
    }

    #[test]
    fn parse_audit_command_does_not_capture_other_input() {
        assert_eq!(parse_audit_command("/auditor review"), None);
        assert_eq!(parse_audit_command("please /audit this"), None);
    }

    #[test]
    fn terminal_audit_result_hides_subagent_evidence_and_parent_reminder() {
        let payload = format!(
            "[task_id=task-1]\n\
[Task: /audit inspect diff via audit @ model] COMPLETED after 1.0s\n\
[Subagent tool evidence]\n\
- read_file({{\"file_path\":\"src/lib.rs\"}}) => fn main() {{}}\n\n\
[Subagent final answer]\n\
No verified findings.\n{}",
            crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER
        );

        assert_eq!(terminal_audit_result(&payload), "No verified findings.");
    }

    #[test]
    fn terminal_audit_result_keeps_compact_error_when_no_final_answer_exists() {
        let payload = "[task_id=task-1]\n[Task: /audit inspect via audit @ model] FAILED after 1.0s\nError: model unavailable\n[subagent evidence omitted]";

        assert_eq!(terminal_audit_result(payload), "[audit] model unavailable");
    }

    #[test]
    fn terminal_audit_result_keeps_compact_error_on_timeout_with_recovery_evidence() {
        // Timeout payloads contain both an error and recovery evidence. Only the lead
        // receives that evidence; the terminal shows the compact error.
        let payload = format!(
            "[task_id=task-1]\n\
[Task: /audit inspect diff via audit @ model] TIMED_OUT after 900.0s\n\
Error: subagent task exceeded hard timeout of 900s\n\
SUBAGENT_TIMEOUT_RECOVERY_V1\n\
status: timed_out\n\
error: subagent task exceeded hard timeout of 900s\n\
last_phase: CallingTool\n\
preserved_child_history: /tmp/audit-timeout.sqlite\n\
\n\
## 中断前已完成的工作（恢复节选）\n\
\n\
AUDIT_CHECKPOINT: checked src/a.rs; finding at src/a.rs:42\n\
\n\
以上是阶段性证据而非完整审计结论。\n{}",
            crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER
        );

        let rendered = terminal_audit_result(&payload);
        assert_eq!(
            rendered,
            "[audit] subagent task exceeded hard timeout of 900s"
        );
        // Recovery evidence and retained-history paths belong only to the lead.
        assert!(!rendered.contains("src/a.rs:42"));
        assert!(!rendered.contains("preserved_child_history"));
    }

    #[test]
    fn terminal_audit_result_recovers_inline_answer_without_marker() {
        // An inline answer with no tool calls has no final-answer marker. Strip
        // metadata and the parent reminder, then preserve the actual conclusion.
        let payload = format!(
            "[task_id=task-1]\n\
[Task: /audit review the diff via audit @ model] COMPLETED after 2.0s\n\
agent_reason=explicit agent override\n\
model_reason=inherited parent agent current model\n\
P0: 空指针解引用在 foo.rs:42。其余改动无阻断问题。\n{}",
            crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER
        );

        assert_eq!(
            terminal_audit_result(&payload),
            "P0: 空指针解引用在 foo.rs:42。其余改动无阻断问题。"
        );
    }

    #[test]
    fn terminal_audit_result_falls_back_to_generic_when_no_final_text_produced() {
        // A placeholder means the subagent produced no conclusion; use the notice.
        let payload = format!(
            "[task_id=task-1]\n\
[Task: /audit review via audit @ model] COMPLETED after 1.0s\n\
agent_reason=explicit agent override\n\
model_reason=inherited parent agent current model\n\
(subagent did not produce any final assistant text)\n{}",
            crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER
        );

        assert!(
            terminal_audit_result(&payload)
                .starts_with("[audit] Audit subagent finished without a final answer")
        );
    }

    #[test]
    fn compose_audit_prompt_requires_incremental_checkpoints_without_changes() {
        let prompt = compose_audit_prompt("review the diff", "");
        assert!(prompt.starts_with("review the diff\n\n"));
        assert!(prompt.contains("AUDIT_CHECKPOINT:"));
        assert!(prompt.contains("Do not wait until the final answer to report findings"));
    }

    #[test]
    fn compose_audit_prompt_appends_changes_context() {
        let prompt = compose_audit_prompt("review the diff", "（示例改动上下文）");
        assert!(prompt.starts_with("review the diff\n\n"));
        assert!(prompt.contains("Files changed by the main agent in this session"));
        assert!(prompt.contains("（示例改动上下文）"));
        assert!(prompt.contains("End of file changes"));
    }

    #[test]
    fn diff_snippet_shows_added_lines_for_created_file() {
        let snippet = diff_snippet(None, Some("line1\nline2\n"), 30).unwrap();
        assert!(snippet.contains("+ line1"));
        assert!(snippet.contains("+ line2"));
        assert!(!snippet.contains("- "));
    }

    #[test]
    fn diff_snippet_shows_removed_lines_for_deleted_file() {
        let snippet = diff_snippet(Some("old\n"), None, 30).unwrap();
        assert!(snippet.contains("- old"));
        assert!(!snippet.contains("+ "));
    }

    #[test]
    fn diff_snippet_strips_common_prefix_and_suffix() {
        let before = "header\nkeep1\nMIDDLE\nkeep2\nfooter\n";
        let after = "header\nkeep1\nNEW\nkeep2\nfooter\n";
        let snippet = diff_snippet(Some(before), Some(after), 30).unwrap();
        assert!(snippet.contains("- MIDDLE"));
        assert!(snippet.contains("+ NEW"));
        assert!(!snippet.contains("header"));
        assert!(!snippet.contains("footer"));
    }

    #[test]
    fn diff_snippet_returns_none_when_unchanged() {
        assert!(diff_snippet(Some("same\n"), Some("same\n"), 30).is_none());
        assert!(diff_snippet(None, None, 30).is_none());
    }

    #[test]
    fn diff_snippet_truncates_large_diff() {
        let before: String = "old\n".repeat(100);
        let after: String = "new\n".repeat(100);
        let snippet = diff_snippet(Some(&before), Some(&after), 10).unwrap();
        assert!(snippet.contains("已展示前 10 行"));
        assert!(snippet.contains("差异共 200 行"));
    }

    #[test]
    fn format_mutation_log_groups_by_path_and_shows_net_change() {
        use crate::ai::tools::storage::mutation_log::MutationEntry;
        let entries = vec![MutationEntry {
            seq: 1,
            ts: "t1".into(),
            path: "/proj/src/a.rs".into(),
            op: "write".into(),
            before: Some("fn a() {}\n".into()),
            before_state: None,
            after: Some("fn a() { return 1; }\n".into()),
            diff: Some("- fn a() {}\n+ fn a() { return 1; }\n".into()),
        }];
        let out = format_mutation_log(&entries);
        assert!(out.contains("src/a.rs"));
        assert!(out.contains("[modified]"));
        assert!(out.contains("1 write"));
        assert!(out.contains("- fn a() {}"));
        assert!(out.contains("+ fn a() { return 1; }"));
    }

    #[test]
    fn audit_preserves_unknown_initial_snapshot_across_later_writes() {
        use crate::ai::tools::storage::mutation_log::{BeforeState, MutationEntry};
        for state in [BeforeState::Present, BeforeState::Unknown] {
            let mut entries = vec![MutationEntry {
                seq: 0,
                ts: "t".into(),
                path: "/proj/existing.rs".into(),
                op: "write".into(),
                before: None,
                before_state: Some(state),
                after: Some("first\n".into()),
                diff: None,
            }];
            entries.push(MutationEntry {
                seq: 1,
                ts: "t".into(),
                path: entries[0].path.clone(),
                op: "write".into(),
                before: Some("first\n".into()),
                before_state: Some(BeforeState::Present),
                after: Some("second\n".into()),
                diff: Some("- first\n+ second\n".into()),
            });
            let out = format_mutation_log(&entries);
            assert!(!out.contains("[created]"), "{out}");
            assert!(
                out.contains(if state == BeforeState::Present {
                    "[modified]"
                } else {
                    "[written]"
                }),
                "{out}"
            );
            assert!(out.contains("before snapshot unavailable"), "{out}");
            assert!(out.contains("2 write"), "{out}");
            assert!(
                !out.lines()
                    .any(|line| line.starts_with('+') || line.starts_with('-')),
                "{out}"
            );
        }
    }

    #[test]
    fn audit_identical_large_write_reports_only_comparison_limit() {
        use crate::ai::tools::storage::mutation_log::{
            BeforeState, MutationEntry, cap_content, entry_diff,
        };
        let content = "same line\n".repeat(200_000);
        let entries = [MutationEntry {
            seq: 0,
            ts: "t".into(),
            path: "/proj/large.rs".into(),
            op: "write".into(),
            before: Some(cap_content(&content)),
            before_state: Some(BeforeState::Present),
            after: Some(cap_content(&content)),
            diff: entry_diff(Some(&content), Some(&content)),
        }];
        let out = format_mutation_log(&entries);
        assert!(out.contains("remaining content not compared"), "{out}");
        assert!(!out.contains("differ beyond"), "{out}");
    }
}
