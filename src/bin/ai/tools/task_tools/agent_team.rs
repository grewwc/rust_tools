//! Agent team deliberation subsystem.
//!
//! Manages multi-agent team phases (start/challenge/synthesize) via parent-mediated
//! message passing. Team members do NOT message each other directly; the parent passes
//! full transcripts between phases.

use serde_json::Value;

use uuid::Uuid;

use crate::ai::model_names;

use aios_kernel::{ChannelId, FutexAddr};

use super::{
    InheritOptions, MAX_AGENT_TEAM_MEMBERS, PreparedSubagentTask, prepare_subagent_task,
    remove_task_entry, spawn_subagent_kernel_task, with_os_kernel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AgentTeamOperation {
    Start,
    Challenge,
    Synthesize,
}

impl AgentTeamOperation {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value
            .unwrap_or("start")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "start" => Ok(Self::Start),
            "challenge" => Ok(Self::Challenge),
            "synthesize" => Ok(Self::Synthesize),
            other => Err(format!(
                "Unknown agent_team operation '{}'. Expected start, challenge, or synthesize.",
                other
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Challenge => "challenge",
            Self::Synthesize => "synthesize",
        }
    }
}

#[derive(Debug)]
pub(crate) struct AgentTeamMemberSpec {
    pub(crate) role: String,
    pub(crate) prompt: String,
    pub(crate) agent: Option<String>,
    pub(crate) model: Option<String>,
}

struct PreparedAgentTeamMember {
    role: String,
    prepared: PreparedSubagentTask,
}

pub(crate) fn execute_agent_team(args: &Value) -> Result<String, String> {
    let operation = AgentTeamOperation::parse(args["operation"].as_str())?;
    let goal = required_nonempty_str(args, "goal")?;
    let inherit = InheritOptions::from_value(&args["inherit"])?;
    let transcript = args["transcript"].as_str().unwrap_or("").trim();
    if matches!(
        operation,
        AgentTeamOperation::Challenge | AgentTeamOperation::Synthesize
    ) && transcript.is_empty()
    {
        return Err(
            "agent_team transcript is required for challenge/synthesize phases.".to_string(),
        );
    }

    let members = parse_agent_team_members(args, operation)?;
    let prepared_members = members
        .iter()
        .map(|member| {
            let team_prompt = build_agent_team_prompt(operation, goal, member, transcript);
            let selection_prompt = build_agent_team_selection_prompt(operation, member);
            let description = format_agent_team_description(operation, &member.role);
            let mut task_args = serde_json::json!({
                "description": description,
                "prompt": selection_prompt,
                "agent": member.agent.as_deref().unwrap_or(""),
                "inherit": inherit.describe(),
            });
            if let Some(model) = member.model.as_deref() {
                task_args["model"] = serde_json::json!(resolve_agent_team_model_override(model)?);
            }
            prepare_subagent_task(&task_args).map(|mut prepared| {
                // 完整 team prompt 可能包含大段 transcript/模板，容易把每个成员都误判
                // 成 heavy 任务而选到高成本模型；agent/model 选择用上面的短 prompt，
                // 实际运行仍使用完整 prompt。
                prepared.prompt = team_prompt;
                PreparedAgentTeamMember {
                    role: member.role.clone(),
                    prepared,
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let team_id = format!("team_{}", Uuid::new_v4().simple());
    let mut launched = Vec::with_capacity(prepared_members.len());
    for member in &prepared_members {
        let spawned = match spawn_subagent_kernel_task(&member.prepared) {
            Ok(spawned) => spawned,
            Err(err) => {
                cleanup_launched_agent_team_members(&launched);
                let partial = if launched.is_empty() {
                    "no members were launched".to_string()
                } else {
                    format!(
                        "cleaned up already launched task_ids: {}",
                        launched
                            .iter()
                            .map(|item: &LaunchedAgentTeamMember| item.task_id.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                return Err(format!(
                    "Failed to launch agent_team member '{}': {} ({})",
                    member.role, err, partial
                ));
            }
        };
        launched.push(LaunchedAgentTeamMember {
            role: member.role.clone(),
            task_id: spawned.task_id,
            pid: spawned.pid,
            result_channel_id: spawned.result_channel_id,
            completion_futex_addr: spawned.completion_futex_addr,
            agent_name: member.prepared.agent_name.clone(),
            model: member.prepared.model.clone(),
        });
    }

    Ok(format_agent_team_launch_result(
        &team_id, operation, goal, inherit, &launched,
    ))
}

struct LaunchedAgentTeamMember {
    role: String,
    task_id: String,
    pid: u64,
    result_channel_id: u64,
    completion_futex_addr: FutexAddr,
    agent_name: String,
    model: String,
}

pub(crate) fn resolve_agent_team_model_override(model: &str) -> Result<String, String> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Err("agent_team member model override cannot be empty".to_string());
    }
    if let Some(def) = model_names::find_by_identifier(trimmed) {
        return Ok(model_names::model_handle(def));
    }
    Err(format!(
        "Unknown agent_team member model '{}'. Team model overrides require an exact model key/name to avoid accidentally selecting an expensive fallback.",
        trimmed
    ))
}

fn cleanup_launched_agent_team_members(launched: &[LaunchedAgentTeamMember]) {
    for member in launched {
        let _ = remove_task_entry(&member.task_id);
        let _ = with_os_kernel(|os| {
            if let Err(err) = os.kill_process(
                member.pid,
                "agent_team launch failed; cleaning up partial phase".to_string(),
            ) {
                eprintln!(
                    "[agent_team] cleanup failed to kill pid {} for task_id {}: {}",
                    member.pid, member.task_id, err
                );
            }
            if let Err(err) = os.channel_close(None, ChannelId(member.result_channel_id)) {
                eprintln!(
                    "[agent_team] cleanup failed to close result channel {} for task_id {}: {}",
                    member.result_channel_id, member.task_id, err
                );
            }
            if let Err(err) = os
                .channel_release_named(ChannelId(member.result_channel_id), "task_result.consumer")
            {
                eprintln!(
                    "[agent_team] cleanup failed to release consumer holder on channel {} for task_id {}: {}",
                    member.result_channel_id, member.task_id, err
                );
            }
            if let Err(err) = os
                .channel_release_named(ChannelId(member.result_channel_id), "task_result.producer")
            {
                eprintln!(
                    "[agent_team] cleanup failed to release producer holder on channel {} for task_id {}: {}",
                    member.result_channel_id, member.task_id, err
                );
            }
            if let Err(err) = os.channel_destroy(None, ChannelId(member.result_channel_id)) {
                eprintln!(
                    "[agent_team] cleanup failed to destroy result channel {} for task_id {}: {}",
                    member.result_channel_id, member.task_id, err
                );
            }
            if !os.futex_destroy(member.completion_futex_addr) {
                eprintln!(
                    "[agent_team] cleanup failed to destroy completion futex {:?} for task_id {}",
                    member.completion_futex_addr, member.task_id
                );
            }
            Ok(())
        });
    }
}

fn required_nonempty_str<'a>(args: &'a Value, field: &str) -> Result<&'a str, String> {
    let value = args[field]
        .as_str()
        .ok_or_else(|| format!("Missing '{}' parameter", field))?
        .trim();
    if value.is_empty() {
        return Err(format!("{} cannot be empty", field));
    }
    Ok(value)
}

pub(crate) fn parse_agent_team_members(
    args: &Value,
    operation: AgentTeamOperation,
) -> Result<Vec<AgentTeamMemberSpec>, String> {
    let values = args["members"]
        .as_array()
        .ok_or("Missing 'members' array parameter")?;
    let min_members = if operation == AgentTeamOperation::Start {
        2
    } else {
        1
    };
    if values.len() < min_members {
        return Err(format!(
            "agent_team operation '{}' requires at least {} member(s).",
            operation.label(),
            min_members
        ));
    }
    if values.len() > MAX_AGENT_TEAM_MEMBERS {
        return Err(format!(
            "agent_team supports at most {} members per phase.",
            MAX_AGENT_TEAM_MEMBERS
        ));
    }

    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let role = value["role"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("members[{index}].role cannot be empty"))?;
            let prompt = value["prompt"]
                .as_str()
                .map(str::trim)
                .unwrap_or("")
                .to_string();
            let agent = value["agent"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string);
            let model = value["model"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string);
            Ok(AgentTeamMemberSpec {
                role: role.to_string(),
                prompt,
                agent,
                model,
            })
        })
        .collect()
}

pub(crate) fn build_agent_team_prompt(
    operation: AgentTeamOperation,
    goal: &str,
    member: &AgentTeamMemberSpec,
    transcript: &str,
) -> String {
    let role_prompt = if member.prompt.trim().is_empty() {
        "(No additional role-specific instructions.)"
    } else {
        member.prompt.trim()
    };
    let mut prompt = format!(
        "You are a member of an AIOS agent team.\n\nTeam goal:\n{goal}\n\nYour role:\n{}\n\nRole-specific instructions:\n{role_prompt}\n\nCommunication contract:\n- Do not wait for direct messages from peer agents.\n- The parent agent coordinates the team and will pass complete transcripts between phases.\n- Make your output self-contained so it can be forwarded to later challenge/synthesis phases.\n",
        member.role
    );
    match operation {
        AgentTeamOperation::Start => {
            prompt.push_str(
                "\nPhase: initial independent analysis.\n- Provide your best answer from this role.\n- State assumptions, evidence, risks, and unresolved questions.\n- Explicitly list points that another agent should challenge.\n",
            );
        }
        AgentTeamOperation::Challenge => {
            prompt.push_str(
                "\nPhase: challenge.\nReview the transcript below. Challenge weak assumptions, missing evidence, contradictions, unsafe proposals, and overconfident conclusions. Keep valid points and propose concrete corrections.\n\nPrior team transcript:\n",
            );
            prompt.push_str(transcript.trim());
        }
        AgentTeamOperation::Synthesize => {
            prompt.push_str(
                "\nPhase: synthesis.\nUse the transcript below to produce the strongest final answer. Resolve disagreements explicitly, cite which arguments survived challenge, and call out residual uncertainty.\n\nPrior team transcript:\n",
            );
            prompt.push_str(transcript.trim());
        }
    }
    prompt
}

pub(crate) fn build_agent_team_selection_prompt(
    operation: AgentTeamOperation,
    member: &AgentTeamMemberSpec,
) -> String {
    let role_prompt = if member.prompt.trim().is_empty() {
        "Use the role name as the main specialization signal."
    } else {
        member.prompt.trim()
    };
    format!(
        "agent_team phase: {}\nrole: {}\nrole instructions: {}",
        operation.label(),
        member.role,
        role_prompt
    )
}

fn format_agent_team_description(operation: AgentTeamOperation, role: &str) -> String {
    let compact_role = role.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("agent_team {} {}", operation.label(), compact_role)
}

fn format_agent_team_launch_result(
    team_id: &str,
    operation: AgentTeamOperation,
    goal: &str,
    inherit: InheritOptions,
    launched: &[LaunchedAgentTeamMember],
) -> String {
    let task_ids = launched
        .iter()
        .map(|member| member.task_id.as_str())
        .collect::<Vec<_>>();
    let mut lines = vec![
        format!(
            "Agent team phase launched: team_id={}, operation={}, members={}, inherit={}",
            team_id,
            operation.label(),
            launched.len(),
            inherit.describe()
        ),
        format!("Goal: {}", goal),
        "Members:".to_string(),
    ];
    for member in launched {
        lines.push(format!(
            "- role='{}' task_id={} pid={} agent={} model={}",
            member.role, member.task_id, member.pid, member.agent_name, member.model
        ));
    }
    lines.push(format!(
        "Next: call task_wait with task_ids=[{}] and wait_policy=\"all\" to collect this phase.",
        task_ids
            .iter()
            .map(|id| format!("\"{}\"", id))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    match operation {
        AgentTeamOperation::Start => lines.push(
            "After collection, call agent_team operation=\"challenge\" with transcript=<all member outputs> to make agents challenge each other."
                .to_string(),
        ),
        AgentTeamOperation::Challenge => lines.push(
            "After collection, call agent_team operation=\"synthesize\" with transcript=<initial outputs + challenges> for the final conclusion."
                .to_string(),
        ),
        AgentTeamOperation::Synthesize => lines.push(
            "After collection, use the synthesis output as the team conclusion; no direct peer messages are expected."
                .to_string(),
        ),
    }
    lines.join("\n")
}
