//! Rust-native persistent Team Runtime.
//!
//! The team keeps one set of persistent state, while actual agent execution keeps reusing
//! the kernel process, channel/futex, result evidence, and cancellation protocol from
//! `task_tools`. Members are logical cross-turn roles; each dispatch is still a bounded
//! leaf subagent, avoiding a second resident-process source of truth or breaking the depth guard.

use super::{
    OwnedTaskPoll, execute_task_cancel, poll_owned_task_result, prepare_subagent_task,
    spawn_subagent_kernel_task,
};
use crate::ai::tools::registry::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec,
};
use crate::ai::tools::storage::file_store::current_session_assets_dir;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

mod graph;

const TEAM_STATE_VERSION: u32 = 1;
const MAX_TEAM_MEMBERS: usize = 32;
const MAX_TEAM_TASKS: usize = 512;
const MAX_TEAM_MESSAGES: usize = 2_048;
const MAX_PROMPT_CHARS: usize = 64_000;
const MAX_RESULT_SUMMARY_CHARS: usize = 5_500;
const DEFAULT_LEASE_SECS: u64 = 300;

static TEAM_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "manage_team",
        description: "",
        execute: execute_manage_team,
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "manage_team",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum TeamLifecycle {
    Active,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum MemberStatus {
    Idle,
    Busy,
    Paused,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum TeamTaskState {
    Pending,
    Claimed,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl TeamTaskState {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TeamBudget {
    pub(super) max_parallel: usize,
    pub(super) max_tasks: usize,
    pub(super) max_total_attempts: usize,
    pub(super) max_messages: usize,
    pub(super) attempts_used: usize,
}

impl Default for TeamBudget {
    fn default() -> Self {
        Self {
            max_parallel: 4,
            max_tasks: 128,
            max_total_attempts: 256,
            max_messages: 512,
            attempts_used: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TeamMember {
    pub(super) id: String,
    pub(super) role: String,
    pub(super) agent: Option<String>,
    pub(super) model: Option<String>,
    pub(super) inherit: String,
    pub(super) capabilities: Vec<String>,
    pub(super) status: MemberStatus,
    pub(super) active_task_id: Option<String>,
    pub(super) last_message_seq: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TaskLease {
    pub(super) member_id: String,
    pub(super) expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TeamTask {
    pub(super) id: String,
    pub(super) title: String,
    pub(super) prompt: String,
    pub(super) assignee: Option<String>,
    pub(super) depends_on: Vec<String>,
    pub(super) state: TeamTaskState,
    pub(super) lease: Option<TaskLease>,
    pub(super) runtime_task_id: Option<String>,
    pub(super) runtime_state: Option<String>,
    pub(super) attempts: usize,
    pub(super) output: Option<String>,
    pub(super) error: Option<String>,
    pub(super) created_at_unix_ms: u64,
    pub(super) updated_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TeamMessage {
    pub(super) seq: u64,
    pub(super) from: String,
    pub(super) to: String,
    pub(super) body: String,
    pub(super) created_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TeamState {
    pub(super) version: u32,
    pub(super) id: String,
    pub(super) session_id: String,
    pub(super) owner_pid: u64,
    pub(super) name: String,
    pub(super) goal: String,
    pub(super) lifecycle: TeamLifecycle,
    pub(super) budget: TeamBudget,
    pub(super) members: FxHashMap<String, TeamMember>,
    pub(super) tasks: FxHashMap<String, TeamTask>,
    pub(super) messages: Vec<TeamMessage>,
    pub(super) next_message_seq: u64,
    pub(super) created_at_unix_ms: u64,
    pub(super) updated_at_unix_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct TeamBudgetInput {
    #[serde(default = "default_max_parallel")]
    max_parallel: usize,
    #[serde(default = "default_max_tasks")]
    max_tasks: usize,
    #[serde(default = "default_max_attempts")]
    max_total_attempts: usize,
    #[serde(default = "default_max_messages")]
    max_messages: usize,
}

impl Default for TeamBudgetInput {
    fn default() -> Self {
        Self {
            max_parallel: default_max_parallel(),
            max_tasks: default_max_tasks(),
            max_total_attempts: default_max_attempts(),
            max_messages: default_max_messages(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TeamMemberInput {
    id: String,
    role: String,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "default_inherit")]
    inherit: String,
    #[serde(default)]
    capabilities: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TeamTaskInput {
    #[serde(default)]
    id: Option<String>,
    title: String,
    prompt: String,
    #[serde(default)]
    assignee: Option<String>,
    #[serde(default)]
    depends_on: Vec<String>,
}

fn default_max_parallel() -> usize {
    4
}
fn default_max_tasks() -> usize {
    128
}
fn default_max_attempts() -> usize {
    256
}
fn default_max_messages() -> usize {
    512
}
fn default_inherit() -> String {
    "cwd,skills".to_string()
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn clip_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut clipped = value.chars().take(max_chars).collect::<String>();
    clipped.push_str("\n…[truncated by Team Runtime]");
    clipped
}

fn validate_id(kind: &str, value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() || value.len() > 96 {
        return Err(format!("{kind} must be 1..=96 bytes"));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(format!(
            "{kind} may contain only ASCII letters, digits, '_', '-', and '.'"
        ));
    }
    Ok(value.to_string())
}

fn validate_checkpoint_id(value: &str) -> Result<String, String> {
    Uuid::parse_str(value.trim())
        .map(|id| id.to_string())
        .map_err(|_| "team_id/graph_id must be a UUID returned by the runtime".to_string())
}

fn active_session_id() -> Result<String, String> {
    let session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    if session_id.is_empty() {
        return Err("Team Runtime requires an active session".to_string());
    }
    Ok(session_id)
}

fn active_owner_pid() -> Result<u64, String> {
    super::current_task_owner_pid()
}

pub(super) fn active_identity() -> Result<(String, u64), String> {
    Ok((active_session_id()?, active_owner_pid()?))
}

pub(super) fn now_ms() -> u64 {
    now_unix_ms()
}

pub(super) fn clip_text(value: &str, max_chars: usize) -> String {
    clip_chars(value, max_chars)
}

pub(super) fn validate_identifier(value: &str, kind: &str) -> Result<(), String> {
    validate_id(kind, value).map(|_| ())
}

pub(super) fn checkpoint_base_dir() -> Result<PathBuf, String> {
    let root = current_session_assets_dir()
        .ok_or("Team Runtime requires an active driver session asset directory")?;
    Ok(root.join("agent_orchestration"))
}

pub(super) fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("checkpoint path has no parent: {}", path.display()))?;
    let _guard = TEAM_FILE_LOCK
        .lock()
        .map_err(|error| format!("failed to lock Team Runtime checkpoint: {error}"))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create checkpoint directory: {error}"))?;
    let temp = parent.join(format!(".graph-{}.tmp", Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode checkpoint: {error}"))?;
    fs::write(&temp, bytes)
        .map_err(|error| format!("failed to write checkpoint temp file: {error}"))?;
    if let Err(error) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(format!("failed to commit checkpoint: {error}"));
    }
    Ok(())
}

pub(super) fn recover_orphaned_team_tasks(team: &mut TeamState) {
    for task in team.tasks.values_mut() {
        if task.state == TeamTaskState::Running {
            task.runtime_task_id = None;
            task.runtime_state = Some("orphaned_after_owner_restart".to_string());
            task.state = if team.budget.attempts_used < team.budget.max_total_attempts {
                TeamTaskState::Pending
            } else {
                task.error =
                    Some("team attempt budget exhausted during owner recovery".to_string());
                TeamTaskState::Failed
            };
            task.updated_at_unix_ms = now_unix_ms();
        }
    }
    for member in team.members.values_mut() {
        member.status = MemberStatus::Idle;
        member.active_task_id = None;
    }
    team.updated_at_unix_ms = now_unix_ms();
}

fn checkpoint_dir(kind: &str) -> Result<PathBuf, String> {
    let root = current_session_assets_dir()
        .ok_or("Team Runtime requires an active driver session asset directory")?;
    Ok(root.join("agent_orchestration").join(kind))
}

pub(super) fn save_checkpoint<T: Serialize>(kind: &str, id: &str, value: &T) -> Result<(), String> {
    let id = validate_checkpoint_id(id)?;
    let path = checkpoint_dir(kind)?.join(format!("{id}.json"));
    atomic_write_json(&path, value)
}

pub(super) fn load_checkpoint<T: for<'de> Deserialize<'de>>(
    kind: &str,
    id: &str,
) -> Result<T, String> {
    let id = validate_checkpoint_id(id)?;
    let path = checkpoint_dir(kind)?.join(format!("{id}.json"));
    let _guard = TEAM_FILE_LOCK
        .lock()
        .map_err(|error| format!("failed to lock Team Runtime checkpoint: {error}"))?;
    let bytes = fs::read(&path)
        .map_err(|error| format!("failed to read checkpoint {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode checkpoint {}: {error}", path.display()))
}

fn validate_budget(input: TeamBudgetInput) -> Result<TeamBudget, String> {
    if !(1..=8).contains(&input.max_parallel) {
        return Err("budget.max_parallel must be between 1 and 8".to_string());
    }
    if !(1..=MAX_TEAM_TASKS).contains(&input.max_tasks) {
        return Err(format!(
            "budget.max_tasks must be between 1 and {MAX_TEAM_TASKS}"
        ));
    }
    if input.max_total_attempts < input.max_tasks || input.max_total_attempts > 4_096 {
        return Err("budget.max_total_attempts must be >= max_tasks and <= 4096".to_string());
    }
    if !(1..=MAX_TEAM_MESSAGES).contains(&input.max_messages) {
        return Err(format!(
            "budget.max_messages must be between 1 and {MAX_TEAM_MESSAGES}"
        ));
    }
    Ok(TeamBudget {
        max_parallel: input.max_parallel,
        max_tasks: input.max_tasks,
        max_total_attempts: input.max_total_attempts,
        max_messages: input.max_messages,
        attempts_used: 0,
    })
}

fn member_from_input(input: TeamMemberInput) -> Result<TeamMember, String> {
    let id = validate_id("member.id", &input.id)?;
    if input.role.trim().is_empty() {
        return Err(format!("member {id} requires a non-empty role"));
    }
    if !matches!(
        input.inherit.as_str(),
        "all" | "none" | "cwd" | "skills" | "memory" | "history" | "cwd,skills" | "skills,cwd"
    ) {
        // `prepare_subagent_task` is the final authority and accepts any valid comma list. This
        // early check deliberately keeps persisted manifests simple and deterministic.
        let allowed = ["history", "memory", "cwd", "skills"];
        let mut seen = FxHashSet::default();
        // Separator parity with InheritOptions::from_value (',' '+' '/'): descriptions show
        // defaults in prose style like "cwd+skills", and a comma-only pre-check here would
        // reject values the final authority accepts. Intentionally stricter than the authority
        // on duplicates: from_value treats a repeated token as idempotent, this pre-check
        // fails it, keeping persisted team manifests canonical.
        for part in input.inherit.split(&[',', '+', '/'][..]).map(str::trim) {
            if !allowed.contains(&part) || !seen.insert(part) {
                return Err(format!("member {id} has invalid inherit value"));
            }
        }
    }
    Ok(TeamMember {
        id,
        role: input.role.trim().to_string(),
        agent: input.agent.filter(|value| !value.trim().is_empty()),
        model: input.model.filter(|value| !value.trim().is_empty()),
        inherit: input.inherit,
        capabilities: input.capabilities,
        status: MemberStatus::Idle,
        active_task_id: None,
        last_message_seq: 0,
    })
}

fn create_team(args: &Value) -> Result<String, String> {
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or("create requires a non-empty name")?;
    let goal = args
        .get("goal")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or("create requires a non-empty goal")?;
    let member_inputs: Vec<TeamMemberInput> = serde_json::from_value(
        args.get("members")
            .cloned()
            .ok_or("create requires members")?,
    )
    .map_err(|error| format!("invalid members: {error}"))?;
    if member_inputs.is_empty() || member_inputs.len() > MAX_TEAM_MEMBERS {
        return Err(format!(
            "members must contain 1..={MAX_TEAM_MEMBERS} entries"
        ));
    }
    let budget_input: TeamBudgetInput = match args.get("budget") {
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|error| format!("invalid budget: {error}"))?,
        None => TeamBudgetInput::default(),
    };
    let budget = validate_budget(budget_input)?;
    let mut members = FxHashMap::default();
    for input in member_inputs {
        let member = member_from_input(input)?;
        if members.insert(member.id.clone(), member).is_some() {
            return Err("member ids must be unique".to_string());
        }
    }
    let now = now_unix_ms();
    let team = TeamState {
        version: TEAM_STATE_VERSION,
        id: Uuid::new_v4().to_string(),
        session_id: active_session_id()?,
        owner_pid: active_owner_pid()?,
        name: name.to_string(),
        goal: goal.to_string(),
        lifecycle: TeamLifecycle::Active,
        budget,
        members,
        tasks: FxHashMap::default(),
        messages: Vec::new(),
        next_message_seq: 1,
        created_at_unix_ms: now,
        updated_at_unix_ms: now,
    };
    save_checkpoint("teams", &team.id, &team)?;
    Ok(render_team_status(&team, "created"))
}

fn recover_team_owner(team: &mut TeamState) -> Result<(), String> {
    let session_id = active_session_id()?;
    if team.session_id != session_id {
        return Err("team checkpoint belongs to another session".to_string());
    }
    let owner_pid = active_owner_pid()?;
    if team.owner_pid == owner_pid {
        return Ok(());
    }

    // After an app restart the kernel task registry no longer exists. Restore orphaned
    // running items to pending; attempts are already billed and the budget is not rolled
    // back, so resume can re-dispatch without treating an interrupted execution as success.
    recover_orphaned_team_tasks(team);
    team.owner_pid = owner_pid;
    team.updated_at_unix_ms = now_unix_ms();
    Ok(())
}

fn load_team(id: &str) -> Result<TeamState, String> {
    let mut team: TeamState = load_checkpoint("teams", id)?;
    if team.version != TEAM_STATE_VERSION {
        return Err(format!(
            "unsupported Team Runtime checkpoint version {}",
            team.version
        ));
    }
    recover_team_owner(&mut team)?;
    Ok(team)
}

fn add_team_task(team: &mut TeamState, input: TeamTaskInput) -> Result<String, String> {
    if team.lifecycle != TeamLifecycle::Active {
        return Err("cannot add tasks to an inactive team".to_string());
    }
    if team.tasks.len() >= team.budget.max_tasks {
        return Err("team task budget is full".to_string());
    }
    let id = match input.id {
        Some(value) => validate_id("task.id", &value)?,
        None => format!("task-{}", Uuid::new_v4().simple()),
    };
    if team.tasks.contains_key(&id) {
        return Err(format!("duplicate task id: {id}"));
    }
    if input.title.trim().is_empty() || input.prompt.trim().is_empty() {
        return Err("task title and prompt must be non-empty".to_string());
    }
    if input.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(format!("task prompt exceeds {MAX_PROMPT_CHARS} characters"));
    }
    if let Some(assignee) = input.assignee.as_deref()
        && !team.members.contains_key(assignee)
    {
        return Err(format!("unknown assignee: {assignee}"));
    }
    let mut dependencies = FxHashSet::default();
    for dependency in &input.depends_on {
        if dependency == &id {
            return Err("a task cannot depend on itself".to_string());
        }
        if !team.tasks.contains_key(dependency) {
            return Err(format!("unknown task dependency: {dependency}"));
        }
        if !dependencies.insert(dependency.clone()) {
            return Err(format!("duplicate task dependency: {dependency}"));
        }
    }
    let now = now_unix_ms();
    team.tasks.insert(
        id.clone(),
        TeamTask {
            id: id.clone(),
            title: input.title.trim().to_string(),
            prompt: input.prompt.trim().to_string(),
            assignee: input.assignee,
            depends_on: input.depends_on,
            state: TeamTaskState::Pending,
            lease: None,
            runtime_task_id: None,
            runtime_state: None,
            attempts: 0,
            output: None,
            error: None,
            created_at_unix_ms: now,
            updated_at_unix_ms: now,
        },
    );
    team.updated_at_unix_ms = now;
    Ok(id)
}

fn dependencies_completed(team: &TeamState, task_id: &str) -> bool {
    let Some(task) = team.tasks.get(task_id) else {
        return false;
    };
    task.depends_on.iter().all(|dependency| {
        team.tasks
            .get(dependency)
            .is_some_and(|task| task.state == TeamTaskState::Completed)
    })
}

fn propagate_failed_dependencies(team: &mut TeamState) {
    loop {
        let mut blocked = team
            .tasks
            .iter()
            .filter_map(|(task_id, task)| {
                if !matches!(task.state, TeamTaskState::Pending | TeamTaskState::Claimed) {
                    return None;
                }
                let mut dependencies = task
                    .depends_on
                    .iter()
                    .filter_map(|dependency_id| {
                        let dependency = team.tasks.get(dependency_id)?;
                        match dependency.state {
                            TeamTaskState::Failed => Some(format!("{dependency_id}=failed")),
                            TeamTaskState::Cancelled => Some(format!("{dependency_id}=cancelled")),
                            _ => None,
                        }
                    })
                    .collect::<Vec<_>>();
                dependencies.sort();
                (!dependencies.is_empty()).then(|| (task_id.clone(), dependencies))
            })
            .collect::<Vec<_>>();
        if blocked.is_empty() {
            break;
        }
        blocked.sort_by(|left, right| left.0.cmp(&right.0));
        let now = now_unix_ms();
        for (task_id, dependencies) in blocked {
            let task = team.tasks.get_mut(&task_id).expect("blocked task exists");
            task.state = TeamTaskState::Failed;
            task.lease = None;
            task.runtime_state = Some("blocked_by_dependency".to_string());
            task.error = Some(format!(
                "blocked by unsuccessful dependencies: {}",
                dependencies.join(", ")
            ));
            task.updated_at_unix_ms = now;
        }
        for member in team.members.values_mut() {
            if member.active_task_id.as_deref().is_some_and(|task_id| {
                team.tasks
                    .get(task_id)
                    .is_some_and(|task| task.state.is_terminal())
            }) {
                member.status = MemberStatus::Idle;
                member.active_task_id = None;
            }
        }
        team.updated_at_unix_ms = now;
    }
}

fn expire_leases(team: &mut TeamState, now: u64) {
    let expired = team
        .tasks
        .iter()
        .filter_map(|(id, task)| {
            (task.state == TeamTaskState::Claimed
                && task
                    .lease
                    .as_ref()
                    .is_some_and(|lease| lease.expires_at_unix_ms <= now))
            .then_some(id.clone())
        })
        .collect::<Vec<_>>();
    for task_id in expired {
        let member_id = team
            .tasks
            .get(&task_id)
            .and_then(|task| task.lease.as_ref())
            .map(|lease| lease.member_id.clone());
        if let Some(task) = team.tasks.get_mut(&task_id) {
            task.state = TeamTaskState::Pending;
            task.lease = None;
            task.updated_at_unix_ms = now;
        }
        if let Some(member_id) = member_id
            && let Some(member) = team.members.get_mut(&member_id)
            && member.active_task_id.as_deref() == Some(task_id.as_str())
        {
            member.active_task_id = None;
            member.status = MemberStatus::Idle;
        }
    }
}

fn claim_team_task(
    team: &mut TeamState,
    member_id: &str,
    requested_task_id: Option<&str>,
    lease_secs: u64,
) -> Result<String, String> {
    if team.lifecycle != TeamLifecycle::Active {
        return Err("cannot claim work from an inactive team".to_string());
    }
    let now = now_unix_ms();
    expire_leases(team, now);
    let member = team
        .members
        .get(member_id)
        .ok_or_else(|| format!("unknown member: {member_id}"))?;
    if member.status != MemberStatus::Idle {
        return Err(format!("member {member_id} is not idle"));
    }
    let task_id = if let Some(task_id) = requested_task_id {
        let task = team
            .tasks
            .get(task_id)
            .ok_or_else(|| format!("unknown task: {task_id}"))?;
        if task.state != TeamTaskState::Pending || !dependencies_completed(team, task_id) {
            return Err(format!("task {task_id} is not ready"));
        }
        if task
            .assignee
            .as_deref()
            .is_some_and(|assignee| assignee != member_id)
        {
            return Err(format!("task {task_id} is assigned to another member"));
        }
        task_id.to_string()
    } else {
        let mut ready = team
            .tasks
            .iter()
            .filter_map(|(id, task)| {
                (task.state == TeamTaskState::Pending
                    && dependencies_completed(team, id)
                    && task
                        .assignee
                        .as_deref()
                        .is_none_or(|assignee| assignee == member_id))
                .then_some(id.clone())
            })
            .collect::<Vec<_>>();
        ready.sort();
        ready
            .into_iter()
            .next()
            .ok_or_else(|| format!("no ready task is available for member {member_id}"))?
    };
    let expires_at_unix_ms = now.saturating_add(lease_secs.saturating_mul(1_000));
    let task = team.tasks.get_mut(&task_id).expect("validated task");
    task.state = TeamTaskState::Claimed;
    task.lease = Some(TaskLease {
        member_id: member_id.to_string(),
        expires_at_unix_ms,
    });
    task.updated_at_unix_ms = now;
    let member = team.members.get_mut(member_id).expect("validated member");
    member.status = MemberStatus::Busy;
    member.active_task_id = Some(task_id.clone());
    team.updated_at_unix_ms = now;
    Ok(task_id)
}

fn team_message_prompt(team: &TeamState, member: &TeamMember) -> String {
    let messages = team
        .messages
        .iter()
        .filter(|message| {
            message.seq > member.last_message_seq && (message.to == "*" || message.to == member.id)
        })
        .map(|message| format!("- #{} {}: {}", message.seq, message.from, message.body))
        .collect::<Vec<_>>();
    if messages.is_empty() {
        String::new()
    } else {
        format!(
            "\nTeam mailbox since last dispatch:\n{}\n",
            messages.join("\n")
        )
    }
}

fn dependency_prompt(team: &TeamState, task: &TeamTask) -> String {
    let dependencies = task
        .depends_on
        .iter()
        .filter_map(|dependency| {
            let upstream = team.tasks.get(dependency)?;
            Some(format!(
                "## {} ({})\n{}",
                upstream.title,
                dependency,
                clip_chars(upstream.output.as_deref().unwrap_or("(no output)"), 8_000)
            ))
        })
        .collect::<Vec<_>>();
    if dependencies.is_empty() {
        String::new()
    } else {
        format!(
            "\nCompleted dependency evidence:\n{}\n",
            dependencies.join("\n\n")
        )
    }
}

fn build_team_task_prompt(team: &TeamState, task: &TeamTask, member: &TeamMember) -> String {
    clip_chars(
        &format!(
            "Team: {}\nTeam goal: {}\nYour persistent role: {}\nTask: {}\n\n{}{}\n{}",
            team.name,
            team.goal,
            member.role,
            task.title,
            team_message_prompt(team, member),
            dependency_prompt(team, task),
            task.prompt
        ),
        MAX_PROMPT_CHARS,
    )
}

fn choose_idle_member(team: &TeamState, task: &TeamTask) -> Option<String> {
    if let Some(lease_member) = task.lease.as_ref().map(|lease| lease.member_id.as_str())
        && team
            .members
            .get(lease_member)
            .is_some_and(|member| member.active_task_id.as_deref() == Some(task.id.as_str()))
    {
        return Some(lease_member.to_string());
    }
    if let Some(assignee) = task.assignee.as_deref() {
        return team
            .members
            .get(assignee)
            .filter(|member| member.status == MemberStatus::Idle)
            .map(|member| member.id.clone());
    }
    let mut idle = team
        .members
        .values()
        .filter(|member| member.status == MemberStatus::Idle)
        .map(|member| member.id.clone())
        .collect::<Vec<_>>();
    idle.sort();
    idle.into_iter().next()
}

pub(super) fn dispatch_ready_team_tasks(team: &mut TeamState) -> Result<Vec<String>, String> {
    dispatch_ready_team_tasks_with(
        team,
        |args| {
            let prepared = prepare_subagent_task(&args)?;
            spawn_subagent_kernel_task(&prepared).map(|spawned| spawned.task_id)
        },
        |team| save_checkpoint("teams", &team.id, team),
    )
}

fn dispatch_ready_team_tasks_with(
    team: &mut TeamState,
    mut launch: impl FnMut(Value) -> Result<String, String>,
    mut persist: impl FnMut(&TeamState) -> Result<(), String>,
) -> Result<Vec<String>, String> {
    refresh_team_lifecycle(team);
    if team.lifecycle != TeamLifecycle::Active {
        return Ok(Vec::new());
    }
    expire_leases(team, now_unix_ms());
    let running = team
        .tasks
        .values()
        .filter(|task| task.state == TeamTaskState::Running)
        .count();
    let mut slots = team.budget.max_parallel.saturating_sub(running);
    if slots == 0 {
        return Ok(Vec::new());
    }
    let mut ready = team
        .tasks
        .iter()
        .filter_map(|(id, task)| {
            (matches!(task.state, TeamTaskState::Pending | TeamTaskState::Claimed)
                && dependencies_completed(team, id))
            .then_some(id.clone())
        })
        .collect::<Vec<_>>();
    ready.sort();
    let mut spawned = Vec::new();
    for task_id in ready {
        if slots == 0 || team.budget.attempts_used >= team.budget.max_total_attempts {
            break;
        }
        let task_snapshot = team.tasks.get(&task_id).expect("ready task").clone();
        let Some(member_id) = choose_idle_member(team, &task_snapshot) else {
            continue;
        };
        let member_snapshot = team.members.get(&member_id).expect("chosen member").clone();
        let prompt = build_team_task_prompt(team, &task_snapshot, &member_snapshot);
        let runtime_task_id = launch(json!({
            "description": format!("{}: {}", team.name, task_snapshot.title),
            "prompt": prompt,
            "agent": member_snapshot.agent,
            "model": member_snapshot.model,
            "inherit": member_snapshot.inherit,
        }))?;
        let now = now_unix_ms();
        let task = team.tasks.get_mut(&task_id).expect("ready task");
        task.state = TeamTaskState::Running;
        task.lease = None;
        task.assignee = Some(member_id.clone());
        task.runtime_task_id = Some(runtime_task_id.clone());
        task.runtime_state = Some("ready".to_string());
        task.attempts += 1;
        task.updated_at_unix_ms = now;
        let member = team.members.get_mut(&member_id).expect("chosen member");
        member.status = MemberStatus::Busy;
        member.active_task_id = Some(task_id.clone());
        member.last_message_seq = team.next_message_seq.saturating_sub(1);
        team.budget.attempts_used += 1;
        team.updated_at_unix_ms = now;
        persist(team).map_err(|error| {
            format!(
                "task {task_id} started as {runtime_task_id}, but its Team Runtime checkpoint could not be committed: {error}"
            )
        })?;
        spawned.push(runtime_task_id);
        slots -= 1;
    }
    Ok(spawned)
}

fn integrate_runtime_result(
    runtime_task_id: &str,
    status: &str,
    summary: &str,
) -> Result<(), String> {
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return Ok(());
    };
    let disposition = if status == "completed" {
        "accepted"
    } else {
        "rejected"
    };
    let summary = clip_chars(summary, MAX_RESULT_SUMMARY_CHARS);
    let found = crate::ai::history::integrate_task_evidence(
        context.app_proto.config.history_file.as_path(),
        &active_session_id()?,
        runtime_task_id,
        disposition,
        &summary,
    )
    .map_err(|error| format!("failed to integrate graph task evidence: {error}"))?;
    if !found {
        return Err(format!(
            "runtime task {runtime_task_id} was collected but missing from durable evidence"
        ));
    }
    Ok(())
}

pub(super) fn integrate_graph_task(
    runtime_task_id: &str,
    disposition: &str,
    summary: &str,
) -> Result<(), String> {
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return Ok(());
    };
    let summary = clip_chars(summary, MAX_RESULT_SUMMARY_CHARS);
    let found = crate::ai::history::integrate_task_evidence(
        context.app_proto.config.history_file.as_path(),
        &active_session_id()?,
        runtime_task_id,
        disposition,
        &summary,
    )
    .map_err(|error| format!("failed to integrate graph task evidence: {error}"))?;
    if !found {
        return Err(format!(
            "runtime task {runtime_task_id} was collected but missing from durable evidence"
        ));
    }
    Ok(())
}

pub(super) fn collect_team_results(team: &mut TeamState) -> Result<Vec<String>, String> {
    let running = team
        .tasks
        .iter()
        .filter_map(|(task_id, task)| {
            (task.state == TeamTaskState::Running)
                .then(|| {
                    task.runtime_task_id
                        .clone()
                        .map(|runtime| (task_id.clone(), runtime))
                })
                .flatten()
        })
        .collect::<Vec<_>>();
    let mut collected = Vec::new();
    for (task_id, runtime_task_id) in running {
        match poll_owned_task_result(&runtime_task_id) {
            Ok(OwnedTaskPoll::Pending { state }) => {
                if let Some(task) = team.tasks.get_mut(&task_id) {
                    task.runtime_state = Some(state);
                }
            }
            Ok(OwnedTaskPoll::Terminal { result, rendered }) => {
                let now = now_unix_ms();
                let member_id = team
                    .tasks
                    .get(&task_id)
                    .and_then(|task| task.assignee.clone());
                let summary = if result.output.trim().is_empty() {
                    result
                        .error
                        .as_deref()
                        .unwrap_or("subagent returned no output")
                } else {
                    result.output.as_str()
                };
                // A failed history integration must not propagate: poll_owned_task_result
                // already consumed the TASK_REGISTRY entry and persisted the evidence. If
                // `?` propagated here, the checkpoint would not be saved, the task on disk
                // would stay Running, and the next advance would hang forever because the
                // task is no longer in the registry (even cancel_team would fail at
                // collect_team_results).
                // So integration is best-effort: the terminal state is already committed
                // in place and a failure only appends a note.
                if let Err(error) = integrate_runtime_result(&runtime_task_id, &result.status, summary) {
                    collected.push(format!("task {task_id} evidence integration failed: {error}"));
                }
                if let Some(task) = team.tasks.get_mut(&task_id) {
                    task.state = if result.status == "completed" {
                        TeamTaskState::Completed
                    } else if result.status == "cancelled" {
                        TeamTaskState::Cancelled
                    } else {
                        TeamTaskState::Failed
                    };
                    task.output = (!result.output.trim().is_empty()).then_some(result.output);
                    task.error = result.error;
                    task.runtime_state = Some(result.status.clone());
                    task.updated_at_unix_ms = now;
                }
                if let Some(member_id) = member_id
                    && let Some(member) = team.members.get_mut(&member_id)
                {
                    member.status = MemberStatus::Idle;
                    member.active_task_id = None;
                }
                collected.push(clip_chars(&rendered, 8_000));
                team.updated_at_unix_ms = now;
            }
            Err(error) => {
                // Mid-round errors must never propagate: when poll fails (task lost /
                // ownership changed), mark the task failed in place and clear the task
                // reference. If `?` propagated, the checkpoint would not be saved, the
                // task on disk would still carry runtime_task_id, and the next advance
                // would re-poll the same lost task and hang forever (even cancel_team
                // would fail at collect_team_results).
                let now = now_unix_ms();
                let member_id = team
                    .tasks
                    .get(&task_id)
                    .and_then(|task| task.assignee.clone());
                if let Some(task) = team.tasks.get_mut(&task_id) {
                    task.state = TeamTaskState::Failed;
                    task.runtime_task_id = None;
                    task.runtime_state = Some("failed".to_string());
                    task.error = Some(format!("lost team task {task_id}: {error}"));
                    task.updated_at_unix_ms = now;
                }
                if let Some(member_id) = member_id
                    && let Some(member) = team.members.get_mut(&member_id)
                {
                    member.status = MemberStatus::Idle;
                    member.active_task_id = None;
                }
                collected.push(format!("task {task_id} poll failed: {error}"));
                team.updated_at_unix_ms = now;
            }
        }
    }
    refresh_team_lifecycle(team);
    Ok(collected)
}

fn refresh_team_lifecycle(team: &mut TeamState) {
    if team.lifecycle != TeamLifecycle::Active || team.tasks.is_empty() {
        return;
    }
    propagate_failed_dependencies(team);
    if team.tasks.values().all(|task| task.state.is_terminal()) {
        team.lifecycle = if team
            .tasks
            .values()
            .all(|task| task.state == TeamTaskState::Completed)
        {
            TeamLifecycle::Completed
        } else {
            TeamLifecycle::Failed
        };
    }
}

fn complete_team_task(
    team: &mut TeamState,
    member_id: &str,
    task_id: &str,
    status: &str,
    output: Option<String>,
    error: Option<String>,
) -> Result<(), String> {
    if !matches!(status, "completed" | "failed" | "cancelled") {
        return Err("status must be completed, failed, or cancelled".to_string());
    }
    let task = team
        .tasks
        .get(task_id)
        .ok_or_else(|| format!("unknown task: {task_id}"))?;
    let owned = task.assignee.as_deref() == Some(member_id)
        || task.lease.as_ref().map(|lease| lease.member_id.as_str()) == Some(member_id);
    if !owned || !matches!(task.state, TeamTaskState::Claimed | TeamTaskState::Running) {
        return Err(format!(
            "task {task_id} is not active for member {member_id}"
        ));
    }
    if task.runtime_task_id.is_some() {
        return Err(
            "running subagent tasks must be completed by advance, not manual complete".to_string(),
        );
    }
    let now = now_unix_ms();
    let task = team.tasks.get_mut(task_id).expect("validated task");
    task.state = match status {
        "completed" => TeamTaskState::Completed,
        "cancelled" => TeamTaskState::Cancelled,
        _ => TeamTaskState::Failed,
    };
    task.output = output;
    task.error = error;
    task.lease = None;
    task.updated_at_unix_ms = now;
    let member = team.members.get_mut(member_id).expect("validated member");
    member.status = MemberStatus::Idle;
    member.active_task_id = None;
    team.updated_at_unix_ms = now;
    refresh_team_lifecycle(team);
    Ok(())
}

fn send_team_message(
    team: &mut TeamState,
    from: &str,
    to: &str,
    body: &str,
) -> Result<u64, String> {
    if team.messages.len() >= team.budget.max_messages {
        return Err("team message budget is full".to_string());
    }
    if from != "parent" && !team.members.contains_key(from) {
        return Err(format!("unknown message sender: {from}"));
    }
    if to != "*" && to != "parent" && !team.members.contains_key(to) {
        return Err(format!("unknown message recipient: {to}"));
    }
    if body.trim().is_empty() || body.chars().count() > 16_000 {
        return Err("message body must be 1..=16000 characters".to_string());
    }
    let seq = team.next_message_seq;
    team.next_message_seq = team.next_message_seq.saturating_add(1);
    let now = now_unix_ms();
    team.messages.push(TeamMessage {
        seq,
        from: from.to_string(),
        to: to.to_string(),
        body: body.trim().to_string(),
        created_at_unix_ms: now,
    });
    team.updated_at_unix_ms = now;
    Ok(seq)
}

fn cancel_team(team: &mut TeamState, reason: &str) -> Result<Vec<String>, String> {
    let runtime_ids = team
        .tasks
        .values()
        .filter_map(|task| task.runtime_task_id.clone())
        .collect::<Vec<_>>();
    let mut notes = Vec::new();
    if !runtime_ids.is_empty() {
        notes.push(execute_task_cancel(&json!({
            "task_ids": runtime_ids,
            "reason": reason,
        }))?);
        notes.extend(collect_team_results(team)?);
    }
    for task in team.tasks.values_mut() {
        if !task.state.is_terminal() {
            task.state = TeamTaskState::Cancelled;
            task.error = Some(reason.to_string());
            task.updated_at_unix_ms = now_unix_ms();
        }
    }
    for member in team.members.values_mut() {
        member.status = MemberStatus::Idle;
        member.active_task_id = None;
    }
    team.lifecycle = TeamLifecycle::Cancelled;
    team.updated_at_unix_ms = now_unix_ms();
    Ok(notes)
}

fn render_team_status(team: &TeamState, event: &str) -> String {
    let mut members = team.members.values().cloned().collect::<Vec<_>>();
    members.sort_by(|left, right| left.id.cmp(&right.id));
    let mut tasks = team.tasks.values().cloned().collect::<Vec<_>>();
    tasks.sort_by(|left, right| left.id.cmp(&right.id));
    let task_views = tasks
        .into_iter()
        .map(|task| {
            json!({
                "id": task.id,
                "title": task.title,
                "assignee": task.assignee,
                "depends_on": task.depends_on,
                "state": task.state,
                "runtime_task_id": task.runtime_task_id,
                "runtime_state": task.runtime_state,
                "attempts": task.attempts,
                "output_preview": task.output.as_deref().map(|value| clip_chars(value, 2_000)),
                "error": task.error,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string_pretty(&json!({
        "event": event,
        "team_id": team.id,
        "name": team.name,
        "goal": team.goal,
        "lifecycle": team.lifecycle,
        "budget": team.budget,
        "members": members,
        "tasks": task_views,
        "messages": team.messages.iter().rev().take(20).cloned().collect::<Vec<_>>(),
        "checkpoint": checkpoint_dir("teams").ok().map(|dir| dir.join(format!("{}.json", team.id))),
    }))
    .unwrap_or_else(|error| format!("failed to render team status: {error}"))
}

fn required_str<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
    args.get(name)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing non-empty '{name}'"))
}

fn execute_manage_team(args: &Value) -> Result<String, String> {
    super::ensure_top_level_task_orchestration("manage_team")?;
    let action = required_str(args, "action")?;
    if action == "create" {
        return create_team(args);
    }
    let team_id = required_str(args, "team_id")?;
    let mut team = load_team(team_id)?;
    let mut event = action.to_string();
    let mut notes = Vec::new();
    match action {
        "add_task" => {
            let input: TeamTaskInput =
                serde_json::from_value(args.get("task").cloned().ok_or("add_task requires task")?)
                    .map_err(|error| format!("invalid task: {error}"))?;
            let task_id = add_team_task(&mut team, input)?;
            event = format!("task_added:{task_id}");
        }
        "claim" => {
            let member_id = required_str(args, "member_id")?;
            let task_id = args.get("task_id").and_then(Value::as_str);
            let lease_secs = args
                .get("lease_secs")
                .and_then(Value::as_u64)
                .unwrap_or(DEFAULT_LEASE_SECS)
                .clamp(1, 86_400);
            let claimed = claim_team_task(&mut team, member_id, task_id, lease_secs)?;
            event = format!("task_claimed:{claimed}");
        }
        "dispatch" => {
            let spawned = dispatch_ready_team_tasks(&mut team)?;
            event = format!("dispatched:{}", spawned.len());
        }
        "advance" => {
            notes.extend(collect_team_results(&mut team)?);
            let dispatch = args
                .get("dispatch")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let spawned = if dispatch {
                dispatch_ready_team_tasks(&mut team)?
            } else {
                Vec::new()
            };
            event = format!(
                "advanced:collected={},spawned={}",
                notes.len(),
                spawned.len()
            );
        }
        "complete" => {
            let member_id = required_str(args, "member_id")?;
            let task_id = required_str(args, "task_id")?;
            let status = required_str(args, "status")?;
            complete_team_task(
                &mut team,
                member_id,
                task_id,
                status,
                args.get("output")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                args.get("error")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            )?;
            event = format!("task_completed:{task_id}:{status}");
        }
        "send_message" => {
            let seq = send_team_message(
                &mut team,
                required_str(args, "from")?,
                required_str(args, "to")?,
                required_str(args, "body")?,
            )?;
            event = format!("message_sent:{seq}");
        }
        "status" => {
            expire_leases(&mut team, now_unix_ms());
            refresh_team_lifecycle(&mut team);
        }
        "cancel" => {
            notes.extend(cancel_team(
                &mut team,
                args.get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("team cancelled by parent"),
            )?);
        }
        other => {
            return Err(format!(
                "unknown action '{other}'; expected create, add_task, claim, dispatch, advance, complete, send_message, status, or cancel"
            ));
        }
    }
    save_checkpoint("teams", &team.id, &team)?;
    let mut rendered = render_team_status(&team, &event);
    if !notes.is_empty() {
        rendered.push_str("\n\nRuntime results:\n");
        rendered.push_str(&notes.join("\n\n---\n\n"));
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_team() -> TeamState {
        let now = now_unix_ms();
        TeamState {
            version: TEAM_STATE_VERSION,
            id: Uuid::new_v4().to_string(),
            session_id: "session".to_string(),
            owner_pid: 1,
            name: "team".to_string(),
            goal: "goal".to_string(),
            lifecycle: TeamLifecycle::Active,
            budget: TeamBudget::default(),
            members: FxHashMap::from_iter([(
                "worker".to_string(),
                TeamMember {
                    id: "worker".to_string(),
                    role: "worker".to_string(),
                    agent: None,
                    model: None,
                    inherit: "none".to_string(),
                    capabilities: Vec::new(),
                    status: MemberStatus::Idle,
                    active_task_id: None,
                    last_message_seq: 0,
                },
            )]),
            tasks: FxHashMap::default(),
            messages: Vec::new(),
            next_message_seq: 1,
            created_at_unix_ms: now,
            updated_at_unix_ms: now,
        }
    }

    #[test]
    fn dependency_blocks_claim_until_upstream_completes() {
        let mut team = sample_team();
        let first = add_team_task(
            &mut team,
            TeamTaskInput {
                id: Some("first".to_string()),
                title: "first".to_string(),
                prompt: "do first".to_string(),
                assignee: None,
                depends_on: Vec::new(),
            },
        )
        .unwrap();
        add_team_task(
            &mut team,
            TeamTaskInput {
                id: Some("second".to_string()),
                title: "second".to_string(),
                prompt: "do second".to_string(),
                assignee: None,
                depends_on: vec![first.clone()],
            },
        )
        .unwrap();

        assert!(claim_team_task(&mut team, "worker", Some("second"), 30).is_err());
        team.tasks.get_mut(&first).unwrap().state = TeamTaskState::Completed;
        assert_eq!(
            claim_team_task(&mut team, "worker", Some("second"), 30).unwrap(),
            "second"
        );
    }

    #[test]
    fn failed_dependency_cascades_and_terminates_team() {
        let mut team = sample_team();
        for (id, dependencies) in [
            ("first", Vec::new()),
            ("second", vec!["first".to_string()]),
            ("third", vec!["second".to_string()]),
        ] {
            add_team_task(
                &mut team,
                TeamTaskInput {
                    id: Some(id.to_string()),
                    title: id.to_string(),
                    prompt: id.to_string(),
                    assignee: None,
                    depends_on: dependencies,
                },
            )
            .unwrap();
        }
        team.tasks.get_mut("first").unwrap().state = TeamTaskState::Failed;

        refresh_team_lifecycle(&mut team);

        assert_eq!(team.tasks["second"].state, TeamTaskState::Failed);
        assert_eq!(team.tasks["third"].state, TeamTaskState::Failed);
        assert!(
            team.tasks["second"]
                .error
                .as_deref()
                .is_some_and(|error| error.contains("first=failed"))
        );
        assert!(
            team.tasks["third"]
                .error
                .as_deref()
                .is_some_and(|error| error.contains("second=failed"))
        );
        assert_eq!(team.lifecycle, TeamLifecycle::Failed);
    }

    #[test]
    fn partial_dispatch_persists_each_successful_runtime_mapping() {
        let mut team = sample_team();
        team.budget.max_parallel = 2;
        let mut second_member = team.members["worker"].clone();
        second_member.id = "worker-2".to_string();
        team.members.insert(second_member.id.clone(), second_member);
        for id in ["first", "second"] {
            add_team_task(
                &mut team,
                TeamTaskInput {
                    id: Some(id.to_string()),
                    title: id.to_string(),
                    prompt: id.to_string(),
                    assignee: None,
                    depends_on: Vec::new(),
                },
            )
            .unwrap();
        }
        let mut launches = 0;
        let mut checkpoints = Vec::new();

        let error = dispatch_ready_team_tasks_with(
            &mut team,
            |_| {
                launches += 1;
                if launches == 1 {
                    Ok("runtime-first".to_string())
                } else {
                    Err("task registry capacity reached".to_string())
                }
            },
            |team| {
                checkpoints.push(team.clone());
                Ok(())
            },
        )
        .unwrap_err();

        assert!(error.contains("task registry capacity reached"));
        assert_eq!(launches, 2);
        assert_eq!(checkpoints.len(), 1);
        let checkpoint = &checkpoints[0];
        assert_eq!(checkpoint.tasks["first"].state, TeamTaskState::Running);
        assert_eq!(
            checkpoint.tasks["first"].runtime_task_id.as_deref(),
            Some("runtime-first")
        );
        assert_eq!(checkpoint.tasks["second"].state, TeamTaskState::Pending);
    }

    #[test]
    fn expired_claim_returns_task_and_member_to_idle() {
        let mut team = sample_team();
        add_team_task(
            &mut team,
            TeamTaskInput {
                id: Some("work".to_string()),
                title: "work".to_string(),
                prompt: "work".to_string(),
                assignee: None,
                depends_on: Vec::new(),
            },
        )
        .unwrap();
        claim_team_task(&mut team, "worker", Some("work"), 1).unwrap();
        let expiry = team.tasks["work"]
            .lease
            .as_ref()
            .unwrap()
            .expires_at_unix_ms;
        expire_leases(&mut team, expiry);
        assert_eq!(team.tasks["work"].state, TeamTaskState::Pending);
        assert_eq!(team.members["worker"].status, MemberStatus::Idle);
    }

    #[test]
    fn checkpoint_id_rejects_path_traversal() {
        assert!(validate_checkpoint_id("../../other").is_err());
    }

    #[test]
    fn budget_validation_rejects_out_of_range_values() {
        let valid = TeamBudgetInput {
            max_parallel: 2,
            max_tasks: 4,
            max_total_attempts: 8,
            max_messages: 16,
        };
        assert!(validate_budget(valid).is_ok());

        let mut input = TeamBudgetInput {
            max_parallel: 0,
            max_tasks: 4,
            max_total_attempts: 8,
            max_messages: 16,
        };
        assert!(validate_budget(input.clone()).is_err());
        input.max_parallel = 9;
        assert!(validate_budget(input).is_err());

        let mut input = TeamBudgetInput {
            max_parallel: 2,
            max_tasks: 0,
            max_total_attempts: 8,
            max_messages: 16,
        };
        assert!(validate_budget(input.clone()).is_err());
        input.max_tasks = MAX_TEAM_TASKS + 1;
        assert!(validate_budget(input).is_err());

        // max_total_attempts must be >= max_tasks and <= 4096
        let mut input = TeamBudgetInput {
            max_parallel: 2,
            max_tasks: 8,
            max_total_attempts: 4,
            max_messages: 16,
        };
        assert!(validate_budget(input.clone()).is_err());
        input.max_total_attempts = 4_097;
        assert!(validate_budget(input).is_err());

        let mut input = TeamBudgetInput {
            max_parallel: 2,
            max_tasks: 4,
            max_total_attempts: 8,
            max_messages: 0,
        };
        assert!(validate_budget(input.clone()).is_err());
        input.max_messages = MAX_TEAM_MESSAGES + 1;
        assert!(validate_budget(input).is_err());
    }

    #[test]
    fn identifier_validation_rejects_illegal_inputs() {
        assert!(validate_id("id", "ok_id-1.2").is_ok());
        assert!(validate_id("id", "").is_err());
        assert!(validate_id("id", "  ").is_err());
        assert!(validate_id("id", &"x".repeat(97)).is_err());
        assert!(validate_id("id", "has space").is_err());
        assert!(validate_id("id", "slash/name").is_err());
    }

    #[test]
    fn member_inherit_rejects_unknown_parts() {
        let base = TeamMemberInput {
            id: "worker".to_string(),
            role: "worker".to_string(),
            agent: None,
            model: None,
            inherit: "none".to_string(),
            capabilities: Vec::new(),
        };
        assert!(member_from_input(base.clone()).is_ok());

        let mut bad = base.clone();
        bad.inherit = "cwd,telepathy".to_string();
        assert!(member_from_input(bad).is_err());

        let mut duplicate = base.clone();
        duplicate.inherit = "cwd,cwd".to_string();
        assert!(member_from_input(duplicate).is_err());

        let mut short = base;
        short.role = " ".to_string();
        assert!(member_from_input(short).is_err());
    }

    #[test]
    fn member_inherit_accepts_plus_and_slash_separators() {
        // Must stay in sync with InheritOptions::from_value: 'cwd+skills' (prose style from
        // tool descriptions) parses like 'cwd,skills' here too.
        let base = TeamMemberInput {
            id: "worker".to_string(),
            role: "worker".to_string(),
            agent: None,
            model: None,
            inherit: "none".to_string(),
            capabilities: Vec::new(),
        };
        let mut ok = base.clone();
        ok.inherit = "cwd+skills".to_string();
        assert!(member_from_input(ok).is_ok());
        let mut ok2 = base.clone();
        ok2.inherit = "history/cwd".to_string();
        assert!(member_from_input(ok2).is_ok());
        // Duplicates are still caught across the lenient separators.
        let mut dup = base;
        dup.inherit = "cwd+cwd".to_string();
        assert!(member_from_input(dup).is_err());
    }

    #[test]
    fn task_prompt_length_is_enforced() {
        let mut team = sample_team();
        let input = TeamTaskInput {
            id: None,
            title: "title".to_string(),
            prompt: "x".repeat(MAX_PROMPT_CHARS + 1),
            assignee: None,
            depends_on: Vec::new(),
        };
        let error = add_team_task(&mut team, input).unwrap_err();
        assert!(error.contains("exceeds"));
        assert!(team.tasks.is_empty());
    }

    #[test]
    fn message_validation_rejects_bad_senders_and_overlong_bodies() {
        let mut team = sample_team();
        assert!(
            send_team_message(&mut team, "ghost", "worker", "hi").is_err(),
            "unknown sender rejected"
        );
        assert!(
            send_team_message(&mut team, "parent", "ghost", "hi").is_err(),
            "unknown recipient rejected"
        );
        assert!(send_team_message(&mut team, "parent", "*", "  ").is_err());
        assert!(
            send_team_message(&mut team, "parent", "*", &"x".repeat(16_001)).is_err(),
            "overlong body rejected"
        );

        team.budget.max_messages = 0;
        assert!(
            send_team_message(&mut team, "parent", "*", "hi").is_err(),
            "full message budget rejected"
        );
    }

    #[test]
    fn cancel_team_terminates_without_runtime_tasks() {
        let mut team = sample_team();
        add_team_task(
            &mut team,
            TeamTaskInput {
                id: Some("work".to_string()),
                title: "work".to_string(),
                prompt: "work".to_string(),
                assignee: None,
                depends_on: Vec::new(),
            },
        )
        .unwrap();

        let notes = cancel_team(&mut team, "no longer needed").unwrap();
        assert!(notes.is_empty(), "no runtime tasks to cancel");
        assert_eq!(team.lifecycle, TeamLifecycle::Cancelled);
        assert_eq!(team.tasks["work"].state, TeamTaskState::Cancelled);
        assert_eq!(
            team.tasks["work"].error.as_deref(),
            Some("no longer needed")
        );
        assert_eq!(team.members["worker"].status, MemberStatus::Idle);
    }
}
