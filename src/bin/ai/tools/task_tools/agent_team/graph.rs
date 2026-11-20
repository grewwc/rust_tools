//! Static Agent Graph and dynamic Graph-of-Agents orchestration.
//!
//! Both modes own only topology, state transitions, checkpoints, and prompt assembly.
//! Execution reuses the parent `task_*` / AIOS kernel protocols for cancellation,
//! evidence, and owner/session isolation.

use super::{
    MAX_PROMPT_CHARS, MemberStatus, StatusViewOptions, TeamBudget, TeamLifecycle, TeamMember,
    TeamState, TeamTask, TeamTaskState, active_identity, atomic_write_json, checkpoint_base_dir,
    clip_text, integrate_graph_task, now_ms, recover_orphaned_team_tasks, validate_identifier,
};
use crate::ai::tools::registry::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec,
};
use rust_tools::commonw::FastMap;
use rust_tools::cw::graph::DirectedGraph;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

use super::super::{
    OwnedTaskPoll, execute_task_cancel, poll_owned_task_result, prepare_subagent_task,
    spawn_subagent_kernel_task,
};

const GRAPH_STATE_VERSION: u32 = 1;
const MAX_GRAPH_NODES: usize = 128;
const MAX_GRAPH_EDGES: usize = 1_024;
const MAX_DYNAMIC_CANDIDATES: usize = 16;

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "run_agent_graph",
        description: "",
        execute: execute_run_agent_graph,
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "run_agent_graph",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::AfterIntegration,
        counts_toward_precision_inline_budget: false,
    },
});

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum GraphLifecycle {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentGraphCheckpoint {
    version: u32,
    #[serde(default, skip_serializing_if = "super::revision_is_zero")]
    revision: u64,
    #[serde(default, skip_serializing_if = "FastMap::is_empty")]
    node_revisions: FastMap<String, u64>,
    graph_id: String,
    session_id: String,
    owner_pid: u64,
    name: String,
    question: String,
    lifecycle: GraphLifecycle,
    max_parallel: usize,
    team: TeamState,
    run: GraphRun,
    final_result: Option<String>,
    created_at_unix_ms: u64,
    updated_at_unix_ms: u64,
}

impl AgentGraphCheckpoint {
    fn configure_revision_tracking(&mut self, enabled: bool) {
        self.team.configure_revision_tracking(enabled);
        if !enabled {
            self.revision = 0;
            self.node_revisions.clear();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum GraphRun {
    Static(StaticGraphRun),
    Dynamic(DynamicGraphRun),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum NodeStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
    Cancelled,
}

impl NodeStatus {
    fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Skipped | Self::Cancelled
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct StaticNodeSpec {
    id: String,
    description: String,
    prompt: String,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "default_inherit")]
    inherit: String,
    #[serde(default)]
    response_schema: Option<Value>,
    #[serde(default)]
    output_key: Option<String>,
    #[serde(default)]
    reducer: Reducer,
    #[serde(default)]
    max_retries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StaticEdgeSpec {
    from: String,
    to: String,
    #[serde(default)]
    condition: EdgeCondition,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum EdgeCondition {
    #[default]
    Always,
    OnSuccess,
    OnFailure,
    OutputContains {
        text: String,
    },
    JsonPointerEquals {
        pointer: String,
        value: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Reducer {
    #[default]
    Replace,
    AppendArray,
    MergeObject,
    Concat,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct StaticNodeRuntime {
    spec: StaticNodeSpec,
    status: NodeStatus,
    runtime_task_id: Option<String>,
    team_task_id: Option<String>,
    attempts: usize,
    output: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StaticGraphRun {
    nodes: FxHashMap<String, StaticNodeRuntime>,
    edges: Vec<StaticEdgeSpec>,
    state: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct DynamicCandidateSpec {
    id: String,
    role: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "default_inherit")]
    inherit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DynamicPolicy {
    #[serde(default = "default_min_selected")]
    min_selected: usize,
    #[serde(default = "default_max_selected")]
    max_selected: usize,
    #[serde(default = "default_relevance_threshold")]
    relevance_threshold: f64,
    #[serde(default = "default_edge_threshold")]
    edge_threshold: f64,
    #[serde(default)]
    pool: PoolStrategy,
    #[serde(default)]
    pool_agent: Option<String>,
    #[serde(default)]
    pool_model: Option<String>,
}

impl Default for DynamicPolicy {
    fn default() -> Self {
        Self {
            min_selected: default_min_selected(),
            max_selected: default_max_selected(),
            relevance_threshold: default_relevance_threshold(),
            edge_threshold: default_edge_threshold(),
            pool: PoolStrategy::default(),
            pool_agent: None,
            pool_model: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PoolStrategy {
    Max,
    #[default]
    Judge,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum DynamicPhase {
    Initial,
    Scoring,
    Forward,
    Reverse,
    Pooling,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DynamicEdge {
    from: String,
    to: String,
    score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct DynamicCandidateRuntime {
    spec: DynamicCandidateSpec,
    task_id: Option<String>,
    team_task_id: Option<String>,
    attempts: usize,
    initial: Option<String>,
    scores: FxHashMap<String, f64>,
    score_complete: bool,
    relevance: f64,
    selected: bool,
    forward: Option<String>,
    reverse: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DynamicGraphRun {
    phase: DynamicPhase,
    policy: DynamicPolicy,
    candidates: FxHashMap<String, DynamicCandidateRuntime>,
    edges: Vec<DynamicEdge>,
    pool_task_id: Option<String>,
    pool_team_task_id: Option<String>,
    result: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GraphToolInput {
    action: String,
    #[serde(default)]
    graph_id: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    question: Option<String>,
    #[serde(default)]
    max_parallel: Option<usize>,
    #[serde(default)]
    static_graph: Option<StaticGraphInput>,
    #[serde(default)]
    dynamic_graph: Option<DynamicGraphInput>,
}

#[derive(Debug, Deserialize)]
struct StaticGraphInput {
    nodes: Vec<StaticNodeSpec>,
    #[serde(default)]
    edges: Vec<StaticEdgeSpec>,
    #[serde(default = "default_graph_state")]
    initial_state: Value,
}

#[derive(Debug, Deserialize)]
struct DynamicGraphInput {
    candidates: Vec<DynamicCandidateSpec>,
    #[serde(default)]
    policy: DynamicPolicy,
}

fn default_inherit() -> String {
    "cwd,skills".to_string()
}

fn default_min_selected() -> usize {
    2
}

fn default_max_selected() -> usize {
    4
}

fn default_relevance_threshold() -> f64 {
    0.2
}

fn default_edge_threshold() -> f64 {
    0.35
}

fn default_graph_state() -> Value {
    Value::Object(Map::new())
}

pub(super) fn execute_run_agent_graph(args: &Value) -> Result<String, String> {
    super::super::ensure_top_level_task_orchestration("run_agent_graph")?;
    let optimizations = crate::ai::config::agent_team_optimizations();
    let optimizations_enabled = optimizations.optimizations_enabled();
    let input: GraphToolInput = serde_json::from_value(args.clone())
        .map_err(|error| format!("invalid run_agent_graph arguments: {error}"))?;
    match input.action.as_str() {
        "start" => start_graph(input, optimizations_enabled),
        "advance" => mutate_graph(input.graph_id.as_deref(), optimizations_enabled, |graph| {
            advance_graph(graph)?;
            Ok(graph_view(graph))
        }),
        "status" => {
            let options = StatusViewOptions::for_optimizations(args, optimizations_enabled)?;
            let graph = load_graph(
                required_graph_id(input.graph_id.as_deref())?,
                optimizations_enabled,
            )?;
            render_json(&graph_view_with_options(&graph, options))
        }
        "cancel" => mutate_graph(input.graph_id.as_deref(), optimizations_enabled, |graph| {
            cancel_graph(graph)?;
            Ok(graph_view(graph))
        }),
        other => Err(format!(
            "unsupported run_agent_graph action {other:?}; expected start, advance, status, or cancel"
        )),
    }
}

fn start_graph(input: GraphToolInput, optimizations_enabled: bool) -> Result<String, String> {
    let (session_id, owner_pid) = active_identity()?;
    let name = input.name.unwrap_or_else(|| "agent-graph".to_string());
    let question = required_non_empty(input.question.as_deref(), "question")?.to_string();
    let max_parallel = input.max_parallel.unwrap_or(4).clamp(1, 8);
    let graph_id = Uuid::new_v4().to_string();
    let now = now_ms();

    let (run, members) = match input.mode.as_deref() {
        Some("static") => build_static_run(
            input
                .static_graph
                .ok_or_else(|| "static mode requires static_graph".to_string())?,
        )?,
        Some("dynamic") => build_dynamic_run(
            input
                .dynamic_graph
                .ok_or_else(|| "dynamic mode requires dynamic_graph".to_string())?,
        )?,
        Some(other) => return Err(format!("unsupported graph mode {other:?}")),
        None => return Err("start requires mode=static or mode=dynamic".to_string()),
    };

    let team_id = Uuid::new_v4().to_string();
    let team = TeamState {
        version: super::TEAM_STATE_VERSION,
        revision: u64::from(optimizations_enabled),
        revision_tracking_enabled: optimizations_enabled,
        id: team_id,
        session_id: session_id.clone(),
        owner_pid,
        name: format!("graph:{name}"),
        goal: question.clone(),
        lifecycle: TeamLifecycle::Active,
        budget: TeamBudget {
            max_parallel,
            max_tasks: 512,
            max_total_attempts: 1_024,
            max_messages: 512,
            attempts_used: 0,
        },
        members,
        tasks: FxHashMap::default(),
        messages: Vec::new(),
        next_message_seq: 1,
        created_at_unix_ms: now,
        updated_at_unix_ms: now,
    };
    let mut checkpoint = AgentGraphCheckpoint {
        version: GRAPH_STATE_VERSION,
        revision: 0,
        node_revisions: FastMap::default(),
        graph_id,
        session_id,
        owner_pid,
        name,
        question,
        lifecycle: GraphLifecycle::Running,
        max_parallel,
        team,
        run,
        final_result: None,
        created_at_unix_ms: now,
        updated_at_unix_ms: now,
    };
    let previous = GraphRevisionSnapshot::Initial;
    advance_graph(&mut checkpoint)?;
    record_graph_revision(&mut checkpoint, &previous);
    save_graph(&checkpoint)?;
    render_json(&graph_view(&checkpoint))
}

fn build_static_run(
    input: StaticGraphInput,
) -> Result<(GraphRun, FxHashMap<String, TeamMember>), String> {
    if input.nodes.is_empty() || input.nodes.len() > MAX_GRAPH_NODES {
        return Err(format!(
            "static graph must contain 1..={MAX_GRAPH_NODES} nodes"
        ));
    }
    if input.edges.len() > MAX_GRAPH_EDGES {
        return Err(format!("static graph exceeds {MAX_GRAPH_EDGES} edges"));
    }

    let mut nodes = FxHashMap::default();
    let mut members = FxHashMap::default();
    for node in input.nodes {
        validate_identifier(&node.id, "node id")?;
        if node.prompt.trim().is_empty() || node.description.trim().is_empty() {
            return Err(format!("node {} requires description and prompt", node.id));
        }
        if nodes.contains_key(&node.id) {
            return Err(format!("duplicate graph node id: {}", node.id));
        }
        members.insert(
            node.id.clone(),
            TeamMember {
                id: node.id.clone(),
                role: node.description.clone(),
                agent: node.agent.clone(),
                model: node.model.clone(),
                inherit: node.inherit.clone(),
                capabilities: vec!["static_graph_node".to_string()],
                status: MemberStatus::Idle,
                active_task_id: None,
                last_message_seq: 0,
            },
        );
        nodes.insert(
            node.id.clone(),
            StaticNodeRuntime {
                spec: node,
                status: NodeStatus::Pending,
                runtime_task_id: None,
                team_task_id: None,
                attempts: 0,
                output: None,
                error: None,
            },
        );
    }
    validate_static_graph(&nodes, &input.edges)?;
    Ok((
        GraphRun::Static(StaticGraphRun {
            nodes,
            edges: input.edges,
            state: input.initial_state,
        }),
        members,
    ))
}

fn build_dynamic_run(
    input: DynamicGraphInput,
) -> Result<(GraphRun, FxHashMap<String, TeamMember>), String> {
    if input.candidates.len() < 2 || input.candidates.len() > MAX_DYNAMIC_CANDIDATES {
        return Err(format!(
            "dynamic graph requires 2..={MAX_DYNAMIC_CANDIDATES} candidates"
        ));
    }
    let mut policy = input.policy;
    policy.min_selected = policy.min_selected.clamp(2, input.candidates.len());
    policy.max_selected = policy
        .max_selected
        .clamp(policy.min_selected, input.candidates.len());
    if !(0.0..=1.0).contains(&policy.relevance_threshold)
        || !(0.0..=1.0).contains(&policy.edge_threshold)
    {
        return Err("dynamic thresholds must be in [0, 1]".to_string());
    }

    let mut candidates = FxHashMap::default();
    let mut members = FxHashMap::default();
    for candidate in input.candidates {
        validate_identifier(&candidate.id, "candidate id")?;
        if candidate.role.trim().is_empty() {
            return Err(format!("candidate {} requires a role", candidate.id));
        }
        if candidates.contains_key(&candidate.id) {
            return Err(format!("duplicate dynamic candidate id: {}", candidate.id));
        }
        members.insert(
            candidate.id.clone(),
            TeamMember {
                id: candidate.id.clone(),
                role: candidate.role.clone(),
                agent: candidate.agent.clone(),
                model: candidate.model.clone(),
                inherit: candidate.inherit.clone(),
                capabilities: vec!["dynamic_graph_candidate".to_string()],
                status: MemberStatus::Idle,
                active_task_id: None,
                last_message_seq: 0,
            },
        );
        candidates.insert(
            candidate.id.clone(),
            DynamicCandidateRuntime {
                spec: candidate,
                task_id: None,
                team_task_id: None,
                attempts: 0,
                initial: None,
                scores: FxHashMap::default(),
                score_complete: false,
                relevance: 0.0,
                selected: false,
                forward: None,
                reverse: None,
                error: None,
            },
        );
    }
    Ok((
        GraphRun::Dynamic(DynamicGraphRun {
            phase: DynamicPhase::Initial,
            policy,
            candidates,
            edges: Vec::new(),
            pool_task_id: None,
            pool_team_task_id: None,
            result: None,
            error: None,
        }),
        members,
    ))
}

fn advance_graph(graph: &mut AgentGraphCheckpoint) -> Result<(), String> {
    if graph.lifecycle != GraphLifecycle::Running {
        return Ok(());
    }
    match &mut graph.run {
        GraphRun::Static(run) => advance_static(
            &graph.graph_id,
            &graph.question,
            graph.max_parallel,
            &mut graph.team,
            run,
        )?,
        GraphRun::Dynamic(run) => advance_dynamic(
            &graph.graph_id,
            &graph.question,
            graph.max_parallel,
            &mut graph.team,
            run,
        )?,
    }
    match &graph.run {
        GraphRun::Static(run) if run.nodes.values().all(|node| node.status.is_terminal()) => {
            graph.lifecycle = if run
                .nodes
                .values()
                .any(|node| node.status == NodeStatus::Failed)
            {
                GraphLifecycle::Failed
            } else {
                GraphLifecycle::Completed
            };
            graph.final_result = Some(render_json(&run.state)?);
        }
        GraphRun::Dynamic(run) if run.error.is_some() => {
            graph.lifecycle = GraphLifecycle::Failed;
            graph.final_result = run.error.clone();
        }
        GraphRun::Dynamic(run) if run.phase == DynamicPhase::Completed => {
            graph.lifecycle = GraphLifecycle::Completed;
            graph.final_result = run.result.clone();
        }
        _ => {}
    }
    if graph.lifecycle == GraphLifecycle::Completed {
        graph.team.lifecycle = TeamLifecycle::Completed;
    } else if graph.lifecycle == GraphLifecycle::Failed {
        graph.team.lifecycle = TeamLifecycle::Failed;
        // Budget exhaustion or spawn failure can leave runtime tasks in flight.
        // Cancel kernel processes and clear references when entering the failed state.
        cancel_active_runtime_tasks(graph, "agent graph failed");
    }
    graph.updated_at_unix_ms = now_ms();
    graph.team.updated_at_unix_ms = graph.updated_at_unix_ms;
    graph.team.bump_revision();
    Ok(())
}

fn advance_static(
    graph_id: &str,
    question: &str,
    max_parallel: usize,
    team: &mut TeamState,
    run: &mut StaticGraphRun,
) -> Result<(), String> {
    let running_ids: Vec<String> = run
        .nodes
        .values()
        .filter_map(|node| node.runtime_task_id.clone())
        .collect();
    for task_id in running_ids {
        match poll_owned_task_result(&task_id) {
            Ok(OwnedTaskPoll::Pending { .. }) => {}
            Err(error) => {
                // Settle poll failures in place and clear the task reference. Propagating
                // the error would skip the checkpoint and repeatedly poll a lost task.
                // Downstream conditional edges observe the failed terminal state.
                let Some(node_id) = run.nodes.iter().find_map(|(id, node)| {
                    (node.runtime_task_id.as_deref() == Some(task_id.as_str())).then(|| id.clone())
                }) else {
                    continue;
                };
                let team_task_id = run.nodes[&node_id].team_task_id.clone();
                settle_graph_team_task(
                    team,
                    team_task_id.as_deref(),
                    "failed",
                    "",
                    Some(error.as_str()),
                );
                let node = run.nodes.get_mut(&node_id).expect("node exists");
                node.status = NodeStatus::Failed;
                node.runtime_task_id = None;
                node.team_task_id = None;
                node.error = Some(format!("lost graph task {task_id}: {error}"));
            }
            Ok(OwnedTaskPoll::Terminal { result, .. }) => {
                let node_id = run
                    .nodes
                    .iter()
                    .find_map(|(id, node)| {
                        (node.runtime_task_id.as_deref() == Some(task_id.as_str()))
                            .then(|| id.clone())
                    })
                    .ok_or_else(|| format!("graph task {task_id} lost its node owner"))?;
                let team_task_id = run.nodes[&node_id].team_task_id.clone();
                settle_graph_team_task(
                    team,
                    team_task_id.as_deref(),
                    &result.status,
                    &result.output,
                    result.error.as_deref(),
                );
                let node = run.nodes.get_mut(&node_id).expect("node exists");
                node.runtime_task_id = None;
                node.team_task_id = None;
                if result.status == "completed" {
                    node.status = NodeStatus::Completed;
                    node.output = Some(result.output.clone());
                    node.error = None;
                    // A reduction error leaves required shared state missing, so mark
                    // the node failed. Poll already consumed the registry entry; an
                    // early return would leave a running checkpoint that cannot resume.
                    if let Err(error) =
                        reduce_static_output(&mut run.state, &node.spec, &result.output)
                    {
                        node.status = NodeStatus::Failed;
                        node.error = Some(format!("static reduce failed: {error}"));
                    }
                    // Poll persisted the evidence; integration failures only add diagnostics.
                    if let Err(error) = integrate_graph_task(
                        &task_id,
                        "accepted",
                        &format!("static graph node {node_id} output applied to graph state"),
                    ) {
                        node.error = Some(match node.error.take() {
                            Some(prev) => format!("{prev}; evidence integration failed: {error}"),
                            None => format!("evidence integration failed: {error}"),
                        });
                    }
                } else if node.attempts <= node.spec.max_retries {
                    node.status = NodeStatus::Pending;
                    node.error = result.error.clone();
                    if let Err(error) = integrate_graph_task(
                        &task_id,
                        "superseded",
                        &format!("static graph node {node_id} scheduled for retry"),
                    ) {
                        node.error = Some(match node.error.take() {
                            Some(prev) => format!("{prev}; evidence integration failed: {error}"),
                            None => format!("evidence integration failed: {error}"),
                        });
                    }
                } else {
                    node.status = NodeStatus::Failed;
                    node.error = result.error.clone();
                    if let Err(error) = integrate_graph_task(
                        &task_id,
                        "rejected",
                        &format!("static graph node {node_id} failed"),
                    ) {
                        node.error = Some(match node.error.take() {
                            Some(prev) => format!("{prev}; evidence integration failed: {error}"),
                            None => format!("evidence integration failed: {error}"),
                        });
                    }
                }
            }
        }
    }

    loop {
        let running = run
            .nodes
            .values()
            .filter(|node| node.status == NodeStatus::Running)
            .count();
        if running >= max_parallel {
            break;
        }
        let mut pending: Vec<String> = run
            .nodes
            .iter()
            .filter_map(|(id, node)| (node.status == NodeStatus::Pending).then(|| id.clone()))
            .collect();
        pending.sort();
        let mut launched = false;
        for node_id in pending {
            match static_node_decision(run, &node_id) {
                Ok(StaticDecision::Wait) => continue,
                Ok(StaticDecision::Skip) => {
                    run.nodes.get_mut(&node_id).unwrap().status = NodeStatus::Skipped;
                    launched = true;
                    break;
                }
                Ok(StaticDecision::Run(upstream)) => {
                    let prompt =
                        static_node_prompt(question, &run.state, &upstream, &run.nodes[&node_id]);
                    let spec = run.nodes[&node_id].spec.clone();
                    let spawned = spawn_graph_member_task(
                        graph_id,
                        team,
                        &node_id,
                        &format!("static:{}", spec.description),
                        &prompt,
                        spec.response_schema.clone(),
                    );
                    let Some((task_id, team_task_id)) = (match spawned {
                        Ok(pair) => pair,
                        Err(error) => {
                            // Settle unrecoverable spawn failures in place. Propagation
                            // would skip the checkpoint and retry the same pending node
                            // indefinitely on subsequent advances.
                            let node = run.nodes.get_mut(&node_id).unwrap();
                            node.status = NodeStatus::Failed;
                            node.error = Some(format!("spawn failed: {error}"));
                            launched = true;
                            break;
                        }
                    }) else {
                        // Exhausted attempts prevent further launches. Fail unfinished
                        // nodes so the graph terminates and its checkpoint is saved.
                        for node in run.nodes.values_mut() {
                            if !node.status.is_terminal() {
                                node.status = NodeStatus::Failed;
                                node.error =
                                    Some("graph team exhausted max_total_attempts".to_string());
                            }
                        }
                        break;
                    };
                    let node = run.nodes.get_mut(&node_id).unwrap();
                    node.status = NodeStatus::Running;
                    node.runtime_task_id = Some(task_id);
                    node.team_task_id = Some(team_task_id);
                    node.attempts += 1;
                    launched = true;
                    break;
                }
                Err(error) => {
                    // Settle decision errors in place so the failed node is checkpointed
                    // rather than left pending for the next advance.
                    let node = run.nodes.get_mut(&node_id).unwrap();
                    node.status = NodeStatus::Failed;
                    node.error = Some(format!("static node decision failed: {error}"));
                    launched = true;
                    break;
                }
            }
        }
        if !launched {
            break;
        }
    }
    Ok(())
}

#[derive(Debug)]
enum StaticDecision {
    Wait,
    Skip,
    Run(Vec<(String, String)>),
}

fn static_node_decision(run: &StaticGraphRun, node_id: &str) -> Result<StaticDecision, String> {
    let incoming: Vec<&StaticEdgeSpec> =
        run.edges.iter().filter(|edge| edge.to == node_id).collect();
    if incoming.is_empty() {
        return Ok(StaticDecision::Run(Vec::new()));
    }
    if incoming
        .iter()
        .any(|edge| !run.nodes[&edge.from].status.is_terminal())
    {
        return Ok(StaticDecision::Wait);
    }
    let mut active = Vec::new();
    for edge in incoming {
        let source = &run.nodes[&edge.from];
        if edge_matches(source, &edge.condition)? {
            active.push((
                edge.from.clone(),
                source.output.clone().unwrap_or_else(|| {
                    source
                        .error
                        .clone()
                        .unwrap_or_else(|| "<no output>".to_string())
                }),
            ));
        }
    }
    if active.is_empty() {
        Ok(StaticDecision::Skip)
    } else {
        Ok(StaticDecision::Run(active))
    }
}

fn edge_matches(node: &StaticNodeRuntime, condition: &EdgeCondition) -> Result<bool, String> {
    match condition {
        EdgeCondition::Always => Ok(true),
        EdgeCondition::OnSuccess => Ok(node.status == NodeStatus::Completed),
        EdgeCondition::OnFailure => Ok(node.status == NodeStatus::Failed),
        EdgeCondition::OutputContains { text } => Ok(node
            .output
            .as_deref()
            .is_some_and(|output| output.contains(text))),
        EdgeCondition::JsonPointerEquals { pointer, value } => {
            let Some(output) = node.output.as_deref() else {
                return Ok(false);
            };
            let Ok(parsed) = serde_json::from_str::<Value>(output) else {
                return Ok(false);
            };
            Ok(parsed.pointer(pointer) == Some(value))
        }
    }
}

fn static_node_prompt(
    question: &str,
    state: &Value,
    upstream: &[(String, String)],
    node: &StaticNodeRuntime,
) -> String {
    let upstream = upstream
        .iter()
        .map(|(id, output)| format!("## {id}\n{}", clip_text(output, 12_000)))
        .collect::<Vec<_>>()
        .join("\n\n");
    clip_text(
        &format!(
            include_str!("prompts/graph_static_node.md"),
            node_id = node.spec.id,
            question = question,
            node_task = node.spec.prompt,
            state = serde_json::to_string_pretty(state).unwrap_or_else(|_| "{}".to_string()),
            upstream = if upstream.is_empty() {
                "<root node>"
            } else {
                &upstream
            }
        ),
        MAX_PROMPT_CHARS,
    )
}

fn reduce_static_output(
    state: &mut Value,
    spec: &StaticNodeSpec,
    output: &str,
) -> Result<(), String> {
    if !state.is_object() {
        *state = Value::Object(Map::new());
    }
    let key = spec.output_key.as_deref().unwrap_or(&spec.id);
    let parsed =
        serde_json::from_str::<Value>(output).unwrap_or_else(|_| Value::String(output.to_string()));
    let object = state.as_object_mut().expect("state normalized to object");
    match spec.reducer {
        Reducer::Replace => {
            object.insert(key.to_string(), parsed);
        }
        Reducer::AppendArray => {
            let slot = object
                .entry(key.to_string())
                .or_insert_with(|| Value::Array(Vec::new()));
            let array = slot.as_array_mut().ok_or_else(|| {
                format!("reducer append_array requires state[{key:?}] to be an array")
            })?;
            array.push(parsed);
        }
        Reducer::MergeObject => {
            let incoming = parsed.as_object().ok_or_else(|| {
                format!(
                    "reducer merge_object requires node {} to return JSON object",
                    spec.id
                )
            })?;
            let slot = object
                .entry(key.to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            let target = slot.as_object_mut().ok_or_else(|| {
                format!("reducer merge_object requires state[{key:?}] to be an object")
            })?;
            for (field, value) in incoming {
                target.insert(field.clone(), value.clone());
            }
        }
        Reducer::Concat => {
            let text = match parsed {
                Value::String(text) => text,
                value => value.to_string(),
            };
            let slot = object
                .entry(key.to_string())
                .or_insert_with(|| Value::String(String::new()));
            let target = slot
                .as_str()
                .ok_or_else(|| format!("reducer concat requires state[{key:?}] to be a string"))?
                .to_string();
            *slot = Value::String(if target.is_empty() {
                text
            } else {
                format!("{target}\n{text}")
            });
        }
    }
    Ok(())
}

fn advance_dynamic(
    graph_id: &str,
    question: &str,
    max_parallel: usize,
    team: &mut TeamState,
    run: &mut DynamicGraphRun,
) -> Result<(), String> {
    collect_dynamic_tasks(team, run)?;
    if run.error.is_some() {
        // Stop launching after an error; advance_graph will mark the graph failed.
        return Ok(());
    }
    loop {
        match run.phase {
            DynamicPhase::Initial => {
                launch_dynamic_phase(graph_id, question, max_parallel, team, run)?;
                if dynamic_initial_done(run) {
                    run.phase = DynamicPhase::Scoring;
                    continue;
                }
            }
            DynamicPhase::Scoring => {
                launch_dynamic_phase(graph_id, question, max_parallel, team, run)?;
                if dynamic_scoring_done(run) {
                    select_dynamic_topology(question, run);
                    run.phase = DynamicPhase::Forward;
                    seed_forward_roots(run);
                    continue;
                }
            }
            DynamicPhase::Forward => {
                launch_dynamic_phase(graph_id, question, max_parallel, team, run)?;
                if dynamic_forward_done(run) {
                    run.phase = DynamicPhase::Reverse;
                    seed_reverse_leaves(run);
                    continue;
                }
            }
            DynamicPhase::Reverse => {
                launch_dynamic_phase(graph_id, question, max_parallel, team, run)?;
                if dynamic_reverse_done(run) {
                    run.phase = DynamicPhase::Pooling;
                    continue;
                }
            }
            DynamicPhase::Pooling => {
                if run.policy.pool == PoolStrategy::Max {
                    run.result = best_dynamic_output(run);
                    run.phase = DynamicPhase::Completed;
                    continue;
                }
                launch_pool_task(graph_id, question, team, run)?;
                if run.result.is_some() {
                    run.phase = DynamicPhase::Completed;
                    continue;
                }
            }
            DynamicPhase::Completed => {}
        }
        break;
    }
    Ok(())
}

fn collect_dynamic_tasks(team: &mut TeamState, run: &mut DynamicGraphRun) -> Result<(), String> {
    let active: Vec<(String, String)> = run
        .candidates
        .iter()
        .filter_map(|(id, candidate)| candidate.task_id.clone().map(|task| (id.clone(), task)))
        .collect();
    for (candidate_id, task_id) in active {
        let poll = poll_owned_task_result(&task_id);
        let OwnedTaskPoll::Terminal { result, .. } = (match poll {
            Ok(poll) => poll,
            Err(error) => {
                // Clear failed task references and reset the member for a later retry.
                // Repeated failures terminate through the attempt budget. Propagating
                // here would skip the checkpoint and repeatedly poll the same lost task.
                let team_task_id = run.candidates[&candidate_id].team_task_id.clone();
                settle_graph_team_task(
                    team,
                    team_task_id.as_deref(),
                    "failed",
                    "",
                    Some(error.as_str()),
                );
                let candidate = run.candidates.get_mut(&candidate_id).unwrap();
                candidate.task_id = None;
                candidate.team_task_id = None;
                candidate.error = Some(format!("lost graph task {task_id}: {error}"));
                continue;
            }
        }) else {
            continue;
        };
        let team_task_id = run.candidates[&candidate_id].team_task_id.clone();
        settle_graph_team_task(
            team,
            team_task_id.as_deref(),
            &result.status,
            &result.output,
            result.error.as_deref(),
        );
        let phase = run.phase.clone();
        let candidate = run.candidates.get_mut(&candidate_id).unwrap();
        candidate.task_id = None;
        candidate.team_task_id = None;
        if result.status != "completed" {
            candidate.error = result.error.clone();
        }
        match phase {
            DynamicPhase::Initial => {
                candidate.initial = Some(if result.status == "completed" {
                    result.output.clone()
                } else {
                    format!(
                        "Candidate failed: {}",
                        result.error.as_deref().unwrap_or("unknown error")
                    )
                });
            }
            DynamicPhase::Scoring => {
                candidate.scores = parse_scores(&result.output).unwrap_or_default();
                candidate.score_complete = true;
            }
            DynamicPhase::Forward => {
                candidate.forward = Some(if result.status == "completed" {
                    result.output.clone()
                } else {
                    candidate.initial.clone().unwrap_or_default()
                });
            }
            DynamicPhase::Reverse => {
                candidate.reverse = Some(if result.status == "completed" {
                    result.output.clone()
                } else {
                    candidate.forward.clone().unwrap_or_default()
                });
            }
            _ => {}
        }
        // Poll consumed the registry entry and persisted evidence. Keep the updated
        // candidate checkpoint even if integration fails; only the parent's evidence
        // receipt is affected, and retrying the consumed task would stall the graph.
        let _ = integrate_graph_task(
            &task_id,
            if result.status == "completed" {
                "accepted"
            } else {
                "rejected"
            },
            &format!("dynamic graph {phase:?} result for candidate {candidate_id} consumed"),
        );
    }

    if let Some(task_id) = run.pool_task_id.clone() {
        match poll_owned_task_result(&task_id) {
            Ok(OwnedTaskPoll::Pending { .. }) => {}
            Ok(OwnedTaskPoll::Terminal { result, .. }) => {
                settle_graph_team_task(
                    team,
                    run.pool_team_task_id.as_deref(),
                    &result.status,
                    &result.output,
                    result.error.as_deref(),
                );
                run.pool_task_id = None;
                run.pool_team_task_id = None;
                run.result = Some(if result.status == "completed" {
                    result.output.clone()
                } else {
                    best_dynamic_output(run).unwrap_or_else(|| {
                        format!(
                            "pooling failed: {}",
                            result.error.as_deref().unwrap_or("unknown error")
                        )
                    })
                });
                // Preserve the settled pooling state if the evidence receipt fails.
                let _ = integrate_graph_task(
                    &task_id,
                    if result.status == "completed" {
                        "accepted"
                    } else {
                        "rejected"
                    },
                    "dynamic graph pooling result consumed",
                );
            }
            Err(error) => {
                // Settle a lost pool task in place and finish using the best candidate.
                let team_task_id = run.pool_team_task_id.clone();
                settle_graph_team_task(
                    team,
                    team_task_id.as_deref(),
                    "failed",
                    "",
                    Some(error.as_str()),
                );
                run.pool_task_id = None;
                run.pool_team_task_id = None;
                run.result = Some(
                    best_dynamic_output(run)
                        .unwrap_or_else(|| format!("pooling task lost: {error}")),
                );
            }
        }
    }
    Ok(())
}

fn launch_dynamic_phase(
    graph_id: &str,
    question: &str,
    max_parallel: usize,
    team: &mut TeamState,
    run: &mut DynamicGraphRun,
) -> Result<(), String> {
    let mut slots = max_parallel.saturating_sub(
        run.candidates
            .values()
            .filter(|candidate| candidate.task_id.is_some())
            .count(),
    );
    if slots == 0 {
        return Ok(());
    }
    let mut ids: Vec<String> = run.candidates.keys().cloned().collect();
    ids.sort();
    for id in ids {
        if slots == 0 {
            break;
        }
        let should_launch = {
            let candidate = &run.candidates[&id];
            candidate.task_id.is_none()
                && match run.phase {
                    DynamicPhase::Initial => candidate.initial.is_none(),
                    DynamicPhase::Scoring => !candidate.score_complete,
                    DynamicPhase::Forward => candidate.selected && candidate.forward.is_none(),
                    DynamicPhase::Reverse => candidate.selected && candidate.reverse.is_none(),
                    _ => false,
                }
        };
        if !should_launch {
            continue;
        }
        let prompt = match dynamic_prompt(question, run, &id) {
            Ok(prompt) => prompt,
            Err(error) => {
                // Record unrecoverable prompt errors so the graph terminates as failed.
                run.error = Some(format!("dynamic prompt failed: {error}"));
                break;
            }
        };
        let schema = (run.phase == DynamicPhase::Scoring).then(score_response_schema);
        let description = format!("dynamic:{:?}:{id}", run.phase).to_lowercase();
        let spawned = spawn_graph_member_task(graph_id, team, &id, &description, &prompt, schema);
        let Some((task_id, team_task_id)) = (match spawned {
            Ok(pair) => pair,
            Err(error) => {
                // Record spawn errors so the graph terminates as failed.
                run.error = Some(format!("spawn failed: {error}"));
                break;
            }
        }) else {
            // Exhausted attempts stop candidate launches. advance_dynamic returns
            // early and advance_graph marks the graph failed.
            run.error = Some("graph team exhausted max_total_attempts".to_string());
            break;
        };
        let candidate = run.candidates.get_mut(&id).unwrap();
        candidate.task_id = Some(task_id);
        candidate.team_task_id = Some(team_task_id);
        candidate.attempts += 1;
        slots -= 1;
    }
    Ok(())
}

fn dynamic_prompt(question: &str, run: &DynamicGraphRun, id: &str) -> Result<String, String> {
    let candidate = &run.candidates[id];
    let role = &candidate.spec.role;
    let prompt = match run.phase {
        DynamicPhase::Initial => format!(
            include_str!("prompts/graph_dynamic_initial.md"),
            special_instructions = candidate.spec.prompt,
            id = id,
            role = role,
            question = question
        ),
        DynamicPhase::Scoring => {
            let answers = ordered_candidate_outputs(run, |candidate| candidate.initial.as_deref());
            format!(
                include_str!("prompts/graph_dynamic_scoring.md"),
                id = id,
                question = question,
                answers = answers
            )
        }
        DynamicPhase::Forward => {
            let messages = run
                .edges
                .iter()
                .filter(|edge| edge.to == id)
                .filter_map(|edge| {
                    run.candidates[&edge.from].initial.as_deref().map(|output| {
                        format!(
                            "## {} (edge {:.3})\n{}",
                            edge.from,
                            edge.score,
                            clip_text(output, 12_000)
                        )
                    })
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            format!(
                include_str!("prompts/graph_dynamic_forward.md"),
                initial_answer = candidate.initial.as_deref().unwrap_or(""),
                inbound_answers = if messages.is_empty() {
                    "<none>"
                } else {
                    &messages
                },
                id = id,
                role = role,
                question = question
            )
        }
        DynamicPhase::Reverse => {
            let messages = run
                .edges
                .iter()
                .filter(|edge| edge.from == id)
                .filter_map(|edge| {
                    run.candidates[&edge.to].forward.as_deref().map(|output| {
                        format!(
                            "## {} (reverse edge {:.3})\n{}",
                            edge.to,
                            edge.score,
                            clip_text(output, 12_000)
                        )
                    })
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            format!(
                include_str!("prompts/graph_dynamic_reverse.md"),
                forward_answer = candidate.forward.as_deref().unwrap_or(""),
                neighbor_feedback = if messages.is_empty() {
                    "<none>"
                } else {
                    &messages
                },
                id = id,
                role = role,
                question = question
            )
        }
        _ => {
            return Err(format!(
                "phase {:?} does not launch candidate tasks",
                run.phase
            ));
        }
    };
    Ok(clip_text(&prompt, MAX_PROMPT_CHARS))
}

fn launch_pool_task(
    graph_id: &str,
    question: &str,
    team: &mut TeamState,
    run: &mut DynamicGraphRun,
) -> Result<(), String> {
    if run.result.is_some() || run.pool_task_id.is_some() {
        return Ok(());
    }
    let Some(member_id) = ranked_selected_ids(run).into_iter().next() else {
        // No selected candidate is a configuration error. Persist the failed state;
        // propagating would skip the checkpoint and leave the graph running forever.
        run.error = Some("dynamic graph has no selected candidate for pooling".to_string());
        return Ok(());
    };
    if let Some(agent) = run.policy.pool_agent.clone() {
        team.members.get_mut(&member_id).unwrap().agent = Some(agent);
    }
    if let Some(model) = run.policy.pool_model.clone() {
        team.members.get_mut(&member_id).unwrap().model = Some(model);
    }
    let answers = ranked_selected_ids(run)
        .into_iter()
        .map(|id| {
            let candidate = &run.candidates[&id];
            format!(
                "## {id} (relevance {:.4})\n{}",
                candidate.relevance,
                clip_text(
                    candidate
                        .reverse
                        .as_deref()
                        .or(candidate.forward.as_deref())
                        .or(candidate.initial.as_deref())
                        .unwrap_or(""),
                    16_000
                )
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let prompt = clip_text(
        &format!(
            include_str!("prompts/graph_pooling_judge.md"),
            question = question,
            answers = answers
        ),
        MAX_PROMPT_CHARS,
    );
    let spawned =
        spawn_graph_member_task(graph_id, team, &member_id, "dynamic:pooling", &prompt, None);
    let Some((task_id, team_task_id)) = (match spawned {
        Ok(pair) => pair,
        Err(error) => {
            // Record unrecoverable pooling spawn errors so the graph terminates.
            run.error = Some(format!("pooling spawn failed: {error}"));
            return Ok(());
        }
    }) else {
        // Record exhausted attempts so the graph fails without launching a pool task.
        run.error = Some("graph team exhausted max_total_attempts".to_string());
        return Ok(());
    };
    run.pool_task_id = Some(task_id);
    run.pool_team_task_id = Some(team_task_id);
    Ok(())
}

fn select_dynamic_topology(question: &str, run: &mut DynamicGraphRun) {
    let ids: Vec<String> = run.candidates.keys().cloned().collect();
    for target in &ids {
        let mut scores = Vec::new();
        for scorer in &ids {
            if let Some(score) = run.candidates[scorer].scores.get(target) {
                scores.push(score.clamp(0.0, 1.0));
            }
        }
        let fallback = lexical_relevance(
            question,
            run.candidates[target].initial.as_deref().unwrap_or(""),
        );
        run.candidates.get_mut(target).unwrap().relevance = if scores.is_empty() {
            fallback
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
    }

    let mut ranked = ranked_all_ids(run);
    let mut selected: Vec<String> = ranked
        .iter()
        .filter(|id| run.candidates[*id].relevance >= run.policy.relevance_threshold)
        .cloned()
        .take(run.policy.max_selected)
        .collect();
    for id in ranked.drain(..) {
        if selected.len() >= run.policy.min_selected {
            break;
        }
        if !selected.contains(&id) {
            selected.push(id);
        }
    }
    let selected_set: FxHashSet<String> = selected.iter().cloned().collect();
    for (id, candidate) in &mut run.candidates {
        candidate.selected = selected_set.contains(id);
    }

    let ranked = ranked_selected_ids(run);
    let mut edges = Vec::new();
    for (from_index, from) in ranked.iter().enumerate() {
        for to in ranked.iter().skip(from_index + 1) {
            let score = pairwise_score(run, from, to);
            if score >= run.policy.edge_threshold {
                edges.push(DynamicEdge {
                    from: from.clone(),
                    to: to.clone(),
                    score,
                });
            }
        }
        if let Some(to) = ranked.get(from_index + 1) {
            if !edges
                .iter()
                .any(|edge| edge.from == *from && edge.to == *to)
            {
                edges.push(DynamicEdge {
                    from: from.clone(),
                    to: to.clone(),
                    score: pairwise_score(run, from, to),
                });
            }
        }
    }
    run.edges = edges;
}

fn pairwise_score(run: &DynamicGraphRun, left: &str, right: &str) -> f64 {
    let lr = run.candidates[left].scores.get(right).copied();
    let rl = run.candidates[right].scores.get(left).copied();
    match (lr, rl) {
        (Some(left), Some(right)) => (left + right) / 2.0,
        (Some(score), None) | (None, Some(score)) => score,
        (None, None) => lexical_relevance(
            run.candidates[left].initial.as_deref().unwrap_or(""),
            run.candidates[right].initial.as_deref().unwrap_or(""),
        ),
    }
    .clamp(0.0, 1.0)
}

fn seed_forward_roots(run: &mut DynamicGraphRun) {
    let ids = ranked_selected_ids(run);
    for id in ids {
        let has_inbound = run.edges.iter().any(|edge| edge.to == id);
        if !has_inbound {
            let initial = run.candidates[&id].initial.clone().unwrap_or_default();
            run.candidates.get_mut(&id).unwrap().forward = Some(initial);
        }
    }
}

fn seed_reverse_leaves(run: &mut DynamicGraphRun) {
    let ids = ranked_selected_ids(run);
    for id in ids {
        let has_outbound = run.edges.iter().any(|edge| edge.from == id);
        if !has_outbound {
            let forward = run.candidates[&id].forward.clone().unwrap_or_default();
            run.candidates.get_mut(&id).unwrap().reverse = Some(forward);
        }
    }
}

fn dynamic_initial_done(run: &DynamicGraphRun) -> bool {
    run.candidates
        .values()
        .all(|candidate| candidate.initial.is_some())
}

fn dynamic_scoring_done(run: &DynamicGraphRun) -> bool {
    run.candidates
        .values()
        .all(|candidate| candidate.score_complete)
}

fn dynamic_forward_done(run: &DynamicGraphRun) -> bool {
    run.candidates
        .values()
        .filter(|candidate| candidate.selected)
        .all(|candidate| candidate.forward.is_some())
}

fn dynamic_reverse_done(run: &DynamicGraphRun) -> bool {
    run.candidates
        .values()
        .filter(|candidate| candidate.selected)
        .all(|candidate| candidate.reverse.is_some())
}

fn best_dynamic_output(run: &DynamicGraphRun) -> Option<String> {
    ranked_selected_ids(run).into_iter().find_map(|id| {
        let candidate = &run.candidates[&id];
        candidate
            .reverse
            .clone()
            .or_else(|| candidate.forward.clone())
            .or_else(|| candidate.initial.clone())
    })
}

fn ranked_all_ids(run: &DynamicGraphRun) -> Vec<String> {
    let mut ids: Vec<String> = run.candidates.keys().cloned().collect();
    ids.sort_by(|left, right| {
        run.candidates[right]
            .relevance
            .partial_cmp(&run.candidates[left].relevance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    ids
}

fn ranked_selected_ids(run: &DynamicGraphRun) -> Vec<String> {
    ranked_all_ids(run)
        .into_iter()
        .filter(|id| run.candidates[id].selected)
        .collect()
}

fn ordered_candidate_outputs(
    run: &DynamicGraphRun,
    output: impl Fn(&DynamicCandidateRuntime) -> Option<&str>,
) -> String {
    let mut ids: Vec<String> = run.candidates.keys().cloned().collect();
    ids.sort();
    ids.into_iter()
        .map(|id| {
            format!(
                "## {id}\n{}",
                clip_text(output(&run.candidates[&id]).unwrap_or("<missing>"), 12_000)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn score_response_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["scores"],
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["candidate_id", "score"],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    }
                }
            }
        }
    })
}

fn parse_scores(output: &str) -> Option<FxHashMap<String, f64>> {
    let value = serde_json::from_str::<Value>(output)
        .ok()
        .or_else(|| extract_json_object(output).and_then(|text| serde_json::from_str(text).ok()))?;
    let scores = value.get("scores")?.as_array()?;
    let mut parsed = FxHashMap::default();
    for score in scores {
        let id = score.get("candidate_id")?.as_str()?.to_string();
        let value = score.get("score")?.as_f64()?.clamp(0.0, 1.0);
        parsed.insert(id, value);
    }
    Some(parsed)
}

fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end >= start).then(|| &text[start..=end])
}

fn lexical_relevance(left: &str, right: &str) -> f64 {
    let left: FxHashSet<String> = left
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.len() > 2)
        .map(str::to_lowercase)
        .collect();
    let right: FxHashSet<String> = right
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.len() > 2)
        .map(str::to_lowercase)
        .collect();
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(&right).count() as f64;
    let union = left.union(&right).count() as f64;
    intersection / union
}

fn spawn_graph_member_task(
    graph_id: &str,
    team: &mut TeamState,
    member_id: &str,
    description: &str,
    prompt: &str,
    response_schema: Option<Value>,
) -> Result<Option<(String, String)>, String> {
    if team.budget.attempts_used >= team.budget.max_total_attempts {
        // Exhausted attempts are a termination condition. Returning None lets callers
        // fail unfinished work and save the checkpoint rather than leaving it running.
        return Ok(None);
    }
    let member = team
        .members
        .get(member_id)
        .ok_or_else(|| format!("unknown graph member {member_id}"))?
        .clone();
    if member.status != MemberStatus::Idle {
        return Err(format!("graph member {member_id} is already busy"));
    }
    let mut args = json!({
        "description": description,
        "prompt": prompt,
        "inherit": member.inherit,
    });
    if let Some(agent) = member.agent {
        args["agent"] = Value::String(agent);
    }
    if let Some(model) = member.model {
        args["model"] = Value::String(model);
    }
    if let Some(schema) = response_schema {
        args["response_schema"] = schema;
    }
    let prepared = prepare_subagent_task(&args)?;
    let spawned = spawn_subagent_kernel_task(&prepared)?;
    let now = now_ms();
    let revision = team.bump_revision();
    let team_task_id = format!("g-{}-{}", &graph_id[..8], Uuid::new_v4().simple());
    team.tasks.insert(
        team_task_id.clone(),
        TeamTask {
            id: team_task_id.clone(),
            revision,
            title: description.to_string(),
            prompt: clip_text(prompt, MAX_PROMPT_CHARS),
            assignee: Some(member_id.to_string()),
            depends_on: Vec::new(),
            state: TeamTaskState::Running,
            lease: None,
            runtime_task_id: Some(spawned.task_id.clone()),
            runtime_state: Some("spawned".to_string()),
            attempts: 1,
            output: None,
            error: None,
            created_at_unix_ms: now,
            updated_at_unix_ms: now,
        },
    );
    let member = team.members.get_mut(member_id).unwrap();
    member.status = MemberStatus::Busy;
    member.active_task_id = Some(team_task_id.clone());
    team.budget.attempts_used += 1;
    Ok(Some((spawned.task_id, team_task_id)))
}

fn settle_graph_team_task(
    team: &mut TeamState,
    team_task_id: Option<&str>,
    status: &str,
    output: &str,
    error: Option<&str>,
) {
    let Some(team_task_id) = team_task_id else {
        return;
    };
    if !team.tasks.contains_key(team_task_id) {
        return;
    }
    let revision = team.bump_revision();
    let Some(task) = team.tasks.get_mut(team_task_id) else {
        return;
    };
    task.state = if status == "completed" {
        TeamTaskState::Completed
    } else if status == "cancelled" {
        TeamTaskState::Cancelled
    } else {
        TeamTaskState::Failed
    };
    task.runtime_task_id = None;
    task.runtime_state = None;
    task.output = (!output.is_empty()).then(|| output.to_string());
    task.error = error.map(str::to_string);
    task.updated_at_unix_ms = now_ms();
    task.revision = revision;
    if let Some(member_id) = task.assignee.clone()
        && let Some(member) = team.members.get_mut(&member_id)
    {
        member.status = MemberStatus::Idle;
        member.active_task_id = None;
    }
}

fn cancel_graph(graph: &mut AgentGraphCheckpoint) -> Result<(), String> {
    let task_ids = active_graph_task_ids(graph);
    if !task_ids.is_empty() {
        execute_task_cancel(&json!({
            "task_ids": task_ids,
            "reason": format!("agent graph {} cancelled", graph.graph_id),
            "force": true,
        }))?;
        match &mut graph.run {
            GraphRun::Static(run) => {
                for node in run.nodes.values_mut() {
                    if node.status == NodeStatus::Running {
                        node.status = NodeStatus::Cancelled;
                        node.runtime_task_id = None;
                        node.team_task_id = None;
                    }
                }
            }
            GraphRun::Dynamic(run) => {
                for candidate in run.candidates.values_mut() {
                    candidate.task_id = None;
                    candidate.team_task_id = None;
                }
                run.pool_task_id = None;
                run.pool_team_task_id = None;
            }
        }
    }
    let revision = graph.team.bump_revision();
    for task in graph.team.tasks.values_mut() {
        if task.state == TeamTaskState::Running {
            task.state = TeamTaskState::Cancelled;
            task.runtime_task_id = None;
            task.revision = revision;
        }
    }
    for member in graph.team.members.values_mut() {
        member.status = MemberStatus::Idle;
        member.active_task_id = None;
    }
    graph.lifecycle = GraphLifecycle::Cancelled;
    graph.team.lifecycle = TeamLifecycle::Cancelled;
    graph.updated_at_unix_ms = now_ms();
    Ok(())
}

fn active_graph_task_ids(graph: &AgentGraphCheckpoint) -> Vec<String> {
    match &graph.run {
        GraphRun::Static(run) => run
            .nodes
            .values()
            .filter_map(|node| node.runtime_task_id.clone())
            .collect(),
        GraphRun::Dynamic(run) => run
            .candidates
            .values()
            .filter_map(|candidate| candidate.task_id.clone())
            .chain(run.pool_task_id.clone())
            .collect(),
    }
}

/// Cancel runtime tasks left in flight by budget exhaustion or spawn failure.
/// The graph has already failed, so its abandoned work bypasses the recent-progress guard.
/// Cancellation uses the shared task protocol and clears references even on failure;
/// session teardown remains responsible for any surviving process groups.
fn cancel_active_runtime_tasks(graph: &mut AgentGraphCheckpoint, reason: &str) {
    let task_ids = active_graph_task_ids(graph);
    if !task_ids.is_empty() {
        let _ = execute_task_cancel(&json!({
            "task_ids": task_ids,
            "reason": format!("{reason}: {}", graph.graph_id),
            "force": true,
        }));
    }
    match &mut graph.run {
        GraphRun::Static(run) => {
            for node in run.nodes.values_mut() {
                if node.runtime_task_id.is_some() {
                    node.runtime_task_id = None;
                    node.team_task_id = None;
                }
            }
        }
        GraphRun::Dynamic(run) => {
            for candidate in run.candidates.values_mut() {
                candidate.task_id = None;
                candidate.team_task_id = None;
            }
            run.pool_task_id = None;
            run.pool_team_task_id = None;
        }
    }
    let revision = graph.team.bump_revision();
    for task in graph.team.tasks.values_mut() {
        if task.state == TeamTaskState::Running {
            task.state = TeamTaskState::Cancelled;
            task.runtime_task_id = None;
            task.revision = revision;
        }
    }
    for member in graph.team.members.values_mut() {
        member.status = MemberStatus::Idle;
        member.active_task_id = None;
    }
}

fn validate_static_graph(
    nodes: &FxHashMap<String, StaticNodeRuntime>,
    edges: &[StaticEdgeSpec],
) -> Result<(), String> {
    let mut graph = DirectedGraph::new();
    for id in nodes.keys() {
        graph.add_node(id.clone());
    }
    for edge in edges {
        if edge.from == edge.to {
            return Err(format!("self edge is not allowed: {}", edge.from));
        }
        if !nodes.contains_key(&edge.from) || !nodes.contains_key(&edge.to) {
            return Err(format!(
                "edge {} -> {} references an unknown node",
                edge.from, edge.to
            ));
        }
        if !graph.add_edge(edge.from.clone(), edge.to.clone()) {
            return Err(format!("duplicate edge: {} -> {}", edge.from, edge.to));
        }
    }
    if graph.has_cycle() {
        return Err("static graph contains a cycle".to_string());
    }
    Ok(())
}

enum GraphRevisionSnapshot {
    Initial,
    Static(FxHashMap<String, StaticNodeRuntime>),
    Dynamic {
        phase: DynamicPhase,
        lifecycle: GraphLifecycle,
        candidates: FxHashMap<String, DynamicCandidateRuntime>,
    },
}

impl GraphRevisionSnapshot {
    fn capture(graph: &AgentGraphCheckpoint) -> Self {
        if !graph.team.revision_tracking_enabled {
            return Self::Initial;
        }
        match &graph.run {
            GraphRun::Static(run) => Self::Static(run.nodes.clone()),
            GraphRun::Dynamic(run) => Self::Dynamic {
                phase: run.phase.clone(),
                lifecycle: graph.lifecycle.clone(),
                candidates: run.candidates.clone(),
            },
        }
    }
}

fn record_graph_revision(graph: &mut AgentGraphCheckpoint, previous: &GraphRevisionSnapshot) {
    if !graph.team.revision_tracking_enabled {
        return;
    }
    // Each committed orchestration transaction advances the cursor. Compare complete
    // node data, including outputs beyond the displayed preview and dynamic phases.
    graph.revision = graph.revision.saturating_add(1);
    match &graph.run {
        GraphRun::Static(run) => {
            let previous_nodes = match previous {
                GraphRevisionSnapshot::Static(nodes) => Some(nodes),
                _ => None,
            };
            for (id, node) in &run.nodes {
                if previous_nodes.and_then(|nodes| nodes.get(id)) != Some(node) {
                    graph.node_revisions.insert(id.clone(), graph.revision);
                }
            }
        }
        GraphRun::Dynamic(run) => {
            let (previous_candidates, phase_changed) = match previous {
                GraphRevisionSnapshot::Dynamic {
                    phase,
                    lifecycle,
                    candidates,
                } => (
                    Some(candidates),
                    phase != &run.phase || lifecycle != &graph.lifecycle,
                ),
                _ => (None, true),
            };
            for (id, candidate) in &run.candidates {
                if phase_changed
                    || previous_candidates.and_then(|candidates| candidates.get(id))
                        != Some(candidate)
                {
                    graph.node_revisions.insert(id.clone(), graph.revision);
                }
            }
        }
    }
}

fn graph_checkpoint_path(graph_id: &str) -> Result<PathBuf, String> {
    Uuid::parse_str(graph_id)
        .map_err(|_| format!("invalid graph_id {graph_id:?}: expected UUID"))?;
    Ok(checkpoint_base_dir()?
        .join("agent_graphs")
        .join(format!("{graph_id}.json")))
}

fn save_graph(graph: &AgentGraphCheckpoint) -> Result<(), String> {
    let path = graph_checkpoint_path(&graph.graph_id)?;
    atomic_write_json(&path, graph)
}

fn load_graph(graph_id: &str, optimizations_enabled: bool) -> Result<AgentGraphCheckpoint, String> {
    let path = graph_checkpoint_path(graph_id)?;
    let bytes = fs::read(&path)
        .map_err(|error| format!("failed to read agent graph {}: {error}", path.display()))?;
    let mut graph: AgentGraphCheckpoint = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode agent graph {}: {error}", path.display()))?;
    if graph.version != GRAPH_STATE_VERSION {
        return Err(format!(
            "unsupported graph checkpoint version {}; expected {GRAPH_STATE_VERSION}",
            graph.version
        ));
    }
    graph.configure_revision_tracking(optimizations_enabled);
    let (session_id, owner_pid) = active_identity()?;
    if graph.session_id != session_id {
        return Err("agent graph belongs to another session".to_string());
    }
    if graph.owner_pid != owner_pid {
        let previous = GraphRevisionSnapshot::capture(&graph);
        graph.owner_pid = owner_pid;
        graph.team.owner_pid = owner_pid;
        recover_orphaned_team_tasks(&mut graph.team);
        recover_orphaned_graph(&mut graph);
        record_graph_revision(&mut graph, &previous);
        save_graph(&graph)?;
    }
    Ok(graph)
}

fn recover_orphaned_graph(graph: &mut AgentGraphCheckpoint) {
    match &mut graph.run {
        GraphRun::Static(run) => {
            for node in run.nodes.values_mut() {
                if node.status == NodeStatus::Running {
                    node.status = if node.attempts <= node.spec.max_retries {
                        NodeStatus::Pending
                    } else {
                        NodeStatus::Failed
                    };
                    node.runtime_task_id = None;
                    node.team_task_id = None;
                    node.error = Some("orphaned by runtime restart".to_string());
                }
            }
        }
        GraphRun::Dynamic(run) => {
            for candidate in run.candidates.values_mut() {
                candidate.task_id = None;
                candidate.team_task_id = None;
            }
            run.pool_task_id = None;
            run.pool_team_task_id = None;
        }
    }
}

fn mutate_graph(
    graph_id: Option<&str>,
    optimizations_enabled: bool,
    operation: impl FnOnce(&mut AgentGraphCheckpoint) -> Result<Value, String>,
) -> Result<String, String> {
    let graph_id = required_graph_id(graph_id)?;
    let mut graph = load_graph(graph_id, optimizations_enabled)?;
    let previous = GraphRevisionSnapshot::capture(&graph);
    let value = operation(&mut graph)?;
    graph.updated_at_unix_ms = now_ms();
    record_graph_revision(&mut graph, &previous);
    save_graph(&graph)?;
    render_json(&value)
}

fn required_graph_id(graph_id: Option<&str>) -> Result<&str, String> {
    graph_id
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| "action requires graph_id".to_string())
}

fn required_non_empty<'a>(value: Option<&'a str>, field: &str) -> Result<&'a str, String> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{field} must not be empty"))
}

fn graph_view(graph: &AgentGraphCheckpoint) -> Value {
    let run = match &graph.run {
        GraphRun::Static(run) => {
            let mut nodes: Vec<Value> = run
                .nodes
                .iter()
                .map(|(id, node)| {
                    json!({
                        "id": id,
                        "status": node.status,
                        "attempts": node.attempts,
                        "runtime_task_id": node.runtime_task_id,
                        "output": node.output.as_deref().map(|text| clip_text(text, 4_000)),
                        "error": node.error,
                    })
                })
                .collect();
            nodes.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
            json!({"mode": "static", "nodes": nodes, "state": run.state})
        }
        GraphRun::Dynamic(run) => {
            let mut candidates: Vec<Value> = run
                .candidates
                .iter()
                .map(|(id, candidate)| {
                    json!({
                        "id": id,
                        "selected": candidate.selected,
                        "relevance": candidate.relevance,
                        "active_task_id": candidate.task_id,
                        "has_initial": candidate.initial.is_some(),
                        "has_forward": candidate.forward.is_some(),
                        "has_reverse": candidate.reverse.is_some(),
                        "error": candidate.error,
                    })
                })
                .collect();
            candidates.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
            json!({
                "mode": "dynamic",
                "phase": run.phase,
                "candidates": candidates,
                "edges": run.edges,
                "pool_task_id": run.pool_task_id,
                "result": run.result.as_deref().map(|text| clip_text(text, 12_000)),
            })
        }
    };
    json!({
        "graph_id": graph.graph_id,
        "name": graph.name,
        "lifecycle": graph.lifecycle,
        "question": graph.question,
        "max_parallel": graph.max_parallel,
        "final_result": graph.final_result.as_deref().map(|text| clip_text(text, 12_000)),
        "run": run,
        "next_action": if graph.lifecycle == GraphLifecycle::Running { "call run_agent_graph action=advance after tasks make progress" } else { "terminal" },
        "updated_at_unix_ms": graph.updated_at_unix_ms,
    })
}

fn dynamic_candidate_status(
    candidate: &DynamicCandidateRuntime,
    phase: &DynamicPhase,
    lifecycle: &GraphLifecycle,
) -> NodeStatus {
    if lifecycle == &GraphLifecycle::Cancelled {
        return NodeStatus::Cancelled;
    }
    if candidate.task_id.is_some() {
        return NodeStatus::Running;
    }
    if !candidate.selected
        && matches!(
            phase,
            DynamicPhase::Forward
                | DynamicPhase::Reverse
                | DynamicPhase::Pooling
                | DynamicPhase::Completed
        )
    {
        return NodeStatus::Skipped;
    }
    let done = match phase {
        DynamicPhase::Initial => candidate.initial.is_some(),
        DynamicPhase::Scoring => candidate.score_complete,
        DynamicPhase::Forward => candidate.forward.is_some(),
        DynamicPhase::Reverse => candidate.reverse.is_some(),
        DynamicPhase::Pooling | DynamicPhase::Completed => true,
    };
    if done {
        NodeStatus::Completed
    } else if lifecycle == &GraphLifecycle::Failed {
        NodeStatus::Failed
    } else {
        NodeStatus::Pending
    }
}

fn graph_view_with_options(graph: &AgentGraphCheckpoint, options: StatusViewOptions) -> Value {
    if options.is_default() {
        return graph_view(graph);
    }
    let includes =
        |id: &String| options.includes(graph.node_revisions.get(id).copied().unwrap_or_default());
    let run = match &graph.run {
        GraphRun::Static(run) => {
            let mut nodes = run
                .nodes
                .iter()
                .filter(|(id, _)| includes(id))
                .map(|(id, node)| {
                    if options.summary {
                        json!({
                            "id": id,
                            "state": node.status,
                            "attempts": node.attempts,
                            "has_error": node.error.is_some(),
                        })
                    } else {
                        json!({
                            "id": id,
                            "status": node.status,
                            "attempts": node.attempts,
                            "runtime_task_id": node.runtime_task_id,
                            "output": node.output.as_deref().map(|text| clip_text(text, 4_000)),
                            "error": node.error,
                        })
                    }
                })
                .collect::<Vec<_>>();
            nodes.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
            let mut view = json!({
                "mode": "static",
                "node_count": run.nodes.len(),
                "returned_node_count": nodes.len(),
                "edge_count": run.edges.len(),
                "nodes": nodes,
            });
            if !options.summary {
                view["state"] = run.state.clone();
            }
            view
        }
        GraphRun::Dynamic(run) => {
            let mut candidates = run
                .candidates
                .iter()
                .filter(|(id, _)| includes(id))
                .map(|(id, candidate)| {
                    if options.summary {
                        json!({
                            "id": id,
                            "state": dynamic_candidate_status(candidate, &run.phase, &graph.lifecycle),
                            "attempts": candidate.attempts,
                            "has_error": candidate.error.is_some(),
                        })
                    } else {
                        json!({
                            "id": id,
                            "selected": candidate.selected,
                            "relevance": candidate.relevance,
                            "active_task_id": candidate.task_id,
                            "has_initial": candidate.initial.is_some(),
                            "has_forward": candidate.forward.is_some(),
                            "has_reverse": candidate.reverse.is_some(),
                            "error": candidate.error,
                        })
                    }
                })
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
            let mut view = json!({
                "mode": "dynamic",
                "phase": run.phase,
                "candidate_count": run.candidates.len(),
                "returned_candidate_count": candidates.len(),
                "edge_count": run.edges.len(),
                "candidates": candidates,
            });
            if options.summary {
                view["has_result"] = json!(run.result.is_some());
            } else {
                view["edges"] = json!(run.edges);
                view["pool_task_id"] = json!(run.pool_task_id);
                view["result"] = json!(run.result.as_deref().map(|text| clip_text(text, 12_000)));
            }
            view
        }
    };
    let mut view = json!({
        "graph_id": graph.graph_id,
        "name": graph.name,
        "lifecycle": graph.lifecycle,
        "max_parallel": graph.max_parallel,
        "revision": graph.revision,
        "summary": options.summary,
        "since": options.since,
        "has_final_result": graph.final_result.is_some(),
        "run": run,
        "next_action": if graph.lifecycle == GraphLifecycle::Running { "call run_agent_graph action=advance after tasks make progress" } else { "terminal" },
        "updated_at_unix_ms": graph.updated_at_unix_ms,
    });
    if !options.summary {
        view["question"] = json!(graph.question);
        view["final_result"] = json!(
            graph
                .final_result
                .as_deref()
                .map(|text| clip_text(text, 12_000))
        );
    }
    view
}

fn render_json(value: &Value) -> Result<String, String> {
    serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to render graph state: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::tools::task_tools::InheritOptions;

    fn node(id: &str) -> StaticNodeSpec {
        StaticNodeSpec {
            id: id.to_string(),
            description: id.to_string(),
            prompt: id.to_string(),
            agent: None,
            model: None,
            inherit: default_inherit(),
            response_schema: None,
            output_key: None,
            reducer: Reducer::Replace,
            max_retries: 0,
        }
    }

    fn runtime_nodes(ids: &[&str]) -> FxHashMap<String, StaticNodeRuntime> {
        ids.iter()
            .map(|id| {
                (
                    (*id).to_string(),
                    StaticNodeRuntime {
                        spec: node(id),
                        status: NodeStatus::Pending,
                        runtime_task_id: None,
                        team_task_id: None,
                        attempts: 0,
                        output: None,
                        error: None,
                    },
                )
            })
            .collect()
    }

    fn status_test_graph(dynamic: bool) -> AgentGraphCheckpoint {
        let mut team = super::super::tests::sample_team();
        let evidence = "graph evidence".repeat(2_000);
        let run = if dynamic {
            let (mut run, members) = build_dynamic_run(
                serde_json::from_value(json!({"candidates": [
                    {"id": "z", "role": "worker"},
                    {"id": "a", "role": "worker"},
                ]}))
                .unwrap(),
            )
            .unwrap();
            team.members = members;
            let GraphRun::Dynamic(dynamic) = &mut run else {
                unreachable!()
            };
            dynamic.candidates.get_mut("a").unwrap().initial = Some(evidence.clone());
            dynamic.result = Some(evidence.clone());
            run
        } else {
            let mut nodes = runtime_nodes(&["z", "a"]);
            let node = nodes.get_mut("a").unwrap();
            node.status = NodeStatus::Completed;
            node.attempts = 2;
            node.output = Some(evidence.clone());
            node.error = Some("integration diagnostic".to_string());
            GraphRun::Static(StaticGraphRun {
                nodes,
                edges: Vec::new(),
                state: json!({"evidence": evidence}),
            })
        };
        let mut graph = AgentGraphCheckpoint {
            version: GRAPH_STATE_VERSION,
            revision: 0,
            node_revisions: FastMap::default(),
            graph_id: Uuid::new_v4().to_string(),
            session_id: team.session_id.clone(),
            owner_pid: team.owner_pid,
            name: "graph".to_string(),
            question: "question".to_string(),
            lifecycle: GraphLifecycle::Running,
            max_parallel: 4,
            team,
            run,
            final_result: Some(evidence),
            created_at_unix_ms: 1,
            updated_at_unix_ms: 1,
        };
        record_graph_revision(&mut graph, &GraphRevisionSnapshot::Initial);
        graph
    }

    fn status_options(summary: bool, since: Option<u64>) -> StatusViewOptions {
        StatusViewOptions { summary, since }
    }

    fn assert_graph_cancellation_bypasses_recent_progress(explicit_cancel: bool) {
        use crate::ai::driver::runtime_ctx;
        use crate::ai::tools::task_tools::{
            AsyncTaskEntry, TASK_PROGRESS_REGISTRY, execute_task_wait, insert_task_entry_for_test,
            lock_unpoisoned, remove_task_entry, with_os_kernel, with_task_entry,
        };
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Instant;

        super::super::tests::with_status_test_context(|| {
            let mut graph = status_test_graph(false);
            (graph.session_id, graph.owner_pid) = active_identity().unwrap();
            graph.lifecycle = if explicit_cancel {
                GraphLifecycle::Running
            } else {
                GraphLifecycle::Failed
            };
            let task_id = format!("graph-cleanup-{}", Uuid::new_v4().simple());
            let (pid, channel, futex) = with_os_kernel(|os| {
                let pid = os.spawn(
                    Some(graph.owner_pid),
                    "worker".into(),
                    "abandoned work".into(),
                    20,
                    4,
                    None,
                    None,
                )?;
                let channel = os.channel_create(Some(graph.owner_pid), 1, "cleanup-result".into());
                let futex = os.futex_create(0, "cleanup-complete".into());
                Ok((pid, channel.raw(), futex))
            })
            .unwrap();
            let cancelled = Arc::new(AtomicBool::new(false));
            insert_task_entry_for_test(
                task_id.clone(),
                AsyncTaskEntry {
                    session_id: graph.session_id.clone(),
                    evidence_history_file: None,
                    owner_pid: graph.owner_pid,
                    pid,
                    result_channel_id: channel,
                    completion_futex_addr: futex,
                    description: "abandoned graph work".into(),
                    agent_name: "worker".into(),
                    backend: None,
                    model: "test-model".into(),
                    is_model_auto_selected: false,
                    auto_model_fallback: None,
                    selection_explanation: "cleanup test".into(),
                    inherit: InheritOptions::default(),
                    abort_handle: None,
                    cancel_stream: cancelled.clone(),
                    started_at: Instant::now(),
                    last_progress_notification_at: None,
                    last_progress_persisted_at: None,
                },
            );
            let slot = runtime_ctx::new_subagent_progress_slot();
            runtime_ctx::publish_subagent_activity_to_slot(&slot, "working", "recent graph work");
            lock_unpoisoned(&TASK_PROGRESS_REGISTRY).insert(task_id.clone(), slot);
            let GraphRun::Static(run) = &mut graph.run else {
                unreachable!()
            };
            let node = run.nodes.get_mut("z").unwrap();
            node.status = NodeStatus::Running;
            node.runtime_task_id = Some(task_id.clone());
            assert!(execute_task_cancel(&json!({"task_ids": [&task_id]})).is_err());

            if explicit_cancel {
                cancel_graph(&mut graph).unwrap();
                assert_eq!(graph.lifecycle, GraphLifecycle::Cancelled);
            } else {
                cancel_active_runtime_tasks(&mut graph, "agent graph failed");
            }

            assert!(cancelled.load(Ordering::Acquire));
            assert!(with_task_entry(&task_id, |_| ()).is_some());
            let GraphRun::Static(run) = &graph.run else {
                unreachable!()
            };
            assert!(run.nodes["z"].runtime_task_id.is_none());
            let output = execute_task_wait(&json!({"task_ids": [&task_id]})).unwrap();
            if explicit_cancel {
                assert!(output.contains(&format!(
                    "Cancellation reason: agent graph {} cancelled",
                    graph.graph_id
                )));
            } else {
                assert!(output.contains("Cancellation reason: agent graph failed"));
            }
            assert!(output.contains("recent graph work"));
            assert!(!output.contains("Failure reason:"));
            remove_task_entry(&task_id);
            lock_unpoisoned(&TASK_PROGRESS_REGISTRY).remove(&task_id);
        });
    }

    #[test]
    fn task_cancel_graph_failure_cleanup_cancels_abandoned_recent_work() {
        assert_graph_cancellation_bypasses_recent_progress(false);
    }

    #[test]
    fn task_cancel_explicit_graph_succeeds_with_recent_progress() {
        assert_graph_cancellation_bypasses_recent_progress(true);
    }

    #[test]
    fn graph_master_off_ignores_status_options_and_omits_revision_checkpoint() {
        super::super::tests::with_status_test_context_with_config("", || {
            assert!(!crate::ai::config::agent_team_optimizations().optimizations_enabled());
            for dynamic in [false, true] {
                let mut graph = status_test_graph(dynamic);
                (graph.session_id, graph.owner_pid) = active_identity().unwrap();
                graph.team.session_id = graph.session_id.clone();
                graph.team.owner_pid = graph.owner_pid;
                assert!(graph.revision > 0);
                save_graph(&graph).unwrap();
                let expected = render_json(&graph_view(&graph)).unwrap();
                for options in [
                    json!({"summary": true, "since": u64::MAX}),
                    json!({"summary": "invalid", "since": -1}),
                ] {
                    let mut args = options;
                    args["action"] = json!("status");
                    args["graph_id"] = json!(graph.graph_id);
                    assert_eq!(execute_run_agent_graph(&args).unwrap(), expected);
                }
                execute_run_agent_graph(&json!({"action": "cancel", "graph_id": graph.graph_id}))
                    .unwrap();
                let bytes = fs::read(graph_checkpoint_path(&graph.graph_id).unwrap()).unwrap();
                let saved: Value = serde_json::from_slice(&bytes).unwrap();
                assert!(saved.get("revision").is_none());
                assert!(saved.get("node_revisions").is_none());
                assert!(saved["team"].get("revision").is_none());
                let mut loaded = load_graph(&graph.graph_id, false).unwrap();
                let previous = GraphRevisionSnapshot::capture(&loaded);
                assert!(matches!(previous, GraphRevisionSnapshot::Initial));
                record_graph_revision(&mut loaded, &previous);
                assert_eq!(loaded.revision, 0);
                assert!(loaded.node_revisions.is_empty());
                assert_eq!(loaded.team.revision, 0);
            }
        });
    }

    #[test]
    fn graph_status_summary_and_default_preserve_full_evidence() {
        super::super::tests::with_status_test_context(|| {
            for dynamic in [false, true] {
                let mut graph = status_test_graph(dynamic);
                (graph.session_id, graph.owner_pid) = active_identity().unwrap();
                graph.team.session_id = graph.session_id.clone();
                graph.team.owner_pid = graph.owner_pid;
                if let GraphRun::Dynamic(run) = &mut graph.run {
                    run.phase = DynamicPhase::Forward;
                    let candidate = run.candidates.get_mut("a").unwrap();
                    candidate.selected = true;
                    candidate.task_id = Some("active-runtime".to_string());
                    candidate.attempts = 2;
                    candidate.error = Some("integration diagnostic".to_string());
                }
                save_graph(&graph).unwrap();
                let args = json!({"action": "status", "graph_id": graph.graph_id});
                let expected = render_json(&graph_view(&graph)).unwrap();
                assert_eq!(execute_run_agent_graph(&args).unwrap(), expected);
                let full: Value = serde_json::from_str(&expected).unwrap();
                assert!(full.get("revision").is_none());
                assert_eq!(
                    full["final_result"],
                    clip_text(&"graph evidence".repeat(2_000), 12_000)
                );
                if !dynamic {
                    assert_eq!(
                        full["run"]["nodes"][0]["output"],
                        clip_text(&"graph evidence".repeat(2_000), 4_000)
                    );
                    assert!(full["run"]["state"]["evidence"].as_str().unwrap().len() > 12_000);
                }
                let summary: Value = serde_json::from_str(
                    &execute_run_agent_graph(&json!({
                        "action": "status", "graph_id": graph.graph_id, "summary": true,
                    }))
                    .unwrap(),
                )
                .unwrap();
                assert_eq!(summary["revision"], graph.revision);
                assert_eq!(summary["has_final_result"], true);
                assert!(summary.get("final_result").is_none());
                assert!(summary["run"].get("state").is_none());
                assert!(summary["run"].get("result").is_none());
                assert!(!summary.to_string().contains("graph evidence"));
                let key = if dynamic { "candidates" } else { "nodes" };
                let rows = summary["run"][key].as_array().unwrap();
                assert_eq!(rows.len(), 2);
                assert!(rows.iter().all(|row| row.as_object().unwrap().len() == 4));
                assert_eq!(rows[0]["id"], "a");
                assert_eq!(rows[0]["attempts"], 2);
                assert_eq!(rows[0]["has_error"], true);
                assert_eq!(
                    rows[0]["state"],
                    if dynamic { "running" } else { "completed" }
                );
                assert_eq!(
                    rows[1]["state"],
                    if dynamic { "skipped" } else { "pending" }
                );
                assert_eq!(
                    execute_run_agent_graph(&json!({
                        "action": "status", "graph_id": graph.graph_id, "summary": false,
                    }))
                    .unwrap(),
                    expected
                );
                assert_eq!(execute_run_agent_graph(&args).unwrap(), expected);
                assert!(
                    execute_run_agent_graph(&json!({
                        "action": "status", "graph_id": graph.graph_id, "since": -1,
                    }))
                    .is_err()
                );
            }
        });
    }

    #[test]
    fn graph_status_since_tracks_full_static_data_and_persisted_cursor() {
        let mut graph = status_test_graph(false);
        let baseline = graph.revision;
        let previous = GraphRevisionSnapshot::capture(&graph);
        let GraphRun::Static(run) = &mut graph.run else {
            unreachable!()
        };
        let node = run.nodes.get_mut("z").unwrap();
        node.status = NodeStatus::Running;
        node.attempts = 1;
        node.runtime_task_id = Some("runtime-z".to_string());
        record_graph_revision(&mut graph, &previous);
        let delta = graph_view_with_options(&graph, status_options(false, Some(baseline)));
        assert_eq!(delta["run"]["node_count"], 2);
        assert_eq!(delta["run"]["returned_node_count"], 1);
        assert_eq!(delta["run"]["nodes"][0]["id"], "z");
        assert!(graph.revision > baseline);

        let saved_revision = graph.revision;
        graph = serde_json::from_slice(&serde_json::to_vec(&graph).unwrap()).unwrap();
        graph.configure_revision_tracking(true);
        assert_eq!(graph.revision, saved_revision);
        assert_eq!(graph.node_revisions["z"], saved_revision);
        let visible_before = graph_view(&graph);
        let previous = GraphRevisionSnapshot::capture(&graph);
        let GraphRun::Static(run) = &mut graph.run else {
            unreachable!()
        };
        run.nodes
            .get_mut("a")
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .push_str("changed beyond the preview");
        record_graph_revision(&mut graph, &previous);
        assert_eq!(graph_view(&graph), visible_before);
        let delta = graph_view_with_options(&graph, status_options(true, Some(saved_revision)));
        assert_eq!(delta["run"]["returned_node_count"], 1);
        assert_eq!(delta["run"]["nodes"][0]["id"], "a");
        assert!(graph.revision > saved_revision);

        let cursor = graph.revision;
        let unchanged = graph_view_with_options(&graph, status_options(false, Some(cursor)));
        assert_eq!(unchanged["run"]["nodes"], json!([]));
        assert_eq!(unchanged["revision"], cursor);
        assert_eq!(graph.revision, cursor);
        let previous = GraphRevisionSnapshot::capture(&graph);
        graph.final_result = Some("new final evidence".to_string());
        record_graph_revision(&mut graph, &previous);
        let metadata_only = graph_view_with_options(&graph, status_options(false, Some(cursor)));
        assert_eq!(metadata_only["run"]["nodes"], json!([]));
        assert_eq!(metadata_only["final_result"], "new final evidence");
        assert!(metadata_only["revision"].as_u64().unwrap() > cursor);
    }

    #[test]
    fn graph_status_since_tracks_dynamic_phase_and_candidate_changes() {
        let mut graph = status_test_graph(true);
        let baseline = graph.revision;
        let previous = GraphRevisionSnapshot::capture(&graph);
        let GraphRun::Dynamic(run) = &mut graph.run else {
            unreachable!()
        };
        let candidate = run.candidates.get_mut("z").unwrap();
        candidate.task_id = Some("runtime-z".to_string());
        candidate.attempts = 1;
        record_graph_revision(&mut graph, &previous);
        let delta = graph_view_with_options(&graph, status_options(true, Some(baseline)));
        assert_eq!(delta["run"]["returned_candidate_count"], 1);
        assert_eq!(delta["run"]["candidates"][0]["id"], "z");
        assert_eq!(delta["run"]["candidates"][0]["state"], "running");

        let cursor = graph.revision;
        let previous = GraphRevisionSnapshot::capture(&graph);
        let GraphRun::Dynamic(run) = &mut graph.run else {
            unreachable!()
        };
        run.candidates.get_mut("z").unwrap().task_id = None;
        run.phase = DynamicPhase::Scoring;
        record_graph_revision(&mut graph, &previous);
        let phase_delta = graph_view_with_options(&graph, status_options(true, Some(cursor)));
        assert_eq!(phase_delta["run"]["phase"], "scoring");
        assert_eq!(phase_delta["run"]["returned_candidate_count"], 2);
        assert_eq!(phase_delta["run"]["candidates"][0]["state"], "pending");

        let cursor = graph.revision;
        let previous = GraphRevisionSnapshot::capture(&graph);
        let visible_before = graph_view(&graph);
        let GraphRun::Dynamic(run) = &mut graph.run else {
            unreachable!()
        };
        let candidate = run.candidates.get_mut("a").unwrap();
        candidate.score_complete = true;
        candidate.scores.insert("z".to_string(), 0.9);
        record_graph_revision(&mut graph, &previous);
        assert_eq!(graph_view(&graph), visible_before);
        graph = serde_json::from_slice(&serde_json::to_vec(&graph).unwrap()).unwrap();
        graph.configure_revision_tracking(true);
        let score_delta = graph_view_with_options(&graph, status_options(true, Some(cursor)));
        assert_eq!(score_delta["run"]["returned_candidate_count"], 1);
        assert_eq!(score_delta["run"]["candidates"][0]["id"], "a");
        assert_eq!(score_delta["run"]["candidates"][0]["state"], "completed");

        let cursor = graph.revision;
        let previous = GraphRevisionSnapshot::capture(&graph);
        cancel_graph(&mut graph).unwrap();
        record_graph_revision(&mut graph, &previous);
        let cancelled = graph_view_with_options(&graph, status_options(true, Some(cursor)));
        assert_eq!(cancelled["run"]["returned_candidate_count"], 2);
        assert!(
            cancelled["run"]["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .all(|row| row["state"] == "cancelled")
        );
    }

    #[test]
    fn graph_checkpoint_without_revision_loads_and_recovers() {
        super::super::tests::with_status_test_context(|| {
            let mut graph = status_test_graph(false);
            (graph.session_id, graph.owner_pid) = active_identity().unwrap();
            graph.team.session_id = graph.session_id.clone();
            graph.team.owner_pid = graph.owner_pid;
            let GraphRun::Static(run) = &mut graph.run else {
                unreachable!()
            };
            let node = run.nodes.get_mut("a").unwrap();
            node.status = NodeStatus::Running;
            node.runtime_task_id = Some("orphaned-runtime".to_string());
            node.spec.max_retries = node.attempts;
            super::super::add_team_task(
                &mut graph.team,
                super::super::TeamTaskInput {
                    id: Some("first".to_string()),
                    title: "first".to_string(),
                    prompt: "first".to_string(),
                    assignee: None,
                    depends_on: Vec::new(),
                },
            )
            .unwrap();
            graph.team.tasks.get_mut("first").unwrap().state = TeamTaskState::Running;
            let mut legacy = serde_json::to_value(&graph).unwrap();
            legacy.as_object_mut().unwrap().remove("revision");
            legacy.as_object_mut().unwrap().remove("node_revisions");
            legacy["team"].as_object_mut().unwrap().remove("revision");
            for task in legacy["team"]["tasks"]
                .as_object_mut()
                .unwrap()
                .values_mut()
            {
                task.as_object_mut().unwrap().remove("revision");
            }
            let decoded: AgentGraphCheckpoint = serde_json::from_value(legacy.clone()).unwrap();
            assert_eq!(decoded.revision, 0);
            assert!(decoded.node_revisions.is_empty());
            assert_eq!(decoded.team.revision, 0);
            assert_eq!(decoded.team.tasks["first"].revision, 0);

            legacy["owner_pid"] = json!(graph.owner_pid + 1);
            atomic_write_json(&graph_checkpoint_path(&graph.graph_id).unwrap(), &legacy).unwrap();
            let recovered: Value = serde_json::from_str(
                &execute_run_agent_graph(&json!({
                    "action": "status", "graph_id": graph.graph_id, "summary": true, "since": 0,
                }))
                .unwrap(),
            )
            .unwrap();
            assert_eq!(recovered["revision"], 1);
            assert_eq!(recovered["run"]["returned_node_count"], 1);
            assert_eq!(recovered["run"]["nodes"][0]["id"], "a");
            assert_eq!(recovered["run"]["nodes"][0]["state"], "pending");
            let saved = load_graph(&graph.graph_id, true).unwrap();
            assert_eq!(saved.revision, 1);
            assert_eq!(saved.node_revisions["a"], 1);
            assert_eq!(
                saved.node_revisions.get("z").copied().unwrap_or_default(),
                0
            );
            assert_eq!(saved.team.revision, 1);
            assert_eq!(saved.team.tasks["first"].revision, 1);
            let GraphRun::Static(run) = saved.run else {
                unreachable!()
            };
            assert_eq!(
                run.nodes["a"].output.as_deref(),
                Some("graph evidence".repeat(2_000).as_str())
            );
        });
    }

    #[test]
    fn graph_inherit_defaults_to_cwd_and_skills() {
        let node: StaticNodeSpec = serde_json::from_value(json!({
            "id": "node",
            "description": "worker",
            "prompt": "complete the task",
        }))
        .unwrap();
        let candidate: DynamicCandidateSpec = serde_json::from_value(json!({
            "id": "candidate",
            "role": "worker",
        }))
        .unwrap();

        for inherit in [node.inherit, candidate.inherit] {
            assert_eq!(inherit, "cwd,skills");
            assert_eq!(
                InheritOptions::from_value(&json!(inherit)).unwrap(),
                InheritOptions {
                    cwd: true,
                    skills: true,
                    history: false,
                    memory: false,
                }
            );
        }
    }

    #[test]
    fn graph_inherit_preserves_explicit_values() {
        for inherit in ["none", "all", "history", "cwd", "skills", "cwd,history"] {
            let node: StaticNodeSpec = serde_json::from_value(json!({
                "id": "node",
                "description": "worker",
                "prompt": "complete the task",
                "inherit": inherit,
            }))
            .unwrap();
            let candidate: DynamicCandidateSpec = serde_json::from_value(json!({
                "id": "candidate",
                "role": "worker",
                "inherit": inherit,
            }))
            .unwrap();
            assert_eq!(node.inherit, inherit);
            assert_eq!(candidate.inherit, inherit);
        }
    }

    #[test]
    fn static_graph_rejects_cycle() {
        let nodes = runtime_nodes(&["a", "b"]);
        let edges = vec![
            StaticEdgeSpec {
                from: "a".to_string(),
                to: "b".to_string(),
                condition: EdgeCondition::Always,
            },
            StaticEdgeSpec {
                from: "b".to_string(),
                to: "a".to_string(),
                condition: EdgeCondition::Always,
            },
        ];
        assert_eq!(
            validate_static_graph(&nodes, &edges).unwrap_err(),
            "static graph contains a cycle"
        );
    }

    #[test]
    fn static_graph_rejects_duplicate_edges() {
        let nodes = runtime_nodes(&["a", "b"]);
        let edge = StaticEdgeSpec {
            from: "a".to_string(),
            to: "b".to_string(),
            condition: EdgeCondition::Always,
        };
        let edges = vec![edge.clone(), edge];
        assert_eq!(
            validate_static_graph(&nodes, &edges).unwrap_err(),
            "duplicate edge: a -> b"
        );
    }

    #[test]
    fn conditional_edge_skips_when_no_branch_matches() {
        let mut nodes = FxHashMap::default();
        nodes.insert(
            "source".to_string(),
            StaticNodeRuntime {
                spec: node("source"),
                status: NodeStatus::Completed,
                runtime_task_id: None,
                team_task_id: None,
                attempts: 1,
                output: Some("no".to_string()),
                error: None,
            },
        );
        nodes.insert(
            "target".to_string(),
            StaticNodeRuntime {
                spec: node("target"),
                status: NodeStatus::Pending,
                runtime_task_id: None,
                team_task_id: None,
                attempts: 0,
                output: None,
                error: None,
            },
        );
        let run = StaticGraphRun {
            nodes,
            edges: vec![StaticEdgeSpec {
                from: "source".to_string(),
                to: "target".to_string(),
                condition: EdgeCondition::OutputContains {
                    text: "yes".to_string(),
                },
            }],
            state: json!({}),
        };
        assert!(matches!(
            static_node_decision(&run, "target").unwrap(),
            StaticDecision::Skip
        ));
    }

    #[test]
    fn reducers_append_and_merge_without_losing_existing_state() {
        let mut state = json!({"items": [1], "object": {"a": 1}});
        let mut append = node("append");
        append.output_key = Some("items".to_string());
        append.reducer = Reducer::AppendArray;
        reduce_static_output(&mut state, &append, "2").unwrap();
        let mut merge = node("merge");
        merge.output_key = Some("object".to_string());
        merge.reducer = Reducer::MergeObject;
        reduce_static_output(&mut state, &merge, r#"{"b":2}"#).unwrap();
        assert_eq!(state, json!({"items": [1, 2], "object": {"a": 1, "b": 2}}));
    }

    #[test]
    fn dynamic_topology_keeps_minimum_and_connects_ranked_agents() {
        let candidates = [
            ("a", 0.9, "alpha answer"),
            ("b", 0.8, "beta answer"),
            ("c", 0.1, "unrelated"),
        ]
        .into_iter()
        .map(|(id, relevance, answer)| {
            (
                id.to_string(),
                DynamicCandidateRuntime {
                    spec: DynamicCandidateSpec {
                        id: id.to_string(),
                        role: id.to_string(),
                        prompt: String::new(),
                        agent: None,
                        model: None,
                        inherit: default_inherit(),
                    },
                    task_id: None,
                    team_task_id: None,
                    attempts: 0,
                    initial: Some(answer.to_string()),
                    scores: FxHashMap::from_iter([
                        ("a".to_string(), 0.9),
                        ("b".to_string(), 0.8),
                        ("c".to_string(), 0.1),
                    ]),
                    score_complete: true,
                    relevance,
                    selected: false,
                    forward: None,
                    reverse: None,
                    error: None,
                },
            )
        })
        .collect();
        let mut run = DynamicGraphRun {
            phase: DynamicPhase::Scoring,
            policy: DynamicPolicy {
                min_selected: 2,
                max_selected: 2,
                relevance_threshold: 0.5,
                edge_threshold: 0.95,
                pool: PoolStrategy::Max,
                pool_agent: None,
                pool_model: None,
            },
            candidates,
            edges: Vec::new(),
            pool_task_id: None,
            pool_team_task_id: None,
            result: None,
            error: None,
        };
        select_dynamic_topology("alpha beta", &mut run);
        assert_eq!(ranked_selected_ids(&run), vec!["a", "b"]);
        assert!(
            run.edges
                .iter()
                .any(|edge| edge.from == "a" && edge.to == "b")
        );
    }
}
