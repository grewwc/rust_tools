//! Static Agent Graph 与动态 Graph-of-Agents 编排器。
//!
//! 两种图都只负责拓扑、状态机、checkpoint 与 prompt 组装；真正执行仍由父模块的
//! `task_*` / AIOS kernel 完成，因此取消、evidence、owner/session 隔离保持一致。

use super::{
    MAX_PROMPT_CHARS, MemberStatus, TeamBudget, TeamLifecycle, TeamMember, TeamState, TeamTask,
    TeamTaskState, active_identity, atomic_write_json, checkpoint_base_dir, clip_text,
    integrate_graph_task, now_ms, recover_orphaned_team_tasks, validate_identifier,
};
use crate::ai::tools::registry::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec,
};
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
        prune: ToolPrunePolicy::Never,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum Reducer {
    #[default]
    Replace,
    AppendArray,
    MergeObject,
    Concat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    "none".to_string()
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
    let input: GraphToolInput = serde_json::from_value(args.clone())
        .map_err(|error| format!("invalid run_agent_graph arguments: {error}"))?;
    match input.action.as_str() {
        "start" => start_graph(input),
        "advance" => mutate_graph(input.graph_id.as_deref(), |graph| {
            advance_graph(graph)?;
            Ok(graph_view(graph))
        }),
        "status" => {
            let graph = load_graph(required_graph_id(input.graph_id.as_deref())?)?;
            render_json(&graph_view(&graph))
        }
        "cancel" => mutate_graph(input.graph_id.as_deref(), |graph| {
            cancel_graph(graph)?;
            Ok(graph_view(graph))
        }),
        other => Err(format!(
            "unsupported run_agent_graph action {other:?}; expected start, advance, status, or cancel"
        )),
    }
}

fn start_graph(input: GraphToolInput) -> Result<String, String> {
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
    advance_graph(&mut checkpoint)?;
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
    validate_static_edges(&nodes, &input.edges)?;
    validate_acyclic(nodes.keys().cloned().collect(), &input.edges)?;
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
        // 图进入 Failed 终态：预算耗尽/启动失败会遗留 Running 节点或在途候选的
        // runtime 任务（进程/channel 未回收），统一取消内核进程并清理引用，避免泄漏。
        cancel_active_runtime_tasks(graph, "agent graph failed");
    }
    graph.updated_at_unix_ms = now_ms();
    graph.team.updated_at_unix_ms = graph.updated_at_unix_ms;
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
                // 轮次中途错误绝不冒泡：poll 失败（任务丢失/所有权变更）时把节点
                // 就地标记为失败并清掉 task 引用。若 `?` 冒泡，checkpoint 不会保存，
                // 磁盘上节点仍带 runtime_task_id，下次 advance 对同一丢失任务重复
                // poll，永久卡死。错误以 Failed 终态呈现，下游条件边按失败跳过。
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
                    // 归约失败是配置性错误：节点已真实完成，但输出未能并入图状态，
                    // 按失败处理以免下游读到缺失的状态键。绝不冒泡——poll 已消费
                    // TASK_REGISTRY 条目，若在这里 `?` 传播，checkpoint 不会保存，
                    // 磁盘上节点仍为 Running，下次 advance 会永久卡死。
                    if let Err(error) =
                        reduce_static_output(&mut run.state, &node.spec, &result.output)
                    {
                        node.status = NodeStatus::Failed;
                        node.error = Some(format!("static reduce failed: {error}"));
                    }
                    // 历史集成尽力而为：证据已由 poll 持久化，失败仅追加诊断。
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
                            // spawn 失败（如 member busy）不可恢复：就地标记节点失败
                            // 并终止本轮。错误冒泡会使 checkpoint 不保存，磁盘上节点
                            // 仍为 Pending，下次 advance 反复重试并再次报错，永久卡死。
                            let node = run.nodes.get_mut(&node_id).unwrap();
                            node.status = NodeStatus::Failed;
                            node.error = Some(format!("spawn failed: {error}"));
                            launched = true;
                            break;
                        }
                    }) else {
                        // 预算耗尽：无法再启动任何节点。把尚未结束的节点标记为失败，
                        // 让图以 Failed 正常终止——错误若冒泡会使 checkpoint 永不
                        // 保存，导致永久卡死。
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
                    // decision 失败（配置性）：就地标记节点失败并终止本轮，
                    // 绝不冒泡——否则 checkpoint 不保存，节点永久 Pending 卡死。
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
            "You are node `{}` in a static agent DAG. Work directly; do not delegate.\n\n# Graph question\n{}\n\n# Node task\n{}\n\n# Current reduced state\n{}\n\n# Active upstream outputs\n{}",
            node.spec.id,
            question,
            node.spec.prompt,
            serde_json::to_string_pretty(state).unwrap_or_else(|_| "{}".to_string()),
            if upstream.is_empty() {
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
        // 预算已耗尽：不再启动新任务，等待 advance_graph 将图标记为 Failed。
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
                // 轮次中途错误绝不冒泡：poll 失败时清掉 task 引用并复位成员，
                // 下一轮 advance 重新 spawn 该候选；反复失败最终由预算耗尽终止。
                // 冒泡会使 checkpoint 不保存，磁盘残留 task_id，下次对同一丢失
                // 任务重复 poll，永久卡死。
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
        // 集成失败不能冒泡：poll 已消费 TASK_REGISTRY 条目并持久化证据，若在此
        // `?` 传播，checkpoint 不会保存，磁盘上候选仍带 runtime_task_id，下次
        // advance 会因任务已不在 registry 而永久卡死。候选状态已就地更新，
        // 集成失败仅影响父级可见的证据回执，忽略即可。
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
                // 同候选分支：pooling 状态已就地提交，集成失败仅丢失父级回执，忽略。
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
                // pool 任务丢失：清掉引用并复位成员，用最优候选输出兜底完成图，
                // 绝不冒泡。
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
                // 不可恢复的配置错误：让图以 Failed 终止，绝不冒泡。
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
                // spawn 失败（如 member busy）：让图以 Failed 终止，绝不冒泡。
                run.error = Some(format!("spawn failed: {error}"));
                break;
            }
        }) else {
            // 预算耗尽：无法再启动任何候选，记录失败并停止启动，由
            // advance_dynamic 的 error 早退 + advance_graph 将图标记为 Failed。
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
            "You are candidate `{id}` in a Graph-of-Agents run. Work directly and independently; do not delegate.\nRole: {role}\nSpecial instructions: {}\n\nQuestion:\n{question}\n\nProduce your best self-contained answer.",
            candidate.spec.prompt
        ),
        DynamicPhase::Scoring => {
            let answers = ordered_candidate_outputs(run, |candidate| candidate.initial.as_deref());
            format!(
                "You are the relevance selector `{id}` in a Graph-of-Agents run. Do not solve the question again. Score every candidate answer from 0.0 to 1.0 for correctness, usefulness, and relevance to the question. Return only the required JSON object.\n\nQuestion:\n{question}\n\nCandidate answers:\n{answers}"
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
                "You are candidate `{id}` in the forward message-passing phase. Work directly; do not delegate.\nRole: {role}\n\nQuestion:\n{question}\n\nYour initial answer:\n{}\n\nMore-relevant inbound answers:\n{}\n\nRefine your answer using useful evidence, while correcting conflicts.",
                candidate.initial.as_deref().unwrap_or(""),
                if messages.is_empty() {
                    "<none>"
                } else {
                    &messages
                }
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
                "You are candidate `{id}` in the reverse propagation phase. Work directly; do not delegate.\nRole: {role}\n\nQuestion:\n{question}\n\nYour forward answer:\n{}\n\nFeedback from less-central neighbors:\n{}\n\nProduce a final corrected answer. Preserve strong conclusions and incorporate valid overlooked details.",
                candidate.forward.as_deref().unwrap_or(""),
                if messages.is_empty() {
                    "<none>"
                } else {
                    &messages
                }
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
        // 无 selected 候选（如 min_selected=0 且阈值过高）：配置性错误，就地标记
        // Failed 落盘。`?` 冒泡会使 checkpoint 不保存，图永久卡在 Running，
        // 每次 mutate 都报同样错误，只能靠 cancel 脱困。
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
            "You are the final judge in a Graph-of-Agents run. Work directly; do not delegate. Synthesize one accurate, self-contained final answer. Resolve disagreements using evidence and the relevance weights; do not mention the orchestration process.\n\nQuestion:\n{question}\n\nCandidate final answers:\n{answers}"
        ),
        MAX_PROMPT_CHARS,
    );
    let spawned =
        spawn_graph_member_task(graph_id, team, &member_id, "dynamic:pooling", &prompt, None);
    let Some((task_id, team_task_id)) = (match spawned {
        Ok(pair) => pair,
        Err(error) => {
            // spawn 失败（如 member busy）：不可恢复，让图以 Failed 终止，绝不冒泡。
            run.error = Some(format!("pooling spawn failed: {error}"));
            return Ok(());
        }
    }) else {
        // 预算耗尽：无法启动 pooling 任务，记录失败并让图以 Failed 终止。
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
        // 预算耗尽不是错误而是终止条件：返回 Ok(None)，由调用方把未完成的工作
        // 标记为失败并让图正常终止。若此处返回 Err，advance 会把错误冒泡且不
        // 保存 checkpoint，导致图永久卡死。
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
    let team_task_id = format!("g-{}-{}", &graph_id[..8], Uuid::new_v4().simple());
    team.tasks.insert(
        team_task_id.clone(),
        TeamTask {
            id: team_task_id.clone(),
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
    for task in graph.team.tasks.values_mut() {
        if task.state == TeamTaskState::Running {
            task.state = TeamTaskState::Cancelled;
            task.runtime_task_id = None;
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

/// 图进入 Failed 终态时的在途任务回收：预算耗尽/启动失败会遗留 Running 节点或
/// 动态候选的 runtime 任务（内核进程/channel 未回收），统一走 execute_task_cancel
/// 取消进程并清理引用。best-effort：取消失败不阻塞 Failed 终态标记，会话 teardown
/// 的进程组回收仍是最兜底防线。
fn cancel_active_runtime_tasks(graph: &mut AgentGraphCheckpoint, reason: &str) {
    let task_ids = active_graph_task_ids(graph);
    if !task_ids.is_empty() {
        let _ = execute_task_cancel(&json!({
            "task_ids": task_ids,
            "reason": format!("{reason}: {}", graph.graph_id),
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
    for task in graph.team.tasks.values_mut() {
        if task.state == TeamTaskState::Running {
            task.state = TeamTaskState::Cancelled;
            task.runtime_task_id = None;
        }
    }
    for member in graph.team.members.values_mut() {
        member.status = MemberStatus::Idle;
        member.active_task_id = None;
    }
}

fn validate_static_edges(
    nodes: &FxHashMap<String, StaticNodeRuntime>,
    edges: &[StaticEdgeSpec],
) -> Result<(), String> {
    let mut seen = FxHashSet::default();
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
        if !seen.insert((edge.from.clone(), edge.to.clone())) {
            return Err(format!("duplicate edge: {} -> {}", edge.from, edge.to));
        }
    }
    Ok(())
}

fn validate_acyclic(nodes: Vec<String>, edges: &[StaticEdgeSpec]) -> Result<(), String> {
    let mut indegree: FxHashMap<String, usize> = nodes.iter().map(|id| (id.clone(), 0)).collect();
    let mut outgoing: FxHashMap<String, Vec<String>> = FxHashMap::default();
    for edge in edges {
        *indegree.get_mut(&edge.to).expect("edge validated") += 1;
        outgoing
            .entry(edge.from.clone())
            .or_default()
            .push(edge.to.clone());
    }
    let mut queue: Vec<String> = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then(|| id.clone()))
        .collect();
    let mut visited = 0;
    while let Some(id) = queue.pop() {
        visited += 1;
        if let Some(children) = outgoing.get(&id) {
            for child in children {
                let degree = indegree.get_mut(child).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    queue.push(child.clone());
                }
            }
        }
    }
    if visited != nodes.len() {
        return Err("static graph contains a cycle".to_string());
    }
    Ok(())
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

fn load_graph(graph_id: &str) -> Result<AgentGraphCheckpoint, String> {
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
    let (session_id, owner_pid) = active_identity()?;
    if graph.session_id != session_id {
        return Err("agent graph belongs to another session".to_string());
    }
    if graph.owner_pid != owner_pid {
        graph.owner_pid = owner_pid;
        graph.team.owner_pid = owner_pid;
        recover_orphaned_team_tasks(&mut graph.team);
        recover_orphaned_graph(&mut graph);
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
    operation: impl FnOnce(&mut AgentGraphCheckpoint) -> Result<Value, String>,
) -> Result<String, String> {
    let graph_id = required_graph_id(graph_id)?;
    let mut graph = load_graph(graph_id)?;
    let value = operation(&mut graph)?;
    graph.updated_at_unix_ms = now_ms();
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

fn render_json(value: &Value) -> Result<String, String> {
    serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to render graph state: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str) -> StaticNodeSpec {
        StaticNodeSpec {
            id: id.to_string(),
            description: id.to_string(),
            prompt: id.to_string(),
            agent: None,
            model: None,
            inherit: "none".to_string(),
            response_schema: None,
            output_key: None,
            reducer: Reducer::Replace,
            max_retries: 0,
        }
    }

    #[test]
    fn static_graph_rejects_cycle() {
        let nodes = vec!["a".to_string(), "b".to_string()];
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
        assert!(validate_acyclic(nodes, &edges).is_err());
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
                        inherit: "none".to_string(),
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
