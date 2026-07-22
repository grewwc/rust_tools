//! Agent routing: skill manifest loading, primary agent activation,
//! auto-routing, hot-reload, and runtime manifest initialization.
//!
//! Extracted from `driver/mod.rs` (review Finding #1, Phase 2).

use std::sync::Arc;

use super::{agent_router, note_search};
use crate::ai::{
    agents::{self, AgentManifest},
    skills::{self, SkillManifest},
    types::App,
};
use crate::commonw::configw;

#[crate::ai::agent_hang_span(
    "pre-fix",
    "S",
    "driver::run:load_all_skills",
    "[DEBUG] loading skills",
    "[DEBUG] loaded skills",
    { "no_skills": no_skills },
    {
        "count": __agent_hang_result.len(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(super) fn load_skill_manifests(no_skills: bool) -> Vec<SkillManifest> {
    if no_skills {
        Vec::new()
    } else {
        skills::load_all_skills()
    }
}

/// Activate a primary agent for the current session.
/// Updates app's current_agent, current_agent_manifest,
/// and switches the model if specified by the agent.
pub(super) fn activate_primary_agent(app: &mut App, agent: &AgentManifest) {
    app.current_agent = agent.name.clone();
    app.current_agent_manifest = Some(agent.clone());
    if let Some(model) = &agent.model {
        app.current_model = model.clone();
    }
}

pub(super) fn has_pending_foreground_process(app: &App) -> bool {
    let os = app.os.lock().unwrap();
    os.list_processes().into_iter().any(|proc| {
        proc.is_foreground
            && !matches!(
                proc.state,
                aios_kernel::kernel::ProcessState::Terminated
                    | aios_kernel::kernel::ProcessState::Ready
            )
    })
}

/// Check if auto-agent routing is enabled in config.
/// Auto-routing selects the best agent based on the question content.
///
/// 默认禁用：TF-IDF + logistic regression 的浅层文本匹配误报率高（如问题中
/// 出现 "skill" 就切到 prompt-skill agent），且 agent 切换会改变 system
/// prompt / 工具集 / model tier，影响整轮对话。用户可通过 `-a <agent>`
/// 手动指定，或配置 `ai.agents.auto_route.enable = true` 重新启用。
pub(super) fn auto_agent_routing_enabled() -> bool {
    !configw::get_all_config()
        .get_opt("ai.agents.auto_route.enable")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("false")
}

/// Get auto-routing strategy from config: "model" or "heuristic".
pub(super) fn auto_route_strategy() -> String {
    configw::get_all_config()
        .get_opt("ai.agents.auto_route.strategy")
        .unwrap_or_else(|| "model".to_string())
        .trim()
        .to_lowercase()
}

/// Auto-route to a different agent based on question content.
/// Activated when:
///   1. No explicit agent specified via CLI (-a/--agent)
///   2. Auto-routing is enabled in config
///   3. A suitable agent is found based on the question
///
/// Uses either:
///   - model strategy: Use ML model to predict best agent
///   - heuristic strategy: Rule-based routing
pub(super) fn maybe_auto_route_agent(
    app: &mut App,
    agent_manifests: &[AgentManifest],
    question: &str,
) {
    if app.cli.agent.is_some() || !auto_agent_routing_enabled() {
        return;
    }

    let history = note_search::read_recent_history(app);
    let decision = match auto_route_strategy().as_str() {
        "heuristic" => {
            let router = agent_router::HeuristicRouter;
            agent_router::AgentRouter::route(
                &router,
                agent_manifests,
                question,
                &history,
                &app.current_agent,
            )
        }
        _ => {
            let model_path = app.config.agent_route_model_path.clone();
            let router = agent_router::ModelRouter::new(model_path);
            agent_router::AgentRouter::route(
                &router,
                agent_manifests,
                question,
                &history,
                &app.current_agent,
            )
        }
    };
    let Some(decision) = decision else {
        return;
    };

    let Some(agent) = agents::find_agent_by_name(agent_manifests, &decision.agent_name) else {
        return;
    };
    if !agent.is_primary() || agent.disabled {
        return;
    }

    let old_agent = app.current_agent.clone();
    activate_primary_agent(app, agent);

    println!(
        "\n[Agent 自动切换: {} → {}] (原因: {})\n",
        old_agent, app.current_agent, decision.reason
    );
}

/// Loads all agents fresh from disk, enabling hot-reload of newly added/modified agents.
/// 发生变化时返回待展示消息，由前台 driver 在结束动态状态行后统一输出。
pub(super) fn reload_agent_manifests(
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
) -> Option<String> {
    let new_agents = agents::load_all_agents();
    let old_fingerprint = agent_manifests_fingerprint(agent_manifests.as_slice());
    let new_fingerprint = agent_manifests_fingerprint(new_agents.as_slice());
    if old_fingerprint == new_fingerprint {
        return None;
    }
    let added = new_agents.len() as i64 - agent_manifests.len() as i64;
    let message = if added > 0 {
        format!("[Agent 发现] 新发现 {} 个 agent(s)，已自动加载", added)
    } else if added < 0 {
        format!(
            "[Agent 发现] 移除 {} 个 agent(s)，共 {} 个",
            -added,
            new_agents.len()
        )
    } else {
        format!(
            "[Agent 发现] 检测到 agent 内容变更，已重新加载，共 {} 个",
            new_agents.len()
        )
    };
    *agent_manifests = Arc::new(new_agents);
    Some(message)
}

/// 基于 manifest 关键字段计算稳定指纹，用于检测增删改三类变更。
pub(super) fn agent_manifests_fingerprint(agents: &[AgentManifest]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut entries: Vec<&AgentManifest> = agents.iter().collect();
    entries.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.source_path.cmp(&b.source_path))
    });
    let mut hasher = Sha256::new();
    for m in entries {
        hasher.update(m.name.as_bytes());
        hasher.update(b"\0");
        hasher.update(m.source_path.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(m.description.as_bytes());
        hasher.update(b"\0");
        hasher.update(m.prompt.as_bytes());
        hasher.update(b"\0");
        hasher.update(m.system_prompt.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(m.model.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", m.mode).as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", m.temperature).as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", m.max_steps).as_bytes());
        hasher.update(b"\0");
        for t in &m.tools {
            hasher.update(t.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        for g in &m.tool_groups {
            hasher.update(g.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        for s in &m.mcp_servers {
            hasher.update(s.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        hasher.update([m.disable_mcp_tools as u8]);
        hasher.update(b"\0");
        hasher.update([m.disabled as u8, m.hidden as u8]);
        hasher.update(b"\0");
        hasher.update(m.color.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"|");
    }
    hasher.finalize().into()
}

pub(super) fn ensure_runtime_manifests_loaded(
    app: &mut App,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
    manifests_loaded: &mut bool,
) {
    if *manifests_loaded {
        return;
    }

    *skill_manifests = Arc::new(load_skill_manifests(app.cli.no_skills));
    *agent_manifests = Arc::new(agents::load_all_agents());

    if !skill_manifests.is_empty() {
        crate::ai::knowledge::indexing::embedder::warm_up();
    }

    if let Some(default_agent) = agents::find_agent_by_name(agent_manifests, &app.current_agent)
        && default_agent.is_primary()
        && !default_agent.disabled
    {
        activate_primary_agent(app, default_agent);
    }

    if let Some(agent_name) = &app.cli.agent {
        if let Some(agent) = agents::find_agent_by_name(agent_manifests, agent_name) {
            if agent.is_primary() && !agent.disabled {
                activate_primary_agent(app, agent);
                println!("[agent] using: {}", agent.name);
            } else {
                eprintln!(
                    "[Warning] Agent '{}' is not available, using default",
                    agent_name
                );
            }
        } else {
            eprintln!("[Warning] Agent '{}' not found, using default", agent_name);
        }
    }

    *manifests_loaded = true;
}
