//! Agent routing: skill manifest loading, primary agent activation,
//! hot-reload, and runtime manifest initialization.
//!
//! Extracted from `driver/mod.rs` (review Finding #1, Phase 2).

use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::ai::{
    agents::{self, AgentManifest},
    skills::{self, SkillManifest},
    types::App,
};

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

/// Loads all agents fresh from disk, enabling hot-reload of newly added/modified agents.
/// Returns a message to show when something changed; the foreground driver
/// prints it after clearing the dynamic status line.
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
    crate::ai::prompt::completion::CommandCompleter::set_agent_manifests(&new_agents);
    *agent_manifests = Arc::new(new_agents);
    Some(message)
}

/// Computes a stable fingerprint from the key manifest fields, used to detect
/// added, removed, and modified agents.
pub(super) fn agent_manifests_fingerprint(agents: &[AgentManifest]) -> [u8; 32] {
    let mut entries: Vec<&AgentManifest> = agents.iter().collect();
    entries.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.source_path.cmp(&b.source_path))
    });
    let mut hasher = Sha256::new();
    for manifest in entries {
        AgentReloadIdentity(manifest).update_hasher(&mut hasher);
    }
    hasher.finalize().into()
}

/// Canonical byte contract for hot-reload identity. Keep this encoder stable:
/// persisted behavior tests intentionally freeze the legacy field order and delimiters.
struct AgentReloadIdentity<'a>(&'a AgentManifest);

impl AgentReloadIdentity<'_> {
    fn update_hasher(&self, hasher: &mut Sha256) {
        let manifest = self.0;
        hasher.update(manifest.name.as_bytes());
        hasher.update(b"\0");
        hasher.update(manifest.source_path.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(manifest.description.as_bytes());
        hasher.update(b"\0");
        hasher.update(manifest.prompt.as_bytes());
        hasher.update(b"\0");
        hasher.update(manifest.system_prompt.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(manifest.model.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", manifest.mode).as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", manifest.temperature).as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", manifest.max_steps).as_bytes());
        hasher.update(b"\0");
        for t in &manifest.tools {
            hasher.update(t.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        for g in &manifest.tool_groups {
            hasher.update(g.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        for s in &manifest.mcp_servers {
            hasher.update(s.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        hasher.update([manifest.disable_mcp_tools as u8]);
        hasher.update(b"\0");
        hasher.update([manifest.disabled as u8, manifest.hidden as u8]);
        hasher.update(b"\0");
        hasher.update(manifest.color.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"|");
    }
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

    let loaded_skill_manifests = Arc::new(load_skill_manifests(app.cli.no_skills));
    install_runtime_manifests(
        app,
        skill_manifests,
        agent_manifests,
        manifests_loaded,
        loaded_skill_manifests,
    );
}

/// Installs the skill snapshot already discovered in the background, avoiding
/// an interactive first-screen wait for the disk scan to finish.
pub(super) fn install_runtime_manifests(
    app: &mut App,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
    manifests_loaded: &mut bool,
    loaded_skill_manifests: Arc<Vec<SkillManifest>>,
) {
    if *manifests_loaded {
        return;
    }

    *skill_manifests = loaded_skill_manifests;
    crate::ai::prompt::completion::CommandCompleter::set_skill_manifests(
        skill_manifests.as_slice(),
    );
    *agent_manifests = Arc::new(agents::load_all_agents());
    crate::ai::prompt::completion::CommandCompleter::set_agent_manifests(
        agent_manifests.as_slice(),
    );

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

#[cfg(test)]
mod tests {
    use super::agent_manifests_fingerprint;
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};

    fn manifest_fixture() -> AgentManifest {
        AgentManifest {
            name: "alpha".to_string(),
            description: "First agent".to_string(),
            mode: AgentMode::Primary,
            model: Some("model-a".to_string()),
            temperature: Some(0.25),
            max_steps: Some(7),
            prompt: "Prompt A".to_string(),
            system_prompt: Some("System A".to_string()),
            tools: vec!["read_file".to_string(), "apply_patch".to_string()],
            tool_groups: vec!["core".to_string(), "executor".to_string()],
            mcp_servers: vec!["browser".to_string(), "excel".to_string()],
            disable_mcp_tools: true,
            model_tier: Some(AgentModelTier::Standard),
            disabled: false,
            hidden: true,
            color: Some("#123456".to_string()),
            source_path: Some("/agents/alpha.agent".to_string()),
        }
    }

    #[test]
    fn agent_manifests_fingerprint_matches_golden_digest() {
        assert_eq!(
            agent_manifests_fingerprint(&[manifest_fixture()]),
            [
                140, 115, 69, 44, 131, 187, 137, 203, 220, 193, 19, 194, 52, 186, 43, 85, 249, 202,
                25, 228, 148, 174, 159, 6, 122, 59, 152, 207, 143, 72, 64, 11,
            ]
        );
    }

    #[test]
    fn agent_manifests_fingerprint_is_independent_of_input_order() {
        let first = manifest_fixture();
        let mut second = manifest_fixture();
        second.name = "beta".to_string();
        second.description = "Second agent".to_string();
        second.source_path = Some("/agents/beta.agent".to_string());

        assert_eq!(
            agent_manifests_fingerprint(&[first.clone(), second.clone()]),
            agent_manifests_fingerprint(&[second, first])
        );
    }

    #[test]
    fn agent_manifests_fingerprint_tracks_each_currently_hashed_field() {
        let fixture = manifest_fixture();
        let baseline = agent_manifests_fingerprint(std::slice::from_ref(&fixture));
        let mutations: &[(&str, fn(&mut AgentManifest))] = &[
            ("name", |m| m.name.push_str("-changed")),
            ("source_path", |m| {
                m.source_path = Some("/agents/changed.agent".to_string())
            }),
            ("description", |m| m.description.push_str(" changed")),
            ("prompt", |m| m.prompt.push_str(" changed")),
            ("system_prompt", |m| {
                m.system_prompt = Some("Changed system".to_string())
            }),
            ("model", |m| m.model = Some("model-b".to_string())),
            ("mode", |m| m.mode = AgentMode::Subagent),
            ("temperature", |m| m.temperature = Some(0.75)),
            ("max_steps", |m| m.max_steps = Some(8)),
            ("tools", |m| m.tools.push("write_file".to_string())),
            ("tool_groups", |m| {
                m.tool_groups.push("knowledge".to_string())
            }),
            ("mcp_servers", |m| {
                m.mcp_servers.push("calendar".to_string())
            }),
            ("disable_mcp_tools", |m| m.disable_mcp_tools = false),
            ("disabled", |m| m.disabled = true),
            ("hidden", |m| m.hidden = false),
            ("color", |m| m.color = Some("#abcdef".to_string())),
        ];

        for (field, mutate) in mutations {
            let mut changed = fixture.clone();
            mutate(&mut changed);
            assert_ne!(
                agent_manifests_fingerprint(&[changed]),
                baseline,
                "changing {field} must change the fingerprint"
            );
        }

        let mut changed_tier = fixture;
        changed_tier.model_tier = Some(AgentModelTier::Heavy);
        assert_eq!(
            agent_manifests_fingerprint(&[changed_tier]),
            baseline,
            "model_tier is intentionally outside the current fingerprint contract"
        );
    }
}
