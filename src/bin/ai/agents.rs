use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use rust_tools::cw::SkipMap;

use crate::commonw::{configw, utils::expanduser};

const BUILTIN_AGENTS: &[(&str, &str)] = &[
    (
        "build.agent",
        include_str!("builtin_agents/build.agent"),
    ),
    (
        "plan.agent",
        include_str!("builtin_agents/plan.agent"),
    ),
    (
        "explore.agent",
        include_str!("builtin_agents/explore.agent"),
    ),
];

/// Categorizes an agent's role: `Primary` for main conversation,
/// `Subagent` for delegated tasks, or `All` for both.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum AgentMode {
    #[serde(rename = "primary")]
    Primary,
    #[serde(rename = "subagent")]
    Subagent,
    #[serde(rename = "all")]
    All,
}

impl Default for AgentMode {
    fn default() -> Self {
        AgentMode::All
    }
}

/// Parsed configuration for an agent, loaded from a `.agent` file
/// with front-matter metadata and a prompt body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AgentManifest {
    pub(super) name: String,
    pub(super) description: String,
    #[serde(default)]
    pub(super) mode: AgentMode,
    #[serde(default)]
    pub(super) model: Option<String>,
    #[serde(default)]
    pub(super) temperature: Option<f64>,
    #[serde(default)]
    pub(super) max_steps: Option<usize>,
    #[serde(default)]
    pub(super) prompt: String,
    #[serde(default)]
    pub(super) system_prompt: Option<String>,
    #[serde(default)]
    pub(super) tools: Vec<String>,
    #[serde(default)]
    pub(super) tool_groups: Vec<String>,
    #[serde(default)]
    pub(super) mcp_servers: Vec<String>,
    #[serde(default)]
    pub(super) disabled: bool,
    #[serde(default)]
    pub(super) hidden: bool,
    #[serde(default)]
    pub(super) color: Option<String>,
    #[serde(skip)]
    pub(super) source_path: Option<String>,
}

impl AgentManifest {
    pub(super) fn build_system_prompt(&self) -> String {
        let mut prompt = self.system_prompt.clone().unwrap_or_default();

        if !self.prompt.is_empty() {
            if !prompt.is_empty() {
                prompt.push_str("\n\n");
            }
            prompt.push_str(self.prompt.as_str());
        }

        prompt
    }

    pub(super) fn is_primary(&self) -> bool {
        matches!(self.mode, AgentMode::Primary | AgentMode::All)
    }

    pub(super) fn is_subagent(&self) -> bool {
        matches!(self.mode, AgentMode::Subagent | AgentMode::All)
    }
}

/// Loads all builtin and user-defined agents, merging them by name
/// (user agents override builtins) and sorting primaries first.
pub(super) fn load_all_agents() -> Vec<AgentManifest> {
    let dir = agents_dir();
    let _ = ensure_seeded_agents_dir(&dir);
    let mut by_name: Box<SkipMap<String, AgentManifest>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);

    for (filename, content) in BUILTIN_AGENTS {
        if let Ok(mut agent) = parse_agent_front_matter(content) {
            agent.source_path = Some(format!("builtin:{filename}"));
            by_name.insert(agent.name.clone(), agent);
        }
    }

    for agent in load_agents_from_dir(&dir) {
        if let Some(existing) = by_name.get_ref(&agent.name)
            && existing
                .source_path
                .as_deref()
                .is_some_and(|p| p.starts_with("builtin:"))
        {
            continue;
        }
        by_name.insert(agent.name.clone(), agent);
    }

    let mut out = (&*by_name)
        .into_iter()
        .map(|(_, v)| v.clone())
        .collect::<Vec<_>>();
    out.sort_by(|a, b| {
        let primary_a = a.is_primary() as i32;
        let primary_b = b.is_primary() as i32;
        primary_b.cmp(&primary_a).then(a.name.cmp(&b.name))
    });
    out
}

/// Filters agents that can serve as primary agents, excluding
/// disabled and hidden ones.
pub(super) fn get_primary_agents(agents: &[AgentManifest]) -> Vec<&AgentManifest> {
    agents.iter().filter(|a| a.is_primary() && !a.disabled && !a.hidden).collect()
}

/// Filters agents that can be spawned as subagents, excluding
/// disabled and hidden ones.
pub(super) fn get_subagents(agents: &[AgentManifest]) -> Vec<&AgentManifest> {
    agents.iter().filter(|a| a.is_subagent() && !a.disabled && !a.hidden).collect()
}

pub(super) fn find_agent_by_name<'a>(agents: &'a [AgentManifest], name: &str) -> Option<&'a AgentManifest> {
    agents.iter().find(|a| a.name == name)
}

pub(super) fn agents_dir() -> PathBuf {
    let cfg = configw::get_all_config();
    let raw = cfg.get_opt("ai.agents.dir").unwrap_or_default();
    let path = if raw.trim().is_empty() {
        "~/.config/rust_tools/agents".to_string()
    } else {
        raw
    };
    PathBuf::from(expanduser(&path).as_ref())
}

fn looks_like_front_matter_agent(content: &str) -> bool {
    content.lines().next().is_some_and(|l| l.trim() == "---")
}

fn parse_agent_front_matter(content: &str) -> Result<AgentManifest, String> {
    let mut lines = content.lines();
    let Some(first) = lines.next() else {
        return Err("empty agent file".to_string());
    };
    if first.trim() != "---" {
        return Err("missing front matter start".to_string());
    }

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut mode: Option<String> = None;
    let mut model: Option<String> = None;
    let mut temperature: Option<f64> = None;
    let mut max_steps: Option<usize> = None;
    let mut system_prompt: Option<String> = None;
    let mut disabled = false;
    let mut hidden = false;
    let mut color: Option<String> = None;
    let mut tools: Vec<String> = Vec::new();
    let mut tool_groups: Vec<String> = Vec::new();
    let mut mcp_servers: Vec<String> = Vec::new();

    let mut body = String::new();
    let mut in_front_matter = true;
    let mut pending_list_key: Option<String> = None;

    for line in lines {
        if in_front_matter {
            if line.trim() == "---" {
                in_front_matter = false;
                pending_list_key = None;
                continue;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(key) = pending_list_key.as_deref()
                && trimmed.starts_with('-')
            {
                let v = trimmed.trim_start_matches('-').trim();
                let v = v.trim_matches('"').trim_matches('\'').to_string();
                if v.is_empty() {
                    continue;
                }
                match key {
                    "tools" => tools.push(v),
                    "tool_groups" => tool_groups.push(v),
                    "mcp_servers" => mcp_servers.push(v),
                    _ => {}
                }
                continue;
            }

            pending_list_key = None;

            let Some((k, v)) = trimmed.split_once(':') else {
                continue;
            };
            let key = k.trim();
            let value = v.trim();

            if value.is_empty() {
                pending_list_key = Some(key.to_string());
                continue;
            }

            let unquoted = value.trim_matches('"').trim_matches('\'');
            match key {
                "name" => name = Some(unquoted.to_string()),
                "description" => description = Some(unquoted.to_string()),
                "mode" => mode = Some(unquoted.to_string()),
                "model" => model = Some(unquoted.to_string()),
                "system_prompt" => system_prompt = Some(unquoted.to_string()),
                "color" => color = Some(unquoted.to_string()),
                "temperature" => {
                    temperature = unquoted.parse::<f64>().ok();
                }
                "max_steps" => {
                    max_steps = unquoted.parse::<usize>().ok();
                }
                "disabled" => {
                    disabled = unquoted.eq_ignore_ascii_case("true");
                }
                "hidden" => {
                    hidden = unquoted.eq_ignore_ascii_case("true");
                }
                "tools" => tools = parse_list_value(unquoted),
                "tool_groups" => tool_groups = parse_list_value(unquoted),
                "mcp_servers" => mcp_servers = parse_list_value(unquoted),
                _ => {}
            }
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }

    if in_front_matter {
        return Err("missing front matter end".to_string());
    }

    let Some(name) = name else {
        return Err("missing name".to_string());
    };

    let Some(description) = description else {
        return Err("missing description".to_string());
    };

    let agent_mode = match mode.as_deref() {
        Some("primary") => AgentMode::Primary,
        Some("subagent") => AgentMode::Subagent,
        Some("all") => AgentMode::All,
        None => AgentMode::All,
        Some(other) => return Err(format!("invalid mode: {}", other)),
    };

    Ok(AgentManifest {
        name,
        description,
        mode: agent_mode,
        model: model.filter(|s| !s.trim().is_empty()),
        temperature,
        max_steps,
        prompt: body.trim().to_string(),
        system_prompt: system_prompt.filter(|s| !s.trim().is_empty()),
        tools,
        tool_groups,
        mcp_servers,
        disabled,
        hidden,
        color: color.filter(|s| !s.trim().is_empty()),
        source_path: None,
    })
}

fn parse_list_value(s: &str) -> Vec<String> {
    let s = s.trim();
    if s.is_empty() {
        return Vec::new();
    }
    let s = s.trim_start_matches('[').trim_end_matches(']');
    s.split(',')
        .map(|x| x.trim().trim_matches('"').trim_matches('\''))
        .filter(|x| !x.is_empty())
        .map(|x| x.to_string())
        .collect()
}

fn parse_agent_front_matter_with_path(content: &str, path: &Path) -> Result<AgentManifest, String> {
    let mut agent = parse_agent_front_matter(content)?;
    agent.source_path = Some(path.display().to_string());
    Ok(agent)
}

fn ensure_seeded_agents_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("failed to create agents dir: {e}"))?;
    Ok(())
}

fn load_agents_from_dir(dir: &Path) -> Vec<AgentManifest> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        if !looks_like_front_matter_agent(&content) {
            continue;
        }
        if let Ok(agent) = parse_agent_front_matter_with_path(&content, &path) {
            out.push(agent);
        }
    }
    out
}
