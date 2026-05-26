use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::ai::skills::SkillManifest;
use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

fn params_discover_skills() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional substring filter applied to skill name, description, tool names, tool groups, MCP servers, and source path."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of skills to return (default: 20, max: 100)."
            },
            "include_capabilities": {
                "type": "boolean",
                "description": "If true, include tool names, tool groups, and MCP servers for each skill."
            }
        }
    })
}

fn skill_matches_query(skill: &SkillManifest, query: &str) -> bool {
    if query.trim().is_empty() {
        return true;
    }
    let query = query.to_ascii_lowercase();
    skill.name.to_ascii_lowercase().contains(&query)
        || skill.description.to_ascii_lowercase().contains(&query)
        || skill
            .triggers
            .iter()
            .any(|item| item.to_ascii_lowercase().contains(&query))
        || skill
            .tools
            .iter()
            .any(|item| item.to_ascii_lowercase().contains(&query))
        || skill
            .tool_groups
            .iter()
            .any(|item| item.to_ascii_lowercase().contains(&query))
        || skill
            .mcp_servers
            .iter()
            .any(|item| item.to_ascii_lowercase().contains(&query))
        || skill
            .source_path
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .contains(&query)
}

fn summarize_skill(skill: &SkillManifest, include_capabilities: bool) -> String {
    let source = skill.source_path.as_deref().unwrap_or("unknown");
    let mut line = format!(
        "- {} | priority={} | source={}",
        skill.name, skill.priority, source
    );
    if !skill.description.trim().is_empty() {
        line.push_str(&format!(" | {}", skill.description.trim()));
    }
    if include_capabilities {
        if !skill.tools.is_empty() {
            line.push_str(&format!(" | tools={}", skill.tools.join(",")));
        }
        if !skill.triggers.is_empty() {
            line.push_str(&format!(" | triggers={}", skill.triggers.join(",")));
        }
        if !skill.tool_groups.is_empty() {
            line.push_str(&format!(" | tool_groups={}", skill.tool_groups.join(",")));
        }
        if !skill.mcp_servers.is_empty() {
            line.push_str(&format!(" | mcp_servers={}", skill.mcp_servers.join(",")));
        }
    }
    line
}

pub(crate) fn execute_discover_skills(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().unwrap_or("").trim();
    let limit = args["limit"].as_u64().unwrap_or(20).clamp(1, 100) as usize;
    let include_capabilities = args["include_capabilities"].as_bool().unwrap_or(false);

    let skills = crate::ai::skills::load_all_skills();
    let filtered = skills
        .into_iter()
        .filter(|skill| skill_matches_query(skill, query))
        .take(limit)
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        return Ok(if query.is_empty() {
            "No skills are currently available.".to_string()
        } else {
            format!("No skills matched query '{}'.", query)
        });
    }

    let mut lines = Vec::with_capacity(filtered.len() + 2);
    if query.is_empty() {
        lines.push(format!("{} skills available:", filtered.len()));
    } else {
        lines.push(format!(
            "{} skills matched query '{}':",
            filtered.len(),
            query
        ));
    }
    lines.extend(
        filtered
            .iter()
            .map(|skill| summarize_skill(skill, include_capabilities)),
    );
    lines.push(
        "This tool returns skill metadata only. Skill prompts stay unloaded until routing selects a skill.\nIf you called this during an active task, do not stop here: continue the turn by selecting the best matching skill, enabling missing tools, or answering directly if no skill is actually needed."
            .to_string(),
    );
    Ok(lines.join("\n"))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "discover_skills",
        description: "List available skills by metadata only. Use this to discover skill names, descriptions, priorities, and optional capability summaries without loading full skill prompts.",
        parameters: params_discover_skills,
        execute: execute_discover_skills,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

fn params_save_skill() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill identifier used in the YAML front matter (and default filename)."
            },
            "description": {
                "type": "string",
                "description": "Short summary shown in skill lists and matching."
            },
            "prompt": {
                "type": "string",
                "description": "Full skill prompt body (Markdown). Saved after the YAML front matter."
            },
            "system_prompt": {
                "type": "string",
                "description": "Optional additional system prompt text to include in the YAML front matter."
            },
            "triggers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Trigger phrases used for matching this skill."
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit tool names that the skill is allowed to use."
            },
            "tool_groups": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tool groups that the skill is allowed to use (e.g. builtin, executor; legacy openclaw is still accepted)."
            },
            "mcp_servers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required MCP server names needed to run this skill."
            },
            "priority": {
                "type": "integer",
                "description": "Optional match priority; higher values take precedence."
            },
            "author": {
                "type": "string",
                "description": "Author string (default: \"agent\")."
            },
            "version": {
                "type": "string",
                "description": "Version string (default: \"1.0.0\")."
            },
            "file_name": {
                "type": "string",
                "description": "Optional output filename; it will be sanitized and forced to end with .skill."
            },
            "overwrite": {
                "type": "boolean",
                "description": "If false, fail when the target file already exists (default: true)."
            }
        },
        "required": ["name", "prompt"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "save_skill",
        description: "Render and save a .skill file (YAML front matter + prompt body) into the configured skills directory.",
        parameters: params_save_skill,
        execute: execute_save_skill,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

fn resolve_configured_skills_dir() -> PathBuf {
    let cfg = crate::commonw::configw::get_all_config();
    let raw = cfg.get_opt(AiConfig::SKILLS_DIR).unwrap_or_default();
    if raw.trim().is_empty() {
        return crate::ai::skills::skills_dir();
    }
    PathBuf::from(crate::commonw::utils::expanduser(&raw).as_ref())
}

fn parse_string_array(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn yaml_quote(s: &str) -> String {
    let escaped = s.replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn safe_skill_file_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 8);
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('-');
        }
    }
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    let out = out.trim_matches('-').to_string();
    let out = if out.is_empty() {
        "skill".to_string()
    } else {
        out
    };
    if out.ends_with(".skill") {
        out
    } else {
        format!("{out}.skill")
    }
}

fn render_string_list_field(out: &mut String, key: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(&format!("{key}:\n"));
    for item in items {
        out.push_str(&format!("  - {}\n", yaml_quote(item)));
    }
}

fn build_skill_file_content(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().ok_or("Missing name")?.trim();
    let prompt = args["prompt"].as_str().ok_or("Missing prompt")?.trim();
    if name.is_empty() {
        return Err("name is empty".to_string());
    }
    if prompt.is_empty() {
        return Err("prompt is empty".to_string());
    }

    let description = args["description"].as_str().unwrap_or("").trim();
    let author = args["author"].as_str().unwrap_or("agent").trim();
    let version = args["version"].as_str().unwrap_or("1.0.0").trim();
    let system_prompt = args["system_prompt"].as_str().unwrap_or("").trim();
    let priority = args["priority"].as_i64().unwrap_or(0);
    let triggers = parse_string_array(&args["triggers"]);
    let tools = parse_string_array(&args["tools"]);
    let tool_groups = parse_string_array(&args["tool_groups"]);
    let mcp_servers = parse_string_array(&args["mcp_servers"]);

    let mut out = String::new();
    out.push_str("---\n");
    out.push_str(&format!("name: {}\n", yaml_quote(name)));
    if !description.is_empty() {
        out.push_str(&format!("description: {}\n", yaml_quote(description)));
    }
    out.push_str(&format!("author: {}\n", yaml_quote(author)));
    out.push_str(&format!("version: {}\n", yaml_quote(version)));
    if !system_prompt.is_empty() {
        out.push_str(&format!("system_prompt: {}\n", yaml_quote(system_prompt)));
    }
    if priority != 0 {
        out.push_str(&format!("priority: {priority}\n"));
    }
    render_string_list_field(&mut out, "triggers", &triggers);
    render_string_list_field(&mut out, "tools", &tools);
    render_string_list_field(&mut out, "tool_groups", &tool_groups);
    render_string_list_field(&mut out, "mcp_servers", &mcp_servers);
    out.push_str("---\n\n");
    out.push_str(prompt);
    out.push('\n');
    Ok(out)
}

pub(crate) fn execute_save_skill(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().ok_or("Missing name")?.trim();
    if name.is_empty() {
        return Err("name is empty".to_string());
    }
    let content = build_skill_file_content(args)?;
    let dir = resolve_configured_skills_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create skills dir: {e}"))?;

    let file_name = args["file_name"]
        .as_str()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| safe_skill_file_name(name));
    let file_name = safe_skill_file_name(&file_name);
    let path = dir.join(file_name);
    let overwrite = args["overwrite"].as_bool().unwrap_or(true);
    if path.exists() && !overwrite {
        return Err(format!(
            "Skill file already exists and overwrite=false: {}",
            path.display()
        ));
    }

    fs::write(&path, content).map_err(|e| format!("Failed to write skill file: {e}"))?;
    Ok(format!(
        "Skill saved: {}\nSkill name: {}",
        path.display(),
        name
    ))
}

// =============================================================================
// save_agent — 仿 save_skill，把 .agent 文件写入 user-level agents_dir。
// 禁止覆盖 builtin agent 文件名（防止"自我改写"误伤系统提示）。
// =============================================================================

fn params_save_agent() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Agent identifier (also default filename)."},
            "description": {"type": "string", "description": "Short summary used in agent listing and routing."},
            "prompt": {"type": "string", "description": "Full agent prompt body (Markdown). Saved after the YAML front matter."},
            "system_prompt": {"type": "string", "description": "Optional additional system prompt text included in the YAML front matter."},
            "mode": {"type": "string", "description": "Agent mode: primary | subagent | all (default: primary)."},
            "model": {"type": "string", "description": "Optional model override."},
            "model_tier": {"type": "string", "description": "Optional model tier: light | standard | heavy."},
            "color": {"type": "string", "description": "Optional UI color tag."},
            "temperature": {"type": "number", "description": "Optional temperature override."},
            "max_steps": {"type": "integer", "description": "Optional max iteration steps for this agent."},
            "tools": {"type": "array", "items": {"type": "string"}},
            "tool_groups": {"type": "array", "items": {"type": "string"}},
            "mcp_servers": {"type": "array", "items": {"type": "string"}},
            "routing_tags": {"type": "array", "items": {"type": "string"}},
            "file_name": {"type": "string", "description": "Optional output filename; sanitized and forced to end with .agent."},
            "overwrite": {"type": "boolean", "description": "If false, fail when target file already exists (default: true)."}
        },
        "required": ["name", "description", "prompt"]
    })
}

fn safe_agent_file_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 8);
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('-');
        }
    }
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    let out = out.trim_matches('-').to_string();
    let out = if out.is_empty() {
        "agent".to_string()
    } else {
        out
    };
    if out.ends_with(".agent") {
        out
    } else {
        format!("{out}.agent")
    }
}

fn build_agent_file_content(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().ok_or("Missing name")?.trim();
    let description = args["description"]
        .as_str()
        .ok_or("Missing description")?
        .trim();
    let prompt = args["prompt"].as_str().ok_or("Missing prompt")?.trim();
    if name.is_empty() {
        return Err("name is empty".to_string());
    }
    if description.is_empty() {
        return Err("description is empty".to_string());
    }
    if prompt.is_empty() {
        return Err("prompt is empty".to_string());
    }

    let system_prompt = args["system_prompt"].as_str().unwrap_or("").trim();
    let mode = args["mode"].as_str().unwrap_or("primary").trim();
    if !matches!(mode, "primary" | "subagent" | "all") {
        return Err(format!(
            "invalid mode '{mode}': expected primary | subagent | all"
        ));
    }
    let model = args["model"].as_str().unwrap_or("").trim();
    let model_tier = args["model_tier"].as_str().unwrap_or("").trim();
    if !model_tier.is_empty() && !matches!(model_tier, "light" | "standard" | "heavy") {
        return Err(format!(
            "invalid model_tier '{model_tier}': expected light | standard | heavy"
        ));
    }
    let color = args["color"].as_str().unwrap_or("").trim();
    let temperature = args["temperature"].as_f64();
    let max_steps = args["max_steps"].as_u64();
    let tools = parse_string_array(&args["tools"]);
    let tool_groups = parse_string_array(&args["tool_groups"]);
    let mcp_servers = parse_string_array(&args["mcp_servers"]);
    let routing_tags = parse_string_array(&args["routing_tags"]);

    let mut out = String::new();
    out.push_str("---\n");
    out.push_str(&format!("name: {}\n", yaml_quote(name)));
    out.push_str(&format!("description: {}\n", yaml_quote(description)));
    out.push_str(&format!("mode: {mode}\n"));
    if !model.is_empty() {
        out.push_str(&format!("model: {}\n", yaml_quote(model)));
    }
    if !model_tier.is_empty() {
        out.push_str(&format!("model_tier: {model_tier}\n"));
    }
    if !color.is_empty() {
        out.push_str(&format!("color: {}\n", yaml_quote(color)));
    }
    if let Some(t) = temperature {
        out.push_str(&format!("temperature: {t}\n"));
    }
    if let Some(s) = max_steps {
        out.push_str(&format!("max_steps: {s}\n"));
    }
    if !system_prompt.is_empty() {
        out.push_str(&format!("system_prompt: {}\n", yaml_quote(system_prompt)));
    }
    render_string_list_field(&mut out, "routing_tags", &routing_tags);
    render_string_list_field(&mut out, "tools", &tools);
    render_string_list_field(&mut out, "tool_groups", &tool_groups);
    render_string_list_field(&mut out, "mcp_servers", &mcp_servers);
    out.push_str("---\n\n");
    out.push_str(prompt);
    out.push('\n');
    Ok(out)
}

pub(crate) fn execute_save_agent(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().ok_or("Missing name")?.trim();
    if name.is_empty() {
        return Err("name is empty".to_string());
    }
    let content = build_agent_file_content(args)?;
    let dir = crate::ai::agents::agents_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create agents dir: {e}"))?;

    let file_name = args["file_name"]
        .as_str()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| safe_agent_file_name(name));
    let file_name = safe_agent_file_name(&file_name);

    // 禁止覆盖 builtin agent 文件名（即使写入用户目录也禁止，避免 user
    // override 把内置 prompt-skill / executor 等改坏）。
    if crate::ai::agents::builtin_agent_filenames()
        .iter()
        .any(|n| *n == file_name.as_str())
    {
        return Err(format!(
            "Refusing to overwrite builtin agent filename '{file_name}'. \
             Use a different name to author a new agent."
        ));
    }

    let path = dir.join(&file_name);
    let overwrite = args["overwrite"].as_bool().unwrap_or(true);
    if path.exists() && !overwrite {
        return Err(format!(
            "Agent file already exists and overwrite=false: {}",
            path.display()
        ));
    }

    fs::write(&path, content).map_err(|e| format!("Failed to write agent file: {e}"))?;
    Ok(format!(
        "Agent saved: {}\nAgent name: {}",
        path.display(),
        name
    ))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "save_agent",
        description: "Render and save a .agent file (YAML front matter + prompt body) into the configured user agents directory. Refuses to overwrite builtin agent filenames.",
        parameters: params_save_agent,
        execute: execute_save_agent,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

#[cfg(test)]
mod tests {
    use super::execute_discover_skills;
    #[test]
    fn discover_skills_returns_builtin_skill_metadata() {
        let args = serde_json::json!({
            "query": "debug",
            "limit": 10
        });
        let out = execute_discover_skills(&args).unwrap();
        assert!(out.contains("debugger"));
        assert!(out.contains("metadata only"));
        assert!(!out.contains("Skill enforcement"));
    }

    #[test]
    fn discover_skills_can_include_capabilities() {
        let args = serde_json::json!({
            "query": "review",
            "limit": 10,
            "include_capabilities": true
        });
        let out = execute_discover_skills(&args).unwrap();
        assert!(out.contains("code-review"));
        assert!(out.contains("tools=") || out.contains("tool_groups="));
    }
}
