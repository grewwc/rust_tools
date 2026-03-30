use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

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
                "description": "Tool groups that the skill is allowed to use (e.g. builtin, openclaw)."
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
        groups: &["builtin"],
    }
});

fn resolve_configured_skills_dir() -> PathBuf {
    let cfg = crate::common::configw::get_all_config();
    let raw = cfg.get_opt("ai.skills.dir").unwrap_or_default();
    if raw.trim().is_empty() {
        return crate::ai::skills::skills_dir();
    }
    PathBuf::from(crate::common::utils::expanduser(&raw).as_ref())
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
