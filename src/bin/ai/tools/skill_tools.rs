use std::fs;
use std::path::PathBuf;
use std::sync::{LazyLock, RwLock};

use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::ai::driver::ScoredSkill;
use crate::ai::skills::SkillManifest;
use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

/// 模型通过 `activate_skill` 工具显式请求激活的 skill 名称（待 driver 在下一个
/// iteration 读取并应用）。
///
/// 工具是纯函数 `fn(&Value) -> Result<String, String>`，拿不到 `App`，因此沿用
/// `enable_tools.rs` 的"工具写全局状态 → driver 读取"桥接模式。这里只需要一个
/// 极小的待激活槽位，故用单个 `RwLock<Option<String>>` 而非完整状态结构。
static PENDING_SKILL_ACTIVATION: LazyLock<RwLock<Option<String>>> =
    LazyLock::new(|| RwLock::new(None));

fn set_pending_skill_activation(name: String) {
    if let Ok(mut slot) = PENDING_SKILL_ACTIVATION.write() {
        *slot = Some(name);
    }
}

/// driver 侧调用：取出并清空待激活的 skill 名称。
pub(crate) fn take_pending_skill_activation() -> Option<String> {
    PENDING_SKILL_ACTIVATION
        .write()
        .ok()
        .and_then(|mut slot| slot.take())
}

fn params_discover_skills() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional relevance query (natural language, any language). Matched semantically against each skill's name, description, and capabilities — not a literal substring filter — so a Chinese query finds English-described skills and vice versa. Leave empty to list all skills."
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
    let haystack = skill_search_haystack(skill);
    let query = query.to_ascii_lowercase();
    if haystack.contains(&query) {
        return true;
    }
    query_tokens(&query)
        .into_iter()
        .any(|token| haystack.contains(token.as_str()))
}

fn skill_search_haystack(skill: &SkillManifest) -> String {
    // 检索仅基于 name/description/能力字段等真实语义来源。
    // source_path/resource_path 是实现细节和纯路径噪音，不应参与检索，
    // 也不应把本地文件系统路径泄露给模型。
    let mut parts = vec![skill.name.clone(), skill.description.clone()];
    parts.extend(skill.tools.iter().cloned());
    parts.extend(skill.tool_groups.iter().cloned());
    parts.extend(skill.mcp_servers.iter().cloned());
    parts.join("\n").to_ascii_lowercase()
}

fn skill_source_label(skill: &SkillManifest) -> &'static str {
    let Some(source) = skill.source_path.as_deref() else {
        return "unknown";
    };
    if source.starts_with("builtin:") {
        "builtin"
    } else if source.contains(".zip!") {
        "package"
    } else if source.contains("/.trae-cn/") {
        "extension"
    } else {
        "user"
    }
}

fn query_tokens(query: &str) -> Vec<String> {
    query
        .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.'))
        .map(str::trim)
        .filter(|token| token.chars().count() >= 2)
        .map(ToString::to_string)
        .collect()
}

fn summarize_skill(skill: &SkillManifest, include_capabilities: bool) -> String {
    let source = skill_source_label(skill);
    let mut line = format!(
        "- {} | priority={} | source={}",
        skill.name, skill.priority, source
    );
    if !skill.description.trim().is_empty() {
        line.push_str(&format!(" | {}", skill.description.trim()));
    }
    if include_capabilities {
        if let Some(resource_path) = skill.resource_path.as_deref()
            && !resource_path.trim().is_empty()
        {
            line.push_str(" | resources=bundled");
        }
        if !skill.tools.is_empty() {
            line.push_str(&format!(" | tools={}", skill.tools.join(",")));
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

/// discover_skills 相关性门槛。比自动路由的 `has_skill_signal`(0.08) 更严：那是
/// 配合 sticky/threshold 的"保留信号"，这里是独立过滤器，必须自己把噪音挡掉。
/// embedding 关闭的降级路径里，char-ngram TF-IDF 对无关 query 也能凑出 ~0.10 的
/// blended 噪音，故 blended 门槛取 0.12 以越过噪音天花板。
const DISCOVER_BLENDED_FLOOR: f64 = 0.12;
/// 预训练 skill_match 模型对该 skill 名的概率：跨语言、且 embedding 关闭时仍可用，
/// 是中文 query 命中英文 builtin skill 的主信号。无关 query 下各 label 概率被 none
/// 吸收（实测 < 0.09），取 0.15 作为置信下限。
const DISCOVER_MODEL_PRIOR_FLOOR: f64 = 0.15;

/// discover_skills 的相关性判定：复用自动路由的语义打分，再叠一条词法兜底。
/// - `lexical_hit`：原子串/token 命中，保证不弱于旧的纯子串行为；
/// - `blended_score`：embedding 与运行时 TF-IDF 的融合分（embedding 开启时为主）；
/// - `model_prior_score`：预训练模型给 builtin skill 的跨语言信号。
fn skill_is_discoverable(item: &ScoredSkill<'_>, lexical_hit: bool) -> bool {
    lexical_hit
        || item.blended_score >= DISCOVER_BLENDED_FLOOR
        || item.model_prior_score >= DISCOVER_MODEL_PRIOR_FLOOR
}

fn render_discovered_skills(
    skills: &[&SkillManifest],
    query: &str,
    include_capabilities: bool,
) -> String {
    let mut lines = Vec::with_capacity(skills.len() + 2);
    if query.is_empty() {
        lines.push(format!("{} skills available:", skills.len()));
    } else {
        lines.push(format!(
            "{} skills matched query '{}':",
            skills.len(),
            query
        ));
    }
    lines.extend(
        skills
            .iter()
            .copied()
            .map(|skill| summarize_skill(skill, include_capabilities)),
    );
    lines.push(
        "This tool returns skill metadata only. Skill prompts stay unloaded until routing selects a skill.\nIf you called this during an active task, do not stop here: continue the turn by selecting the best matching skill, enabling missing tools, or answering directly if no skill is actually needed."
            .to_string(),
    );
    lines.join("\n")
}

pub(crate) fn execute_discover_skills(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().unwrap_or("").trim();
    let limit = args["limit"].as_u64().unwrap_or(20).clamp(1, 100) as usize;
    let include_capabilities = args["include_capabilities"].as_bool().unwrap_or(false);

    let skills = crate::ai::skills::load_all_skills();

    // 空 query：保持原行为，按加载顺序列出全部（截到 limit）。
    if query.is_empty() {
        let listed = skills.iter().take(limit).collect::<Vec<_>>();
        if listed.is_empty() {
            return Ok("No skills are currently available.".to_string());
        }
        return Ok(render_discovered_skills(
            &listed,
            query,
            include_capabilities,
        ));
    }

    // 非空 query：复用自动路由的语义打分器（embedding + 运行时 TF-IDF + 预训练
    // skill_match 模型融合），解决纯 ASCII 子串匹配"中文 query 搜不到英文 skill"
    // 的跨语言失效问题。embedding 失败时 rank_skills_locally 内部已 unwrap_or_default
    // 降级为 TF-IDF/词法，无网络强依赖；结果按相关性降序，最相关的在前。
    // 词法命中（旧子串行为）作为兜底信号纳入，故召回是旧行为的严格超集。
    let ranked = crate::ai::driver::rank_skills_locally(&skills, query);
    let matched = ranked
        .iter()
        .filter(|item| skill_is_discoverable(item, skill_matches_query(item.skill, query)))
        .map(|item| item.skill)
        .take(limit)
        .collect::<Vec<_>>();

    if matched.is_empty() {
        return Ok(format!("No skills matched query '{}'.", query));
    }

    Ok(render_discovered_skills(
        &matched,
        query,
        include_capabilities,
    ))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "discover_skills",
        description: "Search available skills by relevance, returning metadata only (names, descriptions, priorities, optional capabilities) without loading full skill prompts. The query is matched semantically across languages, so prefer a natural-language `query` describing the task (e.g. the workflow, tool, product, or domain) over guessing exact names. Call this early whenever a task might map to a specialized or installed workflow; then use the returned metadata to decide whether a matching skill should be activated or whether the current tool set is enough.",
        parameters: params_discover_skills,
        execute: execute_discover_skills,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

fn params_activate_skill() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact name of the skill to activate."
            }
        },
        "required": ["name"]
    })
}

pub(crate) fn execute_activate_skill(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().unwrap_or("").trim();
    if name.is_empty() {
        return Err("activate_skill requires a non-empty 'name'.".to_string());
    }

    // 校验"别乱用"：请求的 skill 名必须真实存在。未命中则拒绝，并回列可用
    // skill 名，引导模型纠正而不是凭空激活。
    let skills = crate::ai::skills::load_all_skills();
    let matched = skills.iter().find(|s| s.name == name);
    let Some(skill) = matched else {
        let available = skills
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "No skill named '{name}'. Available skills: {available}"
        ));
    };

    set_pending_skill_activation(skill.name.clone());
    Ok(format!(
        "Skill '{}' will be activated on the next step: its prompt and tool set load into the current turn. \
         Only continue if this skill clearly matches the user's task; otherwise proceed without it.",
        skill.name
    ))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "activate_skill",
        description: "Activate a specific skill by name so its full prompt and tool set load into the current turn. \
                      Only use this when one specific skill clearly matches the user's task. \
                      Do not activate a skill speculatively or for tasks that need no skill.",
        parameters: params_activate_skill,
        execute: execute_activate_skill,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

fn params_load_skill() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact name of the skill to read."
            }
        },
        "required": ["name"]
    })
}

/// 渲染 load_skill 的返回：头部元信息 + skill 正文（+ 可选 bundled 资源目录）。
fn render_loaded_skill(skill: &SkillManifest) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Skill: {}\n", skill.name));
    if !skill.description.trim().is_empty() {
        out.push_str(&format!("description: {}\n", skill.description.trim()));
    }
    out.push_str(&format!("version: {}\n", skill.version));
    if let Some(system_prompt) = skill.system_prompt.as_deref()
        && !system_prompt.trim().is_empty()
    {
        out.push_str("\n## system_prompt\n");
        out.push_str(system_prompt.trim());
        out.push('\n');
    }
    out.push_str("\n## prompt\n");
    if skill.prompt.trim().is_empty() {
        out.push_str("(this skill has an empty prompt body)\n");
    } else {
        out.push_str(&skill.prompt);
        if !skill.prompt.ends_with('\n') {
            out.push('\n');
        }
    }
    // 只有当 skill 真带 bundled 资源时才暴露其目录。这是显式 load 单个 skill 的
    // 合法用途（agent 需要读 references/*），与 discover_skills 刻意隐藏所有路径
    // 的防泄漏语义不冲突。
    if let Some(resource_path) = skill.resource_path.as_deref()
        && !resource_path.trim().is_empty()
    {
        out.push_str(&format!(
            "\n## resources\nBundled resource directory: {resource_path}\n"
        ));
    }
    out
}

pub(crate) fn execute_load_skill(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().unwrap_or("").trim();
    if name.is_empty() {
        return Err("load_skill requires a non-empty 'name'.".to_string());
    }

    let skills = crate::ai::skills::load_all_skills();
    let Some(skill) = skills.iter().find(|s| s.name == name) else {
        let available = skills
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "No skill named '{name}'. Available skills: {available}"
        ));
    };

    Ok(render_loaded_skill(skill))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "load_skill",
        description: "Read a skill's full contents (its prompt body, system prompt, and bundled resource directory if any) by name, without changing the current turn. \
                      Use this when you need to inspect, learn from, or modify an existing skill (e.g. authoring or debugging skills) — it only returns text, it does NOT activate the skill or alter your tool set. \
                      Reading a skill does not change the current turn or tool set.",
        parameters: params_load_skill,
        execute: execute_load_skill,
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
    use super::{
        build_skill_file_content, execute_activate_skill, execute_discover_skills,
        execute_load_skill, query_tokens, render_loaded_skill, skill_matches_query,
        summarize_skill, take_pending_skill_activation,
    };
    use crate::ai::skills::SkillManifest;
    use std::sync::{LazyLock, Mutex};

    // activate_skill 系列测试共享同一个全局待激活槽位，串行化避免并发污染。
    static ACTIVATION_TEST_GUARD: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
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

    #[test]
    fn discover_query_extracts_meaningful_tokens_from_sentence() {
        assert!(query_tokens("帮我查一个 argos 日志").contains(&"argos".to_string()));
    }

    #[test]
    fn discover_skills_matches_chinese_query_against_english_skill() {
        // 中文 query 搜英文 builtin skill：纯子串匹配会失效，语义打分应能召回。
        let args = serde_json::json!({
            "query": "帮我审查一下这段代码",
            "limit": 20
        });
        let out = execute_discover_skills(&args).unwrap();
        assert!(
            out.contains("code-review"),
            "expected cross-lingual semantic match to surface code-review, got:\n{out}"
        );
    }

    #[test]
    fn discover_skills_reports_no_match_for_irrelevant_query() {
        // 完全无关的 query 不应倾倒全部 skill 清单。
        let args = serde_json::json!({
            "query": "zzz_totally_unrelated_token_qwxyz",
            "limit": 20
        });
        let out = execute_discover_skills(&args).unwrap();
        assert!(
            out.contains("No skills matched"),
            "expected no-match message for irrelevant query, got:\n{out}"
        );
    }

    #[test]
    fn skill_query_does_not_match_source_path_noise() {
        let mut skill = test_skill("feishu-upload", "Upload markdown into Feishu docs");
        skill.source_path = Some("/tmp/argos.skill".to_string());
        assert!(!skill_matches_query(&skill, "帮我查一个 argos 日志"));
    }

    #[test]
    fn summarize_skill_hides_local_source_and_resource_paths() {
        let mut skill = test_skill("feishu-upload", "Upload markdown into Feishu docs");
        skill.source_path =
            Some("/Users/bytedance/.config/rust_tools/skills/feishu-upload-md.skill".to_string());
        skill.resource_path = Some("/tmp/feishu-upload/resources".to_string());

        let out = summarize_skill(&skill, true);

        assert!(out.contains("source=user"));
        assert!(out.contains("resources=bundled"));
        assert!(!out.contains(".config/rust_tools/skills"));
        assert!(!out.contains("/tmp/feishu-upload/resources"));
    }

    #[test]
    fn activate_skill_rejects_empty_name() {
        let _g = ACTIVATION_TEST_GUARD.lock().unwrap();
        let err = execute_activate_skill(&serde_json::json!({"name": "  "})).unwrap_err();
        assert!(err.contains("non-empty"));
        assert!(take_pending_skill_activation().is_none());
    }

    #[test]
    fn activate_skill_rejects_unknown_name() {
        let _g = ACTIVATION_TEST_GUARD.lock().unwrap();
        let err = execute_activate_skill(&serde_json::json!({"name": "definitely-not-a-skill"}))
            .unwrap_err();
        assert!(err.contains("No skill named"));
        // 未命中不应写入待激活槽位，避免乱激活。
        assert!(take_pending_skill_activation().is_none());
    }

    #[test]
    fn activate_skill_queues_existing_skill() {
        let _g = ACTIVATION_TEST_GUARD.lock().unwrap();
        // 取一个真实存在的 builtin skill 名字。
        let skills = crate::ai::skills::load_all_skills();
        let Some(name) = skills.first().map(|s| s.name.clone()) else {
            return;
        };
        let out = execute_activate_skill(&serde_json::json!({"name": name})).unwrap();
        assert!(out.contains(&name));
        assert_eq!(
            take_pending_skill_activation().as_deref(),
            Some(name.as_str())
        );
        // take 应清空槽位。
        assert!(take_pending_skill_activation().is_none());
    }

    #[test]
    fn save_skill_ignores_legacy_triggers_argument() {
        let out = build_skill_file_content(&serde_json::json!({
            "name": "demo-skill",
            "description": "demo",
            "prompt": "body",
            "triggers": ["legacy", "exact-match"],
            "tools": ["read_file"]
        }))
        .unwrap();
        assert!(out.contains("name: \"demo-skill\""));
        assert!(out.contains("tools:\n  - \"read_file\""));
        assert!(!out.contains("triggers:"));
    }

    #[test]
    fn load_skill_rejects_empty_name() {
        let err = execute_load_skill(&serde_json::json!({"name": "  "})).unwrap_err();
        assert!(err.contains("non-empty"));
    }

    #[test]
    fn load_skill_rejects_unknown_name() {
        let err =
            execute_load_skill(&serde_json::json!({"name": "definitely-not-a-skill"})).unwrap_err();
        assert!(err.contains("No skill named"));
        assert!(err.contains("discover_skills"));
    }

    #[test]
    fn load_skill_returns_prompt_body_for_builtin() {
        // 取一个真实存在、prompt 非空的 skill（builtin debugger 一定在）。
        let out = execute_load_skill(&serde_json::json!({"name": "debugger"})).unwrap();
        assert!(out.contains("# Skill: debugger"));
        assert!(out.contains("## prompt"));
    }

    #[test]
    fn render_loaded_skill_includes_body_and_resources() {
        let mut skill = test_skill("demo", "demo description");
        skill.prompt = "line one\nline two".to_string();
        skill.resource_path = Some("/tmp/demo/resources".to_string());
        let out = render_loaded_skill(&skill);
        assert!(out.contains("# Skill: demo"));
        assert!(out.contains("description: demo description"));
        assert!(out.contains("## prompt"));
        assert!(out.contains("line one\nline two"));
        // 有 bundled 资源时才暴露目录
        assert!(out.contains("Bundled resource directory: /tmp/demo/resources"));
    }

    #[test]
    fn render_loaded_skill_omits_resources_when_absent() {
        let mut skill = test_skill("demo", "demo description");
        skill.prompt = "body".to_string();
        let out = render_loaded_skill(&skill);
        assert!(!out.contains("Bundled resource directory"));
        assert!(!out.contains("## resources"));
    }

    fn test_skill(name: &str, description: &str) -> SkillManifest {
        SkillManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: description.to_string(),
            author: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            skip_recall: false,
            disable_builtin_tools: false,
            disable_mcp_tools: false,
            prompt: String::new(),
            system_prompt: None,
            priority: 0,
            excludes: Vec::new(),
            source_path: None,
            resource_path: None,
        }
    }
}
