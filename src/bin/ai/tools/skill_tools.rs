use std::fs;
use std::path::PathBuf;
use std::sync::{LazyLock, RwLock};

use rust_tools::commonw::{FastMap, FastSet};
use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::ai::skills::SkillManifest;
use crate::ai::tools::common::{ToolRegistration, ToolSpec};

/// 模型通过 `activate_skill` 工具显式请求激活的 skill 名称（待 driver 在下一个
/// iteration 读取并应用）。
///
/// 工具是纯函数 `fn(&Value) -> Result<String, String>`，拿不到 `App`，因此沿用
/// `enable_tools.rs` 的"工具写全局状态 → driver 读取"桥接模式。这里只需要一个
/// 极小的待激活槽位，故用单个 `RwLock<Option<String>>` 而非完整状态结构。
pub(crate) static PENDING_SKILL_ACTIVATION: LazyLock<RwLock<FastMap<(String, usize), String>>> =
    LazyLock::new(|| RwLock::new(FastMap::default()));

/// `request_user_input` 在本轮内记录的明确交互边界。按 `(session_id, turn_id)` 隔离，
/// 避免并行 subagent 或其他 session 的请求污染当前 foreground turn。
static PENDING_USER_INPUT_REQUESTS: LazyLock<RwLock<FastSet<(String, usize)>>> =
    LazyLock::new(|| RwLock::new(FastSet::default()));

fn current_turn_identity() -> (String, usize) {
    crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .try_with(Clone::clone)
        .unwrap_or_default()
}

fn set_pending_skill_activation(name: String) {
    if let Ok(mut slot) = PENDING_SKILL_ACTIVATION.write() {
        slot.insert(current_turn_identity(), name);
    }
}

/// driver 侧调用：取出并清空待激活的 skill 名称。
pub(crate) fn take_pending_skill_activation() -> Option<String> {
    PENDING_SKILL_ACTIVATION
        .write()
        .ok()
        .and_then(|mut slot| slot.remove(&current_turn_identity()))
}

pub(crate) fn clear_pending_user_input_request() {
    if let Ok(mut requests) = PENDING_USER_INPUT_REQUESTS.write() {
        requests.remove(&current_turn_identity());
    }
}

/// driver 侧调用：查询并清空本轮的显式用户输入请求。
pub(crate) fn take_pending_user_input_request() -> bool {
    PENDING_USER_INPUT_REQUESTS
        .write()
        .map(|mut requests| requests.remove(&current_turn_identity()))
        .unwrap_or(false)
}

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
         It is scoped to this user turn and unloads automatically when the turn ends.",
        skill.name
    ))
}

fn params_request_user_input() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The concise information, decision, or confirmation needed from the user."
            }
        },
        "required": ["question"]
    })
}

/// 标记当前 skill 正在等待用户输入。工具结果仅供模型接续生成面向用户的问题；
/// 真正的跨轮状态由 driver 在本 turn 结束时保存，避免工具层直接修改会话状态。
pub(crate) fn execute_request_user_input(args: &Value) -> Result<String, String> {
    let question = args["question"].as_str().unwrap_or("").trim();
    if question.is_empty() {
        return Err("request_user_input requires a non-empty 'question'.".to_string());
    }

    if let Ok(mut requests) = PENDING_USER_INPUT_REQUESTS.write() {
        requests.insert(current_turn_identity());
    }

    Ok(format!(
        "User input has been requested: {question}\n\
         Ask the user this question in your final response, then stop. The active skill will be restored only for the user's immediately following normal message."
    ))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "activate_skill",
        description: "Activate a specific skill by name so its full prompt and tool set load into the current turn. \
                      If the appropriate skill is not known, proactively use `list_skills` after identifying a concrete specialized need or when the user asks about available skills; do not infer such a need from technical keywords or a routine source-code, repository, file, or terminal investigation alone. \
                      Activate only when one listed skill clearly and materially improves the user's task; do not activate for generic work, loose keyword overlap, or just in case. \
                      Activation is scoped to the current user turn and unloads automatically at turn end.",
        parameters: params_activate_skill,
        execute: execute_activate_skill,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        // skill 发现/激活是低频能力：默认不随每轮 core 展开常驻，模型按需经
        // `enable_tools` 启用，压缩每轮 tools schema token。仍保留 builtin 组，
        // 保证可被动态启用。
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "request_user_input",
        description: "When an active skill needs information, a decision, or confirmation from the user before it can continue, use this tool instead of merely ending with a question. The driver restores that skill only for the user's immediately following normal message.",
        parameters: params_request_user_input,
        execute: execute_request_user_input,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        // 这是 driver 直接按名称注入的控制工具，不能通过 manifest 的 tool_groups 暴露。
        groups: &[],
    }
});

const DEFAULT_SKILL_LIST_LIMIT: usize = 50;
const MAX_SKILL_LIST_LIMIT: usize = 100;

fn params_list_skills() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional concise keyword or phrase to filter skill names and descriptions. Omit to browse the installed catalog."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matching skills to return (default 50, maximum 100)."
            }
        }
    })
}

fn skill_list_limit(args: &Value) -> usize {
    args["limit"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(DEFAULT_SKILL_LIST_LIMIT)
        .clamp(1, MAX_SKILL_LIST_LIMIT)
}

fn render_skill_catalog(skills: &[SkillManifest], query: &str, limit: usize) -> String {
    let query = query.trim().to_lowercase();
    let mut matches = skills
        .iter()
        .filter(|skill| {
            query.is_empty()
                || skill.name.to_lowercase().contains(&query)
                || skill.description.to_lowercase().contains(&query)
        })
        .collect::<Vec<_>>();
    // catalog 是发现入口而非排序候选，固定按名字列出以免把 priority 误解为推荐分数。
    matches.sort_by(|a, b| a.name.cmp(&b.name));

    if matches.is_empty() {
        return if query.is_empty() {
            "No installed skills are available. Continue with the current tools.".to_string()
        } else {
            format!(
                "No installed skills matched '{query}'. Refine the query or continue with the current tools."
            )
        };
    }

    let total = matches.len();
    let shown = total.min(limit);
    let mut out = format!("Installed skills ({shown} shown of {total}):\n");
    for skill in matches.into_iter().take(shown) {
        let description = skill
            .description
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if description.is_empty() {
            out.push_str(&format!("- `{}`\n", skill.name));
        } else {
            out.push_str(&format!("- `{}` — {description}\n", skill.name));
        }
    }
    if total > shown {
        out.push_str(
            "Results are sorted by name and truncated; refine `query` to browse further.\n",
        );
    }
    out.push_str(
        "This catalog is metadata only and does not activate a skill. Call `activate_skill(name=...)` only when one listed skill clearly and materially helps the current task.",
    );
    out
}

pub(crate) fn execute_list_skills(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().unwrap_or("");
    let skills = crate::ai::skills::load_all_skills();
    Ok(render_skill_catalog(&skills, query, skill_list_limit(args)))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "list_skills",
        description: "Browse installed skill names and descriptions without activating anything. \
                      Use this proactively when the user asks about available skills, or after identifying a concrete, genuinely specialized need for domain context, an established workflow, bundled resources, or dedicated tools and you need to identify the right skill. \
                      Do not browse merely because a task contains technical keywords or involves a routine source-code, repository, file, or terminal investigation.",
        parameters: params_list_skills,
        execute: execute_list_skills,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin"],
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
    // 合法用途（agent 需要读 references/*）；默认运行时不会因为列举 skill 元信息
    // 而泄露本地路径。
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
        groups: &["builtin"],
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
        groups: &["builtin"],
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

#[cfg(test)]
mod tests {
    use super::{
        build_skill_file_content, clear_pending_user_input_request, execute_activate_skill,
        execute_load_skill, execute_request_user_input, render_loaded_skill, render_skill_catalog,
        take_pending_skill_activation, take_pending_user_input_request,
    };
    use crate::ai::driver::runtime_ctx::TURN_IDENTITY;
    use crate::ai::skills::SkillManifest;
    use std::sync::{LazyLock, Mutex};

    // activate_skill 系列测试共享同一个全局待激活槽位，串行化避免并发污染。
    static ACTIVATION_TEST_GUARD: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

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
    fn request_user_input_is_turn_scoped_and_one_shot() {
        let _g = ACTIVATION_TEST_GUARD.lock().unwrap();
        let identity_a = ("session-a".to_string(), 11);
        let identity_b = ("session-b".to_string(), 12);

        TURN_IDENTITY.sync_scope(identity_a.clone(), || {
            clear_pending_user_input_request();
            let out = execute_request_user_input(&serde_json::json!({
                "question": "Which region should I query?"
            }))
            .unwrap();
            assert!(out.contains("Which region should I query?"));
        });

        TURN_IDENTITY.sync_scope(identity_b, || {
            clear_pending_user_input_request();
            assert!(!take_pending_user_input_request());
        });

        TURN_IDENTITY.sync_scope(identity_a, || {
            assert!(take_pending_user_input_request());
            assert!(!take_pending_user_input_request());
        });
    }

    #[test]
    fn skill_discovery_descriptions_preserve_proactive_boundary() {
        let list_skills = crate::ai::tools::registry::common::get_tool_spec("list_skills")
            .expect("list_skills should be registered");
        assert!(list_skills.description.contains("Use this proactively"));
        assert!(list_skills.description.contains("technical keywords"));
        assert!(
            list_skills
                .description
                .contains("routine source-code, repository, file, or terminal investigation")
        );

        let activate_skill = crate::ai::tools::registry::common::get_tool_spec("activate_skill")
            .expect("activate_skill should be registered");
        assert!(
            activate_skill
                .description
                .contains("proactively use `list_skills`")
        );
        assert!(activate_skill.description.contains("technical keywords"));
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
    }

    #[test]
    fn skill_catalog_is_alphabetical_and_exposes_only_metadata() {
        let mut excel = test_skill("excel-analysis", "Analyze local workbooks");
        excel.resource_path = Some("/private/excel/resources".to_string());
        let general = test_skill("general-review", "Review a document");
        let catalog = render_skill_catalog(&[general, excel], "", 50);

        let excel_pos = catalog.find("`excel-analysis`").unwrap();
        let general_pos = catalog.find("`general-review`").unwrap();
        assert!(excel_pos < general_pos);
        assert!(catalog.contains("Analyze local workbooks"));
        assert!(!catalog.contains("/private/excel/resources"));
        assert!(!catalog.contains("## prompt"));
    }

    #[test]
    fn skill_catalog_filters_by_name_or_description() {
        let excel = test_skill("excel-analysis", "Analyze local workbooks");
        let general = test_skill("general-review", "Review a document");
        let catalog = render_skill_catalog(&[general, excel], "workbook", 50);

        assert!(catalog.contains("`excel-analysis`"));
        assert!(!catalog.contains("`general-review`"));
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
