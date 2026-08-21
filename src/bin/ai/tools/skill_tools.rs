use std::fs;
use std::path::PathBuf;
use std::sync::{LazyLock, RwLock};

use rust_tools::commonw::{FastMap, FastSet};
use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::ai::skills::SkillManifest;
use crate::ai::tools::common::{ToolRegistration, ToolSpec};

/// 模型通过 `activate_skill` / `deactivate_skill` 工具请求的 skill 变更动作（待
/// driver 在下一个 iteration 读取并应用）。
///
/// 工具是纯函数 `fn(&Value) -> Result<String, String>`，拿不到 `App`，因此沿用
/// `enable_tools.rs` 的"工具写全局状态 → driver 读取"桥接模式。这里只需要一个
/// 极小的待激活槽位，故用单个 `RwLock<FastMap>` 而非完整状态结构。支持多 skill
/// 同时激活：`Add` 追加到活动集，`Remove` 从活动集移除。同一 turn 内多次调用会
/// 按调用顺序累积为队列，driver 一次性全部应用（不做"后写覆盖"）。

/// 模型请求的 skill 变更动作。driver 在下一个 iteration 读取并应用到当前活动集。
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PendingSkillAction {
    /// 追加一个 skill 到当前活动集（已存在则忽略）
    Add(String),
    /// 从当前活动集移除一个 skill
    Remove(String),
}

pub(crate) static PENDING_SKILL_ACTIVATION: LazyLock<RwLock<FastMap<(String, usize), Vec<PendingSkillAction>>>> =
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

fn set_pending_skill_action(action: PendingSkillAction) {
    if let Ok(mut slot) = PENDING_SKILL_ACTIVATION.write() {
        slot.entry(current_turn_identity()).or_default().push(action);
    }
}

/// driver 侧调用：取出并清空本 turn 的全部待处理 skill 变更动作。
pub(crate) fn take_pending_skill_action() -> Vec<PendingSkillAction> {
    PENDING_SKILL_ACTIVATION
        .write()
        .ok()
        .and_then(|mut slot| slot.remove(&current_turn_identity()))
        .unwrap_or_default()
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

    set_pending_skill_action(PendingSkillAction::Add(skill.name.clone()));
    Ok(format!(
        "Skill '{}' added to the active skill set for this turn. Its prompt and tools merge with any other active skills. \
         If another skill is already active, both are active simultaneously (skills compose additively as equal peers; none is primary). \
         The skill set is scoped to this user turn and unloads automatically when the turn ends. \
         Use `deactivate_skill` to remove a skill from the active set.",
        skill.name
    ))
}

pub(crate) fn execute_deactivate_skill(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().unwrap_or("").trim();
    if name.is_empty() {
        return Err("deactivate_skill requires a non-empty 'name'.".to_string());
    }

    set_pending_skill_action(PendingSkillAction::Remove(name.to_string()));
    Ok(format!(
        "Skill '{}' will be removed from the active skill set on the next step. \
         If it was the only active skill, no skill will be active afterwards.",
        name
    ))
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
        description: "",

        execute: execute_activate_skill,
        // skill 发现/激活是低频能力：默认不随每轮 core 展开常驻，模型按需经
        // `enable_tools` 启用，压缩每轮 tools schema token。仍保留 builtin 组，
        // 保证可被动态启用。
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "deactivate_skill",
        description: "",

        execute: execute_deactivate_skill,
        // 与 activate_skill 同属低频控制工具：默认不常驻，按需经 enable_tools 启用。
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "request_user_input",
        description: "",

        execute: execute_request_user_input,
        // 这是 driver 直接按名称注入的控制工具，不能通过 manifest 的 tool_groups 暴露。
        groups: &[],
    }
});

const DEFAULT_SKILL_LIST_LIMIT: usize = 50;
const MAX_SKILL_LIST_LIMIT: usize = 100;

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
    // 统计父 skill -> 子 skill 数量，用于给父 skill 加标注
    let mut parent_has_children: FastSet<String> = FastSet::default();
    for s in skills.iter() {
        if let Some(p) = s.parent.as_deref() {
            parent_has_children.insert(p.to_string());
        }
    }
    for skill in matches.into_iter().take(shown) {
        let description = skill
            .description
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let is_parent = parent_has_children.contains(skill.name.as_str());
        let name_display = if let Some(parent) = skill.parent.as_deref() {
            format!("{} (sub-skill of {})", skill.name, parent)
        } else if is_parent {
            format!("{} [has sub-skills]", skill.name)
        } else {
            skill.name.clone()
        };
        if description.is_empty() {
            out.push_str(&format!("- `{}`\n", name_display));
        } else {
            out.push_str(&format!("- `{}` — {description}\n", name_display));
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
    if parent_has_children.is_empty() {
        // 无子 skill 时不额外提示
    } else {
        out.push_str(" Sub-skills can be activated independently via `activate_skill`; parent skills list their sub-skills in `load_skill`.");
    }
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
        description: "",

        execute: execute_list_skills,
        groups: &["builtin"],
    }
});

/// 采集 `resource_path` 下的文件列表（相对路径），用于 `load_skill` / 资源工具的展示。
/// `subdir` 需为安全的相对路径（仅允许单级 category 或空），内部会校验不穿越且 canonical 后仍在 base 内。
fn collect_resource_files(resource_path: &str, subdir: Option<&str>) -> Vec<String> {
    use std::path::Path;
    let base = Path::new(resource_path);
    // 先校验 base 存在且可 canonicalize（防 symlink 攻击前置检查）
    let Ok(canonical_base) = std::fs::canonicalize(base) else {
        // base 不存在或不可解析时返回空（调用方已处理 resource_path 不存在的情况）
        if !base.is_dir() {
            return Vec::new();
        }
        // 退化：base 存在但 canonicalize 失败（如权限），仍尝试直接列
        let mut files = Vec::new();
        collect_files_recursive(base, base, &mut files, 120);
        rust_tools::sortw::stable_sort_by(&mut files, |a, b| a.cmp(b));
        return files;
    };
    let target_base = match subdir {
        Some(s) if !s.trim().is_empty() => {
            let cleaned = s.trim().trim_matches('/').replace('\\', "/");
            // 严格校验：不允许 `..`、绝对路径、含 `/` 的多级穿越
            if cleaned.is_empty() || cleaned.contains("..") || Path::new(&cleaned).is_absolute() {
                return Vec::new();
            }
            // category 仅允许单段且为常见资源目录
            if cleaned.contains('/') {
                return Vec::new();
            }
            // 白名单校验（与文档一致）
            const ALLOWED_CATEGORIES: &[&str] = &["references", "examples", "scripts"];
            if !ALLOWED_CATEGORIES.contains(&cleaned.as_str()) {
                return Vec::new();
            }
            let joined = base.join(&cleaned);
            // canonical 校验：若目标存在则必须仍在 base 内；不存在则视为合法空目录
            match std::fs::canonicalize(&joined) {
                Ok(canonical_joined) if !canonical_joined.starts_with(&canonical_base) => {
                    return Vec::new();
                }
                Ok(canonical_joined) => canonical_joined,
                Err(_) => {
                    // 目标不存在时检查 joined 的 parent 是否仍在 base 内（防 `references/../etc` 类构造）
                    // 已通过 `..` 检查，此处认为不存在即空
                    return Vec::new();
                }
            }
        }
        _ => canonical_base.clone(),
    };
    if !target_base.is_dir() {
        return Vec::new();
    }
    let mut files = Vec::new();
    collect_files_recursive(&target_base, &target_base, &mut files, 120);
    rust_tools::sortw::stable_sort_by(&mut files, |a, b| a.cmp(b));
    files
}

fn collect_files_recursive(root: &std::path::Path, cur: &std::path::Path, out: &mut Vec<String>, cap: usize) {
    if out.len() >= cap {
        return;
    }
    let Ok(rd) = std::fs::read_dir(cur) else { return; };
    let mut entries = rd.flatten().collect::<Vec<_>>();
    entries.sort_by(|a, b| a.path().cmp(&b.path()));
    for entry in entries {
        if out.len() >= cap { break; }
        let path = entry.path();
        let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if file_name.starts_with('.') { continue; }
        if path.is_dir() {
            collect_files_recursive(root, &path, out, cap);
        } else if path.is_file() {
            if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.display().to_string());
            }
        }
    }
}

/// 渲染 load_skill 的返回：头部元信息 + skill 正文（+ 可选 bundled 资源目录）。
fn render_loaded_skill_with_all(skill: &SkillManifest, all: &[SkillManifest]) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Skill: {}\n", skill.name));
    if !skill.description.trim().is_empty() {
        out.push_str(&format!("description: {}\n", skill.description.trim()));
    }
    out.push_str(&format!("version: {}\n", skill.version));
    if let Some(parent) = skill.parent.as_deref()
        && !parent.trim().is_empty()
    {
        out.push_str(&format!("parent: {parent}\nsub-skill of: {parent}\n"));
    }
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
    // 只有当 skill 真带 bundled 资源时才暴露其目录，并列出 references/examples/scripts 等文件。
    if let Some(resource_path) = skill.resource_path.as_deref()
        && !resource_path.trim().is_empty()
    {
        out.push_str(&format!(
            "\n## resources\nBundled resource directory: {resource_path}\n"
        ));
        let list = collect_resource_files(resource_path, None);
        if list.is_empty() {
            out.push_str("(no bundled files found under references/examples/scripts)\n");
        } else {
            out.push_str("Bundled files (relative to resource dir):\n");
            for rel in list.iter().take(80) {
                out.push_str(&format!("- {rel}\n"));
            }
            if list.len() > 80 {
                out.push_str(&format!("... and {} more files\n", list.len() - 80));
            }
            out.push_str("Use `read_skill_resource(name, path)` or `read_file` with the absolute resource directory to read any file.\n");
        }
        // 列出子 skill（由调用方传入的 all，避免全量 IO 重复）
    }
    let children: Vec<&SkillManifest> = all.iter().filter(|s| s.parent.as_deref() == Some(skill.name.as_str())).collect();
    if !children.is_empty() {
        out.push_str("\n## sub-skills\nThis skill contains sub-skills:\n");
        for ch in children {
            let desc = ch.description.split_whitespace().collect::<Vec<_>>().join(" ");
            if desc.is_empty() {
                out.push_str(&format!("- `{}`\n", ch.name));
            } else {
                out.push_str(&format!("- `{}` — {desc}\n", ch.name));
            }
        }
        if skill.resource_path.is_some() {
            out.push_str("Activate any sub-skill via `activate_skill(name=...)` or load its details via `load_skill`.\n");
        } else {
            out.push_str("Activate any sub-skill via `activate_skill(name=...)`.\n");
        }
    }
    out
}

fn render_loaded_skill(skill: &SkillManifest) -> String {
    let all = crate::ai::skills::load_all_skills();
    render_loaded_skill_with_all(skill, &all)
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

    Ok(render_loaded_skill_with_all(skill, &skills))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "load_skill",
        description: "",

        execute: execute_load_skill,
        groups: &["builtin"],
    }
});

// ===== 新增：读取 skill 资源的工具 =====

fn resolve_skill_for_resource(name: &str) -> Result<SkillManifest, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("skill name is empty".to_string());
    }
    let skills = crate::ai::skills::load_all_skills();
    skills
        .into_iter()
        .find(|s| s.name == name)
        .ok_or_else(|| format!("No skill named '{name}'"))
}

fn validate_resource_relative_path(path: &str) -> Result<String, String> {
    let p = path.trim().trim_matches('/').replace('\\', "/");
    if p.is_empty() {
        return Err("resource path is empty".to_string());
    }
    if p.contains("..") {
        return Err("resource path must not contain '..'".to_string());
    }
    if std::path::Path::new(&p).is_absolute() {
        return Err("resource path must be relative".to_string());
    }
    Ok(p)
}

pub(crate) fn execute_list_skill_resources(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().unwrap_or("").trim();
    let category = args["category"].as_str().unwrap_or("").trim();
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(80)
        .clamp(1, 200);
    // 严格校验 category：仅允许空或白名单，且不含路径穿越字符
    if !category.is_empty() {
        const ALLOWED: &[&str] = &["references", "examples", "scripts"];
        if category.contains("..") || category.contains('/') || category.contains('\\') {
            return Err(format!("Invalid category '{category}': must be one of references/examples/scripts and must not contain path separators"));
        }
        if !ALLOWED.contains(&category) {
            return Err(format!("Invalid category '{category}': allowed values are references, examples, scripts"));
        }
    }
    let skill = resolve_skill_for_resource(name)?;
    let resource_path = skill
        .resource_path
        .as_deref()
        .ok_or_else(|| format!("Skill '{name}' has no bundled resource directory (single-file skill)"))?;
    let subdir = if category.is_empty() { None } else { Some(category) };
    let files = collect_resource_files(resource_path, subdir);
    if files.is_empty() {
        let dir_label = if let Some(c) = subdir {
            format!("{resource_path}/{c}")
        } else {
            resource_path.to_string()
        };
        return Ok(format!("No files found under {dir_label}"));
    }
    let total = files.len();
    let shown = total.min(limit);
    let mut out = format!(
        "Resources for skill '{}' under '{}' ({} shown of {}):\n",
        skill.name,
        subdir.unwrap_or("."),
        shown,
        total
    );
    for rel in files.iter().take(shown) {
        // 尝试推断大小
        let abs = std::path::Path::new(resource_path).join(rel);
        let size = std::fs::metadata(&abs).map(|m| m.len()).unwrap_or(0);
        out.push_str(&format!("- {rel} ({size} bytes)\n"));
    }
    if total > shown {
        out.push_str(&format!("... and {} more files (use limit to see more)\n", total - shown));
    }
    out.push_str("Use `read_skill_resource(name, path)` to read a file.\n");
    Ok(out)
}

pub(crate) fn execute_read_skill_resource(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().unwrap_or("").trim();
    let path = args["path"].as_str().unwrap_or("").trim();
    if name.is_empty() || path.is_empty() {
        return Err("read_skill_resource requires non-empty 'name' and 'path'".to_string());
    }
    let skill = resolve_skill_for_resource(name)?;
    let resource_path = skill
        .resource_path
        .as_deref()
        .ok_or_else(|| format!("Skill '{name}' has no bundled resource directory"))?;
    let rel = validate_resource_relative_path(path)?;
    let abs = std::path::Path::new(resource_path).join(&rel);
    // 规范化后确保仍在 resource_path 内（防路径穿越）
    let canonical_base = std::fs::canonicalize(resource_path)
        .unwrap_or_else(|_| std::path::PathBuf::from(resource_path));
    let canonical_target = std::fs::canonicalize(&abs)
        .map_err(|e| format!("Failed to resolve resource path '{rel}': {e}"))?;
    if !canonical_target.starts_with(&canonical_base) {
        return Err("resource path escapes skill resource directory".to_string());
    }
    let content = std::fs::read_to_string(&canonical_target)
        .map_err(|e| format!("Failed to read resource '{rel}': {e}"))?;
    const MAX_READ: usize = 64 * 1024;
    if content.len() > MAX_READ {
        let truncated = &content[..MAX_READ];
        Ok(format!(
            "{truncated}\n\n... truncated: {} bytes total, showing first {} bytes. Use read_file with offset/limit for paging.",
            content.len(),
            MAX_READ
        ))
    } else {
        Ok(content)
    }
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "list_skill_resources",
        description: "",
        execute: execute_list_skill_resources,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_skill_resource",
        description: "",
        execute: execute_read_skill_resource,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "save_skill",
        description: "",

        execute: execute_save_skill,
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

/// 共享的 skill 名净化核心：小写、保留 alnum/`-`/`_`/` .`，其余转 `-`，合并 `--`，trim `-`/` .`。
fn sanitize_skill_basename(name: &str) -> String {
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
    let out = out.trim_matches(|c| c == '-' || c == '.').to_string();
    if out.is_empty() {
        "skill".to_string()
    } else {
        out
    }
}

fn safe_skill_file_name(name: &str) -> String {
    let base = sanitize_skill_basename(name);
    // 使用 strip_suffix 精确判断，避免 trim_end_matches 誤刪字符集
    if base.to_ascii_lowercase().ends_with(".skill") {
        base
    } else {
        format!("{base}.skill")
    }
}

fn safe_skill_dir_name(name: &str) -> String {
    let mut base = sanitize_skill_basename(name);
    // 去除 .skill 後綴（若有），保持文件與目錄 basename 一致：a.b.skill ↔ a.b
    if base.to_ascii_lowercase().ends_with(".skill") {
        if let Some(stripped) = base.get(..base.len() - ".skill".len()) {
            base = stripped.trim_end_matches('-').trim_end_matches('.').to_string();
            if base.is_empty() {
                base = "skill".to_string();
            }
        }
    }
    base
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
    let parent = args["parent"].as_str().unwrap_or("").trim();
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
    if !parent.is_empty() {
        out.push_str(&format!("parent: {}\n", yaml_quote(parent)));
    }
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

// ===== save_skill：支持 package 布局（references/examples/scripts + subskills）=====

fn parse_resources_arg(args: &Value) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::new();
    // 通用字段 `resources: [{path, content}]`
    if let Some(arr) = args.get("resources").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(obj) = item.as_object() {
                let path = obj
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                let content = obj
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if path.is_empty() {
                    return Err("resources[]: each entry requires non-empty 'path'".to_string());
                }
                validate_resource_relative_path(path)?;
                out.push((path.to_string(), content.to_string()));
            } else {
                return Err("resources[] entries must be objects with 'path' and 'content'".to_string());
            }
        }
    }
    // 分类快捷字段：references / examples / scripts，各为同形数组，若 path 不含目录则自动补前缀
    for (key, prefix) in [("references", "references"), ("examples", "examples"), ("scripts", "scripts")] {
        if let Some(arr) = args.get(key).and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(obj) = item.as_object() {
                    let raw_path = obj.get("path").and_then(|v| v.as_str()).unwrap_or("").trim();
                    let content = obj.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    let path = if raw_path.is_empty() {
                        return Err(format!("{key}[]: each entry requires 'path'"));
                    } else if raw_path.contains('/') {
                        raw_path.to_string()
                    } else {
                        format!("{prefix}/{raw_path}")
                    };
                    validate_resource_relative_path(&path)?;
                    out.push((path, content.to_string()));
                } else {
                    return Err(format!("{key}[] entries must be objects with 'path' and 'content'"));
                }
            }
        }
    }
    Ok(out)
}

fn parse_subskills_arg(args: &Value) -> Result<Vec<Value>, String> {
    let Some(arr) = args.get("subskills").and_then(|v| v.as_array()) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for (idx, item) in arr.iter().enumerate() {
        let obj = item.as_object().ok_or_else(|| format!("subskills[{idx}] must be an object"))?;
        let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("").trim();
        let prompt = obj.get("prompt").and_then(|v| v.as_str()).unwrap_or("").trim();
        if name.is_empty() {
            return Err(format!("subskills[{idx}].name is required and must be non-empty"));
        }
        if prompt.is_empty() {
            return Err(format!("subskills[{idx}].prompt is required and must be non-empty"));
        }
        out.push(item.clone());
    }
    Ok(out)
}

fn write_resource_files(base_dir: &std::path::Path, resources: &[(String, String)]) -> Result<(), String> {
    write_resource_files_with_overwrite(base_dir, resources, true)
}

fn write_resource_files_with_overwrite(
    base_dir: &std::path::Path,
    resources: &[(String, String)],
    overwrite: bool,
) -> Result<(), String> {
    // 确保 base_dir 存在并可 canonicalize（前置校验）
    fs::create_dir_all(base_dir)
        .map_err(|e| format!("Failed to create dir {}: {e}", base_dir.display()))?;
    let canonical_base = fs::canonicalize(base_dir)
        .unwrap_or_else(|_| base_dir.to_path_buf());
    for (rel, content) in resources {
        let rel_clean = rel.trim().trim_matches('/').replace('\\', "/");
        // 复用已有校验（含 ..、绝对路径）
        let validated = validate_resource_relative_path(&rel_clean)
            .map_err(|e| format!("Invalid resource path '{rel}': {e}"))?;
        let target = base_dir.join(&validated);
        // 创建父目录前先做路径穿越预检：target 的 parent canonical 必须在 base 内
        if let Some(parent) = target.parent() {
            // parent 可能尚不存在，校验其“预期”路径的前缀字符级穿越
            // 使用组件级检查：若 validated 含 `..` 已在上一步拦截，此处再做 canonical 校验
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create dir {}: {e}", parent.display()))?;
            // 对已创建的 parent 做 canonical 校验
            if let Ok(canonical_parent) = fs::canonicalize(parent) {
                if !canonical_parent.starts_with(&canonical_base) {
                    return Err(format!("Resource path escapes package directory: {validated}"));
                }
            }
        }
        if target.exists() && !overwrite {
            return Err(format!(
                "Resource already exists and overwrite=false: {}",
                target.display()
            ));
        }
        fs::write(&target, content).map_err(|e| format!("Failed to write resource {validated}: {e}"))?;
        // 写入后二次校验（处理 symlink 竞态）
        if let (Ok(cb), Ok(ct)) = (fs::canonicalize(&canonical_base), fs::canonicalize(&target)) {
            if !ct.starts_with(&cb) {
                let _ = fs::remove_file(&target);
                return Err(format!("Resource path escapes package directory: {validated}"));
            }
        }
    }
    Ok(())
}

pub(crate) fn execute_save_skill(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().ok_or("Missing name")?.trim().to_string();
    if name.is_empty() {
        return Err("name is empty".to_string());
    }
    // 提前解析可选的复杂包能力
    let resources = parse_resources_arg(args)?;
    let subskills = parse_subskills_arg(args)?;
    let has_package_features = !resources.is_empty() || !subskills.is_empty();

    let dir = resolve_configured_skills_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create skills dir: {e}"))?;
    let overwrite = args["overwrite"].as_bool().unwrap_or(true);

    if !has_package_features {
        // 兼容旧行为：单文件 `*.skill`
        let content = build_skill_file_content(args)?;
        let file_name = args["file_name"]
            .as_str()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| safe_skill_file_name(&name));
        let file_name = safe_skill_file_name(&file_name);
        let path = dir.join(file_name);
        if path.exists() && !overwrite {
            return Err(format!(
                "Skill file already exists and overwrite=false: {}",
                path.display()
            ));
        }
        fs::write(&path, content).map_err(|e| format!("Failed to write skill file: {e}"))?;
        return Ok(format!("Skill saved: {}\nSkill name: {}", path.display(), name));
    }

    // ===== 包布局：`skills_dir/<package_dir>/SKILL.md` + resources + subskills =====
    // 事前校验：子 skill 去重与环检测
    {
        use rustc_hash::FxHashSet;
        let mut seen_names: FxHashSet<String> = FxHashSet::default();
        let mut seen_dirs: FxHashSet<String> = FxHashSet::default();
        for sub in &subskills {
            let sub_name = sub["name"].as_str().unwrap_or("").trim().to_string();
            if sub_name.is_empty() {
                continue;
            }
            if sub_name == name {
                return Err(format!("Subskill name must not equal parent skill name: '{sub_name}'"));
            }
            if let Some(ep) = sub.get("parent").and_then(|v| v.as_str()) {
                let ep = ep.trim();
                if !ep.is_empty() && ep == sub_name {
                    return Err(format!("Subskill '{sub_name}' parent must not be itself"));
                }
            }
            if !seen_names.insert(sub_name.clone()) {
                return Err(format!("Duplicate subskill name: '{sub_name}'"));
            }
            let dir_name = safe_skill_dir_name(&sub_name);
            if !seen_dirs.insert(dir_name.clone()) {
                return Err(format!("Duplicate subskill directory name (sanitized collision): '{sub_name}' -> '{dir_name}'"));
            }
        }
    }
    let package_dir_name = safe_skill_dir_name(&name);
    let package_dir = dir.join(&package_dir_name);
    let legacy_file = dir.join(safe_skill_file_name(&name));
    if package_dir.exists() && !overwrite {
        return Err(format!("Skill package already exists and overwrite=false: {}", package_dir.display()));
    }
    if legacy_file.exists() && !overwrite {
        return Err(format!("Legacy skill file blocks package creation and overwrite=false: {}", legacy_file.display()));
    }
    let main_content = build_skill_file_content(args)?;
    let tmp_dir = dir.join(format!(".tmp-{}-{}", package_dir_name, std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    let write_result: Result<Vec<String>, String> = (|| {
        fs::create_dir_all(&tmp_dir).map_err(|e| format!("Failed to create temp skill package dir: {e}"))?;
        let main_manifest = tmp_dir.join("SKILL.md");
        fs::write(&main_manifest, &main_content).map_err(|e| format!("Failed to write skill manifest: {e}"))?;
        if !resources.is_empty() {
            write_resource_files(&tmp_dir, &resources)?;
        }
        let mut created_subskills: Vec<String> = Vec::new();
        for sub in &subskills {
            let sub_name = sub["name"].as_str().unwrap_or("").trim().to_string();
            let sub_obj = sub.as_object().unwrap();
            let mut sub_args = sub.clone();
            if sub_obj.get("parent").and_then(|v| v.as_str()).unwrap_or("").trim().is_empty() {
                if let Some(map) = sub_args.as_object_mut() {
                    map.insert("parent".to_string(), Value::String(name.clone()));
                }
            }
            let sub_content = build_skill_file_content(&sub_args)?;
            let sub_dir_name = safe_skill_dir_name(&sub_name);
            let sub_dir = tmp_dir.join("subskills").join(&sub_dir_name);
            fs::create_dir_all(&sub_dir).map_err(|e| format!("Failed to create subskill dir {}: {e}", sub_dir.display()))?;
            let sub_manifest = sub_dir.join("SKILL.md");
            fs::write(&sub_manifest, sub_content).map_err(|e| format!("Failed to write subskill {sub_name}: {e}"))?;
            if let Some(arr) = sub.get("resources").and_then(|v| v.as_array()) {
                let mut sub_resources: Vec<(String, String)> = Vec::new();
                for item in arr {
                    if let Some(obj) = item.as_object() {
                        let p = obj.get("path").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                        let c = obj.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        validate_resource_relative_path(&p)?;
                        sub_resources.push((p, c));
                    }
                }
                if !sub_resources.is_empty() {
                    write_resource_files(&sub_dir, &sub_resources)?;
                }
            }
            for (key, prefix) in [("references", "references"), ("examples", "examples"), ("scripts", "scripts")] {
                if let Some(arr) = sub.get(key).and_then(|v| v.as_array()) {
                    let mut cat_resources: Vec<(String, String)> = Vec::new();
                    for item in arr {
                        if let Some(obj) = item.as_object() {
                            let raw = obj.get("path").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                            let c = obj.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let path = if raw.contains('/') { raw } else { format!("{prefix}/{raw}") };
                            validate_resource_relative_path(&path)?;
                            cat_resources.push((path, c));
                        }
                    }
                    if !cat_resources.is_empty() {
                        write_resource_files(&sub_dir, &cat_resources)?;
                    }
                }
            }
            created_subskills.push(sub_name);
        }
        Ok(created_subskills)
    })();
    let created_subskills = match write_result {
        Ok(v) => v,
        Err(e) => {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(e);
        }
    };
    // 原子发布：若目标已存在，先备份再替换，避免 remove+rename 窗口内崩溃导致数据丢失
    if package_dir.exists() {
        let backup = dir.join(format!(".bak-{}-{}", package_dir_name, std::process::id()));
        if backup.exists() {
            let _ = fs::remove_dir_all(&backup);
        }
        fs::rename(&package_dir, &backup)
            .map_err(|e| format!("Failed to backup existing skill package {}: {e}", package_dir.display()))?;
        match fs::rename(&tmp_dir, &package_dir) {
            Ok(()) => {
                let _ = fs::remove_dir_all(&backup);
            }
            Err(e) => {
                // 回滚：尽力恢复原目录
                let _ = fs::rename(&backup, &package_dir);
                let _ = fs::remove_dir_all(&tmp_dir);
                return Err(format!("Failed to publish skill package {}: {e}", package_dir.display()));
            }
        }
    } else {
        fs::rename(&tmp_dir, &package_dir)
            .map_err(|e| format!("Failed to publish skill package {}: {e}", package_dir.display()))?;
    }
    if legacy_file.exists() {
        let _ = fs::remove_file(&legacy_file);
    }
    let mut msg = format!("Skill package saved: {}\nSkill name: {}\nManifest: SKILL.md", package_dir.display(), name);
    if !resources.is_empty() {
        msg.push_str(&format!("\nResources: {} file(s)", resources.len()));
    }
    if !created_subskills.is_empty() {
        msg.push_str(&format!("\nSub-skills: {}", created_subskills.join(", ")));
    }
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::{
        build_skill_file_content, clear_pending_user_input_request, execute_activate_skill,
        execute_deactivate_skill, execute_load_skill, execute_request_user_input,
        render_loaded_skill, render_skill_catalog, take_pending_skill_action,
        take_pending_user_input_request,
    };
    use super::PendingSkillAction;
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
        assert!(take_pending_skill_action().is_empty());
    }

    #[test]
    fn activate_skill_rejects_unknown_name() {
        let _g = ACTIVATION_TEST_GUARD.lock().unwrap();
        let err = execute_activate_skill(&serde_json::json!({"name": "definitely-not-a-skill"}))
            .unwrap_err();
        assert!(err.contains("No skill named"));
        // 未命中不应写入待激活槽位，避免乱激活。
        assert!(take_pending_skill_action().is_empty());
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
        let actions = take_pending_skill_action();
        assert_eq!(actions, vec![PendingSkillAction::Add(name.clone())]);
        // take 应清空槽位。
        assert!(take_pending_skill_action().is_empty());
    }

    #[test]
    fn multiple_actions_in_one_turn_accumulate_in_order() {
        let _g = ACTIVATION_TEST_GUARD.lock().unwrap();
        // 同一 turn 内连续多次变更动作应按顺序累积，而不是后写覆盖（回归：多
        // skill 叠加时代价槽位曾是单个 action，第二次调用会静默丢失第一次）。
        execute_deactivate_skill(&serde_json::json!({"name": "alpha"})).unwrap();
        execute_deactivate_skill(&serde_json::json!({"name": "beta"})).unwrap();
        let actions = take_pending_skill_action();
        assert_eq!(
            actions,
            vec![
                PendingSkillAction::Remove("alpha".to_string()),
                PendingSkillAction::Remove("beta".to_string()),
            ]
        );
        assert!(take_pending_skill_action().is_empty());
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
        let list_spec = crate::ai::tools::registry::common::get_tool_spec("list_skills")
            .expect("list_skills should be registered");
        let list_skills = crate::ai::tools::registry::tool_metadata::tool_description(
            list_spec.name,
            list_spec.description,
        );
        assert!(list_skills.contains("Use proactively"));
        assert!(list_skills.contains("technical keywords"));
        assert!(
            list_skills
                .contains("routine source-code, repository, file, or terminal investigation")
        );

        let activate_spec = crate::ai::tools::registry::common::get_tool_spec("activate_skill")
            .expect("activate_skill should be registered");
        let activate_skill = crate::ai::tools::registry::tool_metadata::tool_description(
            activate_spec.name,
            activate_spec.description,
        );
        assert!(activate_skill.contains("use `list_skills`"));
        assert!(activate_skill.contains("technical keywords"));
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
            parent: None,
            source_path: None,
            resource_path: None,
        }
    }
}
