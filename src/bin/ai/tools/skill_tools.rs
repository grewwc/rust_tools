use std::fs;
use std::path::PathBuf;
use std::sync::{LazyLock, RwLock};

use rust_tools::commonw::{FastMap, FastSet};
use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::ai::skills::SkillManifest;
use crate::ai::tools::common::{
    ToolParallelPolicy, ToolParallelPolicyRegistration, ToolRegistration, ToolRetryPolicy,
    ToolRetryPolicyRegistration, ToolSpec,
};

/// Skill change actions requested by the model via the `activate_skill` / `deactivate_skill` tools (to be
/// read and applied by the driver on the next iteration).
///
/// Tools are pure functions `fn(&Value) -> Result<String, String>` with no access to `App`, so this follows
/// the `enable_tools.rs` bridging pattern of "tool writes global state → driver reads it". Only a tiny
/// pending-activation slot is needed, hence a single `RwLock<FastMap>` instead of a full state struct. Multiple skills
/// can be activated at once: `Add` appends to the active set and `Remove` removes from it. Multiple calls within one turn
/// accumulate into a queue in call order, and the driver applies all of them at once (no "last write wins").

/// A skill change action requested by the model. The driver reads it on the next iteration and applies it to the current active set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PendingSkillAction {
    /// Append a skill to the current active set (ignored if already present)
    Add(String),
    /// Remove a skill from the current active set
    Remove(String),
}

pub(crate) static PENDING_SKILL_ACTIVATION: LazyLock<
    RwLock<FastMap<(String, usize), Vec<PendingSkillAction>>>,
> = LazyLock::new(|| RwLock::new(FastMap::default()));

/// Explicit interaction boundary recorded by `request_user_input` for the current turn. Isolated by `(session_id, turn_id)`
/// so requests from parallel subagents or other sessions cannot pollute the current foreground turn.
static PENDING_USER_INPUT_REQUESTS: LazyLock<RwLock<FastSet<(String, usize)>>> =
    LazyLock::new(|| RwLock::new(FastSet::default()));

fn current_turn_identity() -> (String, usize) {
    crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .try_with(Clone::clone)
        .unwrap_or_default()
}

fn set_pending_skill_action(action: PendingSkillAction) {
    if let Ok(mut slot) = PENDING_SKILL_ACTIVATION.write() {
        slot.entry(current_turn_identity())
            .or_default()
            .push(action);
    }
}

/// Driver-side call: take and clear all pending skill change actions for this turn.
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

/// Driver-side call: query and clear this turn's explicit user input request.
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

    // Guard against misuse: the requested skill name must actually exist. On a miss, refuse and list the available
    // skill names so the model corrects itself instead of activating something out of thin air.
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

/// Mark the current skill as waiting for user input. The tool result only lets the model continue generating a user-facing question;
/// the real cross-turn state is saved by the driver at the end of the turn, keeping session state out of the tool layer.
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
         Ask the user this question in your final response, then stop. Present the question once in your final response; if you already wrote context in the tool-call turn, do not repeat it. If a skill is active, it will be restored only for the user's immediately following normal message."
    ))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "activate_skill",
        description: "",

        execute: execute_activate_skill,
        // Skill discovery/activation is a low-frequency capability: it does not ship in the per-turn core set by default; the model enables it on demand via
        // `enable_tools`, trimming per-turn tools schema tokens. The builtin group is kept
        // so it can still be enabled dynamically.
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "deactivate_skill",
        description: "",

        execute: execute_deactivate_skill,
        // A low-frequency control tool like activate_skill: not resident by default, enabled on demand via enable_tools.
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "request_user_input",
        description: "",

        execute: execute_request_user_input,
        // This is a control tool injected directly by name from the driver; it must not be exposed via manifest tool_groups.
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
    // The catalog is a discovery entry point, not a ranked candidate list; always list by name so priority is not mistaken for a recommendation score.
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
    // Count parent skill -> child skill edges, used to annotate parent skills
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
        // No extra hint when there are no child skills
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
    }
});

inventory::submit!(ToolParallelPolicyRegistration {
    name: "list_skills",
    policy: ToolParallelPolicy::ReadOnly,
});

inventory::submit!(ToolRetryPolicyRegistration {
    name: "list_skills",
    policy: ToolRetryPolicy::RetrySafe,
});

/// Collect the file listing (relative paths) under `resource_path` for `load_skill` / resource tool display.
/// `subdir` must be a safe relative path (only a single-level category or empty); internally validated against traversal and canonicalized to stay inside base.
fn collect_resource_files(resource_path: &str, subdir: Option<&str>) -> Vec<String> {
    use std::path::Path;
    let base = Path::new(resource_path);
    // First verify base exists and is canonicalizable (front-running check against symlink attacks)
    let Ok(canonical_base) = std::fs::canonicalize(base) else {
        // Return empty when base is missing or unresolvable (the caller already handles a missing resource_path)
        if !base.is_dir() {
            return Vec::new();
        }
        // Degraded case: base exists but canonicalize fails (e.g. permissions); still try listing directly
        let mut files = Vec::new();
        collect_files_recursive(base, base, &mut files, 120);
        rust_tools::sortw::stable_sort_by(&mut files, |a, b| a.cmp(b));
        return files;
    };
    let target_base = match subdir {
        Some(s) if !s.trim().is_empty() => {
            let cleaned = s.trim().trim_matches('/').replace('\\', "/");
            // Strict validation: no `..`, absolute paths, or multi-level traversal containing `/`
            if cleaned.is_empty() || cleaned.contains("..") || Path::new(&cleaned).is_absolute() {
                return Vec::new();
            }
            // category must be a single segment from the common resource dirs
            if cleaned.contains('/') {
                return Vec::new();
            }
            // Whitelist validation (consistent with the docs)
            const ALLOWED_CATEGORIES: &[&str] = &["references", "examples", "scripts"];
            if !ALLOWED_CATEGORIES.contains(&cleaned.as_str()) {
                return Vec::new();
            }
            let joined = base.join(&cleaned);
            // Canonical check: if the target exists it must still be inside base; if missing, treat as a legitimately empty dir
            match std::fs::canonicalize(&joined) {
                Ok(canonical_joined) if !canonical_joined.starts_with(&canonical_base) => {
                    return Vec::new();
                }
                Ok(canonical_joined) => canonical_joined,
                Err(_) => {
                    // When the target is missing, check that the joined parent is still inside base (blocks `references/../etc`-style constructs)
                    // already covered by the `..` check; treat missing as empty here
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

fn collect_files_recursive(
    root: &std::path::Path,
    cur: &std::path::Path,
    out: &mut Vec<String>,
    cap: usize,
) {
    if out.len() >= cap {
        return;
    }
    let Ok(rd) = std::fs::read_dir(cur) else {
        return;
    };
    let mut entries = rd.flatten().collect::<Vec<_>>();
    entries.sort_by(|a, b| a.path().cmp(&b.path()));
    for entry in entries {
        if out.len() >= cap {
            break;
        }
        let path = entry.path();
        let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if file_name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_files_recursive(root, &path, out, cap);
        } else if path.is_file() {
            if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.display().to_string());
            }
        }
    }
}

/// Render the load_skill return: header metadata + skill body (+ optional bundled resource listing).
fn render_loaded_skill_with_all(skill: &SkillManifest, all: &[SkillManifest]) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Skill: {}\n", skill.name));
    if !skill.description.trim().is_empty() {
        out.push_str(&format!("description: {}\n", skill.description.trim()));
    }
    out.push_str(&format!("version: {}\n", skill.version));
    if let Some(source) = skill.source_path.as_deref()
        && !source.trim().is_empty()
        && !source.starts_with("builtin:")
    {
        out.push_str(&format!("source: {source}\n"));
    }
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
    // Only expose the resource dir when the skill actually bundles resources, listing files like references/examples/scripts.
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
        // List child skills (passed in by the caller as all, avoiding duplicated full IO)
    }
    let children: Vec<&SkillManifest> = all
        .iter()
        .filter(|s| s.parent.as_deref() == Some(skill.name.as_str()))
        .collect();
    if !children.is_empty() {
        out.push_str("\n## sub-skills\nThis skill contains sub-skills:\n");
        for ch in children {
            let desc = ch
                .description
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
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
    }
});

// ===== New: tools for reading skill resources =====

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

const MAX_SKILL_RESOURCE_READ_BYTES: usize = 64 * 1024;

fn skill_resource_read_limit(args: &Value) -> usize {
    args.get("limit")
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(MAX_SKILL_RESOURCE_READ_BYTES)
        .clamp(1, MAX_SKILL_RESOURCE_READ_BYTES)
}

fn utf8_prefix_at_most(content: &str, max_bytes: usize) -> &str {
    let mut end = max_bytes.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    &content[..end]
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
    // Strictly validate category: only empty or whitelisted values, with no path traversal characters
    if !category.is_empty() {
        const ALLOWED: &[&str] = &["references", "examples", "scripts"];
        if category.contains("..") || category.contains('/') || category.contains('\\') {
            return Err(format!(
                "Invalid category '{category}': must be one of references/examples/scripts and must not contain path separators"
            ));
        }
        if !ALLOWED.contains(&category) {
            return Err(format!(
                "Invalid category '{category}': allowed values are references, examples, scripts"
            ));
        }
    }
    let skill = resolve_skill_for_resource(name)?;
    let resource_path = skill.resource_path.as_deref().ok_or_else(|| {
        format!("Skill '{name}' has no bundled resource directory (single-file skill)")
    })?;
    let subdir = if category.is_empty() {
        None
    } else {
        Some(category)
    };
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
        // Try to infer the size
        let abs = std::path::Path::new(resource_path).join(rel);
        let size = std::fs::metadata(&abs).map(|m| m.len()).unwrap_or(0);
        out.push_str(&format!("- {rel} ({size} bytes)\n"));
    }
    if total > shown {
        out.push_str(&format!(
            "... and {} more files (use limit to see more)\n",
            total - shown
        ));
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
    // After normalization, ensure the path is still inside resource_path (anti path traversal)
    let canonical_base = std::fs::canonicalize(resource_path)
        .unwrap_or_else(|_| std::path::PathBuf::from(resource_path));
    let canonical_target = std::fs::canonicalize(&abs)
        .map_err(|e| format!("Failed to resolve resource path '{rel}': {e}"))?;
    if !canonical_target.starts_with(&canonical_base) {
        return Err("resource path escapes skill resource directory".to_string());
    }
    let content = std::fs::read_to_string(&canonical_target)
        .map_err(|e| format!("Failed to read resource '{rel}': {e}"))?;
    let limit = skill_resource_read_limit(args);
    if content.len() > limit {
        let truncated = utf8_prefix_at_most(&content, limit);
        Ok(format!(
            "{truncated}\n\n... truncated: {} bytes total, showing first {} UTF-8 bytes (limit {}). Use read_file with offset/limit for paging.",
            content.len(),
            truncated.len(),
            limit
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
    }
});

inventory::submit!(ToolParallelPolicyRegistration {
    name: "list_skill_resources",
    policy: ToolParallelPolicy::ReadOnly,
});

inventory::submit!(ToolRetryPolicyRegistration {
    name: "list_skill_resources",
    policy: ToolRetryPolicy::RetrySafe,
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_skill_resource",
        description: "",
        execute: execute_read_skill_resource,
    }
});

inventory::submit!(ToolParallelPolicyRegistration {
    name: "read_skill_resource",
    policy: ToolParallelPolicy::ReadOnly,
});

inventory::submit!(ToolRetryPolicyRegistration {
    name: "read_skill_resource",
    policy: ToolRetryPolicy::RetrySafe,
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "save_skill",
        description: "",

        execute: execute_save_skill,
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

/// Shared skill name sanitization core: lowercase, keep alnum/`-`/`_`/` .`, convert the rest to `-`, collapse `--`, trim `-`/` .`.
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
    // Use strip_suffix for an exact check; trim_end_matches would wrongly strip whole character sets
    if base.to_ascii_lowercase().ends_with(".skill") {
        base
    } else {
        format!("{base}.skill")
    }
}

fn safe_skill_dir_name(name: &str) -> String {
    let mut base = sanitize_skill_basename(name);
    // Strip the .skill suffix (if any) so the file and dir basenames match: a.b.skill ↔ a.b
    if base.to_ascii_lowercase().ends_with(".skill") {
        if let Some(stripped) = base.get(..base.len() - ".skill".len()) {
            base = stripped
                .trim_end_matches('-')
                .trim_end_matches('.')
                .to_string();
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

// ===== save_skill: package layout support (references/examples/scripts + subskills) =====

const MAX_IMPORTED_RESOURCE_FILES: usize = 1024;
const MAX_IMPORTED_RESOURCE_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone)]
enum ResourceSource {
    /// UTF-8 text supplied directly in the tool arguments.
    Text(String),
    /// Base64-encoded bytes supplied directly in the tool arguments.
    Base64(String),
    /// Copy an existing file, or a directory tree, from disk.
    Copy {
        from: String,
        resolved: PathBuf,
        is_dir: bool,
    },
}

#[derive(Debug, Clone)]
struct SkillResource {
    path: String,
    source: ResourceSource,
    executable: bool,
}

/// Parse `resources` plus the `references`/`examples`/`scripts` aliases into a
/// single destination list. Each entry has `path` and exactly one source:
/// inline `content`, inline `content_base64`, or a filesystem `from`.
fn parse_resources_arg(args: &Value, label_prefix: &str) -> Result<Vec<SkillResource>, String> {
    let mut out: Vec<SkillResource> = Vec::new();
    if let Some(arr) = args.get("resources").and_then(|v| v.as_array()) {
        for (idx, item) in arr.iter().enumerate() {
            out.push(parse_resource_entry(
                item,
                &format!("{label_prefix}resources[{idx}]"),
                None,
            )?);
        }
    }
    // A bare alias path gets the category prefix; a path that already carries a
    // directory is kept as written so callers can nest below the category.
    for (key, prefix) in [
        ("references", "references"),
        ("examples", "examples"),
        ("scripts", "scripts"),
    ] {
        if let Some(arr) = args.get(key).and_then(|v| v.as_array()) {
            for (idx, item) in arr.iter().enumerate() {
                out.push(parse_resource_entry(
                    item,
                    &format!("{label_prefix}{key}[{idx}]"),
                    Some(prefix),
                )?);
            }
        }
    }
    validate_resource_layout(&out)?;
    Ok(out)
}

fn parse_resource_entry(
    item: &Value,
    label: &str,
    default_prefix: Option<&str>,
) -> Result<SkillResource, String> {
    let obj = item.as_object().ok_or_else(|| {
        format!(
            "{label} must be an object with 'path' and exactly one of 'content', 'content_base64', or 'from'"
        )
    })?;
    let raw_path = obj
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if raw_path.is_empty() {
        return Err(format!("{label}: each entry requires non-empty 'path'"));
    }
    let path = match default_prefix {
        Some(prefix) if !raw_path.contains('/') && !raw_path.contains('\\') => {
            format!("{prefix}/{raw_path}")
        }
        _ => raw_path.to_string(),
    };
    let path = validate_resource_path(&path).map_err(|e| format!("{label}: {e}"))?;
    let content = obj.get("content").and_then(|v| v.as_str());
    let content_base64 = obj.get("content_base64").and_then(|v| v.as_str());
    let from = obj.get("from").and_then(|v| v.as_str());
    let provided = [content.is_some(), content_base64.is_some(), from.is_some()]
        .into_iter()
        .filter(|present| *present)
        .count();
    if provided != 1 {
        return Err(format!(
            "{label} '{path}': provide exactly one of 'content', 'content_base64', or 'from'"
        ));
    }
    let source = if let Some(text) = content {
        ResourceSource::Text(text.to_string())
    } else if let Some(encoded) = content_base64 {
        ResourceSource::Base64(encoded.to_string())
    } else {
        let raw = from.unwrap_or("").trim();
        if raw.is_empty() {
            return Err(format!("{label} '{path}': 'from' is empty"));
        }
        let (resolved, is_dir) =
            resolve_import_source(raw).map_err(|e| format!("{label} '{path}': {e}"))?;
        ResourceSource::Copy {
            from: raw.to_string(),
            resolved,
            is_dir,
        }
    };
    let executable = obj
        .get("executable")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if executable && matches!(&source, ResourceSource::Copy { is_dir: true, .. }) {
        return Err(format!(
            "{label} '{path}': 'executable' only applies to one file; directory imports preserve source modes"
        ));
    }
    Ok(SkillResource {
        path,
        source,
        executable,
    })
}

fn resolve_import_source(raw: &str) -> Result<(PathBuf, bool), String> {
    let expanded = crate::commonw::utils::expanduser(raw);
    let path = PathBuf::from(expanded.as_ref());
    let resolved = if path.is_absolute() {
        path
    } else {
        crate::ai::driver::runtime_ctx::effective_cwd()
            .map_err(|e| format!("cannot resolve relative path '{raw}': {e}"))?
            .join(path)
    };
    let meta = fs::symlink_metadata(&resolved)
        .map_err(|e| format!("import source '{raw}' is not accessible: {e}"))?;
    if meta.file_type().is_symlink() {
        return Err(format!("import source '{raw}' must not be a symlink"));
    }
    if !meta.is_file() && !meta.is_dir() {
        return Err(format!(
            "import source '{raw}' is neither a regular file nor a directory"
        ));
    }
    Ok((resolved, meta.is_dir()))
}

/// Validate a resource destination. `SKILL.md` and the `subskills/` tree are
/// generated by `save_skill`, so a resource may not overwrite them.
fn validate_resource_path(path: &str) -> Result<String, String> {
    let raw = path.trim().replace('\\', "/");
    if raw.is_empty() {
        return Err("resource path is empty".to_string());
    }
    // Check absoluteness before trimming separators. Trimming first would hide
    // a leading `/` and make the absolute-path rejection unreachable.
    let has_drive_prefix =
        raw.len() >= 2 && raw.as_bytes()[0].is_ascii_alphabetic() && raw.as_bytes()[1] == b':';
    if raw.starts_with('/') || has_drive_prefix || std::path::Path::new(&raw).is_absolute() {
        return Err("resource path must be relative".to_string());
    }
    let p = raw.trim_end_matches('/');
    if p.is_empty() {
        return Err("resource path is empty".to_string());
    }
    if p.split('/').any(|component| component == "..") {
        return Err("resource path must not contain '..'".to_string());
    }
    if p.split('/')
        .any(|component| component.is_empty() || component == ".")
    {
        return Err("resource path must not contain empty or '.' components".to_string());
    }
    if p.eq_ignore_ascii_case("SKILL.md") {
        return Err("'SKILL.md' is reserved for the generated skill manifest".to_string());
    }
    if p.split('/')
        .next()
        .is_some_and(|first| first.eq_ignore_ascii_case("subskills"))
    {
        return Err("paths under 'subskills/' are reserved for generated sub-skills".to_string());
    }
    Ok(p.to_string())
}

fn validate_resource_layout(resources: &[SkillResource]) -> Result<(), String> {
    let mut roots: Vec<String> = Vec::with_capacity(resources.len());
    for resource in resources {
        let key = resource.path.to_ascii_lowercase();
        for existing in &roots {
            if *existing == key {
                return Err(format!("Duplicate resource path: '{}'", resource.path));
            }
            if key.starts_with(&format!("{existing}/"))
                || existing.starts_with(&format!("{key}/"))
            {
                return Err(format!(
                    "Resource destination '{}' overlaps '{}'",
                    resource.path, existing
                ));
            }
        }
        roots.push(key);
    }
    Ok(())
}

fn parse_subskills_arg(args: &Value) -> Result<Vec<Value>, String> {
    let Some(arr) = args.get("subskills").and_then(|v| v.as_array()) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for (idx, item) in arr.iter().enumerate() {
        let obj = item
            .as_object()
            .ok_or_else(|| format!("subskills[{idx}] must be an object"))?;
        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let prompt = obj
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if name.is_empty() {
            return Err(format!(
                "subskills[{idx}].name is required and must be non-empty"
            ));
        }
        if prompt.is_empty() {
            return Err(format!(
                "subskills[{idx}].prompt is required and must be non-empty"
            ));
        }
        out.push(item.clone());
    }
    Ok(out)
}

#[derive(Debug)]
enum PlannedSource {
    Text(String),
    Bytes(Vec<u8>),
    File(PathBuf),
}

#[derive(Debug)]
struct PlannedResourceFile {
    path: String,
    source: PlannedSource,
    executable: bool,
}

#[derive(Default)]
struct ResourcePlan {
    files: Vec<PlannedResourceFile>,
    seen: rustc_hash::FxHashSet<String>,
    total_bytes: u64,
}

impl ResourcePlan {
    fn push(
        &mut self,
        path: String,
        source: PlannedSource,
        size: u64,
        executable: bool,
    ) -> Result<(), String> {
        if !self.seen.insert(path.to_ascii_lowercase()) {
            return Err(format!("Duplicate resource path: '{path}'"));
        }
        self.total_bytes = self
            .total_bytes
            .checked_add(size)
            .ok_or_else(|| "Resource size overflow".to_string())?;
        if self.total_bytes > MAX_IMPORTED_RESOURCE_BYTES {
            return Err(format!(
                "Resources exceed the {} MiB package limit",
                MAX_IMPORTED_RESOURCE_BYTES / (1024 * 1024)
            ));
        }
        self.files.push(PlannedResourceFile {
            path,
            source,
            executable,
        });
        if self.files.len() > MAX_IMPORTED_RESOURCE_FILES {
            return Err(format!(
                "Resources exceed the {} file limit",
                MAX_IMPORTED_RESOURCE_FILES
            ));
        }
        Ok(())
    }
}

/// Expand every resource into concrete files without writing anything. The
/// same plan drives validation, dry-run counts, and the actual copy step.
fn plan_resource_files(resources: &[SkillResource]) -> Result<Vec<PlannedResourceFile>, String> {
    use base64::Engine as _;
    let mut plan = ResourcePlan::default();
    for resource in resources {
        match &resource.source {
            ResourceSource::Text(text) => {
                plan.push(
                    resource.path.clone(),
                    PlannedSource::Text(text.clone()),
                    text.len() as u64,
                    resource.executable,
                )?;
            }
            ResourceSource::Base64(encoded) => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .map_err(|e| format!("Resource '{}': invalid base64: {e}", resource.path))?;
                let size = bytes.len() as u64;
                plan.push(
                    resource.path.clone(),
                    PlannedSource::Bytes(bytes),
                    size,
                    resource.executable,
                )?;
            }
            ResourceSource::Copy {
                from,
                resolved,
                is_dir,
            } => {
                if *is_dir {
                    let mut rels = Vec::new();
                    collect_import_files(resolved, resolved, &mut rels)?;
                    if rels.is_empty() {
                        return Err(format!(
                            "Resource '{}': import directory '{}' contains no files",
                            resource.path, from
                        ));
                    }
                    for rel in rels {
                        let rel_str = rel.to_string_lossy().replace('\\', "/");
                        let dest =
                            validate_resource_path(&format!("{}/{}", resource.path, rel_str))?;
                        let src = resolved.join(&rel);
                        let size = fs::metadata(&src)
                            .map_err(|e| format!("Cannot stat import file {}: {e}", src.display()))?
                            .len();
                        plan.push(dest, PlannedSource::File(src), size, false)?;
                    }
                } else {
                    let size = fs::metadata(resolved)
                        .map_err(|e| {
                            format!("Cannot stat import file '{}': {e}", resolved.display())
                        })?
                        .len();
                    plan.push(
                        resource.path.clone(),
                        PlannedSource::File(resolved.clone()),
                        size,
                        resource.executable,
                    )?;
                }
            }
        }
    }
    Ok(plan.files)
}

fn collect_import_files(
    source_root: &std::path::Path,
    dir: &std::path::Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    if out.len() >= MAX_IMPORTED_RESOURCE_FILES {
        return Err(format!(
            "Import directory '{}' exceeds the {} file limit",
            source_root.display(),
            MAX_IMPORTED_RESOURCE_FILES
        ));
    }
    let mut entries = fs::read_dir(dir)
        .map_err(|e| format!("Failed to read import directory {}: {e}", dir.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to read import directory {}: {e}", dir.display()))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if matches!(name.as_ref(), ".git" | ".hg" | ".svn" | ".DS_Store") {
            continue;
        }
        let path = entry.path();
        let meta = fs::symlink_metadata(&path)
            .map_err(|e| format!("Failed to inspect import entry {}: {e}", path.display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!(
                "Import directory contains a symlink, which is not supported: {}",
                path.display()
            ));
        }
        if meta.is_dir() {
            collect_import_files(source_root, &path, out)?;
        } else if meta.is_file() {
            let rel = path
                .strip_prefix(source_root)
                .map_err(|_| format!("Import path escapes source directory: {}", path.display()))?;
            if out.len() >= MAX_IMPORTED_RESOURCE_FILES {
                return Err(format!(
                    "Import directory '{}' exceeds the {} file limit",
                    source_root.display(),
                    MAX_IMPORTED_RESOURCE_FILES
                ));
            }
            out.push(rel.to_path_buf());
        } else {
            return Err(format!(
                "Import directory contains a non-regular entry: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn write_planned_resource_files(
    base_dir: &std::path::Path,
    files: &[PlannedResourceFile],
) -> Result<(), String> {
    fs::create_dir_all(base_dir)
        .map_err(|e| format!("Failed to create dir {}: {e}", base_dir.display()))?;
    let canonical_base = fs::canonicalize(base_dir).unwrap_or_else(|_| base_dir.to_path_buf());
    for file in files {
        let target = base_dir.join(&file.path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create dir {}: {e}", parent.display()))?;
            if let Ok(canonical_parent) = fs::canonicalize(parent) {
                if !canonical_parent.starts_with(&canonical_base) {
                    return Err(format!(
                        "Resource path escapes package directory: {}",
                        file.path
                    ));
                }
            }
        }
        if target.exists() {
            return Err(format!("Resource already exists: {}", target.display()));
        }
        match &file.source {
            PlannedSource::Text(text) => {
                fs::write(&target, text)
                    .map_err(|e| format!("Failed to write resource {}: {e}", file.path))?;
            }
            PlannedSource::Bytes(bytes) => {
                fs::write(&target, bytes)
                    .map_err(|e| format!("Failed to write resource {}: {e}", file.path))?;
            }
            PlannedSource::File(source) => {
                fs::copy(source, &target).map_err(|e| {
                    format!(
                        "Failed to copy resource {} -> {}: {e}",
                        source.display(),
                        file.path
                    )
                })?;
            }
        }
        #[cfg(unix)]
        if file.executable {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&target)
                .map_err(|e| format!("Failed to inspect resource {}: {e}", file.path))?
                .permissions();
            perms.set_mode(perms.mode() | 0o111);
            fs::set_permissions(&target, perms)
                .map_err(|e| format!("Failed to mark resource {} executable: {e}", file.path))?;
        }
        if let (Ok(cb), Ok(ct)) = (fs::canonicalize(&canonical_base), fs::canonicalize(&target)) {
            if !ct.starts_with(&cb) {
                let _ = fs::remove_file(&target);
                return Err(format!(
                    "Resource path escapes package directory: {}",
                    file.path
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn execute_save_skill(args: &Value) -> Result<String, String> {
    let name = args["name"]
        .as_str()
        .ok_or("Missing name")?
        .trim()
        .to_string();
    if name.is_empty() {
        return Err("name is empty".to_string());
    }
    // Resolve the optional complex-package capabilities up front
    let resources = parse_resources_arg(args, "")?;
    let subskills = parse_subskills_arg(args)?;
    let has_package_features = !resources.is_empty() || !subskills.is_empty();
    let resource_plan = plan_resource_files(&resources)?;

    let dir = resolve_configured_skills_dir();
    let overwrite = args["overwrite"].as_bool().unwrap_or(true);
    // Preview mode: validate and render without writing, so a session-distilled
    // draft can be iterated into a generic skill before saving.
    if args["dry_run"].as_bool().unwrap_or(false) {
        return render_skill_dry_run_preview(args, &name, &resources, &subskills);
    }
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create skills dir: {e}"))?;

    let package_dir_name = safe_skill_dir_name(&name);
    let package_dir = dir.join(&package_dir_name);
    let legacy_file = dir.join(safe_skill_file_name(&name));
    if !has_package_features {
        if package_dir.exists() {
            if !overwrite {
                return Err(format!(
                    "Skill package already exists and overwrite=false: {}",
                    package_dir.display()
                ));
            }
            // A manifest-only save is the common update path for a package:
            // replace SKILL.md in place and keep its resources/subskills.
            if args.get("resources").is_none() && args.get("subskills").is_none() {
                let content = build_skill_file_content(args)?;
                let manifest = package_dir.join("SKILL.md");
                fs::write(&manifest, content).map_err(|e| {
                    format!(
                        "Failed to update skill manifest {}: {e}",
                        manifest.display()
                    )
                })?;
                return Ok(format!(
                    "Skill package manifest updated: {}\nSkill name: {name}",
                    package_dir.display()
                ));
            }
            return Err(format!(
                "Skill package already exists for '{name}'; pass non-empty `resources`/`subskills` to rebuild it, or omit both fields to update only SKILL.md: {}",
                package_dir.display()
            ));
        }
        // Legacy behavior compatibility: single-file `*.skill`
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
        return Ok(format!(
            "Skill saved: {}\nSkill name: {}",
            path.display(),
            name
        ));
    }

    // ===== Package layout: `skills_dir/<package_dir>/SKILL.md` + resources + subskills =====
    // Upfront validation: child skill dedup and cycle detection
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
                return Err(format!(
                    "Subskill name must not equal parent skill name: '{sub_name}'"
                ));
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
                return Err(format!(
                    "Duplicate subskill directory name (sanitized collision): '{sub_name}' -> '{dir_name}'"
                ));
            }
        }
    }
    // Expand sub-skill imports before creating the temp package so a malformed
    // source fails without leaving partial output.
    let mut sub_resource_plans: Vec<Vec<PlannedResourceFile>> = Vec::with_capacity(subskills.len());
    for (idx, sub) in subskills.iter().enumerate() {
        let sub_resources = parse_resources_arg(sub, &format!("subskills[{idx}]."))?;
        sub_resource_plans.push(plan_resource_files(&sub_resources)?);
    }
    if package_dir.exists() && !overwrite {
        return Err(format!(
            "Skill package already exists and overwrite=false: {}",
            package_dir.display()
        ));
    }
    if legacy_file.exists() && !overwrite {
        return Err(format!(
            "Legacy skill file blocks package creation and overwrite=false: {}",
            legacy_file.display()
        ));
    }
    let main_content = build_skill_file_content(args)?;
    let tmp_dir = dir.join(format!(".tmp-{}-{}", package_dir_name, std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    let write_result: Result<Vec<String>, String> = (|| {
        fs::create_dir_all(&tmp_dir)
            .map_err(|e| format!("Failed to create temp skill package dir: {e}"))?;
        let main_manifest = tmp_dir.join("SKILL.md");
        fs::write(&main_manifest, &main_content)
            .map_err(|e| format!("Failed to write skill manifest: {e}"))?;
        if !resource_plan.is_empty() {
            write_planned_resource_files(&tmp_dir, &resource_plan)?;
        }
        let mut created_subskills: Vec<String> = Vec::new();
        for (idx, sub) in subskills.iter().enumerate() {
            let sub_name = sub["name"].as_str().unwrap_or("").trim().to_string();
            let sub_obj = sub.as_object().unwrap();
            let mut sub_args = sub.clone();
            if sub_obj
                .get("parent")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .is_empty()
            {
                if let Some(map) = sub_args.as_object_mut() {
                    map.insert("parent".to_string(), Value::String(name.clone()));
                }
            }
            let sub_content = build_skill_file_content(&sub_args)?;
            let sub_dir_name = safe_skill_dir_name(&sub_name);
            let sub_dir = tmp_dir.join("subskills").join(&sub_dir_name);
            fs::create_dir_all(&sub_dir)
                .map_err(|e| format!("Failed to create subskill dir {}: {e}", sub_dir.display()))?;
            let sub_manifest = sub_dir.join("SKILL.md");
            fs::write(&sub_manifest, sub_content)
                .map_err(|e| format!("Failed to write subskill {sub_name}: {e}"))?;
            if let Some(plan) = sub_resource_plans.get(idx) {
                if !plan.is_empty() {
                    write_planned_resource_files(&sub_dir, plan)?;
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
    // Atomic publish: if the target exists, back it up before replacing, so a crash inside the remove+rename window cannot lose data
    if package_dir.exists() {
        let backup = dir.join(format!(".bak-{}-{}", package_dir_name, std::process::id()));
        if backup.exists() {
            let _ = fs::remove_dir_all(&backup);
        }
        fs::rename(&package_dir, &backup).map_err(|e| {
            format!(
                "Failed to backup existing skill package {}: {e}",
                package_dir.display()
            )
        })?;
        match fs::rename(&tmp_dir, &package_dir) {
            Ok(()) => {
                let _ = fs::remove_dir_all(&backup);
            }
            Err(e) => {
                // Rollback: restore the original directory best-effort
                let _ = fs::rename(&backup, &package_dir);
                let _ = fs::remove_dir_all(&tmp_dir);
                return Err(format!(
                    "Failed to publish skill package {}: {e}",
                    package_dir.display()
                ));
            }
        }
    } else {
        fs::rename(&tmp_dir, &package_dir).map_err(|e| {
            format!(
                "Failed to publish skill package {}: {e}",
                package_dir.display()
            )
        })?;
    }
    if legacy_file.exists() {
        let _ = fs::remove_file(&legacy_file);
    }
    let mut msg = format!(
        "Skill package saved: {}\nSkill name: {}\nManifest: SKILL.md",
        package_dir.display(),
        name
    );
    if !resource_plan.is_empty() {
        msg.push_str(&format!("\nResources: {} file(s)", resource_plan.len()));
    }
    if !created_subskills.is_empty() {
        msg.push_str(&format!("\nSub-skills: {}", created_subskills.join(", ")));
    }
    Ok(msg)
}

// ===== Skill validation (shared by save_skill dry_run and validate_skill) =====
//
// Read-only checks: `from` imports are statted or enumerated so a broken
// import fails before anything is persisted.
fn validate_resources_arg(
    args: &Value,
    label_prefix: &str,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
) {
    match parse_resources_arg(args, label_prefix) {
        Ok(items) => {
            for item in &items {
                if let ResourceSource::Text(text) = &item.source {
                    if text.trim().is_empty() {
                        warnings.push(format!("resource '{}' is empty", item.path));
                    }
                }
            }
            if let Err(e) = plan_resource_files(&items) {
                errors.push(e);
            }
        }
        Err(e) => errors.push(e),
    }
}

fn validate_skill_args(args: &Value) -> (Vec<String>, Vec<String>) {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let name = args["name"].as_str().unwrap_or("").trim();
    if name.is_empty() {
        errors.push("name is empty".to_string());
    }
    let prompt = args["prompt"].as_str().unwrap_or("");
    if prompt.trim().is_empty() {
        errors.push("prompt is empty".to_string());
    } else {
        // Generic-skill hygiene: session-distilled drafts often leak
        // machine-specific details that block reuse on another machine.
        for marker in ["/Users/", "/home/", "C:\\", "/private/tmp", "/var/folders/"] {
            if prompt.contains(marker) {
                warnings.push(format!(
                    "prompt contains absolute path marker '{marker}'; prefer relative paths or placeholders for a reusable skill"
                ));
                break;
            }
        }
        if prompt.trim().len() < 50 {
            warnings.push(
                "prompt is very short; a reusable skill usually states goal, inputs, steps, and done criteria"
                    .to_string(),
            );
        }
    }
    if args["description"].as_str().unwrap_or("").trim().is_empty() {
        warnings.push("description is empty; skill routing relies on it".to_string());
    }
    let tools = parse_string_array(&args["tools"]);
    if let Some(unknown) = crate::ai::tools::manifest_unknown_tool_names_warning(&tools) {
        warnings.push(format!("unknown tool name(s): {unknown}"));
    }
    for group in parse_string_array(&args["tool_groups"]) {
        if crate::ai::tools::ToolGroup::from_name(&group).is_none() {
            errors.push(format!("unknown tool group: '{group}'"));
        }
    }
    validate_resources_arg(args, "", &mut errors, &mut warnings);
    match parse_subskills_arg(args) {
        Ok(subs) => {
            use rustc_hash::FxHashSet;
            let mut seen: FxHashSet<String> = FxHashSet::default();
            let mut seen_dirs: FxHashSet<String> = FxHashSet::default();
            for (idx, sub) in subs.iter().enumerate() {
                let sub_name = sub["name"].as_str().unwrap_or("").trim();
                if sub_name.is_empty() {
                    errors.push("subskill name is empty".to_string());
                    continue;
                }
                if sub_name == name {
                    errors.push(format!(
                        "Subskill name must not equal parent skill name: '{sub_name}'"
                    ));
                }
                if !seen.insert(sub_name.to_string()) {
                    errors.push(format!("Duplicate subskill name: '{sub_name}'"));
                }
                if let Some(parent) = sub["parent"].as_str() {
                    let parent = parent.trim();
                    if !parent.is_empty() && parent == sub_name {
                        errors.push(format!("Subskill '{sub_name}' parent must not be itself"));
                    }
                }
                let dir_name = safe_skill_dir_name(sub_name);
                if !seen_dirs.insert(dir_name.clone()) {
                    errors.push(format!(
                        "Duplicate subskill directory name (sanitized collision): '{sub_name}' -> '{dir_name}'"
                    ));
                }
                if sub["prompt"].as_str().unwrap_or("").trim().is_empty() {
                    errors.push(format!("Subskill '{sub_name}' prompt is empty"));
                }
                validate_resources_arg(
                    sub,
                    &format!("subskills[{idx}]."),
                    &mut errors,
                    &mut warnings,
                );
            }
        }
        Err(e) => errors.push(e),
    }
    (errors, warnings)
}

/// Preview-only rendering for `save_skill(dry_run=true)`: validates and renders
/// without writing. `from` imports may be read to count the files they expand to.
fn render_skill_dry_run_preview(
    args: &Value,
    name: &str,
    resources: &[SkillResource],
    subskills: &[Value],
) -> Result<String, String> {
    let content = build_skill_file_content(args)?;
    let (errors, warnings) = validate_skill_args(args);
    if !errors.is_empty() {
        return Err(format!("Skill draft invalid:\n- {}", errors.join("\n- ")));
    }
    let mut out = String::from("Skill dry-run preview (nothing written):\n");
    if subskills.is_empty() && resources.is_empty() {
        out.push_str(&format!(
            "Target: {}\n---\n{content}\n---",
            safe_skill_file_name(name)
        ));
    } else {
        out.push_str(&format!(
            "Target package: {}/\nManifest (SKILL.md):\n---\n{content}\n---",
            safe_skill_dir_name(name)
        ));
        if !resources.is_empty() {
            let planned = plan_resource_files(resources).unwrap_or_default();
            out.push_str(&format!("\nResources: {} file(s)", planned.len()));
            for resource in resources {
                match &resource.source {
                    ResourceSource::Text(_) => {
                        out.push_str(&format!("\n  - {} (inline text)", resource.path));
                    }
                    ResourceSource::Base64(_) => {
                        out.push_str(&format!("\n  - {} (inline base64)", resource.path));
                    }
                    ResourceSource::Copy { from, is_dir, .. } if *is_dir => {
                        out.push_str(&format!(
                            "\n  - {}/ (import directory {from})",
                            resource.path
                        ));
                    }
                    ResourceSource::Copy { from, .. } => {
                        out.push_str(&format!("\n  - {} (import file {from})", resource.path));
                    }
                }
            }
        }
        if !subskills.is_empty() {
            out.push_str(&format!("\nSub-skills: {} skill(s)", subskills.len()));
            for sub in subskills {
                out.push_str(&format!(
                    "\n  - {}",
                    sub["name"].as_str().unwrap_or("").trim()
                ));
            }
        }
    }
    if !warnings.is_empty() {
        out.push_str(&format!("\nWarnings:\n- {}", warnings.join("\n- ")));
    }
    out.push_str("\nRe-run save_skill without dry_run to persist.");
    Ok(out)
}

pub(crate) fn execute_validate_skill(args: &Value) -> Result<String, String> {
    let (errors, warnings) = validate_skill_args(args);
    if !errors.is_empty() {
        return Err(format!(
            "Skill validation failed:\n- {}",
            errors.join("\n- ")
        ));
    }
    let name = args["name"].as_str().unwrap_or("").trim();
    let mut out = format!("Skill '{name}' is valid.");
    if !warnings.is_empty() {
        out.push_str(&format!("\nWarnings:\n- {}", warnings.join("\n- ")));
    }
    Ok(out)
}

/// Remove one user skill by name from the configured skills directory.
/// Only the sanitized single-file (`*.skill`) and package-directory entries
/// are candidates; builtin skills are compiled in and cannot be removed here.
fn delete_skill_entries(dir: &std::path::Path, name: &str) -> Result<Vec<String>, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("delete_skill requires a non-empty 'name'.".to_string());
    }
    let file_target = dir.join(safe_skill_file_name(trimmed));
    let dir_target = dir.join(safe_skill_dir_name(trimmed));
    let canonical_base =
        fs::canonicalize(dir).map_err(|e| format!("Cannot resolve skills dir: {e}"))?;
    let mut removed: Vec<String> = Vec::new();
    for target in [&file_target, &dir_target] {
        if !target.exists() && !target.is_symlink() {
            continue;
        }
        // Fail closed on containment: never remove anything outside the skills dir.
        let canonical_target = fs::canonicalize(target)
            .map_err(|e| format!("Cannot resolve skill path {}: {e}", target.display()))?;
        if canonical_target == canonical_base || !canonical_target.starts_with(&canonical_base) {
            return Err(format!(
                "Refusing to remove path outside the skills dir: {}",
                target.display()
            ));
        }
        if target.is_symlink() {
            fs::remove_file(target)
                .map_err(|e| format!("Failed to remove skill link {}: {e}", target.display()))?;
        } else if target.is_dir() {
            fs::remove_dir_all(target)
                .map_err(|e| format!("Failed to remove skill package {}: {e}", target.display()))?;
        } else {
            fs::remove_file(target)
                .map_err(|e| format!("Failed to remove skill file {}: {e}", target.display()))?;
        }
        removed.push(target.display().to_string());
    }
    if removed.is_empty() {
        return Err(format!(
            "No skill named '{trimmed}' found in {}.",
            dir.display()
        ));
    }
    Ok(removed)
}

pub(crate) fn execute_delete_skill(args: &Value) -> Result<String, String> {
    let name = args["name"].as_str().unwrap_or("").trim().to_string();
    let dir = resolve_configured_skills_dir();
    match delete_skill_entries(&dir, &name) {
        Ok(removed) => Ok(format!("Deleted skill '{name}':\n- {}", removed.join("\n- "))),
        Err(e) => Err(e),
    }
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "validate_skill",
        description: "",

        execute: execute_validate_skill,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "delete_skill",
        description: "",

        execute: execute_delete_skill,
    }
});

#[cfg(test)]
mod tests {
    use super::PendingSkillAction;
    use super::{
        build_skill_file_content, clear_pending_user_input_request, delete_skill_entries,
        execute_activate_skill, execute_deactivate_skill, execute_load_skill,
        execute_request_user_input, execute_save_skill, execute_validate_skill,
        parse_resources_arg, plan_resource_files, render_loaded_skill, render_skill_catalog,
        skill_resource_read_limit, take_pending_skill_action, take_pending_user_input_request,
        utf8_prefix_at_most, validate_skill_args, write_planned_resource_files,
    };
    use crate::ai::driver::runtime_ctx::TURN_IDENTITY;
    use crate::ai::skills::SkillManifest;
    use std::sync::{LazyLock, Mutex};

    // activate_skill tests share one global pending-activation slot; serialize to avoid concurrent pollution.
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
        // A miss must not write into the pending-activation slot, avoiding stray activations.
        assert!(take_pending_skill_action().is_empty());
    }

    #[test]
    fn activate_skill_queues_existing_skill() {
        let _g = ACTIVATION_TEST_GUARD.lock().unwrap();
        // Take a builtin skill name that actually exists.
        let skills = crate::ai::skills::load_all_skills();
        let Some(name) = skills.first().map(|s| s.name.clone()) else {
            return;
        };
        let out = execute_activate_skill(&serde_json::json!({"name": name})).unwrap();
        assert!(out.contains(&name));
        let actions = take_pending_skill_action();
        assert_eq!(actions, vec![PendingSkillAction::Add(name.clone())]);
        // take should clear the slot.
        assert!(take_pending_skill_action().is_empty());
    }

    #[test]
    fn multiple_actions_in_one_turn_accumulate_in_order() {
        let _g = ACTIVATION_TEST_GUARD.lock().unwrap();
        // Multiple change actions within one turn must accumulate in order, not last-write-wins (regression: with multiple
        // skills stacked, the slot used to hold a single action and the second call silently dropped the first).
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
        assert!(activate_skill.contains("Use `list_skills`"));
        assert!(activate_skill.contains("loose keyword overlap"));
        assert!(activate_skill.contains("equal peers"));
    }

    #[test]
    fn skill_resource_limit_defaults_and_clamps_to_the_documented_range() {
        assert_eq!(skill_resource_read_limit(&serde_json::json!({})), 64 * 1024);
        assert_eq!(
            skill_resource_read_limit(&serde_json::json!({"limit": 0})),
            1
        );
        assert_eq!(
            skill_resource_read_limit(&serde_json::json!({"limit": 100_000})),
            64 * 1024
        );
    }

    #[test]
    fn skill_resource_truncation_respects_utf8_boundaries() {
        let content = "aé文";
        assert_eq!(utf8_prefix_at_most(content, 3), "aé");
        assert_eq!(utf8_prefix_at_most(content, 2), "a");
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
        skill.source_path = Some("/tmp/demo/SKILL.md".to_string());
        let out = render_loaded_skill(&skill);
        assert!(out.contains("# Skill: demo"));
        assert!(out.contains("description: demo description"));
        assert!(out.contains("source: /tmp/demo/SKILL.md"));
        assert!(out.contains("## prompt"));
        assert!(out.contains("line one\nline two"));
        // Expose the dir only when bundled resources exist
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

    #[test]
    fn validate_skill_accepts_clean_definition() {
        let (errors, warnings) = validate_skill_args(&serde_json::json!({
            "name": "review-checklist",
            "description": "Review checklist for documents",
            "prompt": "Goal: review a document thoroughly. Inputs: the document text. Steps: read, note issues, summarize. Done when every section has a verdict.",
            "tools": ["read_file"],
            "tool_groups": ["skills"],
        }));
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        let out = execute_validate_skill(&serde_json::json!({
            "name": "review-checklist",
            "description": "Review checklist for documents",
            "prompt": "Goal: review a document thoroughly. Inputs: the document text. Steps: read, note issues, summarize. Done when every section has a verdict.",
        }))
        .unwrap();
        assert!(out.contains("is valid"));
    }

    #[test]
    fn validate_skill_rejects_empty_prompt_and_unknown_group() {
        let (errors, _) = validate_skill_args(&serde_json::json!({
            "name": "broken",
            "description": "broken skill",
            "prompt": "  ",
            "tool_groups": ["not-a-group"],
        }));
        assert!(errors.iter().any(|e| e.contains("prompt is empty")));
        assert!(errors.iter().any(|e| e.contains("unknown tool group")));
        let err = execute_validate_skill(&serde_json::json!({
            "name": "broken",
            "prompt": "  ",
        }))
        .unwrap_err();
        assert!(err.contains("prompt is empty"));
    }

    #[test]
    fn validate_skill_warns_on_session_specific_leakage() {
        let (errors, warnings) = validate_skill_args(&serde_json::json!({
            "name": "leaky",
            "prompt": "Goal: handle the thing with enough detail to be reusable across machines. Always read /Users/someone/work/notes.md first.",
            "tools": ["definitely-not-a-tool"],
        }));
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert!(warnings.iter().any(|w| w.contains("absolute path")));
        assert!(warnings.iter().any(|w| w.contains("unknown tool")));
        assert!(warnings.iter().any(|w| w.contains("description is empty")));
    }

    #[test]
    fn save_skill_dry_run_renders_without_writing() {
        let out = execute_save_skill(&serde_json::json!({
            "name": "dry-run-demo",
            "description": "Dry run demo skill",
            "prompt": "Goal: demonstrate dry run rendering. Inputs: none. Steps: call with dry_run, inspect output. Done when the preview looks right.",
            "dry_run": true,
        }))
        .unwrap();
        assert!(out.contains("nothing written"));
        assert!(out.contains("dry-run-demo"));
    }

    #[test]
    fn save_skill_dry_run_rejects_invalid_draft() {
        let err = execute_save_skill(&serde_json::json!({
            "name": "dry-run-broken",
            "description": "broken draft",
            "prompt": "",
            "dry_run": true,
        }))
        .unwrap_err();
        assert!(err.contains("prompt is empty"));
    }

    #[test]
    fn save_skill_updates_package_manifest_without_dropping_resources() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let base = std::env::temp_dir().join(format!(
            "skill-package-update-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let skills = base.join("skills");
        std::fs::create_dir_all(skills.join("demo")).unwrap();
        std::fs::create_dir_all(skills.join("demo/resources")).unwrap();
        std::fs::write(
            skills.join("demo/SKILL.md"),
            "---\nname: demo\nprompt: old\n---\n",
        )
        .unwrap();
        std::fs::write(skills.join("demo/resources/keep.txt"), "keep me").unwrap();
        let cfg_path = base.join("test.configw");
        std::fs::write(
            &cfg_path,
            format!("ai.skills.dir = \"{}\"\n", skills.display()),
        )
        .unwrap();
        let old_cfg = std::env::var_os("CONFIGW_PATH");
        unsafe { std::env::set_var("CONFIGW_PATH", &cfg_path) };
        crate::commonw::configw::refresh();
        assert_eq!(
            crate::commonw::configw::get_config("ai.skills.dir", ""),
            skills.display().to_string()
        );

        let out = execute_save_skill(&serde_json::json!({
            "name": "demo",
            "prompt": "new prompt"
        }))
        .unwrap();

        assert!(out.contains("manifest updated"), "{out}");
        assert!(!skills.join("demo.skill").exists());
        let manifest = std::fs::read_to_string(skills.join("demo/SKILL.md")).unwrap();
        assert!(manifest.contains("new prompt"), "{manifest}");
        assert!(manifest.contains("name: \"demo\""), "{manifest}");
        assert_eq!(
            std::fs::read_to_string(skills.join("demo/resources/keep.txt")).unwrap(),
            "keep me"
        );

        let err = execute_save_skill(&serde_json::json!({
            "name": "demo",
            "prompt": "again",
            "overwrite": false
        }))
        .unwrap_err();
        assert!(err.contains("overwrite=false"), "{err}");

        match old_cfg {
            Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn delete_skill_entries_removes_file_and_package() {
        let base = std::env::temp_dir().join(format!(
            "skill-delete-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let file_skill = base.join("demo.skill");
        std::fs::write(&file_skill, "demo").unwrap();
        let package_skill = base.join("pack");
        std::fs::create_dir_all(package_skill.join("sub")).unwrap();
        std::fs::write(package_skill.join("SKILL.md"), "pack").unwrap();

        let removed = delete_skill_entries(&base, "demo").unwrap();
        assert_eq!(removed.len(), 1);
        assert!(!file_skill.exists());
        let removed = delete_skill_entries(&base, "pack").unwrap();
        assert_eq!(removed.len(), 1);
        assert!(!package_skill.exists());
        let err = delete_skill_entries(&base, "missing").unwrap_err();
        assert!(err.contains("No skill named"));
        let err = delete_skill_entries(&base, "   ").unwrap_err();
        assert!(err.contains("non-empty"));
        let _ = std::fs::remove_dir_all(&base);
    }

    fn resource_test_dir(label: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "rust_tools_skill_resource_{label}_{}_{}",
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn resource_directory_import_copies_tree_and_skips_vcs_metadata() {
        let src = resource_test_dir("dir_src");
        let scripts = src.join("scripts");
        std::fs::create_dir_all(scripts.join("lib")).unwrap();
        std::fs::write(scripts.join("lib/util.py"), "print('x')\n").unwrap();
        std::fs::write(scripts.join("run.sh"), "#!/bin/sh\necho hi\n").unwrap();
        std::fs::create_dir_all(scripts.join(".git")).unwrap();
        std::fs::write(scripts.join(".git/config"), "ignored").unwrap();

        let resources = parse_resources_arg(
            &serde_json::json!({
                "resources": [{
                    "path": "scripts",
                    "from": scripts.to_string_lossy()
                }]
            }),
            "",
        )
        .unwrap();
        let plan = plan_resource_files(&resources).unwrap();
        assert_eq!(plan.len(), 2);

        let out = resource_test_dir("dir_out");
        write_planned_resource_files(&out, &plan).unwrap();
        assert_eq!(
            std::fs::read_to_string(out.join("scripts/run.sh")).unwrap(),
            "#!/bin/sh\necho hi\n"
        );
        assert!(out.join("scripts/lib/util.py").exists());
        assert!(!out.join("scripts/.git").exists());

        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn resource_import_is_binary_safe_and_base64_decodes() {
        let src = resource_test_dir("bin_src");
        let binary = [0u8, 159, 146, 150, 255];
        std::fs::write(src.join("archive.bin"), binary).unwrap();

        let resources = parse_resources_arg(
            &serde_json::json!({
                "resources": [
                    {"path": "assets/archive.bin", "from": src.join("archive.bin").to_string_lossy()},
                    {"path": "assets/encoded.bin", "content_base64": "aGk="}
                ]
            }),
            "",
        )
        .unwrap();
        let plan = plan_resource_files(&resources).unwrap();
        let out = resource_test_dir("bin_out");
        write_planned_resource_files(&out, &plan).unwrap();
        assert_eq!(
            std::fs::read(out.join("assets/archive.bin")).unwrap(),
            binary
        );
        assert_eq!(
            std::fs::read(out.join("assets/encoded.bin")).unwrap(),
            b"hi"
        );

        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn resource_rejects_reserved_and_duplicate_paths() {
        let err = parse_resources_arg(
            &serde_json::json!({"resources": [{"path": "SKILL.md", "content": "x"}]}),
            "",
        )
        .unwrap_err();
        assert!(err.contains("reserved"), "{err}");

        let err = parse_resources_arg(
            &serde_json::json!({
                "resources": [{"path": "subskills/child/SKILL.md", "content": "x"}]
            }),
            "",
        )
        .unwrap_err();
        assert!(err.contains("reserved"), "{err}");

        let err = parse_resources_arg(
            &serde_json::json!({"resources": [
                {"path": "a.md", "content": "x"},
                {"path": "A.MD", "content": "y"}
            ]}),
            "",
        )
        .unwrap_err();
        assert!(err.contains("Duplicate"), "{err}");

        let err = parse_resources_arg(
            &serde_json::json!({"resources": [
                {"path": "scripts", "content": "x"},
                {"path": "scripts/run.sh", "content": "y"}
            ]}),
            "",
        )
        .unwrap_err();
        assert!(err.contains("overlaps"), "{err}");
    }

    #[test]
    fn resource_requires_exactly_one_source() {
        let err = parse_resources_arg(
            &serde_json::json!({
                "resources": [{"path": "a.md", "content": "x", "from": "whatever"}]
            }),
            "",
        )
        .unwrap_err();
        assert!(err.contains("exactly one"), "{err}");

        let err = parse_resources_arg(&serde_json::json!({"resources": [{"path": "a.md"}]}), "")
            .unwrap_err();
        assert!(err.contains("exactly one"), "{err}");
    }

    #[test]
    fn resource_rejects_absolute_destinations() {
        for path in ["/etc/notes.md", "//server/share/notes.md", "C:\\notes.md"] {
            let err = parse_resources_arg(
                &serde_json::json!({
                    "resources": [{"path": path, "content": "x"}]
                }),
                "",
            )
            .unwrap_err();
            assert!(
                err.contains("must be relative"),
                "absolute destination {path} was not rejected: {err}"
            );
        }
    }

    #[test]
    fn validate_skill_reports_sanitized_subskill_collision() {
        let (errors, _) = validate_skill_args(&serde_json::json!({
            "name": "parent",
            "description": "parent skill",
            "prompt": "Goal: parent. Inputs: none. Steps: delegate. Done when the child finishes.",
            "subskills": [
                {"name": "a b", "prompt": "Goal: child one. Done when done."},
                {"name": "a-b", "prompt": "Goal: child two. Done when done."}
            ]
        }));
        assert!(
            errors.iter().any(|e| e.contains("sanitized collision")),
            "expected a sanitized-directory collision error, got {errors:?}"
        );
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
