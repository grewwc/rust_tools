use std::fs;
use std::path::PathBuf;
use std::sync::{LazyLock, RwLock};

use rust_tools::commonw::{FastMap, FastSet};
use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::ai::skills::SkillManifest;
use crate::ai::tools::common::{ToolRegistration, ToolSpec};

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
         Ask the user this question in your final response, then stop. The active skill will be restored only for the user's immediately following normal message."
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

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_skill_resource",
        description: "",
        execute: execute_read_skill_resource,
    }
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

fn parse_resources_arg(args: &Value) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::new();
    // Generic field `resources: [{path, content}]`
    if let Some(arr) = args.get("resources").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(obj) = item.as_object() {
                let path = obj
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                let content = obj.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if path.is_empty() {
                    return Err("resources[]: each entry requires non-empty 'path'".to_string());
                }
                validate_resource_relative_path(path)?;
                out.push((path.to_string(), content.to_string()));
            } else {
                return Err(
                    "resources[] entries must be objects with 'path' and 'content'".to_string(),
                );
            }
        }
    }
    // Category shortcut fields: references / examples / scripts, same-shaped arrays each; a path without a directory gets the prefix auto-added
    for (key, prefix) in [
        ("references", "references"),
        ("examples", "examples"),
        ("scripts", "scripts"),
    ] {
        if let Some(arr) = args.get(key).and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(obj) = item.as_object() {
                    let raw_path = obj
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim();
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
                    return Err(format!(
                        "{key}[] entries must be objects with 'path' and 'content'"
                    ));
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

fn write_resource_files(
    base_dir: &std::path::Path,
    resources: &[(String, String)],
) -> Result<(), String> {
    write_resource_files_with_overwrite(base_dir, resources, true)
}

fn write_resource_files_with_overwrite(
    base_dir: &std::path::Path,
    resources: &[(String, String)],
    overwrite: bool,
) -> Result<(), String> {
    // Ensure base_dir exists and is canonicalizable (upfront validation)
    fs::create_dir_all(base_dir)
        .map_err(|e| format!("Failed to create dir {}: {e}", base_dir.display()))?;
    let canonical_base = fs::canonicalize(base_dir).unwrap_or_else(|_| base_dir.to_path_buf());
    for (rel, content) in resources {
        let rel_clean = rel.trim().trim_matches('/').replace('\\', "/");
        // Reuse the existing validation (covers .. and absolute paths)
        let validated = validate_resource_relative_path(&rel_clean)
            .map_err(|e| format!("Invalid resource path '{rel}': {e}"))?;
        let target = base_dir.join(&validated);
        // Pre-check for path traversal before creating parent dirs: the canonical parent of target must be inside base
        if let Some(parent) = target.parent() {
            // The parent may not exist yet; validate the "expected" path prefix at character level for traversal
            // Component-level check: `..` in validated was already blocked above; do the canonical check here
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create dir {}: {e}", parent.display()))?;
            // Canonical check on the already-created parent
            if let Ok(canonical_parent) = fs::canonicalize(parent) {
                if !canonical_parent.starts_with(&canonical_base) {
                    return Err(format!(
                        "Resource path escapes package directory: {validated}"
                    ));
                }
            }
        }
        if target.exists() && !overwrite {
            return Err(format!(
                "Resource already exists and overwrite=false: {}",
                target.display()
            ));
        }
        fs::write(&target, content)
            .map_err(|e| format!("Failed to write resource {validated}: {e}"))?;
        // Post-write re-validation (handles symlink races)
        if let (Ok(cb), Ok(ct)) = (fs::canonicalize(&canonical_base), fs::canonicalize(&target)) {
            if !ct.starts_with(&cb) {
                let _ = fs::remove_file(&target);
                return Err(format!(
                    "Resource path escapes package directory: {validated}"
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
    let resources = parse_resources_arg(args)?;
    let subskills = parse_subskills_arg(args)?;
    let has_package_features = !resources.is_empty() || !subskills.is_empty();

    let dir = resolve_configured_skills_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create skills dir: {e}"))?;
    let overwrite = args["overwrite"].as_bool().unwrap_or(true);

    if !has_package_features {
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
    let package_dir_name = safe_skill_dir_name(&name);
    let package_dir = dir.join(&package_dir_name);
    let legacy_file = dir.join(safe_skill_file_name(&name));
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
        if !resources.is_empty() {
            write_resource_files(&tmp_dir, &resources)?;
        }
        let mut created_subskills: Vec<String> = Vec::new();
        for sub in &subskills {
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
            if let Some(arr) = sub.get("resources").and_then(|v| v.as_array()) {
                let mut sub_resources: Vec<(String, String)> = Vec::new();
                for item in arr {
                    if let Some(obj) = item.as_object() {
                        let p = obj
                            .get("path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        let c = obj
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        validate_resource_relative_path(&p)?;
                        sub_resources.push((p, c));
                    }
                }
                if !sub_resources.is_empty() {
                    write_resource_files(&sub_dir, &sub_resources)?;
                }
            }
            for (key, prefix) in [
                ("references", "references"),
                ("examples", "examples"),
                ("scripts", "scripts"),
            ] {
                if let Some(arr) = sub.get(key).and_then(|v| v.as_array()) {
                    let mut cat_resources: Vec<(String, String)> = Vec::new();
                    for item in arr {
                        if let Some(obj) = item.as_object() {
                            let raw = obj
                                .get("path")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .trim()
                                .to_string();
                            let c = obj
                                .get("content")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let path = if raw.contains('/') {
                                raw
                            } else {
                                format!("{prefix}/{raw}")
                            };
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
    use super::PendingSkillAction;
    use super::{
        build_skill_file_content, clear_pending_user_input_request, execute_activate_skill,
        execute_deactivate_skill, execute_load_skill, execute_request_user_input,
        render_loaded_skill, render_skill_catalog, skill_resource_read_limit,
        take_pending_skill_action, take_pending_user_input_request, utf8_prefix_at_most,
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
        let out = render_loaded_skill(&skill);
        assert!(out.contains("# Skill: demo"));
        assert!(out.contains("description: demo description"));
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
