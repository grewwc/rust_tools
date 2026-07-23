// =============================================================================
// AIOS Skills - Agent Capabilities System
// =============================================================================
// Skills are the equivalent of commands in a CLI - they provide capabilities
// that AI agents can invoke to perform specific tasks.
//
// Similar to agents:
//   - Loaded from .skill files or package directories/zips (with YAML front-matter)
//   - Have name, description, prompt, system_prompt
//   - Support tools, tool_groups, mcp_servers
//
// Key differences from agents:
//   - Skills are invoked via tool calls, not conversation
//   - Lower priority in skill matching wins ties
//   - Don't have model/temperature (use agent's settings)
//   - No routing tags or disabled/hidden flags
//
// User skills are searched from the configured skills directory.
// =============================================================================

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use rust_tools::cw::SkipMap;

use crate::ai::config_schema::AiConfig;
use crate::commonw::{configw, utils::expanduser};

const BUILTIN_SKILLS: &[(&str, &str)] = &[];

fn default_skill_version() -> String {
    "1.0.0".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SkillDiscoveryLevel {
    Builtin = 0,
    External = 1,
    User = 2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SkillManifest {
    pub(super) name: String,
    #[serde(default = "default_skill_version")]
    pub(super) version: String,
    /// 技能描述：用于模型路由选择的核心字段
    #[serde(default)]
    pub(super) description: String,
    #[serde(default)]
    pub(super) author: Option<String>,
    #[serde(default)]
    pub(super) tools: Vec<String>,
    #[serde(default)]
    pub(super) tool_groups: Vec<String>,
    #[serde(default)]
    pub(super) mcp_servers: Vec<String>,
    #[serde(default)]
    pub(super) skip_recall: bool,
    #[serde(default)]
    pub(super) disable_builtin_tools: bool,
    #[serde(default)]
    pub(super) disable_mcp_tools: bool,
    #[serde(default)]
    pub(super) prompt: String,
    #[serde(default)]
    pub(super) system_prompt: Option<String>,
    /// 优先级：仅在多个技能匹配度相近时使用
    #[serde(default)]
    pub(super) priority: i32,
    #[serde(default)]
    pub(super) excludes: Vec<String>,
    #[serde(skip)]
    pub(super) source_path: Option<String>,
    #[serde(skip)]
    pub(super) resource_path: Option<String>,
}

impl SkillManifest {
    /// 构建系统提示词
    pub(super) fn build_system_prompt(&self) -> String {
        let mut prompt = self.system_prompt.clone().unwrap_or_default();

        if !self.prompt.is_empty() {
            if !prompt.is_empty() {
                prompt.push_str("\n\n");
            }
            // Avoid extra clone when prompt is already allocated
            prompt.push_str(self.prompt.as_str());
        }

        prompt
    }

    pub(super) fn routing_source_hash(&self) -> String {
        let payload = serde_json::json!({
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tools": self.tools,
            "tool_groups": self.tool_groups,
            "mcp_servers": self.mcp_servers,
            "skip_recall": self.skip_recall,
            "disable_builtin_tools": self.disable_builtin_tools,
            "disable_mcp_tools": self.disable_mcp_tools,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "priority": self.priority,
            "source_path": self.source_path,
            "resource_path": self.resource_path,
        });
        let mut hasher = Sha256::new();
        hasher.update(payload.to_string().as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

pub(super) fn load_all_skills() -> Vec<SkillManifest> {
    let dir = skills_dir();
    let _ = ensure_seeded_skills_dir(&dir);
    let mut by_name: Box<SkipMap<String, (SkillDiscoveryLevel, SkillManifest)>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);

    for (filename, content) in BUILTIN_SKILLS {
        if let Ok(mut skill) = parse_skill_front_matter(content) {
            skill.source_path = Some(format!("builtin:{filename}"));
            by_name.insert(skill.name.clone(), (SkillDiscoveryLevel::Builtin, skill));
        }
    }

    for skill in load_external_skills() {
        if should_insert_skill(&by_name, &skill.name, SkillDiscoveryLevel::External) {
            by_name.insert(skill.name.clone(), (SkillDiscoveryLevel::External, skill));
        }
    }

    for skill in load_skills_from_dir(&dir) {
        if should_insert_skill(&by_name, &skill.name, SkillDiscoveryLevel::User) {
            by_name.insert(skill.name.clone(), (SkillDiscoveryLevel::User, skill));
        }
    }

    let mut out = (&*by_name)
        .into_iter()
        .map(|(_, (_, v))| v.clone())
        .collect::<Vec<_>>();
    out.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.name.cmp(&b.name)));
    out
}

fn should_insert_skill(
    by_name: &SkipMap<String, (SkillDiscoveryLevel, SkillManifest)>,
    name: &str,
    incoming_level: SkillDiscoveryLevel,
) -> bool {
    match by_name.get_ref(&name.to_string()) {
        Some((SkillDiscoveryLevel::Builtin, _)) => false,
        Some((level, _)) => incoming_level > *level,
        None => true,
    }
}

pub(super) fn skills_dir() -> PathBuf {
    let cfg = configw::get_all_config();
    let raw = cfg.get_opt(AiConfig::SKILLS_DIR).unwrap_or_default();
    let path = if raw.trim().is_empty() {
        "~/.config/rust_tools/skills".to_string()
    } else {
        raw
    };
    PathBuf::from(expanduser(&path).as_ref())
}

/// 返回需要监听的根目录。用户技能目录始终监听；Trae 外部技能都位于同一根目录下，
/// 监听该根目录可以发现运行期间新增的外部包。
pub(super) fn skill_watch_roots() -> Vec<PathBuf> {
    let user_skills_dir = skills_dir();
    let external_root = PathBuf::from(expanduser("~/.trae-cn").as_ref());
    if external_root.is_dir() && external_root != user_skills_dir {
        vec![user_skills_dir, external_root]
    } else {
        vec![user_skills_dir]
    }
}

fn looks_like_front_matter_skill(content: &str) -> bool {
    content.lines().next().is_some_and(|l| l.trim() == "---")
}

fn file_looks_like_front_matter_skill(path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    looks_like_front_matter_skill(&content)
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

fn parse_bool_value(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Some(true),
        "false" | "no" | "off" | "0" => Some(false),
        _ => None,
    }
}

fn parse_skill_front_matter(content: &str) -> Result<SkillManifest, String> {
    let mut lines = content.lines();
    let Some(first) = lines.next() else {
        return Err("empty skill file".to_string());
    };
    if first.trim() != "---" {
        return Err("missing front matter start".to_string());
    }

    let mut name: Option<String> = None;
    let mut version: Option<String> = None;
    let mut description: Option<String> = None;
    let mut author: Option<String> = None;
    let mut tools: Vec<String> = Vec::new();
    let mut tool_groups: Vec<String> = Vec::new();
    let mut mcp_servers: Vec<String> = Vec::new();
    let mut excludes: Vec<String> = Vec::new();
    let mut skip_recall = false;
    let mut disable_builtin_tools = false;
    let mut disable_mcp_tools = false;
    let mut system_prompt: Option<String> = None;
    let mut priority: i32 = 0;

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
                    "excludes" => excludes.push(v),
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
                "version" => version = Some(unquoted.to_string()),
                "description" => description = Some(unquoted.to_string()),
                "author" => author = Some(unquoted.to_string()),
                "skip_recall" => skip_recall = parse_bool_value(unquoted).unwrap_or(false),
                "disable_builtin_tools" => {
                    disable_builtin_tools = parse_bool_value(unquoted).unwrap_or(false)
                }
                "disable_mcp_tools" => {
                    disable_mcp_tools = parse_bool_value(unquoted).unwrap_or(false)
                }
                "system_prompt" => system_prompt = Some(unquoted.to_string()),
                "priority" => {
                    priority = unquoted.parse::<i32>().unwrap_or(0);
                }
                "tools" => tools = parse_list_value(unquoted),
                "tool_groups" => tool_groups = parse_list_value(unquoted),
                "mcp_servers" => mcp_servers = parse_list_value(unquoted),
                "excludes" => excludes = parse_list_value(unquoted),
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

    Ok(SkillManifest {
        name,
        version: version.unwrap_or_else(default_skill_version),
        description: description.unwrap_or_default(),
        author: author.filter(|s| !s.trim().is_empty()),
        tools,
        tool_groups,
        mcp_servers,
        skip_recall,
        disable_builtin_tools,
        disable_mcp_tools,
        prompt: body.trim().to_string(),
        system_prompt: system_prompt.filter(|s| !s.trim().is_empty()),
        priority,
        excludes,
        source_path: None,
        resource_path: None,
    })
}

fn parse_skill_front_matter_with_path(content: &str, path: &Path) -> Result<SkillManifest, String> {
    let mut skill = parse_skill_front_matter(content)?;
    skill.source_path = Some(path.display().to_string());
    Ok(skill)
}

fn parse_skill_package_manifest(
    manifest_path: &Path,
    resource_root: &Path,
    source_label: Option<String>,
) -> Result<SkillManifest, String> {
    let content = fs::read_to_string(manifest_path)
        .map_err(|e| format!("failed to read skill package manifest: {e}"))?;
    let mut skill = parse_skill_front_matter(&content)?;
    skill.source_path = Some(source_label.unwrap_or_else(|| manifest_path.display().to_string()));
    skill.resource_path = Some(resource_root.display().to_string());
    Ok(skill)
}

fn ensure_seeded_skills_dir(dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("failed to create skills dir: {e}"))?;
    Ok(())
}

fn load_external_skills() -> Vec<SkillManifest> {
    discover_external_skill_dirs()
        .into_iter()
        .flat_map(|dir| load_skills_from_package_dir(&dir))
        .collect()
}

fn discover_external_skill_dirs() -> Vec<PathBuf> {
    let mut dirs = BTreeSet::new();
    for pattern in external_skill_glob_patterns() {
        let expanded = expanduser(pattern);
        let Ok(paths) = rust_tools::terminalw::glob_paths(&expanded, ".") else {
            continue;
        };
        for entry in &paths {
            let p = PathBuf::from(entry);
            if p.is_dir() {
                dirs.insert(p);
            }
        }
    }
    dirs.into_iter().collect()
}

fn external_skill_glob_patterns() -> &'static [&'static str] {
    &[
        "~/.trae-cn/builtin_skills/*",
        "~/.trae-cn/skills/*",
        "~/.trae-cn/builtin/**/skills/*",
        "~/.trae-cn/extensions/*/skills/*",
    ]
}

fn load_skills_from_dir(dir: &Path) -> Vec<SkillManifest> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if is_ignored_package_entry_name(name) {
            continue;
        }

        if path.is_dir() {
            out.extend(load_skills_from_package_dir(&path));
            continue;
        }

        if !path.is_file() {
            continue;
        }

        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"))
        {
            out.extend(load_skills_from_zip_package(&path, dir));
            continue;
        }

        if let Some(skill) = load_skill_from_file(&path) {
            out.push(skill);
        }
    }
    out
}

fn load_skill_from_file(path: &Path) -> Option<SkillManifest> {
    let content = fs::read_to_string(path).ok()?;
    if !looks_like_front_matter_skill(&content) {
        return None;
    }
    parse_skill_front_matter_with_path(&content, path).ok()
}

fn load_skills_from_package_dir(dir: &Path) -> Vec<SkillManifest> {
    collect_skill_packages(dir)
        .into_iter()
        .filter_map(|(resource_root, manifest_path)| {
            parse_skill_package_manifest(&manifest_path, &resource_root, None).ok()
        })
        .collect()
}

fn load_skills_from_zip_package(zip_path: &Path, skills_dir: &Path) -> Vec<SkillManifest> {
    let Ok(extract_root) = extracted_zip_package_root(zip_path, skills_dir) else {
        return Vec::new();
    };
    collect_skill_packages(&extract_root)
        .into_iter()
        .filter_map(|(resource_root, manifest_path)| {
            let relative_manifest = manifest_path
                .strip_prefix(&extract_root)
                .unwrap_or(manifest_path.as_path());
            let source_label = format!("{}!{}", zip_path.display(), relative_manifest.display());
            parse_skill_package_manifest(&manifest_path, &resource_root, Some(source_label)).ok()
        })
        .collect()
}

fn is_ignored_package_entry_name(name: &str) -> bool {
    name.starts_with('.') || name == "__MACOSX"
}

/// 集合下钻的最大目录深度。feishu 集合布局解压后最深为 `feishu/skills/<pkg>`（3 层），
/// 取 4 留出余量，同时避免在异常深的目录树上无限递归。
const MAX_SKILL_PACKAGE_DEPTH: usize = 4;

/// 收集 `root` 下的所有 skill 包，返回每个包的 `(resource_root, manifest_path)`。
///
/// - 若 `root` 自身就是一个包（直接含 manifest），只返回它本身，且**不再向下递归**：
///   单包目录 / 单包 zip（如 argos-tools）保持原有行为，包内 `references/*.skill`
///   等资源文件不会被误判为独立 skill。
/// - 否则把 `root` 当作"集合"（如 feishu，其布局为 `feishu/skills/<pkg>/SKILL.md`），
///   逐层下钻收集所有**最上层**的包；命中某个目录是包后即停止深入该目录。
///
/// 磁盘目录集合与解压后的 zip 集合通过同一条逻辑处理，无需对 `skills/` 之类的容器
/// 名做硬编码判断。
fn collect_skill_packages(root: &Path) -> Vec<(PathBuf, PathBuf)> {
    // 单包短路：root 直接是一个包，按原语义返回单个，不下钻。
    if let Some(manifest) = find_skill_manifest_in_package_dir(root) {
        return vec![(root.to_path_buf(), manifest)];
    }
    let mut out = Vec::new();
    collect_skill_packages_recursive(root, MAX_SKILL_PACKAGE_DEPTH, &mut out);
    rust_tools::sortw::stable_sort_by(&mut out, |a, b| a.0.cmp(&b.0));
    out
}

fn collect_skill_packages_recursive(
    dir: &Path,
    depth_budget: usize,
    out: &mut Vec<(PathBuf, PathBuf)>,
) {
    if depth_budget == 0 {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    let mut children = rd
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let name = path.file_name()?.to_str()?;
            if is_ignored_package_entry_name(name) {
                return None;
            }
            Some(path)
        })
        .collect::<Vec<_>>();
    rust_tools::sortw::stable_sort_by(&mut children, |a, b| a.cmp(b));
    for child in children {
        if let Some(manifest) = find_skill_manifest_in_package_dir(&child) {
            // child 本身是一个包：收集并停止深入（包内是资源，不再当 skill 扫）。
            out.push((child, manifest));
        } else {
            collect_skill_packages_recursive(&child, depth_budget - 1, out);
        }
    }
}

fn find_skill_manifest_in_package_dir(dir: &Path) -> Option<PathBuf> {
    let mut files = fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let name = path.file_name()?.to_str()?;
            if is_ignored_package_entry_name(name) {
                return None;
            }
            Some(path)
        })
        .collect::<Vec<_>>();
    rust_tools::sortw::stable_sort_by(&mut files, |a, b| a.cmp(b));

    if let Some(path) = files.iter().find(|path| {
        path.file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("SKILL.md"))
    }) {
        return Some(path.clone());
    }

    let package_file = dir
        .file_name()
        .and_then(|s| s.to_str())
        .map(|name| format!("{name}.skill"));
    if let Some(package_file) = package_file
        && let Some(path) = files.iter().find(|path| {
            path.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case(&package_file))
        })
    {
        return Some(path.clone());
    }

    files.into_iter().find(|path| {
        path.extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("skill"))
    })
}

fn extracted_zip_package_root(zip_path: &Path, skills_dir: &Path) -> Result<PathBuf, String> {
    let stem = zip_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("skill");
    let digest = file_sha256_hex(zip_path)?;
    let short_digest = digest.chars().take(16).collect::<String>();
    let cache_parent = skills_dir.join(".cache").join("skill-packages");
    let extract_root = cache_parent.join(format!("{stem}-{short_digest}"));
    if extract_root.is_dir() {
        return Ok(extract_root);
    }

    fs::create_dir_all(&cache_parent)
        .map_err(|e| format!("failed to create skill package cache dir: {e}"))?;
    let temp_root = cache_parent.join(format!(".tmp-{stem}-{short_digest}-{}", std::process::id()));
    if temp_root.exists() {
        let _ = fs::remove_dir_all(&temp_root);
    }
    fs::create_dir_all(&temp_root)
        .map_err(|e| format!("failed to create skill package temp dir: {e}"))?;

    let result = extract_zip_to_dir(zip_path, &temp_root).and_then(|()| {
        if extract_root.exists() {
            let _ = fs::remove_dir_all(&temp_root);
            return Ok(extract_root.clone());
        }
        fs::rename(&temp_root, &extract_root)
            .map_err(|e| format!("failed to publish extracted skill package: {e}"))?;
        Ok(extract_root.clone())
    });

    if result.is_err() {
        let _ = fs::remove_dir_all(&temp_root);
    }
    result
}

fn file_sha256_hex(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read skill zip: {e}"))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_zip_to_dir(zip_path: &Path, target_dir: &Path) -> Result<(), String> {
    let file = fs::File::open(zip_path).map_err(|e| format!("failed to open skill zip: {e}"))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("failed to read skill zip: {e}"))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("failed to read skill zip entry: {e}"))?;
        let Some(enclosed_path) = entry.enclosed_name() else {
            continue;
        };
        let out_path = target_dir.join(enclosed_path);
        if entry.is_dir() {
            fs::create_dir_all(&out_path)
                .map_err(|e| format!("failed to create skill package dir: {e}"))?;
            continue;
        }

        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create skill package parent dir: {e}"))?;
        }
        let mut out_file = fs::File::create(&out_path)
            .map_err(|e| format!("failed to create skill package file: {e}"))?;
        std::io::copy(&mut entry, &mut out_file)
            .map_err(|e| format!("failed to extract skill package file: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "skills_tests.rs"]
mod tests;
