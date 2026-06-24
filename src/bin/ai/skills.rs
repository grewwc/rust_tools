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
    fs,
    path::{Path, PathBuf},
};

use rust_tools::cw::SkipMap;

use crate::ai::config_schema::AiConfig;
use crate::commonw::{configw, utils::expanduser};

const BUILTIN_SKILLS: &[(&str, &str)] = &[
    (
        "debugger.skill",
        include_str!("builtin_skills/debugger.skill"),
    ),
    (
        "code-review.skill",
        include_str!("builtin_skills/code-review.skill"),
    ),
    (
        "refactor.skill",
        include_str!("builtin_skills/refactor.skill"),
    ),
];

fn default_skill_version() -> String {
    "1.0.0".to_string()
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
    pub(super) triggers: Vec<String>,
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
            "triggers": self.triggers,
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
    let mut by_name: Box<SkipMap<String, SkillManifest>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);

    for (filename, content) in BUILTIN_SKILLS {
        if let Ok(mut skill) = parse_skill_front_matter(content) {
            skill.source_path = Some(format!("builtin:{filename}"));
            by_name.insert(skill.name.clone(), skill);
        }
    }

    for skill in load_skills_from_dir(&dir) {
        if let Some(existing) = by_name.get_ref(&skill.name)
            && existing
                .source_path
                .as_deref()
                .is_some_and(|p| p.starts_with("builtin:"))
        {
            continue;
        }
        by_name.insert(skill.name.clone(), skill);
    }

    let mut out = (&*by_name)
        .into_iter()
        .map(|(_, v)| v.clone())
        .collect::<Vec<_>>();
    out.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.name.cmp(&b.name)));
    out
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
    let mut triggers: Vec<String> = Vec::new();
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
                    "triggers" => triggers.push(v),
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
                "triggers" => triggers = parse_list_value(unquoted),
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
        triggers,
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
            if let Some(skill) = load_skill_from_package_dir(&path) {
                out.push(skill);
            }
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
            if let Some(skill) = load_skill_from_zip_package(&path, dir) {
                out.push(skill);
            }
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

fn load_skill_from_package_dir(dir: &Path) -> Option<SkillManifest> {
    let manifest_path = find_skill_manifest_in_package_dir(dir)?;
    parse_skill_package_manifest(&manifest_path, dir, None).ok()
}

fn load_skill_from_zip_package(zip_path: &Path, skills_dir: &Path) -> Option<SkillManifest> {
    let extract_root = extracted_zip_package_root(zip_path, skills_dir).ok()?;
    let (resource_root, manifest_path) = find_skill_package_root(&extract_root)?;
    let relative_manifest = manifest_path
        .strip_prefix(&extract_root)
        .unwrap_or(manifest_path.as_path());
    let source_label = format!("{}!{}", zip_path.display(), relative_manifest.display());
    parse_skill_package_manifest(&manifest_path, &resource_root, Some(source_label)).ok()
}

fn is_ignored_package_entry_name(name: &str) -> bool {
    name.starts_with('.') || name == "__MACOSX"
}

fn find_skill_package_root(root: &Path) -> Option<(PathBuf, PathBuf)> {
    if let Some(manifest) = find_skill_manifest_in_package_dir(root) {
        return Some((root.to_path_buf(), manifest));
    }

    let mut child_matches = fs::read_dir(root)
        .ok()?
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
            find_skill_manifest_in_package_dir(&path).map(|manifest| (path, manifest))
        })
        .collect::<Vec<_>>();
    rust_tools::sortw::stable_sort_by(&mut child_matches, |a, b| a.0.cmp(&b.0));
    if child_matches.len() == 1 {
        child_matches.pop()
    } else {
        None
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
mod tests {
    use super::*;
    use rust_tools::cw::SkipSet;
    use std::io::Write;

    fn write_zip_package(path: &Path, entries: &[(&str, &str)]) {
        let file = fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, content) in entries {
            zip.start_file(name, options).unwrap();
            zip.write_all(content.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
    }

    #[test]
    fn seed_skills_dir_creates_dir_but_does_not_copy_builtins() {
        let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
        ensure_seeded_skills_dir(&dir).unwrap();
        let skills = load_skills_from_dir(&dir);
        assert_eq!(skills.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_all_skills_includes_builtin_even_when_user_has_custom_skill() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let home = std::env::temp_dir()
            .join(format!("rust-tools-home-{}", uuid::Uuid::new_v4()))
            .display()
            .to_string();
        let old_home = std::env::var("HOME").ok();
        unsafe {
            std::env::set_var("HOME", &home);
        }

        let dir = skills_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("custom.skill"),
            r#"---
name: custom-skill
description: custom
priority: 1
---

custom"#,
        )
        .unwrap();

        std::fs::write(
            dir.join("code-review.skill"),
            r#"---
name: code-review
description: override
priority: 999
---

override"#,
        )
        .unwrap();

        let skills = load_all_skills();
        let mut names = SkipSet::new(8);
        for s in &skills {
            names.insert(s.name.clone());
        }
        let debugger = "debugger".to_string();
        let code_review = "code-review".to_string();
        let refactor = "refactor".to_string();
        let custom = "custom-skill".to_string();
        assert!(names.contains(&debugger));
        assert!(names.contains(&code_review));
        assert!(names.contains(&refactor));
        assert!(names.contains(&custom));
        let code_review = skills.iter().find(|s| s.name == "code-review").unwrap();
        assert!(
            code_review
                .source_path
                .as_deref()
                .is_some_and(|p| p.starts_with("builtin:"))
        );

        match old_home {
            Some(v) => unsafe {
                std::env::set_var("HOME", v);
            },
            None => unsafe {
                std::env::remove_var("HOME");
            },
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn test_simple_format_parsing() {
        let content = r#"---
name: test-skill
description: test skill for helping users
tools:
  - read_file
  - write_file
priority: 50
---

test prompt"#;

        let skill = parse_skill_front_matter(content).unwrap();
        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.description, "test skill for helping users");
        assert_eq!(skill.tools, vec!["read_file", "write_file"]);
        assert_eq!(skill.priority, 50);
    }

    #[test]
    fn load_skills_from_dir_supports_package_directory() {
        let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
        let package_dir = dir.join("superpower");
        std::fs::create_dir_all(package_dir.join("references")).unwrap();
        std::fs::write(
            package_dir.join("SKILL.md"),
            r#"---
name: superpower
description: packaged skill
---

Read references/guide.md before acting."#,
        )
        .unwrap();
        std::fs::write(package_dir.join("references").join("guide.md"), "resource").unwrap();

        let skills = load_skills_from_dir(&dir);
        let skill = skills.iter().find(|s| s.name == "superpower").unwrap();
        assert_eq!(skill.description, "packaged skill");
        assert_eq!(
            skill.resource_path.as_deref(),
            Some(package_dir.display().to_string().as_str())
        );
        assert!(skill.source_path.as_deref().unwrap().ends_with("SKILL.md"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_skills_from_dir_supports_zip_package_with_wrapped_root() {
        let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let zip_path = dir.join("superpower.zip");
        write_zip_package(
            &zip_path,
            &[
                (
                    "superpower/SKILL.md",
                    r#"---
name: superpower
description: zipped skill
priority: 9
---

Use bundled references."#,
                ),
                ("superpower/references/guide.md", "resource"),
            ],
        );

        let skills = load_skills_from_dir(&dir);
        let skill = skills.iter().find(|s| s.name == "superpower").unwrap();
        let resource_path = PathBuf::from(skill.resource_path.as_deref().unwrap());
        assert_eq!(skill.description, "zipped skill");
        assert_eq!(skill.priority, 9);
        assert!(skill.source_path.as_deref().unwrap().contains(".zip!"));
        assert!(resource_path.join("references").join("guide.md").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
