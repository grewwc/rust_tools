// =============================================================================
// AIOS Skills - Agent Capabilities System
// =============================================================================
// Skills are the equivalent of commands in a CLI - they provide capabilities
// that AI agents can invoke to perform specific tasks.
// 
// Similar to agents:
//   - Loaded from .skill files (with YAML front-matter)
//   - Have name, description, prompt, system_prompt
//   - Support tools, tool_groups, mcp_servers
// 
// Key differences from agents:
//   - Skills are invoked via tool calls, not conversation
//   - Lower priority in skill matching wins ties
//   - Don't have model/temperature (use agent's settings)
//   - No routing tags or disabled/hidden flags
// 
// Skill files are searched in same directories as agents.
// =============================================================================

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

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
    #[serde(skip)]
    pub(super) source_path: Option<String>,
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
        source_path: None,
    })
}

fn parse_skill_front_matter_with_path(content: &str, path: &Path) -> Result<SkillManifest, String> {
    let mut skill = parse_skill_front_matter(content)?;
    skill.source_path = Some(path.display().to_string());
    Ok(skill)
}

fn ensure_seeded_skills_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("failed to create skills dir: {e}"))?;
    Ok(())
}

fn load_skills_from_dir(dir: &Path) -> Vec<SkillManifest> {
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
        if !looks_like_front_matter_skill(&content) {
            continue;
        }
        if let Ok(skill) = parse_skill_front_matter_with_path(&content, &path) {
            out.push(skill);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_tools::cw::SkipSet;

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
}
