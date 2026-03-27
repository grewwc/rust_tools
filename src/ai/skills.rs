use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const BUILTIN_SKILLS: &[(&str, &str)] = &[
    (
        "openclaw.skill",
        include_str!("builtin_skills/openclaw.skill"),
    ),
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
    pub(super) prompt: String,
    #[serde(default)]
    pub(super) system_prompt: Option<String>,
    #[serde(default)]
    pub(super) examples: Vec<SkillExample>,
    #[serde(default)]
    pub(super) triggers: Vec<String>,
    #[serde(default)]
    pub(super) priority: i32,
    #[serde(skip)]
    pub(super) source_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SkillExample {
    pub(super) user: String,
    pub(super) assistant: String,
}

impl SkillManifest {
    pub(super) fn build_system_prompt(&self) -> String {
        let mut prompt = if let Some(sys) = &self.system_prompt {
            sys.clone()
        } else {
            String::new()
        };

        if !self.prompt.is_empty() {
            if !prompt.is_empty() {
                prompt.push_str("\n\n");
            }
            prompt.push_str(&self.prompt);
        }

        prompt
    }
}

pub(super) fn load_all_skills() -> Vec<SkillManifest> {
    let dir = skills_dir();
    let _ = ensure_seeded_skills_dir(&dir);
    let mut by_name = std::collections::BTreeMap::<String, SkillManifest>::new();

    for (filename, content) in BUILTIN_SKILLS {
        if let Ok(mut skill) = parse_skill_front_matter(content) {
            skill.source_path = Some(format!("builtin:{filename}"));
            by_name.insert(skill.name.clone(), skill);
        }
    }

    for skill in load_skills_from_dir(&dir) {
        by_name.insert(skill.name.clone(), skill);
    }

    let mut out = by_name.into_values().collect::<Vec<_>>();
    out.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.name.cmp(&b.name)));
    out
}

pub(super) fn skills_dir() -> PathBuf {
    PathBuf::from(crate::common::utils::expanduser("~/.config/rust_tools/skills").as_ref())
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
    let mut system_prompt: Option<String> = None;
    let mut triggers: Vec<String> = Vec::new();
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
                    "triggers" => triggers.push(v),
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
                "system_prompt" => system_prompt = Some(unquoted.to_string()),
                "priority" => {
                    priority = unquoted.parse::<i32>().unwrap_or(0);
                }
                "tools" => tools = parse_list_value(unquoted),
                "tool_groups" => tool_groups = parse_list_value(unquoted),
                "mcp_servers" => mcp_servers = parse_list_value(unquoted),
                "triggers" => triggers = parse_list_value(unquoted),
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
        prompt: body.trim().to_string(),
        system_prompt: system_prompt.filter(|s| !s.trim().is_empty()),
        examples: Vec::new(),
        triggers,
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

    let mut has_skill = false;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            if file_looks_like_front_matter_skill(&p) {
                has_skill = true;
                break;
            }
        }
    }

    if has_skill {
        return Ok(());
    }

    for (name, content) in BUILTIN_SKILLS {
        let path = dir.join(name);
        if path.exists() {
            continue;
        }
        std::fs::write(&path, content).map_err(|e| format!("failed to seed builtin skill: {e}"))?;
    }
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

    #[test]
    fn seed_skills_dir_writes_skills_when_empty() {
        let dir = std::env::temp_dir().join(format!("rust-tools-skills-{}", uuid::Uuid::new_v4()));
        ensure_seeded_skills_dir(&dir).unwrap();
        let skills = load_skills_from_dir(&dir);
        assert_eq!(skills.len(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_all_skills_includes_builtin_even_when_user_has_custom_skill() {
        let home = std::env::temp_dir()
            .join(format!("rust-tools-home-{}", uuid::Uuid::new_v4()))
            .display()
            .to_string();
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
triggers:
  - custom
priority: 1
---

custom"#,
        )
        .unwrap();

        let skills = load_all_skills();
        let names = skills
            .iter()
            .map(|s| s.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(names.contains("openclaw"));
        assert!(names.contains("debugger"));
        assert!(names.contains("code-review"));
        assert!(names.contains("refactor"));
        assert!(names.contains("custom-skill"));

        let _ = std::fs::remove_dir_all(&home);
    }
}
