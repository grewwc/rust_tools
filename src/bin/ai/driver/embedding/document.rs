use crate::ai::skills::SkillManifest;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SkillEmbeddingDocumentSection {
    Identity,
    Capability,
    Behavior,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SkillEmbeddingDocument {
    pub skill_name: String,
    pub source_key: String,
    pub source_hash: String,
    pub identity_text: String,
    pub capability_text: String,
    pub behavior_text: String,
}

impl SkillEmbeddingDocument {
    pub fn from_skill(skill: &SkillManifest) -> Self {
        let source_key = skill
            .source_path
            .clone()
            .unwrap_or_else(|| format!("skill:{}", skill.name));
        let identity_text = build_identity_text(skill);
        let capability_text = build_capability_text(skill);
        let behavior_text = build_behavior_text(skill);
        let source_hash = skill.routing_source_hash();
        Self {
            skill_name: skill.name.clone(),
            source_key,
            source_hash,
            identity_text,
            capability_text,
            behavior_text,
        }
    }

    pub fn sections(&self) -> [(SkillEmbeddingDocumentSection, &str); 3] {
        [
            (SkillEmbeddingDocumentSection::Identity, self.identity_text.as_str()),
            (
                SkillEmbeddingDocumentSection::Capability,
                self.capability_text.as_str(),
            ),
            (SkillEmbeddingDocumentSection::Behavior, self.behavior_text.as_str()),
        ]
    }
}

fn build_identity_text(skill: &SkillManifest) -> String {
    let mut lines = vec![skill.name.clone(), skill.description.clone()];
    if !skill.triggers.is_empty() {
        lines.push(skill.triggers.join("\n"));
    }
    normalize_section(lines)
}

fn build_capability_text(skill: &SkillManifest) -> String {
    let mut lines = Vec::new();
    if !skill.tools.is_empty() {
        lines.push(skill.tools.join("\n"));
    }
    if !skill.tool_groups.is_empty() {
        lines.push(skill.tool_groups.join("\n"));
    }
    if !skill.mcp_servers.is_empty() {
        lines.push(skill.mcp_servers.join("\n"));
    }
    if skill.skip_recall {
        lines.push("skip recall".to_string());
    }
    if skill.disable_builtin_tools {
        lines.push("disable builtin tools".to_string());
    }
    if skill.disable_mcp_tools {
        lines.push("disable mcp tools".to_string());
    }
    normalize_section(lines)
}

fn build_behavior_text(skill: &SkillManifest) -> String {
    let mut lines = Vec::new();
    if let Some(system_prompt) = &skill.system_prompt {
        lines.push(summarize_text(system_prompt, 12, 800));
    }
    if !skill.prompt.trim().is_empty() {
        lines.push(summarize_text(&skill.prompt, 18, 1400));
    }
    normalize_section(lines)
}

fn normalize_section(lines: Vec<String>) -> String {
    lines
        .into_iter()
        .flat_map(|line| {
            line.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn summarize_text(text: &str, max_lines: usize, max_chars: usize) -> String {
    let mut kept = Vec::new();
    let mut total_chars = 0usize;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "---" {
            continue;
        }
        let compact = trimmed
            .trim_start_matches('#')
            .trim_start_matches('-')
            .trim_start_matches('*')
            .trim();
        if compact.is_empty() {
            continue;
        }
        total_chars += compact.chars().count();
        if total_chars > max_chars {
            break;
        }
        kept.push(compact.to_string());
        if kept.len() >= max_lines {
            break;
        }
    }
    kept.join("\n")
}
