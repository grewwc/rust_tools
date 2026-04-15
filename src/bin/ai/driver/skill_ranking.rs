use std::path::Path;

use crate::ai::skills::SkillManifest;
use rust_tools::commonw::FastMap;

use super::{
    TextSimilarityFeatures, UserIntent, is_intent_excluded, normalize_text_for_similarity,
    score_skill_smart, skill_match_model,
};

const LOCAL_MODEL_SCORE_SCALE: f64 = 8.0;

#[derive(Debug, Clone)]
pub struct ScoredSkill<'a> {
    pub skill: &'a SkillManifest,
    pub score: f64,
    pub heuristic_score: f64,
    pub model_score: f64,
}

pub fn rank_skills_locally<'a>(
    skills: &'a [SkillManifest],
    input: &str,
    intent: Option<&UserIntent>,
) -> Vec<ScoredSkill<'a>> {
    if input.trim().is_empty() || skills.is_empty() {
        return Vec::new();
    }

    let input_lower = input.to_lowercase();
    let model_path = skill_match_model::default_model_path();
    let model_probs = skill_match_model::predict_skill(input, &model_path);
    let mut ranked = Vec::new();

    for skill in skills {
        if let Some(intent_ref) = intent
            && is_intent_excluded(skill, intent_ref)
        {
            continue;
        }

        let heuristic_score = score_skill_smart(skill, &input_lower, intent);
        let model_score = match &model_probs {
            Some(result) => result
                .probabilities
                .iter()
                .find(|(label, _)| label == &skill.name)
                .map(|(_, prob)| *prob)
                .unwrap_or(0.0),
            None => 0.0,
        };
        let priority_bonus = (skill.priority.max(0) as f64).min(100.0) / 100.0;
        let score = heuristic_score + model_score * LOCAL_MODEL_SCORE_SCALE + priority_bonus;
        ranked.push(ScoredSkill {
            skill,
            score,
            heuristic_score,
            model_score,
        });
    }

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.model_score
                    .partial_cmp(&a.model_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.skill.priority.cmp(&a.skill.priority))
            .then_with(|| a.skill.name.cmp(&b.skill.name))
    });
    ranked
}

pub fn rank_skills_locally_with_model_path<'a>(
    skills: &'a [SkillManifest],
    input: &str,
    intent: Option<&UserIntent>,
    model_path: &Path,
) -> Vec<ScoredSkill<'a>> {
    if input.trim().is_empty() || skills.is_empty() {
        return Vec::new();
    }

    let input_lower = input.to_lowercase();
    let model_probs = skill_match_model::predict_skill(input, model_path);
    let runtime_model = RuntimeSkillModel::build(skills);
    let query_features = TextSimilarityFeatures::from_text(&input_lower);
    let mut ranked = Vec::new();

    for skill in skills {
        if let Some(intent_ref) = intent
            && is_intent_excluded(skill, intent_ref)
        {
            continue;
        }

        let heuristic_score = score_skill_smart(skill, &input_lower, intent);
        let static_model_score = match &model_probs {
            Some(result) => result
                .probabilities
                .iter()
                .find(|(label, _)| label == &skill.name)
                .map(|(_, prob)| *prob)
                .unwrap_or(0.0),
            None => 0.0,
        };
        let runtime_model_score = runtime_model.similarity(skill, &query_features);
        let model_score = static_model_score.max(runtime_model_score);
        let priority_bonus = (skill.priority.max(0) as f64).min(100.0) / 100.0;
        let score = heuristic_score + model_score * LOCAL_MODEL_SCORE_SCALE + priority_bonus;
        ranked.push(ScoredSkill {
            skill,
            score,
            heuristic_score,
            model_score,
        });
    }

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.model_score
                    .partial_cmp(&a.model_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.skill.priority.cmp(&a.skill.priority))
            .then_with(|| a.skill.name.cmp(&b.skill.name))
    });
    ranked
}

struct RuntimeSkillDoc<'a> {
    skill: &'a SkillManifest,
    tf: FastMap<String, f64>,
}

struct RuntimeSkillModel<'a> {
    docs: Vec<RuntimeSkillDoc<'a>>,
    idf: FastMap<String, f64>,
}

impl<'a> RuntimeSkillModel<'a> {
    fn build(skills: &'a [SkillManifest]) -> Self {
        let mut docs = Vec::with_capacity(skills.len());
        let mut df: FastMap<String, usize> = FastMap::default();
        for skill in skills {
            let text = skill_document_text(skill);
            let features = TextSimilarityFeatures::from_text(&text);
            let tf = features.ngram_tf.clone();
            let unique = tf.keys().cloned().collect::<Vec<_>>();
            for token in unique {
                *df.entry(token).or_insert(0) += 1;
            }
            docs.push(RuntimeSkillDoc { skill, tf });
        }

        let total_docs = skills.len().max(1) as f64;
        let mut idf = FastMap::default();
        for (token, freq) in df {
            let value = ((1.0 + total_docs) / (1.0 + freq as f64)).ln() + 1.0;
            idf.insert(token, value);
        }
        Self { docs, idf }
    }

    fn similarity(&self, skill: &SkillManifest, query: &TextSimilarityFeatures) -> f64 {
        if query.ngram_tf.is_empty() {
            return 0.0;
        }
        let Some(doc) = self.docs.iter().find(|doc| doc.skill.name == skill.name) else {
            return 0.0;
        };
        cosine_tfidf_similarity(&query.ngram_tf, &doc.tf, &self.idf)
    }
}

fn skill_document_text(skill: &SkillManifest) -> String {
    let mut parts = vec![skill.name.clone(), skill.description.clone()];
    if !skill.tools.is_empty() {
        parts.push(skill.tools.join(" "));
    }
    if !skill.tool_groups.is_empty() {
        parts.push(skill.tool_groups.join(" "));
    }
    if !skill.mcp_servers.is_empty() {
        parts.push(skill.mcp_servers.join(" "));
    }
    if !skill.prompt.trim().is_empty() {
        parts.push(truncate_chars(&skill.prompt, 6000));
    }
    normalize_text_for_similarity(&parts.join("\n"))
}

fn cosine_tfidf_similarity(
    query_tf: &FastMap<String, f64>,
    doc_tf: &FastMap<String, f64>,
    idf: &FastMap<String, f64>,
) -> f64 {
    let mut dot = 0.0;
    let mut query_norm = 0.0;
    let mut doc_norm = 0.0;

    for (token, tf) in query_tf {
        let weight = *tf * idf.get(token.as_str()).copied().unwrap_or(1.0);
        query_norm += weight * weight;
        if let Some(doc_tf) = doc_tf.get(token.as_str()) {
            let doc_weight = *doc_tf * idf.get(token.as_str()).copied().unwrap_or(1.0);
            dot += weight * doc_weight;
        }
    }
    for (token, tf) in doc_tf {
        let weight = *tf * idf.get(token.as_str()).copied().unwrap_or(1.0);
        doc_norm += weight * weight;
    }
    if query_norm <= f64::EPSILON || doc_norm <= f64::EPSILON {
        return 0.0;
    }
    dot / (query_norm.sqrt() * doc_norm.sqrt())
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    input.chars().take(max_chars).collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::driver::intent_recognition::{CoreIntent, UserIntent};

    fn skill(name: &str, description: &str) -> SkillManifest {
        SkillManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: description.to_string(),
            author: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            prompt: String::new(),
            system_prompt: None,
            priority: 0,
            source_path: Some(format!("custom:{name}.skill")),
        }
    }

    #[test]
    fn unknown_dynamic_skill_still_gets_runtime_model_score() {
        let skills = vec![
            skill("code-review", "Review code quality and bugs"),
            skill("my-custom-slides", "生成幻灯片 PPT 演示文稿 slides presentation"),
        ];
        let ranked = rank_skills_locally_with_model_path(
            &skills,
            "帮我生成一份 PPT 幻灯片",
            Some(&UserIntent::new(CoreIntent::RequestAction)),
            &skill_match_model::default_model_path(),
        );
        let top = ranked.first().expect("expected ranked result");
        assert_eq!(top.skill.name, "my-custom-slides");
        assert!(
            top.model_score > 0.0,
            "expected dynamic skill to receive runtime model score"
        );
    }
}
