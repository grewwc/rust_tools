use std::path::Path;

use crate::ai::skills::SkillManifest;
use crate::commonw::configw;
use rust_tools::commonw::FastMap;

use super::{
    embedding::{
        document::SkillEmbeddingDocument,
        index::{SkillEmbeddingHit, SkillEmbeddingIndex},
    },
    TextSimilarityFeatures, UserIntent, build_idf_from_documents, cosine_tfidf_similarity,
    is_intent_excluded, normalize_text_for_similarity, skill_match_model,
};

const LOCAL_MODEL_SCORE_SCALE: f64 = 8.0;
pub const SKILL_NONE_LABEL: &str = "none";

#[derive(Debug, Clone)]
pub struct ScoredSkill<'a> {
    pub skill: &'a SkillManifest,
    pub score: f64,
    pub embedding_score: f64,
    pub embedding_identity_score: f64,
    pub embedding_capability_score: f64,
    pub embedding_behavior_score: f64,
    pub fallback_semantic_score: f64,
    pub model_prior_score: f64,
    pub blended_score: f64,
    pub none_score: f64,
}

pub fn rank_skills_locally<'a>(
    skills: &'a [SkillManifest],
    input: &str,
    intent: Option<&UserIntent>,
) -> Vec<ScoredSkill<'a>> {
    if input.trim().is_empty() || skills.is_empty() {
        return Vec::new();
    }

    let model_path = skill_match_model::default_model_path();
    rank_skills_locally_with_model_path(skills, input, intent, &model_path)
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
    let candidates = skills
        .iter()
        .filter(|skill| {
            intent
                .map(|intent_ref| !is_intent_excluded(skill, intent_ref))
                .unwrap_or(true)
        })
        .cloned()
        .collect::<Vec<_>>();
    let embedding_hits = build_embedding_hits(&candidates, input).unwrap_or_default();
    let mut ranked = Vec::new();

    for skill in skills {
        if let Some(intent_ref) = intent
            && is_intent_excluded(skill, intent_ref)
        {
            continue;
        }

        let model_prior_score = match &model_probs {
            Some(result) => probability_for_label(result, &skill.name),
            None => 0.0,
        };
        let fallback_semantic_score = runtime_model.similarity(skill, &query_features);
        let (embedding_score, embedding_identity_score, embedding_capability_score, embedding_behavior_score) =
            embedding_hits
                .get(skill.name.as_str())
                .map(|hit| {
                    (
                        hit.score,
                        hit.identity_score,
                        hit.capability_score,
                        hit.behavior_score,
                    )
                })
                .unwrap_or((0.0, 0.0, 0.0, 0.0));
        let blended_score = embedding_score
            .max(fallback_semantic_score)
            .max(model_prior_score);
        let none_score = match &model_probs {
            Some(result) => probability_for_label(result, SKILL_NONE_LABEL),
            None => 0.0,
        };
        let priority_bonus = (skill.priority.max(0) as f64).min(100.0) / 100.0;
        let score = blended_score * LOCAL_MODEL_SCORE_SCALE + priority_bonus;
        ranked.push(ScoredSkill {
            skill,
            score,
            embedding_score,
            embedding_identity_score,
            embedding_capability_score,
            embedding_behavior_score,
            fallback_semantic_score,
            model_prior_score,
            blended_score,
            none_score,
        });
    }

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.blended_score
                    .partial_cmp(&a.blended_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.skill.priority.cmp(&a.skill.priority))
            .then_with(|| a.skill.name.cmp(&b.skill.name))
    });
    ranked
}

fn build_embedding_hits(
    skills: &[SkillManifest],
    input: &str,
) -> Result<FastMap<String, SkillEmbeddingHit>, String> {
    if skills.is_empty() || !skill_embedding_routing_enabled() {
        return Ok(FastMap::default());
    }
    let documents = skills
        .iter()
        .map(SkillEmbeddingDocument::from_skill)
        .collect::<Vec<_>>();
    let index = SkillEmbeddingIndex::build(&documents)?;
    let hits = index.search(input, documents.len())?;
    Ok(hits
        .into_iter()
        .map(|hit| (hit.skill_name.clone(), hit))
        .collect())
}

fn skill_embedding_routing_enabled() -> bool {
    configw::get_all_config()
        .get_opt("ai.skills.embedding_routing")
        .map(|value| value.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false)
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
        for skill in skills {
            let text = skill_document_text(skill);
            let features = TextSimilarityFeatures::from_text(&text);
            let tf = features.ngram_tf.clone();
            docs.push(RuntimeSkillDoc { skill, tf });
        }

        let doc_refs = docs.iter().map(|doc| &doc.tf).collect::<Vec<_>>();
        let idf = build_idf_from_documents(&doc_refs);
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
    if !skill.triggers.is_empty() {
        parts.push(skill.triggers.join(" "));
    }
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

fn truncate_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    input.chars().take(max_chars).collect::<String>()
}

fn semantic_score_for_skill(skill: &SkillManifest, input_lower: &str) -> f64 {
    let docs = [skill_document_text(skill)];
    let query_features = TextSimilarityFeatures::from_text(input_lower);
    let doc_features = docs
        .iter()
        .map(|doc| TextSimilarityFeatures::from_text(doc))
        .collect::<Vec<_>>();
    let doc_refs = doc_features.iter().map(|doc| &doc.ngram_tf).collect::<Vec<_>>();
    let idf = build_idf_from_documents(&doc_refs);
    doc_features
        .iter()
        .map(|doc| cosine_tfidf_similarity(&query_features.ngram_tf, &doc.ngram_tf, &idf))
        .fold(0.0, f64::max)
}

fn probability_for_label(result: &skill_match_model::SkillMatchResult, label: &str) -> f64 {
    result
        .probabilities
        .iter()
        .find(|(candidate, _)| candidate == label)
        .map(|(_, prob)| *prob)
        .unwrap_or(0.0)
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
            triggers: Vec::new(),
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            skip_recall: false,
            disable_builtin_tools: false,
            disable_mcp_tools: false,
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
            top.blended_score > 0.0,
            "expected dynamic skill to receive runtime model score"
        );
    }

    #[test]
    fn triggers_participate_in_runtime_ranking() {
        let mut slide = skill("slides", "Create presentation artifacts");
        slide.triggers = vec!["ppt".to_string(), "幻灯片".to_string()];
        let skills = vec![
            skill("code-review", "Review code quality and bugs"),
            slide,
        ];
        let ranked = rank_skills_locally_with_model_path(
            &skills,
            "帮我做一份 ppt 汇报",
            Some(&UserIntent::new(CoreIntent::RequestAction)),
            &skill_match_model::default_model_path(),
        );
        assert_eq!(ranked.first().map(|item| item.skill.name.as_str()), Some("slides"));
    }

    #[test]
    fn none_label_probability_is_exposed_in_ranked_results() {
        let skills = vec![
            skill("code-review", "Review code quality and bugs"),
            skill("debugger", "Debug runtime failures"),
        ];
        let ranked = rank_skills_locally_with_model_path(
            &skills,
            "你好，今天天气怎么样",
            Some(&UserIntent::new(CoreIntent::Casual)),
            &skill_match_model::default_model_path(),
        );
        assert!(!ranked.is_empty());
        assert!(ranked[0].none_score > 0.0);
    }
}
