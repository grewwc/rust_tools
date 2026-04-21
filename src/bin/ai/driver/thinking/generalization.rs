use std::collections::HashMap;

use chrono::Local;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralizedPrinciple {
    pub id: String,
    pub principle: String,
    pub source_experiences: Vec<String>,
    pub domain: String,
    pub abstraction_level: u8,
    pub confidence: f64,
    pub created_at: String,
    pub last_reinforced: String,
    pub reinforcement_count: u32,
    pub cross_domain_links: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RawExperience {
    id: String,
    category: String,
    note: String,
    tags: Vec<String>,
    source: Option<String>,
}

pub struct ExperienceGeneralizer {
    principles: Vec<GeneralizedPrinciple>,
    pub(crate) experience_buffer: Vec<RawExperience>,
    max_buffer_size: usize,
    min_experiences_for_generalization: usize,
}

impl ExperienceGeneralizer {
    pub fn new() -> Self {
        let mut generalizer = Self {
            principles: Vec::new(),
            experience_buffer: Vec::new(),
            max_buffer_size: 50,
            min_experiences_for_generalization: 3,
        };
        generalizer.load_principles_from_store();
        generalizer
    }

    fn load_principles_from_store(&mut self) {
        let store = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config()
        })) {
            Ok(s) => s,
            Err(_) => return,
        };
        if let Ok(results) = store.search("generalized principle", 50) {
            for (entry, _score) in results {
                if entry.category != "generalized_principle" {
                    continue;
                }
                let principle = GeneralizedPrinciple {
                    id: entry.id.unwrap_or_default(),
                    principle: entry.note.clone(),
                    source_experiences: vec![],
                    domain: entry.tags.iter()
                        .find(|t| t.as_str() != "generalized" && t.as_str() != "principle")
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string()),
                    abstraction_level: 1,
                    confidence: 0.6,
                    created_at: entry.timestamp.clone(),
                    last_reinforced: entry.timestamp.clone(),
                    reinforcement_count: 1,
                    cross_domain_links: Vec::new(),
                };
                self.principles.push(principle);
            }
        }
    }

    pub fn ingest_experience(
        &mut self,
        category: &str,
        note: &str,
        tags: &[String],
        source: Option<&str>,
    ) {
        let experience = RawExperience {
            id: format!("exp_{}", uuid::Uuid::new_v4().simple()),
            category: category.to_string(),
            note: note.to_string(),
            tags: tags.to_vec(),
            source: source.map(|s| s.to_string()),
        };
        self.experience_buffer.push(experience);
        if self.experience_buffer.len() > self.max_buffer_size {
            self.experience_buffer.remove(0);
        }
    }

    pub fn try_generalize(&mut self) -> Option<GeneralizedPrinciple> {
        if self.experience_buffer.len() < self.min_experiences_for_generalization {
            return None;
        }

        let grouped = self.group_by_semantic_similarity();
        let best_group = grouped
            .into_values()
            .filter(|g| g.len() >= self.min_experiences_for_generalization)
            .max_by_key(|g| g.len())?;

        let domain = self.infer_domain(&best_group);
        let source_ids: Vec<String> = best_group.iter().map(|e| e.id.clone()).collect();
        let principle_text = self.synthesize_principle(&best_group, &domain)?;

        let existing = self.find_similar_principle(&principle_text);
        if let Some(existing) = existing {
            let updated = GeneralizedPrinciple {
                reinforcement_count: existing.reinforcement_count + 1,
                last_reinforced: Local::now().to_rfc3339(),
                confidence: (existing.confidence + 0.1).min(1.0),
                source_experiences: {
                    let mut sources = existing.source_experiences.clone();
                    for id in source_ids {
                        if !sources.contains(&id) {
                            sources.push(id);
                        }
                    }
                    sources
                },
                ..existing.clone()
            };
            if let Some(pos) = self.principles.iter().position(|p| p.id == updated.id) {
                self.principles[pos] = updated.clone();
            }
            return Some(updated);
        }

        let principle = GeneralizedPrinciple {
            id: format!("principle_{}", uuid::Uuid::new_v4().simple()),
            principle: principle_text,
            source_experiences: source_ids,
            domain: domain.clone(),
            abstraction_level: 1,
            confidence: 0.6,
            created_at: Local::now().to_rfc3339(),
            last_reinforced: Local::now().to_rfc3339(),
            reinforcement_count: 1,
            cross_domain_links: Vec::new(),
        };
        self.principles.push(principle.clone());
        Some(principle)
    }

    pub fn try_cross_domain_link(&mut self) -> Option<(String, String)> {
        if self.principles.len() < 2 {
            return None;
        }
        let mut best_pair: Option<(usize, usize, f64)> = None;
        for i in 0..self.principles.len() {
            for j in (i + 1)..self.principles.len() {
                if self.principles[i].domain == self.principles[j].domain {
                    continue;
                }
                let similarity = self.compute_text_similarity(
                    &self.principles[i].principle,
                    &self.principles[j].principle,
                );
                if similarity > 0.4 {
                    match best_pair {
                        Some((_, _, best_sim)) if similarity <= best_sim => {}
                        _ => best_pair = Some((i, j, similarity)),
                    }
                }
            }
        }
        if let Some((i, j, sim)) = best_pair {
            let id_i = self.principles[i].id.clone();
            let id_j = self.principles[j].id.clone();
            let level_i = self.principles[i].abstraction_level;
            let level_j = self.principles[j].abstraction_level;
            self.principles[i].cross_domain_links.push(id_j.clone());
            self.principles[j].cross_domain_links.push(id_i.clone());
            if sim > 0.7 {
                self.principles[i].abstraction_level = (level_i + 1).min(5);
                self.principles[j].abstraction_level = (level_j + 1).min(5);
            }
            return Some((id_i, id_j));
        }
        None
    }

    pub fn persist_principle(&self, principle: &GeneralizedPrinciple) -> crate::ai::driver::observer::MemoryEntry {
        crate::ai::driver::observer::MemoryEntry {
            category: "generalized_principle".to_string(),
            note: format!(
                "[domain={}] [abstraction={}] [confidence={:.2}] [reinforced={}] {}\nCross-domain links: {}",
                principle.domain,
                principle.abstraction_level,
                principle.confidence,
                principle.reinforcement_count,
                principle.principle,
                principle.cross_domain_links.join(", ")
            ),
            tags: vec!["generalized".to_string(), "principle".to_string(), principle.domain.clone()],
            source: Some("experience_generalizer".to_string()),
            priority: self.priority_for_abstraction(principle.abstraction_level),
        }
    }

    pub fn generate_generalization_prompt(&self, experiences: &[RawExperience]) -> String {
        let mut exp_text = String::new();
        for (i, exp) in experiences.iter().enumerate() {
            exp_text.push_str(&format!("{}. [{}] {}\n", i + 1, exp.category, exp.note));
        }
        format!(
            "You are an experience generalization engine. Given these specific experiences, \
             derive a higher-level principle that captures the common pattern.\n\n\
             Experiences:\n{}\n\n\
             Rules:\n\
             - The principle must be abstract enough to apply across similar situations\n\
             - It must be specific enough to be actionable\n\
             - Frame it as a 'Do:' or 'Avoid:' guideline\n\
             - Identify the domain (e.g., 'error_handling', 'async_patterns', 'api_design')\n\n\
             Output STRICT JSON: {{\"principle\":\"...\",\"domain\":\"...\",\"confidence\":0.7}}",
            exp_text
        )
    }

    fn group_by_semantic_similarity(&self) -> HashMap<String, Vec<&RawExperience>> {
        let mut groups: HashMap<String, Vec<&RawExperience>> = HashMap::new();
        for exp in &self.experience_buffer {
            let key = self.semantic_group_key(exp);
            groups.entry(key).or_default().push(exp);
        }
        groups
    }

    fn semantic_group_key(&self, exp: &RawExperience) -> String {
        let category_key = exp.category.to_lowercase();
        let tag_key = exp
            .tags
            .iter()
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .take(3)
            .collect::<Vec<_>>()
            .join(":");
        if tag_key.is_empty() {
            category_key
        } else {
            format!("{}:{}", category_key, tag_key)
        }
    }

    fn infer_domain(&self, experiences: &[&RawExperience]) -> String {
        let categories: Vec<&str> = experiences.iter().map(|e| e.category.as_str()).collect();
        let tags: Vec<&str> = experiences
            .iter()
            .flat_map(|e| e.tags.iter().map(|t| t.as_str()))
            .collect();
        let all_text = format!("{} {}", categories.join(" "), tags.join(" "));
        let lower = all_text.to_lowercase();
        if lower.contains("async") || lower.contains("concurrent") || lower.contains("race") {
            return "async_patterns".to_string();
        }
        if lower.contains("error") || lower.contains("fail") || lower.contains("bug") {
            return "error_handling".to_string();
        }
        if lower.contains("api") || lower.contains("endpoint") || lower.contains("request") {
            return "api_design".to_string();
        }
        if lower.contains("test") || lower.contains("verify") || lower.contains("assert") {
            return "testing".to_string();
        }
        if lower.contains("security") || lower.contains("auth") || lower.contains("permission") {
            return "security".to_string();
        }
        if lower.contains("performance") || lower.contains("optim") || lower.contains("speed") {
            return "performance".to_string();
        }
        if lower.contains("refactor") || lower.contains("architect") || lower.contains("design") {
            return "architecture".to_string();
        }
        "general_engineering".to_string()
    }

    fn synthesize_principle(&self, experiences: &[&RawExperience], domain: &str) -> Option<String> {
        let notes: Vec<&str> = experiences.iter().map(|e| e.note.as_str()).collect();
        let common_keywords = self.find_common_keywords(&notes);
        if common_keywords.is_empty() {
            return None;
        }
        let do_items: Vec<&str> = notes
            .iter()
            .filter(|n| {
                n.to_lowercase().starts_with("do:") || n.to_lowercase().starts_with("always")
            })
            .map(|n| *n)
            .collect();
        let avoid_items: Vec<&str> = notes
            .iter()
            .filter(|n| {
                n.to_lowercase().starts_with("avoid:") || n.to_lowercase().starts_with("never")
            })
            .map(|n| *n)
            .collect();
        let mut principle = format!("In {}, ", domain.replace('_', " "));
        if !do_items.is_empty() {
            principle.push_str(&format!("Do: {}; ", do_items.join("; ")));
        }
        if !avoid_items.is_empty() {
            principle.push_str(&format!("Avoid: {}", avoid_items.join("; ")));
        }
        if do_items.is_empty() && avoid_items.is_empty() {
            principle.push_str(&format!("key pattern: {}", common_keywords.join(", ")));
        }
        Some(principle)
    }

    fn find_common_keywords(&self, notes: &[&str]) -> Vec<String> {
        let stop_words: &[&str] = &[
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again", "further", "then", "once",
            "and", "but", "or", "nor", "not", "so", "yet", "both", "either", "neither", "each",
            "every", "all", "any", "few", "more", "most", "other", "some", "such", "no", "only",
            "own", "same", "than", "too", "very", "just", "because", "if", "when", "this", "that",
            "these", "those", "it", "its", "i", "me", "my", "we", "our",
        ];
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for note in notes {
            for word in note.to_lowercase().split_whitespace() {
                let w = word.trim_matches(|c: char| !c.is_alphanumeric());
                if w.len() < 3 || stop_words.contains(&w) {
                    continue;
                }
                *word_counts.entry(w.to_string()).or_insert(0) += 1;
            }
        }
        let mut keywords: Vec<(String, usize)> = word_counts.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));
        keywords.truncate(5);
        keywords.into_iter().map(|(w, _)| w).collect()
    }

    fn find_similar_principle(&self, text: &str) -> Option<&GeneralizedPrinciple> {
        let text_lower = text.to_lowercase();
        self.principles
            .iter()
            .filter(|p| {
                self.compute_text_similarity(&p.principle.to_lowercase(), &text_lower) > 0.6
            })
            .max_by(|a, b| {
                let sim_a = self.compute_text_similarity(&a.principle.to_lowercase(), &text_lower);
                let sim_b = self.compute_text_similarity(&b.principle.to_lowercase(), &text_lower);
                sim_a
                    .partial_cmp(&sim_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    fn compute_text_similarity(&self, a: &str, b: &str) -> f64 {
        let words_a: std::collections::HashSet<String> =
            a.split_whitespace().map(|w| w.to_lowercase()).collect();
        let words_b: std::collections::HashSet<String> =
            b.split_whitespace().map(|w| w.to_lowercase()).collect();
        if words_a.is_empty() || words_b.is_empty() {
            return 0.0;
        }
        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();
        intersection as f64 / union as f64
    }

    fn priority_for_abstraction(&self, level: u8) -> u8 {
        match level {
            0 => 150,
            1 => 180,
            2 => 200,
            3 => 220,
            4..=5 => 240,
            _ => 255,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_and_generalize() {
        let mut generalizer = ExperienceGeneralizer::new();
        generalizer.ingest_experience(
            "error_handling",
            "Do: always check Option before unwrap in async code",
            &["async".to_string()],
            None,
        );
        generalizer.ingest_experience(
            "error_handling",
            "Do: validate async results before chaining",
            &["async".to_string()],
            None,
        );
        generalizer.ingest_experience(
            "error_handling",
            "Avoid: unwrap on async task results",
            &["async".to_string()],
            None,
        );
        let principle = generalizer.try_generalize();
        assert!(principle.is_some());
        let p = principle.unwrap();
        assert!(p.domain == "async_patterns" || p.domain == "error_handling");
    }

    #[test]
    fn cross_domain_linking() {
        let mut generalizer = ExperienceGeneralizer::new();
        generalizer.principles.push(GeneralizedPrinciple {
            id: "p1".to_string(),
            principle: "Always validate inputs before processing in API handlers".to_string(),
            source_experiences: vec![],
            domain: "api_design".to_string(),
            abstraction_level: 1,
            confidence: 0.7,
            created_at: Local::now().to_rfc3339(),
            last_reinforced: Local::now().to_rfc3339(),
            reinforcement_count: 1,
            cross_domain_links: vec![],
        });
        generalizer.principles.push(GeneralizedPrinciple {
            id: "p2".to_string(),
            principle: "Always validate inputs before processing in async handlers".to_string(),
            source_experiences: vec![],
            domain: "async_patterns".to_string(),
            abstraction_level: 1,
            confidence: 0.7,
            created_at: Local::now().to_rfc3339(),
            last_reinforced: Local::now().to_rfc3339(),
            reinforcement_count: 1,
            cross_domain_links: vec![],
        });
        let link = generalizer.try_cross_domain_link();
        assert!(link.is_some());
    }

    #[test]
    fn too_few_experiences_no_generalization() {
        let mut generalizer = ExperienceGeneralizer::new();
        generalizer.ingest_experience("test", "one note", &[], None);
        generalizer.ingest_experience("test", "two note", &[], None);
        assert!(generalizer.try_generalize().is_none());
    }
}
