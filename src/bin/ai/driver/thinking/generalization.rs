use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use chrono::Local;
use serde::{Deserialize, Serialize};

const PERSIST_PREFIX_DOMAIN: &str = "[domain=";
const PERSIST_PREFIX_ABSTRACTION: &str = "[abstraction=";
const PERSIST_PREFIX_CONFIDENCE: &str = "[confidence=";
const PERSIST_PREFIX_REINFORCED: &str = "[reinforced=";
const PERSIST_SUFFIX_LINKS: &str = "\nCross-domain links: ";

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

impl GeneralizedPrinciple {
    /// 基于 `last_reinforced` 的时间衰减；保守半衰期 30 天。
    /// 30 天衰减到 0.5 倍，60 天到 0.25 倍。下界 0.2，避免完全淹没。
    pub fn effective_confidence(&self) -> f64 {
        let days_since_reinforced = chrono::DateTime::parse_from_rfc3339(&self.last_reinforced)
            .map(|t| {
                let now = chrono::Local::now().with_timezone(t.offset());
                (now - t).num_days().max(0) as f64
            })
            .unwrap_or(0.0);
        let half_life_days: f64 = 30.0;
        let decay = 0.5_f64.powf(days_since_reinforced / half_life_days);
        (self.confidence * decay).max(self.confidence * 0.2)
    }
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
        generalizer.load_experience_buffer_from_store();
        generalizer
    }

    fn load_principles_from_store(&mut self) {
        let store = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config()
        })) {
            Ok(s) => s,
            Err(_) => return,
        };
        if let Ok(entries) = store.entries_by_category("generalized_principle", 200) {
            let mut deduped = HashMap::new();
            for entry in entries {
                let principle = Self::decode_persisted_principle(&entry);
                match deduped.get(&principle.id) {
                    Some(existing) if !Self::should_replace_loaded_principle(existing, &principle) => {}
                    _ => {
                        deduped.insert(principle.id.clone(), principle);
                    }
                }
            }
            let mut principles = deduped.into_values().collect::<Vec<_>>();
            principles.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
            self.principles.extend(principles);
        }
    }

    /// 从存储加载未泛化的经验缓冲区，恢复跨会话的待泛化数据
    fn load_experience_buffer_from_store(&mut self) {
        let store = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config()
        })) {
            Ok(s) => s,
            Err(_) => return,
        };
        if let Ok(entries) = store.entries_by_category("raw_experience", self.max_buffer_size) {
            for entry in entries {
                // 优先从专门标签 `cat:<原 category>` 还原真实 category；
                // 老数据兼容：回退到 tags 中第一个非内置标签；再不行用 "general"
                let category = entry
                    .tags
                    .iter()
                    .find_map(|t| t.strip_prefix("cat:").map(str::to_string))
                    .or_else(|| {
                        entry
                            .tags
                            .iter()
                            .find(|t| {
                                let s = t.as_str();
                                s != "raw_experience" && !s.starts_with("cat:")
                            })
                            .cloned()
                    })
                    .unwrap_or_else(|| "general".to_string());
                let exp = RawExperience {
                    id: entry.id.clone().unwrap_or_else(|| {
                        format!("exp_{}", uuid::Uuid::new_v4().simple())
                    }),
                    category,
                    note: entry.note.clone(),
                    tags: entry.tags.clone(),
                    source: entry.source.clone(),
                };
                self.experience_buffer.push(exp);
                if self.experience_buffer.len() >= self.max_buffer_size {
                    break;
                }
            }
        }
    }

    /// 持久化单条经验到 store，使经验缓冲区跨进程保留
    fn persist_experience(&self, exp: &RawExperience) {
        let entry = crate::ai::tools::storage::memory_store::AgentMemoryEntry {
            id: Some(exp.id.clone()),
            timestamp: chrono::Local::now().to_rfc3339(),
            category: "raw_experience".to_string(),
            note: exp.note.clone(),
            tags: {
                let mut t = exp.tags.clone();
                if !t.iter().any(|x| x == "raw_experience") {
                    t.push("raw_experience".to_string());
                }
                // 把真实 category 编码成 `cat:<value>` 标签，避免 reload 时丢失
                if !exp.category.is_empty() && exp.category != "raw_experience" {
                    let cat_tag = format!("cat:{}", exp.category);
                    if !t.iter().any(|x| x == &cat_tag) {
                        t.push(cat_tag);
                    }
                }
                t
            },
            source: exp.source.clone(),
            priority: Some(80),
            owner_pid: None,
            owner_pgid: None,
        };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
            let _ = store.append(&entry);
        }));
    }

    /// 从存储中删除一条已泛化的经验，避免重复消费
    fn forget_experience(&self, exp_id: &str) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
            let _ = store.delete_by_id(exp_id);
        }));
    }

    pub fn ingest_experience(
        &mut self,
        category: &str,
        note: &str,
        tags: &[String],
        source: Option<&str>,
    ) {
        const MAX_NOTE_CHARS: usize = 2000;
        let truncated_note: String = if note.chars().count() > MAX_NOTE_CHARS {
            note.chars().take(MAX_NOTE_CHARS).collect::<String>() + "...[truncated]"
        } else {
            note.to_string()
        };
        let experience = RawExperience {
            id: format!("exp_{}", uuid::Uuid::new_v4().simple()),
            category: category.to_string(),
            note: truncated_note,
            tags: tags.to_vec(),
            source: source.map(|s| s.to_string()),
        };
        self.persist_experience(&experience);
        self.experience_buffer.push(experience);
        if self.experience_buffer.len() > self.max_buffer_size {
            let evicted = self.experience_buffer.remove(0);
            self.forget_experience(&evicted.id);
        }
    }

    pub fn try_generalize(&mut self) -> Option<GeneralizedPrinciple> {
        if self.experience_buffer.len() < self.min_experiences_for_generalization {
            return None;
        }
        // Fast path: if no experience carries an explicit structured prefix,
        // synthesize_principle would return None anyway — skip the grouping work.
        let has_structured = self.experience_buffer.iter().any(|e| {
            let lower = e.note.trim_start().to_lowercase();
            lower.starts_with("do:") || lower.starts_with("avoid:")
                || lower.starts_with("always") || lower.starts_with("never")
        });
        if !has_structured {
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
                    for id in &source_ids {
                        if !sources.contains(id) {
                            sources.push(id.clone());
                        }
                    }
                    sources
                },
                ..existing.clone()
            };
            if let Some(pos) = self.principles.iter().position(|p| p.id == updated.id) {
                self.principles[pos] = updated.clone();
            }
            self.consume_experiences(&source_ids);
            return Some(updated);
        }

        let principle = GeneralizedPrinciple {
            id: format!("principle_{}", uuid::Uuid::new_v4().simple()),
            principle: principle_text,
            source_experiences: source_ids.clone(),
            domain: domain.clone(),
            abstraction_level: 1,
            confidence: 0.6,
            created_at: Local::now().to_rfc3339(),
            last_reinforced: Local::now().to_rfc3339(),
            reinforcement_count: 1,
            cross_domain_links: Vec::new(),
        };
        self.principles.push(principle.clone());
        self.consume_experiences(&source_ids);
        Some(principle)
    }

    /// 已被泛化吸收的经验从 buffer 和 store 中移除，避免重复参与下一次泛化
    fn consume_experiences(&mut self, ids: &[String]) {
        self.experience_buffer.retain(|e| !ids.contains(&e.id));
        for id in ids {
            self.forget_experience(id);
        }
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
                if self.principles[i].cross_domain_links.contains(&self.principles[j].id) {
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
            let updated_i = self.principles[i].clone();
            let updated_j = self.principles[j].clone();
            self.persist_principle(&updated_i);
            self.persist_principle(&updated_j);
            return Some((id_i, id_j));
        }
        None
    }

    pub fn persist_principle(&self, principle: &GeneralizedPrinciple) {
        let entry = crate::ai::tools::storage::memory_store::AgentMemoryEntry {
            id: Some(principle.id.clone()),
            timestamp: chrono::Local::now().to_rfc3339(),
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
            priority: Some(self.priority_for_principle(principle)),
            owner_pid: None,
            owner_pgid: None,
        };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
            // 先删除同 id 的旧记录，再追加，避免每次 reinforce 都向 JSONL 追加重复条目
            let _ = store.delete_by_id(&principle.id);
            let _ = store.append(&entry);
        }));
    }

    fn decode_persisted_principle(
        entry: &crate::ai::tools::storage::memory_store::AgentMemoryEntry,
    ) -> GeneralizedPrinciple {
        let (note_text, cross_domain_links) = Self::split_cross_domain_links(&entry.note);
        let (
            principle_body,
            domain,
            abstraction_level,
            confidence,
            reinforcement_count,
        ) = Self::parse_persisted_principle_note(note_text, &entry.tags);

        GeneralizedPrinciple {
            id: entry
                .id
                .clone()
                .filter(|id| !id.trim().is_empty())
                .unwrap_or_else(|| Self::legacy_principle_id(&entry.timestamp, &principle_body)),
            principle: principle_body,
            source_experiences: vec![],
            domain,
            abstraction_level,
            confidence,
            created_at: entry.timestamp.clone(),
            last_reinforced: entry.timestamp.clone(),
            reinforcement_count,
            cross_domain_links,
        }
    }

    fn split_cross_domain_links(note: &str) -> (&str, Vec<String>) {
        if let Some(idx) = note.find(PERSIST_SUFFIX_LINKS) {
            let links = note[idx + PERSIST_SUFFIX_LINKS.len()..]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            (&note[..idx], links)
        } else {
            (note, Vec::new())
        }
    }

    fn parse_persisted_principle_note(
        note_text: &str,
        tags: &[String],
    ) -> (String, String, u8, f64, u32) {
        let fallback_domain = tags
            .iter()
            .find(|t| t.as_str() != "generalized" && t.as_str() != "principle")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let Some(rest) = note_text.strip_prefix(PERSIST_PREFIX_DOMAIN) else {
            return (note_text.to_string(), fallback_domain, 1, 0.6, 1);
        };
        let Some((domain, rest)) = rest.split_once("] [") else {
            return (note_text.to_string(), fallback_domain, 1, 0.6, 1);
        };
        let Some(rest) = rest.strip_prefix(PERSIST_PREFIX_ABSTRACTION) else {
            return (note_text.to_string(), fallback_domain, 1, 0.6, 1);
        };
        let Some((abstraction, rest)) = rest.split_once("] [") else {
            return (note_text.to_string(), fallback_domain, 1, 0.6, 1);
        };
        let Some(rest) = rest.strip_prefix(PERSIST_PREFIX_CONFIDENCE) else {
            return (note_text.to_string(), fallback_domain, 1, 0.6, 1);
        };
        let Some((confidence, rest)) = rest.split_once("] [") else {
            return (note_text.to_string(), fallback_domain, 1, 0.6, 1);
        };
        let Some(rest) = rest.strip_prefix(PERSIST_PREFIX_REINFORCED) else {
            return (note_text.to_string(), fallback_domain, 1, 0.6, 1);
        };
        let Some((reinforced, principle_body)) = rest.split_once("] ") else {
            return (note_text.to_string(), fallback_domain, 1, 0.6, 1);
        };

        (
            principle_body.to_string(),
            domain.to_string(),
            abstraction.parse().unwrap_or(1),
            confidence.parse().unwrap_or(0.6),
            reinforced.parse().unwrap_or(1),
        )
    }

    fn should_replace_loaded_principle(
        existing: &GeneralizedPrinciple,
        candidate: &GeneralizedPrinciple,
    ) -> bool {
        (
            candidate.cross_domain_links.len(),
            candidate.reinforcement_count,
            candidate.abstraction_level,
            candidate.last_reinforced.as_str(),
            candidate.created_at.as_str(),
        ) > (
            existing.cross_domain_links.len(),
            existing.reinforcement_count,
            existing.abstraction_level,
            existing.last_reinforced.as_str(),
            existing.created_at.as_str(),
        )
    }

    fn legacy_principle_id(timestamp: &str, principle_body: &str) -> String {
        let mut hasher = DefaultHasher::new();
        timestamp.hash(&mut hasher);
        principle_body.hash(&mut hasher);
        format!("legacy_principle_{:016x}", hasher.finish())
    }

    #[cfg(test)]
    pub(crate) fn inject_principles_for_test(&mut self, principles: Vec<GeneralizedPrinciple>) {
        self.principles = principles;
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
        // No more keyword guessing. Prefer explicit tags; fall back to
        // category; fall back to a single neutral bucket.
        // 排除内置/管线性质的 tag（包括 raw_experience 和 `cat:` 编码标签），
        // 否则 reload 后会把 "raw_experience" 当成 domain 显示出来。
        for e in experiences {
            for tag in &e.tags {
                let t = tag.trim();
                if t.is_empty() {
                    continue;
                }
                if matches!(t, "agent" | "policy" | "generalized" | "principle" | "raw_experience") {
                    continue;
                }
                if t.starts_with("cat:") {
                    continue;
                }
                return t.to_string();
            }
        }
        if let Some(first) = experiences.first()
            && !first.category.is_empty()
            && first.category != "raw_experience"
        {
            return first.category.clone();
        }
        "general_engineering".to_string()
    }

    fn synthesize_principle(&self, experiences: &[&RawExperience], domain: &str) -> Option<String> {
        // Only synthesize when there are explicit structured signals
        // (Do:/Avoid:/Always/Never prefixes). If the experiences carry no
        // such structure, we refuse to fabricate a "principle" out of
        // keyword co-occurrence — that would be pattern theater, not
        // generalization.
        let notes: Vec<&str> = experiences.iter().map(|e| e.note.as_str()).collect();
        // 剥掉每条经验自带的极性前缀，避免拼接出 "Do: Do: xxx" / "Avoid: Avoid: xxx"
        fn strip_polarity_prefix(note: &str) -> String {
            let trimmed = note.trim();
            let lower = trimmed.to_lowercase();
            for prefix in ["do:", "avoid:", "always:", "never:", "always ", "never "] {
                if lower.starts_with(prefix) {
                    return trimmed[prefix.len()..].trim_start().to_string();
                }
            }
            trimmed.to_string()
        }
        let do_items: Vec<String> = notes
            .iter()
            .filter(|n| {
                let lower = n.trim_start().to_lowercase();
                lower.starts_with("do:") || lower.starts_with("always")
            })
            .map(|n| strip_polarity_prefix(n))
            .collect();
        let avoid_items: Vec<String> = notes
            .iter()
            .filter(|n| {
                let lower = n.trim_start().to_lowercase();
                lower.starts_with("avoid:") || lower.starts_with("never")
            })
            .map(|n| strip_polarity_prefix(n))
            .collect();

        if do_items.is_empty() && avoid_items.is_empty() {
            return None;
        }

        let mut principle = format!("In {}, ", domain.replace('_', " "));
        if !do_items.is_empty() {
            principle.push_str(&format!("Do: {}; ", do_items.join("; ")));
        }
        if !avoid_items.is_empty() {
            principle.push_str(&format!("Avoid: {}", avoid_items.join("; ")));
        }
        Some(principle)
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
        let token_score = if words_a.is_empty() || words_b.is_empty() {
            0.0
        } else {
            let intersection = words_a.intersection(&words_b).count();
            let union = words_a.union(&words_b).count();
            intersection as f64 / union as f64
        };
        // C2 CJK 友好：当至少一侧 token 数 < 4（典型 CJK / 短句 token 化稀疏）时，
        // 用 char-trigram Jaccard 兜底，并取两者最大值。阈值由调用方维持不变。
        let needs_fallback = words_a.len() < 4 || words_b.len() < 4;
        if !needs_fallback {
            return token_score;
        }
        let trig_a = char_trigrams(a);
        let trig_b = char_trigrams(b);
        if trig_a.is_empty() || trig_b.is_empty() {
            return token_score;
        }
        let inter = trig_a.intersection(&trig_b).count();
        let uni = trig_a.union(&trig_b).count();
        let trig_score = inter as f64 / uni as f64;
        token_score.max(trig_score)
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

    /// 在抽象层级基础上叠加 effective_confidence 衰减。
    /// effective_confidence 在 [0.2*conf, conf] 区间，乘到优先级上。
    /// 长期未强化的 principle 召回排序自然靠后，新 principle 维持原值。
    fn priority_for_principle(&self, principle: &GeneralizedPrinciple) -> u8 {
        let base = self.priority_for_abstraction(principle.abstraction_level) as f64;
        // 衰减系数：confidence 越低 / 越久未强化越接近 0.5
        // 保守下界 0.5，确保即使最老的 principle 也不会沉到普通 self_note 之下。
        let factor = (principle.effective_confidence().max(0.0).min(1.0) * 0.5 + 0.5).clamp(0.5, 1.0);
        (base * factor).round().clamp(0.0, 255.0) as u8
    }
}

/// 用于 CJK / 短文本兜底相似度：把字符串切成 char-trigram 集合（首尾用空格 padding 一格）。
fn char_trigrams(s: &str) -> std::collections::HashSet<String> {
    let chars: Vec<char> = format!(" {} ", s.to_lowercase()).chars().collect();
    let mut out = std::collections::HashSet::new();
    if chars.len() < 3 {
        return out;
    }
    for w in chars.windows(3) {
        out.insert(w.iter().collect::<String>());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::test_support::ENV_LOCK;

    #[test]
    fn ingest_and_generalize() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = std::env::temp_dir().join(format!(
            "rt_generalization_ingest_{}_{}.jsonl",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let mut generalizer = ExperienceGeneralizer::new();
        // 隔离潜在的跨进程持久化残留，确保仅以本测试 ingest 的三条经验为准
        generalizer.experience_buffer.clear();
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
        // Domain inference now prefers explicit tags → category → neutral
        // bucket, so "async" (the tag) wins over the keyword-guessed
        // "async_patterns" from before.
        assert_eq!(p.domain, "async");

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
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
    fn cross_domain_link_is_not_repeated_after_reload() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = std::env::temp_dir().join(format!(
            "rt_generalization_reload_{}_{}.jsonl",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let ts = Local::now().to_rfc3339();
        let principle_a = GeneralizedPrinciple {
            id: "p1".to_string(),
            principle: "Always validate inputs before processing in API handlers".to_string(),
            source_experiences: vec![],
            domain: "api_design".to_string(),
            abstraction_level: 1,
            confidence: 0.7,
            created_at: ts.clone(),
            last_reinforced: ts.clone(),
            reinforcement_count: 1,
            cross_domain_links: vec![],
        };
        let principle_b = GeneralizedPrinciple {
            id: "p2".to_string(),
            principle: "Always validate inputs before processing in async handlers".to_string(),
            source_experiences: vec![],
            domain: "async_patterns".to_string(),
            abstraction_level: 1,
            confidence: 0.7,
            created_at: ts.clone(),
            last_reinforced: ts,
            reinforcement_count: 1,
            cross_domain_links: vec![],
        };

        let generalizer = ExperienceGeneralizer::new();
        generalizer.persist_principle(&principle_a);
        generalizer.persist_principle(&principle_b);

        let mut reloaded = ExperienceGeneralizer::new();
        assert!(reloaded.try_cross_domain_link().is_some());

        let mut reloaded_again = ExperienceGeneralizer::new();
        assert!(reloaded_again.try_cross_domain_link().is_none());

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn too_few_experiences_no_generalization() {
        let mut generalizer = ExperienceGeneralizer::new();
        // 防御：跨进程持久化的 raw_experience 可能让 buffer 已经超过最小阈值，
        // 这里手动清空，确保仅以本测试 ingest 的两条经验为准。
        generalizer.experience_buffer.clear();
        generalizer.ingest_experience("test", "one note", &[], None);
        generalizer.ingest_experience("test", "two note", &[], None);
        assert!(generalizer.try_generalize().is_none());
    }

    #[test]
    fn synthesize_principle_strips_polarity_prefixes() {
        let generalizer = ExperienceGeneralizer::new();
        let exps = vec![
            RawExperience {
                id: "a".into(),
                category: "async_handling".into(),
                note: "Do: verify async results before use".into(),
                tags: vec![],
                source: None,
            },
            RawExperience {
                id: "b".into(),
                category: "async_handling".into(),
                note: "Do: check None before unwrap".into(),
                tags: vec![],
                source: None,
            },
            RawExperience {
                id: "c".into(),
                category: "async_handling".into(),
                note: "Avoid: skip validation in concurrent code".into(),
                tags: vec![],
                source: None,
            },
        ];
        let refs: Vec<&RawExperience> = exps.iter().collect();
        let principle = generalizer
            .synthesize_principle(&refs, "async_handling")
            .expect("principle");
        // 不应再出现 "Do: Do:" / "Avoid: Avoid:"
        assert!(!principle.contains("Do: Do:"), "got: {principle}");
        assert!(!principle.contains("Avoid: Avoid:"), "got: {principle}");
        // 应保留前缀剥掉之后的真实正文
        assert!(principle.contains("verify async results before use"));
        assert!(principle.contains("skip validation in concurrent code"));
    }

    #[test]
    fn infer_domain_ignores_pipeline_tags() {
        let generalizer = ExperienceGeneralizer::new();
        // 模拟 reload 后的情形：tags 中包含管线标签 raw_experience 和编码后的 cat:tool_failure
        let exp = RawExperience {
            id: "e".into(),
            category: "tool_failure".into(),
            note: "Avoid: invoking text_grep with filesystem root".into(),
            tags: vec!["raw_experience".into(), "cat:tool_failure".into()],
            source: Some("text_grep".into()),
        };
        let refs = vec![&exp];
        let domain = generalizer.infer_domain(&refs);
        assert_ne!(domain, "raw_experience", "管线标签不应被当作 domain");
        assert!(!domain.starts_with("cat:"), "cat: 编码标签不应直接作为 domain");
        assert_eq!(domain, "tool_failure");
    }
}
