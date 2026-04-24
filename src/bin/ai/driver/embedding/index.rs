use std::path::PathBuf;

use dirs::config_dir;
use rust_tools::commonw::{FastMap, FastSet};

use crate::ai::knowledge::storage::vector_store::{VectorEntry, VectorStore};

use super::document::{SkillEmbeddingDocument, SkillEmbeddingDocumentSection};

const SKILL_ROUTING_CATEGORY: &str = "skill-routing";

#[derive(Debug, Clone)]
pub struct SkillEmbeddingHit {
    pub skill_name: String,
    pub score: f64,
    pub identity_score: f64,
    pub capability_score: f64,
    pub behavior_score: f64,
}

pub struct SkillEmbeddingIndex {
    store: VectorStore,
    section_count: usize,
}

impl SkillEmbeddingIndex {
    pub fn build(documents: &[SkillEmbeddingDocument]) -> Result<Self, String> {
        let store = VectorStore::with_global_provider(&default_skill_index_path())?;
        sync_documents(&store, documents)?;
        Ok(Self {
            store,
            section_count: documents.len() * 3,
        })
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SkillEmbeddingHit>, String> {
        let query_embedding = self.store.embed_text(query)?;
        let raw_hits = self.store.semantic_search(
            &query_embedding,
            self.section_count.max(limit * 3),
            Some(SKILL_ROUTING_CATEGORY),
        )?;
        let mut by_skill: FastMap<String, SkillEmbeddingHit> = FastMap::default();
        for (entry, score) in raw_hits {
            let Some((skill_name, section)) = parse_skill_entry(&entry) else {
                continue;
            };
            let score = score as f64;
            let item = by_skill
                .entry(skill_name.clone())
                .or_insert_with(|| SkillEmbeddingHit {
                    skill_name,
                    score: 0.0,
                    identity_score: 0.0,
                    capability_score: 0.0,
                    behavior_score: 0.0,
                });
            match section {
                SkillEmbeddingDocumentSection::Identity => {
                    item.identity_score = item.identity_score.max(score)
                }
                SkillEmbeddingDocumentSection::Capability => {
                    item.capability_score = item.capability_score.max(score)
                }
                SkillEmbeddingDocumentSection::Behavior => {
                    item.behavior_score = item.behavior_score.max(score)
                }
            }
            item.score = item
                .identity_score
                .max(item.capability_score * 0.9)
                .max(item.behavior_score * 0.75);
        }
        let mut hits = by_skill.into_values().collect::<Vec<_>>();
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        if hits.len() > limit {
            hits.truncate(limit);
        }
        Ok(hits)
    }
}

fn sync_documents(
    store: &VectorStore,
    documents: &[SkillEmbeddingDocument],
) -> Result<(), String> {
    let mut desired_ids: FastSet<String> = FastSet::default();
    for doc in documents {
        for (section, text) in doc.sections() {
            let id = skill_section_id(doc, section);
            desired_ids.insert(id.clone());
            let entry = build_vector_entry(doc, section, text);
            let needs_upsert = store
                .get(&id)?
                .map(|existing| !entry_matches_document(&existing, doc))
                .unwrap_or(true);
            if needs_upsert {
                let embedding = store.embed_text(&entry.content)?;
                store.upsert(VectorEntry { embedding, ..entry })?;
            }
        }
    }

    for id in store.list_ids()? {
        if id.starts_with("skill-routing:")
            && !desired_ids.contains(&id)
        {
            let _ = store.delete(&id)?;
        }
    }
    Ok(())
}

fn default_skill_index_path() -> PathBuf {
    config_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("rust_tools")
        .join("cache")
        .join("ai")
        .join("skill_routing_vectors")
}

fn build_vector_entry(
    doc: &SkillEmbeddingDocument,
    section: SkillEmbeddingDocumentSection,
    text: &str,
) -> VectorEntry {
    let section_name = section_name(section);
    let content = if text.trim().is_empty() {
        doc.skill_name.clone()
    } else {
        text.to_string()
    };
    VectorEntry {
        id: skill_section_id(doc, section),
        content,
        category: SKILL_ROUTING_CATEGORY.to_string(),
        tags: vec![
            format!("skill:{}", doc.skill_name),
            format!("source:{}", doc.source_key),
            format!("hash:{}", doc.source_hash),
            format!("section:{section_name}"),
        ],
        embedding: Vec::new(),
        timestamp: 0,
    }
}

fn entry_matches_document(entry: &VectorEntry, doc: &SkillEmbeddingDocument) -> bool {
    entry.category == SKILL_ROUTING_CATEGORY
        && entry
            .tags
            .iter()
            .any(|tag| tag == &format!("hash:{}", doc.source_hash))
}

fn parse_skill_entry(entry: &VectorEntry) -> Option<(String, SkillEmbeddingDocumentSection)> {
    let skill_name = entry
        .tags
        .iter()
        .find_map(|tag| tag.strip_prefix("skill:").map(ToOwned::to_owned))?;
    let section = entry.tags.iter().find_map(|tag| {
        let name = tag.strip_prefix("section:")?;
        match name {
            "identity" => Some(SkillEmbeddingDocumentSection::Identity),
            "capability" => Some(SkillEmbeddingDocumentSection::Capability),
            "behavior" => Some(SkillEmbeddingDocumentSection::Behavior),
            _ => None,
        }
    })?;
    Some((skill_name, section))
}

fn skill_section_id(doc: &SkillEmbeddingDocument, section: SkillEmbeddingDocumentSection) -> String {
    format!("skill-routing:{}:{}", doc.source_key, section_name(section))
}

fn section_name(section: SkillEmbeddingDocumentSection) -> &'static str {
    match section {
        SkillEmbeddingDocumentSection::Identity => "identity",
        SkillEmbeddingDocumentSection::Capability => "capability",
        SkillEmbeddingDocumentSection::Behavior => "behavior",
    }
}
