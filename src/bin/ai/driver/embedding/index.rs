use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use dirs::config_dir;
use rust_tools::cw::{SkipMap, SkipSet};
use rust_tools::cw::LruCache;

use crate::ai::knowledge::storage::vector_store::{VectorEntry, VectorStore};

use super::document::{SkillEmbeddingDocument, SkillEmbeddingDocumentSection};

const SKILL_ROUTING_CATEGORY: &str = "skill-routing";
const QUERY_EMBEDDING_CACHE_LIMIT: usize = 32;

/// 短词条 query embedding 缓存：discover_skills / skill_match 每个 turn 都会查。
/// 用项目自带的 cw::LruCache（FxHashMap O(1) 查询 + 双向链表 O(1) 淘汰），
/// 替代之前 Vec 线性扫描 + remove(0) 双 O(n) 实现。
/// Value 用 Arc 避免命中时 clone 整个 f32 向量。
static QUERY_EMBEDDING_CACHE: LazyLock<Mutex<LruCache<String, Arc<Vec<f32>>>>> =
    LazyLock::new(|| Mutex::new(LruCache::new(QUERY_EMBEDDING_CACHE_LIMIT)));

/// 同时缓存最近若干 skills_hash 对应的 VectorStore + snapshot。
/// 单槽缓存在 user 来回切 agent（每个 agent 暴露不同 skill 子集）时会
/// thrash：一旦 hash 变了就要重建 SQLite + embedding。改为小型 LRU
/// 后，常用的 4 套 skill 子集都能命中而无需重建。
const CACHED_INDEX_CAPACITY: usize = 4;

#[derive(Clone)]
struct CachedIndex {
    store: Arc<VectorStore>,
    section_count: usize,
    snapshot: Arc<Vec<(VectorEntry, SkillEmbeddingDocumentSection, String)>>,
}

static CACHED_INDEX: LazyLock<Mutex<LruCache<String, CachedIndex>>> =
    LazyLock::new(|| Mutex::new(LruCache::new(CACHED_INDEX_CAPACITY)));

#[derive(Debug, Clone)]
pub struct SkillEmbeddingHit {
    pub skill_name: String,
    pub score: f64,
    pub identity_score: f64,
    pub capability_score: f64,
    pub behavior_score: f64,
}

pub struct SkillEmbeddingIndex {
    store: Arc<VectorStore>,
    section_count: usize,
    snapshot: Arc<Vec<(VectorEntry, SkillEmbeddingDocumentSection, String)>>,
}

impl SkillEmbeddingIndex {
    pub fn build(documents: &[SkillEmbeddingDocument]) -> Result<Self, String> {
        let skills_hash = compute_skills_hash(documents);
        let section_count = documents.len() * 3;

        if let Ok(mut cache) = CACHED_INDEX.lock()
            && let Some(cached) = cache.get_ref(&skills_hash)
        {
            return Ok(Self {
                store: Arc::clone(&cached.store),
                section_count: cached.section_count,
                snapshot: Arc::clone(&cached.snapshot),
            });
        }

        let store = Arc::new(VectorStore::with_global_provider(
            &default_skill_index_path(),
        )?);
        sync_documents(&store, documents)?;
        let snapshot = Arc::new(load_snapshot(&store)?);

        if let Ok(mut cache) = CACHED_INDEX.lock() {
            cache.put(
                skills_hash,
                CachedIndex {
                    store: Arc::clone(&store),
                    section_count,
                    snapshot: Arc::clone(&snapshot),
                },
            );
        }

        Ok(Self {
            store,
            section_count,
            snapshot,
        })
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SkillEmbeddingHit>, String> {
        if self.snapshot.is_empty() {
            return Ok(Vec::new());
        }
        let trimmed = query.trim();
        if trimmed.chars().count() < 4 {
            return Ok(Vec::new());
        }
        let expanded_query = expand_query_bilingual(trimmed);
        let query_embedding = cached_embed(&self.store, &expanded_query)?;

        let mut by_skill: SkipMap<String, SkillEmbeddingHit> = SkipMap::default();
        for (entry, section, skill_name) in self.snapshot.iter() {
            let sim = cosine_similarity_f32(&query_embedding, &entry.embedding);
            let score = sim as f64;
            let item = by_skill
                .entry(skill_name.clone())
                .or_insert_with(|| SkillEmbeddingHit {
                    skill_name: skill_name.clone(),
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
                .max(item.capability_score * 0.3)
                .max(item.behavior_score);
        }
        let mut hits = by_skill.drain().map(|(_, v)| v).collect::<Vec<_>>();
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        if hits.len() > limit {
            hits.truncate(limit);
        }
        Ok(hits)
    }
}

fn compute_skills_hash(documents: &[SkillEmbeddingDocument]) -> String {
    let mut combined = String::new();
    for doc in documents {
        combined.push_str(&doc.source_hash);
        combined.push('\n');
    }
    combined
}

fn load_snapshot(
    store: &VectorStore,
) -> Result<Vec<(VectorEntry, SkillEmbeddingDocumentSection, String)>, String> {
    let mut snapshot = Vec::new();
    for id in store.list_ids()? {
        if !id.starts_with("skill-routing:") {
            continue;
        }
        let Some(entry) = store.get(&id)? else {
            continue;
        };
        if entry.embedding.is_empty() {
            continue;
        }
        let Some((skill_name, section)) = parse_skill_entry(&entry) else {
            continue;
        };
        snapshot.push((entry, section, skill_name));
    }
    Ok(snapshot)
}

fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

fn sync_documents(store: &VectorStore, documents: &[SkillEmbeddingDocument]) -> Result<(), String> {
    let mut desired_ids: SkipSet<String> = SkipSet::default();
    let mut texts_to_embed: Vec<(String, VectorEntry)> = Vec::new();

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
                texts_to_embed.push((entry.content.clone(), entry));
            }
        }
    }

    if !texts_to_embed.is_empty() {
        let texts: Vec<String> = texts_to_embed.iter().map(|(t, _)| t.clone()).collect();
        let embeddings = store.embed_texts(&texts)?;
        for ((_, entry), embedding) in texts_to_embed.into_iter().zip(embeddings.into_iter()) {
            store.upsert(VectorEntry { embedding, ..entry })?;
        }
    }

    for id in store.list_ids()? {
        if id.starts_with("skill-routing:") && !desired_ids.contains(&id) {
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

fn skill_section_id(
    doc: &SkillEmbeddingDocument,
    section: SkillEmbeddingDocumentSection,
) -> String {
    format!("skill-routing:{}:{}", doc.source_key, section_name(section))
}

fn section_name(section: SkillEmbeddingDocumentSection) -> &'static str {
    match section {
        SkillEmbeddingDocumentSection::Identity => "identity",
        SkillEmbeddingDocumentSection::Capability => "capability",
        SkillEmbeddingDocumentSection::Behavior => "behavior",
    }
}

fn cached_embed(store: &VectorStore, text: &str) -> Result<Vec<f32>, String> {
    let cache_key = normalize_cache_key(text);
    if let Ok(mut cache) = QUERY_EMBEDDING_CACHE.lock()
        && let Some(arc) = cache.get_ref(&cache_key)
    {
        return Ok((*arc).clone());
    }

    let embedding = store.embed_text(&cache_key)?;
    let arc = Arc::new(embedding);

    if let Ok(mut cache) = QUERY_EMBEDDING_CACHE.lock() {
        cache.put(cache_key, Arc::clone(&arc));
    }

    Ok((*arc).clone())
}

fn normalize_cache_key(text: &str) -> String {
    let normalized: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() > 256 {
        normalized.chars().take(256).collect()
    } else {
        normalized
    }
}

fn expand_query_bilingual(query: &str) -> String {
    static BILINGUAL_MAP: &[(&str, &str)] = &[
        ("review", "代码审查 code review"),
        ("审查", "review code review"),
        ("代码审查", "review code review"),
        ("调试", "debug debugging"),
        ("debug", "调试 debugging"),
        ("重构", "refactor restructure"),
        ("refactor", "重构 restructure"),
        ("修复", "fix repair bug"),
        ("fix", "修复 repair bug"),
        ("报错", "error compile error runtime error"),
        ("error", "报错 错误"),
        ("panic", "崩溃 crash panic"),
        ("崩溃", "crash panic"),
        ("crash", "崩溃 panic"),
        ("测试", "test unit test"),
        ("test", "测试 unit test"),
        ("性能", "performance optimization"),
        ("performance", "性能 优化"),
        ("优化", "optimize performance improvement"),
        ("安全", "security vulnerability"),
        ("security", "安全 漏洞"),
        ("部署", "deploy deployment CI/CD"),
        ("deploy", "部署 deployment"),
        ("文档", "documentation docs"),
        ("documentation", "文档"),
        ("生成", "generate create"),
        ("generate", "生成 创建"),
        ("分析", "analyze analysis"),
        ("analyze", "分析"),
    ];

    let truncated: String = if query.chars().count() > 120 {
        query.chars().take(120).collect()
    } else {
        query.to_string()
    };
    let lower = truncated.to_lowercase();
    let mut expansions: Vec<&str> = Vec::new();

    for (keyword, expansion) in BILINGUAL_MAP {
        if lower.contains(keyword) {
            expansions.push(expansion);
            if expansions.len() >= 3 {
                break;
            }
        }
    }

    if expansions.is_empty() {
        return truncated;
    }

    format!("{} {}", truncated, expansions.join(" "))
}
