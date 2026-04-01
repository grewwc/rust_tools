/// 知识更新管理工具
/// 
/// 提供智能的知识检索和更新策略：
/// 1. 检查缓存是否过期
/// 2. 如果过期，重新检索并更新
/// 3. 根据知识类型决定更新策略

use std::collections::HashMap;
use serde_json::Value;
use crate::ai::tools::storage::knowledge_cache::{
    SessionKnowledgeCache, CachedKnowledge, KnowledgeType, make_cache_key
};
use crate::ai::tools::storage::knowledge_fingerprint::{
    create_fingerprint_for_topic, KnowledgeFingerprint
};
use crate::ai::tools::storage::knowledge_types::{
    KnowledgeType as NewKnowledgeType, KnowledgeMetadata, ValidationStrategy,
    create_time_sensitive_metadata, create_external_metadata, create_session_metadata
};
use crate::ai::tools::storage::memory_store::MemoryStore;
use crate::commonw::configw;

/// 检查并获取知识（自动处理缓存和过期）
/// 
/// # Arguments
/// * `topic` - 知识主题（如 "project_structure", "coding_guidelines"）
/// * `context` - 上下文信息（如项目路径、相关文件等）
/// * `force_refresh` - 是否强制刷新
/// 
/// # Returns
/// * `Ok(String)` - 知识内容
/// * `Err(String)` - 错误信息
pub fn get_knowledge_with_cache(
    topic: &str,
    context: &HashMap<String, String>,
    force_refresh: bool,
) -> Result<String, String> {
    let mut cache = SessionKnowledgeCache::new();
    cache.load().ok(); // 加载失败则使用空缓存
    
    let cache_key = make_cache_key(topic, context);
    
    // 检查是否需要刷新
    let needs_refresh = force_refresh || cache.needs_refresh(&cache_key);
    
    if needs_refresh {
        // 重新检索知识
        let knowledge = retrieve_knowledge_for_topic(topic, context)?;
        
        // 推断知识类型（支持新知识类型）
        let description = context.get("description").map(|s| s.as_str()).unwrap_or(topic);
        let inferred_type = NewKnowledgeType::infer_from_description(description);
        
        // 创建元数据（包含验证策略）
        let mut metadata = KnowledgeMetadata::new(
            inferred_type.clone(),
            context.clone(),
            Some(description.to_string()),
        );
        
        // 对于 FileBased 类型，尝试创建指纹
        let fingerprint = if matches!(inferred_type, NewKnowledgeType::FileBased) {
            create_fingerprint_for_topic(topic, context).ok()
        } else {
            None
        };
        
        let cached = CachedKnowledge::new_with_metadata(
            knowledge.clone(),
            metadata,
            fingerprint,
        );
        
        cache.set(cache_key, cached);
        if let Err(e) = cache.save() {
            eprintln!("knowledge_cache save failed: {}", e);
        }
        
        Ok(knowledge)
    } else {
        // 使用缓存
        match cache.get(&cache_key) {
            Some(entry) => {
                // 额外检查指纹（双重验证）
                if entry.needs_refresh() {
                    // 指纹变化，重新检索
                    let knowledge = retrieve_knowledge_for_topic(topic, context)?;
                    let knowledge_type = KnowledgeType::from_category(
                        context.get("category").map(|s| s.as_str()).unwrap_or("other")
                    );
                    
                    let fingerprint = create_fingerprint_for_topic(topic, context).ok();
                    
                    let cached = if let Some(fp) = fingerprint {
                        CachedKnowledge::new_with_fingerprint(
                            knowledge.clone(),
                            knowledge_type,
                            context.clone(),
                            fp,
                        )
                    } else {
                        CachedKnowledge::new(
                            knowledge.clone(),
                            knowledge_type,
                            context.clone(),
                        )
                    };
                    
                    cache.set(cache_key, cached);
                    if let Err(e) = cache.save() {
                        eprintln!("knowledge_cache save failed: {}", e);
                    }
                    
                    Ok(knowledge)
                } else {
                    Ok(entry.content.clone())
                }
            },
            None => {
                // 缓存未命中，重新检索并写回缓存
                let knowledge = retrieve_knowledge_for_topic(topic, context)?;
                let description = context.get("description").map(|s| s.as_str()).unwrap_or(topic);
                let inferred_type = NewKnowledgeType::infer_from_description(description);
                let metadata = KnowledgeMetadata::new(
                    inferred_type.clone(),
                    context.clone(),
                    Some(description.to_string()),
                );
                let fingerprint = if matches!(inferred_type, NewKnowledgeType::FileBased) {
                    create_fingerprint_for_topic(topic, context).ok()
                } else {
                    None
                };
                let cached = CachedKnowledge::new_with_metadata(
                    knowledge.clone(),
                    metadata,
                    fingerprint,
                );
                cache.set(cache_key, cached);
                if let Err(e) = cache.save() {
                    eprintln!("knowledge_cache save failed: {}", e);
                }
                Ok(knowledge)
            }
        }
    }
}

/// 为特定主题检索知识
fn retrieve_knowledge_for_topic(
    topic: &str,
    context: &HashMap<String, String>,
) -> Result<String, String> {
    let store = MemoryStore::from_env_or_config();
    
    // 根据主题和上下文构建查询
    let query = context.get("query").map(|s| s.as_str()).unwrap_or(topic);
    let category = context.get("category");
    let limit = context.get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    
    // 从 memory 中检索（使用简单关键词搜索）
    // 如果有 category，将其加入查询
    let search_query = if let Some(cat) = category {
        format!("{} {}", query, cat)
    } else {
        query.to_string()
    };
    
    let entries = store.search(&search_query, limit)?;
    
    if entries.is_empty() {
        return Ok(format!("[No knowledge found for: {}]", topic));
    }
    
    // 格式化结果
    let mut result = String::new();
    result.push_str(&format!("=== {} ===\n", topic.to_uppercase().replace('_', " ")));
    
    for (idx, entry) in entries.iter().enumerate() {
        result.push_str(&format!(
            "\n{}. [{}] {}\n",
            idx + 1,
            entry.category,
            entry.note
        ));
        
        if !entry.tags.is_empty() {
            result.push_str(&format!("   Tags: {}\n", entry.tags.join(", ")));
        }
        
        if let Some(source) = &entry.source {
            result.push_str(&format!("   Source: {}\n", source));
        }
        
        result.push_str(&format!("   Cached: {}\n", entry.timestamp));
    }
    
    Ok(result)
}

/// 清除易变知识的缓存
/// 
/// 当项目结构发生变化时调用此函数
pub fn clear_volatile_knowledge() -> Result<String, String> {
    let mut cache = SessionKnowledgeCache::new();
    cache.load()?;
    
    let stats = cache.stats();
    cache.clear_volatile();
    cache.save()?;
    
    Ok(format!(
        "Cleared volatile knowledge cache. Before: {} total ({} volatile, {} stable), After: {} remaining",
        stats.total,
        stats.volatile,
        stats.stable,
        cache.stats().total
    ))
}

/// 获取缓存统计信息
pub fn get_cache_stats() -> Result<String, String> {
    let cache = SessionKnowledgeCache::new();
    let cache = {
        let mut c = cache;
        c.load().map_err(|e| format!("Failed to load cache: {}", e))?;
        c
    };
    
    let stats = cache.stats();
    
    Ok(format!(
        "Knowledge Cache Stats:\n\
         - Total entries: {}\n\
         - Volatile (time-limited): {}\n\
         - Stable (permanent): {}\n\
         - Expired (pending cleanup): {}",
        stats.total,
        stats.volatile,
        stats.stable,
        stats.expired
    ))
}

/// 执行工具：knowledge_cache_manage
pub fn execute_knowledge_cache_manage(args: &Value) -> Result<String, String> {
    let action = args["action"].as_str().unwrap_or("stats");
    
    match action {
        "clear_volatile" => clear_volatile_knowledge(),
        "stats" => get_cache_stats(),
        "refresh" => {
            // 强制刷新特定主题
            let topic = args["topic"].as_str()
                .ok_or("topic is required for refresh action")?;
            
            let mut context = HashMap::new();
            if let Some(cat) = args["category"].as_str() {
                context.insert("category".to_string(), cat.to_string());
            }
            if let Some(query) = args["query"].as_str() {
                context.insert("query".to_string(), query.to_string());
            }
            if let Some(limit) = args["limit"].as_str() {
                context.insert("limit".to_string(), limit.to_string());
            }
            
            let content = get_knowledge_with_cache(topic, &context, true)?;
            Ok(format!("Refreshed knowledge for '{}':\n{}", topic, content))
        }
        _ => Err(format!("Unknown action: {}. Use: stats, clear_volatile, refresh", action)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_key_generation() {
        let mut ctx1 = HashMap::new();
        ctx1.insert("project".to_string(), "rust_tools".to_string());
        
        let mut ctx2 = HashMap::new();
        ctx2.insert("project".to_string(), "rust_tools".to_string());
        
        // 相同的上下文应该生成相同的键
        assert_eq!(
            make_cache_key("project_structure", &ctx1),
            make_cache_key("project_structure", &ctx2)
        );
        
        // 不同的主题应该生成不同的键
        let key1 = make_cache_key("project_structure", &ctx1);
        let key2 = make_cache_key("coding_guidelines", &ctx1);
        assert_ne!(key1, key2);
    }
}
