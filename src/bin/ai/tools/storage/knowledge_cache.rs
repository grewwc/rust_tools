/// 会话级知识缓存管理
/// 
/// 用于管理易变知识（如项目结构、代码内容）的缓存和过期检测
/// 
/// 策略：
/// 1. 项目结构/代码信息 → 会话级缓存，30 分钟过期
/// 2. 编码规范/用户偏好 → 长期记忆，不过期
/// 3. 每次会话开始时检查缓存是否过期
/// 4. 如果过期，重新检索并更新缓存

use std::collections::HashMap;
use std::time::{SystemTime, Duration, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use crate::ai::tools::storage::knowledge_fingerprint::{KnowledgeFingerprint, FingerprintVerificationResult};
use crate::ai::tools::storage::knowledge_types::{KnowledgeMetadata, ValidationStrategy, ValidationResult, ValidationSuggestion, KnowledgeType as NewKnowledgeType};

/// 缓存的知识条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedKnowledge {
    /// 知识内容
    pub content: String,
    /// 缓存时间戳
    pub cached_at: u64,
    /// 过期时间（秒）
    pub ttl_seconds: u64,
    /// 知识类型（旧版，保留兼容）
    pub knowledge_type: KnowledgeType,
    /// 关联的上下文（如项目路径、文件列表等）
    pub context: HashMap<String, String>,
    /// 文件指纹（用于检测实际变化，仅 FileBased 类型）
    pub fingerprint: Option<KnowledgeFingerprint>,
    /// 知识元数据（新版，包含验证策略）
    pub metadata: Option<KnowledgeMetadata>,
}

/// 知识类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeType {
    /// 项目结构（易变）
    ProjectStructure,
    /// 代码内容（易变）
    CodeContent,
    /// 项目配置（中等变化）
    ProjectConfig,
    /// 编码规范（稳定）
    CodingGuideline,
    /// 用户偏好（稳定）
    UserPreference,
    /// 其他
    Other,
}

impl KnowledgeType {
    /// 获取默认 TTL（秒）
    pub fn default_ttl(&self) -> u64 {
        match self {
            KnowledgeType::ProjectStructure => 1800, // 30 分钟
            KnowledgeType::CodeContent => 1800,      // 30 分钟
            KnowledgeType::ProjectConfig => 3600,    // 60 分钟
            KnowledgeType::CodingGuideline => u64::MAX, // 永久
            KnowledgeType::UserPreference => u64::MAX,  // 永久
            KnowledgeType::Other => 3600,            // 默认 60 分钟
        }
    }
    
    /// 从分类字符串推断知识类型
    pub fn from_category(category: &str) -> Self {
        match category.to_lowercase().as_str() {
            "project_structure" | "project_info" => KnowledgeType::ProjectStructure,
            "code_content" | "code_snippet" => KnowledgeType::CodeContent,
            "project_config" | "config" => KnowledgeType::ProjectConfig,
            "coding_guideline" | "best_practice" => KnowledgeType::CodingGuideline,
            "user_preference" | "preference" => KnowledgeType::UserPreference,
            _ => KnowledgeType::Other,
        }
    }
}

/// 将新知识类型转换为旧知识类型（兼容层）
fn convert_knowledge_type(new_type: &NewKnowledgeType) -> KnowledgeType {
    match new_type {
        NewKnowledgeType::FileBased => KnowledgeType::ProjectStructure,
        NewKnowledgeType::TimeSensitive => KnowledgeType::Other,
        NewKnowledgeType::ExternalDependent => KnowledgeType::Other,
        NewKnowledgeType::SessionScoped => KnowledgeType::Other,
        NewKnowledgeType::Stable => KnowledgeType::CodingGuideline,
        NewKnowledgeType::Other => KnowledgeType::Other,
    }
}

impl CachedKnowledge {
    /// 创建新的缓存知识（基础版本，不带指纹和元数据）
    pub fn new(
        content: String,
        knowledge_type: KnowledgeType,
        context: HashMap<String, String>,
    ) -> Self {
        let ttl = knowledge_type.default_ttl();
        Self {
            content,
            cached_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs(),
            ttl_seconds: ttl,
            knowledge_type,
            context,
            fingerprint: None,
            metadata: None,
        }
    }
    
    /// 创建带指纹的缓存知识（适用于 FileBased 类型）
    pub fn new_with_fingerprint(
        content: String,
        knowledge_type: KnowledgeType,
        context: HashMap<String, String>,
        fingerprint: KnowledgeFingerprint,
    ) -> Self {
        let ttl = knowledge_type.default_ttl();
        Self {
            content,
            cached_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs(),
            ttl_seconds: ttl,
            knowledge_type,
            context,
            fingerprint: Some(fingerprint),
            metadata: None,
        }
    }
    
    /// 创建带元数据的缓存知识（推荐，支持所有验证策略）
    pub fn new_with_metadata(
        content: String,
        metadata: KnowledgeMetadata,
        fingerprint: Option<KnowledgeFingerprint>,
    ) -> Self {
        let ttl = metadata.knowledge_type.default_ttl();
        Self {
            content,
            cached_at: metadata.created_at,
            ttl_seconds: ttl,
            knowledge_type: convert_knowledge_type(&metadata.knowledge_type),
            context: metadata.context.clone(),
            fingerprint,
            metadata: Some(metadata),
        }
    }
    
    /// 检查是否过期（仅基于时间）
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        
        if self.ttl_seconds == u64::MAX {
            return false; // 永不过期
        }
        
        now > self.cached_at + self.ttl_seconds
    }
    
    /// 检查指纹是否有效（检测实际文件变化）
    pub fn verify_fingerprint(&self) -> FingerprintVerificationResult {
        if let Some(ref fp) = self.fingerprint {
            fp.verify()
        } else {
            // 没有指纹，假设有效
            FingerprintVerificationResult {
                is_valid: true,
                changed_files: Vec::new(),
                missing_files: Vec::new(),
                unchanged_count: 0,
                total_files: 0,
            }
        }
    }
    
    /// 检查是否需要刷新（综合所有验证策略）
    pub fn needs_refresh(&self) -> bool {
        // 优先使用新版元数据验证
        if let Some(ref metadata) = self.metadata {
            return !metadata.is_valid();
        }
        
        // 回退到旧版验证逻辑
        // 时间过期
        if self.is_expired() {
            return true;
        }
        
        // 指纹变化（实际文件已修改）
        if let Some(ref fp) = self.fingerprint {
            let verification = fp.verify();
            if !verification.is_valid {
                return true;
            }
        }
        
        false
    }
    
    /// 执行验证并返回详细结果
    pub fn validate(&self) -> ValidationResult {
        if let Some(ref metadata) = self.metadata {
            let is_valid = metadata.is_valid();
            
            let (validation_type, details, suggestion) = match &metadata.validation {
                ValidationStrategy::Fingerprint { files, git_commit } => {
                    let fp_result = self.fingerprint.as_ref().map(|f| f.verify()).unwrap_or_else(|| {
                        FingerprintVerificationResult {
                            is_valid: true,
                            changed_files: Vec::new(),
                            missing_files: Vec::new(),
                            unchanged_count: 0,
                            total_files: 0,
                        }
                    });
                    
                    if fp_result.is_valid {
                        (
                            "fingerprint".to_string(),
                            format!("{} files verified, {} unchanged", fp_result.total_files, fp_result.unchanged_count),
                            ValidationSuggestion::UseCache,
                        )
                    } else {
                        (
                            "fingerprint".to_string(),
                            format!("{} files changed, {} missing", fp_result.changed_files.len(), fp_result.missing_files.len()),
                            ValidationSuggestion::Refresh,
                        )
                    }
                },
                
                ValidationStrategy::TimeRange { valid_from, valid_until } => {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or(Duration::ZERO)
                        .as_secs();
                    let remaining = valid_until.saturating_sub(now);
                    
                    if is_valid {
                        (
                            "time_range".to_string(),
                            format!("Valid for {} more seconds", remaining),
                            ValidationSuggestion::UseCache,
                        )
                    } else {
                        (
                            "time_range".to_string(),
                            "Time range expired".to_string(),
                            ValidationSuggestion::Refresh,
                        )
                    }
                },
                
                ValidationStrategy::ExternalCheck { source, last_check, check_interval } => {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or(Duration::ZERO)
                        .as_secs();
                    let elapsed = now.saturating_sub(*last_check);
                    
                    if is_valid {
                        (
                            "external_check".to_string(),
                            format!("Last checked {}s ago, next check in {}s", elapsed, check_interval.saturating_sub(elapsed)),
                            ValidationSuggestion::UseCache,
                        )
                    } else {
                        (
                            "external_check".to_string(),
                            format!("Source '{}' needs recheck", source),
                            ValidationSuggestion::ExternalCheckRequired,
                        )
                    }
                },
                
                ValidationStrategy::SessionBound { session_id } => {
                    (
                        "session_bound".to_string(),
                        format!("Bound to session: {}", session_id),
                        ValidationSuggestion::UseCache,
                    )
                },
                
                ValidationStrategy::None => {
                    (
                        "none".to_string(),
                        "No validation required (stable knowledge)".to_string(),
                        ValidationSuggestion::UseCache,
                    )
                },
            };
            
            ValidationResult {
                is_valid,
                validation_type,
                details,
                suggestion,
            }
        } else {
            // 旧版验证逻辑
            let is_valid = !self.needs_refresh();
            ValidationResult {
                is_valid,
                validation_type: "legacy".to_string(),
                details: "Using legacy TTL + fingerprint validation".to_string(),
                suggestion: if is_valid {
                    ValidationSuggestion::UseCache
                } else {
                    ValidationSuggestion::Refresh
                },
            }
        }
    }
    
    /// 获取剩余生存时间（秒）
    pub fn ttl_remaining(&self) -> u64 {
        if self.ttl_seconds == u64::MAX {
            return u64::MAX;
        }
        
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        
        let elapsed = now.saturating_sub(self.cached_at);
        self.ttl_seconds.saturating_sub(elapsed)
    }
}

/// 会话知识缓存管理器
pub struct SessionKnowledgeCache {
    /// 缓存的知识
    cache: HashMap<String, CachedKnowledge>,
    /// 缓存配置文件路径
    cache_file: std::path::PathBuf,
}

impl SessionKnowledgeCache {
    /// 创建新的缓存管理器
    pub fn new() -> Self {
        let cache_file = dirs::config_dir().unwrap_or_else(|| std::path::PathBuf::from("~/.config"))
            .join("rust_tools")
            .join("knowledge_cache.json");
        
        Self {
            cache: HashMap::new(),
            cache_file,
        }
    }
    
    /// 从文件加载缓存
    pub fn load(&mut self) -> Result<(), String> {
        if !self.cache_file.exists() {
            return Ok(());
        }
        
        let content = std::fs::read_to_string(&self.cache_file)
            .map_err(|e| format!("Failed to read cache file: {}", e))?;
        
        let cache: HashMap<String, CachedKnowledge> = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse cache file: {}", e))?;
        
        // 过滤掉过期的条目
        self.cache = cache
            .into_iter()
            .filter(|(_, v)| !v.is_expired())
            .collect();
        
        Ok(())
    }
    
    /// 保存缓存到文件
    pub fn save(&self) -> Result<(), String> {
        if let Some(parent) = self.cache_file.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache dir: {}", e))?;
        }
        
        let content = serde_json::to_string_pretty(&self.cache)
            .map_err(|e| format!("Failed to serialize cache: {}", e))?;
        
        std::fs::write(&self.cache_file, content)
            .map_err(|e| format!("Failed to write cache file: {}", e))?;
        
        Ok(())
    }
    
    /// 获取缓存的知识
    pub fn get(&self, key: &str) -> Option<&CachedKnowledge> {
        self.cache.get(key).filter(|v| !v.is_expired())
    }
    
    /// 设置缓存的知识
    pub fn set(&mut self, key: String, knowledge: CachedKnowledge) {
        self.cache.insert(key, knowledge);
    }
    
    /// 清除过期的缓存
    pub fn cleanup_expired(&mut self) -> usize {
        let before = self.cache.len();
        self.cache.retain(|_, v| !v.is_expired());
        before - self.cache.len()
    }
    
    /// 清除所有易变知识的缓存
    pub fn clear_volatile(&mut self) {
        self.cache.retain(|_, v| {
            matches!(
                v.knowledge_type,
                KnowledgeType::CodingGuideline | KnowledgeType::UserPreference
            )
        });
    }
    
    /// 检查是否需要重新检索某个主题
    pub fn needs_refresh(&self, key: &str) -> bool {
        match self.get(key) {
            None => true, // 没有缓存，需要检索
            Some(entry) => entry.is_expired(), // 过期了，需要刷新
        }
    }
    
    /// 获取缓存统计信息
    pub fn stats(&self) -> CacheStats {
        let total = self.cache.len();
        let expired = self.cache.values().filter(|v| v.is_expired()).count();
        let volatile = self.cache.values().filter(|v| v.ttl_seconds != u64::MAX).count();
        let stable = total - volatile;
        
        CacheStats {
            total,
            expired,
            volatile,
            stable,
        }
    }
}

/// 缓存统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total: usize,
    pub expired: usize,
    pub volatile: usize,
    pub stable: usize,
}

impl Default for SessionKnowledgeCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 生成缓存键
pub fn make_cache_key(topic: &str, context: &HashMap<String, String>) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    topic.hash(&mut hasher);
    
    // 对 context 排序后哈希，保证一致性
    let mut sorted_context: Vec<_> = context.iter().collect();
    sorted_context.sort_by(|a, b| a.0.cmp(b.0));
    for (k, v) in sorted_context {
        k.hash(&mut hasher);
        v.hash(&mut hasher);
    }
    
    format!("{}_{}", topic, hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_knowledge_type_ttl() {
        assert_eq!(KnowledgeType::ProjectStructure.default_ttl(), 1800);
        assert_eq!(KnowledgeType::CodingGuideline.default_ttl(), u64::MAX);
    }
    
    #[test]
    fn test_cache_expiry() {
        let mut context = HashMap::new();
        context.insert("project".to_string(), "rust_tools".to_string());
        
        let knowledge = CachedKnowledge::new(
            "test content".to_string(),
            KnowledgeType::ProjectStructure,
            context,
        );
        
        // 刚创建，不应该过期
        assert!(!knowledge.is_expired());
        assert!(knowledge.ttl_remaining() <= 1800);
    }
}
