/// 通用知识类型与验证策略
/// 
/// 解决不同类型知识的缓存验证问题：
/// 1. 文件/代码类 → 文件指纹验证
/// 2. 时间敏感类 → 时间范围验证
/// 3. 外部依赖类 → 外部状态检查
/// 4. 会话相关类 → 会话绑定
/// 5. 稳定知识类 → 永不过期

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH, Duration};

/// 知识类型（决定验证策略）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeType {
    /// 基于文件的知识（项目结构、代码内容）
    /// 验证：文件指纹
    FileBased,
    
    /// 时间敏感知识（天气、日期、新闻）
    /// 验证：时间范围
    TimeSensitive,
    
    /// 外部依赖知识（API 状态、股票价格）
    /// 验证：外部检查
    ExternalDependent,
    
    /// 会话级知识（对话历史、临时状态）
    /// 验证：会话绑定
    SessionScoped,
    
    /// 稳定知识（编码规范、最佳实践）
    /// 验证：永不过期
    Stable,
    
    /// 其他（使用默认 TTL）
    Other,
}

impl KnowledgeType {
    /// 获取默认 TTL（秒）
    pub fn default_ttl(&self) -> u64 {
        match self {
            KnowledgeType::FileBased => 1800,        // 30 分钟
            KnowledgeType::TimeSensitive => 300,     // 5 分钟
            KnowledgeType::ExternalDependent => 600, // 10 分钟
            KnowledgeType::SessionScoped => u64::MAX, // 会话结束前有效
            KnowledgeType::Stable => u64::MAX,       // 永久
            KnowledgeType::Other => 3600,            // 60 分钟
        }
    }
    
    /// 从描述推断知识类型
    pub fn infer_from_description(desc: &str) -> Self {
        let desc_lower = desc.to_lowercase();
        
        // 时间敏感关键词
        if desc_lower.contains("天气") || desc_lower.contains("weather") ||
           desc_lower.contains("今天") || desc_lower.contains("today") ||
           desc_lower.contains("明天") || desc_lower.contains("tomorrow") ||
           desc_lower.contains("现在") || desc_lower.contains("now") ||
           desc_lower.contains("当前") || desc_lower.contains("current") {
            return KnowledgeType::TimeSensitive;
        }
        
        // 外部依赖关键词
        if desc_lower.contains("api") || desc_lower.contains("状态") ||
           desc_lower.contains("status") || desc_lower.contains("价格") ||
           desc_lower.contains("price") || desc_lower.contains("股票") ||
           desc_lower.contains("stock") {
            return KnowledgeType::ExternalDependent;
        }
        
        // 文件/代码关键词
        if desc_lower.contains("项目结构") || desc_lower.contains("project structure") ||
           desc_lower.contains("代码") || desc_lower.contains("code") ||
           desc_lower.contains("文件") || desc_lower.contains("file") {
            return KnowledgeType::FileBased;
        }
        
        // 稳定知识关键词
        if desc_lower.contains("规范") || desc_lower.contains("guideline") ||
           desc_lower.contains("最佳实践") || desc_lower.contains("best practice") ||
           desc_lower.contains("原则") || desc_lower.contains("principle") {
            return KnowledgeType::Stable;
        }
        
        KnowledgeType::Other
    }
}

/// 验证策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStrategy {
    /// 文件指纹验证
    Fingerprint {
        files: Vec<String>,
        git_commit: Option<String>,
    },
    
    /// 时间范围验证
    TimeRange {
        valid_from: u64,
        valid_until: u64,
    },
    
    /// 外部状态验证
    ExternalCheck {
        source: String,
        last_check: u64,
        check_interval: u64,
    },
    
    /// 会话绑定
    SessionBound {
        session_id: String,
    },
    
    /// 无验证（永不过期）
    None,
}

/// 知识元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeMetadata {
    /// 知识类型
    pub knowledge_type: KnowledgeType,
    
    /// 验证策略
    pub validation: ValidationStrategy,
    
    /// 创建时间
    pub created_at: u64,
    
    /// 最后验证时间
    pub last_verified: Option<u64>,
    
    /// 验证失败次数（用于退避策略）
    pub validation_failures: u32,
    
    /// 额外上下文
    pub context: HashMap<String, String>,
    
    /// 人类可读的描述
    pub description: Option<String>,
}

impl KnowledgeMetadata {
    /// 创建新的元数据（基于知识类型）
    pub fn new(
        knowledge_type: KnowledgeType,
        context: HashMap<String, String>,
        description: Option<String>,
    ) -> Self {
        let now = Self::current_timestamp();
        
        let validation = match &knowledge_type {
            KnowledgeType::FileBased => ValidationStrategy::Fingerprint {
                files: Vec::new(),
                git_commit: None,
            },
            
            KnowledgeType::TimeSensitive => {
                let ttl = knowledge_type.default_ttl();
                ValidationStrategy::TimeRange {
                    valid_from: now,
                    valid_until: now + ttl,
                }
            },
            
            KnowledgeType::ExternalDependent => {
                let check_interval = 60; // 1 分钟检查一次
                ValidationStrategy::ExternalCheck {
                    source: context.get("source").cloned().unwrap_or_default(),
                    last_check: now,
                    check_interval,
                }
            },
            
            KnowledgeType::SessionScoped => ValidationStrategy::SessionBound {
                session_id: context.get("session_id").cloned().unwrap_or_else(|| {
                    format!("session_{}", now)
                }),
            },
            
            KnowledgeType::Stable | KnowledgeType::Other => ValidationStrategy::None,
        };
        
        Self {
            knowledge_type,
            validation,
            created_at: now,
            last_verified: Some(now),
            validation_failures: 0,
            context,
            description,
        }
    }
    
    /// 检查是否有效
    pub fn is_valid(&self) -> bool {
        match &self.validation {
            ValidationStrategy::Fingerprint { .. } => {
                // 文件指纹需要外部验证
                true // 假设有效，实际验证在外部进行
            },
            
            ValidationStrategy::TimeRange { valid_until, .. } => {
                Self::current_timestamp() < *valid_until
            },
            
            ValidationStrategy::ExternalCheck { 
                last_check, 
                check_interval,
                ..
            } => {
                let elapsed = Self::current_timestamp().saturating_sub(*last_check);
                elapsed < *check_interval
            },
            
            ValidationStrategy::SessionBound { .. } => {
                // 会话绑定需要外部验证会话是否活跃
                true // 假设有效
            },
            
            ValidationStrategy::None => true,
        }
    }
    
    /// 获取剩余有效时间（秒）
    pub fn time_remaining(&self) -> Option<u64> {
        match &self.validation {
            ValidationStrategy::TimeRange { valid_until, .. } => {
                let now = Self::current_timestamp();
                if now >= *valid_until {
                    Some(0)
                } else {
                    Some(valid_until.saturating_sub(now))
                }
            },
            
            ValidationStrategy::ExternalCheck { 
                last_check, 
                check_interval,
                ..
            } => {
                let elapsed = Self::current_timestamp().saturating_sub(*last_check);
                Some(check_interval.saturating_sub(elapsed))
            },
            
            _ => None, // 其他类型没有时间概念
        }
    }
    
    /// 更新验证时间
    pub fn mark_verified(&mut self) {
        self.last_verified = Some(Self::current_timestamp());
        self.validation_failures = 0;
        
        // 更新时间范围
        if let ValidationStrategy::TimeRange { valid_from, valid_until } = &mut self.validation {
            let ttl = self.knowledge_type.default_ttl();
            *valid_from = Self::current_timestamp();
            *valid_until = *valid_from + ttl;
        }
        
        // 更新外部检查时间
        if let ValidationStrategy::ExternalCheck { last_check, .. } = &mut self.validation {
            *last_check = Self::current_timestamp();
        }
    }
    
    /// 记录验证失败
    pub fn mark_validation_failed(&mut self) {
        self.validation_failures += 1;
    }
    
    /// 获取当前时间戳
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs()
    }
}

/// 验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// 是否有效
    pub is_valid: bool,
    
    /// 验证类型
    pub validation_type: String,
    
    /// 详细信息
    pub details: String,
    
    /// 建议操作
    pub suggestion: ValidationSuggestion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSuggestion {
    /// 使用缓存
    UseCache,
    
    /// 刷新缓存
    Refresh,
    
    /// 需要外部检查
    ExternalCheckRequired,
    
    /// 需要用户确认
    UserConfirmRequired,
}

/// 为时间敏感知识创建元数据
pub fn create_time_sensitive_metadata(
    description: &str,
    context: HashMap<String, String>,
    custom_ttl: Option<u64>,
) -> KnowledgeMetadata {
    let mut metadata = KnowledgeMetadata::new(
        KnowledgeType::TimeSensitive,
        context,
        Some(description.to_string()),
    );
    
    // 自定义 TTL
    if let Some(ttl) = custom_ttl {
        if let ValidationStrategy::TimeRange { valid_from, valid_until } = &mut metadata.validation {
            *valid_until = *valid_from + ttl;
        }
    }
    
    metadata
}

/// 为外部依赖知识创建元数据
pub fn create_external_metadata(
    source: &str,
    description: &str,
    context: HashMap<String, String>,
    check_interval: Option<u64>,
) -> KnowledgeMetadata {
    let mut ctx = context.clone();
    ctx.insert("source".to_string(), source.to_string());
    
    let mut metadata = KnowledgeMetadata::new(
        KnowledgeType::ExternalDependent,
        ctx,
        Some(description.to_string()),
    );
    
    // 自定义检查间隔
    if let Some(interval) = check_interval {
        if let ValidationStrategy::ExternalCheck { check_interval, .. } = &mut metadata.validation {
            *check_interval = interval;
        }
    }
    
    metadata
}

/// 为会话知识创建元数据
pub fn create_session_metadata(
    session_id: &str,
    description: &str,
    context: HashMap<String, String>,
) -> KnowledgeMetadata {
    let mut ctx = context.clone();
    ctx.insert("session_id".to_string(), session_id.to_string());
    
    KnowledgeMetadata::new(
        KnowledgeType::SessionScoped,
        ctx,
        Some(description.to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_knowledge_type_inference() {
        assert_eq!(
            KnowledgeType::infer_from_description("今天天气怎么样"),
            KnowledgeType::TimeSensitive
        );
        
        assert_eq!(
            KnowledgeType::infer_from_description("项目结构"),
            KnowledgeType::FileBased
        );
        
        assert_eq!(
            KnowledgeType::infer_from_description("API 状态"),
            KnowledgeType::ExternalDependent
        );
        
        assert_eq!(
            KnowledgeType::infer_from_description("编码规范"),
            KnowledgeType::Stable
        );
    }
    
    #[test]
    fn test_time_sensitive_validation() {
        let mut ctx = HashMap::new();
        let mut metadata = create_time_sensitive_metadata(
            "今天天气",
            ctx.clone(),
            Some(300), // 5 分钟
        );
        
        // 刚创建，应该有效
        assert!(metadata.is_valid());
        
        // 剩余时间应该在 5 分钟左右
        let remaining = metadata.time_remaining().unwrap();
        assert!(remaining <= 300);
    }
}
