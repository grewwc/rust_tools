use crate::ai::tools::storage::knowledge_fingerprint::{
    FingerprintVerificationResult, KnowledgeFingerprint,
};
use crate::ai::tools::storage::knowledge_types::{
    KnowledgeMetadata, KnowledgeType as NewKnowledgeType, ValidationResult, ValidationStrategy,
    ValidationSuggestion,
};
use crate::commonw::utils::get_config_dir;

/// Session-level knowledge cache management
///
/// Manages caching and expiry detection for volatile knowledge, such as
/// project structure and code content.
///
/// Strategy:
/// 1. Project structure / code info -> session-level cache, 30-minute expiry
/// 2. Coding guidelines / user preferences -> long-term memory, never expires
/// 3. At the start of each session, check whether the cache has expired
/// 4. If expired, re-fetch and update the cache
use rust_tools::cw::SkipMap;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A cached knowledge entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedKnowledge {
    /// Knowledge content
    pub content: String,
    /// Cache timestamp
    pub cached_at: u64,
    /// Expiry time (seconds)
    pub ttl_seconds: u64,
    /// Knowledge type (legacy, kept for compatibility)
    pub knowledge_type: KnowledgeType,
    /// Associated context (such as project path, file list, etc.)
    pub context: SkipMap<String, String>,
    /// File fingerprint (for detecting actual changes; FileBased types only)
    pub fingerprint: Option<KnowledgeFingerprint>,
    /// Knowledge metadata (new version, includes the validation strategy)
    pub metadata: Option<KnowledgeMetadata>,
}

/// Knowledge type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeType {
    /// Project structure (volatile)
    ProjectStructure,
    /// Code content (volatile)
    CodeContent,
    /// Project configuration (moderate change frequency)
    ProjectConfig,
    /// Coding guideline (stable)
    CodingGuideline,
    /// User preference (stable)
    UserPreference,
    /// Other
    Other,
}

impl KnowledgeType {
    /// Returns the default TTL (seconds)
    pub fn default_ttl(&self) -> u64 {
        match self {
            KnowledgeType::ProjectStructure => 1800,    // 30 minutes
            KnowledgeType::CodeContent => 1800,         // 30 minutes
            KnowledgeType::ProjectConfig => 3600,       // 60 minutes
            KnowledgeType::CodingGuideline => u64::MAX, // permanent
            KnowledgeType::UserPreference => u64::MAX,  // permanent
            KnowledgeType::Other => 3600,               // default 60 minutes
        }
    }

    /// Infers the knowledge type from a category string
    pub fn from_category(category: &str) -> Self {
        match category.to_lowercase().as_str() {
            "project_structure" | "project_info" => KnowledgeType::ProjectStructure,
            "code_content" | "code_snippet" => KnowledgeType::CodeContent,
            "project_config" | "config" => KnowledgeType::ProjectConfig,
            "coding_guideline" | "best_practice" | "common_sense" => KnowledgeType::CodingGuideline,
            "user_preference" | "preference" => KnowledgeType::UserPreference,
            _ => KnowledgeType::Other,
        }
    }
}

/// Converts the new knowledge type to the legacy knowledge type (compatibility layer)
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
    /// Creates a new cached knowledge entry (basic version, without fingerprint
    /// or metadata)
    pub fn new(
        content: String,
        knowledge_type: KnowledgeType,
        context: SkipMap<String, String>,
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

    /// Creates a cached knowledge entry with a fingerprint (for FileBased types)
    pub fn new_with_fingerprint(
        content: String,
        knowledge_type: KnowledgeType,
        context: SkipMap<String, String>,
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

    /// Creates a cached knowledge entry with metadata (recommended; supports
    /// all validation strategies)
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

    /// Checks whether the entry has expired (time-based only)
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();

        if self.ttl_seconds == u64::MAX {
            return false; // never expires
        }

        now > self.cached_at + self.ttl_seconds
    }

    /// Checks whether the fingerprint is still valid (detects actual file changes)
    pub fn verify_fingerprint(&self) -> FingerprintVerificationResult {
        if let Some(ref fp) = self.fingerprint {
            fp.verify()
        } else {
            // No fingerprint; assume valid
            FingerprintVerificationResult {
                is_valid: true,
                changed_files: Vec::new(),
                missing_files: Vec::new(),
                unchanged_count: 0,
                total_files: 0,
            }
        }
    }

    /// Checks whether a refresh is needed (combining all validation strategies)
    pub fn needs_refresh(&self) -> bool {
        // The new metadata validation takes precedence, with TTL / fingerprint
        // as supplementary checks.
        if let Some(ref metadata) = self.metadata {
            match &metadata.validation {
                // Fingerprint-based: the fingerprint must still match, or the
                // TTL must have expired.
                ValidationStrategy::Fingerprint { .. } => {
                    if self.is_expired() {
                        return true;
                    }
                    let fp_ok = self
                        .fingerprint
                        .as_ref()
                        .map(|f| f.verify().is_valid)
                        .unwrap_or(true);
                    return !fp_ok;
                }
                // Time range / external check / session bound / no validation:
                // rely on the metadata decision, with TTL as a fallback.
                _ => {
                    if !metadata.is_valid() {
                        return true;
                    }
                    return self.is_expired();
                }
            }
        }

        // Legacy compatibility: TTL and fingerprint only
        if self.is_expired() {
            return true;
        }
        if let Some(ref fp) = self.fingerprint {
            let verification = fp.verify();
            if !verification.is_valid {
                return true;
            }
        }
        false
    }

    /// Runs validation and returns the detailed result
    pub fn validate(&self) -> ValidationResult {
        if let Some(ref metadata) = self.metadata {
            let is_valid = metadata.is_valid();

            let (validation_type, details, suggestion) = match &metadata.validation {
                ValidationStrategy::Fingerprint {
                    files: _,
                    git_commit: _,
                } => {
                    let fp_result = self
                        .fingerprint
                        .as_ref()
                        .map(|f| f.verify())
                        .unwrap_or_else(|| FingerprintVerificationResult {
                            is_valid: true,
                            changed_files: Vec::new(),
                            missing_files: Vec::new(),
                            unchanged_count: 0,
                            total_files: 0,
                        });

                    if fp_result.is_valid {
                        (
                            "fingerprint".to_string(),
                            format!(
                                "{} files verified, {} unchanged",
                                fp_result.total_files, fp_result.unchanged_count
                            ),
                            ValidationSuggestion::UseCache,
                        )
                    } else {
                        (
                            "fingerprint".to_string(),
                            format!(
                                "{} files changed, {} missing",
                                fp_result.changed_files.len(),
                                fp_result.missing_files.len()
                            ),
                            ValidationSuggestion::Refresh,
                        )
                    }
                }

                ValidationStrategy::TimeRange {
                    valid_from: _,
                    valid_until,
                } => {
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
                }

                ValidationStrategy::ExternalCheck {
                    source,
                    last_check,
                    check_interval,
                } => {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or(Duration::ZERO)
                        .as_secs();
                    let elapsed = now.saturating_sub(*last_check);

                    if is_valid {
                        (
                            "external_check".to_string(),
                            format!(
                                "Last checked {}s ago, next check in {}s",
                                elapsed,
                                check_interval.saturating_sub(elapsed)
                            ),
                            ValidationSuggestion::UseCache,
                        )
                    } else {
                        (
                            "external_check".to_string(),
                            format!("Source '{}' needs recheck", source),
                            ValidationSuggestion::ExternalCheckRequired,
                        )
                    }
                }

                ValidationStrategy::SessionBound { session_id } => (
                    "session_bound".to_string(),
                    format!("Bound to session: {}", session_id),
                    ValidationSuggestion::UseCache,
                ),

                ValidationStrategy::None => (
                    "none".to_string(),
                    "No validation required (stable knowledge)".to_string(),
                    ValidationSuggestion::UseCache,
                ),
            };

            ValidationResult {
                is_valid,
                validation_type,
                details,
                suggestion,
            }
        } else {
            // Legacy validation logic
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

    /// Returns the remaining time to live (seconds)
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

/// Session knowledge cache manager
pub struct SessionKnowledgeCache {
    /// Cached knowledge
    cache: SkipMap<String, CachedKnowledge>,
    /// Cache config file path
    cache_file: std::path::PathBuf,
}

impl SessionKnowledgeCache {
    /// Creates a new cache manager
    pub fn new() -> Self {
        let cache_file = get_config_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("~/.config"))
            .join("rust_tools")
            .join("knowledge_cache.json");

        Self {
            cache: SkipMap::default(),
            cache_file,
        }
    }

    /// Loads the cache from file
    pub fn load(&mut self) -> Result<(), String> {
        if !self.cache_file.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&self.cache_file)
            .map_err(|e| format!("Failed to read cache file: {}", e))?;

        let cache: SkipMap<String, CachedKnowledge> = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse cache file: {}", e))?;

        // Filter out expired entries
        self.cache = cache.into_iter().filter(|(_, v)| !v.is_expired()).collect();

        Ok(())
    }

    /// Saves the cache to file
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

    /// Gets a cached knowledge entry
    pub fn get(&self, key: &str) -> Option<&CachedKnowledge> {
        self.cache
            .get_ref(&key.to_string())
            .filter(|v| !v.is_expired())
    }

    /// Sets a cached knowledge entry
    pub fn set(&mut self, key: String, knowledge: CachedKnowledge) {
        self.cache.insert(key, knowledge);
    }

    /// Removes expired cache entries
    pub fn cleanup_expired(&mut self) -> usize {
        let before = self.cache.len();
        self.cache.retain(|_, v| !v.is_expired());
        before - self.cache.len()
    }

    /// Clears all volatile knowledge from the cache
    pub fn clear_volatile(&mut self) {
        self.cache.retain(|_, v| {
            matches!(
                v.knowledge_type,
                KnowledgeType::CodingGuideline | KnowledgeType::UserPreference
            )
        });
    }

    /// Checks whether a topic needs to be re-fetched
    pub fn needs_refresh(&self, key: &str) -> bool {
        match self.get(key) {
            None => true,                         // no cache, needs re-fetch
            Some(entry) => entry.needs_refresh(), // combined refresh decision
        }
    }

    /// Returns cache statistics
    pub fn stats(&self) -> CacheStats {
        let total = self.cache.len();
        let expired = self.cache.values().filter(|v| v.is_expired()).count();
        let volatile = self
            .cache
            .values()
            .filter(|v| v.ttl_seconds != u64::MAX)
            .count();
        let stable = total - volatile;

        CacheStats {
            total,
            expired,
            volatile,
            stable,
        }
    }
}

/// Cache statistics
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

/// Generates a cache key
pub fn make_cache_key(topic: &str, context: &SkipMap<String, String>) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    topic.hash(&mut hasher);

    // Hash the context after sorting, for consistency
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
    use crate::ai::tools::storage::knowledge_fingerprint::KnowledgeFingerprint;
    use crate::ai::tools::storage::knowledge_types::{
        KnowledgeMetadata, KnowledgeType as NewKnowledgeType, ValidationStrategy,
        create_time_sensitive_metadata,
    };
    use rust_tools::cw::SkipMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_knowledge_type_ttl() {
        assert_eq!(KnowledgeType::ProjectStructure.default_ttl(), 1800);
        assert_eq!(KnowledgeType::CodingGuideline.default_ttl(), u64::MAX);
    }

    #[test]
    fn test_cache_expiry() {
        let mut context = SkipMap::default();
        context.insert("project".to_string(), "rust_tools".to_string());

        let knowledge = CachedKnowledge::new(
            "test content".to_string(),
            KnowledgeType::ProjectStructure,
            context,
        );

        // Just created; must not be expired
        assert!(!knowledge.is_expired());
        assert!(knowledge.ttl_remaining() <= 1800);
    }

    #[test]
    fn test_needs_refresh_fingerprint_change() {
        let tmp = std::env::temp_dir();
        let file = tmp.join(format!(
            "rt_kc_{}.txt",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&file, "a").unwrap();
        let mut fp = KnowledgeFingerprint::new(&SkipMap::default());
        fp.add_file(&file, true).unwrap();
        let metadata = KnowledgeMetadata::new(
            NewKnowledgeType::FileBased,
            SkipMap::default(),
            Some("file".to_string()),
        );
        let ck = CachedKnowledge::new_with_metadata("x".to_string(), metadata, Some(fp));
        assert!(!ck.needs_refresh());
        fs::write(&file, "b").unwrap();
        assert!(ck.needs_refresh());
        let _ = fs::remove_file(&file);
    }

    #[test]
    fn test_needs_refresh_time_range_expired() {
        let mut md = create_time_sensitive_metadata("ts", SkipMap::default(), Some(1));
        if let ValidationStrategy::TimeRange {
            valid_from,
            valid_until,
        } = &mut md.validation
        {
            *valid_from = 0;
            *valid_until = 0;
        }
        let ck = CachedKnowledge::new_with_metadata("x".to_string(), md, None);
        assert!(ck.needs_refresh());
    }

    #[test]
    fn test_session_cache_needs_refresh_delegation() {
        let tmp = std::env::temp_dir();
        let file = tmp.join(format!(
            "rt_kc_d_{}.txt",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&file, "a").unwrap();
        let mut fp = KnowledgeFingerprint::new(&SkipMap::default());
        fp.add_file(&file, true).unwrap();
        let metadata = KnowledgeMetadata::new(
            NewKnowledgeType::FileBased,
            SkipMap::default(),
            Some("file".to_string()),
        );
        let ck = CachedKnowledge::new_with_metadata("x".to_string(), metadata, Some(fp));
        let mut cache = SessionKnowledgeCache::new();
        let key = make_cache_key("project_structure", &SkipMap::default());
        cache.set(key.clone(), ck);
        fs::write(&file, "b").unwrap();
        assert!(cache.needs_refresh(&key));
        let _ = fs::remove_file(&file);
    }
}
