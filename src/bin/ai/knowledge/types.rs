/// Unified category and knowledge type definitions.
/// Single source of truth — replaces scattered string lists.
use serde::{Deserialize, Serialize};

/// All valid knowledge categories.
/// Replaces the 5 scattered string lists across the codebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    // Guidelines (agent behavior rules)
    SafetyRules,
    UserPreference,
    Preference,
    CodingGuideline,
    BestPractice,
    CommonSense,
    SelfNote,

    // Knowledge (project facts, user memories)
    UserMemory,
    ProjectInfo,
    Architecture,
    DecisionLog,

    // Internal / system
    ToolCache,
    ProjectWriteback,

    // Fallback for unknown categories
    #[serde(other)]
    Other,
}

impl Category {
    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "safety_rules" => Self::SafetyRules,
            "user_preference" => Self::UserPreference,
            "preference" => Self::Preference,
            "coding_guideline" => Self::CodingGuideline,
            "best_practice" => Self::BestPractice,
            "common_sense" => Self::CommonSense,
            "self_note" => Self::SelfNote,
            "user_memory" => Self::UserMemory,
            "project_info" => Self::ProjectInfo,
            "architecture" => Self::Architecture,
            "decision_log" => Self::DecisionLog,
            "tool_cache" => Self::ToolCache,
            "project_writeback" => Self::ProjectWriteback,
            _ => Self::Other,
        }
    }

    /// Whether this is a guideline category (used for persistent guidelines)
    pub fn is_guideline(&self) -> bool {
        matches!(
            self,
            Self::SafetyRules
                | Self::UserPreference
                | Self::Preference
                | Self::CodingGuideline
                | Self::BestPractice
                | Self::CommonSense
                | Self::SelfNote
        )
    }

    /// Whether this is a knowledge category (used for auto-recall)
    pub fn is_knowledge(&self) -> bool {
        !self.is_guideline() && !matches!(self, Self::ToolCache)
    }

    /// Default priority for this category
    pub fn default_priority(&self) -> u8 {
        match self {
            Self::CommonSense
            | Self::CodingGuideline
            | Self::BestPractice
            | Self::UserPreference
            | Self::Preference => 210,
            Self::SafetyRules => 255,
            _ => 150,
        }
    }

    /// Knowledge type for TTL and validation
    pub fn knowledge_type(&self) -> KnowledgeType {
        match self {
            Self::ProjectInfo | Self::Architecture => KnowledgeType::FileBased,
            Self::DecisionLog => KnowledgeType::LongTerm,
            Self::ToolCache => KnowledgeType::ShortLived,
            Self::UserMemory => KnowledgeType::UserDirected,
            Self::SafetyRules => KnowledgeType::Persistent,
            Self::CodingGuideline | Self::BestPractice => KnowledgeType::LongTerm,
            Self::CommonSense | Self::UserPreference | Self::Preference => KnowledgeType::LongTerm,
            Self::SelfNote => KnowledgeType::ShortLived,
            Self::ProjectWriteback => KnowledgeType::FileBased,
            Self::Other => KnowledgeType::General,
        }
    }

    /// Convert to string for serialization
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SafetyRules => "safety_rules",
            Self::UserPreference => "user_preference",
            Self::Preference => "preference",
            Self::CodingGuideline => "coding_guideline",
            Self::BestPractice => "best_practice",
            Self::CommonSense => "common_sense",
            Self::SelfNote => "self_note",
            Self::UserMemory => "user_memory",
            Self::ProjectInfo => "project_info",
            Self::Architecture => "architecture",
            Self::DecisionLog => "decision_log",
            Self::ToolCache => "tool_cache",
            Self::ProjectWriteback => "project_writeback",
            Self::Other => "other",
        }
    }
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Knowledge type — determines TTL and validation strategy.
/// Merges the old and new KnowledgeType enums into one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeType {
    /// Tied to specific files; invalidated when files change
    FileBased,
    /// Valid for a limited time (API docs, code snippets)
    TimeSensitive,
    /// Long-lived knowledge (best practices, architecture decisions)
    LongTerm,
    /// Short-lived (tool cache, temporary notes)
    ShortLived,
    /// User-directed knowledge (explicitly saved)
    UserDirected,
    /// Never expires (safety rules, core principles)
    Persistent,
    /// General purpose
    General,
}

impl KnowledgeType {
    /// Default TTL in seconds
    pub fn default_ttl(&self) -> u64 {
        match self {
            Self::FileBased => 1800,        // 30 min
            Self::TimeSensitive => 600,     // 10 min
            Self::LongTerm => 3600,         // 1 hour
            Self::ShortLived => 300,        // 5 min
            Self::UserDirected => u64::MAX, // never expires
            Self::Persistent => u64::MAX,   // never expires
            Self::General => 3600,          // 1 hour
        }
    }

    /// Whether this type needs validation
    pub fn needs_validation(&self) -> bool {
        matches!(self, Self::FileBased | Self::TimeSensitive)
    }
}

/// Guidelines group for ranking in persistent guidelines
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuidelineGroup {
    Safety = 0,
    Preferences = 1,
    SelfNotes = 2,
    Other = 3,
}

impl GuidelineGroup {
    pub fn from_category(cat: &Category) -> Self {
        match cat {
            Category::SafetyRules => Self::Safety,
            Category::UserPreference
            | Category::Preference
            | Category::CodingGuideline
            | Category::BestPractice
            | Category::CommonSense => Self::Preferences,
            Category::SelfNote => Self::SelfNotes,
            _ => Self::Other,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}
