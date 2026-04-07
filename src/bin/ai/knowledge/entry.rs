/// Unified knowledge entry — replaces AgentMemoryEntry.
/// Single data structure for all knowledge/memory storage.
use serde::{Deserialize, Serialize};

use super::types::Category;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    #[serde(default)]
    pub id: Option<String>,
    pub timestamp: String,
    pub category: String,
    pub note: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
    /// Priority level: 0-255. Higher = more important. 255 = permanent.
    #[serde(default = "default_priority")]
    pub priority: Option<u8>,
}

fn default_priority() -> Option<u8> {
    Some(100)
}

impl KnowledgeEntry {
    pub fn category_enum(&self) -> Category {
        Category::from_str(&self.category)
    }

    pub fn priority_value(&self) -> u8 {
        self.priority.unwrap_or(100)
    }

    /// Create a search text from all fields
    pub fn search_text(&self) -> String {
        let mut text = format!("{}: {}", self.category, self.note);
        if !self.tags.is_empty() {
            text.push_str(&format!(" [tags: {}]", self.tags.join(", ")));
        }
        if let Some(src) = &self.source {
            text.push_str(&format!(" (source: {})", src));
        }
        text
    }

    /// Whether this entry mentions a project hint
    pub fn mentions_project(&self, project: &str) -> bool {
        let hint = project.trim().to_lowercase();
        if hint.is_empty() {
            return false;
        }
        self.category.to_lowercase().contains(&hint)
            || self.note.to_lowercase().contains(&hint)
            || self
                .source
                .as_deref()
                .unwrap_or("")
                .to_lowercase()
                .contains(&hint)
            || self
                .tags
                .iter()
                .any(|tag| tag.to_lowercase().contains(&hint))
    }

    /// Unique key for deduplication
    pub fn dedup_key(&self) -> String {
        format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}",
            self.id.as_deref().unwrap_or(""),
            self.timestamp,
            self.category,
            self.note
        )
    }
}

/// Alias for backward compatibility during migration
pub type AgentMemoryEntry = KnowledgeEntry;
