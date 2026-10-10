/// Memory importance scoring — used for active learning and automatic forgetting
#[derive(Debug, Clone)]
pub struct MemoryImportance {
    /// Reference count
    pub frequency: u32,
    /// Time-decay factor (0.0 - 1.0, higher when more recent)
    pub recency: f64,
    /// Breadth of applicability (0.0 - 1.0)
    pub generality: f64,
    /// Whether the user has confirmed it
    pub user_validated: bool,
}

impl MemoryImportance {
    pub fn new() -> Self {
        Self {
            frequency: 0,
            recency: 1.0,
            generality: 0.5,
            user_validated: false,
        }
    }

    /// Compute the composite importance score (0.0 - 1.0)
    pub fn score(&self) -> f64 {
        let freq_score = (self.frequency as f64).min(10.0) / 10.0; // 0-1
        let recency_score = self.recency.clamp(0.0, 1.0);
        let generality_score = self.generality.clamp(0.0, 1.0);
        let validation_bonus = if self.user_validated { 0.2 } else { 0.0 };

        // Weights: frequency 30%, recency 30%, generality 20%, user confirmation 20%
        (freq_score * 0.3 + recency_score * 0.3 + generality_score * 0.2 + validation_bonus)
            .min(1.0)
    }

    /// Increment the reference count
    pub fn increment_frequency(&mut self) {
        self.frequency += 1;
    }

    /// Update the time decay
    pub fn update_recency(&mut self, created_at: &str) {
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(created_at) {
            let now = chrono::Utc::now();
            let age_days = (now - dt.with_timezone(&chrono::Utc)).num_seconds() as f64 / 86400.0;
            // Exponential decay: 30-day half-life
            self.recency = (-age_days * std::f64::consts::LN_2 / 30.0).exp();
        }
    }

    /// Evaluate generality (based on category and tags)
    pub fn evaluate_generality(&mut self, category: &str, tags: &[String]) {
        let general_categories = [
            "common_sense",
            "best_practice",
            "coding_guideline",
            "safety_rules",
        ];

        let general_tags = ["general", "universal", "fundamental", "core"];

        let category_score = if general_categories.contains(&category.as_ref()) {
            1.0
        } else {
            0.5
        };

        let tag_score = tags
            .iter()
            .filter(|t| general_tags.contains(&t.as_str()))
            .count() as f64
            / tags.len().max(1) as f64;

        self.generality = (category_score * 0.7 + tag_score * 0.3).clamp(0.0, 1.0);
    }

    /// Mark as user-confirmed
    pub fn mark_user_validated(&mut self) {
        self.user_validated = true;
    }

    /// Decide whether this should be forgotten (low-value memory)
    pub fn should_prune(&self, min_score: f64) -> bool {
        self.score() < min_score
    }
}

impl Default for MemoryImportance {
    fn default() -> Self {
        Self::new()
    }
}
