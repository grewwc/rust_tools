mod background;

use serde::{Deserialize, Serialize};

pub(crate) use background::assess_learning_note_quality;

#[derive(Debug, Clone)]
struct ReflectionQuality {
    actionable: bool,
    specific: bool,
    generalizable: bool,
}

impl ReflectionQuality {
    fn score(&self) -> u8 {
        let mut score = 0;
        if self.actionable {
            score += 1;
        }
        if self.specific {
            score += 1;
        }
        if self.generalizable {
            score += 1;
        }
        score
    }

    fn is_high_quality(&self) -> bool {
        // Long-term retention requires meeting both "actionable" and
        // "generalizable" baselines.
        self.actionable && self.generalizable
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LearningNoteAssessment {
    pub(crate) actionable: bool,
    pub(crate) specific: bool,
    pub(crate) generalizable: bool,
    pub(crate) score: u8,
    pub(crate) high_quality: bool,
    pub(crate) char_count: usize,
    pub(crate) word_count: usize,
    pub(crate) nonempty_lines: usize,
    pub(crate) unique_token_ratio: f32,
    pub(crate) directive_signals: usize,
    pub(crate) code_signals: usize,
    pub(crate) artifact_signals: usize,
    pub(crate) abstraction_signals: usize,
    pub(crate) condition_signals: usize,
    pub(crate) one_off_signals: usize,
}

impl LearningNoteAssessment {
    pub(crate) fn confidence(&self) -> f64 {
        (self.score as f64 / 3.0).clamp(0.0, 1.0)
    }
}
