use super::{LearningNoteAssessment, ReflectionQuality};

/// 共享的显式知识保存质量评估。
///
/// 该评估只在用户或模型显式调用知识工具时运行，不会主动读取或召回知识库。
pub(crate) fn assess_learning_note_quality(note: &str) -> LearningNoteAssessment {
    let features = LearningNoteQualityFeatures::from_note(note);
    let actionable = features.directive_signals > 0
        || (features.condition_signals > 0 && features.word_count >= 8)
        || (features.nonempty_lines >= 2 && features.word_count >= 10);
    let specific = features.code_signals > 0
        || features.artifact_signals >= 2
        || (features.directive_signals > 0
            && features.word_count >= 6
            && features.unique_token_ratio >= 0.75)
        || (features.char_count >= 64
            && features.word_count >= 10
            && features.unique_token_ratio >= 0.55);
    let generalizable = features.one_off_signals == 0
        && (features.condition_signals > 0
            || features.abstraction_signals > 0
            || (features.word_count >= 10
                && features.unique_token_ratio >= 0.55
                && features.nonempty_lines >= 2));

    let quality = ReflectionQuality {
        actionable,
        specific,
        generalizable,
    };
    LearningNoteAssessment {
        actionable: quality.actionable,
        specific: quality.specific,
        generalizable: quality.generalizable,
        score: quality.score(),
        high_quality: quality.is_high_quality(),
        char_count: features.char_count,
        word_count: features.word_count,
        nonempty_lines: features.nonempty_lines,
        unique_token_ratio: features.unique_token_ratio,
        directive_signals: features.directive_signals,
        code_signals: features.code_signals,
        artifact_signals: features.artifact_signals,
        abstraction_signals: features.abstraction_signals,
        condition_signals: features.condition_signals,
        one_off_signals: features.one_off_signals,
    }
}

struct LearningNoteQualityFeatures {
    char_count: usize,
    word_count: usize,
    nonempty_lines: usize,
    unique_token_ratio: f32,
    directive_signals: usize,
    code_signals: usize,
    artifact_signals: usize,
    abstraction_signals: usize,
    condition_signals: usize,
    one_off_signals: usize,
}

impl LearningNoteQualityFeatures {
    fn from_note(note: &str) -> Self {
        let trimmed = note.trim();
        if trimmed.is_empty() {
            return Self {
                char_count: 0,
                word_count: 0,
                nonempty_lines: 0,
                unique_token_ratio: 0.0,
                directive_signals: 0,
                code_signals: 0,
                artifact_signals: 0,
                abstraction_signals: 0,
                condition_signals: 0,
                one_off_signals: 1,
            };
        }

        let lower = trimmed.to_lowercase();
        let tokens = quality_tokens(trimmed);
        let word_count = tokens.len();
        let unique_token_ratio = if word_count == 0 {
            0.0
        } else {
            let unique = tokens.iter().collect::<rust_tools::cw::SkipSet<_>>().len();
            unique as f32 / word_count as f32
        };

        let directive_signals = count_contains(
            &lower,
            &[
                "do:", "avoid:", "prefer ", "should ", "must ", "always ", "never ", "ensure ",
                "应该", "必须", "不要", "避免", "优先", "确保",
            ],
        );
        let condition_signals = count_contains(
            &lower,
            &[
                "when ",
                "if ",
                "before ",
                "after ",
                "instead ",
                "rather than ",
                "unless ",
                "当",
                "如果",
                "之前",
                "之后",
                "而不是",
                "否则",
            ],
        );
        let abstraction_signals = count_contains(
            &lower,
            &[
                "habit",
                "policy",
                "pattern",
                "rule",
                "guideline",
                "principle",
                "workflow",
                "strategy",
                "习惯",
                "规则",
                "准则",
                "原则",
                "策略",
                "流程",
            ],
        );
        let code_signals = count_code_signals(trimmed);
        let artifact_signals = count_artifact_signals(trimmed, &tokens);
        let one_off_signals = count_one_off_signals(trimmed, &tokens);

        Self {
            char_count: trimmed.chars().count(),
            word_count,
            nonempty_lines: trimmed
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count(),
            unique_token_ratio,
            directive_signals,
            code_signals,
            artifact_signals,
            abstraction_signals,
            condition_signals,
            one_off_signals,
        }
    }
}

fn quality_tokens(note: &str) -> Vec<String> {
    note.split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.'))
        .map(str::trim)
        .filter(|token| token.chars().count() >= 2)
        .map(|token| token.to_lowercase())
        .collect()
}

fn count_contains(haystack: &str, needles: &[&str]) -> usize {
    needles
        .iter()
        .filter(|needle| haystack.contains(**needle))
        .count()
}

fn count_code_signals(note: &str) -> usize {
    let markers = ['`', '/', '\\', '(', ')', ':', '[', ']', '{', '}'];
    let mut count = markers
        .iter()
        .filter(|marker| note.contains(**marker))
        .count();
    if note.contains("::") {
        count += 1;
    }
    if note.contains("->") || note.contains("=>") {
        count += 1;
    }
    count
}

fn count_artifact_signals(note: &str, tokens: &[String]) -> usize {
    let mut count = 0usize;
    if note.contains(".rs") || note.contains(".ts") || note.contains(".py") || note.contains(".md") {
        count += 1;
    }
    count += tokens
        .iter()
        .filter(|token| token.contains('_') || token.contains("::") || token.ends_with("()"))
        .count();
    count += tokens
        .iter()
        .filter(|token| token.chars().any(|ch| ch.is_ascii_digit()))
        .count()
        .min(1);
    count
}

fn count_one_off_signals(note: &str, tokens: &[String]) -> usize {
    let mut count = 0usize;
    if crate::ai::knowledge::entry::note_has_local_env_path_leak(note) {
        count += 1;
    }
    if note.contains("session:") || note.contains("tmp") || note.contains("/var/") {
        count += 1;
    }
    count += tokens
        .iter()
        .filter(|token| {
            token.len() >= 12 && token.chars().filter(|ch| ch.is_ascii_digit()).count() >= 4
        })
        .count();
    count += tokens
        .iter()
        .filter(|token| {
            token.len() >= 16 && token.chars().all(|ch| ch.is_ascii_hexdigit() || ch == '-')
        })
        .count();
    count
}
