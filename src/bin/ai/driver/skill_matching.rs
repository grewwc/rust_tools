use crate::ai::skills::SkillManifest;

pub use super::intent_recognition::{CoreIntent, UserIntent};
use super::rank_skills_locally;

/// 动态阈值配置
mod thresholds {
    pub fn skill_count_threshold(skill_count: usize) -> f64 {
        if skill_count > 10 {
            5.0
        } else if skill_count > 5 {
            4.0
        } else {
            3.5
        }
    }
}

/// 简化的技能匹配逻辑：基于语义和意图的智能匹配
/// 仅作为模型路由的 fallback，当模型路由失败时使用
/// intent 参数可选：如果提供，会使用意图信息进行更精确的匹配
pub fn match_skill<'a>(
    skills: &'a [SkillManifest],
    input: &str,
    intent: Option<&UserIntent>,
) -> Option<&'a SkillManifest> {
    if input.trim().is_empty() || skills.is_empty() {
        return None;
    }

    let ranked = rank_skills_locally(skills, input, intent);
    let Some(best) = ranked.first() else {
        return None;
    };
    let best_match = Some(best.skill);
    let best_score = best.score;

    // 动态阈值：根据技能数量和匹配质量调整，提高底线防止误触发
    let base_threshold = thresholds::skill_count_threshold(skills.len());
    // 提高阈值防止通用词导致误匹配
    let threshold = if let Some(intent_ref) = intent {
        match intent_ref.core {
            CoreIntent::RequestAction => base_threshold.max(5.0),
            CoreIntent::SeekSolution => base_threshold.max(4.5),
            CoreIntent::QueryConcept => base_threshold.max(6.0),
            CoreIntent::Casual => base_threshold.max(6.0),
        }
    } else {
        base_threshold.max(5.0)
    };

    if best_score >= threshold {
        best_match
    } else {
        None
    }
}

/// 检查技能描述是否明确排除了某种意图
pub(super) fn is_intent_excluded(_skill: &SkillManifest, intent: &UserIntent) -> bool {
    intent.is_searching_resource("skill")
}

/// 语义评分：由 local model + runtime semantic index 提供。
pub(super) fn score_skill_smart(
    _skill: &SkillManifest,
    _input_lower: &str,
    _intent: Option<&UserIntent>,
) -> f64 {
    0.0
}
