use crate::ai::skills::SkillManifest;

pub fn match_skill<'a>(skills: &'a [SkillManifest], input: &str) -> Option<&'a SkillManifest> {
    let input_lower = input.to_lowercase();
    let mut matched: Vec<&SkillManifest> = Vec::new();

    for skill in skills {
        for trigger in &skill.triggers {
            if !trigger.trim().is_empty() && input_lower.contains(&trigger.to_lowercase()) {
                matched.push(skill);
                break;
            }
        }
    }

    // Return the highest priority matched skill
    matched.into_iter().max_by_key(|s| s.priority)
}
