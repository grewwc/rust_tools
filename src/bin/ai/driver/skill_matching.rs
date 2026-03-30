use crate::ai::skills::SkillManifest;

/// 简化的技能匹配逻辑：基于 description 和 name 的语义匹配
/// 仅作为模型路由的 fallback，当模型路由失败时使用
pub fn match_skill<'a>(skills: &'a [SkillManifest], input: &str) -> Option<&'a SkillManifest> {
    if input.trim().is_empty() || skills.is_empty() {
        return None;
    }

    let input_lower = input.to_lowercase();
    
    // 简单匹配：检查技能名称或描述是否包含输入中的关键词
    let mut best_match: Option<&SkillManifest> = None;
    let mut best_score = 0;

    for skill in skills {
        let score = score_skill(skill, &input_lower);
        if score > best_score {
            best_score = score;
            best_match = Some(skill);
        }
    }

    // 设置一个简单的阈值，避免误匹配
    if best_score >= 2 {
        best_match
    } else {
        None
    }
}

/// 简单的技能评分：基于名称和描述的 token 重叠
fn score_skill(skill: &SkillManifest, input_lower: &str) -> i32 {
    let mut score = 0;

    // 技能名称匹配（权重较高）
    let name_lower = skill.name.to_lowercase();
    if input_lower.contains(&name_lower) {
        score += 5;
    }

    // 描述匹配：检查描述中的关键词是否出现在输入中
    let description_lower = skill.description.to_lowercase();
    
    // 提取描述中的关键词（简单的分词）
    for word in description_lower.split_whitespace() {
        // 跳过太短的词
        if word.len() < 2 {
            continue;
        }
        // 跳过常见的停用词
        if is_stop_word(word) {
            continue;
        }
        if input_lower.contains(word) {
            score += 1;
        }
    }

    // 中文描述：检查中文字符
    // 常见中文停用字（高频但无意义）
    let chinese_stop_chars: &[char] = &[
        '的', '了', '是', '在', '有', '和', '就', '不', '都', '一',
        '上', '也', '很', '到', '说', '要', '去', '会', '着', '这',
        '那', '他', '她', '它', '们', '个', '我', '你', '他', '与',
        '及', '等', '者', '其', '之', '而', '于', '为', '以', '于',
    ];
    
    for ch in description_lower.chars() {
        if ch.is_ascii_alphabetic() || ch.is_whitespace() {
            continue;
        }
        // 跳过停用字
        if chinese_stop_chars.contains(&ch) {
            continue;
        }
        // 有意义的中文字符匹配
        if input_lower.contains(ch) {
            score += 1;
        }
    }

    score
}

/// 常见停用词
fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "the" | "a" | "an" | "is" | "are" | "was" | "were" | "be" | "been"
            | "have" | "has" | "had" | "do" | "does" | "did" | "will" | "would"
            | "could" | "should" | "may" | "might" | "must" | "can" | "need"
            | "to" | "of" | "in" | "for" | "on" | "with" | "at" | "by" | "from"
            | "as" | "into" | "through" | "and" | "but" | "or" | "if" | "then"
            | "的" | "了" | "是" | "在" | "有" | "和" | "就" | "不" | "都"
            | "一" | "上" | "也" | "很" | "到" | "说" | "要" | "去" | "会"
            | "着" | "这" | "那" | "他" | "她" | "它" | "们" | "这个" | "那个"
            | "怎么" | "如何" | "为什么" | "请" | "可以" | "能" | "吗" | "呢"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(name: &str, description: &str, priority: i32) -> SkillManifest {
        SkillManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: description.to_string(),
            author: Some("system".to_string()),
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            prompt: String::new(),
            system_prompt: None,
            priority,
            source_path: None,
        }
    }

    #[test]
    fn test_no_match_for_irrelevant_input() {
        let skills = vec![skill("openclaw", "agent mode for complex tasks", 30)];
        assert!(match_skill(&skills, "随便聊聊今天吃什么").is_none());
    }

    #[test]
    fn test_name_match() {
        let skills = vec![skill("refactor", "code refactoring expert", 65)];
        let matched = match_skill(&skills, "帮我 refactor 这段代码");
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().name, "refactor");
    }

    #[test]
    fn test_description_match() {
        let skills = vec![skill(
            "debugger",
            "调试专家：帮助定位和修复代码中的 bug、错误、异常",
            70,
        )];
        let matched = match_skill(&skills, "代码有 bug，帮我调试一下");
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().name, "debugger");
    }

    #[test]
    fn test_priority_not_primary_factor() {
        let skills = vec![
            skill("openclaw", "agent mode for complex tasks", 30),
            skill("debugger", "调试专家：帮助定位和修复代码中的 bug", 90),
        ];

        // 输入明确指向 openclaw，应该匹配 openclaw 而不是高优先级的 debugger
        let matched = match_skill(&skills, "开启 openclaw 模式");
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().name, "openclaw");
    }

    #[test]
    fn test_nba_query_does_not_match_refactor() {
        // 验证：查询 NBA 比赛和排名不应该匹配到 refactor 技能
        let skills = vec![skill(
            "refactor",
            "重构专家：重构代码结构、命名、可读性与可测试性。适用于代码整理、提取函数、去重复、优化结构等场景。不适用于修复报错、调试、处理异常等情况。",
            65,
        )];
        let input = "帮我看一下今天 nba 的比赛，和现在东西部的所有球队排名";
        let matched = match_skill(&skills, input);
        // 应该不匹配，因为输入与重构无关
        assert!(matched.is_none(), "NBA 查询不应该匹配到 refactor 技能");
    }
}
