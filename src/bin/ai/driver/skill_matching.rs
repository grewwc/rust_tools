use crate::ai::skills::SkillManifest;
use regex::Regex;
use std::collections::BTreeSet;

const SKILL_MATCH_THRESHOLD: i64 = 30;
const NEGATIVE_TRIGGER_PENALTY: i64 = -200;
const CONTEXT_KEYWORD_BONUS: i64 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SkillScore {
    evidence: i64,
    total: i64,
}

pub fn match_skill<'a>(skills: &'a [SkillManifest], input: &str) -> Option<&'a SkillManifest> {
    let input_norm = normalize(input);
    if input_norm.is_empty() {
        return None;
    }
    let input_tokens = tokenize(&input_norm);

    let mut best: Option<(&SkillManifest, SkillScore)> = None;

    for skill in skills {
        // 方案二：先检查负面触发器，如果命中则直接跳过
        if has_negative_trigger(skill, &input_norm) {
            continue;
        }

        let score = score_skill(skill, &input_norm, &input_tokens);
        if score.evidence < SKILL_MATCH_THRESHOLD {
            continue;
        }
        match best {
            None => best = Some((skill, score)),
            Some((best_skill, best_score)) => {
                if score.total > best_score.total
                    || (score.total == best_score.total && skill.priority > best_skill.priority)
                {
                    best = Some((skill, score));
                }
            }
        }
    }

    best.map(|(skill, _)| skill)
}

/// 方案二：检查是否命中负面触发器（支持正则表达式）
fn has_negative_trigger(skill: &SkillManifest, input_norm: &str) -> bool {
    // 从 skill 的 triggers 中读取 negative triggers
    // 格式：negative:pattern 表示负面触发器，pattern 可以是普通字符串或正则表达式
    for trigger in &skill.triggers {
        if trigger.starts_with("negative:") {
            let pattern = &trigger[9..];
            if !pattern.is_empty() {
                // 尝试将 pattern 作为正则表达式匹配
                if match_regex_pattern(pattern, input_norm) {
                    return true;
                }
            }
        }
    }
    false
}

/// 使用正则表达式或普通字符串匹配
fn match_regex_pattern(pattern: &str, text: &str) -> bool {
    // 首先尝试作为正则表达式匹配
    // 如果 pattern 包含正则特殊字符，则作为正则处理
    let has_regex_chars = pattern.contains(|c| matches!(c, '.' | '*' | '+' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\'));
    
    if has_regex_chars {
        // 包含正则特殊字符，尝试编译为正则可以
        if let Ok(regex) = Regex::new(pattern) {
            return regex.is_match(text);
        }
        // 如果正则编译失败，降级为普通字符串匹配
    }
    
    // 普通字符串包含匹配（已 normalized）
    text.contains(pattern)
}

/// 方案三：检查是否包含必要的上下文关键词
fn has_required_context(skill: &SkillManifest, input_norm: &str) -> bool {
    // 从 triggers 中读取 required_context 格式：context:keyword1,keyword2
    for trigger in &skill.triggers {
        if trigger.starts_with("context:") {
            let keywords_str = &trigger[8..];
            let keywords: Vec<&str> = keywords_str.split(',').collect();
            if keywords.is_empty() {
                continue;
            }

            let mut found_count = 0;
            for keyword in keywords {
                let keyword_norm = normalize(keyword.trim());
                if !keyword_norm.is_empty() && input_norm.contains(&keyword_norm) {
                    found_count += 1;
                }
            }

            // 至少找到一个上下文关键词
            if found_count > 0 {
                return true;
            }
        }
    }
    // 如果没有定义 context 要求，默认返回 true
    true
}

fn score_skill(skill: &SkillManifest, input_norm: &str, input_tokens: &[String]) -> SkillScore {
    let mut evidence = 0i64;
    let mut matched_trigger_count = 0i64;
    let mut has_context_bonus = false;

    for trigger in &skill.triggers {
        // 跳过特殊格式的 trigger
        if trigger.starts_with("negative:") || trigger.starts_with("context:") {
            continue;
        }

        let trigger_norm = normalize(trigger);
        if trigger_norm.is_empty() {
            continue;
        }

        if input_norm.contains(&trigger_norm) {
            matched_trigger_count += 1;
            evidence += 100;
            // Reward longer exact-phrase triggers a bit more; this stays generic.
            evidence += (trigger_norm.chars().count() as i64).min(40);

            // 方案三：如果命中 trigger 且有上下文关键词，给予额外奖励
            if has_required_context(skill, input_norm) {
                has_context_bonus = true;
            }
            continue;
        }

        let trigger_tokens = tokenize(&trigger_norm);
        let overlap = token_overlap(input_tokens, &trigger_tokens);
        if overlap >= 2 {
            let mut overlap_score = overlap as i64 * 18;

            // 方案三：如果有上下文关键词，增加 token 重叠的权重
            if has_required_context(skill, input_norm) {
                overlap_score += CONTEXT_KEYWORD_BONUS;
                has_context_bonus = true;
            }

            evidence += overlap_score;
        }
    }

    if matched_trigger_count > 1 {
        evidence += (matched_trigger_count - 1) * 20;
    }

    // 方案三：上下文关键词奖励
    if has_context_bonus {
        evidence += CONTEXT_KEYWORD_BONUS;
    }

    let mut skill_text = String::new();
    skill_text.push_str(&skill.name);
    skill_text.push(' ');
    skill_text.push_str(&skill.description);
    skill_text.push(' ');
    for t in &skill.triggers {
        // 跳过特殊格式
        if !t.starts_with("negative:") && !t.starts_with("context:") {
            skill_text.push_str(t);
            skill_text.push(' ');
        }
    }
    let skill_tokens = tokenize(&normalize(&skill_text));
    let overlap = token_overlap(input_tokens, &skill_tokens);
    evidence += overlap as i64 * 10;

    SkillScore {
        evidence,
        total: evidence + skill.priority as i64,
    }
}

fn token_overlap(a: &[String], b: &[String]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let b = b.iter().collect::<BTreeSet<_>>();
    let mut count = 0usize;
    let mut seen = BTreeSet::new();
    for token in a {
        if token.len() <= 1 {
            continue;
        }
        if !seen.insert(token) {
            continue;
        }
        if b.contains(token) {
            count += 1;
        }
    }
    count
}

fn tokenize(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut ascii = String::new();
    let mut cjk = String::new();

    let flush_ascii = |buf: &mut String, out: &mut Vec<String>| {
        if !buf.is_empty() {
            out.push(std::mem::take(buf));
        }
    };
    let flush_cjk = |buf: &mut String, out: &mut Vec<String>| {
        if buf.is_empty() {
            return;
        }
        let segment = std::mem::take(buf);
        let chars = segment.chars().collect::<Vec<_>>();
        if chars.len() == 1 {
            out.push(segment);
            return;
        }
        for window in chars.windows(2) {
            out.push(window.iter().collect());
        }
        out.push(chars.iter().collect());
    };

    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            flush_cjk(&mut cjk, &mut out);
            ascii.push(ch);
        } else if is_cjk(ch) {
            flush_ascii(&mut ascii, &mut out);
            cjk.push(ch);
        } else {
            flush_ascii(&mut ascii, &mut out);
            flush_cjk(&mut cjk, &mut out);
        }
    }
    flush_ascii(&mut ascii, &mut out);
    flush_cjk(&mut cjk, &mut out);
    out
}

fn normalize(input: &str) -> String {
    input
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect::<String>()
}

fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(name: &str, description: &str, triggers: &[&str], priority: i32) -> SkillManifest {
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
            triggers: triggers.iter().map(|s| s.to_string()).collect(),
            priority,
            source_path: None,
        }
    }

    #[test]
    fn priority_alone_does_not_match_skill() {
        let skills = vec![skill("openclaw", "agent", &["openclaw"], 30)];
        assert!(match_skill(&skills, "随便聊聊今天吃什么").is_none());
    }

    #[test]
    fn review_style_input_prefers_code_review_over_priority_fallback() {
        let skills = vec![
            skill(
                "openclaw",
                "OpenClaw-like autonomous tool-using agent",
                &["openclaw"],
                30,
            ),
            skill(
                "code-review",
                "Review code for quality, security, and best practices",
                &["帮我看一下", "看一下", "有问题吗"],
                10,
            ),
        ];
        let matched = match_skill(&skills, "你帮我看看这个 makefile 是不是有问题").unwrap();
        assert_eq!(matched.name, "code-review");
    }

    #[test]
    fn explicit_openclaw_trigger_still_matches() {
        let skills = vec![skill(
            "openclaw",
            "agent",
            &["openclaw 模式", "开启 openclaw"],
            30,
        )];
        let matched = match_skill(&skills, "帮我开启 openclaw 模式").unwrap();
        assert_eq!(matched.name, "openclaw");
    }

    #[test]
    fn negative_trigger_prevents_match() {
        // 方案二测试：负面触发器应该阻止匹配
        let skills = vec![skill(
            "prompt-optimizer",
            "优化提示词",
            &["优化提示词", "negative:为什么会调用", "negative:技能匹配"],
            20,
        )];

        // 这个输入包含负面触发器，不应该匹配
        let input = "为什么下面这一个 prompt，会调用到 prompt-optimizer.skill 呢？";
        assert!(match_skill(&skills, input).is_none());

        // 这个输入不包含负面触发器，应该匹配
        let input2 = "帮我优化这个提示词";
        assert!(match_skill(&skills, input2).is_some());
    }

    #[test]
    fn negative_trigger_with_regex_pattern() {
        // 测试正则表达式模式的 negative trigger
        let skills = vec![skill(
            "prompt-optimizer",
            "优化提示词",
            &[
                "优化提示词",
                "negative:为什么.*调用",  // 正则模式：为什么...调用
                "negative:为什么会",       // 普通字符串模式
                "negative:技能匹配",       // 普通字符串模式
            ],
            20,
        )];

        // 测试用例 1：匹配正则模式 "为什么.*调用"
        let input1 = "为什么这个 prompt 会调用到 prompt-optimizer 呢？";
        assert!(match_skill(&skills, input1).is_none(), "应该被正则模式匹配并排除");

        // 测试用例 2：匹配普通字符串模式 "为什么会"
        let input2 = "为什么会命中这个 skill 呢？";
        assert!(match_skill(&skills, input2).is_none(), "应该被普通字符串模式匹配并排除");

        // 测试用例 3：匹配普通字符串模式 "技能匹配"
        let input3 = "这是什么技能匹配逻辑？";
        assert!(match_skill(&skills, input3).is_none(), "应该被普通字符串模式匹配并排除");

        // 测试用例 4：不匹配任何 negative trigger，应该正常匹配
        let input4 = "帮我优化这个提示词";
        assert!(match_skill(&skills, input4).is_some(), "应该正常匹配");

        // 测试用例 5：用户实际问题复现
        let input5 = "为什么会命中 /Users/bytedance/.config/rust_tools/skills/prompt-optimizer.skill 这个 skill 呢？";
        assert!(match_skill(&skills, input5).is_none(), "用户实际问题应该被排除");
    }

    #[test]
    fn context_keywords_improve_matching() {
        // 方案三测试：上下文关键词应该提高匹配分数
        let skills = vec![
            skill(
                "prompt-optimizer",
                "优化提示词",
                &["优化提示词", "context:优化，改进，建议"],
                20,
            ),
            skill("code-review", "代码审查", &["帮我看一下代码"], 15),
        ];

        // 包含上下文关键词，应该匹配 prompt-optimizer
        let input = "帮我优化这个提示词，让它更准确";
        let matched = match_skill(&skills, input);
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().name, "prompt-optimizer");

        // 不包含上下文关键词，但包含 trigger，也应该匹配（只是分数低一些）
        let input2 = "优化提示词";
        let matched2 = match_skill(&skills, input2);
        assert!(matched2.is_some());
    }

    #[test]
    fn regex_pattern_with_complex_cases() {
        // 测试更复杂的正则模式
        let skills = vec![skill(
            "debugger",
            "调试代码",
            &[
                "调试",
                "negative:排查.*agent",      // 正则：排查...agent
                "negative:选择到.*skill",    // 正则：选择到...skill
                "negative:错误.*触发",       // 正则：错误...触发
            ],
            20,
        )];

        // 匹配正则模式
        assert!(match_skill(&skills, "排查一下 agent 的问题").is_none());
        assert!(match_skill(&skills, "为什么选择到这个 skill").is_none());
        assert!(match_skill(&skills, "这个错误触发了什么问题").is_none());

        // 不匹配 negative trigger，应该正常匹配
        assert!(match_skill(&skills, "帮我调试这段代码").is_some());
    }
}
