use crate::ai::skills::SkillManifest;
use regex::Regex;
use std::collections::BTreeSet;

// ==================== 配置常量 ====================

// 匹配阈值：技能必须达到的最低证据分
const SKILL_MATCH_THRESHOLD: i64 = 80;

// 负面触发器惩罚：命中负面触发器直接排除
const _NEGATIVE_TRIGGER_PENALTY: i64 = -1000;

// 上下文关键词奖励：命中必需的上下文关键词
const CONTEXT_KEYWORD_BONUS: i64 = 40;

// 精确短语匹配奖励：完整 trigger 短语匹配
const EXACT_PHRASE_BONUS: i64 = 100;

// Token 重叠基础分：每个重叠 token 的分数
const TOKEN_OVERLAP_BASE_SCORE: i64 = 15;

// 长短语额外奖励：trigger 长度超过一定字符数的额外奖励
const LONG_PHRASE_BONUS_CAP: i64 = 50;

// 多 trigger 匹配奖励：匹配多个不同 trigger 的额外奖励
const MULTI_TRIGGER_BONUS: i64 = 25;

// 优先级权重：优先级在总分中的权重（降低优先级影响）
const PRIORITY_WEIGHT: f64 = 0.3;

// 常见停用词（中文和英文），降低这些词的权重
const STOP_WORDS: &[&str] = &[
    // 中文停用词
    "的", "了", "是", "在", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "会", "着", "没有", "看", "好", "自己", "这", "那",
    "他", "她", "它", "们", "这个", "那个", "怎么",
    "如何", "为什么", "请", "一下", "可以", "能",
    "吗", "呢", "吧", "啊", "哦", "嗯",
    // 英文停用词
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "and",
    "but", "if", "or", "because", "until", "while", "about",
    "against", "this", "that", "these", "those", "am", "it", "its",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "they",
    "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "any", "both", "her", "his", "yours", "mine",
    // 通用编程术语（过于常见）
    "code", "代码", "function", "函数", "file", "文件", "help", "帮助",
    "make", "让", "get", "获取", "use", "使用", "write", "写", "read", "读",
];

// ==================== 数据结构 ====================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SkillScore {
    evidence: i64,      // 证据分（基于匹配度）
    total: i64,         // 总分（证据分 + 优先级加权）
    matched_triggers: i64, // 匹配的 trigger 数量
    has_context_bonus: bool, // 是否有上下文奖励
}

// ==================== 主匹配函数 ====================

pub fn match_skill<'a>(skills: &'a [SkillManifest], input: &str) -> Option<&'a SkillManifest> {
    let input_norm = normalize(input);
    if input_norm.is_empty() {
        return None;
    }

    let input_tokens = tokenize(&input_norm);
    let stop_words_set = build_stop_words_set();

    // 第一步：过滤掉命中负面触发器的技能
    let candidate_skills: Vec<&SkillManifest> = skills
        .iter()
        .filter(|skill| !has_negative_trigger(skill, &input_norm))
        .collect();

    if candidate_skills.is_empty() {
        return None;
    }

    // 第二步：对所有候选技能评分
    let mut scored_skills: Vec<(&SkillManifest, SkillScore)> = candidate_skills
        .into_iter()
        .filter_map(|skill| {
            let score = score_skill(skill, &input_norm, &input_tokens, &stop_words_set);
            if score.evidence >= SKILL_MATCH_THRESHOLD {
                Some((skill, score))
            } else {
                None
            }
        })
        .collect();

    if scored_skills.is_empty() {
        return None;
    }

    // 第三步：排序选择最佳技能
    // 排序规则：
    // 1. 优先选择证据分高的（匹配度优先）
    // 2. 证据分相同时，选择匹配 trigger 数量多的
    // 3. 仍相同时，选择有上下文奖励的
    // 4. 最后才考虑优先级
    scored_skills.sort_by(|a, b| {
        let (skill_a, score_a) = a;
        let (skill_b, score_b) = b;

        // 首先比较证据分（匹配度）
        score_b
            .evidence
            .cmp(&score_a.evidence)
            // 其次比较匹配的 trigger 数量
            .then_with(|| score_b.matched_triggers.cmp(&score_a.matched_triggers))
            // 再比较是否有上下文奖励
            .then_with(|| {
                score_b
                    .has_context_bonus
                    .cmp(&score_a.has_context_bonus)
            })
            // 最后比较优先级（权重已降低）
            .then_with(|| score_b.total.cmp(&score_a.total))
            // 如果都相同，按技能名称字母序（保证确定性）
            .then_with(|| skill_a.name.cmp(&skill_b.name))
    });

    // 返回最佳匹配
    scored_skills.first().map(|(skill, _)| *skill)
}

// ==================== 负面触发器检查 ====================

/// 检查是否命中负面触发器（支持正则表达式）
fn has_negative_trigger(skill: &SkillManifest, input_norm: &str) -> bool {
    for trigger in &skill.triggers {
        if let Some(pattern) = trigger.strip_prefix("negative:") {
            if !pattern.is_empty() && match_regex_pattern(pattern, input_norm) {
                return true;
            }
        }
    }
    false
}

/// 使用正则表达式或普通字符串匹配
fn match_regex_pattern(pattern: &str, text: &str) -> bool {
    // 判断是否包含正则特殊字符
    let has_regex_chars = pattern.contains(|c| matches!(c, '.' | '*' | '+' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\'));

    if has_regex_chars {
        // 尝试编译为正则
        if let Ok(regex) = Regex::new(pattern) {
            return regex.is_match(text);
        }
        // 正则编译失败，降级为普通字符串匹配
    }

    // 普通字符串包含匹配
    text.contains(pattern)
}

// ==================== 技能评分函数 ====================

fn score_skill(
    skill: &SkillManifest,
    input_norm: &str,
    input_tokens: &[String],
    stop_words_set: &BTreeSet<String>,
) -> SkillScore {
    let mut evidence = 0i64;
    let mut matched_trigger_count = 0i64;
    let mut has_context_bonus = false;

    // 检查是否有上下文关键词要求
    let context_required = check_context_requirement(skill, input_norm);
    if context_required {
        has_context_bonus = true;
    }

    // 遍历所有 trigger 进行匹配
    for trigger in &skill.triggers {
        // 跳过特殊格式的 trigger
        if trigger.starts_with("negative:") || trigger.starts_with("context:") {
            continue;
        }

        let trigger_norm = normalize(trigger);
        if trigger_norm.is_empty() {
            continue;
        }

        // 1. 精确短语匹配（最高优先级）
        if input_norm.contains(&trigger_norm) {
            matched_trigger_count += 1;
            evidence += EXACT_PHRASE_BONUS;

            // 长 trigger 额外奖励（越具体越长，匹配度越高）
            let length_bonus = (trigger_norm.chars().count() as i64 / 2).min(LONG_PHRASE_BONUS_CAP);
            evidence += length_bonus;

            // 如果有上下文关键词，给予额外奖励
            if has_context_bonus {
                evidence += CONTEXT_KEYWORD_BONUS;
            }
            continue;
        }

        // 2. Token 重叠匹配
        let trigger_tokens = tokenize(&trigger_norm);
        let overlap = weighted_token_overlap(input_tokens, &trigger_tokens, stop_words_set);

        if overlap >= 2 {
            // 基础分 + 重叠数量分
            let overlap_score = TOKEN_OVERLAP_BASE_SCORE + (overlap as i64 * TOKEN_OVERLAP_BASE_SCORE);

            // 如果有上下文关键词，增加权重
            if has_context_bonus {
                evidence += overlap_score + CONTEXT_KEYWORD_BONUS / 2;
            } else {
                evidence += overlap_score;
            }
        }
    }

    // 多 trigger 匹配奖励（匹配越多不同的 trigger，置信度越高）
    if matched_trigger_count > 1 {
        evidence += (matched_trigger_count - 1) * MULTI_TRIGGER_BONUS;
    }

    // 上下文关键词奖励
    if has_context_bonus {
        evidence += CONTEXT_KEYWORD_BONUS;
    }

    // 技能描述和名称的 token 重叠（作为辅助证据）
    let mut skill_text = String::new();
    skill_text.push_str(&skill.name);
    skill_text.push(' ');
    skill_text.push_str(&skill.description);
    for t in &skill.triggers {
        if !t.starts_with("negative:") && !t.starts_with("context:") {
            skill_text.push(' ');
            skill_text.push_str(t);
        }
    }
    let skill_tokens = tokenize(&normalize(&skill_text));
    let description_overlap = weighted_token_overlap(input_tokens, &skill_tokens, stop_words_set);
    evidence += description_overlap as i64 * 8; // 描述匹配的权重较低

    // 计算总分：证据分 + 优先级（降低权重）
    let priority_bonus = (skill.priority as f64 * PRIORITY_WEIGHT) as i64;
    let total = evidence + priority_bonus;

    SkillScore {
        evidence,
        total,
        matched_triggers: matched_trigger_count,
        has_context_bonus,
    }
}

// ==================== 上下文关键词检查 ====================

/// 检查是否满足技能的上下文关键词要求
fn check_context_requirement(skill: &SkillManifest, input_norm: &str) -> bool {
    for trigger in &skill.triggers {
        if let Some(keywords_str) = trigger.strip_prefix("context:") {
            let keywords: Vec<&str> = keywords_str.split(',').collect();
            if keywords.is_empty() {
                continue;
            }

            // 至少匹配一个上下文关键词
            for keyword in keywords {
                let keyword_norm = normalize(keyword.trim());
                if !keyword_norm.is_empty() && input_norm.contains(&keyword_norm) {
                    return true;
                }
            }
        }
    }
    // 没有定义 context 要求，默认返回 true（不强制）
    true
}

// ==================== Token 重叠计算 ====================

/// 带权重的 token 重叠计算（考虑停用词）
fn weighted_token_overlap(
    a: &[String],
    b: &[String],
    stop_words_set: &BTreeSet<String>,
) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let b_set: BTreeSet<&String> = b.iter().collect();
    let mut count = 0;
    let mut seen = BTreeSet::new();

    for token in a {
        // 跳过单字符 token
        if token.len() <= 1 {
            continue;
        }

        // 跳过已处理的重复 token
        if !seen.insert(token) {
            continue;
        }

        // 跳过停用词
        if stop_words_set.contains(token) {
            continue;
        }

        // 计算重叠
        if b_set.contains(token) {
            count += 1;
        }
    }

    count
}

// ==================== 停用词表构建 ====================

fn build_stop_words_set() -> BTreeSet<String> {
    STOP_WORDS.iter().map(|s| normalize(s)).collect()
}

// ==================== Token 化 ====================

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
        // CJK 字符使用双字窗口
        for window in chars.windows(2) {
            out.push(window.iter().collect());
        }
        // 添加完整短语
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

// ==================== 文本标准化 ====================

fn normalize(input: &str) -> String {
    input
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_punctuation() {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
}

// ==================== CJK 字符判断 ====================

fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF
    )
}

// ==================== 测试用例 ====================

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
    fn test_no_match_for_irrelevant_input() {
        // 测试：无关输入不应该匹配任何技能
        let skills = vec![skill("openclaw", "agent", &["openclaw"], 30)];
        assert!(match_skill(&skills, "随便聊聊今天吃什么").is_none());
    }

    #[test]
    fn test_explicit_trigger_match() {
        // 测试：明确触发器应该匹配
        let skills = vec![skill(
            "openclaw",
            "agent",
            &["openclaw 模式", "开启 openclaw"],
            30,
        )];
        let matched = match_skill(&skills, "帮我开启 openclaw 模式");
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().name, "openclaw");
    }

    #[test]
    fn test_negative_trigger_prevents_match() {
        // 测试：负面触发器应该阻止匹配
        let skills = vec![skill(
            "prompt-optimizer",
            "优化提示词",
            &[
                "优化提示词",
                "negative:为什么.*调用",
                "negative:为什么会",
                "negative:技能匹配",
            ],
            90, // 高优先级
        )];

        // 包含负面触发器，不应该匹配
        let input = "为什么下面这一个 prompt，会调用到 prompt-optimizer.skill 呢？";
        assert!(match_skill(&skills, input).is_none());

        // 不包含负面触发器，应该匹配
        let input2 = "帮我优化这个提示词";
        assert!(match_skill(&skills, input2).is_some());
    }

    #[test]
    fn test_context_keywords_improve_matching() {
        // 测试：上下文关键词提高匹配准确度
        let skills = vec![
            skill(
                "prompt-optimizer",
                "优化提示词",
                &[
                    "优化提示词",
                    "改进提示",
                    "context:优化，改进，建议，更好",
                ],
                80,
            ),
            skill(
                "debugger",
                "调试代码",
                &["调试", "修复代码", "代码有问题"],
                70,
            ),
        ];

        // 包含上下文关键词"优化"，应该匹配 prompt-optimizer
        let input = "帮我优化这个提示词，让它更准确";
        let matched = match_skill(&skills, input);
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().name, "prompt-optimizer");

        // 只有 trigger 匹配，没有上下文关键词，也应该匹配
        let input2 = "优化提示词";
        let matched2 = match_skill(&skills, input2);
        assert!(matched2.is_some());
    }

    #[test]
    fn test_priority_should_not_override_low_evidence() {
        // 测试：高优先级不应该覆盖低匹配度
        let skills = vec![
            skill(
                "openclaw",
                "OpenClaw 代理",
                &["openclaw", "openclaw 模式"],
                30,
            ),
            skill(
                "prompt-optimizer",
                "优化提示词",
                &["优化提示词"],
                90, // 高优先级
            ),
        ];

        // 输入明确提到 openclaw，应该匹配 openclaw 而不是高优先级的 prompt-optimizer
        let input = "帮我开启 openclaw 模式";
        let matched = match_skill(&skills, input);
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().name, "openclaw");
    }

    #[test]
    fn test_multi_trigger_bonus() {
        // 测试：匹配多个 trigger 应该获得额外奖励
        let skills = vec![
            skill(
                "code-review",
                "代码审查",
                &["帮我看看代码", "代码审查", "代码有问题吗"],
                80,
            ),
            skill(
                "debugger",
                "调试代码",
                &["调试"],
                70,
            ),
        ];

        // 输入匹配多个 code-review 的 trigger
        let input = "帮我看看代码，代码有问题吗？";
        let matched = match_skill(&skills, input);
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().name, "code-review");
    }

    #[test]
    fn test_stop_words_ignored() {
        // 测试：停用词不应该影响匹配
        let skills = vec![skill(
            "debugger",
            "调试代码",
            &["调试代码", "修复 bug", "调试"],
            70,
        )];

        // 输入包含大量停用词，但核心词"调试"匹配
        let input = "请帮我调试一下这段代码可以吗";
        let matched = match_skill(&skills, input);
        // 由于"调试"是精确匹配，应该能匹配上
        assert!(matched.is_some());
    }

    #[test]
    fn test_threshold_prevents_weak_match() {
        // 测试：阈值应该阻止弱匹配
        let skills = vec![skill(
            "debugger",
            "调试代码修复 bug",
            &["调试代码", "修复 bug", "代码有问题"],
            70,
        )];

        // 输入只有少量重叠 token，不应该匹配
        let input = "我有一段代码";
        assert!(match_skill(&skills, input).is_none());
    }

    #[test]
    fn test_regex_negative_trigger() {
        // 测试：正则负面触发器
        let skills = vec![skill(
            "debugger",
            "调试代码",
            &[
                "调试",
                "negative:排查.*agent",
                "negative:选择到.*skill",
            ],
            70,
        )];

        // 匹配正则模式
        assert!(match_skill(&skills, "排查一下 agent 的问题").is_none());
        assert!(match_skill(&skills, "为什么选择到这个 skill").is_none());

        // 不匹配负面 trigger，应该正常匹配
        assert!(match_skill(&skills, "帮我调试这段代码").is_some());
    }

    #[test]
    fn test_similar_skills_differentiation() {
        // 测试：区分相似技能
        let skills = vec![
            skill(
                "code-review",
                "代码审查，检查代码质量",
                &["代码审查", "帮我看看代码", "代码质量检查", "有什么改进建议"],
                80,
            ),
            skill(
                "debugger",
                "调试代码，修复错误",
                &["调试", "修复代码", "代码报错", "为什么这段代码不工作"],
                70,
            ),
        ];

        // 审查请求应该匹配 code-review
        let input1 = "帮我看看这段代码有什么改进建议";
        let matched1 = match_skill(&skills, input1);
        assert_eq!(matched1.unwrap().name, "code-review");

        // 调试请求应该匹配 debugger
        let input2 = "这段代码报错了，帮我调试一下";
        let matched2 = match_skill(&skills, input2);
        assert_eq!(matched2.unwrap().name, "debugger");
    }
}
