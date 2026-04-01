use crate::ai::skills::SkillManifest;
use rust_tools::cw::SkipSet;
use rust_tools::commonw::FastMap;

pub use super::intent_recognition::{CoreIntent, UserIntent};

/// 评分权重常量
mod weights {
    pub const NAME_MATCH: f64 = 10.0;
    pub const PHRASE_MATCH_BASE: f64 = 2.0;
    pub const PHRASE_LENGTH_BONUS: f64 = 0.5;
    pub const KEYWORD_BASE: f64 = 1.0;
    pub const KEYWORD_LENGTH_BONUS_EN: f64 = 0.2;
    pub const KEYWORD_LENGTH_BONUS_CN: f64 = 0.3;
    pub const CHINESE_WORD_MATCH_BASE: f64 = 1.5;
    pub const CHINESE_WORD_LENGTH_BONUS: f64 = 0.3;
    pub const CHINESE_PARTIAL_MATCH: f64 = 0.5;
    pub const INTENT_MATCH_HIGH: f64 = 5.0;
    pub const INTENT_MATCH_MEDIUM: f64 = 1.5;
    pub const NEGATION_PENALTY: f64 = 0.5;
    pub const EXCLUDED_INTENT_PENALTY: f64 = -5.0;
}

/// 动态阈值配置
mod thresholds {
    pub fn skill_count_threshold(skill_count: usize) -> f64 {
        if skill_count > 5 { 3.0 } else { 2.0 }
    }

    pub const MIN_PHRASE_LENGTH: usize = 5;
    pub const MIN_KEYWORD_LENGTH: usize = 3;
    pub const MIN_CHINESE_WORD_LENGTH: usize = 2;
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

    let input_lower = input.to_lowercase();

    // 为每个技能评分
    let mut best_match: Option<&SkillManifest> = None;
    let mut best_score = 0.0;

    for skill in skills {
        // 如果技能描述明确排除了当前意图，直接跳过（如果有 intent 的话）
        if let Some(intent_ref) = intent {
            if is_intent_excluded(skill, intent_ref) {
                continue;
            }
        }

        let score = score_skill_smart(skill, &input_lower, intent);
        if score > best_score {
            best_score = score;
            best_match = Some(skill);
        }
    }

    // 动态阈值：根据技能数量和匹配质量调整，但保证一个较高的最低底线
    let base_threshold = thresholds::skill_count_threshold(skills.len());
    // 如果意图只是请求行动，但没有明确指向现有技能，提高阈值防止 "代码" 等通用词导致误触发
    let threshold = if let Some(intent_ref) = intent {
        if intent_ref.core == CoreIntent::RequestAction {
            base_threshold.max(4.5)
        } else {
            base_threshold.max(3.0)
        }
    } else {
        base_threshold.max(3.0)
    };

    if best_score >= threshold {
        best_match
    } else {
        None
    }
}

/// 检查技能描述是否明确排除了某种意图
fn is_intent_excluded(skill: &SkillManifest, intent: &UserIntent) -> bool {
    let desc_lower = skill.description.to_lowercase();

    match &intent.core {
        CoreIntent::QueryConcept => {
            // 如果技能描述明确说"不适用于询问概念"之类
            let exclusion_patterns = [
                "如果用户询问的不是",
                "切勿选择",
                "不适用于询问",
                "not for questions",
                "don't use for",
                "only for",
            ];
            exclusion_patterns.iter().any(|p| desc_lower.contains(p))
        }
        CoreIntent::RequestAction | CoreIntent::SeekSolution | CoreIntent::Casual => {
            // 如果用户是在搜索资源（如"找几个 skill"），而不是请求执行技能
            // 所有具体执行类技能都应该排除
            if intent.is_searching_resource("skill") {
                let execution_keywords = [
                    "审查", "重构", "调试", "评估", "分析",
                    "review", "refactor", "debug", "analyze",
                ];
                let name_lower = skill.name.to_lowercase();
                execution_keywords.iter().any(|kw| desc_lower.contains(kw))
                    && (name_lower.contains("review")
                        || name_lower.contains("refactor")
                        || name_lower.contains("debug")
                        || name_lower.contains("analyzer"))
            } else {
                false
            }
        }
    }
}

/// 智能评分：结合 TF-IDF 思想、短语匹配、意图匹配
fn score_skill_smart(
    skill: &SkillManifest,
    input_lower: &str,
    intent: Option<&UserIntent>,
) -> f64 {
    let mut score = 0.0;

    // 1. 技能名称匹配（权重最高）
    let name_lower = skill.name.to_lowercase();
    if input_lower.contains(&name_lower) {
        score += weights::NAME_MATCH;
    }

    // 2. 检查技能描述中的关键短语是否在输入中出现
    let description_lower = skill.description.to_lowercase();

    // 提取技能描述中的关键短语（2-4 个词的短语）
    let skill_phrases = extract_meaningful_phrases(&description_lower);

    // 提取输入中的关键短语
    let input_phrases = extract_meaningful_phrases(input_lower);

    // 短语匹配（权重较高）
    for phrase in skill_phrases.iter() {
        if input_phrases.contains(phrase) {
            // 短语越长，权重越高
            let phrase_weight = weights::KEYWORD_BASE
                + (phrase.split_whitespace().count() as f64) * weights::PHRASE_LENGTH_BONUS;
            score += phrase_weight * weights::PHRASE_MATCH_BASE;
        }
    }

    // 3. 关键词匹配（TF-IDF 风格：罕见词权重更高）
    let skill_keywords = extract_keywords(&description_lower);
    let input_keywords = extract_keywords(input_lower);

    for (keyword, idf_weight) in &skill_keywords {
        if input_keywords.contains_key(keyword) {
            // IDF 权重：在技能描述中出现少但在输入中出现的词权重高
            score += *idf_weight;
        }
    }

    // 4. 中文字符匹配（更智能：优先匹配连续的中文字词）
    score += score_chinese_semantic(&description_lower, input_lower);

    // 5. 意图匹配度加分（如果有 intent）
    if let Some(intent_ref) = intent {
        score += intent_match_bonus(skill, intent_ref, input_lower);
    }

    // 6. 否定词惩罚
    if has_negation_context(input_lower, &description_lower) {
        score *= weights::NEGATION_PENALTY;
    }

    score
}

/// 提取有意义的短语（2-4 个词）
fn extract_meaningful_phrases(text: &str) -> SkipSet<String> {
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| w.len() >= thresholds::MIN_KEYWORD_LENGTH && !is_stop_word(w))
        .collect();

    let mut phrases = SkipSet::new(16);

    // 2 词短语
    for i in 0..words.len().saturating_sub(1) {
        let phrase = format!("{} {}", words[i], words[i + 1]);
        if phrase.len() >= thresholds::MIN_PHRASE_LENGTH {
            phrases.insert(phrase);
        }
    }

    // 3 词短语
    for i in 0..words.len().saturating_sub(2) {
        let phrase = format!("{} {} {}", words[i], words[i + 1], words[i + 2]);
        if phrase.len() >= 7 {
            phrases.insert(phrase);
        }
    }

    // 4 词短语
    for i in 0..words.len().saturating_sub(3) {
        let phrase = format!(
            "{} {} {} {}",
            words[i],
            words[i + 1],
            words[i + 2],
            words[i + 3]
        );
        if phrase.len() >= 9 {
            phrases.insert(phrase);
        }
    }

    // 中文短语：按标点分割
    for segment in
        text.split(|c| c == '，' || c == '。' || c == '；' || c == ':' || c == ',' || c == '、')
    {
        let segment = segment.trim();
        if segment.len() >= 4 && segment.len() <= 15 {
            // 检查是否包含有意义的中文
            let chinese_chars: String = segment.chars().filter(|c| !c.is_ascii()).collect();
            if chinese_chars.len() >= thresholds::MIN_CHINESE_WORD_LENGTH {
                phrases.insert(segment.to_string());
            }
        }
    }

    phrases
}

/// 提取关键词并计算 IDF 权重
fn extract_keywords(text: &str) -> FastMap<String, f64> {
    let mut keywords = FastMap::default();

    // 英文关键词
    for word in text.split_whitespace() {
        let word = word.trim_matches(|c: char| c.is_ascii_punctuation());
        if word.len() >= thresholds::MIN_KEYWORD_LENGTH && !is_stop_word(word) {
            // 权重：词越长通常越具体
            let weight =
                weights::KEYWORD_BASE + (word.len() as f64) * weights::KEYWORD_LENGTH_BONUS_EN;
            *keywords.entry(word.to_string()).or_insert(0.0) += weight;
        }
    }

    // 中文关键词：提取连续的中文字符（2 个字符以上）
    let mut chinese_word = String::new();
    for ch in text.chars() {
        if !ch.is_ascii() && !ch.is_whitespace() && !ch.is_ascii_punctuation() {
            if !is_chinese_stop_char(ch) {
                chinese_word.push(ch);
            }
        } else {
            if chinese_word.len() >= thresholds::MIN_CHINESE_WORD_LENGTH {
                let weight = weights::KEYWORD_BASE
                    + (chinese_word.len() as f64) * weights::KEYWORD_LENGTH_BONUS_CN;
                *keywords.entry(chinese_word.clone()).or_insert(0.0) += weight;
            }
            chinese_word.clear();
        }
    }
    // 处理末尾
    if chinese_word.len() >= thresholds::MIN_CHINESE_WORD_LENGTH {
        let weight =
            weights::KEYWORD_BASE + (chinese_word.len() as f64) * weights::KEYWORD_LENGTH_BONUS_CN;
        *keywords.entry(chinese_word).or_insert(0.0) += weight;
    }

    keywords
}

/// 中文语义匹配：优先匹配连续的中文字词
fn score_chinese_semantic(description: &str, input: &str) -> f64 {
    let mut score = 0.0;

    // 提取描述中的中文词（连续 2 个以上中文字符）
    let desc_chinese_words = extract_chinese_words(description);
    let input_chinese_words = extract_chinese_words(input);

    // 完全匹配权重高
    for word in &desc_chinese_words {
        if input_chinese_words.contains(word) {
            score += weights::CHINESE_WORD_MATCH_BASE
                + (word.len() as f64) * weights::CHINESE_WORD_LENGTH_BONUS;
        }
    }

    // 部分匹配（子串）权重低
    for word in &desc_chinese_words {
        if word.len() >= 3 {
            for input_word in &input_chinese_words {
                if input_word.contains(word.as_str()) || word.contains(input_word.as_str()) {
                    score += weights::CHINESE_PARTIAL_MATCH;
                }
            }
        }
    }

    score
}

/// 提取中文词（连续 2 个以上非停用字的中文字符）
fn extract_chinese_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current_word = String::new();

    for ch in text.chars() {
        if !ch.is_ascii() && !ch.is_whitespace() && !ch.is_ascii_punctuation() {
            if !is_chinese_stop_char(ch) {
                current_word.push(ch);
            } else if current_word.len() >= thresholds::MIN_CHINESE_WORD_LENGTH {
                words.push(current_word.clone());
                current_word.clear();
            } else {
                current_word.clear();
            }
        } else {
            if current_word.len() >= thresholds::MIN_CHINESE_WORD_LENGTH {
                words.push(current_word.clone());
            }
            current_word.clear();
        }
    }

    if current_word.len() >= thresholds::MIN_CHINESE_WORD_LENGTH {
        words.push(current_word);
    }

    words
}

/// 意图匹配加分
fn intent_match_bonus(skill: &SkillManifest, intent: &UserIntent, input: &str) -> f64 {
    let desc_lower = skill.description.to_lowercase();
    let name_lower = skill.name.to_lowercase();

    // 如果是搜索资源类查询，给执行类技能负分
    if intent.is_searching_resource("skill") {
        let execution_keywords = [
            "审查", "重构", "调试", "评估", "分析",
            "review", "refactor", "debug", "analyze",
        ];

        if execution_keywords.iter().any(|kw| desc_lower.contains(kw))
            && (name_lower.contains("review")
                || name_lower.contains("refactor")
                || name_lower.contains("debug")
                || name_lower.contains("analyzer"))
        {
            return weights::EXCLUDED_INTENT_PENALTY;
        }
    }

    match &intent.core {
        CoreIntent::QueryConcept => {
            // 如果技能是用于"询问概念"的，加分
            // 但大多数代码技能不适用于概念询问，所以这里通常是 0 或负分
            if desc_lower.contains("询问") || desc_lower.contains("question") {
                weights::INTENT_MATCH_MEDIUM
            } else if desc_lower.contains("切勿") || desc_lower.contains("not for") {
                weights::EXCLUDED_INTENT_PENALTY // 明确排除的，给负分
            } else {
                0.0
            }
        }
        CoreIntent::RequestAction => {
            let mut score = 0.0;

            // 基础意图匹配：降低单纯行动意图的加分，避免只要有"帮我做"就容易过阈值
            if desc_lower.contains("进行")
                || desc_lower.contains("评估")
                || desc_lower.contains("审查")
                || desc_lower.contains("重构")
                || desc_lower.contains("调试")
                || desc_lower.contains("help")
            {
                score += weights::INTENT_MATCH_MEDIUM * 0.5; // 减弱通用匹配权重
            }

            // 强匹配：审查
            if (input.contains("审查") || input.contains("review"))
                && (desc_lower.contains("审查") || name_lower.contains("code-review"))
            {
                score += weights::INTENT_MATCH_HIGH;
            }
            // 强匹配：重构
            if (input.contains("重构") || input.contains("refactor"))
                && (desc_lower.contains("重构") || name_lower.contains("refactor"))
            {
                score += weights::INTENT_MATCH_HIGH;
            }
            // 强匹配：调试/修复 bug
            if (input.contains("调试")
                || input.contains("报错")
                || input.contains("error")
                || input.contains("panic")
                || input.contains("bug")
                || input.contains("修复"))
                && (desc_lower.contains("调试")
                    || desc_lower.contains("bug")
                    || desc_lower.contains("修复")
                    || desc_lower.contains("错误")
                    || name_lower.contains("debugger"))
            {
                score += weights::INTENT_MATCH_HIGH;
            }

            score
        }
        CoreIntent::SeekSolution => {
            let mut score = 0.0;
            if desc_lower.contains("解决")
                || desc_lower.contains("修复")
                || desc_lower.contains("fix")
                || desc_lower.contains("debug")
            {
                score += weights::INTENT_MATCH_MEDIUM;
            }
            if (input.contains("报错")
                || input.contains("错误")
                || input.contains("panic")
                || input.contains("exception")
                || input.contains("crash")
                || input.contains("error")
                || input.contains("bug"))
                && (name_lower.contains("debugger")
                    || desc_lower.contains("调试")
                    || desc_lower.contains("bug"))
            {
                score += weights::INTENT_MATCH_HIGH - weights::INTENT_MATCH_MEDIUM;
            }
            score
        }
        CoreIntent::Casual => 0.0,
    }
}

/// 检查是否有否定上下文
fn has_negation_context(input: &str, _description: &str) -> bool {
    let negation_patterns = [
        "不要",
        "不是",
        "别",
        "不用",
        "无需",
        "不适合",
        "don't",
        "not",
        "no need",
        "not suitable",
    ];
    negation_patterns.iter().any(|p| input.contains(p))
}

/// 常见停用词
fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "a"
            | "an"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "have"
            | "has"
            | "had"
            | "do"
            | "does"
            | "did"
            | "will"
            | "would"
            | "could"
            | "should"
            | "may"
            | "might"
            | "must"
            | "can"
            | "need"
            | "to"
            | "of"
            | "in"
            | "for"
            | "on"
            | "with"
            | "at"
            | "by"
            | "from"
            | "as"
            | "into"
            | "through"
            | "and"
            | "but"
            | "or"
            | "if"
            | "then"
            | "的"
            | "了"
            | "是"
            | "在"
            | "有"
            | "和"
            | "就"
            | "不"
            | "都"
            | "一"
            | "上"
            | "也"
            | "很"
            | "到"
            | "说"
            | "要"
            | "去"
            | "会"
            | "着"
            | "这"
            | "那"
            | "个"
            | "怎么"
            | "如何"
            | "为什么"
            | "请"
            | "可以"
            | "能"
            | "吗"
            | "呢"
            | "我"
            | "你"
            | "我们"
            | "你们"
            | "此"
            | "其"
            | "之"
            | "而"
            | "于"
            | "为"
            | "以"
            | "与"
            | "及"
            | "等"
            | "者"
    )
}

/// 中文停用字
fn is_chinese_stop_char(ch: char) -> bool {
    matches!(
        ch,
        '的' | '了'
            | '是'
            | '在'
            | '有'
            | '和'
            | '就'
            | '不'
            | '都'
            | '一'
            | '上'
            | '也'
            | '很'
            | '到'
            | '说'
            | '要'
            | '去'
            | '会'
            | '着'
            | '这'
            | '那'
            | '个'
            | '我'
            | '你'
            | '与'
            | '及'
            | '等'
            | '者'
            | '其'
            | '之'
            | '而'
            | '于'
            | '为'
            | '以'
    )
}
