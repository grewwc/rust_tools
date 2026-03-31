use crate::ai::skills::SkillManifest;
use std::collections::{HashMap, HashSet};

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
pub fn match_skill<'a>(skills: &'a [SkillManifest], input: &str) -> Option<&'a SkillManifest> {
    if input.trim().is_empty() || skills.is_empty() {
        return None;
    }

    let input_lower = input.to_lowercase();
    
    // 第一步：意图识别 - 判断用户是否在询问概念/定义
    let input_intent = detect_intent(&input_lower);
    
    // 第二步：为每个技能评分
    let mut best_match: Option<&SkillManifest> = None;
    let mut best_score = 0.0;

    for skill in skills {
        // 如果技能描述明确排除了当前意图，直接跳过
        if is_intent_excluded(skill, &input_intent) {
            continue;
        }
        
        let score = score_skill_smart(skill, &input_lower, &input_intent);
        if score > best_score {
            best_score = score;
            best_match = Some(skill);
        }
    }

    // 动态阈值：根据技能数量和匹配质量调整，但保证一个较高的最低底线
    // 因为误触发的代价通常大于不触发（模型也能正常完成任务）
    let base_threshold = thresholds::skill_count_threshold(skills.len());
    // 如果意图只是请求行动，但没有明确指向现有技能，提高阈值防止 "代码" 等通用词导致误触发
    let threshold = if input_intent == UserIntent::RequestAction {
        base_threshold.max(4.5)
    } else {
        base_threshold.max(3.0)
    };

    if best_score >= threshold {
        best_match
    } else {
        None
    }
}

/// 用户意图类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserIntent {
    /// 询问概念/定义（"这是什么"、"是什么意思"）
    QueryConcept,
    /// 请求操作（"帮我做 X"）
    RequestAction,
    /// 寻求解决方案（"怎么处理"、"如何解决"）
    SeekSolution,
    /// 闲聊/其他
    Casual,
}

/// 检测用户意图
pub fn detect_intent(input: &str) -> UserIntent {
    // 询问概念的关键词
    let concept_patterns = [
        "是什么", "什么意思", "含义", "定义", "解释", "说明",
        "what is", "what's", "meaning", "define", "explain",
        "是啥", "啥意思", "咋回事", "指的是什么", "代表什么",
    ];
    
    // 寻求解决方案的关键词（优先检查，因为"怎么"、"如何"更具体）
    let seek_solution_patterns = [
        "怎么", "如何", "怎么办", "怎么处理", "如何解决",
        "how to", "how do i", "what should i do",
        "为什么", "为啥", "原因",
    ];
    
    // 请求行动的关键词
    let request_action_patterns = [
        "帮我", "给我", "请", "帮我做", "帮我写", "帮我改",
        "帮我检查", "帮我调试", "帮我重构", "帮我审查",
        "help me", "please", "do this", "fix", "review", "refactor",
        "优化", "改进", "整理", "重写", "运行", "执行",
    ];
    
    // 优先匹配最具体的模式
    if concept_patterns.iter().any(|p| input.contains(p)) {
        return UserIntent::QueryConcept;
    }
    
    if seek_solution_patterns.iter().any(|p| input.contains(p)) {
        return UserIntent::SeekSolution;
    }

    if request_action_patterns.iter().any(|p| input.contains(p)) {
        return UserIntent::RequestAction;
    }
    
    UserIntent::Casual
}

/// 检查技能描述是否明确排除了某种意图
fn is_intent_excluded(skill: &SkillManifest, intent: &UserIntent) -> bool {
    let desc_lower = skill.description.to_lowercase();
    
    match intent {
        UserIntent::QueryConcept => {
            // 如果技能描述明确说"不适用于询问概念"之类
            let exclusion_patterns = [
                "如果用户询问的不是", "切勿选择", "不适用于询问",
                "not for questions", "don't use for", "only for",
            ];
            exclusion_patterns.iter().any(|p| desc_lower.contains(p))
        }
        _ => false,
    }
}

/// 智能评分：结合 TF-IDF 思想、短语匹配、意图匹配
fn score_skill_smart(skill: &SkillManifest, input_lower: &str, intent: &UserIntent) -> f64 {
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
    for phrase in &skill_phrases {
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
    
    // 5. 意图匹配度加分
    score += intent_match_bonus(skill, intent, input_lower);
    
    // 6. 否定词惩罚
    if has_negation_context(input_lower, &description_lower) {
        score *= weights::NEGATION_PENALTY;
    }
    
    score
}

/// 提取有意义的短语（2-4 个词）
fn extract_meaningful_phrases(text: &str) -> HashSet<String> {
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| w.len() >= thresholds::MIN_KEYWORD_LENGTH && !is_stop_word(w))
        .collect();
    
    let mut phrases = HashSet::new();
    
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
        let phrase = format!("{} {} {} {}", words[i], words[i + 1], words[i + 2], words[i + 3]);
        if phrase.len() >= 9 {
            phrases.insert(phrase);
        }
    }
    
    // 中文短语：按标点分割
    for segment in text.split(|c| c == '，' || c == '。' || c == '；' || c == ':' || c == ',' || c == '、') {
        let segment = segment.trim();
        if segment.len() >= 4 && segment.len() <= 15 {
            // 检查是否包含有意义的中文
            let chinese_chars: String = segment.chars().filter(|c| c.is_ascii() == false).collect();
            if chinese_chars.len() >= thresholds::MIN_CHINESE_WORD_LENGTH {
                phrases.insert(segment.to_string());
            }
        }
    }
    
    phrases
}

/// 提取关键词并计算 IDF 权重
fn extract_keywords(text: &str) -> HashMap<String, f64> {
    let mut keywords = HashMap::new();
    
    // 英文关键词
    for word in text.split_whitespace() {
        let word = word.trim_matches(|c: char| c.is_ascii_punctuation());
        if word.len() >= thresholds::MIN_KEYWORD_LENGTH && !is_stop_word(word) {
            // 权重：词越长通常越具体
            let weight = weights::KEYWORD_BASE 
                + (word.len() as f64) * weights::KEYWORD_LENGTH_BONUS_EN;
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
        let weight = weights::KEYWORD_BASE 
            + (chinese_word.len() as f64) * weights::KEYWORD_LENGTH_BONUS_CN;
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
    
    match intent {
        UserIntent::QueryConcept => {
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
        UserIntent::RequestAction => {
            let mut score = 0.0;
            
            // 基础意图匹配：降低单纯行动意图的加分，避免只要有“帮我做”就容易过阈值
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
                && (desc_lower.contains("调试") || desc_lower.contains("bug") || desc_lower.contains("修复") || desc_lower.contains("错误") || name_lower.contains("debugger"))
            {
                score += weights::INTENT_MATCH_HIGH;
            }

            score
        }
        UserIntent::SeekSolution => {
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
                && (name_lower.contains("debugger") || desc_lower.contains("调试") || desc_lower.contains("bug"))
            {
                score += weights::INTENT_MATCH_HIGH - weights::INTENT_MATCH_MEDIUM;
            }
            score
        }
        UserIntent::Casual => 0.0,
    }
}

/// 检查是否有否定上下文
fn has_negation_context(input: &str, _description: &str) -> bool {
    let negation_patterns = [
        "不要", "不是", "别", "不用", "无需", "不适合",
        "don't", "not", "no need", "not suitable",
    ];
    negation_patterns.iter().any(|p| input.contains(p))
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
            | "着" | "这" | "那" | "个" | "怎么" | "如何" | "为什么" | "请"
            | "可以" | "能" | "吗" | "呢" | "我" | "你" | "我们" | "你们"
            | "此" | "其" | "之" | "而" | "于" | "为" | "以" | "与" | "及"
            | "等" | "者"
    )
}

/// 中文停用字
fn is_chinese_stop_char(ch: char) -> bool {
    matches!(
        ch,
        '的' | '了' | '是' | '在' | '有' | '和' | '就' | '不' | '都' | '一'
            | '上' | '也' | '很' | '到' | '说' | '要' | '去' | '会' | '着' | '这'
            | '那' | '个' | '我' | '你' | '与' | '及' | '等' | '者' | '其'
            | '之' | '而' | '于' | '为' | '以'
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

    #[test]
    fn test_python_flask_question_does_not_match_code_review() {
        // 验证：询问 Python Flask 中的变量含义不应该匹配到 code-review 技能
        // 这是一个回归测试，防止标点符号（如 .）导致误匹配
        let skills = vec![skill(
            "code-review",
            "代码审查专家：仅针对源代码文件进行质量、安全、性能与可维护性评估。ONLY for reviewing SOURCE CODE files (e.g., .rs, .py, .js, .java, etc.). 如果用户询问的不是代码（如新闻、体育、数据、文档等），切勿选择此技能。",
            70,
        )];
        let input = "python flask 中 g.request_base_info_dict 这个是什么";
        let matched = match_skill(&skills, input);
        // 应该不匹配，因为这是在询问变量含义，不是请求代码审查
        assert!(matched.is_none(), "Python Flask 变量询问不应该匹配到 code-review 技能");
    }

    #[test]
    fn test_generate_code_does_not_match_code_review() {
        let skills = vec![skill(
            "code-review",
            "代码审查专家：仅针对源代码文件进行质量、安全、性能与可维护性评估。ONLY for reviewing SOURCE CODE files (e.g., .rs, .py, .js, .java, etc.). 如果用户询问的不是代码（如新闻、体育、数据、文档等），切勿选择此技能。",
            70,
        )];
        let input = "你帮我生成一段大约200行的python代码。需要可以运行，不需要依赖任何三方库";
        let matched = match_skill(&skills, input);
        assert!(matched.is_none(), "生成代码不应该匹配到 code-review");
    }
    
    #[test]
    fn test_intent_detection_query_concept() {
        assert_eq!(detect_intent("这是什么意思"), UserIntent::QueryConcept);
        assert_eq!(detect_intent("python flask 中 g.request 是什么"), UserIntent::QueryConcept);
        assert_eq!(detect_intent("what is this"), UserIntent::QueryConcept);
        assert_eq!(detect_intent("这个变量代表什么"), UserIntent::QueryConcept);
    }
    
    #[test]
    fn test_intent_detection_request_action() {
        assert_eq!(detect_intent("帮我重构这段代码"), UserIntent::RequestAction);
        assert_eq!(detect_intent("请帮我审查代码"), UserIntent::RequestAction);
        assert_eq!(detect_intent("help me fix this"), UserIntent::RequestAction);
        assert_eq!(detect_intent("帮我运行测试"), UserIntent::RequestAction);
    }
    
    #[test]
    fn test_intent_detection_seek_solution() {
        assert_eq!(detect_intent("怎么解决这个问题"), UserIntent::SeekSolution);
        assert_eq!(detect_intent("如何处理这个错误"), UserIntent::SeekSolution);
        assert_eq!(detect_intent("how to fix this"), UserIntent::SeekSolution);
        assert_eq!(detect_intent("为什么会出现这个错误"), UserIntent::SeekSolution);
    }
    
    #[test]
    fn test_code_review_should_match_review_request() {
        // 验证：明确的代码审查请求应该匹配 code-review
        let skills = vec![skill(
            "code-review",
            "代码审查专家：仅针对源代码文件进行质量、安全、性能与可维护性评估。ONLY for reviewing SOURCE CODE files (e.g., .rs, .py, .js, .java, etc.). 如果用户询问的不是代码（如新闻、体育、数据、文档等），切勿选择此技能。",
            70,
        )];
        let input = "帮我审查一下这段代码的质量";
        let matched = match_skill(&skills, input);
        assert!(matched.is_some(), "代码审查请求应该匹配 code-review 技能");
        assert_eq!(matched.unwrap().name, "code-review");
    }
    
    #[test]
    fn test_refactor_should_match_refactor_request() {
        // 验证：明确的重构请求应该匹配 refactor
        let skills = vec![skill(
            "refactor",
            "代码重构专家：ONLY for refactoring SOURCE CODE (improve structure, naming, readability, testability without changing behavior). 如果用户询问的不是代码（如数据整理、文档优化、新闻体育等），切勿选择此技能。不适用于修复报错、调试、处理异常等情况。",
            65,
        )];
        let input = "帮我重构这个函数，提高可读性";
        let matched = match_skill(&skills, input);
        assert!(matched.is_some(), "重构请求应该匹配 refactor 技能");
        assert_eq!(matched.unwrap().name, "refactor");
    }
    
    #[test]
    fn test_debugger_should_match_debug_request() {
        // 验证：调试请求应该匹配 debugger
        let skills = vec![skill(
            "debugger",
            "代码调试专家：ONLY for debugging SOURCE CODE issues (compile errors, runtime bugs, test failures, panic, exceptions). 如果用户询问的不是代码问题（如数据分析、业务问题、新闻体育等），切勿选择此技能。",
            70,
        )];
        let input = "代码编译报错了，帮我调试一下";
        let matched = match_skill(&skills, input);
        assert!(matched.is_some(), "调试请求应该匹配 debugger 技能");
        assert_eq!(matched.unwrap().name, "debugger");
    }
    
    #[test]
    fn test_question_about_code_not_match_action_skills() {
        // 验证：询问代码概念不应该匹配需要行动的技能
        let skills = vec![
            skill(
                "code-review",
                "代码审查专家：仅针对源代码文件进行质量、安全、性能与可维护性评估。如果用户询问的不是代码，切勿选择此技能。",
                70,
            ),
            skill(
                "refactor",
                "代码重构专家：ONLY for refactoring SOURCE CODE. 如果用户询问的不是代码，切勿选择此技能。",
                65,
            ),
        ];
        
        // 询问变量含义
        let input1 = "g.request_base_info_dict 这个变量是什么意思";
        assert!(match_skill(&skills, input1).is_none(), "询问变量含义不应该匹配任何技能");
        
        // 询问函数定义
        let input2 = "flask 中的 g 对象是什么";
        assert!(match_skill(&skills, input2).is_none(), "询问概念不应该匹配任何技能");
    }

    #[test]
    fn test_phrase_matching() {
        let skills = vec![skill(
            "refactor",
            "重构专家：提取函数、减少嵌套、优化代码结构，专门用于重构代码",
            65,
        )];
        
        // 输入包含技能描述中的短语
        let input = "帮我重构这段代码，提取函数，减少嵌套";
        let matched = match_skill(&skills, input);
        assert!(matched.is_some(), "短语匹配应该成功");
    }

    #[test]
    fn test_chinese_semantic_matching() {
        let skills = vec![skill(
            "debugger",
            "调试专家：帮助定位和修复代码中的 bug、错误、异常，专门用于调试",
            70,
        )];
        
        // 输入包含中文关键词
        let input = "代码有错误，帮我调试定位一下";
        let matched = match_skill(&skills, input);
        assert!(matched.is_some(), "中文语义匹配应该成功");
    }

    #[test]
    fn test_multiple_skills_selection() {
        // 验证：当有多个技能时，选择最匹配的
        let skills = vec![
            skill("refactor", "代码重构专家：优化代码结构、提取函数、去重复", 65),
            skill("debugger", "调试专家：修复 bug、错误、异常", 70),
            skill("code-review", "代码审查专家：评估代码质量、安全、性能", 65),
        ];
        
        // 明确的重构请求
        let input1 = "帮我重构这段代码，提取函数";
        let matched1 = match_skill(&skills, input1);
        assert_eq!(matched1.unwrap().name, "refactor");
        
        // 明确的调试请求（包含 bug 关键词）
        let input2 = "代码有 bug，帮我修复";
        let matched2 = match_skill(&skills, input2);
        assert_eq!(matched2.unwrap().name, "debugger");
        
        // 明确的审查请求
        let input3 = "帮我审查代码质量";
        let matched3 = match_skill(&skills, input3);
        assert_eq!(matched3.unwrap().name, "code-review");
    }
}

#[cfg(test)]
mod command_validation_tests {
    use crate::ai::tools::command_tools;

    #[test]
    fn execute_command_blocks_dangerous_programs() {
        assert!(command_tools::validate_execute_command("rm -rf /").is_err());
        assert!(command_tools::validate_execute_command("mv a b").is_err());
        assert!(command_tools::validate_execute_command("sudo ls").is_err());
    }
}
