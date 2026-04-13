use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
};

use rust_tools::commonw::FastMap;
use serde::{Deserialize, Serialize};

use crate::ai::{
    agents::{self, AgentManifest},
    history::Message,
};

pub struct RoutingDecision {
    pub agent_name: String,
    pub reason: &'static str,
}

pub trait AgentRouter: Send + Sync {
    fn route(
        &self,
        agents: &[AgentManifest],
        question: &str,
        history: &[Message],
        current_agent: &str,
    ) -> Option<RoutingDecision>;
}

const DEFAULT_AGENT_ROUTE_MODEL_RELATIVE_PATH: &str =
    "src/bin/ai/config/agent_route/agent_route_model.json";

static AGENT_ROUTE_MODEL_CACHE: LazyLock<Mutex<FastMap<PathBuf, Arc<AgentRouteModelFile>>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentRouteModelFile {
    version: u32,
    labels: Vec<String>,
    feature_config: FeatureConfig,
    bias: Vec<f64>,
    features: Vec<RouteFeature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureConfig {
    char_ngram_min: usize,
    char_ngram_max: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RouteFeature {
    token: String,
    idf: f64,
    weights: Vec<f64>,
}

fn default_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_AGENT_ROUTE_MODEL_RELATIVE_PATH)
}

fn load_route_model(path: &Path) -> Option<Arc<AgentRouteModelFile>> {
    let path_buf = path.to_path_buf();
    if let Ok(cache) = AGENT_ROUTE_MODEL_CACHE.lock()
        && let Some(model) = cache.get(&path_buf)
    {
        return Some(Arc::clone(model));
    }

    let text = fs::read_to_string(path).ok()?;
    let model: AgentRouteModelFile = serde_json::from_str(&text).ok()?;
    let model = Arc::new(model);

    if let Ok(mut cache) = AGENT_ROUTE_MODEL_CACHE.lock() {
        cache.insert(path_buf, Arc::clone(&model));
    }

    Some(model)
}

fn predict_agent(input: &str, model: &AgentRouteModelFile) -> Option<(String, f64)> {
    let normalized = normalize_text(input);
    if normalized.trim().is_empty() {
        return None;
    }

    let tf = extract_tfidf_features(&normalized, &model.feature_config);
    if tf.is_empty() {
        return None;
    }

    let mut scores = model.bias.clone();
    for feature in &model.features {
        if let Some(tf_value) = tf.get(feature.token.as_str()) {
            let weighted_tf = tf_value * feature.idf;
            for (idx, score) in scores.iter_mut().enumerate() {
                *score += weighted_tf * feature.weights[idx];
            }
        }
    }

    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();

    let mut best_idx = 0usize;
    let mut best_prob = probs[0];
    for (idx, prob) in probs.iter().enumerate().skip(1) {
        if *prob > best_prob {
            best_idx = idx;
            best_prob = *prob;
        }
    }

    let label = model.labels.get(best_idx)?.clone();
    Some((label, best_prob))
}

fn extract_tfidf_features(input: &str, cfg: &FeatureConfig) -> FastMap<String, f64> {
    let mut counts = FastMap::default();
    for ngram in extract_char_ngrams(input, cfg.char_ngram_min, cfg.char_ngram_max) {
        *counts.entry(ngram).or_insert(0.0) += 1.0;
    }

    let total = counts.values().sum::<f64>();
    if total <= f64::EPSILON {
        return counts;
    }

    for value in counts.values_mut() {
        *value /= total;
    }
    counts
}

fn extract_char_ngrams(input: &str, min_n: usize, max_n: usize) -> Vec<String> {
    let padded = format!("^{input}$");
    let chars = padded.chars().collect::<Vec<_>>();
    let mut out = Vec::new();

    for n in min_n..=max_n {
        if chars.len() < n {
            continue;
        }
        for window in chars.windows(n) {
            let token = window.iter().collect::<String>();
            if token.trim().is_empty() {
                continue;
            }
            out.push(token);
        }
    }

    out
}

fn normalize_text(input: &str) -> String {
    let mut normalized = String::new();
    let mut prev_space = false;

    for ch in input.to_lowercase().chars() {
        if ch == '\r' {
            continue;
        }
        if ch.is_whitespace() {
            if !prev_space {
                normalized.push(' ');
            }
            prev_space = true;
        } else {
            normalized.push(ch);
            prev_space = false;
        }
    }

    normalized.trim().to_string()
}

pub struct ModelRouter {
    model_path: PathBuf,
    confidence_threshold: f64,
}

impl ModelRouter {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            confidence_threshold: 0.45,
        }
    }
}

impl AgentRouter for ModelRouter {
    fn route(
        &self,
        agent_manifests: &[AgentManifest],
        question: &str,
        _history: &[Message],
        current_agent: &str,
    ) -> Option<RoutingDecision> {
        let model = load_route_model(&self.model_path)?;

        let (predicted_agent, confidence) = predict_agent(question, &model)?;

        let agent = agents::find_agent_by_name(agent_manifests, &predicted_agent)?;
        if !agent.is_primary() || agent.disabled || agent.hidden {
            return None;
        }

        if predicted_agent == current_agent {
            return None;
        }

        if confidence < self.confidence_threshold {
            let fallback = agents::find_agent_by_name(agent_manifests, "build")
                .filter(|a| a.is_primary() && !a.disabled && !a.hidden)
                .map(|a| a.name.clone())
                .unwrap_or_else(|| current_agent.to_string());

            if fallback == current_agent {
                return None;
            }

            return Some(RoutingDecision {
                agent_name: fallback,
                reason: "model-low-confidence",
            });
        }

        Some(RoutingDecision {
            agent_name: predicted_agent,
            reason: "model-predict",
        })
    }
}

pub struct HeuristicRouter;

impl AgentRouter for HeuristicRouter {
    fn route(
        &self,
        agent_manifests: &[AgentManifest],
        question: &str,
        history: &[Message],
        current_agent: &str,
    ) -> Option<RoutingDecision> {
        let intent = super::intent_recognition::detect_intent_with_model_path(
            question,
            &super::config::load_config().ok()?.intent_model_path,
        );

        let legacy_target = if should_auto_route_to_executor(&intent, question) {
            Some("executor")
        } else {
            None
        };

        let context_target = select_best_agent_by_context(agent_manifests, question, history);

        let fallback_agent_name = agents::find_agent_by_name(agent_manifests, "build")
            .filter(|agent| agent.is_primary() && !agent.disabled && !agent.hidden)
            .map(|agent| agent.name.clone())
            .unwrap_or_else(|| current_agent.to_string());

        let target_agent_name: String = if let Some(name) = legacy_target {
            if let Some(ctx_agent) = context_target {
                if agents::canonical_agent_name(&ctx_agent.name) == name {
                    name.to_string()
                } else {
                    ctx_agent.name.clone()
                }
            } else {
                name.to_string()
            }
        } else if let Some(ctx_agent) = context_target {
            ctx_agent.name.clone()
        } else {
            fallback_agent_name
        };

        if current_agent == target_agent_name {
            return None;
        }

        let agent = agents::find_agent_by_name(agent_manifests, &target_agent_name)?;
        if !agent.is_primary() || agent.disabled {
            return None;
        }

        let reason = if legacy_target.is_some() && legacy_target.unwrap() == target_agent_name {
            "complex-execution"
        } else if let Some(ctx) = context_target {
            if ctx.name == target_agent_name {
                "context-match"
            } else {
                "unknown"
            }
        } else {
            "fallback"
        };

        Some(RoutingDecision {
            agent_name: target_agent_name,
            reason,
        })
    }
}

fn should_auto_route_to_executor(
    intent: &super::intent_recognition::UserIntent,
    question: &str,
) -> bool {
    if intent.is_search_query() {
        return false;
    }
    if !matches!(
        intent.core,
        super::intent_recognition::CoreIntent::RequestAction
            | super::intent_recognition::CoreIntent::SeekSolution
    ) {
        return false;
    }

    let question = question.trim();
    if question.is_empty() || !contains_code_action_marker(question) {
        return false;
    }

    let char_count = question.chars().count();
    let line_count = question.lines().count();
    char_count >= auto_executor_length_threshold()
        || line_count >= 2
        || contains_complex_execution_marker(question)
}

fn select_best_agent_by_context<'a>(
    agent_manifests: &'a [AgentManifest],
    question: &str,
    history: &[Message],
) -> Option<&'a AgentManifest> {
    let question_lower = question.to_lowercase();

    let mut best: Option<&AgentManifest> = None;
    let mut best_score: f64 = 0.0;

    for agent in agent_manifests.iter() {
        if !agent.is_primary() || agent.disabled || agent.hidden {
            continue;
        }

        let score = score_agent_for_question(agent, &question_lower, history);
        if score > best_score {
            best_score = score;
            best = Some(agent);
        }
    }

    if best_score >= 5.0 {
        best
    } else {
        None
    }
}

fn score_agent_for_question(
    agent: &AgentManifest,
    question_lower: &str,
    history: &[Message],
) -> f64 {
    let question_ascii_terms = ascii_word_tokens(question_lower);
    let desc_lower = agent.description.to_lowercase();
    let desc_terms = description_terms(&desc_lower);
    let mut current_turn_score = 0.0;

    let agent_name_lower = agent.name.to_lowercase();
    if question_lower.contains(&agent_name_lower) {
        current_turn_score += 20.0;
    }

    for tag in agent.routing_tags_normalized() {
        if contains_tag_match(question_lower, &tag) {
            current_turn_score += 3.0;
        }
    }

    let desc_overlap = description_overlap_count(&question_ascii_terms, &desc_terms);
    if desc_overlap > 0 {
        current_turn_score += desc_overlap.min(2) as f64;
    }

    let mut score = current_turn_score;

    if current_turn_score > 0.0 && !history.is_empty() {
        let recent_entries: Vec<_> = history.iter().rev().take(4).collect();
        for entry in recent_entries {
            if let Some(text) = extract_text_from_message(entry) {
                let entry_lower = text.to_lowercase();
                let agent_name_lower = agent.name.to_lowercase();
                if entry_lower.contains(&agent_name_lower) {
                    score += 2.0;
                }
                let history_terms = ascii_word_tokens(&entry_lower);
                let history_overlap = description_overlap_count(&history_terms, &desc_terms);
                if history_overlap > 0 {
                    score += history_overlap.min(2) as f64 * 0.5;
                }
            }
        }
    }

    let word_count = question_ascii_terms.len();
    if word_count <= 5
        && agent
            .model_tier
            .as_ref()
            .is_some_and(|t| matches!(t, agents::AgentModelTier::Light))
    {
        score += 2.0;
    }
    if word_count >= 20
        && agent
            .model_tier
            .as_ref()
            .is_some_and(|t| matches!(t, agents::AgentModelTier::Heavy))
    {
        score += 2.0;
    }

    score
}

fn auto_executor_length_threshold() -> usize {
    let cfg = crate::commonw::configw::get_all_config();
    cfg.get_opt("ai.agents.auto_route.executor_min_chars")
        .or_else(|| cfg.get_opt("ai.agents.auto_route.openclaw_min_chars"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(48)
}

fn contains_complex_execution_marker(question: &str) -> bool {
    let lower = question.to_lowercase();
    [
        "然后", "同时", "顺便", "一步步", "分步骤", "自动", "完整", "端到端", "闭环", "multi-step",
        "end-to-end", "step by step", "across", "implement", "refactor", "debug", "fix", "repair",
        "integrate", "migrate",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn contains_code_action_marker(question: &str) -> bool {
    let lower = question.to_lowercase();
    [
        "帮我", "修", "修复", "修改", "改一下", "实现", "添加", "扩展", "重构", "排查", "调试", "处理",
        "完成", "补", "优化", "迁移", "接入", "联调", "修一下", "报错", "panic", "error", "failing",
        "test", "build", "cargo", "fix", "implement", "add", "extend", "refactor", "debug",
        "update", "wire", "integrate", "migrate",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn extract_text_from_message(msg: &Message) -> Option<String> {
    use serde_json::Value;
    match &msg.content {
        Value::String(s) => Some(s.clone()),
        Value::Array(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|part| {
                    if let Value::Object(obj) = part {
                        if let Some(Value::String(s)) = obj.get("text") {
                            return Some(s.clone());
                        }
                    }
                    None
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(" "))
            }
        }
        _ => None,
    }
}

fn ascii_word_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn contains_tag_match(question_lower: &str, tag: &str) -> bool {
    if tag.is_ascii()
        && tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        ascii_word_tokens(question_lower)
            .iter()
            .any(|token| token == tag)
    } else {
        question_lower.contains(tag)
    }
}

fn description_terms(desc_lower: &str) -> HashSet<String> {
    ascii_word_tokens(desc_lower)
        .into_iter()
        .filter(|term| term.len() >= 5)
        .collect()
}

fn description_overlap_count(question_terms: &[String], desc_terms: &HashSet<String>) -> usize {
    question_terms
        .iter()
        .filter(|term| term.len() >= 5)
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|term| desc_terms.contains(*term))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
    use crate::ai::driver::intent_recognition::{CoreIntent, IntentModifiers, UserIntent};

    fn primary_agent(name: &str, description: &str, routing_tags: &[&str]) -> AgentManifest {
        AgentManifest {
            name: name.to_string(),
            description: description.to_string(),
            mode: AgentMode::Primary,
            model: None,
            temperature: None,
            max_steps: None,
            prompt: String::new(),
            system_prompt: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            routing_tags: routing_tags.iter().map(|tag| (*tag).to_string()).collect(),
            model_tier: Some(AgentModelTier::Heavy),
            disabled: false,
            hidden: false,
            color: None,
            source_path: None,
        }
    }

    #[test]
    fn auto_routes_complex_execution_requests_to_executor() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let question = "帮我实现这个 agent 的自动执行能力，然后跑检查并修掉相关报错";
        assert!(should_auto_route_to_executor(&intent, question));
    }

    #[test]
    fn does_not_route_simple_concept_questions_to_executor() {
        let intent = UserIntent::new(CoreIntent::QueryConcept);
        assert!(!should_auto_route_to_executor(&intent, "Rust 的 crate 是什么？"));
    }

    #[test]
    fn does_not_route_search_queries_to_executor() {
        let intent = UserIntent {
            core: CoreIntent::RequestAction,
            modifiers: IntentModifiers {
                is_search_query: true,
                target_resource: Some("tool".to_string()),
                negation: false,
            },
        };
        assert!(!should_auto_route_to_executor(&intent, "帮我找几个调试工具"));
    }

    #[test]
    fn simple_file_line_query_does_not_get_pulled_to_plan_by_history() {
        let build = primary_agent("build", "Default agent for development work", &["fix", "debug"]);
        let plan = primary_agent(
            "plan",
            "Read-only agent for planning and analysis without making changes",
            &["plan", "planning", "review", "analyze", "analysis", "总结", "分析"],
        );

        let history = vec![
            Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String(
                    "plan agent can analyze and summarize".to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: serde_json::Value::String("继续用 plan 分析".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let agents = [build, plan];
        let selected = select_best_agent_by_context(
            &agents,
            "@/Users/bytedance/rust_tools/src/bin/ai/models.rs 这个文件有几行",
            &history,
        );

        assert!(selected.is_none());
    }

    #[test]
    fn history_alone_cannot_route_to_plan_without_current_turn_signal() {
        let build = primary_agent("build", "Default agent for development work", &["fix", "debug"]);
        let plan = primary_agent(
            "plan",
            "Read-only agent for planning and analysis without making changes",
            &["plan", "planning", "review", "analyze", "analysis", "总结", "分析"],
        );

        let history = vec![
            Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String(
                    "plan plan plan analyze summarize".to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: serde_json::Value::String("继续 plan".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let agents = [build, plan];
        let selected = select_best_agent_by_context(&agents, "你好", &history);

        assert!(selected.is_none());
    }
}
