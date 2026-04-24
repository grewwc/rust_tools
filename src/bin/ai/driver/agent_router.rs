use std::{
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

use super::{
    build_idf_from_documents, cosine_tfidf_similarity, normalize_text_for_similarity,
    TextSimilarityFeatures,
};

pub struct RoutingDecision {
    pub agent_name: String,
    pub reason: &'static str,
}

const ROUTE_REASON_MODEL_LOW_CONFIDENCE: &str = "model-low-confidence";
const ROUTE_REASON_MODEL_PREDICT: &str = "model-predict";
const ROUTE_REASON_SEMANTIC_MATCH: &str = "semantic-match";
const ROUTE_REASON_SEMANTIC_FALLBACK: &str = "semantic-fallback";
const MODEL_CONFIDENCE_THRESHOLD: f64 = 0.45;
const SEMANTIC_SWITCH_THRESHOLD: f64 = 0.085;
const SEMANTIC_SWITCH_MARGIN: f64 = 0.015;
const CURRENT_TURN_SEMANTIC_FLOOR: f64 = 0.05;
const CURRENT_TURN_ADVANTAGE_MARGIN: f64 = 0.04;

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
        history: &[Message],
        current_agent: &str,
    ) -> Option<RoutingDecision> {
        let model = load_route_model(&self.model_path)?;

        let (predicted_agent, confidence) = predict_agent(question, &model)?;
        let scored = rank_agents_by_semantics(
            agent_manifests,
            question,
            history,
            Some((predicted_agent.as_str(), confidence)),
        );
        if confidence < self.confidence_threshold {
            return choose_semantic_route(scored, current_agent, ROUTE_REASON_MODEL_LOW_CONFIDENCE);
        }

        let agent = agents::find_agent_by_name(agent_manifests, &predicted_agent)
            .filter(|agent| agent.is_primary() && !agent.disabled && !agent.hidden)?;
        if predicted_agent != current_agent {
            let semantic_decision =
                choose_semantic_route(scored.clone(), current_agent, ROUTE_REASON_MODEL_PREDICT);
            if semantic_decision
                .as_ref()
                .is_some_and(|decision| decision.agent_name == predicted_agent)
            {
                return semantic_decision;
            }
        }

        let semantic_fallback =
            choose_semantic_route(scored, current_agent, ROUTE_REASON_SEMANTIC_FALLBACK);
        if semantic_fallback.is_some() {
            return semantic_fallback;
        }
        if agent.name == current_agent {
            return None;
        }
        Some(RoutingDecision {
            agent_name: predicted_agent,
            reason: ROUTE_REASON_MODEL_PREDICT,
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
        let scored = rank_agents_by_semantics(agent_manifests, question, history, None);
        choose_semantic_route(scored, current_agent, ROUTE_REASON_SEMANTIC_MATCH)
    }
}

#[derive(Clone)]
struct ScoredAgent<'a> {
    agent: &'a AgentManifest,
    score: f64,
    question_score: f64,
    history_score: f64,
}

fn rank_agents_by_semantics<'a>(
    agent_manifests: &'a [AgentManifest],
    question: &str,
    history: &[Message],
    model_prior: Option<(&str, f64)>,
) -> Vec<ScoredAgent<'a>> {
    let candidates = agent_manifests
        .iter()
        .filter(|agent| agent.is_primary() && !agent.disabled && !agent.hidden)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Vec::new();
    }

    let query = TextSimilarityFeatures::from_text(question);
    let history_text = recent_history_text(history);
    let history_features = TextSimilarityFeatures::from_text(&history_text);
    let docs = candidates
        .iter()
        .map(|agent| TextSimilarityFeatures::from_text(&agent_document_text(agent)))
        .collect::<Vec<_>>();
    let doc_refs = docs.iter().map(|doc| &doc.ngram_tf).collect::<Vec<_>>();
    let idf = build_idf_from_documents(&doc_refs);

    let mut ranked = candidates
        .into_iter()
        .zip(docs)
        .map(|(agent, doc)| {
            let question_score = cosine_tfidf_similarity(&query.ngram_tf, &doc.ngram_tf, &idf);
            let history_score =
                cosine_tfidf_similarity(&history_features.ngram_tf, &doc.ngram_tf, &idf);
            let prior_boost = model_prior
                .filter(|(label, _)| *label == agent.name)
                .map(|(_, confidence)| confidence * 0.15)
                .unwrap_or(0.0);
            ScoredAgent {
                agent,
                score: question_score + history_score * 0.35 + prior_boost,
                question_score,
                history_score,
            }
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    ranked
}

fn choose_semantic_route(
    ranked: Vec<ScoredAgent<'_>>,
    current_agent: &str,
    reason: &'static str,
) -> Option<RoutingDecision> {
    let best = ranked.first()?;
    if best.score < SEMANTIC_SWITCH_THRESHOLD {
        return None;
    }
    if best.question_score < CURRENT_TURN_SEMANTIC_FLOOR {
        return None;
    }
    if best.agent.name == current_agent {
        return None;
    }
    let current_score = ranked
        .iter()
        .find(|item| item.agent.name == current_agent)
        .map(|item| item.score)
        .unwrap_or(0.0);
    let current_question_score = ranked
        .iter()
        .find(|item| item.agent.name == current_agent)
        .map(|item| item.question_score)
        .unwrap_or(0.0);
    if best.score <= current_score + SEMANTIC_SWITCH_MARGIN {
        return None;
    }
    if best.question_score <= current_question_score + CURRENT_TURN_ADVANTAGE_MARGIN {
        return None;
    }
    Some(RoutingDecision {
        agent_name: best.agent.name.clone(),
        reason,
    })
}

fn agent_document_text(agent: &AgentManifest) -> String {
    let mut parts = vec![agent.name.clone(), agent.description.clone()];
    if !agent.routing_tags.is_empty() {
        parts.push(agent.routing_tags.join(" "));
    }
    if !agent.tools.is_empty() {
        parts.push(agent.tools.join(" "));
    }
    if !agent.tool_groups.is_empty() {
        parts.push(agent.tool_groups.join(" "));
    }
    if !agent.mcp_servers.is_empty() {
        parts.push(agent.mcp_servers.join(" "));
    }
    if !agent.prompt.trim().is_empty() {
        parts.push(agent.prompt.chars().take(3000).collect());
    }
    if let Some(system_prompt) = &agent.system_prompt
        && !system_prompt.trim().is_empty()
    {
        parts.push(system_prompt.chars().take(1200).collect());
    }
    normalize_text_for_similarity(&parts.join("\n"))
}

fn recent_history_text(history: &[Message]) -> String {
    history
        .iter()
        .rev()
        .take(4)
        .filter_map(extract_text_from_message)
        .collect::<Vec<_>>()
        .join("\n")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
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
    fn semantic_router_prefers_planning_agent_for_analysis_request() {
        let build = primary_agent("build", "Default agent for development work", &["fix", "debug"]);
        let plan = primary_agent(
            "plan",
            "Read-only agent for planning and analysis without making changes",
            &["plan", "planning", "review", "analyze", "analysis", "总结", "分析"],
        );
        let agents = [build, plan];
        let ranked = rank_agents_by_semantics(
            &agents,
            "先别改代码，帮我分析这次重构方案和风险",
            &[],
            None,
        );
        assert_eq!(ranked.first().map(|item| item.agent.name.as_str()), Some("plan"));
    }

    #[test]
    fn history_alone_cannot_override_current_turn_semantics() {
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
        let ranked = rank_agents_by_semantics(
            &agents,
            "@/Users/bytedance/rust_tools/src/bin/ai/models.rs 这个文件有几行",
            &history,
            None,
        );
        let decision = choose_semantic_route(ranked, "build", ROUTE_REASON_SEMANTIC_MATCH);

        assert!(decision.is_none());
    }
}
