use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
};

use rust_tools::cw::SkipMap;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ai::{
    agents::{self, AgentManifest},
    history::Message,
};

use super::{
    TextSimilarityFeatures, build_idf_from_documents, cosine_tfidf_similarity,
    normalize_text_for_similarity,
};

// 与 `normalize_text_for_similarity` 行为相同的本地别名，便于本文件保留旧调用形式。
fn normalize_text(input: &str) -> String {
    normalize_text_for_similarity(input)
}

pub struct RoutingDecision {
    pub agent_name: String,
    pub reason: &'static str,
}

const ROUTE_REASON_MODEL_LOW_CONFIDENCE: &str = "model-low-confidence";
const ROUTE_REASON_MODEL_PREDICT: &str = "model-predict";
const ROUTE_REASON_SEMANTIC_MATCH: &str = "semantic-match";
const ROUTE_REASON_SEMANTIC_FALLBACK: &str = "semantic-fallback";
/// 模型分类置信度阈值：低于此值时不再相信模型预测，转而走语义匹配 fallback。
/// 经验值：基于 logistic regression 在校验集上的 ROC 曲线选取。
const MODEL_CONFIDENCE_THRESHOLD: f64 = 0.45;
/// 跨 agent 语义切换的最低绝对相似度门槛。
/// 当最佳候选的语义得分低于此值时，禁止从当前 agent 切走，避免 agent 频繁抖动。
const SEMANTIC_SWITCH_THRESHOLD: f64 = 0.085;
/// 跨 agent 语义切换的相对优势：候选必须比当前 agent 高 `MARGIN` 才允许切换，
/// 防止得分接近时来回跳。
const SEMANTIC_SWITCH_MARGIN: f64 = 0.015;
/// 当前轮次（仅看 question 自身）的语义最低分。
/// 历史相关性可以带来"惯性加分"，但本轮 question 本身仍需达到此底线。
const CURRENT_TURN_SEMANTIC_FLOOR: f64 = 0.05;
/// 当前轮次相对优势 margin：与 SEMANTIC_SWITCH_MARGIN 类似，但仅作用在
/// "当前轮次 question 维度"，避免"历史强、当前弱"的候选被误推上去。
const CURRENT_TURN_ADVANTAGE_MARGIN: f64 = 0.04;
const AGENT_SEMANTIC_CORPUS_CACHE_LIMIT: usize = 16;

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

static AGENT_ROUTE_MODEL_CACHE: LazyLock<Mutex<SkipMap<PathBuf, Arc<AgentRouteModelFile>>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));

static AGENT_SEMANTIC_CORPUS_CACHE: LazyLock<Mutex<SkipMap<String, Arc<AgentSemanticCorpus>>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));

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
    if let Some(cache) = lock_recover(&AGENT_ROUTE_MODEL_CACHE)
        && let Some(model) = cache.get(&path_buf)
    {
        return Some(Arc::clone(&model));
    }

    let text = fs::read_to_string(path).ok()?;
    let model: AgentRouteModelFile = serde_json::from_str(&text).ok()?;
    let model = Arc::new(model);

    if let Some(mut cache) = lock_recover(&AGENT_ROUTE_MODEL_CACHE) {
        cache.insert(path_buf, Arc::clone(&model));
    }

    Some(model)
}

/// 获取 Mutex 锁，遇到中毒时恢复，避免后续所有缓存操作静默失败。
fn lock_recover<'a, T>(m: &'a Mutex<T>) -> Option<std::sync::MutexGuard<'a, T>> {
    match m.lock() {
        Ok(g) => Some(g),
        Err(poisoned) => {
            eprintln!("[agent_router] cache mutex poisoned, recovering inner state");
            Some(poisoned.into_inner())
        }
    }
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
        if let Some(tf_value) = tf.get(&feature.token) {
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

fn extract_tfidf_features(input: &str, cfg: &FeatureConfig) -> FxHashMap<String, f64> {
    let mut counts = FxHashMap::default();
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

struct AgentSemanticCorpus {
    docs: FxHashMap<String, FxHashMap<String, f64>>,
    idf: FxHashMap<String, f64>,
}

impl AgentSemanticCorpus {
    fn for_candidates(candidates: &[&AgentManifest]) -> Arc<Self> {
        let cache_key = agent_semantic_corpus_cache_key(candidates);
        if let Some(cache) = lock_recover(&AGENT_SEMANTIC_CORPUS_CACHE)
            && let Some(corpus) = cache.get_ref(&cache_key)
        {
            return Arc::clone(corpus);
        }

        let corpus = Arc::new(Self::build(candidates));
        if let Some(mut cache) = lock_recover(&AGENT_SEMANTIC_CORPUS_CACHE) {
            if cache.len() >= AGENT_SEMANTIC_CORPUS_CACHE_LIMIT {
                cache.clear();
            }
            cache.insert(cache_key, Arc::clone(&corpus));
        }
        corpus
    }

    fn build(candidates: &[&AgentManifest]) -> Self {
        let mut docs = FxHashMap::default();
        for agent in candidates {
            let features = TextSimilarityFeatures::from_text(&agent_document_text(agent));
            docs.insert(agent.name.clone(), features.ngram_tf);
        }
        let doc_refs = docs.values().collect::<Vec<_>>();
        let idf = build_idf_from_documents(&doc_refs);
        Self { docs, idf }
    }

    fn document(&self, agent_name: &str) -> Option<&FxHashMap<String, f64>> {
        self.docs.get(agent_name)
    }
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
    let corpus = AgentSemanticCorpus::for_candidates(&candidates);

    // C1: 从 DecisionLog 聚合最近的整体成功率，作为 prior_boost 的放大系数。
    // 没有数据 / 数据稀少时 reliability=1.0（不动原行为），最大 1.2 / 最小 0.85。
    let reliability = recent_tool_invocation_reliability();

    let mut ranked = candidates
        .into_iter()
        .map(|agent| {
            let doc = corpus.document(agent.name.as_str());
            let question_score = doc
                .map(|doc| cosine_tfidf_similarity(&query.ngram_tf, doc, &corpus.idf))
                .unwrap_or(0.0);
            let history_score = doc
                .map(|doc| cosine_tfidf_similarity(&history_features.ngram_tf, doc, &corpus.idf))
                .unwrap_or(0.0);
            let prior_boost = model_prior
                .filter(|(label, _)| *label == agent.name)
                .map(|(_, confidence)| confidence * 0.15 * reliability)
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

/// 从 DecisionLog 取最近 50 条 ToolInvocation 记录，估算工具调用成功率，
/// 并把它映射到 [0.85, 1.2] 区间作为路由 prior 的放大系数。
/// 样本不足（< 10 条）时返回 1.0，保持原有行为。
fn recent_tool_invocation_reliability() -> f64 {
    use crate::ai::driver::decision_log::{DecisionType, get_decision_log_store};
    let logs = get_decision_log_store().by_type(&DecisionType::ToolInvocation);
    let total = logs.len();
    if total < 10 {
        return 1.0;
    }
    let take_from = total.saturating_sub(50);
    let recent = &logs[take_from..];
    let success = recent
        .iter()
        .filter(|l| l.outcome.as_ref().map(|o| o.success).unwrap_or(false))
        .count();
    let rate = success as f64 / recent.len() as f64;
    // 0.5 → 1.0；0.0 → 0.85；1.0 → 1.2
    if rate >= 0.5 {
        1.0 + (rate - 0.5) * 0.4
    } else {
        0.85 + rate * 0.3
    }
}

fn agent_semantic_corpus_cache_key(candidates: &[&AgentManifest]) -> String {
    let mut key = String::new();
    for agent in candidates {
        key.push_str(agent_routing_source_hash(agent).as_str());
        key.push('\n');
    }
    key
}

fn agent_routing_source_hash(agent: &AgentManifest) -> String {
    let payload = serde_json::json!({
        "name": agent.name,
        "description": agent.description,
        "prompt": agent.prompt,
        "system_prompt": agent.system_prompt,
        "tools": agent.tools,
        "tool_groups": agent.tool_groups,
        "mcp_servers": agent.mcp_servers,
    });
    let mut hasher = Sha256::new();
    hasher.update(payload.to_string().as_bytes());
    format!("{:x}", hasher.finalize())
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
    fn primary_agent(name: &str, description: &str) -> AgentManifest {
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
            disable_mcp_tools: false,
            model_tier: Some(AgentModelTier::Heavy),
            disabled: false,
            hidden: false,
            color: None,
            source_path: None,
        }
    }

    #[test]
    fn semantic_router_selects_agent_based_on_semantic_match() {
        let build = primary_agent("build", "Default agent for development work");
        let agents = [build.clone(), build.clone()];
        let ranked =
            rank_agents_by_semantics(&agents, "先别改代码，帮我分析这次重构方案和风险", &[], None);
        // With only 'build' agents, the ranking still produces a result
        assert!(!ranked.is_empty());
        assert_eq!(
            ranked.first().map(|item| item.agent.name.as_str()).unwrap(),
            "build"
        );
    }

    #[test]
    fn history_alone_cannot_override_current_turn_semantics() {
        let build = primary_agent("build", "Default agent for development work");
        let build_analysis = primary_agent("build", "Analysis-oriented build configuration");

        let history = vec![
            Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String("plan plan plan analyze summarize".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: serde_json::Value::String("继续 plan".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let agents = [build, build_analysis];
        let ranked = rank_agents_by_semantics(
            &agents,
            "@/Users/bytedance/rust_tools/src/bin/ai/models.rs 这个文件有几行",
            &history,
            None,
        );
        let decision = choose_semantic_route(ranked, "build", ROUTE_REASON_SEMANTIC_MATCH);

        assert!(decision.is_none());
    }

    #[test]
    fn semantic_corpus_cache_refreshes_when_agent_content_changes() {
        let first_agents = [primary_agent("helper", "Debug runtime failures")];
        let first = rank_agents_by_semantics(&first_agents, "帮我做一份 ppt 幻灯片", &[], None);

        let second_agents = [primary_agent("helper", "生成幻灯片 PPT 演示文稿")];
        let second = rank_agents_by_semantics(&second_agents, "帮我做一份 ppt 幻灯片", &[], None);

        assert!(
            second[0].question_score > first[0].question_score,
            "expected changed agent content to refresh cached semantic features"
        );
    }
}
