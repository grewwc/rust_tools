use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
};

use rust_tools::commonw::FastMap;
use serde::{Deserialize, Serialize};

use super::intent_recognition::{CoreIntent, IntentModifiers, UserIntent};

const DEFAULT_INTENT_MODEL_RELATIVE_PATH: &str = "src/bin/ai/config/intent/intent_model.json";

static INTENT_MODEL_CACHE: LazyLock<Mutex<FastMap<PathBuf, Arc<IntentModelFile>>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IntentModelFile {
    pub(crate) version: u32,
    pub(crate) labels: Vec<String>,
    pub(crate) feature_config: FeatureConfig,
    pub(crate) runtime_rules: RuntimeRules,
    pub(crate) bias: Vec<f64>,
    pub(crate) features: Vec<IntentFeature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FeatureConfig {
    pub(crate) char_ngram_min: usize,
    pub(crate) char_ngram_max: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IntentFeature {
    pub(crate) token: String,
    pub(crate) idf: f64,
    pub(crate) weights: Vec<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RuntimeRules {
    #[serde(default)]
    pub(crate) search_patterns: Vec<String>,
    #[serde(default)]
    pub(crate) negation_patterns: Vec<String>,
    #[serde(default)]
    pub(crate) resource_keywords: Vec<ResourceKeywordRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ResourceKeywordRule {
    pub(crate) pattern: String,
    pub(crate) resource: String,
}

pub(crate) fn default_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_INTENT_MODEL_RELATIVE_PATH)
}

pub(crate) fn detect_intent(input: &str, model_path: Option<&Path>) -> UserIntent {
    let normalized = normalize_text(input);
    let default_path;
    let model_path = if let Some(model_path) = model_path {
        model_path
    } else {
        default_path = default_model_path();
        default_path.as_path()
    };
    let Some(model) = load_model_cached(model_path) else {
        return UserIntent::new(CoreIntent::Casual);
    };

    let modifiers = detect_modifiers(&normalized, &model.runtime_rules);
    let mut core = predict_core_intent(&normalized, &model);
    if modifiers.is_search_query && core == CoreIntent::Casual {
        core = CoreIntent::RequestAction;
    }
    UserIntent { core, modifiers }
}

pub(crate) fn load_model_file(path: &Path) -> Result<IntentModelFile, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read intent model {}: {err}", path.display()))?;
    let model: IntentModelFile = serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse intent model {}: {err}", path.display()))?;
    validate_model(&model)?;
    Ok(model)
}

fn load_model_cached(path: &Path) -> Option<Arc<IntentModelFile>> {
    let path = path.to_path_buf();
    if let Ok(cache) = INTENT_MODEL_CACHE.lock()
        && let Some(model) = cache.get(&path)
    {
        return Some(Arc::clone(model));
    }

    let model = load_model_file(&path).ok()?;
    let model = Arc::new(model);

    if let Ok(mut cache) = INTENT_MODEL_CACHE.lock() {
        cache.insert(path, Arc::clone(&model));
    }

    Some(model)
}

fn validate_model(model: &IntentModelFile) -> Result<(), String> {
    if model.labels.is_empty() {
        return Err("intent model labels are empty".to_string());
    }
    if model.bias.len() != model.labels.len() {
        return Err("intent model bias size does not match labels".to_string());
    }
    if model.feature_config.char_ngram_min == 0
        || model.feature_config.char_ngram_max < model.feature_config.char_ngram_min
    {
        return Err("invalid char ngram config".to_string());
    }
    for feature in &model.features {
        if feature.weights.len() != model.labels.len() {
            return Err(format!(
                "feature '{}' weight size does not match labels",
                feature.token
            ));
        }
        if feature.token.trim().is_empty() {
            return Err("feature token cannot be empty".to_string());
        }
    }
    Ok(())
}

fn detect_modifiers(input: &str, rules: &RuntimeRules) -> IntentModifiers {
    let mut modifiers = IntentModifiers::default();

    if rules.search_patterns.iter().any(|p| input.contains(p)) {
        modifiers.is_search_query = true;
        modifiers.target_resource = extract_target_resource(input, rules);
    }

    modifiers.negation = rules.negation_patterns.iter().any(|p| input.contains(p));
    modifiers
}

fn extract_target_resource(input: &str, rules: &RuntimeRules) -> Option<String> {
    rules
        .resource_keywords
        .iter()
        .find(|rule| !rule.pattern.trim().is_empty() && input.contains(rule.pattern.as_str()))
        .map(|rule| rule.resource.clone())
}

fn predict_core_intent(input: &str, model: &IntentModelFile) -> CoreIntent {
    if input.trim().is_empty() {
        return CoreIntent::Casual;
    }

    let tf = extract_tfidf_features(input, &model.feature_config);
    if tf.is_empty() {
        return CoreIntent::Casual;
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

    let mut best_idx = 0usize;
    let mut best_score = scores[0];
    for (idx, score) in scores.iter().enumerate().skip(1) {
        if *score > best_score {
            best_idx = idx;
            best_score = *score;
        }
    }
    label_to_core(model.labels.get(best_idx).map(|s| s.as_str()))
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

fn label_to_core(label: Option<&str>) -> CoreIntent {
    match label.unwrap_or("casual") {
        "query_concept" => CoreIntent::QueryConcept,
        "request_action" => CoreIntent::RequestAction,
        "seek_solution" => CoreIntent::SeekSolution,
        _ => CoreIntent::Casual,
    }
}
