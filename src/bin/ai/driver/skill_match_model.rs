use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
};

use rust_tools::commonw::FastMap;
use serde::{Deserialize, Serialize};

const DEFAULT_SKILL_MATCH_MODEL_RELATIVE_PATH: &str =
    "src/bin/ai/config/skill_match/skill_match_model.json";

static SKILL_MATCH_MODEL_CACHE: LazyLock<Mutex<FastMap<PathBuf, Arc<SkillMatchModelFile>>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SkillMatchModelFile {
    version: u32,
    labels: Vec<String>,
    feature_config: FeatureConfig,
    bias: Vec<f64>,
    features: Vec<SkillMatchFeature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureConfig {
    char_ngram_min: usize,
    char_ngram_max: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SkillMatchFeature {
    token: String,
    idf: f64,
    weights: Vec<f64>,
}

pub(crate) fn default_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_SKILL_MATCH_MODEL_RELATIVE_PATH)
}

fn load_model_file(path: &Path) -> Result<SkillMatchModelFile, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read skill match model {}: {err}", path.display()))?;
    let model: SkillMatchModelFile = serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse skill match model {}: {err}", path.display()))?;
    validate_model(&model)?;
    Ok(model)
}

fn load_model_cached(path: &Path) -> Option<Arc<SkillMatchModelFile>> {
    let path_buf = path.to_path_buf();
    if let Ok(cache) = SKILL_MATCH_MODEL_CACHE.lock()
        && let Some(model) = cache.get(&path_buf)
    {
        return Some(Arc::clone(model));
    }

    let model = load_model_file(&path).ok()?;
    let model = Arc::new(model);

    if let Ok(mut cache) = SKILL_MATCH_MODEL_CACHE.lock() {
        cache.insert(path_buf, Arc::clone(&model));
    }

    Some(model)
}

fn validate_model(model: &SkillMatchModelFile) -> Result<(), String> {
    if model.labels.is_empty() {
        return Err("skill match model labels are empty".to_string());
    }
    if model.bias.len() != model.labels.len() {
        return Err("skill match model bias size does not match labels".to_string());
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

pub(crate) struct SkillMatchResult {
    pub(crate) label: String,
    pub(crate) confidence: f64,
    pub(crate) probabilities: Vec<(String, f64)>,
}

pub(crate) fn predict_skill(input: &str, model_path: &Path) -> Option<SkillMatchResult> {
    let model = load_model_cached(model_path)?;
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
    let probabilities = model
        .labels
        .iter()
        .zip(probs.iter())
        .map(|(l, p)| (l.clone(), *p))
        .collect();

    Some(SkillMatchResult {
        label,
        confidence: best_prob,
        probabilities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_exists() {
        let path = default_model_path();
        assert!(
            path.exists(),
            "missing bundled skill match model: {}",
            path.display()
        );
    }

    #[test]
    fn test_default_model_loads() {
        let path = default_model_path();
        let model = load_model_file(&path);
        assert!(model.is_ok(), "failed to load bundled skill match model");
    }

    #[test]
    fn test_predict_none_for_image_request_without_dynamic_skill() {
        let path = default_model_path();
        let result = predict_skill("帮我生成一张风景图", &path);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(
            r.label, "none",
            "expected none, got {} (confidence={:.4})",
            r.label, r.confidence
        );
    }

    #[test]
    fn test_predict_returns_label_for_skill_like_request() {
        let path = default_model_path();
        let result = predict_skill("帮我 review 这段代码", &path);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(
            !r.label.is_empty(),
            "expected non-empty label (confidence={:.4})",
            r.confidence
        );
    }

    #[test]
    fn test_predict_none_for_general_coding_request() {
        let path = default_model_path();
        let result = predict_skill("帮我实现这个接口，顺便补上测试", &path);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(
            r.label, "none",
            "expected none, got {} (confidence={:.4})",
            r.label, r.confidence
        );
    }

    #[test]
    fn test_predict_none_for_casual_chat() {
        let path = default_model_path();
        let result = predict_skill("你好，今天怎么样", &path);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(
            r.label, "none",
            "expected none, got {} (confidence={:.4})",
            r.label, r.confidence
        );
    }
}
