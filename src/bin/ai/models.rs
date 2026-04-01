use super::cli::ParsedCli;
use super::model_names;

pub(super) fn is_vl_model(model: &str) -> bool {
    model_names::find_by_name(model)
        .map(|m| m.is_vl)
        .unwrap_or(false)
}

pub(super) fn search_enabled(model: &str) -> bool {
    model_names::find_by_name(model)
        .map(|m| m.search_enabled)
        .unwrap_or(true)
}

pub(super) fn tools_enabled(model: &str) -> bool {
    model_names::find_by_name(model)
        .map(|m| m.tools_default_enabled)
        .unwrap_or(true)
}

pub(super) fn enable_thinking(model: &str) -> bool {
    model_names::find_by_name(model)
        .map(|m| m.enable_thinking)
        .unwrap_or(false)
}

fn all_model_names() -> Vec<String> {
    model_names::all().iter().map(|m| m.name.clone()).collect()
}

fn vl_model_names() -> Vec<String> {
    model_names::all()
        .iter()
        .filter(|m| m.is_vl)
        .map(|m| m.name.clone())
        .collect()
}

fn default_model() -> String {
    model_names::all().first().map(|m| m.name.as_str().to_owned()).unwrap_or_else(|| {
            eprintln!("[model_names] models.json is empty");
            std::process::exit(1);
        })
}

fn default_vl_model() -> String {
    model_names::all()
        .iter()
        .find(|m| m.is_vl)
        .map(|m| m.name.as_str().to_owned())
        .unwrap_or_else(default_model)
}

pub(super) fn forced_deepseek_model() -> String {
    model_names::find_by_key("DEEPSEEK_V3")
        .map(|m| m.name.as_str().to_owned())
        .unwrap_or_else(default_model)
}

pub(super) fn initial_model(cli: &ParsedCli) -> String {
    if let Some(ref model) = cli.model
        && !model.trim().is_empty()
    {
        return determine_model(model);
    }
    let cfg = crate::commonw::configw::get_all_config();
    cfg.get_opt("ai.model.default")
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(default_model)
}

pub(super) fn determine_model(model: &str) -> String {
    let model = model.trim().to_lowercase();
    if model.is_empty() {
        return default_model();
    }
    best_match_model_name(&model, all_model_names().into_iter(), default_model())
}

pub(super) fn determine_vl_model(model: &str) -> String {
    let model = model.trim().to_lowercase();
    if model.is_empty() {
        return default_vl_model();
    }

    if let Ok(idx) = model.parse::<usize>() {
        let all = model_names::all();
        let vl = all.iter().filter(|m| m.is_vl).nth(idx);
        if let Some(vl) = vl {
            return vl.name.as_str().to_owned();
        }
        return default_vl_model();
    }

    if let Some(def) = model_names::find_by_name(&model)
        && def.is_vl
    {
        return def.name.as_str().to_owned();
    }

    best_match_model_name(&model, vl_model_names().into_iter(), default_vl_model())
}

fn best_match_model_name(
    input_lowercase: &str,
    candidates: impl Iterator<Item = String>,
    default: String,
) -> String {
    let mut best = default;
    let mut best_dist = f32::MAX;
    for candidate in candidates {
        let candidate_lower = candidate.to_ascii_lowercase();
        let dist = levenshtein(input_lowercase.as_bytes(), candidate_lower.as_bytes()) as f32
            / (input_lowercase.len() + candidate_lower.len()) as f32;
        if dist < best_dist {
            best_dist = dist;
            best = candidate;
        }
    }
    best
}

fn levenshtein(left: &[u8], right: &[u8]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let mut prev: Vec<usize> = (0..=right.len()).collect();
    let mut curr = vec![0usize; right.len() + 1];
    for (i, left_byte) in left.iter().enumerate() {
        curr[0] = i + 1;
        for (j, right_byte) in right.iter().enumerate() {
            let cost = if left_byte == right_byte { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[right.len()]
}
