use super::cli::Cli;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuiltinModel {
    DeepseekV3,
    Glm,
    QwenPlusLatest,
    Qwen3Max,
    QwenCoderPlusLatest,
    Kimi,
    QwenFlash,
    QwenVlFlash,
    QwenVlMax,
    MiniMax,
}

impl BuiltinModel {
    const ALL: &'static [BuiltinModel] = &[
        BuiltinModel::DeepseekV3,
        BuiltinModel::Glm,
        BuiltinModel::QwenPlusLatest,
        BuiltinModel::Qwen3Max,
        BuiltinModel::QwenCoderPlusLatest,
        BuiltinModel::Kimi,
        BuiltinModel::QwenFlash,
        BuiltinModel::QwenVlFlash,
        BuiltinModel::QwenVlMax,
        BuiltinModel::MiniMax,
    ];

    fn name(self) -> &'static str {
        match self {
            BuiltinModel::DeepseekV3 => "deepseek-v3.2",
            BuiltinModel::Glm => "glm-5",
            BuiltinModel::QwenPlusLatest => "qwen3.5-plus",
            BuiltinModel::Qwen3Max => "qwen3-max",
            BuiltinModel::QwenCoderPlusLatest => "qwen3-coder-plus",
            BuiltinModel::Kimi => "kimi-k2.5",
            BuiltinModel::QwenFlash => "qwen3.5-flash",
            BuiltinModel::QwenVlFlash => "qwen3-vl-flash",
            BuiltinModel::QwenVlMax => "qwen3-vl-plus",
            BuiltinModel::MiniMax => "minimax-m2.5",
        }
    }

    fn is_vl(self) -> bool {
        matches!(
            self,
            BuiltinModel::QwenVlFlash | BuiltinModel::QwenVlMax | BuiltinModel::MiniMax
        )
    }

    fn search_enabled(self) -> bool {
        matches!(
            self,
            BuiltinModel::DeepseekV3
                | BuiltinModel::Glm
                | BuiltinModel::QwenPlusLatest
                | BuiltinModel::Qwen3Max
                | BuiltinModel::QwenFlash
        )
    }

    fn tools_default_enabled(self) -> bool {
        match self {
            BuiltinModel::QwenFlash | BuiltinModel::QwenVlFlash | BuiltinModel::MiniMax => false,
            BuiltinModel::DeepseekV3
            | BuiltinModel::Glm
            | BuiltinModel::QwenPlusLatest
            | BuiltinModel::Qwen3Max
            | BuiltinModel::QwenCoderPlusLatest
            | BuiltinModel::Kimi
            | BuiltinModel::QwenVlMax => true,
        }
    }
}

fn find_model(name: &str) -> Option<BuiltinModel> {
    let name = name.trim();
    BuiltinModel::ALL
        .iter()
        .copied()
        .find(|m| m.name().eq_ignore_ascii_case(name))
}

fn all_model_names() -> impl Iterator<Item = &'static str> {
    BuiltinModel::ALL.iter().map(|m| m.name())
}

fn vl_model_names() -> impl Iterator<Item = &'static str> {
    BuiltinModel::ALL
        .iter()
        .copied()
        .filter(|m| m.is_vl())
        .map(|m| m.name())
}

pub(super) fn qwen3_max() -> &'static str {
    BuiltinModel::Qwen3Max.name()
}

pub(super) fn deepseek_v3() -> &'static str {
    BuiltinModel::DeepseekV3.name()
}

pub(super) fn qwen_vl_flash() -> &'static str {
    BuiltinModel::QwenVlFlash.name()
}

pub(super) fn qwen_vl_max() -> &'static str {
    BuiltinModel::QwenVlMax.name()
}

pub(super) fn minimax_vl() -> &'static str {
    BuiltinModel::MiniMax.name()
}

pub(super) fn is_vl_model(model: &str) -> bool {
    find_model(model).is_some_and(|m| m.is_vl())
}

pub(super) fn search_enabled(model: &str) -> bool {
    find_model(model).is_some_and(|m| m.search_enabled())
}

pub(super) fn tools_enabled(model: &str) -> bool {
    find_model(model)
        .map(|m| m.tools_default_enabled())
        .unwrap_or(true)
}

pub(super) fn initial_model(cli: &Cli) -> String {
    if !cli.model.trim().is_empty() {
        return determine_model(&cli.model);
    }
    let cfg = crate::common::configw::get_all_config();
    cfg.get_opt("ai.model.default")
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| qwen3_max().to_string())
}

pub(super) fn determine_model(model: &str) -> String {
    let model = model.trim().to_lowercase();
    if model.is_empty() {
        return qwen3_max().to_string();
    }
    best_match_model_name(&model, all_model_names(), qwen3_max()).to_string()
}

pub(super) fn determine_vl_model(model: &str) -> String {
    let model = model.trim().to_lowercase();
    if model.is_empty() {
        return qwen_vl_flash().to_string();
    }

    match model.as_str() {
        "0" => return qwen_vl_flash().to_string(),
        "1" => return qwen_vl_max().to_string(),
        "2" => return minimax_vl().to_string(),
        _ => {}
    }

    if is_vl_model(&model) {
        return model;
    }

    best_match_model_name(&model, vl_model_names(), qwen_vl_flash()).to_string()
}

fn best_match_model_name(
    input_lowercase: &str,
    candidates: impl Iterator<Item = &'static str>,
    default: &'static str,
) -> &'static str {
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
            let cost = usize::from(left_byte != right_byte);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[right.len()]
}
