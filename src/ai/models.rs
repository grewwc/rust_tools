use super::cli::Cli;

const DEEPSEEK_V3: &str = "deepseek-v3.1";
const DEEPSEEK_R1: &str = "deepseek-r1";
const QWEN_MAX_LATEST: &str = "qwen-max-latest";
const QWEN_PLUS_LATEST: &str = "qwen3.5-plus";
const QWEN_MAX: &str = "qwen-max";
const QWEN_CODER_PLUS_LATEST: &str = "qwen3-coder-plus";
const QWEN_LONG: &str = "qwen-long";
const QWQ: &str = "qwq-plus-latest";
const QWEN_FLASH: &str = "qwen-flash";
const QWEN3_MAX: &str = "qwen3-max";
const QWEN_VL_FLASH: &str = "qwen3-vl-flash";
const QWEN_VL_MAX: &str = "qwen3-vl-plus";
const QWEN_VL_OCR: &str = "qwen-vl-ocr-latest";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Model {
    DeepseekV3,
    DeepseekR1,
    QwenMaxLatest,
    QwenPlusLatest,
    QwenMax,
    QwenCoderPlusLatest,
    QwenLong,
    Qwq,
    QwenFlash,
    Qwen3Max,
    QwenVlFlash,
    QwenVlMax,
    QwenVlOcr,
}

impl Model {
    pub(super) const ALL: &'static [Model] = &[
        Model::DeepseekV3,
        Model::DeepseekR1,
        Model::QwenMaxLatest,
        Model::QwenPlusLatest,
        Model::QwenMax,
        Model::QwenCoderPlusLatest,
        Model::QwenLong,
        Model::Qwq,
        Model::QwenFlash,
        Model::Qwen3Max,
        Model::QwenVlFlash,
        Model::QwenVlMax,
        Model::QwenVlOcr,
    ];

    pub(super) const VL: &'static [Model] =
        &[Model::QwenVlFlash, Model::QwenVlMax, Model::QwenVlOcr];

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Model::DeepseekV3 => DEEPSEEK_V3,
            Model::DeepseekR1 => DEEPSEEK_R1,
            Model::QwenMaxLatest => QWEN_MAX_LATEST,
            Model::QwenPlusLatest => QWEN_PLUS_LATEST,
            Model::QwenMax => QWEN_MAX,
            Model::QwenCoderPlusLatest => QWEN_CODER_PLUS_LATEST,
            Model::QwenLong => QWEN_LONG,
            Model::Qwq => QWQ,
            Model::QwenFlash => QWEN_FLASH,
            Model::Qwen3Max => QWEN3_MAX,
            Model::QwenVlFlash => QWEN_VL_FLASH,
            Model::QwenVlMax => QWEN_VL_MAX,
            Model::QwenVlOcr => QWEN_VL_OCR,
        }
    }

    pub(super) fn is_vl(self) -> bool {
        matches!(
            self,
            Model::QwenVlFlash | Model::QwenVlMax | Model::QwenVlOcr
        )
    }

    pub(super) fn search_enabled(self) -> bool {
        matches!(
            self,
            Model::QwenMax
                | Model::QwenMaxLatest
                | Model::QwenPlusLatest
                | Model::QwenFlash
                | Model::DeepseekV3
                | Model::Qwen3Max
        )
    }
}

pub(super) fn qwen_long() -> &'static str {
    QWEN_LONG
}

pub(super) fn qwen3_max() -> &'static str {
    QWEN3_MAX
}

pub(super) fn qwen_coder_plus_latest() -> &'static str {
    QWEN_CODER_PLUS_LATEST
}

pub(super) fn deepseek_v3() -> &'static str {
    DEEPSEEK_V3
}

pub(super) fn deepseek_r1() -> &'static str {
    DEEPSEEK_R1
}

pub(super) fn qwen_vl_flash() -> &'static str {
    QWEN_VL_FLASH
}

pub(super) fn qwen_vl_max() -> &'static str {
    QWEN_VL_MAX
}

pub(super) fn qwen_vl_ocr() -> &'static str {
    QWEN_VL_OCR
}

pub(super) fn is_vl_model(model: &str) -> bool {
    Model::ALL
        .iter()
        .find(|m| m.as_str() == model)
        .is_some_and(|m| m.is_vl())
}

pub(super) fn search_enabled(model: &str) -> bool {
    Model::ALL
        .iter()
        .find(|m| m.as_str() == model)
        .is_some_and(|m| m.search_enabled())
}

pub(super) fn initial_model(cli: &Cli) -> String {
    if cli.code {
        return qwen_coder_plus_latest().to_string();
    }
    if cli.deepseek {
        return if cli.thinking {
            deepseek_r1().to_string()
        } else {
            deepseek_v3().to_string()
        };
    }
    if let Some(selector) = selected_model_number(cli) {
        return model_from_selector(selector, cli.thinking)
            .as_str()
            .to_string();
    }
    if !cli.model.trim().is_empty() {
        return determine_model(&cli.model);
    }
    let cfg = crate::common::configw::get_all_config();
    cfg.get_opt("ai.model.default")
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| qwen3_max().to_string())
}

pub(super) fn selected_model_number(cli: &Cli) -> Option<u8> {
    [
        cli.model_0,
        cli.model_1,
        cli.model_2,
        cli.model_3,
        cli.model_4,
        cli.model_5,
        cli.model_6,
    ]
    .into_iter()
    .enumerate()
    .find_map(|(idx, enabled)| enabled.then_some(idx as u8))
}

pub(super) fn model_from_selector(selector: u8, thinking_mode: bool) -> Model {
    match selector {
        0 => Model::Qwq,
        1 => Model::QwenPlusLatest,
        2 => Model::QwenMax,
        3 => Model::Qwen3Max,
        4 => Model::QwenCoderPlusLatest,
        5 => {
            if thinking_mode {
                Model::DeepseekR1
            } else {
                Model::DeepseekV3
            }
        }
        6 => Model::QwenFlash,
        _ => Model::Qwen3Max,
    }
}

pub(super) fn determine_model(model: &str) -> String {
    let model = model.trim().to_lowercase();
    if model.is_empty() {
        return qwen3_max().to_string();
    }
    let mut best = qwen3_max();
    let mut best_dist = f32::MAX;
    for candidate in Model::ALL {
        let candidate = candidate.as_str();
        let dist = levenshtein(model.as_bytes(), candidate.as_bytes()) as f32
            / (model.len() + candidate.len()) as f32;
        if dist < best_dist {
            best_dist = dist;
            best = candidate;
        }
    }
    best.to_string()
}

pub(super) fn determine_vl_model(model: &str) -> String {
    let model = model.trim().to_lowercase();
    if model.is_empty() {
        return qwen_vl_flash().to_string();
    }

    match model.as_str() {
        "0" => return qwen_vl_flash().to_string(),
        "1" => return qwen_vl_max().to_string(),
        "2" => return qwen_vl_ocr().to_string(),
        _ => {}
    }

    if is_vl_model(&model) {
        return model;
    }

    let mut best = qwen_vl_flash();
    let mut best_dist = f32::MAX;
    for candidate in Model::VL {
        let candidate = candidate.as_str();
        let dist = levenshtein(model.as_bytes(), candidate.as_bytes()) as f32
            / (model.len() + candidate.len()) as f32;
        if dist < best_dist {
            best_dist = dist;
            best = candidate;
        }
    }
    best.to_string()
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
