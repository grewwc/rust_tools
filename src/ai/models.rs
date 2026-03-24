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

const TOOLS_ON: bool = false;
const TOOLS_OFF: bool = false;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Model {
    DeepseekV3(bool),
    DeepseekR1(bool),
    QwenMaxLatest(bool),
    QwenPlusLatest(bool),
    QwenMax(bool),
    QwenCoderPlusLatest(bool),
    QwenLong(bool),
    Qwq(bool),
    QwenFlash(bool),
    Qwen3Max(bool),
    QwenVlFlash(bool),
    QwenVlMax(bool),
    QwenVlOcr(bool),
}

impl Model {
    pub(super) const ALL: &'static [Model] = &[
        Model::DeepseekV3(TOOLS_ON),
        Model::DeepseekR1(TOOLS_ON),
        Model::QwenMaxLatest(TOOLS_ON),
        Model::QwenPlusLatest(TOOLS_ON),
        Model::QwenMax(TOOLS_ON),
        Model::QwenCoderPlusLatest(TOOLS_ON),
        Model::QwenLong(TOOLS_ON),
        Model::Qwq(TOOLS_ON),
        Model::QwenFlash(TOOLS_OFF),
        Model::Qwen3Max(TOOLS_ON),
        Model::QwenVlFlash(TOOLS_OFF),
        Model::QwenVlMax(TOOLS_ON),
        Model::QwenVlOcr(TOOLS_OFF),
    ];

    pub(super) const VL: &'static [Model] = &[
        Model::QwenVlFlash(TOOLS_ON),
        Model::QwenVlMax(TOOLS_ON),
        Model::QwenVlOcr(TOOLS_ON),
    ];

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Model::DeepseekV3(_) => DEEPSEEK_V3,
            Model::DeepseekR1(_) => DEEPSEEK_R1,
            Model::QwenMaxLatest(_) => QWEN_MAX_LATEST,
            Model::QwenPlusLatest(_) => QWEN_PLUS_LATEST,
            Model::QwenMax(_) => QWEN_MAX,
            Model::QwenCoderPlusLatest(_) => QWEN_CODER_PLUS_LATEST,
            Model::QwenLong(_) => QWEN_LONG,
            Model::Qwq(_) => QWQ,
            Model::QwenFlash(_) => QWEN_FLASH,
            Model::Qwen3Max(_) => QWEN3_MAX,
            Model::QwenVlFlash(_) => QWEN_VL_FLASH,
            Model::QwenVlMax(_) => QWEN_VL_MAX,
            Model::QwenVlOcr(_) => QWEN_VL_OCR,
        }
    }

    pub(super) fn is_vl(self) -> bool {
        matches!(
            self,
            Model::QwenVlFlash(_) | Model::QwenVlMax(_) | Model::QwenVlOcr(_)
        )
    }

    pub(super) fn search_enabled(self) -> bool {
        matches!(
            self,
            Model::QwenMax(_)
                | Model::QwenMaxLatest(_)
                | Model::QwenPlusLatest(_)
                | Model::QwenFlash(_)
                | Model::DeepseekV3(_)
                | Model::Qwen3Max(_)
        )
    }

    pub(super) fn tools_enabled(self) -> bool {
        match self {
            Model::DeepseekV3(v)
            | Model::DeepseekR1(v)
            | Model::QwenMaxLatest(v)
            | Model::QwenPlusLatest(v)
            | Model::QwenMax(v)
            | Model::QwenCoderPlusLatest(v)
            | Model::QwenLong(v)
            | Model::Qwq(v)
            | Model::QwenFlash(v)
            | Model::Qwen3Max(v)
            | Model::QwenVlFlash(v)
            | Model::QwenVlMax(v)
            | Model::QwenVlOcr(v) => v,
        }
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

pub(super) fn tools_enabled(model: &str) -> bool {
    Model::ALL
        .iter()
        .find(|m| m.as_str() == model)
        .is_none_or(|m| m.tools_enabled())
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
        0 => Model::Qwq(TOOLS_ON),
        1 => Model::QwenPlusLatest(TOOLS_ON),
        2 => Model::QwenMax(TOOLS_ON),
        3 => Model::Qwen3Max(TOOLS_ON),
        4 => Model::QwenCoderPlusLatest(TOOLS_ON),
        5 => {
            if thinking_mode {
                Model::DeepseekR1(TOOLS_ON)
            } else {
                Model::DeepseekV3(TOOLS_ON)
            }
        }
        6 => Model::QwenFlash(TOOLS_OFF),
        _ => Model::Qwen3Max(TOOLS_ON),
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
