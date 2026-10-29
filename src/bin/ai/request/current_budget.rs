//! Context occupancy, never cache-discounted rate usage. All limits use tokens.

use serde::{Deserialize, Serialize};

use super::{PromptTokenFeedback, builder};
use crate::ai::models;

/// Completion and reasoning headroom that compression tries to preserve.
/// Provider output limits remain wire ceilings; reserving an entire 128K+
/// ceiling here would force history compression around half of the window.
const PREFERRED_OUTPUT_RESERVE_TOKENS: u32 = 32_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptCountSource {
    NormalizedEstimate,
    CompatibleActualPlusEstimatedGrowth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextLimits {
    pub(crate) physical_context_tokens: usize,
    /// Preferred headroom for compression decisions, not the wire output cap.
    pub(crate) output_reserve_tokens: usize,
    pub(crate) safety_margin_tokens: usize,
    pub(crate) input_allowance_tokens: usize,
    pub(crate) soft_target_tokens: usize,
}

impl ContextLimits {
    fn new(physical: usize, output: usize, safety: usize) -> Self {
        let input = physical.saturating_sub(output).saturating_sub(safety);
        Self {
            physical_context_tokens: physical,
            output_reserve_tokens: output,
            safety_margin_tokens: safety,
            input_allowance_tokens: input,
            // 70% of the input allowance: lossy mechanical compression starts
            // earlier than the previous 80%, while the LLM summary gate keys
            // off the input allowance itself (see apply_request_budget), so
            // this ratio only tunes the cheaper mechanical passes.
            soft_target_tokens: input.saturating_mul(7).saturating_div(10),
        }
    }

    /// Model-derived limits shared by budget measurement and the water
    /// meter's display band, so the two can never disagree on the allowance.
    pub(crate) fn for_model(model: &str, output_override: Option<u32>) -> Self {
        Self::new(
            models::context_window_tokens(model),
            preferred_output_reserve_tokens(models::max_output_tokens(model), output_override),
            builder::CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS,
        )
    }
}

/// Display-only context occupancy band for the water meter note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WaterBand {
    Comfortable,
    Elevated,
    Pressure,
    Critical,
}

impl WaterBand {
    /// Derive guidance from the same calibrated upper interval and allowance
    /// that compression uses, so model-visible pressure cannot contradict the
    /// runtime's current request budget.
    pub(crate) fn for_budget(budget: CurrentRequestBudget) -> Self {
        Self::for_upper(budget.upper_tokens(), budget.limits.input_allowance_tokens)
    }

    /// Fixed band edges as permille of the input allowance: 600/800/1000.
    /// Pure function of a request upper bound: prompt growth is append-only
    /// within a stable calibration chain, so the band is monotonic and no
    /// hysteresis state is needed (edges sit 20% of the allowance apart
    /// while the note itself is ~50 tokens).
    pub(crate) fn for_upper(upper_tokens: usize, allowance_tokens: usize) -> Self {
        if allowance_tokens == 0 {
            return Self::Critical;
        }
        let permille =
            (upper_tokens as u128).saturating_mul(1000) / (allowance_tokens as u128);
        if permille < 600 {
            Self::Comfortable
        } else if permille < 800 {
            Self::Elevated
        } else if permille < 1000 {
            Self::Pressure
        } else {
            Self::Critical
        }
    }
}

/// Fixed water meter note per band. Byte-stable within a band by
/// construction (only fixed edge labels, no measured numbers): the
/// calibration prefix fingerprint covers all messages, so a
/// per-round-changing note would break usage calibration on every send. A
/// band transition costs exactly one prefix break, then the chain
/// re-stabilizes; transitions are rare (see [`WaterBand::for_upper`]).
pub(crate) fn water_note(band: WaterBand) -> &'static str {
    match band {
        WaterBand::Comfortable => "[context-water-meter: comfortable] Conservative context occupancy is below 60% of the input allowance. No action needed.",
        WaterBand::Elevated => "[context-water-meter: elevated] Conservative context occupancy is at 60-80% of the input allowance. Prefer narrow reads and smaller result sets.",
        WaterBand::Pressure => "[context-water-meter: pressure] Conservative context occupancy is at 80-100% of the input allowance. Checkpoint key findings now, avoid spawning subagents, and narrow further reads.",
        WaterBand::Critical => "[context-water-meter: critical] Conservative context occupancy meets or exceeds the input allowance; compression is imminent or active. Checkpoint key findings immediately and converge the answer.",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CurrentRequestBudget {
    pub(crate) prompt_tokens: usize,
    /// Pessimistic twin of `prompt_tokens` for the water meter's display
    /// interval. Budgets, clamps, and compression keep reading `prompt_tokens`
    /// only. Legacy snapshots deserialize this as 0; use
    /// [`upper_tokens`](Self::upper_tokens) so the interval can
    /// never invert.
    #[serde(default)]
    pub(crate) prompt_tokens_upper: usize,
    pub(crate) source: PromptCountSource,
    pub(crate) limits: ContextLimits,
    /// Inline cap for a single tool result under this budget's model. Kept
    /// inside the budget (rather than re-derived from `app.current_model` at
    /// compression time) so that model-fallback paths offload results with the
    /// *fallback* model's window, not the original one.
    #[serde(default = "default_tool_inline_cap_chars")]
    pub(crate) inline_cap_chars: usize,
    /// Message chars per prompt token (thousandths) observed for this request,
    /// see [`PromptTokenFeedback::chars_per_token_milli`]. Converts token
    /// deficits into char targets instead of a flat 2 chars/token, which
    /// over-compressed CJK histories and under-compressed ASCII ones.
    #[serde(default = "default_chars_per_token_milli")]
    pub(crate) chars_per_token_milli: usize,
}

fn default_chars_per_token_milli() -> usize {
    super::prompt_feedback::DEFAULT_CHARS_PER_TOKEN_MILLI
}

/// Conservative default for externally deserialized budgets (tests, future
/// persisted snapshots) that do not carry a model-derived inline cap: the
/// smallest model line (128K window -> 32K chars). Zero would be wrong — it
/// would classify every tool result as oversized and spill all of them.
fn default_tool_inline_cap_chars() -> usize {
    32_000
}

impl CurrentRequestBudget {
    pub(super) fn measure(
        model: &str,
        output_override: Option<u32>,
        current: &PromptTokenFeedback,
        previous: Option<&PromptTokenFeedback>,
    ) -> Self {
        let compatible = previous.is_some_and(|prior| current.compatible_usage(prior).is_some());
        let inline_cap_chars = crate::ai::driver::turn_runtime::max_tool_result_inline_chars(model);
        let prompt_tokens = current.context_prompt_tokens(previous);
        let prompt_tokens_upper = current.context_prompt_tokens_upper(previous);
        Self {
            prompt_tokens,
            prompt_tokens_upper,
            source: if compatible {
                PromptCountSource::CompatibleActualPlusEstimatedGrowth
            } else {
                PromptCountSource::NormalizedEstimate
            },
            limits: ContextLimits::for_model(model, output_override),
            inline_cap_chars,
            chars_per_token_milli: current.chars_per_token_milli(prompt_tokens),
        }
    }

    pub(crate) fn exceeds_soft_target(self) -> bool {
        self.prompt_tokens > self.limits.soft_target_tokens
    }

    pub(crate) fn exceeds_input_allowance(self) -> bool {
        self.prompt_tokens > self.limits.input_allowance_tokens
    }

    /// Display-only upper bound of the prompt-token interval. Clamped to
    /// `prompt_tokens` so legacy snapshots (whose field deserializes as 0)
    /// degrade to a degenerate interval instead of an inverted one.
    pub(crate) fn upper_tokens(self) -> usize {
        self.prompt_tokens_upper.max(self.prompt_tokens)
    }

    /// Message chars that `tokens` prompt tokens correspond to at this
    /// request's observed ratio, rounded up so a deficit is never understated.
    pub(crate) fn tokens_to_chars(self, tokens: usize) -> usize {
        let chars = (tokens as u128)
            .saturating_mul(self.chars_per_token_milli as u128)
            .div_ceil(1_000);
        usize::try_from(chars).unwrap_or(usize::MAX)
    }

    /// Translate a token deficit to a mechanical character target, not a claim
    /// that the replacement will fit. Fixed tool/system overhead may make this
    /// target unattainable; remeasure the normalized request after replacement.
    pub(crate) fn compression_target_chars(self, current_chars: usize) -> usize {
        current_chars.saturating_sub(
            self.tokens_to_chars(
                self.prompt_tokens
                    .saturating_sub(self.limits.soft_target_tokens),
            ),
        )
    }
}

/// Minimum output-token reserve so a model with a tiny declared `max_output`
/// (or none) still leaves room for a complete reply.
fn preferred_output_reserve_tokens(model_max: Option<u32>, output_override: Option<u32>) -> usize {
    // Keep enough headroom for a useful completion while allowing the request
    // builder to clamp the provider's larger wire ceiling against the actual
    // remaining window. Unknown caps retain a minimum reserve without adding
    // max_tokens to the wire body.
    model_max
        .map(|cap| cap.max(builder::MIN_OUTPUT_TOKENS_FLOOR))
        .map(|cap| cap.min(PREFERRED_OUTPUT_RESERVE_TOKENS))
        .map(|reserve| output_override.map_or(reserve, |value| value.min(reserve)))
        .unwrap_or(builder::MIN_OUTPUT_TOKENS_FLOOR) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::history::Message;

    #[test]
    fn preferred_output_reserve_is_independent_from_large_wire_caps() {
        let reserve = preferred_output_reserve_tokens(Some(128_000), None);
        assert_eq!(reserve, 32_000);
        let limits = ContextLimits::new(256_000, reserve, 2_048);
        assert_eq!(limits.input_allowance_tokens, 221_952);
        assert_eq!(limits.soft_target_tokens, 155_366);

        assert_eq!(preferred_output_reserve_tokens(Some(16_000), None), 16_000);
        assert_eq!(
            preferred_output_reserve_tokens(Some(128_000), Some(8_000)),
            8_000
        );
        assert_eq!(
            preferred_output_reserve_tokens(Some(128_000), Some(64_000)),
            32_000
        );
        assert_eq!(preferred_output_reserve_tokens(None, Some(512)), 1_024);
    }

    #[test]
    fn measured_budget_uses_preferred_reserve_for_registry_models() {
        let model = crate::ai::model_names::all()
            .into_iter()
            .find(|entry| {
                models::max_output_tokens(&entry.key)
                    .is_some_and(|cap| cap > PREFERRED_OUTPUT_RESERVE_TOKENS)
            })
            .expect("registry must contain an output cap above the preferred reserve");
        let feedback = PromptTokenFeedback::capture_preview(
            "session",
            &model.key,
            "",
            &[],
            &serde_json::Map::new(),
            None,
            None,
            None,
            None,
            None,
            false,
            false,
            1_234,
            1_234,
            Default::default(),
        );

        let budget = CurrentRequestBudget::measure(&model.key, None, &feedback, None);

        assert_eq!(budget.prompt_tokens, 1_234);
        assert_eq!(budget.chars_per_token_milli, 2_000);
        assert_eq!(budget.limits.output_reserve_tokens, 32_000);
        assert_eq!(
            budget.limits.input_allowance_tokens,
            models::context_window_tokens(&model.key)
                .saturating_sub(32_000)
                .saturating_sub(2_048)
        );
    }

    #[test]
    fn current_budget_reserve_and_tiny_window_boundaries() {
        let limits = ContextLimits::new(10_000, 2_000, 1_000);
        assert_eq!(limits.input_allowance_tokens, 7_000);
        assert_eq!(limits.soft_target_tokens, 4_900);
        for physical in [0, 1, 1_024, 2_048] {
            let limits = ContextLimits::new(physical, 1_024, 2_048);
            assert_eq!(limits.input_allowance_tokens, 0);
            assert_eq!(limits.soft_target_tokens, 0);
        }
        let budget = CurrentRequestBudget {
            prompt_tokens: usize::MAX,
            prompt_tokens_upper: usize::MAX,
            source: PromptCountSource::NormalizedEstimate,
            limits,
            inline_cap_chars: 32_000,
            chars_per_token_milli: 2_000,
        };
        assert!(budget.exceeds_soft_target());
        assert!(budget.exceeds_input_allowance());
        assert_eq!(budget.compression_target_chars(usize::MAX), 0);
    }

    fn preview_budget(model: &str, messages: &[Message]) -> CurrentRequestBudget {
        let prompt = builder::measure_request_prompt_from_definitions(messages, None, None);
        let feedback = PromptTokenFeedback::capture_preview(
            "session",
            model,
            "",
            messages,
            &serde_json::Map::new(),
            None,
            None,
            None,
            None,
            None,
            false,
            false,
            prompt.lower_tokens(),
            prompt.upper_tokens(),
            prompt.message_counts(),
        );
        CurrentRequestBudget::measure(model, None, &feedback, None)
    }

    #[test]
    fn current_budget_chars_per_token_follows_script_mix() {
        let model = crate::ai::model_names::all()[0].key.clone();
        let text = |body: String| Message {
            role: "user".to_owned(),
            content: serde_json::Value::String(body),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        let ascii = preview_budget(&model, &[text("plain ascii prose ".repeat(1_000))]);
        let cjk = preview_budget(&model, &[text("中文上下文压缩".repeat(1_000))]);
        assert_eq!(ascii.chars_per_token_milli, 3_000);
        assert_eq!(cjk.chars_per_token_milli, 1_000);
        // A 1K-token deficit frees 3K ASCII chars but only 1K CJK chars; the
        // flat ratio under-compressed the former and over-compressed the latter.
        assert_eq!(ascii.tokens_to_chars(1_000), 3_000);
        assert_eq!(cjk.tokens_to_chars(1_000), 1_000);
    }

    #[test]
    fn budget_upper_bound_is_valid_and_script_aware() {
        let model = crate::ai::model_names::all()[0].key.clone();
        let text = |body: String| Message {
            role: "user".to_owned(),
            content: serde_json::Value::String(body),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        let ascii = preview_budget(&model, &[text("plain ascii prose ".repeat(1_000))]);
        let cjk = preview_budget(&model, &[text("中文上下文压缩".repeat(1_000))]);
        // The interval never inverts, with or without calibration history.
        assert!(ascii.upper_tokens() >= ascii.prompt_tokens);
        assert!(cjk.upper_tokens() >= cjk.prompt_tokens);
        // ASCII carries ratio slack (3 vs 2 chars/token); CJK is already 1:1
        // in both projections, so its interval is degenerate.
        assert!(ascii.upper_tokens() > ascii.prompt_tokens);
        assert_eq!(cjk.upper_tokens(), cjk.prompt_tokens);
    }

    #[test]
    fn water_band_edges_follow_fixed_percille() {
        use super::{WaterBand, water_note};
        let band = |upper: usize| WaterBand::for_upper(upper, 10_000);
        assert_eq!(band(0), WaterBand::Comfortable);
        assert_eq!(band(5_999), WaterBand::Comfortable);
        assert_eq!(band(6_000), WaterBand::Elevated);
        assert_eq!(band(7_999), WaterBand::Elevated);
        assert_eq!(band(8_000), WaterBand::Pressure);
        assert_eq!(band(9_999), WaterBand::Pressure);
        assert_eq!(band(10_000), WaterBand::Critical);
        assert_eq!(band(50_000), WaterBand::Critical);
        // Degenerate allowance degrades to critical, never panics.
        assert_eq!(WaterBand::for_upper(0, 0), WaterBand::Critical);
        // Notes are fixed per band: byte-identical across calls and ASCII-only
        // so text-only normalization is a no-op for them.
        for b in [
            WaterBand::Comfortable,
            WaterBand::Elevated,
            WaterBand::Pressure,
            WaterBand::Critical,
        ] {
            assert_eq!(water_note(b), water_note(b));
            assert!(water_note(b).is_ascii());
        }
        assert_ne!(
            water_note(WaterBand::Comfortable),
            water_note(WaterBand::Critical)
        );
    }
}
