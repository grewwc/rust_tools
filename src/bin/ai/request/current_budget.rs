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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CurrentRequestBudget {
    pub(crate) prompt_tokens: usize,
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
        let output =
            preferred_output_reserve_tokens(models::max_output_tokens(model), output_override);
        let inline_cap_chars = crate::ai::driver::turn_runtime::max_tool_result_inline_chars(model);
        let prompt_tokens = current.context_prompt_tokens(previous);
        Self {
            prompt_tokens,
            source: if compatible {
                PromptCountSource::CompatibleActualPlusEstimatedGrowth
            } else {
                PromptCountSource::NormalizedEstimate
            },
            limits: ContextLimits::new(
                models::context_window_tokens(model),
                output,
                builder::CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS,
            ),
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
        let (estimated, counts) =
            builder::estimate_request_prompt_from_definitions(messages, None, None);
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
            estimated,
            counts,
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
}
