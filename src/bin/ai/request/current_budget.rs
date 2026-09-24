//! Context occupancy, never cache-discounted rate usage. All limits use tokens.

use serde::{Deserialize, Serialize};

use super::{PromptTokenFeedback, builder};
use crate::ai::models;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptCountSource {
    NormalizedEstimate,
    CompatibleActualPlusEstimatedGrowth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextLimits {
    pub(crate) physical_context_tokens: usize,
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
            soft_target_tokens: input.saturating_sub(input / 5),
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
        // Reserve output independently of the prompt clamp; the clamp's floor
        // cannot establish that input fits. Unknown model output caps retain a
        // minimum reserve without adding a max_tokens field to the wire body.
        let output = models::max_output_tokens(model)
            .map(|cap| cap.max(builder::MIN_OUTPUT_TOKENS_FLOOR))
            .map(|cap| output_override.map_or(cap, |value| value.min(cap)))
            .unwrap_or(builder::MIN_OUTPUT_TOKENS_FLOOR) as usize;
        let inline_cap_chars =
            crate::ai::driver::turn_runtime::max_tool_result_inline_chars(model);
        Self {
            prompt_tokens: current.context_prompt_tokens(previous),
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
        }
    }

    pub(crate) fn exceeds_soft_target(self) -> bool {
        self.prompt_tokens > self.limits.soft_target_tokens
    }

    pub(crate) fn exceeds_input_allowance(self) -> bool {
        self.prompt_tokens > self.limits.input_allowance_tokens
    }

    /// Translate a token deficit to a mechanical character target, not a claim
    /// that the replacement will fit. Fixed tool/system overhead may make this
    /// target unattainable; remeasure the normalized request after replacement.
    pub(crate) fn compression_target_chars(self, current_chars: usize) -> usize {
        current_chars.saturating_sub(
            self.prompt_tokens
                .saturating_sub(self.limits.soft_target_tokens)
                .saturating_mul(2),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_budget_reserve_and_tiny_window_boundaries() {
        let limits = ContextLimits::new(10_000, 2_000, 1_000);
        assert_eq!(limits.input_allowance_tokens, 7_000);
        assert_eq!(limits.soft_target_tokens, 5_600);
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
        };
        assert!(budget.exceeds_soft_target());
        assert!(budget.exceeds_input_allowance());
        assert_eq!(budget.compression_target_chars(usize::MAX), 0);
    }
}