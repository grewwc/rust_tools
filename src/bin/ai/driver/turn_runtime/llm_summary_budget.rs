//! A last-resort summary allowance owned by one turn, including its retries.

use crate::ai::{
    driver::decision_log::{DecisionLog, DecisionType, get_decision_log_store},
    request::CurrentRequestBudget,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct LlmSummaryPolicy {
    max_model_calls: usize,
    max_attempts: usize,
    min_changed_chars: usize,
    min_round_gap: usize,
}

impl Default for LlmSummaryPolicy {
    fn default() -> Self {
        Self {
            max_model_calls: 1,
            max_attempts: 2,
            min_changed_chars: 20_000,
            min_round_gap: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptOutcome {
    InFlight,
    StructuralNoOp,
    ModelCalled,
}

#[derive(Debug, Clone, Copy)]
struct LastAttempt {
    round: usize,
    chars_after: usize,
    outcome: AttemptOutcome,
}

#[derive(Debug, Default)]
pub(super) struct LlmSummaryBudget {
    policy: LlmSummaryPolicy,
    attempts: usize,
    model_calls: usize,
    last_attempt: Option<LastAttempt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LlmSummaryReason {
    BelowAllowance,
    AttemptLimit,
    GrowthGate,
    RoundGate,
    Allowed,
    StructuralNoOp,
    ModelCalled,
}

impl LlmSummaryReason {
    fn label(self) -> &'static str {
        match self {
            Self::BelowAllowance => "below-allowance",
            Self::AttemptLimit => "attempt-limit",
            Self::GrowthGate => "growth-gate",
            Self::RoundGate => "round-gate",
            Self::Allowed => "allowed",
            Self::StructuralNoOp => "structural-no-op",
            Self::ModelCalled => "model-called",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LlmSummaryDecision {
    pub(super) reason: LlmSummaryReason,
    pub(super) attempt_index: usize,
}

pub(super) struct LlmSummaryGate<'a> {
    pub(super) session_id: &'a str,
    pub(super) model: &'a str,
    pub(super) trigger: &'static str,
    pub(super) round: usize,
    pub(super) request_budget: CurrentRequestBudget,
    pub(super) chars_before: usize,
}

pub(super) struct LlmSummaryAttempt<'a> {
    gate: LlmSummaryGate<'a>,
    index: usize,
}

impl LlmSummaryBudget {
    pub(super) fn decide(
        &self,
        exceeds_input_allowance: bool,
        chars: usize,
        round: usize,
    ) -> LlmSummaryDecision {
        let reason = if !exceeds_input_allowance {
            LlmSummaryReason::BelowAllowance
        } else if self.model_calls >= self.policy.max_model_calls
            || self.attempts >= self.policy.max_attempts
            || self
                .last_attempt
                .is_some_and(|last| last.outcome != AttemptOutcome::StructuralNoOp)
        {
            LlmSummaryReason::AttemptLimit
        } else if let Some(last) = self.last_attempt {
            if round.saturating_sub(last.round) < self.policy.min_round_gap {
                LlmSummaryReason::RoundGate
            } else if chars.abs_diff(last.chars_after) < self.policy.min_changed_chars {
                LlmSummaryReason::GrowthGate
            } else {
                LlmSummaryReason::Allowed
            }
        } else {
            LlmSummaryReason::Allowed
        };
        LlmSummaryDecision {
            reason,
            attempt_index: self.attempts.saturating_add(1),
        }
    }

    /// Reserve before awaiting the summary. Cancellation leaves the reservation
    /// spent: a wrap-up or retry cannot repeat a model call whose result was lost.
    pub(super) fn begin_attempt<'a>(
        &mut self,
        gate: LlmSummaryGate<'a>,
    ) -> Option<LlmSummaryAttempt<'a>> {
        let decision = self.decide(
            gate.request_budget.exceeds_input_allowance(),
            gate.chars_before,
            gate.round,
        );
        if decision.reason == LlmSummaryReason::Allowed {
            self.attempts += 1;
            self.last_attempt = Some(LastAttempt {
                round: gate.round,
                chars_after: gate.chars_before,
                outcome: AttemptOutcome::InFlight,
            });
        }
        self.record_decision(&gate, decision, gate.chars_before, None);
        (decision.reason == LlmSummaryReason::Allowed).then_some(LlmSummaryAttempt {
            gate,
            index: decision.attempt_index,
        })
    }

    /// A completed structural no-op frees the model allowance, but not its gate
    /// attempt. Failed, rejected and ineffective model calls all spend it.
    pub(super) fn finish_attempt(
        &mut self,
        attempt: LlmSummaryAttempt<'_>,
        chars_after: usize,
        model_called: bool,
    ) {
        debug_assert_eq!(attempt.index, self.attempts);
        self.model_calls += usize::from(model_called);
        self.last_attempt = Some(LastAttempt {
            round: attempt.gate.round,
            chars_after,
            outcome: if model_called {
                AttemptOutcome::ModelCalled
            } else {
                AttemptOutcome::StructuralNoOp
            },
        });
        self.record_decision(
            &attempt.gate,
            LlmSummaryDecision {
                reason: if model_called {
                    LlmSummaryReason::ModelCalled
                } else {
                    LlmSummaryReason::StructuralNoOp
                },
                attempt_index: attempt.index,
            },
            chars_after,
            Some(model_called),
        );
    }

    fn record_decision(
        &self,
        gate: &LlmSummaryGate<'_>,
        decision: LlmSummaryDecision,
        chars_after: usize,
        model_called: Option<bool>,
    ) {
        let allowed = matches!(
            decision.reason,
            LlmSummaryReason::Allowed
                | LlmSummaryReason::StructuralNoOp
                | LlmSummaryReason::ModelCalled
        );
        get_decision_log_store().log(DecisionLog {
            timestamp: 0,
            session_id: gate.session_id.to_string(),
            turn_id: gate.round,
            decision_type: DecisionType::ContextProjection,
            context: serde_json::json!({
                "schema": "llm-summary-budget-v1",
                "model": gate.model,
                "task_pid": crate::ai::driver::current_task_pid(),
                "trigger": gate.trigger,
                "round": gate.round,
                "allowed": allowed,
                "reason": decision.reason.label(),
                "attempt_index": decision.attempt_index,
                "attempts": self.attempts,
                "model_calls": self.model_calls,
                "max_attempts": self.policy.max_attempts,
                "max_model_calls": self.policy.max_model_calls,
                "chars_before": gate.chars_before,
                "chars_after": chars_after,
                "model_called": model_called,
                "request_budget": gate.request_budget,
            })
            .to_string(),
            alternatives_considered: Vec::new(),
            chosen_option: decision.reason.label().to_string(),
            reasoning: "Measured input allowance and a shared turn-scoped summary budget; no prompt content."
                .to_string(),
            confidence: None,
            outcome: None,
            execution_time_ms: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(chars: usize, round: usize) -> LlmSummaryGate<'static> {
        LlmSummaryGate {
            session_id: "llm-summary-policy-test",
            model: "test-model",
            trigger: "test",
            round,
            chars_before: chars,
            request_budget: serde_json::from_value(serde_json::json!({
                "prompt_tokens": 11_000, "source": "normalized_estimate",
                "limits": { "physical_context_tokens": 12_000, "output_reserve_tokens": 1_000,
                    "safety_margin_tokens": 1_000, "input_allowance_tokens": 10_000, "soft_target_tokens": 8_000 }
            })).unwrap(),
        }
    }

    #[test]
    fn llm_summary_policy_below_allowance_preserves_first_attempt() {
        let mut budget = LlmSummaryBudget::default();
        assert_eq!(
            budget.decide(false, 500_000, 1).reason,
            LlmSummaryReason::BelowAllowance
        );
        assert_eq!(budget.attempts, 0);
        // Characters never replace measured token pressure, even on the first attempt.
        let first = budget.begin_attempt(gate(1, 1)).unwrap();
        assert_eq!(first.index, 1);
    }

    #[test]
    fn llm_summary_policy_model_call_cannot_reopen_after_growth_or_shrink() {
        for after in [100_000, 299_999, 300_000, 350_000] {
            let mut budget = LlmSummaryBudget::default();
            let first = budget.begin_attempt(gate(300_000, 1)).unwrap();
            budget.finish_attempt(first, after, true);
            assert_eq!(budget.model_calls, 1);
            assert_eq!(
                budget.decide(true, after + 40_000, 2).reason,
                LlmSummaryReason::AttemptLimit
            );
            assert_eq!(
                budget.decide(true, after.saturating_sub(40_000), 10).reason,
                LlmSummaryReason::AttemptLimit
            );
        }
        assert_eq!(
            LlmSummaryBudget::default().decide(true, 300_000, 1).reason,
            LlmSummaryReason::Allowed
        );
    }

    #[test]
    fn llm_summary_policy_structural_no_op_keeps_one_model_call_available() {
        let mut budget = LlmSummaryBudget::default();
        let first = budget.begin_attempt(gate(300_000, 1)).unwrap();
        budget.finish_attempt(first, 300_000, false);
        assert_eq!(budget.model_calls, 0);
        let second = budget.begin_attempt(gate(320_000, 2)).unwrap();
        assert_eq!(second.index, 2);
        budget.finish_attempt(second, 250_000, true);
        assert_eq!(budget.model_calls, 1);
        assert_eq!(
            budget.decide(true, 400_000, 3).reason,
            LlmSummaryReason::AttemptLimit
        );
    }

    #[test]
    fn llm_summary_policy_no_op_retry_requires_growth_and_new_round() {
        let mut budget = LlmSummaryBudget::default();
        let first = budget.begin_attempt(gate(300_000, 4)).unwrap();
        budget.finish_attempt(first, 300_000, false);
        assert_eq!(
            budget.decide(true, 340_000, 4).reason,
            LlmSummaryReason::RoundGate
        );
        assert_eq!(
            budget.decide(true, 319_999, 5).reason,
            LlmSummaryReason::GrowthGate
        );
        assert_eq!(
            budget.decide(true, 280_001, 5).reason,
            LlmSummaryReason::GrowthGate
        );
        assert_eq!(
            budget.decide(true, 320_000, 5).reason,
            LlmSummaryReason::Allowed
        );
        assert_eq!(
            budget.decide(true, 280_000, 5).reason,
            LlmSummaryReason::Allowed
        );
    }

    #[test]
    fn llm_summary_policy_two_structural_no_ops_exhaust_gate_attempts() {
        let mut budget = LlmSummaryBudget::default();
        let first = budget.begin_attempt(gate(300_000, 1)).unwrap();
        budget.finish_attempt(first, 300_000, false);
        let second = budget.begin_attempt(gate(320_000, 2)).unwrap();
        budget.finish_attempt(second, 320_000, false);
        assert_eq!(budget.model_calls, 0);
        assert_eq!(
            budget.decide(true, 400_000, 3).reason,
            LlmSummaryReason::AttemptLimit
        );
    }

    #[test]
    fn llm_summary_policy_cancelled_attempt_stays_reserved() {
        let mut budget = LlmSummaryBudget::default();
        let _cancelled = budget.begin_attempt(gate(300_000, 1)).unwrap();
        assert_eq!(
            budget.decide(true, 400_000, 2).reason,
            LlmSummaryReason::AttemptLimit
        );
    }
}
