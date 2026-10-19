//! Unified admission and observability for turn-health signals.
//!
//! The individual detectors still decide whether their own pattern matched,
//! but every resulting intervention passes through one budget and one decision
//! record here. This keeps the detectors from stacking independent one-shot
//! prompts on the same turn, and makes a later hit-rate review possible
//! without re-deriving state from model-visible text.

use crate::ai::driver::decision_log::{DecisionLog, DecisionType, get_decision_log_store};

/// Maximum non-forcing interventions a single turn may inject.
///
/// Hard stops bypass this budget: a detected loop must still converge even
/// after the soft-note allowance is exhausted. The cap only limits advisory
/// notes, which are the part that can pile up across detectors.
const MAX_ADVISORY_SIGNALS_PER_TURN: usize = 8;

/// Severity of a turn-health signal, decided by the detector that matched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::ai::driver::turn_runtime) enum SignalKind {
    /// Convergence enforcement (loop hard-stop, no-tool handoff). Always
    /// admitted and recorded as `force_final`.
    HardStop,
    /// A one-shot nudge that consumes the shared per-turn advisory budget
    /// and is recorded as `inject_note`. Suppression only drops a nudge:
    /// convergence never depends on it because every hard stop is detected
    /// independently of whether its soft precursor was injected.
    Advisory,
}

#[derive(Debug)]
pub(in crate::ai::driver::turn_runtime) struct TurnSignalBudget {
    remaining: usize,
}

impl Default for TurnSignalBudget {
    fn default() -> Self {
        Self {
            remaining: MAX_ADVISORY_SIGNALS_PER_TURN,
        }
    }
}

impl TurnSignalBudget {
    /// Admit one signal and record the decision.
    ///
    /// [`SignalKind::HardStop`] signals always pass. Advisory signals consume one unit of the
    /// per-turn allowance; once exhausted they are suppressed and the
    /// suppression is logged as the chosen outcome.
    pub(in crate::ai::driver::turn_runtime) fn admit(
        &mut self,
        session_id: &str,
        turn_id: usize,
        iteration: usize,
        name: &'static str,
        kind: SignalKind,
    ) -> bool {
        let hard = matches!(kind, SignalKind::HardStop);
        let admitted = hard || self.remaining > 0;
        if admitted && !hard {
            self.remaining -= 1;
        }
        let chosen = if admitted {
            if hard { "force_final" } else { "inject_note" }
        } else {
            "suppress_budget_exhausted"
        };
        get_decision_log_store().log(DecisionLog {
            timestamp: chrono::Local::now().timestamp_millis(),
            session_id: session_id.to_string(),
            turn_id,
            decision_type: DecisionType::TurnSignal,
            context: format!("turn-health signal `{name}` at turn {turn_id} iteration {iteration}"),
            alternatives_considered: vec![
                "force_final".to_string(),
                "inject_note".to_string(),
                "suppress_budget_exhausted".to_string(),
            ],
            chosen_option: chosen.to_string(),
            reasoning: if hard {
                "hard-stop signals bypass the advisory budget".to_string()
            } else {
                format!(
                    "advisory signals share one per-turn budget; {remaining} remaining after admission",
                    remaining = self.remaining
                )
            },
            confidence: Some(1.0),
            outcome: None,
            execution_time_ms: None,
        });
        admitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advisory_budget_is_bounded_but_hard_signals_always_pass() {
        let mut budget = TurnSignalBudget::default();
        for _ in 0..MAX_ADVISORY_SIGNALS_PER_TURN {
            assert!(budget.admit("test-session", 7, 1, "coarse", SignalKind::Advisory));
        }
        assert!(!budget.admit(
            "test-session",
            7,
            1,
            "target_repeat",
            SignalKind::Advisory
        ));
        assert!(budget.admit("test-session", 7, 1, "hard", SignalKind::HardStop));
        assert!(budget.admit(
            "test-session",
            7,
            1,
            "coarse_hard",
            SignalKind::HardStop
        ));
    }
}