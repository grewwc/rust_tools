//! Thin adapter over the existing TraeCLI argv, protocol, retries, and parser.

use std::sync::{Arc, Mutex, atomic::AtomicBool};

use super::super::backend::{
    Admission, Backend, BackendFuture, BackendId, BackendText, Continuation, ContinueContext,
    Descriptor, Events, ExecutionIdentity, ModelContext, ModelDiscovery, Outcome, RunContext,
    SideNoteMode, validate_external_skill_context,
};
use super::super::{backend_session, traecli as runner, traecli_models};

pub(super) const DESCRIPTOR: Descriptor = Descriptor {
    id: BackendId::TRAECLI,
    task_names: &["traecli", "trae-cli", "coco"],
    use_names: &["traecli"],
    supports_model_pick: true,
    supports_worker_model: true,
    supports_resume: true,
    side_notes: SideNoteMode::DeferredContinuation,
    display_order: 1,
    use_order: 0,
    model_discovery: Some(ModelDiscovery {
        binary: "traecli",
        binary_env: "TRAECLI_BIN",
        arguments: &["models"],
        cache_key: "models",
    }),
    text: BackendText {
        display_name: "TraeCLI",
        default_progress: "traecli",
        worker_model_label: Some("TraeCLI worker model"),
        model_reason: Some("model_reason=backend model '{model}'"),
        preflight_error: Some(
            "traecli backend requested but binary '{binary}' was not found; install traecli or set TRAECLI_BIN",
        ),
        continue_target_error: Some(runner::CONTINUE_TARGET_ERROR),
        session_footer: Some("Backend session: {session} (resume it via `backend_continue_from`)"),
        worker_model_schema: Some(
            "Model for backend 'traecli' (see `traecli models`); ignored otherwise. An explicit value overrides the /use session worker model, then TRAECLI_MODEL, ai.traecli.default_model, and built-in 'gpt-5.5' defaults.",
        ),
        continuation_schema: Some(
            "Continue a previous traecli run: pass the earlier task's `task_id` to resume its traecli session, or a raw traecli session UUID. Only for backend 'traecli'; the referenced run must have completed.",
        ),
    },
};

pub(super) struct TraeCliBackend;

impl Backend for TraeCliBackend {
    fn descriptor(&self) -> &'static Descriptor {
        &DESCRIPTOR
    }

    fn preflight(&self, context: &Admission<'_>) -> Result<(), String> {
        if !runner::traecli_binary_available() {
            return Err(DESCRIPTOR
                .text
                .preflight_error
                .unwrap()
                .replace("{binary}", &runner::traecli_binary()));
        }
        if let Some(spec) = context.external_spec {
            validate_external_skill_context(spec)?;
        }
        Ok(())
    }

    fn attribution(&self, context: &ModelContext<'_>) -> ExecutionIdentity {
        let model =
            runner::traecli_model_for(context.backend_model.or(context.session_worker_model));
        let agent_reason = context
            .selection_explanation
            .split_once('\n')
            .map(|(agent_reason, _)| agent_reason)
            .unwrap_or(context.selection_explanation);
        let model_reason = DESCRIPTOR
            .text
            .model_reason
            .unwrap()
            .replace("{model}", &model);
        ExecutionIdentity {
            backend: DESCRIPTOR.id,
            model,
            reasoning_effort: context
                .backend_reasoning_effort
                .and_then(runner::normalize_traecli_reasoning_effort)
                .map(str::to_string),
            selection_explanation: format!("{agent_reason}\n{model_reason}"),
        }
    }

    fn continuation(&self, context: &ContinueContext<'_>) -> Result<Continuation, String> {
        backend_session::prepare_continuation(
            DESCRIPTOR.id,
            context,
            runner::looks_like_session_id,
            DESCRIPTOR.text.continue_target_error.unwrap(),
        )
    }

    fn model_names(&self) -> Vec<String> {
        traecli_models::traecli_model_names()
    }

    fn run<'a>(
        &'a self,
        context: RunContext,
        mut events: Events,
        cancel: Arc<AtomicBool>,
    ) -> BackendFuture<'a, Outcome> {
        Box::pin(async move {
            let RunContext::External(mut spec) = context else {
                return Outcome::failed(
                    "traecli backend requires an external run context".to_string(),
                    &events,
                    DESCRIPTOR.text.default_progress,
                );
            };
            if let Some(slot) = &events.phase_slot {
                spec.phase_slot = Some(slot.clone());
            } else {
                events.phase_slot = spec.phase_slot.clone();
            }
            let slot = events
                .session_slot
                .get_or_insert_with(|| Arc::new(Mutex::new(None)))
                .clone();
            if let Err(error) = validate_external_skill_context(&spec) {
                return Outcome::failed(error, &events, DESCRIPTOR.text.default_progress);
            }
            let evidence_slot = Arc::new(Mutex::new(None));
            let result = runner::run_traecli_subagent_with_events(
                &spec,
                cancel,
                slot,
                evidence_slot.clone(),
            )
            .await;
            let evidence = super::super::registry::lock_unpoisoned(&evidence_slot).clone();
            execution_outcome(result, evidence, &events)
        })
    }
}

fn execution_outcome(
    result: Result<runner::TraeCliOutcome, String>,
    evidence: Option<runner::TraeCliOutcome>,
    events: &Events,
) -> Outcome {
    let (mut outcome, final_body) = match result {
        Ok(final_body) => (
            Outcome {
                progress_snapshot: events
                    .phase_slot
                    .as_ref()
                    .and_then(crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot),
                backend_session: events
                    .session_slot
                    .as_ref()
                    .and_then(backend_session::captured_session),
                ..Outcome::default()
            },
            Some(final_body),
        ),
        Err(error) => (
            Outcome::failed(error, events, DESCRIPTOR.text.default_progress),
            evidence,
        ),
    };
    if let Some(final_body) = final_body {
        outcome.parent_payload = final_body.result.clone();
        outcome.validation_body = final_body.result;
        outcome.progress = final_body
            .num_turns
            .map(|turns| format!("traecli turns {turns}"))
            .unwrap_or_else(|| DESCRIPTOR.text.default_progress.to_string());
        outcome.backend_session = final_body.session_id.or(outcome.backend_session);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_execution_keeps_final_body_and_streamed_session() {
        let events = Events {
            session_slot: Some(Arc::new(Mutex::new(Some("streamed-session".to_string())))),
            ..Events::default()
        };
        let evidence = runner::TraeCliOutcome {
            result: "  final evidence\n".to_string(),
            is_error: true,
            num_turns: Some(2),
            ..runner::TraeCliOutcome::default()
        };
        let outcome =
            execution_outcome(Err("original failure".to_string()), Some(evidence), &events);
        assert_eq!(outcome.execution_error.as_deref(), Some("original failure"));
        assert_eq!(outcome.parent_payload, "  final evidence\n");
        assert_eq!(outcome.validation_body, "  final evidence\n");
        assert_eq!(outcome.progress, "traecli turns 2");
        assert_eq!(outcome.backend_session.as_deref(), Some("streamed-session"));
    }
}
