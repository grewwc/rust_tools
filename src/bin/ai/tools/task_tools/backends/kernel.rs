//! Kernel adapter; the driver retains ownership of turn setup and scopes.

use std::sync::{Arc, atomic::AtomicBool};

use super::super::backend::{
    Admission, Backend, BackendFuture, BackendId, BackendText, Continuation, ContinueContext,
    Descriptor, Events, ExecutionIdentity, ExecutionMode, ModelContext, Outcome, RunContext,
    SideNoteMode,
};

pub(super) const DESCRIPTOR: Descriptor = Descriptor {
    id: BackendId::KERNEL,
    task_names: &["kernel"],
    use_names: &["kernel"],
    supports_model_pick: true,
    supports_worker_model: false,
    supports_resume: false,
    side_notes: SideNoteMode::Live,
    display_order: 0,
    use_order: 1,
    model_discovery: None,
    text: BackendText {
        display_name: "Kernel",
        default_progress: "working",
        worker_model_label: None,
        model_reason: None,
        preflight_error: None,
        continue_target_error: None,
        session_footer: None,
        worker_model_schema: None,
        continuation_schema: None,
    },
};

pub(super) struct KernelBackend;

impl Backend for KernelBackend {
    fn descriptor(&self) -> &'static Descriptor {
        &DESCRIPTOR
    }

    fn execution_mode(&self) -> ExecutionMode {
        ExecutionMode::KernelDriver
    }

    fn preflight(&self, _: &Admission<'_>) -> Result<(), String> {
        Ok(())
    }

    fn attribution(&self, context: &ModelContext<'_>) -> ExecutionIdentity {
        ExecutionIdentity {
            backend: DESCRIPTOR.id,
            model: context.kernel_model.to_string(),
            reasoning_effort: context.kernel_reasoning_effort.map(str::to_string),
            selection_explanation: context.selection_explanation.to_string(),
        }
    }

    fn continuation(&self, context: &ContinueContext<'_>) -> Result<Continuation, String> {
        if context
            .continue_from
            .is_some_and(|value| !value.trim().is_empty())
        {
            return Err(super::backend_registry().resume_required_error());
        }
        Ok(Continuation::fresh(context.prompt))
    }

    fn model_names(&self) -> Vec<String> {
        // Completion candidates must be selectors a user can type: encrypted
        // registry names are ciphertext, so map through `model_handle` and drop
        // repeats that resolve to the same selector.
        let mut seen = rust_tools::commonw::FastSet::default();
        crate::ai::model_names::all()
            .into_iter()
            .filter_map(|model| {
                let handle = crate::ai::model_names::model_handle(model);
                (!handle.is_empty() && seen.insert(handle.clone())).then_some(handle)
            })
            .collect()
    }

    fn run<'a>(
        &'a self,
        context: RunContext,
        events: Events,
        cancel: Arc<AtomicBool>,
    ) -> BackendFuture<'a, Outcome> {
        match context {
            RunContext::Kernel(run) => run(events, cancel),
            RunContext::External(_) => Box::pin(async move {
                Outcome::failed(
                    "kernel backend has no external runner".to_string(),
                    &events,
                    DESCRIPTOR.text.default_progress,
                )
            }),
        }
    }
}
