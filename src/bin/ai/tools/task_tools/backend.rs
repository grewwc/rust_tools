//! Shared subagent backend contract. Backends register their capabilities and
//! execution adapters in `backends`; legacy names remain compatibility aliases.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::Value;

use crate::ai::driver::runtime_ctx::{self, SubagentPhaseSlot, SubagentProgressSnapshot};

use super::backend_session::{DeferredNotes, SessionSlot};

/// Canonical registration key. Aliases belong to descriptors, never to this key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BackendId(&'static str);

impl BackendId {
    pub(crate) const KERNEL: Self = Self::new("kernel");
    pub(crate) const TRAECLI: Self = Self::new("traecli");

    // Keep source compatibility for the existing runner and focused tests.
    #[allow(non_upper_case_globals)]
    pub(crate) const Kernel: Self = Self::KERNEL;
    #[allow(non_upper_case_globals)]
    pub(crate) const TraeCli: Self = Self::TRAECLI;

    pub(crate) const fn id(self) -> Self {
        self
    }

    pub(crate) const fn new(canonical: &'static str) -> Self {
        Self(canonical)
    }

    pub(crate) const fn as_str(self) -> &'static str {
        self.0
    }

    pub(crate) fn resolve(self) -> Arc<dyn Backend> {
        super::backends::backend_registry()
            .get(self)
            .expect("selected subagent backend must be registered")
    }

    pub(crate) fn parse_optional(value: Option<&Value>) -> Result<Option<Self>, String> {
        super::backends::backend_registry().parse_optional(value)
    }

    pub(crate) fn parse(value: Option<&Value>) -> Result<Self, String> {
        Ok(Self::parse_optional(value)?.unwrap_or(Self::KERNEL))
    }

    /// Stored values are canonical IDs. Legacy aliases still fall back to kernel.
    pub(crate) fn from_stored(value: Option<&str>) -> Self {
        value
            .and_then(|name| super::backends::backend_registry().canonical_id(name))
            .unwrap_or(Self::KERNEL)
    }

    /// The session's kernel default is represented by an absent stored field.
    pub(crate) fn stored_value(self) -> Option<String> {
        (self != Self::default()).then(|| self.as_str().to_string())
    }

    #[cfg(test)]
    pub(crate) fn is_traecli(self) -> bool {
        self == Self::TRAECLI
    }
}

impl Default for BackendId {
    fn default() -> Self {
        Self::KERNEL
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SideNoteMode {
    Live,
    DeferredContinuation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionMode {
    KernelDriver,
    External,
}

/// Existing backend wording, retained as templates for migrated consumers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BackendText {
    pub(crate) display_name: &'static str,
    pub(crate) default_progress: &'static str,
    pub(crate) worker_model_label: Option<&'static str>,
    pub(crate) model_reason: Option<&'static str>,
    pub(crate) preflight_error: Option<&'static str>,
    pub(crate) continue_target_error: Option<&'static str>,
    pub(crate) session_footer: Option<&'static str>,
    pub(crate) worker_model_schema: Option<&'static str>,
    pub(crate) continuation_schema: Option<&'static str>,
}

/// Shell completion invokes this command lazily, independently of registration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelDiscovery {
    pub(crate) binary: &'static str,
    pub(crate) binary_env: &'static str,
    pub(crate) arguments: &'static [&'static str],
    /// Retains existing shell cache variable names for compatibility.
    pub(crate) cache_key: &'static str,
}

/// Static metadata. Constructing the registry must not launch model discovery.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Descriptor {
    pub(crate) id: BackendId,
    /// Includes the canonical name and every accepted task-field alias.
    pub(crate) task_names: &'static [&'static str],
    /// Includes every accepted `/use` name, independently of task-field aliases.
    pub(crate) use_names: &'static [&'static str],
    pub(crate) supports_worker_model: bool,
    pub(crate) supports_resume: bool,
    pub(crate) side_notes: SideNoteMode,
    pub(crate) display_order: u16,
    pub(crate) use_order: u16,
    pub(crate) model_discovery: Option<ModelDiscovery>,
    pub(crate) text: BackendText,
}

pub(crate) struct Admission<'a> {
    /// Backend-independent depth, permissions, and task limits remain upstream.
    pub(crate) external_spec: Option<&'a ExternalSubagentSpec>,
}

pub(crate) struct ModelContext<'a> {
    pub(crate) kernel_model: &'a str,
    pub(crate) backend_model: Option<&'a str>,
    pub(crate) session_worker_model: Option<&'a str>,
    /// Already-resolved kernel display label and external wire effort.
    pub(crate) kernel_reasoning_effort: Option<&'a str>,
    pub(crate) backend_reasoning_effort: Option<&'a str>,
    pub(crate) selection_explanation: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExecutionIdentity {
    pub(crate) backend: BackendId,
    pub(crate) model: String,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) selection_explanation: String,
}

pub(crate) struct ContinueContext<'a> {
    pub(crate) continue_from: Option<&'a str>,
    pub(crate) parent_history: &'a std::path::Path,
    pub(crate) prompt: &'a str,
}

#[derive(Debug, Clone)]
pub(crate) struct Continuation {
    pub(crate) session_id: Option<String>,
    pub(crate) prompt: String,
    pub(crate) deferred_notes: DeferredNotes,
}

impl Continuation {
    pub(crate) fn fresh(prompt: &str) -> Self {
        Self {
            session_id: None,
            prompt: prompt.to_string(),
            deferred_notes: DeferredNotes::default(),
        }
    }

    pub(crate) fn into_parts(self) -> (Option<String>, String, Vec<PathBuf>) {
        (self.session_id, self.prompt, self.deferred_notes.files)
    }
}

pub(crate) type BackendFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The callback owns App, MCP, skills, history, and the driver's scoped turn
/// inputs. Keeping their setup in the driver preserves both kernel execution
/// paths while allowing either one to route through the same backend contract.
pub(crate) type KernelRun =
    Box<dyn FnOnce(Events, Arc<AtomicBool>) -> BackendFuture<'static, Outcome> + Send>;

pub(crate) enum RunContext {
    Kernel(KernelRun),
    External(ExternalSubagentSpec),
}

impl RunContext {
    pub(crate) fn kernel(
        run: impl FnOnce(Events, Arc<AtomicBool>) -> BackendFuture<'static, Outcome> + Send + 'static,
    ) -> Self {
        Self::Kernel(Box::new(run))
    }
}

/// Explicit handles also work on blocking workers, which lack task-local slots.
#[derive(Clone, Default)]
pub(crate) struct Events {
    pub(crate) phase_slot: Option<SubagentPhaseSlot>,
    pub(crate) session_slot: Option<SessionSlot>,
}

/// Execution failure and evidence are independent. A failed run can still have
/// reusable payload, progress checkpoints, and a captured backend session.
#[derive(Debug, Clone, Default)]
pub(crate) struct Outcome {
    pub(crate) execution_error: Option<String>,
    pub(crate) parent_payload: String,
    pub(crate) validation_body: String,
    pub(crate) progress: String,
    pub(crate) progress_snapshot: Option<SubagentProgressSnapshot>,
    pub(crate) backend_session: Option<String>,
}

impl Outcome {
    pub(crate) fn failed(error: String, events: &Events, default_progress: &str) -> Self {
        let progress_snapshot = events
            .phase_slot
            .as_ref()
            .and_then(runtime_ctx::subagent_progress_state_snapshot);
        Self {
            execution_error: Some(error),
            parent_payload: progress_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.checkpoint_excerpt.clone())
                .unwrap_or_default(),
            progress: progress_snapshot
                .as_ref()
                .map(SubagentProgressSnapshot::display_status_line)
                .unwrap_or_else(|| default_progress.to_string()),
            progress_snapshot,
            backend_session: events
                .session_slot
                .as_ref()
                .and_then(super::backend_session::captured_session),
            ..Self::default()
        }
    }
}

/// Object-safe contract shared by kernel and external backend implementations.
pub(crate) trait Backend: Send + Sync {
    fn descriptor(&self) -> &'static Descriptor;
    /// CLI adapters use the external spec; the kernel opts into the driver's
    /// scoped callback without coupling execution to model or resume support.
    fn execution_mode(&self) -> ExecutionMode {
        ExecutionMode::External
    }
    fn preflight(&self, context: &Admission<'_>) -> Result<(), String>;
    fn attribution(&self, context: &ModelContext<'_>) -> ExecutionIdentity;
    fn continuation(&self, context: &ContinueContext<'_>) -> Result<Continuation, String>;
    fn model_names(&self) -> Vec<String>;
    fn run<'a>(
        &'a self,
        context: RunContext,
        events: Events,
        cancel: Arc<AtomicBool>,
    ) -> BackendFuture<'a, Outcome>;
}

/// Source compatibility only; execution and parsing use the registered ID.
pub(crate) type SubagentBackend = BackendId;

pub(crate) fn backend_launch_tag(backend: BackendId, show: bool) -> String {
    if show {
        format!(" [backend={}]", backend.as_str())
    } else {
        String::new()
    }
}

/// Resolve the existing optional task-record backend field.
pub(crate) fn task_backend(backend: Option<BackendId>) -> Arc<dyn Backend> {
    backend.unwrap_or_default().resolve()
}

/// Parse the optional per-call `backend_model` task field: a model name in
/// the selected backend's own registry, or `None` for the backend default.
/// Empty and null values mean "no override".
pub(crate) fn parse_backend_model(value: Option<&Value>) -> Result<Option<String>, String> {
    parse_optional_string_field(value, "backend_model")
}

/// Parse the optional per-call `backend_continue_from` task field: a previous
/// task's `task_id` whose backend session should be resumed, or a raw backend
/// session id. Resolution against recorded sessions happens at dispatch time.
pub(crate) fn parse_backend_continue_from(value: Option<&Value>) -> Result<Option<String>, String> {
    parse_optional_string_field(value, "backend_continue_from")
}

/// Shared shape for optional neutral string fields: missing, null, and blank
/// all mean "unset".
fn parse_optional_string_field(
    value: Option<&Value>,
    field: &str,
) -> Result<Option<String>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let text = value
        .as_str()
        .ok_or_else(|| format!("'{field}' must be a string"))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
}

/// Input spec for one external subagent run.
#[derive(Debug, Clone)]
pub(crate) struct ExternalSubagentSpec {
    pub(crate) prompt: String,
    pub(crate) cwd: PathBuf,
    /// Whether the parent granted workspace access (`inherit` contains `cwd`).
    /// Backends map this to their own isolation control (TraeCLI: sandbox).
    pub(crate) workspace_allowed: bool,
    /// Optional backend model override. Never fed from the parent model name:
    /// registries use different naming. `None` selects the backend default.
    pub(crate) model: Option<String>,
    /// Rendered instructions for the parent turn's active skills (already
    /// gated by `inherit.skills` and scoped to the active set at dispatch).
    /// Backends present it as instructions.
    pub(crate) skills: Option<String>,
    /// Resume target for session continuity: a backend session id resolved at
    /// dispatch time from `backend_continue_from`, whose deferred side-notes
    /// have already been prepended to `prompt`. `None` starts a fresh run.
    pub(crate) resume_session: Option<String>,
    /// Deferred side-note queue files whose contents are already in `prompt`.
    /// Keep them queued until the backend process spawns successfully.
    pub(crate) side_note_files: Vec<PathBuf>,
    /// Parent's resolved reasoning-effort tier, in the provider-neutral wire
    /// vocabulary (`low`, `medium`, `high`, `xhigh`, `max`). Backends map it to
    /// their own control; `None` keeps the backend default.
    pub(crate) reasoning_effort: Option<String>,
    /// Original image files attached to the parent turn. External runners
    /// forward them to their image-input flag on the first request and keep
    /// the absolute paths available for later re-reads.
    pub(crate) image_files: Vec<String>,
    /// Live phase slot owned by the spawning task. External runners execute on
    /// blocking threads, so they publish progress through this explicit handle
    /// instead of the task-local phase slot.
    pub(crate) phase_slot: Option<crate::ai::driver::runtime_ctx::SubagentPhaseSlot>,
}

/// Generic outcome for external subagent runners. `output` carries the final
/// answer text (mapped to the result-channel payload), while `progress` feeds
/// the same progress slot the kernel runner uses. `session_id` carries the
/// backend session id when the backend reports one, so follow-up tasks can
/// resume it via `backend_continue_from`.
#[derive(Debug, Clone, Default)]
pub(crate) struct ExternalSubagentOutcome {
    pub(crate) output: String,
    pub(crate) progress: String,
    pub(crate) session_id: Option<String>,
}

/// Compatibility interface for the existing runner and its focused tests.
/// Production dispatch uses the registered object-safe [`Backend`] adapters.
pub(crate) trait ExternalSubagentRunner: Send + Sync {
    fn backend(&self) -> SubagentBackend;
    fn run(
        &self,
        spec: &ExternalSubagentSpec,
        cancel: Arc<AtomicBool>,
    ) -> impl std::future::Future<Output = Result<ExternalSubagentOutcome, String>> + Send;
}

/// Upper bound on the rendered skill overlay for one external run. External
/// backends have hard input limits (TraeCLI rejects turns above roughly one
/// million characters), and the task prompt plus the backend's own baseline
/// must fit alongside the overlay. Fail closed with an actionable message
/// instead of silently dropping skill instructions.
pub(crate) const MAX_EXTERNAL_SKILL_CONTEXT_CHARS: usize = 262_144;

pub(super) fn validate_external_skill_context(spec: &ExternalSubagentSpec) -> Result<(), String> {
    // Resume sends only the follow-up; the session already holds the overlay.
    if spec.resume_session.is_some() {
        return Ok(());
    }
    let Some(skills) = spec.skills.as_deref() else {
        return Ok(());
    };
    let chars = skills.chars().count();
    if chars > MAX_EXTERNAL_SKILL_CONTEXT_CHARS {
        return Err(format!(
            "external subagent skill context is {chars} characters, exceeding the \
             {MAX_EXTERNAL_SKILL_CONTEXT_CHARS}-character limit; deactivate some skills \
             or use the kernel backend"
        ));
    }
    Ok(())
}

/// Dispatch one external subagent run to the runner for `backend`. Unknown
/// backends fail closed instead of silently running the wrong runner.
pub(crate) async fn run_external_subagent(
    backend: SubagentBackend,
    spec: &ExternalSubagentSpec,
    cancel: Arc<AtomicBool>,
) -> Result<ExternalSubagentOutcome, String> {
    validate_external_skill_context(spec)?;
    let outcome = backend
        .id()
        .resolve()
        .run(
            RunContext::External(spec.clone()),
            Events::default(),
            cancel,
        )
        .await;
    if let Some(error) = outcome.execution_error {
        return Err(error);
    }
    Ok(ExternalSubagentOutcome {
        output: outcome.parent_payload,
        progress: outcome.progress,
        session_id: outcome.backend_session,
    })
}

/// Follow-up hint appended to rendered results of external-backend runs that
/// report a resumable session. Appended post-validation so schema contracts
/// on the raw output are unaffected.
pub(crate) fn backend_session_footer(session: &str) -> String {
    format!("Backend session: {session} (resume it via `backend_continue_from`)")
}

pub(crate) fn backend_session_footer_for(backend: &dyn Backend, session: &str) -> String {
    backend
        .descriptor()
        .text
        .session_footer
        .map(|template| template.replace("{session}", session))
        .unwrap_or_else(|| backend_session_footer(session))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_parse_defaults_to_kernel() {
        assert_eq!(
            SubagentBackend::parse(None).unwrap(),
            SubagentBackend::Kernel
        );
        assert_eq!(
            SubagentBackend::parse(Some(&Value::Null)).unwrap(),
            SubagentBackend::Kernel
        );
        assert_eq!(
            SubagentBackend::parse(Some(&Value::from(""))).unwrap(),
            SubagentBackend::Kernel
        );
        assert_eq!(
            SubagentBackend::parse(Some(&Value::from("traecli"))).unwrap(),
            SubagentBackend::TraeCli
        );
    }

    #[test]
    fn backend_parse_rejects_unknown() {
        assert!(SubagentBackend::parse(Some(&Value::from("remote"))).is_err());
    }

    #[test]
    fn backend_from_stored_falls_back_to_kernel() {
        assert_eq!(SubagentBackend::from_stored(None), SubagentBackend::Kernel);
        assert_eq!(
            SubagentBackend::from_stored(Some("traecli")),
            SubagentBackend::TraeCli
        );
        assert_eq!(
            SubagentBackend::from_stored(Some("future-backend")),
            SubagentBackend::Kernel
        );
    }

    #[test]
    fn model_parse_trims_and_defaults_to_none() {
        assert_eq!(parse_backend_model(None).unwrap(), None);
        assert_eq!(parse_backend_model(Some(&Value::Null)).unwrap(), None);
        assert_eq!(parse_backend_model(Some(&Value::from(""))).unwrap(), None);
        assert_eq!(
            parse_backend_model(Some(&Value::from("  Seed-Code  "))).unwrap(),
            Some("Seed-Code".to_string())
        );
        assert!(parse_backend_model(Some(&Value::from(7))).is_err());
    }

    #[test]
    fn external_skill_context_budget_fails_closed_on_fresh_runs() {
        let oversized = "x".repeat(MAX_EXTERNAL_SKILL_CONTEXT_CHARS + 1);
        let mut spec = ExternalSubagentSpec {
            prompt: "do work".to_string(),
            cwd: std::env::temp_dir(),
            workspace_allowed: false,
            model: None,
            skills: Some(oversized),
            resume_session: None,
            side_note_files: Vec::new(),
            reasoning_effort: None,
            image_files: Vec::new(),
            phase_slot: None,
        };
        let err = validate_external_skill_context(&spec).unwrap_err();
        assert!(err.contains("262144"), "{err}");
        // Resume sends only the follow-up; the session already holds the overlay.
        spec.resume_session = Some("session-id".to_string());
        assert!(validate_external_skill_context(&spec).is_ok());
    }

    #[test]
    fn external_skill_context_budget_allows_normal_context() {
        let mut spec = ExternalSubagentSpec {
            prompt: "do work".to_string(),
            cwd: std::env::temp_dir(),
            workspace_allowed: false,
            model: None,
            skills: Some("x".repeat(1024)),
            resume_session: None,
            side_note_files: Vec::new(),
            reasoning_effort: None,
            image_files: Vec::new(),
            phase_slot: None,
        };
        assert!(validate_external_skill_context(&spec).is_ok());
        spec.skills = None;
        assert!(validate_external_skill_context(&spec).is_ok());
    }
}
