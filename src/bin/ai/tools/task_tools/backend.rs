//! Neutral subagent-backend abstraction: backend selection plus the
//! external runner interface shared by all headless backends.
//!
//! The in-process kernel runner stays inline in the dispatch sites because it
//! needs the full turn context (App, MCP, skills, history, memory); external
//! backends (TraeCLI today, managed sessions or other CLIs tomorrow)
//! implement [`ExternalSubagentRunner`] against the small
//! [`ExternalSubagentSpec`] input. Backend-specific code (CLI surface, arg
//! building, output parsing, session bookkeeping) lives in the backend's own
//! module, never here.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::Value;

/// Subagent execution backend. Defaults to the in-process kernel runner so
/// existing calls without a `backend` field keep their exact behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SubagentBackend {
    #[default]
    Kernel,
    TraeCli,
}

impl SubagentBackend {
    pub(crate) fn parse_optional(value: Option<&Value>) -> Result<Option<Self>, String> {
        let Some(value) = value else {
            return Ok(None);
        };
        if value.is_null() {
            return Ok(None);
        }
        match value.as_str() {
            Some("") => Ok(None),
            Some("kernel") => Ok(Some(Self::Kernel)),
            Some("traecli") | Some("trae-cli") | Some("coco") => Ok(Some(Self::TraeCli)),
            Some(other) => Err(format!(
                "unknown subagent backend '{other}' (expected 'kernel' or 'traecli')"
            )),
            None => Err("'backend' must be a string".to_string()),
        }
    }

    pub(crate) fn parse(value: Option<&Value>) -> Result<Self, String> {
        Ok(Self::parse_optional(value)?.unwrap_or(Self::Kernel))
    }

    pub(crate) fn from_stored(value: Option<&str>) -> Self {
        match value {
            Some("traecli") => Self::TraeCli,
            _ => Self::Kernel,
        }
    }

    pub(crate) fn is_traecli(self) -> bool {
        matches!(self, Self::TraeCli)
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::TraeCli => "traecli",
        }
    }
}

pub(crate) fn backend_launch_tag(backend: SubagentBackend, show: bool) -> String {
    if show {
        format!(" [backend={}]", backend.as_str())
    } else {
        String::new()
    }
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
    /// dispatch time from `backend_continue_from`. `None` starts a fresh run.
    pub(crate) resume_session: Option<String>,
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

/// Interface for headless external subagent backends (TraeCLI today, managed
/// sessions or other CLIs tomorrow). The in-process kernel runner stays inline
/// in the dispatch sites because it needs the full turn context (App, MCP,
/// skills, history, memory); external runners share the small
/// [`ExternalSubagentSpec`]-shaped input (prompt, cwd, workspace flag, model,
/// rendered skills) plus cooperative cancellation, so new backends only add a
/// runner plus a dispatcher arm without touching dispatch logic.
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

fn validate_external_skill_context(spec: &ExternalSubagentSpec) -> Result<(), String> {
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
    match backend {
        SubagentBackend::TraeCli => super::traecli::TraeCliRunner.run(spec, cancel).await,
        SubagentBackend::Kernel => Err("kernel backend has no external runner".to_string()),
    }
}

/// Follow-up hint appended to rendered results of external-backend runs that
/// report a resumable session. Appended post-validation so schema contracts
/// on the raw output are unaffected.
pub(crate) fn backend_session_footer(session: &str) -> String {
    format!("Backend session: {session} (resume it via `backend_continue_from`)")
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
            reasoning_effort: None,
            image_files: Vec::new(),
            phase_slot: None,
        };
        assert!(validate_external_skill_context(&spec).is_ok());
        spec.skills = None;
        assert!(validate_external_skill_context(&spec).is_ok());
    }
}
