//! Shared external subagent input mapping; callers own runtime scopes and execution lifecycle.

use std::path::PathBuf;

use crate::ai::{skills::SkillManifest, tools::task_tools};

use super::runtime_ctx;

/// Callers supply the active skill names from their own dispatch scope or stored task goal.
pub(super) struct ExternalSubagentLaunch<'a> {
    pub(super) prompt: String,
    pub(super) cwd: PathBuf,
    pub(super) workspace_allowed: bool,
    pub(super) model: Option<String>,
    pub(super) skill_manifests: &'a [SkillManifest],
    pub(super) active_skill_names: &'a [String],
    pub(super) resume_session: Option<String>,
    pub(super) side_note_files: Vec<PathBuf>,
    pub(super) reasoning_effort: Option<String>,
    pub(super) image_files: Vec<String>,
    pub(super) phase_slot: Option<runtime_ctx::SubagentPhaseSlot>,
}

impl ExternalSubagentLaunch<'_> {
    pub(super) fn into_spec(self) -> task_tools::ExternalSubagentSpec {
        task_tools::ExternalSubagentSpec {
            prompt: self.prompt,
            cwd: self.cwd,
            workspace_allowed: self.workspace_allowed,
            model: self.model,
            skills: crate::ai::skills::render_external_skill_context(
                self.skill_manifests,
                self.active_skill_names,
            ),
            resume_session: self.resume_session,
            side_note_files: self.side_note_files,
            reasoning_effort: self.reasoning_effort,
            image_files: self.image_files,
            phase_slot: self.phase_slot,
        }
    }
}

/// Resolve cwd while the caller still holds the parent's runtime scope.
pub(super) fn resolve_external_subagent_cwd(
    workspace_allowed: bool,
    inherited_cwd_dir: Option<&PathBuf>,
    private_cwd_dir: Option<&PathBuf>,
) -> PathBuf {
    if workspace_allowed {
        inherited_cwd_dir
            .cloned()
            .or_else(|| runtime_ctx::effective_cwd().ok())
            .unwrap_or_else(|| PathBuf::from("."))
    } else {
        private_cwd_dir.cloned().unwrap_or_else(std::env::temp_dir)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn dispatch_launch_external_spec_preserves_inputs() {
        let skills: Vec<SkillManifest> = serde_json::from_value(serde_json::json!([
            {"name": "inactive", "prompt": "INACTIVE_BODY"},
            {"name": "stored", "prompt": "STORED_BODY"},
            {"name": "agent-team", "prompt": "PARENT_ORCHESTRATION_BODY"},
        ]))
        .unwrap();

        for active_skill_names in [
            Vec::new(),
            vec!["stored".to_string(), "agent-team".to_string()],
        ] {
            for (model, reasoning_effort, resume_session, workspace_allowed) in [
                (None, None, None, false),
                (
                    Some("caller-model"),
                    Some("xhigh"),
                    Some("resolved-session"),
                    true,
                ),
            ] {
                let prompt = "resolved prompt with deferred notes".to_string();
                let prompt_ptr = prompt.as_ptr();
                let image_files = vec![
                    "/task/second.png".to_string(),
                    "/task/first.png".to_string(),
                ];
                let image_files_ptr = image_files.as_ptr();
                let side_note_files =
                    vec![PathBuf::from("second.note"), PathBuf::from("first.note")];
                let side_note_files_ptr = side_note_files.as_ptr();
                let phase_slot = runtime_ctx::new_subagent_progress_slot();
                let spec = ExternalSubagentLaunch {
                    prompt,
                    cwd: PathBuf::from("captured-cwd"),
                    workspace_allowed,
                    model: model.map(str::to_string),
                    skill_manifests: &skills,
                    active_skill_names: &active_skill_names,
                    resume_session: resume_session.map(str::to_string),
                    side_note_files,
                    reasoning_effort: reasoning_effort.map(str::to_string),
                    image_files,
                    phase_slot: Some(phase_slot.clone()),
                }
                .into_spec();

                assert_eq!(spec.prompt, "resolved prompt with deferred notes");
                assert_eq!(spec.prompt.as_ptr(), prompt_ptr);
                assert_eq!(spec.cwd, std::path::Path::new("captured-cwd"));
                assert_eq!(spec.workspace_allowed, workspace_allowed);
                assert_eq!(spec.model.as_deref(), model);
                assert_eq!(
                    spec.skills,
                    crate::ai::skills::render_external_skill_context(&skills, &active_skill_names)
                );
                assert_eq!(spec.resume_session.as_deref(), resume_session);
                assert_eq!(
                    spec.side_note_files,
                    [PathBuf::from("second.note"), PathBuf::from("first.note")]
                );
                assert_eq!(spec.side_note_files.as_ptr(), side_note_files_ptr);
                assert_eq!(spec.reasoning_effort.as_deref(), reasoning_effort);
                assert_eq!(spec.image_files, ["/task/second.png", "/task/first.png"]);
                assert_eq!(spec.image_files.as_ptr(), image_files_ptr);
                assert!(Arc::ptr_eq(spec.phase_slot.as_ref().unwrap(), &phase_slot));
            }
        }
    }

    #[test]
    fn dispatch_launch_cwd_preserves_inheritance_and_isolation() {
        let ambient = PathBuf::from("ambient-parent-cwd");
        let inherited = PathBuf::from("captured-parent-cwd");
        let private = PathBuf::from("private-task-cwd");

        runtime_ctx::SUBAGENT_CWD.sync_scope(ambient.clone(), || {
            assert_eq!(
                resolve_external_subagent_cwd(true, Some(&inherited), Some(&private)),
                inherited
            );
            assert_eq!(
                resolve_external_subagent_cwd(true, None, Some(&private)),
                ambient
            );
            assert_eq!(
                resolve_external_subagent_cwd(false, Some(&inherited), Some(&private)),
                private
            );
            assert_eq!(
                resolve_external_subagent_cwd(false, Some(&inherited), None),
                std::env::temp_dir()
            );
        });
    }
}
