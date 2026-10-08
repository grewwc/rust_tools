//! Session-distill agent tool: distill a session archive zip into knowledge.
//!
//! Kept separate from `knowledge_tools.rs` on purpose: knowledge_* tools are
//! small CRUD operations on the live store, while this is a file-driven
//! pipeline (zip in, up to N entries out) that only runs when the user
//! explicitly asked to distill an archive. It stays out of the resident
//! schema (`groups: ["knowledge"]` only) and loads via `enable_tools`.

use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::session_distill::{
    DEFAULT_DISTILL_LIMIT, run_distill_command,
};

pub(in crate::ai) async fn execute_with_app(app: &crate::ai::types::App, args: &Value) -> Result<String, String> {
    let archive = args["archive"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or("Missing 'archive'. Provide the session archive zip path.")?;
    let limit = args["limit"]
        .as_u64()
        .map(|v| v.clamp(1, 100) as usize)
        .unwrap_or(DEFAULT_DISTILL_LIMIT);
    let dry_run = args["dry_run"].as_bool().unwrap_or(false);

    let path = resolve_archive_path(archive)?;
    let report = run_distill_command(app, &path, limit, dry_run).await?;
    Ok(crate::ai::tools::service::session_distill::format_report(&report, &path))
}

fn resolve_archive_path(archive: &str) -> Result<std::path::PathBuf, String> {
    let path = std::path::PathBuf::from(archive);
    if path.is_absolute() {
        return Ok(path);
    }
    crate::ai::driver::runtime_ctx::effective_cwd()
        .map(|cwd| cwd.join(path))
        .map_err(|error| format!("Cannot resolve archive relative to effective cwd: {error}"))
}

fn execute_session_distill(_args: &Value) -> Result<String, String> {
    Err("session_distill requires the asynchronous driver execution context".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::driver::runtime_ctx::SUBAGENT_CWD;

    #[test]
    fn session_distill_relative_archive_uses_effective_cwd() {
        let cwd = std::env::temp_dir().join("distill-cwd-override");
        assert_ne!(cwd, std::env::current_dir().unwrap());
        SUBAGENT_CWD.sync_scope(cwd.clone(), || {
            assert_eq!(resolve_archive_path("archives/session.zip").unwrap(), cwd.join("archives/session.zip"));
        });
    }

    #[test]
    fn session_distill_absolute_archive_ignores_cwd_override() {
        let archive = std::env::temp_dir().join("session.zip");
        SUBAGENT_CWD.sync_scope(archive.join("other-cwd"), || {
            assert_eq!(resolve_archive_path(archive.to_str().unwrap()).unwrap(), archive);
        });
    }

    #[test]
    fn session_distill_relative_archive_without_override_uses_process_cwd() {
        assert_eq!(resolve_archive_path("session.zip").unwrap(), std::env::current_dir().unwrap().join("session.zip"));
    }
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "session_distill",
        description: "",

        execute: execute_session_distill,
    }
});
