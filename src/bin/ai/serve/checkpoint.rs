//! Remote checkpoint routes used by `serve-chat`.

use super::*;
use crate::ai::driver::commands::{
    SERVE_CHECKPOINT_CONFIRM_ENV, SERVE_CHECKPOINT_CONFIRM_MARKER, SERVE_CHECKPOINT_CONFIRM_PROBE,
    SERVE_CHECKPOINT_CONFIRM_YES,
};
use crate::ai::history::CheckpointStore;

#[derive(Debug, Deserialize)]
pub(crate) struct CheckpointCommandReq {
    pub(crate) command: String,
    #[serde(default)]
    pub(crate) confirmed: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct CheckpointListResp {
    pub(crate) checkpoints: Vec<String>,
    pub(crate) branches: Vec<String>,
    pub(crate) checkpoint_status: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CheckpointCommandResp {
    pub(crate) output: String,
    pub(crate) confirmation: Option<String>,
    pub(crate) checkpoints: Vec<String>,
    pub(crate) branches: Vec<String>,
    pub(crate) checkpoint_status: Option<String>,
}

fn checkpoint_names(state: &ServeState, id: &str) -> Result<Vec<String>, String> {
    CheckpointStore::new(state.history_file.as_path(), id)
        .list()
        .map(|items| items.into_iter().map(|item| item.name).collect())
        .map_err(|err| err.to_string())
}

fn branch_names(state: &ServeState, id: &str) -> Result<Vec<String>, String> {
    CheckpointStore::new(state.history_file.as_path(), id)
        .list_branches()
        .map(|items| items.into_iter().map(|item| item.name).collect())
        .map_err(|err| err.to_string())
}

fn current_checkpoint_status(state: &ServeState, id: &str) -> Result<Option<String>, String> {
    CheckpointStore::new(state.history_file.as_path(), id)
        .current_status_label()
        .map_err(|err| err.to_string())
}

fn validate_checkpoint_command(command: &str) -> Result<(), String> {
    let first = command.split_whitespace().next().unwrap_or_default();
    if matches!(first, "/git" | ":git") {
        Ok(())
    } else {
        Err("expected a /git command".to_string())
    }
}

fn run_checkpoint_child(
    id: &str,
    command: &str,
    confirmed: bool,
    cwd: &std::path::Path,
) -> std::io::Result<std::process::Output> {
    let mut child = std::process::Command::new(runtime_exe()?);
    child
        .arg("--session")
        .arg(id)
        .arg(command)
        // The checkpoint runs in the session's virtual cwd, exactly like
        // every turn child, so cwd-relative reads resolve per session
        // rather than against the daemon's directory.
        .current_dir(cwd)
        .env(background::SERVE_CHILD_ENV, "1")
        .env(
            SERVE_CHECKPOINT_CONFIRM_ENV,
            if confirmed {
                SERVE_CHECKPOINT_CONFIRM_YES
            } else {
                SERVE_CHECKPOINT_CONFIRM_PROBE
            },
        )
        .output()
}

fn split_checkpoint_confirmation(stdout: &str) -> (String, Option<String>) {
    let mut output = Vec::new();
    let mut confirmation = None;
    for line in stdout.lines() {
        if let Some(prompt) = line.strip_prefix(SERVE_CHECKPOINT_CONFIRM_MARKER) {
            confirmation = Some(prompt.to_string());
        } else if confirmation.is_none()
            || !matches!(
                line,
                "Canceled: nothing was changed." | "Checkpoint clear canceled by user."
            )
        {
            output.push(line);
        }
    }
    let output = if output.is_empty() {
        String::new()
    } else {
        format!("{}\n", output.join("\n"))
    };
    (output, confirmation)
}

pub(crate) async fn list_checkpoints(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    if !SessionStore::new(state.history_file.as_path())
        .session_history_file(&id)
        .exists()
    {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "session not found"})),
        )
            .into_response();
    }
    match checkpoint_names(&state, &id) {
        Ok(checkpoints) => Json(CheckpointListResp {
            checkpoints,
            branches: branch_names(&state, &id).unwrap_or_default(),
            checkpoint_status: current_checkpoint_status(&state, &id).unwrap_or_default(),
        })
        .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err})),
        )
            .into_response(),
    }
}

pub(crate) async fn post_checkpoint_command(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<CheckpointCommandReq>,
) -> impl IntoResponse {
    if let Err(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    if let Err(err) = validate_checkpoint_command(&req.command) {
        return bad_request(err).into_response();
    }

    let store = SessionStore::new(state.history_file.as_path());
    if !store.session_history_file(&id).exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "session not found"})),
        )
            .into_response();
    }

    let lock = session_lock(&state, &id).await;
    let _guard = lock.lock().await;
    let session_cwd = session_cwd(&state, &id);
    let result = match run_checkpoint_child(&id, req.command.trim(), req.confirmed, &session_cwd)
    {
        Ok(result) => result,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("failed to spawn checkpoint helper: {err}")
                })),
            )
                .into_response();
        }
    };
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    if !result.status.success() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                serde_json::json!({"error": format!("checkpoint command failed: {stderr}{stdout}")}),
            ),
        )
            .into_response();
    }
    let (mut output, confirmation) = split_checkpoint_confirmation(&stdout);
    if !stderr.is_empty() {
        output.push_str(&stderr);
    }
    let checkpoints = checkpoint_names(&state, &id).unwrap_or_default();
    let branches = branch_names(&state, &id).unwrap_or_default();
    Json(CheckpointCommandResp {
        output,
        confirmation,
        checkpoints,
        branches,
        checkpoint_status: current_checkpoint_status(&state, &id).unwrap_or_default(),
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirmation_marker_is_removed_from_remote_output() {
        let text = format!(
            "{SERVE_CHECKPOINT_CONFIRM_MARKER}Discard history? (y/n): \nCanceled: nothing was changed.\n"
        );
        let (output, confirmation) = split_checkpoint_confirmation(&text);
        assert!(output.is_empty());
        assert_eq!(confirmation.as_deref(), Some("Discard history? (y/n): "));
    }

    #[test]
    fn only_checkpoint_commands_are_accepted() {
        assert!(validate_checkpoint_command("/git tree").is_ok());
        assert!(validate_checkpoint_command(":git list").is_ok());
        assert!(validate_checkpoint_command("/cp tree").is_err());
        assert!(validate_checkpoint_command("/history").is_err());
    }
}
