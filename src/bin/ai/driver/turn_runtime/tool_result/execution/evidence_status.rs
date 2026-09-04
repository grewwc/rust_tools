//! Evidence-status annotation for tool results.
//!
//! A `role=tool` result tells the model *where* output came from (which tool,
//! which call id) but not *what epistemic status* the content has: it does not
//! distinguish live current state from historical or reference data. When the
//! runtime can determine that status from the tool arguments and runtime state
//! alone, it prepends a `[reference: ...]` marker to the model-visible result,
//! so the model does not mistake past state for current fact.
//!
//! The rules are generic — driven by the canonical session-storage path layout,
//! the stale patch-target ledger, and version-control history subcommands —
//! rather than hardcoded to one tool or file type. The markers only annotate the
//! projected model view (`PreparedToolResult::content_for_model`); canonical
//! history keeps the raw provider content untouched.

use super::*;

/// Marker for content read from this agent's own stored session data: session
/// DBs, overflow archives, folded tool groups, checkpoints, or the history
/// replay. Reading them back is legitimate, but the content is a snapshot of
/// the past, not the live conversation or the current filesystem.
pub(in crate::ai::driver::turn_runtime) const SESSION_HISTORY_MARKER: &str = "[reference: session-history - content read from this agent's own stored session data (a session DB, archive, or checkpoint file); it reflects a past state, not the live conversation or current filesystem]";

/// Marker for content read from a file the runtime knows is stale: a patch
/// target whose last apply_patch attempt failed and has not been re-read since.
/// The on-disk truth may differ from earlier tool results.
pub(in crate::ai::driver::turn_runtime) const STALE_FILE_MARKER: &str = "[reference: stale-file - this file is a known-stale patch target; on-disk content may differ from earlier tool results. Re-read before treating it as current.]";

/// Marker for version-control history subcommands (`git log` / `git show` /
/// `git blame`): the output is a point-in-time snapshot of past commits or
/// revisions, not the current working tree.
pub(in crate::ai::driver::turn_runtime) const GIT_HISTORY_MARKER: &str = "[reference: git-history - this output describes past commits or revisions, not the current working tree]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum EvidenceStatus {
    SessionHistory,
    StaleFile,
    GitHistory,
}

/// Classify the epistemic status of a tool result from the tool name, its
/// arguments, and runtime state. `None` means the runtime cannot determine a
/// reference status and the result is left unannotated.
pub(in crate::ai::driver::turn_runtime) fn classify_evidence_status(
    app: &App,
    tool_call: &ToolCall,
    content: &str,
) -> Option<EvidenceStatus> {
    if content.trim_start().starts_with("Error:") || content.trim_start().starts_with("Failed:") {
        // Errors carry no usable reference content; a marker would only add noise.
        return None;
    }
    let tool_name = tool_call.function.name.as_str();
    let arguments = tool_call.function.arguments.as_str();

    // 1. Session storage. The canonical layout is `<parent>/<file-stem>.sessions`
    //    (default `~/.history_file.sessions`) with per-session `<id>.sqlite` and
    //    `<id>.assets/` below it, so any path component ending in `.sessions`
    //    marks the content as session storage for some history file — including
    //    this agent's own. This covers session DBs, overflow archives, folded
    //    tool groups, and checkpoints read through any tool.
    if arguments_reference_session_storage(arguments) {
        return Some(EvidenceStatus::SessionHistory);
    }

    // 2. Stale file. A read of a known-stale patch target may not reflect the
    //    on-disk truth. Both sides are normalized through FileStore, so relative
    //    / absolute / `~` spellings cannot bypass the match.
    if tool_name == "read_file"
        && file_tool_target_path(tool_call)
            .is_some_and(|path| app.stale_patch_targets.contains(&path))
    {
        return Some(EvidenceStatus::StaleFile);
    }

    // 3. Git history subcommands produce point-in-time snapshots of the past.
    if matches!(
        tool_name,
        "execute_command" | "run_command" | "shell" | "bash"
    ) && command_from_arguments(arguments)
        .is_some_and(|command| is_git_history_command(&command))
    {
        return Some(EvidenceStatus::GitHistory);
    }

    None
}

/// Prepend the reference marker for `tool_call`'s result to the model-visible
/// content, when the runtime can classify it. Idempotent: an already-marked
/// content is left untouched.
pub(in crate::ai::driver::turn_runtime) fn annotate_tool_result_evidence_status(
    app: &App,
    tool_call: &ToolCall,
    content: &str,
    prepared: &mut PreparedToolResult,
) {
    let Some(status) = classify_evidence_status(app, tool_call, content) else {
        return;
    };
    let marker = match status {
        EvidenceStatus::SessionHistory => SESSION_HISTORY_MARKER,
        EvidenceStatus::StaleFile => STALE_FILE_MARKER,
        EvidenceStatus::GitHistory => GIT_HISTORY_MARKER,
    };
    if prepared.content_for_model.starts_with(marker) {
        return;
    }
    prepared.content_for_model = format!("{marker}\n{}", prepared.content_for_model);
}

/// Whether any string value inside the tool-call arguments references a path
/// under session storage. Works for every tool: `file_path` (read_file,
/// write_file, apply_patch), `command` (execute_command), or any other string
/// field carrying a path.
fn arguments_reference_session_storage(arguments: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return false;
    };
    let mut references = false;
    visit_string_values(&value, &mut |text| {
        if is_session_storage_path_text(text) {
            references = true;
        }
    });
    references
}

fn visit_string_values(value: &serde_json::Value, visit: &mut dyn FnMut(&str)) {
    match value {
        serde_json::Value::String(text) => visit(text),
        serde_json::Value::Array(items) => {
            for item in items {
                visit_string_values(item, visit);
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values() {
                visit_string_values(value, visit);
            }
        }
        _ => {}
    }
}

/// A text references session storage when it contains a path component ending
/// in `.sessions`. The component boundary (split on `/`) keeps quotes and
/// grep-style search terms (`grep foo.sessions src/`) from matching, while
/// `~/.history_file.sessions/...` and any other `<stem>.sessions/...` layout do.
fn is_session_storage_path_text(text: &str) -> bool {
    text.replace('\\', "/")
        .split('/')
        .any(|component| !component.is_empty() && component.ends_with(".sessions"))
}

fn command_from_arguments(arguments: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(arguments).ok()?;
    value
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// Version-control subcommands whose output is a point-in-time snapshot of the
/// past rather than the current working tree. `git status` and bare `git diff`
/// describe the live tree and are intentionally excluded.
fn is_git_history_command(command: &str) -> bool {
    let command = command.trim();
    // Tolerate a leading directory change: `cd <dir> && git log ...`.
    let command = command
        .strip_prefix("cd ")
        .and_then(|rest| rest.split_once("&&"))
        .map_or(command, |(_, rest)| rest.trim());
    let mut words = command.split_whitespace();
    matches!(
        (words.next(), words.next()),
        (Some(cmd), Some(subcommand)) if cmd == "git" && matches!(subcommand, "log" | "show" | "blame")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::cli::ParsedCli;
    use crate::ai::{
        history::SessionStore,
        types::{App, AppConfig, FunctionCall, ToolCall},
    };
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    fn tool_call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn test_app(history_file: PathBuf) -> App {
        let mut app = App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                base_history_file: history_file.clone(),
                history_file: history_file.clone(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 24_000,
                history_keep_last: 256,
                history_summary_max_chars: 4_000,
                intent_model: None,
            },
            session_id: "test".to_string(),
            session_history_file: history_file.clone(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skills: Vec::new(),
            forced_skill_source: None,
            pending_skill_continuation: None,
            forced_question: None,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            prompt_editor: None,
            agent_context: None,
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(
                crate::ai::driver::thinking::ThinkingOrchestrator::new(),
            )],
            last_known_prompt_tokens: None,
            last_known_cached_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
            last_turn_interrupted: false,
            prune_marks: Default::default(),
            turn_reasoning_items: Default::default(),
            stale_patch_targets: Default::default(),
            tool_middlewares: Vec::new(),
            llm_middlewares: Vec::new(),
            hooks: Default::default(),
        };
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        app.session_history_file = store.session_history_file(&app.session_id);
        std::fs::write(&app.session_history_file, b"test").unwrap();
        app
    }

    #[test]
    fn classifies_session_history_from_read_file_path() {
        let app = test_app(std::env::temp_dir().join("evidence-status-read.sqlite"));
        let call = tool_call(
            "c1",
            "read_file",
            json!({ "file_path": "/home/u/.history_file.sessions/abc-123.sqlite" }),
        );
        assert_eq!(
            classify_evidence_status(&app, &call, "content"),
            Some(EvidenceStatus::SessionHistory)
        );
    }

    #[test]
    fn classifies_session_history_from_execute_command_sqlite() {
        let app = test_app(std::env::temp_dir().join("evidence-status-sqlite3.sqlite"));
        let call = tool_call(
            "c2",
            "execute_command",
            json!({
                "command": "sqlite3 /home/u/.history_file.sessions/abc-123.sqlite \"SELECT content FROM messages\"",
                "cwd": "/tmp"
            }),
        );
        assert_eq!(
            classify_evidence_status(&app, &call, "row content"),
            Some(EvidenceStatus::SessionHistory)
        );
    }

    #[test]
    fn classifies_session_history_from_archive_path() {
        let app = test_app(std::env::temp_dir().join("evidence-status-archive.sqlite"));
        let call = tool_call(
            "c3",
            "read_file",
            json!({ "file_path": "/home/u/.history_file.sessions/abc-123.assets/folded-tool-groups/xyz.md" }),
        );
        assert_eq!(
            classify_evidence_status(&app, &call, "stub content"),
            Some(EvidenceStatus::SessionHistory)
        );
    }

    #[test]
    fn classifies_stale_file_read() {
        let mut app = test_app(std::env::temp_dir().join("evidence-status-stale.sqlite"));
        let path = "/data00/rust_tools/src/foo.rs";
        let call = tool_call("c4", "read_file", json!({ "file_path": path }));
        let normalized = file_tool_target_path(&call).unwrap();
        app.stale_patch_targets.insert(normalized);
        assert_eq!(
            classify_evidence_status(&app, &call, "old content"),
            Some(EvidenceStatus::StaleFile)
        );
    }

    #[test]
    fn classifies_git_history_command() {
        let app = test_app(std::env::temp_dir().join("evidence-status-git.sqlite"));
        for command in [
            "git log --oneline -5",
            "cd /tmp && git show HEAD",
            "git blame src/x.rs",
        ] {
            let call = tool_call("c5", "execute_command", json!({ "command": command }));
            assert_eq!(
                classify_evidence_status(&app, &call, "commit output"),
                Some(EvidenceStatus::GitHistory),
                "command: {command}"
            );
        }
    }

    #[test]
    fn does_not_classify_live_project_reads() {
        let app = test_app(std::env::temp_dir().join("evidence-status-live.sqlite"));
        let read = tool_call("c6", "read_file", json!({ "file_path": "src/main.rs" }));
        assert_eq!(classify_evidence_status(&app, &read, "source"), None);
        // `git status` describes the live working tree, not a past snapshot.
        let status = tool_call("c7", "execute_command", json!({ "command": "git status" }));
        assert_eq!(classify_evidence_status(&app, &status, "clean"), None);
        // grep search terms must not trip the session-storage path rule.
        let grep = tool_call(
            "c8",
            "execute_command",
            json!({ "command": "grep -rn foo.sessions src/" }),
        );
        assert_eq!(classify_evidence_status(&app, &grep, "line"), None);
    }

    #[test]
    fn does_not_classify_error_results() {
        let app = test_app(std::env::temp_dir().join("evidence-status-error.sqlite"));
        let call = tool_call(
            "c9",
            "read_file",
            json!({ "file_path": "/home/u/.history_file.sessions/abc-123.sqlite" }),
        );
        assert_eq!(
            classify_evidence_status(&app, &call, "Error: file not found"),
            None
        );
    }

    #[test]
    fn annotate_prepends_session_marker() {
        let app = test_app(std::env::temp_dir().join("evidence-status-annotate.sqlite"));
        let call = tool_call(
            "c10",
            "read_file",
            json!({ "file_path": "/home/u/.history_file.sessions/abc-123.sqlite" }),
        );
        let mut prepared = PreparedToolResult {
            content_for_model: "row content".to_string(),
            content_for_terminal: String::new(),
        };
        annotate_tool_result_evidence_status(&app, &call, "row content", &mut prepared);
        assert!(
            prepared
                .content_for_model
                .starts_with(SESSION_HISTORY_MARKER),
            "got: {}",
            prepared.content_for_model
        );
        assert!(prepared.content_for_model.ends_with("row content"));
    }

    #[test]
    fn annotate_is_idempotent() {
        let app = test_app(std::env::temp_dir().join("evidence-status-idempotent.sqlite"));
        let call = tool_call(
            "c11",
            "read_file",
            json!({ "file_path": "/home/u/.history_file.sessions/abc-123.sqlite" }),
        );
        let mut prepared = PreparedToolResult {
            content_for_model: format!("{SESSION_HISTORY_MARKER}\nrow content"),
            content_for_terminal: String::new(),
        };
        annotate_tool_result_evidence_status(&app, &call, "row content", &mut prepared);
        assert_eq!(
            prepared.content_for_model,
            format!("{SESSION_HISTORY_MARKER}\nrow content")
        );
    }
}
