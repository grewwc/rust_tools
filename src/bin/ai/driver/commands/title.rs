//! `/title` command: set or show the current session's title while
//! interacting with the agent in interactive mode.
//!
//! A title set through `/title` is persisted with the `User` origin, so the
//! background auto-generation never overwrites it (see
//! `should_generate_model_session_title`).

use std::error::Error;

use crate::ai::history::{SessionStore, SessionTitleOrigin};
use crate::ai::prompt::notify_session_title_updated;
use crate::ai::types::App;

/// Handle `/title [text]` (alias `:title`).
///
/// Bare `/title` prints the current title. `/title <text>` persists the given
/// text as the current session's title with the `User` origin and notifies the
/// foreground input editor so its topic line updates immediately. Returns
/// `Ok(true)` when the input was a `/title` command (even if it failed to
/// apply), `Ok(false)` otherwise.
pub fn try_handle_title_command(app: &mut App, input: &str) -> Result<bool, Box<dyn Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    let mut parts = normalized.split_whitespace();
    if parts.next() != Some("title") {
        return Ok(false);
    }
    let title = parts.collect::<Vec<_>>().join(" ");
    let store = SessionStore::new(app.config.history_file.as_path());

    if title.is_empty() {
        match store.read_session_title(&app.session_id) {
            Ok(Some(current)) => println!("{current}"),
            Ok(None) => println!("no title set for session '{}'", app.session_id),
            Err(err) => eprintln!("[title] failed to read session title: {err}"),
        }
        return Ok(true);
    }

    match store.write_session_title_with_origin(&app.session_id, &title, SessionTitleOrigin::User) {
        Ok(()) => {
            notify_session_title_updated(&app.session_id, &title);
            println!("Session title set: {title}");
        }
        Err(err) => eprintln!("[title] failed to set session title: {err}"),
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    fn test_app(root: &std::path::Path) -> App {
        let history_file = root.join("history.sqlite");
        let session_store = SessionStore::new(history_file.as_path());
        let session_id = "sess-title".to_string();
        App {
            cli: crate::ai::cli::ParsedCli::default(),
            scoped_preflight_required: Vec::new(),
            config: crate::ai::types::AppConfig {
                api_key: String::new(),
                base_history_file: history_file.clone(),
                history_file: history_file.clone(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 8000,
                history_keep_last: 10,
                history_summary_max_chars: 4000,
                intent_model: None,
            },
            session_id: session_id.clone(),
            session_history_file: session_store.session_history_file(&session_id),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: "test".to_string(),
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
            observers: Vec::new(),
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
        }
    }

    fn test_history_root() -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("rust_tools-title-tests-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn non_title_input_is_not_handled() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        for input in ["hello world", "/help", "/sessions", "/titlex foo", ":"] {
            assert!(
                !try_handle_title_command(&mut app, input).unwrap(),
                "input '{input}' must not be consumed as a title command"
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn title_persists_with_user_origin_and_overwrites() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());
        let id = app.session_id.clone();

        // No title yet.
        assert_eq!(store.read_session_title(&id).unwrap(), None);

        // Setting a title persists it with the User origin.
        assert!(try_handle_title_command(&mut app, "/title 修复登录 bug").unwrap());
        let saved = store.read_session_title_with_origin(&id).unwrap().unwrap();
        assert_eq!(saved.text, "修复登录 bug");
        assert_eq!(saved.origin, SessionTitleOrigin::User);

        // A later `/title` overwrites the previous one.
        assert!(try_handle_title_command(&mut app, "/title 排查性能问题").unwrap());
        assert_eq!(
            store.read_session_title_with_origin(&id).unwrap().unwrap().text,
            "排查性能问题"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn colon_alias_and_whitespace_collapsing() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());
        let id = app.session_id.clone();

        assert!(try_handle_title_command(&mut app, ":title   总结   本周  工作").unwrap());
        assert_eq!(store.read_session_title(&id).unwrap().unwrap(), "总结 本周 工作");

        let _ = std::fs::remove_dir_all(root);
    }
}
