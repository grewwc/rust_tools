//! The isolated serve child's single, tool-free turn.

use std::io;

use serde_json::{Value, json};

use crate::ai::{
    history::{Message, append_history_messages_for_model},
    request,
    serve::commit_message::MAX_MESSAGE_BYTES,
    types::App,
};

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    Finished,
    UnexpectedToolCall,
}

pub(super) async fn run(app: &App, prompt: String) -> io::Result<Outcome> {
    // Go straight to the text-only JSON transport. The prompt never passes
    // through CLI expansions, history projection, MCP, or the tool loop.
    let messages = [json!({"role": "user", "content": &prompt})];
    let response = request::do_request_json(app, &app.current_model, &messages, false, true)
        .await
        .map_err(|_| io::Error::other("Commit-message model request failed."))?;

    let unexpected_tool_call = contains_tool_call(&response);
    let mut rows = vec![text_message("user", prompt)];
    if !unexpected_tool_call
        && let Some(text) = assistant_text(&response)
        && text.trim().len() <= MAX_MESSAGE_BYTES
    {
        rows.push(text_message("assistant", text.trim().to_owned()));
    }
    append_history_messages_for_model(&app.session_history_file, &rows, &app.current_model)
        .map_err(|_| io::Error::other("Could not save the commit-message session."))?;
    Ok(if unexpected_tool_call {
        Outcome::UnexpectedToolCall
    } else {
        Outcome::Finished
    })
}

fn text_message(role: &str, text: String) -> Message {
    Message {
        role: role.to_owned(),
        content: Value::String(text),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn contains_tool_call(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().any(contains_tool_call),
        Value::Object(object) => {
            let kind = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            matches!(kind, "tool_use" | "server_tool_use" | "tool_result")
                || kind.ends_with("_call")
                || matches!(
                    object.get("role").and_then(Value::as_str),
                    Some("tool" | "function")
                )
                || ["finish_reason", "stop_reason"].iter().any(|field| {
                    matches!(
                        object.get(*field).and_then(Value::as_str),
                        Some("tool_calls" | "tool_use" | "function_call")
                    )
                })
                || ["tool_calls", "tool_call", "function_call"]
                    .iter()
                    .any(|field| {
                        object.get(*field).is_some_and(|calls| {
                            !calls.is_null() && !calls.as_array().is_some_and(Vec::is_empty)
                        })
                    })
                || ["choices", "output", "message", "content"]
                    .iter()
                    .any(|field| object.get(*field).is_some_and(contains_tool_call))
        }
        _ => false,
    }
}

fn assistant_text(response: &Value) -> Option<String> {
    if let Some(choices) = response
        .get("choices")
        .or_else(|| response.get("output")?.get("choices"))
    {
        let [choice] = choices.as_array()?.as_slice() else {
            return None;
        };
        if matches!(
            choice.get("finish_reason").and_then(Value::as_str),
            Some("length" | "content_filter")
        ) {
            return None;
        }
        return message_text(choice.get("message")?);
    }
    if response.get("content").is_some() {
        if matches!(
            response.get("stop_reason").and_then(Value::as_str),
            Some("max_tokens" | "refusal")
        ) {
            return None;
        }
        return message_text(response);
    }
    // Responses can carry reasoning items next to the single assistant
    // message, or expose the assembled text as a shortcut.
    if response
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| status != "completed")
    {
        return None;
    }
    if let Some(output) = response.get("output").and_then(Value::as_array) {
        let mut messages = output
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"));
        if let Some(message) = messages.next() {
            return if messages.next().is_none() {
                message_text(message)
            } else {
                None
            };
        }
    }
    response
        .get("output_text")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn message_text(message: &Value) -> Option<String> {
    if message
        .get("role")
        .and_then(Value::as_str)
        .is_some_and(|role| role != "assistant")
    {
        return None;
    }
    let content = message.get("content")?;
    if let Some(text) = content.as_str() {
        return Some(text.to_owned());
    }
    let mut text = String::new();
    for part in content.as_array()? {
        match part.get("type").and_then(Value::as_str) {
            Some("text" | "output_text") | None => text.push_str(part.get("text")?.as_str()?),
            Some("thinking" | "redacted_thinking") => {}
            _ => return None,
        }
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::history::SessionStore;
    use axum::{Json, Router, routing::post};
    use std::{
        path::PathBuf,
        sync::{Arc, Mutex},
        time::Duration,
    };

    struct TestConfig {
        directory: PathBuf,
        old_config: Option<std::ffi::OsString>,
    }

    impl TestConfig {
        fn new() -> Self {
            let directory = std::env::temp_dir()
                .join(format!("a-commit-message-test-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&directory).unwrap();
            let old_config = std::env::var_os("CONFIGW_PATH");
            // The caller holds ENV_LOCK for this fixture's whole lifetime.
            unsafe { std::env::set_var("CONFIGW_PATH", directory.join("configW")) };
            crate::commonw::configw::refresh();
            Self {
                directory,
                old_config,
            }
        }
    }

    impl Drop for TestConfig {
        fn drop(&mut self) {
            match self.old_config.take() {
                Some(config) => unsafe { std::env::set_var("CONFIGW_PATH", config) },
                None => unsafe { std::env::remove_var("CONFIGW_PATH") },
            }
            crate::commonw::configw::refresh();
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    #[tokio::test]
    async fn commit_message_child_sends_literal_text_without_tools_and_persists_reply() {
        let _env = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let fixture = TestConfig::new();
        let mut app = crate::ai::driver::tests::test_app("build");
        let history_file = fixture.directory.join("history.sqlite");
        let store = SessionStore::new(&history_file);
        app.session_id = "commit-message-test".into();
        app.session_history_file = store.session_history_file(&app.session_id);
        app.config.history_file = history_file;
        app.config.api_key = "test-key".into();
        app.current_model = format!("commit-message-test-{}", uuid::Uuid::new_v4());
        app.client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();

        let seen = Arc::new(Mutex::new(Vec::new()));
        let observed = seen.clone();
        let router = Router::new().route("/v1/chat/completions", post(move |Json(body): Json<Value>| {
            let observed = observed.clone();
            async move {
                let mut requests = observed.lock().unwrap();
                requests.push(body);
                if requests.len() == 1 {
                    Json(json!({"choices": [{"message": {"role": "assistant", "content": "fix: keep input literal"}, "finish_reason": "stop"}]}))
                } else {
                    Json(json!({"choices": [{"message": {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "execute_command", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}))
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        app.config.endpoint = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let prompt = "Summarize this patch.\n\n+ @reply:last @skills:build @private.txt\n  - literal spacing\n";
        assert_eq!(
            run(&app, prompt.to_owned()).await.unwrap(),
            Outcome::Finished
        );
        let rows = store.read_stored_messages(&app.session_id, 0).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].message.role, "user");
        assert_eq!(rows[0].message.content, prompt);
        assert_eq!(rows[1].message.role, "assistant");
        assert_eq!(rows[1].message.content, "fix: keep input literal");

        app.session_id = "commit-message-tool-rejection".into();
        app.session_history_file = store.session_history_file(&app.session_id);
        assert_eq!(
            run(&app, prompt.to_owned()).await.unwrap(),
            Outcome::UnexpectedToolCall
        );
        let rejected = store.read_stored_messages(&app.session_id, 0).unwrap();
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].message.role, "user");
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 2);
        for request in &*requests {
            assert_eq!(
                request["messages"],
                json!([{"role": "user", "content": prompt}])
            );
            assert!(request.get("tools").is_none());
            assert!(request.get("tool_choice").is_none());
            assert!(request.get("enable_search").is_none());
        }
        server.abort();
    }

    #[test]
    fn commit_message_response_requires_one_complete_reply() {
        assert_eq!(assistant_text(&json!({"role": "assistant", "content": [{"type": "text", "text": "fix: use messages"}]})).as_deref(), Some("fix: use messages"));
        assert_eq!(assistant_text(&json!({"status": "completed", "output": [{"type": "reasoning"}, {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "fix: use responses"}]}]})).as_deref(), Some("fix: use responses"));
        assert!(
            assistant_text(&json!({"status": "incomplete", "output_text": "fix: incomplete"}))
                .is_none()
        );
        assert!(assistant_text(&json!({"choices": [{"message": {"content": "feat: first"}}, {"message": {"content": "feat: second"}}]})).is_none());
        assert!(contains_tool_call(
            &json!({"output": [{"type": "web_search_call"}]})
        ));
        assert!(contains_tool_call(
            &json!({"content": [{"type": "tool_use"}]})
        ));
    }
}
