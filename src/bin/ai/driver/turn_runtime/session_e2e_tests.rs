//! End-to-end tests that drive a complete turn against a scripted model server.
//!
//! The server is a loopback HTTP endpoint that replays OpenAI-compatible SSE
//! chunks in a fixed order and records every request body. Pointing an
//! unregistered model at it exercises the real request builder, the real SSE
//! parser, tool execution, the iteration loop, and final-response gates without
//! touching a provider. Tests assert the request projection the runtime actually
//! sent, not just the final text.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{Value, json};

use super::run_turn;
use crate::ai::driver::turn_runtime::TurnOutcome;
use crate::ai::history::{SessionStore, build_message_arr_with_models};
use crate::ai::mcp::{McpClient, SharedMcpClient};
use crate::ai::types::App;

/// One scripted model response. Tool calls continue the loop; final text can
/// finish the turn once the runtime's final-response gates accept it.
enum ScriptedTurn {
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    Final {
        text: String,
    },
    ProviderControlLeak {
        id: String,
        name: String,
        arguments: Value,
    },
}

impl ScriptedTurn {
    fn tool_call(id: &str, name: &str, arguments: Value) -> Self {
        Self::ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
        }
    }

    fn final_text(text: &str) -> Self {
        Self::Final {
            text: text.to_string(),
        }
    }

    fn provider_control_leak(id: &str, name: &str, arguments: Value) -> Self {
        Self::ProviderControlLeak {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
        }
    }

    fn sse_body(&self) -> String {
        let mut body = String::new();
        match self {
            Self::ToolCall {
                id,
                name,
                arguments,
            } => {
                push_sse_event(
                    &mut body,
                    json!({"choices": [{"delta": {"tool_calls": [{
                        "index": 0,
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }]}}]}),
                );
                push_sse_event(
                    &mut body,
                    json!({"choices": [{"delta": {"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": arguments.to_string()},
                    }]}}]}),
                );
                push_sse_event(
                    &mut body,
                    json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
                );
            }
            Self::Final { text } => {
                push_sse_event(
                    &mut body,
                    json!({"choices": [{"delta": {"content": text}}]}),
                );
                push_sse_event(
                    &mut body,
                    json!({
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                    }),
                );
            }
            Self::ProviderControlLeak {
                id,
                name,
                arguments,
            } => {
                push_sse_event(
                    &mut body,
                    json!({"choices": [{"delta": {"tool_calls": [{
                        "index": 0,
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments.to_string()},
                    }]}}]}),
                );
                push_sse_event(
                    &mut body,
                    json!({"choices": [{"delta": {"reasoning_content":
                        "No reasoning available.<|end|>assistant to=read_file<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"file_path\">/tmp/never-read</rfl:parameter>"
                    }}]}),
                );
            }
        }
        body.push_str("data: [DONE]\n\n");
        body
    }
}

fn push_sse_event(body: &mut String, event: Value) {
    body.push_str("data: ");
    body.push_str(&event.to_string());
    body.push_str("\n\n");
}

/// Loopback server that replays one scripted SSE response per request.
struct ScriptedServer {
    endpoint: String,
    requests: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ScriptedServer {
    fn start(script: Vec<ScriptedTurn>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind scripted server");
        listener
            .set_nonblocking(true)
            .expect("nonblocking scripted server");
        let addr = listener.local_addr().expect("scripted server address");
        let endpoint = format!("http://{addr}/v1/chat/completions");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_requests = Arc::clone(&requests);
        let thread_stop = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let mut next_response = 0usize;
            while !thread_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = stream.set_nonblocking(false);
                        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                        let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
                        let request = read_http_request(&mut stream);
                        thread_requests.lock().unwrap().push(request);
                        let body = match script.get(next_response) {
                            Some(turn) => turn.sse_body(),
                            None => {
                                // A request beyond the script means the runtime performed an
                                // unexpected extra model call (for example an unplanned gate
                                // reopen). Record it and fail the request so the test does not
                                // silently accept the extra turn.
                                write_http_response(
                                    &mut stream,
                                    500,
                                    "application/json",
                                    r#"{"error":"unexpected scripted request"}"#,
                                );
                                next_response += 1;
                                continue;
                            }
                        };
                        next_response += 1;
                        write_http_response(&mut stream, 200, "text/event-stream", &body);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            endpoint,
            requests,
            stop,
            handle: Some(handle),
        }
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ScriptedServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn read_http_request(stream: &mut TcpStream) -> Value {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let read = match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(_) => return json!({"__error": "request read failed"}),
        };
        bytes.extend_from_slice(&buffer[..read]);
        let Some(header_end) = find_bytes(&bytes, b"\r\n\r\n") else {
            continue;
        };
        let header_text = String::from_utf8_lossy(&bytes[..header_end]);
        let content_length = header_text.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        });
        let Some(content_length) = content_length else {
            return json!({"__error": "request has no Content-Length"});
        };
        let body_start = header_end + 4;
        while bytes.len() < body_start + content_length {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => bytes.extend_from_slice(&buffer[..read]),
                Err(_) => return json!({"__error": "request body read failed"}),
            }
        }
        if bytes.len() < body_start + content_length {
            return json!({"__error": "request body ended early"});
        }
        return serde_json::from_slice(&bytes[body_start..body_start + content_length])
            .unwrap_or_else(|_| json!({"__error": "request body is not JSON"}));
    }
    json!({"__error": "request has no headers"})
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn write_http_response(stream: &mut TcpStream, status: u16, content_type: &str, body: &str) {
    let reason = if status == 200 { "OK" } else { "Error" };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

/// Restores `CONFIGW_PATH` after the fixture runs, so one test cannot change the
/// configuration source for the rest of the binary's tests.
struct ConfigPathGuard {
    previous: Option<std::ffi::OsString>,
}

impl ConfigPathGuard {
    fn set(path: &std::path::Path) -> Self {
        let previous = std::env::var_os("CONFIGW_PATH");
        // Callers hold `ENV_LOCK` while this guard changes the process-global
        // configuration path and refreshes configw.
        unsafe { std::env::set_var("CONFIGW_PATH", path) };
        crate::commonw::configw::refresh();
        Self { previous }
    }
}

impl Drop for ConfigPathGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();
    }
}

struct TempDirGuard(std::path::PathBuf);

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let path = std::env::temp_dir().join(format!("{prefix}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct TurnFixture {
    app: App,
    server: ScriptedServer,
    root: TempDirGuard,
    mcp: SharedMcpClient,
    model: String,
    _config_path: ConfigPathGuard,
}

impl TurnFixture {
    fn new(script: Vec<ScriptedTurn>) -> Self {
        let root = TempDirGuard::new("ai-scripted-e2e");
        std::fs::create_dir_all(root.path().join("history.sessions")).expect("create fixture root");
        // A nonexistent path makes configw fall back to defaults; it also keeps the
        // developer's real ~/.configW (and any ai.history.max_chars override) out of
        // the test.
        let config_path = root.path().join("configW");
        let config_guard = ConfigPathGuard::set(&config_path);

        let mut app = crate::ai::driver::tests::test_app("build");
        app.config.base_history_file = root.path().join("history.sqlite");
        app.config.history_file = root.path().join("history.sqlite");
        app.session_id = "scripted-e2e".to_string();
        // An explicit session keeps the one-shot turn persisted for history assertions.
        app.cli.session = Some(app.session_id.clone());
        let store = SessionStore::new(&app.config.history_file);
        app.session_history_file = store.session_history_file(&app.session_id);
        // Finalization otherwise asks the model for a session title, which would
        // consume an extra scripted response after the turn's real last request.
        store
            .write_session_title(&app.session_id, "scripted turn")
            .expect("seed scripted session title");
        app.config.endpoint = String::new();
        // Unregistered model: endpoint_for_model() then uses app.config.endpoint,
        // while request_model_name() keeps the unique name on the wire.
        let model = format!("scripted-{}", uuid::Uuid::new_v4());
        app.current_model = model.clone();
        app.client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("build scripted client");

        let server = ScriptedServer::start(script);
        app.config.endpoint = server.endpoint.clone();

        Self {
            app,
            server,
            root,
            mcp: Arc::new(Mutex::new(McpClient::new())),
            model,
            _config_path: config_guard,
        }
    }

    async fn run(&mut self, question: &str) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
        let mut question = question.to_string();
        let selectors = crate::ai::history::reply_reference::extract_selectors(&mut question)?;
        let future = run_turn(
            &mut self.app,
            &self.mcp,
            &[],
            0,
            question,
            String::new(),
            selectors,
            self.model.clone(),
            None,
            None,
            true,
            false,
        );
        crate::ai::driver::runtime_ctx::SUPPRESS_TERMINAL_OUTPUT
            .scope(true, future)
            .await
    }

    async fn run_loop(&mut self, question: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.run_loop_with_skill_loading(question, false).await
    }

    async fn run_loop_with_skill_loading(
        &mut self,
        question: &str,
        load_skills: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.app.cli.args = vec![question.to_string()];
        self.app.cli.model = Some(self.model.clone());
        self.app.cli.no_skills = !load_skills;
        self.app.shutdown.store(false, Ordering::Relaxed);
        let mut skills = Arc::new(Vec::new());
        let mut agents = Arc::new(Vec::new());
        let probe = crate::ai::driver::mcp_init::McpConfigProbe {
            config_path: self.root.path().join("mcp.json").display().to_string(),
            exists: false,
            server_count: 0,
        };
        let future =
            crate::ai::driver::run_loop(&mut self.app, &self.mcp, probe, &mut skills, &mut agents);
        crate::ai::driver::runtime_ctx::SUPPRESS_TERMINAL_OUTPUT
            .scope(
                true,
                crate::ai::driver::runtime_ctx::SUBAGENT_MEMORY_PATH
                    .scope(self.root.path().join("memory.jsonl"), future),
            )
            .await
    }

    fn history(&self) -> Vec<(crate::ai::history::Message, Option<String>)> {
        build_message_arr_with_models(&self.app.session_history_file)
            .expect("read scripted session history")
    }
}

fn request_messages(request: &Value) -> &[Value] {
    request["messages"]
        .as_array()
        .expect("request messages array")
}

fn text_content(message: &Value) -> &str {
    message["content"].as_str().unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn skills_pin_trailing_question_answers_and_survives_next_turn() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let question = "自动补全加上了吗？\n请检查 WebUI（包括手机端和电脑端）的 skills 系列自动补全。";
    let next_question = "再检查一下手机端。";
    let replies = [
        "会为 WebUI 的 skills 系列添加自动补全。",
        "手机端也会包含 skills 自动补全。",
    ];
    let mut fixture = TurnFixture::new(
        replies
            .iter()
            .map(|reply| ScriptedTurn::final_text(reply))
            .collect(),
    );
    let skills_dir = fixture.root.path().join("skills");
    std::fs::write(
        fixture.root.path().join("configW"),
        format!("ai.skills.dir={}\n", skills_dir.display()),
    )
    .expect("isolate the user skill directory");
    crate::commonw::configw::refresh();
    assert_eq!(crate::ai::skills::skills_dir(), skills_dir);
    let code_prompt = crate::ai::skills::builtin_skill_body("code").expect("builtin code skill");
    let project = fixture.root.path().to_path_buf();

    for (index, input) in [
        format!("/skills pin code\n {question} \n"),
        next_question.to_string(),
    ]
    .iter()
    .enumerate()
    {
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD
            .scope(
                project.clone(),
                tokio::time::timeout(
                    Duration::from_secs(20),
                    fixture.run_loop_with_skill_loading(input, true),
                ),
            )
            .await
            .expect("pinned-skill run_loop timed out")
            .expect("pinned-skill run_loop failed");
        assert_eq!(fixture.app.session_default_skills, ["code"]);
        assert!(fixture.app.forced_question.is_none());

        let requests = fixture.server.requests();
        assert_eq!(requests.len(), index + 1, "each input must produce a reply");
        let messages = request_messages(&requests[index]);
        assert!(text_content(&messages[0]).contains(code_prompt.trim()));
        let sent = messages
            .iter()
            .rev()
            .find(|message| message["role"] == "user")
            .expect("question sent to the model");
        let sent_question = crate::ai::history::value_to_string(&sent["content"]);
        let expected_question = [question, next_question][index];
        assert!(sent_question.contains(expected_question), "{sent_question}");
        assert!(!sent_question.contains("/skills pin code"));
        assert!(fixture.history().iter().any(|(message, _)| {
            message.role == "assistant"
                && crate::ai::history::value_to_string(&message.content) == replies[index]
        }));
    }

    let events =
        crate::ai::history::read_skill_activation_events_sqlite(&fixture.app.session_history_file)
            .expect("read pinned skill activation events");
    assert_eq!(
        events
            .iter()
            .filter(|event| event.source == "session-default"
                && event.injected_skill.as_deref() == Some("code")
                && event.outcome == "injected")
            .count(),
        2,
        "the pinned skill must be injected on both turns"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scripted_turn_executes_tool_round_and_keeps_request_prefix() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let root = std::env::temp_dir().join(format!("ai-scripted-sentinel-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).expect("create sentinel root");
    let sentinel_path = root.join("sentinel.txt");
    std::fs::write(&sentinel_path, "SCRIPTED_SENTINEL\n").expect("write sentinel");
    let sentinel = sentinel_path.to_string_lossy().to_string();

    let mut fixture = TurnFixture::new(vec![
        ScriptedTurn::tool_call("call_read_1", "read_file", json!({"file_path": sentinel})),
        ScriptedTurn::final_text("The sentinel file was read successfully."),
    ]);

    let outcome = tokio::time::timeout(
        Duration::from_secs(15),
        fixture.run("Read the test file and report the sentinel."),
    )
    .await
    .expect("scripted turn timed out")
    .expect("scripted turn failed");
    assert_eq!(outcome, TurnOutcome::Continue);

    let requests = fixture.server.requests();
    assert_eq!(
        requests.len(),
        2,
        "a tool-call round plus a final answer must produce exactly two model requests; got {}",
        requests.len()
    );
    assert_eq!(requests[0]["stream"], json!(true));

    let first = request_messages(&requests[0]);
    let second = request_messages(&requests[1]);
    assert_eq!(
        &second[..first.len()],
        first,
        "the second request must keep the first request's prefix byte-for-byte"
    );

    let tools = requests[0]["tools"]
        .as_array()
        .expect("first request must carry tool definitions");
    assert!(
        tools
            .iter()
            .any(|tool| tool["function"]["name"] == "read_file"),
        "read_file must be visible in the request tool schema"
    );

    let tool_result = second
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("second request must carry the tool result");
    assert!(
        text_content(tool_result).contains("SCRIPTED_SENTINEL"),
        "tool result must contain the file body: {tool_result}"
    );
    assert!(
        second.iter().any(|message| {
            message["role"] == "assistant"
                && message["tool_calls"][0]["function"]["name"] == "read_file"
        }),
        "second request must carry the assistant tool call"
    );

    let history = fixture.history();
    assert!(
        history.iter().any(|(message, _)| {
            message.role == "tool"
                && message
                    .content
                    .as_str()
                    .is_some_and(|content| content.contains("SCRIPTED_SENTINEL"))
        }),
        "the tool result must be persisted to canonical history: {history:#?}"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loop_detector_note_reaches_next_request_without_extra_gate_reopen() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let root = std::env::temp_dir().join(format!("ai-scripted-loop-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).expect("create loop root");
    let target_path = root.join("loop.txt");
    std::fs::write(&target_path, "LOOP_SENTINEL\n").expect("write loop target");
    let target = target_path.to_string_lossy().to_string();

    // Four byte-identical rounds trip TOOL_LOOP_SOFT_WINDOW. The next response is
    // the model's final answer; if the final gate spuriously reopened, the script
    // would need a sixth response and the request count/index assertions fail.
    let mut script: Vec<ScriptedTurn> = (0..4)
        .map(|_| {
            ScriptedTurn::tool_call("call_read_loop", "read_file", json!({"file_path": target}))
        })
        .collect();
    script.push(ScriptedTurn::final_text(
        "Stopped repeating the read after the loop notice.",
    ));

    let mut fixture = TurnFixture::new(script);
    let outcome = tokio::time::timeout(
        Duration::from_secs(20),
        fixture.run("Read the same file repeatedly."),
    )
    .await
    .expect("loop-detector turn timed out")
    .expect("loop-detector turn failed");
    assert_eq!(outcome, TurnOutcome::Continue);

    let requests = fixture.server.requests();
    assert_eq!(
        requests.len(),
        5,
        "the soft loop signal must inject one note and the next response must end the turn; \
         an extra request would mean a gate reopened unexpectedly"
    );

    let fifth = request_messages(&requests[4]);
    // Runtime notes use internal_note before normalization, which projects them
    // as system content; assert on the note body rather than its role.
    let saw_loop_note = fifth
        .iter()
        .any(|message| text_content(message).contains("[loop-detected]"));
    assert!(
        saw_loop_note,
        "the request after the soft loop signal must carry the loop-breaker note: {fifth:#?}"
    );

    let history = fixture.history();
    let tool_calls = history
        .iter()
        .flat_map(|(message, _)| message.tool_calls.iter().flatten())
        .filter(|call| call.function.name == "read_file")
        .count();
    assert_eq!(
        tool_calls, 4,
        "exactly four read_file rounds should execute before the scripted final answer"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_control_leak_discards_tools_and_retries_only_once() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    let leaked = || {
        ScriptedTurn::provider_control_leak(
            "call_must_not_run",
            "read_file",
            json!({"file_path": "/tmp/never-read"}),
        )
    };
    let mut fixture = TurnFixture::new(vec![leaked(), leaked()]);

    let outcome = tokio::time::timeout(
        Duration::from_secs(15),
        fixture.run("Inspect a file, but never execute malformed tool output."),
    )
    .await
    .expect("provider-control-leak turn timed out")
    .expect("provider-control-leak turn failed");
    assert_eq!(outcome, TurnOutcome::Continue);

    let requests = fixture.server.requests();
    assert_eq!(
        requests.len(),
        2,
        "the malformed response gets exactly one clean replay"
    );
    assert!(
        request_messages(&requests[1]).iter().any(|message| {
            text_content(message).contains("tool_followup:provider_control_leak")
        })
    );

    let history = fixture.history();
    assert!(
        history
            .iter()
            .all(|(message, _)| message.role != "tool" && message.tool_calls.is_none()),
        "no tool from either malformed response may execute or persist: {history:#?}"
    );
}
/// Number of equal leading message values shared by two request bodies,
/// used to check stability of the message prefix.
fn shared_message_prefix(previous: &[Value], next: &[Value]) -> usize {
    previous
        .iter()
        .zip(next)
        .take_while(|(left, right)| left == right)
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cross_turn_requests_extend_the_previous_request_prefix() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    // A project with a scoped instruction file next to the file the model reads.
    // The turn's effective cwd is pinned to the project root below.
    let project_dir = TempDirGuard::new("ai-scripted-scoped-project");
    let project = project_dir.path().to_path_buf();
    let nested = project.join("src/driver");
    std::fs::create_dir_all(project.join(".git")).expect("create project marker");
    std::fs::create_dir_all(&nested).expect("create nested dir");
    std::fs::write(project.join("AGENTS.md"), "Root project rules.\n").expect("root rules");
    std::fs::write(nested.join("AGENTS.md"), "SCOPED_DRIVER_RULES\n").expect("scoped rules");
    let target = nested.join("turn.rs");
    std::fs::write(&target, "// DRIVER_SOURCE\n").expect("target file");
    let read_args = json!({"file_path": target.to_string_lossy()});
    let mut fixture = TurnFixture::new(vec![
        ScriptedTurn::tool_call("call_turn1_read", "read_file", read_args.clone()),
        ScriptedTurn::final_text("Turn one done."),
        ScriptedTurn::tool_call("call_turn2_read", "read_file", read_args),
        ScriptedTurn::final_text("Turn two done."),
    ]);

    for question in [
        "Inspect `src/driver/turn.rs`.",
        "Inspect `src/driver/turn.rs` once more.",
    ] {
        let outcome = crate::ai::driver::runtime_ctx::SUBAGENT_CWD
            .scope(
                project.clone(),
                tokio::time::timeout(Duration::from_secs(20), fixture.run(question)),
            )
            .await
            .expect("scripted turn timed out")
            .expect("scripted turn failed");
        assert_eq!(outcome, TurnOutcome::Continue);
    }

    let requests = fixture.server.requests();
    assert_eq!(
        requests.len(),
        4,
        "two turns of one tool round plus a final answer each"
    );
    let messages: Vec<&[Value]> = requests.iter().map(|r| request_messages(r)).collect();

    // Every request after the first must carry the scoped rules the model already
    // followed: losing them at a turn boundary would change agent behavior.
    for (index, request) in messages.iter().enumerate().skip(1) {
        assert!(
            text_content(&request[0]).contains("SCOPED_DRIVER_RULES"),
            "request {index} dropped the scoped rules loaded earlier in the session"
        );
    }

    // Within a turn the first touch of a scoped directory necessarily rewrites
    // the system prompt once (the model must see the new rules). After that,
    // the next turn's first request must reuse everything through the previous
    // tool round byte for byte. The only exception is the last pre-request
    // working-memory note, which is request-only and never persisted.
    let shared: Vec<usize> = (1..messages.len())
        .map(|index| shared_message_prefix(messages[index - 1], messages[index]))
        .collect();
    let first_difference = |previous: &[Value], next: &[Value]| {
        let index = shared_message_prefix(previous, next);
        let render = |message: Option<&Value>| {
            message
                .map(|message| match message["content"].as_str() {
                    Some(text) => text.to_string(),
                    None => message.to_string(),
                })
                .unwrap_or_default()
        };
        let (left, right) = (render(previous.get(index)), render(next.get(index)));
        let split = left
            .char_indices()
            .zip(right.chars())
            .find(|((_, a), b)| a != b)
            .map(|((offset, _), _)| offset)
            .unwrap_or_else(|| left.len().min(right.len()));
        let window = |text: &str| {
            let start = text.floor_char_boundary(split.saturating_sub(200));
            let end = text.ceil_char_boundary((split + 400).min(text.len()));
            text[start..end].to_string()
        };
        format!(
            "first differing message #{index} at byte {split}\nprevious: …{}…\nnext: …{}…",
            window(&left),
            window(&right)
        )
    };
    assert_eq!(
        shared[1],
        messages[1].len() - 1,
        "the next turn must reuse everything through the previous tool round; shared {shared:?} of lengths {:?}\n{}",
        messages.iter().map(|m| m.len()).collect::<Vec<_>>(),
        first_difference(messages[1], messages[2])
    );
    assert_eq!(
        shared[2],
        messages[2].len(),
        "the tool-round request must extend the turn's first request: shared {shared:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reply_reference_request_keeps_user_boundary_and_replays_frozen_snapshot() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let source_text = "REPLY_REFERENCE_SOURCE: the committed answer is forty two.";
    let mut fixture = TurnFixture::new(vec![
        ScriptedTurn::final_text(source_text),
        ScriptedTurn::final_text("The referenced answer can be explained directly."),
        ScriptedTurn::final_text(
            "The quoted content remains available from the earlier user turn.",
        ),
    ]);
    fixture.run("What is the answer?").await.unwrap();
    let store = SessionStore::new(fixture.app.config.history_file.as_path());
    let stored = store
        .read_stored_messages(&fixture.app.session_id, 0)
        .unwrap();
    let source = stored
        .iter()
        .find(|row| {
            row.message.role == "assistant" && row.message.content.as_str() == Some(source_text)
        })
        .unwrap();
    let source_id = source.message_id;
    fixture
        .run(&format!(
            "@reply:m{source_id} @reply:last Explain this answer."
        ))
        .await
        .unwrap();
    fixture.run("Continue with the explanation.").await.unwrap();
    let requests = fixture.server.requests();
    assert_eq!(requests.len(), 3);
    let text = |value: &Value| crate::ai::history::value_to_string(value);
    let second = request_messages(&requests[1]);
    for message in second.iter().filter(|message| message["role"] == "system") {
        assert!(!text(&message["content"]).contains(source_text));
    }
    let prior_assistant = second
        .iter()
        .find(|message| message["role"] == "assistant")
        .unwrap();
    assert_eq!(prior_assistant["content"], source_text);
    let current_user = second
        .iter()
        .rev()
        .find(|message| message["role"] == "user")
        .unwrap();
    let quoted = text(&current_user["content"]);
    assert!(quoted.contains(source_text), "{quoted}");
    assert!(quoted.contains(&format!("[Referenced assistant reply: m{source_id}]")));
    assert!(quoted.contains("Explain this answer."));
    assert!(!current_user.to_string().contains("\"type\":\"reference\""));
    let history = store
        .read_stored_messages(&fixture.app.session_id, 0)
        .unwrap();
    let referenced_user = history
        .iter()
        .find(|row| {
            row.message
                .content
                .as_array()
                .is_some_and(|parts| parts.iter().any(|part| part["kind"] == "message"))
        })
        .unwrap();
    let parts = referenced_user.message.content.as_array().unwrap();
    let references = parts
        .iter()
        .filter(|part| part["type"] == "reference" && part["kind"] == "message")
        .collect::<Vec<_>>();
    assert_eq!(references.len(), 1);
    assert_eq!(references[0]["text"], source_text);
    assert_eq!(references[0]["source_message_id"], source_id);
    assert_eq!(parts.last().unwrap()["text"], "Explain this answer.");
    let replayed = request_messages(&requests[2])
        .iter()
        .find(|message| {
            message["role"] == "user"
                && text(&message["content"]).contains("[Referenced assistant reply:")
        })
        .unwrap();
    assert_eq!(replayed["content"], current_user["content"]);
    assert_eq!(
        history
            .iter()
            .find(|row| row.message_id == source_id)
            .unwrap()
            .message
            .content,
        json!(source_text)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reply_reference_deleted_identity_rejects_before_model_or_user_commit() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut fixture = TurnFixture::new(vec![ScriptedTurn::final_text("source reply")]);
    fixture.run("first question").await.unwrap();
    let conn = rusqlite::Connection::open(&fixture.app.session_history_file).unwrap();
    let id: i64 = conn
        .query_row(
            "SELECT id FROM messages WHERE role = 'assistant' ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    conn.execute("DELETE FROM messages WHERE id = ?1", [id])
        .unwrap();
    let before = fixture.history().len();
    let error = fixture
        .run(&format!("@reply:m{id} Explain it"))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("unavailable"));
    assert_eq!(fixture.server.requests().len(), 1);
    assert_eq!(fixture.history().len(), before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reply_reference_command_text_is_persisted_without_dispatch() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let commands = [
        "compact",
        "distill-session",
        "audit",
        "audit --deep",
        "audit inspect the quoted answer",
        "audit --deep inspect the quoted answer",
    ];
    let mut script = vec![ScriptedTurn::final_text("Frozen source answer.")];
    for _ in 0..commands.len() * 2 {
        script.push(ScriptedTurn::final_text(
            "The quoted answer remains user context.",
        ));
    }
    let mut fixture = TurnFixture::new(script);
    fixture.run("Seed the assistant reply.").await.unwrap();
    let store = SessionStore::new(fixture.app.config.history_file.as_path());
    let source_id = store
        .read_stored_messages(&fixture.app.session_id, 0)
        .unwrap()
        .into_iter()
        .find(|row| row.message.role == "assistant")
        .unwrap()
        .message_id;
    let mut expected_requests = 1;
    for prefix in ["/", ":"] {
        for command in commands {
            let question = format!("{prefix}{command}");
            assert_eq!(
                fixture
                    .run(&format!("@reply:m{source_id} {question}"))
                    .await
                    .unwrap(),
                TurnOutcome::Continue
            );
            expected_requests += 1;
            let requests = fixture.server.requests();
            assert_eq!(requests.len(), expected_requests, "{question}");
            let sent = request_messages(requests.last().unwrap())
                .iter()
                .rev()
                .find(|message| message["role"] == "user")
                .unwrap();
            let text = crate::ai::history::value_to_string(&sent["content"]);
            assert!(text.contains(&question), "{text}");
            assert!(text.contains("Frozen source answer."), "{text}");
            let history = fixture.history();
            let user = history
                .iter()
                .rev()
                .find(|(message, _)| message.role == "user")
                .unwrap();
            let parts = user.0.content.as_array().unwrap();
            assert_eq!(parts.last().unwrap()["text"], question);
            assert_eq!(parts[0]["source_message_id"], source_id);
            assert_eq!(parts[0]["text"], "Frozen source answer.");
            assert_eq!(
                history
                    .iter()
                    .filter(|(message, _)| message.role == "user")
                    .count(),
                expected_requests
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reply_reference_plain_compact_and_distill_still_dispatch() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut fixture = TurnFixture::new(Vec::new());
    // Validation and help are real command outcomes that need no auxiliary model.
    for prefix in ["/", ":"] {
        for command in ["compact unexpected", "distill-session --help"] {
            assert_eq!(
                fixture.run(&format!("{prefix}{command}")).await.unwrap(),
                TurnOutcome::Continue
            );
            assert!(fixture.server.requests().is_empty());
            assert!(fixture.history().is_empty());
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reply_reference_run_loop_preserves_command_text_across_dispatchers() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    // The memo dispatcher loops even in one-shot mode. Keep a broken guard from
    // blocking on stdin, and restore the process marker even if an assertion fails.
    struct ServeChildGuard(bool);
    impl Drop for ServeChildGuard {
        fn drop(&mut self) {
            crate::ai::background::set_serve_child_for_test(self.0);
        }
    }
    let _serve_child = ServeChildGuard(crate::ai::background::is_serve_child());
    crate::ai::background::set_serve_child_for_test(true);

    let commands = [
        "title quoted title",
        "skills help",
        "memo",
        "compact",
        "distill-session --help",
        "audit --deep inspect the quoted answer",
    ];
    // A distinct fixture per input keeps @reply:last bound to the seeded answer.
    for prefix in ["/", ":"] {
        for command in commands {
            let mut fixture = TurnFixture::new(vec![
                ScriptedTurn::final_text("Frozen loop source answer."),
                ScriptedTurn::final_text("The quoted answer remains user context."),
            ]);
            tokio::time::timeout(
                Duration::from_secs(20),
                fixture.run_loop("Seed the assistant reply."),
            )
            .await
            .expect("seed run_loop timed out")
            .unwrap();
            let store = SessionStore::new(&fixture.app.config.history_file);
            let source_id = store
                .read_stored_messages(&fixture.app.session_id, 0)
                .unwrap()
                .into_iter()
                .find(|row| row.message.role == "assistant")
                .unwrap()
                .message_id;
            let question = format!("{prefix}{command}");
            tokio::time::timeout(
                Duration::from_secs(20),
                fixture.run_loop(&format!("@reply:last {question}")),
            )
            .await
            .expect("referenced run_loop timed out")
            .unwrap();

            let requests = fixture.server.requests();
            assert_eq!(requests.len(), 2, "{question}");
            let sent = request_messages(&requests[1])
                .iter()
                .rev()
                .find(|message| message["role"] == "user")
                .unwrap();
            let text = crate::ai::history::value_to_string(&sent["content"]);
            assert!(text.contains(&question), "{text}");
            assert!(text.contains("Frozen loop source answer."), "{text}");
            assert!(
                text.contains(&format!("[Referenced assistant reply: m{source_id}]")),
                "{text}"
            );
            let history = fixture.history();
            assert_eq!(
                history
                    .iter()
                    .filter(|(message, _)| message.role == "user")
                    .count(),
                2,
                "{question}"
            );
            let user = history
                .iter()
                .rev()
                .find(|(message, _)| message.role == "user")
                .unwrap();
            let parts = user.0.content.as_array().unwrap();
            assert_eq!(parts.last().unwrap()["text"], question);
            assert_eq!(parts[0]["type"], "reference");
            assert_eq!(parts[0]["kind"], "message");
            assert_eq!(parts[0]["source_message_id"], source_id);
            assert_eq!(parts[0]["text"], "Frozen loop source answer.");
            assert_eq!(
                store
                    .read_session_title(&fixture.app.session_id)
                    .unwrap()
                    .as_deref(),
                Some("scripted turn")
            );
            assert!(
                !fixture.root.path().join("memory.jsonl").exists(),
                "{question}"
            );
        }
    }

    // Both pre-turn dispatchers still consume unreferenced commands. The title
    // mutation proves local dispatch; /skills help is owned by interactive dispatch.
    let mut fixture = TurnFixture::new(Vec::new());
    for question in ["/title plain title", "/skills help"] {
        tokio::time::timeout(Duration::from_secs(20), fixture.run_loop(question))
            .await
            .expect("plain command run_loop timed out")
            .unwrap();
        assert!(fixture.server.requests().is_empty(), "{question}");
        assert!(fixture.history().is_empty(), "{question}");
    }
    assert_eq!(
        SessionStore::new(&fixture.app.config.history_file)
            .read_session_title(&fixture.app.session_id)
            .unwrap()
            .as_deref(),
        Some("plain title")
    );
}
