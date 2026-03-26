use std::{
    io::Write,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use clap::Parser;
use colored::Colorize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::ai::{
    cli::{Cli, normalize_single_dash_long_opts},
    config,
    history::{
        COLON, Message, NEWLINE, SessionStore, append_history, build_message_arr,
        compress_messages_for_context, delete_history_artifacts,
    },
    mcp::McpClient,
    models,
    prompt::PromptEditor,
    request,
    skills::{self, SkillManifest},
    stream,
    types::{AgentContext, App, StreamOutcome},
};
use crate::common::prompt::{prompt_yes_or_no_interruptible, read_line};

pub mod input;
pub mod mcp_init;
pub mod model;
pub mod params;
pub mod print;
pub mod signal;
pub mod skill_matching;
pub mod tools;

pub use mcp_init::*;
pub use model::*;
pub use print::*;
pub use signal::*;
pub use skill_matching::*;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse_from(normalize_single_dash_long_opts(std::env::args()));
    let config = config::load_config()?;
    let session_store = SessionStore::new(config.history_file.as_path());
    if let Err(err) = session_store.migrate_legacy_if_needed(&config.history_file) {
        eprintln!("[Warning] Failed to migrate legacy history: {}", err);
    }
    let session_arg = cli.session.clone().unwrap_or_default();
    let session_id = if session_arg.trim().is_empty() {
        Uuid::new_v4().to_string()
    } else {
        session_arg.trim().to_string()
    };
    if let Err(err) = session_store.ensure_root_dir() {
        eprintln!("[Warning] Failed to create sessions dir: {}", err);
    }
    if cli.clear {
        let _ = session_store.clear_session(&session_id);
        println!("History cleared. (session: {})", session_id);
        return Ok(());
    }

    let shutdown = Arc::new(AtomicBool::new(false));
    let streaming = Arc::new(AtomicBool::new(false));
    let cancel_stream = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&shutdown);
    let streaming_flag = Arc::clone(&streaming);
    let cancel_stream_flag = Arc::clone(&cancel_stream);
    ctrlc::set_handler(move || {
        handle_sigint(
            signal_flag.as_ref(),
            streaming_flag.as_ref(),
            cancel_stream_flag.as_ref(),
        );
    })?;

    let writer = config::open_output_writer(cli.out.as_deref())?;
    let current_model = models::initial_model(&cli);
    let client = reqwest::blocking::Client::builder().build()?;
    let prompt_editor = if cli.args.is_empty() {
        Some(PromptEditor::new())
    } else {
        None
    };

    let mut app = App {
        pending_files: if cli.files.trim().is_empty() {
            None
        } else {
            Some(cli.files.clone())
        },
        pending_clipboard: cli.clipboard,
        pending_short_output: cli.short_output,
        current_model,
        session_id: session_id.clone(),
        session_history_file: session_store.session_history_file(&session_id),
        cli,
        config,
        client,
        attached_image_files: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        writer,
        prompt_editor,
        agent_context: Some(AgentContext {
            tools: super::tools::get_builtin_tool_definitions(),
            max_iterations: 6,
            ..Default::default()
        }),
    };

    let mut mcp_client = McpClient::new();
    let skill_manifests = skills::create_default_skills();
    let mcp_report = init_mcp(&mut app, &mut mcp_client);

    if app.cli.list_tools {
        print_builtin_tools(&app);
        return Ok(());
    }
    if app.cli.list_skills {
        print_skills(&skill_manifests);
        return Ok(());
    }
    if app.cli.list_mcp_tools {
        print_mcp_tools(&mcp_report, &mcp_client);
        return Ok(());
    }

    if mcp_report.loaded {
        println!(
            "[mcp] {} servers, {} tools (config: {})",
            mcp_report.server_count, mcp_report.tool_count, mcp_report.config_path
        );
    }

    run_loop(&mut app, &mut mcp_client, &skill_manifests)
}

fn run_loop(
    app: &mut App,
    mcp_client: &mut McpClient,
    skill_manifests: &[SkillManifest],
) -> Result<(), Box<dyn std::error::Error>> {
    let one_shot_mode = !app.cli.args.is_empty();
    let mut should_quit = one_shot_mode;

    let cleanup_one_shot = |app: &App| {
        if one_shot_mode {
            let _ = delete_history_artifacts(&app.session_history_file);
        }
    };
    loop {
        if app.shutdown.load(Ordering::Acquire) {
            cleanup_one_shot(app);
            return Ok(());
        }

        let Some(ctx) = input::next_question(app)? else {
            cleanup_one_shot(app);
            return Ok(());
        };
        if ctx.question.trim().is_empty() {
            should_quit = false;
            continue;
        }

        let mut question = ctx.question;
        if try_handle_help_command(&question) {
            if should_quit {
                cleanup_one_shot(app);
                return Ok(());
            }
            should_quit = false;
            continue;
        }
        if try_handle_session_command(app, &question)? {
            if should_quit {
                cleanup_one_shot(app);
                return Ok(());
            }
            should_quit = false;
            continue;
        }
        if try_handle_feishu_auth_command(mcp_client, &question)? {
            if should_quit {
                cleanup_one_shot(app);
                return Ok(());
            }
            should_quit = false;
            continue;
        }
        let next_model = resolve_model_for_input(app, &mut question);
        app.current_model = next_model.clone();

        app.cancel_stream.store(false, Ordering::SeqCst);
        let skill = match_skill(skill_manifests, &question);
        let openclaw_active = skill
            .as_ref()
            .is_some_and(|s| s.name.as_str() == "openclaw");

        let mut restore_agent_context = None;
        if openclaw_active {
            if let Some(ctx) = app.agent_context.as_mut() {
                restore_agent_context = Some((ctx.tools.clone(), ctx.max_iterations));
                ctx.tools = super::tools::get_openclaw_tool_definitions();
                ctx.max_iterations = 12;
            }
        }
        let system_prompt = if let Some(skill) = skill {
            let mut p = "You are a helpful assistant.".to_string();
            let extra = skill.build_system_prompt();
            if !extra.trim().is_empty() {
                p.push_str("\n\n");
                p.push_str(extra.trim());
            }
            p
        } else {
            "You are a helpful assistant.".to_string()
        };
        let system_prompt = if openclaw_active && !system_prompt.contains("OpenClaw") {
            format!(
                "{}\n\n{}",
                system_prompt,
                "OpenClaw mode:\n- Plan before acting.\n- Prefer reading/searching before editing.\n- Make minimal, reversible edits.\n- Verify by running checks/tests.\n- If a tool is unsafe or ambiguous, ask the user."
            )
        } else {
            system_prompt
        };
        let system_prompt = if is_feishu_docs_search_intent(&question) {
            format!(
                "{}\n\n{}",
                system_prompt,
                "Feishu mode:\n- If the user asks to search Feishu/Lark cloud docs, call the MCP tool mcp_*_docs_search.\n- If authorization is required, follow the MCP OAuth flow (oauth_authorize_url -> oauth_wait_local_code -> oauth_exchange_code) and then retry.\n- If the MCP tool is not available, tell the user to configure MCP first."
            )
        } else {
            system_prompt
        };
        let system_prompt = if mcp_client
            .get_all_tools()
            .iter()
            .any(|t| t.function.name.contains("mcp_feishu_"))
        {
            format!(
                "{}\n\n{}",
                system_prompt,
                "If the user asks anything related to Feishu/Lark docs, prefer calling the available Feishu MCP tools instead of saying you cannot access the account."
            )
        } else {
            system_prompt
        };

        let mut messages = Vec::new();
        messages.push(Message {
            role: "system".to_string(),
            content: Value::String(system_prompt),
            tool_calls: None,
            tool_call_id: None,
        });
        let history = build_message_arr(ctx.history_count, &app.session_history_file)?;
        let history = if app.config.history_max_chars == 0 {
            history
        } else {
            compress_messages_for_context(
                history,
                app.config.history_max_chars,
                app.config.history_keep_last,
                app.config.history_summary_max_chars,
            )
        };
        messages.extend(history);
        messages.push(Message {
            role: "user".to_string(),
            content: request::build_content(&next_model, &question, &app.attached_image_files)?,
            tool_calls: None,
            tool_call_id: None,
        });

        let max_iterations = app
            .agent_context
            .as_ref()
            .map(|c| c.max_iterations)
            .unwrap_or(0);
        let max_iterations = max_iterations.max(1);

        let mut iteration = 0usize;
        let mut final_assistant_text = String::new();
        loop {
            iteration += 1;
            let mut current_history = String::new();
            app.streaming.store(true, Ordering::Release);
            let mut response =
                match request::do_request_messages(app, &next_model, messages.clone(), true) {
                    Ok(response) => response,
                    Err(err) => {
                        app.streaming.store(false, Ordering::Release);
                        cleanup_one_shot(app);
                        return Err(err);
                    }
                };
            if app.cancel_stream.swap(false, Ordering::AcqRel) {
                app.streaming.store(false, Ordering::Release);
                println!("\nInterrupted.");
                if should_quit {
                    cleanup_one_shot(app);
                    return Ok(());
                }
                break;
            }
            request::print_info(&next_model);
            let stream_result =
                match stream::stream_response(app, &mut response, &mut current_history) {
                    Ok(result) => result,
                    Err(err) => {
                        app.streaming.store(false, Ordering::Release);
                        cleanup_one_shot(app);
                        return Err(err);
                    }
                };
            app.streaming.store(false, Ordering::Release);

            if stream_result.outcome == StreamOutcome::Cancelled {
                println!("\nInterrupted.");
                if should_quit {
                    cleanup_one_shot(app);
                    return Ok(());
                }
                break;
            }
            if app.shutdown.load(Ordering::Acquire) {
                println!();
                cleanup_one_shot(app);
                return Ok(());
            }
            drain_response(&mut response)?;

            if stream_result.outcome != StreamOutcome::ToolCall {
                let assistant_msg = Message {
                    role: "assistant".to_string(),
                    content: Value::String(stream_result.assistant_text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                };
                messages.push(assistant_msg);
                final_assistant_text = stream_result.assistant_text;
                break;
            }

            println!("\n{}", "[Tool Calls]".yellow());
            for tool_call in &stream_result.tool_calls {
                println!(
                    "  - {}({})",
                    tool_call.function.name.cyan(),
                    tool_call.function.arguments.dimmed()
                );
            }

            let exec_result = tools::execute_tool_calls(mcp_client, &stream_result.tool_calls)?;

            let assistant_msg = Message {
                role: "assistant".to_string(),
                content: Value::String(stream_result.assistant_text.clone()),
                tool_calls: Some(exec_result.executed_tool_calls.clone()),
                tool_call_id: None,
            };
            messages.push(assistant_msg);

            for (tool_call, result) in exec_result
                .executed_tool_calls
                .iter()
                .zip(exec_result.tool_results.iter())
            {
                println!("\n{}", "[Tool Result]".green());
                if tool_call.function.name == "web_search" {
                    let mut preview = String::new();
                    let mut truncated = false;
                    for (lines, line) in result.content.lines().enumerate() {
                        if lines >= 12 || preview.len() >= 1200 {
                            truncated = true;
                            break;
                        }
                        preview.push_str(line);
                        preview.push('\n');
                    }
                    print!("{}", preview);
                    if truncated {
                        println!("... (truncated)");
                    }
                } else {
                    println!("{}", result.content);
                }
                messages.push(Message {
                    role: "tool".to_string(),
                    content: Value::String(result.content.clone()),
                    tool_calls: None,
                    tool_call_id: Some(result.tool_call_id.clone()),
                });
            }

            if iteration >= max_iterations {
                final_assistant_text = "Agent stopped: too many tool iterations.".to_string();
                break;
            }
        }

        if !final_assistant_text.is_empty() {
            if !one_shot_mode {
                let history_line = format!(
                    "user{COLON}{question}{NEWLINE}assistant{COLON}{final_assistant_text}{NEWLINE}"
                );
                // 忽略历史保存错误，避免因为权限问题导致程序异常退出
                if let Err(e) = append_history(&app.session_history_file, &history_line) {
                    eprintln!("[Warning] Failed to save history: {}", e);
                }
            }
            println!();
        }

        if let Some((tools, max_iterations)) = restore_agent_context.take() {
            if let Some(ctx) = app.agent_context.as_mut() {
                ctx.tools = tools;
                ctx.max_iterations = max_iterations;
            }
        }
        if should_quit {
            cleanup_one_shot(app);
            return Ok(());
        }
        if let Some(writer) = app.writer.as_mut() {
            writer.write_all(b"\n---\n")?;
            writer.flush()?;
        }
    }
}

fn try_handle_feishu_auth_command(
    mcp_client: &mut McpClient,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
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
    if normalized != "feishu-auth" && normalized != "feishu auth" && normalized != "feishu_auth" {
        return Ok(false);
    }

    let mut server = None;
    for tool in mcp_client.get_all_tools() {
        if let Some((server_name, tool_name)) =
            mcp_client.parse_tool_name_for_known_server(&tool.function.name)
        {
            if tool_name == "oauth_authorize_url" {
                server = Some(server_name);
                break;
            }
        }
    }
    let Some(server) = server else {
        println!("未检测到飞书 OAuth MCP 工具（oauth_authorize_url）。");
        println!("- 先运行：cargo run --bin a -- --list-mcp-tools");
        println!("- 再按文档配置：docs/mcp-feishu.md");
        return Ok(true);
    };

    let scope = read_line("OAuth scope (default: offline_access): ");
    let scope = if scope.trim().is_empty() {
        "offline_access".to_string()
    } else {
        scope.trim().to_string()
    };

    let port_input = read_line("Local callback port (default: 8711): ");
    let port = port_input
        .trim()
        .parse::<u16>()
        .ok()
        .filter(|p| *p > 0)
        .unwrap_or(8711);
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    let url = mcp_client.call_tool(
        &server,
        "oauth_authorize_url",
        json!({
            "redirect_uri": redirect_uri,
            "scope": scope,
            "prompt": "consent",
            "state": "rust-tools-ai"
        }),
    )?;
    let url = url.trim().to_string();
    println!("\n授权链接：\n{url}\n");

    let open_now = prompt_yes_or_no_interruptible("Open browser now? (y/n): ");
    if open_now == Some(true) {
        let program = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
        let _ = Command::new(program).arg(&url).status();
    }

    println!("等待授权回调（{redirect_uri}）...");
    let code_out = mcp_client.call_tool(
        &server,
        "oauth_wait_local_code",
        json!({
            "port": port,
            "timeout_sec": 180
        }),
    )?;
    let code = extract_code_from_wait_output(&code_out).unwrap_or_default();
    if code.is_empty() {
        println!("未获取到 code，原始输出：\n{code_out}");
        return Ok(true);
    }

    let exchange = mcp_client.call_tool(&server, "oauth_exchange_code", json!({ "code": code }))?;
    println!("{exchange}");
    Ok(true)
}

fn extract_code_from_wait_output(s: &str) -> Option<String> {
    for line in s.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("code:") {
            let v = rest.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

fn is_feishu_docs_search_intent(input: &str) -> bool {
    let q = input.trim();
    if q.is_empty() {
        return false;
    }
    let lower = q.to_lowercase();
    let mentions_feishu = q.contains("飞书") || lower.contains("feishu") || lower.contains("lark");
    let mentions_docs = q.contains("云文档") || q.contains("文档");
    let mentions_search = q.contains("搜索");
    mentions_feishu && mentions_docs && mentions_search
}

fn try_handle_help_command(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return false;
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return false;
    };
    if normalized != "help" && normalized != "h" {
        return false;
    }
    println!("interactive commands:");
    println!("  /help");
    println!("  /feishu-auth");
    println!("  /sessions");
    println!("  /sessions list");
    println!("  /sessions current");
    println!("  /sessions new");
    println!("  /sessions use <id>");
    println!("  /sessions delete <id>");
    println!("  /sessions clear-all");
    true
}

fn try_handle_session_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
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
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if cmd != "sessions" && cmd != "session" {
        return Ok(false);
    }
    let action = parts.next().unwrap_or("list");
    let store = SessionStore::new(app.config.history_file.as_path());
    let _ = store.ensure_root_dir();

    match action {
        "help" | "h" => {
            println!("sessions commands:");
            println!("  /sessions");
            println!("  /sessions list");
            println!("  /sessions current");
            println!("  /sessions new");
            println!("  /sessions use <id>");
            println!("  /sessions delete <id>");
            println!("  /sessions clear-all");
        }
        "list" | "ls" => {
            let sessions = store.list_sessions()?;
            if sessions.is_empty() {
                println!("No sessions.");
            } else {
                for s in sessions {
                    let mark = if s.id == app.session_id { "*" } else { " " };
                    let time = s
                        .modified_local
                        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let prompt = s
                        .first_user_prompt
                        .as_deref()
                        .map(sanitize_session_prompt)
                        .filter(|v| !v.is_empty())
                        .unwrap_or_else(|| "-".to_string());
                    let prompt = truncate_session_prompt(&prompt, 80);
                    println!(
                        "{mark} {:<36}  {}  {:>8}B  {}",
                        s.id, time, s.size_bytes, prompt
                    );
                }
            }
        }
        "current" | "cur" => {
            println!("session: {}", app.session_id);
            println!("history: {}", app.session_history_file.display());
            let first = store.first_user_prompt(&app.session_id).unwrap_or(None);
            if let Some(v) = first {
                let prompt = sanitize_session_prompt(&v);
                if !prompt.is_empty() {
                    println!("first: {}", truncate_session_prompt(&prompt, 160));
                }
            }
        }
        "new" | "create" => {
            let new_id = Uuid::new_v4().to_string();
            app.session_id = new_id.clone();
            app.session_history_file = store.session_history_file(&new_id);
            println!("Switched to new session: {}", new_id);
        }
        "use" | "select" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions use <id>");
                return Ok(true);
            };
            app.session_id = id.to_string();
            app.session_history_file = store.session_history_file(id);
            println!("Switched session: {}", id);
            let first = store.first_user_prompt(id).unwrap_or(None);
            if let Some(v) = first {
                let prompt = sanitize_session_prompt(&v);
                if !prompt.is_empty() {
                    println!("first: {}", truncate_session_prompt(&prompt, 160));
                }
            }
        }
        "delete" | "del" | "rm" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions delete <id>");
                return Ok(true);
            };
            let deleted = store.delete_session(id)?;
            if deleted {
                if id == app.session_id {
                    let new_id = Uuid::new_v4().to_string();
                    app.session_id = new_id.clone();
                    app.session_history_file = store.session_history_file(&new_id);
                    println!(
                        "Deleted current session. Switched to new session: {}",
                        new_id
                    );
                } else {
                    println!("Deleted session: {}", id);
                }
            } else {
                println!("Session not found: {}", id);
            }
        }
        "clear-all" | "clear_all" | "clear" | "wipe" => {
            let confirm = crate::common::prompt::prompt_yes_or_no_interruptible(
                "Delete ALL sessions? (y/n): ",
            );
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }

            let deleted = store.clear_all_sessions()?;
            let new_id = Uuid::new_v4().to_string();
            app.session_id = new_id.clone();
            app.session_history_file = store.session_history_file(&new_id);
            println!("Deleted {deleted} session(s). Switched to new session: {new_id}");
        }
        _ => {
            println!("unknown action: {}. try: /sessions help", action);
        }
    }
    Ok(true)
}

fn sanitize_session_prompt(s: &str) -> String {
    let mut out = String::new();
    let mut last_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(ch);
            last_space = false;
        }
    }
    out.trim().to_string()
}

fn truncate_session_prompt(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push_str("...");
    out
}
