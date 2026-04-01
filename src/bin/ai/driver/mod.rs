use std::{
    io::Write,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use colored::Colorize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::ai::{
    cli::{self},
    config,
    history::{
        Message, SessionStore, append_history_messages, build_message_arr,
        compress_messages_for_context,
    },
    mcp::McpClient,
    models,
    prompt::PromptEditor,
    request,
    skills::{self, SkillManifest},
    stream,
    types::{AgentContext, App, StreamOutcome, StreamResult},
};
use crate::common::configw;
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

const DEFAULT_MAX_ITERATIONS: usize = 256;
const OPENCLAW_MAX_ITERATIONS: usize = 256;

fn print_assistant_banner() {
    println!("\n{}", "[Assistant]".bright_blue().bold());
}

fn print_tool_output_block(content: &str) {
    if content.trim().is_empty() {
        println!("  {} {}", "│".bright_black(), "(empty)".dimmed());
        return;
    }
    for line in content.lines() {
        if line.is_empty() {
            println!("  {}", "│".bright_black());
        } else {
            println!("  {} {}", "│".bright_black(), line.dimmed());
        }
    }
}

fn builtin_tools_for_skill(
    prompt_optimizer_active: bool,
    skill: Option<&SkillManifest>,
) -> Vec<crate::ai::types::ToolDefinition> {
    if prompt_optimizer_active {
        return Vec::new();
    }
    if let Some(skill) = skill {
        if !skill.tool_groups.is_empty() {
            let groups: Vec<&str> = skill.tool_groups.iter().map(|s| s.as_str()).collect();
            return super::tools::tool_definitions_for_groups(&groups);
        }
        if !skill.tools.is_empty() {
            return super::tools::get_tool_definitions_by_names(&skill.tools);
        }
    }
    super::tools::get_builtin_tool_definitions()
}

fn tool_uses_mcp_server(tool_name: &str, allowed_servers: &[String]) -> bool {
    if !tool_name.starts_with("mcp_") {
        return false;
    }

    let mut names = allowed_servers
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));

    names.into_iter().any(|server_name| {
        let prefix = format!("mcp_{server_name}_");
        tool_name
            .strip_prefix(&prefix)
            .is_some_and(|tool_part| !tool_part.is_empty())
    })
}

fn mcp_tools_for_skill(
    mcp_client: &McpClient,
    prompt_optimizer_active: bool,
    skill: Option<&SkillManifest>,
) -> Vec<crate::ai::types::ToolDefinition> {
    if prompt_optimizer_active {
        return Vec::new();
    }

    let all_tools = mcp_client.get_all_tools();
    let Some(skill) = skill else {
        return all_tools;
    };
    if skill.mcp_servers.is_empty() {
        return all_tools;
    }

    all_tools
        .into_iter()
        .filter(|tool| tool_uses_mcp_server(&tool.function.name, &skill.mcp_servers))
        .collect()
}

#[allow(clippy::type_complexity)]
async fn prepare_skill_for_turn(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    question: &str,
) -> (
    String,
    Option<(Vec<crate::ai::types::ToolDefinition>, usize)>,
    Option<String>,
) {
    let cfg = configw::get_all_config();
    let router_enabled = !cfg
        .get_opt("ai.skills.router")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");

    let router_selected = if router_enabled {
        let model = app.current_model.clone();
        request::select_skill_via_model(app, &model, question, skill_manifests).await
    } else {
        None
    };

    let heuristic_skill = match_skill(skill_manifests, question);
    let router_skill = router_selected
        .as_deref()
        .and_then(|name| skill_manifests.iter().find(|s| s.name == name));
    // 优先使用模型路由结果，其次使用启发式匹配作为 fallback
    // 当模型路由失败（返回 None）时，才使用启发式匹配
    let skill = router_skill.or(heuristic_skill);
    let matched_skill_name = skill.map(|s| s.name.clone());
    let prompt_optimizer_active = skill
        .as_ref()
        .is_some_and(|s| s.name.as_str() == "prompt-optimizer");
    let openclaw_active = skill.as_ref().is_some_and(|s| {
        s.name.as_str() == "openclaw" || s.tool_groups.iter().any(|g| g == "openclaw")
    });

    let builtin_tools = builtin_tools_for_skill(prompt_optimizer_active, skill);
    let mcp_tools = mcp_tools_for_skill(mcp_client, prompt_optimizer_active, skill);

    let mut restore_agent_context = None;
    if let Some(ctx) = app.agent_context.as_mut() {
        let mut all_tools = builtin_tools;
        all_tools.extend(mcp_tools);
        let prev_tools = std::mem::replace(&mut ctx.tools, all_tools);
        let prev_max_iterations = std::mem::replace(
            &mut ctx.max_iterations,
            if openclaw_active {
                OPENCLAW_MAX_ITERATIONS
            } else {
                DEFAULT_MAX_ITERATIONS
            },
        );
        restore_agent_context = Some((prev_tools, prev_max_iterations));
    }

    let mut system_prompt = if let Some(skill) = skill {
        let mut p = "You are a helpful assistant.".to_string();
        p.push_str("\n\n");
        p.push_str("Skill enforcement:\n- You MUST follow the active skill instructions precisely.\n- Do not ignore, weaken, or bypass the skill behavior.\n- If the user request conflicts with the skill, ask a brief clarification aligned with the skill.");
        let extra = skill.build_system_prompt();
        if !extra.trim().is_empty() {
            p.push_str("\n\n");
            p.push_str(extra.trim());
        }
        p
    } else {
        "You are a helpful assistant.".to_string()
    };

    system_prompt = format!(
        "{}\n\n{}",
        system_prompt,
        "Tool recovery mode:\n- If a tool call fails, read the error message and correct course before answering.\n- Prefer retrying with corrected arguments or switching to a more appropriate tool.\n- Do not repeat the exact same failing tool call unless the error indicates a transient retry is appropriate.\n- If a URL-based docs fetch tool says the URL is unsupported, switch to a search tool or ask for a supported docs URL instead of retrying the same call.\n- Only stop and ask the user when the error is ambiguous or missing required information."
    );

    (system_prompt, restore_agent_context, matched_skill_name)
}

#[cfg(test)]
mod tests {
    use super::tool_uses_mcp_server;

    #[test]
    fn mcp_server_filter_matches_longest_server_name_prefix() {
        let allowed = vec!["foo".to_string(), "foo_bar".to_string()];
        assert!(tool_uses_mcp_server("mcp_foo_bar_search", &allowed));
        assert!(tool_uses_mcp_server("mcp_foo_lookup", &allowed));
        assert!(!tool_uses_mcp_server("mcp_bar_search", &allowed));
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli::parse_cli_args(std::env::args());
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

    if cli.help {
        cli::print_help();
        return Ok(());
    }

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
    let client = reqwest::Client::builder().build()?;
    let prompt_editor = if cli.args.is_empty() {
        Some(PromptEditor::new(
            &session_id,
            config.history_file.as_path(),
        ))
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
            max_iterations: DEFAULT_MAX_ITERATIONS,
            ..Default::default()
        }),
    };

    let mut mcp_client = McpClient::new();
    let skill_manifests = if app.cli.no_skills {
        Vec::new()
    } else {
        skills::load_all_skills()
    };
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

    run_loop(&mut app, &mut mcp_client, &skill_manifests).await
}

async fn run_loop(
    app: &mut App,
    mcp_client: &mut McpClient,
    skill_manifests: &[SkillManifest],
) -> Result<(), Box<dyn std::error::Error>> {
    let one_shot_mode = !app.cli.args.is_empty();
    let mut should_quit = one_shot_mode;

    let cleanup_one_shot = |app: &App| {
        if one_shot_mode {
            let store = SessionStore::new(app.config.history_file.as_path());
            let _ = store.delete_session(&app.session_id);
        }
    };
    loop {
        if app.shutdown.load(Ordering::Relaxed) {
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

        app.cancel_stream.store(false, Ordering::Relaxed);
        let (system_prompt, mut restore_agent_context, matched_skill_name) =
            prepare_skill_for_turn(app, mcp_client, skill_manifests, &question).await;
        if let Some(name) = matched_skill_name.as_deref() {
            println!("[skill: {}]", name.cyan());
        }

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
        let mut messages = Vec::with_capacity(history.len() + 2);
        messages.push(Message {
            role: "system".to_string(),
            content: Value::String(system_prompt),
            tool_calls: None,
            tool_call_id: None,
        });
        messages.extend(history);
        let user_message = Message {
            role: "user".to_string(),
            content: request::build_content(&next_model, &question, &app.attached_image_files)?,
            tool_calls: None,
            tool_call_id: None,
        };
        messages.push(user_message.clone());
        let mut turn_messages = Vec::with_capacity(8);
        turn_messages.push(user_message);

        let max_iterations = app
            .agent_context
            .as_ref()
            .map(|c| c.max_iterations)
            .unwrap_or(0);
        let max_iterations = max_iterations.max(1);

        let mut iteration = 0usize;
        let mut force_final_response = false;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        loop {
            iteration += 1;
            let mut current_history = String::new();
            app.streaming.store(true, Ordering::Relaxed);
            if force_final_response {
                messages.push(Message {
                    role: "system".to_string(),
                    content: Value::String(
                        "Tool limit reached. Do not call any more tools. Provide the best possible final answer using the information already collected.".to_string(),
                    ),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }

            let saved_tools = if force_final_response {
                app.agent_context
                    .as_mut()
                    .map(|ctx| std::mem::take(&mut ctx.tools))
            } else {
                None
            };

            let request_result =
                request::do_request_messages(app, &next_model, &messages, true).await;

            if let Some(saved_tools) = saved_tools
                && let Some(ctx) = app.agent_context.as_mut()
            {
                ctx.tools = saved_tools;
            }

            let mut response = match request_result {
                Ok(response) => response,
                Err(err) => {
                    app.streaming.store(false, Ordering::Relaxed);
                    if request::is_transient_error(&err) {
                        eprintln!("[Warning] {}", err);
                        if should_quit {
                            cleanup_one_shot(app);
                            return Ok(());
                        }
                        break;
                    }
                    cleanup_one_shot(app);
                    return Err(err.into());
                }
            };
            if app.cancel_stream.swap(false, Ordering::Relaxed) {
                app.streaming.store(false, Ordering::Relaxed);
                println!("\nInterrupted.");
                if should_quit {
                    cleanup_one_shot(app);
                    return Ok(());
                }
                break;
            }
            request::print_info(&next_model);
            print_assistant_banner();
            let stream_result =
                match stream::stream_response(app, &mut response, &mut current_history).await {
                    Ok(result) => result,
                    Err(err) => {
                        app.streaming.store(false, Ordering::Relaxed);
                        eprintln!("\n[Error] 流式响应处理失败：{}", err);
                        eprintln!("[Info] 尝试继续对话...");
                        let _ = drain_response(&mut response).await;
                        StreamResult {
                            outcome: StreamOutcome::Completed,
                            tool_calls: Vec::new(),
                            assistant_text: "[响应解析失败，请重试]".to_string(),
                        }
                    }
                };

            input::clear_stdin_buffer();

            if stream_result.outcome == StreamOutcome::Cancelled {
                println!("\nInterrupted.");
                if should_quit {
                    cleanup_one_shot(app);
                    return Ok(());
                }
                break;
            }
            if app.shutdown.load(Ordering::Relaxed) {
                println!();
                cleanup_one_shot(app);
                return Ok(());
            }
            drain_response(&mut response).await?;

            if stream_result.outcome != StreamOutcome::ToolCall {
                let assistant_msg = Message {
                    role: "assistant".to_string(),
                    content: Value::String(stream_result.assistant_text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                };
                messages.push(assistant_msg.clone());
                turn_messages.push(assistant_msg);
                final_assistant_text = stream_result.assistant_text;
                final_assistant_recorded = true;
                break;
            }

            let exec_result = tools::execute_tool_calls(mcp_client, &stream_result.tool_calls)?;

            let assistant_msg = Message {
                role: "assistant".to_string(),
                content: Value::String(stream_result.assistant_text.clone()),
                tool_calls: Some(exec_result.executed_tool_calls.clone()),
                tool_call_id: None,
            };
            messages.push(assistant_msg.clone());
            turn_messages.push(assistant_msg);

            for (tool_call, result) in exec_result
                .executed_tool_calls
                .iter()
                .zip(exec_result.tool_results.iter())
            {
                println!(
                    "\n{} {}",
                    "[Tool]".bright_green().bold(),
                    tool_call.function.name.bright_cyan().bold()
                );
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
                    print_tool_output_block(&preview);
                    if truncated {
                        print_tool_output_block("... (truncated)");
                    }
                } else {
                    print_tool_output_block(&result.content);
                }
                let tool_message = Message {
                    role: "tool".to_string(),
                    content: Value::String(result.content.clone()),
                    tool_calls: None,
                    tool_call_id: Some(result.tool_call_id.clone()),
                };
                messages.push(tool_message.clone());
                turn_messages.push(tool_message);
            }

            input::clear_stdin_buffer();

            if iteration >= max_iterations {
                if force_final_response {
                    final_assistant_text = format!(
                        "Agent reached the tool iteration limit ({max_iterations}) without producing a final answer."
                    );
                    break;
                }
                force_final_response = true;
            }
        }

        if !final_assistant_text.trim().is_empty() {
            if !final_assistant_recorded {
                println!("\n{}", final_assistant_text.yellow());
                turn_messages.push(Message {
                    role: "assistant".to_string(),
                    content: Value::String(final_assistant_text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            if !one_shot_mode
                && let Err(e) = append_history_messages(&app.session_history_file, &turn_messages)
            {
                eprintln!("[Warning] Failed to save history: {}", e);
            }
            println!();
        } else {
            println!("{}", "(no response)".dimmed());
        }

        if let Some((tools, max_iterations)) = restore_agent_context.take()
            && let Some(ctx) = app.agent_context.as_mut()
        {
            ctx.tools = tools;
            ctx.max_iterations = max_iterations;
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
            && tool_name == "oauth_authorize_url"
        {
            server = Some(server_name);
            break;
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
        let program = if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };
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
    println!("  /sessions export <id> [output.md]");
    println!("  /sessions export-current [output.md]");
    println!("  /sessions export-last [output.md]");
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
            println!("  /sessions export <id> [output.md]");
            println!("  /sessions export-current [output.md]");
            println!("  /sessions export-last [output.md]");
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
        "export" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions export <id> [output.md]");
                return Ok(true);
            };
            let output_path = parts.next().unwrap_or("session_export.md");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_to_markdown(id, output_path) {
                Ok(()) => {
                    println!("Exported session '{}' to '{}'", id, output_path.display());
                }
                Err(err) => {
                    eprintln!("Failed to export session: {}", err);
                }
            }
        }
        "export-current" | "export-cur" => {
            let output_path = parts.next().unwrap_or("session_export.md");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_to_markdown(&app.session_id, output_path) {
                Ok(()) => {
                    println!(
                        "Exported current session '{}' to '{}'",
                        app.session_id,
                        output_path.display()
                    );
                }
                Err(err) => {
                    eprintln!("Failed to export session: {}", err);
                }
            }
        }
        "export-last" | "export-latest" => {
            let sessions = store.list_sessions()?;
            let Some(last) = sessions.first() else {
                println!("No sessions found to export.");
                return Ok(true);
            };
            let output_path = parts.next().unwrap_or("session_export.md");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_to_markdown(&last.id, output_path) {
                Ok(()) => {
                    println!(
                        "Exported latest session '{}' to '{}'",
                        last.id,
                        output_path.display()
                    );
                }
                Err(err) => {
                    eprintln!("Failed to export session: {}", err);
                }
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
