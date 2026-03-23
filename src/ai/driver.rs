use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use clap::Parser;
use colored::Colorize;
use reqwest::blocking::Response;
use serde_json::Value;

use crate::{clipboard::string_content, strw::split::split_space_keep_symbol};

use super::{
    cli::{Cli, normalize_single_dash_long_opts},
    config, files,
    history::{COLON, NEWLINE, Message, append_history, build_message_arr},
    mcp,
    mcp_example_server,
    models,
    prompt::{PromptEditor, trim_trailing_newline},
    request, stream,
    skills,
    tools,
    types::{AgentContext, App, LoopOverrides, QuestionContext, StreamOutcome},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SigintAction {
    CancelStream,
    Shutdown,
    Exit,
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse_from(normalize_single_dash_long_opts(std::env::args()));

    if cli.example_mcp_server {
        return mcp_example_server::run_example_mcp_server();
    }

    let config = config::load_config()?;

    if cli.clear {
        config::clear_history_file(&config.history_file);
        println!("History cleared.");
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
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
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
        raw_args,
        cli,
        config,
        client,
        attached_image_files: Vec::new(),
        attached_binary_files: Vec::new(),
        uploaded_file_ids: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        writer,
        prompt_editor,
        agent_context: Some(AgentContext {
            tools: tools::get_builtin_tool_definitions(),
            max_iterations: 6,
            ..Default::default()
        }),
    };

    let mut mcp_client = mcp::McpClient::new();
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
            mcp_report.server_count,
            mcp_report.tool_count,
            mcp_report.config_path
        );
    }

    run_loop(&mut app, &mut mcp_client, &skill_manifests)
}

pub(super) fn handle_sigint(
    shutdown: &AtomicBool,
    streaming: &AtomicBool,
    cancel_stream: &AtomicBool,
) {
    match sigint_action(streaming, cancel_stream) {
        SigintAction::CancelStream => {
            cancel_stream.store(true, Ordering::SeqCst);
        }
        SigintAction::Shutdown => {
            shutdown.store(true, Ordering::SeqCst);
        }
        SigintAction::Exit => {
            shutdown.store(true, Ordering::SeqCst);
            #[cfg(unix)]
            unsafe {
                let _ = libc::close(libc::STDIN_FILENO);
            }
            #[cfg(not(test))]
            std::process::exit(130);
        }
    }
}

pub(super) fn sigint_action(streaming: &AtomicBool, cancel_stream: &AtomicBool) -> SigintAction {
    if streaming.load(Ordering::SeqCst) {
        if cancel_stream.load(Ordering::SeqCst) {
            SigintAction::Shutdown
        } else {
            SigintAction::CancelStream
        }
    } else {
        SigintAction::Exit
    }
}

fn run_loop(
    app: &mut App,
    mcp_client: &mut mcp::McpClient,
    skill_manifests: &[skills::SkillManifest],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut should_quit = !app.cli.args.is_empty();
    loop {
        if app.shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        let Some(ctx) = next_question(app)? else {
            return Ok(());
        };
        if ctx.question.trim().is_empty() {
            should_quit = false;
            continue;
        }

        let mut question = ctx.question;
        let next_model = resolve_model_for_input(app, &mut question);
        app.current_model = next_model.clone();

        app.cancel_stream.store(false, Ordering::SeqCst);
        let skill = match_skill(skill_manifests, &question);
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

        let mut messages = Vec::new();
        messages.push(Message {
            role: "system".to_string(),
            content: Value::String(system_prompt),
            tool_calls: None,
            tool_call_id: None,
        });
        messages.extend(build_message_arr(ctx.history_count, &app.config.history_file)?);
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
            app.streaming.store(true, Ordering::SeqCst);
            let mut response = match request::do_request_messages(app, &next_model, messages.clone(), true) {
                Ok(response) => response,
                Err(err) => {
                    app.streaming.store(false, Ordering::SeqCst);
                    return Err(err);
                }
            };
            if app.cancel_stream.swap(false, Ordering::SeqCst) {
                app.streaming.store(false, Ordering::SeqCst);
                println!("\nInterrupted.");
                if should_quit {
                    return Ok(());
                }
                break;
            }
            request::print_info(&next_model);
            let stream_result = match stream::stream_response(app, &mut response, &mut current_history) {
                Ok(result) => result,
                Err(err) => {
                    app.streaming.store(false, Ordering::SeqCst);
                    return Err(err);
                }
            };
            app.streaming.store(false, Ordering::SeqCst);

            if stream_result.outcome == StreamOutcome::Cancelled {
                println!("\nInterrupted.");
                if should_quit {
                    return Ok(());
                }
                break;
            }
            if app.shutdown.load(Ordering::SeqCst) {
                println!();
                return Ok(());
            }
            drain_response(&mut response)?;

            let assistant_msg = Message {
                role: "assistant".to_string(),
                content: Value::String(stream_result.assistant_text.clone()),
                tool_calls: if stream_result.tool_calls.is_empty() {
                    None
                } else {
                    Some(stream_result.tool_calls.clone())
                },
                tool_call_id: None,
            };
            messages.push(assistant_msg);

            if stream_result.outcome != StreamOutcome::ToolCall {
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

            let tool_results = execute_tool_calls(mcp_client, &stream_result.tool_calls)?;
            for result in &tool_results {
                println!("\n{}", "[Tool Result]".green());
                println!("{}", result.content);
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
            let history_line = format!(
                "user{COLON}{question}{NEWLINE}assistant{COLON}{final_assistant_text}{NEWLINE}"
            );
            append_history(&app.config.history_file, &history_line)?;
            println!();
        }

        if should_quit {
            return Ok(());
        }
        if let Some(writer) = app.writer.as_mut() {
            writer.write_all(b"\n---\n")?;
            writer.flush()?;
        }
    }
}

fn execute_tool_calls(
    mcp_client: &mut mcp::McpClient,
    tool_calls: &[super::types::ToolCall],
) -> Result<Vec<super::types::ToolResult>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    
    for tool_call in tool_calls {
        let result = if let Some((server_name, tool_name)) =
            mcp_client.parse_tool_name_for_known_server(&tool_call.function.name)
        {
            let raw_args = tool_call.function.arguments.trim();
            let args: Value = if raw_args.is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(raw_args)?
            };
            let content = mcp_client.call_tool(&server_name, &tool_name, args)?;
            super::types::ToolResult {
                tool_call_id: tool_call.id.clone(),
                content,
            }
        } else {
            tools::execute_tool_call(tool_call)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
        };
        println!("\n[Executed] {}", tool_call.function.name.green());
        results.push(result);
    }
    
    Ok(results)
}

fn match_skill<'a>(
    skills: &'a [skills::SkillManifest],
    input: &str,
) -> Option<&'a skills::SkillManifest> {
    let input_lower = input.to_lowercase();
    skills.iter().find(|s| {
        s.triggers
            .iter()
            .any(|t| !t.trim().is_empty() && input_lower.contains(&t.to_lowercase()))
    })
}

fn init_mcp(app: &mut App, mcp_client: &mut mcp::McpClient) -> McpInitReport {
    init_builtin_example_mcp(app, mcp_client);

    let cfg = crate::common::configw::get_all_config();
    let mcp_path = if !app.cli.mcp_config.trim().is_empty() {
        app.cli.mcp_config.trim().to_string()
    } else {
        cfg.get_opt("ai.mcp.config")
            .unwrap_or_else(|| "~/.config/mcp.json".to_string())
    };
    let mcp_path = crate::common::utils::expanduser(&mcp_path);
    let mcp_path = mcp_path.as_ref();

    let mut report = McpInitReport {
        config_path: mcp_path.to_string(),
        loaded: false,
        server_count: 0,
        tool_count: 0,
        failures: Vec::new(),
    };
    if std::fs::metadata(mcp_path).is_err() {
        return report;
    }

    let servers = match mcp::load_mcp_config_from_file(mcp_path) {
        Ok(s) => s,
        Err(_) => return report,
    };

    for (name, server_cfg) in &servers {
        if let Err(err) = mcp_client.connect_server(name, server_cfg) {
            report.failures.push(format!("{}: {}", name, err));
        }
    }

    if let Some(ctx) = app.agent_context.as_mut() {
        ctx.mcp_servers = servers;
        ctx.tools.extend(mcp_client.get_all_tools());
    }
    report.loaded = true;
    report.server_count = app
        .agent_context
        .as_ref()
        .map(|c| c.mcp_servers.len())
        .unwrap_or(0);
    report.tool_count = mcp_client.get_all_tools().len();
    report
}

#[derive(Debug, Clone)]
struct McpInitReport {
    config_path: String,
    loaded: bool,
    server_count: usize,
    tool_count: usize,
    failures: Vec<String>,
}

fn init_builtin_example_mcp(app: &mut App, mcp_client: &mut mcp::McpClient) {
    let cfg = crate::common::configw::get_all_config();
    let disabled = cfg
        .get_opt("ai.mcp.example.disabled")
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("true");
    if disabled {
        return;
    }

    let _ = mcp_client.connect_inprocess_example_server("example");
    if let Some(ctx) = app.agent_context.as_mut() {
        ctx.tools.extend(mcp_client.get_all_tools());
    }
}

fn print_builtin_tools(app: &App) {
    println!("{}", "[builtin tools]".yellow());
    let tools = app
        .agent_context
        .as_ref()
        .map(|c| c.tools.clone())
        .unwrap_or_default();
    for t in tools {
        if t.function.name.starts_with("mcp_") {
            continue;
        }
        println!(" - {}: {}", t.function.name.cyan(), t.function.description);
    }
}

fn print_skills(skill_manifests: &[skills::SkillManifest]) {
    println!("{}", "[skills]".yellow());
    for s in skill_manifests {
        println!(" - {}: {}", s.name.cyan(), s.description);
    }
}

fn print_mcp_tools(report: &McpInitReport, mcp_client: &mcp::McpClient) {
    if !report.loaded {
        println!("{}", "[mcp] no config or failed to load".yellow());
        println!("config path: {}", report.config_path);
        return;
    }
    println!(
        "{}",
        format!(
            "[mcp] {} servers, {} tools",
            report.server_count, report.tool_count
        )
        .yellow()
    );
    for failure in &report.failures {
        println!(" - {}", format!("[mcp failed] {failure}").red());
    }
    for t in mcp_client.get_all_tools() {
        println!(" - {}: {}", t.function.name.cyan(), t.function.description);
    }
}

fn drain_response(response: &mut Response) -> Result<(), Box<dyn std::error::Error>> {
    response.copy_to(&mut io::sink())?;
    Ok(())
}

fn next_question(app: &mut App) -> Result<Option<QuestionContext>, Box<dyn std::error::Error>> {
    if !app.cli.args.is_empty() {
        let base_question = if app.cli.raw {
            app.raw_args.clone()
        } else {
            let question = app.cli.args.join(" ");
            app.cli.args.clear();
            question
        };
        app.cli.args.clear();
        let (question, overrides) = parse_loop_overrides(&base_question);
        let history_count = overrides
            .history_count
            .unwrap_or_else(|| base_history_count(app.cli.history, app.cli.no_history));
        let ctx = finalize_question(
            app,
            question,
            history_count,
            overrides.short_output,
        )?;
        return Ok(Some(ctx));
    }

    let question = match prompt_user(app) {
        Ok(v) => v,
        Err(_) if app.shutdown.load(Ordering::SeqCst) => {
            return Ok(None);
        }
        Err(err) => return Err(err.into()),
    };
    let Some(question) = question else {
        app.shutdown.store(true, Ordering::SeqCst);
        return Ok(None);
    };
    let (question, overrides) = parse_loop_overrides(&question);
    let history_count = overrides
        .history_count
        .unwrap_or_else(|| base_history_count(app.cli.history, app.cli.no_history));
    let ctx = finalize_question(app, question, history_count, overrides.short_output)?;
    Ok(Some(ctx))
}

fn base_history_count(history: usize, no_history: bool) -> usize {
    if no_history { 0 } else { history }
}

fn finalize_question(
    app: &mut App,
    mut question: String,
    history_count: usize,
    loop_short_output: bool,
) -> Result<QuestionContext, Box<dyn std::error::Error>> {
    if let Some(files) = app.pending_files.take() {
        let parsed = files::parse_files(&files);
        if !parsed.text_files.is_empty() {
            let prefix = files::text_file_contents(&parsed.text_files)?;
            if !prefix.is_empty() {
                question = format!("{prefix}\n{question}");
            }
        }
        if !parsed.image_files.is_empty() {
            app.attached_image_files = parsed.image_files;
        }
        if !parsed.binary_files.is_empty() {
            app.attached_binary_files = parsed.binary_files;
        }
    }

    if app.pending_clipboard {
        let clipboard = string_content::get_clipboard_content();
        question = format!("{clipboard}{question}");
        app.pending_clipboard = false;
    }

    if app.pending_short_output || loop_short_output {
        if !question.ends_with('\n') {
            question.push('\n');
        }
        question.push_str("Be Concise.");
        app.pending_short_output = false;
    }

    Ok(QuestionContext {
        question,
        history_count,
    })
}

fn prompt_user(app: &mut App) -> io::Result<Option<String>> {
    if let Some(editor) = app.prompt_editor.as_mut() {
        if app.cli.multi_line {
            return editor.read_multi_line();
        }
        return editor.read_single_line();
    }

    let multiline = app.cli.multi_line;
    let stdin = io::stdin();
    let mut stdin = stdin.lock();

    if !multiline {
        print!("> ");
        io::stdout().flush()?;
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(trim_trailing_newline(line))),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                println!("Exit.");
                Ok(None)
            }
            Err(err) => Err(err),
        }
    } else {
        let mut lines = Vec::new();
        loop {
            print!("  ");
            io::stdout().flush()?;
            let mut line = String::new();
            match stdin.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => lines.push(trim_trailing_newline(line)),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                    println!("Exit.");
                    return Ok(None);
                }
                Err(err) => return Err(err),
            }
        }
        if lines.is_empty() {
            Ok(None)
        } else {
            Ok(Some(lines.join("\n")))
        }
    }
}

pub(super) fn loop_overrides(question: &str) -> LoopOverrides {
    parse_loop_overrides(question).1
}

pub(super) fn parse_loop_overrides(question: &str) -> (String, LoopOverrides) {
    let tokens = split_space_keep_symbol(question, "\"'").collect::<Vec<_>>();
    let mut out_tokens = Vec::with_capacity(tokens.len());

    let mut short_output = false;
    let has_x = tokens.iter().any(|t| *t == "-x");
    let mut history_count = has_x.then_some(0);

    let mut idx = 0usize;
    while idx < tokens.len() {
        match tokens[idx] {
            "-s" => {
                short_output = true;
                idx += 1;
            }
            "-x" => {
                idx += 1;
            }
            "--history" => {
                if let Some(next) = tokens.get(idx + 1)
                    && let Ok(value) = next.parse::<usize>()
                {
                    if !has_x {
                        history_count = Some(value);
                    }
                    idx += 2;
                } else {
                    out_tokens.push(tokens[idx].to_string());
                    idx += 1;
                }
            }
            _ => {
                out_tokens.push(tokens[idx].to_string());
                idx += 1;
            }
        }
    }

    (
        out_tokens.join(" "),
        LoopOverrides {
            short_output,
            history_count,
        },
    )
}

fn resolve_model_for_input(app: &App, question: &mut String) -> String {
    if let Some(model) = attachment_forced_model(
        &app.current_model,
        !app.attached_image_files.is_empty(),
        !app.attached_binary_files.is_empty(),
        &app.config.vl_default_model,
    ) {
        return model;
    }

    let trimmed = question.trim_end().to_string();
    if let Some(stripped) = trimmed.strip_suffix(" -code") {
        *question = stripped.to_string();
        return models::qwen_coder_plus_latest().to_string();
    }
    if let Some(stripped) = trimmed.strip_suffix(" -d") {
        *question = stripped.to_string();
        return models::deepseek_v3().to_string();
    }
    if let Some(selector) = trailing_model_selector(&trimmed) {
        if trimmed.len() >= 3 {
            *question = trimmed[..trimmed.len() - 3].trim_end().to_string();
        }
        return models::model_from_selector(selector, app.cli.thinking)
            .as_str()
            .to_string();
    }
    app.current_model.clone()
}

pub(super) fn attachment_forced_model(
    current_model: &str,
    has_image_files: bool,
    has_binary_files: bool,
    vl_default_model: &str,
) -> Option<String> {
    if current_model == models::qwen_long() {
        return Some(models::qwen_long().to_string());
    }
    if has_binary_files {
        return Some(models::qwen_long().to_string());
    }
    if has_image_files && !models::is_vl_model(current_model) {
        return Some(models::determine_vl_model(vl_default_model));
    }
    None
}

pub(super) fn trailing_model_selector(input: &str) -> Option<u8> {
    let bytes = input.as_bytes();
    if bytes.len() < 3 {
        return None;
    }
    let dash_idx = bytes.len() - 2;
    if bytes[dash_idx] != b'-' || !bytes[dash_idx + 1].is_ascii_digit() {
        return None;
    }
    if dash_idx == 0 || bytes[dash_idx - 1] != b' ' {
        return None;
    }
    Some(bytes[dash_idx + 1] - b'0')
}
