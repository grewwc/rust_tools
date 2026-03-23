use std::{
    io::{self, Write, BufRead},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use clap::Parser;
use colored::Colorize;
use serde_json::Value;

use crate::{
    ai::{
        cli::{Cli, normalize_single_dash_long_opts},
        config,
        history::{COLON, Message, NEWLINE, append_history, build_message_arr},
        mcp::McpClient,
        mcp_example_server,
        models,
        prompt::PromptEditor,
        request,
        skills::{self, SkillManifest},
        stream,
        types::{AgentContext, App, StreamOutcome},
    }
};

pub mod signal;
pub mod mcp_init;
pub mod print;
pub mod tools;
pub mod skill_matching;
pub mod input;
pub mod params;
pub mod model;

pub use signal::*;
pub use mcp_init::*;
pub use print::*;
pub use tools::*;
pub use skill_matching::*;
pub use input::*;
pub use model::*;

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
    let mut should_quit = !app.cli.args.is_empty();
    loop {
        if app.shutdown.load(Ordering::Acquire) {
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
        messages.extend(build_message_arr(
            ctx.history_count,
            &app.config.history_file,
        )?);
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
                        return Err(err);
                    }
                };
            if app.cancel_stream.swap(false, Ordering::AcqRel) {
                app.streaming.store(false, Ordering::Release);
                println!("\nInterrupted.");
                if should_quit {
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
                        return Err(err);
                    }
                };
            app.streaming.store(false, Ordering::Release);

            if stream_result.outcome == StreamOutcome::Cancelled {
                println!("\nInterrupted.");
                if should_quit {
                    return Ok(());
                }
                break;
            }
            if app.shutdown.load(Ordering::Acquire) {
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
            // 忽略历史保存错误，避免因为权限问题导致程序异常退出
            if let Err(e) = append_history(&app.config.history_file, &history_line) {
                eprintln!("[Warning] Failed to save history: {}", e);
            }
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
