use std::{
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use uuid::Uuid;

use crate::ai::{
    agents::{self, AgentManifest},
    cli::{self},
    config,
    history::SessionStore,
    mcp::McpClient,
    models,
    prompt::PromptEditor,
    skills::{self, SkillManifest},
    types::{AgentContext, App},
};

pub mod commands;
pub mod decision_log;
pub mod input;
pub mod intent_recognition;
pub mod mcp_init;
pub mod model;
pub mod params;
pub mod print;
pub mod reflection;
pub mod signal;
pub mod skill_matching;
pub mod skill_runtime;
pub mod tools;
pub mod turn_runtime;

pub use commands::{
    try_handle_agent_command, try_handle_feishu_auth_command, try_handle_help_command,
    try_handle_session_command, try_handle_share_command,
};
pub use mcp_init::*;
pub use model::*;
#[cfg(test)]
pub use signal::{sigint_action, SigintAction};
pub use skill_matching::*;

const DEFAULT_MAX_ITERATIONS: usize = 1024;
const OPENCLAW_MAX_ITERATIONS: usize = 64;

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
        signal::handle_sigint(
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
        current_agent: "build".to_string(),
        session_id: session_id.clone(),
        session_history_file: session_store.session_history_file(&session_id),
        cli,
        config,
        client,
        attached_image_files: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        ignore_next_prompt_interrupt: false,
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
        print::print_builtin_tools(&app);
        return Ok(());
    }
    if app.cli.list_skills {
        print::print_skills(&skill_manifests);
        return Ok(());
    }
    if app.cli.list_mcp_tools {
        print::print_mcp_tools(&mcp_report, &mcp_client);
        return Ok(());
    }

    let agent_manifests = agents::load_all_agents();

    if app.cli.list_agents {
        commands::help::print_agents_list(&agent_manifests);
        return Ok(());
    }

    if let Some(agent_name) = &app.cli.agent {
        if let Some(agent) = agents::find_agent_by_name(&agent_manifests, agent_name) {
            if agent.is_primary() && !agent.disabled {
                app.current_agent = agent.name.clone();
                if let Some(model) = &agent.model {
                    app.current_model = model.clone();
                }
                println!("[agent] using: {}", agent.name);
            } else {
                eprintln!(
                    "[Warning] Agent '{}' is not available, using default",
                    agent_name
                );
            }
        } else {
            eprintln!("[Warning] Agent '{}' not found, using default", agent_name);
        }
    }

    if mcp_report.loaded {
        println!(
            "[mcp] {} servers, {} tools (config: {})",
            mcp_report.server_count, mcp_report.tool_count, mcp_report.config_path
        );
    }

    run_loop(&mut app, &mut mcp_client, &skill_manifests, &agent_manifests).await
}

async fn run_loop(
    app: &mut App,
    mcp_client: &mut McpClient,
    skill_manifests: &[SkillManifest],
    agent_manifests: &[AgentManifest],
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
        if try_handle_agent_command(app, &question, agent_manifests)? {
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
        if try_handle_share_command(app, &question)? {
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
        let turn_outcome = match turn_runtime::run_turn(
            app,
            mcp_client,
            skill_manifests,
            ctx.history_count,
            question,
            next_model,
            one_shot_mode,
            should_quit,
        )
        .await
        {
            Ok(outcome) => outcome,
            Err(err) => {
                eprintln!("[Error] 当前轮请求失败：{}", err);
                if one_shot_mode || should_quit {
                    cleanup_one_shot(app);
                    return Err(err);
                }
                eprintln!("[Info] 会话保持运行，请继续输入下一条消息。\n");
                should_quit = false;
                continue;
            }
        };

        if matches!(turn_outcome, turn_runtime::TurnOutcome::Quit) || should_quit {
            cleanup_one_shot(app);
            return Ok(());
        }
        if let Some(writer) = app.writer.as_mut() {
            writer.write_all(b"\n---\n")?;
            writer.flush()?;
        }
    }
}
