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
use crate::commonw::configw;

pub mod agent_router;
pub mod commands;
pub mod decision_log;
pub mod input;
pub mod intent_model;
pub mod intent_recognition;
pub mod mcp_init;
pub mod model;
pub mod params;
pub mod print;
pub mod reflection;
pub mod signal;
pub mod skill_matching;
pub mod skill_match_model;
pub mod skill_ranking;
pub mod skill_runtime;
pub mod text_similarity;
pub mod tools;
pub mod turn_runtime;

pub use commands::try_handle_interactive_command;
pub use mcp_init::*;
pub use model::*;
pub use skill_matching::*;
pub use skill_ranking::*;
pub use text_similarity::*;

const DEFAULT_MAX_ITERATIONS: usize = 1024;
const EXECUTOR_MAX_ITERATIONS: usize = 64;

#[crate::ai::agent_hang_span(
    "pre-fix",
    "S",
    "driver::run:load_all_skills",
    "[DEBUG] loading skills",
    "[DEBUG] loaded skills",
    { "no_skills": no_skills },
    {
        "count": __agent_hang_result.len(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
fn load_skill_manifests(no_skills: bool) -> Vec<SkillManifest> {
    if no_skills {
        Vec::new()
    } else {
        skills::load_all_skills()
    }
}

fn activate_primary_agent(app: &mut App, agent: &AgentManifest) {
    app.current_agent = agent.name.clone();
    app.current_agent_manifest = Some(agent.clone());
    if let Some(model) = &agent.model {
        app.current_model = model.clone();
    }
}

fn auto_agent_routing_enabled() -> bool {
    !configw::get_all_config()
        .get_opt("ai.agents.auto_route.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false")
}

fn auto_route_strategy() -> String {
    configw::get_all_config()
        .get_opt("ai.agents.auto_route.strategy")
        .unwrap_or_else(|| "model".to_string())
        .trim()
        .to_lowercase()
}

fn maybe_auto_route_agent(
    app: &mut App,
    agent_manifests: &[AgentManifest],
    question: &str,
) {
    if app.cli.agent.is_some() || !auto_agent_routing_enabled() {
        return;
    }

    let history = read_recent_history(app);
    let decision = match auto_route_strategy().as_str() {
        "heuristic" => {
            let router = agent_router::HeuristicRouter;
            agent_router::AgentRouter::route(&router, agent_manifests, question, &history, &app.current_agent)
        }
        _ => {
            let model_path = app.config.agent_route_model_path.clone();
            let router = agent_router::ModelRouter::new(model_path);
            agent_router::AgentRouter::route(&router, agent_manifests, question, &history, &app.current_agent)
        }
    };
    let Some(decision) = decision else {
        return;
    };

    let Some(agent) = agents::find_agent_by_name(agent_manifests, &decision.agent_name) else {
        return;
    };
    if !agent.is_primary() || agent.disabled {
        return;
    }

    let old_agent = app.current_agent.clone();
    activate_primary_agent(app, agent);

    println!(
        "\n[Agent 自动切换: {} → {}] (原因: {})\n",
        old_agent, app.current_agent, decision.reason
    );
}

fn read_recent_history(app: &App) -> Vec<crate::ai::history::Message> {
    use crate::ai::history::build_message_arr;
    match build_message_arr(usize::MAX, &app.session_history_file) {
        Ok(entries) => entries.into_iter().rev().take(10).collect(),
        Err(_) => Vec::new(),
    }
}

/// Loads all agents fresh from disk, enabling hot-reload of newly added/modified agents.
/// Returns the updated manifests.
fn reload_agent_manifests(agent_manifests: &mut Vec<AgentManifest>) {
    let new_agents = agents::load_all_agents();
    if new_agents.len() != agent_manifests.len() {
        let added = new_agents.len() as i64 - agent_manifests.len() as i64;
        if added > 0 {
            println!("[Agent 发现] 新发现 {} 个 agent(s)，已自动加载", added);
        } else {
            println!("[Agent 发现] agent 列表已更新，共 {} 个", new_agents.len());
        }
        *agent_manifests = new_agents;
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli::parse_cli_args(std::env::args());
    let config = config::load_config()?;
    let session_store = SessionStore::new(config.history_file.as_path());
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
        pending_short_output: cli.short_output,
        current_model,
        current_agent: "build".to_string(),
        current_agent_manifest: None,
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
            tools: super::tools::tool_definitions_for_groups(&["core"]),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            ..Default::default()
        }),
        last_skill_bias: None,
        agent_reload_counter: None,
    };

    let mut mcp_client = McpClient::new();
    let skill_manifests = load_skill_manifests(app.cli.no_skills);
    let mcp_report = init_mcp(&mut app, &mut mcp_client).await;

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

    let mut agent_manifests = agents::load_all_agents();
    if app.cli.list_agents {
        commands::help::print_agents_list(&agent_manifests);
        return Ok(());
    }

    if let Some(default_agent) = agents::find_agent_by_name(&agent_manifests, &app.current_agent)
        && default_agent.is_primary()
        && !default_agent.disabled
    {
        activate_primary_agent(&mut app, default_agent);
    }

    if let Some(agent_name) = &app.cli.agent {
        if let Some(agent) = agents::find_agent_by_name(&agent_manifests, agent_name) {
            if agent.is_primary() && !agent.disabled {
                activate_primary_agent(&mut app, agent);
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
            "{}",
            print::format_section_header(
                "mcp",
                Some(&format!(
                    "{} servers, {} tools",
                    mcp_report.server_count, mcp_report.tool_count
                ))
            )
        );
    }
    run_loop(&mut app, &mut mcp_client, &skill_manifests, &mut agent_manifests).await
}

async fn run_loop(
    app: &mut App,
    mcp_client: &mut McpClient,
    skill_manifests: &[SkillManifest],
    agent_manifests: &mut Vec<AgentManifest>,
) -> Result<(), Box<dyn std::error::Error>> {
    let one_shot_mode = !app.cli.args.is_empty();
    let mut should_quit = one_shot_mode;

    let cleanup_one_shot = |app: &App| {
        if one_shot_mode {
            let store = SessionStore::new(app.config.history_file.as_path());
            let _ = store.delete_session(&app.session_id);
        }
    };
    let handle_post_command = |app: &App, should_quit: &mut bool| {
        if *should_quit {
            cleanup_one_shot(app);
            true
        } else {
            *should_quit = false;
            false
        }
    };

    loop {
        // Agent hot-discovery: check for new agents every 5 turns
        if let Some(counter) = app.agent_reload_counter.as_mut() {
            *counter += 1;
            if *counter % 5 == 0 {
                reload_agent_manifests(agent_manifests);
            }
        } else {
            app.agent_reload_counter = Some(0);
        }

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
        if try_handle_interactive_command(app, mcp_client, &question, agent_manifests)? {
            if handle_post_command(app, &mut should_quit) {
                return Ok(());
            }
            continue;
        }
        maybe_auto_route_agent(app, &*agent_manifests, &question);
        let precomputed_ocr = if !app.attached_image_files.is_empty()
            && !crate::ai::models::is_vl_model(&app.current_model)
        {
            crate::ai::driver::model::ocr_images_for_attached_input(
                mcp_client,
                &app.attached_image_files,
            )
            .ok()
            .flatten()
        } else {
            None
        };
        let ocr_succeeded_for_images = precomputed_ocr
            .as_ref()
            .map(|ocr| ocr.images.iter().all(|image| image.error.is_none()))
            .unwrap_or(false);
        let next_model = resolve_model_for_input(app, ocr_succeeded_for_images, &mut question);
        app.current_model = next_model.clone();

        app.cancel_stream.store(false, Ordering::Relaxed);
        crate::ai::tools::registry::common::clear_tool_cancel();
        let turn_outcome = match turn_runtime::run_turn(
            app,
            mcp_client,
            skill_manifests,
            ctx.history_count,
            question,
            next_model,
            precomputed_ocr,
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

#[cfg(test)]
mod tests {
    use super::maybe_auto_route_agent;
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
    use crate::ai::cli::ParsedCli;
    use crate::ai::types::{AgentContext, App, AppConfig};
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    fn primary_agent(name: &str, description: &str, routing_tags: &[&str]) -> AgentManifest {
        AgentManifest {
            name: name.to_string(),
            description: description.to_string(),
            mode: AgentMode::Primary,
            model: None,
            temperature: None,
            max_steps: None,
            prompt: String::new(),
            system_prompt: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            routing_tags: routing_tags.iter().map(|tag| (*tag).to_string()).collect(),
            model_tier: Some(AgentModelTier::Heavy),
            disabled: false,
            hidden: false,
            color: None,
            source_path: None,
        }
    }

    fn test_app(current_agent: &str) -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 12000,
                history_keep_last: 8,
                history_summary_max_chars: 4000,
                intent_model: None,
                intent_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/intent/intent_model.json"),
                agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            client: reqwest::Client::new(),
            current_model: "test-model".to_string(),
            current_agent: current_agent.to_string(),
            current_agent_manifest: None,
            pending_files: None,
            pending_short_output: false,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            writer: None,
            prompt_editor: None,
            agent_context: Some(AgentContext {
                tools: Vec::new(),
                mcp_servers: Default::default(),
                max_iterations: super::DEFAULT_MAX_ITERATIONS,
            }),
            last_skill_bias: None,
            agent_reload_counter: None,
        }
    }

    #[test]
    fn auto_route_falls_back_to_build_instead_of_current_agent() {
        let build = primary_agent("build", "Default agent for development work", &["fix", "debug"]);
        let prompt_skill = primary_agent(
            "prompt-skill",
            "Specialized agent for optimizing and generating prompts and skills",
            &["prompt", "skill", "optimize"],
        );
        let mut app = test_app("prompt-skill");

        maybe_auto_route_agent(
            &mut app,
            &[build.clone(), prompt_skill.clone()],
            "这个问题为什么会这样？",
        );

        assert_eq!(app.current_agent, "build");
        assert_eq!(
            app.current_agent_manifest.as_ref().map(|agent| agent.name.as_str()),
            Some("build")
        );
    }
}
