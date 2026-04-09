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
pub mod skill_runtime;
pub mod tools;
pub mod turn_runtime;

pub use commands::try_handle_interactive_command;
pub use mcp_init::*;
pub use model::*;
pub use skill_matching::*;

const DEFAULT_MAX_ITERATIONS: usize = 1024;
const OPENCLAW_MAX_ITERATIONS: usize = 64;

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

fn auto_openclaw_length_threshold() -> usize {
    configw::get_all_config()
        .get_opt("ai.agents.auto_route.openclaw_min_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(48)
}

fn contains_complex_execution_marker(question: &str) -> bool {
    let lower = question.to_lowercase();
    [
        "然后", "同时", "顺便", "一步步", "分步骤", "自动", "完整", "端到端", "闭环", "multi-step",
        "end-to-end", "step by step", "across", "implement", "refactor", "debug", "fix", "repair",
        "integrate", "migrate",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn contains_code_action_marker(question: &str) -> bool {
    let lower = question.to_lowercase();
    [
        "帮我", "修", "修复", "修改", "改一下", "实现", "添加", "扩展", "重构", "排查", "调试", "处理",
        "完成", "补", "优化", "迁移", "接入", "联调", "修一下", "报错", "panic", "error", "failing",
        "test", "build", "cargo", "fix", "implement", "add", "extend", "refactor", "debug",
        "update", "wire", "integrate", "migrate",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn should_auto_route_to_openclaw(
    intent: &intent_recognition::UserIntent,
    question: &str,
) -> bool {
    if intent.is_search_query() {
        return false;
    }
    if !matches!(
        intent.core,
        intent_recognition::CoreIntent::RequestAction | intent_recognition::CoreIntent::SeekSolution
    ) {
        return false;
    }

    let question = question.trim();
    if question.is_empty() || !contains_code_action_marker(question) {
        return false;
    }

    let char_count = question.chars().count();
    let line_count = question.lines().count();
    char_count >= auto_openclaw_length_threshold()
        || line_count >= 2
        || contains_complex_execution_marker(question)
}

fn maybe_auto_route_agent(
    app: &mut App,
    agent_manifests: &[AgentManifest],
    question: &str,
) {
    if app.cli.agent.is_some() || !auto_agent_routing_enabled() {
        return;
    }

    let intent =
        intent_recognition::detect_intent_with_model_path(question, &app.config.intent_model_path);
    let target_agent_name = if should_auto_route_to_openclaw(&intent, question) {
        "openclaw"
    } else {
        "build"
    };

    if app.current_agent == target_agent_name {
        return;
    }

    let Some(agent) = agents::find_agent_by_name(agent_manifests, target_agent_name) else {
        return;
    };
    if !agent.is_primary() || agent.disabled {
        return;
    }

    let old_agent = app.current_agent.clone();
    activate_primary_agent(app, agent);
    println!(
        "[agent auto-routed: {} -> {}]",
        old_agent, app.current_agent
    );
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
            tools: super::tools::get_builtin_tool_definitions(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            ..Default::default()
        }),
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

    let agent_manifests = agents::load_all_agents();
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
        maybe_auto_route_agent(app, agent_manifests, &question);
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

#[cfg(test)]
mod tests {
    use super::should_auto_route_to_openclaw;
    use crate::ai::driver::intent_recognition::{CoreIntent, IntentModifiers, UserIntent};

    #[test]
    fn auto_routes_complex_execution_requests_to_openclaw() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let question = "帮我实现这个 agent 的自动执行能力，然后跑检查并修掉相关报错";
        assert!(should_auto_route_to_openclaw(&intent, question));
    }

    #[test]
    fn does_not_route_simple_concept_questions_to_openclaw() {
        let intent = UserIntent::new(CoreIntent::QueryConcept);
        assert!(!should_auto_route_to_openclaw(&intent, "Rust 的 crate 是什么？"));
    }

    #[test]
    fn does_not_route_search_queries_to_openclaw() {
        let intent = UserIntent {
            core: CoreIntent::RequestAction,
            modifiers: IntentModifiers {
                is_search_query: true,
                target_resource: Some("tool".to_string()),
                negation: false,
            },
        };
        assert!(!should_auto_route_to_openclaw(&intent, "帮我找几个调试工具"));
    }
}
