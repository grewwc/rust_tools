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

fn auto_executor_length_threshold() -> usize {
    let cfg = configw::get_all_config();
    cfg.get_opt("ai.agents.auto_route.executor_min_chars")
        .or_else(|| cfg.get_opt("ai.agents.auto_route.openclaw_min_chars"))
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

/// Extracts plain text content from a Message for context scoring.
fn extract_text_from_message(msg: &crate::ai::history::Message) -> Option<String> {
    use serde_json::Value;
    match &msg.content {
        Value::String(s) => Some(s.clone()),
        Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().filter_map(|part| {
                if let Value::Object(obj) = part {
                    if let Some(Value::String(s)) = obj.get("text") {
                        return Some(s.clone());
                    }
                }
                None
            }).collect();
            if parts.is_empty() { None } else { Some(parts.join(" ")) }
        }
        _ => None,
    }
}

/// Computes a context-aware routing score for each candidate agent.
/// Returns the best matching agent, or None if none qualifies.
fn select_best_agent_by_context<'a>(
    agent_manifests: &'a [AgentManifest],
    question: &str,
    history: &[crate::ai::history::Message],
) -> Option<&'a AgentManifest> {
    let question_lower = question.to_lowercase();

    let mut best: Option<&AgentManifest> = None;
    let mut best_score: f64 = 0.0;

    for agent in agent_manifests.iter() {
        if !agent.is_primary() || agent.disabled || agent.hidden {
            continue;
        }

        let score = score_agent_for_question(agent, &question_lower, history);
        if score > best_score {
            best_score = score;
            best = Some(agent);
        }
    }

    // Only switch if the score exceeds a minimum threshold
    if best_score >= 5.0 {
        best
    } else {
        None
    }
}

/// Scores a single agent for the given question and conversation history.
fn score_agent_for_question(
    agent: &AgentManifest,
    question_lower: &str,
    history: &[crate::ai::history::Message],
) -> f64 {
    let mut score = 0.0;

    // 1. Direct name mention in the question (highest signal)
    let agent_name_lower = agent.name.to_lowercase();
    if question_lower.contains(&agent_name_lower) {
        score += 20.0;
    }

    // 2. Routing tags matching question keywords
    for tag in agent.routing_tags_normalized() {
        if question_lower.contains(&tag) {
            score += 3.0;
        }
    }

    // 3. Description keyword match
    let desc_lower = agent.description.to_lowercase();
    for word in question_lower.split_whitespace().filter(|w| w.len() >= 3) {
        if desc_lower.contains(word) {
            score += 1.5;
        }
    }

    // 4. Context carry-over: if recent conversation was about the same agent's
    //    domain, give a bonus
    if !history.is_empty() {
        let recent_entries: Vec<_> = history.iter().rev().take(4).collect();
        for entry in recent_entries {
            if let Some(text) = extract_text_from_message(entry) {
                let entry_lower = text.to_lowercase();
                let agent_name_lower = agent.name.to_lowercase();
                if entry_lower.contains(&agent_name_lower) {
                    score += 2.0;
                }
                let desc_lower = agent.description.to_lowercase();
                for word in entry_lower.split_whitespace().filter(|w| w.len() >= 4) {
                    if desc_lower.contains(word) {
                        score += 0.5;
                    }
                }
            }
        }
    }

    // 5. Model tier alignment: prefer lightweight agents for simple queries
    let word_count = question_lower.split_whitespace().count();
    if word_count <= 5
        && agent
            .model_tier
            .as_ref()
            .is_some_and(|t| matches!(t, agents::AgentModelTier::Light))
    {
        score += 2.0;
    }
    if word_count >= 20
        && agent
            .model_tier
            .as_ref()
            .is_some_and(|t| matches!(t, agents::AgentModelTier::Heavy))
    {
        score += 2.0;
    }

    score
}

fn should_auto_route_to_executor(
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
    char_count >= auto_executor_length_threshold()
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

    // Phase 1: Legacy hardcoded routing (executor vs build) for backward compatibility
    let legacy_target = if should_auto_route_to_executor(&intent, question) {
        Some("executor")
    } else {
        None
    };

    // Phase 2: Context-aware routing using agent routing_tags and conversation history
    let history_entries = read_recent_history(app);
    let context_target = select_best_agent_by_context(agent_manifests, question, &history_entries);

    let fallback_agent_name = agents::find_agent_by_name(agent_manifests, "build")
        .filter(|agent| agent.is_primary() && !agent.disabled && !agent.hidden)
        .map(|agent| agent.name.clone())
        .unwrap_or_else(|| app.current_agent.clone());

    // Phase 3: Resolve final target — context-aware routing takes precedence
    // unless the legacy hard-coded routing explicitly triggers the executor agent.
    let target_agent_name: String = if let Some(name) = legacy_target {
        // Legacy executor trigger wins only if no better context match exists
        // with a significantly higher score
        if let Some(ctx_agent) = context_target {
            if agents::canonical_agent_name(&ctx_agent.name) == name {
                name.to_string()
            } else {
                // Let context-aware routing decide
                ctx_agent.name.clone()
            }
        } else {
            name.to_string()
        }
    } else if let Some(ctx_agent) = context_target {
        ctx_agent.name.clone()
    } else {
        // Fallback: return to the default build agent instead of keeping sticky context.
        fallback_agent_name
    };

    if app.current_agent == target_agent_name {
        return;
    }

    let Some(agent) = agents::find_agent_by_name(agent_manifests, &target_agent_name) else {
        return;
    };
    if !agent.is_primary() || agent.disabled {
        return;
    }

    let old_agent = app.current_agent.clone();
    activate_primary_agent(app, agent);

    // Determine switch reason for logging
    let reason = if legacy_target.is_some() && legacy_target.unwrap() == target_agent_name {
        "complex-execution"
    } else if let Some(ctx) = context_target {
        if ctx.name == target_agent_name {
            "context-match"
        } else {
            "unknown"
        }
    } else {
        "fallback"
    };

    println!(
        "\n[Agent 自动切换: {} → {}] (原因: {})\n",
        old_agent, app.current_agent, reason
    );
}

/// Reads the most recent conversation history entries for the current session.
/// Used by context-aware routing to understand what the conversation has been about.
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
                    "{} servers, {} tools (config: {})",
                    mcp_report.server_count, mcp_report.tool_count, mcp_report.config_path
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
    use super::{maybe_auto_route_agent, should_auto_route_to_executor};
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
    use crate::ai::cli::ParsedCli;
    use crate::ai::driver::intent_recognition::{CoreIntent, IntentModifiers, UserIntent};
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
                    .join("config/intent/intent_model.json"),
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
    fn auto_routes_complex_execution_requests_to_executor() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let question = "帮我实现这个 agent 的自动执行能力，然后跑检查并修掉相关报错";
        assert!(should_auto_route_to_executor(&intent, question));
    }

    #[test]
    fn does_not_route_simple_concept_questions_to_executor() {
        let intent = UserIntent::new(CoreIntent::QueryConcept);
        assert!(!should_auto_route_to_executor(&intent, "Rust 的 crate 是什么？"));
    }

    #[test]
    fn does_not_route_search_queries_to_executor() {
        let intent = UserIntent {
            core: CoreIntent::RequestAction,
            modifiers: IntentModifiers {
                is_search_query: true,
                target_resource: Some("tool".to_string()),
                negation: false,
            },
        };
        assert!(!should_auto_route_to_executor(&intent, "帮我找几个调试工具"));
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
