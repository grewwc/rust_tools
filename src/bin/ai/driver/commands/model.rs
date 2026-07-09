use crate::ai::{model_names, models, provider::ReasoningEffort, types::App};

fn print_model_help() {
    println!("Model commands:");
    println!();
    println!("  /model                              list available models");
    println!("  /model current                      show current model & effort");
    println!("  /model <selector>                   switch to a model");
    println!("                                      e.g. /model deepseek-v4-flash-opencode");
    println!("  /model effort                       show current reasoning effort");
    println!("  /model effort <minimal|low|medium|high>");
    println!("                                      override reasoning effort");
    println!("  /model effort off|none|auto         clear override (use model default)");
    println!();
}

/// 计算当前生效的推理强度（与 [`request::resolve_reasoning_effort`] 同语义，
/// 但本模块不依赖 request.rs 内部结构，所以在这里复刻一份纯查询逻辑）。
fn effective_effort(app: &App, model: &str) -> Option<ReasoningEffort> {
    if let Some(override_value) = app.cli.reasoning_effort_override.as_ref() {
        return *override_value;
    }
    models::default_reasoning_effort(model)
}

fn format_effort(effort: Option<ReasoningEffort>) -> &'static str {
    match effort {
        Some(e) => e.as_str(),
        None => "auto",
    }
}

fn model_handle(model: &model_names::ModelDef) -> String {
    model_names::model_handle(model)
}

fn print_model_list(app: &App) {
    println!(
        "Current model: {}",
        models::model_display_label(&app.current_model)
    );
    println!(
        "Reasoning effort: {} (override: {})",
        format_effort(effective_effort(app, &app.current_model)),
        match app.cli.reasoning_effort_override {
            None => "none".to_string(),
            Some(None) => "off".to_string(),
            Some(Some(e)) => e.as_str().to_string(),
        }
    );
    println!();
    println!("Available models:");
    let current = model_names::find_by_identifier(&app.current_model)
        .map(model_handle)
        .unwrap_or_else(|| app.current_model.trim().to_string())
        .to_ascii_lowercase();
    for model in model_names::all() {
        let handle = model_handle(model);
        let mark = if handle.eq_ignore_ascii_case(&current) {
            ">>>"
        } else {
            "   "
        };
        let label = models::model_display_label(&handle);
        let flags = [
            model.is_vl.then_some("vl"),
            model.search_enabled.then_some("search"),
            model.tools_default_enabled.then_some("tools"),
            model.enable_thinking.then_some("thinking"),
            model.reasoning_effort.map(|e| match e {
                ReasoningEffort::Minimal => "effort:minimal",
                ReasoningEffort::Low => "effort:low",
                ReasoningEffort::Medium => "effort:medium",
                ReasoningEffort::High => "effort:high",
            }),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ");
        if flags.is_empty() {
            println!("  {} {}", mark, label);
        } else {
            println!("  {} {} [{:?} | {}]", mark, label, model.provider, flags);
        }
    }
    println!();
}

pub fn try_handle_model_command(
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
    if cmd != "model" {
        return Ok(false);
    }

    let remainder = normalized[cmd.len()..].trim();
    if remainder.is_empty() || matches!(remainder, "list" | "ls") {
        print_model_list(app);
        return Ok(true);
    }
    if matches!(remainder, "help" | "h") {
        print_model_help();
        return Ok(true);
    }
    if matches!(remainder, "current" | "cur") {
        println!(
            "Current model: {}",
            models::model_display_label(&app.current_model)
        );
        if let Some(def) = model_names::find_by_identifier(&app.current_model) {
            println!("Provider: {:?}", def.provider);
            println!("Quality tier: {:?}", def.quality_tier);
            println!("Selector: {}", model_names::model_handle(def));
            if !def.aliases.is_empty() {
                println!("Aliases: {}", def.aliases.join(", "));
            }
            println!("Model name: {}", def.name);
            println!("Vision: {}", if def.is_vl { "yes" } else { "no" });
            println!("Search: {}", if def.search_enabled { "yes" } else { "no" });
            println!(
                "Tools default enabled: {}",
                if def.tools_default_enabled {
                    "yes"
                } else {
                    "no"
                }
            );
            println!(
                "Thinking: {}",
                if def.enable_thinking { "yes" } else { "no" }
            );
            println!(
                "Reasoning effort: {} (model default: {}, override: {})",
                format_effort(effective_effort(app, &app.current_model)),
                format_effort(def.reasoning_effort),
                match app.cli.reasoning_effort_override {
                    None => "none".to_string(),
                    Some(None) => "off".to_string(),
                    Some(Some(e)) => e.as_str().to_string(),
                }
            );
            println!(
                "Endpoint: {}",
                models::endpoint_for_model(&model_handle(def), "")
            );
        }
        return Ok(true);
    }

    // /model effort [<value>]
    if let Some(rest) = remainder.strip_prefix("effort") {
        let arg = rest.trim();
        if arg.is_empty() {
            println!(
                "Reasoning effort: {} (override: {})",
                format_effort(effective_effort(app, &app.current_model)),
                match app.cli.reasoning_effort_override {
                    None => "none".to_string(),
                    Some(None) => "off".to_string(),
                    Some(Some(e)) => e.as_str().to_string(),
                }
            );
            return Ok(true);
        }
        match arg.to_ascii_lowercase().as_str() {
            "auto" | "clear" | "default" | "reset" => {
                app.cli.reasoning_effort_override = None;
                println!(
                    "Cleared reasoning_effort override; now using model default ({}).",
                    format_effort(models::default_reasoning_effort(&app.current_model))
                );
                return Ok(true);
            }
            "off" | "none" | "no" | "false" | "disable" | "disabled" => {
                app.cli.reasoning_effort_override = Some(None);
                println!("Reasoning effort disabled (no field will be sent).");
                return Ok(true);
            }
            _ => {}
        }
        match ReasoningEffort::parse(arg) {
            Some(level) => {
                app.cli.reasoning_effort_override = Some(Some(level));
                println!("Reasoning effort overridden: {}", level.as_str());
            }
            None => {
                println!(
                    "Unknown reasoning effort '{}'. Allowed: minimal, low, medium, high, off, auto.",
                    arg
                );
            }
        }
        return Ok(true);
    }

    let target = remainder;

    if target.is_empty() {
        println!("missing model selector. try: /model <name-provider>");
        print_model_list(app);
        return Ok(true);
    }

    let Some(model) = model_names::find_by_identifier(target) else {
        println!("Model not found: {}", target);
        print_model_list(app);
        return Ok(true);
    };

    let old_model = app.current_model.clone();
    let next_model = model_handle(model);
    let old_handle = model_names::find_by_identifier(&old_model)
        .map(model_handle)
        .unwrap_or_else(|| old_model.trim().to_string());
    if old_handle.eq_ignore_ascii_case(&next_model) {
        println!(
            "Model unchanged: {}",
            models::model_display_label(&next_model)
        );
        return Ok(true);
    }

    app.current_model = next_model.clone();
    app.cli.model = Some(next_model.clone());
    println!(
        "Switched model: {} -> {}",
        models::model_display_label(&old_model),
        models::model_display_label(&next_model)
    );
    println!("Provider: {:?}", model.provider);
    println!(
        "Capabilities: {}{}{}{}",
        if model.is_vl { "vl " } else { "" },
        if model.search_enabled { "search " } else { "" },
        if model.tools_default_enabled {
            "tools "
        } else {
            ""
        },
        if model.enable_thinking {
            "thinking"
        } else {
            ""
        },
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{cli::ParsedCli, types::AppConfig};
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    fn test_app() -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                base_history_file: PathBuf::new(),
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 12000,
                history_keep_last: 8,
                history_summary_max_chars: 4000,
                intent_model: None,
                agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::new(),
            current_model: crate::ai::model_names::all()
                .first()
                .map(|m| crate::ai::model_names::model_handle(m))
                .expect("models.json is empty"),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skill: None,
            forced_question: None,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            prompt_editor: None,
            agent_context: None,
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(
                crate::ai::driver::thinking::ThinkingOrchestrator::new(),
            )],
            last_known_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
        }
    }

    #[test]
    fn model_command_switches_current_model() {
        let models = crate::ai::model_names::all();
        if models.len() < 2 {
            return;
        }
        let mut app = test_app();
        let target = crate::ai::model_names::model_handle(models[1]);

        let handled = try_handle_model_command(&mut app, &format!("/model {}", target)).unwrap();

        assert!(handled);
        assert_eq!(app.current_model, target);
        assert_eq!(app.cli.model.as_deref(), Some(app.current_model.as_str()));
    }

    #[test]
    fn model_command_does_not_accept_removed_action_aliases() {
        let models = crate::ai::model_names::all();
        if models.len() < 2 {
            return;
        }
        let mut app = test_app();
        let original = app.current_model.clone();
        let target = crate::ai::model_names::model_handle(models[1]);

        let handled =
            try_handle_model_command(&mut app, &format!("/model use {}", target)).unwrap();

        assert!(handled);
        assert_eq!(app.current_model, original);
        assert!(app.cli.model.is_none());
    }
}
