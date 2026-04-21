use crate::ai::{model_names, models, types::App};

fn print_model_help() {
    println!("Model commands:");
    println!();
    println!("  /model                     list available models");
    println!("  /model current             show current model");
    println!("  /model <name>              switch to a model");
    println!();
}

fn print_model_list(app: &App) {
    println!("Current model: {}", app.current_model);
    println!();
    println!("Available models:");
    for model in model_names::all() {
        let mark = if model.name == app.current_model {
            ">>>"
        } else {
            "   "
        };
        let flags = [
            model.is_vl.then_some("vl"),
            model.search_enabled.then_some("search"),
            model.tools_default_enabled.then_some("tools"),
            model.enable_thinking.then_some("thinking"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ");
        if flags.is_empty() {
            println!("  {} {}", mark, model.name);
        } else {
            println!(
                "  {} {} [{:?} | {}]",
                mark, model.name, model.provider, flags
            );
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
        println!("Current model: {}", app.current_model);
        if let Some(def) = model_names::find_by_name(&app.current_model) {
            println!("Provider: {:?}", def.provider);
            println!("Quality tier: {:?}", def.quality_tier);
            println!("Vision: {}", if def.is_vl { "yes" } else { "no" });
            println!("Search: {}", if def.search_enabled { "yes" } else { "no" });
            println!(
                "Tools default enabled: {}",
                if def.tools_default_enabled { "yes" } else { "no" }
            );
            println!("Thinking: {}", if def.enable_thinking { "yes" } else { "no" });
            println!("Endpoint: {}", models::endpoint_for_model(&def.name, ""));
        }
        return Ok(true);
    }

    let target = if let Some(rest) = remainder.strip_prefix("use ") {
        rest.trim()
    } else if let Some(rest) = remainder.strip_prefix("select ") {
        rest.trim()
    } else if let Some(rest) = remainder.strip_prefix("switch ") {
        rest.trim()
    } else {
        remainder
    };

    if target.is_empty() {
        println!("missing model name. try: /model <name>");
        print_model_list(app);
        return Ok(true);
    }

    let Some(model) = model_names::find_by_name(target) else {
        println!("Model not found: {}", target);
        print_model_list(app);
        return Ok(true);
    };

    let old_model = app.current_model.clone();
    if old_model.eq_ignore_ascii_case(&model.name) {
        println!("Model unchanged: {}", model.name);
        return Ok(true);
    }

    app.current_model = model.name.clone();
    app.cli.model = Some(model.name.clone());
    println!("Switched model: {} -> {}", old_model, model.name);
    println!("Provider: {:?}", model.provider);
    println!(
        "Capabilities: {}{}{}{}",
        if model.is_vl { "vl " } else { "" },
        if model.search_enabled { "search " } else { "" },
        if model.tools_default_enabled { "tools " } else { "" },
        if model.enable_thinking { "thinking" } else { "" },
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
            current_model: crate::ai::model_names::all()
                .first()
                .expect("models.json is empty")
                .name
                .clone(),
            current_agent: "build".to_string(),
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
            agent_context: None,
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(crate::ai::driver::thinking::ThinkingOrchestrator::new())],
        }
    }

    #[test]
    fn model_command_switches_current_model() {
        let models = crate::ai::model_names::all();
        if models.len() < 2 {
            return;
        }
        let mut app = test_app();
        let target = models[1].name.clone();

        let handled = try_handle_model_command(&mut app, &format!("/model {}", target)).unwrap();

        assert!(handled);
        assert_eq!(app.current_model, target);
        assert_eq!(app.cli.model.as_deref(), Some(app.current_model.as_str()));
    }
}
