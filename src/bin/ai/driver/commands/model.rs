use crate::ai::{model_names, models, provider::ReasoningEffort, types::App};

fn print_model_help() {
    println!("Model commands:");
    println!();
    println!("  /model                              list available models");
    println!("  /model current                      show current model & effort");
    println!("  /model <selector> [<question>...]   switch to a model");
    println!("                                      e.g. /model deepseek-v4-flash-opencode");
    println!(
        "                                      (text after the selector is asked on this turn)"
    );
    println!("  /model effort                       show current reasoning effort");
    println!("  /model effort <minimal|low|medium|high|xhigh|max>");
    println!("                                      override reasoning effort");
    println!("  /model effort off|none|auto         clear override (use model default)");
    println!("  /effort <level|off|auto>            standalone shortcut for /model effort");
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
                ReasoningEffort::None => "effort:none",
                ReasoningEffort::Minimal => "effort:minimal",
                ReasoningEffort::Low => "effort:low",
                ReasoningEffort::Medium => "effort:medium",
                ReasoningEffort::High => "effort:high",
                ReasoningEffort::XHigh => "effort:xhigh",
                ReasoningEffort::Max => "effort:max",
            }),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ");
        if flags.is_empty() {
            println!("  {} {}", mark, label);
        } else {
            println!(
                "  {} {} [platform:{} adapter:{} | {}]",
                mark,
                label,
                model_names::platform_label(model),
                model_names::adapter_slug(model.adapter),
                flags
            );
        }
    }
    println!();
}

/// Handle the reasoning-effort override, shared by `/model effort <level>` and the
/// standalone `/effort <level>` command. `arg` is everything after the subcommand:
///
/// - empty → show the current effective effort and override state
/// - `auto|clear|default|reset` → clear the override (use the model default)
/// - `off|none|no|false|disable|disabled` → force the effort field off entirely
/// - a tier name → set the override to that tier
///
/// Returns `true` because a recognized effort command always consumes the input line.
fn handle_effort_arg(app: &mut App, arg: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let arg = arg.trim();
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
                "Unknown reasoning effort '{}'. Allowed: minimal, low, medium, high, xhigh, max, off, auto.",
                arg
            );
        }
    }
    Ok(true)
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
    // `/effort <level>` (or `:effort <level>`) is a standalone shortcut for
    // `/model effort <level>`; everything after the word is the effort argument.
    if cmd == "effort" {
        return handle_effort_arg(app, &normalized[cmd.len()..]);
    }
    if cmd != "model" && cmd != "models" {
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
            println!("Platform: {}", model_names::platform_label(def));
            println!("Adapter: {}", model_names::adapter_slug(def.adapter));
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
        return handle_effort_arg(app, rest);
    }

    let raw = remainder;

    if raw.is_empty() {
        println!("missing model selector. try: /model <name-platform>");
        print_model_list(app);
        return Ok(true);
    }

    // 支持行内问题：`/model <selector> [<question>...]`，question 可直接换行跟在
    // selector 之后（与 `/skills <name>... <question>` 一致）。模型 selector 是
    // 单 token：先整体尝试命中（保留空格归一化 selector 的既有行为），未命中时取
    // 首个 token 作为 selector，其余文本作为本轮问题。
    let mut selector = raw;
    let mut inline_question: Option<String> = None;
    if let Some(first) = raw.split_whitespace().next()
        && model_names::find_by_identifier(raw).is_none()
        && model_names::find_by_identifier(first).is_some()
    {
        selector = first;
        let rest = raw[first.len()..].trim();
        if !rest.is_empty() {
            inline_question = Some(rest.to_string());
        }
    }

    let Some(model) = model_names::find_by_identifier(selector) else {
        println!("Model not found: {}", raw);
        print_model_list(app);
        return Ok(true);
    };

    let old_model = app.current_model.clone();
    let next_model = model_handle(model);
    let old_handle = model_names::find_by_identifier(&old_model)
        .map(model_handle)
        .unwrap_or_else(|| old_model.trim().to_string());
    if old_handle.eq_ignore_ascii_case(&next_model) {
        if let Some(question) = inline_question {
            // 模型未变但带了行内问题：问题仍照常送入本轮。
            app.forced_question = Some(question);
            return Ok(true);
        }
        println!(
            "Model unchanged: {}",
            models::model_display_label(&next_model)
        );
        return Ok(true);
    }

    app.current_model = next_model.clone();
    app.cli.model = Some(next_model.clone());
    println!(
        "Switched model: {} -> {}\nPlatform: {} | Adapter: {} | Capabilities: {}{}{}{}",
        models::model_display_label(&old_model),
        models::model_display_label(&next_model),
        model_names::platform_label(model),
        model_names::adapter_slug(model.adapter),
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
    if let Some(question) = inline_question {
        app.forced_question = Some(question);
    }
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
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::new(),
            current_model: crate::ai::model_names::all()
                .first()
                .map(|m| crate::ai::model_names::model_handle(m))
                .expect("model registry is empty"),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skills: Vec::new(),
            forced_skill_source: None,
            pending_skill_continuation: None,
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
            last_known_cached_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
            last_turn_interrupted: false,
            prune_marks: Default::default(),
            turn_reasoning_items: Default::default(),
            stale_patch_targets: Default::default(),
            tool_middlewares: Vec::new(),
            llm_middlewares: Vec::new(),
            hooks: Default::default(),
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

    #[test]
    fn model_command_with_inline_question_switches_and_forces_question() {
        let models = crate::ai::model_names::all();
        if models.len() < 2 {
            return;
        }
        let mut app = test_app();
        let target = crate::ai::model_names::model_handle(models[1]);
        let question = "帮我看看这段代码";

        let handled =
            try_handle_model_command(&mut app, &format!("/model {target} {question}")).unwrap();

        assert!(handled);
        assert_eq!(app.current_model, target);
        assert_eq!(app.cli.model.as_deref(), Some(target.as_str()));
        assert_eq!(app.forced_question.as_deref(), Some(question));
    }

    #[test]
    fn model_command_multiline_question() {
        // /model <selector>\n<question>：换行后的文本作为本轮问题
        let models = crate::ai::model_names::all();
        if models.len() < 2 {
            return;
        }
        let mut app = test_app();
        let target = crate::ai::model_names::model_handle(models[1]);

        let handled =
            try_handle_model_command(&mut app, &format!("/model {target}\n帮我检查最近的变更"))
                .unwrap();

        assert!(handled);
        assert_eq!(app.current_model, target);
        assert_eq!(app.forced_question.as_deref(), Some("帮我检查最近的变更"));
    }

    #[test]
    fn model_command_no_question_does_not_force() {
        let models = crate::ai::model_names::all();
        if models.len() < 2 {
            return;
        }
        let mut app = test_app();
        let target = crate::ai::model_names::model_handle(models[1]);

        let handled = try_handle_model_command(&mut app, &format!("/model {target}")).unwrap();

        assert!(handled);
        assert_eq!(app.current_model, target);
        assert!(app.forced_question.is_none());
    }

    #[test]
    fn model_command_unknown_selector_with_question_is_not_found() {
        let mut app = test_app();
        let original = app.current_model.clone();

        let handled =
            try_handle_model_command(&mut app, "/model no-such-model-xyz 帮我看看").unwrap();

        assert!(handled);
        assert_eq!(app.current_model, original);
        assert!(app.forced_question.is_none());
    }

    #[test]
    fn model_command_unchanged_model_with_question_still_forces_question() {
        let mut app = test_app();
        let current = app.current_model.clone();

        let handled =
            try_handle_model_command(&mut app, &format!("/model {current} 帮我看看")).unwrap();

        assert!(handled);
        assert_eq!(app.current_model, current);
        assert_eq!(app.forced_question.as_deref(), Some("帮我看看"));
    }

    #[test]
    fn models_alias_switches_model() {
        let models = crate::ai::model_names::all();
        if models.len() < 2 {
            return;
        }
        let mut app = test_app();
        let target = crate::ai::model_names::model_handle(models[1]);

        let handled = try_handle_model_command(&mut app, &format!("/models {target}")).unwrap();

        assert!(handled);
        assert_eq!(app.current_model, target);
        assert_eq!(app.cli.model.as_deref(), Some(target.as_str()));
    }

    #[test]
    fn models_alias_with_inline_question_forces_question() {
        let models = crate::ai::model_names::all();
        if models.len() < 2 {
            return;
        }
        let mut app = test_app();
        let target = crate::ai::model_names::model_handle(models[1]);
        let question = "帮我看看这段代码";

        let handled =
            try_handle_model_command(&mut app, &format!("/models {target} {question}")).unwrap();

        assert!(handled);
        assert_eq!(app.current_model, target);
        assert_eq!(app.forced_question.as_deref(), Some(question));
    }

    #[test]
    fn effort_command_sets_override() {
        let mut app = test_app();

        let handled = try_handle_model_command(&mut app, "/effort high").unwrap();

        assert!(handled);
        assert_eq!(
            app.cli.reasoning_effort_override,
            Some(Some(ReasoningEffort::High))
        );
    }

    #[test]
    fn effort_command_off_disables() {
        let mut app = test_app();

        let handled = try_handle_model_command(&mut app, "/effort off").unwrap();

        assert!(handled);
        assert_eq!(app.cli.reasoning_effort_override, Some(None));
    }

    #[test]
    fn effort_command_auto_clears_override() {
        let mut app = test_app();
        try_handle_model_command(&mut app, "/effort high").unwrap();

        let handled = try_handle_model_command(&mut app, "/effort auto").unwrap();

        assert!(handled);
        assert_eq!(app.cli.reasoning_effort_override, None);
    }

    #[test]
    fn effort_command_without_arg_keeps_override() {
        let mut app = test_app();

        let handled = try_handle_model_command(&mut app, "/effort").unwrap();

        assert!(handled);
        assert_eq!(app.cli.reasoning_effort_override, None);
    }

    #[test]
    fn effort_command_unknown_value_keeps_override() {
        let mut app = test_app();

        let handled = try_handle_model_command(&mut app, "/effort banana").unwrap();

        assert!(handled);
        assert_eq!(app.cli.reasoning_effort_override, None);
    }

    #[test]
    fn model_effort_still_works_via_model_command() {
        let mut app = test_app();

        let handled = try_handle_model_command(&mut app, "/model effort high").unwrap();

        assert!(handled);
        assert_eq!(
            app.cli.reasoning_effort_override,
            Some(Some(ReasoningEffort::High))
        );
    }
}
