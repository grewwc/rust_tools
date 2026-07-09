    use super::{ToolRegistration, ToolSpec};
    use super::{
        AgentTeamMemberSpec, AgentTeamOperation, AsyncTaskEntry, InheritOptions, OsTaskGoal,
        SelectedSubagent, StoredTaskResult, TASK_REGISTRY, WaitManySource,
        append_current_process_cancel_source, build_agent_team_prompt,
        build_agent_team_selection_prompt, build_selection_explanation, encode_os_task_goal,
        epoll_wait_many, epoll_wait_many_channels, format_task_result, is_encoded_task_goal,
        parse_agent_team_members, prepare_subagent_task, remove_task_entry,
        resolve_agent_team_model_override, select_subagent, wait_sources_for_channel_and_futex,
        with_task_entry_by_pid,
    };
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
    use crate::ai::cli::ParsedCli;
    use crate::ai::driver::runtime_ctx::{DRIVER_CTX, DriverContext};
    use crate::ai::mcp::McpClient;
    use crate::ai::types::{App, AppConfig};
    use serde_json::Value;
    use aios_kernel::{
        kernel::{EventId, KernelInternal, Syscall, WaitPolicy},
        local::LocalOS,
        primitives::{FutexAddr, FutexOps, IpcOps},
    };
    use std::sync::{Arc, atomic::AtomicBool};
    use std::time::Instant;

    fn manifest(name: &str, description: &str, mode: AgentMode) -> AgentManifest {
        AgentManifest {
            name: name.to_string(),
            description: description.to_string(),
            mode,
            model: None,
            temperature: None,
            max_steps: None,
            prompt: String::new(),
            system_prompt: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            disable_mcp_tools: false,
            routing_tags: Vec::new(),
            model_tier: Some(AgentModelTier::Standard),
            disabled: false,
            hidden: false,
            color: None,
            source_path: None,
        }
    }

    fn test_app_with_model(current_model: String) -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                base_history_file: std::path::PathBuf::new(),
                history_file: std::path::PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 12000,
                history_keep_last: 8,
                history_summary_max_chars: 4000,
                intent_model: None,
                agent_route_model_path: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
            },
            session_id: String::new(),
            session_history_file: std::path::PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::new(),
            current_model,
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
            observers: Vec::new(),
            last_known_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
        }
    }

    #[test]
    fn auto_select_prefers_explore_for_codebase_investigation() {
        let mut build = manifest("build", "Main build agent", AgentMode::Primary);
        build.routing_tags = vec!["implement".to_string(), "fix".to_string()];
        build.model_tier = Some(AgentModelTier::Heavy);
        let mut explore = manifest(
            "explore",
            "Read-only codebase exploration agent",
            AgentMode::Subagent,
        );
        explore.routing_tags = vec![
            "find".to_string(),
            "search".to_string(),
            "read-only".to_string(),
            "understand".to_string(),
        ];
        explore.model_tier = Some(AgentModelTier::Light);
        let mut review = manifest("review", "Read-only review agent", AgentMode::Subagent);
        review.routing_tags = vec!["review".to_string(), "audit".to_string()];

        let all_agents = vec![build, explore, review];

        let selected = select_subagent(
            &all_agents,
            None,
            "Locate routing logic",
            "Find where automatic agent routing happens and summarize the files involved.",
        )
        .unwrap();

        assert_eq!(selected.agent.name, "explore");
        assert!(selected.auto_selected);
        assert!(!selected.matched_tags.is_empty());
    }

    #[test]
    fn prepare_subagent_task_inherits_parent_model_without_auto_fallback() {
        let parent_model = crate::ai::model_names::all()
            .first()
            .map(|model| crate::ai::model_names::model_handle(model))
            .expect("models.json must contain at least one model");
        let mut explore = manifest(
            "explore",
            "Read-only codebase exploration agent",
            AgentMode::Subagent,
        );
        explore.routing_tags = vec!["find".to_string(), "search".to_string()];
        let ctx = DriverContext::new(
            test_app_with_model(parent_model.clone()),
            Arc::new(std::sync::Mutex::new(McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(vec![explore]),
        );
        let args = serde_json::json!({
            "description": "Locate task tool",
            "prompt": "Find where task spawning is implemented.",
            "agent": "explore"
        });

        let prepared = DRIVER_CTX
            .sync_scope(ctx, || prepare_subagent_task(&args))
            .unwrap();

        assert_eq!(prepared.model, parent_model);
        assert!(!prepared.is_model_auto_selected);
        assert!(prepared.auto_model_fallback.is_none());
        assert!(
            prepared
                .selection_explanation
                .contains("model_reason=inherited parent agent current model")
        );
    }

    #[test]
    fn explicit_primary_agent_is_rejected_for_task_tool() {
        let mut build = manifest("build", "Main build agent", AgentMode::Primary);
        build.routing_tags = vec!["implement".to_string()];
        let mut explore = manifest(
            "explore",
            "Read-only codebase exploration agent",
            AgentMode::Subagent,
        );
        explore.routing_tags = vec!["find".to_string(), "search".to_string()];
        let all_agents = vec![build, explore];

        let err = select_subagent(&all_agents, Some("build"), "Inspect code", "Look up files")
            .unwrap_err();

        assert!(err.contains("not a subagent"));
    }

    #[test]
    fn routing_tags_drive_auto_selection_without_name_special_cases() {
        let mut explore = manifest(
            "navigator",
            "Read-only codebase exploration agent",
            AgentMode::Subagent,
        );
        explore.routing_tags = vec![
            "find".to_string(),
            "search".to_string(),
            "locate".to_string(),
        ];
        explore.model_tier = Some(AgentModelTier::Light);

        let mut review = manifest("critic", "Code review agent", AgentMode::Subagent);
        review.routing_tags = vec!["review".to_string(), "audit".to_string()];
        let all_agents = vec![explore, review];

        let selected = select_subagent(
            &all_agents,
            None,
            "Find handler",
            "Search the codebase and locate where the request handler is defined.",
        )
        .unwrap();

        assert_eq!(selected.agent.name, "navigator");
    }

    #[test]
    fn selection_explanation_mentions_quality_tier_for_auto_model_choice() {
        // 之前这里硬编码 "qwen3-max"；该模型已经从 models.json 移除。
        // 改为从真实条目中找一个 Alibaba+flagship 的模型，确保解释里出现
        // "flagship" 和 "alibaba" 这两个 tier/provider 关键字。
        use crate::ai::provider::{ApiProvider, ModelQualityTier};
        let model = crate::ai::model_names::all()
            .iter()
            .find(|m| {
                m.provider == ApiProvider::Alibaba && m.quality_tier == ModelQualityTier::Flagship
            })
            .map(|m| m.name.clone());
        let Some(model) = model else {
            eprintln!(
                "[test] skipping selection_explanation_mentions_quality_tier_for_auto_model_choice: \
                 no Alibaba+Flagship model present in models.json"
            );
            return;
        };

        let agent = manifest("build", "Main build agent", AgentMode::Subagent);
        let selected = SelectedSubagent {
            agent: &agent,
            auto_selected: true,
            matched_tags: vec!["implement".to_string(), "fix".to_string()],
            score: 48,
        };

        let explanation = build_selection_explanation(&selected, &model, None, false);

        assert!(explanation.contains("routing_tags [implement, fix]"));
        assert!(explanation.contains("quality_tier"));
        assert!(explanation.contains("flagship"));
        assert!(explanation.contains("alibaba"));
    }

    #[test]
    fn selection_explanation_mentions_explicit_overrides() {
        let agent = manifest(
            "explore",
            "Read-only codebase exploration agent",
            AgentMode::Subagent,
        );
        let selected = SelectedSubagent {
            agent: &agent,
            auto_selected: false,
            matched_tags: Vec::new(),
            score: 0,
        };

        let explanation = build_selection_explanation(&selected, "gpt-4o", Some("gpt-4o"), false);

        assert!(explanation.contains("explicit agent override"));
        assert!(explanation.contains("explicit model override"));
    }

    #[test]
    fn blank_model_override_is_treated_as_auto_selection() {
        let agent = manifest(
            "explore",
            "Read-only codebase exploration agent",
            AgentMode::Subagent,
        );
        let selected = SelectedSubagent {
            agent: &agent,
            auto_selected: true,
            matched_tags: Vec::new(),
            score: 0,
        };

        let explanation =
            build_selection_explanation(&selected, "deepseek-v4-flash", Some(" "), false);

        assert!(explanation.contains("auto-selected"));
        assert!(!explanation.contains("explicit model override"));
    }

    #[test]
    fn agent_team_start_requires_multiple_members() {
        let args = serde_json::json!({
            "operation": "start",
            "goal": "Decide the safest implementation",
            "members": [
                { "role": "reviewer" }
            ]
        });

        let err = parse_agent_team_members(&args, AgentTeamOperation::Start).unwrap_err();

        assert!(err.contains("requires at least 2"));
    }

    #[test]
    fn agent_team_challenge_prompt_uses_parent_mediated_transcript() {
        let member = AgentTeamMemberSpec {
            role: "skeptic".to_string(),
            prompt: "Focus on concurrency and missing evidence.".to_string(),
            agent: None,
            model: None,
        };

        let prompt = build_agent_team_prompt(
            AgentTeamOperation::Challenge,
            "Review the design",
            &member,
            "member A: looks safe\nmember B: maybe races",
        );

        assert!(prompt.contains("Do not wait for direct messages from peer agents"));
        assert!(prompt.contains("parent agent coordinates"));
        assert!(prompt.contains("Phase: challenge"));
        assert!(prompt.contains("member A: looks safe"));
        assert!(prompt.contains("concurrency"));
    }

    #[test]
    fn agent_team_selection_prompt_stays_cost_aware() {
        let member = AgentTeamMemberSpec {
            role: "skeptic".to_string(),
            prompt: "Focus on assumptions and risks.".to_string(),
            agent: None,
            model: None,
        };

        let selection_prompt =
            build_agent_team_selection_prompt(AgentTeamOperation::Challenge, &member);

        assert!(selection_prompt.contains("agent_team phase: challenge"));
        assert!(selection_prompt.contains("role: skeptic"));
        assert!(!selection_prompt.contains("Prior team transcript"));
        assert!(!selection_prompt.contains("Team goal"));
    }

    #[test]
    fn agent_team_model_override_requires_exact_match() {
        let err = resolve_agent_team_model_override("__missing_model_for_team__").unwrap_err();

        assert!(err.contains("exact model key/name"));
        assert!(err.contains("expensive fallback"));
    }

    #[test]
    fn epoll_wait_many_channels_returns_ready_without_suspending() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let channel = os.channel_create(Some(root), 1, "task-ready".to_string());
        os.channel_send(Some(root), channel, "payload".to_string())
            .unwrap();

        let wait = epoll_wait_many_channels(
            &mut os,
            "task_wait:test-ready",
            &[channel.raw()],
            WaitPolicy::All,
            None,
        )
        .unwrap();

        assert_eq!(
            wait.ready_sources,
            vec![WaitManySource::Channel(channel.raw())]
        );
        assert!(wait.pending_sources.is_empty());
        assert!(!wait.suspended);
        assert!(wait.event_ids.is_empty());
    }

    #[test]
    fn epoll_wait_many_channels_preserves_all_wait_suspension() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let channel_a = os.channel_create(Some(root), 1, "task-a".to_string());
        let channel_b = os.channel_create(Some(root), 1, "task-b".to_string());

        let wait = epoll_wait_many_channels(
            &mut os,
            "task_wait:test-suspend",
            &[channel_a.raw(), channel_b.raw()],
            WaitPolicy::All,
            None,
        )
        .unwrap();

        assert!(wait.ready_sources.is_empty());
        assert_eq!(
            wait.pending_sources,
            vec![
                WaitManySource::Channel(channel_a.raw()),
                WaitManySource::Channel(channel_b.raw())
            ]
        );
        assert_eq!(wait.event_ids.len(), 2);
        assert!(wait.suspended);
        assert!(os.current_process_id().is_none());
    }

    #[test]
    fn epoll_wait_many_supports_mixed_ready_sources() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let channel = os.channel_create(Some(root), 1, "mixed-channel".to_string());
        let futex = os.futex_create(0, "mixed-futex".to_string());
        let event = EventId::new(77);
        os.channel_send(Some(root), channel, "payload".to_string())
            .unwrap();
        let _ = os.futex_store(futex, 1);
        os.notify_events_completed(&[event]);

        let wait = epoll_wait_many(
            &mut os,
            "mixed-ready",
            &[
                WaitManySource::Channel(channel.raw()),
                WaitManySource::Futex {
                    addr: futex,
                    expected: 0,
                },
                WaitManySource::Event(event),
            ],
            WaitPolicy::Any,
            None,
        )
        .unwrap();

        assert_eq!(
            wait.ready_sources,
            vec![
                WaitManySource::Channel(channel.raw()),
                WaitManySource::Futex {
                    addr: futex,
                    expected: 0
                },
                WaitManySource::Event(event),
            ]
        );
        assert!(wait.pending_sources.is_empty());
        assert!(!wait.suspended);
    }

    #[test]
    fn epoll_wait_many_supports_mixed_all_wait_suspension() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let channel = os.channel_create(Some(root), 1, "mixed-channel".to_string());
        let futex = os.futex_create(0, "mixed-futex".to_string());
        let event = EventId::new(88);
        os.notify_events_completed(&[event]);

        let wait = epoll_wait_many(
            &mut os,
            "mixed-all",
            &[
                WaitManySource::Channel(channel.raw()),
                WaitManySource::Futex {
                    addr: futex,
                    expected: 0,
                },
                WaitManySource::Event(event),
            ],
            WaitPolicy::All,
            None,
        )
        .unwrap();

        assert_eq!(wait.ready_sources, vec![WaitManySource::Event(event)]);
        assert_eq!(
            wait.pending_sources,
            vec![
                WaitManySource::Channel(channel.raw()),
                WaitManySource::Futex {
                    addr: futex,
                    expected: 0
                },
            ]
        );
        assert_eq!(wait.event_ids.len(), 2);
        assert!(wait.suspended);
        assert!(os.current_process_id().is_none());
    }

    #[test]
    fn task_wait_low_level_wait_wakes_on_task_result_even_with_cancel_source() {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let channel = os.channel_create(Some(root), 1, "task-result".to_string());
        let futex = os.futex_create(0, "task-complete".to_string());
        let mut sources =
            wait_sources_for_channel_and_futex(&mut os, channel.raw(), Some(futex)).unwrap();
        append_current_process_cancel_source(&mut os, &mut sources).unwrap();

        let wait = epoll_wait_many(
            &mut os,
            "task_wait:with-cancel-source",
            &sources,
            WaitPolicy::Any,
            None,
        )
        .unwrap();

        assert!(wait.suspended);
        assert!(os.current_process_id().is_none());

        os.channel_send(Some(root), channel, "payload".to_string())
            .unwrap();

        let root_proc = os.get_process(root).unwrap();
        assert_eq!(root_proc.state, aios_kernel::kernel::ProcessState::Ready);
    }

    #[test]
    fn task_wait_formats_empty_subagent_result_explicitly() {
        let entry = AsyncTaskEntry {
            pid: 42,
            result_channel_id: 1,
            completion_futex_addr: FutexAddr(1),
            description: "verify behavior".to_string(),
            agent_name: "explore".to_string(),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: true,
            auto_model_fallback: None,
            selection_explanation: "model_reason=auto-selected".to_string(),
            inherit: InheritOptions::default(),
            started_at: Instant::now(),
        };
        let result = StoredTaskResult {
            status: "failed".to_string(),
            output: String::new(),
            error: Some("request timed out waiting for response headers".to_string()),
        };

        let output = format_task_result(&entry, result);

        assert!(output.contains("FAILED"));
        assert!(output.contains("Error: request timed out waiting for response headers"));
        assert!(output.contains("(subagent did not produce any final assistant text)"));
    }

    #[test]
    fn encoded_task_goal_prefix_is_detectable_without_decoding() {
        let goal = encode_os_task_goal(&OsTaskGoal {
            task_id: "task_test".to_string(),
            result_channel_id: 7,
            completion_futex_addr: 9,
            description: "inspect".to_string(),
            prompt: "look around".to_string(),
            agent_name: "explore".to_string(),
            model: "qwen3.7-max-alibaba".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit agent/model override".to_string(),
        })
        .unwrap();

        assert!(is_encoded_task_goal(&goal));
        assert!(!is_encoded_task_goal("plain foreground goal"));
    }

    #[test]
    fn task_entry_can_be_looked_up_by_pid() {
        let task_id = format!("task_test_{}", uuid::Uuid::new_v4().simple());
        {
            let mut registry = TASK_REGISTRY.lock().unwrap();
            registry.insert(
                task_id.clone(),
                AsyncTaskEntry {
                    pid: 4242,
                    result_channel_id: 11,
                    completion_futex_addr: FutexAddr(13),
                    description: "inspect".to_string(),
                    agent_name: "explore".to_string(),
                    model: "qwen3.7-max-alibaba".to_string(),
                    is_model_auto_selected: false,
                    auto_model_fallback: None,
                    selection_explanation: "explicit override".to_string(),
                    inherit: InheritOptions::default(),
                    started_at: Instant::now(),
                },
            );
        }

        let looked_up = with_task_entry_by_pid(4242, |entry| {
            (entry.agent_name.clone(), entry.model.clone(), entry.result_channel_id)
        });

        assert_eq!(
            looked_up,
            Some((
                "explore".to_string(),
                "qwen3.7-max-alibaba".to_string(),
                11
            ))
        );

        let _ = remove_task_entry(&task_id);
    }

fn params_question() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user."
            },
            "header": {
                "type": "string",
                "description": "Very short label (max 30 chars) for context."
            },
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Display text for this option (1-5 words)."
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief explanation of what this option means."
                        }
                    },
                    "required": ["label", "description"]
                },
                "description": "Available choices for the user."
            },
            "multiple": {
                "type": "boolean",
                "description": "Allow selecting multiple choices (default: false)."
            }
        },
        "required": ["question", "header", "options"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "question",
        description: "Ask the user questions during execution. Use this to gather preferences, clarify ambiguous instructions, get decisions on implementation choices, or offer choices about direction. Returns the user's selected answer(s).",
        parameters: params_question,
        execute: execute_question,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

pub(crate) fn execute_question(args: &Value) -> Result<String, String> {
    let question = args["question"]
        .as_str()
        .ok_or("Missing 'question' parameter")?;

    let header = args["header"]
        .as_str()
        .ok_or("Missing 'header' parameter")?;

    let options = args["options"]
        .as_array()
        .ok_or("Missing 'options' parameter (must be an array)")?;

    if options.is_empty() {
        return Err("options array cannot be empty".to_string());
    }

    let multiple = args["multiple"].as_bool().unwrap_or(false);

    println!("\n--- Question: {} ---", header);
    println!("{}", question);
    println!();

    for (i, opt) in options.iter().enumerate() {
        let label = opt["label"].as_str().unwrap_or("?");
        let desc = opt["description"].as_str().unwrap_or("");
        println!("  {}. {} - {}", i + 1, label, desc);
    }
    println!();

    if multiple {
        println!("Enter option numbers separated by commas (or type your own answer):");
    } else {
        println!("Enter option number (or type your own answer):");
    }

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|e| format!("Failed to read input: {}", e))?;

    let input = input.trim();

    if input.is_empty() {
        return Err("No answer provided".to_string());
    }

    if multiple {
        let selections: Vec<&str> = input.split(',').map(|s| s.trim()).collect();
        let mut selected_labels = Vec::new();

        for sel in &selections {
            if let Ok(idx) = sel.parse::<usize>() {
                if idx > 0 && idx <= options.len() {
                    if let Some(label) = options[idx - 1]["label"].as_str() {
                        selected_labels.push(label.to_string());
                    }
                } else {
                    return Ok(format!("[User answer] {}", input));
                }
            } else {
                return Ok(format!("[User answer] {}", input));
            }
        }

        Ok(format!("[User selected] {}", selected_labels.join(", ")))
    } else {
        if let Ok(idx) = input.parse::<usize>() {
            if idx > 0 && idx <= options.len() {
                if let Some(label) = options[idx - 1]["label"].as_str() {
                    return Ok(format!("[User selected] {}", label));
                }
            }
        }

        Ok(format!("[User answer] {}", input))
    }
}
