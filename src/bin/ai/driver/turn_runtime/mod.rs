// =============================================================================
// AIOS Turn Runtime - Core Execution Engine
// =============================================================================
// This module handles the core execution loop where the LLM repeatedly calls tools.
//
// The turn execution follows this flow:
//   1. Prepare: Build messages, select skills, initial request
//   2. Iterate: LLM generates response with potential tool calls
//   3. Execute: Run each tool call and collect results
//   4. Finalize: Build final response and persist history
//
// Submodules:
//   - prepare: Prepare turn (build messages, select skills)
//   - iteration: Execute one LLM turn (call LLM, execute tools)
//   - orchestrator: run_turn() - main turn coordination
//   - tool_result: Handle tool execution results
//   - finalize: Build final response, persist history
//   - types: Outcome types (TurnOutcome, etc)
//   - debug: Hang/debug reporting
//   - persistence: SQLite history management
// =============================================================================

mod debug;
mod finalize;
mod iteration;
mod orchestrator;
mod persistence;
mod prepare;
mod tool_result;
mod types;

pub(super) use orchestrator::run_turn;
#[cfg(test)]
use persistence::persist_pending_turn_messages;
#[cfg(test)]
use tool_result::prepare_tool_result;
pub(super) use types::TurnOutcome;

const MAX_TOOL_RESULT_INLINE_CHARS: usize = 32_000;
const TOOL_OVERFLOW_PREVIEW_CHARS: usize = 800;
/// 中等大输出阈值：超过此值但未到 overflow 阈值的工具结果，对结构化的 read/grep/tree
/// 类工具走"头 + 关键命中 + 尾"的按行裁剪，避免完整 32KB 全部进上下文。
const MAX_TOOL_RESULT_LINE_TRIM_CHARS: usize = 8_000;

/// Mid-turn 渐进式压缩：messages 总字符数超过该阈值时，在 iteration loop 内
/// 复用跨 turn 压缩管线，避免单 turn 长链工具调用把上下文撑爆。
///
/// 阈值默认按 `app.config.history_max_chars` 动态计算（详见
/// [`mid_turn_compress_soft_threshold`] / [`mid_turn_compress_hard_threshold`]）。
/// 这两个常量仅作为 floor 兜底（防止用户把 history_max_chars 设得过小，
/// 导致 mid-turn 压缩在单条 tool 结果就被触发，反而不停 no-op）。
pub(in crate::ai::driver::turn_runtime) const MID_TURN_COMPRESS_SOFT_FLOOR: usize = 36_000;
/// Mid-turn LLM 摘要硬阈值 floor：经过无损/弱损压缩后仍超过该值，触发 LLM summary
/// 兜底（会调用一次模型，并显示 "🗜️ compressing context..." 状态行）。
pub(in crate::ai::driver::turn_runtime) const MID_TURN_COMPRESS_HARD_FLOOR: usize = 80_000;

/// 软阈值：min 36K，否则取 history_max_chars * 1.5。
/// history_max_chars 默认 120K，对应软阈值 180K。
pub(in crate::ai::driver::turn_runtime) fn mid_turn_compress_soft_threshold(
    history_max_chars: usize,
) -> usize {
    history_max_chars
        .saturating_mul(3)
        .saturating_div(2)
        .max(MID_TURN_COMPRESS_SOFT_FLOOR)
}

/// 硬阈值：min 80K，否则取 history_max_chars * 3.5。
/// history_max_chars 默认 120K，对应硬阈值 420K（远超模型 context window，
/// 实际硬阈值会被模型 4xx 之前的 normalize_messages_for_request 拦截）。
/// 但相对软阈值留出明显 gap，避免软阈值边界连续触发 LLM summary。
pub(in crate::ai::driver::turn_runtime) fn mid_turn_compress_hard_threshold(
    history_max_chars: usize,
) -> usize {
    history_max_chars
        .saturating_mul(7)
        .saturating_div(2)
        .max(MID_TURN_COMPRESS_HARD_FLOOR)
}
/// LLM 摘要兜底时保留尾部窗口的 user 起始轮数。早期超过此窗口的对话被压成
/// 一条 internal_note 摘要插入到尾部窗口前。
pub(in crate::ai::driver::turn_runtime) const MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS: usize = 2;
/// LLM 摘要文本的最大字符数。
pub(in crate::ai::driver::turn_runtime) const MID_TURN_LLM_SUMMARY_MAX_CHARS: usize = 4_000;
/// Mid-turn 压缩冷却：触发一次后至少间隔 N 轮才再次重判，避免在阈值附近徘徊
/// 时每轮都跑一次（实际无变化）。
pub(in crate::ai::driver::turn_runtime) const MID_TURN_COMPRESS_COOLDOWN_ITERATIONS: usize = 2;
/// Mid-turn 压缩增量门槛：自上次压缩后 messages 增量小于此值则跳过（避免
/// 大 tool result 留在 messages 末尾时反复触发 no-op 压缩）。
pub(in crate::ai::driver::turn_runtime) const MID_TURN_COMPRESS_DELTA_THRESHOLD: usize = 4_000;

pub(in crate::ai) use debug::report_agent_hang_debug;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    use serde_json::Value;

    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        history::{Message, SessionStore, build_message_arr},
        types::{App, AppConfig},
    };

    fn test_app(history_file: PathBuf) -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                history_file: history_file.clone(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 24_000,
                history_keep_last: 256,
                history_summary_max_chars: 4_000,
                intent_model: None,
                intent_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/intent/intent_model.json"),
                agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
            },
            session_id: "test".to_string(),
            session_history_file: history_file,
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            pending_short_output: false,
            forced_skill: None,
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
            observers: vec![Box::new(
                crate::ai::driver::thinking::ThinkingOrchestrator::new(),
            )],
        }
    }

    fn extract_stub_path(stub: &str) -> Option<PathBuf> {
        stub.lines()
            .find_map(|line| line.strip_prefix("- file_path: "))
            .map(PathBuf::from)
    }

    #[test]
    fn persist_pending_turn_messages_only_appends_new_entries() {
        let path =
            std::env::temp_dir().join(format!("ai-turn-history-{}.sqlite", uuid::Uuid::new_v4()));
        let app = test_app(path.clone());

        let mut turn_messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let mut persisted = 0usize;

        persist_pending_turn_messages(&app, false, &turn_messages, &mut persisted);
        assert_eq!(persisted, 1);

        turn_messages.push(Message {
            role: "tool".to_string(),
            content: Value::String("tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        });
        turn_messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String("done".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });

        persist_pending_turn_messages(&app, false, &turn_messages, &mut persisted);
        assert_eq!(persisted, 3);

        let loaded = build_message_arr(16, &path).unwrap();
        assert_eq!(loaded, turn_messages);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn prepare_tool_result_spills_large_output_to_session_file() {
        let history_file =
            std::env::temp_dir().join(format!("ai-tool-overflow-{}.sqlite", uuid::Uuid::new_v4()));
        let app = test_app(history_file.clone());
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        std::fs::write(store.session_history_file(&app.session_id), b"test").unwrap();

        let content = "x".repeat(MAX_TOOL_RESULT_INLINE_CHARS + 256);
        let prepared = prepare_tool_result(&app, "mcp_big_payload", &content);

        assert!(
            prepared
                .content_for_model
                .contains("Output too large; full result saved")
        );
        let path = extract_stub_path(&prepared.content_for_model).unwrap();
        assert!(path.is_absolute());
        assert!(path.exists());
        let saved = std::fs::read_to_string(&path).unwrap();
        assert_eq!(saved, content);

        let _ = store.delete_session(&app.session_id);
        assert!(!path.exists());
    }

    #[test]
    fn prepare_tool_result_json_stub_includes_keys_and_samples() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-overflow-json-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file.clone());
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        std::fs::write(store.session_history_file(&app.session_id), b"test").unwrap();

        let payload = serde_json::json!({
            "id": 123,
            "name": "example payload",
            "items": [
                { "kind": "doc", "token": "abc", "size": 42 }
            ],
            "meta": {
                "source": "mcp",
                "ok": true
            }
        });
        let content = format!("{}{}", payload, " ".repeat(MAX_TOOL_RESULT_INLINE_CHARS));
        let prepared = prepare_tool_result(&app, "mcp_json_payload", &content);

        assert!(prepared.content_for_model.contains("- top_level_keys:"));
        assert!(prepared.content_for_model.contains("id"));
        assert!(prepared.content_for_model.contains("name"));
        assert!(prepared.content_for_model.contains("- field_samples:"));
        assert!(prepared.content_for_model.contains("items:"));
        assert!(prepared.content_for_model.contains("meta:"));

        let _ = store.delete_session(&app.session_id);
    }

    #[test]
    fn prepare_tool_result_truncates_terminal_preview_but_keeps_model_content() {
        let history_file =
            std::env::temp_dir().join(format!("ai-tool-preview-{}.sqlite", uuid::Uuid::new_v4()));
        let app = test_app(history_file.clone());

        let mut content = String::new();
        for i in 0..160usize {
            content.push_str(&format!("{}→{}\n", i, "x".repeat(120)));
        }
        assert!(content.chars().count() < MAX_TOOL_RESULT_INLINE_CHARS);

        let prepared = prepare_tool_result(&app, "read_file_lines", &content);

        eprintln!("DEBUG: content chars = {}", content.chars().count());
        eprintln!("DEBUG: content lines = {}", content.lines().count());
        eprintln!(
            "DEBUG: terminal preview len = {}",
            prepared.content_for_terminal.len()
        );
        eprintln!(
            "DEBUG: terminal preview first 300 chars:\n{}",
            &prepared.content_for_terminal[..300.min(prepared.content_for_terminal.len())]
        );

        assert_eq!(prepared.content_for_model, content);
        assert!(
            prepared
                .content_for_terminal
                .contains("truncated for terminal preview")
        );
        assert!(prepared.content_for_terminal.len() < prepared.content_for_model.len());
        assert!(prepared.content_for_terminal.contains("0→"));
        assert!(prepared.content_for_terminal.contains("159→"));
    }

    #[test]
    fn read_file_lines_uses_shorter_terminal_preview_policy() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-preview-read-file-lines-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);

        let mut content = String::new();
        for i in 0..90usize {
            content.push_str(&format!("{}→{}\n", i, "x".repeat(100)));
        }

        let prepared = prepare_tool_result(&app, "read_file_lines", &content);

        assert_eq!(prepared.content_for_model, content);
        assert!(
            prepared
                .content_for_terminal
                .contains("truncated for terminal preview")
        );
        assert!(prepared.content_for_terminal.contains("0→"));
        assert!(prepared.content_for_terminal.contains("89→"));
        assert!(!prepared.content_for_terminal.contains("39→"));
        assert!(prepared.content_for_terminal.len() < 3000);
    }

    #[test]
    fn web_search_uses_summary_first_terminal_preview() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-preview-web-search-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);

        let mut content = String::new();
        for i in 0..40usize {
            content.push_str(&format!("result {}: title {}\n", i, "x".repeat(60)));
        }

        let prepared = prepare_tool_result(&app, "web_search", &content);

        assert_eq!(prepared.content_for_model, content);
        assert!(
            prepared
                .content_for_terminal
                .contains("summary-first terminal preview")
        );
        assert!(prepared.content_for_terminal.contains("result 0"));
        assert!(!prepared.content_for_terminal.contains("result 39"));
    }
}
