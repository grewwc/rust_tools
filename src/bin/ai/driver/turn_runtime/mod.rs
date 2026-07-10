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

mod context_budget;
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
pub(crate) use prepare::QuestionShape;
#[cfg(test)]
use tool_result::prepare_tool_result;
pub(super) use types::TurnOutcome;

const MAX_TOOL_RESULT_INLINE_CHARS: usize = 32_000;
const TOOL_OVERFLOW_PREVIEW_CHARS: usize = 800;
/// 首次 overflow stub 中的 head 预览字符数。
/// 与 mid-turn 压缩 stub 的 head 8 行预览保持一致的信息密度。
const TOOL_OVERFLOW_HEAD_CHARS: usize = 800;
/// 中等大输出阈值：超过此值但未到 overflow 阈值的工具结果，仅对非精确概览类
/// 工具走"头 + 关键命中 + 尾"的按行裁剪，避免完整 32KB 全部进上下文。
/// grep/code_search/search_files/read_file(_lines) 等精确证据工具不走该有损路径。
const MAX_TOOL_RESULT_LINE_TRIM_CHARS: usize = 8_000;

/// 单条工具结果 inline（不 offload 到文件）的字符上限，按模型 context window 动态计算。
///
/// - 基准 32K（`MAX_TOOL_RESULT_INLINE_CHARS`），适合 128K token 窗口的模型。
/// - 大窗口模型按比例放宽：`context_window * chars_per_token / 8`，即窗口的 ~12.5%
///   预留给单条工具结果。256K token 模型 → 64K 字符，200K → 50K，128K → 32K。
/// - 上限 64K：避免单条工具结果占用过多上下文，即使模型窗口很大。
/// - 下限 32K：不小于基准值，确保小窗口模型也不会频繁 offload。
pub(in crate::ai::driver::turn_runtime) fn max_tool_result_inline_chars(model: &str) -> usize {
    const CHARS_PER_TOKEN: usize = 2;
    let window = crate::ai::models::context_window_tokens(model);
    window
        .saturating_mul(CHARS_PER_TOKEN)
        .saturating_div(8)
        .clamp(MAX_TOOL_RESULT_INLINE_CHARS, 64_000)
}

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
/// history_max_chars 默认 90K，对应软阈值 135K。
///
/// 但字符阈值与模型的 token 窗口是两套单位：一个高占用 prompt 可能远未触及
/// 180K 字符阈值，却已逼近模型 token 窗口。[`token_window_char_ceiling`] 给出
/// 该模型「安全字符预算」，二者取 min，确保接近 token 窗口时压缩必然更早触发。
pub(in crate::ai::driver::turn_runtime) fn mid_turn_compress_soft_threshold(
    model: &str,
    history_max_chars: usize,
) -> usize {
    history_max_chars
        .saturating_mul(3)
        .saturating_div(2)
        .max(MID_TURN_COMPRESS_SOFT_FLOOR)
        .min(token_window_char_ceiling(model))
}

/// 硬阈值：min 80K，否则取 history_max_chars * 3.5。
/// history_max_chars 默认 90K，对应硬阈值 315K（远超模型 context window，
/// 实际硬阈值会被模型 4xx 之前的 normalize_messages_for_request 拦截）。
/// 但相对软阈值留出明显 gap，避免软阈值边界连续触发 LLM summary。
/// 按 LLM 摘要字符阈值收口——LLM 摘要只在上下文接近模型实际 context window
/// 时触发（见 [`llm_summary_char_threshold`]），而非 60% 窗口处过早触发。
pub(in crate::ai::driver::turn_runtime) fn mid_turn_compress_hard_threshold(
    model: &str,
    history_max_chars: usize,
) -> usize {
    history_max_chars
        .saturating_mul(7)
        .saturating_div(2)
        .max(MID_TURN_COMPRESS_HARD_FLOOR)
        .min(llm_summary_char_threshold(model))
}
/// LLM 摘要兜底时保留尾部窗口的 user 起始轮数。早期超过此窗口的对话被压成
/// 一条 internal_note 摘要插入到尾部窗口前。
pub(in crate::ai::driver::turn_runtime) const MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS: usize = 2;
/// LLM 摘要文本的最大字符数。
pub(in crate::ai::driver::turn_runtime) const MID_TURN_LLM_SUMMARY_MAX_CHARS: usize = 4_000;
/// Pre-request LLM 摘要阈值：在每次发送 LLM 请求前，如果无损+弱损压缩后
/// 仍超过此阈值，触发 LLM 摘要兜底（把早期对话压成单条 internal_note）。
/// 按 LLM 摘要字符阈值收口——LLM 摘要只在上下文接近模型实际 context window
/// 时触发，避免在远低于窗口上限时就频繁调用 LLM 摘要（旧设计用 0.6 窗口
/// 收口导致小窗口模型在 76K 字符处就不停触发 LLM 摘要却压不动）。
pub(in crate::ai::driver::turn_runtime) fn pre_request_llm_summary_threshold(
    model: &str,
    history_max_chars: usize,
) -> usize {
    history_max_chars
        .saturating_mul(2)
        .max(MID_TURN_COMPRESS_HARD_FLOOR)
        .min(llm_summary_char_threshold(model))
}

/// LLM 摘要字符阈值：`context_window_tokens * CHARS_PER_TOKEN`。
///
/// 与 [`token_window_char_ceiling`]（0.6 窗口，用于无损压缩提前裁剪）不同，
/// 此阈值代表「history 已撑满模型实际 context window」——此时 LLM 摘要
///（昂贵，需额外调用一次模型）才真正必要。默认 100K token 模型 → 200K 字符。
///
/// `CHARS_PER_TOKEN = 2` 本身已偏保守（中文约 1-2 字符/token，英文约 3-4），
/// 不再额外乘 fraction，避免小窗口模型阈值过低导致 LLM 摘要空转。
pub(in crate::ai::driver::turn_runtime) fn llm_summary_char_threshold(model: &str) -> usize {
    const CHARS_PER_TOKEN: usize = 2;
    crate::ai::models::context_window_tokens(model)
        .saturating_mul(CHARS_PER_TOKEN)
        .max(MID_TURN_COMPRESS_HARD_FLOOR)
}

/// 模型 token 窗口换算出的「安全字符预算」：`window * chars_per_token * fraction`。
/// - `chars_per_token = 2`：与 [`request`] 侧 max_tokens 钳制保持同一保守换算。
/// - `fraction = 0.6`：只用窗口的 ~60% 给历史 prompt，剩余留给系统 prompt、
///   本轮 user、工具 schema 以及模型输出，避免压缩阈值本身贴着窗口上沿。
pub(in crate::ai::driver::turn_runtime) fn token_window_char_ceiling(model: &str) -> usize {
    const CHARS_PER_TOKEN: usize = 2;
    let window = crate::ai::models::context_window_tokens(model);
    window
        .saturating_mul(CHARS_PER_TOKEN)
        .saturating_mul(3)
        .saturating_div(5)
        .max(MID_TURN_COMPRESS_SOFT_FLOOR)
}
/// Pre-request LLM 摘要重触发最小增量：自上次 LLM 摘要后 messages 增量
/// 小于此值则跳过，避免摘要失败时每轮重复调用 LLM。
pub(in crate::ai::driver::turn_runtime) const PRE_REQUEST_LLM_SUMMARY_MIN_GROWTH: usize = 20_000;
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
                base_history_file: history_file.clone(),
                history_file: history_file.clone(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 24_000,
                history_keep_last: 256,
                history_summary_max_chars: 4_000,
                intent_model: None,
                agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
            },
            session_id: "test".to_string(),
            session_history_file: history_file,
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
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
            last_turn_interrupted: false,
            prune_marks: Default::default(),
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
        let mut app = test_app(history_file.clone());
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        app.session_history_file = store.session_history_file(&app.session_id);
        std::fs::write(&app.session_history_file, b"test").unwrap();

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
        let expected_dir = store
            .session_assets_dir(&app.session_id)
            .join("tool-overflow");
        // macOS 上 /tmp 是 /private/tmp 的符号链接；overflow 文件路径经过
        // canonicalize，比较前需对 expected_dir 做同样处理，否则 starts_with 失败。
        let expected_dir = expected_dir.canonicalize().unwrap_or(expected_dir);
        let nested_dir = SessionStore::new(app.session_history_file.as_path())
            .session_assets_dir(&app.session_id)
            .join("tool-overflow");
        assert!(path.starts_with(&expected_dir));
        assert!(!nested_dir.exists());
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
    fn precision_search_tools_keep_medium_output_exact_for_model() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-preview-grep-exact-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);

        let mut content = String::new();
        for i in 0..160usize {
            content.push_str(&format!(
                "src/example_{i}.rs:{}: matched precise line {}\n",
                i + 1,
                "x".repeat(90)
            ));
        }
        assert!(content.chars().count() > MAX_TOOL_RESULT_LINE_TRIM_CHARS);
        assert!(content.chars().count() < MAX_TOOL_RESULT_INLINE_CHARS);

        for tool_name in ["find_path", "code_search", "search_files"] {
            let prepared = prepare_tool_result(&app, tool_name, &content);
            assert_eq!(prepared.content_for_model, content);
            assert!(!prepared.content_for_model.contains("middle trimmed"));
        }
    }

    #[test]
    fn precision_search_tools_offload_large_output_instead_of_lossy_trimming() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-overflow-grep-exact-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file.clone());
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        std::fs::write(store.session_history_file(&app.session_id), b"test").unwrap();

        let content = (0..420usize)
            .map(|i| {
                format!(
                    "src/example_{i}.rs:{}: matched precise line {}\n",
                    i + 1,
                    "x".repeat(90)
                )
            })
            .collect::<String>();
        assert!(content.chars().count() > MAX_TOOL_RESULT_INLINE_CHARS);

        let prepared = prepare_tool_result(&app, "find_path", &content);

        assert!(
            prepared
                .content_for_model
                .contains("Output too large; full result saved")
        );
        assert!(!prepared.content_for_model.contains("middle trimmed"));
        let path = extract_stub_path(&prepared.content_for_model).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), content);

        let _ = store.delete_session(&app.session_id);
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
