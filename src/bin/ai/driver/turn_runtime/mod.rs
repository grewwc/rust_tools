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
mod context_memory;
mod context_metrics;
mod debug;
mod finalize;
mod iteration;
mod llm_summary_budget;
mod orchestrator;
mod output_postprocess;
mod persistence;
mod prepare;
pub(in crate::ai) mod tool_result;
mod turn_signals;
mod types;

pub(super) use finalize::SUBAGENT_PARENT_AGENT_TEAM_MODE;
pub(super) use orchestrator::run_turn;
// `/history last` replay (driver/input.rs, serve/chat.rs) paints the stored
// body through the same display-only post-processing as the live turn path
// (finalize.rs).
pub(in crate::ai) use output_postprocess::postprocess_terminal_text;
// Earlier steps declared checkpoint/progress as orchestrator-private submodules, but execution.rs references them as
// `turn_runtime::checkpoint|progress`; these re-exports restore turn_runtime-level visibility.
pub(crate) use orchestrator::{checkpoint, progress};
#[cfg(test)]
use persistence::persist_pending_turn_messages;
pub(crate) use prepare::QuestionShape;
pub(in crate::ai) use tool_result::is_evidence_gated_audit_agent;
pub(in crate::ai) use tool_result::stale_patch_targets_from_messages;
#[cfg(test)]
use tool_result::{prepare_recent_tool_result, prepare_tool_result};
pub(super) use types::TurnOutcome;

/// Merges adjacent compression phases into a single status line so each phase does not occupy its own line.
#[derive(Default)]
pub(super) struct CompressionReport {
    entries: Vec<String>,
}

impl CompressionReport {
    pub(super) fn record(&mut self, label: impl Into<String>, before: usize, after: usize) {
        self.entries
            .push(format!("{}: {before} → {after} chars", label.into()));
    }

    /// `effective` only means the net savings reached the summary-effective threshold; when false, the hard-budget fallback may
    /// still have shrunk and replaced the request context, so it must not be misreported as skipped.
    /// `llm_summary_inserted` says whether this compression actually ran and injected `[mid-turn-summary]`.
    /// When false and `after < before`, the reduction came entirely from mechanical paths (folding/truncation/offload),
    /// reported as "skipped (no LLM summary), mechanical-only", so purely mechanical compression is not
    /// misreported as "an LLM summary ran" (users previously saw `pre-request LLM ... chars` when in fact
    /// no LLM was triggered — an observability bug).
    pub(super) fn record_llm_summary_attempt(
        &mut self,
        label: impl Into<String>,
        before: usize,
        after: usize,
        effective: bool,
        llm_summary_inserted: bool,
    ) {
        let label = label.into();
        if llm_summary_inserted {
            if effective {
                self.record(label, before, after);
            } else {
                self.record(
                    format!("{label} partial (below effective-savings threshold)"),
                    before,
                    after,
                );
            }
        } else if after < before {
            self.record(
                format!("{label} skipped (no LLM summary), mechanical-only"),
                before,
                after,
            );
        } else {
            self.note(format!(
                "{label} skipped (no reducible context or summary call failed); \
                 agent may hit context limit"
            ));
        }
    }

    pub(super) fn note(&mut self, note: impl Into<String>) {
        self.entries.push(note.into());
    }

    pub(super) fn render(&self) -> Option<String> {
        (!self.entries.is_empty()).then(|| self.entries.join(" | "))
    }

    pub(super) fn emit(self) {
        if let Some(line) = self.render() {
            crate::ai::driver::print::print_tool_note_line("compress", &line);
        }
    }
}

pub(super) async fn maybe_generate_session_title(app: &super::App, run_in_background: bool) {
    finalize::maybe_generate_session_title(app, run_in_background).await;
}

pub(super) async fn maybe_generate_session_title_for_input(app: &super::App, user_input: &str) {
    finalize::maybe_generate_session_title_for_input(app, user_input).await;
}

pub(crate) use finalize::SessionTitleSeed;
#[cfg(feature = "serve")]
pub(crate) use finalize::{SERVE_TITLE_SEED_ENV, SERVE_TITLE_SESSION_ENV};

/// Generate the model session title for a served session, outside its turn
/// child. See [`finalize::generate_session_title_outside_turn`] for why this
/// must run outside the child. The title policy stays distinct from the child's input.
pub(crate) async fn generate_session_title_outside_turn(
    history_file: &std::path::Path,
    session_id: &str,
    current_model: &str,
    title_seed: &SessionTitleSeed,
) -> bool {
    finalize::generate_session_title_outside_turn(
        history_file,
        session_id,
        current_model,
        title_seed,
    )
    .await
}

const MAX_TOOL_RESULT_INLINE_CHARS: usize = 32_000;
const TOOL_OVERFLOW_PREVIEW_CHARS: usize = 800;
/// Number of head preview characters in the first overflow stub.
/// Matches the information density of the head-8-line preview in mid-turn compression stubs.
const TOOL_OVERFLOW_HEAD_CHARS: usize = 800;
/// Medium-large output threshold: tool results above this but below the overflow threshold are trimmed line-wise
/// ("head + key hits + tail") only for non-precise overview tools, avoiding the full 32KB entering context.
/// Precise evidence tools such as grep/read_file(_lines) never take this lossy path.
const MAX_TOOL_RESULT_LINE_TRIM_CHARS: usize = 8_000;

/// Per-result inline (not offloaded to file) character cap, computed dynamically from the model context window.
///
/// - Baseline 32K (`MAX_TOOL_RESULT_INLINE_CHARS`), suited to 128K token window models.
/// - Large-window models scale up proportionally: `context_window * chars_per_token / 12`, i.e. ~8.3% of the window
///   reserved for a single tool result. 256K token model → 42K chars, 200K → 33K, 128K → 21K (raised to the 32K floor).
/// - Cap 64K: keeps a single tool result from consuming too much context even on very large windows.
/// - Floor 32K: never below the baseline so small-window models do not offload too often.
pub(crate) fn max_tool_result_inline_chars(model: &str) -> usize {
    const CHARS_PER_TOKEN: usize = 2;
    let window = crate::ai::models::context_window_tokens(model);
    window
        .saturating_mul(CHARS_PER_TOKEN)
        .saturating_div(12)
        .clamp(MAX_TOOL_RESULT_INLINE_CHARS, 64_000)
}

/// Character floors for the window-derived compaction thresholds below.
///
/// The thresholds default to a dynamic value from `app.config.history_max_chars` (see
/// [`mid_turn_compress_soft_threshold`] / [`mid_turn_compress_hard_threshold`]).
/// These two constants only act as floor guards (so a user-set too-small history_max_chars cannot cause
/// compaction to trigger on a single tool result and no-op repeatedly). In-turn compaction itself
/// runs only at the request boundary, gated by measured token pressure.
pub(in crate::ai::driver::turn_runtime) const MID_TURN_COMPRESS_SOFT_FLOOR: usize = 36_000;
/// Mid-turn LLM summary hard-threshold floor: if content still exceeds this after lossless/weak-loss compression, an LLM summary
/// fallback runs (one model call, followed by a single merged compression status line).
pub(in crate::ai::driver::turn_runtime) const MID_TURN_COMPRESS_HARD_FLOOR: usize = 80_000;

/// Soft threshold: min 36K, otherwise history_max_chars * 1.5.
/// history_max_chars is a fixed default when unset (see
/// `config::history_max_chars_for_model`; 200K chars for every model).
///
/// But character thresholds and the model token window are different units: a heavily loaded prompt may be far below
/// the 400K character threshold yet already close to the model token window. [`token_window_char_ceiling`] gives
/// the model's "safe character budget"; taking the min of both ensures compression triggers earlier as the window nears.
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

/// Hard threshold: min 80K, otherwise history_max_chars * 3.5.
/// history_max_chars defaults to 200K, giving a 700K hard threshold, far beyond the lossless soft
/// threshold; it is capped at the model's context window ([`llm_summary_char_threshold`]) and in
/// practice intercepted by normalize_messages_for_request before the model returns 4xx.
/// It leaves a clear gap above the soft threshold so LLM summary does not trigger repeatedly at the soft boundary.
/// Gated by the LLM summary character threshold — LLM summary only runs when the context approaches the model's actual context window
/// (see [`llm_summary_char_threshold`]), not prematurely at 60% of the window.
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
pub(in crate::ai::driver::turn_runtime) use crate::ai::history::compress::{
    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS, MID_TURN_LLM_SUMMARY_MAX_CHARS,
};
/// Pre-request LLM summary threshold: before each LLM request is sent, if lossless + weak-loss compression still leaves
/// content above this threshold, the LLM summary fallback runs (compressing earlier conversation into a single internal_note).
/// Gated by the LLM summary character threshold — LLM summary only runs when the context approaches the model's actual context window,
/// avoiding frequent LLM summary calls far below the window cap (the old design gated at 0.6 of the window,
/// which made small-window models trigger LLM summary repeatedly at 76K chars without any effect).
pub(in crate::ai::driver::turn_runtime) fn pre_request_llm_summary_threshold(
    model: &str,
    history_max_chars: usize,
) -> usize {
    history_max_chars
        .saturating_mul(2)
        .max(MID_TURN_COMPRESS_HARD_FLOOR)
        .min(llm_summary_char_threshold(model))
}

/// LLM summary character threshold: `context_window_tokens * CHARS_PER_TOKEN`.
///
/// Unlike [`token_window_char_ceiling`] (0.6 window, used to trim early during lossless compression),
/// this threshold means "history has filled the model's actual context window" — only then is an LLM summary
/// (expensive, one extra model call) truly necessary. A 100K token model defaults to 200K characters.
///
/// `CHARS_PER_TOKEN = 2` is already conservative (Chinese ≈ 1-2 chars/token, English ≈ 3-4),
/// so no extra fraction is multiplied, avoiding a too-low threshold that makes LLM summary spin uselessly on small-window models.
pub(in crate::ai::driver::turn_runtime) fn llm_summary_char_threshold(model: &str) -> usize {
    const CHARS_PER_TOKEN: usize = 2;
    crate::ai::models::context_window_tokens(model)
        .saturating_mul(CHARS_PER_TOKEN)
        .max(MID_TURN_COMPRESS_HARD_FLOOR)
}

/// "Safe character budget" derived from the model token window: `window * chars_per_token * fraction`.
/// - `chars_per_token = 2`: same conservative conversion as the max_tokens clamp on the [`request`] side.
/// - `fraction = 0.6`: only ~60% of the window goes to the history prompt, leaving the rest for the system prompt,
///   this turn's user message, tool schemas, and model output, so the compression threshold never hugs the top of the window.
pub(in crate::ai::driver::turn_runtime) fn token_window_char_ceiling(model: &str) -> usize {
    const CHARS_PER_TOKEN: usize = 2;
    let window = crate::ai::models::context_window_tokens(model);
    window
        .saturating_mul(CHARS_PER_TOKEN)
        .saturating_mul(3)
        .saturating_div(5)
        .max(MID_TURN_COMPRESS_SOFT_FLOOR)
}
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

    #[test]
    fn compression_report_merges_adjacent_stages_into_one_line() {
        let mut report = CompressionReport::default();
        report.record("mid-turn", 182_743, 182_259);
        report.record("pre-request LLM (limit 180000)", 182_259, 89_984);

        assert_eq!(
            report.render().as_deref(),
            Some(
                "mid-turn: 182743 → 182259 chars | \
                 pre-request LLM (limit 180000): 182259 → 89984 chars"
            )
        );
    }

    #[test]
    fn llm_summary_partial_reduction_is_not_reported_as_skipped() {
        let mut report = CompressionReport::default();
        report.record_llm_summary_attempt(
            "pre-request LLM (limit 180000)",
            182_259,
            181_000,
            false,
            true,
        );

        assert_eq!(
            report.render().as_deref(),
            Some(
                "pre-request LLM (limit 180000) partial \
                 (below effective-savings threshold): 182259 → 181000 chars"
            )
        );
    }

    #[test]
    fn llm_summary_mechanical_only_reduction_is_not_reported_as_llm() {
        // User-observed problem: in single-turn sessions with huge tool output, the LLM summary was actually skipped
        // (no old conversation to summarize), yet once the net reduction reached the effective threshold it was reported as `... LLM ... chars`,
        // looking like an LLM call. This verifies that mechanical-only reduction is marked as skipped (no LLM summary).
        let mut report = CompressionReport::default();
        report.record_llm_summary_attempt(
            "pre-request LLM (limit 180000)",
            581_560,
            569_677,
            true,
            false,
        );

        assert_eq!(
            report.render().as_deref(),
            Some(
                "pre-request LLM (limit 180000) skipped (no LLM summary), \
                 mechanical-only: 581560 → 569677 chars"
            )
        );
    }

    fn test_app(history_file: PathBuf) -> App {
        App {
            cli: ParsedCli::default(),
            scoped_preflight_required: Vec::new(),
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
            },
            session_id: "test".to_string(),
            session_history_file: history_file,
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: "build".to_string(),
            session_default_subagent_backend: Default::default(),
            session_default_subagent_model: None,
            session_default_skills: Vec::new(),
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
            turn_tool_errors: Default::default(),
            stale_patch_targets: Default::default(),
            tool_middlewares: Vec::new(),
            llm_middlewares: Vec::new(),
            hooks: Default::default(),
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
        // Live tool-overflow stubs carry a session-asset alias; resolve it
        // against this session's assets for the filesystem assertions below.
        let locator = path.to_string_lossy().into_owned();
        assert!(
            locator.starts_with("session-asset:tool-overflow/"),
            "{locator}"
        );
        let assets = store.session_assets_dir(&app.session_id);
        let path =
            crate::ai::tools::storage::file_store::session_asset_for_alias(&locator, &assets)
                .expect("alias must resolve under this session's assets");
        // Match the canonicalized comparison below (/tmp -> /private/tmp).
        let path = path.canonicalize().unwrap_or(path);
        assert!(path.is_absolute());
        assert!(path.exists());
        let expected_dir = store
            .session_assets_dir(&app.session_id)
            .join("tool-overflow");
        // On macOS /tmp is a symlink to /private/tmp; overflow file paths go through canonicalize,
        // so expected_dir must be canonicalized the same way before comparison, or starts_with fails.
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
    fn prepare_recent_tool_result_keeps_large_output_raw_for_model() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-overflow-recent-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let mut app = test_app(history_file.clone());
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        app.session_history_file = store.session_history_file(&app.session_id);
        std::fs::write(&app.session_history_file, b"test").unwrap();

        let content = "x".repeat(MAX_TOOL_RESULT_INLINE_CHARS + 256);
        let prepared = prepare_recent_tool_result(&app, "mcp_big_payload", &content);

        assert_eq!(prepared.content_for_model, content);
        assert!(
            prepared
                .content_for_terminal
                .contains("Saved full output to"),
            "terminal preview should still keep overflow ergonomics"
        );

        let _ = store.delete_session(&app.session_id);
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

        let prepared = prepare_tool_result(&app, "read_file", &content);

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
    fn read_file_uses_shorter_terminal_preview_policy() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-preview-read-file-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);

        let mut content = String::new();
        for i in 0..90usize {
            content.push_str(&format!("{}→{}\n", i, "x".repeat(100)));
        }

        let prepared = prepare_tool_result(&app, "read_file", &content);

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

        let prepared = prepare_tool_result(&app, "read_file", &content);
        assert_eq!(prepared.content_for_model, content);
        assert!(!prepared.content_for_model.contains("middle trimmed"));
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

        let prepared = prepare_tool_result(&app, "read_file", &content);

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
}
