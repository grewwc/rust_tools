/// Centralized configuration schema for the AI agent.
///
/// All configuration keys are defined here as constants to avoid
/// scattered string literals across the codebase.
///
/// Usage:
/// ```rust
/// use crate::ai::config_schema::AiConfig;
/// let cfg = configw::get_all_config();
/// let endpoint = cfg.get_opt(AiConfig::MODEL_ENDPOINT);
/// ```
pub struct AiConfig;

impl AiConfig {
    // ── Model ──────────────────────────────────────────────
    pub const MODEL_ENDPOINT: &str = "ai.model.endpoint";
    pub const MODEL_API_KEY: &str = "api_key";
    pub const MODEL_OPENCODE_API_KEY: &str = "opencode.api_key";
    pub const MODEL_OPENROUTER_API_KEY: &str = "openrouter.api_key";
    pub const MODEL_COMPATIBLE_API_KEY: &str = "compatible.api_key";
    pub const MODEL_ALIBABA_API_KEY: &str = "alibaba.api_key";
    pub const MODEL_ALIYUN_API_KEY: &str = "aliyun.api_key";
    pub const MODEL_OPENAI_API_KEY: &str = "openai.api_key";
    pub const MODEL_DEFAULT: &str = "ai.model.default";
    pub const MODEL_VL_DEFAULT: &str = "ai.model.vl_default";
    pub const MODEL_THINKING: &str = "ai.model.thinking";
    pub const MODEL_AUTO_THINKING_ENABLE: &str = "ai.model.auto_thinking.enable";
    pub const MODEL_AUTO_THINKING_THRESHOLD: &str = "ai.model.auto_thinking.threshold";
    /// Comma-separated model keys/names excluded from automatic model selection.
    /// Explicit model overrides still resolve normally.
    pub const MODEL_DISABLED: &str = "ai.model.disabled";

    // ── History ────────────────────────────────────────────
    pub const HISTORY_MAX_CHARS: &str = "ai.history.max_chars";
    pub const HISTORY_KEEP_LAST: &str = "ai.history.keep_last";
    pub const HISTORY_SUMMARY_MAX_CHARS: &str = "ai.history.summary_max_chars";

    // ── Intent ─────────────────────────────────────────────
    pub const INTENT_MODEL: &str = "ai.intent_model";

    // ── MCP ────────────────────────────────────────────────
    pub const MCP_CONFIG: &str = "ai.mcp.config";

    // ── Skills ─────────────────────────────────────────────
    pub const SKILLS_DIR: &str = "ai.skills.dir";
    pub const SKILLS_DEBUG: &str = "ai.skills.debug";
    pub const SKILLS_ROUTER: &str = "ai.skills.router";
    pub const SKILLS_ROUTER_THRESHOLD: &str = "ai.skills.router_threshold";

    // ── Agents ─────────────────────────────────────────────
    pub const AGENTS_DIR: &str = "ai.agents.dir";

    // ── Memory ─────────────────────────────────────────────
    pub const MEMORY_FILE: &str = "ai.memory.file";
    pub const MEMORY_SEARCH_ARCHIVES_ENABLE: &str = "ai.memory.search_archives.enable";
    pub const MEMORY_SEARCH_ARCHIVES_KEEP_LAST: &str = "ai.memory.search_archives.keep_last";
    pub const MEMORY_AUTO_ROTATE_MAX_BYTES: &str = "ai.memory.auto_rotate.max_bytes";
    pub const MEMORY_AUTO_GC_DAYS: &str = "ai.memory.auto_gc.days";
    pub const MEMORY_AUTO_GC_MIN_KEEP: &str = "ai.memory.auto_gc.min_keep";
    pub const MEMORY_AUTO_MAINTAIN_PROBABILITY: &str = "ai.memory.auto_maintain.probability";
    pub const MEMORY_ARCHIVES_RETAIN_DAYS: &str = "ai.memory.archives.retain_days";
    pub const MEMORY_ARCHIVES_KEEP_LAST: &str = "ai.memory.archives.keep_last";
    pub const MEMORY_ARCHIVES_MAX_BYTES: &str = "ai.memory.archives.max_bytes";

    // ── Knowledge ──────────────────────────────────────────
    pub const KNOWLEDGE_MIN_SCORE_GUIDELINE: &str = "ai.knowledge.thresholds.min_score_guideline";
    pub const KNOWLEDGE_MIN_SCORE_KNOWLEDGE: &str = "ai.knowledge.thresholds.min_score_knowledge";
    pub const KNOWLEDGE_GUIDELINES_MAX_CHARS: &str =
        "ai.knowledge.maintenance.guidelines_max_chars";
    pub const KNOWLEDGE_SEARCH_DEFAULT_LIMIT: &str = "ai.knowledge.search.default_limit";

    // ── Embedding (remote provider) ────────────────────────
    pub const EMBEDDING_ENABLE: &str = "ai.embedding.enable";
    pub const EMBEDDING_ENDPOINT: &str = "ai.embedding.endpoint";
    pub const EMBEDDING_API_KEY: &str = "ai.embedding.api_key";
    pub const EMBEDDING_MODEL: &str = "ai.embedding.model";
    pub const EMBEDDING_TIMEOUT_MS: &str = "ai.embedding.timeout_ms";

    // ── Reflection ─────────────────────────────────────────
    pub const REFLECTION_ENABLE: &str = "ai.reflection.enable";
    pub const REFLECTION_INTEGRATED: &str = "ai.reflection.integrated";
    pub const REFLECTION_TIMEOUT_MS: &str = "ai.reflection.timeout_ms";
    pub const REFLECTION_MODEL_GATE_ENABLE: &str = "ai.reflection.model_gate.enable";
    pub const REFLECTION_MODEL_GATE_TIMEOUT_MS: &str = "ai.reflection.model_gate.timeout_ms";
    pub const REFLECTION_FILTER_ENABLE: &str = "ai.reflection.filter.enable";
    pub const REFLECTION_FILTER_MIN_QUESTION_CHARS: &str =
        "ai.reflection.filter.min_question_chars";
    pub const REFLECTION_FILTER_MIN_ANSWER_CHARS: &str = "ai.reflection.filter.min_answer_chars";
    pub const REFLECTION_FILTER_REQUIRE_TOOL_OR_LONG: &str =
        "ai.reflection.filter.require_tool_or_long";

    // ── Critic & Revise ────────────────────────────────────
    pub const CRITIC_REVISE_ENABLE: &str = "ai.critic_revise.enable";
    pub const CRITIC_REVISE_INTEGRATED: &str = "ai.critic_revise.integrated";
    pub const CRITIC_REVISE_TIMEOUT_MS: &str = "ai.critic_revise.timeout_ms";
    pub const CRITIC_REVISE_ONLY_FOR_CODE: &str = "ai.critic_revise.only_for_code";
    pub const CRITIC_REVISE_MODEL: &str = "ai.critic_revise.model";
    pub const CRITIC_REVISE_FILTER_MIN_QUESTION_CHARS: &str =
        "ai.critic_revise.filter.min_question_chars";
    pub const CRITIC_REVISE_FILTER_MIN_ANSWER_CHARS: &str =
        "ai.critic_revise.filter.min_answer_chars";

    // ── Experience Generalization (LLM 二次提炼) ────────────
    pub const GENERALIZE_LLM_REFINE_ENABLE: &str = "ai.generalize.llm_refine.enable";
    pub const GENERALIZE_LLM_REFINE_MODEL: &str = "ai.generalize.llm_refine.model";
    pub const GENERALIZE_LLM_REFINE_TIMEOUT_MS: &str = "ai.generalize.llm_refine.timeout_ms";

    // ── Project Writeback ──────────────────────────────────
    pub const PROJECT_WRITEBACK_ENABLE: &str = "ai.project_writeback.enable";
    pub const PROJECT_WRITEBACK_MODEL: &str = "ai.project_writeback.model";
    pub const PROJECT_WRITEBACK_TIMEOUT_MS: &str = "ai.project_writeback.timeout_ms";

    // ── Sandbox ────────────────────────────────────────────
    /// Comma-separated extra program names to block in `execute_command`
    /// (merged with the built-in deny list). Empty = no extras.
    pub const SANDBOX_BLOCKED_COMMANDS: &str = "ai.sandbox.blocked_commands";
    /// Default timeout (seconds) applied to `execute_command` when the call
    /// does not specify one. Falls back to 60 when unset/invalid.
    pub const SANDBOX_COMMAND_TIMEOUT_DEFAULT: &str = "ai.sandbox.command_timeout_default";
    /// Hard upper bound (seconds) for any `execute_command` timeout. Falls
    /// back to 300 when unset/invalid.
    pub const SANDBOX_COMMAND_TIMEOUT_MAX: &str = "ai.sandbox.command_timeout_max";
    /// Comma-separated absolute roots that file read/write tools are confined
    /// to. Empty = no extra confinement (preserves default behavior).
    pub const SANDBOX_ALLOWED_ROOTS: &str = "ai.sandbox.allowed_roots";
    /// Comma-separated extra path substrings to treat as sensitive (blocked
    /// for file read/write), merged with the built-in sensitive list.
    pub const SANDBOX_EXTRA_SENSITIVE_PATHS: &str = "ai.sandbox.extra_sensitive_paths";

    // ── Hooks (lifecycle) ──────────────────────────────────
    /// Shell command executed before each turn starts. Receives env
    /// `AI_HOOK_EVENT=on_turn_start`. Empty = disabled.
    pub const HOOK_ON_TURN_START: &str = "ai.hooks.on_turn_start";
    /// Shell command executed after each turn finishes. Env
    /// `AI_HOOK_EVENT=on_turn_end`. Empty = disabled.
    pub const HOOK_ON_TURN_END: &str = "ai.hooks.on_turn_end";
    /// Shell command executed before each tool call. Env `AI_HOOK_EVENT`,
    /// `AI_TOOL_NAME`. Empty = disabled.
    pub const HOOK_BEFORE_TOOL: &str = "ai.hooks.before_tool";
    /// Shell command executed after each tool call. Env `AI_HOOK_EVENT`,
    /// `AI_TOOL_NAME`, `AI_TOOL_OK` (true/false). Empty = disabled.
    pub const HOOK_AFTER_TOOL: &str = "ai.hooks.after_tool";
    /// Shell command executed when the interactive session ends. Env
    /// `AI_HOOK_EVENT=on_session_end`. Empty = disabled.
    pub const HOOK_ON_SESSION_END: &str = "ai.hooks.on_session_end";
    /// Timeout (seconds) for each lifecycle hook command. Defaults to 30.
    pub const HOOK_TIMEOUT_SECS: &str = "ai.hooks.timeout_secs";

    // ── Prompt cache ───────────────────────────────────────
    /// When true, inject an Anthropic-style `cache_control` breakpoint on the
    /// system prompt for gateways that honor it (e.g. OpenRouter→Anthropic).
    /// Default false (OpenAI/DashScope cache automatically server-side).
    pub const PROMPT_CACHE_ENABLE: &str = "ai.prompt_cache.enable";
    /// When true, print prompt-cache hit metrics (cached tokens / hit rate)
    /// after each request when the provider reports them. Default true.
    pub const PROMPT_CACHE_SHOW_METRICS: &str = "ai.prompt_cache.show_metrics";

    // ── Output ─────────────────────────────────────────────
    /// Thinking 流式输出在 terminal 中最多保留多少行可见窗口。默认 8。
    /// 设为 0 可关闭折叠，恢复完整实时输出。仅影响终端展示，不影响模型上下文。
    pub const OUTPUT_THINKING_MAX_VISIBLE_LINES: &str = "ai.output.thinking.max_visible_lines";

    // ── Token usage stats ──────────────────────────────────
    /// When true, persist the per-session DecisionLog sidecar JSONL file.
    /// Default false; the in-memory DecisionLogStore remains enabled.
    pub const DECISION_LOG_PERSIST_ENABLE: &str = "ai.decision_log.persist.enable";
    /// When false, disable recording LLM token usage to the SQLite stats
    /// table. Default true.
    pub const TOKEN_USAGE_ENABLE: &str = "ai.token_usage.enable";
    /// SQLite database path for token usage stats. Defaults to
    /// `~/.config/rust_tools/token_usage.db`.
    pub const TOKEN_USAGE_DB: &str = "ai.token_usage.db";
    /// Retain token usage rows for this many days; older rows are purged
    /// during periodic cleanup. Defaults to 90.
    pub const TOKEN_USAGE_RETAIN_DAYS: &str = "ai.token_usage.retain_days";

    // ── Scheduler ──────────────────────────────────────────
    pub const SCHEDULER_BASE_BATCH: &str = "ai.scheduler.base_batch";
    pub const SCHEDULER_MAX_BATCH: &str = "ai.scheduler.max_batch";
    pub const SCHEDULER_EXECUTE_MAX: &str = "ai.scheduler.execute_max";
    pub const SCHEDULER_FAIL_STREAK_THRESHOLD: &str = "ai.scheduler.fail_streak_threshold";
    pub const SCHEDULER_COOLDOWN_EPOCHS: &str = "ai.scheduler.cooldown_epochs";
    pub const SCHEDULER_EVAL_PERIOD_EPOCHS: &str = "ai.scheduler.eval_period_epochs";
    pub const SCHEDULER_EVAL_MIN_SAMPLES: &str = "ai.scheduler.eval_min_samples";
    pub const SCHEDULER_COST_PENALTY_DIVISOR_MICROS: &str =
        "ai.scheduler.cost_penalty_divisor_micros";
    pub const SCHEDULER_TOKEN_PENALTY_DIVISOR: &str = "ai.scheduler.token_penalty_divisor";
}
