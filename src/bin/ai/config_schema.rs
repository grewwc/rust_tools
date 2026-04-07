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
    pub const MODEL_DEFAULT: &str = "ai.model.default";
    pub const MODEL_VL_DEFAULT: &str = "ai.model.vl_default";
    pub const MODEL_THINKING: &str = "ai.model.thinking";

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

    // ── Project Writeback ──────────────────────────────────
    pub const PROJECT_WRITEBACK_ENABLE: &str = "ai.project_writeback.enable";
    pub const PROJECT_WRITEBACK_MODEL: &str = "ai.project_writeback.model";
    pub const PROJECT_WRITEBACK_TIMEOUT_MS: &str = "ai.project_writeback.timeout_ms";
}
