use colored::Colorize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};

use crate::ai::mcp::SharedMcpClient;
use crate::ai::{
    code_discovery_policy::{
        CodeDiscoveryRecord, parse_confidence, parse_kind, parse_record_line, recall_limit,
        recall_rank, render_record,
    },
    driver::{print::print_ocr_summary, reflection, skill_runtime},
    history::{Message, build_context_history},
    request,
    tools::storage::memory_store::{AgentMemoryEntry, MemoryStore},
    types::App,
};

use super::types::TurnPreparation;

const CODE_DISCOVERY_PREFIX: &str = "code_discovery:";
const CODE_DISCOVERY_CATEGORY: &str = "code_discovery";
const SESSION_CODE_DISCOVERY_RECALL_PREFIX: &str = "Recent session code discoveries:";
const RECENT_MEMORY_CACHE_TTL: Duration = Duration::from_secs(60);

static RECENT_MEMORY_CACHE: LazyLock<Mutex<Option<RecentMemoryCacheEntry>>> =
    LazyLock::new(|| Mutex::new(None));

#[derive(Debug, Clone, Copy, Default)]
struct QuestionShape {
    char_count: usize,
    nonempty_line_count: usize,
    artifact_token_count: usize,
    has_code_fence: bool,
    has_inline_code: bool,
    has_namespace_path: bool,
    has_list_marker: bool,
}

impl QuestionShape {
    fn analyze(question: &str) -> Self {
        let trimmed = question.trim();
        let mut shape = QuestionShape {
            char_count: trimmed.chars().count(),
            has_code_fence: trimmed.contains("```"),
            has_inline_code: trimmed.contains('`'),
            has_namespace_path: trimmed.contains("::"),
            ..QuestionShape::default()
        };

        for line in trimmed.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            shape.nonempty_line_count += 1;
            shape.has_list_marker |= line_has_list_marker(line);
            shape.artifact_token_count += line
                .split_whitespace()
                .filter(|token| is_artifact_like_token(token))
                .count();
        }

        shape
    }

    fn has_code_or_repo_artifact(self) -> bool {
        self.has_code_fence
            || self.has_inline_code
            || self.has_namespace_path
            || self.artifact_token_count > 0
    }

    fn has_reflection_shape(self) -> bool {
        self.has_code_or_repo_artifact()
            || self.nonempty_line_count >= 2
            || self.has_list_marker
            || self.char_count >= 80
    }

    fn is_complex_task(self) -> bool {
        if self.char_count < 12 {
            return false;
        }
        self.nonempty_line_count >= 3
            || self.has_list_marker
            || self.char_count >= 180
            || self.artifact_token_count >= 2
    }
}

fn line_has_list_marker(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
        || starts_with_ordered_list_marker(trimmed)
}

fn starts_with_ordered_list_marker(line: &str) -> bool {
    let mut chars = line.char_indices().peekable();
    let mut digit_count = 0;
    while let Some((_, ch)) = chars.peek().copied() {
        if !ch.is_ascii_digit() {
            break;
        }
        digit_count += 1;
        chars.next();
    }
    if digit_count == 0 {
        return false;
    }
    let Some((_, marker)) = chars.next() else {
        return false;
    };
    if marker != '.' && marker != ')' {
        return false;
    }
    chars.next().is_some_and(|(_, ch)| ch.is_ascii_whitespace())
}

fn is_artifact_like_token(token: &str) -> bool {
    let token = trim_artifact_token(token);
    if token.is_empty() {
        return false;
    }
    if token.contains('/') || token.contains('\\') {
        return true;
    }
    let path = Path::new(token);
    let has_stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| !stem.trim().is_empty());
    let has_extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(is_probable_file_extension);
    has_stem && has_extension
}

fn trim_artifact_token(token: &str) -> &str {
    token.trim_matches(|ch: char| {
        ch.is_ascii_whitespace()
            || matches!(
                ch,
                '`' | '\'' | '"' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>'
            )
    })
}

fn is_probable_file_extension(extension: &str) -> bool {
    let len = extension.chars().count();
    (1..=8).contains(&len)
        && extension.chars().all(|ch| ch.is_ascii_alphanumeric())
        && extension.chars().any(|ch| ch.is_ascii_alphabetic())
}

fn should_inject_integrated_reflection(
    question: &str,
    intent: &crate::ai::driver::intent_recognition::UserIntent,
) -> bool {
    matches!(
        intent.core,
        crate::ai::driver::intent_recognition::CoreIntent::RequestAction
            | crate::ai::driver::intent_recognition::CoreIntent::SeekSolution
    ) && QuestionShape::analyze(question).has_reflection_shape()
}

fn sync_recall_enabled() -> bool {
    crate::commonw::configw::get_all_config()
        .get_opt("ai.prepare.sync_recall")
        .map(|value| value.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn sync_prepare_observers_enabled() -> bool {
    crate::commonw::configw::get_all_config()
        .get_opt("ai.prepare.sync_observers")
        .map(|value| value.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[derive(Clone, PartialEq, Eq)]
struct RecentMemoryCacheKey {
    path: std::path::PathBuf,
    limit: usize,
    file_len: Option<u64>,
    modified_unix_ms: Option<u128>,
}

struct RecentMemoryCacheEntry {
    created_at: Instant,
    key: RecentMemoryCacheKey,
    entries: Vec<AgentMemoryEntry>,
}

#[crate::ai::agent_hang_span(
    "post-fix",
    "K",
    "turn_runtime::run_turn:prepare_turn",
    "[DEBUG] preparing turn",
    "[DEBUG] prepared turn",
    {
        "history_count": history_count,
        "question_len": question.chars().count(),
        "model": next_model,
    },
    {
        "message_count": __agent_hang_result.as_ref().map(|v| v.messages.len()).unwrap_or(0),
        "turn_message_count": __agent_hang_result
            .as_ref()
            .map(|v| v.turn_messages.len())
            .unwrap_or(0),
        "max_iterations": __agent_hang_result
            .as_ref()
            .map(|v| v.max_iterations)
            .unwrap_or(0),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(super) async fn prepare_turn(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    question: &str,
    attachments_text: &str,
    next_model: &str,
    precomputed_ocr: Option<crate::ai::driver::model::OcrExtraction>,
) -> Result<TurnPreparation, Box<dyn std::error::Error>> {
    let overflow_dir = {
        use crate::ai::history::SessionStore;
        let store = SessionStore::new(app.config.history_file.as_path());
        Some(store.session_assets_dir(&app.session_id))
    };
    crate::ai::driver::runtime_ctx::publish_subagent_phase("preparing context");
    let history = build_context_history(
        history_count,
        &app.session_history_file,
        app.config.history_max_chars,
        app.config.history_keep_last,
        app.config.history_summary_max_chars,
        overflow_dir,
    )?;
    let mut skill_turn = {
        let mc = mcp_client.lock().unwrap();
        skill_runtime::prepare_skill_for_turn(app, &mc, skill_manifests, question)
    };

    crate::ai::driver::runtime_ctx::publish_subagent_phase("recognizing intent");
    // LLM 意图识别 fallback：当本地 TF-IDF 把 question 划到 Casual 但内容具备
    // 非闲聊形态（结构化文本、路径/代码片段、问句或长度足够）时，调用大模型
    // 二次判定。LLM 调用本身在 stderr 打印 [intent:llm] 标识，方便用户区分
    // "本地分类" vs "大模型分类"。
    {
        let local = skill_turn.intent().clone();
        let upgraded = crate::ai::driver::intent_recognition::upgrade_intent_via_model(
            app,
            question,
            local.clone(),
        )
        .await;
        if upgraded.core != local.core {
            skill_turn.set_intent(upgraded);
        }
    }

    {
        let now = chrono::Local::now();
        let date_str = now.format("%Y-%m-%d").to_string();
        skill_turn.push_labeled_section(
            skill_runtime::ContextKind::Fact,
            "Current Date",
            &format!("Today's date is {}.", date_str),
        );
    }

    let mut messages = Vec::with_capacity(history.len() + 2);

    {
        let integrated = crate::commonw::configw::get_all_config()
            .get_opt("ai.critic_revise.integrated")
            .unwrap_or_else(|| "true".to_string())
            .trim()
            .ne("false");
        let reflect_integrated = crate::commonw::configw::get_all_config()
            .get_opt("ai.reflection.integrated")
            .unwrap_or_else(|| "true".to_string())
            .trim()
            .ne("false");
        let intent_needs_reflection =
            should_inject_integrated_reflection(question, skill_turn.intent());
        if (integrated || reflect_integrated) && intent_needs_reflection {
            let mut sys = String::new();
            if integrated {
                sys.push_str("Before replying, internally perform a brief CRITIC→REVISE pass to ensure correctness, missing steps, and clear structure. Do not output the critic. Output only the final improved answer.\n");
            }
            if reflect_integrated {
                sys.push_str("At the very end of your message, include a compact self experience note enclosed within <meta:self_note> and </meta:self_note>. The note should be 2-6 short bullets grouped under 'Do:' and 'Avoid:'. Do not mention these tags in the visible content.\n");
            }
            if !sys.is_empty() {
                skill_turn.push_section(skill_runtime::ContextKind::Behavior, &sys);
            }
        }
    }

    let observer_outputs: Vec<crate::ai::driver::observer::PrepareOutput> =
        if sync_prepare_observers_enabled() {
            app.observers.iter_mut().filter_map(|obs| {
            if obs.is_poisoned() {
                return None;
            }
            let ctx = crate::ai::driver::observer::PrepareContext {
                question: question.to_string(),
            };
            let obs_name = obs.name().to_string();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                obs.on_prepare_rich(&ctx)
            })) {
                Ok(out) => Some(out),
                Err(_) => {
                    eprintln!("[Warning] observer '{}' panicked in on_prepare; disabling for rest of conversation.", obs_name);
                    obs.mark_poisoned();
                    None
                }
            }
        }).collect()
        } else {
            Vec::new()
        };
    for output in &observer_outputs {
        for (kind, label, content) in &output.sections {
            match kind {
                crate::ai::driver::observer::SectionKind::Behavior => {
                    skill_turn.push_section(skill_runtime::ContextKind::Behavior, content);
                    let _ = label;
                }
                crate::ai::driver::observer::SectionKind::Fact => {
                    skill_turn.push_labeled_section(
                        skill_runtime::ContextKind::Fact,
                        label,
                        content,
                    );
                }
            }
        }
    }
    let suggested_tool_calls_aggregated: Vec<_> = observer_outputs
        .iter()
        .flat_map(|o| o.suggested_tool_calls.clone())
        .collect();
    if !suggested_tool_calls_aggregated.is_empty() {
        let mut block = String::from(
            "Thinking engine proposes the following verification-driven tool calls BEFORE answering. \
             Consider them as high-priority candidates:\n",
        );
        for sc in &suggested_tool_calls_aggregated {
            block.push_str(&format!(
                "- {} (rationale: {})\n  args: {}\n",
                sc.tool_name, sc.rationale, sc.arguments
            ));
        }
        skill_turn.push_section(skill_runtime::ContextKind::Behavior, &block);
    }

    let recall_intent = skill_turn.intent().clone();
    let skip_recall_for_skill_context = skill_turn.skip_recall_by_skill();
    let matched_skill_name = skill_turn.matched_skill_name().map(|name| name.to_string());
    let should_run_general_recall = sync_recall_enabled()
        && should_run_general_recall(
            question,
            &recall_intent,
            matched_skill_name.as_deref(),
            skip_recall_for_skill_context,
        );
    if should_run_general_recall {
        let recall_bundle = reflection::build_recall_bundle(question, 1200, 2000);
        if let Some(guidelines) = recall_bundle.guidelines {
            if !guidelines.trim().is_empty() {
                skill_turn.push_labeled_section(
                    skill_runtime::ContextKind::Fact,
                    "Guidelines",
                    &guidelines,
                );
            }
        }
        if let Some(recalled) = recall_bundle.recalled
            && !recalled.content.trim().is_empty()
        {
            let project_part = recalled
                .project_hint
                .as_deref()
                .map(|project| format!(" project={project}"))
                .unwrap_or_default();
            let category_part = if recalled.categories.is_empty() {
                String::new()
            } else {
                format!(" categories={}", recalled.categories.join(","))
            };
            let confidence_part = if recalled.high_confidence_project_memory {
                " high_confidence=true"
            } else {
                " high_confidence=false"
            };
            println!(
                "{} count={}{}{}{}",
                "[Memory] recalled".bright_blue().bold(),
                recalled.entry_count,
                project_part,
                category_part,
                confidence_part
            );
            skill_turn.push_labeled_section(
                skill_runtime::ContextKind::Fact,
                "Recalled Knowledge",
                &recalled.content,
            );
            if recalled.high_confidence_project_memory {
                skill_turn.push_section(
                    skill_runtime::ContextKind::Policy,
                    high_confidence_project_memory_policy(),
                );
            } else {
                skill_turn.push_section(
                    skill_runtime::ContextKind::Policy,
                    recalled_knowledge_usage_policy(),
                );
            }
        }
    }

    if sync_recall_enabled()
        && should_run_session_code_discovery_recall(
            question,
            &recall_intent,
            matched_skill_name.as_deref(),
            skip_recall_for_skill_context,
        )
        && let Some(code_discovery_recall) = build_session_code_discovery_recall(app, &history)
    {
        println!(
            "{} session={}",
            "[Memory] code_discovery recalled".bright_blue().bold(),
            app.session_id
        );
        skill_turn.push_labeled_section(
            skill_runtime::ContextKind::Fact,
            "Code Discovery",
            &code_discovery_recall,
        );
    }

    // C3: 复杂任务自动提示（不强制激活 Thinking 引擎，仅作为软引导）
    // 仅依据多行、列表、长度和多 artifact 等形态信号判断，避免词面关键词误触发。
    if detect_complex_task(question) {
        skill_turn.push_section(
            skill_runtime::ContextKind::Policy,
            "Complex task hint:\n\
             - This request looks multi-step (refactor / design / architecture / cross-module).\n\
             - Before editing, briefly outline the plan: what to read, what to change, how to verify.\n\
             - Prefer small reversible steps; surface assumptions explicitly.",
        );
    }

    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(skill_turn.system_prompt().to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    messages.extend(history);
    // Per-turn context reminder (Current Date / Recalled Knowledge / Code
    // Discovery, …) used to be injected as a synthetic user+assistant pair
    // between `history` and the current user message. Because the reminder
    // text changes every turn, that pair sat right between two cache-stable
    // segments and caused providers to lose the prompt-cache hit on
    // everything from the reminder onward. Fold it into the **current**
    // user message instead: the current message is always a cache miss
    // anyway, so reminder churn no longer truncates the cached prefix.
    // The `turn_messages` list (what gets persisted to long-term history)
    // intentionally keeps the original user question without the reminder.
    let context_reminder = skill_turn.context_reminder();
    let user_message = Message {
        role: "user".to_string(),
        content: {
            let has_images = !app.attached_image_files.is_empty();
            let mut final_question = if attachments_text.is_empty() {
                question.to_string()
            } else if attachments_text.ends_with('\n') {
                format!("{}{}", attachments_text, question)
            } else {
                format!("{}\n{}", attachments_text, question)
            };
            if has_images
                && !crate::ai::models::supports_image_input(next_model)
                && let Some(ocr) = precomputed_ocr
                && ocr.has_usable_text()
            {
                print_ocr_summary(&ocr);
                final_question = format!(
                    "{}\n\n[Auto OCR From Attached Images via {}]\n{}",
                    final_question, ocr.tool_name, ocr.content
                );
            }
            request::build_content(next_model, &final_question, &app.attached_image_files)?
        },
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    let request_user_message = if let Some(reminder) = context_reminder.as_deref() {
        let mut decorated = user_message.clone();
        decorated.content = match decorated.content {
            Value::String(text) => Value::String(format!("{}\n\n{}", reminder, text)),
            Value::Array(mut parts) => {
                parts.insert(
                    0,
                    serde_json::json!({
                        "type": "text",
                        "text": reminder,
                    }),
                );
                Value::Array(parts)
            }
            other => other,
        };
        decorated
    } else {
        user_message.clone()
    };
    messages.push(request_user_message);
    let mut turn_messages = Vec::with_capacity(8);
    turn_messages.push(user_message);

    let max_iterations = app
        .agent_context
        .as_ref()
        .map(|c| c.max_iterations)
        .unwrap_or(0)
        .max(1);

    Ok(TurnPreparation {
        skill_turn,
        messages,
        turn_messages,
        persisted_turn_messages: 0,
        max_iterations,
    })
}

fn should_run_general_recall(
    question: &str,
    intent: &crate::ai::driver::intent_recognition::UserIntent,
    matched_skill_name: Option<&str>,
    skip_recall_for_skill_context: bool,
) -> bool {
    if skip_recall_for_skill_context {
        return false;
    }
    if matched_skill_name.is_some() {
        return true;
    }

    let question = question.trim();
    if question.is_empty() {
        return false;
    }

    let question_len = question.chars().count();
    let simple_concept_turn = question_len <= 120
        && matches!(
            intent.core,
            crate::ai::driver::intent_recognition::CoreIntent::Casual
                | crate::ai::driver::intent_recognition::CoreIntent::QueryConcept
        )
        && !intent.is_search_query()
        && !looks_like_code_or_repo_question(question);

    !simple_concept_turn
}

fn should_run_session_code_discovery_recall(
    question: &str,
    intent: &crate::ai::driver::intent_recognition::UserIntent,
    matched_skill_name: Option<&str>,
    skip_recall_for_skill_context: bool,
) -> bool {
    if skip_recall_for_skill_context {
        return false;
    }
    if matched_skill_name.is_some() {
        return true;
    }
    matches!(
        intent.core,
        crate::ai::driver::intent_recognition::CoreIntent::RequestAction
            | crate::ai::driver::intent_recognition::CoreIntent::SeekSolution
    ) && looks_like_code_or_repo_question(question)
}

fn looks_like_code_or_repo_question(question: &str) -> bool {
    QuestionShape::analyze(question).has_code_or_repo_artifact()
}

/// C3: 复杂任务检测——仅基于结构信号的轻量启发式。
/// 命中后只会注入一段 Policy 提示鼓励 agent 自行拆解，不强制激活 Thinking 引擎。
fn detect_complex_task(question: &str) -> bool {
    QuestionShape::analyze(question).is_complex_task()
}

fn build_session_code_discovery_recall(app: &App, history: &[Message]) -> Option<String> {
    let existing = extract_existing_code_discoveries(history);
    let entries = recent_memory_entries(200)?;
    let discoveries = collect_session_code_discovery_records(
        &entries,
        &format!("session:{}", app.session_id),
        &existing,
    );
    render_session_code_discovery_recall(&discoveries)
}

fn recent_memory_entries(limit: usize) -> Option<Vec<AgentMemoryEntry>> {
    let store = MemoryStore::from_env_or_config();
    let key = recent_memory_cache_key(store.path(), limit);
    if let Some(entries) = try_get_recent_memory_cache(&key) {
        return Some(entries);
    }

    let entries = store.recent(limit).ok()?;
    store_recent_memory_cache(key, entries.clone());
    Some(entries)
}

fn recent_memory_cache_key(path: &Path, limit: usize) -> RecentMemoryCacheKey {
    let metadata = std::fs::metadata(path).ok();
    RecentMemoryCacheKey {
        path: path.to_path_buf(),
        limit,
        file_len: metadata.as_ref().map(|m| m.len()),
        modified_unix_ms: metadata
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(system_time_millis),
    }
}

fn system_time_millis(value: SystemTime) -> Option<u128> {
    value
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis())
}

fn try_get_recent_memory_cache(key: &RecentMemoryCacheKey) -> Option<Vec<AgentMemoryEntry>> {
    let Ok(mut cache) = RECENT_MEMORY_CACHE.lock() else {
        return None;
    };
    if let Some(entry) = cache.as_ref()
        && entry.created_at.elapsed() < RECENT_MEMORY_CACHE_TTL
        && &entry.key == key
    {
        return Some(entry.entries.clone());
    }
    *cache = None;
    None
}

fn store_recent_memory_cache(key: RecentMemoryCacheKey, entries: Vec<AgentMemoryEntry>) {
    let Ok(mut cache) = RECENT_MEMORY_CACHE.lock() else {
        return;
    };
    *cache = Some(RecentMemoryCacheEntry {
        created_at: Instant::now(),
        key,
        entries,
    });
}

fn extract_existing_code_discoveries(messages: &[Message]) -> BTreeSet<CodeDiscoveryRecord> {
    let mut out = BTreeSet::new();
    for message in messages {
        let Value::String(content) = &message.content else {
            continue;
        };
        if !content.starts_with(CODE_DISCOVERY_PREFIX) {
            continue;
        }
        for line in content[CODE_DISCOVERY_PREFIX.len()..].lines() {
            if let Some(record) = parse_record_line(line) {
                out.insert(record);
            }
        }
    }
    out
}

fn collect_session_code_discovery_records(
    entries: &[AgentMemoryEntry],
    session_source: &str,
    existing_records: &BTreeSet<CodeDiscoveryRecord>,
) -> Vec<CodeDiscoveryRecord> {
    let mut seen = existing_records.clone();
    let mut discoveries = Vec::new();
    for entry in entries {
        if entry.category != CODE_DISCOVERY_CATEGORY {
            continue;
        }
        if entry.source.as_deref() != Some(session_source) {
            continue;
        }
        let Some(record) = code_discovery_record_from_memory_entry(entry) else {
            continue;
        };
        if !seen.insert(record.clone()) {
            continue;
        }
        discoveries.push(record);
    }
    discoveries.sort_by(|a, b| {
        recall_rank(b)
            .cmp(&recall_rank(a))
            .then_with(|| a.finding.cmp(&b.finding))
    });
    discoveries.truncate(recall_limit());
    discoveries
}

fn render_session_code_discovery_recall(discoveries: &[CodeDiscoveryRecord]) -> Option<String> {
    if discoveries.is_empty() {
        return None;
    }
    let mut out = String::from(SESSION_CODE_DISCOVERY_RECALL_PREFIX);
    out.push('\n');
    for record in discoveries {
        out.push_str(&render_record(record));
        out.push('\n');
    }
    out.push_str(
        "Treat these as stable findings from earlier in this session. Prioritize high-confidence items, use medium-confidence as support, and reuse them before rerunning equivalent repo inspection unless verification or a narrower slice is needed.\n",
    );
    Some(out)
}

fn high_confidence_project_memory_policy() -> &'static str {
    "Memory-first project answer policy:\n- High-confidence project memory is available. Answer from it first only when it already covers the ask and the answer does not depend on current repository state.\n- If the answer depends on current code, files, configs, command results, or any potentially changed runtime/project state, verify with file/search/inspection tools before concluding.\n- Only skip repo/tool verification when the recalled knowledge is sufficient and the request is not state-sensitive."
}

fn recalled_knowledge_usage_policy() -> &'static str {
    "Knowledge usage policy:\n- Recalled knowledge is relevant for this turn; use it as context, not as a substitute for current-state verification.\n- If the answer depends on current code, files, configs, command results, or any potentially changed runtime/project state, verify with file/repo tools before concluding.\n- Use file/repo tools when key requested details are missing, ambiguous, or state-sensitive; avoid full re-scan when recall is already sufficient."
}

fn code_discovery_record_from_memory_entry(
    entry: &AgentMemoryEntry,
) -> Option<CodeDiscoveryRecord> {
    let mut confidence = None;
    let mut kind = None;
    for tag in &entry.tags {
        if let Some(value) = tag.strip_prefix("confidence:") {
            confidence = parse_confidence(value.trim());
        } else if let Some(value) = tag.strip_prefix("kind:") {
            kind = parse_kind(value.trim());
        }
    }
    Some(CodeDiscoveryRecord {
        finding: entry.note.trim().to_string(),
        confidence: confidence?,
        kind: kind?,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        collect_session_code_discovery_records, extract_existing_code_discoveries,
        high_confidence_project_memory_policy, looks_like_code_or_repo_question,
        recalled_knowledge_usage_policy, render_session_code_discovery_recall,
        should_inject_integrated_reflection, should_run_general_recall,
        should_run_session_code_discovery_recall,
    };
    use crate::ai::code_discovery_policy::parse_record_line;
    use crate::ai::driver::intent_recognition::{CoreIntent, UserIntent};
    use crate::ai::history::Message;
    use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
    use serde_json::Value;
    use std::collections::BTreeSet;

    fn memory_entry(
        note: impl Into<String>,
        source: &str,
        kind: &str,
        confidence: &str,
    ) -> AgentMemoryEntry {
        AgentMemoryEntry {
            id: None,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            category: "code_discovery".to_string(),
            note: note.into(),
            tags: vec![format!("kind:{kind}"), format!("confidence:{confidence}")],
            source: Some(source.to_string()),
            priority: Some(180),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        }
    }

    #[test]
    fn extract_existing_code_discovery_lines_reads_history_messages() {
        let history = vec![Message {
            role: "system".to_string(),
            content: Value::String(
                "code_discovery:\n- [confidence=high kind=error_site] code_search(operation=text_search, query=panic) => src/main.rs:42: panic!(\"boom\")\n- [confidence=high kind=symbol] read_file_lines(file=src/main.rs, lines=40..50) => fn crash() {".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let lines = extract_existing_code_discoveries(&history);
        assert!(lines.contains(&parse_record_line("- [confidence=high kind=error_site] code_search(operation=text_search, query=panic) => src/main.rs:42: panic!(\"boom\")").unwrap()));
        assert!(lines.contains(&parse_record_line("- [confidence=high kind=symbol] read_file_lines(file=src/main.rs, lines=40..50) => fn crash() {").unwrap()));
    }

    #[test]
    fn collect_session_code_discovery_lines_filters_by_session_and_dedupes() {
        let entries = vec![
            AgentMemoryEntry {
                id: None,
                timestamp: "2026-01-01T00:00:00Z".to_string(),
                category: "code_discovery".to_string(),
                note: "read_file_lines(file=src/main.rs, lines=40..50) => fn crash() {".to_string(),
                tags: vec!["kind:symbol".to_string(), "confidence:high".to_string()],
                source: Some("session:abc".to_string()),
                priority: Some(180),
                owner_pid: None,
                owner_pgid: None,
                image_path: None,
            },
            AgentMemoryEntry {
                id: None,
                timestamp: "2026-01-01T00:01:00Z".to_string(),
                category: "code_discovery".to_string(),
                note: "code_search(operation=text_search, query=panic) => src/main.rs:42: panic!(\"boom\")".to_string(),
                tags: vec!["kind:error_site".to_string(), "confidence:high".to_string()],
                source: Some("session:abc".to_string()),
                priority: Some(180),
                owner_pid: None,
                owner_pgid: None,
                image_path: None,
            },
            AgentMemoryEntry {
                id: None,
                timestamp: "2026-01-01T00:02:00Z".to_string(),
                category: "code_discovery".to_string(),
                note: "read_file_lines(file=src/main.rs, lines=40..50) => fn crash() {".to_string(),
                tags: vec!["kind:symbol".to_string(), "confidence:high".to_string()],
                source: Some("session:xyz".to_string()),
                priority: Some(180),
                owner_pid: None,
                owner_pgid: None,
                image_path: None,
            },
        ];
        let mut existing = BTreeSet::new();
        existing.insert(
            parse_record_line(
                "- [confidence=high kind=symbol] read_file_lines(file=src/main.rs, lines=40..50) => fn crash() {",
            )
            .unwrap(),
        );

        let lines = collect_session_code_discovery_records(&entries, "session:abc", &existing);
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0],
            parse_record_line(
                "- [confidence=high kind=error_site] code_search(operation=text_search, query=panic) => src/main.rs:42: panic!(\"boom\")",
            )
            .unwrap()
        );
    }

    #[test]
    fn collect_session_code_discovery_ranks_all_recent_candidates_before_truncating() {
        let mut entries = (0..16)
            .map(|idx| {
                memory_entry(
                    format!("find_path(query=todo-{idx}) => TODO item {idx}"),
                    "session:abc",
                    "todo",
                    "low",
                )
            })
            .collect::<Vec<_>>();
        entries.push(memory_entry(
            "read_file_lines(file=src/main.rs, lines=10..20) => root cause: missing config",
            "session:abc",
            "root_cause",
            "high",
        ));

        let lines =
            collect_session_code_discovery_records(&entries, "session:abc", &BTreeSet::new());
        assert!(
            lines
                .iter()
                .any(|record| { record.finding.contains("root cause: missing config") })
        );
    }

    #[test]
    fn render_session_code_discovery_recall_formats_system_note() {
        let note = render_session_code_discovery_recall(&[parse_record_line(
            "- [confidence=high kind=error_site] code_search(operation=text_search, query=panic) => src/main.rs:42: panic!(\"boom\")",
        )
        .unwrap()])
        .expect("note");

        assert!(note.contains("Recent session code discoveries:"));
        assert!(note.contains("confidence=high kind=error_site"));
        assert!(note.contains("Treat these as stable findings"));
    }

    #[test]
    fn high_confidence_project_memory_policy_requires_state_verification() {
        let policy = high_confidence_project_memory_policy();
        assert!(policy.contains("does not depend on current repository state"));
        assert!(policy.contains("verify with file/search/inspection tools before concluding"));
        assert!(policy.contains("request is not state-sensitive"));
    }

    #[test]
    fn recalled_knowledge_usage_policy_treats_memory_as_context_not_ground_truth() {
        let policy = recalled_knowledge_usage_policy();
        assert!(policy.contains("use it as context, not as a substitute"));
        assert!(policy.contains("verify with file/repo tools before concluding"));
        assert!(policy.contains("missing, ambiguous, or state-sensitive"));
    }

    #[test]
    fn simple_concept_turn_skips_general_recall() {
        let intent = UserIntent::new(CoreIntent::QueryConcept);
        assert!(!should_run_general_recall(
            "Rust 的 trait 是什么？",
            &intent,
            None,
            false
        ));
    }

    #[test]
    fn simple_common_sense_turn_skips_integrated_reflection_even_if_misclassified() {
        let intent = UserIntent::new(CoreIntent::SeekSolution);
        assert!(!should_inject_integrated_reflection(
            "为什么天是蓝的？",
            &intent
        ));
    }

    #[test]
    fn simple_technical_concept_turn_skips_integrated_reflection_even_if_misclassified() {
        let intent = UserIntent::new(CoreIntent::SeekSolution);
        assert!(!should_inject_integrated_reflection(
            "Rust 的函数是什么？",
            &intent
        ));
    }

    #[test]
    fn coding_task_keeps_integrated_reflection() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        assert!(should_inject_integrated_reflection(
            "帮我处理 `build check` 的 failure",
            &intent
        ));
    }

    #[test]
    fn plain_action_words_do_not_trigger_reflection_without_structure() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        assert!(!should_inject_integrated_reflection(
            "帮我处理这个问题",
            &intent
        ));
    }

    #[test]
    fn generic_file_extension_counts_as_code_or_repo_artifact() {
        assert!(looks_like_code_or_repo_question(
            "看一下 schema.proto 的生成逻辑"
        ));
    }

    #[test]
    fn numeric_decimal_does_not_count_as_code_or_repo_artifact() {
        assert!(!looks_like_code_or_repo_question("圆周率约等于 3.14"));
    }

    #[test]
    fn code_request_keeps_session_code_discovery_recall() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        assert!(should_run_session_code_discovery_recall(
            "帮我看下 src/main.rs 这里的 panic",
            &intent,
            None,
            false
        ));
    }

    #[test]
    fn matched_skill_keeps_general_recall_enabled() {
        let intent = UserIntent::new(CoreIntent::Casual);
        assert!(should_run_general_recall(
            "简短请求",
            &intent,
            Some("debugger"),
            false
        ));
    }
}
