use colored::Colorize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};

use crate::ai::mcp::McpClient;
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
const RECENT_MEMORY_CACHE_TTL: Duration = Duration::from_secs(10);

static RECENT_MEMORY_CACHE: LazyLock<Mutex<Option<RecentMemoryCacheEntry>>> =
    LazyLock::new(|| Mutex::new(None));

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
    mcp_client: &mut McpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    question: &str,
    next_model: &str,
    precomputed_ocr: Option<crate::ai::driver::model::OcrExtraction>,
) -> Result<TurnPreparation, Box<dyn std::error::Error>> {
    let history = build_context_history(
        history_count,
        &app.session_history_file,
        app.config.history_max_chars,
        app.config.history_keep_last,
        app.config.history_summary_max_chars,
    )?;
    let mut skill_turn =
        skill_runtime::prepare_skill_for_turn(app, mcp_client, skill_manifests, question)
            .await;

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
        if integrated || reflect_integrated {
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

    let recall_intent = skill_turn.intent().clone();
    let skip_recall_for_skill_context = skill_turn.skip_recall_by_skill();
    let matched_skill_name = skill_turn.matched_skill_name().map(|name| name.to_string());
    let should_run_general_recall = should_run_general_recall(
        question,
        &recall_intent,
        matched_skill_name.as_deref(),
        skip_recall_for_skill_context,
    );
    if should_run_general_recall {
        let recall_bundle = reflection::build_recall_bundle(question, 1200, 2000);
        if let Some(guidelines) = recall_bundle.guidelines {
            if !guidelines.trim().is_empty() {
                skill_turn.push_labeled_section(skill_runtime::ContextKind::Fact, "Guidelines", &guidelines);
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
            skill_turn.push_labeled_section(skill_runtime::ContextKind::Fact, "Recalled Knowledge", &recalled.content);
            if recalled.high_confidence_project_memory {
                skill_turn.push_section(
                    skill_runtime::ContextKind::Policy,
                    "Memory-first project answer policy:\n- High-confidence project memory is available. Answer from it first when it already covers the ask.\n- Only use file/search tools for missing details, ambiguity, or explicit verification against current code.",
                );
            } else {
                skill_turn.push_section(
                    skill_runtime::ContextKind::Policy,
                    "Knowledge usage policy:\n- Recalled knowledge is relevant for this turn; build the answer primarily from it.\n- Use file/repo tools only when key requested details are missing; avoid full re-scan when recall is sufficient.",
                );
            }
        }
    }

    if should_run_session_code_discovery_recall(
        question,
        &recall_intent,
        matched_skill_name.as_deref(),
        skip_recall_for_skill_context,
    ) && let Some(code_discovery_recall) = build_session_code_discovery_recall(app, &history)
    {
        println!(
            "{} session={}",
            "[Memory] code_discovery recalled".bright_blue().bold(),
            app.session_id
        );
        skill_turn.push_labeled_section(skill_runtime::ContextKind::Fact, "Code Discovery", &code_discovery_recall);
    }

    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(skill_turn.system_prompt().to_string()),
        tool_calls: None,
        tool_call_id: None,
    });
    messages.extend(history);
    let context_reminder = skill_turn.context_reminder();
    if let Some(reminder) = &context_reminder {
        messages.push(Message {
            role: "user".to_string(),
            content: Value::String(reminder.clone()),
            tool_calls: None,
            tool_call_id: None,
        });
        messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String("Understood. I will use the provided context when relevant.".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    let user_message = Message {
        role: "user".to_string(), 
        content: {
            let has_images = !app.attached_image_files.is_empty();
            let mut final_question = question.to_string();
            if has_images
                && !crate::ai::models::supports_image_input(next_model)
                && let Some(ocr) = precomputed_ocr
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
    };
    messages.push(user_message.clone());
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
    let question = question.trim();
    if question.is_empty() {
        return false;
    }

    if question.contains("```") || question.contains('`') || question.contains("::") {
        return true;
    }

    question.split_whitespace().any(|token| {
        token.contains('/')
            || token.contains('\\')
            || token.ends_with(".rs")
            || token.ends_with(".ts")
            || token.ends_with(".tsx")
            || token.ends_with(".js")
            || token.ends_with(".jsx")
            || token.ends_with(".py")
            || token.ends_with(".go")
            || token.ends_with(".java")
            || token.ends_with(".json")
            || token.ends_with(".yaml")
            || token.ends_with(".yml")
            || token.ends_with(".toml")
    })
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
        if discoveries.len() >= 16 {
            break;
        }
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

fn code_discovery_record_from_memory_entry(entry: &AgentMemoryEntry) -> Option<CodeDiscoveryRecord> {
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
        render_session_code_discovery_recall, should_run_general_recall,
        should_run_session_code_discovery_recall,
    };
    use crate::ai::driver::intent_recognition::{CoreIntent, UserIntent};
    use crate::ai::code_discovery_policy::parse_record_line;
    use crate::ai::history::Message;
    use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
    use serde_json::Value;
    use std::collections::BTreeSet;

    #[test]
    fn extract_existing_code_discovery_lines_reads_history_messages() {
        let history = vec![Message {
            role: "system".to_string(),
            content: Value::String(
                "code_discovery:\n- [confidence=high kind=error_site] code_search(operation=text_search, query=panic) => src/main.rs:42: panic!(\"boom\")\n- [confidence=high kind=symbol] read_file_lines(file=src/main.rs, lines=40..50) => fn crash() {".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
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
            },
            AgentMemoryEntry {
                id: None,
                timestamp: "2026-01-01T00:01:00Z".to_string(),
                category: "code_discovery".to_string(),
                note: "code_search(operation=text_search, query=panic) => src/main.rs:42: panic!(\"boom\")".to_string(),
                tags: vec!["kind:error_site".to_string(), "confidence:high".to_string()],
                source: Some("session:abc".to_string()),
                priority: Some(180),
            },
            AgentMemoryEntry {
                id: None,
                timestamp: "2026-01-01T00:02:00Z".to_string(),
                category: "code_discovery".to_string(),
                note: "read_file_lines(file=src/main.rs, lines=40..50) => fn crash() {".to_string(),
                tags: vec!["kind:symbol".to_string(), "confidence:high".to_string()],
                source: Some("session:xyz".to_string()),
                priority: Some(180),
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
