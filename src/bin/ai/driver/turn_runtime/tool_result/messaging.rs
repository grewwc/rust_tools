use rust_tools::cw::{SkipMap, SkipSet};
use serde_json::Value;
use std::collections::{BTreeSet};

use crate::ai::{
    code_discovery_policy::{
        CodeDiscoveryRecord, classify_finding, confidence_label, kind_label, persistence_limit,
        priority_for_confidence, render_record, should_persist,
    },
    driver::{
        print::{print_tool_note_line, print_tool_output_block},
        tools::ExecuteToolCallsResult,
    },
    history::{Message, ROLE_INTERNAL_NOTE, is_system_like_role},
    types::ToolCall,
    types::App,
};

use super::execution::prepare_tool_result;
use super::super::types::PreparedToolResult;

const CODE_INSPECTION_MEMORY_PREFIX: &str = "Current code-inspection working memory:";
const CODE_DISCOVERY_PREFIX: &str = "code_discovery:";
const CODE_DISCOVERY_CATEGORY: &str = "code_discovery";

#[derive(Debug, Clone)]
struct RepoInspectionFinding {
    tool_name: String,
    rendered: String,
    highlight: String,
}

pub(super) fn append_message_pair(
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    message: Message,
) {
    messages.push(message.clone());
    turn_messages.push(message);
}

pub(super) fn record_hidden_self_note(app: &App, turn_messages: &mut Vec<Message>, hidden_meta: &str) {
    let hidden_meta = hidden_meta.trim();
    if hidden_meta.is_empty() {
        return;
    }

    let record = Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!("self_note:\n{hidden_meta}")),
        tool_calls: None,
        tool_call_id: None,
    };
    turn_messages.push(record);

    let entry = crate::ai::tools::storage::memory_store::AgentMemoryEntry {
        id: None,
        timestamp: chrono::Local::now().to_rfc3339(),
        category: "self_note".to_string(),
        note: hidden_meta.to_string(),
        tags: vec!["agent".to_string(), "policy".to_string()],
        source: Some(format!("session:{}", app.session_id)),
        priority: Some(255),
        owner_pid: None,
        owner_pgid: None,
    };
    let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

pub(super) fn append_cached_tool_results_note(
    exec_result: &ExecuteToolCallsResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) {
    if !exec_result.cached_hits.iter().any(|hit| *hit) {
        return;
    }

    let cached_names = exec_result
        .executed_tool_calls
        .iter()
        .zip(exec_result.cached_hits.iter())
        .filter_map(|(tool_call, cached)| cached.then_some(tool_call.function.name.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let cache_note = Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!(
            "Context note: reused cached tool results from the current session for identical calls within the recent TTL. Treat these results as already verified context unless the user asks to refresh. Tools: {cached_names}"
        )),
        tool_calls: None,
        tool_call_id: None,
    };
    append_message_pair(messages, turn_messages, cache_note);
}

pub(super) fn print_tool_result_preview(_tool_name: &str, prepared: &PreparedToolResult) {
    print_tool_note_line("output", "tool result");
    print_tool_output_block(&prepared.content_for_terminal);
}

pub(super) fn append_tool_result_messages(
    app: &mut App,
    stream_assistant_text: &str,
    exec_result: &ExecuteToolCallsResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) {
    let assistant_msg = Message {
        role: "assistant".to_string(),
        content: Value::String(stream_assistant_text.to_string()),
        tool_calls: Some(exec_result.executed_tool_calls.clone()),
        tool_call_id: None,
    };
    append_message_pair(messages, turn_messages, assistant_msg);

    for (tool_call, result) in exec_result
        .executed_tool_calls
        .iter()
        .zip(exec_result.tool_results.iter())
    {
        let prepared = prepare_tool_result(app, &tool_call.function.name, &result.content);
        for obs in app.observers.iter_mut() {
            if obs.is_poisoned() {
                continue;
            }
            let ctx = crate::ai::driver::observer::ToolResultContext {
                tool_name: tool_call.function.name.clone(),
                result_content: result.content.clone(),
                success: {
                    let content_lower = result.content.to_lowercase();
                    let is_execution_tool = tool_call.function.name == "execute_command"
                        || tool_call.function.name == "run_command"
                        || tool_call.function.name == "shell"
                        || tool_call.function.name == "bash";
                    if is_execution_tool {
                        !content_lower.contains("error:") && !content_lower.contains("exit code")
                            && !content_lower.contains("command not found")
                            && !content_lower.contains("permission denied")
                    } else {
                        !content_lower.starts_with("error:") && !content_lower.starts_with("failed:")
                    }
                },
            };
            let obs_name = obs.name().to_string();
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                obs.on_tool_result(&ctx);
            })).is_err() {
                eprintln!("[Warning] observer '{}' panicked in on_tool_result; disabling for rest of conversation.", obs_name);
                obs.mark_poisoned();
            }
        }
        let tool_message = Message {
            role: "tool".to_string(),
            content: Value::String(prepared.content_for_model),
            tool_calls: None,
            tool_call_id: Some(result.tool_call_id.clone()),
        };
        append_message_pair(messages, turn_messages, tool_message);
    }
}

pub(super) fn append_code_inspection_working_memory(
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
) {
    let Some(note) = build_code_inspection_working_memory(turn_messages) else {
        return;
    };

    let duplicate = messages.iter().rev().find_map(|message| {
        if !is_system_like_role(&message.role) {
            return None;
        }
        match &message.content {
            Value::String(content) if content.starts_with(CODE_INSPECTION_MEMORY_PREFIX) => {
                Some(content == &note)
            }
            _ => None,
        }
    });
    if duplicate == Some(true) {
        return;
    }

    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
    });
}

pub(super) fn record_persistent_code_discoveries(
    app: &App,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) {
    let discoveries = build_persistent_code_discoveries(turn_messages);
    if discoveries.is_empty() {
        return;
    }

    let body = discoveries
        .iter()
        .map(render_record)
        .collect::<Vec<_>>()
        .join("\n");
    if persistent_code_discovery_already_present(messages, &body)
        || persistent_code_discovery_already_present(turn_messages, &body)
    {
        return;
    }

    let record = Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!("{CODE_DISCOVERY_PREFIX}\n{body}")),
        tool_calls: None,
        tool_call_id: None,
    };
    append_message_pair(messages, turn_messages, record);

    let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
    let session_source = format!("session:{}", app.session_id);
    for discovery in &discoveries {
        let entry = crate::ai::tools::storage::memory_store::AgentMemoryEntry {
            id: None,
            timestamp: chrono::Local::now().to_rfc3339(),
            category: CODE_DISCOVERY_CATEGORY.to_string(),
            note: discovery.finding.clone(),
            tags: vec![
                "code".to_string(),
                "debug".to_string(),
                "session".to_string(),
                format!("kind:{}", kind_label(discovery.kind)),
                format!("confidence:{}", confidence_label(discovery.confidence)),
            ],
            source: Some(session_source.clone()),
            priority: Some(priority_for_confidence(discovery.confidence)),
            owner_pid: None,
            owner_pgid: None,
        };
        let _ = store.append(&entry);
    }
    store.maintain_after_append();
}

pub(super) fn record_final_stream_response(
    app: &App,
    stream_result: crate::ai::types::StreamResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    final_assistant_text: &mut String,
    final_assistant_recorded: &mut bool,
) {
    let assistant_msg = Message {
        role: "assistant".to_string(),
        content: Value::String(stream_result.assistant_text.clone()),
        tool_calls: None,
        tool_call_id: None,
    };
    append_message_pair(messages, turn_messages, assistant_msg);
    *final_assistant_text = stream_result.assistant_text;
    *final_assistant_recorded = true;
    record_hidden_self_note(app, turn_messages, &stream_result.hidden_meta);
}

fn build_code_inspection_working_memory(turn_messages: &[Message]) -> Option<String> {
    let findings = collect_repo_inspection_findings(turn_messages);

    let mut raw_repo_tool_count = 0usize;
    let mut code_search_count = 0usize;
    for message in turn_messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            let tool_name = tool_call.function.name.as_str();
            if !is_repo_inspection_tool(tool_name) {
                continue;
            }
            if is_raw_repo_tool(tool_name) {
                raw_repo_tool_count += 1;
            }
            if tool_name == "code_search" {
                code_search_count += 1;
            }
        }
    }

    if findings.is_empty() && raw_repo_tool_count < 2 {
        return None;
    }

    let mut note = String::from(CODE_INSPECTION_MEMORY_PREFIX);
    note.push('\n');
    for finding in findings.iter().rev().take(6).rev() {
        note.push_str(&finding.rendered);
        note.push('\n');
    }
    note.push_str(
        "Treat these findings as already-known context for the current debugging turn. Avoid re-running the same raw file/search reads unless you need verification or a narrower slice.\n",
    );
    if raw_repo_tool_count >= 2 && code_search_count == 0 {
        note.push_str(
            "Code-navigation correction: you have used raw inspection tools without `code_search`. STOP using `read_file`, `read_file_lines`, `search_files`, or `grep_search` for exploration. Use `code_search` first to locate the relevant file/symbol/definition before reading specific lines.\n",
        );
    } else if raw_repo_tool_count >= 3 && code_search_count <= 1 {
        note.push_str(
            "Code-navigation correction: too many raw reads/searches. Use `code_search` for the next step instead of another `read_file_lines` or `grep_search`.\n",
        );
    }
    Some(truncate_note(&note, 1800))
}

fn build_persistent_code_discoveries(turn_messages: &[Message]) -> Vec<CodeDiscoveryRecord> {
    collect_repo_inspection_findings(turn_messages)
        .into_iter()
        .filter_map(|finding| classify_code_discovery(&finding))
        .rev()
        .take(persistence_limit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn collect_repo_inspection_findings(turn_messages: &[Message]) -> Vec<RepoInspectionFinding> {
    let tool_outputs = turn_messages
        .iter()
        .filter_map(|message| {
            message.tool_call_id.as_ref().and_then(|id| {
                let content = match &message.content {
                    Value::String(content) => content.clone(),
                    other => other.to_string(),
                };
                Some((id.clone(), content))
            })
        })
        .collect::<SkipMap<_, _>>();

    let mut findings = Vec::new();
    let mut seen = SkipSet::new(16);

    for message in turn_messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            let tool_name = tool_call.function.name.as_str();
            if !is_repo_inspection_tool(tool_name) {
                continue;
            }

            let tool_call_id = &tool_call.id;
            let Some(content) = tool_outputs.get(tool_call_id) else {
                continue;
            };
            let scope = describe_tool_call(tool_call);
            let highlight = summarize_tool_result(tool_name, &content);
            if highlight.is_empty() {
                continue;
            }
            let line = format!("- {}{} => {}", tool_name, scope, highlight);
            if seen.insert(line.clone()) {
                findings.push(RepoInspectionFinding {
                    tool_name: tool_name.to_string(),
                    rendered: line,
                    highlight,
                });
            }
        }
    }
    findings
}

fn is_repo_inspection_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "code_search"
            | "read_file"
            | "read_file_lines"
            | "search_files"
            | "grep_search"
            | "list_directory"
    )
}

fn is_raw_repo_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file" | "read_file_lines" | "search_files" | "grep_search" | "list_directory"
    )
}

fn persistent_code_discovery_already_present(messages: &[Message], body: &str) -> bool {
    messages.iter().any(|message| match &message.content {
        Value::String(content) => {
            content.starts_with(CODE_DISCOVERY_PREFIX)
                && content[CODE_DISCOVERY_PREFIX.len()..].trim_start() == body
        }
        _ => false,
    })
}

fn classify_code_discovery(finding: &RepoInspectionFinding) -> Option<CodeDiscoveryRecord> {
    let record = classify_finding(&finding.tool_name, &finding.highlight, &finding.rendered)?;
    should_persist(record.confidence).then_some(record)
}

fn describe_tool_call(tool_call: &ToolCall) -> String {
    let Ok(args) = serde_json::from_str::<Value>(&tool_call.function.arguments) else {
        return String::new();
    };
    match tool_call.function.name.as_str() {
        "code_search" => {
            let operation = args.get("operation").and_then(|v| v.as_str()).unwrap_or("search");
            let mut parts = vec![format!("operation={operation}")];
            for key in ["query", "symbol", "file_path", "path", "intent"] {
                if let Some(value) = args.get(key).and_then(|v| v.as_str()) && !value.is_empty() {
                    parts.push(format!("{key}={}", truncate_inline(value, 48)));
                }
            }
            format!("({})", parts.join(", "))
        }
        "read_file" | "read_file_lines" => {
            let path = args
                .get("file_path")
                .or_else(|| args.get("path"))
                .and_then(|v| v.as_str())
                .map(|v| truncate_inline(v, 64))
                .unwrap_or_else(|| "?".to_string());
            let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(1);
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(0);
            if limit > 0 {
                format!("(file={}, lines={}..{})", path, offset, offset + limit.saturating_sub(1))
            } else {
                format!("(file={path})")
            }
        }
        "grep_search" | "search_files" => {
            let query = args
                .get("query")
                .or_else(|| args.get("pattern"))
                .and_then(|v| v.as_str())
                .map(|v| truncate_inline(v, 48))
                .unwrap_or_else(|| "?".to_string());
            format!("(query={query})")
        }
        "list_directory" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|path| format!("(path={})", truncate_inline(path, 64)))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn summarize_tool_result(tool_name: &str, content: &str) -> String {
    let mut lines = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with("code_search route="))
        .filter(|line| !line.starts_with("- tail_preview:"))
        .collect::<Vec<_>>();
    if let Some(summary_line) = lines
        .iter()
        .find(|line| line.starts_with("- summary:") || line.starts_with("summary:"))
    {
        return truncate_inline(summary_line, 160);
    }

    if let Some(error_like) = lines.iter().find(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("error")
            || lower.contains("failed")
            || lower.contains("panic")
            || lower.contains("missing")
    }) {
        return truncate_inline(error_like, 160);
    }

    if tool_name == "code_search" {
        lines.retain(|line| {
            !line.starts_with("No exact symbol")
                && !line.starts_with("No exact matches")
                && !line.starts_with("No files matched")
        });
    }

    lines
        .into_iter()
        .next()
        .map(|line| truncate_inline(line, 160))
        .unwrap_or_default()
}

fn truncate_inline(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in value.chars().enumerate() {
        if idx >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn truncate_note(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    truncate_inline(value, max_chars)
}

#[cfg(test)]
mod tests {
    use super::{
        build_code_inspection_working_memory, build_persistent_code_discoveries,
        classify_code_discovery, is_repo_inspection_tool, persistent_code_discovery_already_present,
        RepoInspectionFinding,
    };
    use crate::ai::code_discovery_policy::{CodeDiscoveryConfidence, CodeDiscoveryKind};
    use crate::ai::history::Message;
    use crate::ai::types::{FunctionCall, ToolCall};
    use serde_json::Value;

    fn tool_call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[test]
    fn repo_inspection_tools_include_code_search() {
        assert!(is_repo_inspection_tool("code_search"));
    }

    #[test]
    fn working_memory_note_includes_findings_and_correction() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![
                    tool_call(
                        "1",
                        "read_file_lines",
                        serde_json::json!({"file_path":"src/lib.rs","offset":10,"limit":20}),
                    ),
                    tool_call(
                        "2",
                        "grep_search",
                        serde_json::json!({"pattern":"panic!","path":"src"}),
                    ),
                    tool_call(
                        "3",
                        "read_file",
                        serde_json::json!({"file_path":"src/main.rs","offset":1,"limit":40}),
                    ),
                ]),
                tool_call_id: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("    10\tfn load_config() {".to_string()),
                tool_calls: None,
                tool_call_id: Some("1".to_string()),
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("src/main.rs:42: panic!(\"boom\")".to_string()),
                tool_calls: None,
                tool_call_id: Some("2".to_string()),
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("     1\tmod main;".to_string()),
                tool_calls: None,
                tool_call_id: Some("3".to_string()),
            },
        ];

        let note = build_code_inspection_working_memory(&turn_messages).expect("note");
        assert!(note.contains("Current code-inspection working memory"));
        assert!(note.contains("read_file_lines(file=src/lib.rs, lines=10..29)"));
        assert!(note.contains("grep_search(query=panic!)"));
        assert!(note.contains("Code-navigation correction"));
        assert!(note.contains("Use `code_search`"));
    }

    #[test]
    fn working_memory_note_uses_code_search_without_correction_when_present() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "1",
                    "code_search",
                    serde_json::json!({"operation":"text_search","query":"load_config"}),
                )]),
                tool_call_id: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String(
                    "code_search route=grep_search operation=text_search\nsrc/lib.rs:10: fn load_config() {"
                        .to_string(),
                ),
                tool_calls: None,
                tool_call_id: Some("1".to_string()),
            },
        ];

        let note = build_code_inspection_working_memory(&turn_messages).expect("note");
        assert!(note.contains("code_search(operation=text_search, query=load_config)"));
        assert!(note.contains("fn load_config()"));
        assert!(!note.contains("Code-navigation correction"));
    }

    #[test]
    fn persistent_code_discoveries_keep_only_high_value_findings() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![
                    tool_call(
                        "1",
                        "read_file_lines",
                        serde_json::json!({"file_path":"src/lib.rs","offset":10,"limit":20}),
                    ),
                    tool_call(
                        "2",
                        "list_directory",
                        serde_json::json!({"path":"src"}),
                    ),
                ]),
                tool_call_id: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("    10\tfn load_config() {".to_string()),
                tool_calls: None,
                tool_call_id: Some("1".to_string()),
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("main.rs\nlib.rs".to_string()),
                tool_calls: None,
                tool_call_id: Some("2".to_string()),
            },
        ];

        let discoveries = build_persistent_code_discoveries(&turn_messages);
        assert_eq!(discoveries.len(), 1);
        assert!(discoveries[0].finding.contains("fn load_config()"));
    }

    #[test]
    fn duplicate_persistent_discovery_is_detected() {
        let messages = vec![Message {
            role: "system".to_string(),
            content: Value::String(
                "code_discovery:\n- read_file_lines(file=src/lib.rs, lines=10..29) => fn load_config() {"
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
        }];

        assert!(persistent_code_discovery_already_present(
            &messages,
            "- read_file_lines(file=src/lib.rs, lines=10..29) => fn load_config() {"
        ));
    }

    #[test]
    fn classify_code_discovery_marks_root_cause() {
        let finding = RepoInspectionFinding {
            tool_name: "read_file_lines".to_string(),
            rendered:
                "- read_file_lines(file=src/main.rs, lines=40..50) => root cause: config cache is empty due to missing APP_ENV"
                    .to_string(),
            highlight: "root cause: config cache is empty due to missing APP_ENV".to_string(),
        };

        let record = classify_code_discovery(&finding).expect("record");
        assert_eq!(record.kind, CodeDiscoveryKind::RootCause);
        assert_eq!(record.confidence, CodeDiscoveryConfidence::High);
    }

    #[test]
    fn classify_code_discovery_marks_entry_point() {
        let finding = RepoInspectionFinding {
            tool_name: "read_file_lines".to_string(),
            rendered:
                "- read_file_lines(file=src/main.rs, lines=1..20) => fn main() calls app::run() as the entry point"
                    .to_string(),
            highlight: "fn main() calls app::run() as the entry point".to_string(),
        };

        let record = classify_code_discovery(&finding).expect("record");
        assert_eq!(record.kind, CodeDiscoveryKind::EntryPoint);
        assert_eq!(record.confidence, CodeDiscoveryConfidence::High);
    }

    #[test]
    fn classify_code_discovery_marks_call_chain() {
        let finding = RepoInspectionFinding {
            tool_name: "code_search".to_string(),
            rendered:
                "- code_search(operation=structural, intent=find_calls, query=load_config) => call chain: main -> bootstrap -> load_config"
                    .to_string(),
            highlight: "call chain: main -> bootstrap -> load_config".to_string(),
        };

        let record = classify_code_discovery(&finding).expect("record");
        assert_eq!(record.kind, CodeDiscoveryKind::CallChain);
        assert_eq!(record.confidence, CodeDiscoveryConfidence::Medium);
    }
}
