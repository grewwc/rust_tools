use serde_json::Value;

use crate::ai::history::append_history_messages;
use crate::ai::knowledge::config::KnowledgeConfig;
use crate::ai::knowledge::retrieval::recall;
use crate::ai::knowledge::storage::jsonl_store::JsonlStore;
use crate::ai::tools::service::memory::execute_memory_update;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::{
    history::Message,
    request::{self, build_content},
    types::App,
};
use crate::commonw::configw;
use chrono::Local;
use serde_json::json;
use std::path::PathBuf;
use uuid::Uuid;

fn current_project_hint() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let name = cwd.file_name()?.to_str()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// 反思触发条件 - 用于主动学习
#[derive(Debug, Clone, PartialEq)]
pub enum ReflectionTrigger {
    /// 工具调用失败
    ToolFailure,
    /// 模型回答置信度低
    LowConfidenceAnswer,
    /// 用户纠正
    UserCorrection,
    /// 重复问题（说明之前没解决）
    RepeatedQuestion,
    /// 超长对话轮次（>10 轮）
    LongTurn,
    /// 常规反思
    Routine,
}

/// 反思质量评估
#[derive(Debug, Clone)]
pub struct ReflectionQuality {
    /// 是否可执行
    pub actionable: bool,
    /// 是否具体
    pub specific: bool,
    /// 是否可推广
    pub generalizable: bool,
}

impl ReflectionQuality {
    pub fn score(&self) -> u8 {
        let mut score = 0;
        if self.actionable {
            score += 1;
        }
        if self.specific {
            score += 1;
        }
        if self.generalizable {
            score += 1;
        }
        score
    }

    pub fn is_high_quality(&self) -> bool {
        self.score() >= 2
    }
}

pub(super) async fn maybe_append_self_reflection(
    app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    turn_messages: &mut Vec<Message>,
) {
    let q = question.trim();
    let a = answer.trim();
    if q.is_empty() || a.is_empty() {
        return;
    }
    let had_tool = turn_has_tool(turn_messages);
    let history_path = app.session_history_file.clone();
    let session_id = app.session_id.clone();
    let model_s = model.to_string();
    let q_s = q.to_string();
    let a_s = a.to_string();
    tokio::spawn(async move {
        run_self_reflection_background(history_path, session_id, model_s, q_s, a_s, had_tool).await;
    });
}

pub(super) async fn maybe_write_back_project_knowledge(
    _app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    turn_messages: &Vec<Message>,
) {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.project_writeback.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return;
    }
    let q = question.trim();
    let a = answer.trim();
    if q.is_empty() || a.is_empty() {
        return;
    }
    if !turn_uses_repo_inspection_tools(turn_messages) {
        return;
    }
    if answer_looks_unstable_for_writeback(a) {
        return;
    }

    let Some(project_name) = current_project_hint() else {
        return;
    };
    let model_s = cfg
        .get_opt("ai.project_writeback.model")
        .unwrap_or_else(|| model.to_string());
    let q_s = q.to_string();
    let a_s = a.to_string();
    run_project_knowledge_writeback_background(project_name, model_s, q_s, a_s).await;
}

fn extract_content(v: &Value) -> Option<String> {
    let choices = v
        .get("choices")
        .or_else(|| v.get("output").and_then(|o| o.get("choices")))?;
    let msg = choices.get(0)?.get("message")?;
    let content = msg.get("content")?;
    match content {
        Value::String(s) => Some(s.to_string()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(s) = part.get("text").and_then(|v| v.as_str()) {
                    out.push_str(s);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

pub(super) fn build_persistent_guidelines(question: &str, max_chars: usize) -> Option<String> {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let config = KnowledgeConfig::from_config_file();
    build_persistent_guidelines_from_parts(&jsonl_store, question, max_chars, &config)
}

pub(super) struct AutoRecalledKnowledge {
    pub(super) content: String,
    pub(super) high_confidence_project_memory: bool,
    pub(super) entry_count: usize,
    pub(super) project_hint: Option<String>,
    pub(super) categories: Vec<String>,
}

pub(super) fn build_auto_recalled_knowledge(
    question: &str,
    max_chars: usize,
) -> Option<AutoRecalledKnowledge> {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let config = KnowledgeConfig::from_config_file();
    build_auto_recalled_knowledge_from_parts(&jsonl_store, question, max_chars, &config)
}

pub(super) fn build_auto_recalled_knowledge_with_project(
    question: &str,
    max_chars: usize,
    project_hint: Option<&str>,
) -> Option<AutoRecalledKnowledge> {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let config = KnowledgeConfig::from_config_file();
    build_auto_recalled_knowledge_with_project_from_parts(
        &jsonl_store,
        question,
        max_chars,
        project_hint,
        &config,
    )
}

pub(super) struct RecallBundle {
    pub(super) guidelines: Option<String>,
    pub(super) recalled: Option<AutoRecalledKnowledge>,
}

pub(super) fn build_recall_bundle(
    question: &str,
    guideline_max_chars: usize,
    recall_max_chars: usize,
) -> RecallBundle {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let config = KnowledgeConfig::from_config_file();
    RecallBundle {
        guidelines: build_persistent_guidelines_from_parts(
            &jsonl_store,
            question,
            guideline_max_chars,
            &config,
        ),
        recalled: build_auto_recalled_knowledge_from_parts(
            &jsonl_store,
            question,
            recall_max_chars,
            &config,
        ),
    }
}

fn build_persistent_guidelines_from_parts(
    jsonl_store: &JsonlStore,
    question: &str,
    max_chars: usize,
    config: &KnowledgeConfig,
) -> Option<String> {
    recall::build_persistent_guidelines(jsonl_store, question, max_chars, config)
}

fn build_auto_recalled_knowledge_from_parts(
    jsonl_store: &JsonlStore,
    question: &str,
    max_chars: usize,
    config: &KnowledgeConfig,
) -> Option<AutoRecalledKnowledge> {
    recall::build_auto_recalled_knowledge(jsonl_store, question, max_chars, config)
        .map(map_auto_recalled_knowledge)
}

fn build_auto_recalled_knowledge_with_project_from_parts(
    jsonl_store: &JsonlStore,
    question: &str,
    max_chars: usize,
    project_hint: Option<&str>,
    config: &KnowledgeConfig,
) -> Option<AutoRecalledKnowledge> {
    recall::build_auto_recalled_knowledge_with_project(
        jsonl_store,
        question,
        max_chars,
        project_hint,
        config,
    )
    .map(map_auto_recalled_knowledge)
}

fn map_auto_recalled_knowledge(
    r: crate::ai::knowledge::retrieval::recall::AutoRecalledKnowledge,
) -> AutoRecalledKnowledge {
    AutoRecalledKnowledge {
        content: r.content,
        high_confidence_project_memory: r.high_confidence_project_memory,
        entry_count: r.entry_count,
        project_hint: r.project_hint,
        categories: r.categories,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ProjectWritebackUpsert, build_auto_recalled_knowledge_with_project,
        build_persistent_guidelines, turn_uses_repo_inspection_tools,
        upsert_project_writeback_entry,
    };
    use crate::ai::history::Message;
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
    use crate::ai::types::{FunctionCall, ToolCall};
    use chrono::Local;
    use serde_json::Value;

    #[test]
    fn persistent_guidelines_include_safety_rules_and_high_priority_entries() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_guidelines_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let store = MemoryStore::from_env_or_config();
        let timestamp = Local::now().to_rfc3339();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "self_note".to_string(),
                note: "Do: validate tool arguments".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(100),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "safety_rules".to_string(),
                note: "Avoid: delete files without double checking".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(255),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "common_sense".to_string(),
                note: "Keep broadly applicable engineering habits in memory.".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(150),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "coding_guideline".to_string(),
                note: "Prefer cargo check before cargo test for quick feedback.".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(150),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "user_preference".to_string(),
                note: "Prefer concise, information-dense answers.".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(150),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "safety_rules".to_string(),
                note: "Do: always ask before risky file operations".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(200),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp,
                category: "user_memory".to_string(),
                note: "Ignore me - this is knowledge not guideline".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(150),
            })
            .unwrap();

        let guidelines =
            build_persistent_guidelines("delete files safely", 1200).expect("guidelines");

        assert!(guidelines.contains("Do: validate tool arguments"));
        assert!(guidelines.contains("Avoid: delete files without double checking"));
        assert!(guidelines.contains("Keep broadly applicable engineering habits in memory."));
        assert!(guidelines.contains("Prefer cargo check before cargo test for quick feedback."));
        assert!(guidelines.contains("Prefer concise, information-dense answers."));
        assert!(guidelines.contains("Do: always ask before risky file operations"));
        assert!(!guidelines.contains("Ignore me"));

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn auto_recalled_knowledge_uses_project_hint_for_this_project_queries() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_auto_recall_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let store = MemoryStore::from_env_or_config();
        let timestamp = Local::now().to_rfc3339();
        store
            .append(&AgentMemoryEntry {
                id: Some("mem_project".to_string()),
                timestamp: timestamp.clone(),
                category: "user_memory".to_string(),
                note: "rust_tools 项目结构：src/bin 放各个入口，src/cw 放通用组件。".to_string(),
                tags: vec!["project".to_string(), "rust_tools".to_string()],
                source: Some("rust_tools".to_string()),
                priority: Some(150),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: Some("mem_noise".to_string()),
                timestamp,
                category: "tool_cache".to_string(),
                note: "{\"tool\":\"read_file\"}".to_string(),
                tags: vec!["read_file".to_string()],
                source: Some("session:test".to_string()),
                priority: Some(80),
            })
            .unwrap();

        let recalled = build_auto_recalled_knowledge_with_project(
            "这个项目的结构是什么？",
            1200,
            Some("rust_tools"),
        )
        .expect("recalled knowledge");

        assert!(recalled.content.contains("Auto-Recalled Knowledge:"));
        assert!(recalled.content.contains("rust_tools 项目结构"));
        assert!(!recalled.content.contains("tool_cache"));
        assert!(!recalled.content.contains("read_file"));
        assert!(recalled.high_confidence_project_memory);

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn repo_inspection_tools_are_detected_from_turn_messages() {
        let messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read_file_lines".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("...".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
            },
        ];
        assert!(turn_uses_repo_inspection_tools(&messages));
    }

    #[test]
    fn project_writeback_replaces_existing_entry_by_source() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_project_writeback_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let store = MemoryStore::from_env_or_config();
        let source = "auto_project_writeback:rust_tools";
        let created = upsert_project_writeback_entry(
            &store,
            source,
            "- rust_tools 项目结构：初始版本",
            vec!["project".to_string(), "rust_tools".to_string()],
            180,
        )
        .unwrap();
        assert!(matches!(created, ProjectWritebackUpsert::Saved));

        let updated = upsert_project_writeback_entry(
            &store,
            source,
            "- rust_tools 项目结构：更新版本",
            vec![
                "project".to_string(),
                "rust_tools".to_string(),
                "structure".to_string(),
            ],
            200,
        )
        .unwrap();
        assert!(matches!(updated, ProjectWritebackUpsert::Updated));

        let entries = store
            .recent(20)
            .unwrap()
            .into_iter()
            .filter(|entry| entry.source.as_deref() == Some(source))
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].note.contains("更新版本"));
        assert_eq!(entries[0].priority, Some(200));
        assert!(entries[0].tags.iter().any(|tag| tag == "structure"));

        let unchanged = upsert_project_writeback_entry(
            &store,
            source,
            "- rust_tools 项目结构：更新版本",
            vec![
                "project".to_string(),
                "rust_tools".to_string(),
                "structure".to_string(),
            ],
            200,
        )
        .unwrap();
        assert!(matches!(unchanged, ProjectWritebackUpsert::Unchanged));
        let entries_after = store
            .recent(20)
            .unwrap()
            .into_iter()
            .filter(|entry| entry.source.as_deref() == Some(source))
            .collect::<Vec<_>>();
        assert_eq!(entries_after.len(), 1);

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    #[ignore]
    fn diagnose_real_memory_recall() {
        let real_path = std::path::PathBuf::from(
            dirs::home_dir()
                .unwrap_or_default()
                .join(".config/rust_tools/agent_memory.jsonl"),
        );
        if !real_path.exists() {
            eprintln!("[DIAG] Memory file not found: {}", real_path.display());
            return;
        }
        unsafe {
            std::env::set_var(
                "RUST_TOOLS_MEMORY_FILE",
                real_path.to_string_lossy().as_ref(),
            );
        }

        let question = "你帮我简单总结一下rust_tools这个项目结构";
        let project_hint = "rust_tools";
        let max_chars = 2000;

        eprintln!(
            "=== DIAG: build_auto_recalled_knowledge (max_chars={}) ===",
            max_chars
        );
        eprintln!("[DIAG] question   = {:?}", question);
        eprintln!("[DIAG] project_hint = {:?}", project_hint);

        let store = MemoryStore::from_env_or_config();
        eprintln!("[DIAG] memory_file = {:?}", store.path());

        let mut query_variants = vec![question.trim().to_string()];
        query_variants.push(format!("{question} {project_hint}"));
        query_variants.push(project_hint.to_string());
        for (i, q) in query_variants.iter().enumerate() {
            eprintln!("[DIAG] query_variant[{}] = {:?} (len={})", i, q, q.len());
            let results = store.search(q, 24).unwrap_or_default();
            eprintln!("[DIAG]   search returned {} results", results.len());
            for (j, (entry, _score)) in results.iter().enumerate() {
                if entry.category == "tool_cache" {
                    continue;
                }
                let note_str = entry.note.as_str();
                let preview: String = if note_str.len() > 80 {
                    let end = note_str
                        .char_indices()
                        .find(|(i, _)| *i >= 80)
                        .map(|(i, _)| i)
                        .unwrap_or(note_str.len());
                    format!("{}...", &note_str[..end])
                } else {
                    entry.note.clone()
                };
                eprintln!(
                    "[DIAG]   [{}] cat={} pri={} src={:?} tags={:?} note={:?}",
                    j,
                    entry.category,
                    entry.priority.unwrap_or(0),
                    entry.source,
                    entry.tags,
                    preview
                );
            }
        }

        let recent_entries = store.recent(80).unwrap_or_default();
        let project_entries: Vec<_> = recent_entries
            .iter()
            .filter(|e| e.category != "tool_cache")
            .filter(|e| {
                let h = project_hint.to_lowercase();
                e.category.to_lowercase().contains(&h)
                    || e.note.to_lowercase().contains(&h)
                    || e.source
                        .as_deref()
                        .is_some_and(|s| s.to_lowercase().contains(&h))
                    || e.tags.iter().any(|t| t.to_lowercase().contains(&h))
            })
            .collect();
        eprintln!(
            "[DIAG] recent(80) non-tool_cache entries mentioning '{}': {}",
            project_hint,
            project_entries.len()
        );
        for (i, entry) in project_entries.iter().enumerate() {
            let note_str = entry.note.as_str();
            let preview: String = if note_str.len() > 100 {
                let end = note_str
                    .char_indices()
                    .find(|(i, _)| *i >= 100)
                    .map(|(i, _)| i)
                    .unwrap_or(note_str.len());
                format!("{}...", &note_str[..end])
            } else {
                entry.note.clone()
            };
            eprintln!(
                "[DIAG]   [{}] cat={} pri={} src={:?} tags={:?} note={:?}",
                i,
                entry.category,
                entry.priority.unwrap_or(0),
                entry.source,
                entry.tags,
                preview
            );
            eprintln!("[DIAG]       note_total_len={}", entry.note.len());
        }

        let result =
            build_auto_recalled_knowledge_with_project(question, max_chars, Some(project_hint));
        match result {
            None => {
                eprintln!("[DIAG] >>> RESULT: None !!!");
            }
            Some(r) => {
                eprintln!("[DIAG] >>> RESULT: Some {{");
                eprintln!("[DIAG] >>>   entry_count  = {}", r.entry_count);
                eprintln!("[DIAG] >>>   categories   = {:?}", r.categories);
                eprintln!(
                    "[DIAG] >>>   high_confidence_project_memory = {}",
                    r.high_confidence_project_memory
                );
                eprintln!("[DIAG] >>>   content_len  = {} chars", r.content.len());
                eprintln!("[DIAG] >>>   content (first 30 lines):");
                for (i, line) in r.content.lines().take(30).enumerate() {
                    eprintln!("[DIAG] >>>     {}: {}", i, line);
                }
                if r.content.lines().count() > 30 {
                    eprintln!(
                        "[DIAG] >>>     ... ({} total lines)",
                        r.content.lines().count()
                    );
                }
                eprintln!("[DIAG] >>> }}");
            }
        }

        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }
}

pub(super) async fn maybe_critic_and_revise(
    app: &mut App,
    model: &str,
    question: &str,
    draft: &str,
) -> Option<(String, String)> {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.critic_revise.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled || question.trim().is_empty() || draft.trim().is_empty() {
        return None;
    }
    if critic_filtered(question, draft) {
        return None;
    }
    let to_ms = cfg
        .get_opt("ai.critic_revise.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(7000);
    let only_for_code = !cfg
        .get_opt("ai.critic_revise.only_for_code")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if only_for_code {
        let gate_fut = model_should_revise(app, model, question, draft);
        let should = match timeout(Duration::from_millis(to_ms / 2), gate_fut).await {
            Ok(v) => v.unwrap_or(false),
            Err(_) => false,
        };
        if !should {
            return None;
        }
    }
    // 禁用工具
    let saved_tools = app
        .agent_context
        .as_mut()
        .map(|ctx| std::mem::replace(&mut ctx.tools, Vec::new()));
    let critic_system = "You are a strict code assistant critic. Review the DRAFT answer for the user QUESTION.\nReturn a compact list of 3-8 actionable points focused on:\n- factual correctness and missing steps\n- tool usage and argument hygiene\n- clarity and structure of final message\nNo markdown fences. Use short bullets.";
    let critic_user = format!("QUESTION:\n{}\n\nDRAFT:\n{}", question.trim(), draft.trim());
    let critic_req = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(critic_system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &critic_user, &[])
                .unwrap_or(Value::String(critic_user.clone())),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let critic_fut = request::do_request_messages(app, model, &critic_req, false);
    let critic_resp = match timeout(Duration::from_millis(to_ms), critic_fut).await {
        Ok(Ok(r)) => r,
        _ => {
            // 恢复工具
            if let Some(mut tools) = saved_tools {
                if let Some(ctx) = app.agent_context.as_mut() {
                    std::mem::swap(&mut ctx.tools, &mut tools);
                }
            }
            return None;
        }
    };
    let critic_text = critic_resp.text().await.ok()?;
    let critic_v: Value = serde_json::from_str(&critic_text).ok()?;
    let critic = extract_content(&critic_v).unwrap_or_default();
    if critic.trim().is_empty() {
        // 恢复工具
        if let Some(mut tools) = saved_tools {
            if let Some(ctx) = app.agent_context.as_mut() {
                std::mem::swap(&mut ctx.tools, &mut tools);
            }
        }
        return None;
    }
    let revise_system = "You are a senior coding assistant. Rewrite the final answer for the QUESTION using the CRITIC points.\nRules:\n- Fix issues; add missing steps; keep answers concise and correct.\n- If code is needed, use proper markdown fences.\n- Do not mention the critic itself.";
    let revise_user = format!(
        "QUESTION:\n{}\n\nCRITIC:\n{}\n\nDRAFT:\n{}",
        question.trim(),
        critic.trim(),
        draft.trim()
    );
    let revise_req = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(revise_system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &revise_user, &[])
                .unwrap_or(Value::String(revise_user.clone())),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let revise_fut = request::do_request_messages(app, model, &revise_req, false);
    let revised_resp = match timeout(Duration::from_millis(to_ms), revise_fut).await {
        Ok(Ok(r)) => r,
        _ => {
            // 恢复工具
            if let Some(mut tools) = saved_tools {
                if let Some(ctx) = app.agent_context.as_mut() {
                    std::mem::swap(&mut ctx.tools, &mut tools);
                }
            }
            return None;
        }
    };
    // 恢复工具
    if let Some(mut tools) = saved_tools {
        if let Some(ctx) = app.agent_context.as_mut() {
            std::mem::swap(&mut ctx.tools, &mut tools);
        }
    }
    let revised_text = revised_resp.text().await.ok()?;
    let revised_v: Value = serde_json::from_str(&revised_text).ok()?;
    let revised = extract_content(&revised_v).unwrap_or_default();
    if revised.trim().is_empty() {
        None
    } else {
        Some((critic, revised))
    }
}

fn reflection_filtered(question: &str, answer: &str, turn_messages: &Vec<Message>) -> bool {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.reflection.filter.enable")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return false;
    }
    let min_q = cfg
        .get_opt("ai.reflection.filter.min_question_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let min_a = cfg
        .get_opt("ai.reflection.filter.min_answer_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80);
    let require_tool = cfg
        .get_opt("ai.reflection.filter.require_tool_or_long")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("true");
    let q = question.trim();
    let a = answer.trim();
    if q.chars().count() < min_q && a.chars().count() < min_a {
        return true;
    }
    if require_tool && !turn_has_tool(turn_messages) && a.chars().count() < min_a {
        return true;
    }
    false
}

fn critic_filtered(question: &str, draft: &str) -> bool {
    let cfg = configw::get_all_config();
    let min_q = cfg
        .get_opt("ai.critic_revise.filter.min_question_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let min_a = cfg
        .get_opt("ai.critic_revise.filter.min_answer_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(120);
    let q = question.trim();
    let a = draft.trim();
    q.chars().count() < min_q && a.chars().count() < min_a
}

async fn model_should_reflect(
    app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    had_tool: bool,
) -> Option<bool> {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let to_ms = cfg
        .get_opt("ai.reflection.model_gate.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2000);
    let system = "You are a binary classifier that decides whether to capture a short 'experience note' for future turns.\nReturn STRICT JSON ONLY with the shape: {\"reflect\": true|false}.\nRules:\n- reflect=true when Q/A contains non-trivial reasoning, code, multi-step instructions, tool usage outcomes, errors/diagnosis, or decisions that should guide future behavior.\n- reflect=false for greetings, acknowledgements, trivial answers, or very short single-turn exchanges with no actionable guidance.\nDo not include explanations or extra text.";
    let user = format!(
        "question:\n{}\n\nanswer:\n{}\n\nhad_tool:\n{}",
        question.trim(),
        answer.trim(),
        if had_tool { "true" } else { "false" }
    );
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &user, &[]).unwrap_or(Value::String(user)),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let fut = request::do_request_messages(app, model, &messages, false);
    let resp = match timeout(Duration::from_millis(to_ms), fut).await {
        Ok(Ok(r)) => r,
        _ => return None,
    };
    let text = resp.text().await.ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_content(&v).unwrap_or_default();
    parse_reflect_flag(&content)
}

fn parse_reflect_flag(s: &str) -> Option<bool> {
    let trimmed = s.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return v.get("reflect").and_then(|b| b.as_bool());
    }
    let l = trimmed.find('{')?;
    let r = trimmed.rfind('}')?;
    if r < l {
        return None;
    }
    let sub = &trimmed[l..=r];
    serde_json::from_str::<Value>(sub)
        .ok()
        .and_then(|v| v.get("reflect").and_then(|b| b.as_bool()))
}

fn turn_uses_repo_inspection_tools(messages: &Vec<Message>) -> bool {
    const REPO_INSPECTION_TOOLS: &[&str] = &[
        "read_file",
        "read_file_lines",
        "list_directory",
        "search_files",
        "grep_search",
        "execute_command",
    ];
    messages.iter().any(|message| {
        message
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls.iter().any(|call| {
                    REPO_INSPECTION_TOOLS
                        .iter()
                        .any(|name| call.function.name == *name)
                })
            })
            .unwrap_or(false)
    })
}

fn answer_looks_unstable_for_writeback(answer: &str) -> bool {
    let lower = answer.to_lowercase();
    if answer.chars().count() < 40 {
        return true;
    }
    [
        "[本轮请求失败",
        "i'm sorry",
        "不确定",
        "可能",
        "猜测",
        "大概",
        "无法确认",
        "need to verify",
        "might be",
    ]
    .iter()
    .any(|needle| lower.contains(&needle.to_lowercase()))
}

fn turn_has_tool(messages: &Vec<Message>) -> bool {
    for m in messages {
        if m.role == "tool" {
            return true;
        }
        if let Some(calls) = m.tool_calls.as_ref() {
            if !calls.is_empty() {
                return true;
            }
        }
    }
    false
}

async fn model_should_revise(
    app: &mut App,
    model: &str,
    question: &str,
    draft: &str,
) -> Option<bool> {
    let system = "You decide if the DRAFT answer should be CRITIC→REVISE refined.\nReturn STRICT JSON ONLY: {\"revise\": true|false}.\nRules:\n- true ONLY for software engineering tasks: code writing/review/debug/refactor, tool execution results, build/test errors, patch proposals.\n- false for general knowledge, Q&A like weather/news/sports/finance, travel, generic suggestions, or casual chat.\n- false when the answer is short and already sufficient without code/steps.\nNo extra text.";
    let user = format!("QUESTION:\n{}\n\nDRAFT:\n{}", question.trim(), draft.trim());
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &user, &[]).unwrap_or(Value::String(user)),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let resp = request::do_request_messages(app, model, &messages, false)
        .await
        .ok()?;
    let text = resp.text().await.ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_content(&v).unwrap_or_default();
    if let Ok(v2) = serde_json::from_str::<Value>(content.trim()) {
        return v2.get("revise").and_then(|b| b.as_bool());
    }
    None
}

pub(super) async fn run_critic_revise_background(
    history_path: PathBuf,
    model: String,
    question: String,
    draft: String,
) {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let to_ms = cfg
        .get_opt("ai.critic_revise.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(7000);
    // Perform critic
    let system_c = "You are a strict code assistant critic. Review the DRAFT answer for the user QUESTION.\nReturn a compact list of 3-8 actionable points focused on:\n- factual correctness and missing steps\n- tool usage and argument hygiene\n- clarity and structure of final message\nNo markdown fences. Use short bullets.";
    let critic_user = format!("QUESTION:\n{}\n\nDRAFT:\n{}", question.trim(), draft.trim());
    let messages_c = vec![
        json!({"role":"system","content":system_c}),
        json!({"role":"user","content":critic_user}),
    ];
    let resp_c = match background_call(&model, &messages_c).await {
        Some(v) => v,
        None => return,
    };
    let content_c = extract_back_content(&resp_c).unwrap_or_default();
    if content_c.trim().is_empty() {
        return;
    }
    // Perform revise
    let system_r = "You are a senior coding assistant. Rewrite the final answer for the QUESTION using the CRITIC points.\nRules:\n- Fix issues; add missing steps; keep answers concise and correct.\n- If code is needed, use proper markdown fences.\n- Do not mention the critic itself.";
    let revise_user = format!(
        "QUESTION:\n{}\n\nCRITIC:\n{}\n\nDRAFT:\n{}",
        question.trim(),
        content_c.trim(),
        draft.trim()
    );
    let messages_r = vec![
        json!({"role":"system","content":system_r}),
        json!({"role":"user","content":revise_user}),
    ];
    let resp_r = match timeout(
        Duration::from_millis(to_ms),
        background_call(&model, &messages_r),
    )
    .await
    {
        Ok(v) => v.and_then(|vv| Some(vv)),
        Err(_) => None,
    };
    let Some(resp_r) = resp_r else {
        return;
    };
    let content_r = extract_back_content(&resp_r).unwrap_or_default();
    if content_r.trim().is_empty() {
        return;
    }
    // Append as a single system record (background meta)
    let record = Message {
        role: "system".to_string(),
        content: Value::String(format!(
            "critic:\n{}\n\nrevised:\n{}",
            content_c.trim(),
            content_r.trim()
        )),
        tool_calls: None,
        tool_call_id: None,
    };
    let _ = append_history_messages(&history_path, &[record]);
}

async fn background_call(model: &str, messages: &Vec<Value>) -> Option<Value> {
    let cfg = configw::get_all_config();
    let endpoint = cfg.get_opt("ai.model.endpoint")?;
    let api_key = cfg.get_opt("api_key")?;
    let body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "enable_thinking": false
    });
    let client = reqwest::Client::new();
    let resp = client
        .post(&endpoint)
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .ok()?;
    let text = resp.text().await.ok()?;
    serde_json::from_str::<Value>(&text).ok()
}

fn extract_back_content(v: &Value) -> Option<String> {
    let choices = v
        .get("choices")
        .or_else(|| v.get("output")?.get("choices"))?;
    let msg = choices.get(0)?.get("message")?;
    let content = msg.get("content")?;
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if let Some(s) = p.get("text").and_then(|x| x.as_str()) {
                    out.push_str(s);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

pub(super) async fn run_self_reflection_background(
    history_path: PathBuf,
    session_id: String,
    model: String,
    question: String,
    answer: String,
    had_tool: bool,
) {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.reflection.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return;
    }
    let q = question.trim();
    let a = answer.trim();
    if q.is_empty() || a.is_empty() {
        return;
    }
    let model_gate_enabled = !cfg
        .get_opt("ai.reflection.model_gate.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if model_gate_enabled {
        let to_ms = cfg
            .get_opt("ai.reflection.model_gate.timeout_ms")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(2000);
        let fut = background_model_should_reflect(&model, q, a, had_tool);
        let should = match timeout(Duration::from_millis(to_ms), fut).await {
            Ok(v) => v.unwrap_or(false),
            Err(_) => false,
        };
        if !should {
            return;
        }
    } else if reflection_filtered_bg(q, a, had_tool) {
        return;
    }
    let system = "You are an introspective meta-optimizer for a coding assistant. Produce a brief self note to improve future turns.\nRules:\n- Output 2-6 compact bullets grouped under 'Do:' and 'Avoid:' tuned to the given Q&A.\n- Focus on planning, tool usage, argument hygiene, and verification habits.\n- No apologies, no explanations, no markdown code fences.\n- Keep under 800 chars.";
    let user_payload = format!("question:\n{}\n\nanswer:\n{}", q, a);
    let messages = vec![
        json!({"role":"system","content":system}),
        json!({"role":"user","content":user_payload}),
    ];
    let to_ms_note = cfg
        .get_opt("ai.reflection.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(3000);
    let resp = match timeout(
        Duration::from_millis(to_ms_note),
        background_call(&model, &messages),
    )
    .await
    {
        Ok(v) => v,
        Err(_) => None,
    };
    let Some(resp) = resp else {
        return;
    };
    let content = extract_back_content(&resp).unwrap_or_default();
    let note = content.trim();
    if note.is_empty() {
        return;
    }
    let record = Message {
        role: "system".to_string(),
        content: Value::String(format!("self_note:\n{}", note)),
        tool_calls: None,
        tool_call_id: None,
    };
    let _ = append_history_messages(&history_path, &[record]);
    let entry = AgentMemoryEntry {
        id: None,
        timestamp: Local::now().to_rfc3339(),
        category: "self_note".to_string(),
        note: note.to_string(),
        tags: vec!["agent".to_string(), "policy".to_string()],
        source: Some(format!("session:{}", session_id)),
        priority: Some(255), // Permanent: agent policies are never deleted
    };
    let store = MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

async fn background_model_should_reflect(
    model: &str,
    question: &str,
    answer: &str,
    had_tool: bool,
) -> Option<bool> {
    let system = "You are a binary classifier that decides whether to capture a short 'experience note' for future turns.\nReturn STRICT JSON ONLY with the shape: {\"reflect\": true|false}.\nRules:\n- reflect=true when Q/A contains non-trivial reasoning, code, multi-step instructions, tool usage outcomes, errors/diagnosis, or decisions that should guide future behavior.\n- reflect=false for greetings, acknowledgements, trivial answers, or very short single-turn exchanges with no actionable guidance.\nDo not include explanations or extra text.";
    let user = format!(
        "question:\n{}\n\nanswer:\n{}\n\nhad_tool:\n{}",
        question.trim(),
        answer.trim(),
        if had_tool { "true" } else { "false" }
    );
    let messages = vec![
        json!({"role":"system","content":system}),
        json!({"role":"user","content":user}),
    ];
    let resp = background_call(model, &messages).await?;
    let text = extract_back_content(&resp).unwrap_or_default();
    parse_reflect_flag(&text)
}

#[derive(Debug)]
struct ProjectWritebackPayload {
    content: String,
    tags: Vec<String>,
    priority: u8,
}

fn parse_project_writeback_payload(s: &str) -> Option<ProjectWritebackPayload> {
    let trimmed = s.trim();
    let candidate = if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        v
    } else {
        let l = trimmed.find('{')?;
        let r = trimmed.rfind('}')?;
        if r < l {
            return None;
        }
        serde_json::from_str::<Value>(&trimmed[l..=r]).ok()?
    };

    if !candidate
        .get("writeback")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return None;
    }
    let content = candidate
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if content.is_empty() {
        return None;
    }
    let tags = candidate
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .take(8)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let priority = candidate
        .get("priority")
        .and_then(|v| v.as_u64())
        .map(|v| if v > u8::MAX as u64 { u8::MAX } else { v as u8 })
        .unwrap_or(180)
        .max(150);
    Some(ProjectWritebackPayload {
        content,
        tags,
        priority,
    })
}

fn find_existing_project_writeback_entry(
    store: &MemoryStore,
    source: &str,
) -> Option<AgentMemoryEntry> {
    store
        .recent(500)
        .ok()
        .unwrap_or_default()
        .into_iter()
        .find(|entry| entry.category == "project_memory" && entry.source.as_deref() == Some(source))
}

enum ProjectWritebackUpsert {
    Saved,
    Updated,
    Unchanged,
}

fn upsert_project_writeback_entry(
    store: &MemoryStore,
    source: &str,
    content: &str,
    tags: Vec<String>,
    priority: u8,
) -> Result<ProjectWritebackUpsert, String> {
    if let Some(existing) = find_existing_project_writeback_entry(store, source) {
        if existing.note.trim() == content.trim() {
            return Ok(ProjectWritebackUpsert::Unchanged);
        }
        let Some(id) = existing.id.as_deref() else {
            return Ok(ProjectWritebackUpsert::Unchanged);
        };
        execute_memory_update(&json!({
            "id": id,
            "content": content,
            "category": "project_memory",
            "tags": tags,
            "source": source,
            "priority": priority,
        }))?;
        return Ok(ProjectWritebackUpsert::Updated);
    }

    let entry = AgentMemoryEntry {
        id: Some(format!("mem_{}", Uuid::new_v4().simple())),
        timestamp: Local::now().to_rfc3339(),
        category: "project_memory".to_string(),
        note: content.to_string(),
        tags,
        source: Some(source.to_string()),
        priority: Some(priority),
    };
    store.append(&entry)?;
    Ok(ProjectWritebackUpsert::Saved)
}

pub(super) async fn run_project_knowledge_writeback_background(
    project_name: String,
    model: String,
    question: String,
    answer: String,
) {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let timeout_ms = cfg
        .get_opt("ai.project_writeback.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(3000);
    let system = "You extract durable project knowledge for future turns.\nReturn STRICT JSON ONLY.\nSchema:\n{\"writeback\":true|false,\"content\":\"...\",\"tags\":[\"...\"],\"priority\":180}\nRules:\n- writeback=true ONLY if the answer contains stable, project-specific facts useful for future Q&A.\n- Focus on repository structure, module responsibilities, build/test workflow, conventions, and architecture.\n- `content` must be 2-6 concise bullet lines of factual memory, no markdown fences.\n- Use only facts explicitly stated in the ANSWER. Do not speculate or add anything new.\n- Exclude transient status, temporary debugging details, one-off user requests, and uncertain statements.\n- If the answer is incomplete, vague, or not worth remembering, return {\"writeback\":false}.";
    let user = format!(
        "PROJECT:\n{}\n\nQUESTION:\n{}\n\nANSWER:\n{}",
        project_name.trim(),
        question.trim(),
        answer.trim()
    );
    let messages = vec![
        json!({"role":"system","content":system}),
        json!({"role":"user","content":user}),
    ];
    let resp = match timeout(
        Duration::from_millis(timeout_ms),
        background_call(&model, &messages),
    )
    .await
    {
        Ok(v) => v,
        Err(_) => None,
    };
    let Some(resp) = resp else {
        return;
    };
    let content = extract_back_content(&resp).unwrap_or_default();
    let Some(payload) = parse_project_writeback_payload(&content) else {
        return;
    };

    let source = format!("auto_project_writeback:{project_name}");
    let store = MemoryStore::from_env_or_config();
    let mut tags = vec![
        "project".to_string(),
        "auto_writeback".to_string(),
        project_name.clone(),
    ];
    for tag in payload.tags {
        if !tags.iter().any(|existing| existing == &tag) {
            tags.push(tag);
        }
    }
    match upsert_project_writeback_entry(&store, &source, &payload.content, tags, payload.priority)
    {
        Ok(ProjectWritebackUpsert::Saved) => {
            println!(
                "[Memory] writeback saved project={} source={} category=project_memory priority={}",
                project_name, source, payload.priority
            );
            store.maintain_after_append();
        }
        Ok(ProjectWritebackUpsert::Updated) => {
            println!(
                "[Memory] writeback updated project={} source={} category=project_memory priority={}",
                project_name, source, payload.priority
            );
            store.maintain_after_append();
        }
        Ok(ProjectWritebackUpsert::Unchanged) => {}
        Err(_) => {}
    }
}

fn reflection_filtered_bg(question: &str, answer: &str, had_tool: bool) -> bool {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.reflection.filter.enable")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return false;
    }
    let min_q = cfg
        .get_opt("ai.reflection.filter.min_question_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let min_a = cfg
        .get_opt("ai.reflection.filter.min_answer_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80);
    let require_tool = cfg
        .get_opt("ai.reflection.filter.require_tool_or_long")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("true");
    let q = question.trim();
    let a = answer.trim();
    if q.chars().count() < min_q && a.chars().count() < min_a {
        return true;
    }
    if require_tool && !had_tool && a.chars().count() < min_a {
        return true;
    }
    false
}
