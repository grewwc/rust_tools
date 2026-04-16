mod background;
mod gates;
mod recall;
mod writeback;

use std::path::PathBuf;

use crate::ai::{history::Message, types::App};

pub(crate) use recall::{AutoRecalledKnowledge, RecallBundle};

pub(super) async fn maybe_append_self_reflection(
    app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    turn_messages: &mut Vec<Message>,
) {
    background::maybe_append_self_reflection(app, model, question, answer, turn_messages).await
}

pub(super) async fn maybe_critic_and_revise(
    app: &mut App,
    model: &str,
    question: &str,
    draft: &str,
) -> Option<(String, String)> {
    background::maybe_critic_and_revise(app, model, question, draft).await
}

pub(super) async fn run_critic_revise_background(
    history_path: PathBuf,
    model: String,
    question: String,
    draft: String,
) {
    background::run_critic_revise_background(history_path, model, question, draft).await
}

pub(super) async fn run_self_reflection_background(
    history_path: PathBuf,
    session_id: String,
    model: String,
    question: String,
    answer: String,
    had_tool: bool,
) {
    background::run_self_reflection_background(
        history_path,
        session_id,
        model,
        question,
        answer,
        had_tool,
    )
    .await
}

pub(super) fn build_persistent_guidelines(question: &str, max_chars: usize) -> Option<String> {
    recall::build_persistent_guidelines(question, max_chars)
}

pub(super) fn build_auto_recalled_knowledge(
    question: &str,
    max_chars: usize,
) -> Option<AutoRecalledKnowledge> {
    recall::build_auto_recalled_knowledge(question, max_chars)
}

pub(super) fn build_auto_recalled_knowledge_with_project(
    question: &str,
    max_chars: usize,
    project_hint: Option<&str>,
) -> Option<AutoRecalledKnowledge> {
    recall::build_auto_recalled_knowledge_with_project(question, max_chars, project_hint)
}

pub(super) fn build_recall_bundle(
    question: &str,
    guideline_max_chars: usize,
    recall_max_chars: usize,
) -> RecallBundle {
    recall::build_recall_bundle(question, guideline_max_chars, recall_max_chars)
}

pub(super) async fn maybe_write_back_project_knowledge(
    app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    turn_messages: &Vec<Message>,
) {
    writeback::maybe_write_back_project_knowledge(app, model, question, answer, turn_messages)
        .await
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReflectionTrigger {
    ToolFailure,
    LowConfidenceAnswer,
    UserCorrection,
    RepeatedQuestion,
    LongTurn,
    Routine,
}

#[derive(Debug, Clone)]
pub struct ReflectionQuality {
    pub actionable: bool,
    pub specific: bool,
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

#[cfg(test)]
mod tests {
    use super::{
        build_auto_recalled_knowledge_with_project, build_persistent_guidelines,
        gates::turn_uses_repo_inspection_tools, writeback::ProjectWritebackUpsert,
        writeback::upsert_project_writeback_entry,
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
                owner_pid: None,
                owner_pgid: None,
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
                owner_pid: None,
                owner_pgid: None,
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
                owner_pid: None,
                owner_pgid: None,
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
                owner_pid: None,
                owner_pgid: None,
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
                owner_pid: None,
                owner_pgid: None,
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
                owner_pid: None,
                owner_pgid: None,
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
                owner_pid: None,
                owner_pgid: None,
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
                owner_pid: None,
                owner_pgid: None,
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
                owner_pid: None,
                owner_pgid: None,
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
