mod background;
mod gates;
mod writeback;

use serde::{Deserialize, Serialize};

pub(crate) use background::assess_learning_note_quality;

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
        // 长期沉淀必须满足"可执行 + 可泛化"两条底线。
        // 仅仅具体但不可迁移的运行时实例（例如原始 tool error / 路径 / 一次性报错）
        // 不应被晋升为长期知识或稳定 guideline。
        self.actionable && self.generalizable
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningNoteAssessment {
    pub actionable: bool,
    pub specific: bool,
    pub generalizable: bool,
    pub score: u8,
    pub high_quality: bool,
    pub char_count: usize,
    pub word_count: usize,
    pub nonempty_lines: usize,
    pub unique_token_ratio: f32,
    pub directive_signals: usize,
    pub code_signals: usize,
    pub artifact_signals: usize,
    pub abstraction_signals: usize,
    pub condition_signals: usize,
    pub one_off_signals: usize,
}

impl LearningNoteAssessment {
    pub fn confidence(&self) -> f64 {
        (self.score as f64 / 3.0).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        assess_learning_note_quality, gates::turn_uses_repo_inspection_tools,
        writeback::ProjectWritebackUpsert, writeback::upsert_project_writeback_entry,
    };
    use crate::ai::history::Message;
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::tools::storage::memory_store::MemoryStore;
    use crate::ai::types::{FunctionCall, ToolCall};
    use serde_json::Value;

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
                        name: "read_file".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("...".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
        ];
        assert!(turn_uses_repo_inspection_tools(&messages));
    }

    #[test]
    fn project_writeback_replaces_existing_entry_by_source() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_project_writeback_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let store = MemoryStore::from_env_or_config();
        let source = "project_writeback:rust_tools";
        let created = upsert_project_writeback_entry(
            &store,
            source,
            "- rust_tools 项目结构：src/bin 放入口，src/cw 放通用组件\n- rust_tools 构建流程：优先使用 cargo check 做快速验证",
            vec!["project".to_string(), "rust_tools".to_string()],
            180,
        )
        .unwrap();
        assert!(matches!(created, ProjectWritebackUpsert::Saved));

        let updated = upsert_project_writeback_entry(
            &store,
            source,
            "- rust_tools 项目结构：src/bin 放入口，src/cw 放通用组件\n- rust_tools 构建流程：优先使用 cargo test --bin a 做行为验证",
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
        assert!(entries[0].note.contains("cargo test --bin a"));
        assert_eq!(entries[0].priority, Some(200));
        assert!(entries[0].tags.iter().any(|tag| tag == "structure"));

        let unchanged = upsert_project_writeback_entry(
            &store,
            source,
            "- rust_tools 项目结构：src/bin 放入口，src/cw 放通用组件\n- rust_tools 构建流程：优先使用 cargo test --bin a 做行为验证",
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
    fn project_writeback_rejects_user_local_skill_path_note() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_project_writeback_reject_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let store = MemoryStore::from_env_or_config();
        let source = "project_writeback:main-test";
        let rejected = upsert_project_writeback_entry(
            &store,
            source,
            "- Skill 文件位置：~/.config/rust_tools/skills/feishu-upload-md.skill\n- 工作流程：读取Markdown → 调用飞书API → 返回文档链接\n- 支持的调用方式：tool_spawn 直接调用或 Python 脚本",
            vec![
                "project".to_string(),
                "main-test".to_string(),
                "skill".to_string(),
            ],
            180,
        )
        .unwrap();

        assert!(matches!(rejected, ProjectWritebackUpsert::Rejected));
        let entries = store
            .recent(20)
            .unwrap()
            .into_iter()
            .filter(|entry| entry.source.as_deref() == Some(source))
            .collect::<Vec<_>>();
        assert!(entries.is_empty());

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn shared_quality_pipeline_rejects_short_generic_note() {
        let assessment = assess_learning_note_quality("be careful");
        assert_eq!(assessment.score, 0);
        assert!(!assessment.high_quality);
    }

    #[test]
    fn shared_quality_pipeline_accepts_actionable_general_rule() {
        let assessment = assess_learning_note_quality(
            "Files should end with a trailing newline before write_file commits the edit",
        );
        assert!(assessment.high_quality);
        assert!(assessment.directive_signals > 0);
    }

    #[test]
    fn shared_quality_pipeline_rejects_user_local_path_note() {
        let assessment = assess_learning_note_quality(
            "- Skill 文件位置：~/.config/rust_tools/skills/feishu-upload-md.skill\n- 工作流程：读取Markdown → 调用飞书API → 返回文档链接\n- 支持的调用方式：tool_spawn 直接调用或 Python 脚本",
        );
        assert!(!assessment.high_quality);
        assert!(assessment.one_off_signals > 0);
    }
}
