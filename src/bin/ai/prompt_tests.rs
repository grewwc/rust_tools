    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn refresh_generated_session_topic_replaces_fallback_topic() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rust_tools_prompt_topic_{}_{}",
            std::process::id(),
            unique
        ));
        let history_file = root.join("history.jsonl");
        let session_id = "topic-test";

        let store = SessionStore::new(&history_file);
        store.ensure_root_dir().unwrap();
        store
            .write_session_title(session_id, "生成后的标题")
            .unwrap();

        let mut editor = PromptEditor::new(session_id, &history_file);
        editor.set_session_topic(Some("首条消息摘要".to_string()));

        assert_eq!(
            editor.refresh_generated_session_topic().unwrap(),
            Some(true)
        );
        assert_eq!(editor.session_topic.as_deref(), Some("生成后的标题"));

        let _ = fs::remove_dir_all(root);
    }
