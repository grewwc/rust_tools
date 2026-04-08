use crate::ai::knowledge::config::KnowledgeConfig;
use crate::ai::knowledge::retrieval::recall;
use crate::ai::knowledge::storage::jsonl_store::JsonlStore;
use crate::ai::tools::storage::memory_store::MemoryStore;

pub(crate) struct AutoRecalledKnowledge {
    pub(crate) content: String,
    pub(crate) high_confidence_project_memory: bool,
    pub(crate) entry_count: usize,
    pub(crate) project_hint: Option<String>,
    pub(crate) categories: Vec<String>,
}

pub(crate) struct RecallBundle {
    pub(crate) guidelines: Option<String>,
    pub(crate) recalled: Option<AutoRecalledKnowledge>,
}

fn current_project_hint() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let name = cwd.file_name()?.to_str()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

pub(crate) fn build_persistent_guidelines(question: &str, max_chars: usize) -> Option<String> {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let config = KnowledgeConfig::from_config_file();
    build_persistent_guidelines_from_parts(&jsonl_store, question, max_chars, &config)
}

pub(crate) fn build_auto_recalled_knowledge(
    question: &str,
    max_chars: usize,
) -> Option<AutoRecalledKnowledge> {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let config = KnowledgeConfig::from_config_file();
    build_auto_recalled_knowledge_from_parts(&jsonl_store, question, max_chars, &config)
}

pub(crate) fn build_auto_recalled_knowledge_with_project(
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

pub(crate) fn build_recall_bundle(
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

pub(super) fn current_project_name_hint() -> Option<String> {
    current_project_hint()
}
