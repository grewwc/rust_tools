//! Typed provenance for verified semantic memories, separate from searchable notes.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use crate::ai::tools::storage::memory_store::AgentMemoryEntry;

pub(crate) const DISTILLED_SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DistilledEvidence {
    pub source_digest: String,
    pub message_id: String,
    pub role: String,
    pub quote: String,
    pub text_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DistilledRevision {
    pub revision: u32,
    pub note: String,
    pub content_digest: String,
    pub evidence: Vec<DistilledEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DistilledMetadata {
    pub schema: u32,
    pub scope: String,
    pub revision: u32,
    pub topic_key: String,
    pub verified: bool,
    pub content_digest: String,
    pub evidence: Vec<DistilledEvidence>,
    pub source_digests: Vec<String>,
    pub previous_revisions: Vec<DistilledRevision>,
}

pub(crate) fn digest(text: impl AsRef<[u8]>) -> String {
    Sha256::digest(text.as_ref()).iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn entry_content_digest(entry: &AgentMemoryEntry) -> String {
    digest(serde_json::to_vec(&(&entry.category, &entry.note, &entry.tags)).unwrap_or_default())
}

/// Old snippet entries, manual edits and incomplete semantic outputs are not recallable.
pub(crate) fn active_distilled_metadata(entry: &AgentMemoryEntry) -> Option<DistilledMetadata> {
    let metadata = entry.distilled.as_ref()?;
    (metadata.schema == DISTILLED_SCHEMA
        && metadata.verified && metadata.revision > 0
        && !metadata.scope.is_empty() && !metadata.topic_key.is_empty()
        && !metadata.evidence.is_empty()
        && metadata.content_digest == entry_content_digest(entry)
        && metadata.evidence.iter().all(|evidence| {
            matches!(evidence.role.as_str(), "user" | "tool")
                && !evidence.quote.trim().is_empty() && !evidence.message_id.is_empty()
                && metadata.source_digests.contains(&evidence.source_digest)
        }))
    .then(|| metadata.clone())
}

/// Scope is the ingestion environment, never a provenance inference from a zip.
pub(crate) fn current_project_scope() -> String {
    let Ok(cwd) = crate::ai::driver::runtime_ctx::effective_cwd() else { return String::new(); };
    let root = cwd.ancestors().find(|path| path.join(".git").exists()).unwrap_or(&cwd);
    root.canonicalize().unwrap_or_else(|_| root.to_path_buf()).to_string_lossy().into_owned()
}