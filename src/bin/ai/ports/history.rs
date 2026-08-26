// =============================================================================
// HistoryStore - history storage port (dependency inversion)
// =============================================================================
// Previously the driver called concrete functions in `crate::ai::history::{blob, sqlite::*}`
// directly, making it impossible to insert cross-cutting logic such as audit/encryption/mock.
// Decoupled via a trait now; the driver depends only on the abstraction.
use std::{io, path::{Path, PathBuf}};

use crate::ai::history::Message;

// =============================================================================
// Compressor - history compression strategy port (pluggable)
// =============================================================================
/// Pluggable compressor: trims/summarizes the loaded messages to fit a budget.
/// The default implementation delegates to the existing `history::compress` logic; the Noop
/// implementation is used for tests/bypass. Object-safe, so it supports `Box<dyn Compressor>`
/// injection.
pub(crate) trait Compressor: Send + Sync {
    fn compress(&self, messages: Vec<Message>, max_chars: usize, keep_last: usize) -> Vec<Message>;
    fn name(&self) -> &str;
}

/// Default compressor: delegates to the existing `history::compress_messages_for_context` (the
/// summary=0 simplified variant). Zero behavior change: the default path still goes through
/// `HistoryStore::build_context`; this implementation only takes effect when explicitly injected.
/// For full summary/overflow support, build a custom Compressor with extra parameters on the
/// calling side.
pub(crate) struct DefaultCompressor;
impl Compressor for DefaultCompressor {
    fn compress(&self, messages: Vec<Message>, max_chars: usize, keep_last: usize) -> Vec<Message> {
        if max_chars == 0 || messages.is_empty() {
            return messages;
        }
        // Forward to the existing compression logic (summary_max_chars=0, no overflow archiving),
        // keeping semantics identical to the hardcoded path; the full path is still owned by
        // HistoryStore::build_context, with zero behavior change.
        crate::ai::history::compress_messages_for_context(messages, max_chars, keep_last, 0, None, None)
    }
    fn name(&self) -> &str { "default" }
}

/// No-op compressor: returns messages as-is without trimming or summarizing, used in tests or to
/// disable compression.
pub(crate) struct NoopCompressor;
impl Compressor for NoopCompressor {
    fn compress(&self, messages: Vec<Message>, _max_chars: usize, _keep_last: usize) -> Vec<Message> {
        messages
    }
    fn name(&self) -> &str { "noop" }
}

/// History storage port: object-safe, minimal, and does not leak SQLite/text dual-backend
/// details. Kept `pub(crate)` to avoid leaking private types such as the internal `RequestError`
/// into the public API.
pub(crate) trait HistoryStore: Send + Sync {
    /// Reads the context projection that can be sent to the model (already compressed, trimmed,
    /// and overflow-archived). `cwd` is used to decide overflow-archive reuse for relative
    /// paths; pass `None` when unavailable (consistent with the `cwd: Option<&Path>` contract of
    /// the underlying `build_context_history`).
    fn build_context(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        history_summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
    ) -> io::Result<Vec<Message>>;

    /// Pluggable compression variant: lets the caller inject a `Compressor` strategy.
    /// The default implementation stays backward compatible - it ignores the compressor and
    /// forwards to `build_context`, guaranteeing zero behavior change. Concrete `HistoryStore`
    /// implementations can override this method to actually apply `compressor.compress`.
    fn build_context_with_compressor(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        history_summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
        compressor: &dyn Compressor,
    ) -> io::Result<Vec<Message>> {
        let _ = compressor;
        self.build_context(
            history_count,
            history_file,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
            overflow_dir,
            cwd,
        )
    }

    /// Convenience overload taking `Box<dyn Compressor>`, so `Pipeline` can inject it by ownership.
    fn build_context_with_boxed_compressor(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        history_summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
        compressor: Box<dyn Compressor>,
    ) -> io::Result<Vec<Message>> {
        self.build_context_with_compressor(
            history_count,
            history_file,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
            overflow_dir,
            cwd,
            compressor.as_ref(),
        )
    }

    /// Appends messages to the canonical history (atomic write + overflow archiving).
    fn append_messages(&self, history_file: &Path, msgs: &[Message]) -> io::Result<()>;

    /// Model-aware append: the sqlite backend additionally records source_model provenance.
    /// The default implementation degrades to `append_messages`, with zero breakage for custom
    /// stores that do not care about model provenance.
    fn append_messages_for_model(
        &self,
        history_file: &Path,
        msgs: &[Message],
        source_model: &str,
    ) -> io::Result<()> {
        let _ = source_model;
        self.append_messages(history_file, msgs)
    }

    /// Loads raw history (for debugging / replay).
    fn load_messages(&self, history_file: &Path) -> io::Result<Vec<Message>>;
}

/// Default implementation: delegates to the concrete functions in the existing `history` module,
/// keeping behavior 100% identical.
pub(crate) struct DefaultHistoryStore;

impl HistoryStore for DefaultHistoryStore {
    fn build_context(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        history_summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
    ) -> io::Result<Vec<Message>> {
        crate::ai::history::build_context_history(
            history_count,
            history_file,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
            overflow_dir,
            cwd,
        )
        .map_err(|e| {
            // Preserve the original io::ErrorKind (e.g. WouldBlock) so upper layers can retry
            // snapshots and similar operations.
            if e.is::<io::Error>() {
                match e.downcast::<io::Error>() {
                    Ok(io_err) => *io_err,
                    Err(e2) => io::Error::new(io::ErrorKind::Other, e2.to_string()),
                }
            } else {
                io::Error::new(io::ErrorKind::Other, e.to_string())
            }
        })
    }

    fn build_context_with_compressor(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        _history_summary_max_chars: usize,
        _overflow_dir: Option<PathBuf>,
        _cwd: Option<&Path>,
        compressor: &dyn Compressor,
    ) -> io::Result<Vec<Message>> {
        // Demonstrate "real pluggability": load the raw history first, then delegate to the
        // compressor. Unlike the full sqlite+snapshot+cache path in `build_context`, this path
        // is used when a strategy is explicitly injected from pipeline/tests; production behavior
        // still goes through `build_context`, with zero behavior change.
        // To avoid conflicting with cache semantics, this path bypasses `build_context_history`'s
        // cache and reads directly.
        let _ = history_count; // the pluggable path does not truncate by history_count; the compressor decides via keep_last
        let messages = self.load_messages(history_file)?;
        Ok(compressor.compress(messages, history_max_chars, history_keep_last))
    }

    fn append_messages(&self, history_file: &Path, msgs: &[Message]) -> io::Result<()> {
        crate::ai::history::append_history_messages(history_file, msgs)
    }

    fn append_messages_for_model(
        &self,
        history_file: &Path,
        msgs: &[Message],
        source_model: &str,
    ) -> io::Result<()> {
        // The sqlite backend writes source_model into the meta column (model provenance); the
        // blob backend degrades to a plain append.
        crate::ai::history::append_history_messages_for_model(history_file, msgs, source_model)
    }

    fn load_messages(&self, history_file: &Path) -> io::Result<Vec<Message>> {
        // Dispatch correctly between the sqlite / blob backends (decided by is_sqlite_path inside
        // build_message_arr), avoiding silent data loss when read_to_string would empty the
        // sqlite binary.
        crate::ai::history::build_message_arr(usize::MAX, history_file).map_err(|e| {
            if e.is::<io::Error>() {
                match e.downcast::<io::Error>() {
                    Ok(io_err) => *io_err,
                    Err(e2) => io::Error::new(io::ErrorKind::Other, e2.to_string()),
                }
            } else {
                io::Error::new(io::ErrorKind::Other, e.to_string())
            }
        })
    }
}

/// In-memory implementation: used in tests / middleware unit tests; never touches the filesystem.
#[cfg(test)]
pub(crate) struct InMemoryHistoryStore {
    pub(crate) messages: std::sync::Mutex<Vec<Message>>,
}

#[cfg(test)]
impl HistoryStore for InMemoryHistoryStore {
    fn build_context(
        &self,
        _history_count: usize,
        _history_file: &Path,
        _history_max_chars: usize,
        _history_keep_last: usize,
        _history_summary_max_chars: usize,
        _overflow_dir: Option<PathBuf>,
        _cwd: Option<&Path>,
    ) -> io::Result<Vec<Message>> {
        Ok(self.messages.lock().unwrap().clone())
    }
    fn append_messages(&self, _history_file: &Path, msgs: &[Message]) -> io::Result<()> {
        self.messages.lock().unwrap().extend_from_slice(msgs);
        Ok(())
    }
    fn load_messages(&self, _history_file: &Path) -> io::Result<Vec<Message>> {
        Ok(self.messages.lock().unwrap().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::history::Message;

    #[test]
    fn in_memory_store_append_messages_for_model_falls_back_to_plain_append() {
        let store = InMemoryHistoryStore {
            messages: std::sync::Mutex::new(Vec::new()),
        };
        let msgs = vec![Message {
            role: "user".into(),
            content: serde_json::json!("hi"),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        // Step 6: the trait's default implementation ignores source_model and degrades to
        // `append_messages`, with zero breakage for custom stores that do not care about model
        // provenance.
        store
            .append_messages_for_model(Path::new("unused"), &msgs, "some-model")
            .unwrap();
        assert_eq!(store.messages.lock().unwrap().len(), 1);
    }
}
