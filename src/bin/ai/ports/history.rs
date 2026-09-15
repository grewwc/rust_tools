// =============================================================================
// HistoryStore - history storage port (dependency inversion)
// =============================================================================
// Previously the driver called concrete functions in `crate::ai::history::{blob, sqlite::*}`
// directly, making it impossible to insert cross-cutting logic such as audit/encryption/mock.
// Decoupled via a trait now; the driver depends only on the abstraction.
use std::{
    io,
    path::{Path, PathBuf},
};

use crate::ai::history::{
    ContextCompressionOutcome, ContextCompressionStatus, Message, messages_total_chars_pub,
};

// =============================================================================
// Compressor - history compression strategy port (pluggable)
// =============================================================================
/// Pluggable compressor: trims/summarizes the loaded messages to fit a budget.
/// The default implementation delegates to the existing `history::compress` logic; the Noop
/// implementation is used for tests/bypass. Object-safe, so it supports `Box<dyn Compressor>`
/// injection.
pub(crate) trait Compressor: Send + Sync {
    fn compress(
        &self,
        messages: Vec<Message>,
        max_chars: usize,
        keep_last: usize,
        summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
    ) -> ContextCompressionOutcome;
    fn name(&self) -> &str;
}

/// Default compressor: preserves the caller's summary budget, archive sink, and working
/// directory while reporting compaction status separately from budget success.
pub(crate) struct DefaultCompressor;
impl Compressor for DefaultCompressor {
    fn compress(
        &self,
        messages: Vec<Message>,
        max_chars: usize,
        keep_last: usize,
        summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
    ) -> ContextCompressionOutcome {
        crate::ai::history::compress_messages_for_context_with_outcome(
            messages,
            max_chars,
            keep_last,
            summary_max_chars,
            overflow_dir,
            cwd,
        )
    }
    fn name(&self) -> &str {
        "default"
    }
}

/// No-op compressor: returns messages as-is without trimming or summarizing, used in tests or to
/// disable compression.
pub(crate) struct NoopCompressor;
impl Compressor for NoopCompressor {
    fn compress(
        &self,
        messages: Vec<Message>,
        max_chars: usize,
        _keep_last: usize,
        _summary_max_chars: usize,
        _overflow_dir: Option<PathBuf>,
        _cwd: Option<&Path>,
    ) -> ContextCompressionOutcome {
        let before_chars = messages_total_chars_pub(&messages);
        ContextCompressionOutcome::new(
            messages,
            before_chars,
            max_chars,
            ContextCompressionStatus::Disabled,
        )
    }
    fn name(&self) -> &str {
        "noop"
    }
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

    /// Compatibility wrapper for injected compression. Reports a diagnostic before returning
    /// messages; callers that need machine-readable status should use the outcome variant.
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
        self.build_context_with_compressor_outcome(
            history_count,
            history_file,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
            overflow_dir,
            cwd,
            compressor,
        )
        .map(ContextCompressionOutcome::into_messages)
    }

    /// Reads raw history and applies the injected strategy with all compression inputs intact.
    /// This bypasses projection caches. `history_count` is not a truncation boundary here:
    /// the compressor must archive any older span before applying its `keep_last` policy.
    fn build_context_with_compressor_outcome(
        &self,
        _history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        history_summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
        compressor: &dyn Compressor,
    ) -> io::Result<ContextCompressionOutcome> {
        let messages = self.load_messages(history_file)?;
        Ok(compressor.compress(
            messages,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
            overflow_dir,
            cwd,
        ))
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

    fn dialogue() -> Vec<Message> {
        [
            ("user", "old request\n".repeat(1_000)),
            ("assistant", "old answer\n".repeat(1_000)),
            ("user", "current request".to_string()),
        ]
        .into_iter()
        .map(|(role, content)| Message {
            role: role.to_string(),
            content: serde_json::json!(content),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        })
        .collect()
    }

    #[test]
    fn default_compressor_archives_before_zero_summary_compaction() {
        let dir = std::env::temp_dir().join(format!("ai-port-archive-{}", uuid::Uuid::new_v4()));
        let messages = dialogue();
        let outcome = DefaultCompressor.compress(
            messages.clone(),
            2_048,
            1,
            0,
            Some(dir.clone()),
            None,
        );
        assert_eq!(outcome.status, ContextCompressionStatus::Complete);
        assert_eq!(outcome.before_chars, messages_total_chars_pub(&messages));
        assert!(outcome.after_chars < outcome.before_chars);
        assert!(outcome.budget_met());
        assert_eq!(outcome.messages.last(), messages.last());
        assert!(!outcome.messages.contains(&messages[0]));
        assert!(!outcome.messages.contains(&messages[1]));
        let archived = std::fs::read_to_string(dir.join("overflow-history.md")).unwrap();
        assert!(archived.contains(messages[0].content.as_str().unwrap()));
        assert!(archived.contains(messages[1].content.as_str().unwrap()));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn default_compressor_missing_sink_preserves_dialogue() {
        let messages = dialogue();
        for summary_max_chars in [0, 2_048] {
            let outcome = DefaultCompressor.compress(
                messages.clone(),
                2_048,
                1,
                summary_max_chars,
                None,
                None,
            );
            assert_eq!(outcome.status, ContextCompressionStatus::MissingArchiveSink);
            assert_eq!(outcome.messages, messages);
            assert_eq!(outcome.before_chars, outcome.after_chars);
            assert!(!outcome.budget_met());
            assert!(outcome.diagnostic().unwrap().contains("no archive sink"));
        }
    }

    #[test]
    fn noop_compressor_reports_disabled_without_archiving() {
        let dir = std::env::temp_dir().join(format!("ai-port-noop-{}", uuid::Uuid::new_v4()));
        let messages = dialogue();
        let outcome = NoopCompressor.compress(
            messages.clone(),
            1,
            1,
            2_048,
            Some(dir.clone()),
            None,
        );
        assert_eq!(outcome.status, ContextCompressionStatus::Disabled);
        assert_eq!(outcome.messages, messages);
        assert_eq!(outcome.before_chars, outcome.after_chars);
        assert!(!outcome.budget_met());
        assert!(!dir.exists());
    }

    #[test]
    fn history_store_injected_compressor_forwards_all_context_options() {
        struct AssertOptions;
        impl Compressor for AssertOptions {
            fn compress(
                &self,
                messages: Vec<Message>,
                max_chars: usize,
                keep_last: usize,
                summary_max_chars: usize,
                overflow_dir: Option<PathBuf>,
                cwd: Option<&Path>,
            ) -> ContextCompressionOutcome {
                assert_eq!((max_chars, keep_last, summary_max_chars), (2_048, 1, 777));
                assert_eq!(overflow_dir.as_deref(), Some(Path::new("explicit-archive")));
                assert_eq!(cwd, Some(Path::new("explicit-cwd")));
                NoopCompressor.compress(messages, max_chars, keep_last, summary_max_chars, overflow_dir, cwd)
            }
            fn name(&self) -> &str {
                "assert-options"
            }
        }
        let messages = dialogue();
        let store = InMemoryHistoryStore {
            messages: std::sync::Mutex::new(messages.clone()),
        };
        let outcome = store
            .build_context_with_compressor_outcome(
                1,
                Path::new("unused"),
                2_048,
                1,
                777,
                Some(PathBuf::from("explicit-archive")),
                Some(Path::new("explicit-cwd")),
                &AssertOptions,
            )
            .unwrap();
        assert_eq!(outcome.messages, messages);
        assert_eq!(outcome.status, ContextCompressionStatus::Disabled);
        let compatible = store
            .build_context_with_boxed_compressor(
                1,
                Path::new("unused"),
                2_048,
                1,
                777,
                Some(PathBuf::from("explicit-archive")),
                Some(Path::new("explicit-cwd")),
                Box::new(AssertOptions),
            )
            .unwrap();
        assert_eq!(compatible, messages);
        assert_eq!(*store.messages.lock().unwrap(), messages);
    }

    #[test]
    fn default_history_store_injected_sink_archives_without_mutating_history() {
        let dir = std::env::temp_dir().join(format!("ai-port-store-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let history_file = dir.join("history.sqlite");
        let archive_dir = dir.join("archive");
        let store = DefaultHistoryStore;
        store.append_messages(&history_file, &dialogue()).unwrap();
        let raw = store.load_messages(&history_file).unwrap();
        let outcome = store
            .build_context_with_compressor_outcome(
                1,
                &history_file,
                2_048,
                1,
                0,
                Some(archive_dir.clone()),
                Some(&dir),
                &DefaultCompressor,
            )
            .unwrap();
        assert_eq!(outcome.status, ContextCompressionStatus::Complete);
        assert!(outcome.budget_met());
        assert!(outcome.after_chars < outcome.before_chars);
        assert_eq!(outcome.messages.last(), raw.last());
        let archived = std::fs::read_to_string(archive_dir.join("overflow-history.md")).unwrap();
        assert!(archived.contains(raw[0].content.as_str().unwrap()));
        assert!(archived.contains(raw[1].content.as_str().unwrap()));
        assert_eq!(store.load_messages(&history_file).unwrap(), raw);
        std::fs::remove_dir_all(dir).unwrap();
    }

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
