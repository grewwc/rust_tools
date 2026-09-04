/// Decision log module - records the AI agent's key decision-making process
///
/// Used for meta-cognition: traces back "why a certain choice was made", aiding debugging and optimization
use chrono::Local;
use rust_tools::cw::SkipMap;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Byte cap for the on-disk decision log; exceeding it triggers one tail-retaining compaction. About 8MB.
const DECISION_LOG_MAX_PERSIST_BYTES: u64 = 8 * 1024 * 1024;
/// Number of recent lines retained after compaction (same order as the in-memory max_capacity, enough to replay the current session).
const DECISION_LOG_RETAIN_LINES: usize = 2000;

/// Decision type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecisionType {
    /// Skill selection
    SkillSelection,
    /// Tool call
    ToolInvocation,
    /// Model routing (which model to choose)
    ModelRouting,
    /// Memory retrieval
    MemoryRetrieval,
    /// Memory save gate
    MemorySave,
    /// Reflection trigger
    ReflectionTrigger,
    /// Scheduler dispatch and evaluation
    SchedulerDispatch,
    /// Auxiliary LLM tasks such as session title generation
    SessionTitle,
    /// Reasoning effort downgrade decision on truncation retry
    TruncationDowngrade,
    /// Root-cause record for a runtime-forced no-tool handoff
    RuntimeStop,
}

/// Decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionLog {
    /// Timestamp (Unix timestamp in milliseconds)
    pub timestamp: i64,
    /// Session ID
    pub session_id: String,
    /// Turn ID
    pub turn_id: usize,
    /// Decision type
    pub decision_type: DecisionType,
    /// Context (user input / current state)
    pub context: String,
    /// Alternatives considered
    pub alternatives_considered: Vec<String>,
    /// Final choice
    pub chosen_option: String,
    /// Rationale for the choice
    pub reasoning: String,
    /// Confidence (0.0 - 1.0)
    pub confidence: Option<f64>,
    /// Post-hoc outcome (filled in after execution)
    pub outcome: Option<Outcome>,
    /// Execution duration (milliseconds)
    pub execution_time_ms: Option<u64>,
}

/// Decision outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub success: bool,
    pub message: String,
    pub user_feedback: Option<UserFeedback>,
}

/// User feedback
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UserFeedback {
    Positive,
    Negative,
    Neutral,
}

/// Decision log store
pub struct DecisionLogStore {
    logs: Arc<Mutex<Vec<DecisionLog>>>,
    max_capacity: usize,
    persist_path: Arc<Mutex<Option<PathBuf>>>,
}

impl DecisionLogStore {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            logs: Arc::new(Mutex::new(Vec::with_capacity(max_capacity))),
            max_capacity,
            persist_path: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_persist_path<P: AsRef<Path>>(&self, path: P) {
        let mut guard = self.persist_path.lock().unwrap();
        *guard = Some(path.as_ref().to_path_buf());
    }

    pub fn clear_persist_path(&self) {
        let mut guard = self.persist_path.lock().unwrap();
        *guard = None;
    }

    fn persist_log_if_enabled(&self, log: &DecisionLog) {
        let path = {
            let guard = self.persist_path.lock().unwrap();
            guard.clone()
        };
        let Some(path) = path else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
            return;
        };
        let Ok(line) = serde_json::to_string(log) else {
            return;
        };
        let _ = writeln!(file, "{}", line);
        drop(file);

        // The decision log is append-only JSONL: the in-memory buffer is bounded by max_capacity, but the on-disk file
        // grows without bound if never rotated, and `replay_recent_from_disk` reads the whole file line by line every time.
        // Use one O(1) metadata probe and compact with tail retention only when the cap is exceeded.
        if let Ok(meta) = fs::metadata(&path)
            && meta.len() > DECISION_LOG_MAX_PERSIST_BYTES
        {
            self.compact_persist_file(&path);
        }
    }

    /// Compacts the on-disk log file down to the most recent `DECISION_LOG_RETAIN_LINES` lines, using a temp file
    /// + atomic rename so readers never see a partial file. Best-effort: any failed step aborts the compaction without
    /// affecting the main flow. Concurrent writes follow last-writer-wins; at most a few not-yet-compacted tail lines are lost.
    fn compact_persist_file(&self, path: &Path) {
        let Ok(file) = std::fs::File::open(path) else {
            return;
        };
        let reader = BufReader::new(file);
        let mut lines: Vec<String> = reader.lines().map_while(Result::ok).collect();
        if lines.len() <= DECISION_LOG_RETAIN_LINES {
            return;
        }
        let start = lines.len() - DECISION_LOG_RETAIN_LINES;
        let retained = lines.split_off(start);

        let tmp_path = path.with_extension("jsonl.tmp");
        let Ok(mut tmp) = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
        else {
            return;
        };
        for line in &retained {
            if writeln!(tmp, "{}", line).is_err() {
                let _ = fs::remove_file(&tmp_path);
                return;
            }
        }
        if tmp.flush().is_err() {
            let _ = fs::remove_file(&tmp_path);
            return;
        }
        drop(tmp);
        let _ = fs::rename(&tmp_path, path);
    }

    /// Record a decision
    pub fn log(&self, mut log: DecisionLog) {
        let mut logs = self.logs.lock().unwrap();

        // Set the timestamp
        log.timestamp = Local::now().timestamp_millis();

        // If over capacity, drop the oldest 10%
        if logs.len() >= self.max_capacity {
            let remove_count = self.max_capacity / 10;
            logs.drain(0..remove_count);
        }

        let persist_copy = log.clone();
        logs.push(log);
        drop(logs);
        self.persist_log_if_enabled(&persist_copy);
    }

    /// Get the most recent N log entries
    pub fn recent(&self, n: usize) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        let start = logs.len().saturating_sub(n);
        logs[start..].to_vec()
    }

    pub fn recent_by_session(&self, session_id: &str, n: usize) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        let filtered = logs
            .iter()
            .filter(|log| log.session_id == session_id)
            .cloned()
            .collect::<Vec<_>>();
        let start = filtered.len().saturating_sub(n);
        filtered[start..].to_vec()
    }

    pub fn replay_recent_from_disk(&self, session_id: &str, n: usize) -> Vec<DecisionLog> {
        let path = {
            let guard = self.persist_path.lock().unwrap();
            guard.clone()
        };
        let Some(path) = path else {
            return Vec::new();
        };
        let Ok(file) = std::fs::File::open(path) else {
            return Vec::new();
        };
        let reader = BufReader::new(file);
        let mut out = Vec::new();
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(log) = serde_json::from_str::<DecisionLog>(&line)
                && log.session_id == session_id
            {
                out.push(log);
            }
        }
        let start = out.len().saturating_sub(n);
        out[start..].to_vec()
    }

    /// Filter log entries by type
    pub fn by_type(&self, decision_type: &DecisionType) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        logs.iter()
            .filter(|log| &log.decision_type == decision_type)
            .cloned()
            .collect()
    }

    /// Get failed decision log entries
    pub fn failures(&self) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        logs.iter()
            .filter(|log| log.outcome.as_ref().map(|o| !o.success).unwrap_or(false))
            .cloned()
            .collect()
    }

    /// Get low-confidence decision log entries
    pub fn low_confidence(&self, threshold: f64) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        logs.iter()
            .filter(|log| log.confidence.map(|c| c < threshold).unwrap_or(false))
            .cloned()
            .collect()
    }

    /// Update the outcome of a decision
    pub fn update_outcome(&self, session_id: &str, turn_id: usize, outcome: Outcome) {
        let mut logs = self.logs.lock().unwrap();
        if let Some(log) = logs
            .iter_mut()
            .find(|log| log.session_id == session_id && log.turn_id == turn_id)
        {
            log.outcome = Some(outcome);
        }
    }

    /// Record user feedback
    pub fn add_feedback(&self, session_id: &str, turn_id: usize, feedback: UserFeedback) {
        let mut logs = self.logs.lock().unwrap();
        if let Some(log) = logs
            .iter_mut()
            .find(|log| log.session_id == session_id && log.turn_id == turn_id)
        {
            if let Some(outcome) = &mut log.outcome {
                outcome.user_feedback = Some(feedback);
            } else {
                log.outcome = Some(Outcome {
                    success: feedback != UserFeedback::Negative,
                    message: String::new(),
                    user_feedback: Some(feedback),
                });
            }
        }
    }

    /// Export as a JSON string
    pub fn export_json(&self, n: Option<usize>) -> String {
        let logs = self.logs.lock().unwrap();
        let logs_to_export = if let Some(n) = n {
            let start = logs.len().saturating_sub(n);
            &logs[start..]
        } else {
            &logs[..]
        };

        serde_json::to_string_pretty(logs_to_export)
            .unwrap_or_else(|e| format!("Error serializing logs: {}", e))
    }

    /// Statistics
    pub fn stats(&self) -> DecisionStats {
        let logs = self.logs.lock().unwrap();

        let total = logs.len();
        let successes = logs
            .iter()
            .filter(|log| log.outcome.as_ref().map(|o| o.success).unwrap_or(false))
            .count();
        let failures = total - successes;

        let by_type: SkipMap<String, usize> = logs
            .iter()
            .map(|log| format!("{:?}", log.decision_type))
            .fold(SkipMap::default(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            });

        let confidence_count = logs.iter().filter(|log| log.confidence.is_some()).count();
        let avg_confidence = if confidence_count > 0 {
            logs.iter().filter_map(|log| log.confidence).sum::<f64>() / (confidence_count as f64)
        } else {
            0.0
        };

        let exec_time_count = logs
            .iter()
            .filter(|log| log.execution_time_ms.is_some())
            .count();
        let avg_execution_time_ms = if exec_time_count > 0 {
            logs.iter()
                .filter_map(|log| log.execution_time_ms)
                .sum::<u64>() as f64
                / (exec_time_count as f64)
        } else {
            0.0
        };

        DecisionStats {
            total,
            successes,
            failures,
            success_rate: if total > 0 {
                successes as f64 / total as f64
            } else {
                0.0
            },
            by_type,
            avg_confidence,
            avg_execution_time_ms,
        }
    }

    /// Clear the log
    pub fn clear(&self) {
        let mut logs = self.logs.lock().unwrap();
        logs.clear();
    }
}

/// Decision statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionStats {
    pub total: usize,
    pub successes: usize,
    pub failures: usize,
    pub success_rate: f64,
    pub by_type: SkipMap<String, usize>,
    pub avg_confidence: f64,
    pub avg_execution_time_ms: f64,
}

/// Helper: create a skill-selection log entry
pub fn log_skill_selection(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    context: &str,
    candidates: Vec<&str>,
    chosen: &str,
    reasoning: &str,
    confidence: Option<f64>,
    execution_time_ms: u64,
) {
    store.log(DecisionLog {
        timestamp: 0, // Will be set by log()
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::SkillSelection,
        context: context.to_string(),
        alternatives_considered: candidates.iter().map(|s| s.to_string()).collect(),
        chosen_option: chosen.to_string(),
        reasoning: reasoning.to_string(),
        confidence,
        outcome: None,
        execution_time_ms: Some(execution_time_ms),
    });
}

/// Helper: create a tool-call log entry
pub fn log_tool_invocation(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    context: &str,
    tool_name: &str,
    reasoning: &str,
    confidence: Option<f64>,
    execution_time_ms: u64,
) {
    store.log(DecisionLog {
        timestamp: 0, // Will be set by log()
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::ToolInvocation,
        context: context.to_string(),
        alternatives_considered: vec![],
        chosen_option: tool_name.to_string(),
        reasoning: reasoning.to_string(),
        confidence,
        outcome: None,
        execution_time_ms: Some(execution_time_ms),
    });
}

/// Helper: record the reasoning effort downgrade decision made on truncation retry.
/// Used for post-hoc auditing of "which session, which turn, and which truncation caused a downgrade (or not)".
pub fn log_truncation_downgrade(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    model: &str,
    consecutive_truncations: usize,
    reasoning_tokens: u64,
    completion_tokens: u64,
    downgraded: bool,
    note: &str,
) {
    store.log(DecisionLog {
        timestamp: 0, // Will be set by log()
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::TruncationDowngrade,
        context: format!(
            "model={} truncation#{} reasoning={}/{} completion tokens",
            model, consecutive_truncations, reasoning_tokens, completion_tokens
        ),
        alternatives_considered: vec![],
        chosen_option: if downgraded {
            "reasoning_effort downgraded".to_string()
        } else {
            "reasoning_effort kept".to_string()
        },
        reasoning: note.to_string(),
        confidence: None,
        outcome: None,
        execution_time_ms: None,
    });
}

/// Record the root cause of a runtime-forced no-tool handoff for post-hoc auditing.
/// Writes only to the decision log (session side channel, never enters model context), so an internal note cannot be promoted to
/// a system message and replayed forever; the stop reason is also injected into the current request projection so the model can wrap up.
pub fn log_runtime_stop(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    reason: &str,
    target: Option<&str>,
    iteration: usize,
) {
    store.log(DecisionLog {
        timestamp: 0, // Will be set by log()
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::RuntimeStop,
        context: format!("reason={reason}, iteration={iteration}"),
        alternatives_considered: vec![],
        chosen_option: "no_tool_handoff".to_string(),
        reasoning: target.map(|t| format!("target={t}")).unwrap_or_default(),
        confidence: None,
        outcome: None,
        execution_time_ms: None,
    });
}

/// Helper: create a memory-retrieval log entry
pub fn log_memory_retrieval(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    query: &str,
    results_count: usize,
    reasoning: &str,
    execution_time_ms: u64,
) {
    store.log(DecisionLog {
        timestamp: 0, // Will be set by log()
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::MemoryRetrieval,
        context: query.to_string(),
        alternatives_considered: vec![],
        chosen_option: format!("Retrieved {} items", results_count),
        reasoning: reasoning.to_string(),
        confidence: None,
        outcome: None,
        execution_time_ms: Some(execution_time_ms),
    });
}

pub fn log_memory_save_assessment(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    requested_category: &str,
    final_category: &str,
    note: &str,
    assessment: &crate::ai::driver::reflection::LearningNoteAssessment,
    downgraded: bool,
) {
    let note_chars = note.chars().count();
    let preview: String = note.chars().take(160).collect();
    let context = serde_json::json!({
        "requested_category": requested_category,
        "final_category": final_category,
        "downgraded": downgraded,
        "note_chars": note_chars,
        "note_preview": preview,
    })
    .to_string();
    let reasoning = serde_json::to_string(assessment).unwrap_or_else(|_| "{}".to_string());
    let outcome_message = if downgraded {
        "memory_save downgraded to short-term self_note"
    } else {
        "memory_save accepted for requested category"
    };
    store.log(DecisionLog {
        timestamp: 0,
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::MemorySave,
        context,
        alternatives_considered: vec![requested_category.to_string(), "self_note".to_string()],
        chosen_option: final_category.to_string(),
        reasoning,
        confidence: Some(assessment.confidence()),
        outcome: Some(Outcome {
            success: !downgraded,
            message: outcome_message.to_string(),
            user_feedback: None,
        }),
        execution_time_ms: None,
    });
}

/// Helper: record a scheduler dispatch decision (including defer/selected and a scoring summary)
pub fn log_scheduler_dispatch(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    context: &str,
    alternatives: Vec<String>,
    chosen: &str,
    reasoning: &str,
    success: bool,
) {
    store.log(DecisionLog {
        timestamp: 0,
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::SchedulerDispatch,
        context: context.to_string(),
        alternatives_considered: alternatives,
        chosen_option: chosen.to_string(),
        reasoning: reasoning.to_string(),
        confidence: None,
        outcome: Some(Outcome {
            success,
            message: if success {
                "scheduler decision accepted".to_string()
            } else {
                "scheduler decision indicates risk".to_string()
            },
            user_feedback: None,
        }),
        execution_time_ms: None,
    });
}

/// Helper: record a session-title generation failure (silent paths such as transport/HTTP/parse errors or low-quality rejection).
/// Title generation failures used to leave only a commented-out eprintln, so users could not tell a "request timeout" from
/// "the model replied but was rejected by quality filtering", leaving the session stuck on a fallback title, unobservable.
pub fn log_session_title_failure(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    reason: &str,
    detail: &str,
) {
    store.log(DecisionLog {
        timestamp: 0,
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::SessionTitle,
        context: "session_title_generation".to_string(),
        alternatives_considered: vec!["fallback_title".to_string()],
        chosen_option: "fallback_title".to_string(),
        reasoning: reason.to_string(),
        confidence: None,
        outcome: Some(Outcome {
            success: false,
            message: detail.to_string(),
            user_feedback: None,
        }),
        execution_time_ms: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_log_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!(
            "rust_tools_{name}_{}_{}.jsonl",
            std::process::id(),
            ts
        ));
        path
    }

    #[test]
    fn test_log_store_basic() {
        let store = DecisionLogStore::new(100);

        store.log(DecisionLog {
            timestamp: 0,
            session_id: "test-session".to_string(),
            turn_id: 1,
            decision_type: DecisionType::SkillSelection,
            context: "test input".to_string(),
            alternatives_considered: vec!["skill_a".to_string(), "skill_b".to_string()],
            chosen_option: "skill_a".to_string(),
            reasoning: "test reasoning".to_string(),
            confidence: Some(0.85),
            outcome: None,
            execution_time_ms: Some(10),
        });

        let recent = store.recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].chosen_option, "skill_a");
    }

    #[test]
    fn test_log_store_capacity() {
        let store = DecisionLogStore::new(10);

        // Add 15 log entries
        for i in 0..15 {
            store.log(DecisionLog {
                timestamp: 0,
                session_id: "test-session".to_string(),
                turn_id: i,
                decision_type: DecisionType::SkillSelection,
                context: format!("input {}", i),
                alternatives_considered: vec![],
                chosen_option: format!("skill_{}", i),
                reasoning: "test".to_string(),
                confidence: None,
                outcome: None,
                execution_time_ms: None,
            });
        }

        // Only the most recent 10 entries should be kept (in practice 9-10 are kept because 10% get dropped)
        let recent = store.recent(100);
        assert!(recent.len() <= 10);
        assert_eq!(recent[0].turn_id, 5); // the oldest one is entry 5
    }

    #[test]
    fn test_outcome_update() {
        let store = DecisionLogStore::new(100);

        store.log(DecisionLog {
            timestamp: 0,
            session_id: "test-session".to_string(),
            turn_id: 1,
            decision_type: DecisionType::ToolInvocation,
            context: "test".to_string(),
            alternatives_considered: vec![],
            chosen_option: "tool_x".to_string(),
            reasoning: "test".to_string(),
            confidence: None,
            outcome: None,
            execution_time_ms: None,
        });

        store.update_outcome(
            "test-session",
            1,
            Outcome {
                success: true,
                message: "Tool executed successfully".to_string(),
                user_feedback: None,
            },
        );

        let recent = store.recent(1);
        assert!(recent[0].outcome.as_ref().unwrap().success);
    }

    #[test]
    fn test_stats() {
        let store = DecisionLogStore::new(100);

        // Add success and failure log entries
        for i in 0..5 {
            store.log(DecisionLog {
                timestamp: 0,
                session_id: "test-session".to_string(),
                turn_id: i,
                decision_type: DecisionType::SkillSelection,
                context: "test".to_string(),
                alternatives_considered: vec![],
                chosen_option: format!("skill_{}", i),
                reasoning: "test".to_string(),
                confidence: Some(0.8),
                outcome: Some(Outcome {
                    success: true,
                    message: "OK".to_string(),
                    user_feedback: None,
                }),
                execution_time_ms: Some(10),
            });
        }

        for i in 5..10 {
            store.log(DecisionLog {
                timestamp: 0,
                session_id: "test-session".to_string(),
                turn_id: i,
                decision_type: DecisionType::ToolInvocation,
                context: "test".to_string(),
                alternatives_considered: vec![],
                chosen_option: format!("tool_{}", i),
                reasoning: "test".to_string(),
                confidence: Some(0.6),
                outcome: Some(Outcome {
                    success: false,
                    message: "Failed".to_string(),
                    user_feedback: None,
                }),
                execution_time_ms: Some(20),
            });
        }

        let stats = store.stats();
        assert_eq!(stats.total, 10);
        assert_eq!(stats.successes, 5);
        assert_eq!(stats.failures, 5);
        assert!((stats.success_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_persist_and_replay_recent_by_session() {
        let store = DecisionLogStore::new(100);
        let path = temp_log_path("decision_log_persist");
        store.set_persist_path(&path);

        for turn in 0..5usize {
            log_scheduler_dispatch(
                &store,
                "sess-a",
                turn,
                "ctx",
                vec!["a".to_string()],
                "chosen",
                "reason",
                true,
            );
        }
        for turn in 0..3usize {
            log_scheduler_dispatch(
                &store,
                "sess-b",
                turn,
                "ctx",
                vec!["b".to_string()],
                "chosen",
                "reason",
                false,
            );
        }

        let replay = store.replay_recent_from_disk("sess-a", 3);
        assert_eq!(replay.len(), 3);
        assert!(replay.iter().all(|item| item.session_id == "sess-a"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_clear_persist_path_disables_disk_write() {
        let store = DecisionLogStore::new(100);
        let path = temp_log_path("decision_log_disabled");
        store.set_persist_path(&path);
        store.clear_persist_path();

        log_scheduler_dispatch(
            &store,
            "sess-a",
            0,
            "ctx",
            vec!["a".to_string()],
            "chosen",
            "reason",
            true,
        );

        assert!(!path.exists());
    }

    #[test]
    fn test_compact_persist_file_retains_recent_tail() {
        let store = DecisionLogStore::new(100);
        let path = temp_log_path("decision_log_compact");

        // Write more lines than the retention cap directly, then trigger compaction manually.
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let total = DECISION_LOG_RETAIN_LINES + 500;
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap();
            for i in 0..total {
                let log = DecisionLog {
                    timestamp: i as i64,
                    session_id: "sess".to_string(),
                    turn_id: i,
                    decision_type: DecisionType::SchedulerDispatch,
                    context: "ctx".to_string(),
                    alternatives_considered: vec![],
                    chosen_option: "c".to_string(),
                    reasoning: "r".to_string(),
                    confidence: None,
                    outcome: None,
                    execution_time_ms: None,
                };
                writeln!(file, "{}", serde_json::to_string(&log).unwrap()).unwrap();
            }
        }

        store.compact_persist_file(&path);

        let reader = BufReader::new(std::fs::File::open(&path).unwrap());
        let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();
        assert_eq!(lines.len(), DECISION_LOG_RETAIN_LINES);
        // The latest tail must be retained: the last line has turn_id == total - 1.
        let last: DecisionLog = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last.turn_id, total - 1);
        // The oldest retained line should be total - RETAIN_LINES.
        let first: DecisionLog = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(first.turn_id, total - DECISION_LOG_RETAIN_LINES);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_log_memory_save_assessment_records_structured_reasoning() {
        let store = DecisionLogStore::new(100);
        let assessment = crate::ai::driver::reflection::LearningNoteAssessment {
            actionable: false,
            specific: false,
            generalizable: false,
            score: 0,
            high_quality: false,
            char_count: 10,
            word_count: 2,
            nonempty_lines: 1,
            unique_token_ratio: 1.0,
            directive_signals: 0,
            code_signals: 0,
            artifact_signals: 0,
            abstraction_signals: 0,
            condition_signals: 0,
            one_off_signals: 0,
        };

        log_memory_save_assessment(
            &store,
            "sess-test",
            7,
            "common_sense",
            "self_note",
            "be careful",
            &assessment,
            true,
        );

        let recent = store.recent(1);
        assert_eq!(recent[0].decision_type, DecisionType::MemorySave);
        assert_eq!(recent[0].chosen_option, "self_note");
        assert!(recent[0].reasoning.contains("\"score\":0"));
        assert!(recent[0].context.contains("requested_category"));
    }
}

// Global singleton access
use std::sync::OnceLock;

static DECISION_LOG_STORE: OnceLock<DecisionLogStore> = OnceLock::new();

/// Get the global decision log store
pub fn get_decision_log_store() -> &'static DecisionLogStore {
    DECISION_LOG_STORE.get_or_init(|| DecisionLogStore::new(1000))
}

/// Initialize the decision log store (optional, for a custom capacity)
pub fn init_decision_log_store(capacity: usize) -> &'static DecisionLogStore {
    DECISION_LOG_STORE.get_or_init(|| DecisionLogStore::new(capacity))
}

pub fn init_decision_log_store_with_path<P: AsRef<Path>>(
    capacity: usize,
    path: P,
) -> &'static DecisionLogStore {
    let store = DECISION_LOG_STORE.get_or_init(|| DecisionLogStore::new(capacity));
    store.set_persist_path(path);
    store
}

pub fn set_decision_log_persist_path<P: AsRef<Path>>(path: P) {
    get_decision_log_store().set_persist_path(path);
}

pub fn clear_decision_log_persist_path() {
    get_decision_log_store().clear_persist_path();
}
