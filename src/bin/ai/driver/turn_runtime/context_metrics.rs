//! Content-free projection measurements. Character budgets are not provider token usage.

use serde::{Deserialize, Serialize};

use crate::ai::{
    driver::decision_log::{DecisionLog, DecisionType, get_decision_log_store},
    history::{Message, ROLE_INTERNAL_NOTE, message_billable_chars},
};

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ContextSizeBreakdown {
    pub(super) messages: usize,
    pub(super) total_chars: usize,
    system_chars: usize,
    user_chars: usize,
    assistant_chars: usize,
    tool_result_chars: usize,
    summary_chars: usize,
    task_evidence_chars: usize,
    recovery_metadata_chars: usize,
    other_chars: usize,
}

impl ContextSizeBreakdown {
    pub(super) fn measure(messages: &[Message]) -> Self {
        let mut result = Self {
            messages: messages.len(),
            ..Self::default()
        };
        for message in messages {
            let chars = message_billable_chars(message);
            let text = message.content.as_str().unwrap_or_default().trim_start();
            let bucket = match message.role.as_str() {
                "assistant" | ROLE_INTERNAL_NOTE if text.starts_with("[task-evidence-ledger]") => {
                    &mut result.task_evidence_chars
                }
                "system" => &mut result.system_chars,
                "user" => &mut result.user_chars,
                "assistant" => &mut result.assistant_chars,
                "tool" => &mut result.tool_result_chars,
                ROLE_INTERNAL_NOTE
                    if crate::ai::history::compress::automatic_summary_body(text).is_some() =>
                {
                    &mut result.summary_chars
                }
                ROLE_INTERNAL_NOTE
                    if crate::ai::history::compress::is_context_checkpoint_marker(message)
                        || crate::ai::history::compress::is_compressed_tool_evidence_note(
                            message,
                        )
                        || crate::ai::history::compress::is_archive_note_text(text)
                        || text.starts_with(
                            crate::ai::history::compress::QUERY_MEMORY_INDEX_PREFIX,
                        ) =>
                {
                    &mut result.recovery_metadata_chars
                }
                _ => &mut result.other_chars,
            };
            *bucket = bucket.saturating_add(chars);
            result.total_chars = result.total_chars.saturating_add(chars);
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ContextRequestMetrics {
    pub(super) before: ContextSizeBreakdown,
    pub(super) after: ContextSizeBreakdown,
    pub(super) provider_overflows: usize,
    pub(super) overflow_retries: usize,
    /// HTTP response received, not semantic task success or completed streaming.
    pub(super) response_received: bool,
    pub(super) elapsed_ms: u64,
}

pub(super) fn record_context_request(
    session_id: &str,
    iteration: usize,
    model: &str,
    metrics: ContextRequestMetrics,
) {
    get_decision_log_store().log(context_decision(session_id, iteration, model, &metrics));
}

fn context_decision(
    session_id: &str,
    iteration: usize,
    model: &str,
    metrics: &ContextRequestMetrics,
) -> DecisionLog {
    DecisionLog {
        timestamp: 0,
        session_id: session_id.to_string(),
        turn_id: iteration,
        decision_type: DecisionType::ContextProjection,
        context: serde_json::json!({
            "schema": "context-projection-v1", "model": model, "metrics": metrics,
        })
        .to_string(),
        alternatives_considered: Vec::new(),
        chosen_option: "request_projection".to_string(),
        reasoning:
            "Projection-only character accounting; no message bodies or task-success inference."
                .to_string(),
        confidence: None,
        outcome: None,
        execution_time_ms: Some(metrics.elapsed_ms),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(role: &str, content: serde_json::Value) -> Message {
        Message {
            role: role.into(),
            content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn context_replay_accounting_is_conservative_and_does_not_mutate_history() {
        let canonical = vec![
            message("system", json!("system instruction")),
            message(
                "user",
                json!([{"type":"image_url", "image_url":{"url": "data:image/png;base64,AAAA"}}, {"type":"text", "text":"private question"}]),
            ),
            message("assistant", json!("analysis result")),
            message("tool", json!("secret source body")),
            message(
                ROLE_INTERNAL_NOTE,
                json!("[mid-turn-summary] a derived conclusion"),
            ),
            message("assistant", json!("[task-evidence-ledger] task-1")),
        ];
        let snapshot = serde_json::to_value(&canonical).unwrap();
        let sizes = ContextSizeBreakdown::measure(&canonical);
        assert_eq!(
            sizes.total_chars,
            crate::ai::history::messages_total_chars_pub(&canonical)
        );
        assert_eq!(
            sizes.total_chars,
            sizes.system_chars
                + sizes.user_chars
                + sizes.assistant_chars
                + sizes.tool_result_chars
                + sizes.summary_chars
                + sizes.task_evidence_chars
                + sizes.recovery_metadata_chars
                + sizes.other_chars
        );
        assert!(sizes.summary_chars > 0 && sizes.task_evidence_chars > 0);
        assert_eq!(serde_json::to_value(&canonical).unwrap(), snapshot);
        assert!(
            !serde_json::to_string(&sizes)
                .unwrap()
                .contains("secret source")
        );
    }

    #[test]
    fn context_replay_metrics_roundtrip_keeps_failure_and_growth_visible() {
        let metrics = ContextRequestMetrics {
            before: ContextSizeBreakdown {
                total_chars: 10,
                ..Default::default()
            },
            after: ContextSizeBreakdown {
                total_chars: 20,
                ..Default::default()
            },
            provider_overflows: 2,
            overflow_retries: 1,
            response_received: false,
            elapsed_ms: 50,
        };
        let log = context_decision("session", 3, "model", &metrics);
        let replay: DecisionLog =
            serde_json::from_str(&serde_json::to_string(&log).unwrap()).unwrap();
        let body: serde_json::Value = serde_json::from_str(&replay.context).unwrap();
        let decoded: ContextRequestMetrics =
            serde_json::from_value(body["metrics"].clone()).unwrap();
        assert_eq!(decoded, metrics);
        assert!(replay.outcome.is_none());
        assert!(decoded.after.total_chars > decoded.before.total_chars);
    }

    #[test]
    fn context_replay_telemetry_does_not_change_decision_quality_or_feedback() {
        use crate::ai::driver::decision_log::{DecisionLogStore, Outcome, UserFeedback};
        let store = DecisionLogStore::new(100);
        let metrics = ContextRequestMetrics {
            before: ContextSizeBreakdown::default(),
            after: ContextSizeBreakdown::default(),
            provider_overflows: 0,
            overflow_retries: 0,
            response_received: true,
            elapsed_ms: 500,
        };
        store.log(context_decision("session", 1, "model", &metrics));
        let mut decision = context_decision("session", 1, "model", &metrics);
        decision.decision_type = DecisionType::ToolInvocation;
        decision.execution_time_ms = Some(10);
        store.log(decision);
        store.update_outcome(
            "session",
            1,
            Outcome {
                success: true,
                message: "tool completed".into(),
                user_feedback: None,
            },
        );
        store.add_feedback("session", 1, UserFeedback::Positive);
        let stats = store.stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.success_rate, 1.0);
        assert_eq!(stats.avg_execution_time_ms, 10.0);
        let logs = store.recent(2);
        assert_eq!(logs.len(), 1);
        assert_eq!(
            logs[0].outcome.as_ref().unwrap().user_feedback,
            Some(UserFeedback::Positive)
        );
        assert!(
            store.by_type(&DecisionType::ContextProjection)[0]
                .outcome
                .is_none()
        );
    }

    #[test]
    fn context_replay_telemetry_retention_cannot_evict_decisions() {
        use crate::ai::driver::decision_log::DecisionLogStore;
        let store = DecisionLogStore::new(10);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "context_metrics_{}_{}.jsonl",
            std::process::id(),
            nonce,
        ));
        store.set_persist_path(&path);
        let metrics = ContextRequestMetrics {
            before: ContextSizeBreakdown::default(),
            after: ContextSizeBreakdown::default(),
            provider_overflows: 0,
            overflow_retries: 0,
            response_received: true,
            elapsed_ms: 1,
        };
        for turn in 0..10 {
            let mut decision = context_decision("session", turn, "model", &metrics);
            decision.decision_type = DecisionType::ToolInvocation;
            store.log(decision);
        }
        let original = serde_json::to_value(store.recent(10)).unwrap();
        let original_disk = std::fs::read(&path).unwrap();
        for turn in 0..30 {
            store.log(context_decision("session", turn, "model", &metrics));
        }
        assert_eq!(serde_json::to_value(store.recent(10)).unwrap(), original);
        assert_eq!(store.stats().total, 10);
        assert_eq!(std::fs::read(&path).unwrap(), original_disk);
        assert_eq!(store.replay_recent_from_disk("session", 10).len(), 10);
        assert_eq!(store.by_type(&DecisionType::ContextProjection).len(), 10);
        assert_eq!(
            std::fs::read_to_string(path.with_extension("context.jsonl"))
                .unwrap()
                .lines()
                .count(),
            30
        );
        std::fs::remove_file(path.with_extension("context.jsonl")).unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
