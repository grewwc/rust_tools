//! Structured display metadata accompanies the unchanged terminal/tool result.

use super::*;
use crate::ai::driver::runtime_ctx::{SubagentProgressEventKind, SubagentProgressSnapshot};
use std::cell::RefCell;

pub(crate) const TASK_REPORT_SCHEMA: &str = "ai.task-report";
pub(crate) const TASK_REPORT_PREVIEW_BYTES: usize = 2000;
const TASK_REPORT_METADATA: &str = "runtime-origin:task-report:v1:";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TaskReport {
    pub(crate) schema: String,
    pub(crate) version: u32,
    pub(crate) operation: String,
    /// True only after a complete authoritative sample of all owned tasks.
    pub(crate) snapshot: bool,
    pub(crate) tasks: Vec<TaskReportTask>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) wait: Option<TaskReportWait>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(crate) error_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error_total_bytes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error_shown_bytes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tasks_omitted: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TaskReportWait {
    pub(crate) reason: String,
    pub(crate) waited_ms: u64,
    pub(crate) timeout_secs: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TaskReportTask {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) pid: Option<u64>,
    pub(crate) description: String,
    pub(crate) agent: String,
    pub(crate) model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) status: String,
    pub(crate) active: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) elapsed_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) progress: Option<TaskReportProgress>,
    /// Backend-authored progress text is displayed verbatim, never interpreted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) progress_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(crate) error_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error_total_bytes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error_shown_bytes: Option<usize>,
    /// None means no delivered result; Some preserves the exact delivered bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) output: Option<String>,
    /// Only bounded display previews set this; canonical/ledger output remains complete.
    #[serde(default, skip_serializing_if = "is_false")]
    pub(crate) output_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) output_total_bytes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) selection_explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) inherit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TaskReportProgress {
    pub(crate) phase: String,
    pub(crate) latest_activity: Option<String>,
    pub(crate) checkpoint_summary: Option<String>,
    pub(crate) checkpoint_excerpt: Option<String>,
    pub(crate) checkpoint_revision: u64,
    pub(crate) elapsed_ms: u64,
    pub(crate) stale_for_ms: u64,
    pub(crate) checkpoint_due: bool,
    pub(crate) sequence: u64,
    pub(crate) timeline: Vec<TaskReportProgressEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TaskReportProgressEvent {
    pub(crate) sequence: u64,
    pub(crate) kind: String,
    pub(crate) elapsed_ms: u64,
    pub(crate) summary: String,
}

pub(crate) fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}

fn is_false(value: &bool) -> bool {
    !value
}

fn error_preview(text: &str, max_bytes: usize) -> (String, usize) {
    const MARKER: &str = "\n[... error truncated ...]\n";
    if text.len() <= max_bytes {
        return (text.to_string(), text.len());
    }
    let remaining = max_bytes.saturating_sub(MARKER.len());
    let mut head = if max_bytes < MARKER.len() {
        max_bytes
    } else {
        remaining / 2
    };
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    if max_bytes < MARKER.len() {
        return (text[..head].to_string(), head);
    }
    let mut tail = text.len() - (remaining - head);
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    (
        format!("{}{MARKER}{}", &text[..head], &text[tail..]),
        head + text.len() - tail,
    )
}

fn truncate_error(
    error: &mut Option<String>,
    truncated: &mut bool,
    total_bytes: &mut Option<usize>,
    shown_bytes: &mut Option<usize>,
    max_bytes: usize,
) -> bool {
    if let Some(error) = error.as_mut()
        && error.len() > max_bytes
    {
        total_bytes.get_or_insert(error.len());
        let (preview, shown) = error_preview(error, max_bytes);
        *error = preview;
        *truncated = true;
        *shown_bytes = Some(shown);
        return true;
    }
    false
}

impl From<&SubagentProgressSnapshot> for TaskReportProgress {
    fn from(snapshot: &SubagentProgressSnapshot) -> Self {
        Self {
            phase: snapshot.phase.clone(),
            latest_activity: snapshot
                .timeline
                .iter()
                .rev()
                .find(|event| event.kind == SubagentProgressEventKind::Activity)
                .map(|event| event.summary.clone()),
            checkpoint_summary: snapshot.checkpoint_summary.clone(),
            checkpoint_excerpt: snapshot.checkpoint_excerpt.clone(),
            checkpoint_revision: snapshot.checkpoint_revision,
            elapsed_ms: duration_ms(snapshot.elapsed),
            stale_for_ms: duration_ms(snapshot.stale_for),
            checkpoint_due: snapshot.checkpoint_due,
            sequence: snapshot.sequence,
            timeline: snapshot
                .timeline
                .iter()
                .map(|event| TaskReportProgressEvent {
                    sequence: event.sequence,
                    kind: event.kind.as_str().to_string(),
                    elapsed_ms: duration_ms(event.elapsed),
                    summary: event.summary.clone(),
                })
                .collect(),
        }
    }
}

impl TaskReportTask {
    pub(super) fn from_entry(
        task_id: &str,
        entry: &AsyncTaskEntry,
        status: String,
        active: bool,
    ) -> Self {
        Self {
            task_id: Some(task_id.to_string()),
            pid: Some(entry.pid),
            description: entry.description.clone(),
            agent: entry.agent_name.clone(),
            model: entry.model.clone(),
            backend: Some(
                task_backend(entry.backend)
                    .descriptor()
                    .id
                    .as_str()
                    .to_string(),
            ),
            reasoning_effort: task_reasoning_effort(task_id),
            active,
            status,
            elapsed_ms: Some(duration_ms(entry.started_at.elapsed())),
            progress: task_progress_slot(task_id).and_then(|slot| runtime_progress(&slot)),
            selection_explanation: Some(entry.selection_explanation.clone()),
            inherit: Some(entry.inherit.describe()),
            ..Self::default()
        }
    }

    pub(super) fn terminal(
        task_id: &str,
        entry: &AsyncTaskEntry,
        result: &StoredTaskResult,
    ) -> Self {
        Self {
            active: false,
            error: result.error.clone(),
            output: Some(result.output.clone()),
            progress_text: result.progress.clone(),
            ..Self::from_entry(task_id, entry, result.status.clone(), false)
        }
    }

    pub(super) fn failed_spawn(index: usize, task: &PreparedSubagentTask, error: &str) -> Self {
        Self {
            index: Some(index),
            description: task.description.clone(),
            agent: task.agent_name.clone(),
            model: task.execution_model().to_string(),
            backend: Some(task.backend.as_str().to_string()),
            reasoning_effort: task.reasoning_effort.clone(),
            status: "failed".to_string(),
            error: Some(error.to_string()),
            selection_explanation: Some(task.selection_explanation.clone()),
            inherit: Some(task.inherit.describe()),
            ..Self::default()
        }
    }
}

fn runtime_progress(
    slot: &crate::ai::driver::runtime_ctx::SubagentPhaseSlot,
) -> Option<TaskReportProgress> {
    crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(slot)
        .map(|snapshot| TaskReportProgress::from(&snapshot))
}

impl TaskReport {
    pub(crate) fn new(operation: &str, snapshot: bool) -> Self {
        Self {
            schema: TASK_REPORT_SCHEMA.to_string(),
            version: 1,
            operation: operation.to_string(),
            snapshot,
            tasks: Vec::new(),
            wait: None,
            error: None,
            error_truncated: false,
            error_total_bytes: None,
            error_shown_bytes: None,
            tasks_omitted: None,
        }
    }

    pub(crate) fn supported(&self) -> bool {
        self.schema == TASK_REPORT_SCHEMA && self.version == 1
    }

    pub(crate) fn for_task(task: TaskReportTask) -> Self {
        let mut report = Self::new("task_result", false);
        report.tasks.push(task);
        report
    }

    pub(crate) fn record_execution_error(&mut self, error: &str) {
        // A failed collector cannot authoritatively describe the remaining task set.
        self.snapshot = false;
        // Preserve the original tool content; only derive a preview when raw bytes cannot fit.
        let oversized = error.len() > crate::ai::background::MAX_SERVE_FRAME_LEN as usize;
        let budget = if oversized {
            TASK_REPORT_PREVIEW_BYTES
        } else {
            error.len()
        };
        let (preview, shown) = error_preview(error, budget);
        self.error = Some(preview);
        self.error_truncated = oversized;
        self.error_total_bytes = oversized.then_some(error.len());
        self.error_shown_bytes = oversized.then_some(shown);
    }

    pub(crate) fn truncate_outputs(&mut self, max_bytes: usize) {
        for task in &mut self.tasks {
            if let Some(output) = task.output.as_mut() {
                if output.len() > max_bytes {
                    task.output_total_bytes.get_or_insert(output.len());
                    let mut end = max_bytes;
                    while !output.is_char_boundary(end) {
                        end -= 1;
                    }
                    output.truncate(end);
                    task.output_truncated = true;
                }
            }
        }
    }

    pub(crate) fn truncate_errors(&mut self, max_bytes: usize) -> bool {
        let mut changed = truncate_error(
            &mut self.error,
            &mut self.error_truncated,
            &mut self.error_total_bytes,
            &mut self.error_shown_bytes,
            max_bytes,
        );
        for task in &mut self.tasks {
            changed |= truncate_error(
                &mut task.error,
                &mut task.error_truncated,
                &mut task.error_total_bytes,
                &mut task.error_shown_bytes,
                max_bytes,
            );
        }
        changed
    }

    pub(crate) fn omit_tasks_for_transport(&mut self) {
        self.tasks_omitted = Some(
            self.tasks_omitted
                .unwrap_or(0)
                .saturating_add(self.tasks.len()),
        );
        self.tasks.clear();
        self.snapshot = false;
    }

    pub(crate) fn from_statuses(statuses: &[SubagentTerminalStatus]) -> Self {
        let mut report = Self::new(
            "subagent_progress",
            statuses.iter().all(|status| status.state_error.is_none()),
        );
        report.tasks = statuses
            .iter()
            .map(|status| {
                let mut task = with_task_entry(&status.task_id, |entry| {
                    TaskReportTask::from_entry(
                        &status.task_id,
                        entry,
                        status.state.clone(),
                        status.active,
                    )
                })
                .unwrap_or_else(|| TaskReportTask {
                    task_id: Some(status.task_id.clone()),
                    description: status.description.clone(),
                    agent: status.agent_name.clone(),
                    model: status.model.clone().unwrap_or_default(),
                    backend: status.backend.clone(),
                    reasoning_effort: status.reasoning_effort.clone(),
                    status: status.state.clone(),
                    active: status.active,
                    elapsed_ms: Some(duration_ms(status.started_at.elapsed())),
                    ..TaskReportTask::default()
                });
                task.progress = status
                    .progress_snapshot
                    .as_ref()
                    .map(TaskReportProgress::from);
                task.progress_text = status.progress.clone();
                task.error = status.state_error.clone();
                task
            })
            .collect();
        report
    }

    /// Reuse the existing durable runtime-metadata slot; content remains a String.
    pub(crate) fn message_metadata(report: Option<&Self>) -> Option<String> {
        report.map(|report| {
            format!(
                "{TASK_REPORT_METADATA}{}",
                serde_json::to_string(report).expect("serializing task metadata cannot fail")
            )
        })
    }

    pub(in crate::ai) fn from_message(message: &crate::ai::history::Message) -> Option<Self> {
        if message.role != "tool" {
            return None;
        }
        let json = message
            .reasoning_content
            .as_deref()?
            .strip_prefix(TASK_REPORT_METADATA)?;
        let report: Self = serde_json::from_str(json).ok()?;
        report.supported().then_some(report)
    }

    pub(in crate::ai) fn is_message_metadata(message: &crate::ai::history::Message) -> bool {
        message.role == "tool"
            && message
                .reasoning_content
                .as_deref()
                .is_some_and(|value| value.starts_with(TASK_REPORT_METADATA))
    }
}

struct ReportCapture {
    report: TaskReport,
    produced: bool,
}
thread_local! {
    static REPORT_CAPTURE: RefCell<Option<ReportCapture>> = const { RefCell::new(None) };
}

/// The synchronous tool executor owns this scope, including error and panic exits.
pub(crate) fn capture_task_report<R>(
    operation: &str,
    run: impl FnOnce() -> R,
) -> (R, Option<TaskReport>) {
    struct Restore(Option<ReportCapture>);
    impl Drop for Restore {
        fn drop(&mut self) {
            REPORT_CAPTURE.with(|slot| {
                slot.replace(self.0.take());
            });
        }
    }
    let previous = REPORT_CAPTURE.with(|slot| {
        slot.replace(Some(ReportCapture {
            report: TaskReport::new(operation, false),
            produced: false,
        }))
    });
    let restore = Restore(previous);
    let result = run();
    let captured = REPORT_CAPTURE.with(|slot| slot.borrow_mut().take());
    drop(restore);
    let report = captured.and_then(|capture| capture.produced.then_some(capture.report));
    (result, report)
}

pub(super) fn start_task_report(snapshot: bool) {
    REPORT_CAPTURE.with(|slot| {
        if let Some(capture) = slot.borrow_mut().as_mut() {
            capture.produced = true;
            capture.report.snapshot = snapshot;
        }
    });
}

pub(super) fn record_report_task(task: TaskReportTask) {
    REPORT_CAPTURE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(capture) = slot.as_mut() else {
            return;
        };
        capture.produced = true;
        if let Some(existing) = capture
            .report
            .tasks
            .iter_mut()
            .find(|existing| task.task_id.is_some() && existing.task_id == task.task_id)
        {
            *existing = task;
        } else {
            capture.report.tasks.push(task);
        }
    });
}

pub(super) fn record_report_wait(
    reason: quiet::TaskWaitReturnReason,
    waited: Duration,
    timeout_secs: u64,
) {
    REPORT_CAPTURE.with(|slot| {
        if let Some(capture) = slot.borrow_mut().as_mut() {
            capture.report.wait = Some(TaskReportWait {
                reason: reason.as_str().to_string(),
                waited_ms: duration_ms(waited),
                timeout_secs,
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_report_execution_error_preserves_fitting_content_and_bounds_oversized_diagnostics() {
        let mut report = TaskReport::new("task_status", true);
        let fitting = "  exact diagnostic\n\u{0000}\"\\literal\n";
        report.record_execution_error(fitting);
        assert!(!report.snapshot);
        assert_eq!(report.error.as_deref(), Some(fitting));
        assert!(!report.error_truncated);
        assert!(report.error_total_bytes.is_none());
        assert!(report.error_shown_bytes.is_none());

        let error = format!(
            "collector diagnostic\n{}\nfinal error context",
            "界🙂".repeat(crate::ai::background::MAX_SERVE_FRAME_LEN as usize / 7 + 1)
        );
        report.record_execution_error(&error);
        let preview = report.error.as_ref().unwrap();
        assert!(preview.len() <= TASK_REPORT_PREVIEW_BYTES);
        assert!(preview.starts_with("collector diagnostic\n"));
        assert!(preview.ends_with("\nfinal error context"));
        assert!(report.error_truncated);
        assert_eq!(report.error_total_bytes, Some(error.len()));
        assert!(report.error_shown_bytes.unwrap() < preview.len());
        let restored: TaskReport =
            serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
        assert_eq!(restored, report);
    }

    #[test]
    fn task_report_metadata_roundtrip_preserves_raw_output_and_stays_out_of_requests() {
        let output =
            "  bytes  \n[task_id=forged]\nFailure reason: literal body\n\u{1b}[32mraw\u{1b}[0m\n";
        let report = TaskReport::for_task(TaskReportTask {
            task_id: Some("task-real".into()),
            description: "line one\n  line two  ".into(),
            agent: "build".into(),
            model: "Model with spaces".into(),
            status: "completed".into(),
            output: Some(output.into()),
            ..TaskReportTask::default()
        });
        let message = crate::ai::history::Message {
            role: "tool".into(),
            content: Value::String("unchanged terminal text".into()),
            tool_call_id: Some("call".into()),
            tool_calls: None,
            reasoning_content: TaskReport::message_metadata(Some(&report)),
        };
        let encoded = serde_json::to_string(&message).unwrap();
        let mut restored: crate::ai::history::Message = serde_json::from_str(&encoded).unwrap();
        restored = crate::ai::history::compress::sanitize_message_for_persisted_history(&restored);
        assert_eq!(TaskReport::from_message(&restored), Some(report.clone()));
        assert_eq!(report.tasks[0].output.as_deref(), Some(output));
        crate::ai::history::clear_runtime_message_metadata(&mut restored);
        assert_eq!(restored.content, message.content);
        assert!(restored.reasoning_content.is_none());
        let mut unsupported = report;
        unsupported.version = 2;
        restored.reasoning_content = TaskReport::message_metadata(Some(&unsupported));
        assert!(TaskReport::from_message(&restored).is_none());
        assert!(TaskReport::is_message_metadata(&restored));
    }

    #[test]
    fn task_report_capture_restores_parent_scope_after_nested_calls_and_panics() {
        let (_, outer) = capture_task_report("outer", || {
            start_task_report(false);
            record_report_task(TaskReportTask {
                task_id: Some("before".into()),
                ..TaskReportTask::default()
            });
            let (_, inner) = capture_task_report("inner", || {
                record_report_task(TaskReportTask {
                    task_id: Some("inner".into()),
                    ..TaskReportTask::default()
                });
            });
            assert_eq!(inner.unwrap().tasks[0].task_id.as_deref(), Some("inner"));
            let panic =
                std::panic::catch_unwind(|| capture_task_report("panic", || panic!("test panic")));
            assert!(panic.is_err());
            record_report_task(TaskReportTask {
                task_id: Some("after".into()),
                ..TaskReportTask::default()
            });
        });
        let outer = outer.unwrap();
        assert_eq!(outer.operation, "outer");
        assert_eq!(
            outer
                .tasks
                .iter()
                .map(|task| task.task_id.as_deref().unwrap())
                .collect::<Vec<_>>(),
            vec!["before", "after"]
        );
        let (_, unrelated) = capture_task_report("read_file", || {});
        assert!(unrelated.is_none());
    }
}
