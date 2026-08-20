//! plan 状态模型：纯数据层（无 I/O、无渲染）。
//!
//! `StepStatus` / `PlanStepState` / `StepTransition` / `PlanState` 只描述状态与状态转换
//! （build / apply_update / 计数）；持久化见 `store`，文本渲染见 `render`。

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 终态被改写为其他状态时返回提示文案；pending/running 之间的双向调整与
/// pending→skipped 放弃属于合法流，不提示。
fn transition_warning(step: u64, from: StepStatus, to: StepStatus) -> Option<String> {
    if from != to && from.is_terminal() {
        Some(format!(
            "Warning: step {step} was marked {}, and is now {} — this overrides a terminal status (only valid for a deliberate redo/retry/reopen).",
            from.name(),
            to.name()
        ))
    } else {
        None
    }
}

/// plan 步骤状态。serde 使用小写字符串，与 `plan_update` 的 `status` 参数一致。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum StepStatus {
    #[default]
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
}

impl StepStatus {
    pub(crate) const ALL: [&'static str; 5] = ["pending", "running", "done", "failed", "skipped"];

    pub(crate) fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "pending" => Self::Pending,
            "running" => Self::Running,
            "done" => Self::Done,
            "failed" => Self::Failed,
            "skipped" => Self::Skipped,
            _ => return None,
        })
    }

    /// 状态名的短字符串（与 plan_update 参数一致，用于变更提示文案）。
    fn name(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }

    /// 是否为终态：进入 done/failed/skipped 后再次改写成其他状态，属于"覆盖终态"
    /// （重做/重试/重开），`apply_update` 会在该场景返回 warning 提示。
    fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Skipped)
    }

}


/// 单个步骤的持久化状态。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PlanStepState {
    pub(crate) step: u64,
    pub(crate) action: String,
    #[serde(default)]
    pub(crate) reason: String,
    #[serde(default)]
    pub(crate) tool: String,
    #[serde(default)]
    pub(crate) delegate: bool,
    #[serde(default)]
    pub(crate) parallelizable: bool,
    #[serde(default)]
    pub(crate) status: StepStatus,
    /// `plan_update` 附带的可选说明（失败原因、跳过理由等）。
    #[serde(default)]
    pub(crate) note: Option<String>,
}

/// `plan_update` 一次状态变更的元信息；`warning` 在覆盖既有终态时给出提示（见
/// `transition_warning`），由 `execute_plan_update` 追加到结果回显中，不持久化。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StepTransition {
    pub(crate) from: StepStatus,
    pub(crate) to: StepStatus,
    pub(crate) warning: Option<String>,
}

/// 会话内当前活跃计划的完整状态。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PlanState {
    pub(crate) schema_version: u32,
    pub(crate) summary: String,
    pub(crate) steps: Vec<PlanStepState>,
    pub(crate) updated_at_ms: u64,
}


impl PlanState {
    /// 由 `plan` 工具原始参数与可选的上一个状态构建；按全字段一致匹配保留既有步骤
    /// 状态，任一字段有变（动作/工具/理由/执行属性）都视为新步骤（回退 pending）。
    pub(crate) fn build(
        summary: &str,
        raw_steps: &[Value],
        previous: Option<&PlanState>,
    ) -> Result<Self, String> {
        let steps = parse_step_specs(raw_steps)?;
        let steps = match previous {
            Some(prev) => steps
                .into_iter()
                .map(|mut spec| {
                    if let Some(carried) = prev
                        .steps
                        .iter()
                        .find(|s| {
                            s.step == spec.step
                                && s.action == spec.action
                                && s.tool == spec.tool
                                && s.reason == spec.reason
                                && s.delegate == spec.delegate
                                && s.parallelizable == spec.parallelizable
                        })
                    {
                        spec.status = carried.status;
                        spec.note = carried.note.clone();
                    }
                    spec
                })
                .collect(),
            None => steps,
        };
        Ok(Self {
            schema_version: 1,
            summary: summary.to_string(),
            steps,
            updated_at_ms: now_ms(),
        })
    }

    /// 更新指定步骤的状态；步骤不存在时返回错误且不改动任何状态。
    ///
    /// 状态转移保持宽松：合法流包括 failed→running 重试、done→running 返工、
    /// running→pending 撤回、pending→skipped 放弃；仅当新状态覆盖既有终态
    /// （done/failed/skipped）时，返回值的 `warning` 携带一次提示，由调用方决定呈现。
    pub(crate) fn apply_update(
        &mut self,
        step: u64,
        status: StepStatus,
        note: Option<String>,
    ) -> Result<StepTransition, String> {
        let target = self
            .steps
            .iter_mut()
            .find(|s| s.step == step)
            .ok_or_else(|| format!("Step {step} not found in the active plan."))?;
        let from = target.status;
        target.status = status;
        target.note = note;
        self.updated_at_ms = now_ms();
        Ok(StepTransition {
            from,
            to: status,
            warning: transition_warning(step, from, status),
        })
    }

    pub(crate) fn done_count(&self) -> usize {
        self.steps.iter().filter(|s| s.status == StepStatus::Done).count()
    }

    pub(crate) fn running_count(&self) -> usize {
        self.steps.iter().filter(|s| s.status == StepStatus::Running).count()
    }

    pub(crate) fn failed_count(&self) -> usize {
        self.steps.iter().filter(|s| s.status == StepStatus::Failed).count()
    }

    pub(crate) fn skipped_count(&self) -> usize {
        self.steps.iter().filter(|s| s.status == StepStatus::Skipped).count()
    }

    /// 是否存在非 pending 步骤（决定是否渲染进度行）。
    pub(crate) fn has_progress(&self) -> bool {
        self.steps
            .iter()
            .any(|s| s.status != StepStatus::Pending)
    }
}

fn parse_step_specs(raw_steps: &[Value]) -> Result<Vec<PlanStepState>, String> {
    let mut out = Vec::with_capacity(raw_steps.len());
    for step in raw_steps {
        let step_obj = step
            .as_object()
            .ok_or("Each step must be a JSON object.")?;
        let step_num = step_obj
            .get("step")
            .and_then(|v| v.as_u64())
            .ok_or("Each step must have an integer 'step' field.")?;
        let action = step_obj
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or("Each step must have an 'action' string field.")?;
        let reason = step_obj.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        // 与 plan_tools 的 execute_plan 保持一致：tool 缺省时渲染为 "unspecified"。
        let tool = step_obj.get("tool").and_then(|v| v.as_str()).unwrap_or("unspecified");
        let delegate = step_obj.get("delegate").and_then(|v| v.as_bool()).unwrap_or(false);
        let parallelizable = step_obj.get("parallelizable").and_then(|v| v.as_bool()).unwrap_or(false);
        out.push(PlanStepState {
            step: step_num,
            action: action.to_string(),
            reason: reason.to_string(),
            tool: tool.to_string(),
            delegate,
            parallelizable,
            status: StepStatus::Pending,
            note: None,
        });
    }
    Ok(out)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_status_parse() {
        assert_eq!(StepStatus::parse("done"), Some(StepStatus::Done));
        assert_eq!(StepStatus::parse("pending"), Some(StepStatus::Pending));
        assert_eq!(StepStatus::parse("bogus"), None);
    }

    #[test]
    fn test_build_keeps_previous_status_by_action() {
        let raw = serde_json::json!([
            { "step": 1, "action": "Read file", "tool": "read_file" },
            { "step": 2, "action": "Patch file", "tool": "apply_patch" }
        ]);
        let steps = raw.as_array().unwrap();
        let first = PlanState::build("T", steps, None).unwrap();
        assert_eq!(first.steps[0].status, StepStatus::Pending);

        let mut updated = first.clone();
        updated
            .apply_update(1, StepStatus::Done, Some("verified".to_string()))
            .unwrap();

        // 全字段一致的计划重发：保留既有状态与 note。
        let again = PlanState::build("T", steps, Some(&updated)).unwrap();
        assert_eq!(again.steps[0].status, StepStatus::Done);
        assert_eq!(again.steps[0].note.as_deref(), Some("verified"));
        // 动作变更的步骤视为新步骤，回退 pending。
        let changed = serde_json::json!([
            { "step": 2, "action": "Rewrite file", "tool": "apply_patch" }
        ]);
        let rebuilt = PlanState::build("T", changed.as_array().unwrap(), Some(&updated)).unwrap();
        assert_eq!(rebuilt.steps[0].status, StepStatus::Pending);
    }

    #[test]
    fn test_build_resets_when_execution_fields_change() {
        // 重规划仅改执行属性/理由时不再继承旧终态：任何字段变化都视为新步骤。
        let raw = serde_json::json!([{
            "step": 1, "action": "Read file", "tool": "read_file",
            "reason": "entry point", "delegate": true, "parallelizable": true
        }]);
        let steps = raw.as_array().unwrap();
        let mut first = PlanState::build("T", steps, None).unwrap();
        first.apply_update(1, StepStatus::Done, None).unwrap();

        let variants = [
            // 仅 delegate 变化。
            serde_json::json!([{ "step": 1, "action": "Read file", "tool": "read_file",
                "reason": "entry point", "delegate": false, "parallelizable": true }]),
            // 仅 parallelizable 变化。
            serde_json::json!([{ "step": 1, "action": "Read file", "tool": "read_file",
                "reason": "entry point", "delegate": true, "parallelizable": false }]),
            // 仅 reason 变化。
            serde_json::json!([{ "step": 1, "action": "Read file", "tool": "read_file",
                "reason": "different motive", "delegate": true, "parallelizable": true }]),
            // 仅 step 序号变化。
            serde_json::json!([{ "step": 2, "action": "Read file", "tool": "read_file",
                "reason": "entry point", "delegate": true, "parallelizable": true }]),
        ];
        for spec in variants {
            let rebuilt = PlanState::build("T", spec.as_array().unwrap(), Some(&first)).unwrap();
            assert_eq!(rebuilt.steps[0].status, StepStatus::Pending, "字段变化应回退 pending");
        }
    }

    #[test]
    fn test_apply_update_transition_warnings() {
        let raw = serde_json::json!([{ "step": 1, "action": "A", "tool": "read_file" }]);
        let mut state = PlanState::build("T", raw.as_array().unwrap(), None).unwrap();

        // 正常推进 pending → done：不提示。
        let t = state.apply_update(1, StepStatus::Done, None).unwrap();
        assert_eq!(t.from, StepStatus::Pending);
        assert_eq!(t.to, StepStatus::Done);
        assert!(t.warning.is_none());

        // 覆盖终态 done → running（返工）：提示。
        let t = state.apply_update(1, StepStatus::Running, None).unwrap();
        assert!(t.warning.as_deref().unwrap().contains("step 1"));

        // running → failed 是正常流；failed → done 直接跳变：提示。
        let t = state.apply_update(1, StepStatus::Failed, None).unwrap();
        assert!(t.warning.is_none());
        let t = state.apply_update(1, StepStatus::Done, None).unwrap();
        assert!(t.warning.as_deref().unwrap().contains("failed"));

        // 同状态重复提交不提示。
        let t = state.apply_update(1, StepStatus::Done, None).unwrap();
        assert!(t.warning.is_none());
    }

    #[test]
    fn test_apply_update_errors() {
        let raw = serde_json::json!([{ "step": 1, "action": "A", "tool": "read_file" }]);
        let mut state = PlanState::build("T", raw.as_array().unwrap(), None).unwrap();
        assert!(state
            .apply_update(9, StepStatus::Done, None)
            .unwrap_err()
            .contains("not found"));
        assert!(state.apply_update(1, StepStatus::Done, None).is_ok());
    }

}
