use std::time::Duration;

/// `/audit` 是用户显式发起的深度审计；允许比普通同步 `task` 更长的前台等待时间。
pub(crate) const AUDIT_SUBAGENT_HARD_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// 给审计子代理预留收口时间，避免到硬超时时丢失已经收集到的结论。
pub(crate) const AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME: Duration = Duration::from_secs(2 * 60);

const SUBAGENT_FINAL_ANSWER_MARKER: &str = "[Subagent final answer]\n";

/// `/audit` 需要在已有 DRIVER_CTX 的 turn 内启动同步子代理，因此这里只识别命令，
/// 实际执行由 turn_runtime 在进入模型循环前完成。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AuditCommand {
    Run(String),
    Usage,
}

pub(crate) fn parse_audit_command(input: &str) -> Option<AuditCommand> {
    let trimmed = input.trim();
    let normalized = trimmed
        .strip_prefix('/')
        .or_else(|| trimmed.strip_prefix(':'))?;
    let remainder = normalized.strip_prefix("audit")?;

    if remainder
        .chars()
        .next()
        .is_some_and(|first| !first.is_whitespace())
    {
        return None;
    }

    let instruction = remainder.trim();
    if instruction.is_empty() {
        Some(AuditCommand::Usage)
    } else {
        Some(AuditCommand::Run(instruction.to_string()))
    }
}

/// 同步 `/audit` 的完整 payload 会作为主 agent 的证据持久化；终端只显示子代理的
/// 最终结论，避免把工具调用记录和给主 agent 的控制提示直接泄漏给用户。
pub(crate) fn terminal_audit_result(payload: &str) -> String {
    if let Some((_, answer)) = payload.split_once(SUBAGENT_FINAL_ANSWER_MARKER) {
        let answer = answer.trim();
        let answer = answer
            .strip_suffix(crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER)
            .unwrap_or(answer)
            .trim();
        if !answer.is_empty() {
            return answer.to_string();
        }
    }

    if let Some(error) = payload
        .lines()
        .find_map(|line| line.trim().strip_prefix("Error: "))
    {
        return format!("[audit] {error}");
    }

    "[audit] Audit subagent finished without a final answer. Its full result was delivered to the main agent."
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        AUDIT_SUBAGENT_HARD_TIMEOUT, AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME, AuditCommand,
        parse_audit_command, terminal_audit_result,
    };

    #[test]
    fn audit_subagent_hard_timeout_is_fifteen_minutes() {
        assert_eq!(AUDIT_SUBAGENT_HARD_TIMEOUT, Duration::from_secs(15 * 60));
    }

    #[test]
    fn audit_subagent_reserves_two_minutes_for_wrap_up() {
        assert_eq!(
            AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME,
            Duration::from_secs(2 * 60)
        );
    }

    #[test]
    fn parse_audit_command_recognizes_instruction_and_alias() {
        assert_eq!(
            parse_audit_command("/audit review the current diff"),
            Some(AuditCommand::Run("review the current diff".to_string()))
        );
        assert_eq!(
            parse_audit_command(":audit inspect src/bin/a.rs"),
            Some(AuditCommand::Run("inspect src/bin/a.rs".to_string()))
        );
    }

    #[test]
    fn parse_audit_command_requires_an_instruction() {
        assert_eq!(parse_audit_command("/audit"), Some(AuditCommand::Usage));
        assert_eq!(parse_audit_command(" :audit   "), Some(AuditCommand::Usage));
    }

    #[test]
    fn parse_audit_command_does_not_capture_other_input() {
        assert_eq!(parse_audit_command("/auditor review"), None);
        assert_eq!(parse_audit_command("please /audit this"), None);
    }

    #[test]
    fn terminal_audit_result_hides_subagent_evidence_and_parent_reminder() {
        let payload = format!(
            "[task_id=task-1]\n\
[Task: /audit inspect diff via audit @ model] COMPLETED after 1.0s\n\
[Subagent tool evidence]\n\
- read_file({{\"file_path\":\"src/lib.rs\"}}) => fn main() {{}}\n\n\
[Subagent final answer]\n\
No verified findings.\n{}",
            crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER
        );

        assert_eq!(terminal_audit_result(&payload), "No verified findings.");
    }

    #[test]
    fn terminal_audit_result_keeps_compact_error_when_no_final_answer_exists() {
        let payload = "[task_id=task-1]\n[Task: /audit inspect via audit @ model] FAILED after 1.0s\nError: model unavailable\n[subagent evidence omitted]";

        assert_eq!(terminal_audit_result(payload), "[audit] model unavailable");
    }
}
