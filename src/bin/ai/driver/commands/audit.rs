use std::path::Path;
use std::time::Duration;

/// `/audit` 是用户显式发起的深度审计；允许比普通同步 `task` 更长的前台等待时间。
pub(crate) const AUDIT_SUBAGENT_HARD_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// 给审计子代理预留收口时间，避免到硬超时时丢失已经收集到的结论。
/// 之前 2 分钟太短：长模型请求来不及被收口信号打断就在硬超时处被 abort，导致 15 分钟
/// 工作产物丢失。5 分钟让收口信号能在请求中途打断并强制产出最终结论。
pub(crate) const AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME: Duration = Duration::from_secs(5 * 60);

/// `/audit --fast` 的硬超时：当前会话模型 + high 思考 + 步数受限，正常一轮快速
/// 审计几分钟内应能完成；8 分钟硬超时兜底，避免长时间占用前台。
pub(crate) const FAST_AUDIT_SUBAGENT_HARD_TIMEOUT: Duration = Duration::from_secs(8 * 60);
/// 快速审计的收口预留：3 分钟足够在请求中途打断并产出最终结论，
/// 同时保留同步 task 的收口保护语义。
pub(crate) const FAST_AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME: Duration = Duration::from_secs(3 * 60);

const SUBAGENT_FINAL_ANSWER_MARKER: &str = "[Subagent final answer]\n";
const AUDIT_PROGRESS_PROTOCOL: &str = "===== 审计增量交付协议 =====\n\
每完成一个独立检查分支，必须在继续调用工具前输出一条以 `AUDIT_CHECKPOINT:` 开头的简短进度记录，包含：已检查范围、带 file:line 的阶段性发现、仍待验证的问题。\n\
不要等到最终回答才首次汇总发现；checkpoint 是可恢复的阶段性证据。\n\
收到收尾信号后立即停止扩展调查，基于已有 checkpoint 和工具证据生成最终审计结论。\n\
===== 审计增量交付协议结束 =====";

/// `/audit` 需要在已有 DRIVER_CTX 的 turn 内启动同步子代理，因此这里只识别命令，
/// 实际执行由 turn_runtime 在进入模型循环前完成。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AuditCommand {
    Run { instruction: String, fast: bool },
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
    } else if let Some(rest) = strip_fast_flag(instruction) {
        if rest.is_empty() {
            // `/audit --fast` 单独出现：没有要审计的指令，同样按用法提示处理。
            Some(AuditCommand::Usage)
        } else {
            Some(AuditCommand::Run {
                instruction: rest.to_string(),
                fast: true,
            })
        }
    } else {
        Some(AuditCommand::Run {
            instruction: instruction.to_string(),
            fast: false,
        })
    }
}

/// 识别指令开头的 `--fast` / `-f` 快速模式标志；不是标志则返回 None。
fn strip_fast_flag(instruction: &str) -> Option<&str> {
    let rest = instruction
        .strip_prefix("--fast")
        .or_else(|| instruction.strip_prefix("-f"))?;
    // 标志必须是完整单词：`--fast` 后跟空白才剥离，`--fastly` 之类不动。
    if instruction.len() != rest.len()
        && rest
            .chars()
            .next()
            .is_some_and(|c| !c.is_whitespace())
    {
        return None;
    }
    Some(rest.trim_start())
}

/// 同步 `/audit` 的完整 payload 会作为主 agent 的证据持久化；终端只显示子代理的
/// 最终结论，避免把工具调用记录和给主 agent 的控制提示直接泄漏给用户。
pub(crate) fn terminal_audit_result(payload: &str) -> String {
    // 首选：子代理产出工具证据时，finalize 用 marker 分隔证据与最终答案。
    if let Some((_, answer)) = payload.split_once(SUBAGENT_FINAL_ANSWER_MARKER) {
        let answer = strip_parent_reminder(answer);
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

    // 回退：子代理零工具调用直接内联作答时 payload 里没有 marker（见 finalize.rs
    // `format_subagent_result_for_parent`：工具证据为空则原样返回最终文本）。审计
    // 提示已把 mutation-log / git diff 注入，内联作答很常见。此时剥离 sync 子代理
    // payload 的固定头部与父提醒尾巴，展示真正的结论，而不是误报「无最终答案」。
    let stripped = strip_subagent_payload_header(payload);
    let body = strip_parent_reminder(&stripped);
    if !body.is_empty() && body != SUBAGENT_NO_FINAL_TEXT_SENTINEL {
        return body.to_string();
    }

    "[audit] Audit subagent finished without a final answer. Its full result was delivered to the main agent."
        .to_string()
}

/// sync 子代理在无最终文本时写入的占位串（见 `sync_task::format_subagent_output`）。
/// 命中它说明确实没有结论，回退到通用提示而非把占位串展示给用户。
const SUBAGENT_NO_FINAL_TEXT_SENTINEL: &str = "(subagent did not produce any final assistant text)";

/// 剥离 payload 末尾的父提醒（finalize / sync 都会附加 `SUBAGENT_PARENT_SUMMARY_REMINDER`）。
fn strip_parent_reminder(text: &str) -> &str {
    text.trim()
        .strip_suffix(crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER)
        .unwrap_or_else(|| text.trim())
        .trim()
}

/// 剥离 sync 子代理 payload 的固定头部：`[task_id=…]` 行、`[Task: …] STATUS after …s`
/// 行，以及 selection_explanation 的 `agent_reason=` / `model_reason=` 行（见
/// `sync_task::format_subagent_output` 与 `task_tools::build_selection_explanation`）。
/// 只跳过开头**连续**的已知元数据行；审计结论（散文 / markdown）不会以这些前缀开头，
/// 因此不会被误删。即便头部格式将来变动，最坏也只是多显示几行头部，绝不会吞掉结论。
fn strip_subagent_payload_header(payload: &str) -> String {
    const HEADER_PREFIXES: &[&str] = &["[task_id=", "[Task: ", "agent_reason=", "model_reason="];
    let mut lines = payload.lines().peekable();
    while let Some(line) = lines.peek() {
        if HEADER_PREFIXES
            .iter()
            .any(|prefix| line.trim_start().starts_with(prefix))
        {
            lines.next();
        } else {
            break;
        }
    }
    lines.collect::<Vec<_>>().join("\n")
}

/// 构建审计子代理的输入提示。审计子代理默认只继承 cwd/skills、不继承父对话历史，
/// 因此把 main agent 本会话通过工具（write_file / apply_patch）改动的文件注入提示，
/// 让子代理知道"改了什么"--经常多个需求并行改动，子代理只有看到本会话的改动才能
/// 判断哪些属于本次审计范围（而非工作区里其他并发需求留下的未提交改动）。
pub(crate) fn build_audit_prompt(instruction: &str) -> String {
    let changes = capture_current_changes_context();
    compose_audit_prompt(instruction, &changes)
}

/// 纯函数：把用户指令与采集到的改动上下文拼成子代理输入，便于单测。
fn compose_audit_prompt(instruction: &str, changes_context: &str) -> String {
    let instruction = format!("{instruction}\n\n{AUDIT_PROGRESS_PROTOCOL}");
    if changes_context.is_empty() {
        return instruction;
    }
    format!(
        "{instruction}\n\n\
         ===== 本会话 main agent 已做的文件改动 =====\n\
         {changes_context}\n\
         ===== 文件改动结束 ====="
    )
}

/// 采集 main agent 本会话的文件改动上下文，供审计子代理判断审计范围。
///
/// 优先读取会话级 mutation log（只含本会话工具级改动，不含并发需求的改动）；
/// 若 mutation log 为空（例如改动经由 execute_command 间接产生），回退到工作区
/// 未提交改动作为尽力而为的上下文--此时可能包含并发需求留下的改动，子代理应结合
/// 审计指令判断范围。采集失败返回空串，不影响审计启动。
fn capture_current_changes_context() -> String {
    let entries = crate::ai::tools::storage::mutation_log::read_all();
    if !entries.is_empty() {
        return format_mutation_log(&entries);
    }
    capture_git_diff_fallback()
}

/// mutation log 为空时的回退：采集工作区未提交改动。非 git 仓库或无改动返回空串。
fn capture_git_diff_fallback() -> String {
    let cwd = match crate::ai::driver::runtime_ctx::effective_cwd() {
        Ok(p) => p,
        Err(_) => return String::new(),
    };
    if !is_inside_git_work_tree(&cwd) {
        return String::new();
    }
    let status = git_output(&cwd, &["status", "--porcelain=v1"]);
    if status.trim().is_empty() {
        return String::new();
    }
    const MAX_DIFF_BYTES: usize = 32_768;
    let diff = git_output(&cwd, &["diff", "HEAD", "--no-color"]);
    let mut sections = vec![format!("## git status --porcelain\n{status}")];
    if diff.len() <= MAX_DIFF_BYTES {
        if !diff.trim().is_empty() {
            sections.push(format!("## git diff HEAD\n{diff}"));
        }
    } else {
        let stat = git_output(&cwd, &["diff", "HEAD", "--stat", "--no-color"]);
        sections.push(format!(
            "## git diff HEAD --stat\n\
             （完整 diff 超过 {MAX_DIFF_BYTES} 字节，仅展示统计；如需完整内容请自行执行 `git diff HEAD`）\n\
             {stat}"
        ));
    }
    let mut out = String::from(
        "（本会话无工具级 mutation log，以下为工作区未提交改动，可能含并发需求的改动）\n\n",
    );
    out.push_str(&sections.join("\n\n"));
    out
}

/// 把 mutation log 条目格式化为审计提示用的改动摘要。
///
/// 按文件路径分组（保留首次改动顺序），取每个文件的原始 before（首条）与最终 after
/// （末条）生成净变更类型与紧凑行差异；输出总量有上限，超出则截断并指向 mutation log
/// 文件本身，子代理可用 read_file 读取完整 before/after。
fn format_mutation_log(
    entries: &[crate::ai::tools::storage::mutation_log::MutationEntry],
) -> String {
    // (path, first_before, last_after, last_op, write_count, delete_count)
    let mut summary: Vec<(String, Option<String>, Option<String>, String, usize, usize)> =
        Vec::new();
    for e in entries {
        if let Some(s) = summary.iter_mut().find(|(p, _, _, _, _, _)| p == &e.path) {
            s.2 = e.after.clone();
            s.3 = e.op.clone();
            if e.op == "write" {
                s.4 += 1;
            } else {
                s.5 += 1;
            }
        } else {
            summary.push((
                e.path.clone(),
                e.before.clone(),
                e.after.clone(),
                e.op.clone(),
                if e.op == "write" { 1 } else { 0 },
                if e.op == "delete" { 1 } else { 0 },
            ));
        }
    }

    let cwd = crate::ai::driver::runtime_ctx::effective_cwd().ok();
    let log_path = crate::ai::tools::storage::mutation_log::log_path();
    let mut out = String::from(
        "以下是 main agent 在本会话通过 write_file / apply_patch 改动的文件（按首次改动顺序）。\n\
         这只包含本会话工具级改动，不含其他并发需求留下的未提交改动。\n\n",
    );
    const CAP: usize = 14_000;
    let mut truncated = false;
    for (i, (path, first_before, last_after, last_op, wc, dc)) in summary.iter().enumerate() {
        if out.len() >= CAP {
            truncated = true;
            break;
        }
        let rel = cwd
            .as_ref()
            .and_then(|c| std::path::Path::new(path).strip_prefix(c).ok())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        let net = match (
            last_op.as_str(),
            first_before.is_some(),
            last_after.is_some(),
        ) {
            ("delete", _, _) => "deleted",
            (_, false, true) => "created",
            _ => "modified",
        };
        let mut counts = String::new();
        if *wc > 0 {
            counts.push_str(&format!("{wc} write"));
        }
        if *dc > 0 {
            if !counts.is_empty() {
                counts.push_str(", ");
            }
            counts.push_str(&format!("{dc} delete"));
        }
        out.push_str(&format!("{}. {}  [{net}]  ({counts})\n", i + 1, rel));
        if let Some(diff) = diff_snippet(first_before.as_deref(), last_after.as_deref(), 30) {
            out.push_str(&diff);
            out.push_str("\n\n");
        }
    }
    if truncated {
        out.push_str("…（更多改动省略）\n\n");
    }
    if let Some(lp) = &log_path {
        out.push_str(&format!(
            "完整 before/after（含每次写入的中间状态）见 mutation log：{}\n\
             可用 read_file 读取该日志获取每个改动的原始与最终内容。\n",
            lp.display()
        ));
    }
    out
}

/// 生成 before -> after 的紧凑行差异片段（剔除公共前缀/后缀行）。
///
/// 适用于单区域编辑；多区域编辑会把中间整段标为差异（由调用方截断）。返回 None 表示
/// 内容未变或前后均无内容。最多展示 `max_lines` 行，超出则截断并提示。
fn diff_snippet(before: Option<&str>, after: Option<&str>, max_lines: usize) -> Option<String> {
    let (b, a): (Vec<&str>, Vec<&str>) = match (before, after) {
        (None, None) => return None,
        (None, Some(a)) => (Vec::new(), a.lines().collect()),
        (Some(b), None) => (b.lines().collect(), Vec::new()),
        (Some(b), Some(a)) => (b.lines().collect(), a.lines().collect()),
    };
    if b == a {
        return None;
    }
    let mut prefix = 0;
    while prefix < b.len() && prefix < a.len() && b[prefix] == a[prefix] {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < b.len() - prefix
        && suffix < a.len() - prefix
        && b[b.len() - 1 - suffix] == a[a.len() - 1 - suffix]
    {
        suffix += 1;
    }
    let mut lines: Vec<String> = Vec::new();
    for l in &b[prefix..b.len() - suffix] {
        lines.push(format!("- {l}"));
    }
    for l in &a[prefix..a.len() - suffix] {
        lines.push(format!("+ {l}"));
    }
    if lines.is_empty() {
        return None;
    }
    let total = lines.len();
    if total <= max_lines {
        Some(format!("```diff\n{}\n```", lines.join("\n")))
    } else {
        let shown: Vec<&str> = lines.iter().take(max_lines).map(|s| s.as_str()).collect();
        Some(format!(
            "```diff\n{}\n```\n（差异共 {total} 行，已展示前 {max_lines} 行；完整内容见 mutation log）",
            shown.join("\n")
        ))
    }
}

fn is_inside_git_work_tree(cwd: &Path) -> bool {
    git_output(cwd, &["rev-parse", "--is-inside-work-tree"])
        .trim()
        .eq_ignore_ascii_case("true")
}

fn git_output(cwd: &Path, args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        AUDIT_SUBAGENT_HARD_TIMEOUT, AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME, AuditCommand,
        FAST_AUDIT_SUBAGENT_HARD_TIMEOUT, FAST_AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME,
        compose_audit_prompt, diff_snippet, format_mutation_log, parse_audit_command,
        terminal_audit_result,
    };

    #[test]
    fn audit_subagent_hard_timeout_is_fifteen_minutes() {
        assert_eq!(AUDIT_SUBAGENT_HARD_TIMEOUT, Duration::from_secs(15 * 60));
    }

    #[test]
    fn audit_subagent_reserves_five_minutes_for_wrap_up() {
        assert_eq!(
            AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME,
            Duration::from_secs(5 * 60)
        );
    }

    #[test]
    fn fast_audit_shorter_timeouts_than_full_audit() {
        assert!(FAST_AUDIT_SUBAGENT_HARD_TIMEOUT < AUDIT_SUBAGENT_HARD_TIMEOUT);
        assert!(
            FAST_AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME < AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME
        );
    }

    #[test]
    fn parse_audit_command_recognizes_instruction_and_alias() {
        assert_eq!(
            parse_audit_command("/audit review the current diff"),
            Some(AuditCommand::Run {
                instruction: "review the current diff".to_string(),
                fast: false,
            })
        );
        assert_eq!(
            parse_audit_command(":audit inspect src/bin/a.rs"),
            Some(AuditCommand::Run {
                instruction: "inspect src/bin/a.rs".to_string(),
                fast: false,
            })
        );
    }

    #[test]
    fn parse_audit_command_recognizes_fast_flag() {
        assert_eq!(
            parse_audit_command("/audit --fast review the diff"),
            Some(AuditCommand::Run {
                instruction: "review the diff".to_string(),
                fast: true,
            })
        );
        assert_eq!(
            parse_audit_command("/audit -f check src/lib.rs"),
            Some(AuditCommand::Run {
                instruction: "check src/lib.rs".to_string(),
                fast: true,
            })
        );
    }

    #[test]
    fn parse_audit_command_fast_flag_without_instruction_is_usage() {
        assert_eq!(parse_audit_command("/audit --fast"), Some(AuditCommand::Usage));
        assert_eq!(parse_audit_command("/audit -f  "), Some(AuditCommand::Usage));
    }

    #[test]
    fn parse_audit_command_does_not_confuse_fast_prefix_with_word() {
        // `--fastly` 不是标志；指令原样保留。
        assert_eq!(
            parse_audit_command("/audit --fastly failing path"),
            Some(AuditCommand::Run {
                instruction: "--fastly failing path".to_string(),
                fast: false,
            })
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

    #[test]
    fn terminal_audit_result_recovers_inline_answer_without_marker() {
        // 子代理零工具调用直接内联作答：payload 无 [Subagent final answer] marker、
        // 状态 COMPLETED、无 Error 行。回退必须剥离头部/父提醒并展示真正结论，
        // 而不是误报「finished without a final answer」。
        let payload = format!(
            "[task_id=task-1]\n\
[Task: /audit review the diff via audit @ model] COMPLETED after 2.0s\n\
agent_reason=explicit agent override\n\
model_reason=inherited parent agent current model\n\
P0: 空指针解引用在 foo.rs:42。其余改动无阻断问题。\n{}",
            crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER
        );

        assert_eq!(
            terminal_audit_result(&payload),
            "P0: 空指针解引用在 foo.rs:42。其余改动无阻断问题。"
        );
    }

    #[test]
    fn terminal_audit_result_falls_back_to_generic_when_no_final_text_produced() {
        // 子代理确实没有产出结论：sync 写入占位串。此时仍回退到通用提示。
        let payload = format!(
            "[task_id=task-1]\n\
[Task: /audit review via audit @ model] COMPLETED after 1.0s\n\
agent_reason=explicit agent override\n\
model_reason=inherited parent agent current model\n\
(subagent did not produce any final assistant text)\n{}",
            crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER
        );

        assert!(
            terminal_audit_result(&payload)
                .starts_with("[audit] Audit subagent finished without a final answer")
        );
    }

    #[test]
    fn compose_audit_prompt_requires_incremental_checkpoints_without_changes() {
        let prompt = compose_audit_prompt("review the diff", "");
        assert!(prompt.starts_with("review the diff\n\n"));
        assert!(prompt.contains("AUDIT_CHECKPOINT:"));
        assert!(prompt.contains("不要等到最终回答才首次汇总发现"));
    }

    #[test]
    fn compose_audit_prompt_appends_changes_context() {
        let prompt = compose_audit_prompt("review the diff", "（示例改动上下文）");
        assert!(prompt.starts_with("review the diff\n\n"));
        assert!(prompt.contains("本会话 main agent 已做的文件改动"));
        assert!(prompt.contains("（示例改动上下文）"));
        assert!(prompt.contains("文件改动结束"));
    }

    #[test]
    fn diff_snippet_shows_added_lines_for_created_file() {
        let snippet = diff_snippet(None, Some("line1\nline2\n"), 30).unwrap();
        assert!(snippet.contains("+ line1"));
        assert!(snippet.contains("+ line2"));
        assert!(!snippet.contains("- "));
    }

    #[test]
    fn diff_snippet_shows_removed_lines_for_deleted_file() {
        let snippet = diff_snippet(Some("old\n"), None, 30).unwrap();
        assert!(snippet.contains("- old"));
        assert!(!snippet.contains("+ "));
    }

    #[test]
    fn diff_snippet_strips_common_prefix_and_suffix() {
        let before = "header\nkeep1\nMIDDLE\nkeep2\nfooter\n";
        let after = "header\nkeep1\nNEW\nkeep2\nfooter\n";
        let snippet = diff_snippet(Some(before), Some(after), 30).unwrap();
        assert!(snippet.contains("- MIDDLE"));
        assert!(snippet.contains("+ NEW"));
        assert!(!snippet.contains("header"));
        assert!(!snippet.contains("footer"));
    }

    #[test]
    fn diff_snippet_returns_none_when_unchanged() {
        assert!(diff_snippet(Some("same\n"), Some("same\n"), 30).is_none());
        assert!(diff_snippet(None, None, 30).is_none());
    }

    #[test]
    fn diff_snippet_truncates_large_diff() {
        let before: String = "old\n".repeat(100);
        let after: String = "new\n".repeat(100);
        let snippet = diff_snippet(Some(&before), Some(&after), 10).unwrap();
        assert!(snippet.contains("已展示前 10 行"));
        assert!(snippet.contains("差异共 200 行"));
    }

    #[test]
    fn format_mutation_log_groups_by_path_and_shows_net_change() {
        use crate::ai::tools::storage::mutation_log::MutationEntry;
        let entries = vec![MutationEntry {
            seq: 1,
            ts: "t1".into(),
            path: "/proj/src/a.rs".into(),
            op: "write".into(),
            before: Some("fn a() {}\n".into()),
            after: Some("fn a() { return 1; }\n".into()),
        }];
        let out = format_mutation_log(&entries);
        assert!(out.contains("src/a.rs"));
        assert!(out.contains("[modified]"));
        assert!(out.contains("1 write"));
        assert!(out.contains("- fn a() {}"));
        assert!(out.contains("+ fn a() { return 1; }"));
    }
}
