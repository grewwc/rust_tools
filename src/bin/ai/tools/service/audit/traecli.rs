//! Audit rules for direct TraeCLI execution.
//!
//! The `task_spawn` tool owns every TraeCLI-backed subagent: the task
//! control block, cancellation, the evidence ledger, and result validation all
//! live there. Running the CLI as an ordinary shell command bypasses that
//! machinery, so the audit layer rejects the program outright — at any path and
//! behind wrappers such as `env`, `nohup`, or `timeout`.
//!
//! Only shell segments are affected: `task_spawn` tool calls
//! themselves never pass through `validate_execute_command`.

/// Executable names that route into the TraeCLI agent runtime.
const DIRECT_TRAECLI_PROGRAMS: &[&str] = &["traecli", "coco", "trae-agent", "ta"];

/// Rejection reason shared by every direct TraeCLI form.
pub(crate) const DIRECT_TRAECLI_BLOCKED: &str = "direct TraeCLI execution is blocked by sandbox policy: subagents must run through the \
     `task_spawn` tool with backend \"traecli\", never by executing the CLI directly; \
     direct CLI execution bypasses the task control block, cancellation, evidence ledger, and \
     result validation";

/// If `program` (a program word, possibly a path) is a TraeCLI entry point,
/// return the rejection reason.
///
/// The basename is compared so `/Users/bytedance/.local/bin/traecli` and
/// `./traecli` are rejected exactly like the bare name.
pub(crate) fn blocked_direct_traecli(program: &str) -> Option<&'static str> {
    let program = program.rsplit('/').next().unwrap_or(program);
    DIRECT_TRAECLI_PROGRAMS
        .iter()
        .any(|name| program.eq_ignore_ascii_case(name))
        .then_some(DIRECT_TRAECLI_BLOCKED)
}
