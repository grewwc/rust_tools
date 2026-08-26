//! Lifecycle hooks.
//!
//! Users can attach arbitrary shell commands to the following events in the ~/.configW config:
//! - `ai.hooks.on_turn_start` / `ai.hooks.on_turn_end`
//! - `ai.hooks.before_tool`   / `ai.hooks.after_tool`
//! - `ai.hooks.on_session_end`
//!
//! Hooks run best-effort: zero overhead when unconfigured; a failure only
//! prints a warning and never interrupts the main flow. Since the underlying
//! `RunCmdOptions` cannot inject environment variables, the event context is
//! prepended to the user command as a safely escaped `export VAR='...'` prefix.

use crate::ai::config_schema::AiConfig;

/// Default hook timeout (seconds).
const DEFAULT_HOOK_TIMEOUT_SECS: u64 = 30;

/// Lifecycle event. `as_event_str` also serves as the `AI_HOOK_EVENT` value passed to hooks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookEvent {
    TurnStart,
    TurnEnd,
    BeforeTool,
    AfterTool,
    SessionEnd,
}

impl HookEvent {
    fn as_event_str(self) -> &'static str {
        match self {
            HookEvent::TurnStart => "on_turn_start",
            HookEvent::TurnEnd => "on_turn_end",
            HookEvent::BeforeTool => "before_tool",
            HookEvent::AfterTool => "after_tool",
            HookEvent::SessionEnd => "on_session_end",
        }
    }

    fn config_key(self) -> &'static str {
        match self {
            HookEvent::TurnStart => AiConfig::HOOK_ON_TURN_START,
            HookEvent::TurnEnd => AiConfig::HOOK_ON_TURN_END,
            HookEvent::BeforeTool => AiConfig::HOOK_BEFORE_TOOL,
            HookEvent::AfterTool => AiConfig::HOOK_AFTER_TOOL,
            HookEvent::SessionEnd => AiConfig::HOOK_ON_SESSION_END,
        }
    }
}

/// Fire the hook for a lifecycle event. Returns immediately when unconfigured (zero overhead).
///
/// - `tool_name`: only meaningful for before/after tool events; passed as `AI_TOOL_NAME`.
/// - `tool_ok`:   only meaningful for after tool events; passed as `AI_TOOL_OK` (`true`/`false`).
pub fn run_lifecycle_hook(event: HookEvent, tool_name: Option<&str>, tool_ok: Option<bool>) {
    let cfg = crate::commonw::configw::get_all_config();
    let command = cfg.get(event.config_key(), "");
    let command = command.trim();
    if command.is_empty() {
        return;
    }

    let timeout_secs = cfg
        .get(AiConfig::HOOK_TIMEOUT_SECS, "")
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|v| *v >= 1)
        .unwrap_or(DEFAULT_HOOK_TIMEOUT_SECS);

    let full_command = build_hook_command(event, tool_name, tool_ok, command);

    // Run under effective_cwd to keep the same working-directory semantics as tools/commands.
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .ok()
        .map(|p| p.to_string_lossy().into_owned());

    match crate::ai::tools::storage::command_runner::run_command(
        &full_command,
        cwd.as_deref(),
        timeout_secs,
    ) {
        Ok(output) => {
            if !output.status.success() {
                eprintln!(
                    "[hooks] {} hook exited with status {}",
                    event.as_event_str(),
                    output.status
                );
            }
        }
        Err(err) => {
            eprintln!("[hooks] {} hook failed: {}", event.as_event_str(), err);
        }
    }
}

/// Build the final command handed to the shell: first `export` the context
/// variables (safe single-quote escaping), then append the user command.
fn build_hook_command(
    event: HookEvent,
    tool_name: Option<&str>,
    tool_ok: Option<bool>,
    user_command: &str,
) -> String {
    let mut prelude = String::new();
    prelude.push_str(&format!(
        "export AI_HOOK_EVENT={}; ",
        shell_single_quote(event.as_event_str())
    ));
    if let Some(name) = tool_name {
        prelude.push_str(&format!(
            "export AI_TOOL_NAME={}; ",
            shell_single_quote(name)
        ));
    }
    if let Some(ok) = tool_ok {
        prelude.push_str(&format!(
            "export AI_TOOL_OK={}; ",
            shell_single_quote(if ok { "true" } else { "false" })
        ));
    }
    prelude.push_str(user_command);
    prelude
}

/// Safely wrap a string in single quotes so it is a single shell literal.
/// A single quote itself is escaped with the `'\''` sequence.
fn shell_single_quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('\'');
    for ch in value.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_single_quote_wraps_plain_value() {
        assert_eq!(shell_single_quote("on_turn_start"), "'on_turn_start'");
    }

    #[test]
    fn shell_single_quote_escapes_embedded_quote() {
        // a'b  ->  'a'\''b'
        assert_eq!(shell_single_quote("a'b"), "'a'\\''b'");
    }

    #[test]
    fn shell_single_quote_neutralizes_injection() {
        // An attempted injection like `; rm -rf /` must stay fully inside the quotes and cannot escape.
        let quoted = shell_single_quote("x; rm -rf /");
        assert_eq!(quoted, "'x; rm -rf /'");
    }

    #[test]
    fn build_hook_command_includes_event_only() {
        let cmd = build_hook_command(HookEvent::TurnStart, None, None, "echo hi");
        assert_eq!(cmd, "export AI_HOOK_EVENT='on_turn_start'; echo hi");
    }

    #[test]
    fn build_hook_command_includes_tool_context() {
        let cmd = build_hook_command(
            HookEvent::AfterTool,
            Some("read_file"),
            Some(false),
            "echo done",
        );
        assert_eq!(
            cmd,
            "export AI_HOOK_EVENT='after_tool'; export AI_TOOL_NAME='read_file'; \
             export AI_TOOL_OK='false'; echo done"
        );
    }

    #[test]
    fn event_strings_are_stable() {
        assert_eq!(HookEvent::TurnStart.as_event_str(), "on_turn_start");
        assert_eq!(HookEvent::TurnEnd.as_event_str(), "on_turn_end");
        assert_eq!(HookEvent::BeforeTool.as_event_str(), "before_tool");
        assert_eq!(HookEvent::AfterTool.as_event_str(), "after_tool");
        assert_eq!(HookEvent::SessionEnd.as_event_str(), "on_session_end");
    }
}
