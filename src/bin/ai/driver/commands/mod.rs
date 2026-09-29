pub mod agent;
pub(crate) mod audit;
pub(crate) mod changes;
pub mod checkpoint;
pub(crate) mod compact;
pub mod export;
pub mod feishu;
pub mod goal;
pub mod help;
pub mod memo;
pub mod model;
pub mod persona;
pub mod proc;
pub mod session;
pub mod share;
pub mod skills;
pub mod status_line;
pub mod theme;
pub mod usage;

use std::sync::Arc;

use crate::ai::{agents::AgentManifest, mcp::SharedMcpClient, skills::SkillManifest, types::App};

pub use agent::try_handle_agent_command;
pub use checkpoint::try_handle_checkpoint_command;
pub use export::try_handle_export_command;
pub use feishu::try_handle_feishu_auth_command;
pub use goal::try_handle_goal_command;
pub use help::try_handle_help_command;
pub use model::try_handle_model_command;
pub use persona::try_handle_persona_command;
pub use proc::try_handle_proc_command;
pub use session::try_handle_clear_command;
pub use session::try_handle_session_command;
pub use share::try_handle_share_command;
pub use skills::try_handle_skills_command;
pub use theme::try_handle_theme_command;
pub use usage::try_handle_usage_command;

/// True when `token` begins a local slash command (e.g. `/sessions`, `:bg`).
///
/// The one-shot CLI (`cli.rs::protect_slash_command_args`) uses this to detect
/// where `a`'s own option parsing ends and verbatim command text begins, so
/// command flags like `--prefix` reach the command dispatcher unmodified. The
/// REPL dispatches raw input directly and never goes through this predicate.
/// Keep the word list in sync with the `try_handle_*_command` matchers in this
/// module: a missing word only degrades one-shot flag pass-through to the old
/// mangled behavior, never misroutes input.
pub fn is_local_command_start(token: &str) -> bool {
    let trimmed = token.trim();
    let Some(rest) = trimmed
        .strip_prefix('/')
        .or_else(|| trimmed.strip_prefix(':'))
    else {
        return false;
    };
    let Some(word) = rest.split_whitespace().next() else {
        return false;
    };
    matches!(
        word,
        "agent" | "agents"
            | "audit"
            | "bg"
            | "changes"
            | "checkpoint"
            | "clear"
            | "close"
            | "compact"
            | "cp"
            | "detach"
            | "diff"
            | "effort"
            | "export"
            | "feishu"
            | "fork"
            | "goal"
            | "h"
            | "help"
            | "history"
            | "mark"
            | "memo"
            | "model"
            | "models"
            | "persona"
            | "personas"
            | "proc"
            | "s"
            | "session"
            | "sessions"
            | "share"
            | "skill"
            | "skills"
            | "ss"
            | "suspend"
            | "susp"
            | "theme"
            | "token"
            | "tokens"
            | "unmark"
            | "usage"
    )
}

/// Handle local slash commands that do not depend on skill/agent manifests.
///
/// These commands (/usage, /help, /model, /goal, ...) can be dispatched before
/// manifests load; when one matches without a `forced_question` injected, the
/// expensive manifest scan is skipped, which notably cuts latency for
/// read-only commands in one-shot mode (e.g. `a /usage`).
pub fn try_handle_local_command(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    if try_handle_help_command(input) {
        return Ok(true);
    }
    if try_handle_clear_command(input) {
        return Ok(true);
    }
    if changes::try_handle_changes_command(input)? {
        return Ok(true);
    }
    if try_handle_goal_command(app, input)? {
        return Ok(true);
    }
    if try_handle_model_command(app, input)? {
        return Ok(true);
    }
    if try_handle_persona_command(app, input)? {
        return Ok(true);
    }
    if try_handle_theme_command(input)? {
        return Ok(true);
    }
    if try_handle_usage_command(input)? {
        return Ok(true);
    }
    if try_handle_session_command(app, input)? {
        return Ok(true);
    }
    if try_handle_proc_command(app, input)? {
        return Ok(true);
    }
    if try_handle_checkpoint_command(app, input)? {
        return Ok(true);
    }
    if try_handle_feishu_auth_command(mcp_client, input)? {
        return Ok(true);
    }
    if try_handle_share_command(app, input)? {
        return Ok(true);
    }
    if try_handle_export_command(app, input)? {
        return Ok(true);
    }
    Ok(false)
}

pub fn try_handle_interactive_command(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    input: &str,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
) -> Result<bool, Box<dyn std::error::Error>> {
    if try_handle_local_command(app, mcp_client, input)? {
        return Ok(true);
    }
    // Async compaction belongs to the turn dispatcher, not skill activation.
    if compact::compact_command_args(input).is_some() {
        return Ok(false);
    }
    // `/audit` must be deferred to a turn that already has DRIVER_CTX, so the
    // session/task identity is not lost during local-command dispatch; it also
    // takes precedence over explicit activation of a skill with the same name.
    if audit::parse_audit_command(input).is_some() {
        return Ok(false);
    }
    if try_handle_agent_command(app, input, agent_manifests)? {
        return Ok(true);
    }
    if try_handle_skills_command(app, input, skill_manifests)? {
        return Ok(true);
    }
    Ok(false)
}
