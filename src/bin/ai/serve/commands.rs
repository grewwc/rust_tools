//! Slash-command catalog for the WebUI composer.
//!
//! The web client and the terminal client accept different command sets: the
//! terminal keeps `/history`, `/resume`, and `/bg` local, while the WebUI
//! forwards server-owned commands and handles `/cd`, `/git`, and the transfer
//! commands in the page itself. Keeping the catalog here gives the composer one
//! source of truth for names, argument hints, and descriptions instead of
//! embedding a second list in `app.html`.

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// One command offered by the WebUI composer completion menu.
#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct ServeCommand {
    /// Command word without the leading `/` or `:`.
    pub name: &'static str,
    /// Argument hint appended after the name, without a leading space.
    pub usage: &'static str,
    /// One-line description shown beside the command.
    pub description: &'static str,
    /// Display grouping; the client keeps catalog order inside each group.
    pub group: &'static str,
}

/// Commands the WebUI can run end-to-end, in menu order.
///
/// `/model`, `/effort`, and `/theme` are intentionally absent: the page owns
/// those settings through its pickers, so suggesting a command that only
/// affects one throw-away turn child would be misleading. Terminal-only
/// commands (`/history`, `/resume`, `/bg`, `/close`, ...) stay absent for the
/// same reason.
pub(crate) const SERVE_COMMANDS: &[ServeCommand] = &[
    ServeCommand {
        name: "audit",
        usage: "[-f|--fast] [--deep] [instruction]",
        description: "Run the audit subagent, then verify and fix findings",
        group: "analysis",
    },
    ServeCommand {
        name: "task",
        usage: "[N] <goal> | status",
        description: "Run a multi-worker task room on the server",
        group: "analysis",
    },
    ServeCommand {
        name: "discuss",
        usage: "[N] <topic> | status",
        description: "Run a multi-member discussion on the server",
        group: "analysis",
    },
    ServeCommand {
        name: "goal",
        usage: "[target]",
        description: "Keep working until the goal is complete",
        group: "analysis",
    },
    ServeCommand {
        name: "changes",
        usage: "[--stat|--json|--patch|--open]",
        description: "Show files changed in this session",
        group: "analysis",
    },
    ServeCommand {
        name: "diff",
        usage: "[--stat|--json|--patch]",
        description: "Show the session diff",
        group: "analysis",
    },
    ServeCommand {
        name: "memo",
        usage: "[text]",
        description: "Save the last conclusion or given text to knowledge",
        group: "analysis",
    },
    ServeCommand {
        name: "distill-session",
        usage: "[session-id|archive.zip] [--dry-run] [--limit N]",
        description: "Distill a session into knowledge",
        group: "analysis",
    },
    ServeCommand {
        name: "compact",
        usage: "",
        description: "Compact the session context without deleting history",
        group: "session",
    },
    ServeCommand {
        name: "title",
        usage: "[text]",
        description: "Set or show the session title",
        group: "session",
    },
    ServeCommand {
        name: "mark",
        usage: "[<id>|current|last] [message]",
        description: "Mark a session as important",
        group: "session",
    },
    ServeCommand {
        name: "unmark",
        usage: "[<id>|current|last|--]",
        description: "Remove an important mark",
        group: "session",
    },
    ServeCommand {
        name: "archive",
        usage: "[<id>|current|last]",
        description: "Toggle the session archive flag",
        group: "session",
    },
    ServeCommand {
        name: "use",
        usage: "[backend] [model]",
        description: "Set the subagent backend and sticky agent-team",
        group: "session",
    },
    ServeCommand {
        name: "subagent",
        usage: "[backend] [model]",
        description: "Set the subagent backend and model",
        group: "session",
    },
    ServeCommand {
        name: "reset",
        usage: "",
        description: "Restore the default backend and clear overrides",
        group: "session",
    },
    ServeCommand {
        name: "agent",
        usage: "[name]",
        description: "List or switch agents",
        group: "session",
    },
    ServeCommand {
        name: "skills",
        usage: "[name]...",
        description: "List skills or activate them for the next message",
        group: "session",
    },
    ServeCommand {
        name: "personas",
        usage: "[list|current|use <name>]",
        description: "Manage personas",
        group: "session",
    },
    ServeCommand {
        name: "cd",
        usage: "[dir]",
        description: "Show or change the session working directory",
        group: "workspace",
    },
    ServeCommand {
        name: "pwd",
        usage: "",
        description: "Show the session working directory",
        group: "workspace",
    },
    ServeCommand {
        name: "dirs",
        usage: "[add|del <DIR>]",
        description: "List or edit writable directories",
        group: "workspace",
    },
    ServeCommand {
        name: "add-dir",
        usage: "<DIR>",
        description: "Add a writable directory to this session",
        group: "workspace",
    },
    ServeCommand {
        name: "git",
        usage: "[commit|checkout|branch|switch|tree|list|show|delete|clear|help]",
        description: "Operate on session checkpoints and branches",
        group: "workspace",
    },
    ServeCommand {
        name: "export",
        usage: "[id|current] [output.zip]",
        description: "Download a session as a ZIP archive",
        group: "transfer",
    },
    ServeCommand {
        name: "export-md",
        usage: "[file.md]",
        description: "Download the last conclusion as Markdown",
        group: "transfer",
    },
    ServeCommand {
        name: "import",
        usage: "<file.zip> [as=<id>]",
        description: "Import a session archive",
        group: "transfer",
    },
    ServeCommand {
        name: "help",
        usage: "[commands|shortcuts]",
        description: "Show interactive commands and shortcuts",
        group: "info",
    },
    ServeCommand {
        name: "usage",
        usage: "[models|today|7d|30d|all|daily]",
        description: "Show token usage statistics",
        group: "info",
    },
    ServeCommand {
        name: "proc",
        usage: "[list|kill <id>]",
        description: "List or stop background subagent processes",
        group: "info",
    },
];

/// `GET /commands`: the WebUI command catalog.
pub(crate) async fn list_commands(
    State(state): State<super::ServeState>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = super::check_auth(&state, &headers) {
        return e.into_response();
    }
    (StatusCode::OK, Json(SERVE_COMMANDS)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn command_catalog_is_well_formed() {
        let mut names = HashSet::new();
        for command in SERVE_COMMANDS {
            assert!(!command.name.is_empty(), "command name must not be empty");
            assert!(
                !command.name.starts_with('/') && !command.name.starts_with(':'),
                "catalog names are stored without a prefix: {}",
                command.name
            );
            assert!(
                command
                    .name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "command names use lowercase words and dashes: {}",
                command.name
            );
            assert!(
                names.insert(command.name),
                "duplicate command name: {}",
                command.name
            );
            assert!(
                !command.description.is_empty(),
                "command {} needs a description",
                command.name
            );
            assert!(
                !command.group.is_empty(),
                "command {} needs a group",
                command.name
            );
        }
    }

    #[test]
    fn command_catalog_covers_web_composer_commands() {
        for name in [
            "audit",
            "task",
            "discuss",
            "goal",
            "cd",
            "git",
            "export",
            "export-md",
            "import",
            "dirs",
            "add-dir",
            "title",
            "skills",
        ] {
            assert!(
                SERVE_COMMANDS.iter().any(|command| command.name == name),
                "missing web composer command: {name}"
            );
        }
    }

    #[test]
    fn command_catalog_serializes_for_the_web_client() {
        let value = serde_json::to_value(SERVE_COMMANDS).expect("catalog serializes");
        let first = value
            .as_array()
            .and_then(|items| items.first())
            .expect("catalog is not empty");
        for key in ["name", "usage", "description", "group"] {
            assert!(first.get(key).is_some(), "missing catalog field: {key}");
        }
    }
}
