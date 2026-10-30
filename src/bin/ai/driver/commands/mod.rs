pub mod agent;
pub(crate) mod audit;
pub(crate) mod changes;
pub mod checkpoint;
pub(crate) mod compact;
pub(crate) mod distill_session;
pub mod cwd;
pub mod discuss;
pub mod export;
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
pub mod task;
pub mod theme;
pub mod title;
pub mod usage;

use std::sync::Arc;

use crate::ai::{agents::AgentManifest, skills::SkillManifest, types::{App, ForcedSkillSource}};

pub use agent::try_handle_agent_command;
pub use checkpoint::try_handle_checkpoint_command;
#[cfg(feature = "serve")]
pub(crate) use checkpoint::{
    SERVE_CHECKPOINT_CONFIRM_ENV, SERVE_CHECKPOINT_CONFIRM_MARKER,
    SERVE_CHECKPOINT_CONFIRM_PROBE, SERVE_CHECKPOINT_CONFIRM_YES,
};
pub use cwd::try_handle_cwd_command;
pub use discuss::try_handle_discuss_command;
pub use export::try_handle_export_command;
pub use goal::try_handle_goal_command;
pub use help::try_handle_help_command;
pub use model::try_handle_model_command;
pub use persona::try_handle_persona_command;
pub use proc::try_handle_proc_command;
pub use session::try_handle_clear_command;
pub use session::try_handle_session_command;
pub use share::try_handle_share_command;
pub use skills::try_handle_skills_command;
pub use task::try_handle_task_command;
pub use theme::try_handle_theme_command;
pub use title::try_handle_title_command;
pub use usage::try_handle_usage_command;

pub(super) fn publish_checkpoint_names(store: &crate::ai::history::CheckpointStore) {
    checkpoint::publish_checkpoint_names(store);
}

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
        "agent"
            | "agents"
            | "archive"
            | "audit"
            | "bg"
            | "cd"
            | "changes"
            | "clear"
            | "close"
            | "compact"
            | "distill-session"
            | "detach"
            | "discuss"
            | "diff"
            | "effort"
            | "export"
            | "fork"
            | "git"
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
            | "pwd"
            | "s"
            | "session"
            | "sessions"
            | "share"
            | "skill"
            | "skills"
            | "ss"
            | "suspend"
            | "susp"
            | "task"
            | "theme"
            | "title"
            | "token"
            | "tokens"
            | "unmark"
            | "usage"
    )
}

/// Commands serve-chat forwards to the turn child even though the local REPL
/// also recognizes them. The child executes them server-side: `/memo` after
/// pre-turn dispatch, `/audit` and `/distill-session` in the turn dispatcher,
/// `/mark`, `/unmark`, `/archive`, and `/title` in the pre-turn local-command
/// path, and `/task`, `/discuss`, and `/goal` inject a `forced_question` into
/// the turn.
fn is_serve_forwarded_command_start(token: &str) -> bool {
    if memo::memo_command_args(token).is_some() {
        return true;
    }
    if !is_local_command_start(token) {
        return false;
    }
    let rest = token
        .trim()
        .strip_prefix('/')
        .or_else(|| token.trim().strip_prefix(':'));
    let word = rest.and_then(|r| r.split_whitespace().next()).unwrap_or("");
    matches!(
        word,
        "task"
            | "discuss"
            | "goal"
            | "distill-session"
            | "mark"
            | "unmark"
            | "archive"
            | "audit"
            | "title"
    )
}

/// True when `token` starts a local command that serve-chat must not forward
/// as a turn. Commands in [`is_serve_forwarded_command_start`] run server-side
/// and are forwarded like a prompt instead.
pub(crate) fn is_serve_local_command_start(token: &str) -> bool {
    is_local_command_start(token) && !is_serve_forwarded_command_start(token)
}

/// Routing decision for one line submitted to a remote client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServeInputRoute {
    /// A local command this client cannot execute: reject without sending.
    Unsupported,
    /// A command the server turn child executes: forward as a prompt.
    ServerCommand,
    /// A slash/colon command shape no handler claims: reject without sending.
    UnknownCommand,
    /// Ordinary prompt text: send as a turn.
    Prompt,
}

/// Classify a submitted line so a remote client never silently turns an
/// unsupported command into a model prompt.
pub(crate) fn classify_serve_input(input: &str) -> ServeInputRoute {
    let trimmed = input.trim();
    if is_serve_local_command_start(trimmed) {
        return ServeInputRoute::Unsupported;
    }
    if is_serve_forwarded_command_start(trimmed) {
        return ServeInputRoute::ServerCommand;
    }
    if looks_like_slash_command(trimmed) {
        return ServeInputRoute::UnknownCommand;
    }
    ServeInputRoute::Prompt
}

/// Dash-flag tokens owned by the named local command's own grammar.
///
/// The one-shot CLI (`cli.rs::protect_slash_command_args`) extracts registered
/// `a` options from behind a slash command, but a token the command itself
/// interprets must stay verbatim: `/changes -s` means `--stat`, not `--sharp`,
/// and `/changes -h` means command help, not global help. Keep each entry in
/// sync with that command's parser (`changes.rs` here): a missing entry
/// silently re-enables the steal. Tokens match without any `=value` suffix, so
/// `--open=vscode` is covered by `--open`.
pub fn local_command_reserved_flags(head: &str) -> &'static [&'static str] {
    match local_command_word(head) {
        "changes" | "diff" => &[
            "-h", "--help", "--stat", "-s", "--json", "--patch", "--open",
        ],
        // `/audit -f` is fast mode (`audit.rs::strip_fast_flag`); `-f` also
        // collides with the global `--files` alias, which would additionally
        // swallow the following token as a file path.
        "audit" => &["-f", "--fast"],
        // `/git commit -m "..."` / `--message "..."`: the checkpoint parser
        // owns both spellings, and `-m` collides with the global `--model`
        // alias, which would swallow the message as its value.
        "git" => &["-m", "--message"],
        "distill-session" => distill_session::FLAGS,
        // `--prefix` is not a registered global today, so this entry changes
        // no current behavior; it only guards the `sessions delete --prefix`
        // grammar against a future global collision of the same shape.
        "sessions" | "session" | "ss" => &["--prefix"],
        _ => &[],
    }
}

/// Dash-flags whose value is one argv element that the command re-tokenizes
/// from a quoted command line.
///
/// The one-shot CLI folds argv back into command text, and that fold splits on
/// whitespace; a value flag's argument must therefore be rendered quoted to
/// stay a single word. Keep each entry in sync with the command's argument
/// parser (`checkpoint.rs::parse_save_args` for `/git`).
pub fn local_command_value_flags(head: &str) -> &'static [&'static str] {
    match local_command_word(head) {
        // `/git commit -m "..."` and `--message "..."` accept a multi-word
        // message.
        "git" => &["-m", "--message"],
        _ => &[],
    }
}

/// The command word a slash-command token names, without its `/` or `:`.
fn local_command_word(head: &str) -> &str {
    let trimmed = head.trim();
    trimmed
        .strip_prefix('/')
        .or_else(|| trimmed.strip_prefix(':'))
        .and_then(|rest| rest.split_whitespace().next())
        .unwrap_or("")
}

/// Handle local slash commands that do not depend on skill/agent manifests.
///
/// These commands (/usage, /help, /model, /goal, ...) can be dispatched before
/// manifests load; when one matches without a `forced_question` injected, the
/// expensive manifest scan is skipped, which notably cuts latency for
/// read-only commands in one-shot mode (e.g. `a /usage`).
pub fn try_handle_local_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    if try_handle_help_command(input) {
        return Ok(true);
    }
    if try_handle_clear_command(input) {
        return Ok(true);
    }
    if try_handle_cwd_command(input) {
        return Ok(true);
    }
    if changes::try_handle_changes_command(input)? {
        return Ok(true);
    }
    // Reject ambiguous multi-main sends before handler order can silently
    // decide: each of /task, /discuss, /goal sets a competing forced
    // question, and the per-handler split would swallow the losing command
    // line into the winner's topic/goal text.
    {
        let conflicts = conflicting_main_commands(input);
        if conflicts.len() >= 2 {
            let mut distinct: Vec<&str> = Vec::new();
            for kind in conflicts {
                if !distinct.contains(&kind) {
                    distinct.push(kind);
                }
            }
            let names = distinct
                .iter()
                .map(|kind| format!("/{kind}"))
                .collect::<Vec<_>>()
                .join(" and ");
            println!(
                "Ambiguous input: this message contains multiple main commands ({names}). Send one main command per message (optional /skills lines are fine); nothing was started."
            );
            return Ok(true);
        }
    }
    if try_handle_goal_command(app, input)? {
        return Ok(true);
    }
    if try_handle_discuss_command(app, input)? {
        return Ok(true);
    }
    if try_handle_task_command(app, input)? {
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
    if try_handle_title_command(app, input)? {
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
    if try_handle_share_command(app, input)? {
        return Ok(true);
    }
    if try_handle_export_command(app, input)? {
        return Ok(true);
    }
    Ok(false)
}

/// True when `input` must skip pre-turn command/activation dispatch and be
/// handled later on the turn path.
///
/// `/compact` and `/distill-session` run in the turn dispatcher and need its
/// persona/session scope; `/audit` needs `DRIVER_CTX` (and takes precedence
/// over a same-named skill); `/memo` is handled just after this dispatcher and
/// needs `&mut App` plus an async model call.
fn is_turn_deferred_command(input: &str) -> bool {
    compact::compact_command_args(input).is_some()
        || distill_session::command_args(input).is_some()
        || audit::parse_audit_command(input).is_some()
        || memo::memo_command_args(input).is_some()
}

pub fn try_handle_interactive_command(
    app: &mut App,
    input: &str,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
) -> Result<bool, Box<dyn std::error::Error>> {
    if try_handle_local_command(app, input)? {
        return Ok(true);
    }
    // Commands that need the turn's session scope, `DRIVER_CTX`, or async
    // model access skip pre-turn dispatch and run inside the turn itself.
    if is_turn_deferred_command(input) {
        return Ok(false);
    }
    if try_handle_agent_command(app, input, agent_manifests)? {
        return Ok(true);
    }
    if try_handle_skills_command(app, input, skill_manifests)? {
        return Ok(true);
    }
    // A slash-looking invocation no handler claimed is a usage error, not a
    // model question: report it locally instead of sending it to the model.
    if looks_like_slash_command(input) {
        let head = input.trim_start().split_whitespace().next().unwrap_or(input);
        println!("Unknown command: '{head}'. Run /help to list available commands.");
        return Ok(true);
    }
    Ok(false)
}

/// True when `input` looks like a slash-command invocation: the first
/// whitespace-separated token starts with `/` or `:` followed by an ASCII
/// alphanumeric, and contains no further `/`. Absolute paths (`/tmp/foo`),
/// `//` comments and `:)` emoticons are not commands. Remote clients use this
/// to reject unknown command-shaped input instead of forwarding it to a model.
pub(crate) fn looks_like_slash_command(input: &str) -> bool {
    let trimmed = input.trim_start();
    let rest = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return false;
    };
    let token = rest.split_whitespace().next().unwrap_or("");
    let mut chars = token.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    !token.contains('/')
}

/// True when the trimmed line starts with `prefix` on a word boundary, so
/// `/tasks` never matches `/task` and `/discussion` never matches `/discuss`.
fn command_line_matches(trimmed_line: &str, prefix: &str) -> bool {
    match trimmed_line.strip_prefix(prefix) {
        None => false,
        Some(rest) => rest.is_empty() || rest.starts_with(char::is_whitespace),
    }
}

/// Names a main command when the trimmed line invokes it. Each of `/task`,
/// `/discuss`, `/goal` sets a competing `forced_question`; `/skills` lines are
/// deliberately not main commands because they are legal companions to any
/// single main command.
fn main_command_kind(trimmed_line: &str) -> Option<&'static str> {
    for (kind, prefixes) in [
        ("task", ["/task", ":task"]),
        ("discuss", ["/discuss", ":discuss"]),
        ("goal", ["/goal", ":goal"]),
    ] {
        if prefixes
            .iter()
            .any(|prefix| command_line_matches(trimmed_line, prefix))
        {
            return Some(kind);
        }
    }
    None
}

/// Parse a fenced-code marker: the fence character (backtick or tilde), the
/// run length, and whatever follows the run. Returns `None` for lines that
/// cannot start a fence (fewer than three fence characters, wrong leading
/// character). Follows CommonMark: a closing fence uses the same character,
/// is at least as long as the opening run, and carries no info string; an
/// opening backtick fence's info string must not contain a backtick.
/// Command scans skip fenced lines so code examples mentioning `/task` or
/// `/skills` stay question text.
fn fence_marker(trimmed_line: &str) -> Option<(char, usize, &str)> {
    let first = trimmed_line.chars().next()?;
    if first != '`' && first != '~' {
        return None;
    }
    let run = trimmed_line.chars().take_while(|c| *c == first).count();
    if run < 3 {
        return None;
    }
    Some((first, run, &trimmed_line[run..]))
}

/// Update fence state for one trimmed line. Returns true when the line is a
/// fence delimiter or fenced content (the caller skips command parsing for
/// it). Tracks the opening character and length so a shorter or
/// cross-character run inside the block (e.g. a ``` example inside a ````
/// block) stays content instead of toggling the state early.
fn feed_fence_line(open: &mut Option<(char, usize)>, trimmed_line: &str) -> bool {
    let Some((ch, len, rest)) = fence_marker(trimmed_line) else {
        return open.is_some();
    };
    match *open {
        None => {
            if ch == '`' && rest.contains('`') {
                return false;
            }
            *open = Some((ch, len));
            true
        }
        Some((open_ch, open_len)) => {
            if ch == open_ch && len >= open_len && rest.trim().is_empty() {
                *open = None;
            }
            true
        }
    }
}

/// Every main-command line in `input`, in order of appearance.
///
/// Engages only when a `/task` / `/discuss` handler would claim the input:
/// the first non-empty line starts with `/` or `:` and is a main command or
/// a `/skills` selection. Plain prose stays a question even when later lines
/// happen to start with a command word, so it is never rejected here.
fn conflicting_main_commands(input: &str) -> Vec<&'static str> {
    let first = input.lines().map(str::trim).find(|line| !line.is_empty());
    let Some(first) = first else {
        return Vec::new();
    };
    if !(first.starts_with('/') || first.starts_with(':')) {
        return Vec::new();
    }
    if main_command_kind(first).is_none()
        && skills::parse_embedded_skills_line(first).is_none()
    {
        return Vec::new();
    }
    let mut open: Option<(char, usize)> = None;
    let mut kinds = Vec::new();
    for line in input.lines() {
        let trimmed = line.trim();
        if feed_fence_line(&mut open, trimmed) {
            continue;
        }
        if let Some(kind) = main_command_kind(trimmed) {
            kinds.push(kind);
        }
    }
    kinds
}

/// Split a combined multi-line submission into its main command input plus
/// embedded `/skills` selections.
///
/// Engages only when the first non-empty line starts with `/` or `:` (plain
/// prose stays a question even when a later line mentions a command) and is
/// either the main command or a skills selection; otherwise returns `None`
/// and the caller leaves the input alone. The main command is the first line
/// starting with one of `prefixes`; the cleaned input puts that line first so
/// the existing single-line parsers apply unchanged, with all non-selection
/// lines preserved after it.
pub(crate) fn split_combined_command(
    input: &str,
    prefixes: &[&str],
) -> Option<(String, Vec<String>)> {
    let first = input.lines().map(str::trim).find(|line| !line.is_empty())?;
    if !(first.starts_with('/') || first.starts_with(':')) {
        return None;
    }
    let starts_main = prefixes.iter().any(|p| command_line_matches(first, p));
    if !starts_main && skills::parse_embedded_skills_line(first).is_none() {
        return None;
    }
    let mut main_line: Option<&str> = None;
    let mut kept: Vec<&str> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut open: Option<(char, usize)> = None;
    for line in input.lines() {
        let trimmed = line.trim();
        if feed_fence_line(&mut open, trimmed) {
            kept.push(line);
            continue;
        }
        if main_line.is_none() && prefixes.iter().any(|p| command_line_matches(trimmed, p)) {
            main_line = Some(line);
            continue;
        }
        if let Some(picked) = skills::parse_embedded_skills_line(trimmed) {
            for name in picked {
                if !names.iter().any(|known: &String| known.eq_ignore_ascii_case(&name)) {
                    names.push(name);
                }
            }
            continue;
        }
        kept.push(line);
    }
    let main = main_line?;
    let mut cleaned = Vec::with_capacity(kept.len() + 1);
    cleaned.push(main);
    cleaned.extend(kept);
    Some((cleaned.join("\n"), names))
}

/// Apply skills extracted from a combined send. A started room activates them
/// on the host turn (mirrors `/skills-inline`); a read-only status view keeps
/// them pending for the next turn (mirrors `/skills use`).
pub(crate) fn apply_embedded_skills(app: &mut App, names: &[String], same_turn: bool) {
    if names.is_empty() {
        return;
    }
    app.forced_skills = names.to_vec();
    app.forced_skill_source = Some(if same_turn {
        ForcedSkillSource::SkillsCommandInline
    } else {
        ForcedSkillSource::SkillsCommandNextTurn
    });
    if same_turn {
        println!("Skills for this session: {}", names.join(", "));
    } else {
        println!("Skills selected for next turn: {}", names.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::looks_like_slash_command;

    #[test]
    fn slash_command_shape_detects_commands_but_not_paths_or_text() {
        assert!(looks_like_slash_command("/cmd"));
        assert!(looks_like_slash_command("/foobar x"));
        assert!(looks_like_slash_command("  :help shortcuts  "));
        assert!(!looks_like_slash_command("/tmp/foo"));
        assert!(!looks_like_slash_command("/api/v1/users"));
        assert!(!looks_like_slash_command("// comment"));
        assert!(!looks_like_slash_command("/"));
        assert!(!looks_like_slash_command(":"));
        assert!(!looks_like_slash_command(":)"));
        assert!(!looks_like_slash_command("plain prompt"));
        assert!(!looks_like_slash_command(""));
    }

    #[test]
    fn serve_input_routing_separates_local_server_unknown_and_prompt() {
        use super::{ServeInputRoute, classify_serve_input};

        assert_eq!(
            classify_serve_input("/history rewind u4"),
            ServeInputRoute::Unsupported
        );
        assert_eq!(
            classify_serve_input("/compact"),
            ServeInputRoute::Unsupported
        );
        assert_eq!(
            classify_serve_input("/distill-session abc --dry-run"),
            ServeInputRoute::ServerCommand
        );
        assert_eq!(
            classify_serve_input("/archive last"),
            ServeInputRoute::ServerCommand
        );
        for input in ["/title", ":title", "/title 排查性能问题"] {
            assert_eq!(
                classify_serve_input(input),
                ServeInputRoute::ServerCommand,
                "{input}"
            );
        }
        for input in ["/mark", ":mark keep this", "/unmark", ":unmark current"] {
            assert_eq!(
                classify_serve_input(input),
                ServeInputRoute::ServerCommand,
                "{input}"
            );
        }
        for input in [
            "/audit",
            ":audit",
            "/audit review the current diff",
            "/audit -f check this change",
            "/audit --fast check this change",
        ] {
            assert_eq!(
                classify_serve_input(input),
                ServeInputRoute::ServerCommand,
                "{input}"
            );
        }
        assert_eq!(
            classify_serve_input("/memo keep this"),
            ServeInputRoute::ServerCommand
        );
        assert_eq!(
            classify_serve_input("/task fix login"),
            ServeInputRoute::ServerCommand
        );
        assert_eq!(
            classify_serve_input("/no-such-command x"),
            ServeInputRoute::UnknownCommand
        );
        assert_eq!(
            classify_serve_input(":no-such-command"),
            ServeInputRoute::UnknownCommand
        );
        assert_eq!(
            classify_serve_input("/auditor review this"),
            ServeInputRoute::UnknownCommand
        );
        // Path-shaped absolute paths and ordinary prose stay prompts.
        assert_eq!(
            classify_serve_input("/tmp/foo"),
            ServeInputRoute::Prompt
        );
        assert_eq!(
            classify_serve_input("plain prompt"),
            ServeInputRoute::Prompt
        );
    }

    #[test]
    fn turn_deferred_commands_skip_pre_turn_dispatch() {
        use super::is_turn_deferred_command;

        for input in [
            "/compact",
            "/distill-session",
            ":distill-session session-1 --dry-run",
            "/audit -f",
            "/memo remember this",
        ] {
            assert!(is_turn_deferred_command(input), "{input}");
        }
        for input in ["/history last", "/archive", "/task x", "plain prompt"] {
            assert!(!is_turn_deferred_command(input), "{input}");
        }
    }

    #[test]
    fn combined_send_splits_main_command_and_skills() {
        use super::split_combined_command;
        // Forward order: main first, skills stripped from the goal.
        let (main, names) =
            split_combined_command("/task fix login\n/skills s1 s2", &["/task", ":task"])
                .expect("combined send must engage");
        assert_eq!(main, "/task fix login");
        assert_eq!(names, vec!["s1", "s2"]);
        // Reverse order: the main line moves first so single-line parsers apply.
        let (main, names) =
            split_combined_command("/skills use s1\n/discuss topic here", &["/discuss", ":discuss"])
                .expect("reverse order must engage");
        assert_eq!(main, "/discuss topic here");
        assert_eq!(names, vec!["s1"]);
        // No skills lines: still engages with an empty selection.
        let (main, names) =
            split_combined_command("/task fix login", &["/task", ":task"])
                .expect("plain command must engage");
        assert_eq!(main, "/task fix login");
        assert!(names.is_empty());
        // Other-command lines and prose are preserved, never claimed.
        let (main, names) =
            split_combined_command("/task x\n/discuss y", &["/task", ":task"])
                .expect("main command must engage");
        assert_eq!(main, "/task x\n/discuss y");
        assert!(names.is_empty());
        assert!(split_combined_command("hello\n/task x", &["/task", ":task"]).is_none());
        assert!(split_combined_command("/help", &["/task", ":task"]).is_none());
        assert!(split_combined_command("", &["/task", ":task"]).is_none());
        // Fenced code examples stay body text: no skills stripped, no main moved.
        let (main, names) = split_combined_command(
            "/task fix login\n```\n/skills s1\n/discuss API options\n```",
            &["/task", ":task"],
        )
        .expect("fenced body must engage");
        assert!(names.is_empty());
        assert!(main.contains("/skills s1"));
        assert!(main.contains("/discuss API options"));
        // Nested fences: a ``` example inside a ```` block stays content, so
        // neither the skills line nor the discuss line escapes the fence.
        let (main, names) = split_combined_command(
            "/task fix login\n````\n```\n/skills s1\n/discuss API options\n````",
            &["/task", ":task"],
        )
        .expect("nested fence must engage");
        assert!(names.is_empty());
        assert!(main.contains("/skills s1"));
        assert!(main.contains("/discuss API options"));
    }

    #[test]
    fn multi_main_send_reports_every_command_line() {
        use super::conflicting_main_commands;
        // The reported case: handler order used to silently start discuss and
        // swallow the /task line into its topic.
        assert_eq!(
            conflicting_main_commands("/skills code-review\n/task fix login\n/discuss API options"),
            vec!["task", "discuss"]
        );
        // Any order and any pair among task/discuss/goal conflicts.
        assert_eq!(
            conflicting_main_commands("/discuss API options\n/task fix login"),
            vec!["discuss", "task"]
        );
        assert_eq!(
            conflicting_main_commands("/goal ship it\n/task fix login"),
            vec!["goal", "task"]
        );
        // A repeated single command is still one send only if it appears once.
        assert!(conflicting_main_commands("/task fix login").len() < 2);
        assert!(conflicting_main_commands("/task a\n/task b").len() >= 2);
        // Single main plus skills is the legal combined send.
        assert!(conflicting_main_commands("/skills s1 s2\n/task fix login").len() < 2);
        assert!(conflicting_main_commands("/task status").len() < 2);
        // Word boundaries and prose are never commands.
        assert!(conflicting_main_commands("/tasks are many").is_empty());
        assert!(conflicting_main_commands("/discussion notes").is_empty());
        assert!(conflicting_main_commands("how do I use /task?\n/discuss this").is_empty());
        assert!(conflicting_main_commands("").is_empty());
        // Fenced code examples are body text, never conflicting commands.
        assert!(conflicting_main_commands(
            "/task fix login\n```\n/discuss API options\n```"
        )
        .len()
            < 2);
        assert!(conflicting_main_commands(
            "/task fix login\n~~~\n/task second\n~~~"
        )
        .len()
            < 2);
        // Nested fences: the inner ``` run is shorter than the ```` opener,
        // so it cannot close the block and the /discuss line stays fenced.
        assert!(
            conflicting_main_commands(
                "/task fix login\n````\n```\n/discuss API options\n````"
            )
            .len()
                < 2
        );
        // Cross-character runs never close: ~~~ inside a ``` block is content.
        assert!(
            conflicting_main_commands("/task fix login\n```\n~~~\n/discuss API options\n```")
                .len()
                < 2
        );
        // A shorter backtick run cannot close a longer opener: the second
        // /task line is still fenced, so no conflict is reported.
        assert!(
            conflicting_main_commands("/task a\n````\n```\n/task b\n````").len() < 2
        );
    }
}
