//! Interactive client for a running serve instance.
//!
//! Local testing REPL that mirrors the single-machine interactive input: the
//! same multiline `PromptEditor` (Enter inserts a newline, Esc or Alt+Enter
//! submits) and progressive output rendering. Each submitted prompt is POSTed
//! to `POST /sessions/{id}/turns/stream`; every SSE event is rendered as it
//! arrives instead of waiting for the whole turn.

use std::{
    collections::VecDeque,
    io::IsTerminal,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::commonw::{
    configw,
    utils::{expanduser, get_config_dir},
};

use super::super::{
    agents,
    cli::ParsedCli,
    config_schema::AiConfig,
    driver::{
        commands::{
            ServeInputRoute, classify_serve_input, serve_command_title_seed,
            use_backend::{parse_use_arguments, validate_reset_arguments},
        },
        history_rewind::{
            HistoryRewindTarget, count_real_users, parse_history_rewind_target, real_user_index,
            real_user_ordinal_entries,
        },
        input::inline_image_filenames,
    },
    history::{SessionStore, current_terminal_key},
    model_names, models,
    prompt::{PromptEditor, completion::CommandCompleter},
    stream::{MarkdownStreamRenderer, side_note_input::SideNoteInputGuard},
};

#[derive(Debug, Deserialize)]
struct CreatedSession {
    id: String,
}

/// Serve-side counterpart of the local suspended-session binding: remembers
/// which remote session this terminal backgrounded with `/bg`, so the next
/// `a --serve-chat` here resumes it instead of opening a fresh session.
///
/// Kept in its own directory, never inside `SuspendedSessionStore`: a remote
/// id stored there would be adopted by plain `a` as a local session on the
/// next launch in that terminal.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct ServeChatSuspendedBinding {
    base: String,
    session_id: String,
    suspended_at: String,
}

fn serve_chat_suspended_root() -> PathBuf {
    // Same override seam as the local suspended store: tests point it at a
    // temp dir. The serve bindings live next to that dir, never inside it.
    if let Ok(dir) = std::env::var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR") {
        let trimmed = dir.trim();
        if !trimmed.is_empty() {
            let dir = PathBuf::from(expanduser(trimmed).as_ref());
            match dir.parent() {
                Some(parent) => return parent.join("serve_chat_suspended"),
                None => return dir.join("serve_chat_suspended"),
            }
        }
    }
    get_config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("rust_tools")
        .join("serve_chat_suspended")
}

fn hex_encode_terminal_key(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

fn serve_chat_binding_path(root: &Path, terminal_key: &str) -> PathBuf {
    root.join(format!(
        "{}.json",
        hex_encode_terminal_key(terminal_key.as_bytes())
    ))
}

fn save_serve_chat_binding_for_key(
    root: &Path,
    terminal_key: &str,
    base: &str,
    session_id: &str,
) -> std::io::Result<()> {
    std::fs::create_dir_all(root)?;
    let binding = ServeChatSuspendedBinding {
        base: base.to_string(),
        session_id: session_id.to_string(),
        suspended_at: chrono::Local::now().to_rfc3339(),
    };
    let content = serde_json::to_string_pretty(&binding).map_err(std::io::Error::other)?;
    std::fs::write(serve_chat_binding_path(root, terminal_key), content)
}

fn load_serve_chat_binding_for_key(
    root: &Path,
    terminal_key: &str,
) -> Option<ServeChatSuspendedBinding> {
    let content = std::fs::read_to_string(serve_chat_binding_path(root, terminal_key)).ok()?;
    serde_json::from_str(&content).ok()
}

/// Record this terminal's backgrounded remote session. Fails when the
/// terminal is not identifiable; callers still exit, with a manual resume hint.
fn save_serve_chat_binding(base: &str, session_id: &str) -> std::io::Result<()> {
    let key = current_terminal_key().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "current terminal is not identifiable",
        )
    })?;
    save_serve_chat_binding_for_key(&serve_chat_suspended_root(), &key, base, session_id)
}

fn load_serve_chat_binding() -> Option<ServeChatSuspendedBinding> {
    let key = current_terminal_key()?;
    load_serve_chat_binding_for_key(&serve_chat_suspended_root(), &key)
}

/// Best-effort cleanup (used after `/close` deletes the bound session).
fn clear_serve_chat_binding() {
    if let Some(key) = current_terminal_key() {
        let _ = std::fs::remove_file(serve_chat_binding_path(&serve_chat_suspended_root(), &key));
    }
}

/// One entry of `GET /sessions` (newest first). `summary` is the generated
/// title when the server has one; otherwise the client falls back to the
/// first user prompt. Optional fields default so older servers still parse.
#[derive(Debug, Clone, Deserialize)]
struct RemoteSession {
    id: String,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    first_user_prompt: Option<String>,
    #[serde(default)]
    marked: bool,
    /// Older servers do not report the formal archive flag; they also have no
    /// archive view, so a missing field degrades to "not archived".
    #[serde(default)]
    archived: bool,
}

/// Fetch all remote sessions (`GET /sessions`, newest first). No client-side
/// cap: the local REPL lists every session, and a capped list would hide
/// exactly the older session a reconnect is looking for. Thin network
/// glue; parsing and display live in the pure helpers below so tests stay
/// offline except for this one loopback round-trip.
fn fetch_remote_sessions(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
) -> Result<Vec<RemoteSession>, String> {
    let resp = auth(client.get(format!("{base}/sessions")), token)
        .send()
        .map_err(|err| format!("cannot reach serve at {base} ({err})"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    parse_remote_sessions(&text)
}

#[derive(Debug, Deserialize)]
struct RemoteCheckpointList {
    checkpoints: Vec<String>,
    #[serde(default)]
    branches: Vec<String>,
    #[serde(default)]
    checkpoint_status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RemoteCheckpointCommand {
    output: String,
    confirmation: Option<String>,
    checkpoints: Vec<String>,
    #[serde(default)]
    branches: Vec<String>,
    #[serde(default)]
    checkpoint_status: Option<String>,
}

fn fetch_remote_checkpoint_names(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
) -> Result<RemoteCheckpointList, String> {
    let resp = auth(
        client.get(format!("{base}/sessions/{session_id}/checkpoints")),
        token,
    )
    .timeout(Duration::from_secs(2))
    .send()
    .map_err(|err| format!("cannot reach serve at {base} ({err})"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    let response: RemoteCheckpointList = serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected checkpoint list: {err}"))?;
    Ok(response)
}

fn post_remote_checkpoint_command(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    command: &str,
    confirmed: bool,
) -> Result<RemoteCheckpointCommand, String> {
    let resp = auth(
        client.post(format!("{base}/sessions/{session_id}/checkpoints")),
        token,
    )
    .json(&serde_json::json!({"command": command, "confirmed": confirmed}))
    .send()
    .map_err(|err| format!("cannot reach serve at {base} ({err})"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected checkpoint response: {err}"))
}

fn run_serve_checkpoint_command(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    command: &str,
) -> Result<Option<String>, String> {
    let mut response =
        post_remote_checkpoint_command(client, base, token, session_id, command, false)?;
    if let Some(prompt) = response.confirmation.take() {
        let confirmed = crate::commonw::prompt::prompt_yes_or_no_danger(&prompt);
        if confirmed != Some(true) {
            println!("Canceled: nothing was changed.");
            return Ok(response.checkpoint_status);
        }
        response = post_remote_checkpoint_command(client, base, token, session_id, command, true)?;
    }
    print!("{}", response.output);
    CommandCompleter::set_checkpoint_name_candidates(response.checkpoints);
    CommandCompleter::set_checkpoint_branch_candidates(response.branches);
    Ok(response.checkpoint_status)
}

/// Parse a `GET /sessions` body. Entries with an empty id are dropped —
/// they can never be resumed.
fn parse_remote_sessions(text: &str) -> Result<Vec<RemoteSession>, String> {
    let mut items: Vec<RemoteSession> = serde_json::from_str(text)
        .map_err(|err| format!("serve returned an unexpected session list: {err}"))?;
    items.retain(|s| !s.id.trim().is_empty());
    Ok(items)
}

/// Runtime labels the connected server turns run with (`GET /info`).
/// Every field defaults so an older server (no `/info` route, or a thinner
/// body) degrades to the previous blank-header behavior instead of failing
/// the whole chat.
#[derive(Debug, Deserialize, Default)]
struct ServerInfo {
    #[serde(default)]
    model: String,
    #[serde(default)]
    model_label: String,
    #[serde(default)]
    agent: String,
    #[serde(default)]
    reasoning_effort: String,
    #[serde(default)]
    version: String,
}

/// Parse a `GET /info` body. Pure helper so tests stay offline.
fn parse_server_info(text: &str) -> Result<ServerInfo, String> {
    serde_json::from_str(text)
        .map_err(|err| format!("serve returned an unexpected info body: {err}"))
}

/// Fetch the server runtime info. `None` means "predates `/info` or
/// unreachable": the caller keeps blank labels (today's behavior).
fn fetch_server_info(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
) -> Option<ServerInfo> {
    let text = auth(client.get(format!("{base}/info")), token)
        .send()
        .ok()
        .filter(|resp| resp.status().is_success())?
        .text()
        .unwrap_or_default();
    parse_server_info(&text).ok()
}

/// Title row for the current remote session from the last listing, mirroring
/// the local REPL's session-topic line (generated title, else first prompt).
fn current_remote_title(listed: &[RemoteSession], session_id: &str) -> Option<String> {
    listed
        .iter()
        .find(|s| s.id == session_id)
        .map(remote_session_title)
}

/// Mirror the local REPL header (`prompt_user` in `driver/input.rs`): model,
/// agent, reasoning-effort and session topic above the input box. The model
/// label stays bare; the editor carries a remote-mode flag so the renderer
/// paints its own distinctly styled `(remote)` marker after the model name.
/// The completion hint keeps the raw server model id.
fn apply_remote_header(editor: &mut PromptEditor, info: &ServerInfo, topic: Option<String>) {
    if !info.model_label.is_empty() {
        editor.set_current_model_label(&info.model_label);
        editor.set_model_remote(true);
    }
    if !info.agent.is_empty() {
        editor.set_current_agent_label(&info.agent);
    }
    if !info.reasoning_effort.is_empty() {
        editor.set_current_reasoning_effort_label(&info.reasoning_effort);
    }
    if !info.model.is_empty() {
        CommandCompleter::set_current_model_hint(&info.model);
    }
    editor.set_session_topic(topic);
}

/// Client-side per-turn selection for serve-chat `/model`/`/effort`/`/agent`
/// switching. The server stays stateless: the selected values travel with
/// every turn request (`TurnReq` overrides) and the child CLI resolves them
/// through the same registry path as the local REPL. `reasoning_effort` is
/// `None` when cleared (server default), `Some("off")` when disabled, else
/// the explicit tier. `skills` holds the pending forced skills from
/// `/skills use` (next turn only, like the local REPL): they ride along as
/// the `skills` turn field and are cleared once the server accepts the turn.
#[derive(Debug, Clone)]
struct ServeTurnSelection {
    model: String,
    agent: String,
    reasoning_effort: Option<String>,
    skills: Vec<String>,
}

impl ServeTurnSelection {
    fn from_info(info: &ServerInfo) -> Self {
        Self {
            model: info.model.clone(),
            agent: info.agent.clone(),
            reasoning_effort: match info.reasoning_effort.trim().to_ascii_lowercase().as_str() {
                "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "off" => {
                    Some(info.reasoning_effort.trim().to_ascii_lowercase())
                }
                _ => None,
            },
            skills: Vec::new(),
        }
    }

    /// Effective effort label for the input-box header: the explicit
    /// override when set, else the registry default for the current model.
    fn effective_effort_label(&self) -> String {
        if let Some(effort) = self.reasoning_effort.as_deref() {
            return effort.to_string();
        }
        models::default_reasoning_effort(&self.model)
            .map(|e| e.as_str().to_string())
            .unwrap_or_else(|| "server default".to_string())
    }

    /// Refresh the input-box header after a local switch, mirroring
    /// `apply_remote_header` but driven by the client selection instead of a
    /// fresh `GET /info` (the server has no session state to update).
    fn apply_header(&self, editor: &mut PromptEditor) {
        if !self.model.is_empty() {
            let label = models::model_display_label(&self.model);
            if !label.is_empty() {
                editor.set_current_model_label(&label);
                editor.set_model_remote(true);
            }
            CommandCompleter::set_current_model_hint(&self.model);
        }
        if !self.agent.is_empty() {
            editor.set_current_agent_label(&self.agent);
        }
        editor.set_current_reasoning_effort_label(self.effective_effort_label());
    }
}

/// Field-wise update for `POST /sessions/{id}/config` (the shape of
/// `serve::types::SessionConfigReq`): an absent field keeps the server's
/// stored value, a blank string clears it, a value sets it. The serve-chat
/// `/model`/`/effort`/`/agent` handlers return one so the caller persists only
/// what actually changed — switching the model must not wipe a stored effort.
#[derive(Debug, Default, Serialize)]
struct ServeConfigPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

impl ServeConfigPatch {
    fn is_empty(&self) -> bool {
        self.model.is_none() && self.agent.is_none() && self.reasoning_effort.is_none()
    }
}

/// Server-side body of `GET /sessions/{id}/config`: an empty field means
/// "server default".
#[derive(Debug, Deserialize, Default)]
struct ServeSessionConfig {
    #[serde(default)]
    model: String,
    #[serde(default)]
    agent: String,
    #[serde(default)]
    reasoning_effort: String,
}

/// Best-effort persist of a selection change to the server's session config,
/// so a later reconnect (`a --serve-chat --session <id>`) resumes the picks
/// instead of server defaults. A failure is reported once and never blocks the
/// chat.
fn persist_serve_config_patch(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    patch: &ServeConfigPatch,
) {
    if patch.is_empty() {
        return;
    }
    let Ok(body) = serde_json::to_string(patch) else {
        return;
    };
    match auth(
        client.post(format!("{base}/sessions/{session_id}/config")),
        token,
    )
    .header("Content-Type", "application/json")
    .body(body)
    .send()
    {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => eprintln!(
            "[serve-chat] failed to persist session picks: {}",
            server_error_message(resp.status(), &resp.text().unwrap_or_default())
        ),
        Err(err) => eprintln!("[serve-chat] failed to persist session picks: {err}"),
    }
}

/// Fold a server-stored session config into the client selection. Stored
/// non-empty picks override the `/info` defaults so a reconnect resumes the
/// last `/model`/`/effort`/`/agent` state. Returns true when something
/// changed (callers refresh the input-box header in that case).
fn merge_restored_session_config(
    cfg: &ServeSessionConfig,
    selection: &mut ServeTurnSelection,
) -> bool {
    let mut changed = false;
    if !cfg.model.is_empty() && cfg.model != selection.model {
        selection.model = cfg.model.clone();
        changed = true;
    }
    if !cfg.agent.is_empty() && cfg.agent != selection.agent {
        selection.agent = cfg.agent.clone();
        changed = true;
    }
    if !cfg.reasoning_effort.is_empty()
        && selection.reasoning_effort.as_deref() != Some(cfg.reasoning_effort.as_str())
    {
        selection.reasoning_effort = Some(cfg.reasoning_effort.clone());
        changed = true;
    }
    changed
}

/// Fetch and apply the server-stored session picks (`GET /sessions/{id}/config`).
/// Best-effort: an unreachable or unknown session keeps the `/info` defaults.
fn restore_serve_session_picks(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    selection: &mut ServeTurnSelection,
) -> bool {
    let cfg = auth(
        client.get(format!("{base}/sessions/{session_id}/config")),
        token,
    )
    .send()
    .ok()
    .filter(|resp| resp.status().is_success())
    .and_then(|resp| resp.text().ok())
    .and_then(|text| serde_json::from_str::<ServeSessionConfig>(&text).ok());
    cfg.as_ref()
        .map(|cfg| merge_restored_session_config(cfg, selection))
        .unwrap_or(false)
}

/// Normalize a non-empty `/effort` argument to the stored override value.
/// Mirrors the local `/effort` and `/model effort` branches
/// (`driver/commands/model.rs::handle_effort_arg`): `auto|clear|default|reset`
/// clears the override (server default), `off`-family disables, a tier name
/// sets the tier. `Err` carries the human-readable rejection.
fn normalize_serve_effort_arg(arg: &str) -> Result<Option<String>, String> {
    match arg.trim().to_ascii_lowercase().as_str() {
        "auto" | "clear" | "default" | "reset" => Ok(None),
        "off" | "none" | "no" | "false" | "disable" | "disabled" => Ok(Some("off".to_string())),
        "minimal" | "low" | "medium" | "high" | "xhigh" | "max" => {
            Ok(Some(arg.trim().to_ascii_lowercase()))
        }
        _ => Err("unknown effort; expected minimal|low|medium|high|xhigh|max|off|auto".to_string()),
    }
}

/// Split a `/model <selector> [question...]` remainder: the first token is
/// the model selector, the rest (when non-empty) is sent as the turn prompt
/// immediately, mirroring the local forced-question form.
fn split_serve_model_inline(arg: &str) -> (String, Option<String>) {
    let arg = arg.trim();
    match arg.split_once(char::is_whitespace) {
        Some((selector, rest)) => {
            let question = rest.trim();
            if question.is_empty() {
                (selector.to_string(), None)
            } else {
                (selector.to_string(), Some(question.to_string()))
            }
        }
        None => (arg.to_string(), None),
    }
}

/// Handle `/model` in serve-chat. Returns a pending prompt when the command
/// carries an inline question (`/model foo <question>` switches and sends).
/// Mirrors the local `/model` branches (`driver/commands/model.rs`):
/// `list|current|help`, `effort <level>`, else a registry selector.
fn run_serve_model_command(
    arg: &str,
    selection: &mut ServeTurnSelection,
    editor: &mut PromptEditor,
) -> (Option<String>, ServeConfigPatch) {
    let arg = arg.trim();
    if arg.is_empty() || arg.eq_ignore_ascii_case("current") || arg.eq_ignore_ascii_case("cur") {
        if selection.model.is_empty() {
            println!("(remote) model: unknown");
        } else if let Some(def) = model_names::find_by_identifier(&selection.model) {
            let handle = model_names::model_handle(def);
            println!(
                "(remote) current model: {}",
                models::model_display_label(&handle)
            );
        } else {
            println!("(remote) current model: {}", selection.model);
        }
        println!("(remote) effort: {}", selection.effective_effort_label());
        return (None, ServeConfigPatch::default());
    }
    if arg.eq_ignore_ascii_case("list") || arg.eq_ignore_ascii_case("ls") {
        let current_handle = model_names::find_by_identifier(&selection.model)
            .map(model_names::model_handle)
            .unwrap_or_else(|| selection.model.trim().to_string())
            .to_ascii_lowercase();
        for def in model_names::all() {
            let handle = model_names::model_handle(def);
            let marker = if handle.eq_ignore_ascii_case(&current_handle) {
                "*"
            } else {
                " "
            };
            println!("{marker} {}", models::model_display_label(&handle));
        }
        return (None, ServeConfigPatch::default());
    }
    if arg.eq_ignore_ascii_case("help") || arg.eq_ignore_ascii_case("h") {
        println!("(remote) /model <selector> [question] - switch remote model");
        println!("(remote) /model list|current|help - inspect remote models");
        println!("(remote) /model effort <level> - same as /effort <level>");
        return (None, ServeConfigPatch::default());
    }
    let (head, head_rest) = match arg.split_once(char::is_whitespace) {
        Some((h, r)) => (h, r.trim()),
        None => (arg, ""),
    };
    if head.eq_ignore_ascii_case("effort") {
        if head_rest.is_empty() {
            println!("(remote) effort: {}", selection.effective_effort_label());
            return (None, ServeConfigPatch::default());
        }
        let patch = run_serve_effort_command(head_rest, selection, editor);
        return (None, patch);
    }
    let (selector, question) = split_serve_model_inline(arg);
    match model_names::find_by_identifier(&selector) {
        Some(def) => {
            let handle = model_names::model_handle(def);
            selection.model = handle.clone();
            selection.apply_header(editor);
            println!(
                "(remote) switched model to {}",
                models::model_display_label(&handle)
            );
            (
                question,
                ServeConfigPatch {
                    model: Some(handle),
                    ..Default::default()
                },
            )
        }
        None => {
            println!("(remote) unknown model {selector:?}; see /model list");
            (None, ServeConfigPatch::default())
        }
    }
}

/// Handle `/effort` in serve-chat (always consumes the line; the switch
/// applies to the next turn).
fn run_serve_effort_command(
    arg: &str,
    selection: &mut ServeTurnSelection,
    editor: &mut PromptEditor,
) -> ServeConfigPatch {
    let arg = arg.trim();
    if arg.is_empty() {
        println!("(remote) effort: {}", selection.effective_effort_label());
        return ServeConfigPatch::default();
    }
    match normalize_serve_effort_arg(arg) {
        Ok(override_value) => {
            selection.reasoning_effort = override_value.clone();
            selection.apply_header(editor);
            println!("(remote) effort: {}", selection.effective_effort_label());
            // Persist the pick server-side: a blank value clears the stored
            // effort (`/effort auto`), a tier or "off" stores it.
            ServeConfigPatch {
                reasoning_effort: Some(override_value.unwrap_or_default()),
                ..Default::default()
            }
        }
        Err(err) => {
            println!("(remote) {err}");
            ServeConfigPatch::default()
        }
    }
}

/// Handle `/agent` in serve-chat. Mirrors the local `/agent` branches
/// (`driver/commands/agent.rs`): bare/list/current/help plus `<name>` (an
/// agent-implied model also moves the model selection, like
/// `activate_primary_agent` does locally). Non-primary and disabled agents
/// are refused exactly like the local `switch_agent`.
fn run_serve_agent_command(
    arg: &str,
    selection: &mut ServeTurnSelection,
    agent_manifests: &[agents::AgentManifest],
    editor: &mut PromptEditor,
) -> ServeConfigPatch {
    let arg = arg.trim();
    if arg.is_empty() || arg.eq_ignore_ascii_case("list") || arg.eq_ignore_ascii_case("ls") {
        let primary = agents::get_primary_agents(agent_manifests);
        for agent in &primary {
            let marker = if agent.name == selection.agent {
                "*"
            } else {
                " "
            };
            println!("{marker} {} - {}", agent.name, agent.description);
        }
        return ServeConfigPatch::default();
    }
    if arg.eq_ignore_ascii_case("current") || arg.eq_ignore_ascii_case("cur") {
        println!("(remote) current agent: {}", selection.agent);
        return ServeConfigPatch::default();
    }
    if arg.eq_ignore_ascii_case("help") || arg.eq_ignore_ascii_case("h") {
        println!("(remote) /agent <name> - switch remote agent (primary only)");
        return ServeConfigPatch::default();
    }
    if arg.eq_ignore_ascii_case("reload") {
        println!("(remote) agent reload is server-side; reconnect to refresh");
        return ServeConfigPatch::default();
    }
    match agents::find_agent_by_name(agent_manifests, arg) {
        Some(manifest) => {
            if !manifest.is_primary() {
                println!("(remote) agent {:?} is not switchable", manifest.name);
                return ServeConfigPatch::default();
            }
            if manifest.disabled {
                println!("(remote) agent {:?} is disabled", manifest.name);
                return ServeConfigPatch::default();
            }
            selection.agent = manifest.name.clone();
            let mut patch = ServeConfigPatch {
                agent: Some(manifest.name.clone()),
                ..Default::default()
            };
            if let Some(model) = manifest.model.as_deref().filter(|m| !m.trim().is_empty()) {
                let resolved = models::determine_model(model);
                selection.model = resolved.clone();
                patch.model = Some(resolved);
            }
            selection.apply_header(editor);
            println!("(remote) switched agent to {}", manifest.name);
            patch
        }
        None => {
            println!("(remote) unknown agent {arg:?}; see /agent list");
            ServeConfigPatch::default()
        }
    }
}

/// One entry of `GET /skills`: only name + description matter to the client
/// (unknown manifest fields are ignored by serde).
#[derive(Debug, Deserialize)]
struct ServeSkillEntry {
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
}

/// Fetch the server's skill list: the names a remote turn can actually force.
/// Errors carry the cause; callers surface them with the `(remote)` prefix.
fn fetch_serve_skills(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
) -> Result<Vec<ServeSkillEntry>, Box<dyn std::error::Error>> {
    let resp = auth(client.get(format!("{base}/skills")), token).send()?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().unwrap_or_default();
        return Err(server_error_message(status, &text).into());
    }
    let entries: Vec<ServeSkillEntry> = resp.json()?;
    Ok(entries
        .into_iter()
        .filter(|e| !e.name.trim().is_empty())
        .collect())
}

/// Push the server skill list into Tab completion so `/skills` and `@skills`
/// suggest remote names. Must run on every successful fetch: completion reads
/// only this cache in serve-chat and never falls back to local disk.
fn apply_serve_skills_to_completion(entries: &[ServeSkillEntry]) {
    CommandCompleter::set_skill_name_descriptions(
        entries
            .iter()
            .map(|entry| (entry.name.clone(), entry.description.clone()))
            .collect(),
    );
}

/// Best-effort skill refresh before each input box (short timeout, errors
/// swallowed): keeps `/skills` completion fresh when server manifests change.
/// Failures keep the previous cache; the prompt is never blocked by this.
fn refresh_serve_skills_quiet(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
) -> Option<Vec<ServeSkillEntry>> {
    let resp = auth(client.get(format!("{base}/skills")), token)
        .timeout(Duration::from_secs(2))
        .send()
        .ok()
        .filter(|resp| resp.status().is_success())?;
    let entries: Vec<ServeSkillEntry> = resp.json().ok()?;
    Some(
        entries
            .into_iter()
            .filter(|entry| !entry.name.trim().is_empty())
            .collect(),
    )
}

/// Print the server skill list, marking the pending next-turn selection.
fn print_serve_skills_list(entries: &[ServeSkillEntry], selected: &[String]) {
    if entries.is_empty() {
        println!("(remote) no skills available on server");
        return;
    }
    for entry in entries {
        let marker = if selected.iter().any(|n| n == &entry.name) {
            "*"
        } else {
            " "
        };
        let desc = entry.description.trim();
        if desc.is_empty() {
            println!("{marker} {}", entry.name);
        } else {
            println!("{marker} {}  · {desc}", entry.name);
        }
    }
}

/// Split a `/skills <names...> [question]` remainder against the server skill
/// list: greedily collect consecutive leading tokens that match a skill name
/// (case-insensitive, canonicalized to the manifest spelling, deduped, input
/// order kept); the rest becomes the turn question, verbatim. Pure:
/// unit-tested below.
fn split_serve_skills_inline(
    known: &[ServeSkillEntry],
    arg: &str,
) -> (Vec<String>, Option<String>) {
    let mut names: Vec<String> = Vec::new();
    let mut rest = arg.trim_start();
    loop {
        let token = rest.split_whitespace().next().unwrap_or("");
        if token.is_empty() {
            break;
        }
        let hit = known.iter().find(|e| e.name.eq_ignore_ascii_case(token));
        match hit {
            Some(entry) => {
                if !names.iter().any(|n| n == &entry.name) {
                    names.push(entry.name.clone());
                }
                rest = rest[token.len()..].trim_start();
            }
            None => break,
        }
    }
    let question = if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    };
    (names, question)
}

/// Split the multi-line `/skills` form: with a newline in the remainder the
/// first line is the selection line and everything after it is the question, so
/// every token on that line must name a server skill. Returns `None` for the
/// single-line form (handled by `split_serve_skills_inline`) and for an explicit
/// `use <names>` head, whose branch owns the whole line. The third element lists
/// tokens that matched no skill: reporting them beats the implicit split
/// silently demoting a misspelled name to the first words of the question.
fn split_serve_skills_multiline(
    known: &[ServeSkillEntry],
    arg: &str,
) -> Option<(Vec<String>, Option<String>, Vec<String>)> {
    let (head, tail) = arg.split_once('\n')?;
    if head
        .split_whitespace()
        .next()
        .is_some_and(|token| token.eq_ignore_ascii_case("use"))
    {
        return None;
    }
    let mut names: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for token in head.split_whitespace() {
        match known.iter().find(|e| e.name.eq_ignore_ascii_case(token)) {
            Some(entry) => {
                if !names.contains(&entry.name) {
                    names.push(entry.name.clone());
                }
            }
            None => missing.push(token.to_string()),
        }
    }
    let question = tail.trim();
    let question = (!question.is_empty()).then(|| question.to_string());
    Some((names, question, missing))
}

/// Handle `/skills` in serve-chat. Mirrors the local `/skills` branches
/// (`driver/commands/skills.rs`): bare/`list` prints the server skill list,
/// `current` shows the pending next-turn selection, `use <names>` pins names
/// for the next turn, `clear` drops the pin, and the implicit
/// `/skills <names...> [question]` form selects and optionally sends
/// immediately. Returns a pending prompt for the inline-question form.
/// Validation runs against the server list: only the server knows which
/// names its own manifests can force.
fn run_serve_skills_command(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    arg: &str,
    selection: &mut ServeTurnSelection,
) -> Option<String> {
    let arg = arg.trim();
    if arg.eq_ignore_ascii_case("current") || arg.eq_ignore_ascii_case("cur") {
        if selection.skills.is_empty() {
            println!("(remote) no skills selected for next turn");
        } else {
            println!(
                "(remote) skills for next turn: {}",
                selection.skills.join(", ")
            );
        }
        return None;
    }
    if arg.eq_ignore_ascii_case("help") || arg.eq_ignore_ascii_case("h") {
        println!(
            "(remote) /skills [list|current|use <name>...|clear|<names...> [question]] - force server skills for the next turn"
        );
        return None;
    }
    if arg.eq_ignore_ascii_case("clear") {
        selection.skills.clear();
        println!("(remote) cleared skills for next turn");
        return None;
    }
    let entries = match fetch_serve_skills(client, base, token) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("[serve-chat] {err}");
            return None;
        }
    };
    apply_serve_skills_to_completion(&entries);
    if arg.is_empty() || arg.eq_ignore_ascii_case("list") || arg.eq_ignore_ascii_case("ls") {
        print_serve_skills_list(&entries, &selection.skills);
        return None;
    }
    if arg.eq_ignore_ascii_case("use") {
        println!("(remote) Usage: /skills use <skill-name> [<skill-name>...]");
        print_serve_skills_list(&entries, &selection.skills);
        return None;
    }
    if let Some((names, question, missing)) = split_serve_skills_multiline(&entries, arg) {
        if !missing.is_empty() {
            println!(
                "(remote) unknown skill(s): {}; see /skills list",
                missing.join(", ")
            );
            return None;
        }
        if names.is_empty() {
            println!("(remote) no skill names on the selection line; see /skills list");
            return None;
        }
        selection.skills = names.clone();
        println!(
            "(remote) skills selected for next turn: {}",
            names.join(", ")
        );
        return question;
    }
    // Explicit `use <names>`: every token must name a server skill (unknown
    // names abort without touching the pending selection, like the local
    // command). The implicit form below instead splits names from a trailing
    // question.
    let (names, question) = match arg.split_once(char::is_whitespace) {
        Some((head, tail)) if head.eq_ignore_ascii_case("use") => {
            let mut names: Vec<String> = Vec::new();
            let mut missing: Vec<String> = Vec::new();
            for token in tail.split_whitespace() {
                match entries.iter().find(|e| e.name.eq_ignore_ascii_case(token)) {
                    Some(entry) => {
                        if !names.iter().any(|n| n == &entry.name) {
                            names.push(entry.name.clone());
                        }
                    }
                    None => missing.push(token.to_string()),
                }
            }
            if !missing.is_empty() {
                println!(
                    "(remote) unknown skill(s): {}; see /skills list",
                    missing.join(", ")
                );
                return None;
            }
            (names, None)
        }
        _ => split_serve_skills_inline(&entries, arg),
    };
    if names.is_empty() {
        println!("(remote) unknown /skills subcommand: {arg:?}; see /skills list");
        return None;
    }
    selection.skills = names.clone();
    println!(
        "(remote) skills selected for next turn: {}",
        names.join(", ")
    );
    question
}

/// Best-effort listing refresh before each input box (short timeout, errors
/// swallowed): picks up server-generated titles after a turn and keeps
/// `/resume` completion fresh. Never disturbs the prompt on failure.
fn refresh_listing_quiet(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
) -> Option<Vec<RemoteSession>> {
    let resp = auth(client.get(format!("{base}/sessions")), token)
        .timeout(Duration::from_secs(2))
        .send()
        .ok()
        .filter(|resp| resp.status().is_success())?;
    let text = resp.text().unwrap_or_default();
    parse_remote_sessions(&text).ok()
}

/// Printable one-row title: generated summary first, else the first user
/// prompt's first line, truncated with an ellipsis.
fn remote_session_title(item: &RemoteSession) -> String {
    const MAX_TITLE_CHARS: usize = 60;
    let raw = item
        .summary
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            item.first_user_prompt
                .as_deref()
                .filter(|s| !s.trim().is_empty())
        })
        .unwrap_or("(untitled)");
    let first_line = raw.lines().next().unwrap_or(raw).trim();
    let mut title: String = first_line.chars().take(MAX_TITLE_CHARS).collect();
    if first_line.chars().count() > MAX_TITLE_CHARS {
        title.push('…');
    }
    if item.marked {
        title.push_str(" ★");
    }
    title
}

/// Resolve a `/resume` argument against the last `/sessions` listing:
/// 1-based row number, unique id prefix, or full id. A full id that was
/// never listed (created after the listing, or pasted from elsewhere) passes
/// through verbatim; the server validates it on the next turn.
fn resolve_session_ref(arg: &str, listed: &[RemoteSession]) -> Result<String, String> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Err("usage: /resume <number|id-prefix|id> (see /sessions)".to_string());
    }
    if let Ok(n) = arg.parse::<usize>() {
        return listed
            .get(n.wrapping_sub(1))
            .map(|s| s.id.clone())
            .ok_or_else(|| {
                if listed.is_empty() {
                    "no session listing yet; run /sessions first".to_string()
                } else {
                    format!("session number out of range (1-{})", listed.len())
                }
            });
    }
    let mut prefix_hits = listed.iter().filter(|s| s.id.starts_with(arg));
    match (prefix_hits.next(), prefix_hits.next()) {
        (Some(one), None) => Ok(one.id.clone()),
        (Some(_), Some(_)) => Err(format!(
            "id prefix {arg:?} matches several listed sessions; use more characters"
        )),
        (None, _) => Ok(arg.to_string()),
    }
}

fn normalize_base(bind: &str) -> Result<String, String> {
    let trimmed = bind.trim().trim_end_matches('/');
    let candidate = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let url = reqwest::Url::parse(&candidate)
        .map_err(|err| format!("invalid serve URL {candidate:?}: {err}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(format!(
            "serve URL must use http:// or https:// and include a host: {candidate:?}"
        ));
    }
    if url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
        return Err(format!(
            "serve URL must not include a path, query, or fragment: {candidate:?}"
        ));
    }
    Ok(candidate)
}

fn build_serve_client(
    ca_cert: &str,
    timeout: Duration,
) -> Result<reqwest::blocking::Client, String> {
    let mut builder = reqwest::blocking::Client::builder().timeout(timeout);
    let ca_cert = ca_cert.trim();
    if !ca_cert.is_empty() {
        let path = PathBuf::from(expanduser(ca_cert).as_ref());
        let pem = std::fs::read(&path)
            .map_err(|err| format!("cannot read serve CA certificate {}: {err}", path.display()))?;
        let certificate = reqwest::Certificate::from_pem(&pem)
            .map_err(|err| format!("invalid PEM serve CA certificate {}: {err}", path.display()))?;
        builder = builder.add_root_certificate(certificate);
    }
    builder
        .build()
        .map_err(|err| format!("cannot build serve HTTP client: {err}"))
}

fn resolve_serve_ca_cert(explicit: &str, configured: &str) -> String {
    if explicit.trim().is_empty() {
        configured.trim().to_string()
    } else {
        explicit.trim().to_string()
    }
}

fn auth(req: reqwest::blocking::RequestBuilder, token: &str) -> reqwest::blocking::RequestBuilder {
    if token.trim().is_empty() {
        req
    } else {
        req.bearer_auth(token.trim())
    }
}

fn server_error_message(status: reqwest::StatusCode, text: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(v) => v
            .get("error")
            .and_then(|e| e.as_str())
            .map(|e| format!("serve returned {status}: {e}"))
            .unwrap_or_else(|| format!("serve returned {status}: {text}")),
        Err(_) => {
            let clipped: String = text.chars().take(500).collect();
            format!("serve returned {status}: {clipped}")
        }
    }
}

fn create_session(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let resp = auth(client.post(format!("{base}/sessions")), token).send()?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text).into());
    }
    let created: CreatedSession = serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected session body: {err}"))?;
    if created.id.trim().is_empty() {
        return Err("serve returned an empty session id".into());
    }
    Ok(created.id)
}

/// Fork the current remote session (`POST /sessions/{id}/fork`). Returns the
/// new session id; the forked copy is title-marked server-side, same as the
/// local `/fork`. Network glue only — the loopback test below covers the
/// wire shape.
fn fork_remote_session(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let resp = auth(
        client.post(format!("{base}/sessions/{session_id}/fork")),
        token,
    )
    .send()?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text).into());
    }
    let created: CreatedSession = serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected fork body: {err}"))?;
    if created.id.trim().is_empty() {
        return Err("serve returned an empty fork session id".into());
    }
    Ok(created.id)
}

#[derive(Debug, Deserialize)]
struct DeletedSession {
    #[serde(default)]
    deleted: bool,
}

/// Delete a remote session (`DELETE /sessions/{id}`). Idempotent: deleting a
/// missing session still succeeds, so a retry after a dropped connection is
/// safe. Returns the server-reported `deleted` flag.
fn delete_remote_session(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let resp = auth(
        client.delete(format!("{base}/sessions/{session_id}")),
        token,
    )
    .send()?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text).into());
    }
    let deleted: DeletedSession = serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected delete body: {err}"))?;
    Ok(deleted.deleted)
}

// --- Remote `/history`: previews render from `GET /sessions/{id}/history`
// and rewinds truncate through `POST /sessions/{id}/rewind`, never through a
// turn child. A turn child would dispatch `/history` as a local command
// against server-side state: read-only forms only waste a model round-trip,
// but `rewind` blocks on an interactive stdin confirm that has no writer on
// the daemon side and hangs the session until restart. ---

/// One row of `GET /sessions/{id}/history`, which serves the canonical
/// message array 1:1, so an index here is a valid `POST /rewind`
/// `message_index`. Unknown fields are ignored so a newer server never breaks
/// this client; `content` is a string for plain messages and structured
/// blocks otherwise.
#[derive(Debug, Deserialize)]
struct RemoteHistoryMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: serde_json::Value,
    #[serde(default)]
    tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    reasoning_content: Option<String>,
    /// Global real-user ordinal attached by the server on filtered view fetches
    /// (`?role`/`?count`), so `u<N>` markers stay exact on a partial
    /// response. Absent on full fetches and older servers; the renderer
    /// falls back to counting locally then.
    #[serde(default)]
    user_ordinal: Option<usize>,
}

const SERVE_HISTORY_DEFAULT_COUNT: usize = 6;
const SERVE_HISTORY_FULL_COUNT: usize = 20;
const SERVE_HISTORY_MAX_CHARS: usize = 160;

/// Fetch the session's full canonical history. No `limit`: user ordinals
/// (`u40`) count every stored user message, so a tail cut would renumber the
/// rows and invalidate rewind anchors. Network glue only; shaping below is
/// pure and tested offline.
fn fetch_remote_history(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
) -> Result<Vec<RemoteHistoryMessage>, String> {
    let resp = auth(
        client.get(format!("{base}/sessions/{session_id}/history")),
        token,
    )
    .send()
    .map_err(|err| format!("history request failed: {err}"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected history body: {err}"))
}

/// Fetch only the per-message roles (`GET /sessions/{id}/history?meta=roles`):
/// a bare array of role strings indexed by canonical position. Rewind
/// ordinal resolution and user-turn counting need nothing else, so the
/// session payload (tool outputs, attachment text, reasoning content) never
/// crosses the wire for `u<N>` rewinds. A server that does not know
/// `meta=roles` returns the full message body, which fails to parse as a
/// role array and surfaces loudly here instead of misresolving ordinals.
fn fetch_remote_history_roles(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
) -> Result<Vec<String>, String> {
    let resp = auth(
        client
            .get(format!("{base}/sessions/{session_id}/history"))
            .query(&[("meta", "roles")]),
        token,
    )
    .send()
    .map_err(|err| format!("history roles request failed: {err}"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected history roles body: {err}"))
}

/// Fetch the bounded preview subset for a view (`?role` + `?count`): the
/// server filters by role and tail-cuts, so only the rows the client will
/// show travel, and it attaches each user row's global `user_ordinal` for
/// exact `u<N>` markers. `grep` views need content scanning, so they keep
/// the full fetch.
fn fetch_remote_history_view(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    view: &ServeHistoryView,
) -> Result<Vec<RemoteHistoryMessage>, String> {
    let mut query: Vec<(&str, String)> = Vec::new();
    if let Some(role) = serve_view_role_param(view.role) {
        query.push(("role", role.to_string()));
    }
    query.push(("count", view.count.to_string()));
    let resp = auth(
        client
            .get(format!("{base}/sessions/{session_id}/history"))
            .query(&query),
        token,
    )
    .send()
    .map_err(|err| format!("history request failed: {err}"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected history body: {err}"))
}

/// Fetch the N-th-most-recent assistant conclusion (`?conclusion=N`): a
/// one-row body the server selected with the same scan rule as the local
/// `/history last`. Not-found surfaces the server's exact error wording
/// verbatim (matching what the local command prints) instead of a transport
/// wrapper.
fn fetch_serve_history_conclusion(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    nth_back: usize,
) -> Result<Vec<RemoteHistoryMessage>, String> {
    fetch_serve_history_selected_conclusion(client, base, token, session_id, "conclusion", nth_back)
}

fn fetch_serve_history_user_conclusion(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    user_ordinal: usize,
) -> Result<Vec<RemoteHistoryMessage>, String> {
    fetch_serve_history_selected_conclusion(
        client,
        base,
        token,
        session_id,
        "user_conclusion",
        user_ordinal,
    )
}

fn fetch_serve_history_selected_conclusion(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    query_key: &str,
    ordinal: usize,
) -> Result<Vec<RemoteHistoryMessage>, String> {
    let resp = auth(
        client
            .get(format!("{base}/sessions/{session_id}/history"))
            .query(&[(query_key, ordinal.to_string())]),
        token,
    )
    .send()
    .map_err(|err| format!("history request failed: {err}"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        if let Ok(body) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(detail) = body.get("error").and_then(|e| e.as_str()) {
                return Err(detail.to_string());
            }
        }
        return Err(server_error_message(status, &text));
    }
    serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected history body: {err}"))
}

/// Searchable text of one stored message, mirroring the local preview:
/// plain strings verbatim, anything else as compact JSON.
fn serve_history_searchable_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(text) => text.clone(),
        other => {
            serde_json::to_string(other).unwrap_or_else(|_| "<non-string content>".to_string())
        }
    }
}

/// Char-based truncation with the same `...` suffix as the local preview.
fn serve_truncate_for_terminal(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = String::new();
    for (idx, ch) in value.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn is_remote_real_user_message(message: &RemoteHistoryMessage) -> bool {
    let message = crate::ai::history::Message {
        role: message.role.clone(),
        content: serde_json::Value::Null,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: message.reasoning_content.clone(),
    };
    message.role == "user" && !crate::ai::history::is_runtime_synthetic_user_message(&message)
}

/// Canonical (0-based) index of the Nth stored real user message.
fn serve_user_message_index(messages: &[RemoteHistoryMessage], ordinal: usize) -> Option<usize> {
    real_user_index(messages, ordinal, is_remote_real_user_message)
}

fn serve_user_message_count(messages: &[RemoteHistoryMessage]) -> usize {
    count_real_users(messages, is_remote_real_user_message)
}

/// Role-array counterparts of the message-based helpers above: the
/// `?meta=roles` response carries no message bodies, only one role string
/// per canonical row. Runtime-synthetic users carry a distinct role sentinel,
/// so only literal `user` rows participate in ordinals.
fn serve_role_user_message_index(roles: &[String], ordinal: usize) -> Option<usize> {
    real_user_index(roles, ordinal, |role| role == "user")
}

fn serve_role_user_message_count(roles: &[String]) -> usize {
    count_real_users(roles, |role| role == "user")
}

/// Wire value for a view's role filter, mirroring `parse_serve_history_view`
/// role set; `All` sends no filter so the server keeps the unfiltered shape.
fn serve_view_role_param(role: ServeHistoryRole) -> Option<&'static str> {
    match role {
        ServeHistoryRole::All => None,
        ServeHistoryRole::User => Some("user"),
        ServeHistoryRole::Assistant => Some("assistant"),
        ServeHistoryRole::Tool => Some("tool"),
        ServeHistoryRole::System => Some("system"),
    }
}

/// Read-only view selector for the remote `/history` preview. Mirrors the
/// local preview grammar (`[N]`, `full`, role words, `grep <keyword>`);
/// `export`/`copy`/other-session stay local-only and report how to get the
/// same result from here.
#[derive(Debug, PartialEq, Eq)]
struct ServeHistoryView {
    count: usize,
    role: ServeHistoryRole,
    full: bool,
    grep: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum ServeHistoryRole {
    All,
    User,
    Assistant,
    Tool,
    System,
}

#[derive(Debug, PartialEq, Eq)]
enum ServeHistoryAction {
    Show,
    Help,
}

fn parse_serve_history_view(
    args: &[&str],
) -> Result<(ServeHistoryView, ServeHistoryAction), String> {
    let mut view = ServeHistoryView {
        count: SERVE_HISTORY_DEFAULT_COUNT,
        role: ServeHistoryRole::All,
        full: false,
        grep: None,
    };
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx] {
            "full" => {
                view.full = true;
                view.count = SERVE_HISTORY_FULL_COUNT;
                idx += 1;
            }
            "user" => {
                view.role = ServeHistoryRole::User;
                idx += 1;
            }
            "assistant" => {
                view.role = ServeHistoryRole::Assistant;
                idx += 1;
            }
            "tool" => {
                view.role = ServeHistoryRole::Tool;
                idx += 1;
            }
            "system" => {
                view.role = ServeHistoryRole::System;
                idx += 1;
            }
            "grep" => {
                let keyword = args[idx + 1..].join(" ");
                if keyword.trim().is_empty() {
                    return Err("`/history grep` requires a keyword".into());
                }
                view.grep = Some(keyword);
                break;
            }
            "export" => {
                return Err("`/history export` would write to the server machine, not this one; print the view with `/history` and save the output locally."
                    .into());
            }
            "copy" => {
                return Err("`/history copy` cannot reach this machine's clipboard from the server; print the view with `/history` and copy it locally."
                    .into());
            }
            "help" => {
                return Ok((view, ServeHistoryAction::Help));
            }
            raw => match raw.parse::<usize>() {
                Ok(count) => {
                    view.count = count.max(1);
                    idx += 1;
                }
                Err(_) if idx == 0 => {
                    return Err(format!(
                        "`/history {raw}` addresses another session, which serve-chat does not support; `/resume {raw}` first, then use `/history`."
                    ));
                }
                Err(_) => {
                    return Err(format!("invalid /history argument: {raw}"));
                }
            },
        }
    }
    Ok((view, ServeHistoryAction::Show))
}

/// Render the preview, mirroring the local `[history] Showing N recent ...`
/// shape. No model tags: the history route serves bare messages without the
/// sqlite `source_model` provenance the local preview reads.
fn render_serve_history(messages: &[RemoteHistoryMessage], view: &ServeHistoryView) -> String {
    let label = match view.role {
        ServeHistoryRole::All => "message(s)",
        ServeHistoryRole::User => "user message(s)",
        ServeHistoryRole::Assistant => "assistant message(s)",
        ServeHistoryRole::Tool => "tool message(s)",
        ServeHistoryRole::System => "system message(s)",
    };
    let grep_suffix = view
        .grep
        .as_deref()
        .map(|grep| format!(" matching \"{grep}\""))
        .unwrap_or_default();
    // Ordinals count real user turns before filtering, like local.
    let mut user_ordinal = 0usize;
    let mut filtered = Vec::new();
    for message in messages {
        // The server attaches each user row's global ordinal on filtered
        // fetches; fall back to local counting on full fetches (grep, older
        // servers), where every user row is present anyway.
        let is_real_user = is_remote_real_user_message(message);
        let ordinal = if is_real_user {
            user_ordinal += 1;
            message.user_ordinal.or(Some(user_ordinal))
        } else {
            None
        };
        let role_matches = match view.role {
            ServeHistoryRole::All => true,
            ServeHistoryRole::User => is_real_user,
            ServeHistoryRole::Assistant => message.role == "assistant",
            ServeHistoryRole::Tool => message.role == "tool",
            // Same rule as the local system filter (system + internal notes).
            ServeHistoryRole::System => message.role == "system" || message.role == "internal_note",
        };
        if !role_matches {
            continue;
        }
        if let Some(needle) = view.grep.as_deref() {
            let haystack = serve_history_searchable_text(&message.content).to_ascii_lowercase();
            if !haystack.contains(&needle.to_ascii_lowercase()) {
                continue;
            }
        }
        filtered.push((message, ordinal));
    }
    let shown = if filtered.len() > view.count {
        &filtered[filtered.len() - view.count..]
    } else {
        &filtered[..]
    };
    if shown.is_empty() {
        return "[history] No recent messages.".to_string();
    }
    let mut out = format!(
        "[history] Showing {} recent {}{}:\n",
        shown.len(),
        label,
        grep_suffix
    );
    for (row, item) in shown.iter().enumerate() {
        let (message, ordinal) = *item;
        let content = if view.full {
            serve_history_searchable_text(&message.content)
        } else {
            serve_truncate_for_terminal(
                &serve_history_searchable_text(&message.content)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" "),
                SERVE_HISTORY_MAX_CHARS,
            )
        };
        let marker = ordinal.map(|n| format!(" (u{n})")).unwrap_or_default();
        out.push_str(&format!(
            "{}. [{}] {}{}\n",
            row + 1,
            message.role,
            content,
            marker
        ));
    }
    out.trim_end().to_string()
}

/// Render the `nth_back`-most-recent (1 = latest) assistant conclusion, full
/// text. Mirrors the local conclusion rule (no tool calls, non-empty text);
/// the model tag is unavailable remotely, so the header carries only the
/// position.
fn render_serve_history_last(
    messages: &[RemoteHistoryMessage],
    nth_back: usize,
) -> Result<String, String> {
    let mut seen = 0usize;
    let mut found = None;
    for message in messages.iter().rev() {
        if message.role != "assistant" {
            continue;
        }
        if !message
            .tool_calls
            .as_ref()
            .is_none_or(|calls| calls.is_empty())
        {
            continue;
        }
        let text = serve_history_searchable_text(&message.content);
        if text.trim().is_empty() {
            continue;
        }
        seen += 1;
        if seen == nth_back {
            found = Some(text);
            break;
        }
    }
    let Some(text) = found else {
        if seen == 0 {
            return Err("[history] No assistant messages yet.".into());
        }
        return Err(format!(
            "[history] Only {seen} assistant message(s) in history."
        ));
    };
    Ok(format!("{}\n{}", serve_history_last_header(nth_back), text))
}

/// Header for the Nth-back assistant conclusion (1 = latest).
fn serve_history_last_header(nth_back: usize) -> String {
    if nth_back == 1 {
        "[history] Latest assistant message:".to_string()
    } else {
        format!("[history] Assistant message {nth_back} back from latest:")
    }
}

/// Print the Nth recent assistant conclusion like local `/history last`:
/// the header stays plain text while the body goes through the same
/// display-only post-processing and terminal markdown renderer as a live
/// turn. `messages` is the one-row server result (`?conclusion=N`), so the
/// header uses the requested `nth_back` rather than the subset position;
/// `render_serve_history_last` stays pure for tests and `/history replay`.
fn print_serve_history_last_rendered(
    messages: &[RemoteHistoryMessage],
    nth_back: usize,
) -> Result<(), String> {
    print_serve_history_selected_rendered(messages, serve_history_last_header(nth_back))
}

fn print_serve_history_user_reply_rendered(
    messages: &[RemoteHistoryMessage],
    user_ordinal: usize,
) -> Result<(), String> {
    print_serve_history_selected_rendered(
        messages,
        format!("[history] Assistant reply to user message u{user_ordinal}:"),
    )
}

fn print_serve_history_selected_rendered(
    messages: &[RemoteHistoryMessage],
    header: String,
) -> Result<(), String> {
    let body = serve_history_searchable_text(&messages[0].content);
    let rendered = format!("{header}\n{body}");
    let (header, body) = rendered.split_once('\n').unwrap_or((rendered.as_str(), ""));
    println!("{header}");
    let body = super::super::driver::turn_runtime::postprocess_terminal_text(body.to_string());
    super::super::stream::render_markdown_block(&body)
        .map_err(|err| format!("history render failed: {err}"))?;
    Ok(())
}

/// Resolve a rewind target to its user ordinal and canonical index, mirroring
/// the local resolver's grammar (`u<N>`, bare `N`, `last`/`latest`, single
/// `grep` match) and error wording.
fn resolve_serve_rewind_target(
    messages: &[RemoteHistoryMessage],
    args: &[&str],
) -> Result<(usize, usize), String> {
    match parse_history_rewind_target(args)? {
        HistoryRewindTarget::LatestUser => {
            let count = serve_user_message_count(messages);
            if count == 0 {
                return Err("no user input found in history".into());
            }
            let index =
                serve_user_message_index(messages, count).expect("counted user message resolves");
            Ok((count, index))
        }
        HistoryRewindTarget::Grep(keyword) => {
            let needle = keyword.to_ascii_lowercase();
            let mut matches = Vec::new();
            for (ordinal, index, message) in
                real_user_ordinal_entries(messages, is_remote_real_user_message)
            {
                let text = serve_history_searchable_text(&message.content);
                if text.to_ascii_lowercase().contains(&needle) {
                    matches.push((ordinal, index, message));
                }
            }
            if matches.is_empty() {
                return Err(format!("no user input matching {keyword:?}"));
            }
            if matches.len() > 1 {
                // Same disambiguation shape as local: up to 8 `u<N>` candidates.
                let mut out = format!(
                    "{} user inputs match {keyword:?}; use /history rewind u<N>:",
                    matches.len()
                );
                for (ord, _, message) in matches.iter().take(8) {
                    let preview = serve_truncate_for_terminal(
                        &serve_history_searchable_text(&message.content)
                            .split_whitespace()
                            .collect::<Vec<_>>()
                            .join(" "),
                        120,
                    );
                    out.push_str(&format!("\n  u{ord} {preview}"));
                }
                return Err(out);
            }
            let (ord, index, _) = matches[0];
            Ok((ord, index))
        }
        HistoryRewindTarget::UserOrdinal(ordinal) => serve_user_message_index(messages, ordinal)
            .map(|index| (ordinal, index))
            .ok_or_else(|| format!("user input u{ordinal} not found")),
    }
}

/// Same resolution grammar as `resolve_serve_rewind_target`, but over the
/// roles-only history shape (`u<N>`, bare `N`, `last`/`latest`). `grep`
/// needs message content, so it stays on the full-message resolver; callers
/// branch on the first verb before fetching anything, and this defensive
/// arm keeps the two resolvers' contracts separate.
fn resolve_serve_rewind_target_from_roles(
    roles: &[String],
    args: &[&str],
) -> Result<(usize, usize), String> {
    if args.first().copied() == Some("grep") {
        return Err(
            "`/history rewind grep` needs message content; the full-history path resolves it"
                .into(),
        );
    }
    match parse_history_rewind_target(args)? {
        HistoryRewindTarget::LatestUser => {
            let count = serve_role_user_message_count(roles);
            if count == 0 {
                return Err("no user input found in history".into());
            }
            let index =
                serve_role_user_message_index(roles, count).expect("counted user message resolves");
            Ok((count, index))
        }
        HistoryRewindTarget::Grep(_) => Err(
            "`/history rewind grep` needs message content; the full-history path resolves it"
                .into(),
        ),
        HistoryRewindTarget::UserOrdinal(ordinal) => serve_role_user_message_index(roles, ordinal)
            .map(|index| (ordinal, index))
            .ok_or_else(|| format!("user input u{ordinal} not found")),
    }
}

/// `POST /sessions/{id}/rewind` result: `{"removed": N, "kept": M}`.
#[derive(Debug, Deserialize, PartialEq, Eq)]
struct RemoteRewindResult {
    #[serde(default)]
    removed: usize,
    #[serde(default)]
    kept: usize,
}

fn post_remote_rewind(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    message_index: usize,
) -> Result<RemoteRewindResult, String> {
    let resp = auth(
        client.post(format!("{base}/sessions/{session_id}/rewind")),
        token,
    )
    .json(&serde_json::json!({"message_index": message_index}))
    .send()
    .map_err(|err| format!("rewind request failed: {err}"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected rewind body: {err}"))
}

/// POST a `/cd`/`:cd` line to the server and render the captured result.
///
/// The switch is recorded per session on the server (`POST
/// /sessions/{id}/cd`), and every turn child of that session is spawned with
/// that directory as its real cwd, so a child-local `/cd` would die with the
/// child. The printed path is the remote session's working directory; this
/// client's own directory is never touched.
fn run_serve_cd_command(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    command: &str,
) -> Result<(), String> {
    let resp = auth(
        client.post(format!("{base}/sessions/{session_id}/cd")),
        token,
    )
    .header("Content-Type", "application/json")
    .body(serde_json::json!({ "command": command }).to_string())
    .send()
    .map_err(|err| format!("cd request failed: {err}"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    let body: serde_json::Value = serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected cd body: {err}"))?;
    if let Some(stdout) = body.get("stdout").and_then(|v| v.as_str()) {
        print!("{stdout}");
    }
    if let Some(stderr) = body.get("stderr").and_then(|v| v.as_str()) {
        eprint!("{stderr}");
    }
    Ok(())
}

fn run_serve_subagent_config_request(
    request: reqwest::blocking::RequestBuilder,
) -> Result<(), String> {
    let response = request
        .send()
        .map_err(|err| format!("subagent config request failed: {err}"))?;
    let status = response.status();
    let text = response.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    let body: serde_json::Value = serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected config body: {err}"))?;
    let output = body
        .get("output")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "serve config response is missing output".to_string())?;
    print!("{output}");
    Ok(())
}

fn run_serve_use_command(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    arg: &str,
) -> Result<(), String> {
    let command_line = arg.split_once('\n').map_or(arg, |(head, _)| head);
    let (backend, model) = parse_use_arguments(command_line)?;
    let url = format!("{base}/sessions/{session_id}/config");
    let request = match backend {
        Some(backend) => client.post(url).json(&serde_json::json!({
            "subagents": {"action": "use", "backend": backend, "model": model},
        })),
        None => client.get(url),
    };
    run_serve_subagent_config_request(auth(request, token))
}

fn run_serve_reset_command(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    arg: &str,
) -> Result<(), String> {
    validate_reset_arguments(arg).map_err(str::to_string)?;
    let request = client
        .post(format!("{base}/sessions/{session_id}/config"))
        .json(&serde_json::json!({"subagents": {"action": "reset"}}));
    run_serve_subagent_config_request(auth(request, token))
}

/// Preserve the child input while extracting title intent from nested commands and skills.
fn serve_use_turn_input(input: &str) -> Option<(&str, String)> {
    let tail = input.trim().split_once('\n')?.1.trim();
    if tail.is_empty() {
        return None;
    }
    Some((tail, serve_command_title_seed(input).unwrap_or_default()))
}

/// Fetch `/cd` Tab-completion candidates for `fragment` from the remote
/// server: directories only, `dir/`-suffixed, in the exact shape the local
/// completer would produce against the server's filesystem.
fn fetch_remote_cd_candidates(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    fragment: &str,
) -> Result<Vec<String>, String> {
    let resp = auth(
        client.get(format!("{base}/sessions/{session_id}/cd-candidates")),
        token,
    )
    .query(&[("frag", fragment)])
    .send()
    .map_err(|err| format!("cd completion request failed: {err}"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    let body: serde_json::Value = serde_json::from_str(&text)
        .map_err(|err| format!("serve returned an unexpected cd-candidates body: {err}"))?;
    Ok(body
        .get("candidates")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default())
}

/// Install the serve-chat `/cd` completion provider: Tab after `/cd` lists
/// directories on the remote server (`GET /cd-candidates`) instead of this
/// client's local disk, always for the *currently selected* session (the
/// shared cell follows `/resume` `/new` `/fork` switches). Cleared when the
/// chat exits.
fn install_serve_cd_completion(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    current_session: Arc<std::sync::Mutex<String>>,
) {
    let client = client.clone();
    let base = base.to_string();
    let token = token.to_string();
    CommandCompleter::set_remote_cd_candidates_provider(Some(Box::new(move |fragment: &str| {
        let session_id = current_session
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        match fetch_remote_cd_candidates(&client, &base, &token, &session_id, fragment) {
            Ok(candidates) => candidates,
            Err(err) => {
                eprintln!("[serve-chat] cd completion unavailable: {err}");
                Vec::new()
            }
        }
    })));
}

fn parse_history_user_reply_selector(args: &[&str]) -> Result<Option<usize>, String> {
    let Some(first) = args.first().copied() else {
        return Ok(None);
    };
    let Some(raw) = first.strip_prefix('u').or_else(|| first.strip_prefix('U')) else {
        return Ok(None);
    };
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Ok(None);
    }
    if args.len() > 1 {
        return Err("`/history u<N>` takes no additional arguments".into());
    }
    let user_ordinal = raw
        .parse::<usize>()
        .map_err(|_| format!("invalid user message selector: {first}"))?;
    if user_ordinal == 0 {
        return Err("user ordinal must be >= 1".into());
    }
    Ok(Some(user_ordinal))
}

/// Remote `/history` entry: first-token verbs dispatch like the local
/// grammar; everything else is a preview. Prints the result; `Err` carries
/// the detail the caller prefixes with `[serve-chat]`.
fn run_serve_history_command(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    arg: &str,
) -> Result<(), String> {
    let args: Vec<&str> = arg.split_whitespace().collect();
    let Some(first) = args.first().copied() else {
        return show_serve_history(client, base, token, session_id, &[]);
    };
    if let Some(user_ordinal) = parse_history_user_reply_selector(&args)? {
        let messages =
            fetch_serve_history_user_conclusion(client, base, token, session_id, user_ordinal)?;
        print_serve_history_user_reply_rendered(&messages, user_ordinal)?;
        return Ok(());
    }
    match first {
        "help" => {
            print_serve_history_help();
            Ok(())
        }
        "rewind" => rewind_serve_history(client, base, token, session_id, &args[1..]),
        // `/history replay` stays raw text (mirrors local replay, which prints
        // text only so the output can be piped or copied verbatim).
        "replay" => {
            if args.len() > 1 {
                return Err("too many arguments. try: /history replay".into());
            }
            let messages = fetch_serve_history_conclusion(client, base, token, session_id, 1)?;
            println!("{}", render_serve_history_last(&messages, 1)?);
            Ok(())
        }
        // `/history last [N]` renders through the terminal markdown renderer
        // (mirrors local `/history last`, which paints the stored conclusion
        // like a live turn).
        "last" => {
            if args.len() > 2 {
                return Err("too many arguments. try: /history last [N]".into());
            }
            let nth_back = match args.get(1) {
                None => 1,
                Some(raw) => raw.parse::<usize>().map_err(|_| {
                    format!("invalid /history last argument: {raw}. try: /history last [N]")
                })?,
            };
            if nth_back == 0 {
                return Err("invalid /history last argument: 0. try: /history last [N]".into());
            }
            let messages =
                fetch_serve_history_conclusion(client, base, token, session_id, nth_back)?;
            print_serve_history_last_rendered(&messages, nth_back)?;
            Ok(())
        }
        _ => show_serve_history(client, base, token, session_id, &args),
    }
}

fn show_serve_history(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    args: &[&str],
) -> Result<(), String> {
    let (view, action) = parse_serve_history_view(args)?;
    if action == ServeHistoryAction::Help {
        print_serve_history_help();
        return Ok(());
    }
    // `grep` searches content, so it needs the full message list and shapes
    // locally like the local preview; every other view is a bounded tail the
    // server filters and cuts before transfer.
    let messages = if view.grep.is_some() {
        fetch_remote_history(client, base, token, session_id)?
    } else {
        fetch_remote_history_view(client, base, token, session_id, &view)?
    };
    println!("{}", render_serve_history(&messages, &view));
    Ok(())
}

fn rewind_serve_history(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    args: &[&str],
) -> Result<(), String> {
    // Pure-syntax check first: usage errors must surface before any network
    // round trip, so a missing or malformed target is never masked by a
    // transport failure. Range checks still need the fetched messages below.
    check_serve_rewind_syntax(args)?;
    // `u<N>`/`last` resolve from the roles-only shape (no message bodies
    // cross the wire); only `grep` needs full content to search, so it keeps
    // the full-history fetch. Both shapes resolve the same canonical index,
    // so the `POST /rewind` contract is unchanged.
    let first = args.first().copied().unwrap_or("");
    let (ordinal, index, total) = if first == "grep" {
        let messages = fetch_remote_history(client, base, token, session_id)?;
        let (ordinal, index) = resolve_serve_rewind_target(&messages, args)?;
        (ordinal, index, messages.len())
    } else {
        let roles = fetch_remote_history_roles(client, base, token, session_id)?;
        let (ordinal, index) = resolve_serve_rewind_target_from_roles(&roles, args)?;
        (ordinal, index, roles.len())
    };
    let removed = total - index;
    if removed == 0 {
        println!("[history] Nothing to rewind.");
        return Ok(());
    }
    // The confirm runs in this client, on this terminal: the turn child this
    // replaces had nobody to answer it, which is what used to hang.
    let confirm = crate::commonw::prompt::prompt_yes_or_no_interruptible(&format!(
        "Rewind from user input u{ordinal} and remove {removed} message(s)? (y/n): "
    ));
    if confirm != Some(true) {
        println!("canceled by user.");
        return Ok(());
    }
    let result = post_remote_rewind(client, base, token, session_id, index)?;
    println!(
        "[history] Rewound from u{ordinal} and removed {} message(s); kept {} message(s).",
        result.removed, result.kept
    );
    Ok(())
}

/// Syntax-only validation of `/history rewind` args: no message list needed,
/// so callers run this before fetching remote history.
fn check_serve_rewind_syntax(args: &[&str]) -> Result<(), String> {
    parse_history_rewind_target(args).map(|_| ())
}

fn print_serve_history_help() {
    println!(
        "/history usage (remote session):\n  /history [N]           Show last N messages (default: {})\n  /history full          Show full messages\n  /history user/assistant/tool/system\n                         Filter by role\n  /history u<N>          Replay the assistant reply to user message u<N>\n  /history grep <keyword>  Search messages\n  /history last [N]      Replay the Nth recent assistant conclusion (full text)\n  /history replay        Replay the last assistant conclusion (text only)\n  /history rewind u<N>   Remove user message u<N> and everything after it\n  /history rewind last   Remove latest user message and everything after it\n  /history rewind grep <keyword>\n                         Rewind the only user message matching keyword\n  /history help          Show this help\n\n`/history copy`/`/history export` and other-session views are local-only: print the view here and save or copy it locally; `/export [path]` runs on the server. Assistant rows carry no model tag; the history route serves bare messages.",
        SERVE_HISTORY_DEFAULT_COUNT
    );
}

#[cfg(test)]
mod serve_history_tests {
    use super::*;
    use crate::ai::driver::commands::is_serve_local_command_start;
    use serde_json::json;

    fn test_message(role: &str, text: &str) -> RemoteHistoryMessage {
        RemoteHistoryMessage {
            role: role.to_string(),
            content: json!(text),
            tool_calls: None,
            reasoning_content: None,
            user_ordinal: None,
        }
    }

    fn sample_history() -> Vec<RemoteHistoryMessage> {
        vec![
            test_message("system", "sys"),
            test_message("user", "first question"),
            test_message("assistant", "first answer"),
            test_message("user", "second question about meego"),
            test_message("assistant", "second answer"),
        ]
    }

    fn default_view() -> ServeHistoryView {
        ServeHistoryView {
            count: SERVE_HISTORY_DEFAULT_COUNT,
            role: ServeHistoryRole::All,
            full: false,
            grep: None,
        }
    }

    #[test]
    fn serve_history_view_defaults() {
        let (view, action) = parse_serve_history_view(&[]).expect("parse");
        assert_eq!(view, default_view());
        assert_eq!(action, ServeHistoryAction::Show);
    }

    #[test]
    fn serve_history_user_reply_selector_matches_local_syntax() {
        assert_eq!(
            parse_history_user_reply_selector(&["u205"]).unwrap(),
            Some(205)
        );
        assert_eq!(parse_history_user_reply_selector(&["U7"]).unwrap(), Some(7));
        assert_eq!(parse_history_user_reply_selector(&["user"]).unwrap(), None);
        assert!(parse_history_user_reply_selector(&["u0"]).is_err());
        assert!(parse_history_user_reply_selector(&["u2", "full"]).is_err());
    }

    #[test]
    fn serve_history_view_full_roles_and_clamp() {
        let (view, _) = parse_serve_history_view(&["full"]).expect("parse");
        assert!(view.full);
        assert_eq!(view.count, SERVE_HISTORY_FULL_COUNT);
        let (view, _) = parse_serve_history_view(&["assistant", "8"]).expect("parse");
        assert_eq!(view.role, ServeHistoryRole::Assistant);
        assert_eq!(view.count, 8);
        let (view, _) = parse_serve_history_view(&["999"]).expect("parse");
        assert_eq!(view.count, 999);
        let (view, _) = parse_serve_history_view(&["0"]).expect("parse");
        assert_eq!(view.count, 1);
    }

    #[test]
    fn serve_history_view_grep_joins_rest() {
        let (view, _) = parse_serve_history_view(&["user", "grep", "a", "b"]).expect("parse");
        assert_eq!(view.role, ServeHistoryRole::User);
        assert_eq!(view.grep.as_deref(), Some("a b"));
        assert!(parse_serve_history_view(&["grep"]).is_err());
        assert!(parse_serve_history_view(&["grep", "  "]).is_err());
    }

    #[test]
    fn serve_history_view_local_only_forms_error() {
        let err = parse_serve_history_view(&["export"]).expect_err("export");
        assert!(err.contains("save the output locally"), "{err}");
        let err = parse_serve_history_view(&["copy"]).expect_err("copy");
        assert!(err.contains("copy it locally"), "{err}");
    }

    #[test]
    fn serve_history_view_help_and_session_selector() {
        let (_, action) = parse_serve_history_view(&["user", "help"]).expect("parse");
        assert_eq!(action, ServeHistoryAction::Help);
        let err = parse_serve_history_view(&["f830931f"]).expect_err("session");
        assert!(err.contains("/resume f830931f"), "{err}");
        let err = parse_serve_history_view(&["user", "bogus"]).expect_err("invalid");
        assert!(err.contains("invalid /history argument: bogus"), "{err}");
    }

    #[test]
    fn serve_user_ordinals_skip_non_user_rows() {
        let messages = sample_history();
        assert_eq!(serve_user_message_count(&messages), 2);
        assert_eq!(serve_user_message_index(&messages, 0), None);
        assert_eq!(serve_user_message_index(&messages, 1), Some(1));
        assert_eq!(serve_user_message_index(&messages, 2), Some(3));
        assert_eq!(serve_user_message_index(&messages, 3), None);
    }

    #[test]
    fn serve_user_ordinals_skip_runtime_synthetic_rows() {
        let synthetic = crate::ai::history::runtime_synthetic_user_message(json!("follow-up"));
        let messages = vec![
            test_message("user", "first"),
            RemoteHistoryMessage {
                role: synthetic.role,
                content: synthetic.content,
                tool_calls: None,
                reasoning_content: synthetic.reasoning_content,
                user_ordinal: None,
            },
            test_message("assistant", "answer"),
            test_message("user", "second"),
        ];
        assert_eq!(serve_user_message_count(&messages), 2);
        assert_eq!(serve_user_message_index(&messages, 2), Some(3));

        let roles = vec![
            "user".to_string(),
            "runtime_synthetic_user".to_string(),
            "assistant".to_string(),
            "user".to_string(),
        ];
        assert_eq!(serve_role_user_message_count(&roles), 2);
        assert_eq!(serve_role_user_message_index(&roles, 2), Some(3));
    }

    #[test]
    fn serve_rewind_grep_uses_real_user_ordinals() {
        let synthetic =
            crate::ai::history::runtime_synthetic_user_message(json!("synthetic cache note"));
        let messages = vec![
            test_message("user", "first"),
            RemoteHistoryMessage {
                role: synthetic.role,
                content: synthetic.content,
                tool_calls: None,
                reasoning_content: synthetic.reasoning_content,
                user_ordinal: None,
            },
            test_message("user", "second cache target"),
        ];

        let (ordinal, index) =
            resolve_serve_rewind_target(&messages, &["grep", "target"]).expect("real user match");
        assert_eq!((ordinal, index), (2, 2));
    }

    #[test]
    fn serve_role_ordinals_match_message_resolver() {
        let messages = sample_history();
        let roles: Vec<String> = messages.iter().map(|m| m.role.clone()).collect();
        assert_eq!(
            serve_role_user_message_count(&roles),
            serve_user_message_count(&messages)
        );
        for ordinal in 0..=4 {
            assert_eq!(
                serve_role_user_message_index(&roles, ordinal),
                serve_user_message_index(&messages, ordinal)
            );
        }
    }

    #[test]
    fn serve_view_role_param_maps_roles() {
        assert_eq!(serve_view_role_param(ServeHistoryRole::All), None);
        assert_eq!(serve_view_role_param(ServeHistoryRole::User), Some("user"));
        assert_eq!(
            serve_view_role_param(ServeHistoryRole::Assistant),
            Some("assistant")
        );
        assert_eq!(serve_view_role_param(ServeHistoryRole::Tool), Some("tool"));
        assert_eq!(
            serve_view_role_param(ServeHistoryRole::System),
            Some("system")
        );
    }

    #[test]
    fn render_serve_history_prefers_attached_ordinals() {
        // A filtered fetch arrives as a tail subset with global ordinals
        // attached; the markers must use them, not renumber the subset.
        let mut first = test_message("user", "q1");
        first.user_ordinal = Some(38);
        let mut second = test_message("user", "q2");
        second.user_ordinal = Some(40);
        let messages = vec![first, second];
        let view = ServeHistoryView {
            count: 2,
            role: ServeHistoryRole::User,
            full: false,
            grep: None,
        };
        let rendered = render_serve_history(&messages, &view);
        assert!(rendered.contains(" (u38)"), "{rendered}");
        assert!(rendered.contains(" (u40)"), "{rendered}");
        assert!(!rendered.contains(" (u1)"), "{rendered}");
        assert!(!rendered.contains(" (u2)"), "{rendered}");
    }

    #[test]
    fn serve_history_last_header_wording() {
        assert_eq!(
            serve_history_last_header(1),
            "[history] Latest assistant message:"
        );
        assert_eq!(
            serve_history_last_header(3),
            "[history] Assistant message 3 back from latest:"
        );
    }

    #[test]
    fn serve_rewind_target_from_roles_resolves_u_n_and_last() {
        let roles: Vec<String> = sample_history().iter().map(|m| m.role.clone()).collect();
        assert_eq!(
            resolve_serve_rewind_target_from_roles(&roles, &["u2"]).expect("u2"),
            (2, 3)
        );
        assert_eq!(
            resolve_serve_rewind_target_from_roles(&roles, &["2"]).expect("2"),
            (2, 3)
        );
        assert_eq!(
            resolve_serve_rewind_target_from_roles(&roles, &["last"]).expect("last"),
            (2, 3)
        );
        assert_eq!(
            resolve_serve_rewind_target_from_roles(&roles, &["latest"]).expect("latest"),
            (2, 3)
        );
        let err = resolve_serve_rewind_target_from_roles(&roles, &["u9"]).expect_err("u9");
        assert!(err.contains("u9 not found"), "{err}");
        let err = resolve_serve_rewind_target_from_roles(&roles, &["u0"]).expect_err("u0");
        assert!(err.contains("must be >= 1"), "{err}");
        let err = resolve_serve_rewind_target_from_roles(&roles, &[]).expect_err("empty");
        assert!(err.contains("missing rewind target"), "{err}");
        let err =
            resolve_serve_rewind_target_from_roles(&roles, &["grep", "foo"]).expect_err("grep");
        assert!(err.contains("needs message content"), "{err}");
    }

    #[test]
    fn serve_history_renders_local_shaped_rows() {
        let rendered = render_serve_history(&sample_history(), &default_view());
        assert!(rendered.starts_with("[history] Showing 5 recent message(s):\n"));
        assert!(rendered.contains("2. [user] first question (u1)\n"));
        assert!(rendered.contains("4. [user] second question about meego (u2)\n"));
        assert!(rendered.contains("3. [assistant] first answer\n"));
        // No model tags: the history route serves bare messages.
        assert!(!rendered.contains("(model:"));
    }

    #[test]
    fn serve_history_preview_truncates_and_collapses() {
        let long = "word ".repeat(60);
        let messages = vec![test_message("user", &format!("a\nb  {long}"))];
        let rendered = render_serve_history(&messages, &default_view());
        let row = rendered.lines().nth(1).expect("row");
        assert!(row.starts_with("1. [user] a b word "), "{row}");
        assert!(row.ends_with("... (u1)"), "{row}");
        assert!(row.chars().count() < 220, "{row}");
    }

    #[test]
    fn serve_history_full_and_filters() {
        let messages = sample_history();
        let (mut view, _) = parse_serve_history_view(&["full", "user"]).expect("parse");
        let rendered = render_serve_history(&messages, &view);
        assert!(rendered.starts_with("[history] Showing 2 recent user message(s):\n"));
        view.grep = Some("MEEGO".to_string());
        let rendered = render_serve_history(&messages, &view);
        assert!(rendered.contains("matching \"MEEGO\""));
        assert!(rendered.contains("(u2)"));
        assert!(!rendered.contains("(u1)"));
        assert_eq!(
            render_serve_history(&[], &default_view()),
            "[history] No recent messages."
        );
    }

    #[test]
    fn serve_history_system_filter_covers_internal_notes() {
        let messages = vec![
            test_message("internal_note", "note"),
            test_message("developer", "dev"),
        ];
        let (view, _) = parse_serve_history_view(&["system"]).expect("parse");
        let rendered = render_serve_history(&messages, &view);
        assert!(rendered.contains("[internal_note] note"));
        assert!(!rendered.contains("[developer]"));
    }

    #[test]
    fn serve_history_last_skips_tool_calls_and_empty_rows() {
        let mut messages = sample_history();
        messages.push(RemoteHistoryMessage {
            role: "assistant".to_string(),
            content: json!("tool answer"),
            tool_calls: Some(vec![json!({"id": "1"})]),
            reasoning_content: None,
            user_ordinal: None,
        });
        messages.push(test_message("assistant", "   "));
        let first = render_serve_history_last(&messages, 1).expect("last");
        assert_eq!(first, "[history] Latest assistant message:\nsecond answer");
        let second = render_serve_history_last(&messages, 2).expect("last 2");
        assert!(second.starts_with("[history] Assistant message 2 back from latest:\n"));
        assert!(second.ends_with("first answer"));
        assert!(render_serve_history_last(&messages, 3).is_err());
        assert!(render_serve_history_last(&[], 1).is_err());
    }

    #[test]
    fn serve_rewind_target_resolution() {
        let messages = sample_history();
        assert!(resolve_serve_rewind_target(&messages, &[]).is_err());
        assert_eq!(
            resolve_serve_rewind_target(&messages, &["last"]).expect("last"),
            (2, 3)
        );
        assert_eq!(
            resolve_serve_rewind_target(&messages, &["latest"]).expect("latest"),
            (2, 3)
        );
        assert_eq!(
            resolve_serve_rewind_target(&messages, &["u1"]).expect("u1"),
            (1, 1)
        );
        assert_eq!(
            resolve_serve_rewind_target(&messages, &["2"]).expect("bare"),
            (2, 3)
        );
        assert_eq!(
            resolve_serve_rewind_target(&messages, &["rewind", "u1"]).expect_err("nested"),
            "invalid rewind target: rewind. try: /history rewind u3"
        );
        assert!(resolve_serve_rewind_target(&messages, &["u0"]).is_err());
        assert!(resolve_serve_rewind_target(&messages, &["u9"]).is_err());
        assert!(resolve_serve_rewind_target(&[], &["last"]).is_err());
    }

    #[test]
    fn serve_rewind_grep_targets() {
        let messages = sample_history();
        assert_eq!(
            resolve_serve_rewind_target(&messages, &["grep", "meego"]).expect("one match"),
            (2, 3)
        );
        assert!(resolve_serve_rewind_target(&messages, &["grep", "missing"]).is_err());
        let err = resolve_serve_rewind_target(&messages, &["grep", "question"]).expect_err("multi");
        assert!(err.contains("2 user inputs match"), "{err}");
        assert!(err.contains("u1") && err.contains("u2"), "{err}");
        assert!(resolve_serve_rewind_target(&messages, &["grep"]).is_err());
    }

    #[test]
    fn serve_rewind_syntax_precheck_matches_resolver() {
        // Usage errors must be identical with and without the message list, so
        // the pre-fetch check never masks or contradicts full resolution.
        let messages = sample_history();
        for args in [
            vec![],
            vec!["bogus"],
            vec!["u0"],
            vec!["grep"],
            vec!["last"],
            vec!["u1"],
            vec!["2"],
            vec!["grep", "meego"],
        ] {
            let pre = check_serve_rewind_syntax(&args);
            let full = resolve_serve_rewind_target(&messages, &args).map(|_| ());
            match (pre, full) {
                (Ok(()), Ok(())) => {}
                (Err(a), Err(b)) => assert_eq!(a, b, "args: {args:?}"),
                (a, b) => panic!("syntax/full mismatch for {args:?}: {a:?} vs {b:?}"),
            }
        }
    }

    #[test]
    fn serve_rewind_result_shape() {
        let result: RemoteRewindResult =
            serde_json::from_str("{\"removed\": 2, \"kept\": 5}").expect("parse");
        assert_eq!(
            result,
            RemoteRewindResult {
                removed: 2,
                kept: 5
            }
        );
    }

    #[test]
    fn serve_chat_intercepts_local_commands() {
        // The `_` catch-all must never forward these as turn prompts.
        assert!(is_serve_local_command_start("/history rewind u40"));
        assert!(is_serve_local_command_start(":history"));
        assert!(is_serve_local_command_start("/compact"));
        // `/pwd` is dispatched by its own arm above, but it is still a local
        // command word: it must never be forwarded as a turn prompt either.
        assert!(is_serve_local_command_start("/pwd"));
        assert!(is_serve_local_command_start(":pwd extra"));
        // `/memo` is executed by the serve turn child itself, so serve-chat
        // must forward it (and `:memo`) like a prompt instead of blocking it.
        assert!(!is_serve_local_command_start("/memo"));
        assert!(!is_serve_local_command_start(":memo 记住这个"));
        // `/task`, `/discuss` and `/goal` only inject a `forced_question` and
        // let the turn continue, so the turn child runs them like a prompt.
        assert!(!is_serve_local_command_start("/task fix login"));
        assert!(!is_serve_local_command_start(":discuss API options"));
        assert!(!is_serve_local_command_start("/goal refactor parser"));
        assert!(!is_serve_local_command_start("/task status"));
        // `/audit`, `/distill-session`, `/mark`, `/unmark`, `/archive`,
        // `/title`, and `/export` execute in the server turn child, so
        // serve-chat forwards them like a prompt instead of blocking them.
        assert!(!is_serve_local_command_start("/audit"));
        assert!(!is_serve_local_command_start(":audit -f review this"));
        assert_eq!(
            classify_serve_input("/audit --fast review this"),
            ServeInputRoute::ServerCommand
        );
        assert!(!is_serve_local_command_start("/distill-session"));
        assert!(!is_serve_local_command_start("/archive"));
        assert_eq!(
            classify_serve_input("/distill-session current"),
            ServeInputRoute::ServerCommand
        );
        assert_eq!(
            classify_serve_input("/archive"),
            ServeInputRoute::ServerCommand
        );
        for input in ["/mark", ":mark keep this", "/unmark", ":unmark current"] {
            assert!(!is_serve_local_command_start(input), "{input}");
            assert_eq!(
                classify_serve_input(input),
                ServeInputRoute::ServerCommand,
                "{input}"
            );
        }
        for input in ["/title", ":title", "/title 排查性能问题"] {
            assert!(!is_serve_local_command_start(input), "{input}");
            assert_eq!(
                classify_serve_input(input),
                ServeInputRoute::ServerCommand,
                "{input}"
            );
        }
        for input in ["/export", ":export", "/export notes/summary.md"] {
            assert!(!is_serve_local_command_start(input), "{input}");
            assert_eq!(
                classify_serve_input(input),
                ServeInputRoute::ServerCommand,
                "{input}"
            );
        }
        // The classifier keeps these routes for compatibility; the client
        // intercepts `/use` and `/reset` with dedicated config branches.
        for input in [
            "/use",
            ":use",
            "/use traecli",
            "/use traecli GPT-5.5",
            "/use kernel",
            "/use\nfix login",
            "/use traecli\nfix login",
            "/use traecli GPT-5.5\nfix login",
            "/reset",
            ":reset",
        ] {
            assert!(!is_serve_local_command_start(input), "{input}");
            assert_eq!(
                classify_serve_input(input),
                ServeInputRoute::ServerCommand,
                "{input}"
            );
        }
        // Unknown command shapes are rejected locally, never forwarded.
        assert_eq!(
            classify_serve_input("/no-such-command x"),
            ServeInputRoute::UnknownCommand
        );
        assert_eq!(
            classify_serve_input(":no-such-command"),
            ServeInputRoute::UnknownCommand
        );
        assert!(!is_serve_local_command_start("/tmp/foo"));
        assert!(!is_serve_local_command_start("plain prompt"));
        assert_eq!(classify_serve_input("/tmp/foo"), ServeInputRoute::Prompt);
        assert_eq!(
            classify_serve_input("plain prompt"),
            ServeInputRoute::Prompt
        );
    }
}

/// Render one SSE turn stream progressively: message lines print as they
/// arrive, `delta` events (live body chunks) print without a trailing
/// newline, thinking frames maintain the same width-wrapped multi-row tail as
/// a local terminal and collapse to one `✓ thinking (N lines)` summary on
/// close, subagent progress redraws one clamped live row in place instead of
/// appending scrollback, error events abort with the server detail, and the
/// done event (or the end of the body) ends the turn.
fn render_turn_stream(resp: reqwest::blocking::Response) -> Result<(), Box<dyn std::error::Error>> {
    render_turn_stream_with_confirm(resp, None)
}

/// Same as [`render_turn_stream`], but answers remote confirmation requests
/// through `confirm` (`None` keeps the fail-closed default and only notes an
/// unanswerable question instead of hanging the turn on it).
fn render_turn_stream_with_confirm(
    resp: reqwest::blocking::Response,
    confirm: Option<&ServeConfirmChannel>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::IsTerminal;
    let tty = std::io::stdout().is_terminal();
    render_turn_stream_to_with_confirm(resp, &mut std::io::stdout(), tty, confirm)
}

/// Same as [`render_turn_stream`], but writes to `out` so tests can assert
/// the exact bytes (line breaks between body, footers and fold summaries).
/// Never answers confirmations (`None` channel): tests stay offline and never
/// touch stdin.
fn render_turn_stream_to(
    resp: reqwest::blocking::Response,
    out: &mut dyn std::io::Write,
    tty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    render_turn_stream_to_with_confirm(resp, out, tty, None)
}

/// Same as [`render_turn_stream_to`], but `confirm` answers remote
/// confirmation requests (`confirm_request` events) with a local y/n prompt
/// and a `POST .../confirm` round-trip, so a serve-chat turn can approve the
/// same gates (today `git commit` / `git push` / `git stash` / `git checkout` / `git switch`) as a local turn.
fn render_turn_stream_to_with_confirm(
    resp: reqwest::blocking::Response,
    out: &mut dyn std::io::Write,
    tty: bool,
    confirm: Option<&ServeConfirmChannel>,
) -> Result<(), Box<dyn std::error::Error>> {
    render_turn_stream_to_with_width(resp, out, tty, confirm, || {
        crossterm::terminal::size()
            .map(|(cols, _)| usize::from(cols))
            .unwrap_or(0)
    })
}

/// Probe each observer row at dispatch time; tests can replay live resizes
/// without changing process-wide terminal state or the SSE protocol.
fn render_turn_stream_to_with_width(
    resp: reqwest::blocking::Response,
    out: &mut dyn std::io::Write,
    tty: bool,
    confirm: Option<&ServeConfirmChannel>,
    mut terminal_width: impl FnMut() -> usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::BufRead;
    // `blocking::Response` implements `Read`, so lines are parsed (and
    // printed) as soon as the server flushes them, not after the whole turn.
    let mut reader = std::io::BufReader::new(resp);
    let mut event_kind = String::new();
    // SSE allows one event to carry several `data:` lines; the server emits
    // the multi-line subagent status block that way. Join them with `\n` and
    // dispatch only when the blank line terminates the event, instead of
    // treating every `data:` line as a complete event.
    let mut event_data = String::new();
    let mut event_has_data = false;
    let mut raw = String::new();
    // Folded-thinking state: retain only a sanitized display tail plus source
    // character/newline counters for the approximate summary. The transient
    // rows never mix with the answer, and display memory stays bounded even
    // for a runaway reasoning stream.
    let mut thinking_active = false;
    let mut thinking_newlines = 0usize;
    let mut thinking_chars = 0usize;
    let configured_thinking_lines = crate::ai::stream::resolve_thinking_fold_max_visible_lines(
        tty,
        configw::get_all_config()
            .get_opt(AiConfig::OUTPUT_THINKING_MAX_VISIBLE_LINES)
            .as_deref(),
    );
    let mut thinking_region = ThinkingLiveRegion::new(configured_thinking_lines, None);
    let mut answer_pending = false;
    let mut answer_pending_drawn = false;
    // Retain the plain rows so a resize can recompute the block's physical
    // height before erasure. Each frame replaces this transient live output.
    let mut subagent_status_rows = Vec::new();
    // Whether the cursor sits mid-line (answer delta without a trailing
    // newline, or a live thinking status). Footer/message lines must start
    // on a fresh row instead of gluing onto the answer's last line.
    let mut line_open = false;
    let mut observer_rows = ObserverRowState::default();
    // Stdout and live-body frames use separate pumps, so a metrics row can
    // overtake the last body delta. Retain it until a body/round boundary.
    let mut pending_metrics = Vec::new();
    // Local-equivalent Markdown rendering: on a TTY the body deltas go
    // through the shared streaming renderer (tables, code blocks, math),
    // exactly like the local turn output. Piped output stays raw so
    // captures keep clean bytes.
    let mut markdown = MarkdownStreamRenderer::new_with_tty(tty);
    // Body bytes already painted by `delta`. `replace` is authoritative and
    // usually repeats them; a terminal cannot rewind the rows, so only the
    // not-yet-rendered suffix is printed.
    let mut streamed_body = String::new();
    // Whether a `replace` has committed a turn on this stream. The turn header
    // is also printed on stream retries, so only a header after a committed
    // turn starts a new one.
    let mut turn_committed = false;
    let stream_result = 'events: loop {
        // Collect one complete event before dispatching it. The server always
        // terminates an event with a blank line; a final event at EOF is still
        // dispatched so a truncated trailing frame is not silently dropped.
        let payload = loop {
            raw.clear();
            match reader.read_line(&mut raw) {
                Ok(0) => {
                    if event_has_data {
                        event_has_data = false;
                        break std::mem::take(&mut event_data);
                    }
                    break 'events Ok(());
                }
                Ok(_) => {}
                Err(err) => break 'events Err(Box::<dyn std::error::Error>::from(err)),
            }
            let line = raw.strip_suffix('\n').unwrap_or(raw.as_str());
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.is_empty() {
                if event_has_data {
                    event_has_data = false;
                    break std::mem::take(&mut event_data);
                }
                event_kind.clear();
                continue;
            }
            if line.starts_with(':') {
                continue; // SSE comment / keep-alive ping.
            }
            if let Some(kind) = line.strip_prefix("event:") {
                event_kind = kind.trim().to_string();
                continue;
            }
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.strip_prefix(' ').unwrap_or(data);
            if event_has_data {
                event_data.push('\n');
            }
            event_data.push_str(data);
            event_has_data = true;
        };
        let payload = payload.as_str();
        if matches!(event_kind.as_str(), "" | "message") && is_metrics_footer(payload) {
            pending_metrics.push(payload.to_string());
            observer_rows.last_header_key = None;
            event_kind.clear();
            continue;
        }
        if matches!(
            event_kind.as_str(),
            "error"
                | "done"
                | "thinking_start"
                | "thinking"
                | "thinking_done"
                | "answer_pending"
                | "delta"
                | "confirm_request"
                | "replace"
                | "subagent_progress"
        ) {
            // Header deduplication is only for adjacent retry copies. Once
            // the stream reports model activity, the next request header is
            // useful context again even when its fields are unchanged.
            observer_rows.last_header_key = None;
            observer_rows.close_tool_row(&mut line_open, out);
            observer_rows.hide_context_budget_row(&mut line_open, tty, out);
        }
        // The subagent status occupies the active row, so any event that
        // renders output or ends the turn supersedes it. Clear it first so
        // answer/footer bytes never land on top of a stale progress row.
        // `tool` frames are display-only and `confirm_done` renders nothing,
        // so keeping the row through them avoids needless flicker.
        if !matches!(
            event_kind.as_str(),
            "subagent_progress" | "tool" | "confirm_done"
        ) {
            hide_subagent_status(&mut subagent_status_rows, &mut line_open, tty, out);
        }
        match event_kind.as_str() {
            "error" => {
                let msg = serde_json::from_str::<serde_json::Value>(payload)
                    .ok()
                    .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
                    .unwrap_or_else(|| payload.to_string());
                break Err(format!("turn failed: {msg}").into());
            }
            "done" => {
                break Ok(());
            }
            "thinking_start" => {
                // A second thinking block in one turn closes the previous
                // summary first (multi-round model calls).
                // Flush first: the previous round's body is complete, and the
                // status line must not glue onto its last row.
                hide_answer_pending_status(&mut answer_pending_drawn, &mut line_open, tty, out);
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_region,
                    tty,
                    out,
                    &mut observer_rows,
                );
                flush_metrics_footers(&mut pending_metrics, &mut line_open, tty, out);
                if tty {
                    close_open_line(&mut line_open, out);
                }
                thinking_active = true;
                thinking_newlines = 0;
                thinking_chars = 0;
                thinking_region.reset();
                if tty {
                    thinking_region.redraw(out);
                    line_open = true;
                }
            }
            "thinking" => {
                // A thinking frame arriving after body output (multi-round
                // calls) must not rewrite the rendered answer's last line.
                // Mid-thinking this flush is a no-op.
                hide_answer_pending_status(&mut answer_pending_drawn, &mut line_open, tty, out);
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                let thinking = serde_json::from_str::<serde_json::Value>(payload).ok();
                let text = thinking
                    .as_ref()
                    .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
                    .unwrap_or_else(|| payload.to_string());
                if !thinking_active {
                    // Tolerate start-less frames (older servers never send
                    // them, but a folded region beats leaked reasoning).
                    if tty {
                        close_open_line(&mut line_open, out);
                    }
                    thinking_active = true;
                    thinking_newlines = 0;
                    thinking_chars = 0;
                    thinking_region.reset();
                }
                thinking_newlines = thinking_newlines.saturating_add(text.matches('\n').count());
                thinking_chars = thinking_chars.saturating_add(text.chars().count());
                if tty {
                    thinking_region.header_rate = thinking
                        .as_ref()
                        .and_then(|v| v.get("rate").and_then(|rate| rate.as_str()))
                        .map(|rate| {
                            rate.chars()
                                .filter(|ch| !ch.is_control())
                                .take(128)
                                .collect()
                        });
                    thinking_region.append(&text);
                    thinking_region.redraw(out);
                    line_open = true;
                }
            }
            "thinking_done" => {
                // The live body (if any) is complete: erase the transient
                // region and retain only the fold summary row.
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_region,
                    tty,
                    out,
                    &mut observer_rows,
                );
                show_answer_pending_status(
                    answer_pending,
                    &mut answer_pending_drawn,
                    thinking_active,
                    &mut line_open,
                    tty,
                    out,
                );
            }
            // Text progress already serves the terminal; do not print web UI JSON.
            "tool" => {}
            "answer_pending" => {
                answer_pending = true;
                show_answer_pending_status(
                    answer_pending,
                    &mut answer_pending_drawn,
                    thinking_active,
                    &mut line_open,
                    tty,
                    out,
                );
            }
            "delta" => {
                // Frames from the FIFO and lines from child stdout travel on
                // separate threads, so `thinking_done` can arrive after body or
                // footer lines (or only at stream end, when the close marker
                // was glued to body text upstream). Body text must never share
                // the live thinking status line, so fold it first here,
                // unconditionally.
                // A late close can arrive after body output started; fold it
                // on a fresh row before the renderer continues.
                hide_answer_pending_status(&mut answer_pending_drawn, &mut line_open, tty, out);
                answer_pending = false;
                if tty && thinking_active {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_region,
                    tty,
                    out,
                    &mut observer_rows,
                );
                let text = serde_json::from_str::<serde_json::Value>(payload)
                    .ok()
                    .and_then(|v| v.get("delta").and_then(|d| d.as_str()).map(str::to_string))
                    .unwrap_or_else(|| payload.to_string());
                if tty {
                    // Same renderer as the local turn output: tables, code
                    // blocks and math render progressively, not as raw source.
                    let _ = markdown.write_chunk_to(out, &text, false);
                } else {
                    let _ = write!(out, "{text}");
                }
                if !text.is_empty() {
                    observer_rows.last_thinking_summary = None;
                    streamed_body.push_str(&text);
                    line_open = !text.ends_with('\n');
                }
                let _ = out.flush();
            }
            "confirm_request" => {
                // Same ordering guarantee as the `_` arm below: the question
                // must start on a fresh row, never glued onto answer deltas.
                hide_answer_pending_status(&mut answer_pending_drawn, &mut line_open, tty, out);
                if tty {
                    let _ = markdown.flush_pending_to(out);
                    if markdown.at_line_start() {
                        line_open = false;
                    }
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_region,
                    tty,
                    out,
                    &mut observer_rows,
                );
                if line_open {
                    let _ = writeln!(out);
                }
                let question = serde_json::from_str::<serde_json::Value>(payload).ok();
                flush_metrics_footers(&mut pending_metrics, &mut line_open, tty, out);
                let parsed = question.as_ref().and_then(|v| {
                    Some((
                        v.get("id")?.as_u64()?,
                        v.get("token")?.as_u64()?,
                        v.get("prompt")?.as_str()?,
                    ))
                });
                match parsed {
                    Some((id, token, prompt)) => match confirm {
                        Some(channel) => channel.answer(id, token, prompt, out),
                        None => {
                            let _ = writeln!(
                                out,
                                "[serve-chat] warning: the remote turn waits for a \
                                 confirmation this client cannot answer (stdin is not \
                                 interactive); the remote command stays blocked until \
                                 the turn ends."
                            );
                        }
                    },
                    None => {
                        let _ = writeln!(
                            out,
                            "[serve-chat] warning: ignoring a malformed confirm_request event."
                        );
                    }
                }
                observer_rows.last_thinking_summary = None;
                line_open = false;
                let _ = out.flush();
                show_answer_pending_status(
                    answer_pending,
                    &mut answer_pending_drawn,
                    thinking_active,
                    &mut line_open,
                    tty,
                    out,
                );
            }
            "confirm_done" => {
                // The child's own resolution frame: an answered question already
                // cleared the slot server-side, so there is nothing to render
                // (and it must not fall into the `_` arm's footer).
            }
            "replace" => {
                // The daemon sends the accepted final body after the
                // provisional `delta` stream. Render only the part beyond what
                // was already painted (typically a final-gate suffix such as a
                // runtime warning); an identical body is a no-op, and a body
                // received without deltas renders in full. This event must
                // never reach the observer-row arm below: that arm prints its
                // payload verbatim, which leaks the `{"text":...}` envelope
                // into the scrollback.
                hide_answer_pending_status(&mut answer_pending_drawn, &mut line_open, tty, out);
                answer_pending = false;
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_region,
                    tty,
                    out,
                    &mut observer_rows,
                );
                let text = serde_json::from_str::<serde_json::Value>(payload)
                    .ok()
                    .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
                    .unwrap_or_default();
                if text.is_empty() {
                    observer_rows.last_thinking_summary = None;
                    // An empty accepted body means the turn produced no
                    // user-visible answer (or the final gate rejected it).
                    // The provisional deltas are not authoritative, and
                    // `/history last` can show an older turn's answer, so say
                    // plainly that this turn has no final answer instead of
                    // pointing at history.
                    if line_open {
                        let _ = writeln!(out);
                    }
                    let _ = writeln!(
                        out,
                        "[serve-chat] this turn ended without a final answer; the provisional draft was not authoritative."
                    );
                    line_open = false;
                    let _ = out.flush();
                } else if let Some(suffix) = text.strip_prefix(streamed_body.as_str()) {
                    if !suffix.is_empty() {
                        observer_rows.last_thinking_summary = None;
                        if tty {
                            let _ = markdown.write_chunk_to(out, suffix, false);
                        } else {
                            let _ = write!(out, "{suffix}");
                        }
                        line_open = !suffix.ends_with('\n');
                        let _ = out.flush();
                    }
                } else {
                    observer_rows.last_thinking_summary = None;
                    // The accepted body is not an extension of the streamed
                    // text (a gate rewrote it). The provisional draft may
                    // never have reached the screen (or may have been
                    // overwritten), so hiding the accepted body would leave
                    // the turn with no visible answer. Print it under a
                    // separator; a duplicate is preferable to a missing
                    // answer.
                    if line_open {
                        let _ = writeln!(out);
                    }
                    let _ = writeln!(
                        out,
                        "[serve-chat] final answer was revised; showing the accepted answer:"
                    );
                    if tty {
                        let _ = markdown.write_chunk_to(out, &text, false);
                    } else {
                        let _ = write!(out, "{text}");
                    }
                    line_open = !text.ends_with('\n');
                    let _ = out.flush();
                }
                streamed_body = text;
                turn_committed = true;
                if tty && !pending_metrics.is_empty() {
                    let _ = markdown.flush_pending_to(out);
                    if markdown.at_line_start() {
                        line_open = false;
                    }
                }
                flush_metrics_footers(&mut pending_metrics, &mut line_open, tty, out);
            }
            "subagent_progress" => {
                // Serve frames carry the whole parent-observed status line
                // (several subagents, plans, elapsed counters). Painting each
                // frame as an observer row leaves one wrapped copy per
                // scheduler tick in the scrollback, so render it as a single
                // in-place live block instead, exactly like the local TTY
                // status block.
                show_subagent_status(
                    payload,
                    &mut subagent_status_rows,
                    answer_pending_drawn,
                    thinking_active,
                    &mut line_open,
                    tty,
                    out,
                );
            }
            _ => {
                // Same ordering guarantee as the `delta` arm: observer footers
                // arriving while a fold is still open must not slip between the
                // answer and its thinking summary.
                hide_answer_pending_status(&mut answer_pending_drawn, &mut line_open, tty, out);
                if tty {
                    // Flush first so the footer starts after the rendered
                    // body; a completed row means no extra guard newline.
                    let _ = markdown.flush_pending_to(out);
                    if markdown.at_line_start()
                        && observer_rows.open_tool_row.is_none()
                        && !observer_rows.context_budget_drawn
                    {
                        line_open = false;
                    }
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_region,
                    tty,
                    out,
                    &mut observer_rows,
                );
                // Unnamed stdout metrics normally take the buffering path above.
                // Tolerate named observer events carrying the same footer too.
                if is_metrics_footer(payload) {
                    pending_metrics.push(payload.to_string());
                    observer_rows.last_header_key = None;
                    observer_rows.close_tool_row(&mut line_open, out);
                    observer_rows.hide_context_budget_row(&mut line_open, tty, out);
                    let _ = out.flush();
                } else {
                    flush_metrics_footers(&mut pending_metrics, &mut line_open, tty, out);
                    let width = if tty { terminal_width() } else { 0 };
                    let previous_tool_text = observer_rows.open_tool_text.take();
                    let decision = decide_observer_row(
                        payload,
                        &mut observer_rows,
                        tty,
                        width,
                        turn_committed,
                    );
                    if decision.close_tool_row {
                        close_open_line(&mut line_open, out);
                    }
                    if !matches!(decision.action, ObserverRowAction::ContextBudget { .. }) {
                        observer_rows.hide_context_budget_row(&mut line_open, tty, out);
                    }
                    // `replace` commits the prior turn; only its following
                    // header starts the pending side-note's next turn. Retry
                    // headers must keep the answer buffer and dedupe key.
                    if decision.new_turn {
                        observer_rows.finish_context_budget_row(&mut line_open, tty, width, out);
                        streamed_body.clear();
                        turn_committed = false;
                    }
                    // The turn header (`print_info` in `request/transport.rs`)
                    // is forwarded as a plain stdout row, so without a marker
                    // the serve-chat scrollback is indistinguishable from a
                    // local run. Clip after appending the marker so the entire
                    // displayed row fits the client's current terminal.
                    let row = if decision.header {
                        annotate_remote_turn_header(payload, tty)
                    } else {
                        payload.to_string()
                    };
                    let row = clamp_observer_row(&row, tty, width);
                    match decision.action {
                        ObserverRowAction::SkipDuplicate => {}
                        ObserverRowAction::ReplaceOpenToolRow => {
                            let _ = write!(out, "\r\x1b[2K");
                            // A row left open at a wider size may have reflowed
                            // before this completion arrives. Erase its whole
                            // current footprint, not just the cursor's row.
                            let rows = previous_tool_text
                                .as_deref()
                                .filter(|_| width > 0)
                                .map(|text| thinking_physical_rows(text, width))
                                .unwrap_or(1);
                            for _ in 1..rows {
                                let _ = write!(out, "\x1b[1A\r\x1b[2K");
                            }
                            let _ = writeln!(out, "{row}");
                            line_open = false;
                        }
                        ObserverRowAction::OpenToolRow => {
                            close_open_line(&mut line_open, out);
                            let _ = write!(out, "{row}");
                            observer_rows.open_tool_text = Some(strip_ansi_escapes(&row));
                            line_open = true;
                        }
                        ObserverRowAction::ContextBudget { replace, draw } => {
                            if replace {
                                let _ = write!(out, "\r\x1b[2K");
                                line_open = false;
                            } else {
                                close_open_line(&mut line_open, out);
                            }
                            // Keep the source for the deferred commit, which
                            // may happen after another terminal resize.
                            observer_rows.context_budget_row = Some(payload.to_string());
                            observer_rows.context_budget_drawn = draw;
                            if draw {
                                let _ = write!(out, "{row}");
                                line_open = true;
                            }
                        }
                        ObserverRowAction::Print => {
                            close_open_line(&mut line_open, out);
                            let _ = writeln!(out, "{row}");
                            line_open = false;
                        }
                    }
                    let _ = out.flush();
                }
                if tty && answer_pending && !thinking_active {
                    observer_rows.close_tool_row(&mut line_open, out);
                    observer_rows.hide_context_budget_row(&mut line_open, tty, out);
                }
                show_answer_pending_status(
                    answer_pending,
                    &mut answer_pending_drawn,
                    thinking_active,
                    &mut line_open,
                    tty,
                    out,
                );
            }
        }
        event_kind.clear();
    };
    // Done, EOF and read/server errors all share cleanup, including open tools.
    observer_rows.close_tool_row(&mut line_open, out);
    observer_rows.hide_context_budget_row(&mut line_open, tty, out);
    hide_subagent_status(&mut subagent_status_rows, &mut line_open, tty, out);
    hide_answer_pending_status(&mut answer_pending_drawn, &mut line_open, tty, out);
    if tty {
        let _ = markdown.flush_pending_to(out);
    }
    close_thinking_status(
        &mut thinking_active,
        &mut line_open,
        thinking_summary(thinking_newlines, thinking_chars),
        &mut thinking_region,
        tty,
        out,
        &mut observer_rows,
    );
    close_open_line(&mut line_open, out);
    let width = if tty { terminal_width() } else { 0 };
    observer_rows.finish_context_budget_row(&mut line_open, tty, width, out);
    flush_metrics_footers(&mut pending_metrics, &mut line_open, tty, out);
    stream_result
}

#[derive(Default)]
struct ObserverRowState {
    last_header_key: Option<String>,
    last_thinking_summary: Option<String>,
    last_context_budget: Option<String>,
    open_tool_row: Option<String>,
    open_tool_text: Option<String>,
    context_budget_row: Option<String>,
    context_budget_drawn: bool,
}

impl ObserverRowState {
    fn close_tool_row(&mut self, line_open: &mut bool, out: &mut dyn std::io::Write) {
        self.open_tool_text = None;
        if self.open_tool_row.take().is_some() {
            close_open_line(line_open, out);
        }
    }

    fn hide_context_budget_row(
        &mut self,
        line_open: &mut bool,
        tty: bool,
        out: &mut dyn std::io::Write,
    ) {
        // Budget notes use the same transient-row lifecycle as live progress.
        if !self.context_budget_drawn {
            return;
        }
        if tty {
            erase_status_rows(1, out);
            let _ = out.flush();
        }
        self.context_budget_drawn = false;
        *line_open = false;
    }

    fn finish_context_budget_row(
        &mut self,
        line_open: &mut bool,
        tty: bool,
        width: usize,
        out: &mut dyn std::io::Write,
    ) {
        if let Some(row) = self.context_budget_row.take() {
            close_open_line(line_open, out);
            let row = clamp_observer_row(&row, tty, width);
            let _ = writeln!(out, "{row}");
            self.last_thinking_summary = None;
            let _ = out.flush();
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ObserverRowAction {
    Print,
    SkipDuplicate,
    OpenToolRow,
    ReplaceOpenToolRow,
    ContextBudget { replace: bool, draw: bool },
}

#[derive(Debug, PartialEq, Eq)]
struct ObserverRowDecision {
    action: ObserverRowAction,
    close_tool_row: bool,
    header: bool,
    new_turn: bool,
}

/// Decide from the complete source text before presentation truncation.
fn decide_observer_row(
    payload: &str,
    state: &mut ObserverRowState,
    tty: bool,
    width: usize,
    turn_committed: bool,
) -> ObserverRowDecision {
    use unicode_width::UnicodeWidthStr;

    let plain = strip_ansi_escapes(payload);
    let normalized = plain.trim();
    let header_text = normalized
        .strip_suffix("(remote)")
        .unwrap_or(normalized)
        .trim_end();
    let header = is_turn_header_row(header_text);
    let new_turn = header && turn_committed;
    if new_turn {
        state.last_header_key = None;
        state.last_thinking_summary = None;
        state.last_context_budget = None;
    } else if !header {
        // Suppress only consecutive copies (normally request retries). Tool,
        // thinking, and other observer rows make a later header informative.
        state.last_header_key = None;
    }
    let thinking_summary = normalized
        .strip_prefix("✓ thinking (")
        .and_then(|tail| tail.strip_suffix(')'))
        .and_then(|tail| tail.split_once(' '))
        .is_some_and(|(count, unit)| {
            count.parse::<usize>().is_ok() && matches!(unit, "line" | "lines" | "line(s)")
        });
    let completion_name = normalized
        .strip_prefix('✓')
        .and_then(|tail| tail.split_whitespace().next());
    let replace_tool = !thinking_summary
        && completion_name.is_some()
        && state.open_tool_row.as_deref() == completion_name;
    let close_tool_row = state.open_tool_row.take().is_some() && !replace_tool;
    let fits = width > 0 && !plain.chars().any(char::is_control) && plain.width() < width;
    let action = if header {
        // Identity is the bracketed model/effort/title, excluding any clocks
        // outside it and the client's `(remote)` suffix. Only adjacent copies
        // are hidden; later model rounds remain visible even when unchanged.
        let key = header_text
            .rfind(']')
            .map(|end| &header_text[..=end])
            .unwrap_or(header_text);
        if state.last_header_key.as_deref() == Some(key) {
            ObserverRowAction::SkipDuplicate
        } else {
            state.last_header_key = Some(key.to_string());
            state.last_thinking_summary = None;
            ObserverRowAction::Print
        }
    } else if thinking_summary {
        if state.last_thinking_summary.as_deref() == Some(normalized) {
            ObserverRowAction::SkipDuplicate
        } else {
            state.last_thinking_summary = Some(normalized.to_string());
            ObserverRowAction::Print
        }
    } else if normalized.starts_with("│ context-budget:") {
        state.last_thinking_summary = None;
        let duplicate = state.last_context_budget.as_deref() == Some(normalized);
        state.last_context_budget = Some(normalized.to_string());
        if tty {
            // A transient note is erased before other output. Only the latest
            // note is committed at the turn boundary; wrapped notes wait for
            // that final print so erasure never crosses into another row.
            let action = ObserverRowAction::ContextBudget {
                replace: state.context_budget_drawn,
                draw: fits,
            };
            state.context_budget_drawn = fits;
            action
        } else if duplicate {
            ObserverRowAction::SkipDuplicate
        } else {
            ObserverRowAction::Print
        }
    } else {
        state.last_thinking_summary = None;
        if replace_tool {
            ObserverRowAction::ReplaceOpenToolRow
        } else if let Some(name) = normalized
            .strip_prefix('●')
            .and_then(|tail| tail.split_whitespace().next())
            .filter(|_| tty && fits)
        {
            state.open_tool_row = Some(name.to_string());
            ObserverRowAction::OpenToolRow
        } else {
            // Pipes keep both rows: they cannot erase an earlier output row.
            ObserverRowAction::Print
        }
    };
    ObserverRowDecision {
        action,
        close_tool_row,
        header,
        new_turn,
    }
}

/// Approximate folded-thinking line count for the summary row.
fn thinking_summary(newlines: usize, chars: usize) -> usize {
    if chars == 0 { 0 } else { newlines + 1 }
}

/// Whether a stdout-derived footer row is an end-of-stream metrics summary.
/// These (`↳ cache …`, `↳ speed …`) are buffered until a body/round boundary;
/// matching on the marker (not the full line) keeps this robust to the
/// child's indent/ANSI styling. Body text never reaches this path (it arrives
/// as `delta` events), so a quoted marker in the answer cannot be
/// misclassified here.
fn is_metrics_footer(payload: &str) -> bool {
    payload.contains("↳ cache") || payload.contains("↳ speed")
}

fn flush_metrics_footers(
    pending: &mut Vec<String>,
    line_open: &mut bool,
    tty: bool,
    out: &mut dyn std::io::Write,
) {
    if pending.is_empty() {
        return;
    }
    close_open_line(line_open, out);
    for row in pending.drain(..) {
        let row = if tty { row } else { strip_ansi_escapes(&row) };
        let _ = writeln!(out, "{row}");
    }
    let _ = out.flush();
}

/// Whether a stdout-derived row is the turn header printed by `print_info`
/// (`request/transport.rs`): `[model (effort: …) · topic]`. The child paints
/// it with theme colors, so detection strips ANSI escapes and matches the
/// shape, not the raw bytes. Body text never reaches this path (it arrives
/// as `delta` events), and the `[` + `(effort:` shape is specific to that
/// header, so tool output cannot realistically collide.
fn is_turn_header_row(payload: &str) -> bool {
    if payload.contains("(remote)") {
        return false;
    }
    let plain = strip_ansi_escapes(payload);
    let trimmed = plain.trim_start();
    trimmed.starts_with('[') && trimmed.contains("(effort:")
}

/// Strip ANSI escape sequences for shape matching only; the display bytes
/// are never rewritten. Handles CSI (`ESC [ … final`) plus the single-char
/// and OSC-style introducers so a half-kept sequence cannot leak into the
/// match. Bare `\r` bytes are dropped: the server already turns interior
/// overwrite controls into row breaks before SSE transport.
fn strip_ansi_escapes(text: &str) -> String {
    let mut plain = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            if c != '\r' {
                plain.push(c);
            }
            continue;
        }
        match chars.next() {
            Some('[') => {
                for next in chars.by_ref() {
                    if ('@'..='~').contains(&next) {
                        break;
                    }
                }
            }
            Some(']' | 'P' | 'X' | '^' | '_') => {
                for next in chars.by_ref() {
                    if next == '\x07' {
                        break;
                    }
                    if next == '\x1b' {
                        let _ = chars.next();
                        break;
                    }
                }
            }
            Some(_) => {}
            None => break,
        }
    }
    plain
}

/// Annotate a forwarded turn-header row as remote. Display bytes stay
/// untouched; the marker is appended after the child's trailing reset so it
/// renders identically on every terminal. TTY uses the same lavender bold
/// as the input-box `(remote)` marker, piped output stays plain text.
fn annotate_remote_turn_header(payload: &str, tty: bool) -> String {
    if payload.contains("(remote)") {
        return payload.to_string();
    }
    if tty {
        format!("{payload}\x1b[1;38;2;196;181;253m (remote)\x1b[0m")
    } else {
        format!("{payload} (remote)")
    }
}

/// Preserve SGR styling while fitting each observer line to the local width.
/// Pipes and unknown geometry retain the original bytes. State/deduplication
/// always sees the complete source row, including any hidden suffix.
fn clamp_observer_row(row: &str, tty: bool, width: usize) -> String {
    if !tty || width == 0 {
        return row.to_string();
    }
    row.split('\n')
        .map(|line| {
            let mut plain = String::with_capacity(line.len());
            let mut styles = Vec::new();
            let mut chars = line.char_indices();
            while let Some((start, ch)) = chars.next() {
                if ch == '\x1b' {
                    match chars.next() {
                        Some((_, '[')) => {
                            for (end, next) in chars.by_ref() {
                                if ('@'..='~').contains(&next) {
                                    if next == 'm'
                                        && line[start + 2..end]
                                            .bytes()
                                            .all(|b| b.is_ascii_digit() || matches!(b, b';' | b':'))
                                    {
                                        styles.push((plain.len(), &line[start..end + 1]));
                                    }
                                    break;
                                }
                            }
                        }
                        Some((_, ']' | 'P' | 'X' | '^' | '_')) => {
                            for (_, next) in chars.by_ref() {
                                if next == '\x07' {
                                    break;
                                }
                                if next == '\x1b' {
                                    let _ = chars.next();
                                    break;
                                }
                            }
                        }
                        _ => {}
                    }
                    continue;
                }
                match ch {
                    '\r' => {}
                    '\t' => plain.push_str("    "),
                    ch if ch.is_control() => plain.extend(ch.escape_default()),
                    ch => plain.push(ch),
                }
            }
            // Reuse the existing plain-row clamp and the shared renderer's
            // terminal_cell_width semantics, including CJK and emoji.
            let clamped = clamp_thinking_row(&plain, width);
            let truncated = clamped != plain;
            let kept = if truncated {
                clamped.len() - '…'.len_utf8()
            } else {
                plain.len()
            };
            let styled = !styles.is_empty();
            let mut result = String::with_capacity(line.len());
            let mut copied = 0;
            for (at, style) in styles.into_iter().take_while(|(at, _)| *at <= kept) {
                result.push_str(&plain[copied..at]);
                result.push_str(style);
                copied = at;
            }
            result.push_str(&clamped[copied..]);
            if truncated && styled {
                // The child's trailing reset may be beyond the cut point.
                result.push_str("\x1b[0m");
            }
            result
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Close the answer's open row at end-of-turn so the scrollback ends on a
/// fresh line. Metrics summaries used to provide this trailing newline as a
/// side effect; with them discarded the terminator is explicit.
fn close_open_line(line_open: &mut bool, out: &mut dyn std::io::Write) {
    if *line_open {
        let _ = writeln!(out);
        *line_open = false;
    }
    let _ = out.flush();
}

/// Dim helper: plain text when piped, so non-terminal captures stay clean.
fn dim(text: &str, tty: bool) -> String {
    if tty {
        format!("\x1b[2m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn show_answer_pending_status(
    pending: bool,
    drawn: &mut bool,
    thinking_active: bool,
    line_open: &mut bool,
    tty: bool,
    out: &mut dyn std::io::Write,
) {
    if !tty || !pending || *drawn || thinking_active {
        return;
    }
    let _ = write!(out, "\r\x1b[K\x1b[2m◌ finalizing response…\x1b[0m");
    let _ = out.flush();
    *drawn = true;
    *line_open = true;
}

fn hide_answer_pending_status(
    drawn: &mut bool,
    line_open: &mut bool,
    tty: bool,
    out: &mut dyn std::io::Write,
) {
    if !*drawn {
        return;
    }
    if tty {
        let _ = write!(out, "\r\x1b[K");
        let _ = out.flush();
    }
    *drawn = false;
    *line_open = false;
}

/// Paint the parent-observed subagent status block as transient terminal rows.
///
/// The server sends the complete team tree in one SSE event. Clamp every line
/// to one physical row and redraw the block in place, matching the local TTY
/// status block. If the tree exceeds the viewport, keep its prefix and mark
/// the omitted bottom rows. Piped output stays clean: the progress block is
/// terminal-only and is never part of the captured transcript.
///
/// The block must not overwrite real output or another live status:
/// - `thinking_active` / `answer_pending_drawn` own the active row;
/// - `line_open` without our own rows means a body delta is mid-line.
fn show_subagent_status(
    text: &str,
    drawn_rows: &mut Vec<String>,
    answer_pending_drawn: bool,
    thinking_active: bool,
    line_open: &mut bool,
    tty: bool,
    out: &mut dyn std::io::Write,
) {
    if !tty || thinking_active || answer_pending_drawn || (*line_open && drawn_rows.is_empty()) {
        return;
    }
    let (cols, viewport_rows) = subagent_terminal_size();
    let rows = subagent_status_rows_at_height(text, viewport_rows);
    if rows.is_empty() {
        hide_subagent_status(drawn_rows, line_open, tty, out);
        return;
    }
    erase_subagent_status_rows_at_size(drawn_rows, cols, viewport_rows, out);
    for (index, row) in rows.iter().enumerate() {
        if index > 0 {
            let _ = write!(out, "\r\n");
        }
        let _ = write!(out, "{}", dim(row, true));
    }
    let _ = out.flush();
    *drawn_rows = rows;
    *line_open = true;
}

fn subagent_terminal_size() -> (usize, usize) {
    let rows = crossterm::terminal::size()
        .map(|(_, rows)| usize::from(rows).max(1))
        .unwrap_or(24);
    (crate::ai::stream::raw_terminal_cols(), rows)
}

fn subagent_status_rows_at_height(text: &str, viewport_rows: usize) -> Vec<String> {
    // Keep one row outside the status region so cursor-up can still reach
    // its beginning after drawing a tall tree at the bottom of the screen.
    let budget = viewport_rows.saturating_sub(1).max(1);
    let total_rows = text.lines().count();
    let kept_rows = if total_rows > budget {
        budget - 1
    } else {
        total_rows
    };
    let mut rows: Vec<String> = text
        .lines()
        .take(kept_rows)
        .enumerate()
        .map(|(index, line)| {
            let line = if index == 0 {
                format!("◌ {line}")
            } else {
                line.to_string()
            };
            crate::ai::stream::clamp_line_to_terminal_row_with_reserve(&line, 1)
        })
        .collect();
    if kept_rows < total_rows {
        let marker = format!("… {} more rows below", total_rows - kept_rows);
        rows.push(crate::ai::stream::clamp_line_to_terminal_row_with_reserve(
            &marker, 1,
        ));
    }
    rows
}

fn erase_subagent_status_rows_at_size(
    drawn_rows: &mut Vec<String>,
    cols: usize,
    viewport_rows: usize,
    out: &mut dyn std::io::Write,
) {
    // Narrowing can wrap previously clamped rows. A height reduction can
    // move their prefix out of the viewport; erase only the visible suffix.
    let physical_rows = drawn_rows
        .iter()
        .map(|row| thinking_physical_rows(row, cols))
        .sum::<usize>()
        .min(viewport_rows.max(1));
    erase_status_rows(physical_rows, out);
    drawn_rows.clear();
}

/// Clear the in-place subagent status block before other output claims the
/// active line. No-op when the block was never painted.
fn hide_subagent_status(
    drawn_rows: &mut Vec<String>,
    line_open: &mut bool,
    tty: bool,
    out: &mut dyn std::io::Write,
) {
    if drawn_rows.is_empty() {
        return;
    }
    if tty {
        let (cols, viewport_rows) = subagent_terminal_size();
        erase_subagent_status_rows_at_size(drawn_rows, cols, viewport_rows, out);
        let _ = out.flush();
    }
    drawn_rows.clear();
    *line_open = false;
}

/// Erase the transient rows ending at the cursor, leaving the cursor at the
/// first erased row. Callers keep the cursor at the end of the last row.
fn erase_status_rows(rows: usize, out: &mut dyn std::io::Write) {
    let _ = write!(out, "\r\x1b[K");
    for _ in 1..rows {
        let _ = write!(out, "\x1b[1A\r\x1b[K");
    }
}

const SERVE_THINKING_BODY_INDENT: &str = "    ";
const SERVE_THINKING_BODY_INDENT_WIDTH: usize = 4;
const SERVE_THINKING_RIGHT_MARGIN_COLS: usize = 2;
const SERVE_THINKING_VIEWPORT_RESERVED_ROWS: usize = 3;
const SERVE_THINKING_TAIL_CHAR_CAP: usize = 131_072;

/// Bounded display-only state for the serve-chat thinking fold.
///
/// Canonical reasoning remains on the server. The client needs only enough
/// sanitized tail text to project the live window plus the plain rows from
/// the prior frame so a resize can recompute their current physical height.
struct ThinkingLiveRegion {
    tail: VecDeque<char>,
    tail_truncated: bool,
    drawn_rows: Vec<String>,
    header_rate: Option<String>,
    max_visible_lines: usize,
    terminal_width_override: Option<usize>,
}

impl ThinkingLiveRegion {
    fn new(max_visible_lines: usize, terminal_width_override: Option<usize>) -> Self {
        Self {
            tail: VecDeque::new(),
            tail_truncated: false,
            drawn_rows: Vec::new(),
            header_rate: None,
            max_visible_lines,
            terminal_width_override,
        }
    }

    fn reset(&mut self) {
        self.tail.clear();
        self.tail_truncated = false;
        self.drawn_rows.clear();
        self.header_rate = None;
    }

    /// Append model text after making every terminal control byte inert.
    /// Newlines remain structural; tabs become four spaces, and every other
    /// control is rendered through its visible Rust escape.
    fn append(&mut self, text: &str) {
        for ch in text.chars() {
            match ch {
                '\n' => self.push_display_char('\n'),
                '\t' => {
                    for _ in 0..4 {
                        self.push_display_char(' ');
                    }
                }
                ch if ch.is_control() => {
                    for escaped in ch.escape_default() {
                        self.push_display_char(escaped);
                    }
                }
                ch => self.push_display_char(ch),
            }
        }
    }

    fn push_display_char(&mut self, ch: char) {
        if self.tail.len() == SERVE_THINKING_TAIL_CHAR_CAP {
            self.tail.pop_front();
            self.tail_truncated = true;
        }
        self.tail.push_back(ch);
    }

    fn terminal_width(&self) -> usize {
        self.terminal_width_override
            .unwrap_or_else(crate::ai::stream::raw_terminal_cols)
            .max(1)
    }

    /// Keep the region inside the live viewport, matching the local fold's
    /// protection against cursor-up being clamped after a tall frame scrolls.
    fn visible_body_row_budget(&self) -> usize {
        let viewport_budget = if self.terminal_width_override.is_some() {
            usize::MAX
        } else {
            crossterm::terminal::size()
                .ok()
                .map(|(_, rows)| {
                    usize::from(rows)
                        .saturating_sub(SERVE_THINKING_VIEWPORT_RESERVED_ROWS)
                        .max(1)
                })
                .unwrap_or(usize::MAX)
        };
        self.max_visible_lines.min(viewport_budget)
    }

    /// Project the latest configured number of physical rows at `width`.
    /// The marker is outside that budget, as in the local fold.
    fn projected_rows(&self, width: usize) -> Vec<String> {
        let content_width = width
            .saturating_sub(SERVE_THINKING_BODY_INDENT_WIDTH + SERVE_THINKING_RIGHT_MARGIN_COLS)
            .max(1);
        let body_budget = self.visible_body_row_budget();
        let tail: String = self.tail.iter().collect();
        let mut visible = VecDeque::new();
        let mut hidden = self.tail_truncated;
        for line in tail.split('\n').filter(|line| !line.trim().is_empty()) {
            for row in wrap_thinking_line(line, content_width) {
                if visible.len() == body_budget {
                    visible.pop_front();
                    hidden = true;
                }
                if body_budget > 0 {
                    visible.push_back(row);
                } else {
                    hidden = true;
                }
            }
        }

        let mut rows = Vec::with_capacity(visible.len() + 2);
        let header = match self.header_rate.as_deref() {
            Some(rate) => format!("  ○ thinking · {rate}"),
            None => "  ○ thinking".to_string(),
        };
        rows.push(clamp_thinking_row(
            &header,
            width
                .saturating_sub(SERVE_THINKING_RIGHT_MARGIN_COLS)
                .max(1),
        ));
        if hidden {
            rows.push(format!(
                "{SERVE_THINKING_BODY_INDENT}{}",
                clamp_thinking_row("… more", content_width)
            ));
        }
        rows.extend(
            visible
                .into_iter()
                .map(|row| format!("{SERVE_THINKING_BODY_INDENT}{row}")),
        );
        rows
    }

    fn rendered_physical_rows(&self, width: usize) -> usize {
        self.drawn_rows
            .iter()
            .map(|row| thinking_physical_rows(row, width))
            .sum()
    }

    fn erase_at_width(&mut self, width: usize, out: &mut dyn std::io::Write) {
        let rows = self.rendered_physical_rows(width);
        if rows > 0 {
            erase_status_rows(rows, out);
        }
        self.drawn_rows.clear();
    }

    fn erase(&mut self, out: &mut dyn std::io::Write) {
        let width = self.terminal_width();
        self.erase_at_width(width, out);
    }

    fn redraw(&mut self, out: &mut dyn std::io::Write) {
        let width = self.terminal_width();
        self.erase_at_width(width, out);
        let rows = self.projected_rows(width);
        for (index, row) in rows.iter().enumerate() {
            if index > 0 {
                let _ = write!(out, "\r\n");
            }
            let _ = write!(out, "{}", dim(row, true));
        }
        let _ = out.flush();
        self.drawn_rows = rows;
    }
}

fn thinking_cell_width(ch: char) -> usize {
    crate::ai::stream::terminal_cell_width(ch)
}

fn wrap_thinking_line(line: &str, cols: usize) -> Vec<String> {
    let cols = cols.max(1);
    if line.is_empty() {
        return vec![String::new()];
    }
    let mut rows = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for ch in line.chars() {
        let width = thinking_cell_width(ch);
        if width > cols {
            if !current.is_empty() {
                rows.push(std::mem::take(&mut current));
                current_width = 0;
            }
            rows.push("?".to_string());
            continue;
        }
        if current_width > 0 && current_width + width > cols {
            rows.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(ch);
        current_width += width;
    }
    rows.push(current);
    rows
}

fn clamp_thinking_row(line: &str, cols: usize) -> String {
    let cols = cols.max(1);
    let width = line.chars().map(thinking_cell_width).sum::<usize>();
    if width <= cols {
        return line.to_string();
    }
    if cols == 1 {
        return "…".to_string();
    }
    let budget = cols - 1;
    let mut row = String::new();
    let mut used = 0usize;
    for ch in line.chars() {
        let width = thinking_cell_width(ch);
        if used + width > budget {
            break;
        }
        row.push(ch);
        used += width;
    }
    row.push('…');
    row
}

fn thinking_physical_rows(line: &str, cols: usize) -> usize {
    let cols = cols.max(1);
    let mut rows = 1usize;
    let mut used = 0usize;
    for ch in line.chars() {
        let width = thinking_cell_width(ch);
        if used > 0 && used + width > cols {
            rows += 1;
            used = width;
        } else {
            used += width;
        }
    }
    rows
}

/// Erase the complete live thinking region (if open) and print only the
/// folded summary, mirroring the local `✓ thinking (N lines)` close state.
/// Every close path (done/error/follow-up block/late delta or footer) shares
/// this helper so the summary is emitted at most once.
fn close_thinking_status(
    active: &mut bool,
    line_open: &mut bool,
    lines: usize,
    region: &mut ThinkingLiveRegion,
    tty: bool,
    out: &mut dyn std::io::Write,
    observer_rows: &mut ObserverRowState,
) {
    if !*active {
        return;
    }
    *active = false;
    if tty {
        region.erase(out);
    }
    region.reset();
    let unit = if lines == 1 { "line" } else { "lines" };
    let summary = format!("✓ thinking ({lines} {unit})");
    // An active close is always a genuine block completion. Clear any prior
    // block's key before classification so equal-sized blocks both print;
    // classification then seeds the key again for the server-mirrored
    // observer row that may follow this close.
    observer_rows.last_thinking_summary = None;
    if decide_observer_row(&summary, observer_rows, tty, 0, false).action
        != ObserverRowAction::SkipDuplicate
    {
        let _ = writeln!(out, "{}", dim(&summary, tty));
    }
    *line_open = false;
    let _ = out.flush();
}

/// Package client-local `[[image:name]]` files for upload with the turn.
/// The server shares no filesystem with this client, so every bare pasted
/// filename that resolves inside the client session assets dir is read and
/// base64-encoded here; the server stages the bytes under the same filename
/// before running, and its placeholder resolution then works unchanged.
/// Entries that are not bare names (explicit user paths stay server-side
/// references) or that have no local file are skipped — the server resolves
/// or reports those exactly as before. Repeated placeholders upload once.
fn collect_image_uploads(
    prompt: &str,
    history_file: &Path,
    session_id: &str,
) -> Vec<serde_json::Value> {
    use base64::Engine as _;

    let assets_dir = SessionStore::new(history_file).session_assets_dir(session_id);
    let mut uploads = Vec::new();
    for name in inline_image_filenames(prompt) {
        if name.contains('/') || name.contains('\\') {
            continue;
        }
        let path = assets_dir.join(&name);
        if !path.is_file() {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if bytes.is_empty() {
            continue;
        }
        uploads.push(serde_json::json!({
            "filename": name,
            "data_base64": base64::engine::general_purpose::STANDARD.encode(&bytes),
        }));
    }
    uploads
}

/// Whether a remote turn is streaming right now, i.e. the client is inside
/// [`post_turn_stream`]. The SIGINT handler runs on its own thread and cannot
/// see the loop's locals, so the flag lives at module scope.
static REMOTE_TURN_IN_FLIGHT: AtomicBool = AtomicBool::new(false);
/// Whether the in-flight turn already got its interrupt request: the first
/// Ctrl+C asks the server to stop it, a further one exits locally — the local
/// REPL's second-press escape hatch.
static REMOTE_TURN_INTERRUPT_SENT: AtomicBool = AtomicBool::new(false);

/// Connection and current session for the SIGINT handler. The session id is
/// rebound once per input box, which covers every path that starts a turn.
struct ServeSigintState {
    client: reqwest::blocking::Client,
    base: String,
    token: String,
    session_id: std::sync::Mutex<String>,
}

impl ServeSigintState {
    fn bind_session(&self, session_id: &str) {
        if let Ok(mut current) = self.session_id.lock() {
            current.clear();
            current.push_str(session_id);
        }
    }

    fn current_session(&self) -> String {
        self.session_id
            .lock()
            .map(|current| current.clone())
            .unwrap_or_default()
    }
}

/// What one Ctrl+C in serve-chat should do: while a remote turn runs and has
/// not been interrupted yet, stop that turn; otherwise leave the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServeSigintAction {
    InterruptRemoteTurn,
    ExitLocal,
}

fn serve_sigint_action(turn_in_flight: bool, interrupt_sent: bool) -> ServeSigintAction {
    if turn_in_flight && !interrupt_sent {
        ServeSigintAction::InterruptRemoteTurn
    } else {
        ServeSigintAction::ExitLocal
    }
}

/// SIGINT handler for `--serve-chat`. The client is a thin shell around the
/// server, so Ctrl+C must reach the remote turn instead of killing this
/// process: without a handler the default action exits the client mid-stream
/// and leaves the turn running to completion on the server.
fn handle_serve_sigint(state: &ServeSigintState) {
    match serve_sigint_action(
        REMOTE_TURN_IN_FLIGHT.load(Ordering::SeqCst),
        REMOTE_TURN_INTERRUPT_SENT.load(Ordering::SeqCst),
    ) {
        ServeSigintAction::InterruptRemoteTurn => {
            REMOTE_TURN_INTERRUPT_SENT.store(true, Ordering::SeqCst);
            let session_id = state.current_session();
            match interrupt_remote_turn(&state.client, &state.base, &state.token, &session_id) {
                // The turn ends through the SSE stream; the loop reports the
                // outcome once the stream returns.
                Ok(true) => {}
                Ok(false) => eprintln!(
                    "\n[serve-chat] no remote turn running for {session_id}; nothing to interrupt."
                ),
                Err(err) => eprintln!("\n[serve-chat] interrupt request failed: {err}"),
            }
        }
        ServeSigintAction::ExitLocal => {
            // Nothing to interrupt (or the turn is already stopping): mirror
            // the local REPL, which exits on Ctrl+C in the idle state.
            eprintln!("\n[serve-chat] exit.");
            std::process::exit(130);
        }
    }
}

/// Ask the server to stop the in-flight turn for `session_id`: the remote
/// counterpart of the local first Ctrl+C. `Ok(false)` means the server had no
/// turn to interrupt (it already finished), which is not an error.
fn interrupt_remote_turn(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let resp = auth(
        client.post(format!("{base}/sessions/{session_id}/interrupt")),
        token,
    )
    // Short timeout: this runs on the SIGINT handler thread while the main
    // thread keeps reading the SSE body, so a wedged server must not pin it.
    .timeout(Duration::from_secs(10))
    .send()?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text).into());
    }
    Ok(serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|value| value.get("interrupted").and_then(|v| v.as_bool()))
        .unwrap_or(false))
}

/// Remote-confirmation answer channel for a serve-chat turn: asks the local
/// user the question a server-side turn child is blocked on, then POSTs the
/// answer to `POST /sessions/{id}/confirm`, where the daemon relays it into
/// the child's stdin. Only built when the turn runs on an interactive
/// terminal (see [`serve_confirm_channel`]); otherwise the turn is sent
/// without `"confirm": true` and the child keeps the fail-closed default.
struct ServeConfirmChannel {
    poster: reqwest::blocking::Client,
    base: String,
    token: String,
    session_id: String,
}

/// Build the [`ServeConfirmChannel`] for a turn, or `None` when this client
/// cannot answer: without a terminal stdin the y/n prompt could neither show
/// nor read (`prompt_yes_or_no` spins forever on EOF), so opting in would
/// hang the turn on a question nobody can see.
fn serve_confirm_channel(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
) -> Option<ServeConfirmChannel> {
    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() {
        return None;
    }
    Some(ServeConfirmChannel {
        poster: client.clone(),
        base: base.to_string(),
        token: token.to_string(),
        session_id: session_id.to_string(),
    })
}

/// Exact body of `POST /sessions/{id}/confirm` (see `turn::ConfirmAnswerReq`):
/// the question id, the token it was shown with, and the verdict. Pure so
/// tests can pin the wire shape without touching stdin or the network.
fn confirm_answer_body(id: u64, token: u64, allow: bool) -> serde_json::Value {
    serde_json::json!({"id": id, "token": token, "allow": allow})
}

impl ServeConfirmChannel {
    /// Ask the local user and POST the answer. Runs on the render thread: the
    /// child cannot proceed without the answer, so pausing the stream here is
    /// correct (pending SSE bytes stay buffered in TCP). The renderer is
    /// already flushed by the caller, so the question starts on a fresh row.
    fn answer(&self, id: u64, token: u64, prompt: &str, out: &mut dyn std::io::Write) {
        let allow = crate::commonw::prompt::prompt_yes_or_no_interruptible(&format!(
            "[serve-chat] remote confirmation: {prompt} [y/n] "
        ));
        let allow = match allow {
            Some(v) => v,
            // Ctrl+C / Esc / read failure must not leave the child blocked
            // forever: deny fail-closed, like the local gate cancelling the
            // command. Send another key (or Ctrl+C during later output) to
            // interrupt the turn itself.
            None => {
                let _ = writeln!(out, "[serve-chat] confirmation dismissed; answering no.");
                false
            }
        };
        self.post_answer(id, token, allow, out);
    }

    /// POST one answer; every outcome is reported, none hangs the loop. A 409
    /// is terminal per the route contract (answered elsewhere, superseded, or
    /// the turn ended), not a retry. Other failures leave the child blocked
    /// until the turn ends, which the warning says plainly.
    fn post_answer(&self, id: u64, token: u64, allow: bool, out: &mut dyn std::io::Write) {
        let url = format!("{}/sessions/{}/confirm", self.base, self.session_id);
        // Reuse the shared client so private-CA trust is preserved, while the
        // per-request timeout prevents a wedged answer from pinning rendering.
        match auth(self.poster.post(url), &self.token)
            .timeout(Duration::from_secs(15))
            .json(&confirm_answer_body(id, token, allow))
            .send()
        {
            Ok(resp) if resp.status().is_success() => {
                let _ = writeln!(
                    out,
                    "[serve-chat] confirmation sent ({}).",
                    if allow { "yes" } else { "no" }
                );
            }
            Ok(resp) if resp.status().as_u16() == 409 => {
                let _ = writeln!(
                    out,
                    "[serve-chat] confirmation already resolved elsewhere; continuing."
                );
            }
            Ok(resp) => {
                let _ = writeln!(
                    out,
                    "[serve-chat] warning: confirmation answer rejected ({}); the remote \
                     command stays blocked until the turn ends.",
                    resp.status()
                );
            }
            Err(err) => {
                let _ = writeln!(
                    out,
                    "[serve-chat] warning: confirmation answer not delivered ({err}); the remote \
                     command stays blocked until the turn ends."
                );
            }
        }
        let _ = out.flush();
    }
}

/// Marks a remote turn as in flight for the SIGINT handler; the mark drops
/// when the stream returns, including the error paths.
struct RemoteTurnScope;

impl RemoteTurnScope {
    fn enter() -> Self {
        REMOTE_TURN_INTERRUPT_SENT.store(false, Ordering::SeqCst);
        REMOTE_TURN_IN_FLIGHT.store(true, Ordering::SeqCst);
        Self
    }
}

impl Drop for RemoteTurnScope {
    fn drop(&mut self) {
        REMOTE_TURN_IN_FLIGHT.store(false, Ordering::SeqCst);
    }
}

/// The text a routed input sends to the server.
///
/// A prompt keeps the bytes the user typed: the server stores the prompt
/// unchanged, so trimming on the way out would make the transcript differ from
/// the input (leading indentation in a pasted block, trailing markdown hard
/// breaks). A command line is trimmed because its grammar owns the words and
/// the handlers expect the command word first.
fn routed_prompt_text<'a>(route: &ServeInputRoute, typed: &'a str, trimmed: &'a str) -> &'a str {
    match route {
        ServeInputRoute::Prompt => typed,
        _ => trimmed,
    }
}

fn post_turn_stream(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    prompt: &str,
    title_seed: Option<&str>,
    history_file: &Path,
    selection: &mut ServeTurnSelection,
) -> Result<(), Box<dyn std::error::Error>> {
    // From here until the stream returns, Ctrl+C belongs to the remote turn.
    let _turn_scope = RemoteTurnScope::enter();
    let images = collect_image_uploads(prompt, history_file, session_id);
    {
        // A bare `[[image:name]]` with no local file uploads nothing, and the
        // server then fails the turn with a bare os-error-2. Warn here where
        // the cause (usually a paste saved under a previous session before
        // `/resume`/`/new`) is still actionable.
        let uploaded: std::collections::HashSet<&str> = images
            .iter()
            .filter_map(|v| v.get("filename").and_then(|f| f.as_str()))
            .collect();
        let mut missing = Vec::new();
        for name in inline_image_filenames(prompt) {
            if name.contains('/') || name.contains('\\') {
                continue;
            }
            if uploaded.contains(name.as_str()) || missing.iter().any(|m| m == &name) {
                continue;
            }
            missing.push(name);
        }
        if !missing.is_empty() {
            let assets_dir = SessionStore::new(history_file).session_assets_dir(session_id);
            eprintln!(
                "[serve-chat] warning: image file(s) {} missing, empty, or unreadable in local session assets ({}); re-paste the image in this session before sending.",
                missing.join(", "),
                assets_dir.display()
            );
        }
    }
    // Remote confirmations are answered on this terminal: opt the child into
    // the confirm channel only when stdin can actually ask (see
    // `serve_confirm_channel`). The daemon relays the answer into the child's
    // stdin, so the same gates as a local turn (`git commit` / `git push` / `git stash` / `git checkout` / `git switch`)
    // become approvable instead of failing closed.
    let confirm_channel = serve_confirm_channel(client, base, token, session_id);
    let resp = auth(
        client.post(format!("{base}/sessions/{session_id}/turns/stream")),
        token,
    )
    .json(&serde_json::json!({
        "prompt": prompt,
        "title_seed": title_seed,
        "images": images,
        "model": if selection.model.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(selection.model.clone())
        },
        "agent": if selection.agent.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(selection.agent.clone())
        },
        "reasoning_effort": match selection.reasoning_effort.as_deref() {
            Some(effort) => serde_json::Value::String(effort.to_string()),
            None => serde_json::Value::Null,
        },
        "skills": selection.skills.clone(),
        "confirm": confirm_channel.is_some(),
        // The width this terminal draws at. The turn child renders the rows
        // the client displays, but its own stdout is a pipe and cannot know
        // that width, so it is reported here.
        "cols": u16::try_from(crate::ai::stream::raw_terminal_cols()).unwrap_or(u16::MAX),
    }))
    .send()?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().unwrap_or_default();
        return Err(server_error_message(status, &text).into());
    }
    // Next-turn-only parity with the local `/skills use`: once the server
    // accepted the turn, the pin is consumed by this turn. A refused send
    // keeps the selection so a retry still carries it.
    selection.skills.clear();
    // Same Ctrl+G side-note composer as the local REPL: stdin is idle while
    // the stream renders, so the listener owns it until the turn ends. Drafts
    // are POSTed to the serve session; the server-side turn child drains them
    // like local notes. Degrades to a no-op on non-tty stdin.
    let _side_note_guard = if super::super::stream::side_note_input::side_note_input_enabled() {
        Some(SideNoteInputGuard::spawn_remote(remote_side_note_sink(
            client, base, token, session_id,
        )))
    } else {
        None
    };
    let outcome = render_turn_stream_with_confirm(resp, confirm_channel.as_ref());
    if REMOTE_TURN_INTERRUPT_SENT.swap(false, Ordering::SeqCst) {
        // Leading newline: `delta` events print without a trailing newline, so
        // the note would otherwise land mid-line.
        println!("\n[serve-chat] interrupt sent; remote turn stopped.");
    }
    outcome
}

/// Build the remote side-note sink for a serve-chat turn: POST one draft to
/// `POST /sessions/{id}/side-notes`. Any failure returns false so the
/// composer keeps the draft for retry instead of silently losing it.
fn remote_side_note_sink(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
) -> Arc<dyn Fn(String) -> bool + Send + Sync> {
    let url = format!("{base}/sessions/{session_id}/side-notes");
    let token = token.to_string();
    // Reuse the shared client so private-CA trust is preserved. The request
    // keeps its own short timeout so a wedged server cannot pin the composer.
    let poster = client.clone();
    Arc::new(move |text: String| {
        auth(poster.post(url.clone()), &token)
            .timeout(Duration::from_secs(15))
            .json(&serde_json::json!({"text": text}))
            .send()
            .map(|resp| resp.status().is_success())
            .unwrap_or(false)
    })
}

/// Shared `--serve-chat` / `--serve-sessions` dial-up: resolve the bind,
/// build the client, and fail fast with a copy-pasteable hint when the
/// server is not up. `tool` names the caller in error hints only.
fn serve_connection(
    cli: &ParsedCli,
    tool: &str,
) -> Result<(reqwest::blocking::Client, String, String), Box<dyn std::error::Error>> {
    let cfg = configw::get_all_config();
    // Same precedence as the server and the management commands: explicit
    // `--serve-bind` wins, then `ai.serve.bind`, then the default.
    let from_cfg = cfg.get_opt(AiConfig::SERVE_BIND).unwrap_or_default();
    let bind = super::resolve_serve_bind(&cli.serve_bind, &from_cfg);
    let base = normalize_base(&bind)?;
    let token = cfg.get_opt(AiConfig::SERVE_TOKEN).unwrap_or_default();
    let ca_from_cfg = cfg.get_opt(AiConfig::SERVE_CA_CERT).unwrap_or_default();
    let ca_cert = resolve_serve_ca_cert(&cli.serve_ca_cert, &ca_from_cfg);
    let client = build_serve_client(&ca_cert, Duration::from_secs(600))?;

    // Fail fast with a copy-pasteable hint when the server is not up.
    let health = client.get(format!("{base}/healthz")).send();
    match health {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => {
            return Err(format!(
                "{tool}: serve at {base} answered {}; is `a --serve` running there?",
                resp.status()
            )
            .into());
        }
        Err(err) => {
            return Err(format!(
                "{tool}: cannot reach serve at {base} ({err}); for HTTPS use `--serve-bind https://<host>:<port>` and, for a private CA, `--serve-ca-cert <ca.pem>`"
            )
            .into());
        }
    }
    Ok((client, base, token))
}

/// Highlight style for remote sessions marked important, mirroring the local
/// `/ss` view: ANSI colors only on a terminal, piped output stays plain.
fn marked_remote_session_style(tty: bool) -> (&'static str, &'static str) {
    if tty {
        (
            crate::ai::theme::current().accent_marked,
            crate::ai::theme::RESET,
        )
    } else {
        ("", "")
    }
}

/// One `/ss` row of a `GET /sessions` listing. With a current session (the
/// interactive client) the row carries a `*`/` ` mark column, like the local
/// view; without one (`None`, the non-interactive `--serve-sessions`) the row
/// keeps the legacy `N. <id>  <title>` bytes so piped output stays stable.
/// Marked rows carry the theme's marked highlight and archived rows render
/// dimmed, both terminal-only like the local view. `tty` is a parameter so
/// tests can pin both renderings without a terminal.
fn format_remote_session_row(
    item: &RemoteSession,
    index: usize,
    current_session_id: Option<&str>,
    tty: bool,
) -> String {
    let archived_suffix = if item.archived { "  [archived]" } else { "" };
    let interactive_tty = current_session_id.is_some() && tty;
    let (style_open, style_close) = if item.marked {
        marked_remote_session_style(interactive_tty)
    } else if item.archived && interactive_tty {
        (crate::ai::theme::DIM, crate::ai::theme::RESET)
    } else {
        ("", "")
    };
    match current_session_id {
        Some(current) => {
            let mark = if item.id == current { "*" } else { " " };
            format!(
                "{style_open}{mark} {}. {}  {}{archived_suffix}{style_close}",
                index + 1,
                item.id,
                remote_session_title(item)
            )
        }
        None => format!(
            "{style_open}{}. {}  {}{archived_suffix}{style_close}",
            index + 1,
            item.id,
            remote_session_title(item)
        ),
    }
}

/// Print one `GET /sessions` listing as numbered rows (newest first) on
/// stdout, so `--serve-sessions` output stays pipeable for scripts and later
/// mobile clients; the interactive client passes its session id so the
/// current row is marked, like the local REPL.
fn print_remote_sessions(
    items: &[RemoteSession],
    empty_message: &str,
    current_session_id: Option<&str>,
) {
    if items.is_empty() {
        println!("{empty_message}");
    } else {
        let tty = std::io::stdout().is_terminal();
        for (i, item) in items.iter().enumerate() {
            println!(
                "{}",
                format_remote_session_row(item, i, current_session_id, tty)
            );
        }
    }
}

/// Non-interactive `--serve-sessions`: print the remote session list and
/// exit, so the session can be picked before opening the interactive client.
pub(in crate::ai) fn run_serve_sessions(cli: &ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    let (client, base, token) = serve_connection(cli, "serve-sessions")?;
    let items = fetch_remote_sessions(&client, &base, &token)
        .map_err(|err| format!("[serve-sessions] {err}"))?;
    print_remote_sessions(&items, "No remote sessions.", None);
    Ok(())
}

/// What `/sessions|/ss|/ls [arg]` should do in serve-chat, mirroring the
/// local REPL subcommands: bare/`list` lists active sessions (archived hidden,
/// current always shown), `all` lists everything, `archive` lists formally
/// archived sessions, `current` (also `cur`/`curr`) shows the current one,
/// `id` prints just its id, and any other argument is rejected instead of
/// silently listing everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServeSessionsArg {
    List,
    All,
    Archive,
    Current,
    Id,
    Unknown,
}

fn classify_serve_sessions_arg(arg: &str) -> ServeSessionsArg {
    match arg {
        "" | "list" | "ls" | "verbose" => ServeSessionsArg::List,
        "all" => ServeSessionsArg::All,
        "archive" => ServeSessionsArg::Archive,
        "current" | "cur" | "curr" => ServeSessionsArg::Current,
        "id" => ServeSessionsArg::Id,
        _ => ServeSessionsArg::Unknown,
    }
}

/// Rows shown by one `/ss` listing view. The default view hides formally
/// archived sessions but keeps the current one visible, so archiving never
/// hides where the user is; `all` shows everything; `archive` shows only
/// archived sessions.
fn sessions_for_view(
    items: Vec<RemoteSession>,
    view: ServeSessionsArg,
    current_session_id: &str,
) -> Vec<RemoteSession> {
    items
        .into_iter()
        .filter(|item| match view {
            ServeSessionsArg::All => true,
            ServeSessionsArg::Archive => item.archived,
            _ => !item.archived || item.id == current_session_id,
        })
        .collect()
}

/// Run the interactive serve chat: connect to a running serve instance and
/// loop over multiline prompts until `/quit`. Input uses the same editor as
/// the local REPL, so Enter/movement/completion behave identically.
pub(in crate::ai) fn run_serve_chat(cli: ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    let (client, base, token) = serve_connection(&cli, "serve-chat")?;
    // Ctrl+C must stop the remote turn rather than this client: without a
    // handler, SIGINT's default action would exit mid-stream and leave the
    // turn running on the server.
    let sigint_state = Arc::new(ServeSigintState {
        client: client.clone(),
        base: base.clone(),
        token: token.clone(),
        session_id: std::sync::Mutex::new(String::new()),
    });
    {
        let sigint_state = Arc::clone(&sigint_state);
        ctrlc::set_handler(move || handle_serve_sigint(&sigint_state))?;
    }
    let app_config = super::super::config::load_config()?;

    let mut session_id = match cli.session {
        Some(id) if !id.trim().is_empty() => id.trim().to_string(),
        // No explicit session: pick up the `/bg` binding this terminal left
        // behind, provided it still points at a live session on this server.
        _ => match load_serve_chat_binding() {
            Some(binding) if binding.base == base => {
                match fetch_remote_sessions(&client, &base, &token) {
                    Ok(items) if items.iter().any(|s| s.id == binding.session_id) => {
                        println!(
                            "Resumed backgrounded remote session {}.",
                            binding.session_id
                        );
                        binding.session_id
                    }
                    Ok(_) => {
                        println!(
                            "Backgrounded remote session {} is gone; starting a new session.",
                            binding.session_id
                        );
                        create_session(&client, &base, &token)?
                    }
                    // Unreachable list: keep the bound id optimistically; the
                    // per-input refresh heals or replaces it once reachable.
                    Err(_) => binding.session_id,
                }
            }
            _ => create_session(&client, &base, &token)?,
        },
    };

    println!("Connected to {base}, session {session_id}.");
    let use_usage = crate::ai::tools::task_tools::backends::backend_registry().use_usage();
    println!(
        "Enter inserts a newline, Esc or Alt+Enter submits; Ctrl+C interrupts the running remote turn (a second Ctrl+C exits); /quit exits, /bg backgrounds the current session and exits, /new starts a new session, /sessions (or /ss, same as the local REPL) lists remote sessions (/sessions current|id shows the current one), /fork branches the current session, /close deletes it and exits. /cd switches the remote working directory (Tab completes remote paths; /pwd prints it). /model, /effort, /agent switch the remote turn like the local REPL. {use_usage} shows or switches the remote session's default subagent backend; /reset restores kernel subagent defaults and clears sticky agent-team. /task and /discuss run multi-worker rooms on the server (/task help, /discuss help)."
    );
    let mut editor = PromptEditor::new(&session_id, &app_config.history_file);
    // Command words the shared completion table does not carry: serve-chat-only
    // commands plus shared ones missing from the trie. Registered per client
    // process, so the local REPL never suggests them.
    CommandCompleter::set_extra_command_words(
        ["/bg", "/new", "/quit", "/resume"]
            .iter()
            .map(|word| (*word).to_string())
            .collect(),
    );
    // `/cd` Tab-completion must list the remote server's directories for the
    // currently selected session (its cwd lives server-side), so completion
    // fetches candidates over HTTP instead of reading this client's disk.
    // The shared cell below follows `/resume` `/new` `/fork` switches.
    let current_session: Arc<std::sync::Mutex<String>> =
        Arc::new(std::sync::Mutex::new(session_id.clone()));
    install_serve_cd_completion(&client, &base, &token, current_session.clone());
    // Skill Tab-completion must list the remote server's skills: the shared
    // cache starts empty (Some) so completion never falls back to this
    // client's local disk, then the quiet refresh below fills it.
    CommandCompleter::set_skill_name_descriptions(Vec::new());
    if let Some(entries) = refresh_serve_skills_quiet(&client, &base, &token) {
        apply_serve_skills_to_completion(&entries);
    }
    // Last `/sessions` view and its rows: backs `/resume <number|id-prefix>`.
    // The silent refresh below re-applies the view, so a filtered listing keeps
    // the row numbers the user just saw.
    let mut listed_view = ServeSessionsArg::List;
    let mut listed: Vec<RemoteSession> = Vec::new();
    // Client-side per-turn selection, seeded from the server (`GET /info`)
    // and switched locally by `/model`/`/effort`/`/agent`; every turn carries
    // the selection as request overrides so the child CLI resolves the same
    // registry path as the local REPL. The model label keeps a persistent
    // `(remote)` marker. An unreachable `/info` keeps blank labels; chat
    // still works.
    let info = fetch_server_info(&client, &base, &token).unwrap_or_default();
    apply_remote_header(&mut editor, &info, None);
    let mut selection = ServeTurnSelection::from_info(&info);
    // A reconnect resumes the session's stored picks (`/model`/`/effort`/
    // `/agent` changes are persisted server-side), so `a --serve-chat
    // --session <id>` after a ctrl+c keeps the last state instead of falling
    // back to server defaults.
    restore_serve_session_picks(&client, &base, &token, &session_id, &mut selection);
    selection.apply_header(&mut editor);
    let agent_manifests = agents::load_all_agents();
    loop {
        sigint_state.bind_session(&session_id);
        // Silent live refresh before every input box: server-generated titles
        // (they appear after the first turn) and fresh `/resume` completion
        // candidates. The last explicit view is re-applied so its numbering
        // stays valid. Failures keep the previous listing; the prompt is never
        // blocked by this.
        if let Some(items) = refresh_listing_quiet(&client, &base, &token) {
            editor.set_session_topic(current_remote_title(&items, &session_id));
            let items = sessions_for_view(items, listed_view, &session_id);
            CommandCompleter::set_serve_session_candidates(
                items
                    .iter()
                    .map(|s| (s.id.clone(), remote_session_title(s)))
                    .collect(),
            );
            listed = items;
        }
        if let Ok(checkpoint_state) =
            fetch_remote_checkpoint_names(&client, &base, &token, &session_id)
        {
            CommandCompleter::set_checkpoint_name_candidates(checkpoint_state.checkpoints);
            CommandCompleter::set_checkpoint_branch_candidates(checkpoint_state.branches);
            editor.set_remote_checkpoint_status_label(checkpoint_state.checkpoint_status);
        }
        if let Some(entries) = refresh_serve_skills_quiet(&client, &base, &token) {
            apply_serve_skills_to_completion(&entries);
        }
        let input = match editor.read_multi_line() {
            Ok(Some(text)) => text,
            Ok(None) => break,
            // Ctrl+C in the input box: the same exit as the local REPL's
            // prompt interrupt, not an error report.
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {
                println!("Exit.");
                break;
            }
            Err(err) => {
                eprintln!("[serve-chat] input error: {err}");
                break;
            }
        };
        let trimmed = input.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        // Split the command word from its argument so `/resume <id>` works;
        // anything else keeps the full text (prompts starting with `/` still
        // send verbatim, exactly as before).
        let mut words = trimmed.splitn(2, char::is_whitespace);
        let cmd = words.next().unwrap_or("");
        let arg = words.next().unwrap_or("").trim();
        match cmd {
            "/quit" | "/exit" | ":q" => break,
            "/help" | "/h" => {
                println!(
                    "Enter inserts a newline, Esc or Alt+Enter submits (same as the local REPL).\n/quit - exit\n/new - start a new session\n/sessions (/ss, same as the local REPL) - list remote sessions (newest first, archived hidden); /sessions all - include archived; /sessions archive - list archived; /sessions current|cur|curr - show the current session; /sessions id - print its id\n/resume <number|id-prefix|id> - continue a listed session (Tab completes after /sessions)\n/fork - branch the current session and switch to it\n/close - delete the current remote session and exit\n/git [commit|checkout|branch|switch|tree|list|show|delete|clear|help] - operate on the remote session checkpoints and branches\n/history [N|full|user|assistant|tool|system|u<N>|grep <kw>|last [N]|replay|rewind <uN|last|grep <kw>>|help] - remote history over HTTP (never a turn)\n/export [path] - export the latest server-side assistant conclusion to a markdown file on the server (relative to the remote cwd; default: _summary.md)\n/cd [dir] - switch the remote working directory (Tab completes remote paths; `-` goes back)\n/pwd - print the remote working directory (same as bare /cd)\n/model <selector> [question] - switch remote model (/model list|current|help; question sends immediately)\n/effort <level> - switch remote reasoning effort (minimal|low|medium|high|xhigh|max|off|auto)\n/agent <name> - switch remote agent (/agent list|current|help)\n{use_usage} - show or switch the remote session's default subagent backend (also :use); a task on a new line runs immediately\n/reset - restore kernel subagent defaults and clear sticky agent-team\n/audit [-f|--fast] [instruction] - run the audit subagent on the server, then verify and fix findings\n/task [N] <goal> - run a multi-worker task room on the server (/task status [room] shows progress; /task help for details)\n/discuss [N] <topic> - run a multi-member discussion on the server (/discuss status [room] shows progress; /discuss help for details)\n/distill-session [session-id|archive.zip] [--dry-run] [--limit N] - distill a session into knowledge on the server (paths resolve on the server)\n/title [text] - set or show the current server-side session title (bare /title shows it)\n/mark [<id>|current|last] [message] - mark a server-side session as important (default: current; leading `--` makes the rest a literal message for current)\n/unmark [<id>|current|last|--] - remove a server-side important mark (default: current; `--` forces current)\n/archive [<id>|current|last] - toggle the server-side formal archive flag (default: current)\n/bg - exit and bind this terminal to the current remote session\nCtrl+C - interrupt the running remote turn (a second Ctrl+C exits the client)"
                );
                continue;
            }
            "/new" => match create_session(&client, &base, &token) {
                Ok(id) => {
                    session_id = id;
                    if let Ok(mut cell) = current_session.lock() {
                        *cell = session_id.clone();
                    }
                    CommandCompleter::clear_checkpoint_candidates();
                    editor.set_session_id(session_id.as_str());
                    println!("New session {session_id}.");
                }
                Err(err) => eprintln!("[serve-chat] {err}"),
            },
            // `/ss` matches the local REPL shorthand (`/ss` === `/sessions`
            // list) and its subcommands (`all`/`archive`/`current`/`id`) so the
            // same muscle memory works on both sides; unknown arguments are
            // rejected instead of silently listing every session.
            "/sessions" | "/ss" | "/ls" => match classify_serve_sessions_arg(arg) {
                view @ (ServeSessionsArg::List
                | ServeSessionsArg::All
                | ServeSessionsArg::Archive) => {
                    match fetch_remote_sessions(&client, &base, &token) {
                        Ok(items) => {
                            let items = sessions_for_view(items, view, &session_id);
                            let empty = if view == ServeSessionsArg::Archive {
                                "No archived sessions."
                            } else {
                                "No remote sessions."
                            };
                            print_remote_sessions(&items, empty, Some(session_id.as_str()));
                            // Back `/resume` Tab-completion and numbering with
                            // the rows just printed; the silent refresh before
                            // the next input re-applies this view.
                            listed_view = view;
                            CommandCompleter::set_serve_session_candidates(
                                items
                                    .iter()
                                    .map(|s| (s.id.clone(), remote_session_title(s)))
                                    .collect(),
                            );
                            listed = items;
                        }
                        Err(err) => eprintln!("[serve-chat] {err}"),
                    }
                }
                ServeSessionsArg::Current => {
                    // Local `/ss current` reads the local store; remotely the
                    // listing carries the same per-session fields, so refresh
                    // it and show the current session's row. If the current
                    // session is missing (created elsewhere after the last
                    // fetch) the id still prints.
                    match fetch_remote_sessions(&client, &base, &token) {
                        Ok(items) => {
                            println!("session: {session_id}");
                            if let Some(item) = items.iter().find(|s| s.id == session_id) {
                                println!("title: {}", remote_session_title(item));
                                println!("marked: {}", if item.marked { "yes" } else { "no" });
                                println!("archived: {}", if item.archived { "yes" } else { "no" });
                            }
                        }
                        Err(err) => eprintln!("[serve-chat] {err}"),
                    }
                }
                ServeSessionsArg::Id => println!("{session_id}"),
                ServeSessionsArg::Unknown => {
                    println!(
                        "unknown /sessions subcommand: '{arg}'. try: /ss list|current|id|all|archive"
                    )
                }
            },
            "/resume" => match resolve_session_ref(arg, &listed) {
                Ok(id) => {
                    let title = listed.iter().find(|s| s.id == id).map(remote_session_title);
                    session_id = id;
                    if let Ok(mut cell) = current_session.lock() {
                        *cell = session_id.clone();
                    }
                    CommandCompleter::clear_checkpoint_candidates();
                    editor.set_session_id(session_id.as_str());
                    if restore_serve_session_picks(
                        &client,
                        &base,
                        &token,
                        &session_id,
                        &mut selection,
                    ) {
                        selection.apply_header(&mut editor);
                    }
                    match title {
                        Some(t) => println!("Resumed session {session_id} ({t})."),
                        None => println!("Resumed session {session_id}."),
                    }
                }
                Err(err) => eprintln!("[serve-chat] {err}"),
            },
            "/fork" => match fork_remote_session(&client, &base, &token, &session_id) {
                Ok(id) => {
                    println!("Forked '{session_id}' -> '{id}', switched to new branch.");
                    session_id = id;
                    if let Ok(mut cell) = current_session.lock() {
                        *cell = session_id.clone();
                    }
                    CommandCompleter::clear_checkpoint_candidates();
                    editor.set_session_id(session_id.as_str());
                }
                Err(err) => eprintln!("[serve-chat] {err}"),
            },
            // Local `/close` deletes the session and exits; it also drops
            // that session's suspended bindings, whose remote equivalent is
            // this terminal's `/bg` binding file (the startup existence check
            // would heal it anyway, but an explicit delete should not leave a
            // dangling pointer).
            "/close" => match delete_remote_session(&client, &base, &token, &session_id) {
                Ok(_) => {
                    println!("Closed remote session {session_id}.");
                    clear_serve_chat_binding();
                    break;
                }
                Err(err) => eprintln!("[serve-chat] {err}"),
            },
            // Local parity for the suspend family (`/bg`, `/suspend`,
            // `/detach`, `/susp`, with `/` or `:` prefix): exit the
            // interactive input and bind this terminal to the current remote
            // session, so the next `a --serve-chat` here resumes it.
            "/bg" | ":bg" | "/suspend" | ":suspend" | "/detach" | ":detach" | "/susp" | ":susp" => {
                let resume = format!("a --serve-chat --session {session_id}");
                match save_serve_chat_binding(&base, &session_id) {
                    Ok(()) => println!(
                        "Backgrounded remote session {session_id}; resume with `{resume}` (or just `a --serve-chat` in this terminal)."
                    ),
                    Err(err) => println!(
                        "Backgrounded remote session {session_id} (binding not saved: {err}); resume with `{resume}`."
                    ),
                }
                break;
            }
            // Remote switching parity: like the local REPL, `/model <sel>
            // [question]` switches and optionally sends, `/effort <level>`
            // and `/agent <name>` switch for the next turn. Both `/` and `:`
            // prefixes work, matching the local command forms.
            "/model" | ":model" | "/models" => {
                let (question, patch) = run_serve_model_command(arg, &mut selection, &mut editor);
                persist_serve_config_patch(&client, &base, &token, &session_id, &patch);
                if let Some(question) = question {
                    if let Err(err) = post_turn_stream(
                        &client,
                        &base,
                        &token,
                        &session_id,
                        &question,
                        None,
                        &app_config.history_file,
                        &mut selection,
                    ) {
                        eprintln!("[serve-chat] {err}");
                    }
                }
            }
            "/effort" | ":effort" => {
                let patch = run_serve_effort_command(arg, &mut selection, &mut editor);
                persist_serve_config_patch(&client, &base, &token, &session_id, &patch);
            }
            "/agent" | ":agent" => {
                let patch =
                    run_serve_agent_command(arg, &mut selection, &agent_manifests, &mut editor);
                persist_serve_config_patch(&client, &base, &token, &session_id, &patch);
            }
            // Remote `/skills`: list/use run against the server skill list
            // (`GET /skills`); the pending selection rides the next turn as
            // `@skills:` tokens and is consumed once the server accepts it.
            "/skills" | ":skills" | "/skill" | ":skill" => {
                if let Some(question) =
                    run_serve_skills_command(&client, &base, &token, arg, &mut selection)
                {
                    if let Err(err) = post_turn_stream(
                        &client,
                        &base,
                        &token,
                        &session_id,
                        &question,
                        None,
                        &app_config.history_file,
                        &mut selection,
                    ) {
                        eprintln!("[serve-chat] {err}");
                    }
                }
            }
            // Remote `/history`: previews and rewinds go through the history
            // HTTP routes, never through a turn child (a child would run the
            // local command against server state; `rewind` would block on a
            // stdin confirm nobody can answer and hang the session).
            "/history" | ":history" => {
                if let Err(err) =
                    run_serve_history_command(&client, &base, &token, &session_id, arg)
                {
                    eprintln!("[serve-chat] {err}");
                }
            }
            "/git" | ":git" => {
                match run_serve_checkpoint_command(&client, &base, &token, &session_id, &trimmed) {
                    Ok(checkpoint_status) => {
                        editor.set_remote_checkpoint_status_label(checkpoint_status);
                    }
                    Err(err) => eprintln!("[serve-chat] {err}"),
                }
            }
            // Remote `/cd`/`/pwd`: the session working directory lives on the
            // server (every turn child is spawned with the session's cwd), so
            // the switch — or the read-only print — runs there over HTTP and
            // the reported result is the remote path; this client's local
            // directory is untouched. The server shares one grammar for both,
            // so the raw line forwards verbatim.
            "/cd" | ":cd" | "/pwd" | ":pwd" => {
                if let Err(err) =
                    run_serve_cd_command(&client, &base, &token, &session_id, &trimmed)
                {
                    eprintln!("[serve-chat] {err}");
                }
            }
            "/use" | ":use" => {
                // Preserve the newline after a bare command: the generic
                // argument view trims it away and would treat the tail as a backend.
                let arg = &trimmed[cmd.len()..];
                match run_serve_use_command(&client, &base, &token, &session_id, arg) {
                    Ok(()) => {
                        if let Some((tail, title_seed)) = serve_use_turn_input(&trimmed) {
                            if let Err(err) = post_turn_stream(
                                &client,
                                &base,
                                &token,
                                &session_id,
                                tail,
                                Some(&title_seed),
                                &app_config.history_file,
                                &mut selection,
                            ) {
                                eprintln!("[serve-chat] {err}");
                            }
                        }
                    }
                    Err(err) => eprintln!("[serve-chat] {err}"),
                }
            }
            "/reset" | ":reset" => {
                if let Err(err) = run_serve_reset_command(&client, &base, &token, &session_id, arg)
                {
                    eprintln!("[serve-chat] {err}");
                }
            }
            _ => {
                let route = classify_serve_input(&trimmed);
                let prompt = routed_prompt_text(&route, &input, &trimmed);
                match route {
                    // A local slash command is never a prompt: the local REPL
                    // would swallow it, so forwarding it as a turn only burns a
                    // model round-trip at best and hangs the session at worst
                    // (`/history rewind` used to block a turn child on stdin).
                    // Anything already handled above never reaches this arm.
                    ServeInputRoute::Unsupported => {
                        let command = trimmed.split_whitespace().next().unwrap_or(&trimmed);
                        eprintln!(
                            "[serve-chat] {command} is a local command and is not supported over serve-chat; it was not sent. Supported here: /cd, /pwd, /git, /history, /export, /model, /effort, /agent, /use, /reset, /skills, /sessions, /resume, /fork, /close, /bg, /new, /audit, /distill-session, /mark, /unmark, /archive, /title (see /help)."
                        );
                    }
                    // Unknown command shapes must not fall through to the
                    // model, which would answer a command the user meant to
                    // execute.
                    ServeInputRoute::UnknownCommand => {
                        let command = trimmed.split_whitespace().next().unwrap_or(&trimmed);
                        eprintln!(
                            "[serve-chat] unknown command: {command}; it was not sent. Run /help to list commands supported over serve-chat."
                        );
                    }
                    // `/memo`, `/task`, `/discuss`, `/goal`, `/audit`,
                    // `/distill-session`, `/mark`, `/unmark`, `/archive`,
                    // `/title` execute in the server turn child; everything
                    // else is ordinary prompt text. State commands have their
                    // own HTTP branches above.
                    ServeInputRoute::ServerCommand | ServeInputRoute::Prompt => {
                        let title_seed = if route == ServeInputRoute::ServerCommand {
                            serve_command_title_seed(&trimmed)
                        } else {
                            None
                        };
                        if let Err(err) = post_turn_stream(
                            &client,
                            &base,
                            &token,
                            &session_id,
                            prompt,
                            title_seed.as_deref(),
                            &app_config.history_file,
                            &mut selection,
                        ) {
                            eprintln!("[serve-chat] {err}");
                        }
                    }
                }
            }
        }
    }
    // A later local REPL in the same process must never complete `/cd`
    // against the (now disconnected) remote server.
    CommandCompleter::set_remote_cd_candidates_provider(None);
    Ok(())
}

#[cfg(test)]
#[path = "chat_test_terminal.rs"]
mod test_terminal;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;

    #[test]
    fn normalize_base_preserves_explicit_https_and_defaults_to_http() {
        assert_eq!(
            normalize_base(" https://10.37.110.250:8080/ ").unwrap(),
            "https://10.37.110.250:8080"
        );
        assert_eq!(
            normalize_base("10.37.110.250:8080").unwrap(),
            "http://10.37.110.250:8080"
        );
    }

    #[test]
    fn normalize_base_rejects_non_base_urls() {
        assert!(normalize_base("https://host:8080/app").is_err());
        assert!(normalize_base("https://host:8080?mode=chat").is_err());
        assert!(normalize_base("ftp://host:8080").is_err());
    }

    #[test]
    fn explicit_serve_ca_certificate_wins_over_config() {
        assert_eq!(
            resolve_serve_ca_cert(" /cli/ca.pem ", "/cfg/ca.pem"),
            "/cli/ca.pem"
        );
        assert_eq!(resolve_serve_ca_cert("", " /cfg/ca.pem "), "/cfg/ca.pem");
    }

    /// Serve one canned HTTP response body, then return its URL. Keeps the
    /// SSE framing tests offline (loopback only, one connection).
    fn serve_body_once(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = vec![0u8; 4096];
            let _ = std::io::Read::read(&mut stream, &mut request);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        });
        format!("http://{addr}/sessions/x/turns/stream")
    }

    /// Same canned-response trick, but also captures the request and JSON body
    /// so routing, auth, and submitted intent can be asserted offline.
    fn serve_capture_once(
        body: &'static str,
    ) -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
        serve_capture_status_once("200 OK", body)
    }

    fn serve_capture_status_once(
        status: &'static str,
        body: &'static str,
    ) -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
        use std::io::BufRead;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let sink = std::sync::Arc::clone(&captured);
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut head = String::new();
            {
                let mut reader = std::io::BufReader::new(stream.try_clone().expect("clone stream"));
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    head.push_str(&line);
                }
                let content_length = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().expect("content length"))
                    })
                    .unwrap_or(0);
                let mut request_body = vec![0; content_length];
                std::io::Read::read_exact(&mut reader, &mut request_body).expect("request body");
                head.push_str("\r\n");
                head.push_str(&String::from_utf8(request_body).expect("request utf8"));
            }
            if let Ok(mut sink) = sink.lock() {
                *sink = head;
            }
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        });
        (format!("http://{addr}"), captured)
    }

    #[test]
    fn sse_message_lines_render_until_done() {
        let url = serve_body_once("data: hello\n\ndata: world\n\nevent: done\ndata: \n\n");
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(buf, b"hello\nworld\n");
    }

    #[test]
    fn confirm_answer_body_matches_server_shape() {
        // Wire shape of `turn::ConfirmAnswerReq`: the daemon matches on
        // (id, token) and relays `allow` as one yes/no line on stdin.
        assert_eq!(
            confirm_answer_body(7, 42, true),
            serde_json::json!({"id": 7, "token": 42, "allow": true})
        );
        assert_eq!(
            confirm_answer_body(7, 42, false),
            serde_json::json!({"id": 7, "token": 42, "allow": false})
        );
    }

    #[test]
    fn sse_confirm_request_without_channel_warns_and_continues() {
        // Tests have no interactive stdin (`None` channel): the event must
        // degrade to a warning, never hang, and never leak as raw JSON.
        let url = serve_body_once(
            "event: confirm_request\ndata: {\"id\": 3, \"prompt\": \"Run `git commit`?\", \"token\": 9}\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        let out = String::from_utf8(buf).expect("utf8");
        assert!(out.contains("cannot answer"), "unexpected: {out}");
        assert!(!out.contains("confirm_request"), "raw event leaked: {out}");
    }

    #[test]
    fn sse_malformed_confirm_request_is_ignored() {
        let url =
            serve_body_once("event: confirm_request\ndata: not-json\n\nevent: done\ndata: \n\n");
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        let out = String::from_utf8(buf).expect("utf8");
        assert!(out.contains("malformed"), "unexpected: {out}");
    }

    #[test]
    fn sse_confirm_done_is_swallowed() {
        // The child's own resolution frame carries nothing to render and must
        // not fall into the `_` arm's footer.
        let url =
            serve_body_once("event: confirm_done\ndata: {\"id\": 3}\n\nevent: done\ndata: \n\n");
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert!(
            buf.is_empty(),
            "unexpected: {}",
            String::from_utf8_lossy(&buf)
        );
    }

    #[test]
    fn sse_replace_event_never_leaks_its_json_envelope() {
        // `replace` repeats the final accepted body after the deltas. The
        // scrollback must not print the `{"text":...}` envelope as a footer,
        // on a TTY or when piped.
        let body = "event: delta\ndata: {\"delta\": \"answer\"}\n\nevent: replace\ndata: {\"text\": \"answer\"}\n\nevent: done\ndata: \n\n";
        for tty in [false, true] {
            let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
            let mut buf = Vec::new();
            render_turn_stream_to(resp, &mut buf, tty).expect("render");
            let out = String::from_utf8(buf).expect("utf8");
            assert!(out.contains("answer"), "missing body: {out}");
            assert!(!out.contains("{\"text\""), "raw replace payload: {out}");
            assert!(!out.contains("\\\"text\\\""), "raw escaped payload: {out}");
        }
    }

    #[test]
    fn sse_replace_event_renders_only_the_appended_suffix() {
        // The accepted body may append a final-gate warning after the streamed
        // prose. Only that suffix is missing from the scrollback.
        let body = "event: delta\ndata: {\"delta\": \"answer\"}\n\nevent: replace\ndata: {\"text\": \"answer\\n\\n[Runtime warning] revised\"}\n\nevent: done\ndata: \n\n";
        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(
            String::from_utf8(buf).expect("utf8"),
            "answer\n\n[Runtime warning] revised\n"
        );
    }

    #[test]
    fn sse_replace_event_without_deltas_renders_the_final_body() {
        let body = "event: replace\ndata: {\"text\": \"full answer\"}\n\nevent: done\ndata: \n\n";
        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(String::from_utf8(buf).expect("utf8"), "full answer\n");
    }

    #[test]
    fn sse_replace_event_with_rewritten_body_prints_the_accepted_answer() {
        // A final gate can rewrite the streamed draft. The client must render
        // the accepted body; a notice alone hides the only authoritative copy.
        let body = "event: delta\ndata: {\"delta\": \"draft\"}\n\nevent: replace\ndata: {\"text\": \"accepted answer\"}\n\nevent: done\ndata: \n\n";
        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        let out = String::from_utf8(buf).expect("utf8");
        assert!(out.contains("accepted answer"), "missing final body: {out}");
        assert!(
            out.contains("final answer was revised"),
            "missing revision notice: {out}"
        );
    }

    #[test]
    fn sse_empty_replace_reports_missing_final_answer() {
        // An empty accepted body clears rejected prose, but the user must not
        // be pointed at `/history last`: that can show an older turn's answer.
        let body = "event: delta\ndata: {\"delta\": \"draft\"}\n\nevent: replace\ndata: {\"text\": \"\"}\n\nevent: done\ndata: \n\n";
        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        let out = String::from_utf8(buf).expect("utf8");
        assert!(
            out.contains("without a final answer"),
            "missing status: {out}"
        );
        assert!(!out.contains("/history last"), "misleading pointer: {out}");
    }

    #[test]
    fn sse_turn_header_resets_the_streamed_body_buffer() {
        // A serve child can flush pending side-notes as a second turn on the
        // same SSE stream. Each turn's `replace` must be compared only with
        // that turn's deltas, not the previous turn's answer.
        let body = concat!(
            "event: delta\ndata: {\"delta\": \"first\"}\n\n",
            "event: replace\ndata: {\"text\": \"first\"}\n\n",
            "data: [test (effort: max) - topic]\n\n",
            "event: delta\ndata: {\"delta\": \"second\"}\n\n",
            "event: replace\ndata: {\"text\": \"second\"}\n\n",
            "event: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        let out = String::from_utf8(buf).expect("utf8");
        assert!(
            !out.contains("final answer was revised"),
            "spurious revision notice: {out}"
        );
        assert!(out.contains("first"), "missing first turn body: {out}");
        assert!(out.contains("second"), "missing second turn body: {out}");
    }

    #[test]
    fn sse_stream_retry_header_does_not_duplicate_the_streamed_answer() {
        // `print_info` runs inside the stream retry loop, so a retry prints
        // the turn header mid-turn. That header must not clear the body
        // already painted for the same turn.
        let body = concat!(
            "event: delta\ndata: {\"delta\": \"part1\"}\n\n",
            "data: [test (effort: max) - retry]\n\n",
            "event: delta\ndata: {\"delta\": \"part2\"}\n\n",
            "event: replace\ndata: {\"text\": \"part1part2\"}\n\n",
            "event: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        let out = String::from_utf8(buf).expect("utf8");
        assert!(
            !out.contains("final answer was revised"),
            "retry header caused a spurious rewrite: {out}"
        );
        assert_eq!(
            out.matches("part1").count(),
            1,
            "duplicated first half: {out}"
        );
        assert_eq!(
            out.matches("part2").count(),
            1,
            "duplicated second half: {out}"
        );
    }

    #[test]
    fn confirm_post_answer_hits_confirm_route_with_auth() {
        let (base, captured) = serve_capture_once("{}");
        let channel = ServeConfirmChannel {
            poster: reqwest::blocking::Client::builder()
                .build()
                .expect("client"),
            base,
            token: "sekret".to_string(),
            session_id: "sess-9".to_string(),
        };
        let mut buf = Vec::new();
        channel.post_answer(3, 9, true, &mut buf);
        let out = String::from_utf8(buf).expect("utf8");
        assert!(out.contains("confirmation sent (yes)"), "unexpected: {out}");
        let head = captured.lock().expect("head").clone();
        assert!(
            head.contains("POST /sessions/sess-9/confirm "),
            "unexpected head: {head}"
        );
        assert!(
            head.to_lowercase().contains("authorization: bearer sekret"),
            "missing bearer auth: {head}"
        );
    }

    #[test]
    fn collect_image_uploads_packages_multiple_local_files() {
        // Client assets layout: <tmp>/<stem>.sessions/<session>.assets/.
        let root = std::env::temp_dir().join(format!("a-serve-chat-img-{}", uuid::Uuid::new_v4()));
        let history = root.join("history.jsonl");
        let assets = SessionStore::new(&history).session_assets_dir("sess-1");
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("paste-a.png"), b"AAA").unwrap();
        std::fs::write(assets.join("paste-b.png"), b"BB").unwrap();
        let prompt = "what [[image:paste-a.png]] and [[image:paste-b.png]] \
            plus [[image:paste-a.png]] again, missing [[image:paste-missing.png]], \
            server-side [[image:/srv/x.png]]?";
        let uploads = collect_image_uploads(prompt, &history, "sess-1");
        assert_eq!(uploads.len(), 2, "unexpected: {uploads:?}");
        assert_eq!(uploads[0]["filename"], serde_json::json!("paste-a.png"));
        // The payload must round-trip to the staged file bytes.
        use base64::Engine as _;
        let raw = base64::engine::general_purpose::STANDARD
            .decode(uploads[0]["data_base64"].as_str().unwrap())
            .unwrap();
        assert_eq!(raw, b"AAA");
        assert_eq!(uploads[1]["filename"], serde_json::json!("paste-b.png"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sse_error_event_surfaces_server_detail() {
        let url = serve_body_once("event: error\ndata: {\"error\": \"boom\"}\n\n");
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let err = render_turn_stream(resp).expect_err("error event must fail");
        assert!(err.to_string().contains("boom"), "unexpected: {err}");
    }

    #[test]
    fn serve_effort_arg_normalizes_like_local() {
        assert_eq!(
            normalize_serve_effort_arg("low"),
            Ok(Some("low".to_string()))
        );
        assert_eq!(
            normalize_serve_effort_arg(" HIGH "),
            Ok(Some("high".to_string()))
        );
        assert_eq!(
            normalize_serve_effort_arg("off"),
            Ok(Some("off".to_string()))
        );
        assert_eq!(
            normalize_serve_effort_arg("disabled"),
            Ok(Some("off".to_string()))
        );
        assert_eq!(normalize_serve_effort_arg("auto"), Ok(None));
        assert_eq!(normalize_serve_effort_arg("clear"), Ok(None));
        assert!(normalize_serve_effort_arg("ultra").is_err());
        assert!(normalize_serve_effort_arg("").is_err());
    }

    #[test]
    fn serve_model_inline_split_matches_local_form() {
        let (selector, question) = split_serve_model_inline("deepseek");
        assert_eq!(selector, "deepseek");
        assert_eq!(question, None);
        let (selector, question) = split_serve_model_inline("deepseek   explain this code");
        assert_eq!(selector, "deepseek");
        assert_eq!(question.as_deref(), Some("explain this code"));
    }

    fn skill_fixture(name: &str) -> ServeSkillEntry {
        ServeSkillEntry {
            name: name.to_string(),
            description: format!("{name} skill"),
        }
    }

    #[test]
    fn serve_skills_inline_split_matches_local_form() {
        let known = vec![skill_fixture("bytedcli"), skill_fixture("code-review")];
        let (names, question) = split_serve_skills_inline(&known, "bytedcli");
        assert_eq!(names, vec!["bytedcli".to_string()]);
        assert_eq!(question, None);
        // Case-insensitive match canonicalizes to the manifest spelling and
        // the trailing question stays verbatim.
        let (names, question) =
            split_serve_skills_inline(&known, "ByteDcli CODE-review   explain this");
        assert_eq!(
            names,
            vec!["bytedcli".to_string(), "code-review".to_string()]
        );
        assert_eq!(question.as_deref(), Some("explain this"));
        // Dedupes keeping input order; a non-skill token stops the split and
        // starts the question.
        let (names, question) = split_serve_skills_inline(&known, "bytedcli bytedcli ??? nope");
        assert_eq!(names, vec!["bytedcli".to_string()]);
        assert_eq!(question.as_deref(), Some("??? nope"));
        // No leading skill token: pure question, no selection.
        let (names, question) = split_serve_skills_inline(&known, "hello there");
        assert!(names.is_empty());
        assert_eq!(question.as_deref(), Some("hello there"));
    }

    #[test]
    fn serve_skills_multiline_selection_line_is_strict() {
        let known = vec![skill_fixture("bytedcli"), skill_fixture("code-review")];
        // Both names on the selection line, question on the following lines.
        let (names, question, missing) =
            split_serve_skills_multiline(&known, "bytedcli code-review\n先看看这个文档\n第二行")
                .expect("multi-line form");
        assert_eq!(
            names,
            vec!["bytedcli".to_string(), "code-review".to_string()]
        );
        assert_eq!(question.as_deref(), Some("先看看这个文档\n第二行"));
        assert!(missing.is_empty());
        // An unknown name is reported, not demoted to question text.
        let (names, _, missing) =
            split_serve_skills_multiline(&known, "bytedcli nope\nquestion").expect("multi-line");
        assert_eq!(names, vec!["bytedcli".to_string()]);
        assert_eq!(missing, vec!["nope".to_string()]);
        // Single-line and explicit `use` heads keep their existing paths.
        assert!(split_serve_skills_multiline(&known, "bytedcli nope").is_none());
        assert!(split_serve_skills_multiline(&known, "use bytedcli\nquestion").is_none());
    }

    fn agent_fixture(
        name: &str,
        mode: agents::AgentMode,
        disabled: bool,
        hidden: bool,
    ) -> agents::AgentManifest {
        agents::AgentManifest {
            name: name.to_string(),
            description: format!("{name} agent"),
            mode,
            model: None,
            temperature: None,
            max_steps: None,
            prompt: String::new(),
            system_prompt: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            disable_mcp_tools: false,
            audit_checkpoints: false,
            model_tier: None,
            disabled,
            hidden,
            auto_select: false,
            pure: false,
            color: None,
            source_path: None,
        }
    }

    #[test]
    fn serve_agent_switch_rejects_disabled_like_local() {
        let manifests = vec![
            agent_fixture("build", agents::AgentMode::Primary, false, false),
            agent_fixture("sharp", agents::AgentMode::Primary, false, false),
            agent_fixture("off", agents::AgentMode::Primary, true, false),
            agent_fixture("helper", agents::AgentMode::Subagent, false, false),
        ];
        let mut selection = ServeTurnSelection {
            model: "m".to_string(),
            agent: "build".to_string(),
            reasoning_effort: None,
            skills: Vec::new(),
        };
        let mut editor = PromptEditor::new(
            "test",
            std::path::Path::new("/tmp/a-serve-agent-test-history"),
        );
        run_serve_agent_command("off", &mut selection, &manifests, &mut editor);
        assert_eq!(selection.agent, "build");
        run_serve_agent_command("helper", &mut selection, &manifests, &mut editor);
        assert_eq!(selection.agent, "build");
        run_serve_agent_command("sharp", &mut selection, &manifests, &mut editor);
        assert_eq!(selection.agent, "sharp");
    }

    #[test]
    fn sse_delta_events_stream_without_newlines() {
        let url = serve_body_once(
            "event: delta\ndata: {\"delta\": \"hel\"}\n\nevent: delta\ndata: {\"delta\": \"lo\"}\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        assert!(render_turn_stream(resp).is_ok());
    }

    #[test]
    fn transactional_serve_body_pending_is_temporary_and_renders_once() {
        let body = "event: answer_pending\ndata: \n\nevent: delta\ndata: {\"delta\": \"accepted answer\"}\n\nevent: done\ndata: \n\n";

        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut piped = Vec::new();
        render_turn_stream_to(resp, &mut piped, false).expect("render piped");
        assert_eq!(String::from_utf8(piped).expect("utf8"), "accepted answer\n");

        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut tty = Vec::new();
        render_turn_stream_to(resp, &mut tty, true).expect("render tty");
        let tty = String::from_utf8(tty).expect("utf8");
        assert!(tty.contains("finalizing response…"));
        assert!(tty.contains("\r\x1b[K"));
        assert!(tty.contains("accepted answer"));
    }

    #[test]
    fn sse_subagent_progress_is_an_in_place_live_row() {
        // The scheduler republishes the whole status line whenever the elapsed
        // counters change, so appending frames as footer rows stacks one
        // wrapped copy per tick in the scrollback. The client must redraw a
        // single live row instead, and piped captures must stay clean.
        let body = "event: subagent_progress\ndata: subagents · 1 active\n\nevent: subagent_progress\ndata: subagents · 1 active\n\nevent: delta\ndata: {\"delta\": \"answer\"}\n\nevent: done\ndata: \n\n";

        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut piped = Vec::new();
        render_turn_stream_to(resp, &mut piped, false).expect("render piped");
        assert_eq!(
            String::from_utf8(piped).expect("utf8"),
            "answer\n",
            "live progress must not enter piped scrollback"
        );

        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut tty = Vec::new();
        render_turn_stream_to(resp, &mut tty, true).expect("render tty");
        let tty = String::from_utf8(tty).expect("utf8");
        assert_eq!(
            tty.matches("\r\x1b[K\x1b[2m◌ subagents · 1 active").count(),
            2,
            "each frame redraws the live row: {tty:?}"
        );
        assert!(
            !tty.contains("subagents · 1 active\x1b[0m\n"),
            "status must never become a scrollback row: {tty:?}"
        );
        assert!(tty.contains("answer"), "missing body: {tty:?}");
    }

    #[test]
    fn sse_multiline_subagent_progress_keeps_the_whole_block() {
        // axum splits embedded newlines into one `data:` line per status row.
        // A one-task team has three rows; they must be joined into one event
        // and painted together, preserving the identity and detail prefixes.
        let body = "event: subagent_progress\ndata: subagents · 1 active\ndata: └─ [backend=traecli] worker · running 5s · task: audit\ndata:    ↳ progress: editing file\n\nevent: subagent_progress\ndata: subagents · 1 active\ndata: └─ [backend=traecli] worker · running 6s · task: audit\ndata:    ↳ progress: reviewing output\n\nevent: done\ndata: \n\n";

        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut piped = Vec::new();
        render_turn_stream_to(resp, &mut piped, false).expect("render piped");
        assert_eq!(
            String::from_utf8(piped).expect("utf8"),
            "",
            "live progress must not enter piped scrollback"
        );

        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut tty = Vec::new();
        render_turn_stream_to(resp, &mut tty, true).expect("render tty");
        let tty = String::from_utf8(tty).expect("utf8");
        assert_eq!(
            tty.matches("\x1b[2m◌ subagents · 1 active").count(),
            2,
            "each complete event paints one header row: {tty:?}"
        );
        assert!(
            tty.contains(
                "\r\n\x1b[2m└─ [backend=traecli] worker · running 5s · task: audit\x1b[0m\r\n\x1b[2m   ↳ progress: editing file\x1b[0m"
            ),
            "all status rows must be painted: {tty:?}"
        );
        assert!(
            tty.contains("\x1b[1A\r\x1b[K\x1b[1A\r\x1b[K\x1b[2m◌ subagents · 1 active"),
            "the previous three-row block must be erased before the next frame: {tty:?}"
        );
    }

    const SUBAGENT_TEAM_BLOCK: &str = concat!(
        "subagents · 3 active\n",
        "├─ [backend=traecli] alpha · running 5s · task: audit\n",
        "│  ↳ progress: reading files\n",
        "├─ [backend=traecli] beta · running 4s · task: implement\n",
        "│  ↳ progress: editing file\n",
        "└─ [backend=traecli] gamma · running 3s · task: validate\n",
        "   ↳ progress: checking tests",
    );

    #[test]
    fn sse_multiline_subagent_progress_draws_all_team_nodes_and_erases_old_rows() {
        let body: &'static str = Box::leak(
            format!(
                "event: subagent_progress\ndata: {}\n\nevent: subagent_progress\ndata: subagents · 1 active\ndata: └─ alpha · running 6s · task: audit\ndata:    ↳ progress: reviewing output\n\nevent: done\ndata: \n\n",
                SUBAGENT_TEAM_BLOCK.replace('\n', "\ndata: ")
            )
            .into_boxed_str(),
        );
        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut piped = Vec::new();
        render_turn_stream_to(resp, &mut piped, false).expect("render piped");
        assert!(piped.is_empty(), "live progress must stay out of captures");

        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        let mut tty = Vec::new();
        render_turn_stream_to(resp, &mut tty, true).expect("render tty");
        let tty = String::from_utf8(tty).expect("utf8");
        let first_frame = tty.split("\x1b[2m◌ subagents · 1 active").next().unwrap();
        assert_eq!(first_frame.matches("\r\n").count(), 6, "{tty:?}");
        let mut previous_end = 0;
        for row in SUBAGENT_TEAM_BLOCK.lines().skip(1) {
            let painted = dim(row, true);
            let start = first_frame.find(&painted).expect("missing team row");
            assert!(start >= previous_end, "team rows must keep their order");
            previous_end = start + painted.len();
        }
        let erase_seven = format!("\r\x1b[K{}", "\x1b[1A\r\x1b[K".repeat(6));
        assert!(
            tty.contains(&format!("{erase_seven}\x1b[2m◌ subagents · 1 active")),
            "all seven old rows must be erased before the smaller frame: {tty:?}"
        );
        let erase_three = format!("\r\x1b[K{}", "\x1b[1A\r\x1b[K".repeat(2));
        assert!(
            tty.ends_with(&erase_three),
            "done must clear the final block: {tty:?}"
        );
    }

    #[test]
    fn subagent_status_height_limit_marks_only_the_omitted_bottom_rows() {
        let full = subagent_status_rows_at_height(SUBAGENT_TEAM_BLOCK, 8);
        assert_eq!(full.len(), 7);
        for (height, kept) in [(6usize, 4usize), (3, 1), (2, 0), (1, 0)] {
            let rows = subagent_status_rows_at_height(SUBAGENT_TEAM_BLOCK, height);
            assert_eq!(&rows[..kept], &full[..kept], "height={height}");
            assert_eq!(rows.len(), kept + 1, "height={height}");
            assert_eq!(rows[kept], format!("… {} more rows below", 7 - kept));
            assert!(rows.len() <= height.saturating_sub(1).max(1));
        }
        assert!(subagent_status_rows_at_height("", 1).is_empty());
    }

    #[test]
    fn subagent_status_rows_bound_unicode_to_one_terminal_row() {
        let cols = crate::ai::stream::raw_terminal_cols();
        let long = "界⚠️😀é".repeat(cols + 1);
        let block = format!(
            "subagents · 3 active\n├─ {long}\n│  ↳ {long}\n├─ {long}\n│  ↳ {long}\n└─ {long}\n   ↳ {long}"
        );
        let rows = subagent_status_rows_at_height(&block, 8);
        assert_eq!(rows.len(), 7);
        assert!(rows.iter().all(|row| {
            row.chars().map(thinking_cell_width).sum::<usize>() <= cols.saturating_sub(1).max(1)
        }));
        assert!(rows[1..].iter().all(|row| row.ends_with('…')));
    }

    #[test]
    fn subagent_status_erases_reflowed_rows_within_the_resized_viewport() {
        // At width 20 these seven rows fit. At width 10 the eleven ASCII
        // cells, six CJK glyphs, and six emoji each occupy two rows: ten total.
        let previous = [
            "12345678901",
            "界界界界界界",
            "⚠️⚠️⚠️⚠️⚠️⚠️",
            "b",
            "c",
            "d",
            "e",
        ];
        for (cols, height, expected) in [(20, 24, 7), (10, 24, 10), (10, 4, 4)] {
            let mut drawn = previous.iter().map(|row| row.to_string()).collect();
            let mut out = Vec::new();
            erase_subagent_status_rows_at_size(&mut drawn, cols, height, &mut out);
            let out = String::from_utf8(out).expect("utf8");
            assert_eq!(out.matches("\x1b[K").count(), expected, "{out:?}");
            assert_eq!(out.matches("\x1b[1A").count(), expected - 1, "{out:?}");
            assert!(drawn.is_empty());
        }
    }

    #[test]
    fn subagent_status_gates_and_empty_updates_preserve_line_ownership() {
        for (tty, thinking, pending, open) in [
            (false, false, false, false),
            (true, true, false, false),
            (true, false, true, false),
            (true, false, false, true),
        ] {
            let mut drawn = Vec::new();
            let mut line_open = open;
            let mut out = Vec::new();
            show_subagent_status(
                SUBAGENT_TEAM_BLOCK,
                &mut drawn,
                pending,
                thinking,
                &mut line_open,
                tty,
                &mut out,
            );
            assert!(out.is_empty());
            assert!(drawn.is_empty());
            assert_eq!(line_open, open);
        }

        let mut drawn = Vec::new();
        let mut line_open = false;
        let mut out = Vec::new();
        show_subagent_status(
            SUBAGENT_TEAM_BLOCK,
            &mut drawn,
            false,
            false,
            &mut line_open,
            true,
            &mut out,
        );
        assert_eq!(drawn.len(), 7);
        assert!(line_open);
        let start = out.len();
        show_subagent_status(
            "blocked",
            &mut drawn,
            true,
            false,
            &mut line_open,
            true,
            &mut out,
        );
        assert_eq!(out.len(), start);
        assert_eq!(drawn.len(), 7);
        assert!(line_open);

        // Our own open line permits replacement, and shrinking must erase
        // every old row before storing the new frame's row count.
        show_subagent_status(
            "subagents · 1 active",
            &mut drawn,
            false,
            false,
            &mut line_open,
            true,
            &mut out,
        );
        let replacement = std::str::from_utf8(&out[start..]).expect("utf8");
        assert_eq!(replacement.matches("\x1b[K").count(), 7);
        assert_eq!(replacement.matches("\x1b[1A").count(), 6);
        assert_eq!(drawn.len(), 1);
        assert!(line_open);

        let start = out.len();
        show_subagent_status("", &mut drawn, false, false, &mut line_open, true, &mut out);
        assert_eq!(&out[start..], b"\r\x1b[K");
        assert!(drawn.is_empty());
        assert!(!line_open);
    }

    #[test]
    fn sse_thinking_frames_fold_and_end_turn() {
        // Thinking lifecycle interleaved with the answer: the renderer must
        // keep piped behavior summary-only and finish on `done`.
        let url = serve_body_once(
            "event: thinking_start\ndata: \n\nevent: thinking\ndata: {\"text\": \"reasoning here\"}\n\nevent: thinking_done\ndata: \n\nevent: delta\ndata: {\"delta\": \"answer\"}\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut out = Vec::new();
        render_turn_stream_to(resp, &mut out, false).expect("render");
        assert_eq!(
            String::from_utf8(out).expect("utf8"),
            "✓ thinking (1 line)\nanswer\n"
        );
    }

    #[test]
    fn thinking_live_region_wraps_sanitizes_and_erases_after_resize() {
        let thought = "hidden-first-line\nmiddle-row-that-also-wraps\nsafe\x1b[31m\r\tend";
        let mut region = ThinkingLiveRegion::new(2, Some(20));
        region.append(thought);

        let frame = region.projected_rows(20);
        assert_eq!(frame[0], "  ○ thinking");
        assert_eq!(frame[1], "    … more");
        assert_eq!(frame.len(), 4, "header + marker + two body rows: {frame:?}");
        let visible = frame.join("\n");
        assert!(!visible.contains("hidden-first-line"), "{visible:?}");
        assert!(!visible.contains("middle-row"), "{visible:?}");
        assert!(
            visible.contains(r"\u{1b}"),
            "ESC must be visible: {visible:?}"
        );
        assert!(visible.contains(r"\r"), "CR must be visible: {visible:?}");
        assert!(
            !visible.chars().any(|ch| matches!(ch, '\x1b' | '\r' | '\t')),
            "projected model text contains a terminal control: {visible:?}"
        );

        let mut out = Vec::new();
        region.redraw(&mut out);
        // Simulate a narrowing between the last frame and close. Erasure must
        // count how the already-written plain rows reflow at the new width.
        let resized_rows = region.rendered_physical_rows(12);
        region.terminal_width_override = Some(12);
        let mut active = true;
        let mut line_open = true;
        close_thinking_status(
            &mut active,
            &mut line_open,
            3,
            &mut region,
            true,
            &mut out,
            &mut ObserverRowState::default(),
        );
        writeln!(&mut out, "answer").unwrap();

        let out = String::from_utf8(out).expect("utf8");
        assert!(!out.contains("hidden-first-line"), "{out:?}");
        assert!(!out.contains("\x1b[31m"), "model ESC leaked: {out:?}");
        assert_eq!(
            out.matches("\x1b[1A\r\x1b[K").count(),
            resized_rows.saturating_sub(1),
            "erase must move up only within its reflowed region: {out:?}"
        );
        assert_eq!(out.matches("✓ thinking (3 lines)").count(), 1, "{out:?}");
        assert!(
            out.ends_with("✓ thinking (3 lines)\x1b[0m\nanswer\n"),
            "summary must be the only retained row before the answer: {out:?}"
        );
        assert!(region.drawn_rows.is_empty());
        assert!(region.tail.is_empty());
    }

    #[test]
    fn thinking_live_region_vs16_rows_match_terminal_oracle_across_resize() {
        // Independent oracle: each `⚠️` is two cells (base + VS16) and `界`
        // is two. At width 16 the ten-cell body fits its ten-cell content
        // budget. At width 12 it wraps into two six-cell-budget body rows.
        const NARROW: usize = 12;
        const WIDE: usize = 16;
        let mut region = ThinkingLiveRegion::new(usize::MAX, Some(WIDE));
        region.append("⚠️⚠️⚠️⚠️界");
        let mut out = Vec::new();

        region.redraw(&mut out);
        let wide_frame = std::str::from_utf8(&out).unwrap();
        assert_eq!(
            wide_frame.matches("\r\n").count() + 1,
            2,
            "wide redraw must emit one header and one body row: {wide_frame:?}"
        );

        let mut offset = out.len();
        region.terminal_width_override = Some(NARROW);
        region.redraw(&mut out);
        {
            let narrow_redraw = std::str::from_utf8(&out[offset..]).unwrap();
            let frame_start = narrow_redraw.find("\x1b[2m").unwrap();
            let erase = &narrow_redraw[..frame_start];
            let frame = &narrow_redraw[frame_start..];
            // The old body was fourteen cells including indentation, so it
            // reflows to two physical rows at width 12; the header is one.
            assert_eq!(erase.matches("\x1b[K").count(), 3, "{erase:?}");
            assert_eq!(erase.matches("\x1b[1A").count(), 2, "{erase:?}");
            assert_eq!(
                frame.matches("\r\n").count() + 1,
                3,
                "narrow redraw must emit one header and two body rows: {frame:?}"
            );
        }

        offset = out.len();
        region.terminal_width_override = Some(WIDE);
        region.redraw(&mut out);
        {
            let wide_redraw = std::str::from_utf8(&out[offset..]).unwrap();
            let frame_start = wide_redraw.find("\x1b[2m").unwrap();
            let erase = &wide_redraw[..frame_start];
            let frame = &wide_redraw[frame_start..];
            // The narrow frame consists of three hard rows; widening must
            // erase all three before replacing them with two wide rows.
            assert_eq!(erase.matches("\x1b[K").count(), 3, "{erase:?}");
            assert_eq!(erase.matches("\x1b[1A").count(), 2, "{erase:?}");
            assert_eq!(frame.matches("\r\n").count() + 1, 2, "{frame:?}");
        }

        offset = out.len();
        let mut active = true;
        let mut line_open = true;
        close_thinking_status(
            &mut active,
            &mut line_open,
            1,
            &mut region,
            true,
            &mut out,
            &mut ObserverRowState::default(),
        );
        let close = std::str::from_utf8(&out[offset..]).unwrap();
        let summary_start = close.find("\x1b[2m").unwrap();
        let erase = &close[..summary_start];
        assert_eq!(erase.matches("\x1b[K").count(), 2, "{erase:?}");
        assert_eq!(erase.matches("\x1b[1A").count(), 1, "{erase:?}");
        assert_eq!(close.matches("✓ thinking (1 line)").count(), 1, "{close:?}");
    }

    #[test]
    fn sse_metrics_footers_follow_the_answer() {
        let url = serve_body_once(
            "event: delta\ndata: {\"delta\": \"hi there\"}\n\ndata: ↳ cache · 1k\n\ndata: ↳ speed · output 10 tok/s\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(
            String::from_utf8(buf).expect("utf8"),
            "hi there\n↳ cache · 1k\n↳ speed · output 10 tok/s\n"
        );
    }

    #[test]
    fn sse_metrics_footer_between_chunks_waits_for_body_boundary() {
        // The stdout pump can overtake live-body frames. Keep the sentence
        // contiguous while retaining the metrics after its final chunk.
        let url = serve_body_once(
            "event: delta\ndata: {\"delta\": \"读\"}\n\ndata: ↳ speed · reasoning 601 tok @ 108 tok/s\n\nevent: delta\ndata: {\"delta\": \"不到。\"}\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(
            String::from_utf8(buf).expect("utf8"),
            "读不到。\n↳ speed · reasoning 601 tok @ 108 tok/s\n"
        );
    }

    #[test]
    fn sse_metrics_multi_round_preserves_each_request_summary() {
        let url = serve_body_once(
            "event: delta\ndata: {\"delta\": \"done\"}\n\ndata: ↳ cache · 1k\n\ndata: ↳ speed · round one\n\ndata: ↳ cache · 2k\n\ndata: ↳ speed · round two\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(
            String::from_utf8(buf).expect("utf8"),
            "done\n↳ cache · 1k\n↳ speed · round one\n↳ cache · 2k\n↳ speed · round two\n"
        );
    }

    #[test]
    fn sse_thinking_rate_shows_server_sample_and_true_footer() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let body = concat!(
            "event: thinking_start\ndata: \n\n",
            "event: thinking\ndata: {\"text\":\"abcd\",\"rate\":\"~8 tok @ 4.00 tok/s\"}\n\n",
            "event: thinking_done\ndata: \n\n",
            "event: delta\ndata: {\"delta\":\"answer\"}\n\n",
            "data: ↳ speed · reasoning 800 tok @ 400 tok/s\n\n",
            "event: done\ndata: \n\n",
        );
        for tty in [false, true] {
            let url = serve_body_once(body);
            let resp = reqwest::blocking::get(url).unwrap();
            let mut out = Vec::new();
            render_turn_stream_to(resp, &mut out, tty).unwrap();
            let text = String::from_utf8(out).unwrap();
            assert_eq!(text.contains("~8 tok @ 4.00 tok/s"), tty, "{text:?}");
            assert_eq!(
                text.matches("↳ speed · reasoning 800 tok @ 400 tok/s")
                    .count(),
                1,
                "{text:?}"
            );
            assert!(
                text.find("answer").unwrap() < text.find("↳ speed").unwrap(),
                "{text:?}"
            );
        }
    }

    #[test]
    fn sse_metrics_flush_before_next_thinking_round_and_strip_pipe_ansi() {
        let url = serve_body_once(concat!(
            "event: delta\ndata: {\"delta\":\"first\"}\n\n",
            "data: \u{1b}[2m↳ speed · output 20 tok @ 10 tok/s\u{1b}[0m\n\n",
            "event: thinking_start\ndata: \n\n",
            "event: thinking\ndata: {\"text\":\"next thought\"}\n\n",
            "event: thinking_done\ndata: \n\n",
            "event: delta\ndata: {\"delta\":\"second\"}\n\n",
            "event: done\ndata: \n\n",
        ));
        let resp = reqwest::blocking::get(url).unwrap();
        let mut out = Vec::new();
        render_turn_stream_to(resp, &mut out, false).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "first\n↳ speed · output 20 tok @ 10 tok/s\n✓ thinking (1 line)\nsecond\n"
        );
    }

    #[test]
    fn sse_turn_header_row_is_marked_remote() {
        // Serve-chat scrollback must show which turns ran remotely: the child
        // header `[model (effort: …) · topic]` is forwarded as a plain stdout
        // row, so the client appends a piped-safe ` (remote)` marker.
        let header = "[deepseek-v4-flash-volcano (effort: max) · [fork] 启动服务]";
        let url = serve_body_once(
            "data: [deepseek-v4-flash-volcano (effort: max) · [fork] 启动服务]\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(
            String::from_utf8(buf).expect("utf8"),
            format!("{header} (remote)\n")
        );
    }

    #[test]
    fn turn_header_detection_ignores_ansi_and_remote() {
        // Unit net for the matcher: theme colors must not hide the header
        // shape, an already-marked row stays idempotent, and ordinary rows
        // (tool output, bracketed text without the effort marker) pass
        // through untouched.
        let colored = "\x1b[2m\x1b[32m[model (effort: max) · topic]\x1b[0m";
        assert!(is_turn_header_row(colored));
        assert!(!is_turn_header_row(
            "[model (effort: max) · topic] (remote)"
        ));
        assert!(!is_turn_header_row("  ✕ execute_command"));
        assert!(!is_turn_header_row("[note] something else"));
        assert_eq!(
            annotate_remote_turn_header(colored, false),
            format!("{colored} (remote)")
        );
        assert_eq!(
            annotate_remote_turn_header(colored, true),
            format!("{colored}\x1b[1;38;2;196;181;253m (remote)\x1b[0m")
        );
    }

    #[test]
    fn observer_rows_dedupe_only_adjacent_identical_headers() {
        let mut state = ObserverRowState::default();
        let header = "[model (effort: max) · topic]";
        assert_eq!(
            decide_observer_row(header, &mut state, true, 80, false).action,
            ObserverRowAction::Print
        );
        for duplicate in [
            header,
            "\x1b[32m[model (effort: max) · topic]\x1b[0m (remote)",
            "[model (effort: max) · topic] 12:34:56",
        ] {
            assert_eq!(
                decide_observer_row(duplicate, &mut state, true, 80, false).action,
                ObserverRowAction::SkipDuplicate,
                "{duplicate:?}"
            );
        }
        assert_eq!(
            decide_observer_row("│ tool output", &mut state, true, 80, false).action,
            ObserverRowAction::Print
        );
        assert_eq!(
            decide_observer_row(header, &mut state, true, 80, false).action,
            ObserverRowAction::Print
        );
    }

    #[test]
    fn observer_rows_keep_changed_header_fields_and_new_turns() {
        let mut state = ObserverRowState::default();
        for header in [
            "[model (effort: max) · topic]",
            "[fallback (effort: max) · topic]",
            "[fallback (effort: off) · topic]",
            "[fallback (effort: off) · next topic]",
        ] {
            assert_eq!(
                decide_observer_row(header, &mut state, false, 80, false).action,
                ObserverRowAction::Print,
                "{header}"
            );
        }
        let next = decide_observer_row(
            "[fallback (effort: off) · next topic]",
            &mut state,
            false,
            80,
            true,
        );
        assert!(next.new_turn);
        assert_eq!(next.action, ObserverRowAction::Print);
    }

    #[test]
    fn observer_rows_replace_matching_tool_completion() {
        let mut state = ObserverRowState::default();
        assert_eq!(
            decide_observer_row(
                "  \x1b[34m●\x1b[0m cmd · target",
                &mut state,
                true,
                80,
                false
            )
            .action,
            ObserverRowAction::OpenToolRow
        );
        assert_eq!(state.open_tool_row.as_deref(), Some("cmd"));
        let done = decide_observer_row(
            "\r\x1b[2K  \x1b[32m✓\x1b[0m cmd · 2s",
            &mut state,
            true,
            80,
            false,
        );
        assert_eq!(done.action, ObserverRowAction::ReplaceOpenToolRow);
        assert!(!done.close_tool_row);
        assert!(state.open_tool_row.is_none());
    }

    #[test]
    fn observer_rows_close_tool_before_unrelated_row() {
        let mut state = ObserverRowState::default();
        assert_eq!(
            decide_observer_row("● cmd", &mut state, true, 80, false).action,
            ObserverRowAction::OpenToolRow
        );
        let other = decide_observer_row("  │ command output", &mut state, true, 80, false);
        assert!(other.close_tool_row, "the started row needs a newline");
        assert_eq!(other.action, ObserverRowAction::Print);
        assert_eq!(
            decide_observer_row("✓ cmd", &mut state, true, 80, false).action,
            ObserverRowAction::Print
        );
    }

    #[test]
    fn observer_rows_keep_piped_and_wrapped_tool_rows() {
        let mut state = ObserverRowState::default();
        for row in ["● cmd", "✓ cmd"] {
            assert_eq!(
                decide_observer_row(row, &mut state, false, 80, false).action,
                ObserverRowAction::Print
            );
        }
        for (row, width) in [
            ("● cmd", 5),
            ("● 文件", 6),
            ("● cmd\targ", 80),
            ("● cmd", 0),
        ] {
            assert_eq!(
                decide_observer_row(row, &mut state, true, width, false).action,
                ObserverRowAction::Print,
                "unsafe single-row erase: {row:?}, width={width}"
            );
        }
    }

    #[test]
    fn observer_rows_suppress_consecutive_thinking_summaries() {
        let mut state = ObserverRowState::default();
        let summary = "✓ thinking (3 lines)";
        assert_eq!(
            decide_observer_row(summary, &mut state, false, 80, false).action,
            ObserverRowAction::Print
        );
        assert_eq!(
            decide_observer_row(
                "\x1b[2m✓ thinking (3 lines)\x1b[0m",
                &mut state,
                false,
                80,
                false
            )
            .action,
            ObserverRowAction::SkipDuplicate
        );
        assert_eq!(
            decide_observer_row("  │ unrelated", &mut state, false, 80, false).action,
            ObserverRowAction::Print
        );
        assert_eq!(
            decide_observer_row(summary, &mut state, false, 80, false).action,
            ObserverRowAction::Print
        );
    }

    #[test]
    fn observer_rows_replace_or_dedupe_context_budget_notes() {
        let first = "  \x1b[2m│ context-budget: projection refreshed: 1 -> 2 chars\x1b[0m";
        let second = "  │ context-budget: projection refreshed: 2 -> 3 chars";
        let mut state = ObserverRowState::default();
        assert_eq!(
            decide_observer_row(first, &mut state, true, 80, false).action,
            ObserverRowAction::ContextBudget {
                replace: false,
                draw: true
            }
        );
        assert_eq!(
            decide_observer_row(second, &mut state, true, 80, false).action,
            ObserverRowAction::ContextBudget {
                replace: true,
                draw: true
            }
        );
        assert_eq!(
            decide_observer_row(first, &mut state, true, 20, false).action,
            ObserverRowAction::ContextBudget {
                replace: true,
                draw: false
            }
        );
        let mut piped = ObserverRowState::default();
        for note in [first, second] {
            assert_eq!(
                decide_observer_row(note, &mut piped, false, 80, false).action,
                ObserverRowAction::Print
            );
        }
        assert_eq!(
            decide_observer_row(second, &mut piped, false, 80, false).action,
            ObserverRowAction::SkipDuplicate
        );
        assert_eq!(
            decide_observer_row("│ ordinary continuation", &mut piped, false, 80, false).action,
            ObserverRowAction::Print
        );
    }

    #[test]
    fn sse_repeated_headers_dedupe_only_without_intervening_activity() {
        let body = concat!(
            "data: [model (effort: max) · topic]\n\n",
            "data: [model (effort: max) · topic]\n\n",
            "event: delta\ndata: {\"delta\": \"first\"}\n\n",
            "data: [model (effort: max) · topic]\n\n",
            "data: [fallback (effort: max) · topic]\n\n",
            "event: replace\ndata: {\"text\": \"first\"}\n\n",
            "data: [fallback (effort: max) · topic]\n\n",
            "event: delta\ndata: {\"delta\": \"second\"}\n\n",
            "event: replace\ndata: {\"text\": \"second\"}\n\n",
            "event: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(serve_body_once(body)).unwrap();
        let mut out = Vec::new();
        render_turn_stream_to(resp, &mut out, false).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            concat!(
                "[model (effort: max) · topic] (remote)\n",
                "first\n",
                "[model (effort: max) · topic] (remote)\n",
                "[fallback (effort: max) · topic] (remote)\n",
                "[fallback (effort: max) · topic] (remote)\n",
                "second\n",
            )
        );
    }

    #[test]
    fn sse_budget_notes_keep_only_latest_tty_note_and_continuations() {
        // Overwide notes are deferred, so this fixture is independent of the
        // test runner's terminal geometry and still checks footer ordering.
        let first = format!("  │ context-budget: old {}", "a".repeat(70_000));
        let latest = format!(
            "  \x1b[2m│ context-budget: latest {}\x1b[0m",
            "b".repeat(70_000)
        );
        let body = format!(
            "data: {first}\n\ndata:   │ ordinary continuation\n\ndata: {latest}\n\nevent: delta\ndata: {{\"delta\": \"answer\"}}\n\ndata: ↳ speed · dropped\n\nevent: done\ndata: \n\n"
        );
        let resp =
            reqwest::blocking::get(serve_body_once(Box::leak(body.into_boxed_str()))).unwrap();
        let mut out = Vec::new();
        render_turn_stream_to(resp, &mut out, true).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(!out.contains("context-budget: old"), "stale note retained");
        assert_eq!(out.matches("context-budget: latest").count(), 1);
        assert!(out.contains("  │ ordinary continuation\n"));
        assert!(
            out.ends_with(&format!("{latest}\n")),
            "final note lost ANSI or newline"
        );
        assert!(!out.contains("↳ speed"));
    }

    #[test]
    fn sse_thinking_summary_dedupe_covers_live_and_observer_rows() {
        let body = concat!(
            "event: thinking_start\ndata: \n\n",
            "event: thinking\ndata: {\"text\": \"first\"}\n\n",
            "event: thinking_done\ndata: \n\n",
            "data: ✓ thinking (1 line)\n\n",
            "event: thinking_start\ndata: \n\n",
            "event: thinking\ndata: {\"text\": \"second\"}\n\n",
            "event: thinking_done\ndata: \n\n",
            "event: delta\ndata: {\"delta\": \"answer\"}\n\n",
            "event: done\ndata: \n\n",
        );
        for tty in [false, true] {
            let resp = reqwest::blocking::get(serve_body_once(body)).unwrap();
            let mut out = Vec::new();
            render_turn_stream_to(resp, &mut out, tty).unwrap();
            let out = String::from_utf8(out).unwrap();
            let summaries = out
                .match_indices("✓ thinking (1 line)")
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            assert_eq!(
                summaries.len(),
                2,
                "each real block prints once while the first mirrored observer row is deduped: {out:?}"
            );
            let answer = out.find("answer").unwrap();
            assert!(
                summaries[0] < summaries[1] && summaries[1] < answer,
                "{out:?}"
            );
            if !tty {
                assert_eq!(out, "✓ thinking (1 line)\n✓ thinking (1 line)\nanswer\n");
            }
        }
    }

    #[test]
    fn sse_answer_and_confirmation_break_thinking_summary_dedupe() {
        for intervening in [
            "event: replace\ndata: {\"text\": \"answer\"}\n\n",
            "event: confirm_request\ndata: {}\n\n",
        ] {
            let body = format!(
                "event: thinking_start\ndata: \n\nevent: thinking\ndata: {{\"text\": \"reasoning\"}}\n\n{intervening}data: ✓ thinking (1 line)\n\nevent: done\ndata: \n\n"
            );
            let resp =
                reqwest::blocking::get(serve_body_once(Box::leak(body.into_boxed_str()))).unwrap();
            let mut out = Vec::new();
            render_turn_stream_to(resp, &mut out, false).unwrap();
            let out = String::from_utf8(out).unwrap();
            assert_eq!(out.matches("✓ thinking (1 line)").count(), 2, "{out:?}");
        }
    }

    #[test]
    fn sse_observer_rows_ellipsis_use_local_width_and_preserve_pipes() {
        const COLS: usize = 40;
        let header = "\x1b[32m[model (effort: max) · topics🙂]\x1b[0m";
        let running = "  \x1b[34m●\x1b[0m execute_command";
        let command = format!(
            "  │ $ \x1b[32mrg -n \"{}\" src/bin/ai/history/sqlite/rollback.rs\x1b[0m",
            "backup|reflink|界🙂|clone_file|".repeat(6)
        );
        let health = format!(
            "  \x1b[2m│ agent-health: required delegation is overdue: {}\x1b[0m",
            "delegate discovery before continuing exploration ".repeat(3)
        );
        let done = "  \x1b[32m✓\x1b[0m execute_command";
        let reading = "  \x1b[34m●\x1b[0m read_file · target";
        let read_done = format!(
            "  \x1b[32m✓\x1b[0m read_file · src/bin/ai/history/{} 1415..1604",
            "文件🙂/".repeat(6)
        );
        let controls = "  │ controls\t中🙂\x08\r\x1b[2Ksafe\x1b]0;hidden-title\x07";
        let child = format!("    │ {}\n    │ child 中🙂", "child 界🙂 ".repeat(10));
        let body: &'static str = Box::leak(
            format!(
                "data: {header}\n\ndata: {header}\n\ndata: {running}\n\ndata: {command}\n\ndata: {health}\n\ndata: {done}\n\ndata: {reading}\n\ndata: {read_done}\n\ndata: {controls}\n\ndata: {}\n\nevent: answer_pending\ndata: \n\nevent: delta\ndata: {{\"delta\":\"answer-one\\n\"}}\n\ndata: {header}\n\nevent: delta\ndata: {{\"delta\":\"answer-two\\n\"}}\n\nevent: replace\ndata: {{\"text\":\"answer-one\\nanswer-two\\n\"}}\n\nevent: done\ndata: \n\n",
                child.replace('\n', "\ndata: ")
            )
            .into_boxed_str(),
        );
        let header_plain = strip_ansi_escapes(header);
        assert!(header_plain.chars().map(thinking_cell_width).sum::<usize>() <= COLS);
        assert!(
            format!("{header_plain} (remote)")
                .chars()
                .map(thinking_cell_width)
                .sum::<usize>()
                > COLS,
            "only appending (remote) should make this header overflow"
        );
        let replay = |tty, width| {
            let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
            test_terminal::render(resp, tty, &[width])
        };
        // Resolve preview rewrites and live rows before counting visible
        // answers; raw ANSI contains both the preview and its final paint.
        let (narrow, rows) = replay(true, COLS);
        for row in &rows {
            assert!(
                row.chars().map(thinking_cell_width).sum::<usize>() <= COLS,
                "observer row would wrap: {row:?}"
            );
        }
        let headers = rows
            .iter()
            .filter(|row| row.starts_with('['))
            .collect::<Vec<_>>();
        assert_eq!(headers.len(), 2, "adjacent retry deduplication: {rows:?}");
        for row in headers {
            assert!(
                row.contains(" (rem"),
                "marker appended before clipping: {row:?}"
            );
            assert!(row.ends_with('…'), "header must end in ellipsis: {row:?}");
        }
        for prefix in [
            "  │ $ rg -n",
            "  │ agent-health:",
            "  ✓ read_file",
            "    │ child 界",
        ] {
            let matches = rows
                .iter()
                .filter(|row| row.starts_with(prefix))
                .collect::<Vec<_>>();
            assert_eq!(
                matches.len(),
                1,
                "missing/duplicated history: {prefix:?}, {rows:?}"
            );
            assert!(
                matches[0].ends_with('…'),
                "long row not clipped: {matches:?}"
            );
        }
        assert!(
            rows.iter().any(|row| row == "    │ child 中🙂"),
            "child indent: {rows:?}"
        );
        assert!(
            rows.iter().any(|row| row == "  ● execute_command"),
            "running history: {rows:?}"
        );
        assert!(
            rows.iter().any(|row| row == "  ✓ execute_command"),
            "completion history: {rows:?}"
        );
        assert!(
            !rows
                .iter()
                .any(|row| row.contains("● read_file") || row.contains("finalizing response")),
            "live row residue: {rows:?}"
        );
        assert!(!narrow.contains("hidden-title"), "OSC control leaked");
        let visible = rows.join("\n");
        for answer in ["answer-one", "answer-two"] {
            assert_eq!(
                visible.matches(answer).count(),
                1,
                "answer duplicated on terminal: {visible:?}"
            );
        }
        assert!(
            !narrow.contains("final answer was revised"),
            "retry cleared the body buffer"
        );
        assert!(narrow.ends_with('\n'), "cursor left on a live row");

        let (wide, wide_rows) = replay(true, 400);
        for row in [
            running,
            command.as_str(),
            health.as_str(),
            done,
            read_done.as_str(),
        ] {
            assert!(wide.contains(row), "wide TTY changed source bytes: {row:?}");
        }
        let wide_plain = wide_rows.join("\n");
        assert_eq!(
            wide_plain
                .matches(&format!("{header_plain} (remote)"))
                .count(),
            2
        );
        assert!(
            !wide_plain
                .lines()
                .any(|line| line.rsplit('\r').next().unwrap().ends_with('…'))
        );

        // Negative control: the exact same old-server replay at unknown
        // geometry retains an overwide row. Removing the clamp therefore
        // fails the narrow assertions above without changing the fixture.
        let (unknown, _) = replay(true, 0);
        assert!(
            unknown.contains(&format!("{command}\n")),
            "unknown width must preserve bytes"
        );
        assert!(
            strip_ansi_escapes(&command)
                .chars()
                .map(thinking_cell_width)
                .sum::<usize>()
                > COLS
        );
        let (piped, _) = replay(false, COLS);
        assert_eq!(
            piped,
            format!(
                "{header} (remote)\n{running}\n{command}\n{health}\n{done}\n{reading}\n{read_done}\n{controls}\n{child}\nanswer-one\n{header} (remote)\nanswer-two\n"
            ),
            "piped observer bytes must remain complete, including controls and ANSI"
        );
    }

    #[test]
    fn sse_observer_rows_ellipsis_follow_resize_and_erase_open_tool() {
        let header = "[model (effort: max) · topics🙂]";
        let history = "  │ history before resize 界🙂";
        let running = format!("  \x1b[34m●\x1b[0m read_file · {}", "文件🙂/".repeat(12));
        let done = format!(
            "  \x1b[32m✓\x1b[0m read_file · {} 640..719",
            "文件🙂/".repeat(14)
        );
        let command = format!("  │ $ rg -n backup|clone_file {}", "目录🙂/".repeat(10));
        let health = "  │ agent-health: required delegation is overdue: delegate discovery before continuing exploration";
        let body: &'static str = Box::leak(
            format!(
                "data: {header}\n\ndata: {history}\n\ndata: {running}\n\ndata: {done}\n\ndata: {health}\n\ndata: {command}\n\ndata: {command}\n\ndata: {header}\n\nevent: answer_pending\ndata: \n\nevent: delta\ndata: {{\"delta\":\"answer-after-resize\\n\"}}\n\nevent: replace\ndata: {{\"text\":\"answer-after-resize\\n\"}}\n\nevent: done\ndata: \n\n"
            ).into_boxed_str(),
        );
        let resp = reqwest::blocking::get(serve_body_once(body)).expect("get canned SSE");
        // Resize before the completion, then grow before another command
        // arrives. There is no server-side geometry update in this replay.
        let (out, rows) = test_terminal::render(resp, true, &[140, 140, 140, 40, 40, 40, 140, 140]);
        let old_plain = strip_ansi_escapes(&running);
        assert!(old_plain.chars().map(thinking_cell_width).sum::<usize>() < 140);
        let old_rows = thinking_physical_rows(&old_plain, 40);
        assert!(old_rows > 1, "fixture must reflow the live running row");
        assert_eq!(
            out.matches("\x1b[1A\r\x1b[2K").count(),
            old_rows - 1,
            "completion must erase every reflowed running row: {out:?}"
        );
        for prefix in ["  ✓ read_file", "  │ agent-health:"] {
            let row = rows
                .iter()
                .find(|row| row.starts_with(prefix))
                .expect("narrow row");
            assert!(
                row.ends_with('…'),
                "resize used stale wide geometry: {row:?}"
            );
            assert!(row.chars().map(thinking_cell_width).sum::<usize>() <= 40);
        }
        let commands = rows
            .iter()
            .filter(|row| row.starts_with("  │ $ rg -n"))
            .collect::<Vec<_>>();
        assert_eq!(
            commands.len(),
            2,
            "each incoming command appears once: {rows:?}"
        );
        assert!(commands[0].ends_with('…'), "narrow command not clamped");
        assert!(commands[0].chars().map(thinking_cell_width).sum::<usize>() <= 40);
        assert_eq!(
            commands[1], &command,
            "growing should display the complete new row"
        );
        assert!(
            rows.iter().any(|row| row == history),
            "committed history rewritten"
        );
        assert_eq!(
            rows.iter()
                .filter(|row| *row == &format!("{header} (remote)"))
                .count(),
            2
        );
        assert!(
            !rows
                .iter()
                .any(|row| row.contains("● read_file") || row.contains("finalizing response")),
            "live row residue: {rows:?}"
        );
        assert_eq!(rows.join("\n").matches("answer-after-resize").count(), 1);
        assert!(out.ends_with('\n'));
    }

    #[test]
    fn observer_row_ellipsis_preserves_sgr_controls_and_cell_boundaries() {
        let row = "  \x1b[38;2;1;2;3m界🙂界🙂\x1b[0m";
        assert_eq!(
            clamp_observer_row(row, true, 10),
            row,
            "exact fit must keep source bytes"
        );
        assert_eq!(
            clamp_observer_row(row, true, 9),
            "  \x1b[38;2;1;2;3m界🙂界…\x1b[0m"
        );
        assert_eq!(clamp_observer_row(row, true, 1), "…\x1b[0m");
        assert_eq!(clamp_observer_row("⚠️⚠️", true, 3), "⚠️…");
        let controls = "  │ a\tb\x08\r\x1b[2Kc\x1b]0;hidden-title\x07\x1b[31";
        assert_eq!(clamp_observer_row(controls, true, 80), "  │ a    b\\u{8}c");
        assert_eq!(clamp_observer_row(controls, true, 0), controls);
        assert_eq!(clamp_observer_row(controls, false, 5), controls);
        let children = clamp_observer_row("    │ child 界🙂\n    │ second 界🙂", true, 12);
        assert_eq!(children.lines().count(), 2);
        for child in children.lines() {
            assert!(
                child.starts_with("    │ ") && child.ends_with('…'),
                "{child:?}"
            );
            assert!(child.chars().map(thinking_cell_width).sum::<usize>() <= 12);
        }
    }

    #[test]
    fn sse_tool_rows_keep_child_indent_verbatim() {
        // Regression net for the serve-chat indent report: footer/message rows
        // (tool status, long command echoes) must reach the terminal
        // byte-identical to the child bytes the server forwarded: leading
        // two-space indent, inline ANSI, and over-long rows included. The
        // client preserves all source bytes when piped or when the row fits
        // the TTY; the narrow-TTY behavior has a separate regression above.
        // Metrics summaries (`↳ cache` /
        // `↳ speed`) are dropped instead, so they can never tear the answer's
        // open line or bury the answer tail; the tool rows stay byte-identical.
        let cache = "  \u{1b}[2m↳ cache · 4.1k/22.0k tokens · 19% hit\u{1b}[0m";
        let speed = "  \u{1b}[2m↳ speed · reasoning 1.6k tok @ 48.4 tok/s\u{1b}[0m";
        let done = "  \u{1b}[32m✓\u{1b}[0m read_file  \u{1b}[2m·\u{1b}[0m target";
        let running = "  \u{1b}[34m●\u{1b}[0m execute_command";
        let long_cmd = format!("  │ $ {}", "cd /data00/x; ".repeat(20));
        assert!(long_cmd.chars().count() > 200);
        let failed = "  \u{1b}[31m✕\u{1b}[0m execute_command";
        let expected = format!("ok\n{done}\n{running}\n{long_cmd}\n{failed}\n");
        let body = format!(
            "event: delta\ndata: {{\"delta\": \"ok\"}}\n\ndata: {cache}\n\ndata: {speed}\n\ndata: {done}\n\ndata: {running}\n\ndata: {long_cmd}\n\ndata: {failed}\n\nevent: done\ndata: \n\n"
        );
        // Piped output: exact bytes, the long row is not clamped.
        // `serve_body_once` keeps the body for the whole test server lifetime.
        let canned: &'static str = Box::leak(body.into_boxed_str());
        let url = serve_body_once(canned);
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(String::from_utf8(buf).expect("utf8"), expected);
        // TTY output: the shared markdown renderer only owns body deltas, so
        // message rows stay byte-identical here too. (Body deltas go through
        // the live preview with styling/redraw sequences that only resolve on
        // a real terminal, so the TTY assertion is containment per row, not
        // whole-buffer equality.)
        let url = serve_body_once(canned);
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to_with_width(resp, &mut buf, true, None, || 400).expect("render");
        let out = String::from_utf8(buf).expect("utf8");
        for row in [done, running, long_cmd.as_str(), failed] {
            assert!(row.starts_with("  "), "fixture lost indent: {row:?}");
            assert!(!row.starts_with("   "), "fixture gained indent: {row:?}");
            assert!(
                out.lines().any(|line| line == row),
                "message row mangled or missing on TTY: {row:?}"
            );
        }
        for dropped in [cache, speed] {
            assert!(
                !out.lines().any(|line| line == dropped),
                "metrics footer leaked onto TTY: {dropped:?}"
            );
        }
    }

    #[test]
    fn sse_preserves_body_spacing_and_empty_deltas() {
        let url = serve_body_once(
            "event: delta\ndata: {\"delta\": \"paragraph\\n\\n| a | b |\\n\"}\n\nevent: delta\ndata: {\"delta\": \"\"}\n\ndata: cache\n\ndata: speed\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(buf, b"paragraph\n\n| a | b |\ncache\nspeed\n");
    }

    #[test]
    fn sse_tty_renders_table_through_shared_markdown_renderer() {
        // Same table source as the piped case above, split across two chunks:
        // on a TTY the body must render as a box table, not raw pipes.
        let url = serve_body_once(
            "event: delta\ndata: {\"delta\": \"| a | b |\\n| --- | --- |\\n\"}\n\nevent: delta\ndata: {\"delta\": \"| 1 | 2 |\\n\"}\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, true).expect("render");
        let out = String::from_utf8(buf).expect("utf8");
        assert!(out.contains('─'), "table must render as a box, got: {out}");
        assert!(
            !out.contains("| ---"),
            "raw separator must not leak, got: {out}"
        );
    }

    #[test]
    fn serve_tool_lifecycle_chat_preserves_answer_without_json() {
        let body = "event: tool\ndata: {\"id\":\"a\",\"name\":\"read_file\",\"phase\":\"started\",\"arguments\":\"{}\"}\n\nevent: delta\ndata: {\"delta\":\"answer\"}\n\nevent: tool\ndata: {\"id\":\"a\",\"name\":\"read_file\",\"phase\":\"finished\",\"output\":\"private preview\"}\n\nevent: done\ndata: \n\n";
        for tty in [false, true] {
            let response = reqwest::blocking::get(serve_body_once(body)).unwrap();
            let mut output = Vec::new();
            render_turn_stream_to(response, &mut output, tty).unwrap();
            let output = String::from_utf8(output).unwrap();
            assert!(output.contains("answer"), "{output}");
            assert!(!output.contains("private preview"), "{output}");
            assert!(!output.contains("read_file"), "{output}");
        }
    }

    #[test]
    fn sse_done_returns_before_connection_close() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let (rendered_tx, rendered_rx) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = [0u8; 4096];
            let _ = std::io::Read::read(&mut stream, &mut request);
            let body = "event: delta\ndata: {\"delta\": \"answer\"}\n\nevent: done\ndata: \n\n";
            // Advertise one more byte than we send so EOF cannot complete the
            // response. The renderer must return on done while HTTP stays open.
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{body}",
                body.len() + 1
            )
            .expect("write SSE");
            stream.flush().expect("flush SSE");
            rendered_rx.recv_timeout(Duration::from_secs(3)).is_ok()
        });
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("client");
        let resp = client.get(format!("http://{addr}/")).send().expect("SSE");
        let mut buf = Vec::new();
        let result = render_turn_stream_to(resp, &mut buf, false);
        let _ = rendered_tx.send(());
        assert!(server.join().expect("server"), "renderer waited for EOF");
        result.expect("render");
        // End-of-turn still terminates the answer's open row, even when no
        // footer follows it.
        assert_eq!(buf, b"answer\n");
    }

    #[test]
    fn sse_duplicate_thinking_done_prints_single_summary() {
        // A late/duplicate close (body-path marker plus CloseThinking frame
        // for the same round) must not print a second summary row. The
        // buffered body is discarded; only the summary row prints once.
        let url = serve_body_once(
            "event: thinking_start\ndata: \n\nevent: thinking\ndata: {\"text\": \"a b\"}\n\nevent: thinking_done\ndata: \n\nevent: thinking_done\ndata: \n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let mut buf = Vec::new();
        render_turn_stream_to(resp, &mut buf, false).expect("render");
        assert_eq!(
            String::from_utf8(buf).expect("utf8"),
            "✓ thinking (1 line)\n"
        );
    }

    #[test]
    fn sse_error_closes_open_thinking_status() {
        // A turn that fails mid-thinking must still terminate (no stuck
        // status line, no hang waiting for `done`).
        let url = serve_body_once(
            "event: thinking_start\ndata: \n\nevent: error\ndata: {\"error\": \"boom\"}\n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        let err = render_turn_stream(resp).expect_err("error event must fail");
        assert!(err.to_string().contains("boom"), "unexpected: {err}");
    }

    #[test]
    fn thinking_close_print_caps_a_long_body() {
        // A 100-line thinking block must not leave any body rows in the
        // scrollback: only the summary row stays, like the local fold.
        let body = (1..=100)
            .map(|i| format!("thought line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut region = ThinkingLiveRegion::new(2, Some(80));
        region.append(&body);
        let mut active = true;
        let mut line_open = true;
        let mut buf = Vec::new();
        close_thinking_status(
            &mut active,
            &mut line_open,
            100,
            &mut region,
            false,
            &mut buf,
            &mut ObserverRowState::default(),
        );
        let out = String::from_utf8(buf).expect("utf8");
        assert_eq!(out, "✓ thinking (100 lines)\n");
        assert!(region.tail.is_empty(), "body must drain exactly once");
    }

    #[test]
    fn thinking_close_print_keeps_a_short_body_whole() {
        // Even a short body leaves no body rows: only the summary stays.
        let mut region = ThinkingLiveRegion::new(2, Some(80));
        region.append("a\nb\nc");
        let mut active = true;
        let mut line_open = true;
        let mut buf = Vec::new();
        close_thinking_status(
            &mut active,
            &mut line_open,
            3,
            &mut region,
            false,
            &mut buf,
            &mut ObserverRowState::default(),
        );
        let out = String::from_utf8(buf).expect("utf8");
        assert_eq!(out, "✓ thinking (3 lines)\n");
    }

    #[test]
    fn fork_remote_session_parses_new_id() {
        // serve_body_once ignores the request path and method, so the
        // canned fork body stands in for POST /sessions/{id}/fork.
        let url = serve_body_once(r#"{"id":"f1"}"#);
        let base = url.split("/sessions/").next().unwrap_or(&url);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("client");
        let id = fork_remote_session(&client, base, "", "cur").expect("fork");
        assert_eq!(id, "f1");
    }

    #[test]
    fn delete_remote_session_reports_server_flag() {
        let url = serve_body_once(r#"{"deleted":true}"#);
        let base = url.split("/sessions/").next().unwrap_or(&url);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("client");
        assert!(delete_remote_session(&client, base, "", "cur").expect("delete"));
    }

    fn listed_session(id: &str, summary: Option<&str>, prompt: Option<&str>) -> RemoteSession {
        RemoteSession {
            id: id.to_string(),
            summary: summary.map(str::to_string),
            first_user_prompt: prompt.map(str::to_string),
            marked: false,
            archived: false,
        }
    }

    #[test]
    fn parse_remote_sessions_drops_empty_ids() {
        let items = parse_remote_sessions(
            r#"[{"id":"a1","summary":"Fix bug","size_bytes":10,"marked":false,"archived":true},{"id":"  "}]"#,
        )
        .expect("parse");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "a1");
        assert_eq!(items[0].summary.as_deref(), Some("Fix bug"));
        assert!(items[0].archived);
        parse_remote_sessions("not json").expect_err("malformed must fail");
        // Older servers without the optional fields still parse.
        let legacy = parse_remote_sessions(r#"[{"id":"old"}]"#).expect("legacy");
        assert!(legacy[0].summary.is_none());
        assert!(!legacy[0].marked);
        assert!(!legacy[0].archived);
    }

    #[test]
    fn serve_sessions_arg_dispatch_mirrors_local_repl() {
        use ServeSessionsArg::*;
        // Bare `/ss` and the list spellings keep the full listing.
        for bare in ["", "list", "ls", "verbose"] {
            assert!(
                matches!(classify_serve_sessions_arg(bare), List),
                "{bare:?}"
            );
        }
        // `current`/`cur`/`curr` show only the current session; `curr` is
        // the spelling the reported serve-chat bug silently listed all
        // sessions for.
        for cur in ["current", "cur", "curr"] {
            assert!(
                matches!(classify_serve_sessions_arg(cur), Current),
                "{cur:?}"
            );
        }
        assert!(matches!(classify_serve_sessions_arg("all"), All));
        assert!(matches!(classify_serve_sessions_arg("archive"), Archive));
        assert!(matches!(classify_serve_sessions_arg("id"), Id));
        // Anything else is rejected, never silently treated as a listing.
        for unknown in ["foo", "use", "delete", "workers"] {
            assert!(
                matches!(classify_serve_sessions_arg(unknown), Unknown),
                "{unknown:?}"
            );
        }
    }

    #[test]
    fn sessions_for_view_hides_archived_except_current() {
        use ServeSessionsArg::*;
        let active = listed_session("active", Some("A"), None);
        let mut old = listed_session("old", Some("O"), None);
        old.archived = true;
        let mut current = listed_session("current", Some("C"), None);
        current.archived = true;
        let items = vec![active, old, current];
        let ids = |rows: Vec<RemoteSession>| -> Vec<String> {
            rows.into_iter().map(|row| row.id).collect()
        };
        assert_eq!(
            ids(sessions_for_view(items.clone(), List, "current")),
            vec!["active", "current"]
        );
        assert_eq!(
            ids(sessions_for_view(items.clone(), All, "current")),
            vec!["active", "old", "current"]
        );
        assert_eq!(
            ids(sessions_for_view(items, Archive, "current")),
            vec!["old", "current"]
        );
    }

    #[test]
    fn archive_view_row_numbers_resolve_against_archived_rows() {
        use ServeSessionsArg::*;
        let active = listed_session("active", Some("A"), None);
        let mut old = listed_session("old", Some("O"), None);
        old.archived = true;
        let mut older = listed_session("older", Some("O2"), None);
        older.archived = true;
        // The rows the archive view prints are the rows `/resume <number>`
        // resolves against, so the visible numbering stays meaningful.
        let view = sessions_for_view(vec![active, old, older], Archive, "active");
        assert_eq!(resolve_session_ref("1", &view).unwrap(), "old");
        assert_eq!(resolve_session_ref("2", &view).unwrap(), "older");
    }

    #[test]
    fn remote_session_title_prefers_summary_then_prompt() {
        assert_eq!(
            remote_session_title(&listed_session("x", Some("T"), Some("P"))),
            "T"
        );
        // First prompt falls back to its first line only.
        assert_eq!(
            remote_session_title(&listed_session("x", None, Some("hello\nworld"))),
            "hello"
        );
        assert_eq!(
            remote_session_title(&listed_session("x", Some("  "), None)),
            "(untitled)"
        );
        // Long titles truncate with an ellipsis at a char boundary.
        let long = "字".repeat(100);
        let title = remote_session_title(&listed_session("x", Some(&long), None));
        assert_eq!(title.chars().count(), 61, "unexpected: {title}");
        assert!(title.ends_with('…'));
        let mut marked = listed_session("x", Some("T"), None);
        marked.marked = true;
        assert_eq!(remote_session_title(&marked), "T ★");
    }

    #[test]
    fn remote_session_row_marks_current_and_highlights_marked() {
        let current = listed_session("cur", Some("C"), None);
        // Interactive rows carry a `*`/` ` mark column, like the local view.
        assert_eq!(
            format_remote_session_row(&current, 0, Some("cur"), false),
            "* 1. cur  C"
        );
        assert_eq!(
            format_remote_session_row(&current, 0, Some("other"), false),
            "  1. cur  C"
        );
        // Without a current session the legacy pipeable bytes are kept.
        assert_eq!(
            format_remote_session_row(&current, 0, None, false),
            "1. cur  C"
        );
        // On a terminal, marked rows carry the theme highlight.
        let mut marked = listed_session("m", Some("T"), None);
        marked.marked = true;
        let row = format_remote_session_row(&marked, 1, Some("cur"), true);
        assert!(
            row.starts_with(&format!(
                "{}  2. m  T ★",
                crate::ai::theme::current().accent_marked
            )),
            "{row:?}"
        );
        assert!(row.ends_with(crate::ai::theme::RESET), "{row:?}");
        // Non-interactive output keeps its legacy bytes even on a terminal.
        assert_eq!(
            format_remote_session_row(&marked, 1, None, true),
            "2. m  T ★"
        );
        // Archived rows render dimmed and keep the explicit suffix.
        let mut archived = listed_session("o", Some("O"), None);
        archived.archived = true;
        let row = format_remote_session_row(&archived, 0, Some("cur"), true);
        assert!(row.contains(crate::ai::theme::DIM), "{row:?}");
        assert!(row.contains("[archived]"), "{row:?}");
        assert_eq!(
            format_remote_session_row(&archived, 0, None, true),
            "1. o  O  [archived]"
        );
        // Piped output stays plain.
        let row = format_remote_session_row(&marked, 0, Some("cur"), false);
        assert!(!row.contains('\u{1b}'), "{row:?}");
    }

    #[test]
    fn resolve_session_ref_accepts_number_prefix_or_id() {
        let listed = vec![
            listed_session("b2b23f0c-aaaa", Some("A"), None),
            listed_session("b2b23f0c-bbbb", Some("B"), None),
        ];
        assert_eq!(resolve_session_ref("1", &listed).unwrap(), "b2b23f0c-aaaa");
        assert_eq!(resolve_session_ref("2", &listed).unwrap(), "b2b23f0c-bbbb");
        assert!(resolve_session_ref("3", &listed).is_err());
        assert!(resolve_session_ref("", &listed).is_err());
        // Unique prefix resolves; ambiguous prefix errors.
        assert_eq!(
            resolve_session_ref("b2b23f0c-aaaa", &listed).unwrap(),
            "b2b23f0c-aaaa"
        );
        assert!(resolve_session_ref("b2b23f0c", &listed).is_err());
        // A full id that was never listed passes through for the server to check.
        assert_eq!(
            resolve_session_ref("elsewhere-id", &listed).unwrap(),
            "elsewhere-id"
        );
        assert!(resolve_session_ref("1", &[]).is_err());
    }

    #[test]
    fn fetch_remote_sessions_parses_canned_list() {
        // serve_body_once ignores the request path, so the canned session
        // list stands in for GET /sessions over loopback.
        let url =
            serve_body_once(r#"[{"id":"s1","summary":"Hello","size_bytes":7,"marked":true}]"#);
        let base = url.rsplit_once('/').map(|(b, _)| b).unwrap_or(&url);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("client");
        let items = fetch_remote_sessions(&client, base, "").expect("fetch");
        assert_eq!(items.len(), 1);
        assert_eq!(remote_session_title(&items[0]), "Hello ★");
    }

    #[test]
    fn parse_server_info_reads_full_body() {
        let info = parse_server_info(
            r#"{"model":"m","model_label":"M","agent":"build","reasoning_effort":"max","version":"0.1.0"}"#,
        )
        .expect("parse");
        assert_eq!(info.model, "m");
        assert_eq!(info.model_label, "M");
        assert_eq!(info.agent, "build");
        assert_eq!(info.reasoning_effort, "max");
        assert_eq!(info.version, "0.1.0");
    }

    #[test]
    fn parse_server_info_tolerates_thin_body() {
        // Older servers have no `/info`: missing fields default to blank so
        // the chat keeps its previous header behavior instead of failing.
        let info = parse_server_info("{}").expect("parse");
        assert!(info.model.is_empty());
        assert!(info.model_label.is_empty());
        assert!(parse_server_info("not json").is_err());
    }

    #[test]
    fn current_remote_title_resolves_listed_session() {
        let listed = vec![
            RemoteSession {
                id: "s1".to_string(),
                summary: Some("Hello".to_string()),
                first_user_prompt: None,
                marked: false,
                archived: false,
            },
            RemoteSession {
                id: "s2".to_string(),
                summary: None,
                first_user_prompt: Some("question\nbody".to_string()),
                marked: false,
                archived: false,
            },
        ];
        assert_eq!(
            current_remote_title(&listed, "s1").as_deref(),
            Some("Hello")
        );
        assert_eq!(
            current_remote_title(&listed, "s2").as_deref(),
            Some("question")
        );
        assert_eq!(current_remote_title(&listed, "missing"), None);
    }

    #[test]
    fn serve_config_patch_serializes_only_set_fields() {
        let empty = ServeConfigPatch::default();
        assert!(empty.is_empty());
        assert_eq!(serde_json::to_string(&empty).unwrap(), "{}");

        // A blank reasoning_effort clears the stored value server-side; absent
        // fields (agent here) keep the stored value untouched.
        let patch = ServeConfigPatch {
            model: Some("m1".to_string()),
            reasoning_effort: Some(String::new()),
            ..Default::default()
        };
        assert!(!patch.is_empty());
        let json = serde_json::to_string(&patch).unwrap();
        assert!(json.contains(r#""model":"m1""#));
        assert!(json.contains(r#""reasoning_effort":"""#));
        assert!(!json.contains("agent"));
    }

    #[test]
    fn merge_restored_session_config_fills_only_nonempty_picks() {
        let mut selection = ServeTurnSelection {
            model: "default-m".to_string(),
            agent: "build".to_string(),
            reasoning_effort: None,
            skills: Vec::new(),
        };
        let cfg = ServeSessionConfig {
            model: "stored-m".to_string(),
            agent: String::new(), // empty -> keep the selection's agent
            reasoning_effort: "high".to_string(),
        };
        assert!(merge_restored_session_config(&cfg, &mut selection));
        assert_eq!(selection.model, "stored-m");
        assert_eq!(selection.agent, "build");
        assert_eq!(selection.reasoning_effort.as_deref(), Some("high"));

        // Restoring the same picks again changes nothing.
        assert!(!merge_restored_session_config(&cfg, &mut selection));
    }

    #[test]
    fn serve_chat_bg_binding_roundtrips_per_terminal() {
        let root = std::env::temp_dir().join(format!(
            "serve-chat-bg-test-{}-roundtrip",
            uuid::Uuid::new_v4().simple()
        ));
        let _ = std::fs::remove_dir_all(&root);
        save_serve_chat_binding_for_key(
            &root,
            "terminal:term-1",
            "http://127.0.0.1:8080",
            "sess-1",
        )
        .expect("save");
        let loaded = load_serve_chat_binding_for_key(&root, "terminal:term-1").expect("load");
        assert_eq!(loaded.base, "http://127.0.0.1:8080");
        assert_eq!(loaded.session_id, "sess-1");
        // A different terminal key sees nothing: bindings are per-terminal,
        // like the local suspended store.
        assert_eq!(
            load_serve_chat_binding_for_key(&root, "terminal:term-2"),
            None
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn serve_chat_bg_binding_rejects_corrupt_file() {
        let root = std::env::temp_dir().join(format!(
            "serve-chat-bg-test-{}-corrupt",
            uuid::Uuid::new_v4().simple()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("mkdir");
        std::fs::write(
            serve_chat_binding_path(&root, "terminal:term-1"),
            "not json",
        )
        .expect("write");
        assert_eq!(
            load_serve_chat_binding_for_key(&root, "terminal:term-1"),
            None
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn serve_chat_suspended_root_honors_override_next_to_it() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "serve-chat-bg-test-{}-override",
            uuid::Uuid::new_v4().simple()
        ));
        let suspended = dir.join("suspended_sessions");
        unsafe {
            std::env::set_var(
                "RUST_TOOLS_SUSPENDED_SESSIONS_DIR",
                suspended.to_str().expect("utf8"),
            );
        }
        let root = serve_chat_suspended_root();
        unsafe {
            std::env::remove_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR");
        }
        assert_eq!(root, dir.join("serve_chat_suspended"));
    }

    #[test]
    fn serve_sigint_stops_the_turn_before_exiting() {
        use super::{ServeSigintAction, serve_sigint_action};
        // A running turn claims the first press; anything else falls through to
        // the local-exit behaviour.
        assert_eq!(
            serve_sigint_action(true, false),
            ServeSigintAction::InterruptRemoteTurn
        );
        assert_eq!(
            serve_sigint_action(true, true),
            ServeSigintAction::ExitLocal
        );
        assert_eq!(
            serve_sigint_action(false, false),
            ServeSigintAction::ExitLocal
        );
        assert_eq!(
            serve_sigint_action(false, true),
            ServeSigintAction::ExitLocal
        );
    }

    /// The interrupt must address the session-scoped route with the bearer
    /// header, and an "already finished" answer must surface as `false`
    /// instead of an error.
    #[test]
    fn interrupt_request_targets_the_session_route() {
        let (base, captured) = serve_capture_once("{\"session_id\":\"s1\",\"interrupted\":true}");
        let client = reqwest::blocking::Client::new();
        let interrupted = interrupt_remote_turn(&client, &base, "tk", "s1").expect("request");
        assert!(interrupted);
        let head = captured.lock().expect("capture").clone().to_lowercase();
        assert!(
            head.starts_with("post /sessions/s1/interrupt http/1.1"),
            "request head: {head}"
        );
        assert!(
            head.contains("authorization: bearer tk"),
            "request head: {head}"
        );

        let (base, _) = serve_capture_once("{\"session_id\":\"s1\",\"interrupted\":false}");
        assert!(!interrupt_remote_turn(&client, &base, "tk", "s1").expect("request"));
    }

    #[test]
    fn checkpoint_requests_target_the_remote_session_route() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let (base, captured) = serve_capture_once(
            r#"{"output":"switched\n","confirmation":null,"checkpoints":["latest","before"],"branches":["main","feature"],"checkpoint_status":"feature@latest"}"#,
        );
        let client = reqwest::blocking::Client::new();
        let response = post_remote_checkpoint_command(
            &client,
            &base,
            "tk",
            "s1",
            "/git switch feature",
            false,
        )
        .expect("checkpoint request");
        assert_eq!(response.output, "switched\n");
        assert_eq!(response.checkpoints, ["latest", "before"]);
        assert_eq!(response.branches, ["main", "feature"]);
        assert_eq!(
            response.checkpoint_status.as_deref(),
            Some("feature@latest")
        );
        let head = captured.lock().expect("capture").clone().to_lowercase();
        assert!(
            head.starts_with("post /sessions/s1/checkpoints http/1.1"),
            "request head: {head}"
        );
        assert!(
            head.contains("authorization: bearer tk"),
            "request head: {head}"
        );

        let (base, _) = serve_capture_once(
            r#"{"checkpoints":["latest","before"],"branches":["main","feature"],"checkpoint_status":"main@latest"}"#,
        );
        let checkpoint_state =
            fetch_remote_checkpoint_names(&client, &base, "tk", "s1").expect("checkpoint list");
        assert_eq!(checkpoint_state.checkpoints, ["latest", "before"]);
        assert_eq!(checkpoint_state.branches, ["main", "feature"]);
        assert_eq!(
            checkpoint_state.checkpoint_status.as_deref(),
            Some("main@latest")
        );
        CommandCompleter::clear_checkpoint_candidates();
    }

    #[test]
    fn cd_command_posts_to_the_remote_cd_route_and_renders_the_result() {
        let (base, captured) = serve_capture_once(
            r#"{"stdout":"Working directory: /srv/work (was /srv)\n","stderr":""}"#,
        );
        let client = reqwest::blocking::Client::new();
        run_serve_cd_command(&client, &base, "tk", "s1", "/cd /srv/work").expect("cd command");
        let head = captured.lock().expect("capture").clone().to_lowercase();
        assert!(
            head.starts_with("post /sessions/s1/cd http/1.1"),
            "request head: {head}"
        );
        assert!(
            head.contains("authorization: bearer tk"),
            "request head: {head}"
        );
        assert!(
            head.contains("content-type: application/json"),
            "request head: {head}"
        );

        // A rejected switch surfaces the server's error message, not a panic.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = vec![0u8; 4096];
            let _ = std::io::Read::read(&mut stream, &mut request);
            let body = r#"{"error":"not a /cd command"}"#;
            let response = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        });
        let err = run_serve_cd_command(
            &client,
            &format!("http://{addr}"),
            "tk",
            "s1",
            "plain prompt",
        )
        .unwrap_err();
        assert!(err.contains("not a /cd command"), "unexpected error: {err}");
    }

    #[test]
    fn serve_chat_subagent_commands_use_config_route() {
        let client = reqwest::blocking::Client::new();
        for (arg, method) in [
            ("", "GET"),
            ("\nfix login", "GET"),
            (" \nfix login", "GET"),
            ("traecli GPT-5.5\nfix login", "POST"),
            ("kernel", "POST"),
        ] {
            let (base, captured) = serve_capture_once(r#"{"output":"server state\n"}"#);
            run_serve_use_command(&client, &base, "tk", "s1", arg).expect("use config");
            let request = captured.lock().unwrap().clone();
            assert!(
                request.starts_with(&format!("{method} /sessions/s1/config HTTP/1.1")),
                "{request}"
            );
            assert!(
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer tk")
            );
            let (_, body) = request.split_once("\r\n\r\n").expect("request body seam");
            if method == "POST" {
                let body: serde_json::Value = serde_json::from_str(body).unwrap();
                assert_eq!(body["subagents"]["action"], "use");
                assert!(!body.to_string().contains("fix login"));
                assert!(body.get("prompt").is_none());
                assert!(body.get("title_seed").is_none());
                if arg.starts_with("traecli") {
                    assert_eq!(body["subagents"]["backend"], "traecli");
                    assert_eq!(body["subagents"]["model"], "GPT-5.5");
                } else {
                    assert_eq!(body["subagents"]["backend"], "kernel");
                    assert!(body["subagents"]["model"].is_null());
                }
            } else {
                assert!(body.is_empty());
            }
        }
        let (base, captured) = serve_capture_once(r#"{"output":"reset state\n"}"#);
        run_serve_reset_command(&client, &base, "tk", "s1", "").expect("reset config");
        let request = captured.lock().unwrap().clone();
        assert!(request.starts_with("POST /sessions/s1/config HTTP/1.1"));
        let body: serde_json::Value =
            serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body, serde_json::json!({"subagents": {"action": "reset"}}));
        assert!(
            run_serve_reset_command(&client, &base, "tk", "s1", "\nquestion")
                .unwrap_err()
                .contains("no arguments or trailing prompt")
        );
        assert!(
            run_serve_use_command(&client, &base, "tk", "s1", "traecli model extra")
                .unwrap_err()
                .contains("Too many arguments")
        );
        let (base, _) = serve_capture_status_once(
            "400 Bad Request",
            r#"{"error":"unknown backend from server"}"#,
        );
        let error = run_serve_use_command(&client, &base, "tk", "s1", "unknown").unwrap_err();
        assert!(error.contains("unknown backend from server"), "{error}");
    }

    #[test]
    fn serve_chat_use_tail_preserves_child_input_and_extracts_nested_question() {
        for (input, tail, seed) in [
            (
                "/use kernel\n/task 3 fix login",
                "/task 3 fix login",
                "fix login",
            ),
            (
                ":use traecli\n:audit --fast inspect login",
                ":audit --fast inspect login",
                "inspect login",
            ),
            (
                "/use kernel\n/task 3 fix @Skill:review login",
                "/task 3 fix @Skill:review login",
                "fix  login",
            ),
            ("/use kernel\n/title t", "/title t", ""),
            (
                "/use kernel\n/use traecli\n/task 3 fix login",
                "/use traecli\n/task 3 fix login",
                "fix login",
            ),
        ] {
            let (child_input, title_seed) = serve_use_turn_input(input).expect("use tail");
            assert_eq!(child_input, tail);
            assert_eq!(title_seed, seed);
        }
        assert!(serve_use_turn_input("/use kernel").is_none());
        assert!(serve_use_turn_input("/use kernel\n  ").is_none());
    }

    #[test]
    fn serve_chat_turn_stream_sends_explicit_title_seed() {
        let client = reqwest::blocking::Client::new();
        let history_file = std::env::temp_dir().join(format!(
            "a-serve-title-wire-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        for (prompt, title_seed) in [
            ("/task 3 fix login", Some("fix login")),
            ("/title t", Some("")),
            ("/mark m", Some("")),
            ("  ordinary question  ", None),
        ] {
            let (base, captured) = serve_capture_once("event: done\ndata: \n\n");
            let mut selection = ServeTurnSelection {
                model: String::new(),
                agent: String::new(),
                reasoning_effort: None,
                skills: vec!["code-review".to_string()],
            };
            let route = classify_serve_input(prompt);
            let extracted = if route == ServeInputRoute::ServerCommand {
                serve_command_title_seed(prompt)
            } else {
                None
            };
            assert_eq!(extracted.as_deref(), title_seed);
            post_turn_stream(
                &client,
                &base,
                "tk",
                "s1",
                prompt,
                extracted.as_deref(),
                &history_file,
                &mut selection,
            )
            .expect("turn stream");
            let request = captured.lock().unwrap().clone();
            assert!(request.starts_with("POST /sessions/s1/turns/stream HTTP/1.1"));
            let body: serde_json::Value =
                serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
            assert_eq!(body["prompt"], prompt);
            assert_eq!(body["title_seed"], serde_json::json!(title_seed));
            assert_eq!(body["skills"], serde_json::json!(["code-review"]));
        }
    }

    #[test]
    fn cd_completion_fetches_remote_candidates_from_the_candidates_route() {
        let (base, captured) = serve_capture_once(r#"{"candidates":["src/","docs/"]}"#);
        let client = reqwest::blocking::Client::new();
        let candidates =
            fetch_remote_cd_candidates(&client, &base, "tk", "s1", "s").expect("cd candidates");
        assert_eq!(candidates, vec!["src/", "docs/"]);
        let head = captured.lock().expect("capture").clone().to_lowercase();
        assert!(
            head.starts_with("get /sessions/s1/cd-candidates?frag=s http/1.1"),
            "request head: {head}"
        );
        assert!(
            head.contains("authorization: bearer tk"),
            "request head: {head}"
        );
    }

    /// The prompt the server stores must be the one the user typed. Only
    /// command lines, whose grammar owns the words, are sent trimmed.
    #[test]
    fn prompt_route_sends_typed_bytes_while_commands_stay_trimmed() {
        let typed = "  fn main() {}\n\n";
        let trimmed = typed.trim();
        assert_eq!(
            routed_prompt_text(&ServeInputRoute::Prompt, typed, trimmed),
            typed,
            "a prompt must reach the server exactly as typed"
        );
        assert_eq!(
            routed_prompt_text(&ServeInputRoute::ServerCommand, typed, trimmed),
            "fn main() {}",
            "a command line keeps its trimmed grammar form"
        );
    }
}
