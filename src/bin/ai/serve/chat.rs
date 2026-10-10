//! Interactive client for a running serve instance.
//!
//! Local testing REPL that mirrors the single-machine interactive input: the
//! same multiline `PromptEditor` (Enter inserts a newline, Esc or Alt+Enter
//! submits) and progressive output rendering. Each submitted prompt is POSTed
//! to `POST /sessions/{id}/turns/stream`; every SSE event is rendered as it
//! arrives instead of waiting for the whole turn.

use std::{
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
    agents, cli::ParsedCli, config_schema::AiConfig, driver::{commands::is_local_command_start, input::inline_image_filenames},
    history::{current_terminal_key, SessionStore},
    model_names, models,
    prompt::{completion::CommandCompleter, PromptEditor},
    stream::{side_note_input::SideNoteInputGuard, MarkdownStreamRenderer},
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
        let _ =
            std::fs::remove_file(serve_chat_binding_path(&serve_chat_suspended_root(), &key));
    }
}

/// One entry of `GET /sessions` (newest first). `summary` is the generated
/// title when the server has one; otherwise the client falls back to the
/// first user prompt. Optional fields default so older servers still parse.
#[derive(Debug, Deserialize)]
struct RemoteSession {
    id: String,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    first_user_prompt: Option<String>,
    #[serde(default)]
    marked: bool,
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
    let resp = auth(
        client.get(format!("{base}/sessions")),
        token,
    )
    .send()
    .map_err(|err| format!("cannot reach serve at {base} ({err})"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(server_error_message(status, &text));
    }
    parse_remote_sessions(&text)
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
fn merge_restored_session_config(cfg: &ServeSessionConfig, selection: &mut ServeTurnSelection) -> bool {
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
    let cfg = auth(client.get(format!("{base}/sessions/{session_id}/config")), token)
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
        _ => Err(
            "unknown effort; expected minimal|low|medium|high|xhigh|max|off|auto"
                .to_string(),
        ),
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
            println!("(remote) current model: {}", models::model_display_label(&handle));
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
            println!("(remote) switched model to {}", models::model_display_label(&handle));
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
            let marker = if agent.name == selection.agent { "*" } else { " " };
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
        println!("(remote) /skills [list|current|use <name>...|clear|<names...> [question]] - force server skills for the next turn");
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
    if arg.is_empty() || arg.eq_ignore_ascii_case("list") || arg.eq_ignore_ascii_case("ls") {
        print_serve_skills_list(&entries, &selection.skills);
        return None;
    }
    if arg.eq_ignore_ascii_case("use") {
        println!("(remote) Usage: /skills use <skill-name> [<skill-name>...]");
        print_serve_skills_list(&entries, &selection.skills);
        return None;
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
    println!("(remote) skills selected for next turn: {}", names.join(", "));
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
        return listed.get(n.wrapping_sub(1)).map(|s| s.id.clone()).ok_or_else(|| {
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

fn normalize_base(bind: &str) -> String {
    let trimmed = bind.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

fn auth(
    req: reqwest::blocking::RequestBuilder,
    token: &str,
) -> reqwest::blocking::RequestBuilder {
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
}

const SERVE_HISTORY_DEFAULT_COUNT: usize = 6;
const SERVE_HISTORY_FULL_COUNT: usize = 20;
const SERVE_HISTORY_MAX_COUNT: usize = 20;
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

/// Canonical (0-based) index of the Nth stored user message, counting every
/// `role == "user"` row exactly like the local rewind resolver so ordinals
/// match on both sides. A runtime-injected row still resolves here; the
/// server rejects it with 400, same rule as local.
fn serve_user_message_index(
    messages: &[RemoteHistoryMessage],
    ordinal: usize,
) -> Option<usize> {
    if ordinal == 0 {
        return None;
    }
    let mut seen = 0usize;
    for (idx, message) in messages.iter().enumerate() {
        if message.role != "user" {
            continue;
        }
        seen += 1;
        if seen == ordinal {
            return Some(idx);
        }
    }
    None
}

fn serve_user_message_count(messages: &[RemoteHistoryMessage]) -> usize {
    messages.iter().filter(|m| m.role == "user").count()
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
                    view.count = count.clamp(1, SERVE_HISTORY_MAX_COUNT);
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
    // Ordinals count every stored user message before filtering, like local.
    let mut user_ordinal = 0usize;
    let mut filtered = Vec::new();
    for message in messages {
        let ordinal = if message.role == "user" {
            user_ordinal += 1;
            Some(user_ordinal)
        } else {
            None
        };
        let role_matches = match view.role {
            ServeHistoryRole::All => true,
            ServeHistoryRole::User => message.role == "user",
            ServeHistoryRole::Assistant => message.role == "assistant",
            ServeHistoryRole::Tool => message.role == "tool",
            // Same rule as the local system filter (system + internal notes).
            ServeHistoryRole::System => {
                message.role == "system" || message.role == "internal_note"
            }
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
    let header = if nth_back == 1 {
        "[history] Latest assistant message:".to_string()
    } else {
        format!("[history] Assistant message {nth_back} back from latest:")
    };
    Ok(format!("{header}\n{text}"))
}

/// Print the Nth recent assistant conclusion like local `/history last`:
/// the header stays plain text while the body goes through the same
/// display-only post-processing and terminal markdown renderer as a live
/// turn. Keeps `render_serve_history_last` pure (it still returns the raw
/// `"header\nbody"` string for tests and `/history replay`).
fn print_serve_history_last_rendered(
    messages: &[RemoteHistoryMessage],
    nth_back: usize,
) -> Result<(), String> {
    let rendered = render_serve_history_last(messages, nth_back)?;
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
    let Some(first) = args.first().copied() else {
        return Err("missing rewind target. try: /history rewind u3".into());
    };
    if first == "last" || first == "latest" {
        let count = serve_user_message_count(messages);
        if count == 0 {
            return Err("no user input found in history".into());
        }
        let index =
            serve_user_message_index(messages, count).expect("counted user message resolves");
        return Ok((count, index));
    }
    if first == "grep" {
        let keyword = args[1..].join(" ");
        if keyword.trim().is_empty() {
            return Err("`/history rewind grep` requires a keyword".into());
        }
        let mut ordinal = 0usize;
        let mut matches = Vec::new();
        for message in messages {
            if message.role != "user" {
                continue;
            }
            ordinal += 1;
            let text = serve_history_searchable_text(&message.content);
            if text
                .to_ascii_lowercase()
                .contains(&keyword.to_ascii_lowercase())
            {
                matches.push((ordinal, message));
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
            for (ord, message) in matches.iter().take(8) {
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
        let (ord, _) = matches[0];
        let index =
            serve_user_message_index(messages, ord).expect("matched user message resolves");
        return Ok((ord, index));
    }
    let raw = first
        .strip_prefix('u')
        .or_else(|| first.strip_prefix('U'))
        .unwrap_or(first);
    let ordinal = raw
        .parse::<usize>()
        .map_err(|_| format!("invalid rewind target: {first}. try: /history rewind u3"))?;
    if ordinal == 0 {
        return Err("user ordinal must be >= 1".into());
    }
    serve_user_message_index(messages, ordinal)
        .map(|index| (ordinal, index))
        .ok_or_else(|| format!("user input u{ordinal} not found"))
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
            let messages = fetch_remote_history(client, base, token, session_id)?;
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
            let messages = fetch_remote_history(client, base, token, session_id)?;
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
    let messages = fetch_remote_history(client, base, token, session_id)?;
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
    let messages = fetch_remote_history(client, base, token, session_id)?;
    let (ordinal, index) = resolve_serve_rewind_target(&messages, args)?;
    let removed = messages.len() - index;
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
    let Some(first) = args.first().copied() else {
        return Err("missing rewind target. try: /history rewind u3".into());
    };
    if first == "last" || first == "latest" {
        return Ok(());
    }
    if first == "grep" {
        if args[1..].join(" ").trim().is_empty() {
            return Err("`/history rewind grep` requires a keyword".into());
        }
        return Ok(());
    }
    let raw = first
        .strip_prefix('u')
        .or_else(|| first.strip_prefix('U'))
        .unwrap_or(first);
    let ordinal = raw
        .parse::<usize>()
        .map_err(|_| format!("invalid rewind target: {first}. try: /history rewind u3"))?;
    if ordinal == 0 {
        return Err("user ordinal must be >= 1".into());
    }
    Ok(())
}

fn print_serve_history_help() {
    println!(
        "/history usage (remote session):\n  /history [N]           Show last N messages (default: {})\n  /history full          Show full messages\n  /history user/assistant/tool/system\n                         Filter by role\n  /history grep <keyword>  Search messages\n  /history last [N]      Replay the Nth recent assistant conclusion (full text)\n  /history replay        Replay the last assistant conclusion (text only)\n  /history rewind u<N>   Remove user message u<N> and everything after it\n  /history rewind last   Remove latest user message and everything after it\n  /history rewind grep <keyword>\n                         Rewind the only user message matching keyword\n  /history help          Show this help\n\nCopy/export and other-session views are local-only: print the view here and save or copy it locally. Assistant rows carry no model tag; the history route serves bare messages.",
        SERVE_HISTORY_DEFAULT_COUNT
    );
}

#[cfg(test)]
mod serve_history_tests {
    use super::*;
    use serde_json::json;

    fn test_message(role: &str, text: &str) -> RemoteHistoryMessage {
        RemoteHistoryMessage {
            role: role.to_string(),
            content: json!(text),
            tool_calls: None,
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
    fn serve_history_view_full_roles_and_clamp() {
        let (view, _) = parse_serve_history_view(&["full"]).expect("parse");
        assert!(view.full);
        assert_eq!(view.count, SERVE_HISTORY_FULL_COUNT);
        let (view, _) = parse_serve_history_view(&["assistant", "8"]).expect("parse");
        assert_eq!(view.role, ServeHistoryRole::Assistant);
        assert_eq!(view.count, 8);
        let (view, _) = parse_serve_history_view(&["999"]).expect("parse");
        assert_eq!(view.count, SERVE_HISTORY_MAX_COUNT);
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
        let err =
            resolve_serve_rewind_target(&messages, &["grep", "question"]).expect_err("multi");
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
        assert!(is_local_command_start("/history rewind u40"));
        assert!(is_local_command_start(":history"));
        assert!(is_local_command_start("/compact"));
        assert!(!is_local_command_start("/tmp/foo"));
        assert!(!is_local_command_start("plain prompt"));
    }
}

/// Render one SSE turn stream progressively: message lines print as they
/// arrive, `delta` events (live body chunks) print without a trailing
/// newline, thinking frames collapse into one live status line while the
/// block is open and print only the `✓ thinking (N lines)` summary on close
/// (local-fold parity: the body is discarded), error events abort with
/// the server detail, the done event (or the end of the body) ends the turn.
fn render_turn_stream(
    resp: reqwest::blocking::Response,
) -> Result<(), Box<dyn std::error::Error>> {
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
/// same gates (today `git commit` / `git stash`) as a local turn.
fn render_turn_stream_to_with_confirm(
    resp: reqwest::blocking::Response,
    out: &mut dyn std::io::Write,
    tty: bool,
    confirm: Option<&ServeConfirmChannel>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::BufRead;
    // `blocking::Response` implements `Read`, so lines are parsed (and
    // printed) as soon as the server flushes them, not after the whole turn.
    let mut reader = std::io::BufReader::new(resp);
    let mut event_kind = String::new();
    let mut raw = String::new();
    // Folded-thinking state: the body text stays buffered in the status
    // buffer until the block closes, so it never mixes with the answer.
    // Line count is approximate (chunks split mid-line); it only feeds the
    // summary row.
    let mut thinking_active = false;
    let mut thinking_newlines = 0usize;
    let mut thinking_chars = 0usize;
    let mut thinking_body = String::new();
    // Whether the cursor sits mid-line (answer delta without a trailing
    // newline, or a live thinking status). Footer/message lines must start
    // on a fresh row instead of gluing onto the answer's last line.
    let mut line_open = false;
    // End-of-stream metrics summaries (`↳ cache` / `↳ speed`) buffered instead
    // of printed inline: a turn can keep streaming body chunks afterwards
    // (multi-round model calls), and the FIFO/stdout pumps can reorder the two,
    // so printing one here would tear the answer's open line mid-sentence.
    // They flush at end-of-turn, like the local client which prints metrics
    // after the answer completes.
    let mut deferred_metrics: Vec<String> = Vec::new();
    // Local-equivalent Markdown rendering: on a TTY the body deltas go
    // through the shared streaming renderer (tables, code blocks, math),
    // exactly like the local turn output. Piped output stays raw so
    // captures keep clean bytes.
    let mut markdown = MarkdownStreamRenderer::new_with_tty(tty);
    loop {
        raw.clear();
        if reader.read_line(&mut raw)? == 0 {
            break;
        }
        let line = raw.strip_suffix('\n').unwrap_or(raw.as_str());
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
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
        let payload = data.strip_prefix(' ').unwrap_or(data);
        match event_kind.as_str() {
            "error" => {
                // Best effort: surface the partial answer before failing.
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_body,
                    tty,
                    out,
                );
                flush_deferred_metrics(&mut deferred_metrics, &mut line_open, out);
                let msg = serde_json::from_str::<serde_json::Value>(payload)
                    .ok()
                    .and_then(|v| {
                        v.get("error")
                            .and_then(|e| e.as_str())
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| payload.to_string());
                return Err(format!("turn failed: {msg}").into());
            }
            "done" => {
                // End of turn: emit whatever the renderer still holds.
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_body,
                    tty,
                    out,
                );
                flush_deferred_metrics(&mut deferred_metrics, &mut line_open, out);
                return Ok(());
            }
            "thinking_start" => {
                // A second thinking block in one turn closes the previous
                // summary first (multi-round model calls).
                // Flush first: the previous round's body is complete, and the
                // status line must not glue onto its last row.
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_body,
                    tty,
                    out,
                );
                thinking_active = true;
                thinking_newlines = 0;
                thinking_chars = 0;
                thinking_body.clear();
                if tty {
                    let _ = write!(out, "\x1b[2m◌ thinking…\x1b[0m");
                    let _ = out.flush();
                    line_open = true;
                }
            }
            "thinking" => {
                // A thinking frame arriving after body output (multi-round
                // calls) must not rewrite the rendered answer's last line.
                // Mid-thinking this flush is a no-op.
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                let text = serde_json::from_str::<serde_json::Value>(payload)
                    .ok()
                    .and_then(|v| {
                        v.get("text")
                            .and_then(|t| t.as_str())
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| payload.to_string());
                if !thinking_active {
                    // Tolerate start-less frames (older servers never send
                    // them, but a folded line beats a leaked one).
                    thinking_active = true;
                    thinking_newlines = 0;
                    thinking_chars = 0;
                    thinking_body.clear();
                }
                thinking_newlines += text.matches('\n').count();
                thinking_chars += text.chars().count();
                thinking_body.push_str(&text);
                if tty {
                    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    let tail: String = flat
                        .chars()
                        .rev()
                        .take(60)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    let _ = write!(out, "\r\x1b[K\x1b[2m◌ thinking… {tail}\x1b[0m");
                    let _ = out.flush();
                    line_open = true;
                }
            }
            "thinking_done" => {
                // The buffered body (if any) is complete: discard it and print
                // only the fold summary row.
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_body,
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
                if tty && thinking_active {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_body,
                    tty,
                    out,
                );
                let text = serde_json::from_str::<serde_json::Value>(payload)
                    .ok()
                    .and_then(|v| {
                        v.get("delta")
                            .and_then(|d| d.as_str())
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| payload.to_string());
                if tty {
                    // Same renderer as the local turn output: tables, code
                    // blocks and math render progressively, not as raw source.
                    let _ = markdown.write_chunk_to(out, &text, false);
                } else {
                    let _ = write!(out, "{text}");
                }
                if !text.is_empty() {
                    line_open = !text.ends_with('\n');
                }
                let _ = out.flush();
            }
            "confirm_request" => {
                // Same ordering guarantee as the `_` arm below: the question
                // must start on a fresh row, never glued onto answer deltas.
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
                    &mut thinking_body,
                    tty,
                    out,
                );
                if line_open {
                    let _ = writeln!(out);
                }
                let question = serde_json::from_str::<serde_json::Value>(payload).ok();
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
                line_open = false;
                let _ = out.flush();
            }
            "confirm_done" => {
                // The child's own resolution frame: an answered question already
                // cleared the slot server-side, so there is nothing to render
                // (and it must not fall into the `_` arm's footer).
            }
            _ => {
                // Same ordering guarantee as the `delta` arm: observer footers
                // arriving while a fold is still open must not slip between the
                // answer and its thinking summary.
                if tty {
                    // Flush first so the footer starts after the rendered
                    // body; a completed row means no extra guard newline.
                    let _ = markdown.flush_pending_to(out);
                    if markdown.at_line_start() {
                        line_open = false;
                    }
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
                    &mut thinking_body,
                    tty,
                    out,
                );
                // Metrics summaries must not tear the answer's open line (see
                // `deferred_metrics`): buffer them for the end-of-turn flush.
                // Every other observer row keeps the previous inline behavior.
                if is_deferred_metrics_footer(payload) {
                    deferred_metrics.push(payload.to_string());
                    let _ = out.flush();
                } else {
                    // Body deltas carry no trailing newline, so the first footer
                    // line would otherwise glue onto the answer's last line.
                    if line_open {
                        let _ = writeln!(out);
                    }
                    // The turn header (`print_info` in `request/transport.rs`)
                    // is forwarded as a plain stdout row, so without a marker
                    // the serve-chat scrollback is indistinguishable from a
                    // local run. Annotate only that row; every other observer
                    // row stays byte-identical.
                    let row = if is_turn_header_row(payload) {
                        annotate_remote_turn_header(payload, tty)
                    } else {
                        payload.to_string()
                    };
                    let _ = writeln!(out, "{row}");
                    // The appended newline closes the output line even when the
                    // stdout-derived payload has no line terminator of its own.
                    line_open = false;
                    let _ = out.flush();
                }
            }
        }
    }
    // Stream ended without a `done` event: same end-of-turn flush.
    if tty {
        let _ = markdown.flush_pending_to(out);
    }
    close_thinking_status(
        &mut thinking_active,
        &mut line_open,
        thinking_summary(thinking_newlines, thinking_chars),
        &mut thinking_body,
        tty,
        out,
    );
    flush_deferred_metrics(&mut deferred_metrics, &mut line_open, out);
    Ok(())
}

/// Approximate folded-thinking line count for the summary row.
fn thinking_summary(newlines: usize, chars: usize) -> usize {
    if chars == 0 {
        0
    } else {
        newlines + 1
    }
}

/// Whether a stdout-derived footer row is an end-of-stream metrics summary.
/// These (`↳ cache …`, `↳ speed …`) describe a finished stream, but the turn
/// may keep going; matching on the marker (not the full line) keeps this
/// robust to the child's indent/ANSI styling. Body text never reaches this
/// path (it arrives as `delta` events), so a quoted marker in the answer
/// cannot be misclassified here.
fn is_deferred_metrics_footer(payload: &str) -> bool {
    payload.contains("↳ cache") || payload.contains("↳ speed")
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

/// Print buffered metrics footers at end-of-turn, each starting on a fresh
/// row. Drains the buffer so every path (done/error/truncated stream) emits
/// each row exactly once, in arrival order.
fn flush_deferred_metrics(
    pending: &mut Vec<String>,
    line_open: &mut bool,
    out: &mut dyn std::io::Write,
) {
    for row in pending.drain(..) {
        if *line_open {
            let _ = writeln!(out);
        }
        let _ = writeln!(out, "{row}");
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

/// Close the live thinking status line (if open) and print only the folded
/// summary, mirroring the local `✓ thinking (N lines)` row. The buffered
/// thinking body is discarded, never printed: the local client erases the
/// fold body entirely, and printing even a bounded recap leaves a wall of
/// dimmed rows in the serve-chat scrollback. No-op when no thinking block
/// is open. Takes the body so every close path
/// (done/error/follow-up block/late delta or footer) renders it exactly
/// once.
fn close_thinking_status(
    active: &mut bool,
    line_open: &mut bool,
    lines: usize,
    body: &mut String,
    tty: bool,
    out: &mut dyn std::io::Write,
) {
    if !*active {
        return;
    }
    *active = false;
    if tty {
        let _ = write!(out, "\r\x1b[K");
    }
    // Discard the buffered body unread: the live status line already hinted
    // at progress while the block was open, and the summary below is the
    // only retained row.
    body.clear();
    let unit = if lines == 1 { "line" } else { "lines" };
    let _ = writeln!(out, "{}", dim(&format!("✓ thinking ({lines} {unit})"), tty));
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
    // Short timeout: the shared turn client waits up to 10 minutes, which
    // would pin the render loop on a wedged connection while the child waits.
    let poster = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap_or_else(|_| client.clone());
    Some(ServeConfirmChannel {
        poster,
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
    fn answer(
        &self,
        id: u64,
        token: u64,
        prompt: &str,
        out: &mut dyn std::io::Write,
    ) {
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
    fn post_answer(
        &self,
        id: u64,
        token: u64,
        allow: bool,
        out: &mut dyn std::io::Write,
    ) {
        let url = format!("{}/sessions/{}/confirm", self.base, self.session_id);
        match auth(self.poster.post(url), &self.token)
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

fn post_turn_stream(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    prompt: &str,
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
            let assets_dir =
                SessionStore::new(history_file).session_assets_dir(session_id);
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
    // stdin, so the same gates as a local turn (`git commit` / `git stash`)
    // become approvable instead of failing closed.
    let confirm_channel = serve_confirm_channel(client, base, token, session_id);
    let resp = auth(
        client.post(format!("{base}/sessions/{session_id}/turns/stream")),
        token,
    )
    .json(&serde_json::json!({
        "prompt": prompt,
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
    let _side_note_guard =
        if super::super::stream::side_note_input::side_note_input_enabled() {
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
    // Short timeout: the shared turn client waits up to 10 minutes, which
    // would pin the composer's persist worker on a wedged connection.
    let poster = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap_or_else(|_| client.clone());
    Arc::new(move |text: String| {
        auth(poster.post(url.clone()), &token)
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
    rust_tools::ensure_rustls_provider();
    let cfg = configw::get_all_config();
    // Same precedence as the server and the management commands: explicit
    // `--serve-bind` wins, then `ai.serve.bind`, then the default.
    let from_cfg = cfg.get_opt(AiConfig::SERVE_BIND).unwrap_or_default();
    let bind = super::resolve_serve_bind(&cli.serve_bind, &from_cfg);
    let base = normalize_base(&bind);
    let token = cfg.get_opt(AiConfig::SERVE_TOKEN).unwrap_or_default();
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

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
                "{tool}: cannot reach serve at {base} ({err}); start it first with `a --serve --serve-bind <addr>`"
            )
            .into());
        }
    }
    Ok((client, base, token))
}

/// Print one `GET /sessions` listing as `N. <id>  <title>` rows (newest
/// first) on stdout, so `--serve-sessions` output stays pipeable for scripts
/// and later mobile clients; the interactive client reuses the same format.
fn print_remote_sessions(items: &[RemoteSession]) {
    if items.is_empty() {
        println!("No remote sessions.");
    } else {
        for (i, item) in items.iter().enumerate() {
            println!("{}. {}  {}", i + 1, item.id, remote_session_title(item));
        }
    }
}

/// Non-interactive `--serve-sessions`: print the remote session list and
/// exit, so the session can be picked before opening the interactive client.
pub(in crate::ai) fn run_serve_sessions(
    cli: &ParsedCli,
) -> Result<(), Box<dyn std::error::Error>> {
    let (client, base, token) = serve_connection(cli, "serve-sessions")?;
    let items = fetch_remote_sessions(&client, &base, &token)
        .map_err(|err| format!("[serve-sessions] {err}"))?;
    print_remote_sessions(&items);
    Ok(())
}

/// Run the interactive serve chat: connect to a running serve instance and
/// loop over multiline prompts until `/quit`. Input uses the same editor as
/// the local REPL, so Enter/movement/completion behave identically.
pub(in crate::ai) fn run_serve_chat(
    cli: ParsedCli,
) -> Result<(), Box<dyn std::error::Error>> {
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
    println!("Enter inserts a newline, Esc or Alt+Enter submits; Ctrl+C interrupts the running remote turn (a second Ctrl+C exits); /quit exits, /bg backgrounds the current session and exits, /new starts a new session, /sessions (or /ss, same as the local REPL) lists remote sessions, /fork branches the current session, /close deletes it and exits. /model, /effort, /agent switch the remote turn like the local REPL.");
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
    // Last `/sessions` output: backs `/resume <number|id-prefix>`.
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
        // candidates. Failures keep the previous listing; the prompt is never
        // blocked by this.
        if let Some(items) = refresh_listing_quiet(&client, &base, &token) {
            CommandCompleter::set_serve_session_candidates(
                items
                    .iter()
                    .map(|s| (s.id.clone(), remote_session_title(s)))
                    .collect(),
            );
            editor.set_session_topic(current_remote_title(&items, &session_id));
            listed = items;
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
                    "Enter inserts a newline, Esc or Alt+Enter submits (same as the local REPL).\n/quit - exit\n/new - start a new session\n/sessions (/ss, same as the local REPL) - list remote sessions (newest first)\n/resume <number|id-prefix|id> - continue a listed session (Tab completes after /sessions)\n/fork - branch the current session and switch to it\n/close - delete the current remote session and exit\n/history [N|full|user|assistant|tool|system|grep <kw>|last [N]|replay|rewind <uN|last|grep <kw>>|help] - remote history over HTTP (never a turn)\n/model <selector> [question] - switch remote model (/model list|current|help; question sends immediately)\n/effort <level> - switch remote reasoning effort (minimal|low|medium|high|xhigh|max|off|auto)\n/agent <name> - switch remote agent (/agent list|current|help)\n/bg - exit and bind this terminal to the current remote session\nCtrl+C - interrupt the running remote turn (a second Ctrl+C exits the client)"
                );
                continue;
            }
            "/new" => match create_session(&client, &base, &token) {
                Ok(id) => {
                    session_id = id;
                    editor.set_session_id(session_id.as_str());
                    println!("New session {session_id}.");
                }
                Err(err) => eprintln!("[serve-chat] {err}"),
            },
            // `/ss` matches the local REPL shorthand (`/ss` === `/sessions`
            // list), so the same muscle memory works on both sides.
            "/sessions" | "/ss" | "/ls" => match fetch_remote_sessions(&client, &base, &token) {
                Ok(items) => {
                    print_remote_sessions(&items);
                    // Back `/resume` Tab-completion until the next listing.
                    CommandCompleter::set_serve_session_candidates(
                        items
                            .iter()
                            .map(|s| (s.id.clone(), remote_session_title(s)))
                            .collect(),
                    );
                    listed = items;
                }
                Err(err) => eprintln!("[serve-chat] {err}"),
            },
            "/resume" => match resolve_session_ref(arg, &listed) {
                Ok(id) => {
                    let title = listed.iter().find(|s| s.id == id).map(remote_session_title);
                    session_id = id;
                    editor.set_session_id(session_id.as_str());
                    if restore_serve_session_picks(&client, &base, &token, &session_id, &mut selection)
                    {
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
                let patch = run_serve_agent_command(arg, &mut selection, &agent_manifests, &mut editor);
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
            _ => {
                // A local slash command is never a prompt: the local REPL
                // would swallow it, so forwarding it as a turn only burns a
                // model round-trip at best and hangs the session at worst
                // (`/history rewind` used to block a turn child on stdin).
                // Anything already handled above never reaches this arm.
                if is_local_command_start(&trimmed) {
                    let command = trimmed.split_whitespace().next().unwrap_or(&trimmed);
                    eprintln!(
                        "[serve-chat] {command} is a local command and is not supported over serve-chat; it was not sent. Supported here: /history, /model, /effort, /agent, /skills, /sessions, /resume, /fork, /close, /bg, /new (see /help)."
                    );
                } else if let Err(err) = post_turn_stream(
                    &client,
                    &base,
                    &token,
                    &session_id,
                    &trimmed,
                    &app_config.history_file,
                    &mut selection,
                ) {
                    eprintln!("[serve-chat] {err}");
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;

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

    /// Same canned-response trick, but also captures the request head (request
    /// line + headers) so the interrupt route and its auth header can be
    /// asserted offline.
    fn serve_capture_once(
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
                let mut reader =
                    std::io::BufReader::new(stream.try_clone().expect("clone stream"));
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    head.push_str(&line);
                }
            }
            if let Ok(mut sink) = sink.lock() {
                *sink = head;
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
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
        assert!(
            !out.contains("confirm_request"),
            "raw event leaked: {out}"
        );
    }

    #[test]
    fn sse_malformed_confirm_request_is_ignored() {
        let url = serve_body_once("event: confirm_request\ndata: not-json\n\nevent: done\ndata: \n\n");
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
        let root =
            std::env::temp_dir().join(format!("a-serve-chat-img-{}", uuid::Uuid::new_v4()));
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
        assert_eq!(normalize_serve_effort_arg("low"), Ok(Some("low".to_string())));
        assert_eq!(normalize_serve_effort_arg(" HIGH "), Ok(Some("high".to_string())));
        assert_eq!(normalize_serve_effort_arg("off"), Ok(Some("off".to_string())));
        assert_eq!(normalize_serve_effort_arg("disabled"), Ok(Some("off".to_string())));
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
        let (selector, question) =
            split_serve_model_inline("deepseek   explain this code");
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
        assert_eq!(names, vec!["bytedcli".to_string(), "code-review".to_string()]);
        assert_eq!(question.as_deref(), Some("explain this"));
        // Dedupes keeping input order; a non-skill token stops the split and
        // starts the question.
        let (names, question) =
            split_serve_skills_inline(&known, "bytedcli bytedcli ??? nope");
        assert_eq!(names, vec!["bytedcli".to_string()]);
        assert_eq!(question.as_deref(), Some("??? nope"));
        // No leading skill token: pure question, no selection.
        let (names, question) = split_serve_skills_inline(&known, "hello there");
        assert!(names.is_empty());
        assert_eq!(question.as_deref(), Some("hello there"));
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
            model_tier: None,
            disabled,
            hidden,
            auto_select: false,
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
        let mut editor =
            PromptEditor::new("test", std::path::Path::new("/tmp/a-serve-agent-test-history"));
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
    fn sse_thinking_frames_fold_and_end_turn() {
        // Thinking lifecycle interleaved with the answer: the renderer must
        // accept it and finish on `done` (under test stdout is piped, so the
        // summary row prints plain without ANSI rewrites).
        let url = serve_body_once(
            "event: thinking_start\ndata: \n\nevent: thinking\ndata: {\"text\": \"reasoning here\"}\n\nevent: thinking_done\ndata: \n\nevent: delta\ndata: {\"delta\": \"answer\"}\n\nevent: done\ndata: \n\n",
        );
        let resp = reqwest::blocking::get(url).expect("get canned SSE");
        assert!(render_turn_stream(resp).is_ok());
    }

    #[test]
    fn sse_footer_after_body_starts_on_fresh_line() {
        // The first footer must start on a new line; subsequent footers must
        // not add another newline after the one the renderer already wrote.
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
    fn sse_metrics_footer_does_not_split_body() {
        // Regression net for the serve-chat interleave report: a metrics
        // footer (`↳ speed`) arriving between two body chunks of one sentence
        // must not tear the sentence apart (`读` / speed / `不到。`). Metrics
        // summaries buffer until `done`, so the body stays contiguous and the
        // footer lands after it, like the local client.
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
        assert!(!is_turn_header_row("[model (effort: max) · topic] (remote)"));
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
    fn sse_tool_rows_keep_child_indent_verbatim() {
        // Regression net for the serve-chat indent report: footer/message rows
        // (tool status, long command echoes) must reach the terminal
        // byte-identical to the child bytes the server forwarded: leading
        // two-space indent, inline ANSI, and over-long rows included. The
        // client neither re-indents nor clamps these rows; soft-wrap of a row
        // wider than the terminal is the terminal's own layout, identical to
        // a local run of the same turn. Metrics summaries (`↳ cache` /
        // `↳ speed`) stay byte-identical too, but flush at end-of-turn (after
        // the tool rows) so they can never tear the answer's open line.
        let cache = "  \u{1b}[2m↳ cache · 4.1k/22.0k tokens · 19% hit\u{1b}[0m";
        let speed = "  \u{1b}[2m↳ speed · reasoning 1.6k tok @ 48.4 tok/s\u{1b}[0m";
        let done = "  \u{1b}[32m✓\u{1b}[0m read_file  \u{1b}[2m·\u{1b}[0m target";
        let running = "  \u{1b}[34m●\u{1b}[0m execute_command";
        let long_cmd = format!("  │ $ {}", "cd /data00/x; ".repeat(20));
        assert!(long_cmd.chars().count() > 200);
        let failed = "  \u{1b}[31m✕\u{1b}[0m execute_command";
        let expected =
            format!("ok\n{done}\n{running}\n{long_cmd}\n{failed}\n{cache}\n{speed}\n");
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
        render_turn_stream_to(resp, &mut buf, true).expect("render");
        let out = String::from_utf8(buf).expect("utf8");
        for row in [cache, speed, done, running, long_cmd.as_str(), failed] {
            assert!(row.starts_with("  "), "fixture lost indent: {row:?}");
            assert!(!row.starts_with("   "), "fixture gained indent: {row:?}");
            assert!(
                out.lines().any(|line| line == row),
                "message row mangled or missing on TTY: {row:?}"
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
        assert!(
            out.contains('─'),
            "table must render as a box, got: {out}"
        );
        assert!(
            !out.contains("| ---"),
            "raw separator must not leak, got: {out}"
        );
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
        assert_eq!(buf, b"answer");
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
        let mut body = (1..=100)
            .map(|i| format!("thought line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut active = true;
        let mut line_open = true;
        let mut buf = Vec::new();
        close_thinking_status(&mut active, &mut line_open, 100, &mut body, false, &mut buf);
        let out = String::from_utf8(buf).expect("utf8");
        assert_eq!(out, "✓ thinking (100 lines)\n");
        assert!(body.is_empty(), "body must drain exactly once");
    }

    #[test]
    fn thinking_close_print_keeps_a_short_body_whole() {
        // Even a short body leaves no body rows: only the summary stays.
        let mut body = "a\nb\nc".to_string();
        let mut active = true;
        let mut line_open = true;
        let mut buf = Vec::new();
        close_thinking_status(&mut active, &mut line_open, 3, &mut body, false, &mut buf);
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
        }
    }

    #[test]
    fn parse_remote_sessions_drops_empty_ids() {
        let items = parse_remote_sessions(
            r#"[{"id":"a1","summary":"Fix bug","size_bytes":10,"marked":false},{"id":"  "}]"#,
        )
        .expect("parse");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "a1");
        assert_eq!(items[0].summary.as_deref(), Some("Fix bug"));
        parse_remote_sessions("not json").expect_err("malformed must fail");
        // Older servers without the optional fields still parse.
        let legacy = parse_remote_sessions(r#"[{"id":"old"}]"#).expect("legacy");
        assert!(legacy[0].summary.is_none());
        assert!(!legacy[0].marked);
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
        let url = serve_body_once(
            r#"[{"id":"s1","summary":"Hello","size_bytes":7,"marked":true}]"#,
        );
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
            },
            RemoteSession {
                id: "s2".to_string(),
                summary: None,
                first_user_prompt: Some("question\nbody".to_string()),
                marked: false,
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
        save_serve_chat_binding_for_key(&root, "terminal:term-1", "http://127.0.0.1:8080", "sess-1")
            .expect("save");
        let loaded =
            load_serve_chat_binding_for_key(&root, "terminal:term-1").expect("load");
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
        assert_eq!(serve_sigint_action(true, true), ServeSigintAction::ExitLocal);
        assert_eq!(
            serve_sigint_action(false, false),
            ServeSigintAction::ExitLocal
        );
        assert_eq!(serve_sigint_action(false, true), ServeSigintAction::ExitLocal);
    }

    /// The interrupt must address the session-scoped route with the bearer
    /// header, and an "already finished" answer must surface as `false`
    /// instead of an error.
    #[test]
    fn interrupt_request_targets_the_session_route() {
        rust_tools::ensure_rustls_provider();
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
}
