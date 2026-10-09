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
    agents, cli::ParsedCli, config_schema::AiConfig, driver::input::inline_image_filenames,
    history::{current_terminal_key, SessionStore},
    model_names, models,
    prompt::{completion::CommandCompleter, PromptEditor},
    stream::MarkdownStreamRenderer,
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
/// the explicit tier.
#[derive(Debug, Clone)]
struct ServeTurnSelection {
    model: String,
    agent: String,
    reasoning_effort: Option<String>,
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
) -> Option<String> {
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
        return None;
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
        return None;
    }
    if arg.eq_ignore_ascii_case("help") || arg.eq_ignore_ascii_case("h") {
        println!("(remote) /model <selector> [question] - switch remote model");
        println!("(remote) /model list|current|help - inspect remote models");
        println!("(remote) /model effort <level> - same as /effort <level>");
        return None;
    }
    let (head, head_rest) = match arg.split_once(char::is_whitespace) {
        Some((h, r)) => (h, r.trim()),
        None => (arg, ""),
    };
    if head.eq_ignore_ascii_case("effort") {
        if head_rest.is_empty() {
            println!("(remote) effort: {}", selection.effective_effort_label());
            return None;
        }
        run_serve_effort_command(head_rest, selection, editor);
        return None;
    }
    let (selector, question) = split_serve_model_inline(arg);
    match model_names::find_by_identifier(&selector) {
        Some(def) => {
            let handle = model_names::model_handle(def);
            selection.model = handle.clone();
            selection.apply_header(editor);
            println!("(remote) switched model to {}", models::model_display_label(&handle));
            question
        }
        None => {
            println!("(remote) unknown model {selector:?}; see /model list");
            None
        }
    }
}

/// Handle `/effort` in serve-chat (always consumes the line; the switch
/// applies to the next turn).
fn run_serve_effort_command(
    arg: &str,
    selection: &mut ServeTurnSelection,
    editor: &mut PromptEditor,
) -> Option<String> {
    let arg = arg.trim();
    if arg.is_empty() {
        println!("(remote) effort: {}", selection.effective_effort_label());
        return None;
    }
    match normalize_serve_effort_arg(arg) {
        Ok(override_value) => {
            selection.reasoning_effort = override_value;
            selection.apply_header(editor);
            println!("(remote) effort: {}", selection.effective_effort_label());
        }
        Err(err) => println!("(remote) {err}"),
    }
    None
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
) {
    let arg = arg.trim();
    if arg.is_empty() || arg.eq_ignore_ascii_case("list") || arg.eq_ignore_ascii_case("ls") {
        let primary = agents::get_primary_agents(agent_manifests);
        for agent in &primary {
            let marker = if agent.name == selection.agent { "*" } else { " " };
            println!("{marker} {} - {}", agent.name, agent.description);
        }
        return;
    }
    if arg.eq_ignore_ascii_case("current") || arg.eq_ignore_ascii_case("cur") {
        println!("(remote) current agent: {}", selection.agent);
        return;
    }
    if arg.eq_ignore_ascii_case("help") || arg.eq_ignore_ascii_case("h") {
        println!("(remote) /agent <name> - switch remote agent (primary only)");
        return;
    }
    if arg.eq_ignore_ascii_case("reload") {
        println!("(remote) agent reload is server-side; reconnect to refresh");
        return;
    }
    match agents::find_agent_by_name(agent_manifests, arg) {
        Some(manifest) => {
            if !manifest.is_primary() {
                println!("(remote) agent {:?} is not switchable", manifest.name);
                return;
            }
            if manifest.disabled {
                println!("(remote) agent {:?} is disabled", manifest.name);
                return;
            }
            selection.agent = manifest.name.clone();
            if let Some(model) = manifest.model.as_deref().filter(|m| !m.trim().is_empty()) {
                selection.model = models::determine_model(model);
            }
            selection.apply_header(editor);
            println!("(remote) switched agent to {}", manifest.name);
        }
        None => println!("(remote) unknown agent {arg:?}; see /agent list"),
    }
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

/// Render one SSE turn stream progressively: message lines print as they
/// arrive, `delta` events (live body chunks) print without a trailing
/// newline, thinking frames collapse into one live status line (matching the
/// local fold: `✓ thinking (N lines)` when closed), error events abort with
/// the server detail, the done event (or the end of the body) ends the turn.
fn render_turn_stream(
    resp: reqwest::blocking::Response,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::IsTerminal;
    let tty = std::io::stdout().is_terminal();
    render_turn_stream_to(resp, &mut std::io::stdout(), tty)
}

/// Same as [`render_turn_stream`], but writes to `out` so tests can assert
/// the exact bytes (line breaks between body, footers and fold summaries).
fn render_turn_stream_to(
    resp: reqwest::blocking::Response,
    out: &mut dyn std::io::Write,
    tty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::BufRead;
    // `blocking::Response` implements `Read`, so lines are parsed (and
    // printed) as soon as the server flushes them, not after the whole turn.
    let mut reader = std::io::BufReader::new(resp);
    let mut event_kind = String::new();
    let mut raw = String::new();
    // Folded-thinking state: the body text stays buffered in the status
    // line, so it never mixes with the answer. Line count is approximate
    // (chunks split mid-line); it only feeds the summary row.
    let mut thinking_active = false;
    let mut thinking_newlines = 0usize;
    let mut thinking_chars = 0usize;
    // Whether the cursor sits mid-line (answer delta without a trailing
    // newline, or a live thinking status). Footer/message lines must start
    // on a fresh row instead of gluing onto the answer's last line.
    let mut line_open = false;
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
                    tty,
                    out,
                );
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
                    tty,
                    out,
                );
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
                    tty,
                    out,
                );
                thinking_active = true;
                thinking_newlines = 0;
                thinking_chars = 0;
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
                }
                thinking_newlines += text.matches('\n').count();
                thinking_chars += text.chars().count();
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
                // The buffered body (if any) is complete: render it before
                // the fold summary row.
                if tty {
                    let _ = markdown.flush_pending_to(out);
                }
                close_thinking_status(
                    &mut thinking_active,
                    &mut line_open,
                    thinking_summary(thinking_newlines, thinking_chars),
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
                    tty,
                    out,
                );
                // Body deltas carry no trailing newline, so the first footer
                // line would otherwise glue onto the answer's last line.
                if line_open {
                    let _ = writeln!(out);
                }
                let _ = writeln!(out, "{payload}");
                // The appended newline closes the output line even when the
                // stdout-derived payload has no line terminator of its own.
                line_open = false;
                let _ = out.flush();
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
        tty,
        out,
    );
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

/// Dim helper: plain text when piped, so non-terminal captures stay clean.
fn dim(text: &str, tty: bool) -> String {
    if tty {
        format!("\x1b[2m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

/// Close the live thinking status line (if open) and print the folded
/// summary, mirroring the local `✓ thinking (N lines)` row. No-op when no
/// thinking block is open.
fn close_thinking_status(
    active: &mut bool,
    line_open: &mut bool,
    lines: usize,
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
    selection: &ServeTurnSelection,
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
    }))
    .send()?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().unwrap_or_default();
        return Err(server_error_message(status, &text).into());
    }
    let outcome = render_turn_stream(resp);
    if REMOTE_TURN_INTERRUPT_SENT.swap(false, Ordering::SeqCst) {
        // Leading newline: `delta` events print without a trailing newline, so
        // the note would otherwise land mid-line.
        println!("\n[serve-chat] interrupt sent; remote turn stopped.");
    }
    outcome
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
                    "Enter inserts a newline, Esc or Alt+Enter submits (same as the local REPL).\n/quit - exit\n/new - start a new session\n/sessions (/ss, same as the local REPL) - list remote sessions (newest first)\n/resume <number|id-prefix|id> - continue a listed session (Tab completes after /sessions)\n/fork - branch the current session and switch to it\n/close - delete the current remote session and exit\n/model <selector> [question] - switch remote model (/model list|current|help; question sends immediately)\n/effort <level> - switch remote reasoning effort (minimal|low|medium|high|xhigh|max|off|auto)\n/agent <name> - switch remote agent (/agent list|current|help)\n/bg - exit and bind this terminal to the current remote session\nCtrl+C - interrupt the running remote turn (a second Ctrl+C exits the client)"
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
                if let Some(question) =
                    run_serve_model_command(arg, &mut selection, &mut editor)
                {
                    if let Err(err) = post_turn_stream(
                        &client,
                        &base,
                        &token,
                        &session_id,
                        &question,
                        &app_config.history_file,
                        &selection,
                    ) {
                        eprintln!("[serve-chat] {err}");
                    }
                }
            }
            "/effort" | ":effort" => {
                run_serve_effort_command(arg, &mut selection, &mut editor);
            }
            "/agent" | ":agent" => {
                run_serve_agent_command(arg, &mut selection, &agent_manifests, &mut editor);
            }
            _ => {
                if let Err(err) =
                    post_turn_stream(
                        &client,
                        &base,
                        &token,
                        &session_id,
                        &trimmed,
                        &app_config.history_file,
                        &selection,
                    )
                {
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
    fn sse_tool_rows_keep_child_indent_verbatim() {
        // Regression net for the serve-chat indent report: footer/message rows
        // (tool status, cache/speed metrics, long command echoes) must reach
        // the terminal byte-identical to the child bytes the server forwarded:
        // leading two-space indent, inline ANSI, and over-long rows included.
        // The client neither re-indents nor clamps these rows; soft-wrap of a
        // row wider than the terminal is the terminal's own layout, identical
        // to a local run of the same turn.
        let cache = "  \u{1b}[2m↳ cache · 4.1k/22.0k tokens · 19% hit\u{1b}[0m";
        let speed = "  \u{1b}[2m↳ speed · reasoning 1.6k tok @ 48.4 tok/s\u{1b}[0m";
        let done = "  \u{1b}[32m✓\u{1b}[0m read_file  \u{1b}[2m·\u{1b}[0m target";
        let running = "  \u{1b}[34m●\u{1b}[0m execute_command";
        let long_cmd = format!("  │ $ {}", "cd /data00/x; ".repeat(20));
        assert!(long_cmd.chars().count() > 200);
        let failed = "  \u{1b}[31m✕\u{1b}[0m execute_command";
        let expected =
            format!("ok\n{cache}\n{speed}\n{done}\n{running}\n{long_cmd}\n{failed}\n");
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
        // for the same round) must not print a second summary row.
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
