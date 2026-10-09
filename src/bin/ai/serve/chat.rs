//! Interactive client for a running serve instance.
//!
//! Local testing REPL that mirrors the single-machine interactive input: the
//! same multiline `PromptEditor` (Enter inserts a newline, Esc or Alt+Enter
//! submits) and progressive output rendering. Each submitted prompt is POSTed
//! to `POST /sessions/{id}/turns/stream`; every SSE event is rendered as it
//! arrives instead of waiting for the whole turn.

use std::{path::Path, time::Duration};

use serde::Deserialize;

use crate::commonw::configw;

use super::super::{
    cli::ParsedCli, config_schema::AiConfig, driver::input::inline_image_filenames,
    history::SessionStore, prompt::PromptEditor,
};
use super::DEFAULT_BIND;

#[derive(Debug, Deserialize)]
struct CreatedSession {
    id: String,
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

/// Max rows `GET /sessions` asks for: enough to find a recent session after
/// a client restart, small enough to print as one screen.
const SESSION_LIST_LIMIT: usize = 20;

/// Fetch the newest remote sessions (`GET /sessions?limit=`). Thin network
/// glue; parsing and display live in the pure helpers below so tests stay
/// offline except for this one loopback round-trip.
fn fetch_remote_sessions(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
) -> Result<Vec<RemoteSession>, String> {
    let resp = auth(
        client.get(format!("{base}/sessions?limit={SESSION_LIST_LIMIT}")),
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
/// never listed (older than the list limit, or pasted from elsewhere) passes
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
                let _ = write!(out, "{text}");
                if !text.is_empty() {
                    line_open = !text.ends_with('\n');
                }
                let _ = out.flush();
            }
            _ => {
                // Same ordering guarantee as the `delta` arm: observer footers
                // arriving while a fold is still open must not slip between the
                // answer and its thinking summary.
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

fn post_turn_stream(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    prompt: &str,
    history_file: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let images = collect_image_uploads(prompt, history_file, session_id);
    let resp = auth(
        client.post(format!("{base}/sessions/{session_id}/turns/stream")),
        token,
    )
    .json(&serde_json::json!({ "prompt": prompt, "images": images }))
    .send()?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().unwrap_or_default();
        return Err(server_error_message(status, &text).into());
    }
    render_turn_stream(resp)
}

/// Run the interactive serve chat: connect to a running serve instance and
/// loop over multiline prompts until `/quit`. Input uses the same editor as
/// the local REPL, so Enter/movement/completion behave identically.
pub(in crate::ai) fn run_serve_chat(
    cli: ParsedCli,
) -> Result<(), Box<dyn std::error::Error>> {
    rust_tools::ensure_rustls_provider();
    let cfg = configw::get_all_config();
    let bind = if !cli.serve_bind.trim().is_empty() {
        cli.serve_bind.trim().to_string()
    } else {
        let from_cfg = cfg.get_opt(AiConfig::SERVE_BIND).unwrap_or_default();
        if from_cfg.trim().is_empty() {
            DEFAULT_BIND.to_string()
        } else {
            from_cfg
        }
    };
    let base = normalize_base(&bind);
    let token = cfg.get_opt(AiConfig::SERVE_TOKEN).unwrap_or_default();
    let app_config = super::super::config::load_config()?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    // Fail fast with a copy-pasteable hint when the server is not up.
    let health = client.get(format!("{base}/healthz")).send();
    match health {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => {
            return Err(format!(
                "serve-chat: serve at {base} answered {}; is `a --serve` running there?",
                resp.status()
            )
            .into());
        }
        Err(err) => {
            return Err(format!(
                "serve-chat: cannot reach serve at {base} ({err}); start it first with `a --serve --serve-bind <addr>`"
            )
            .into());
        }
    }

    let mut session_id = match cli.session {
        Some(id) if !id.trim().is_empty() => id.trim().to_string(),
        _ => create_session(&client, &base, &token)?,
    };

    println!("Connected to {base}, session {session_id}.");
    println!("Enter inserts a newline, Esc or Alt+Enter submits; /quit exits, /new starts a new session, /sessions (or /ss, same as the local REPL) lists remote sessions, /fork branches the current session, /close deletes it and exits.");
    let mut editor = PromptEditor::new(&session_id, &app_config.history_file);
    // Last `/sessions` output: backs `/resume <number|id-prefix>`.
    let mut listed: Vec<RemoteSession> = Vec::new();
    loop {
        let input = match editor.read_multi_line() {
            Ok(Some(text)) => text,
            Ok(None) => break,
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
                    "Enter inserts a newline, Esc or Alt+Enter submits (same as the local REPL).\n/quit - exit\n/new - start a new session\n/sessions (/ss, same as the local REPL) - list remote sessions (newest first)\n/resume <number|id-prefix|id> - continue a listed session\n/fork - branch the current session and switch to it\n/close - delete the current remote session and exit"
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
                    if items.is_empty() {
                        println!("No remote sessions.");
                    } else {
                        for (i, item) in items.iter().enumerate() {
                            println!("{}. {}  {}", i + 1, item.id, remote_session_title(item));
                        }
                    }
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
            // Local `/close` deletes the session and exits; the suspended-
            // binding cleanup it also does is local-terminal state with no
            // remote equivalent, so DELETE + break is the full parity here.
            "/close" => match delete_remote_session(&client, &base, &token, &session_id) {
                Ok(_) => {
                    println!("Closed remote session {session_id}.");
                    break;
                }
                Err(err) => eprintln!("[serve-chat] {err}"),
            },
            _ => {
                if let Err(err) =
                    post_turn_stream(
                        &client,
                        &base,
                        &token,
                        &session_id,
                        &trimmed,
                        &app_config.history_file,
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
}
