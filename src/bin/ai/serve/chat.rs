//! Interactive client for a running serve instance.
//!
//! Local testing REPL that mirrors the single-machine interactive input: the
//! same multiline `PromptEditor` (Enter inserts a newline, Esc or Alt+Enter
//! submits) and progressive output rendering. Each submitted prompt is POSTed
//! to `POST /sessions/{id}/turns/stream`; every SSE event is rendered as it
//! arrives instead of waiting for the whole turn.

use std::time::Duration;

use serde::Deserialize;

use crate::commonw::configw;

use super::super::{cli::ParsedCli, config_schema::AiConfig, prompt::PromptEditor};
use super::DEFAULT_BIND;

#[derive(Debug, Deserialize)]
struct CreatedSession {
    id: String,
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

fn post_turn_stream(
    client: &reqwest::blocking::Client,
    base: &str,
    token: &str,
    session_id: &str,
    prompt: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let resp = auth(
        client.post(format!("{base}/sessions/{session_id}/turns/stream")),
        token,
    )
    .json(&serde_json::json!({ "prompt": prompt }))
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
    println!("Enter inserts a newline, Esc or Alt+Enter submits; /quit exits, /new starts a new session.");
    let mut editor = PromptEditor::new(&session_id, &app_config.history_file);
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
        match trimmed.as_str() {
            "/quit" | "/exit" | ":q" => break,
            "/help" | "/h" => {
                println!(
                    "Enter inserts a newline, Esc or Alt+Enter submits (same as the local REPL).\n/quit - exit\n/new - start a new session"
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
            _ => {
                if let Err(err) =
                    post_turn_stream(&client, &base, &token, &session_id, &trimmed)
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
}
