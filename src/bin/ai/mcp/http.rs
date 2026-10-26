//! HTTP transports for MCP servers.
//!
//! Two wire shapes are supported:
//!
//! * Streamable HTTP (MCP 2025-03-26): every JSON-RPC message is POSTed to the
//!   configured URL. The server answers with either one JSON document or an SSE
//!   stream carrying the response. `Mcp-Session-Id` is captured and replayed.
//! * Legacy HTTP+SSE (MCP 2024-11-05): a long-lived GET stream first emits an
//!   `endpoint` event; JSON-RPC messages are POSTed to that endpoint and the
//!   responses arrive on the same GET stream.
//!
//! The rest of the client is synchronous, so this module uses `reqwest::blocking`
//! and bounded channel waits. The connection is only created from blocking-safe
//! call sites (MCP init runs in `spawn_blocking`; tool execution uses
//! `block_in_place`).

use std::{
    io::{BufRead, BufReader},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::{Duration, Instant},
};

use reqwest::{
    Url,
    blocking::{Client, Response},
    header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use rust_tools::cw::SkipMap;
use serde_json::Value;

use crate::ai::types::{McpServerConfig, McpTransportKind};

use super::{
    io::{tool_cancel_error, tool_cancel_requested},
    jsonrpc::{InboundJsonRpc, classify_inbound_value, unsupported_server_request_payload},
};

const ACCEPT_JSON_OR_SSE: &str = "application/json, text/event-stream";
const ACCEPT_SSE: &str = "text/event-stream";
const STREAMABLE_PROTOCOL_VERSION: &str = "2025-03-26";
const LEGACY_PROTOCOL_VERSION: &str = "2024-11-05";
const CANCEL_POLL_MS: u64 = 100;
const ERROR_BODY_LIMIT: usize = 400;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpMode {
    Streamable,
    LegacySse,
}

/// HTTP transport state kept inside a `McpServerConnection`. The legacy SSE
/// receiver is `Option` because streamable HTTP has no long-lived stream.
pub(in crate::ai) struct HttpTransport {
    mode: HttpMode,
    client: Client,
    url: Url,
    post_url: Url,
    headers: HeaderMap,
    session_id: Option<String>,
    legacy: Option<LegacySse>,
}

struct LegacySse {
    rx: Receiver<ReaderEvent>,
    stop: Arc<AtomicBool>,
}

enum SseFrame {
    Endpoint(String),
    Message(String),
}

enum ReaderEvent {
    Endpoint(Url),
    Message(Value),
    Error(String),
}

impl HttpTransport {
    pub(in crate::ai) fn connect(config: &McpServerConfig) -> Result<Self, String> {
        let kind = config.transport_kind()?;
        if kind == McpTransportKind::Stdio {
            return Err("HTTP transport selected for a stdio MCP server".to_string());
        }

        let url = Url::parse(config.url.trim())
            .map_err(|e| format!("Invalid MCP HTTP URL '{}': {}", config.url, e))?;
        if url.scheme() != "http" && url.scheme() != "https" {
            return Err(format!(
                "MCP HTTP URL must use http or https, got '{}'",
                url.scheme()
            ));
        }
        let headers = build_headers(&config.headers)?;
        let client = Client::builder()
            .user_agent(concat!("rust-tools-ai/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| format!("Failed to create MCP HTTP client: {}", e))?;
        let timeout_ms = config.request_timeout_ms.max(100);

        match kind {
            McpTransportKind::Http => Ok(Self {
                mode: HttpMode::Streamable,
                client,
                post_url: url.clone(),
                url,
                headers,
                session_id: None,
                legacy: None,
            }),
            McpTransportKind::Sse => {
                let (legacy, endpoint) =
                    LegacySse::connect(client.clone(), url.clone(), headers.clone(), timeout_ms)?;
                Ok(Self {
                    mode: HttpMode::LegacySse,
                    client,
                    url,
                    post_url: endpoint,
                    headers,
                    session_id: None,
                    legacy: Some(legacy),
                })
            }
            McpTransportKind::Stdio => unreachable!("stdio was rejected above"),
        }
    }

    pub(in crate::ai) fn protocol_version(&self) -> &'static str {
        match self.mode {
            HttpMode::Streamable => STREAMABLE_PROTOCOL_VERSION,
            HttpMode::LegacySse => LEGACY_PROTOCOL_VERSION,
        }
    }

    pub(in crate::ai) fn request(
        &mut self,
        id: u64,
        body: &str,
        timeout_ms: u64,
    ) -> Result<Value, String> {
        match self.mode {
            HttpMode::Streamable => self.request_streamable(id, body, timeout_ms),
            HttpMode::LegacySse => self.request_legacy(id, body, timeout_ms),
        }
    }

    pub(in crate::ai) fn notify(&mut self, body: &str, timeout_ms: u64) -> Result<(), String> {
        let response = self
            .post(body, timeout_ms)
            .map_err(|e| format!("MCP HTTP notification failed: {}", e))?;
        self.capture_session_id(&response);
        ensure_success(response)
    }

    pub(in crate::ai) fn shutdown(&mut self) {
        if let Some(legacy) = &self.legacy {
            legacy.stop.store(true, Ordering::Relaxed);
        }
        if self.mode != HttpMode::Streamable {
            return;
        }
        let Some(session_id) = self.session_id.take() else {
            return;
        };
        let mut headers = self.headers.clone();
        if let Ok(value) = HeaderValue::from_str(&session_id) {
            headers.insert("mcp-session-id", value);
        }
        // Best-effort session termination. A failure here must not mask the
        // original transport error or delay shutdown for long.
        let _ = self
            .client
            .delete(self.url.clone())
            .headers(headers)
            .timeout(Duration::from_millis(1000))
            .send();
    }

    fn request_streamable(
        &mut self,
        id: u64,
        body: &str,
        timeout_ms: u64,
    ) -> Result<Value, String> {
        let response = self.post(body, timeout_ms)?;
        self.capture_session_id(&response);
        if !response.status().is_success() {
            return Err(http_error(response));
        }
        if response.status() == reqwest::StatusCode::NO_CONTENT {
            return Err("MCP HTTP server returned no response body".to_string());
        }

        if is_event_stream(&response) {
            return self.await_streamable_sse(response, id, timeout_ms);
        }

        let text = response
            .text()
            .map_err(|e| format!("MCP HTTP response read failed: {}", e))?;
        self.parse_json_response(&text, id, timeout_ms)
    }

    fn request_legacy(&mut self, id: u64, body: &str, timeout_ms: u64) -> Result<Value, String> {
        self.post_no_wait(body, timeout_ms)?;
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let rx = &self
            .legacy
            .as_ref()
            .ok_or("MCP SSE transport is not connected")?
            .rx;

        loop {
            if tool_cancel_requested() {
                return Err(tool_cancel_error());
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(format!("MCP response timeout after {} ms", timeout_ms));
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(Duration::from_millis(CANCEL_POLL_MS));
            match rx.recv_timeout(wait) {
                Ok(ReaderEvent::Message(value)) => {
                    if let Some(result) = self.handle_inbound(value, id, timeout_ms)? {
                        return Ok(result);
                    }
                }
                Ok(ReaderEvent::Error(err)) => return Err(err),
                Ok(ReaderEvent::Endpoint(_)) => {}
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("MCP SSE stream closed unexpectedly".to_string());
                }
            }
        }
    }

    fn await_streamable_sse(
        &self,
        response: Response,
        id: u64,
        timeout_ms: u64,
    ) -> Result<Value, String> {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_reader = Arc::clone(&stop);
        thread::spawn(move || {
            read_streamable_sse(response, tx, stop_reader);
        });

        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let result = self.await_response(&rx, id, deadline, timeout_ms);
        stop.store(true, Ordering::Relaxed);
        result
    }

    fn await_response(
        &self,
        rx: &Receiver<ReaderEvent>,
        id: u64,
        deadline: Instant,
        timeout_ms: u64,
    ) -> Result<Value, String> {
        loop {
            if tool_cancel_requested() {
                return Err(tool_cancel_error());
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(format!("MCP response timeout after {} ms", timeout_ms));
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(Duration::from_millis(CANCEL_POLL_MS));
            match rx.recv_timeout(wait) {
                Ok(ReaderEvent::Message(value)) => {
                    if let Some(result) = self.handle_inbound(value, id, timeout_ms)? {
                        return Ok(result);
                    }
                }
                Ok(ReaderEvent::Error(err)) => return Err(err),
                Ok(ReaderEvent::Endpoint(_)) => {}
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("MCP SSE stream closed unexpectedly".to_string());
                }
            }
        }
    }

    fn parse_json_response(
        &self,
        text: &str,
        expected_id: u64,
        timeout_ms: u64,
    ) -> Result<Value, String> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| format!("Failed to parse MCP HTTP response: {}", e))?;
        let messages = match value {
            Value::Array(items) => items,
            other => vec![other],
        };
        for message in messages {
            if let Some(result) = self.handle_inbound(message, expected_id, timeout_ms)? {
                return Ok(result);
            }
        }
        Err("MCP HTTP response did not contain a matching JSON-RPC response".to_string())
    }

    fn handle_inbound(
        &self,
        value: Value,
        expected_id: u64,
        timeout_ms: u64,
    ) -> Result<Option<Value>, String> {
        match classify_inbound_value(value)? {
            InboundJsonRpc::Notification { .. } => Ok(None),
            InboundJsonRpc::Request { id, method } => {
                self.reject_server_request(id, &method, timeout_ms)?;
                Ok(None)
            }
            InboundJsonRpc::Response(response) => match response.id {
                Some(id) if id == expected_id => {
                    if let Some(error) = response.error {
                        Err(format!("MCP error {}: {}", error.code, error.message))
                    } else {
                        Ok(Some(response.result.unwrap_or(Value::Null)))
                    }
                }
                Some(id) => Err(format!(
                    "MCP response ID mismatch: expected {}, got {}",
                    expected_id, id
                )),
                // Some servers emit a spurious initialized acknowledgment with
                // a null id; it must not consume the pending request slot.
                None => Ok(None),
            },
        }
    }

    fn reject_server_request(
        &self,
        id: Value,
        method: &str,
        timeout_ms: u64,
    ) -> Result<(), String> {
        let payload = unsupported_server_request_payload(id, method);
        let body = serde_json::to_string(&payload)
            .map_err(|e| format!("Failed to serialize MCP server-request rejection: {}", e))?;
        self.post_no_wait(&body, timeout_ms)
            .map_err(|e| format!("Failed to reject MCP server request: {}", e))
    }

    fn post(&self, body: &str, timeout_ms: u64) -> Result<Response, String> {
        self.client
            .post(self.post_url.clone())
            .timeout(Duration::from_millis(timeout_ms))
            .headers(self.request_headers())
            .body(body.to_string())
            .send()
            .map_err(|e| format!("MCP HTTP request failed: {}", e))
    }

    fn post_no_wait(&self, body: &str, timeout_ms: u64) -> Result<(), String> {
        let response = self.post(body, timeout_ms)?;
        ensure_success(response)
    }

    fn request_headers(&self) -> HeaderMap {
        let mut headers = self.headers.clone();
        headers.insert(ACCEPT, HeaderValue::from_static(ACCEPT_JSON_OR_SSE));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if self.mode == HttpMode::Streamable {
            headers.insert(
                HeaderName::from_static("mcp-protocol-version"),
                HeaderValue::from_static(STREAMABLE_PROTOCOL_VERSION),
            );
        }
        if let Some(session_id) = &self.session_id
            && let Ok(value) = HeaderValue::from_str(session_id)
        {
            headers.insert("mcp-session-id", value);
        }
        headers
    }

    fn capture_session_id(&mut self, response: &Response) {
        let Some(value) = response.headers().get("mcp-session-id") else {
            return;
        };
        let Ok(value) = value.to_str() else {
            return;
        };
        if !value.trim().is_empty() {
            self.session_id = Some(value.to_string());
        }
    }
}

impl LegacySse {
    fn connect(
        client: Client,
        url: Url,
        headers: HeaderMap,
        timeout_ms: u64,
    ) -> Result<(Self, Url), String> {
        let mut request_headers = headers.clone();
        request_headers.insert(ACCEPT, HeaderValue::from_static(ACCEPT_SSE));
        let response = client
            .get(url.clone())
            .headers(request_headers)
            .send()
            .map_err(|e| format!("MCP SSE connection failed: {}", e))?;
        if !response.status().is_success() {
            return Err(http_error(response));
        }
        if !is_event_stream(&response) {
            let content_type = content_type(&response);
            return Err(format!(
                "MCP SSE endpoint returned unexpected content type: {}",
                content_type
            ));
        }

        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_reader = Arc::clone(&stop);
        let endpoint_slot = Arc::new(Mutex::new(None));
        let endpoint_reader = Arc::clone(&endpoint_slot);
        thread::spawn(move || {
            read_legacy_sse(
                response,
                tx,
                stop_reader,
                client,
                headers,
                endpoint_reader,
                url,
                timeout_ms,
            );
        });

        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            let now = Instant::now();
            if now >= deadline {
                stop.store(true, Ordering::Relaxed);
                return Err("MCP SSE endpoint event timeout".to_string());
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(Duration::from_millis(CANCEL_POLL_MS));
            match rx.recv_timeout(wait) {
                Ok(ReaderEvent::Endpoint(endpoint)) => {
                    *endpoint_slot.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(endpoint.clone());
                    return Ok((Self { rx, stop }, endpoint));
                }
                Ok(ReaderEvent::Error(err)) => return Err(err),
                Ok(ReaderEvent::Message(_)) => {}
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("MCP SSE stream closed before endpoint event".to_string());
                }
            }
        }
    }
}

fn read_legacy_sse(
    response: Response,
    tx: Sender<ReaderEvent>,
    stop: Arc<AtomicBool>,
    client: Client,
    headers: HeaderMap,
    endpoint_slot: Arc<Mutex<Option<Url>>>,
    base_url: Url,
    timeout_ms: u64,
) {
    let mut reader = BufReader::new(response);
    let mut parser = SseParser::default();
    let mut line = String::new();
    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                let _ = tx.send(ReaderEvent::Error(
                    "MCP SSE stream closed unexpectedly".to_string(),
                ));
                return;
            }
            Ok(_) => {}
            Err(err) => {
                let _ = tx.send(ReaderEvent::Error(format!("MCP SSE read failed: {}", err)));
                return;
            }
        }
        let Some(frame) = parser.feed_line(&line) else {
            continue;
        };
        match frame {
            SseFrame::Endpoint(data) => match base_url.join(&data) {
                Ok(endpoint) => {
                    *endpoint_slot.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(endpoint.clone());
                    if tx.send(ReaderEvent::Endpoint(endpoint)).is_err() {
                        return;
                    }
                }
                Err(err) => {
                    let _ = tx.send(ReaderEvent::Error(format!(
                        "Invalid MCP SSE endpoint '{}': {}",
                        data, err
                    )));
                    return;
                }
            },
            SseFrame::Message(data) => {
                let value: Value = match serde_json::from_str(&data) {
                    Ok(value) => value,
                    Err(err) => {
                        let _ = tx.send(ReaderEvent::Error(format!(
                            "Failed to parse MCP SSE message: {}",
                            err
                        )));
                        continue;
                    }
                };
                match classify_inbound_value(value.clone()) {
                    Ok(InboundJsonRpc::Response(_)) => {
                        if tx.send(ReaderEvent::Message(value)).is_err() {
                            return;
                        }
                    }
                    Ok(InboundJsonRpc::Request { id, method }) => {
                        let endpoint = endpoint_slot
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .clone();
                        if let Some(endpoint) = endpoint {
                            let payload = serde_json::to_string(
                                &unsupported_server_request_payload(id, &method),
                            )
                            .unwrap_or_default();
                            let _ = post_json(&client, &endpoint, &headers, &payload, timeout_ms);
                        }
                    }
                    Ok(InboundJsonRpc::Notification { .. }) => {}
                    Err(err) => {
                        let _ = tx.send(ReaderEvent::Error(err));
                    }
                }
            }
        }
    }
}

fn read_streamable_sse(response: Response, tx: Sender<ReaderEvent>, stop: Arc<AtomicBool>) {
    let mut reader = BufReader::new(response);
    let mut parser = SseParser::default();
    let mut line = String::new();
    let mut sent_response = false;
    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(err) => {
                let _ = tx.send(ReaderEvent::Error(format!("MCP SSE read failed: {}", err)));
                return;
            }
        }
        let Some(frame) = parser.feed_line(&line) else {
            continue;
        };
        let SseFrame::Message(data) = frame else {
            continue;
        };
        let value: Value = match serde_json::from_str(&data) {
            Ok(value) => value,
            Err(err) => {
                let _ = tx.send(ReaderEvent::Error(format!(
                    "Failed to parse MCP SSE message: {}",
                    err
                )));
                return;
            }
        };
        let is_response =
            value.get("method").is_none() && value.get("id").and_then(Value::as_u64).is_some();
        if tx.send(ReaderEvent::Message(value)).is_err() {
            return;
        }
        if is_response {
            sent_response = true;
            break;
        }
    }
    if !sent_response && !stop.load(Ordering::Relaxed) {
        let _ = tx.send(ReaderEvent::Error(
            "MCP SSE stream closed before a response arrived".to_string(),
        ));
    }
}

fn build_headers(raw: &SkipMap<String, String>) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();
    for (name, value) in raw {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| format!("Invalid MCP HTTP header name '{}': {}", name, e))?;
        let mut header_value = HeaderValue::from_str(value)
            .map_err(|e| format!("Invalid MCP HTTP header value for '{}': {}", name, e))?;
        header_value.set_sensitive(true);
        headers.insert(header_name, header_value);
    }
    Ok(headers)
}

fn is_event_stream(response: &Response) -> bool {
    content_type(response).contains("text/event-stream")
}

fn content_type(response: &Response) -> String {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn ensure_success(response: Response) -> Result<(), String> {
    if response.status().is_success() {
        Ok(())
    } else {
        Err(http_error(response))
    }
}

fn http_error(response: Response) -> String {
    let status = response.status();
    let body = response.text().unwrap_or_default();
    let snippet: String = body.chars().take(ERROR_BODY_LIMIT).collect();
    if status == reqwest::StatusCode::NOT_FOUND {
        format!(
            "MCP HTTP session expired or endpoint not found (HTTP 404): {}",
            snippet
        )
    } else {
        format!("MCP HTTP {}: {}", status, snippet)
    }
}

fn post_json(
    client: &Client,
    url: &Url,
    headers: &HeaderMap,
    body: &str,
    timeout_ms: u64,
) -> Result<(), String> {
    let response = client
        .post(url.clone())
        .timeout(Duration::from_millis(timeout_ms))
        .headers(headers.clone())
        .header(CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .map_err(|e| format!("MCP HTTP request failed: {}", e))?;
    ensure_success(response)
}

#[derive(Default)]
struct SseParser {
    event: String,
    data: String,
    has_data: bool,
}

impl SseParser {
    fn feed_line(&mut self, line: &str) -> Option<SseFrame> {
        let line = line.trim_end_matches(|c| c == '\r' || c == '\n');
        if line.is_empty() {
            if !self.has_data {
                self.event.clear();
                return None;
            }
            let event = std::mem::take(&mut self.event);
            let data = std::mem::take(&mut self.data);
            self.has_data = false;
            return Some(if event == "endpoint" {
                SseFrame::Endpoint(data)
            } else {
                SseFrame::Message(data)
            });
        }
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => self.event = value.to_string(),
            "data" => {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
            _ => {}
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::{TcpListener, TcpStream},
    };

    fn config(url: String, kind: McpTransportKind) -> McpServerConfig {
        McpServerConfig {
            command: String::new(),
            args: Vec::new(),
            env: SkipMap::default(),
            url,
            headers: SkipMap::default(),
            transport: Some(kind),
            request_timeout_ms: 2000,
            disabled: false,
        }
    }

    fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    fn read_request(stream: &mut TcpStream) -> String {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = stream.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            bytes.extend_from_slice(&buf[..n]);
            let Some(header_end) = find_subslice(&bytes, b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.strip_prefix("Content-Length: ")
                        .or_else(|| line.strip_prefix("content-length: "))
                })
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if bytes.len() >= header_end + 4 + content_length {
                break;
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn collect(input: &str) -> Vec<(String, String)> {
        let mut parser = SseParser::default();
        let mut events = Vec::new();
        for line in input.lines() {
            if let Some(frame) = parser.feed_line(line) {
                match frame {
                    super::SseFrame::Endpoint(data) => events.push(("endpoint".to_string(), data)),
                    super::SseFrame::Message(data) => events.push(("message".to_string(), data)),
                }
            }
        }
        events
    }

    #[test]
    fn parses_endpoint_and_message_events() {
        let events = collect(
            "event: endpoint\ndata: /messages?session=abc\n\n\
             event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n",
        );
        assert_eq!(
            events,
            vec![
                ("endpoint".to_string(), "/messages?session=abc".to_string()),
                (
                    "message".to_string(),
                    "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}".to_string()
                ),
            ]
        );
    }

    #[test]
    fn joins_multiline_data_and_ignores_comments() {
        let events = collect(": ping\n\nevent: message\ndata: line one\ndata: line two\n\n");
        assert_eq!(
            events,
            vec![("message".to_string(), "line one\nline two".to_string())]
        );
    }

    #[test]
    fn ignores_events_without_data() {
        let events = collect("event: message\n\nevent: message\ndata: ok\n\n");
        assert_eq!(events, vec![("message".to_string(), "ok".to_string())]);
    }

    #[test]
    fn streamable_http_posts_json_and_captures_session_id() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_request(&mut stream);
            assert!(request.contains("\"method\":\"initialize\""));
            assert!(request.contains("mcp-protocol-version: 2025-03-26"));
            let body = r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26"}}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nMcp-Session-Id: session-1\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });

        let mut transport = HttpTransport::connect(&config(
            format!("http://{}/mcp", addr),
            McpTransportKind::Http,
        ))
        .unwrap();
        let result = transport
            .request(
                1,
                r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
                2000,
            )
            .unwrap();
        assert_eq!(result["protocolVersion"], "2025-03-26");
        assert_eq!(transport.session_id.as_deref(), Some("session-1"));

        server.join().unwrap();
        drop(transport);
    }

    #[test]
    fn legacy_sse_uses_endpoint_event_and_returns_stream_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut sse, _) = listener.accept().unwrap();
            let request = read_request(&mut sse);
            assert!(request.starts_with("GET /sse"));
            write!(
                sse,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n"
            )
            .unwrap();
            write!(sse, "event: endpoint\ndata: http://{}/messages\n\n", addr).unwrap();
            sse.flush().unwrap();

            let (mut post, _) = listener.accept().unwrap();
            let request = read_request(&mut post);
            assert!(request.contains("\"method\":\"initialize\""));
            write!(
                post,
                "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            post.flush().unwrap();

            let body = r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#;
            write!(sse, "event: message\ndata: {}\n\n", body).unwrap();
            sse.flush().unwrap();
            thread::sleep(Duration::from_millis(50));
        });

        let mut transport = HttpTransport::connect(&config(
            format!("http://{}/sse", addr),
            McpTransportKind::Sse,
        ))
        .unwrap();
        let result = transport
            .request(
                1,
                r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
                2000,
            )
            .unwrap();
        assert_eq!(result["ok"], true);

        drop(transport);
        server.join().unwrap();
    }
}
