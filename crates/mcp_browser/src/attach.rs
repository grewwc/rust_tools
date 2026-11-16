//! Guard chromiumoxide's automatic target cleanup on external CDP connections.
//!
//! The upstream handler closes even foreign targets after initialization failures.
//! A loopback WebSocket bridge enforces ownership before requests reach Chrome.

use chromiumoxide::Connection;
use chromiumoxide::cdp::CdpEventMessage;
use chromiumoxide::cdp::browser_protocol::target::SessionId;
use chromiumoxide::types::{CallId, CdpJsonEventMessage, Message};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

pub(crate) struct AttachGuard {
    pub(crate) url: String,
    task: JoinHandle<()>,
}

impl AttachGuard {
    pub(crate) async fn connect(upstream: &str) -> Result<Self, String> {
        let connection = Connection::<CdpEventMessage>::connect(upstream)
            .await
            .map_err(|e| format!("CDP attach connection failed: {e}"))?;
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| format!("CDP attach guard listener failed: {e}"))?;
        let url = format!(
            "ws://{}/",
            listener.local_addr().map_err(|e| e.to_string())?
        );
        let task = tokio::spawn(async move {
            if let Err(e) = bridge(listener, connection).await {
                eprintln!("CDP attach guard stopped: {e}");
            }
        });
        Ok(Self { url, task })
    }
}

impl Drop for AttachGuard {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct AbortTask(JoinHandle<()>);

impl Drop for AbortTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn allowed(request: &Value, owned: &[String]) -> bool {
    match request["method"].as_str() {
        Some("Browser.close") => false,
        Some("Target.closeTarget") => request["params"]["targetId"]
            .as_str()
            .is_some_and(|id| owned.iter().any(|target| target == id)),
        _ => true,
    }
}

async fn bridge(
    listener: TcpListener,
    mut upstream: Connection<CdpEventMessage>,
) -> Result<(), String> {
    let (mut local, _) = listener.accept().await.map_err(|e| e.to_string())?;
    handshake(&mut local).await?;
    let (mut reader, mut writer) = local.into_split();
    let mut pending: Vec<(CallId, Value)> = Vec::new();
    let mut owned = Vec::new();
    // Keep the frame reader alive across select iterations: read_exact is not
    // cancellation safe, and upstream events may arrive halfway through a frame.
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let reader_task = AbortTask(tokio::spawn(async move {
        while let Ok(frame) = read_frame(&mut reader).await {
            if tx.send(frame).await.is_err() {
                break;
            }
        }
    }));
    let outcome = async {
        loop {
            tokio::select! {
                frame = rx.recv() => {
                    let Some((opcode, bytes)) = frame else { break };
                    if opcode == 8 { break; }
                    if opcode == 9 {
                        write_frame(&mut writer, 10, &bytes).await?;
                        continue;
                    }
                    if opcode != 1 { return Err("unsupported CDP WebSocket frame".to_string()); }
                    let request: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                    if !allowed(&request, &owned) {
                        let mut response = json!({"id": request["id"], "error": {
                            "code": -32000, "message": "attach guard: target is not owned by this session"
                        }});
                        if let Some(session) = request.get("sessionId") {
                            response["sessionId"] = session.clone();
                        }
                        write_json(&mut writer, &response).await?;
                        continue;
                    }
                    let method = request["method"].as_str().ok_or("CDP method missing")?.to_string();
                    let session = request["sessionId"].as_str().map(|id| SessionId::from(id.to_string()));
                    let id = upstream.submit_command(method.into(), session, request["params"].clone())
                        .map_err(|e| e.to_string())?;
                    pending.push((id, request));
                }
                message = upstream.next() => {
                    let Some(message) = message else { break };
                    match message.map_err(|e| e.to_string())? {
                        Message::Response(response) => {
                            let Some(index) = pending.iter().position(|(id, _)| *id == response.id) else { continue };
                            let (_, request) = pending.remove(index);
                            if request["method"] == "Target.createTarget"
                                && let Some(id) = response.result.as_ref().and_then(|v| v["targetId"].as_str()) {
                                owned.push(id.to_string());
                            }
                            let mut output = json!({"id": request["id"]});
                            if let Some(result) = response.result { output["result"] = result; }
                            if let Some(error) = response.error {
                                output["error"] = json!({"code": error.code, "message": error.message});
                            }
                            if let Some(session) = request.get("sessionId") { output["sessionId"] = session.clone(); }
                            write_json(&mut writer, &output).await?;
                        }
                        Message::Event(event) => {
                            let event: CdpJsonEventMessage = event.try_into().map_err(|e: serde_json::Error| e.to_string())?;
                            let mut output = json!({"method": event.method, "params": event.params});
                            if let Some(session) = event.session_id { output["sessionId"] = json!(session); }
                            write_json(&mut writer, &output).await?;
                        }
                    }
                }
            }
        }
        Ok(())
    }.await;
    drop(reader_task);
    outcome
}

// Only the server half of RFC 6455 is needed here. The existing CDP Connection
// owns the remote WebSocket (including TLS); the local peer is chromiumoxide.
async fn handshake(stream: &mut TcpStream) -> Result<(), String> {
    let mut header = Vec::new();
    while !header.ends_with(b"\r\n\r\n") {
        if header.len() >= 8192 {
            return Err("CDP guard handshake too large".into());
        }
        header.push(stream.read_u8().await.map_err(|e| e.to_string())?);
    }
    let header = String::from_utf8(header).map_err(|e| e.to_string())?;
    let key = header
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("sec-websocket-key")
                .then_some(value.trim())
        })
        .ok_or("CDP guard WebSocket key missing")?;
    let accept = websocket_accept(key);
    stream.write_all(format!("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n").as_bytes())
        .await.map_err(|e| e.to_string())
}

async fn read_frame(reader: &mut tokio::net::tcp::OwnedReadHalf) -> Result<(u8, Vec<u8>), String> {
    let first = reader.read_u8().await.map_err(|e| e.to_string())?;
    let second = reader.read_u8().await.map_err(|e| e.to_string())?;
    if first & 0x80 == 0 || second & 0x80 == 0 {
        return Err("CDP guard requires complete masked frames".into());
    }
    let length = match second & 0x7f {
        126 => reader.read_u16().await.map_err(|e| e.to_string())? as u64,
        127 => reader.read_u64().await.map_err(|e| e.to_string())?,
        n => n as u64,
    };
    if length > 64 * 1024 * 1024 {
        return Err("CDP guard frame too large".into());
    }
    let mut mask = [0; 4];
    reader
        .read_exact(&mut mask)
        .await
        .map_err(|e| e.to_string())?;
    let mut bytes = vec![0; length as usize];
    reader
        .read_exact(&mut bytes)
        .await
        .map_err(|e| e.to_string())?;
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte ^= mask[i % 4];
    }
    Ok((first & 0xf, bytes))
}

async fn write_json(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    value: &Value,
) -> Result<(), String> {
    write_frame(writer, 1, value.to_string().as_bytes()).await
}

async fn write_frame(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    opcode: u8,
    bytes: &[u8],
) -> Result<(), String> {
    let mut frame = vec![0x80 | opcode];
    match bytes.len() {
        n @ 0..=125 => frame.push(n as u8),
        n @ 126..=65535 => {
            frame.push(126);
            frame.extend_from_slice(&(n as u16).to_be_bytes());
        }
        n => {
            frame.push(127);
            frame.extend_from_slice(&(n as u64).to_be_bytes());
        }
    }
    frame.extend_from_slice(bytes);
    writer.write_all(&frame).await.map_err(|e| e.to_string())
}

fn websocket_accept(key: &str) -> String {
    let mut bytes = format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11").into_bytes();
    let bits = (bytes.len() as u64) * 8;
    bytes.push(0x80);
    while bytes.len() % 64 != 56 {
        bytes.push(0);
    }
    bytes.extend_from_slice(&bits.to_be_bytes());
    let mut hash = [
        0x67452301u32,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0,
    ];
    for chunk in bytes.chunks_exact(64) {
        let mut words = [0u32; 80];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            words[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..80 {
            words[i] = (words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = hash;
        for (i, word) in words.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5a827999u32),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc),
                _ => (b ^ c ^ d, 0xca62c1d6),
            };
            let next = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }
        for (h, v) in hash.iter_mut().zip([a, b, c, d, e]) {
            *h = h.wrapping_add(v);
        }
    }
    let digest: Vec<u8> = hash.into_iter().flat_map(u32::to_be_bytes).collect();
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::new();
    for chunk in digest.chunks(3) {
        let n = ((chunk[0] as u32) << 16)
            | ((chunk.get(1).copied().unwrap_or(0) as u32) << 8)
            | chunk.get(2).copied().unwrap_or(0) as u32;
        encoded.push(alphabet[((n >> 18) & 63) as usize] as char);
        encoded.push(alphabet[((n >> 12) & 63) as usize] as char);
        encoded.push(if chunk.len() > 1 {
            alphabet[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            alphabet[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_fixes_attach_guard_rejects_foreign_cleanup() {
        let owned = vec!["owned".to_string()];
        assert!(allowed(
            &json!({"method":"Target.closeTarget","params":{"targetId":"owned"}}),
            &owned
        ));
        assert!(!allowed(
            &json!({"method":"Target.closeTarget","params":{"targetId":"foreign"}}),
            &owned
        ));
        assert!(!allowed(&json!({"method":"Target.closeTarget"}), &owned));
        assert!(!allowed(&json!({"method":"Browser.close"}), &owned));
        assert!(allowed(&json!({"method":"Target.createTarget"}), &owned));
        assert_eq!(
            websocket_accept("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }
}
