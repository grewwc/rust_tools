use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, io::{self, Read, Write}, os::unix::net::UnixStream,
    path::Path, time::Duration};

pub(super) const REQUEST: u8 = 1;
pub(super) const REPLY: u8 = 2;
pub(super) const OUTPUT: u8 = 3;
pub(super) const INPUT: u8 = 4;
pub(super) const RESIZE: u8 = 5;
pub(super) const DETACHED: u8 = 6;
pub(super) const EXIT: u8 = 7;
pub(super) const MAX_FRAME: usize = 64 * 1024;
pub(super) const MAX_QUEUE: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub(super) struct Window { pub rows: u16, pub cols: u16 }

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum Request {
    Attach { terminal: String, window: Window },
    Register { token: String, session: String },
    Claim { token: String, session: String },
    Commit { token: String },
    Rollback { token: String },
    Detach { token: String },
    Terminal { token: String },
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct Reply {
    pub error: Option<String>,
    pub terminal: Option<String>,
    pub session: Option<String>,
}

pub(super) fn frame(kind: u8, payload: &[u8]) -> io::Result<Vec<u8>> {
    if payload.len() > MAX_FRAME { return Err(io::Error::other("PTY frame too large")); }
    let mut bytes = Vec::with_capacity(5 + payload.len());
    bytes.push(kind);
    bytes.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

pub(super) fn write_json(writer: &mut impl Write, kind: u8, value: &impl Serialize) -> io::Result<()> {
    writer.write_all(&frame(kind, &serde_json::to_vec(value)?)?)
}

pub(super) fn read_frame(reader: &mut impl Read) -> io::Result<(u8, Vec<u8>)> {
    let mut header = [0; 5];
    reader.read_exact(&mut header)?;
    let len = u32::from_be_bytes(header[1..].try_into().unwrap()) as usize;
    if len > MAX_FRAME { return Err(io::Error::other("PTY frame too large")); }
    let mut payload = vec![0; len];
    reader.read_exact(&mut payload)?;
    Ok((header[0], payload))
}

pub(super) fn rpc(socket: &Path, request: Request) -> io::Result<Reply> {
    let mut stream = UnixStream::connect(socket)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    write_json(&mut stream, REQUEST, &request)?;
    let (kind, data) = read_frame(&mut stream)?;
    if kind != REPLY { return Err(io::Error::other("invalid PTY reply")); }
    let reply: Reply = serde_json::from_slice(&data)?;
    if let Some(error) = &reply.error { return Err(io::Error::other(error.clone())); }
    Ok(reply)
}

#[derive(Default)]
pub(super) struct Decoder { data: Vec<u8> }
impl Decoder {
    pub(super) fn push(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.data.len() + bytes.len() > MAX_QUEUE { return Err(io::Error::other("PTY input overflow")); }
        self.data.extend_from_slice(bytes);
        Ok(())
    }
    pub(super) fn next(&mut self) -> io::Result<Option<(u8, Vec<u8>)>> {
        if self.data.len() < 5 { return Ok(None); }
        let len = u32::from_be_bytes(self.data[1..5].try_into().unwrap()) as usize;
        if len > MAX_FRAME { return Err(io::Error::other("PTY frame too large")); }
        if self.data.len() < 5 + len { return Ok(None); }
        let kind = self.data[0];
        let payload = self.data[5..5 + len].to_vec();
        self.data.drain(..5 + len);
        Ok(Some((kind, payload)))
    }
}

#[derive(Default)]
pub(super) struct Queue { chunks: VecDeque<Vec<u8>>, offset: usize, size: usize }
impl Queue {
    pub(super) fn is_empty(&self) -> bool { self.chunks.is_empty() }
    pub(super) fn push(&mut self, bytes: Vec<u8>) -> io::Result<()> {
        if self.size + bytes.len() > MAX_QUEUE { return Err(io::Error::other("PTY output overflow")); }
        self.size += bytes.len();
        self.chunks.push_back(bytes);
        Ok(())
    }
    pub(super) fn framed(&mut self, kind: u8, data: &[u8]) -> io::Result<()> { self.push(frame(kind, data)?) }
    pub(super) fn json(&mut self, kind: u8, value: &impl Serialize) -> io::Result<()> {
        self.framed(kind, &serde_json::to_vec(value)?)
    }
    pub(super) fn flush(&mut self, writer: &mut impl Write) -> io::Result<()> {
        while let Some(chunk) = self.chunks.front() {
            match writer.write(&chunk[self.offset..]) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => {
                    self.offset += n;
                    self.size -= n;
                    if self.offset == chunk.len() { self.chunks.pop_front(); self.offset = 0; }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}