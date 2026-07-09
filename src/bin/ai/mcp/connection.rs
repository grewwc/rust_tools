use std::{
    io::{BufReader, Read, Write},
    process::{Child, ChildStderr, ChildStdin, ChildStdout},
    sync::{Arc, Mutex},
    thread,
};

use crate::ai::types::{McpPrompt, McpResource, McpTool};

use super::io::read_line_with_timeout_process;

const STDERR_TAIL_LIMIT: usize = 16 * 1024;
const STDERR_SNIPPET_LIMIT: usize = 400;

pub(in crate::ai) struct McpServerConnection {
    pub(in crate::ai) config: crate::ai::types::McpServerConfig,
    pub(in crate::ai) process: Child,
    pub(in crate::ai) stdin: ChildStdin,
    pub(in crate::ai) stdout: BufReader<ChildStdout>,
    pub(in crate::ai) stderr_tail: Arc<Mutex<String>>,
    pub(in crate::ai) request_timeout_ms: u64,
    pub(in crate::ai) tools: Vec<McpTool>,
    pub(in crate::ai) resources: Vec<McpResource>,
    pub(in crate::ai) prompts: Vec<McpPrompt>,
}

pub(in crate::ai) fn spawn_stderr_drain(stderr: ChildStderr) -> Arc<Mutex<String>> {
    let stderr_tail = Arc::new(Mutex::new(String::new()));
    let sink = stderr_tail.clone();
    thread::spawn(move || drain_stderr_into_tail(stderr, sink));
    stderr_tail
}

impl McpServerConnection {
    pub(in crate::ai) fn stdin_mut(&mut self) -> &mut dyn Write {
        &mut self.stdin
    }

    pub(super) fn read_response_line(&mut self) -> Result<String, String> {
        self.read_response_line_with_timeout(self.request_timeout_ms)
    }

    pub(super) fn read_response_line_with_timeout(
        &mut self,
        timeout_ms: u64,
    ) -> Result<String, String> {
        let mut response_line = String::new();
        read_line_with_timeout_process(&mut self.stdout, timeout_ms, &mut response_line)
            .map_err(|err| self.decorate_transport_error(err))?;
        Ok(response_line)
    }

    pub(super) fn decorate_transport_error(&mut self, err: String) -> String {
        let mut detail = err;

        if let Ok(Some(status)) = self.process.try_wait() {
            detail.push_str(&format!(" | process exited with status {}", status));
        }

        let stderr = self.read_stderr_snippet();
        if !stderr.is_empty() {
            detail.push_str(" | stderr: ");
            detail.push_str(&stderr);
        }

        detail
    }

    fn read_stderr_snippet(&self) -> String {
        let text = self
            .stderr_tail
            .lock()
            .map(|tail| tail.trim().to_string())
            .unwrap_or_default();
        if text.chars().count() > STDERR_SNIPPET_LIMIT {
            let truncated = text.chars().take(STDERR_SNIPPET_LIMIT).collect::<String>();
            format!("{}...", truncated)
        } else {
            text
        }
    }
}

fn drain_stderr_into_tail(mut stderr: ChildStderr, sink: Arc<Mutex<String>>) {
    let mut buf = [0u8; 1024];
    loop {
        match stderr.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => append_stderr_tail(&sink, &buf[..n]),
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

fn append_stderr_tail(sink: &Arc<Mutex<String>>, chunk: &[u8]) {
    let text = String::from_utf8_lossy(chunk);
    let Ok(mut tail) = sink.lock() else {
        return;
    };
    tail.push_str(&text);
    if tail.len() <= STDERR_TAIL_LIMIT {
        return;
    }

    let trim_from = tail.len().saturating_sub(STDERR_TAIL_LIMIT);
    let keep_start = tail
        .char_indices()
        .find(|(idx, _)| *idx >= trim_from)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    tail.drain(..keep_start);
}
