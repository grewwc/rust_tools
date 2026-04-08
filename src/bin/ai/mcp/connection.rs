use std::{
    io::{BufReader, Write},
    process::{Child, ChildStderr, ChildStdin, ChildStdout},
};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

use crate::ai::types::{McpPrompt, McpResource, McpTool};

use super::io::{read_available_buf, read_line_with_timeout_process};

pub(in crate::ai) struct McpServerConnection {
    pub(in crate::ai) config: crate::ai::types::McpServerConfig,
    pub(in crate::ai) process: Child,
    pub(in crate::ai) stdin: ChildStdin,
    pub(in crate::ai) stdout: BufReader<ChildStdout>,
    pub(in crate::ai) stderr: BufReader<ChildStderr>,
    pub(in crate::ai) request_timeout_ms: u64,
    pub(in crate::ai) tools: Vec<McpTool>,
    pub(in crate::ai) resources: Vec<McpResource>,
    pub(in crate::ai) prompts: Vec<McpPrompt>,
}

impl McpServerConnection {
    pub(in crate::ai) fn stdin_mut(&mut self) -> &mut dyn Write {
        &mut self.stdin
    }

    pub(super) fn read_response_line(&mut self) -> Result<String, String> {
        let mut response_line = String::new();
        read_line_with_timeout_process(
            &mut self.stdout,
            self.request_timeout_ms,
            &mut response_line,
        )
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

    fn read_stderr_snippet(&mut self) -> String {
        let fd = self.stderr.get_ref().as_raw_fd();
        let bytes = read_available_buf(&mut self.stderr, fd);
        let text = String::from_utf8_lossy(&bytes).trim().to_string();
        if text.chars().count() > 400 {
            let truncated = text.chars().take(400).collect::<String>();
            format!("{}...", truncated)
        } else {
            text
        }
    }
}
