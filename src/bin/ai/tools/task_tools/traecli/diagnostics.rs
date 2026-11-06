//! Bounded incremental stderr capture, including bytes received before EOF.

use std::io::{self, Read};
use std::sync::{Arc, Mutex};

use super::{lock_unpoisoned, permanent_traecli_error_marker};

const MAX_STDERR_BYTES: usize = 16 * 1024;

#[derive(Default)]
pub(super) struct StderrTail {
    bytes: Vec<u8>,
    total_bytes: usize,
    permanent_marker: Option<&'static str>,
}

impl StderrTail {
    fn append(&mut self, chunk: &[u8]) {
        self.total_bytes = self.total_bytes.saturating_add(chunk.len());
        if chunk.len() >= MAX_STDERR_BYTES {
            self.bytes.clear();
            self.bytes
                .extend_from_slice(&chunk[chunk.len() - MAX_STDERR_BYTES..]);
        } else {
            let keep = MAX_STDERR_BYTES - chunk.len();
            if self.bytes.len() > keep {
                self.bytes.drain(..self.bytes.len() - keep);
            }
            self.bytes.extend_from_slice(chunk);
        }
        // Keep the classifier's observed evidence even if later stderr evicts
        // that text. Each read is smaller than the retained window, so markers
        // split across reads are also recognized before eviction.
        if self.permanent_marker.is_none() {
            self.permanent_marker =
                permanent_traecli_error_marker(&String::from_utf8_lossy(&self.bytes));
        }
    }

    pub(super) fn snapshot(&self, max_chars: usize) -> String {
        let text = String::from_utf8_lossy(&self.bytes);
        let chars = text.chars().count();
        let shown = if chars > max_chars {
            let start = text
                .char_indices()
                .nth(chars - max_chars)
                .map(|(index, _)| index)
                .unwrap_or(text.len());
            &text[start..]
        } else {
            text.as_ref()
        };
        let snapshot = if self.total_bytes > self.bytes.len() || chars > max_chars {
            format!(
                "[stderr truncated: retained last {} of {} bytes; showing {} chars]\n{shown}",
                self.bytes.len(),
                self.total_bytes,
                chars.min(max_chars),
            )
        } else {
            shown.to_string()
        };
        if let Some(marker) = self.permanent_marker
            && permanent_traecli_error_marker(&snapshot).is_none()
        {
            format!("[earlier stderr diagnostic: {marker}]\n{snapshot}")
        } else {
            snapshot
        }
    }
}

pub(super) fn read_stderr(mut reader: impl Read, shared: &Arc<Mutex<StderrTail>>) {
    let mut buffer = [0_u8; 8192];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(len) => lock_unpoisoned(shared).append(&buffer[..len]),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}
