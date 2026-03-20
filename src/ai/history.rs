use std::{
    fs::{self, OpenOptions},
    io,
    path::PathBuf,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::strw::split::split_by_str_keep_quotes;

const MAX_HISTORY_LINES: usize = 100;
pub(super) const COLON: char = '\0';
pub(super) const NEWLINE: char = '\x01';

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct Message {
    pub(super) role: String,
    pub(super) content: Value,
}

pub(super) fn build_message_arr(
    history_count: usize,
    history_file: &PathBuf,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let history = match fs::read_to_string(history_file) {
        Ok(history) => history,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };

    let newline = NEWLINE.to_string();
    let lines = split_by_str_keep_quotes(&history, &newline, "\"", false);
    let mut messages = Vec::new();

    for line in &lines {
        if line.is_empty() {
            continue;
        }
        let Some(last_colon) = line.rfind(COLON) else {
            continue;
        };
        if last_colon == 0 || last_colon + COLON.len_utf8() >= line.len() {
            continue;
        }
        let role = &line[..last_colon];
        if role != "user" && role != "assistant" {
            continue;
        }
        let content = &line[last_colon + COLON.len_utf8()..];
        messages.push(Message {
            role: role.to_string(),
            content: Value::String(content.to_string()),
        });
    }

    if lines.len() > MAX_HISTORY_LINES {
        let start = lines.len() - MAX_HISTORY_LINES;
        let trimmed = lines[start..].join(&newline);
        fs::write(history_file, trimmed)?;
    }

    if history_count >= messages.len() {
        return Ok(messages);
    }
    Ok(messages[messages.len() - history_count..].to_vec())
}

pub(super) fn append_history(path: &PathBuf, content: &str) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.create(true).append(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o664);
    }
    let mut file = options.open(path)?;
    use std::io::Write;
    file.write_all(content.as_bytes())
}
