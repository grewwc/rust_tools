use std::io::{self, BufRead, BufReader, Write};
use std::collections::HashMap;

use colored::Colorize;
use serde_json;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::{
    request::StreamChunk,
    types::{App, StreamOutcome, StreamResult, ToolCall, take_stream_cancelled},
};

pub(super) fn stream_response(
    app: &mut App,
    response: &mut reqwest::blocking::Response,
    current_history: &mut String,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(response);
    let thinking_tag = "<thinking>".yellow().to_string();
    let end_thinking_tag = "<end thinking>".yellow().to_string();
    let mut thinking_open = false;
    let mut markdown = MarkdownStreamRenderer::new();
    let mut line = String::new();
    let mut tool_calls_map: HashMap<usize, ToolCallBuilder> = HashMap::new();
    let mut finish_reason: Option<String> = None;
    let mut assistant_text = String::new();

    while !app.shutdown.load(std::sync::atomic::Ordering::SeqCst) {
        if take_stream_cancelled(app) {
            return Ok(StreamResult {
                outcome: StreamOutcome::Cancelled,
                tool_calls: Vec::new(),
                finish_reason: None,
                assistant_text: String::new(),
            });
        }
        line.clear();
        let n = match reader.read_line(&mut line) {
            Ok(n) => n,
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                ) =>
            {
                if take_stream_cancelled(app) {
                    return Ok(StreamResult {
                        outcome: StreamOutcome::Cancelled,
                        tool_calls: Vec::new(),
                        finish_reason: None,
                        assistant_text: String::new(),
                    });
                }
                if app.shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                    return Ok(StreamResult {
                        outcome: StreamOutcome::Cancelled,
                        tool_calls: Vec::new(),
                        finish_reason: None,
                        assistant_text: String::new(),
                    });
                }
                continue;
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                if take_stream_cancelled(app) {
                    return Ok(StreamResult {
                        outcome: StreamOutcome::Cancelled,
                        tool_calls: Vec::new(),
                        finish_reason: None,
                        assistant_text: String::new(),
                    });
                }
                continue;
            }
            Err(err) => return Err(err.into()),
        };
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if !trimmed.starts_with("data:") {
            continue;
        }
        let payload = trimmed.trim_start_matches("data:").trim();
        if payload.is_empty() {
            continue;
        }
        if payload == "[DONE]" {
            break;
        }

        let chunk: StreamChunk = match serde_json::from_str(payload) {
            Ok(chunk) => chunk,
            Err(err) => {
                eprintln!("handleResponse error {err}");
                eprintln!("======> response: ");
                eprintln!("{payload}");
                eprintln!("<======");
                continue;
            }
        };

        if let Some(choice) = chunk.choices.first() {
            if let Some(ref reason) = choice.finish_reason {
                finish_reason = Some(reason.clone());
            }

            for stream_tool_call in &choice.delta.tool_calls {
                let index = stream_tool_call.index;
                let builder = tool_calls_map.entry(index).or_insert_with(ToolCallBuilder::new);
                
                if !stream_tool_call.id.is_empty() {
                    builder.id = stream_tool_call.id.clone();
                }
                if !stream_tool_call.tool_type.is_empty() {
                    builder.tool_type = stream_tool_call.tool_type.clone();
                }
                if !stream_tool_call.function.name.is_empty() {
                    builder.function_name = stream_tool_call.function.name.clone();
                }
                builder.arguments.push_str(&stream_tool_call.function.arguments);
            }
        }

        let content =
            extract_chunk_text(&chunk, &thinking_tag, &end_thinking_tag, &mut thinking_open);
        if content.is_empty() {
            continue;
        }
        write_stream_content(content.as_str(), app.writer.as_mut(), &mut markdown)?;
        if thinking_open {
            continue;
        }
        let text = content.replace(&end_thinking_tag, "");
        let text = text.trim_matches('\n');
        current_history.push_str(text);
        assistant_text.push_str(text);
    }

    markdown.flush_pending()?;

    if take_stream_cancelled(app) {
        return Ok(StreamResult {
            outcome: StreamOutcome::Cancelled,
            tool_calls: Vec::new(),
            finish_reason: None,
            assistant_text: String::new(),
        });
    }

    let tool_calls: Vec<ToolCall> = tool_calls_map
        .into_iter()
        .map(|(_, builder)| builder.build())
        .collect();

    let outcome = if !tool_calls.is_empty() {
        StreamOutcome::ToolCall
    } else {
        StreamOutcome::Completed
    };

    Ok(StreamResult {
        outcome,
        tool_calls,
        finish_reason,
        assistant_text,
    })
}

#[derive(Default)]
struct ToolCallBuilder {
    id: String,
    tool_type: String,
    function_name: String,
    arguments: String,
}

impl ToolCallBuilder {
    fn new() -> Self {
        Self {
            id: String::new(),
            tool_type: "function".to_string(),
            function_name: String::new(),
            arguments: String::new(),
        }
    }

    fn build(self) -> ToolCall {
        use super::types::{FunctionCall, ToolCall};
        ToolCall {
            id: self.id,
            tool_type: self.tool_type,
            function: FunctionCall {
                name: self.function_name,
                arguments: self.arguments,
            },
        }
    }
}

pub(super) fn extract_chunk_text(
    chunk: &StreamChunk,
    thinking_tag: &str,
    end_thinking_tag: &str,
    thinking_open: &mut bool,
) -> String {
    let Some(choice) = chunk.choices.first() else {
        return String::new();
    };
    let delta = &choice.delta;

    if delta.content.is_empty() && !delta.reasoning_content.is_empty() {
        if !*thinking_open {
            *thinking_open = true;
            return format!("\n{thinking_tag}\n{}", delta.reasoning_content);
        }
        return delta.reasoning_content.clone();
    }

    if *thinking_open {
        *thinking_open = false;
        return format!("\n{end_thinking_tag}\n{}", delta.content);
    }
    delta.content.clone()
}

fn write_stream_content(
    content: &str,
    mut writer: Option<&mut std::fs::File>,
    markdown: &mut MarkdownStreamRenderer,
) -> io::Result<()> {
    if let Some(file) = writer.as_mut() {
        file.write_all(content.as_bytes())?;
        file.flush()?;
    }

    if markdown.should_render(content) {
        markdown.write_chunk(content)?;
    } else {
        print!("{content}");
    }
    io::stdout().flush()
}

pub(super) struct MarkdownStreamRenderer {
    tty: bool,
    enabled: bool,
    in_code_block: bool,
    bol: bool,
    line_buf: String,
    line_preview_emitted: bool,
    line_preview_height: usize,
    table_state: TableState,
}

impl MarkdownStreamRenderer {
    fn new() -> Self {
        use std::io::IsTerminal;
        Self::new_with_tty(io::stdout().is_terminal())
    }

    pub(super) fn new_with_tty(tty: bool) -> Self {
        Self {
            tty,
            enabled: true,
            in_code_block: false,
            bol: false,
            line_buf: String::new(),
            line_preview_emitted: false,
            line_preview_height: 0,
            table_state: TableState::None,
        }
    }

    fn should_render(&mut self, chunk: &str) -> bool {
        if !self.tty {
            return false;
        }
        if chunk.contains("\x1b[") {
            return false;
        }
        self.enabled = true;
        true
    }

    fn redraw_inline_preview(&mut self, out: &mut std::io::Stdout) -> io::Result<()> {
        if !self.tty {
            return Ok(());
        }
        if self.line_preview_emitted {
            let up = self.line_preview_height.saturating_sub(1);
            if up > 0 {
                out.write_all(format!("\x1b[{up}A\r\x1b[0J").as_bytes())?;
            } else {
                out.write_all(b"\r\x1b[0J")?;
            }
        }

        let rendered = if self.in_code_block {
            if self.line_buf.is_empty() {
                String::new()
            } else {
                format!("\x1b[97m{}\x1b[0m", self.line_buf)
            }
        } else {
            let (indent, rest) = split_indent(&self.line_buf);
            format!("{indent}{}", render_inline_md(rest, ""))
        };
        out.write_all(rendered.as_bytes())?;
        self.line_preview_emitted = true;
        self.line_preview_height = table_preview_height(&self.line_buf).max(1);
        Ok(())
    }

    fn write_chunk(&mut self, chunk: &str) -> io::Result<()> {
        let mut out = io::stdout();
        for ch in chunk.chars() {
            if ch == '\n' {
                if self.line_preview_emitted {
                    out.write_all(b"\n")?;
                    self.bol = true;
                }
                let line = std::mem::take(&mut self.line_buf);
                let rendered = self.consume_line(&line, self.line_preview_emitted);
                self.line_preview_emitted = false;
                self.line_preview_height = 0;
                if !rendered.is_empty() {
                    out.write_all(rendered.as_bytes())?;
                    self.bol = rendered.ends_with('\n');
                }
                continue;
            }
            self.line_buf.push(ch);

            if self.should_emit_table_preview_live() {
                if !self.line_preview_emitted {
                    out.write_all(self.line_buf.as_bytes())?;
                    self.line_preview_emitted = true;
                    self.line_preview_height = table_preview_height(&self.line_buf).max(1);
                } else {
                    let mut buf = [0u8; 4];
                    out.write_all(ch.encode_utf8(&mut buf).as_bytes())?;
                    self.line_preview_height = table_preview_height(&self.line_buf).max(1);
                }
            } else {
                self.redraw_inline_preview(&mut out)?;
            }
            self.bol = false;
        }
        Ok(())
    }

    fn flush_pending(&mut self) -> io::Result<()> {
        let mut out = io::stdout();
        if !self.line_buf.is_empty() {
            if self.line_preview_emitted {
                out.write_all(b"\n")?;
                self.bol = true;
            }
            let line = std::mem::take(&mut self.line_buf);
            let rendered = self.consume_line(&line, self.line_preview_emitted);
            self.line_preview_emitted = false;
            self.line_preview_height = 0;
            if !rendered.is_empty() {
                out.write_all(rendered.as_bytes())?;
                self.bol = rendered.ends_with('\n');
            }
        }

        let state = std::mem::replace(&mut self.table_state, TableState::None);
        let rendered = match state {
            TableState::None => String::new(),
            TableState::PendingHeader { .. } => String::new(),
            TableState::InTable {
                indent,
                header,
                align,
                rows,
                preview_height,
            } => self.rewrite_table_preview(&indent, preview_height, &header, &align, &rows),
        };
        if !rendered.is_empty() {
            out.write_all(rendered.as_bytes())?;
            self.bol = rendered.ends_with('\n');
        }
        out.flush()
    }

    fn should_emit_table_preview_live(&self) -> bool {
        matches!(
            self.table_state,
            TableState::PendingHeader { .. } | TableState::InTable { .. }
        ) && line_looks_like_table_preview(&self.line_buf)
    }

    pub(super) fn consume_line(&mut self, line: &str, preview_emitted: bool) -> String {
        let state = std::mem::replace(&mut self.table_state, TableState::None);
        match state {
            TableState::None => {
                if is_table_row_candidate(line) {
                    let mut out = String::new();
                    if !self.bol {
                        out.push('\n');
                        self.bol = true;
                    }
                    let (indent, rest) = split_indent(line);
                    let raw = format!("{indent}{}", rest.trim_end());
                    let mut preview_height = table_preview_height(&raw);
                    if out.starts_with('\n') {
                        preview_height += 1;
                    }
                    self.table_state = TableState::PendingHeader {
                        indent: indent.to_string(),
                        header_line: rest.trim_end().to_string(),
                        preview_height,
                    };
                    if preview_emitted {
                        return String::new();
                    }
                    out.push_str(&raw);
                    out.push('\n');
                    return out;
                }
                let rendered = self.render_line_no_table(line);
                if preview_emitted && !line.is_empty() {
                    let preview_height = table_preview_height(line);
                    return format!("\x1b[{preview_height}A\r\x1b[0J{rendered}");
                }
                rendered
            }
            TableState::PendingHeader {
                indent,
                header_line,
                mut preview_height,
            } => {
                if is_table_separator(line) {
                    let raw = line.trim_end().to_string();
                    let mut out = String::new();
                    if !preview_emitted {
                        out.push_str(&raw);
                        out.push('\n');
                    }
                    preview_height += table_preview_height(&raw);

                    let header_cells = parse_table_row(&header_line);
                    let align = parse_table_align(line, header_cells.len());
                    self.table_state = TableState::InTable {
                        indent,
                        header: header_cells,
                        align,
                        rows: Vec::new(),
                        preview_height,
                    };
                    return out;
                }

                let _ = preview_height;
                let _ = indent;
                let _ = header_line;
                self.consume_line(line, preview_emitted)
            }
            TableState::InTable {
                indent,
                header,
                align,
                mut rows,
                mut preview_height,
            } => {
                if is_table_row(line) {
                    rows.push(parse_table_row(line));
                    let raw = line.trim_end().to_string();
                    let mut out = String::new();
                    if !preview_emitted {
                        out.push_str(&raw);
                        out.push('\n');
                    }
                    preview_height += table_preview_height(&raw);
                    self.table_state = TableState::InTable {
                        indent,
                        header,
                        align,
                        rows,
                        preview_height,
                    };
                    return out;
                }

                let mut out = String::new();
                let move_up = preview_height
                    + if preview_emitted {
                        self.line_preview_height
                    } else {
                        0
                    };
                out.push_str(&self.rewrite_table_preview(&indent, move_up, &header, &align, &rows));
                out.push_str(&self.consume_line(line, false));
                out
            }
        }
    }

    fn rewrite_table_preview(
        &self,
        indent: &str,
        move_up: usize,
        header: &[String],
        align: &[TableAlign],
        rows: &[Vec<String>],
    ) -> String {
        let cols = header
            .len()
            .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
        if cols < 2 || move_up == 0 {
            return String::new();
        }

        let widths = compute_table_widths(indent, header, rows);
        let mut final_table = String::new();
        final_table.push_str(&render_table_top(indent, &widths));
        final_table.push_str(&render_table_header(indent, header, align, &widths));
        final_table.push_str(&render_table_mid(indent, &widths));
        for row in rows {
            let row_cells = row.iter().cloned().collect::<Vec<_>>();
            final_table.push_str(&render_table_row(indent, &row_cells, align, &widths));
        }
        final_table.push_str(&render_table_bottom(indent, &widths));

        let mut out = String::new();
        out.push_str(&format!("\x1b[{move_up}A\r\x1b[0J"));
        out.push_str(&final_table);
        out
    }

    fn render_line_no_table(&mut self, line: &str) -> String {
        let (indent, rest) = split_indent(line);
        let trimmed = rest.trim_start_matches([' ', '\t']);

        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            self.in_code_block = !self.in_code_block;
            return format!("{indent}\x1b[2m{trimmed}\x1b[0m\n");
        }

        if self.in_code_block {
            if line.is_empty() {
                return "\n".to_string();
            }
            return format!("\x1b[97m{line}\x1b[0m\n");
        }

        if let Some((level, title)) = parse_heading(trimmed) {
            let (base, underline_char) = match level {
                1 => ("\x1b[1m\x1b[35m", Some('═')),
                2 => ("\x1b[1m\x1b[36m", Some('─')),
                3 => ("\x1b[1m\x1b[34m", None),
                _ => ("\x1b[1m\x1b[36m", None),
            };
            let mut out = String::new();
            if !self.bol {
                out.push('\n');
                self.bol = true;
            }
            out.push_str(indent);
            out.push_str(base);
            out.push_str(&render_inline_md(title, base));
            out.push_str("\x1b[0m\n");

            if let Some(ch) = underline_char {
                let len = title.chars().count().max(3).min(80);
                out.push_str(indent);
                out.push_str("\x1b[2m\x1b[36m");
                out.push_str(&std::iter::repeat(ch).take(len).collect::<String>());
                out.push_str("\x1b[0m\n");
            }
            return out;
        }

        if let Some((p_indent, prefix, body)) = split_list_prefix(line) {
            let mut out = String::new();
            out.push_str(p_indent);
            out.push_str("\x1b[36m");
            out.push_str(prefix);
            out.push_str("\x1b[0m");
            out.push_str(&render_inline_md(body, ""));
            out.push('\n');
            return out;
        }

        if line.is_empty() {
            return "\n".to_string();
        }
        format!("{}{}\n", indent, render_inline_md(rest, ""))
    }
}

enum TableState {
    None,
    PendingHeader {
        indent: String,
        header_line: String,
        preview_height: usize,
    },
    InTable {
        indent: String,
        header: Vec<String>,
        align: Vec<TableAlign>,
        rows: Vec<Vec<String>>,
        preview_height: usize,
    },
}

#[derive(Clone, Copy)]
enum TableAlign {
    Left,
    Center,
    Right,
}

fn table_preview_height(line: &str) -> usize {
    let cols = terminal_width().max(1);
    let width = UnicodeWidthStr::width(line);
    let width = width.max(1);
    (width + cols - 1) / cols
}

fn split_indent(s: &str) -> (&str, &str) {
    let mut idx = 0usize;
    for (i, ch) in s.char_indices() {
        if ch == ' ' || ch == '\t' {
            idx = i + ch.len_utf8();
            continue;
        }
        idx = i;
        break;
    }
    if s.chars().all(|c| c == ' ' || c == '\t') {
        return (s, "");
    }
    s.split_at(idx)
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i] == b'#' {
        i += 1;
    }
    if i == 0 || i > 6 {
        return None;
    }
    if i >= bytes.len() || bytes[i] != b' ' {
        return None;
    }
    Some((i, line[i + 1..].trim_end()))
}

fn split_list_prefix(line: &str) -> Option<(&str, &str, &str)> {
    let (indent, rest) = split_indent(line);
    let rest = rest.trim_end();
    if rest.starts_with("- ") || rest.starts_with("* ") || rest.starts_with("+ ") {
        return Some((indent, &rest[..2], &rest[2..]));
    }
    let bytes = rest.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        if i > 4 {
            break;
        }
    }
    if i == 0 || i + 1 >= bytes.len() {
        return None;
    }
    if bytes[i] == b'.' && bytes[i + 1] == b' ' {
        return Some((indent, &rest[..i + 2], &rest[i + 2..]));
    }
    None
}

fn render_inline_md(s: &str, base: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    let mut bold = false;
    let mut code = false;

    fn apply_style(out: &mut String, base: &str, bold: bool, code: bool) {
        out.push_str("\x1b[0m");
        out.push_str(base);
        if bold {
            out.push_str("\x1b[1m");
        }
        if code {
            out.push_str("\x1b[96m");
        }
    }

    while i < bytes.len() {
        if bytes[i] == b'`' {
            code = !code;
            apply_style(&mut out, base, bold, code);
            i += 1;
            continue;
        }

        if !code && bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            bold = !bold;
            apply_style(&mut out, base, bold, code);
            i += 2;
            continue;
        }

        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    out.push_str("\x1b[0m");
    out
}

fn is_table_row_candidate(line: &str) -> bool {
    if line.trim().is_empty() {
        return false;
    }
    let (_, rest) = split_indent(line);
    let s = rest.trim_end();
    if !s.contains('|') {
        return false;
    }
    if s.starts_with("```") || s.starts_with("~~~") {
        return false;
    }
    let cells = parse_table_row(s);
    cells.len() >= 2
}

pub(super) fn line_looks_like_table_preview(line: &str) -> bool {
    if line.trim().is_empty() {
        return false;
    }
    let (_, rest) = split_indent(line);
    let s = rest.trim_end();
    if s.starts_with("```") || s.starts_with("~~~") {
        return false;
    }
    s.contains('|')
}

fn is_table_row(line: &str) -> bool {
    let (_, rest) = split_indent(line);
    let s = rest.trim_end();
    if s.trim().is_empty() {
        return false;
    }
    if is_table_separator(s) {
        return false;
    }
    let cells = parse_table_row(s);
    cells.len() >= 2
}

fn is_table_separator(line: &str) -> bool {
    let (_, rest) = split_indent(line);
    let mut s = rest.trim();
    if s.starts_with('|') {
        s = &s[1..];
    }
    if s.ends_with('|') && s.len() >= 1 {
        s = &s[..s.len() - 1];
    }
    let parts = s.split('|').map(|p| p.trim()).filter(|p| !p.is_empty());
    let mut count = 0usize;
    for p in parts {
        count += 1;
        let p = p.trim_matches(' ');
        let core = p.trim_matches(':');
        if core.len() < 3 || !core.chars().all(|c| c == '-') {
            return false;
        }
    }
    count >= 2
}

fn parse_table_row(line: &str) -> Vec<String> {
    let (_, rest) = split_indent(line);
    let s = rest.trim();
    let mut raw = s.split('|').map(|p| p.trim()).collect::<Vec<_>>();
    if s.starts_with('|') && !raw.is_empty() {
        if raw.first().is_some_and(|x| x.is_empty()) {
            raw.remove(0);
        }
    }
    if s.ends_with('|') && !raw.is_empty() {
        if raw.last().is_some_and(|x| x.is_empty()) {
            raw.pop();
        }
    }
    raw.into_iter().map(|x| x.to_string()).collect()
}

fn parse_table_align(line: &str, cols: usize) -> Vec<TableAlign> {
    let (_, rest) = split_indent(line);
    let s = rest.trim();
    let mut raw = s.split('|').map(|p| p.trim()).collect::<Vec<_>>();
    if s.starts_with('|') && !raw.is_empty() {
        if raw.first().is_some_and(|x| x.is_empty()) {
            raw.remove(0);
        }
    }
    if s.ends_with('|') && !raw.is_empty() {
        if raw.last().is_some_and(|x| x.is_empty()) {
            raw.pop();
        }
    }
    let mut out = Vec::with_capacity(cols);
    for i in 0..cols {
        let seg = raw.get(i).copied().unwrap_or("");
        let seg = seg.trim();
        let left = seg.starts_with(':');
        let right = seg.ends_with(':');
        out.push(match (left, right) {
            (true, true) => TableAlign::Center,
            (false, true) => TableAlign::Right,
            _ => TableAlign::Left,
        });
    }
    out
}

fn strip_inline_md_markers(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            i += 1;
            continue;
        }
        if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            continue;
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn visible_width(s: &str) -> usize {
    UnicodeWidthStr::width(strip_inline_md_markers(s).as_str())
}

fn pad_cell(s: &str, width: usize, align: TableAlign) -> String {
    let w = visible_width(s);
    let pad = width.saturating_sub(w);
    match align {
        TableAlign::Left => format!("{s}{}", " ".repeat(pad)),
        TableAlign::Right => format!("{}{}", " ".repeat(pad), s),
        TableAlign::Center => {
            let left = pad / 2;
            let right = pad - left;
            format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
        }
    }
}

fn wrap_md_cell(s: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    if s.trim().is_empty() {
        return vec![String::new()];
    }

    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut bold = false;
    let mut code = false;
    let mut cur = String::new();
    let mut cur_w = 0usize;
    let mut lines: Vec<String> = Vec::new();

    let start_new_line = |cur: &mut String, cur_w: &mut usize, bold: bool, code: bool| {
        if bold {
            cur.push_str("**");
        }
        if code {
            cur.push('`');
        }
        *cur_w = 0;
    };

    let close_line = |lines: &mut Vec<String>, cur: &mut String, bold: bool, code: bool| {
        if code {
            cur.push('`');
        }
        if bold {
            cur.push_str("**");
        }
        lines.push(std::mem::take(cur));
    };

    start_new_line(&mut cur, &mut cur_w, bold, code);

    while i < bytes.len() {
        if bytes[i] == b'`' {
            code = !code;
            cur.push('`');
            i += 1;
            continue;
        }
        if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            bold = !bold;
            cur.push_str("**");
            i += 2;
            continue;
        }
        let ch = s[i..].chars().next().unwrap();
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if cur_w > 0 && cur_w + w > width {
            close_line(&mut lines, &mut cur, bold, code);
            start_new_line(&mut cur, &mut cur_w, bold, code);
        }
        cur.push(ch);
        cur_w += w;
        i += ch.len_utf8();
    }

    close_line(&mut lines, &mut cur, bold, code);
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn compute_table_widths(indent: &str, header: &[String], rows: &[Vec<String>]) -> Vec<usize> {
    let cols = header
        .len()
        .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
    if cols == 0 {
        return Vec::new();
    }

    let mut widths = vec![3usize; cols];
    for (i, cell) in header.iter().enumerate() {
        widths[i] = widths[i].max(visible_width(cell));
    }
    for row in rows {
        for i in 0..cols {
            let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
            widths[i] = widths[i].max(visible_width(cell));
        }
    }
    for w in &mut widths {
        *w = (*w).max(3);
    }

    let term_cols = terminal_width();
    let indent_w = UnicodeWidthStr::width(indent);
    let max_total = term_cols.saturating_sub(indent_w).max(20);
    let avail = max_total.saturating_sub(3 * cols + 1);
    if avail == 0 {
        return widths;
    }

    let min_w = 3usize;
    let mut sum = widths.iter().sum::<usize>();
    if avail < min_w * cols {
        let base = (avail / cols).max(1);
        let mut rem = avail.saturating_sub(base * cols);
        for w in &mut widths {
            *w = base;
            if rem > 0 {
                *w += 1;
                rem -= 1;
            }
        }
        return widths;
    }

    if sum > avail {
        let mut indices = (0..cols).collect::<Vec<_>>();
        indices.sort_by_key(|&i| std::cmp::Reverse(widths[i]));
        let mut excess = sum - avail;
        while excess > 0 {
            let mut changed = false;
            for &i in &indices {
                if excess == 0 {
                    break;
                }
                if widths[i] > min_w {
                    let reducible = widths[i] - min_w;
                    let delta = reducible.min(excess);
                    widths[i] -= delta;
                    excess -= delta;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            sum = widths.iter().sum::<usize>();
            if sum <= avail {
                break;
            }
        }
    }

    widths
}

fn render_table_top(indent: &str, widths: &[usize]) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(indent);
    out.push('┌');
    for i in 0..cols {
        out.push_str(&"─".repeat(widths[i] + 2));
        out.push(if i + 1 == cols { '┐' } else { '┬' });
    }
    out.push('\n');
    out
}

fn render_table_mid(indent: &str, widths: &[usize]) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(indent);
    out.push('├');
    for i in 0..cols {
        out.push_str(&"─".repeat(widths[i] + 2));
        out.push(if i + 1 == cols { '┤' } else { '┼' });
    }
    out.push('\n');
    out
}

fn render_table_bottom(indent: &str, widths: &[usize]) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(indent);
    out.push('└');
    for i in 0..cols {
        out.push_str(&"─".repeat(widths[i] + 2));
        out.push(if i + 1 == cols { '┘' } else { '┴' });
    }
    out.push('\n');
    out
}

fn render_table_header(
    indent: &str,
    header: &[String],
    align: &[TableAlign],
    widths: &[usize],
) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }

    let header_lines = header
        .iter()
        .enumerate()
        .map(|(i, cell)| wrap_md_cell(cell, *widths.get(i).unwrap_or(&3)))
        .collect::<Vec<_>>();
    let header_height = header_lines.iter().map(|c| c.len()).max().unwrap_or(1);

    let mut out = String::new();
    for line_idx in 0..header_height {
        out.push_str(indent);
        out.push('│');
        for i in 0..cols {
            let cell_line = header_lines
                .get(i)
                .and_then(|ls| ls.get(line_idx))
                .map(|s| s.as_str())
                .unwrap_or("");
            let padded = pad_cell(
                cell_line,
                widths[i],
                align.get(i).copied().unwrap_or(TableAlign::Left),
            );
            out.push(' ');
            out.push_str("\x1b[1m\x1b[36m");
            out.push_str(&render_inline_md(&padded, "\x1b[1m\x1b[36m"));
            out.push_str("\x1b[0m");
            out.push(' ');
            out.push('│');
        }
        out.push('\n');
    }
    out
}

fn render_table_row(
    indent: &str,
    row: &[String],
    align: &[TableAlign],
    widths: &[usize],
) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }

    let wrapped = (0..cols)
        .map(|i| wrap_md_cell(row.get(i).map(|s| s.as_str()).unwrap_or(""), widths[i]))
        .collect::<Vec<_>>();
    let height = wrapped.iter().map(|c| c.len()).max().unwrap_or(1);

    let mut out = String::new();
    for line_idx in 0..height {
        out.push_str(indent);
        out.push('│');
        for i in 0..cols {
            let cell_line = wrapped
                .get(i)
                .and_then(|ls| ls.get(line_idx))
                .map(|s| s.as_str())
                .unwrap_or("");
            let padded = pad_cell(
                cell_line,
                widths[i],
                align.get(i).copied().unwrap_or(TableAlign::Left),
            );
            out.push(' ');
            out.push_str(&render_inline_md(&padded, ""));
            out.push(' ');
            out.push('│');
        }
        out.push('\n');
    }
    out
}

fn terminal_width() -> usize {
    if let Some(cols) = std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        && cols > 0
    {
        return cols;
    }

    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = std::io::stdout().as_raw_fd();
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) };
        if rc == 0 && ws.ws_col > 0 {
            return ws.ws_col as usize;
        }
    }

    80
}
