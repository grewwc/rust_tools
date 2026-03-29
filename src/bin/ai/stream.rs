use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};

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
    let mut assistant_text = String::new();
    let mut internal_tool_call_idx: usize = 0;

    while !app.shutdown.load(std::sync::atomic::Ordering::SeqCst) {
        if take_stream_cancelled(app) {
            return Ok(StreamResult {
                outcome: StreamOutcome::Cancelled,
                tool_calls: Vec::new(),
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
                        assistant_text: String::new(),
                    });
                }
                if app.shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                    return Ok(StreamResult {
                        outcome: StreamOutcome::Cancelled,
                        tool_calls: Vec::new(),
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

        let mut reached_finish_reason = false;
        if let Some(choice) = chunk.choices.first() {
            reached_finish_reason = choice.finish_reason.is_some();

            for stream_tool_call in &choice.delta.tool_calls {
                let index = stream_tool_call.index;
                let builder = tool_calls_map.entry(index).or_default();

                if !stream_tool_call.id.is_empty() {
                    builder.id = stream_tool_call.id.clone();
                }
                if !stream_tool_call.tool_type.is_empty() {
                    builder.tool_type = stream_tool_call.tool_type.clone();
                }
                if !stream_tool_call.function.name.is_empty() {
                    builder.function_name = stream_tool_call.function.name.clone();
                }
                builder
                    .arguments
                    .push_str(&stream_tool_call.function.arguments);
            }
        }

        let (content, internal_tool_calls) = extract_chunk_text_with_tools(
            &chunk,
            &thinking_tag,
            &end_thinking_tag,
            &mut thinking_open,
        );
        
        for tc in internal_tool_calls {
            let builder = tool_calls_map.entry(internal_tool_call_idx).or_default();
            builder.id = tc.id;
            builder.tool_type = tc.tool_type;
            builder.function_name = tc.function_name;
            builder.arguments = tc.arguments;
            internal_tool_call_idx += 1;
        }
        
        if content.is_empty() {
            if reached_finish_reason {
                break;
            }
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

        // Some providers may not send a trailing [DONE] frame promptly.
        // If finish_reason is present, we already have the terminal chunk.
        if reached_finish_reason {
            break;
        }
    }

    // If the stream ended (cut or [DONE]) while still inside a thinking block,
    // close it cleanly so the terminal isn't left hanging open.
    if thinking_open {
        write_stream_content(
            &format!("\n{end_thinking_tag}\n"),
            app.writer.as_mut(),
            &mut markdown,
        )?;
    }

    markdown.flush_pending()?;

    if take_stream_cancelled(app) {
        return Ok(StreamResult {
            outcome: StreamOutcome::Cancelled,
            tool_calls: Vec::new(),
            assistant_text: String::new(),
        });
    }

    let tool_calls: Vec<ToolCall> = tool_calls_map
        .into_values()
        .map(|builder| builder.build())
        .collect();

    let outcome = if !tool_calls.is_empty() {
        StreamOutcome::ToolCall
    } else {
        StreamOutcome::Completed
    };

    Ok(StreamResult {
        outcome,
        tool_calls,
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
    let (content, _) = extract_chunk_text_with_tools(chunk, thinking_tag, end_thinking_tag, thinking_open);
    content
}

struct InternalToolCall {
    id: String,
    tool_type: String,
    function_name: String,
    arguments: String,
}

fn extract_chunk_text_with_tools(
    chunk: &StreamChunk,
    thinking_tag: &str,
    end_thinking_tag: &str,
    thinking_open: &mut bool,
) -> (String, Vec<InternalToolCall>) {
    let Some(choice) = chunk.choices.first() else {
        return (String::new(), Vec::new());
    };
    let delta = &choice.delta;

    if delta.content.is_empty() && !delta.reasoning_content.is_empty() {
        let (cleaned, tool_calls) = extract_internal_tool_calls(&delta.reasoning_content);
        if cleaned.is_empty() && tool_calls.is_empty() {
            return (String::new(), Vec::new());
        }
        let dimmed = cleaned.dimmed().to_string();
        if !*thinking_open {
            *thinking_open = true;
            return (format!("\n{thinking_tag}\n{dimmed}"), tool_calls);
        }
        return (dimmed, tool_calls);
    }

    if *thinking_open {
        *thinking_open = false;
        return (format!("\n{end_thinking_tag}\n{}", delta.content), Vec::new());
    }
    (delta.content.clone(), Vec::new())
}

fn extract_internal_tool_calls(s: &str) -> (String, Vec<InternalToolCall>) {
    let mut result = String::with_capacity(s.len());
    let mut tool_calls = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            let marker_start = i;
            let mut marker_end = i + 2;
            while marker_end < bytes.len() {
                if bytes[marker_end] == b'|' && marker_end + 1 < bytes.len() && bytes[marker_end + 1] == b'>' {
                    break;
                }
                marker_end += 1;
            }
            
            let marker = &s[marker_start..marker_end + 2];
            
            if marker == "<|tool_call_begin|>" {
                let (name, consumed) = parse_tool_call_name(s, marker_end + 2);
                if let Some(name) = name {
                    let mut tc = InternalToolCall {
                        id: format!("internal_{}", tool_calls.len()),
                        tool_type: "function".to_string(),
                        function_name: name,
                        arguments: String::new(),
                    };
                    
                    let remaining_start = marker_end + 2 + consumed;
                    if let Some((args, args_consumed)) = parse_tool_call_args(s, remaining_start) {
                        tc.arguments = args;
                        i = remaining_start + args_consumed;
                    } else {
                        i = remaining_start;
                    }
                    
                    tool_calls.push(tc);
                    continue;
                }
            } else if marker == "<|tool_call_end|>" || marker == "<|tool_calls_section_end|>" {
                i = marker_end + 2;
                continue;
            } else if marker == "<|tool_call_argument_begin|>" {
                i = marker_end + 2;
                continue;
            }
            
            i = marker_end + 2;
            continue;
        }
        
        let ch = s[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }
    
    (result, tool_calls)
}

fn parse_tool_call_name(s: &str, start: usize) -> (Option<String>, usize) {
    let bytes = s.as_bytes();
    let mut i = start;
    let mut name = String::new();
    
    while i < bytes.len() {
        let ch = s[i..].chars().next().unwrap();
        if ch == '<' || ch == '{' {
            break;
        }
        name.push(ch);
        i += ch.len_utf8();
    }
    
    let name = name.trim().to_string();
    if name.is_empty() {
        (None, 0)
    } else {
        (Some(name), i - start)
    }
}

fn parse_tool_call_args(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    let mut i = start;
    
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t') {
        i += 1;
    }
    
    if i >= bytes.len() || bytes[i] != b'{' {
        return None;
    }
    
    let json_start = i;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;
    
    while i < bytes.len() {
        let b = bytes[i];
        
        if escape {
            escape = false;
            i += 1;
            continue;
        }
        
        match b {
            b'\\' if in_string => escape = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    i += 1;
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    
    let json_str = s[json_start..i].to_string();
    Some((json_str, i - start))
}

fn strip_ansi_codes(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut result = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            i += 2;
            while i < bytes.len() {
                let b = bytes[i];
                i += 1;
                if (b as char) >= '@' && (b as char) <= '~' {
                    break;
                }
            }
            continue;
        }
        let ch = s[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }
    result
}

fn write_stream_content(
    content: &str,
    mut writer: Option<&mut std::fs::File>,
    markdown: &mut MarkdownStreamRenderer,
) -> io::Result<()> {
    if let Some(file) = writer.as_mut() {
        let clean = strip_ansi_codes(content);
        file.write_all(clean.as_bytes())?;
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
    in_math_block: bool,
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
            in_math_block: false,
            bol: false,
            line_buf: String::new(),
            line_preview_emitted: false,
            line_preview_height: 0,
            table_state: TableState::None,
        }
    }

    fn should_render(&mut self, _chunk: &str) -> bool {
        if !self.tty {
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
        let preview_height = preview_height_for_rendered(&rendered).max(1);
        out.write_all(rendered.as_bytes())?;
        self.line_preview_emitted = true;
        self.line_preview_height = preview_height;
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
                    let preview_height = self.line_preview_height.max(1);
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
            let row_cells = row.to_vec();
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

        if trimmed == "$$" || trimmed == "\\[" || trimmed == "\\]" {
            self.in_math_block = !self.in_math_block;
            return "\n".to_string();
        }

        if self.in_math_block {
            if line.is_empty() {
                return "\n".to_string();
            }
            let math = render_math_tex_to_unicode(rest.trim_end());
            return format!("{indent}\x1b[95m{math}\x1b[0m\n");
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
                let len = title.chars().count().clamp(3, 80);
                out.push_str(indent);
                out.push_str("\x1b[2m\x1b[36m");
                out.push_str(&std::iter::repeat_n(ch, len).collect::<String>());
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
    width.div_ceil(cols)
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
    let mut math = false;
    let mut math_delim = "$";
    let mut math_buf = String::new();

    fn apply_style(out: &mut String, base: &str, bold: bool, code: bool, math: bool) {
        out.push_str("\x1b[0m");
        out.push_str(base);
        if bold {
            out.push_str("\x1b[1m");
        }
        if code {
            out.push_str("\x1b[96m");
        }
        if math {
            out.push_str("\x1b[95m");
        }
    }

    fn is_url_start(bytes: &[u8], i: usize) -> bool {
        bytes
            .get(i..i + 8)
            .is_some_and(|s| s.eq_ignore_ascii_case(b"https://"))
            || bytes
                .get(i..i + 7)
                .is_some_and(|s| s.eq_ignore_ascii_case(b"http://"))
    }

    fn url_raw_end(bytes: &[u8], start: usize) -> usize {
        let mut end = start;
        while end < bytes.len() {
            let b = bytes[end];
            if b.is_ascii_whitespace()
                || b == b'<'
                || b == b'"'
                || b == b'\''
                || b == b'`'
                || b == b'\\'
            {
                break;
            }
            end += 1;
        }
        end
    }

    while i < bytes.len() {
        if bytes[i] == b'`' {
            code = !code;
            apply_style(&mut out, base, bold, code, math);
            i += 1;
            continue;
        }

        if !code && !math && bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            bold = !bold;
            apply_style(&mut out, base, bold, code, math);
            i += 2;
            continue;
        }

        if !code && bytes[i] == b'$' {
            let is_double = i + 1 < bytes.len() && bytes[i + 1] == b'$';
            let delim = if is_double { "$$" } else { "$" };

            if math {
                if delim == math_delim {
                    let rendered = render_math_tex_to_unicode(math_buf.trim());
                    out.push_str(&rendered);
                    math_buf.clear();
                    math = false;
                    apply_style(&mut out, base, bold, code, math);
                    i += delim.len();
                    continue;
                }
            } else {
                math = true;
                math_delim = delim;
                apply_style(&mut out, base, bold, code, math);
                i += delim.len();
                continue;
            }
        }

        if !math && is_url_start(bytes, i) {
            let raw_end = url_raw_end(bytes, i);
            let mut end = raw_end;
            while end > i {
                match bytes[end - 1] {
                    b'.' | b',' | b';' | b':' | b')' | b']' => end -= 1,
                    _ => break,
                }
            }
            let url = &s[i..end];
            let trail = &s[end..raw_end];

            out.push_str("\x1b[0m");
            out.push_str(base);
            if bold {
                out.push_str("\x1b[1m");
            }
            out.push_str("\x1b[4m\x1b[34m");
            out.push_str(url);
            apply_style(&mut out, base, bold, code, math);
            out.push_str(trail);

            i = raw_end;
            continue;
        }

        let ch = s[i..].chars().next().unwrap();
        if math && !code {
            math_buf.push(ch);
        } else {
            out.push(ch);
        }
        i += ch.len_utf8();
    }

    if math && !math_buf.is_empty() {
        out.push_str(&render_math_tex_to_unicode(math_buf.trim()));
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
    if s.ends_with('|') && !s.is_empty() {
        s = &s[..s.len() - 1];
    }
    let parts = split_table_segments(s)
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty());
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
    let mut raw = split_table_segments(s);
    if s.starts_with('|') && !raw.is_empty() && raw.first().is_some_and(|x| x.is_empty()) {
        raw.remove(0);
    }
    if s.ends_with('|') && !raw.is_empty() && raw.last().is_some_and(|x| x.is_empty()) {
        raw.pop();
    }
    raw.into_iter().map(|x| x.trim().to_string()).collect()
}

fn parse_table_align(line: &str, cols: usize) -> Vec<TableAlign> {
    let (_, rest) = split_indent(line);
    let s = rest.trim();
    let mut raw = split_table_segments(s);
    if s.starts_with('|') && !raw.is_empty() && raw.first().is_some_and(|x| x.is_empty()) {
        raw.remove(0);
    }
    if s.ends_with('|') && !raw.is_empty() && raw.last().is_some_and(|x| x.is_empty()) {
        raw.pop();
    }
    let mut out = Vec::with_capacity(cols);
    for seg in raw
        .iter()
        .map(|s| s.as_str())
        .chain(std::iter::repeat(""))
        .take(cols)
    {
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

fn split_table_segments(s: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut in_code = false;
    let mut in_math = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }

        if ch == '\\' {
            current.push(ch);
            escaped = true;
            continue;
        }

        if ch == '`' && !in_math {
            in_code = !in_code;
            current.push(ch);
            continue;
        }

        if ch == '$' && !in_code {
            if chars.peek().copied() == Some('$') {
                chars.next();
                in_math = !in_math;
                current.push('$');
                current.push('$');
                continue;
            }

            in_math = !in_math;
            current.push(ch);
            continue;
        }

        if ch == '|' && !in_code && !in_math {
            segments.push(std::mem::take(&mut current));
            continue;
        }

        current.push(ch);
    }

    segments.push(current);
    segments
}

fn strip_inline_md_markers(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    let mut code = false;
    let mut math = false;
    let mut math_delim = "$";
    let mut math_buf = String::new();
    while i < bytes.len() {
        if bytes[i] == b'`' {
            code = !code;
            i += 1;
            continue;
        }
        if !code && bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            continue;
        }
        if !code && bytes[i] == b'$' {
            let is_double = i + 1 < bytes.len() && bytes[i + 1] == b'$';
            let delim = if is_double { "$$" } else { "$" };
            if math {
                if delim == math_delim {
                    out.push_str(&render_math_tex_to_unicode(math_buf.trim()));
                    math_buf.clear();
                    math = false;
                    i += delim.len();
                    continue;
                }
            } else {
                math = true;
                math_delim = delim;
                i += delim.len();
                continue;
            }
        }
        let ch = s[i..].chars().next().unwrap();
        if math && !code {
            math_buf.push(ch);
        } else {
            out.push(ch);
        }
        i += ch.len_utf8();
    }
    if math && !math_buf.is_empty() {
        out.push_str(&render_math_tex_to_unicode(math_buf.trim()));
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

    let mut bold = false;
    let mut cur = String::new();
    let mut cur_w = 0usize;
    let mut lines: Vec<String> = Vec::new();

    let start_new_line = |cur: &mut String, cur_w: &mut usize, bold: bool| {
        if bold {
            cur.push_str("**");
        }
        *cur_w = 0;
    };

    let close_line = |lines: &mut Vec<String>, cur: &mut String, bold: bool| {
        if bold {
            cur.push_str("**");
        }
        lines.push(std::mem::take(cur));
    };

    let mut i = 0usize;
    start_new_line(&mut cur, &mut cur_w, bold);

    while i < s.len() {
        let rest = &s[i..];

        if rest.starts_with("**") {
            bold = !bold;
            cur.push_str("**");
            i += 2;
            continue;
        }

        if let Some((piece, next)) = take_atomic_markdown_span(s, i) {
            let piece_width = visible_width(&piece);
            if cur_w > 0 && cur_w + piece_width > width {
                close_line(&mut lines, &mut cur, bold);
                start_new_line(&mut cur, &mut cur_w, bold);
            }
            cur.push_str(&piece);
            cur_w += piece_width;
            i = next;
            continue;
        }

        let ch = rest.chars().next().unwrap();
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if cur_w > 0 && cur_w + w > width {
            close_line(&mut lines, &mut cur, bold);
            start_new_line(&mut cur, &mut cur_w, bold);
        }
        cur.push(ch);
        cur_w += w;
        i += ch.len_utf8();
    }

    close_line(&mut lines, &mut cur, bold);
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn take_atomic_markdown_span(s: &str, start: usize) -> Option<(String, usize)> {
    let rest = &s[start..];

    if rest.starts_with('`') {
        let end = find_unescaped_delim(s, start + 1, "`")?;
        return Some((s[start..end].to_string(), end));
    }

    if rest.starts_with("$$") {
        let end = find_unescaped_delim(s, start + 2, "$$")?;
        return Some((s[start..end].to_string(), end));
    }

    if rest.starts_with('$') {
        let end = find_unescaped_delim(s, start + 1, "$")?;
        return Some((s[start..end].to_string(), end));
    }

    if let Some(stripped) = rest.strip_prefix('\\') {
        let next = stripped.chars().next()?;
        let end = start + 1 + next.len_utf8();
        return Some((s[start..end].to_string(), end));
    }

    None
}

fn find_unescaped_delim(s: &str, mut i: usize, delim: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    while i < bytes.len() {
        if s[i..].starts_with(delim) && !is_escaped_at(s, i) {
            return Some(i + delim.len());
        }
        let ch = s[i..].chars().next()?;
        i += ch.len_utf8();
    }
    None
}

fn is_escaped_at(s: &str, idx: usize) -> bool {
    if idx == 0 {
        return false;
    }

    let mut backslashes = 0usize;
    let mut i = idx;
    while i > 0 {
        let prev = s[..i].chars().next_back().unwrap();
        if prev != '\\' {
            break;
        }
        backslashes += 1;
        i -= prev.len_utf8();
    }
    backslashes % 2 == 1
}

fn render_math_tex_to_unicode(s: &str) -> String {
    use regex::Regex;
    use std::sync::LazyLock;

    static RE_MATHBB: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\\mathbb\{([A-Za-z])\}").unwrap());

    let mut t = s.to_string();

    t = t.replace("\\left", "");
    t = t.replace("\\right", "");
    t = t.replace("\\,", " ");
    t = t.replace("\\;", " ");
    t = t.replace("\\:", " ");
    t = t.replace("\\!", "");
    t = t.replace("\\ ", " ");

    fn read_group_braced(s: &str, start: usize) -> Option<(String, usize)> {
        let bytes = s.as_bytes();
        if start >= bytes.len() || bytes[start] != b'{' {
            return None;
        }
        let mut i = start + 1;
        let mut depth = 1usize;
        let mut out = String::new();
        while i < bytes.len() {
            let ch = match s.get(i..) {
                Some(rest) => match rest.chars().next() {
                    Some(ch) => ch,
                    None => break,
                },
                None => break,
            };
            i += ch.len_utf8();
            match ch {
                '{' => {
                    depth += 1;
                    out.push(ch);
                }
                '}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some((out, i));
                    }
                    out.push(ch);
                }
                _ => out.push(ch),
            }
        }
        None
    }

    fn read_group_bracketed(s: &str, start: usize) -> Option<(String, usize)> {
        let bytes = s.as_bytes();
        if start >= bytes.len() || bytes[start] != b'[' {
            return None;
        }
        let mut i = start + 1;
        let mut depth = 1usize;
        let mut out = String::new();
        while i < bytes.len() {
            let ch = match s.get(i..) {
                Some(rest) => match rest.chars().next() {
                    Some(ch) => ch,
                    None => break,
                },
                None => break,
            };
            i += ch.len_utf8();
            match ch {
                '[' => {
                    depth += 1;
                    out.push(ch);
                }
                ']' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some((out, i));
                    }
                    out.push(ch);
                }
                _ => out.push(ch),
            }
        }
        None
    }

    fn needs_parens(s: &str) -> bool {
        let s = s.trim();
        if s.is_empty() {
            return false;
        }
        if s.starts_with('-') {
            return true;
        }
        if s.chars().count() <= 1 {
            return false;
        }
        for ch in s.chars() {
            if ch.is_whitespace() {
                return true;
            }
            if matches!(
                ch,
                '+' | '-' | '*' | '/' | '=' | '±' | '∓' | '×' | '·' | '÷' | '→' | '←' | '↔'
            ) {
                return true;
            }
        }
        false
    }

    fn wrap_parens(s: &str) -> String {
        let s = s.trim();
        if needs_parens(s) {
            format!("({s})")
        } else {
            s.to_string()
        }
    }

    fn replace_structural_tex(mut s: String) -> String {
        let mut changed = true;
        while changed {
            changed = false;
            let bytes = s.as_bytes();
            let mut out = String::with_capacity(s.len());
            let mut i = 0usize;
            while i < bytes.len() {
                if s[i..].starts_with("\\frac") {
                    let mut j = i + "\\frac".len();
                    while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                        j += 1;
                    }
                    if let Some((num, j2)) = read_group_braced(&s, j) {
                        let mut k = j2;
                        while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t') {
                            k += 1;
                        }
                        if let Some((den, k2)) = read_group_braced(&s, k) {
                            let num = replace_structural_tex(num);
                            let den = replace_structural_tex(den);
                            let num = wrap_parens(&num);
                            let den = wrap_parens(&den);
                            out.push_str(&format!("{num}/{den}"));
                            i = k2;
                            changed = true;
                            continue;
                        }
                    }
                }
                if s[i..].starts_with("\\sqrt") {
                    let mut j = i + "\\sqrt".len();
                    while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                        j += 1;
                    }
                    if j < bytes.len()
                        && bytes[j] == b'['
                        && let Some((_, j2)) = read_group_bracketed(&s, j)
                    {
                        j = j2;
                        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                            j += 1;
                        }
                    }
                    if let Some((rad, j2)) = read_group_braced(&s, j) {
                        let rad = replace_structural_tex(rad);
                        let rad = rad.trim();
                        out.push_str(&format!("√({rad})"));
                        i = j2;
                        changed = true;
                        continue;
                    }
                }
                let ch = s[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
            }
            s = out;
        }
        s
    }

    t = replace_structural_tex(t);

    t = t.replace("\\times", "×");
    t = t.replace("\\cdot", "·");
    t = t.replace("\\div", "÷");
    t = t.replace("\\pm", "±");
    t = t.replace("\\mp", "∓");

    t = t.replace("\\leq", "≤");
    t = t.replace("\\geq", "≥");
    t = t.replace("\\neq", "≠");
    t = t.replace("\\approx", "≈");
    t = t.replace("\\equiv", "≡");
    t = t.replace("\\to", "→");
    t = t.replace("\\rightarrow", "→");
    t = t.replace("\\leftarrow", "←");
    t = t.replace("\\leftrightarrow", "↔");

    t = t.replace("\\infty", "∞");
    t = t.replace("\\sum", "∑");
    t = t.replace("\\prod", "∏");
    t = t.replace("\\int", "∫");

    t = t.replace("\\in", "∈");
    t = t.replace("\\notin", "∉");
    t = t.replace("\\subset", "⊂");
    t = t.replace("\\subseteq", "⊆");
    t = t.replace("\\supset", "⊃");
    t = t.replace("\\supseteq", "⊇");
    t = t.replace("\\cup", "∪");
    t = t.replace("\\cap", "∩");

    t = t.replace("\\alpha", "α");
    t = t.replace("\\beta", "β");
    t = t.replace("\\gamma", "γ");
    t = t.replace("\\delta", "δ");
    t = t.replace("\\epsilon", "ε");
    t = t.replace("\\zeta", "ζ");
    t = t.replace("\\eta", "η");
    t = t.replace("\\theta", "θ");
    t = t.replace("\\iota", "ι");
    t = t.replace("\\kappa", "κ");
    t = t.replace("\\lambda", "λ");
    t = t.replace("\\mu", "μ");
    t = t.replace("\\nu", "ν");
    t = t.replace("\\xi", "ξ");
    t = t.replace("\\pi", "π");
    t = t.replace("\\rho", "ρ");
    t = t.replace("\\sigma", "σ");
    t = t.replace("\\tau", "τ");
    t = t.replace("\\upsilon", "υ");
    t = t.replace("\\phi", "φ");
    t = t.replace("\\chi", "χ");
    t = t.replace("\\psi", "ψ");
    t = t.replace("\\omega", "ω");

    t = t.replace("\\Gamma", "Γ");
    t = t.replace("\\Delta", "Δ");
    t = t.replace("\\Theta", "Θ");
    t = t.replace("\\Lambda", "Λ");
    t = t.replace("\\Xi", "Ξ");
    t = t.replace("\\Pi", "Π");
    t = t.replace("\\Sigma", "Σ");
    t = t.replace("\\Phi", "Φ");
    t = t.replace("\\Psi", "Ψ");
    t = t.replace("\\Omega", "Ω");

    t = RE_MATHBB
        .replace_all(&t, |caps: &regex::Captures| {
            let v = &caps[1];
            match v {
                "R" => "ℝ".to_string(),
                "N" => "ℕ".to_string(),
                "Z" => "ℤ".to_string(),
                "Q" => "ℚ".to_string(),
                "C" => "ℂ".to_string(),
                other => other.to_string(),
            }
        })
        .to_string();

    t = t.replace("\\_", "_");
    t = t.replace("\\{", "{");
    t = t.replace("\\}", "}");

    t = apply_super_subscripts(&t);
    t = t.replace('{', "");
    t = t.replace('}', "");

    t
}

fn apply_super_subscripts(s: &str) -> String {
    fn map_sup(ch: char) -> Option<char> {
        match ch {
            '0' => Some('⁰'),
            '1' => Some('¹'),
            '2' => Some('²'),
            '3' => Some('³'),
            '4' => Some('⁴'),
            '5' => Some('⁵'),
            '6' => Some('⁶'),
            '7' => Some('⁷'),
            '8' => Some('⁸'),
            '9' => Some('⁹'),
            '+' => Some('⁺'),
            '-' => Some('⁻'),
            '=' => Some('⁼'),
            '(' => Some('⁽'),
            ')' => Some('⁾'),
            'n' => Some('ⁿ'),
            'i' => Some('ⁱ'),
            _ => None,
        }
    }

    fn map_sub(ch: char) -> Option<char> {
        match ch {
            '0' => Some('₀'),
            '1' => Some('₁'),
            '2' => Some('₂'),
            '3' => Some('₃'),
            '4' => Some('₄'),
            '5' => Some('₅'),
            '6' => Some('₆'),
            '7' => Some('₇'),
            '8' => Some('₈'),
            '9' => Some('₉'),
            '+' => Some('₊'),
            '-' => Some('₋'),
            '=' => Some('₌'),
            '(' => Some('₍'),
            ')' => Some('₎'),
            'a' => Some('ₐ'),
            'e' => Some('ₑ'),
            'h' => Some('ₕ'),
            'i' => Some('ᵢ'),
            'j' => Some('ⱼ'),
            'k' => Some('ₖ'),
            'l' => Some('ₗ'),
            'm' => Some('ₘ'),
            'n' => Some('ₙ'),
            'o' => Some('ₒ'),
            'p' => Some('ₚ'),
            'r' => Some('ᵣ'),
            's' => Some('ₛ'),
            't' => Some('ₜ'),
            'u' => Some('ᵤ'),
            'v' => Some('ᵥ'),
            'x' => Some('ₓ'),
            _ => None,
        }
    }

    fn read_braced(s: &str, start: usize) -> Option<(String, usize)> {
        let bytes = s.as_bytes();
        if start >= bytes.len() || bytes[start] != b'{' {
            return None;
        }
        let mut i = start + 1;
        let mut depth = 1usize;
        let mut out = String::new();
        while i < bytes.len() {
            let ch = s[i..].chars().next().unwrap();
            i += ch.len_utf8();
            match ch {
                '{' => {
                    depth += 1;
                    out.push(ch);
                }
                '}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some((out, i));
                    }
                    out.push(ch);
                }
                _ => out.push(ch),
            }
        }
        None
    }

    fn convert_group(group: &str, sup: bool) -> Option<String> {
        let mut out = String::new();
        for ch in group.chars() {
            let mapped = if sup { map_sup(ch) } else { map_sub(ch) }?;
            out.push(mapped);
        }
        Some(out)
    }

    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let ch = s[i..].chars().next().unwrap();
        if ch == '^' || ch == '_' {
            let sup = ch == '^';
            i += ch.len_utf8();
            if i >= bytes.len() {
                out.push(ch);
                break;
            }
            if bytes[i] == b'{'
                && let Some((group, next)) = read_braced(s, i)
            {
                if let Some(converted) = convert_group(group.trim(), sup) {
                    out.push_str(&converted);
                } else {
                    out.push(if sup { '^' } else { '_' });
                    out.push('(');
                    out.push_str(group.trim());
                    out.push(')');
                }
                i = next;
                continue;
            }
            let next_ch = s[i..].chars().next().unwrap();
            if let Some(mapped) = if sup {
                map_sup(next_ch)
            } else {
                map_sub(next_ch)
            } {
                out.push(mapped);
            } else {
                out.push(if sup { '^' } else { '_' });
                out.push(next_ch);
            }
            i += next_ch.len_utf8();
            continue;
        }
        out.push(ch);
        i += ch.len_utf8();
    }
    out
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
        for (width, cell) in widths
            .iter_mut()
            .zip(row.iter().map(|s| s.as_str()).chain(std::iter::repeat("")))
        {
            *width = (*width).max(visible_width(cell));
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
    for (i, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(*width + 2));
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
    for (i, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(*width + 2));
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
    for (i, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(*width + 2));
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
        for (i, width) in widths.iter().enumerate() {
            let cell_line = header_lines
                .get(i)
                .and_then(|ls| ls.get(line_idx))
                .map(|s| s.as_str())
                .unwrap_or("");
            let padded = pad_cell(
                cell_line,
                *width,
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

    let wrapped = widths
        .iter()
        .enumerate()
        .map(|(i, width)| wrap_md_cell(row.get(i).map(|s| s.as_str()).unwrap_or(""), *width))
        .collect::<Vec<_>>();
    let height = wrapped.iter().map(|c| c.len()).max().unwrap_or(1);

    let mut out = String::new();
    for line_idx in 0..height {
        out.push_str(indent);
        out.push('│');
        for (i, width) in widths.iter().enumerate() {
            let cell_line = wrapped
                .get(i)
                .and_then(|ls| ls.get(line_idx))
                .map(|s| s.as_str())
                .unwrap_or("");
            let padded = pad_cell(
                cell_line,
                *width,
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

fn preview_height_for_rendered(rendered: &str) -> usize {
    let cols = terminal_width().max(1);
    let mut plain = String::new();
    let bytes = rendered.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            i += 2;
            while i < bytes.len() {
                let b = bytes[i];
                i += 1;
                if (b as char) >= '@' && (b as char) <= '~' {
                    break;
                }
            }
            continue;
        }
        let ch = rendered[i..].chars().next().unwrap();
        if ch != '\r' {
            plain.push(ch);
        }
        i += ch.len_utf8();
    }
    let width = UnicodeWidthStr::width(plain.as_str()).max(1);
    width.div_ceil(cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_preview_height_uses_rendered_width_not_raw_markdown() {
        unsafe { std::env::set_var("COLUMNS", "6") };
        let raw = "**hello**";
        assert_eq!(table_preview_height(raw), 2);
        let rendered = render_inline_md(raw, "");
        assert_eq!(preview_height_for_rendered(&rendered), 1);
    }

    #[test]
    fn consume_line_move_up_matches_preview_height() {
        unsafe { std::env::set_var("COLUMNS", "6") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.line_preview_height = 1;
        let out = renderer.consume_line("**hello**", true);
        assert!(out.contains("\x1b[1A\r\x1b[0J"));
        assert!(!out.contains("\x1b[2A\r\x1b[0J"));
    }

    #[test]
    fn parse_table_row_ignores_embedded_pipes() {
        assert_eq!(
            parse_table_row(r#"| `a|b` | x \| y | $p|q$ |"#),
            vec!["`a|b`", r#"x \| y"#, "$p|q$"]
        );
    }

    #[test]
    fn wrap_md_cell_uses_visible_width_for_math_and_code_spans() {
        let math = wrap_md_cell(r#"$\frac{1}{2}$"#, 5);
        assert_eq!(math, vec![r#"$\frac{1}{2}$"#]);

        let code = wrap_md_cell(r#"`a|b`"#, 3);
        assert_eq!(code, vec![r#"`a|b`"#]);
    }

    #[test]
    fn compute_table_widths_does_not_add_columns_for_embedded_pipes() {
        let header = parse_table_row("| name | value |");
        let rows = vec![parse_table_row(r#"| `a|b` | $\frac{1}{2}$ |"#)];
        let widths = compute_table_widths("", &header, &rows);

        assert_eq!(widths.len(), 2);
        assert!(widths[0] >= visible_width("`a|b`"));
        assert!(widths[1] >= visible_width(r#"$\frac{1}{2}$"#));
    }
}
