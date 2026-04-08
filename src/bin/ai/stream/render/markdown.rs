use std::io::{self, Write};

use crate::ai::stream::render::code::{
    MONOKAI_BG, MONOKAI_COMMENT, highlight_code_line, parse_code_block_language,
};
use crate::ai::stream::render::inline::render_inline_md;
use crate::ai::stream::render::table::{
    TableAlign, TableState, is_table_row, is_table_row_candidate, is_table_separator,
    line_looks_like_table_preview, parse_table_align, parse_table_row, render_table_bottom,
    render_table_header, render_table_mid, render_table_row, render_table_top, split_indent,
    table_preview_height, compute_table_widths,
};

pub(in crate::ai) struct MarkdownStreamRenderer {
    tty: bool,
    enabled: bool,
    in_code_block: bool,
    code_block_lang: Option<String>,
    in_math_block: bool,
    bol: bool,
    line_buf: String,
    line_preview_emitted: bool,
    line_preview_height: usize,
    table_state: TableState,
    dimmed: bool,
}

impl MarkdownStreamRenderer {
    pub(in crate::ai::stream) fn new() -> Self {
        use std::io::IsTerminal;
        Self::new_with_tty(io::stdout().is_terminal())
    }

    pub(in crate::ai) fn new_with_tty(tty: bool) -> Self {
        Self {
            tty,
            enabled: true,
            in_code_block: false,
            code_block_lang: None,
            in_math_block: false,
            bol: false,
            line_buf: String::new(),
            line_preview_emitted: false,
            line_preview_height: 0,
            table_state: TableState::None,
            dimmed: false,
        }
    }

    pub(in crate::ai::stream) fn should_render(&mut self, _chunk: &str) -> bool {
        if !self.tty {
            return false;
        }
        self.enabled = true;
        true
    }

    pub(in crate::ai::stream) fn write_chunk(&mut self, chunk: &str, dimmed: bool) -> io::Result<()> {
        self.dimmed = dimmed;
        let mut out = io::stdout();
        for ch in chunk.chars() {
            if ch == '\n' {
                self.handle_newline(&mut out)?;
                continue;
            }

            self.line_buf.push(ch);
            self.handle_char(&mut out, ch)?;
            self.bol = false;
        }
        out.flush()?;
        Ok(())
    }

    pub(in crate::ai::stream::render) fn line_preview_height(&self) -> usize {
        self.line_preview_height
    }

    pub(in crate::ai::stream::render) fn set_line_preview_height(&mut self, height: usize) {
        self.line_preview_height = height;
    }

    pub(in crate::ai::stream::render) fn code_block_lang(&self) -> Option<&str> {
        self.code_block_lang.as_deref()
    }

    fn handle_newline(&mut self, out: &mut std::io::Stdout) -> io::Result<()> {
        if self.line_preview_emitted {
            out.write_all(b"\n")?;
            out.flush()?;
            self.bol = true;
        }

        let line = std::mem::take(&mut self.line_buf);
        let rendered = self.consume_line(&line, self.line_preview_emitted);

        self.line_preview_emitted = false;
        self.line_preview_height = 0;

        if !rendered.is_empty() {
            out.write_all(rendered.as_bytes())?;
            out.flush()?;
            self.bol = rendered.ends_with('\n');
        }
        Ok(())
    }

    fn handle_char(&mut self, out: &mut std::io::Stdout, ch: char) -> io::Result<()> {
        if self.should_emit_table_preview_live() {
            self.handle_table_preview(out, ch)
        } else {
            self.handle_realtime_output(out, ch)
        }
    }

    fn handle_table_preview(&mut self, out: &mut std::io::Stdout, ch: char) -> io::Result<()> {
        if !self.line_preview_emitted {
            if self.dimmed {
                out.write_all(b"\x1b[2m")?;
            }
            out.write_all(self.line_buf.as_bytes())?;
            out.flush()?;
            self.line_preview_emitted = true;
            self.line_preview_height = table_preview_height(&self.line_buf).max(1);
        } else {
            self.emit_char(out, ch)?;
            self.line_preview_height = table_preview_height(&self.line_buf).max(1);
        }
        Ok(())
    }

    fn handle_realtime_output(&mut self, out: &mut std::io::Stdout, ch: char) -> io::Result<()> {
        if self.line_buf.chars().count() == 1 && self.dimmed {
            out.write_all(b"\x1b[2m")?;
        }
        self.emit_char(out, ch)?;
        self.line_preview_emitted = true;
        self.line_preview_height = table_preview_height(&self.line_buf).max(1);
        Ok(())
    }

    fn emit_char(&mut self, out: &mut std::io::Stdout, ch: char) -> io::Result<()> {
        let mut buf = [0u8; 4];
        out.write_all(ch.encode_utf8(&mut buf).as_bytes())?;
        out.flush()?;
        Ok(())
    }

    pub(in crate::ai::stream) fn flush_pending(&mut self) -> io::Result<()> {
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
            TableState::PendingHeader {
                indent,
                header_line,
                preview_height: _,
            } => self.consume_line(&format!("{indent}{header_line}"), false),
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

    pub(in crate::ai) fn consume_line(&mut self, line: &str, preview_emitted: bool) -> String {
        let state = std::mem::replace(&mut self.table_state, TableState::None);
        match state {
            TableState::None => {
                if !self.in_code_block && is_table_row_candidate(line) && !is_table_separator(line)
                {
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

                let move_up = preview_height
                    + if preview_emitted {
                        self.line_preview_height
                    } else {
                        0
                    };
                let mut out = String::new();
                if move_up > 0 {
                    out.push_str(&format!("\x1b[{move_up}A\r\x1b[0J"));
                }
                self.table_state = TableState::None;
                out.push_str(&self.render_line_no_table(&format!("{indent}{header_line}")));
                out.push_str(&self.consume_line(line, false));
                out
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

        let actual_table_height = final_table.lines().count().max(1);
        let clear_height = move_up.max(actual_table_height);

        let mut out = String::new();
        out.push_str(&format!("\x1b[{clear_height}A\r\x1b[0J"));
        out.push_str(&final_table);
        out
    }

    fn render_line_no_table(&mut self, line: &str) -> String {
        let (indent, rest) = split_indent(line);
        let trimmed = rest.trim_start_matches([' ', '\t']);

        let base = if self.dimmed { "\x1b[2m" } else { "" };

        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            if self.in_code_block {
                self.in_code_block = false;
                self.code_block_lang = None;
            } else {
                self.in_code_block = true;
                self.code_block_lang = parse_code_block_language(trimmed);
            }
            return format!("{indent}{MONOKAI_BG}{MONOKAI_COMMENT}{trimmed}\x1b[0m\n");
        }

        if self.in_code_block {
            if line.is_empty() {
                return "\n".to_string();
            }
            return format!(
                "{indent}{MONOKAI_BG}{base}{}\x1b[0m\n",
                highlight_code_line(rest, self.code_block_lang.as_deref())
            );
        }

        if trimmed == "$$" || trimmed == "\\[" || trimmed == "\\]" {
            self.in_math_block = !self.in_math_block;
            return "\n".to_string();
        }

        if self.in_math_block {
            if line.is_empty() {
                return "\n".to_string();
            }
            let math = crate::ai::stream::render_math_tex_to_unicode(rest.trim_end());
            return format!("{indent}{base}\x1b[95m{math}\x1b[0m\n");
        }

        if let Some((level, title)) = parse_heading(trimmed) {
            let (h_base, underline_char) = match level {
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
            out.push_str(h_base);
            let combined_base = format!("{}{}", base, h_base);
            out.push_str(&render_inline_md(title, &combined_base));
            out.push_str("\x1b[0m\n");

            if let Some(ch) = underline_char {
                let len = title.chars().count().clamp(3, 80);
                out.push_str(indent);
                out.push_str(base);
                out.push_str("\x1b[2m\x1b[36m");
                out.push_str(&std::iter::repeat_n(ch, len).collect::<String>());
                out.push_str("\x1b[0m\n");
            }
            return out;
        }

        if let Some((p_indent, prefix, checkbox, body)) = split_list_prefix(line) {
            let mut out = String::new();
            out.push_str(p_indent);
            out.push_str(base);
            out.push_str("\x1b[36m");
            out.push_str(prefix);
            out.push_str("\x1b[0m");
            if let Some(checked) = checkbox {
                let mark = if checked { "✅ " } else { "⬜ " };
                out.push_str(mark);
            }
            out.push_str(&render_inline_md(body, base));
            out.push('\n');
            return out;
        }

        if line.is_empty() {
            return "\n".to_string();
        }
        format!("{}{}{}\n", indent, base, render_inline_md(rest, base))
    }
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

fn split_list_prefix(line: &str) -> Option<(&str, &str, Option<bool>, &str)> {
    let (indent, rest) = split_indent(line);
    let rest = rest.trim_end();

    // Task list: - [ ] / - [x] / - [X]
    if rest.starts_with("- [ ] ") {
        return Some((indent, "- ", Some(false), &rest[6..]));
    }
    if rest.starts_with("- [x] ") || rest.starts_with("- [X] ") {
        return Some((indent, "- ", Some(true), &rest[6..]));
    }

    // Bullet list
    if rest.starts_with("- ") || rest.starts_with("* ") || rest.starts_with("+ ") {
        return Some((indent, &rest[..2], None, &rest[2..]));
    }

    // Ordered list
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
        return Some((indent, &rest[..i + 2], None, &rest[i + 2..]));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consume_line_move_up_matches_preview_height() {
        unsafe { std::env::set_var("COLUMNS", "6") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.set_line_preview_height(1);
        let out = renderer.consume_line("**hello**", true);
        assert!(out.contains("\x1b[1A\r\x1b[0J"));
        assert!(!out.contains("\x1b[2A\r\x1b[0J"));
    }

    #[test]
    fn test_write_chunk_preview_height() {
        unsafe { std::env::set_var("COLUMNS", "10") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.write_chunk("123456789012345", false);
        let first_height = renderer.line_preview_height();
        assert!(first_height >= 1);

        let _ = renderer.write_chunk("678901", false);
        assert!(renderer.line_preview_height() >= first_height);
    }

    #[test]
    fn test_pending_header_restore() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let _ = renderer.consume_line("| Header A | Header B |", false);
        let _ = renderer.consume_line("Not a separator", false);
    }
}
