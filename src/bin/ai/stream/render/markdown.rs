use std::io::{self, Write};

use crate::ai::stream::extract::strip_ansi_codes;
use crate::ai::theme::{
    ACCENT_MUTED, ACCENT_PRIMARY, ACCENT_RULE, ACCENT_SECONDARY, ACCENT_SUCCESS,
};
use crate::ai::stream::state::{END_THINKING_TAG_TEXT, THINKING_TAG_TEXT};
use crate::ai::stream::render::code::{
    MONOKAI_BG, MONOKAI_DIM, highlight_code_line, parse_code_block_language,
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
    show_line_gutter: bool,
    code_block_indent: String,
    code_block_lang: Option<String>,
    code_line_number: usize,
    in_math_block: bool,
    bol: bool,
    line_buf: String,
    line_preview_emitted: bool,
    line_preview_height: usize,
    table_state: TableState,
    dimmed: bool,
    code_preview_segment_width: usize,
}

impl MarkdownStreamRenderer {
    pub(in crate::ai::stream) fn new() -> Self {
        use std::io::IsTerminal;
        Self::new_with_tty(io::stdout().is_terminal())
    }

    pub(in crate::ai) fn new_with_tty(tty: bool) -> Self {
        Self {
            tty,
            show_line_gutter: false,
            enabled: true,
            in_code_block: false,
            code_block_indent: String::new(),
            code_block_lang: None,
            code_line_number: 0,
            in_math_block: false,
            bol: false,
            line_buf: String::new(),
            line_preview_emitted: false,
            line_preview_height: 0,
            table_state: TableState::None,
            dimmed: false,
            code_preview_segment_width: 0,
        }
    }

    pub(in crate::ai::stream) fn should_render(&mut self, _chunk: &str) -> bool {
        if !self.tty {
            return false;
        }
        self.enabled = true;
        true
    }

    pub(in crate::ai::stream) fn write_chunk(
        &mut self,
        chunk: &str,
        dimmed: bool,
    ) -> io::Result<()> {
        let mut out = io::stdout();
        self.write_chunk_to(&mut out, chunk, dimmed)
    }

    fn write_chunk_to(
        &mut self,
        out: &mut dyn Write,
        chunk: &str,
        dimmed: bool,
    ) -> io::Result<()> {
        self.dimmed = dimmed;
        for ch in chunk.chars() {
            if ch == '\n' {
                self.handle_newline(out)?;
                continue;
            }

            self.line_buf.push(ch);
            self.handle_char(out, ch)?;
            self.bol = false;
        }
        out.flush()?;
        Ok(())
    }

    #[cfg(test)]
    pub fn write_chunk_for_test(&mut self, chunk: &str, dimmed: bool) -> io::Result<String> {
        let mut out = Vec::new();
        self.write_chunk_to(&mut out, chunk, dimmed)?;
        Ok(String::from_utf8_lossy(&out).into_owned())
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

    pub(in crate::ai::stream) fn has_unfinished_line(&self) -> bool {
        !self.line_buf.is_empty()
    }

    pub(in crate::ai::stream::render) fn code_line_number(&self) -> usize {
        self.code_line_number
    }

    pub(in crate::ai::stream::render) fn reset_code_line_number(&mut self) {
        self.code_line_number = 0;
    }

    #[cfg(test)]
    fn set_show_line_gutter(&mut self, show_line_gutter: bool) {
        self.show_line_gutter = show_line_gutter;
    }

    fn handle_newline(&mut self, out: &mut dyn Write) -> io::Result<()> {
        if self.line_preview_emitted {
            if self.in_code_block {
                out.write_all(b"\x1b[0m\n")?;
            } else {
                out.write_all(b"\n")?;
            }
            out.flush()?;
            self.bol = true;
        }

        let line = std::mem::take(&mut self.line_buf);
        let rendered = self.consume_line(&line, self.line_preview_emitted);

        self.line_preview_emitted = false;
        self.line_preview_height = 0;
        self.code_preview_segment_width = 0;

        if !rendered.is_empty() {
            out.write_all(rendered.as_bytes())?;
            out.flush()?;
            self.bol = rendered.ends_with('\n');
        }
        Ok(())
    }

    fn handle_char(&mut self, out: &mut dyn Write, ch: char) -> io::Result<()> {
        if self.should_emit_table_preview_live() {
            self.handle_table_preview(out, ch)
        } else {
            self.handle_realtime_output(out, ch)
        }
    }

    fn handle_table_preview(&mut self, out: &mut dyn Write, ch: char) -> io::Result<()> {
        if !self.line_preview_emitted {
            if self.dimmed {
                out.write_all(b"\x1b[2m")?;
            }
            out.write_all(self.line_buf.as_bytes())?;
            out.flush()?;
            self.line_preview_emitted = true;
            self.line_preview_height = self.current_line_preview_height();
        } else {
            self.emit_char(out, ch)?;
            self.line_preview_height = self.current_line_preview_height();
        }
        Ok(())
    }

    fn handle_realtime_output(&mut self, out: &mut dyn Write, ch: char) -> io::Result<()> {
        if self.in_code_block {
            self.handle_code_block_realtime_output(out, ch)?;
            self.line_preview_emitted = true;
            self.line_preview_height = self.current_line_preview_height();
            return Ok(());
        }
        if self.line_buf.chars().count() == 1 && self.dimmed {
            out.write_all(b"\x1b[2m")?;
        }
        self.emit_char(out, ch)?;
        self.line_preview_emitted = true;
        self.line_preview_height = self.current_line_preview_height();
        Ok(())
    }

    fn handle_code_block_realtime_output(
        &mut self,
        out: &mut dyn Write,
        ch: char,
    ) -> io::Result<()> {
        let block_indent = self.code_block_indent.clone();
        let line_num_str = format!("{:>3}", self.code_line_number + 1);
        let available_width =
            code_block_content_width(&block_indent, &line_num_str, self.show_line_gutter).max(1);

        // Keep the realtime emit path aligned with `code_block_preview_height`
        // (which strips `block_indent`) and with `render_line_no_table` (which
        // also strips `block_indent` before wrapping). If the character just
        // pushed into `line_buf` still belongs to the outer block_indent
        // prefix, we must not emit it nor count it towards the segment width
        // — the prefix is already produced by `code_block_preview_prefix`.
        if !block_indent.is_empty()
            && self.line_buf.len() <= block_indent.len()
            && block_indent.starts_with(self.line_buf.as_str())
        {
            if !self.line_preview_emitted {
                out.write_all(
                    code_block_preview_prefix(
                        &block_indent,
                        &line_num_str,
                        self.dimmed,
                        self.show_line_gutter,
                    )
                    .as_bytes(),
                )?;
                out.flush()?;
            }
            return Ok(());
        }

        let ch_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);

        if !self.line_preview_emitted {
            out.write_all(
                code_block_preview_prefix(&block_indent, &line_num_str, self.dimmed, self.show_line_gutter)
                    .as_bytes(),
            )?;
        } else if self.code_preview_segment_width > 0
            && self.code_preview_segment_width + ch_width > available_width
        {
            out.write_all(b"\x1b[0m\n")?;
            out.write_all(
                code_block_preview_continuation_prefix(&block_indent, self.dimmed, self.show_line_gutter).as_bytes(),
            )?;
            self.code_preview_segment_width = 0;
        }

        self.emit_char(out, ch)?;
        self.code_preview_segment_width += ch_width;
        Ok(())
    }

    fn current_line_preview_height(&self) -> usize {
        if self.in_code_block {
            return self.code_block_preview_height(&self.line_buf);
        }
        live_preview_cursor_rows(&self.line_buf)
    }

    fn code_block_preview_height(&self, line: &str) -> usize {
        let block_indent = self.code_block_indent.as_str();
        let line_num_str = format!("{:>3}", self.code_line_number + 1);
        let code_text = line.strip_prefix(block_indent).unwrap_or(line);
        wrap_code_block_text(
            code_text,
            code_block_content_width(block_indent, &line_num_str, self.show_line_gutter),
        )
        .len()
        .max(1)
    }

    fn emit_char(&mut self, out: &mut dyn Write, ch: char) -> io::Result<()> {
        let mut buf = [0u8; 4];
        out.write_all(ch.encode_utf8(&mut buf).as_bytes())?;
        out.flush()?;
        Ok(())
    }

    pub(in crate::ai::stream) fn flush_pending(&mut self) -> io::Result<()> {
        let mut out = io::stdout();

        if !self.line_buf.is_empty() {
            if self.line_preview_emitted {
                if self.in_code_block {
                    out.write_all(b"\x1b[0m\n")?;
                } else {
                    out.write_all(b"\n")?;
                }
                self.bol = true;
            }
            let line = std::mem::take(&mut self.line_buf);
            let rendered = self.consume_line(&line, self.line_preview_emitted);
            self.line_preview_emitted = false;
            self.line_preview_height = 0;
            self.code_preview_segment_width = 0;
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

        let mut out = String::new();
        if move_up > 0 {
            out.push_str(&format!("\x1b[{move_up}A\r\x1b[0J"));
        } else {
            out.push_str("\r\x1b[0J");
        }
        out.push_str(&final_table);
        out
    }

    fn render_line_no_table(&mut self, line: &str) -> String {
        let (indent, rest) = split_indent(line);
        let trimmed = rest.trim_start_matches([' ', '\t']);

        let base = if self.dimmed { "\x1b[2m" } else { "" };

        if trimmed == THINKING_TAG_TEXT || trimmed == END_THINKING_TAG_TEXT {
            self.in_code_block = false;
            self.code_block_indent.clear();
            self.code_block_lang = None;
            self.code_line_number = 0;

            let label = if trimmed == THINKING_TAG_TEXT {
                "thinking"
            } else {
                "done thinking"
            };
            let glyph = if trimmed == THINKING_TAG_TEXT { "╭─" } else { "╰─" };
            return format!(
                "{indent}{ACCENT_RULE}{glyph}\x1b[0m {ACCENT_MUTED}{label}\x1b[0m\n"
            );
        }

        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            if self.in_code_block {
                self.in_code_block = false;
                self.code_block_lang = None;
                let block_indent = std::mem::take(&mut self.code_block_indent);
                let border = "─".repeat(22);
                return format!("{block_indent}{MONOKAI_BG}{MONOKAI_DIM}╰{border}\x1b[0m\n");
            } else {
                self.in_code_block = true;
                self.code_block_indent = indent.to_string();
                self.code_block_lang = parse_code_block_language(trimmed);
                self.code_line_number = 0;
                let lang = self.code_block_lang.as_deref().unwrap_or("code");
                return format!("{indent}{MONOKAI_BG}{MONOKAI_DIM}╭─ {lang}\x1b[0m\n");
            }
        }

        if self.in_code_block {
            self.code_line_number += 1;
            let line_num = format!("{}", self.code_line_number);
            let line_num_str = format!("{:>3}", line_num);
            let block_indent = self.code_block_indent.as_str();
            let code_text = line.strip_prefix(block_indent).unwrap_or(line);
            if code_text.is_empty() {
                if self.show_line_gutter {
                    return format!(
                        "{block_indent}{MONOKAI_BG}{MONOKAI_DIM}{} │\x1b[0m\n",
                        line_num_str
                    );
                }
                return format!("{block_indent}{MONOKAI_BG}\x1b[0m\n");
            }
            let wrapped = wrap_code_block_text(
                code_text,
                code_block_content_width(block_indent, &line_num_str, self.show_line_gutter),
            );
            let mut out = String::new();
            for (idx, segment) in wrapped.iter().enumerate() {
                out.push_str(block_indent);
                out.push_str(MONOKAI_BG);
                if self.show_line_gutter {
                    let gutter = if idx == 0 {
                        line_num_str.as_str()
                    } else {
                        "   "
                    };
                    out.push_str(MONOKAI_DIM);
                    out.push_str(gutter);
                    out.push_str(" │");
                }
                out.push_str(base);
                out.push_str(&highlight_code_line(segment, self.code_block_lang.as_deref()));
                out.push_str("\x1b[0m\n");
            }
            return out;
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
            return format!("{indent}{base}{ACCENT_SECONDARY}{math}\x1b[0m\n");
        }

        if let Some((level, title)) = parse_heading(trimmed) {
            let (heading_style, underline_char, underline_style) = match level {
                1 => ("\x1b[1m\x1b[38;2;191;219;254m", Some('━'), ACCENT_RULE),
                2 => ("\x1b[1m\x1b[38;2;125;211;252m", Some('─'), ACCENT_RULE),
                3 => ("\x1b[1m\x1b[38;2;165;180;252m", None, ACCENT_RULE),
                _ => ("\x1b[1m\x1b[38;2;148;163;184m", None, ACCENT_RULE),
            };
            let mut out = String::new();
            if !self.bol {
                out.push('\n');
                self.bol = true;
            }
            out.push_str(indent);
            out.push_str(base);
            out.push_str(heading_style);
            let combined_base = format!("{}{}", base, heading_style);
            out.push_str(&render_inline_md(title, &combined_base));
            out.push_str("\x1b[0m\n");

            if let Some(ch) = underline_char {
                let len = title.chars().count().clamp(3, 80);
                out.push_str(indent);
                out.push_str(base);
                out.push_str("\x1b[2m");
                out.push_str(underline_style);
                out.push_str(&std::iter::repeat_n(ch, len).collect::<String>());
                out.push_str("\x1b[0m\n");
            }
            return out;
        }

        if is_thematic_break(trimmed) {
            return format!("{indent}{base}{ACCENT_RULE}{}\x1b[0m\n", "─".repeat(28));
        }

        if let Some(body) = parse_blockquote(trimmed) {
            return format!(
                "{indent}{base}{ACCENT_MUTED}▍\x1b[0m {base}{}\n",
                render_inline_md(body, base)
            );
        }

        if let Some((p_indent, prefix, checkbox, body)) = split_list_prefix(line) {
            let mut out = String::new();
            out.push_str(p_indent);
            if let Some(checked) = checkbox {
                out.push_str(base);
                if checked {
                    out.push_str(ACCENT_SUCCESS);
                    out.push('✓');
                } else {
                    out.push_str(ACCENT_MUTED);
                    out.push('○');
                }
                out.push_str("\x1b[0m ");
            } else if prefix.ends_with(". ") {
                out.push_str(base);
                out.push_str(ACCENT_MUTED);
                out.push_str(prefix.trim_end());
                out.push_str("\x1b[0m ");
            } else {
                out.push_str(base);
                out.push_str(ACCENT_PRIMARY);
                out.push('•');
                out.push_str("\x1b[0m ");
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

fn live_preview_cursor_rows(line: &str) -> usize {
    let cols = preview_terminal_width().max(1);
    let visible = strip_ansi_codes(line);
    let width: usize = visible.chars().map(|c| unicode_width::UnicodeWidthChar::width(c).unwrap_or(0)).sum();
    if width == 0 {
        1
    } else {
        1 + ((width - 1) / cols)
    }
}

fn preview_terminal_width() -> usize {
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

fn code_block_gutter_width(
    block_indent: &str,
    line_num_str: &str,
    show_line_gutter: bool,
) -> usize {
    let mut width = unicode_width::UnicodeWidthStr::width(block_indent);
    if show_line_gutter {
        width += unicode_width::UnicodeWidthStr::width(line_num_str)
            + unicode_width::UnicodeWidthStr::width(" │");
    }
    width
}

fn code_block_content_width(
    block_indent: &str,
    line_num_str: &str,
    show_line_gutter: bool,
) -> usize {
    preview_terminal_width()
        .saturating_sub(code_block_gutter_width(
            block_indent,
            line_num_str,
            show_line_gutter,
        ))
        .max(1)
}

fn wrap_code_block_text(text: &str, content_width: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for ch in text.chars() {
        let ch_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if current_width > 0 && current_width + ch_width > content_width {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(ch);
        current_width += ch_width;
    }

    if current.is_empty() {
        lines.push(String::new());
    } else {
        lines.push(current);
    }
    lines
}

fn code_block_preview_prefix(
    block_indent: &str,
    line_num_str: &str,
    dimmed: bool,
    show_line_gutter: bool,
) -> String {
    let mut out = String::new();
    out.push_str(block_indent);
    out.push_str(MONOKAI_BG);
    if show_line_gutter {
        out.push_str(MONOKAI_DIM);
        out.push_str(line_num_str);
        out.push_str(" │\x1b[0m");
    } else {
        out.push_str("\x1b[0m");
    }
    out.push_str(MONOKAI_BG);
    if dimmed {
        out.push_str("\x1b[2m");
    }
    out
}

fn code_block_preview_continuation_prefix(
    block_indent: &str,
    dimmed: bool,
    show_line_gutter: bool,
) -> String {
    code_block_preview_prefix(block_indent, "   ", dimmed, show_line_gutter)
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

fn is_thematic_break(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 3 {
        return false;
    }
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !matches!(first, '-' | '*' | '_') {
        return false;
    }
    chars.all(|ch| ch == first)
}

fn parse_blockquote(line: &str) -> Option<&str> {
    let body = line.strip_prefix("> ")?;
    Some(body.trim_end())
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
    use crate::ai::test_support::ENV_LOCK;

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn strip_ansi_for_test(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                if chars.peek() == Some(&'[') {
                    let _ = chars.next();
                    while let Some(c) = chars.next() {
                        if c == 'm' {
                            break;
                        }
                    }
                    continue;
                }
            }
            out.push(ch);
        }
        out
    }

    #[test]
    fn consume_line_move_up_matches_preview_height() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "6") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.set_line_preview_height(1);
        let out = renderer.consume_line("**hello**", true);
        assert!(out.contains("\x1b[1A\r\x1b[0J"));
        assert!(!out.contains("\x1b[2A\r\x1b[0J"));
    }

    #[test]
    fn test_write_chunk_preview_height() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "10") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.write_chunk_for_test("123456789012345", false);
        let first_height = renderer.line_preview_height();
        assert!(first_height >= 1);

        let _ = renderer.write_chunk_for_test("678901", false);
        assert!(renderer.line_preview_height() >= first_height);
    }

    #[test]
    fn test_pending_header_restore() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let _ = renderer.consume_line("| Header A | Header B |", false);
        let _ = renderer.consume_line("Not a separator", false);
    }

    #[test]
    fn code_block_keeps_inner_indentation_without_visible_gutter() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```rust", false);
        let out = renderer.consume_line("    let x = 1;", false);

        let visible = strip_ansi_for_test(&out);
        assert_eq!(visible, "    let x = 1;\n");
    }

    #[test]
    fn code_block_nested_indent_is_stable() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("  ```rust", false);
        let out = renderer.consume_line("      let x = 1;", false);

        let visible = strip_ansi_for_test(&out);
        assert_eq!(visible, "      let x = 1;\n");
    }

    #[test]
    fn task_list_uses_minimal_markers_instead_of_emoji() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let checked = renderer.consume_line("- [x] done", false);
        let unchecked = renderer.consume_line("- [ ] todo", false);

        let checked_visible = strip_ansi_for_test(&checked);
        let unchecked_visible = strip_ansi_for_test(&unchecked);
        assert!(checked_visible.contains("✓ done"));
        assert!(unchecked_visible.contains("○ todo"));
        assert!(!checked_visible.contains("✅"));
        assert!(!unchecked_visible.contains("⬜"));
    }

    #[test]
    fn blockquote_and_rule_render_with_cleaner_structure() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let quote = renderer.consume_line("> note", false);
        let rule = renderer.consume_line("---", false);

        let quote_visible = strip_ansi_for_test(&quote);
        let rule_visible = strip_ansi_for_test(&rule);
        assert!(quote_visible.contains("▍ note"));
        assert!(rule_visible.contains(&"─".repeat(28)));
    }

    #[test]
    fn thinking_markers_render_cleanly_without_leaking_ansi_bytes() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let start = renderer.consume_line(THINKING_TAG_TEXT, false);
        let end = renderer.consume_line(END_THINKING_TAG_TEXT, false);

        let start_visible = strip_ansi_for_test(&start);
        let end_visible = strip_ansi_for_test(&end);
        assert_eq!(start_visible, "╭─ thinking\n");
        assert_eq!(end_visible, "╰─ done thinking\n");
    }

    #[test]
    fn thinking_marker_breaks_out_of_code_block_rendering() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let _ = renderer.consume_line("```rust", false);
        let out = renderer.consume_line(END_THINKING_TAG_TEXT, false);

        let visible = strip_ansi_for_test(&out);
        assert_eq!(visible, "╰─ done thinking\n");
    }

    #[test]
    fn unfinished_line_state_tracks_pending_inline_content() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        renderer.write_chunk_for_test("hello", false).unwrap();
        assert!(renderer.has_unfinished_line());

        renderer.write_chunk_for_test("\n", false).unwrap();
        assert!(!renderer.has_unfinished_line());
    }

    #[test]
    fn html_code_block_preview_height_matches_plain_width_without_gutter() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "7") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let _ = renderer.consume_line("```html", false);
        renderer.write_chunk_for_test("<x>", false).unwrap();

        assert!(
            renderer.line_preview_height() == live_preview_cursor_rows("<x>"),
            "code-block preview height should match the visible code width when gutter is hidden"
        );
    }

    #[test]
    fn html_code_block_line_remains_visible_after_rewrite() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "16") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```html", false);
        let out = renderer.consume_line(r#"<div class="input-area"></div>"#, true);

        let visible = strip_ansi_for_test(&out);
        let flattened = visible.replace('\n', "");
        assert_eq!(flattened, r#"<div class="input-area"></div>"#);
        assert!(visible.lines().count() > 1, "{visible:?}");
        assert!(!visible.contains('│'), "{visible:?}");
    }

    #[test]
    fn live_preview_cursor_rows_counts_exact_width_cjk_boundary() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "20") };
        assert_eq!(live_preview_cursor_rows("你好你好你好你好你好"), 1);
    }

    #[test]
    fn exact_width_cjk_preview_updates_renderer_height_to_one_row() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "20") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.write_chunk_for_test("你好你好你好你好你好", true).unwrap();

        assert_eq!(renderer.line_preview_height(), 1);
    }

    #[test]
    fn exact_width_cjk_rewrite_moves_up_one_row() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "20") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.set_line_preview_height(1);

        let out = renderer.consume_line("你好你好你好你好你好", true);
        assert!(out.contains("\x1b[1A\r\x1b[0J"), "{out:?}");
        assert!(!out.contains("\x1b[2A\r\x1b[0J"), "{out:?}");
    }

    #[test]
    fn code_block_long_line_wraps_without_gutter_prefix() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "10") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```text", false);

        let out = renderer.consume_line("abcdefghijkl", false);
        let visible = strip_ansi_for_test(&out);
        let lines = visible.lines().collect::<Vec<_>>();

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "abcdefghij");
        assert_eq!(lines[1], "kl");
    }

    #[test]
    fn code_block_streaming_preview_wraps_before_newline() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "10") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```text", false);

        renderer.write_chunk_for_test("abcdefghijkl", false).unwrap();

        assert_eq!(renderer.line_preview_height(), 2);
    }

    #[test]
    fn optional_code_block_gutter_still_renders_line_numbers() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.set_show_line_gutter(true);
        let _ = renderer.consume_line("```rust", false);

        let out = renderer.consume_line("let x = 1;", false);
        let visible = strip_ansi_for_test(&out);

        assert!(visible.starts_with("  1 │let x = 1;"), "{visible:?}");
    }

    #[test]
    fn nested_code_block_streaming_preview_does_not_double_emit_block_indent() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("  ```rust", false);

        let streamed = renderer
            .write_chunk_for_test("      if len", false)
            .unwrap();
        let visible = strip_ansi_for_test(&streamed);

        assert!(
            visible.starts_with("      if len"),
            "streamed preview should render the source line verbatim once, got {visible:?}"
        );
        assert!(
            !visible.starts_with("            "),
            "block_indent must not be emitted twice during realtime preview, got {visible:?}"
        );
        assert_eq!(renderer.line_preview_height(), 1);
    }

    #[test]
    fn nested_code_block_preview_height_matches_final_render_height() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "14") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("  ```rust", false);

        renderer
            .write_chunk_for_test("      if len_is_long", false)
            .unwrap();
        let streamed_height = renderer.line_preview_height();

        let final_out = renderer.consume_line("      if len_is_long", true);
        let move_up = format!("\x1b[{streamed_height}A\r\x1b[0J");
        assert!(
            final_out.contains(&move_up),
            "final rewrite must move the cursor up by exactly the streamed preview height \
             ({streamed_height}); got {final_out:?}"
        );
    }
}
