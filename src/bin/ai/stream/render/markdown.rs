use std::io::{self, Write};

use crate::ai::stream::extract::strip_ansi_codes;
use crate::ai::stream::render::code::{
    MONOKAI_BG, MONOKAI_DIM, highlight_code_line, parse_code_block_language,
};
use crate::ai::stream::render::inline::render_inline_md;
use crate::ai::stream::render::table::{
    TableAlign, TableState, compute_table_widths, is_table_row, is_table_row_candidate,
    is_table_separator, line_looks_like_table_preview, parse_table_align, parse_table_row,
    render_table_bottom, render_table_header, render_table_mid, render_table_row, render_table_top,
    split_indent, table_preview_height,
};
use crate::ai::stream::render::html::{
    contains_close_table_tag, contains_open_table_tag, parse_html_table, render_html_table,
};
use crate::ai::stream::state::{END_THINKING_TAG_TEXT, THINKING_TAG_TEXT};
use crate::ai::theme::{
    ACCENT_MUTED, ACCENT_PRIMARY, ACCENT_RULE, ACCENT_SECONDARY, ACCENT_SUCCESS,
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
    // HTML 表格缓冲状态
    in_html_table: bool,
    html_table_buf: String,
    html_table_preview_height: usize,
    html_table_indent: String,
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
            in_html_table: false,
            html_table_buf: String::new(),
            html_table_preview_height: 0,
            html_table_indent: String::new(),
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

    fn write_chunk_to(&mut self, out: &mut dyn Write, chunk: &str, dimmed: bool) -> io::Result<()> {
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

    #[cfg(test)]
    fn flush_pending_for_test(&mut self) -> io::Result<String> {
        let mut out = Vec::new();

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
                preview_height,
            } => self.rewrite_plain_preview(&format!("{indent}{header_line}"), preview_height),
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
        out.flush()?;
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
                code_block_preview_prefix(
                    &block_indent,
                    &line_num_str,
                    self.dimmed,
                    self.show_line_gutter,
                )
                .as_bytes(),
            )?;
        } else if self.code_preview_segment_width > 0
            && self.code_preview_segment_width + ch_width > available_width
        {
            out.write_all(b"\x1b[0m\n")?;
            out.write_all(
                code_block_preview_continuation_prefix(
                    &block_indent,
                    self.dimmed,
                    self.show_line_gutter,
                )
                .as_bytes(),
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

    fn streamed_or_measured_preview_height(&self, line: &str, preview_emitted: bool) -> usize {
        if preview_emitted {
            self.line_preview_height.max(1)
        } else {
            table_preview_height(line)
        }
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
        out.write_all(ch.encode_utf8(&mut buf).as_bytes())
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

        // 流式输出在 HTML 表格缓冲中途结束——尝试解析已有内容
        if self.in_html_table {
            let buf = std::mem::take(&mut self.html_table_buf);
            let indent = std::mem::take(&mut self.html_table_indent);
            let preview_height = std::mem::take(&mut self.html_table_preview_height);
            self.in_html_table = false;

            let rendered = parse_html_table(&buf)
                .map(|t| render_html_table(&indent, &t))
                .unwrap_or(buf);
            let move_up = preview_height;
            let final_out = if move_up > 0 {
                format!("\x1b[{move_up}A\r\x1b[0J{rendered}")
            } else {
                rendered
            };
            if !final_out.is_empty() {
                out.write_all(final_out.as_bytes())?;
                self.bol = final_out.ends_with('\n');
            }
        }

        let state = std::mem::replace(&mut self.table_state, TableState::None);
        let rendered = match state {
            TableState::None => String::new(),
            TableState::PendingHeader {
                indent,
                header_line,
                preview_height,
            } => self.rewrite_plain_preview(&format!("{indent}{header_line}"), preview_height),
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
        // HTML 表格缓冲：正在收集 <table>...</table> 内容
        if self.in_html_table {
            return self.consume_html_table_line(line, preview_emitted);
        }

        let state = std::mem::replace(&mut self.table_state, TableState::None);
        match state {
            TableState::None => {
                // 检测 HTML <table> 开标签（非代码块、非 markdown 表格上下文）
                if !self.in_code_block && contains_open_table_tag(line) {
                    return self.start_html_table(line, preview_emitted);
                }
                if !self.in_code_block && is_table_row_candidate(line) && !is_table_separator(line)
                {
                    let mut out = String::new();
                    if !self.bol {
                        out.push('\n');
                        self.bol = true;
                    }
                    let (indent, rest) = split_indent(line);
                    let raw = format!("{indent}{}", rest.trim_end());
                    let mut preview_height =
                        self.streamed_or_measured_preview_height(&raw, preview_emitted);
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
                preview_height,
            } => {
                if is_table_separator(line) {
                    let raw = line.trim_end().to_string();
                    let header_cells = parse_table_row(&header_line);
                    let align = parse_table_align(line, header_cells.len());
                    let separator_preview_height =
                        self.streamed_or_measured_preview_height(&raw, preview_emitted);
                    let move_up = preview_height
                        + if preview_emitted {
                            separator_preview_height
                        } else {
                            0
                        };
                    let rendered_height =
                        self.rendered_table_height(&indent, &header_cells, &align, &[]);
                    let rewrite = self.rewrite_table_preview(
                        &indent,
                        move_up,
                        &header_cells,
                        &align,
                        &[],
                    );
                    self.table_state = TableState::InTable {
                        indent,
                        header: header_cells,
                        align,
                        rows: Vec::new(),
                        preview_height: rendered_height,
                    };
                    return rewrite;
                }

                let move_up = preview_height
                    + if preview_emitted {
                        self.line_preview_height
                    } else {
                        0
                    };
                let mut out =
                    self.rewrite_plain_preview(&format!("{indent}{header_line}"), move_up);
                self.table_state = TableState::None;
                out.push_str(&self.consume_line(line, false));
                out
            }
            TableState::InTable {
                indent,
                header,
                align,
                mut rows,
                preview_height,
            } => {
                if is_table_row(line) {
                    rows.push(parse_table_row(line));
                    let move_up = preview_height
                        + if preview_emitted {
                            self.line_preview_height.max(1)
                        } else {
                            0
                        };
                    let rendered_height =
                        self.rendered_table_height(&indent, &header, &align, &rows);
                    let out =
                        self.rewrite_table_preview(&indent, move_up, &header, &align, &rows);
                    self.table_state = TableState::InTable {
                        indent,
                        header,
                        align,
                        rows,
                        preview_height: rendered_height,
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

    // ── HTML 表格缓冲 ──────────────────────────────────────────

    /// 检测到 `<table>` 开标签，开始缓冲 HTML 内容。
    fn start_html_table(&mut self, line: &str, preview_emitted: bool) -> String {
        self.in_html_table = true;
        self.html_table_buf = line.to_string();

        let (indent, rest) = split_indent(line);
        self.html_table_indent = indent.to_string();

        // 单行表格（<table>...</table> 在同一行）
        if contains_close_table_tag(line) {
            return self.finalize_html_table(preview_emitted);
        }

        // 开始缓冲——记录预览高度
        let raw = format!("{indent}{}", rest.trim_end());
        self.html_table_preview_height =
            self.streamed_or_measured_preview_height(&raw, preview_emitted);

        if preview_emitted {
            return String::new();
        }
        format!("{}\n", raw)
    }

    /// 缓冲 HTML 表格的后续行。
    fn consume_html_table_line(&mut self, line: &str, preview_emitted: bool) -> String {
        self.html_table_buf.push('\n');
        self.html_table_buf.push_str(line);

        // 检测 </table> 闭标签——解析并渲染最终表格
        if contains_close_table_tag(line) {
            return self.finalize_html_table(preview_emitted);
        }

        // 继续缓冲——累加预览高度
        let raw = line.trim_end();
        self.html_table_preview_height +=
            self.streamed_or_measured_preview_height(raw, preview_emitted);

        if preview_emitted {
            return String::new();
        }
        format!("{}\n", raw)
    }

    /// HTML 表格缓冲完成，解析并渲染为终端表格。
    fn finalize_html_table(&mut self, preview_emitted: bool) -> String {
        let buf = std::mem::take(&mut self.html_table_buf);
        let indent = std::mem::take(&mut self.html_table_indent);
        let preview_height = std::mem::take(&mut self.html_table_preview_height);
        self.in_html_table = false;

        let table = parse_html_table(&buf);
        let rendered = match &table {
            Some(t) => render_html_table(&indent, t),
            None => {
                // 解析失败——回退为原始文本
                buf
            }
        };

        let move_up = preview_height
            + if preview_emitted {
                self.line_preview_height
            } else {
                0
            };

        if move_up > 0 {
            format!("\x1b[{move_up}A\r\x1b[0J{rendered}")
        } else {
            rendered
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
        let final_table = self.render_table_block(indent, header, align, rows);
        if final_table.is_empty() || move_up == 0 {
            return String::new();
        }

        let mut out = String::new();
        if move_up > 0 {
            out.push_str(&format!("\x1b[{move_up}A\r\x1b[0J"));
        } else {
            out.push_str("\r\x1b[0J");
        }
        out.push_str(&final_table);
        out
    }

    fn render_table_block(
        &self,
        indent: &str,
        header: &[String],
        align: &[TableAlign],
        rows: &[Vec<String>],
    ) -> String {
        let cols = header
            .len()
            .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
        if cols == 0 {
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
        final_table
    }

    fn rendered_table_height(
        &self,
        indent: &str,
        header: &[String],
        align: &[TableAlign],
        rows: &[Vec<String>],
    ) -> usize {
        self.render_table_block(indent, header, align, rows)
            .lines()
            .count()
    }

    fn rewrite_plain_preview(&mut self, line: &str, move_up: usize) -> String {
        let rendered = self.render_line_no_table(line);
        if move_up > 0 {
            format!("\x1b[{move_up}A\r\x1b[0J{rendered}")
        } else {
            rendered
        }
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
            let glyph = if trimmed == THINKING_TAG_TEXT {
                "╭─"
            } else {
                "╰─"
            };
            return format!("{indent}{ACCENT_RULE}{glyph}\x1b[0m {ACCENT_MUTED}{label}\x1b[0m\n");
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
                out.push_str(&highlight_code_line(
                    segment,
                    self.code_block_lang.as_deref(),
                ));
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

/// 把一行文本按终端 **真实** 列宽硬截断到「最多占一个物理行」，超出部分用 `…` 收尾。
///
/// 折叠窗口（thinking / 工具输出）用它保证每条可见行只占一个物理行，使 cursor-up
/// 擦除的行数与逻辑行数严格相等，彻底摆脱 `live_preview_cursor_rows` 对自动折行的
/// 预测——tab / 全角字符 / 超长行 / 终端 resize 都不会再让擦除行数算少而残留旧内容
/// （表现为 header 反复堆叠、大段空白）。传入文本视为不含 ANSI 的纯文本。
pub(in crate::ai) fn clamp_line_to_terminal_row(line: &str) -> String {
    clamp_line_to_terminal_row_with_reserve(line, 0)
}

/// 同 [`clamp_line_to_terminal_row`]，但先从终端列宽预留 `reserve_cols` 列给行首装饰
/// （如折叠行的 `  │ ` 前缀），保证「前缀 + clamp 后正文」合起来仍不超过一个物理行。
pub(in crate::ai) fn clamp_line_to_terminal_row_with_reserve(
    line: &str,
    reserve_cols: usize,
) -> String {
    let cols = raw_terminal_cols().saturating_sub(reserve_cols).max(1);
    let mut total = 0usize;
    for ch in line.chars() {
        total += unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
    }
    if total <= cols {
        return line.to_string();
    }

    // 需要截断：预留 1 列给省略号，保证含省略号后仍不超过 cols。
    let budget = cols.saturating_sub(1).max(1);
    let mut out = String::with_capacity(line.len());
    let mut col = 0usize;
    for ch in line.chars() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if col + w > budget {
            break;
        }
        out.push(ch);
        col += w;
    }
    out.push('…');
    out
}

pub(in crate::ai) fn live_preview_cursor_rows(line: &str) -> usize {
    // 预览行是逐字符原样写入终端、由终端按 **真实** 列宽自动折行的，所以这里必须用
    // raw_terminal_cols（而非保留右边距的 preview_terminal_width）来数物理行数。
    // 否则窄于真实宽度的列数会把"恰好一行"的预览算成两行，cursor-up 多移一行，
    // 越界清掉表格上方内容、或在重写后残留预览碎片。
    // 折行规则与终端 DECAWM 一致：全角字符放不下右边一列时提前折行（col+w>cols）。
    let cols = raw_terminal_cols().max(1);
    let visible = strip_ansi_codes(line);
    let mut lines = 1usize;
    let mut col = 0usize;
    for ch in visible.chars() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if col > 0 && col + w > cols {
            lines += 1;
            col = w;
        } else {
            col += w;
        }
    }
    lines
}

/// 终端可用列数（已扣除右侧安全边距）。
///
/// 大多数终端开启 DECAWM（auto-margin）：当输出列号 == 列数时会触发隐式换行，
/// 对全角字符（CJK / emoji）更敏感，会让代码块/表格的边框贴边或被截断，进而
/// 让 cursor up + clear 重写时对不上行数，出现"残留色块 / 残留边框"。
/// 这里统一保留 4 列安全边距，下限 20 防止极窄终端崩盘。
const RIGHT_MARGIN: usize = 4;
const MIN_PREVIEW_WIDTH: usize = 20;

fn preview_terminal_width() -> usize {
    raw_terminal_cols()
        .saturating_sub(RIGHT_MARGIN)
        .max(MIN_PREVIEW_WIDTH)
}

fn raw_terminal_cols() -> usize {
    // 优先用 ioctl(TIOCGWINSZ) 拿 **实时** 列数：`a` 是常驻进程，运行在 VS Code 等
    // 面板里时，环境变量 COLUMNS 是进程启动那一刻的快照（shell 只在每次提示符前刷新
    // 它），面板被拖窄后 COLUMNS 往往比真实宽度大。若用过大的列数计算预览行数 / 表格
    // 宽度，会导致 cursor-up 重写的行数算少（残留预览）以及表格超宽被终端硬折行
    // （边框错位）。因此真实 tty 一律以 ioctl 为准，COLUMNS 仅作为非 tty（如测试、
    // 管道）时的回退。
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

    if let Some(cols) = std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        && cols > 0
    {
        return cols;
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
        let bytes = s.as_bytes();
        let mut out = String::with_capacity(s.len());
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
            let Some(ch) = s[i..].chars().next() else {
                break;
            };
            if ch != '\r' {
                out.push(ch);
            }
            i += ch.len_utf8();
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
    fn pending_header_flush_rewrites_to_plain_markdown_line() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.bol = true;

        let first = renderer
            .write_chunk_for_test("| **Header** | Value |\n", false)
            .unwrap();
        assert_eq!(first, "| **Header** | Value |\n");

        let flushed = renderer.flush_pending_for_test().unwrap();
        assert!(flushed.contains("\x1b[1A\r\x1b[0J"));

        let visible = strip_ansi_for_test(&flushed);
        assert_eq!(visible, "| Header | Value |\n");
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
        renderer
            .write_chunk_for_test("你好你好你好你好你好", true)
            .unwrap();

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
        // preview_terminal_width 引入 RIGHT_MARGIN=4 与 MIN_PREVIEW_WIDTH=20
        // 之后，最小有效宽度是 20。这里给 30 列模拟终端宽度，扣除右边距得到
        // 26 列内容宽度（block_indent="" + gutter 1 列），实际 wrap 阈值约 25。
        unsafe { std::env::set_var("COLUMNS", "30") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```text", false);

        // 30 个字符确保超过当前内容宽度，会被强制 wrap 至两行。
        let out = renderer.consume_line(&"a".repeat(30), false);
        let visible = strip_ansi_for_test(&out);
        let lines = visible.lines().collect::<Vec<_>>();

        assert!(
            lines.len() >= 2,
            "expected wrap to >=2 lines, got {lines:?}"
        );
        // 拼回所有行后字符总数仍应等于原输入（仅 wrap，不应丢字符）。
        let joined: String = lines.concat();
        assert_eq!(joined, "a".repeat(30));
    }

    #[test]
    fn code_block_streaming_preview_wraps_before_newline() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "30") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```text", false);

        renderer
            .write_chunk_for_test(&"a".repeat(30), false)
            .unwrap();

        assert!(
            renderer.line_preview_height() >= 2,
            "expected preview height >=2, got {}",
            renderer.line_preview_height()
        );
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

    /// 极简 VT100 网格模拟器：复现"流式预览 → cursor-up 重写"实际落到终端后的可见结果。
    /// 仅实现本渲染器会发出的控制序列：可打印字符（含 CJK 宽度）、`\n`(CRLF)、`\r`、
    /// CSI nA（光标上移）、CSI 0J（清到屏幕末尾）、SGR（忽略）。DECAWM 自动换行按真实
    /// 终端语义建模：全角字符放不下右边一列时提前折到下一行（留空位）。
    struct VtGrid {
        width: usize,
        rows: Vec<Vec<char>>,
        row: usize,
        col: usize,
    }

    impl VtGrid {
        fn new(width: usize) -> Self {
            Self {
                width,
                rows: vec![vec![' '; width]],
                row: 0,
                col: 0,
            }
        }

        fn ensure_row(&mut self, r: usize) {
            while self.rows.len() <= r {
                self.rows.push(vec![' '; self.width]);
            }
        }

        fn newline(&mut self) {
            self.row += 1;
            self.col = 0;
            self.ensure_row(self.row);
        }

        fn put(&mut self, ch: char) {
            let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            if w == 0 {
                return;
            }
            if self.col + w > self.width {
                self.newline();
            }
            self.ensure_row(self.row);
            self.rows[self.row][self.col] = ch;
            for k in 1..w {
                if self.col + k < self.width {
                    self.rows[self.row][self.col + k] = '\0';
                }
            }
            self.col += w;
        }

        fn feed(&mut self, s: &str) {
            let mut chars = s.chars().peekable();
            while let Some(ch) = chars.next() {
                match ch {
                    '\n' => self.newline(),
                    '\r' => self.col = 0,
                    '\x1b' => {
                        if chars.peek() == Some(&'[') {
                            chars.next();
                            let mut num = String::new();
                            let mut final_byte = '\0';
                            for c in chars.by_ref() {
                                if c.is_ascii_digit() {
                                    num.push(c);
                                } else {
                                    final_byte = c;
                                    break;
                                }
                            }
                            let n: usize = num.parse().unwrap_or(0);
                            match final_byte {
                                'A' => self.row = self.row.saturating_sub(n.max(1)),
                                'J' => {
                                    // 0J: 清到屏幕末尾
                                    for c in self.col..self.width {
                                        self.rows[self.row][c] = ' ';
                                    }
                                    let start = self.row + 1;
                                    self.rows.truncate(start.max(1));
                                    self.ensure_row(self.row);
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => self.put(ch),
                }
            }
        }

        fn screen(&self) -> Vec<String> {
            self.rows
                .iter()
                .map(|r| {
                    r.iter()
                        .filter(|c| **c != '\0')
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                })
                .collect()
        }
    }

    fn render_full_stream(markdown: &str, dimmed: bool) -> String {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let mut bytes = String::new();
        for ch in markdown.chars() {
            let mut buf = [0u8; 4];
            bytes.push_str(
                &renderer
                    .write_chunk_for_test(ch.encode_utf8(&mut buf), dimmed)
                    .unwrap(),
            );
        }
        bytes.push_str(&renderer.flush_pending_for_test().unwrap());
        bytes
    }

    #[test]
    fn streamed_cjk_table_has_no_residual_fragments_on_screen() {
        let _guard = env_guard();

        let md = "\
下面是一个对照表，用来说明不同协议的默认 Endpoint 与典型后端服务的对应关系：
| 协议 | 默认 Endpoint | 典型后端 |
| --- | --- | --- |
| Compatible | dashscope.aliyuncs.com/compatible-mode | 阿里云百炼（DashScope）的 OpenAI 兼容模式 |
| OpenAi | api.openai.com/v1/chat/completions | OpenAI 官方 API |
";
        let para = "下面是一个对照表，用来说明不同协议的默认 Endpoint 与典型后端服务的对应关系：";

        for cols in [80usize, 72, 64, 60, 56, 52, 48] {
            unsafe { std::env::set_var("COLUMNS", cols.to_string()) };

            let stream = render_full_stream(md, false);
            let mut grid = VtGrid::new(cols);
            grid.feed(&stream);
            let screen = grid.screen();
            let joined: String = screen.join("");

            // 表格上方的引导段落必须完整保留（不能被 cursor-up 越界清掉）。
            // 终端可能按列宽自动折行，所以拼接所有行后再比对去掉换行的原文。
            let para_joined: String = para.chars().filter(|c| !c.is_whitespace()).collect();
            let screen_joined: String = joined.chars().filter(|c| !c.is_whitespace()).collect();
            assert!(
                screen_joined.contains(&para_joined),
                "COLUMNS={cols}: leading paragraph was clobbered:\n{}",
                screen.join("\n")
            );

            // 表格区域不得残留原始 markdown 预览碎片（裸 `|`，区别于盒框 `│`）。
            // 残留说明 cursor-up 行数与实际渲染行数不一致，预览没被完全覆盖。
            for line in &screen {
                if line.is_empty() || para.contains(line.trim()) {
                    continue;
                }
                assert!(
                    !line.contains('|'),
                    "COLUMNS={cols}: residual raw markdown pipe: {line:?}\nfull screen:\n{}",
                    screen.join("\n")
                );
            }
        }
    }

    #[test]
    fn table_confirmation_and_flush_use_streamed_preview_height_for_rewrite() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.bol = true;

        renderer.set_line_preview_height(5);
        assert_eq!(renderer.consume_line("| Header | Value |", true), "");

        renderer.set_line_preview_height(7);
        let header_rewrite = renderer.consume_line("| --- | --- |", true);
        assert!(
            header_rewrite.contains("\x1b[12A\r\x1b[0J"),
            "expected separator confirmation to rewrite using header+separator preview height; got {header_rewrite:?}"
        );

        renderer.set_line_preview_height(9);
        let row_rewrite = renderer.consume_line("| foo | bar |", true);
        assert!(
            row_rewrite.contains("\x1b[13A\r\x1b[0J"),
            "expected streamed table row to rewrite existing table plus live row preview; got {row_rewrite:?}"
        );

        let flushed = renderer.flush_pending_for_test().unwrap();
        assert!(
            flushed.contains("\x1b[5A\r\x1b[0J"),
            "expected final flush to clear the latest rendered table height; got {flushed:?}"
        );
    }

    #[test]
    fn heading_followed_by_cjk_table_leaves_no_raw_header_fragment() {
        let _guard = env_guard();

        let md = "\
## 常用选项

| 选项 | 作用 |
| --- | --- |
| -b a | 对所有行编号（包括空行），等价于 `cat -n` |
| -b t | 仅对非空行编号（默认行为） |
";

        for cols in [96usize, 80, 72, 64] {
            unsafe { std::env::set_var("COLUMNS", cols.to_string()) };

            let stream = render_full_stream(md, false);
            let mut grid = VtGrid::new(cols);
            grid.feed(&stream);
            let screen = grid.screen();

            assert!(
                !screen.iter().any(|line| line.contains("| 选项 | 作用 |")),
                "COLUMNS={cols}: raw table header leaked onto final screen:\n{}",
                screen.join("\n")
            );
        }
    }

    #[test]
    fn streamed_single_column_table_renders_as_table() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "64") };

        let md = "\
| 函数签名 |
| --- |
| `processOrder(orderId: string)` |
";
        let stream = render_full_stream(md, false);
        let mut grid = VtGrid::new(64);
        grid.feed(&stream);
        let screen = grid.screen();
        let joined = screen.join("\n");

        assert!(joined.contains('┌'), "{joined}");
        assert!(joined.contains('│'), "{joined}");
        assert!(
            !joined.contains("| 函数签名 |"),
            "raw single-column markdown table leaked:\n{joined}"
        );
    }

    #[test]
    fn streamed_single_column_long_json_row_rewrites_without_raw_preview_residue() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "72") };

        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let mut stream = String::new();
        let md = "\
| 事件载荷 |
| --- |
| {\"event\":\"order.completed\",\"payload\":{\"orderId\":\"ORD-20260703-000001\",\"status\":\"paid\",\"items\":[{\"sku\":\"long-sku-code-001\",\"quantity\":2}],\"traceId\":\"trace-abcdefghijklmnopqrstuvwxyz\"}} |
";

        for ch in md.chars() {
            let mut buf = [0u8; 4];
            stream.push_str(
                &renderer
                    .write_chunk_for_test(ch.encode_utf8(&mut buf), false)
                    .unwrap(),
            );
        }

        let mut grid = VtGrid::new(72);
        grid.feed(&stream);
        let screen = grid.screen();
        let joined = screen.join("\n");

        assert!(joined.contains('┌'), "{joined}");
        assert!(joined.contains("事件载荷"), "{joined}");
        for line in &screen {
            assert!(
                !line.contains('|'),
                "raw markdown table preview leaked after row rewrite: {line:?}\n{}",
                screen.join("\n")
            );
            if line.contains("event") || line.contains("payload") || line.contains("trace") {
                assert!(
                    line.starts_with('│'),
                    "JSON table content must stay inside the rendered table: {line:?}\n{}",
                    screen.join("\n")
                );
            }
        }
    }

    #[test]
    fn confirmed_table_row_still_streams_realtime_before_newline() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "72") };

        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let mut stream = String::new();
        for chunk in ["| 事件载荷 |\n", "| --- |\n", "| {\"event\":\"order"] {
            stream.push_str(&renderer.write_chunk_for_test(chunk, false).unwrap());
        }

        assert!(
            stream.contains("{\"event\":\"order"),
            "confirmed table rows must keep realtime raw preview before newline; got {stream:?}"
        );
        assert!(
            !stream.ends_with('┘'),
            "partial row should not wait for final table rendering before newline"
        );
    }

    #[test]
    fn multi_column_table_with_overlong_code_span_stays_within_terminal_width() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "80") };

        let renderer = MarkdownStreamRenderer::new_with_tty(true);
        let header = vec![
            "函数签名".to_string(),
            "说明".to_string(),
            "返回值".to_string(),
        ];
        let align = vec![TableAlign::Left; 3];
        let rows = vec![vec![
            "`async processOrder(orderId: string, options?: { retry?: number; timeout?: number; callback?: (result: OrderResult) => void; metadata?: Record<string, string> }) => Promise<OrderResult>`".to_string(),
            "异步处理订单。该函数会依次执行：校验订单、锁定库存、调用支付网关、写入流水、释放锁、发送通知。".to_string(),
            "`{ success, data, error, traceId }`".to_string(),
        ]];

        let rendered = renderer.render_table_block("", &header, &align, &rows);
        assert!(!rendered.is_empty());
        for line in rendered.lines() {
            let visible = crate::ai::stream::extract::strip_ansi_codes(line);
            let width = unicode_width::UnicodeWidthStr::width(visible.as_str());
            assert!(
                width <= 80,
                "rendered table line exceeds terminal width ({width}):\n{visible}"
            );
            assert!(
                visible.starts_with(['┌', '├', '└', '│']),
                "table line should not be a wrapped continuation:\n{visible}"
            );
        }
    }
}
