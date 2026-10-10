use std::io::{self, Write};

use crate::ai::stream::render::code::{
    highlight_code_line, parse_code_block_language,
};
use crate::ai::stream::render::html::{
    contains_close_table_tag, contains_open_table_tag, parse_html_table, render_html_table,
};
use crate::ai::stream::render::inline::{render_inline_md, terminal_cell_width};
use crate::ai::stream::render::table::{
    TableAlign, TableState, compute_table_widths, is_table_row, is_table_row_candidate,
    is_table_separator, line_looks_like_table_preview, parse_table_align, parse_table_row,
    render_table_bottom, render_table_header, render_table_mid, render_table_row, render_table_top,
    split_indent, table_column_ranges, table_preview_height,
};
use crate::ai::stream::state::{END_THINKING_TAG_TEXT, THINKING_TAG_TEXT};
use crate::ai::theme;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MathBlockDelimiter {
    Dollars,
    Brackets,
    /// A fenced code block tagged `latex` / `tex` / `math`. The fence open/close
    /// lines act as the delimiters; the content is buffered and rendered through
    /// the math pipeline on close, so formulas appear like `$$...$$` blocks
    /// instead of a code box. Carries the opener's fence char and marker run so
    /// the closer is validated by the same rule as regular code blocks.
    Fenced {
        fence_char: char,
        fence_run: usize,
    },
}

impl MathBlockDelimiter {
    fn opening(line: &str) -> Option<Self> {
        match line {
            "$$" => Some(Self::Dollars),
            "\\[" => Some(Self::Brackets),
            _ => None,
        }
    }

    fn closes(self, line: &str) -> bool {
        match (self, line) {
            (Self::Dollars, "$$") | (Self::Brackets, "\\]") => true,
            (Self::Fenced { fence_char, fence_run }, line) => {
                // Close only on a bare fence of the same char with a run at least
                // as long as the opener: an inner fence line such as ` ```rust `
                // inside a fenced latex dump is formula content, not a terminator.
                // Same rule as the code-block state machine in `render_line_no_table`.
                is_closing_fence_line(line, fence_char, fence_run)
            }
            _ => false,
        }
    }
}

/// Classifies a fence-candidate line (already leading-trimmed): returns the fence
/// char, the homogeneous marker run length (>= 3), and the info-string remainder.
/// Per CommonMark, the info string of a backtick fence must not contain a
/// backtick, so a paragraph line like ````think`/```x` is plain text, not a
/// fence; tilde fences may contain backticks in their info string.
fn parse_fence_marker(trimmed: &str) -> Option<(char, usize, &str)> {
    let fence_char = trimmed.chars().next()?;
    if fence_char != '`' && fence_char != '~' {
        return None;
    }
    let run = trimmed.chars().take_while(|c| *c == fence_char).count();
    if run < 3 {
        return None;
    }
    let info = &trimmed[run..];
    if fence_char == '`' && info.contains('`') {
        return None;
    }
    Some((fence_char, run, info))
}

/// True when `trimmed` closes the open block that was started with `fence_char`
/// and an opening marker run of `open_len`. CommonMark closers are bare: the
/// same fence char, a run at least as long as the opener, and only whitespace
/// after. An info string (e.g. ` ```think ` inside an open block) marks content,
/// not a boundary — otherwise a nested-fence dump closes the outer block early
/// and the rest of the document leaks out of the box.
fn is_closing_fence_line(trimmed: &str, fence_char: char, open_len: usize) -> bool {
    matches!(
        parse_fence_marker(trimmed),
        Some((ch, run, info)) if ch == fence_char && run >= open_len && info.trim().is_empty()
    )
}

/// Incrementally classified tail of `line_buf.trim_start()`, advanced per arriving
/// char so the math-candidate decision stays O(1) instead of re-scanning the whole
/// buffered line each time. States cover every value `trim_start` can yield while
/// its length is ≤ 2; once that is impossible (`Decided`), appending more chars can
/// never bring the trimmed line back into the candidate set because the trimmed
/// tail only ever grows, so the classification is stable for any input sequence.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MathCandidateTail {
    /// Every char seen so far is whitespace, i.e. `trim_start(line_buf)` is "".
    Blank,
    /// Trimmed tail is exactly `$`.
    Dollar,
    /// Trimmed tail is exactly `\`.
    Backslash,
    /// Trimmed tail is exactly `$$`.
    DollarDollar,
    /// Trimmed tail is exactly `\[`.
    BracketOpen,
    /// Trimmed tail is exactly `\]`.
    BracketClose,
    /// The trimmed tail can no longer be one of the candidate tokens.
    Decided,
}

impl MathCandidateTail {
    fn advance(self, ch: char) -> Self {
        // Leading whitespace is invisible to `trim_start`, so whitespace keeps the
        // blank state. Whitespace *after* non-whitespace content extends past the
        // short tokens (only the leading run is trimmed), which rules them out.
        if ch.is_whitespace() {
            return if self == Self::Blank {
                Self::Blank
            } else {
                Self::Decided
            };
        }
        match self {
            Self::Blank => match ch {
                '$' => Self::Dollar,
                '\\' => Self::Backslash,
                _ => Self::Decided,
            },
            Self::Dollar => {
                if ch == '$' {
                    Self::DollarDollar
                } else {
                    Self::Decided
                }
            }
            Self::Backslash => match ch {
                '[' => Self::BracketOpen,
                ']' => Self::BracketClose,
                _ => Self::Decided,
            },
            _ => Self::Decided,
        }
    }

    fn is_math_candidate(self) -> bool {
        self != Self::Decided
    }
}

fn indent_math_block(rendered: &str, indent: &str) -> String {
    rendered
        .split('\n')
        .map(|line| format!("{indent}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(in crate::ai) struct MarkdownStreamRenderer {
    tty: bool,
    enabled: bool,
    in_code_block: bool,
    show_line_gutter: bool,
    code_block_indent: String,
    code_block_lang: Option<String>,
    /// Fence marker of the open block (char + opening run length); only
    /// meaningful while `in_code_block`. A close requires a bare run of the same
    /// char at least this long (CommonMark), so inner shorter or info-carrying
    /// fence lines stay content instead of terminating the block.
    code_block_fence_char: char,
    code_block_fence_len: usize,
    code_line_number: usize,
    math_block_delimiter: Option<MathBlockDelimiter>,
    math_block_indent: String,
    math_line_candidate_buffered: bool,
    bol: bool,
    line_buf: String,
    line_preview_emitted: bool,
    line_preview_height: usize,
    table_state: TableState,
    // 表格缓冲期已在屏幕上显示了一行占位提示（“⋯ 生成表格中”）。表格就绪一次性
    // 渲染时需先用确定的 `\x1b[1A\r\x1b[0J` 上移恰好 1 行清掉它——占位强制单行且
    // 短于终端宽，故这个“1 行”不依赖任何折行预测，不会重现残像叠表。
    table_placeholder_shown: bool,
    dimmed: bool,
    code_preview_segment_width: usize,
    // Per-line cache for code-block realtime rendering: `block_indent`, the line
    // number label, and the ioctl'd terminal width are constant within a line, so
    // recomputing them on every character cost 2 heap allocations + 1 syscall per
    // char. Refreshed lazily by `refresh_code_block_line_ctx` when the line number
    // or indent changes.
    code_block_ctx_line: usize,
    code_block_ctx_indent: String,
    code_block_ctx_num_str: String,
    code_block_ctx_width: usize,
    // 已缓存但尚未落地的纯空行数。正文/收尾常带尾随空行，逐行直出会在屏幕上
    // 堆叠成多余空白（尤其在正文结束到工具状态行之间）。缓存后：有真实内容跟进
    // 就照数补回（段间空行不受影响），若直到 flush 仍无内容则作为尾随空行丢弃。
    deferred_blank_lines: usize,
    // HTML 表格缓冲状态
    in_html_table: bool,
    html_table_buf: String,
    html_table_preview_height: usize,
    html_table_indent: String,
    // 数学块缓冲区：积累 $$/\[/\] 之间的内容，块结束后一次性渲染
    math_block_buf: Vec<String>,
    /// Number of chars currently buffered in `line_buf`. Maintained incrementally
    /// at every mutation point (the push/take helpers below are the only ones) so
    /// per-char handling stays O(1) instead of re-scanning the whole growing line.
    line_char_count: usize,
    /// Incremental classification of `line_buf.trim_start()`; see [`MathCandidateTail`].
    math_candidate_tail: MathCandidateTail,
    /// Whether `line_preview_height` no longer matches the already-echoed portion
    /// of `line_buf`. Echo paths mark this stale instead of recomputing the
    /// ANSI-aware height after every char; the height is recomputed lazily right
    /// before it is consumed or read. That yields the same value as eager
    /// recomputation because `line_buf` cannot change between the last echo of a
    /// line and its consumption.
    line_preview_height_stale: bool,
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
            code_block_fence_char: '`',
            code_block_fence_len: 3,
            code_line_number: 0,
            math_block_delimiter: None,
            math_block_indent: String::new(),
            math_line_candidate_buffered: false,
            bol: false,
            line_buf: String::new(),
            line_preview_emitted: false,
            line_preview_height: 0,
            table_state: TableState::None,
            table_placeholder_shown: false,
            dimmed: false,
            code_preview_segment_width: 0,
            code_block_ctx_line: usize::MAX,
            code_block_ctx_indent: String::new(),
            code_block_ctx_num_str: String::new(),
            code_block_ctx_width: 1,
            deferred_blank_lines: 0,
            in_html_table: false,
            html_table_buf: String::new(),
            html_table_preview_height: 0,
            html_table_indent: String::new(),
            math_block_buf: Vec::new(),
            line_char_count: 0,
            math_candidate_tail: MathCandidateTail::Blank,
            line_preview_height_stale: false,
        }
    }

    pub(in crate::ai::stream) fn should_render(&mut self, _chunk: &str) -> bool {
        if !self.tty {
            return false;
        }
        self.enabled = true;
        true
    }

    /// Whether the rendered output currently sits at a line start. Callers
    /// that interleave their own lines with renderer output use this to
    /// decide if a guard newline is still needed after a flush.
    pub(in crate::ai) fn at_line_start(&self) -> bool {
        self.bol
    }

    pub(in crate::ai::stream) fn write_chunk(
        &mut self,
        chunk: &str,
        dimmed: bool,
    ) -> io::Result<()> {
        let mut out = io::stdout();
        self.write_chunk_to(&mut out, chunk, dimmed)
    }

    pub(in crate::ai) fn write_block(&mut self, text: &str, dimmed: bool) -> io::Result<()> {
        let mut out = io::stdout();
        self.write_block_to(&mut out, text, dimmed)
    }

    pub(in crate::ai) fn write_chunk_to(
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

            self.push_line_char(ch);
            self.handle_char(out, ch)?;
            self.bol = false;
        }
        out.flush()?;
        Ok(())
    }

    fn write_block_to(&mut self, out: &mut dyn Write, text: &str, dimmed: bool) -> io::Result<()> {
        self.dimmed = dimmed;
        for segment in text.split_inclusive('\n') {
            if let Some(line) = segment.strip_suffix('\n') {
                self.push_line_str(line);
                self.handle_newline(out)?;
            } else {
                self.push_line_str(segment);
            }
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
    fn write_block_for_test(&mut self, text: &str, dimmed: bool) -> io::Result<String> {
        let mut out = Vec::new();
        self.write_block_to(&mut out, text, dimmed)?;
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    #[cfg(test)]
    fn flush_pending_for_test(&mut self) -> io::Result<String> {
        let mut out = Vec::new();

        // Blank lines still cached at flush time are trailing blanks; discard (do not emit).
        self.deferred_blank_lines = 0;

        if !self.line_buf.is_empty() {
            if self.line_preview_emitted {
                if self.in_code_block {
                    out.write_all(b"\x1b[0m\n")?;
                } else {
                    out.write_all(b"\n")?;
                }
                self.bol = true;
            }
            self.refresh_stale_line_preview_height();
            let line = self.take_line_buf();
            let rendered = self.consume_line(&line, self.line_preview_emitted);
            self.math_line_candidate_buffered = false;
            self.line_preview_emitted = false;
            self.line_preview_height = 0;
            self.line_preview_height_stale = false;
            self.code_preview_segment_width = 0;
            if !rendered.is_empty() {
                out.write_all(rendered.as_bytes())?;
                self.bol = rendered.ends_with('\n');
            }
        }

        if self.math_block_delimiter.take().is_some() {
            let indent = std::mem::take(&mut self.math_block_indent);
            let rendered = indent_math_block(
                &super::math::render_math_block(&self.math_block_buf),
                &indent,
            );
            self.math_block_buf.clear();
            if !rendered.is_empty() {
                out.write_all(rendered.as_bytes())?;
                out.write_all(b"\n")?;
                self.bol = true;
            }
        }

        // The block ended while an HTML table was still buffered: emit what arrived.
        let rendered = self.take_html_table_output();
        if !rendered.is_empty() {
            out.write_all(rendered.as_bytes())?;
            self.bol = rendered.ends_with('\n');
        }

        let state = std::mem::replace(&mut self.table_state, TableState::None);
        // Before finishing, clear any leftover table placeholder (always 1 line up; no wrap prediction).
        let mut rendered = self.clear_table_placeholder();
        rendered.push_str(&match state {
            TableState::None => String::new(),
            // The stream ended after the header row and no separator ever arrived — just an
            // ordinary text line containing `|`; nothing was echoed, so render it as a plain line.
            TableState::PendingHeader {
                indent,
                header_line,
            } => self.render_buffered_plain(&indent, &header_line),
            // Table finishing at end of stream: draw the final boxed table in one shot, no cursor-up.
            TableState::InTable {
                indent,
                header,
                align,
                rows,
            } => self.render_table_block(&indent, &header, &align, &rows),
        });
        if !rendered.is_empty() {
            out.write_all(rendered.as_bytes())?;
            self.bol = rendered.ends_with('\n');
        }
        out.flush()?;
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    pub(in crate::ai::stream::render) fn line_preview_height(&mut self) -> usize {
        self.refresh_stale_line_preview_height();
        self.line_preview_height
    }

    pub(in crate::ai::stream::render) fn set_line_preview_height(&mut self, height: usize) {
        self.line_preview_height_stale = false;
        self.line_preview_height = height;
    }

    pub(in crate::ai::stream::render) fn code_block_lang(&self) -> Option<&str> {
        self.code_block_lang.as_deref()
    }

    pub(in crate::ai::stream) fn has_unfinished_line(&self) -> bool {
        !self.line_buf.is_empty()
    }

    /// The only append path into `line_buf`, keeping [`Self::line_char_count`] and
    /// the incremental math-candidate classification in sync with the buffer.
    fn push_line_char(&mut self, ch: char) {
        self.line_buf.push(ch);
        self.line_char_count += 1;
        self.math_candidate_tail = self.math_candidate_tail.advance(ch);
    }

    fn push_line_str(&mut self, s: &str) {
        for ch in s.chars() {
            self.push_line_char(ch);
        }
    }

    /// The only reset path for `line_buf`; clears the derived incremental state so
    /// the counters describe exactly the chars still in the buffer.
    fn take_line_buf(&mut self) -> String {
        self.line_char_count = 0;
        self.math_candidate_tail = MathCandidateTail::Blank;
        std::mem::take(&mut self.line_buf)
    }

    /// Recomputes the preview height if an echo path marked it stale; see the
    /// `line_preview_height_stale` field for why this is equivalent to the former
    /// per-char eager recomputation.
    fn refresh_stale_line_preview_height(&mut self) {
        if !self.line_preview_height_stale {
            return;
        }
        self.line_preview_height = self.current_line_preview_height();
        self.line_preview_height_stale = false;
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

        self.refresh_stale_line_preview_height();
        let line = self.take_line_buf();
        self.math_line_candidate_buffered = false;

        // 纯空行且不在代码块/数学块/表格上下文：先缓存，不立即落地。等真实内容跟进
        // 时照数补回（段间空行不受影响）；若直到 flush 仍无内容，则作为尾随空行丢弃，
        // 避免正文结束到工具状态行之间堆叠出多余空白。
        if !self.line_preview_emitted
            && line.trim().is_empty()
            && !self.in_code_block
            && self.math_block_delimiter.is_none()
            && !self.in_html_table
            && matches!(self.table_state, TableState::None)
        {
            self.deferred_blank_lines += 1;
            self.line_preview_height = 0;
            self.line_preview_height_stale = false;
            self.code_preview_segment_width = 0;
            return Ok(());
        }

        let rendered = self.consume_line(&line, self.line_preview_emitted);

        self.line_preview_emitted = false;
        self.line_preview_height = 0;
        self.line_preview_height_stale = false;
        self.code_preview_segment_width = 0;

        if !rendered.is_empty() {
            self.flush_deferred_blank_lines(out)?;
            out.write_all(rendered.as_bytes())?;
            out.flush()?;
            self.bol = rendered.ends_with('\n');
        }
        Ok(())
    }

    /// 把缓存的纯空行照数补回（有真实内容跟进时调用），保证段落间的空行不受影响。
    fn flush_deferred_blank_lines(&mut self, out: &mut dyn Write) -> io::Result<()> {
        if self.deferred_blank_lines == 0 {
            return Ok(());
        }
        for _ in 0..self.deferred_blank_lines {
            out.write_all(b"\n")?;
        }
        self.deferred_blank_lines = 0;
        self.bol = true;
        Ok(())
    }

    fn handle_char(&mut self, out: &mut dyn Write, ch: char) -> io::Result<()> {
        // Restore deferred paragraph spacing before the first content character.
        if self.deferred_blank_lines > 0 && self.line_char_count == 1 {
            self.flush_deferred_blank_lines(out)?;
        }
        // Buffer table candidates without echoing. consume_line/flush renders the
        // finished table once, avoiding cursor-up estimates and leftover previews.
        if self.should_buffer_table_line() {
            return Ok(());
        }

        // Buffer math blocks and delimiter candidates to avoid echoing raw TeX
        // before the rendered formula. Replay the buffered prefix once it can
        // no longer be a math delimiter, using the same color as ordinary prose.
        if !self.in_code_block {
            let is_math_candidate =
                self.math_block_delimiter.is_some() || self.math_candidate_tail.is_math_candidate();
            if is_math_candidate {
                self.math_line_candidate_buffered = true;
                return Ok(());
            }
            if self.math_line_candidate_buffered {
                self.math_line_candidate_buffered = false;
                out.write_all(
                    if self.dimmed {
                        theme::current().accent_muted.as_bytes()
                    } else {
                        theme::current().markdown_body.as_bytes()
                    },
                )?;
                out.write_all(self.line_buf.as_bytes())?;
                self.line_preview_emitted = true;
                self.line_preview_height_stale = true;
                return Ok(());
            }
        }
        self.handle_realtime_output(out, ch)
    }

    fn handle_realtime_output(&mut self, out: &mut dyn Write, ch: char) -> io::Result<()> {
        if self.in_code_block {
            self.handle_code_block_realtime_output(out, ch)?;
            self.line_preview_emitted = true;
            self.line_preview_height_stale = true;
            return Ok(());
        }
        if self.line_char_count == 1 {
            out.write_all(
                if self.dimmed {
                    theme::current().accent_muted.as_bytes()
                } else {
                    theme::current().markdown_body.as_bytes()
                },
            )?;
        }
        self.emit_char(out, ch)?;
        self.line_preview_emitted = true;
        self.line_preview_height_stale = true;
        Ok(())
    }

    fn handle_code_block_realtime_output(
        &mut self,
        out: &mut dyn Write,
        ch: char,
    ) -> io::Result<()> {
        self.refresh_code_block_line_ctx();
        let block_indent = self.code_block_ctx_indent.as_str();
        let line_num_str = self.code_block_ctx_num_str.as_str();
        let available_width = self.code_block_ctx_width;

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

        let ch_width = terminal_cell_width(ch);

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

        let mut buf = [0u8; 4];
        out.write_all(ch.encode_utf8(&mut buf).as_bytes())?;
        self.code_preview_segment_width += ch_width;
        self.line_preview_emitted = true;
        self.line_preview_height_stale = true;
        Ok(())
    }

    fn current_line_preview_height(&self) -> usize {
        if self.in_code_block {
            return self.code_block_preview_height(&self.line_buf);
        }
        live_preview_cursor_rows(&self.line_buf)
    }

    /// Refresh the per-line code-block context (indent copy, line-number label,
    /// terminal width) when the current line or indent changed. The width re-queries
    /// the terminal once per line instead of per character; the finished line's
    /// height is still recomputed with a fresh ioctl in `code_block_preview_height`,
    /// so a mid-line resize only affects wrapping of the current partial line.
    fn refresh_code_block_line_ctx(&mut self) {
        if self.code_block_ctx_line != self.code_line_number
            || self.code_block_ctx_indent != self.code_block_indent
        {
            self.code_block_ctx_line = self.code_line_number;
            self.code_block_ctx_indent = self.code_block_indent.clone();
            self.code_block_ctx_num_str = format!("{:>3}", self.code_line_number + 1);
            self.code_block_ctx_width = code_block_content_width(
                &self.code_block_ctx_indent,
                &self.code_block_ctx_num_str,
                self.show_line_gutter,
            )
            .max(1);
        }
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

    pub(in crate::ai) fn flush_pending(&mut self) -> io::Result<()> {
        let mut out = io::stdout();
        self.flush_pending_to(&mut out)
    }

    /// Same body as `flush_pending`, but writes to the given sink instead of
    /// stdout, so callers rendering into another buffer (serve chat output)
    /// share the exact streaming behavior.
    pub(in crate::ai) fn flush_pending_to(&mut self, out: &mut dyn Write) -> io::Result<()> {

        // Blank lines still cached at flush time are trailing blanks; discard (do not emit).
        self.deferred_blank_lines = 0;

        if !self.line_buf.is_empty() {
            if self.line_preview_emitted {
                if self.in_code_block {
                    out.write_all(b"\x1b[0m\n")?;
                } else {
                    out.write_all(b"\n")?;
                }
                self.bol = true;
            }
            self.refresh_stale_line_preview_height();
            let line = self.take_line_buf();
            let rendered = self.consume_line(&line, self.line_preview_emitted);
            self.math_line_candidate_buffered = false;
            self.line_preview_emitted = false;
            self.line_preview_height = 0;
            self.line_preview_height_stale = false;
            self.code_preview_segment_width = 0;
            if !rendered.is_empty() {
                out.write_all(rendered.as_bytes())?;
                self.bol = rendered.ends_with('\n');
            }
        }

        // 先消费最后一段无换行文本，再收尾未闭合数学块；否则末行会漏出为原始 TeX。
        if self.math_block_delimiter.take().is_some() {
            let indent = std::mem::take(&mut self.math_block_indent);
            let rendered = indent_math_block(
                &super::math::render_math_block(&self.math_block_buf),
                &indent,
            );
            self.math_block_buf.clear();
            if !rendered.is_empty() {
                let line = format!("{}{rendered}\x1b[0m\n", theme::current().accent_secondary);
                out.write_all(line.as_bytes())?;
                self.bol = true;
            }
        }

        // The stream ended while an HTML table was still buffered: emit what arrived.
        let rendered = self.take_html_table_output();
        if !rendered.is_empty() {
            out.write_all(rendered.as_bytes())?;
            self.bol = rendered.ends_with('\n');
        }

        let state = std::mem::replace(&mut self.table_state, TableState::None);
        // Before finishing, clear any leftover table placeholder (always 1 line up; no wrap prediction).
        let mut rendered = self.clear_table_placeholder();
        rendered.push_str(&match state {
            TableState::None => String::new(),
            // The stream ended after the header row and no separator ever arrived — just an
            // ordinary text line containing `|`; nothing was echoed, so render it as a plain line.
            TableState::PendingHeader {
                indent,
                header_line,
            } => self.render_buffered_plain(&indent, &header_line),
            // Table finishing at end of stream: draw the final boxed table in one shot, no cursor-up.
            TableState::InTable {
                indent,
                header,
                align,
                rows,
            } => self.render_table_block(&indent, &header, &align, &rows),
        });
        if !rendered.is_empty() {
            out.write_all(rendered.as_bytes())?;
            self.bol = rendered.ends_with('\n');
        }
        out.flush()
    }

    fn should_buffer_table_line(&self) -> bool {
        // 已在表格上下文（表头待确认 / 表格中）：后续所有行都静默缓冲。
        if matches!(
            self.table_state,
            TableState::PendingHeader { .. } | TableState::InTable { .. }
        ) {
            return true;
        }
        // 尚未进入表格，但当前这行看起来像表格行（含 `|`）：乐观缓冲，不逐字 echo。
        // 若整行落定后并非表格，consume_line 会按普通行渲染它（从未 echo，无残留）。
        // `!line_preview_emitted` 保证：一旦本行已经开始实时 echo（如首 token 不含
        // `|` 的裸表头），就不再中途切换到缓冲，避免半 echo 半缓冲的错位。
        !self.in_code_block
            && !self.line_preview_emitted
            && line_looks_like_table_preview(&self.line_buf)
    }

    pub(in crate::ai) fn consume_line(&mut self, line: &str, preview_emitted: bool) -> String {
        // 数学块中的 `|` 是公式内容，不得先进入 Markdown 表格状态机。
        if self.math_block_delimiter.is_some() {
            let rendered = self.render_line_no_table(line);
            if preview_emitted && !line.is_empty() {
                let preview_height = self.line_preview_height.max(1);
                return format!("\x1b[{preview_height}A\r\x1b[0J{rendered}");
            }
            return rendered;
        }

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
                    // 表头行落定：记录状态并显示单行占位提示（表格生成期不再空窗）。
                    let (indent, rest) = split_indent(line);
                    // 裸表头（无前导 `|`）或带缩进的表头，其行首在识别为表头之前已被逐字
                    // echo（首 token 不含 `|`，should_buffer_table_line 拒绝中途切缓冲）。
                    // 若不回收这些已 echo 的物理行，原始 markdown 会泄漏在成品表格上方。
                    let mut out = String::new();
                    if preview_emitted && !line.is_empty() {
                        let preview_height = self.line_preview_height.max(1);
                        out.push_str(&format!("\x1b[{preview_height}A\r\x1b[0J"));
                    }
                    out.push_str(&self.table_placeholder_line(indent));
                    self.table_state = TableState::PendingHeader {
                        indent: indent.to_string(),
                        header_line: rest.trim_end().to_string(),
                    };
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
            } => {
                if is_table_separator(line) {
                    // 分隔行确认这是真表格：进入 InTable，继续静默缓冲，占位保留。
                    let header_cells = parse_table_row(&header_line);
                    let align = parse_table_align(line, header_cells.len());
                    self.table_state = TableState::InTable {
                        indent,
                        header: header_cells,
                        align,
                        rows: Vec::new(),
                    };
                    return String::new();
                }

                // 表头行后并非分隔行——说明先前那行只是含 `|` 的普通文本。先清掉占位
                // 提示，再把它当普通行渲染，最后处理当前行。
                let mut out = self.clear_table_placeholder();
                out.push_str(&self.render_buffered_plain(&indent, &header_line));
                self.table_state = TableState::None;
                out.push_str(&self.consume_line(line, false));
                out
            }
            TableState::InTable {
                indent,
                header,
                align,
                mut rows,
            } => {
                if is_table_row(line) {
                    // 表格数据行：累积到缓冲，暂不输出（等表格结束一次性画出）。
                    rows.push(parse_table_row(line));
                    self.table_state = TableState::InTable {
                        indent,
                        header,
                        align,
                        rows,
                    };
                    return String::new();
                }

                // 遇到非表格行：先清占位，一次性画出成品盒框表，再处理当前行。
                let mut out = self.clear_table_placeholder();
                out.push_str(&self.render_table_block(&indent, &header, &align, &rows));
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

    /// HTML table buffering finished: parse it into a terminal table.
    fn finalize_html_table(&mut self, preview_emitted: bool) -> String {
        let buf = std::mem::take(&mut self.html_table_buf);
        let indent = std::mem::take(&mut self.html_table_indent);
        let preview_height = std::mem::take(&mut self.html_table_preview_height);
        self.in_html_table = false;

        let rendered = match parse_html_table(&buf) {
            Some(table) => render_html_table(&indent, &table),
            // No `</table>` arrived: keep the model's raw text (see `ensure_trailing_newline`).
            None => Self::ensure_trailing_newline(buf),
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

    /// Emit an HTML table buffer that the stream or the block ended on. The buffer is parsed and
    /// rendered as a table when it is complete, otherwise it is kept as raw text.
    ///
    /// Rows echoed while buffering are erased with a deterministic cursor-up of
    /// `html_table_preview_height`, so the final form is drawn exactly once.
    fn take_html_table_output(&mut self) -> String {
        if !self.in_html_table {
            return String::new();
        }
        let buf = std::mem::take(&mut self.html_table_buf);
        let indent = std::mem::take(&mut self.html_table_indent);
        let preview_height = std::mem::take(&mut self.html_table_preview_height);
        self.in_html_table = false;

        let rendered = match parse_html_table(&buf) {
            Some(table) => render_html_table(&indent, &table),
            // No `</table>` arrived: keep the model's raw text (see `ensure_trailing_newline`).
            None => Self::ensure_trailing_newline(buf),
        };

        if preview_height > 0 {
            format!("\x1b[{preview_height}A\r\x1b[0J{rendered}")
        } else {
            rendered
        }
    }

    /// Keeps a raw fallback line-terminated.
    ///
    /// Rendered blocks always end with `\n`. A fallback that stopped mid-row would leave the
    /// terminal cursor on that row, and the multiline input box anchors on the live cursor row
    /// (`prompt/multiline/multiline_ui.rs`, `fixed_viewport_area`), so the input line would then
    /// be drawn over the last output row.
    fn ensure_trailing_newline(mut text: String) -> String {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text
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

        let ranges = table_column_ranges(indent, cols);
        if ranges.len() > 1 {
            let mut final_table = String::new();
            for (idx, range) in ranges.into_iter().enumerate() {
                if idx > 0 {
                    final_table.push('\n');
                    final_table.push_str(&self.render_table_column_block_continuation(
                        indent, header, align, rows, range,
                    ));
                } else {
                    final_table.push_str(
                        &self.render_table_column_block(indent, header, align, rows, range),
                    );
                }
            }
            return final_table;
        }

        self.render_table_column_block(indent, header, align, rows, 0..cols)
    }

    /// 续接列块：不画顶部边框，用 mid separator 衔接上一块，避免宽表分列时
    /// 每块都像独立新表（header 反复出现）。
    fn render_table_column_block_continuation(
        &self,
        indent: &str,
        header: &[String],
        align: &[TableAlign],
        rows: &[Vec<String>],
        range: std::ops::Range<usize>,
    ) -> String {
        let header = range
            .clone()
            .map(|idx| header.get(idx).cloned().unwrap_or_default())
            .collect::<Vec<_>>();
        let align = range
            .clone()
            .map(|idx| align.get(idx).copied().unwrap_or(TableAlign::Left))
            .collect::<Vec<_>>();
        let rows = rows
            .iter()
            .map(|row| {
                range
                    .clone()
                    .map(|idx| row.get(idx).cloned().unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let widths = compute_table_widths(indent, &header, &rows);
        let mut out = String::new();
        // 续接块：用 mid 代替 top border，视觉上表示"承接上一块"
        out.push_str(&render_table_mid(indent, &widths));
        out.push_str(&render_table_header(indent, &header, &align, &widths));
        out.push_str(&render_table_mid(indent, &widths));
        for row in &rows {
            out.push_str(&render_table_row(indent, row, &align, &widths));
        }
        out.push_str(&render_table_bottom(indent, &widths));
        out
    }

    fn render_table_column_block(
        &self,
        indent: &str,
        header: &[String],
        align: &[TableAlign],
        rows: &[Vec<String>],
        range: std::ops::Range<usize>,
    ) -> String {
        let header = range
            .clone()
            .map(|idx| header.get(idx).cloned().unwrap_or_default())
            .collect::<Vec<_>>();
        let align = range
            .clone()
            .map(|idx| align.get(idx).copied().unwrap_or(TableAlign::Left))
            .collect::<Vec<_>>();
        let rows = rows
            .iter()
            .map(|row| {
                range
                    .clone()
                    .map(|idx| row.get(idx).cloned().unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let widths = compute_table_widths(indent, &header, &rows);
        let mut final_table = String::new();
        final_table.push_str(&render_table_top(indent, &widths));
        final_table.push_str(&render_table_header(indent, &header, &align, &widths));
        final_table.push_str(&render_table_mid(indent, &widths));
        for row in &rows {
            final_table.push_str(&render_table_row(indent, row, &align, &widths));
        }
        final_table.push_str(&render_table_bottom(indent, &widths));
        final_table
    }

    /// 渲染一整行静默缓冲的普通文本（此前被当作潜在表头，最终未成表）。
    /// 因全程未 echo，无需 cursor-up，直接走通用渲染即可。
    fn render_buffered_plain(&mut self, indent: &str, rest: &str) -> String {
        self.render_line_no_table(&format!("{indent}{rest}"))
    }

    /// 表格缓冲期的单行占位提示。仅在 tty 且尚未显示时返回一行短提示并置位标志。
    /// 强制单行、短于终端宽，保证只占 1 个物理行，清除时上移量恒为 1。
    fn table_placeholder_line(&mut self, indent: &str) -> String {
        if !self.tty || self.table_placeholder_shown {
            return String::new();
        }
        self.table_placeholder_shown = true;
        format!("{indent}{}⋯ 生成表格中\x1b[0m\n", theme::current().accent_muted)
    }

    /// 若占位提示正显示在屏幕上，返回清除它的确定序列（上移 1 行并清到屏末）。
    /// 上移量恒为 1，不依赖任何折行预测，故不会重现残像叠表。
    fn clear_table_placeholder(&mut self) -> String {
        if !self.table_placeholder_shown {
            return String::new();
        }
        self.table_placeholder_shown = false;
        "\x1b[1A\r\x1b[0J".to_string()
    }

    fn render_line_no_table(&mut self, line: &str) -> String {
        let (indent, rest) = split_indent(line);
        let trimmed = rest.trim_start_matches([' ', '\t']);

        let base = if self.dimmed { "\x1b[2m" } else { "" };

        if trimmed == THINKING_TAG_TEXT || trimmed == END_THINKING_TAG_TEXT {
            self.in_code_block = false;
            self.code_block_indent.clear();
            self.code_block_lang = None;
            self.code_block_fence_char = '`';
            self.code_block_fence_len = 3;
            self.code_line_number = 0;

            let label = if trimmed == THINKING_TAG_TEXT {
                "○ thinking"
            } else {
                "✓ thinking"
            };
            return format!("{indent}{}{label}\x1b[0m\n", theme::current().accent_muted);
        }

        // Fence boundaries follow CommonMark: inside an open block only a *bare*
        // fence of the same char and at least the opening run length closes it,
        // so an inner dump like ` ```think ` stays content instead of closing the
        // block and spilling the rest of the document outside the box. A ```/~~~
        // line that is not a valid fence (e.g. a backtick-fence info string that
        // itself contains backticks) is ordinary text, never a boundary.
        let fence_boundary = if self.math_block_delimiter.is_none() {
            if self.in_code_block {
                is_closing_fence_line(
                    trimmed,
                    self.code_block_fence_char,
                    self.code_block_fence_len,
                )
            } else {
                parse_fence_marker(trimmed).is_some()
            }
        } else {
            // A math block may already be open (e.g. a fenced latex block); its
            // closing fence line must reach the math-close path below, not
            // re-enter the code fence state machine here.
            false
        };
        if fence_boundary {
            if self.in_code_block {
                self.in_code_block = false;
                self.code_block_lang = None;
                self.code_block_fence_char = '`';
                self.code_block_fence_len = 3;
                let block_indent = std::mem::take(&mut self.code_block_indent);
                let border = "─".repeat(22);
                return format!(
                    "{block_indent}{}{}╰{border}\x1b[0m\n",
                    theme::current().code_background,
                    theme::current().code_dim
                );
            } else {
                let (fence_char, fence_len, _) = parse_fence_marker(trimmed)
                    .expect("fence_boundary implies a valid fence marker");
                let lang = parse_code_block_language(trimmed);
                // `latex` / `tex` / `math` fences carry formulas: buffer the lines and
                // render them through the math pipeline on close, exactly like `$$...$$`.
                if matches!(lang.as_deref(), Some("latex" | "tex" | "math")) {
                    self.math_block_delimiter =
                        Some(MathBlockDelimiter::Fenced { fence_char, fence_run: fence_len });
                    self.math_block_indent = indent.to_string();
                    self.math_block_buf.clear();
                    return String::new();
                }
                self.in_code_block = true;
                self.code_block_indent = indent.to_string();
                self.code_block_lang = lang;
                self.code_block_fence_char = fence_char;
                self.code_block_fence_len = fence_len;
                self.code_line_number = 0;
                let lang = self.code_block_lang.as_deref().unwrap_or("code");
                return format!(
                    "{indent}{}{}╭─ {lang}\x1b[0m\n",
                    theme::current().code_background,
                    theme::current().code_dim
                );
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
                        "{block_indent}{}{}{} │\x1b[0m\n",
                        theme::current().code_background,
                        theme::current().code_dim,
                        line_num_str
                    );
                }
                return format!("{block_indent}{}\x1b[0m\n", theme::current().code_background);
            }
            let wrapped = wrap_code_block_text(
                code_text,
                code_block_content_width(block_indent, &line_num_str, self.show_line_gutter),
            );
            let mut out = String::new();
            for (idx, segment) in wrapped.iter().enumerate() {
                out.push_str(block_indent);
                out.push_str(theme::current().code_background);
                if self.show_line_gutter {
                    let gutter = if idx == 0 {
                        line_num_str.as_str()
                    } else {
                        "   "
                    };
                    out.push_str(theme::current().code_dim);
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

        if let Some(delimiter) = self.math_block_delimiter {
            if delimiter.closes(trimmed) {
                self.math_block_delimiter = None;
                let block_indent = std::mem::take(&mut self.math_block_indent);
                let rendered = indent_math_block(
                    &super::math::render_math_block(&self.math_block_buf),
                    &block_indent,
                );
                self.math_block_buf.clear();
                if !rendered.is_empty() {
                    return format!("{base}{}{rendered}\x1b[0m\n", theme::current().accent_secondary);
                }
                return String::new();
            }

            self.math_block_buf.push(rest.trim_end().to_string());
            return String::new();
        }

        if let Some(delimiter) = MathBlockDelimiter::opening(trimmed) {
            self.math_block_delimiter = Some(delimiter);
            self.math_block_indent = indent.to_string();
            self.math_block_buf.clear();
            return String::new();
        }

        // Dimmed prose is the thinking channel: use the theme's muted color directly instead of
        // dimming markdown_body (SGR 2m applied to the body color is barely distinguishable from
        // plain body text on many terminals). accent_muted contrasts with markdown_body in every
        // builtin theme, so thinking stays visibly distinct from the final answer.
        let prose_base = if self.dimmed {
            theme::current().accent_muted
        } else {
            theme::current().markdown_body
        };
        let base = prose_base;

        if let Some((level, title)) = parse_heading(trimmed) {
            let underline_char = match level {
                1 => Some('━'),
                2 => Some('─'),
                _ => None,
            };
            let mut out = String::new();
            if !self.bol {
                out.push('\n');
                self.bol = true;
            }
            out.push_str(indent);
            let combined_base = format!("{base}\x1b[1m{}", theme::current().markdown_heading);
            out.push_str(&render_inline_md(title, &combined_base));
            out.push_str("\x1b[0m\n");

            if let Some(ch) = underline_char {
                let len = title.chars().count().clamp(3, 80);
                out.push_str(indent);
                out.push_str(base);
                out.push_str("\x1b[2m");
                out.push_str(theme::current().accent_rule);
                out.push_str(&std::iter::repeat_n(ch, len).collect::<String>());
                out.push_str("\x1b[0m\n");
            }
            return out;
        }

        if is_thematic_break(trimmed) {
            return format!(
                "{indent}{base}{}{}\x1b[0m\n",
                theme::current().accent_rule,
                "─".repeat(28)
            );
        }

        if let Some(body) = parse_blockquote(trimmed) {
            let quote_base = format!("{base}{}", theme::current().accent_muted);
            // Prepend the prefix in place so the rendered body is not copied a
            // second time into a fresh `format!` buffer.
            let mut out = render_inline_md(body, &quote_base);
            out.insert_str(
                0,
                &format!("{indent}{base}{}▍\x1b[0m ", theme::current().accent_muted),
            );
            out.push('\n');
            return out;
        }

        if let Some((p_indent, prefix, checkbox, body)) = split_list_prefix(line) {
            let mut out = String::new();
            out.push_str(p_indent);
            if let Some(checked) = checkbox {
                out.push_str(base);
                if checked {
                    out.push_str(theme::current().accent_success);
                    out.push('✓');
                } else {
                    out.push_str(theme::current().accent_muted);
                    out.push('○');
                }
                out.push_str("\x1b[0m ");
            } else if prefix.ends_with(". ") {
                out.push_str(base);
                out.push_str(theme::current().accent_muted);
                out.push_str(prefix.trim_end());
                out.push_str("\x1b[0m ");
            } else {
                out.push_str(base);
                out.push_str(theme::current().accent_primary);
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
        format!("{}{}\n", indent, render_inline_md(rest, base))
    }
}

mod width;
mod parser;
#[cfg(test)]
mod tests;

// Re-export externally-visible / test-visible measurement helpers through the
// `markdown` module root so `render::markdown::X` paths (stream runtime) and the
// inline `tests` module (via `use super::*`) keep their pre-split names.
pub(in crate::ai) use width::{
    clamp_line_to_terminal_row, clamp_line_to_terminal_row_with_reserve,
    live_preview_cursor_rows, raw_terminal_rows, wrap_line_to_terminal_rows_with_reserve,
};
// Impl-only helpers: imported privately for the methods above; not part of the
// public `markdown` surface.
use width::{
    code_block_content_width, code_block_preview_continuation_prefix, code_block_preview_prefix,
    wrap_code_block_text,
};
use parser::{is_thematic_break, parse_blockquote, parse_heading, split_list_prefix};
