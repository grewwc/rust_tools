//! Terminal width / measurement helpers for live-region and code-block preview
//! rendering. Pure measurement functions over terminal geometry; the streaming
//! state machine lives in the parent `markdown` module.

use crate::ai::stream::extract::strip_ansi_codes;
use crate::ai::stream::render::inline::terminal_cell_width;
use crate::ai::theme;

/// Wrapped width for **live regions**: the live terminal width. Live regions are the rows the renderer
/// redraws in place with relative cursor moves (thinking/subagent fold body and markers, fold status
/// header, tool-output preview, waiting hints). The answer body, code blocks and tables are not live
/// regions and keep their own widths.
///
/// Live rows deliberately use the same width the terminal uses to lay them out, so a long logical row
/// still shows as much text as before. Narrowing makes the terminal re-wrap rows it has already drawn
/// (xterm.js reflows immediately, while the new PTY winsize can reach this process one or more redraws
/// later), so the erase side must never assume one physical row per logical row: it recomputes the
/// footprint from the stored plain text at the current width instead
/// (`thinking_fold_rendered_body_rows` / `thinking_fold_header_rendered_rows` and the waiting-hint
/// erase in `stream/runtime.rs`).
pub(super) fn live_region_cols() -> usize {
    raw_terminal_cols()
}

/// Hard-truncate one line of text to a single physical terminal row at the live-region column width,
/// ending an over-long line with `…`.
///
/// Live regions use it so every drawn row occupies exactly one physical row and the cursor-up erase
/// row count matches the logical row count, instead of predicting terminal auto-wrap (tabs, full-width
/// characters, over-long lines). The truncation width is the live terminal width, so a wide terminal
/// still shows the whole line; if the terminal narrows afterwards it re-wraps that row, and the erase
/// side recomputes the footprint from the stored text (see [`live_region_cols`]). Input is plain text
/// without ANSI.
pub(in crate::ai) fn clamp_line_to_terminal_row(line: &str) -> String {
    clamp_line_to_terminal_row_with_reserve(line, 0)
}

/// Same as [`clamp_line_to_terminal_row`], but reserves `reserve_cols` columns for a line prefix
/// (such as the fold's `  │ ` indent) so that prefix plus clamped text still fits one physical row.
pub(in crate::ai) fn clamp_line_to_terminal_row_with_reserve(
    line: &str,
    reserve_cols: usize,
) -> String {
    let cols = live_region_cols().saturating_sub(reserve_cols).max(1);
    let mut total = 0usize;
    for ch in line.chars() {
        total += terminal_cell_width(ch);
    }
    if total <= cols {
        return line.to_string();
    }

    // Truncation: with a single remaining column only the ellipsis fits; otherwise reserve one column
    // for the ellipsis so the result with the ellipsis still fits within `cols`.
    if cols == 1 {
        return "…".to_string();
    }
    let budget = cols - 1;
    let mut out = String::with_capacity(line.len());
    let mut col = 0usize;
    for ch in line.chars() {
        let w = terminal_cell_width(ch);
        if col + w > budget {
            break;
        }
        out.push(ch);
        col += w;
    }
    out.push('…');
    out
}

/// 按当前终端宽度把单个逻辑行拆成多条可见行，并为行首装饰预留列宽。
///
/// Unlike [`clamp_line_to_terminal_row_with_reserve`] this never truncates content; callers usually
/// re-apply the same prefix/indent to every returned segment so manually wrapped rows stay in one
/// block.
pub(in crate::ai) fn wrap_line_to_terminal_rows_with_reserve(
    line: &str,
    reserve_cols: usize,
) -> Vec<String> {
    let cols = live_region_cols().saturating_sub(reserve_cols).max(1);
    if line.is_empty() {
        return vec![String::new()];
    }

    let mut rows = Vec::new();
    let mut current = String::new();
    let mut col = 0usize;
    for ch in line.chars() {
        let w = terminal_cell_width(ch);
        // With a single available column a wide character cannot avoid terminal auto-wrap; substitute a
        // one-column ASCII placeholder to keep the "each returned segment is exactly one physical row"
        // redraw invariant.
        if w > cols {
            if !current.is_empty() {
                rows.push(std::mem::take(&mut current));
                col = 0;
            }
            rows.push("?".to_string());
            continue;
        }
        if col > 0 && col + w > cols {
            rows.push(std::mem::take(&mut current));
            col = 0;
        }
        current.push(ch);
        col += w;
    }
    rows.push(current);
    rows
}

pub(in crate::ai) fn live_preview_cursor_rows(line: &str) -> usize {
    // Preview characters are written verbatim and wrap at the terminal's actual width,
    // so count physical rows with raw_terminal_cols, not the margin-reduced preview width.
    // A narrower width can count a one-row preview as two, moving the cursor up too far
    // and clearing content above the table or leaving preview fragments after a redraw.
    live_preview_cursor_rows_at(line, raw_terminal_cols())
}

/// Physical rows `line` occupies on a terminal `cols` columns wide.
///
/// Split out from [`live_preview_cursor_rows`] so a caller that must judge a row at a width other than
/// the live one shares the same rule: the thinking fold asks how tall a region it has already written
/// would be at a width this process learns only after the terminal reflowed it.
/// Match DECAWM wrapping: wide characters wrap early when col + w > cols.
pub(in crate::ai) fn live_preview_cursor_rows_at(line: &str, cols: usize) -> usize {
    let cols = cols.max(1);
    let visible = strip_ansi_codes(line);
    let mut lines = 1usize;
    let mut col = 0usize;
    for ch in visible.chars() {
        let w = terminal_cell_width(ch);
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

pub(super) fn preview_terminal_width() -> usize {
    raw_terminal_cols()
        .saturating_sub(RIGHT_MARGIN)
        .max(MIN_PREVIEW_WIDTH)
}

pub(in crate::ai) fn raw_terminal_cols() -> usize {
    // Emulator-truth width cached by the fold's `CSI 18 t` refresh (see
    // `side_note_input::refresh_true_width`): while xterm.js has rewrapped the rows it
    // already drew but the PTY winsize has not reached this process, the kernel width
    // below is stale, and every cursor-up span measured from it stops short, stacking
    // the previous `○ thinking` header under the redraw. A cache miss keeps this
    // function's ioctl/COLUMNS behavior unchanged.
    if let Some(cols) = crate::ai::stream::side_note_input::fresh_true_width_cols() {
        return cols as usize;
    }
    // Prefer ioctl(TIOCGWINSZ) for the live width: the long-running `a` process inherits
    // COLUMNS at startup, so narrowing a terminal panel can leave it larger than reality.
    // An overstated width undercounts preview rows during cursor-up redraws, leaving
    // fragments, and makes tables wrap at the terminal edge, misaligning their borders.
    // Use the ioctl result when available; fall back to COLUMNS when no usable size is
    // returned, such as in non-TTY tests or pipelines.
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

/// 终端可见视口的物理行数（**实时**行高）。
///
/// 与 [`raw_terminal_cols`] 同源：`a` 是常驻进程，`LINES` 环境变量只是启动那一刻
/// 的快照，面板被拖矮后往往比真实高度大。折叠预览窗靠相对光标（`\x1b[nA`）就地
/// 重绘，而 `CUU` 在视口顶部会被终端钳制、够不到已滚入 scrollback 的旧行；因此
/// 必须用实时行高把窗口高度钳制在视口内，真实 tty 一律以 ioctl 为准，`LINES`
/// 仅作非 tty（测试 / 管道）回退。
pub(in crate::ai) fn raw_terminal_rows() -> usize {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = std::io::stdout().as_raw_fd();
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) };
        if rc == 0 && ws.ws_row > 0 {
            return ws.ws_row as usize;
        }
    }

    if let Some(rows) = std::env::var("LINES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        && rows > 0
    {
        return rows;
    }

    24
}

pub(super) fn code_block_gutter_width(
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

pub(super) fn code_block_content_width(
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

pub(super) fn wrap_code_block_text(text: &str, content_width: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for ch in text.chars() {
        let ch_width = terminal_cell_width(ch);
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

pub(super) fn code_block_preview_prefix(
    block_indent: &str,
    line_num_str: &str,
    dimmed: bool,
    show_line_gutter: bool,
) -> String {
    let mut out = String::new();
    out.push_str(block_indent);
    out.push_str(theme::current().code_background);
    if show_line_gutter {
        out.push_str(theme::current().code_dim);
        out.push_str(line_num_str);
        out.push_str(" │\x1b[0m");
    } else {
        out.push_str("\x1b[0m");
    }
    out.push_str(theme::current().code_background);
    if dimmed {
        out.push_str("\x1b[2m");
    }
    out
}

pub(super) fn code_block_preview_continuation_prefix(
    block_indent: &str,
    dimmed: bool,
    show_line_gutter: bool,
) -> String {
    code_block_preview_prefix(block_indent, "   ", dimmed, show_line_gutter)
}
