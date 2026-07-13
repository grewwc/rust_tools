use std::io;
use std::time::Duration;

use crossterm::{
    cursor,
    event::{self, DisableBracketedPaste, EnableBracketedPaste, Event},
    execute,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size as terminal_size},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use tui_textarea::TextArea;

use super::{
    MultilineHistoryState,
    completion_panel::{CompletionPanel, PendingTabCompletion},
    events::{EventLoopAction, RecentTextInput, handle_multiline_event},
    render::render_multiline_popup,
};
use crate::ai::prompt::{PromptEditor, interrupted_error};

/// viewport 最大高度（textarea + chrome），随终端尺寸动态缩放，上限 10 行。
const MAX_VIEWPORT_HEIGHT: u16 = 10;
/// textarea 最大行数 = MAX_VIEWPORT_HEIGHT - VIEWPORT_CHROME_LINES。
const MAX_TEXTAREA_LINES: u16 = 7;
/// chrome 固定行数：model(1) + help(2)，top_margin=0 且无 spacer。
const VIEWPORT_CHROME_LINES: u16 = 3;
/// textarea 最小行数，用于 clamp 计算。
const MIN_TEXTAREA_LINES: u16 = 2;
/// 空输入时保持更紧凑：只保留 1 行输入区 + 固定 chrome，减少上一轮输出与
/// model/help 区之间的空白。
const EMPTY_VIEWPORT_HEIGHT: u16 = 1 + VIEWPORT_CHROME_LINES;

/// 补全面板一次最多显示的候选行数，与 `render::COMPLETION_WINDOW` 对齐。
const PANEL_COMPLETION_WINDOW: u16 = 12;
/// 补全面板激活时的保底 chrome：textarea 最小行(1) + 压缩后的帮助行(1) = 2。
/// 补全态会隐藏 model/session 行，优先把高度让给候选列表。
const PANEL_CHROME_LINES: u16 = 2;
/// 补全态允许比普通编辑态更高的 inline viewport，这样大终端里可以一次看到更多候选。
const MAX_COMPLETION_VIEWPORT_HEIGHT: u16 = PANEL_CHROME_LINES + PANEL_COMPLETION_WINDOW + 2;

/// `Terminal::clear()` 在 inline viewport 下会先把真实终端 cursor 挪到 viewport 顶部
/// 再执行 `FromCursorDown`。如果 resize 突发期间又立刻来一轮 autoresize，ratatui 会
/// 以这个被挪到顶部的 cursor 重新计算 viewport 锚点，导致输入框/光标跳到 terminal
/// 顶部。这里清屏后把 cursor 放回清屏前的位置，只让它承担“清屏”职责，不污染下一轮
/// viewport 定位基准。
fn clear_inline_viewport_preserving_cursor<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
) {
    let cursor_before_clear = terminal.get_cursor_position().ok();
    let _ = terminal.clear();
    if let Some(cursor_before_clear) = cursor_before_clear {
        let _ = terminal.set_cursor_position(cursor_before_clear);
    }
}

fn multiline_viewport_height(terminal_rows: u16, prefill: Option<&str>) -> u16 {
    let available_rows = terminal_rows.saturating_sub(2).max(1);
    // 空输入默认保持紧凑，避免把 cursor 放在一个很高的空白 textarea 左上角。
    if prefill.is_none_or(str::is_empty) {
        return EMPTY_VIEWPORT_HEIGHT
            .min(available_rows)
            .min(MAX_VIEWPORT_HEIGHT);
    }
    // textarea 基础行数：取 terminal 可用行数的 1/4，至少 MIN、最多 MAX
    let base_textarea = (available_rows / 4).clamp(MIN_TEXTAREA_LINES, MAX_TEXTAREA_LINES);
    // 有预填内容时，textarea 行数不小于 base 且能容纳内容，最多 MAX_TEXTAREA_LINES
    let content_rows = prefill.map(|text| text.lines().count().max(1)).unwrap_or(1) as u16;
    let textarea = content_rows.clamp(base_textarea, MAX_TEXTAREA_LINES);
    let viewport = textarea.saturating_add(VIEWPORT_CHROME_LINES);
    viewport.min(available_rows).min(MAX_VIEWPORT_HEIGHT)
}

/// 补全面板激活时所需的 viewport 高度：在保持输入框（textarea）行数不变的前提下，
/// 额外为面板腾出空间。面板期望行数 = min(候选数, PANEL_COMPLETION_WINDOW) + 上下边框(2)，
/// 再加上 PANEL_CHROME_LINES（textarea 最小行 + 压缩后的帮助行）。
/// 未超过 base_height（无面板时的高度）时直接用 base_height，避免面板很小反而缩了 viewport。
fn viewport_height_with_completion(
    terminal_rows: u16,
    base_height: u16,
    completion_items: Option<usize>,
) -> u16 {
    let available_rows = terminal_rows.saturating_sub(2).max(1);
    let base = base_height.min(available_rows);
    let Some(items) = completion_items else {
        return base;
    };
    let visible = (items.min(PANEL_COMPLETION_WINDOW as usize) as u16).max(1);
    let panel_lines = visible.saturating_add(2); // 上下边框
    let desired = panel_lines.saturating_add(PANEL_CHROME_LINES);
    desired
        .max(base)
        .min(MAX_COMPLETION_VIEWPORT_HEIGHT)
        .min(available_rows)
}

type MultilineTerminal = Terminal<CrosstermBackend<io::Stdout>>;

fn build_inline_terminal(height: u16) -> io::Result<MultilineTerminal> {
    let backend = CrosstermBackend::new(io::stdout());
    Terminal::with_options(
        backend,
        ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Inline(height.max(1)),
        },
    )
    .map_err(|err| io::Error::other(err.to_string()))
}

/// 补全面板开/关时，inline viewport 需要的高度会变化，而 ratatui 的 inline viewport
/// 高度在创建时固定、无法原地修改。这里通过“清屏归位光标 -> 用新高度重建 Terminal”
/// 完成切换：`clear()` 会把光标移回 viewport 顶部并擦除其下内容，重建时 ratatui 以同一
/// 顶部锚点向下 `append_lines` 展开，因此放大/收回都不会污染 scrollback。
/// 输入框（textarea）的行数不受影响——多出/收回的高度只作用于补全面板区域。
fn resize_inline_viewport(terminal: &mut MultilineTerminal, new_height: u16) -> io::Result<()> {
    let _ = terminal.hide_cursor();
    let _ = terminal.clear();
    *terminal = build_inline_terminal(new_height)?;
    Ok(())
}

/// 处理 `Event::Resize` 连续触发的情况：终端窗口拖动时 Resize 事件会快速连发，
/// 如果每次都重绘 inline viewport，会导致"画面撕裂"或闪烁。
/// 本函数先 drain 掉后续堆积的 Resize 事件，再做一次 full redraw。
fn handle_resize_burst<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>) -> io::Result<()> {
    // 1) drain 掉所有后续的 Resize 事件，避免重复触发重绘
    while event::poll(Duration::ZERO).unwrap_or(false) {
        match event::read() {
            Ok(Event::Resize(_, _)) => continue,
            // 其他 case：resize 突发期间夹杂的非 Resize 事件直接丢弃，
            // 回到 UI 主循环重新处理
            Ok(_) | Err(_) => break,
        }
    }

    // 2) 通知 ratatui 终端尺寸已变化（宽度变化影响文本折行），
    //    然后清除 inline viewport。不再手动 MoveUp + Clear：
    //    viewport_height 是启动时的快照，resize 后手动 MoveUp 的行数
    //    与 ratatui 内部光标状态不一致，导致 clear() 清错区域，旧
    //    viewport 内容（包括 cursor 块 ▌）残留在 scrollback 中反复堆叠。
    let _ = terminal.hide_cursor();
    let _ = terminal.autoresize();
    clear_inline_viewport_preserving_cursor(terminal);
    Ok(())
}

fn submitted_input_preview_lines(content: &str) -> Vec<String> {
    let mut rendered = Vec::new();
    let mut lines = content.lines();
    if let Some(first) = lines.next() {
        rendered.push(format!("\x1b[2m> {first}\x1b[0m"));
        for line in lines {
            rendered.push(format!("\x1b[2m  {line}\x1b[0m"));
        }
    }
    rendered
}

fn print_submitted_input_preview(content: &str) {
    for line in submitted_input_preview_lines(content) {
        println!("{line}");
    }
}

impl PromptEditor {
    pub(in crate::ai::prompt) fn read_multi_line_tui(&mut self) -> io::Result<Option<String>> {
        enable_raw_mode()?;

        // SSH 下禁用 bracketed paste：终端截获 Ctrl+V 后，如果剪贴板内容是图片（二进制），
        // 无法通过 bracketed paste 传输，导致 paste event 为空或不触发。
        // 禁用后 Ctrl+V 直接产生 Event::Key(Ctrl+V)，由 handler 通过 OSC52 通路读取剪贴板。
        let is_ssh = std::env::var("SSH_CONNECTION").is_ok()
            || std::env::var("SSH_CLIENT").is_ok()
            || std::env::var("SSH_TTY").is_ok();
        if is_ssh {
            let _ = execute!(io::stdout(), DisableBracketedPaste);
        } else {
            let _ = execute!(io::stdout(), EnableBracketedPaste);
        }

        // Inline viewport 初始化会通过 append_lines() 真实撑开终端区域。空输入默认保持
        // 紧凑，避免每轮回答后和下一轮光标之间出现大段空白；编辑已有内容时再按预填行数
        // 放大，给 textarea 保留足够空间。
        let mut base_viewport_height = terminal_size()
            .map(|(_, h)| multiline_viewport_height(h, self.pending_prefill.as_deref()))
            .unwrap_or(6);

        let mut terminal = match build_inline_terminal(base_viewport_height) {
            Ok(terminal) => terminal,
            Err(err) => {
                let _ = disable_raw_mode();
                return Err(err);
            }
        };

        // 退出时按“最后一次实际渲染到哪里”来清理 viewport；不能依赖创建 Terminal
        // 那一刻的 cursor 位置，因为补全面板/textarea 扩容会重建 inline viewport。
        let mut last_viewport_top_row: Option<u16> = None;

        let result: io::Result<Option<String>> = (|| {
            // 预填内容（编辑已有 memo 场景）：按行载入 textarea，读取后清空。
            let mut textarea: TextArea = match self.pending_prefill.take() {
                Some(text) => TextArea::from(text.lines().map(|l| l.to_string())),
                None => TextArea::default(),
            };
            let mut history = MultilineHistoryState::new(self.multiline_history_entries());
            let mut status_msg: Option<String> = None;
            let mut pending_tab_completion: Option<PendingTabCompletion> = None;
            let mut completion_panel: Option<CompletionPanel> = None;
            let mut recent_text_input: Option<RecentTextInput> = None;
            // 记录当前 viewport 已适配的补全候选数：None 表示无面板（base 高度）。
            // 面板出现/消失/候选数变化时，据此重建 viewport 让面板获得足够高度，
            // 而输入框行数保持不变。
            let mut fitted_completion_items: Option<usize> = None;

            loop {
                // 面板状态变化时，重建 inline viewport 以匹配面板所需高度。
                let current_items = completion_panel.as_ref().map(|p| p.items.len());
                if current_items != fitted_completion_items {
                    let terminal_rows = terminal_size().map(|(_, h)| h).unwrap_or(0);
                    let new_height = viewport_height_with_completion(
                        terminal_rows,
                        base_viewport_height,
                        current_items,
                    );
                    resize_inline_viewport(&mut terminal, new_height)?;
                    fitted_completion_items = current_items;
                }

                // 内容超出 textarea 容量时自动扩展 viewport（只扩不缩，避免频繁闪烁）。
                let content_lines = textarea.lines().len() as u16;
                let textarea_capacity = base_viewport_height.saturating_sub(VIEWPORT_CHROME_LINES);
                if content_lines > textarea_capacity && base_viewport_height < MAX_VIEWPORT_HEIGHT {
                    let terminal_rows = terminal_size().map(|(_, h)| h).unwrap_or(0);
                    let available = terminal_rows.saturating_sub(2).max(1);
                    let new_height = content_lines
                        .saturating_add(VIEWPORT_CHROME_LINES)
                        .min(MAX_VIEWPORT_HEIGHT)
                        .min(available);
                    if new_height > base_viewport_height {
                        resize_inline_viewport(&mut terminal, new_height)?;
                        base_viewport_height = new_height;
                    }
                }

                terminal
                    .draw(|f| {
                        last_viewport_top_row = Some(f.area().y);
                        render_multiline_popup(
                            f,
                            &mut textarea,
                            status_msg.as_deref(),
                            completion_panel.as_ref(),
                            &self.current_model_label,
                            self.session_topic.as_deref(),
                        );
                    })
                    .map_err(|e| io::Error::other(e.to_string()))?;

                let event = event::read().map_err(|e| io::Error::other(e.to_string()))?;
                if let Event::Resize(_, _) = &event {
                    handle_resize_burst(&mut terminal)?;
                }

                match handle_multiline_event(
                    event,
                    &mut textarea,
                    &mut history,
                    &mut status_msg,
                    &mut pending_tab_completion,
                    &mut completion_panel,
                    &mut recent_text_input,
                    &self.session_image_dir,
                )? {
                    EventLoopAction::Continue => {}
                    EventLoopAction::Submit(result) => break Ok(result),
                }
            }
        })();

        // 退出 TUI：先让 ratatui 按“当前实际 viewport 状态”清一次，
        // 再用最后一次渲染时记录的顶行做兜底清除，确保补全面板/扩容 textarea
        // 都不会留下残影。
        let _ = terminal.hide_cursor();
        let _ = terminal.clear();
        drop(terminal);
        if let Some(top_row) = last_viewport_top_row {
            let _ = execute!(
                io::stdout(),
                cursor::MoveTo(0, top_row),
                Clear(ClearType::FromCursorDown),
            );
        } else {
            let _ = execute!(io::stdout(), Clear(ClearType::FromCursorDown));
        }
        let _ = execute!(io::stdout(), cursor::Show);
        let _ = execute!(io::stdout(), DisableBracketedPaste);
        let _ = disable_raw_mode();

        let result = match result {
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                return interrupted_error();
            }
            Err(err) => return Err(err),
            Ok(result) => result,
        };
        if let Some(content) = &result {
            self.save_history_entry(content);
            print_submitted_input_preview(content);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        clear_inline_viewport_preserving_cursor, multiline_viewport_height,
        submitted_input_preview_lines, viewport_height_with_completion,
    };
    use ratatui::{
        Terminal, TerminalOptions, Viewport,
        backend::TestBackend,
        layout::{Position, Rect},
        widgets::{Paragraph, Widget},
    };

    #[test]
    fn clear_inline_viewport_preserves_existing_cursor_anchor() {
        let backend = TestBackend::new(24, 8);
        let mut terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(3),
            },
        )
        .unwrap();

        terminal
            .insert_before(2, |buf| {
                Paragraph::new(vec!["line 1".into(), "line 2".into()]).render(buf.area, buf);
            })
            .unwrap();

        let mut viewport_area = Rect::ZERO;
        terminal
            .draw(|f| {
                viewport_area = f.area();
                f.render_widget(Paragraph::new("prompt"), f.area());
                f.set_cursor_position((viewport_area.x + 4, viewport_area.y + 1));
            })
            .unwrap();

        let expected = Position::new(viewport_area.x + 4, viewport_area.y + 1);
        terminal.backend_mut().assert_cursor_position(expected);

        clear_inline_viewport_preserving_cursor(&mut terminal);

        terminal.backend_mut().assert_cursor_position(expected);
    }

    #[test]
    fn multiline_viewport_height_scales_with_terminal() {
        // 空输入：更紧凑，viewport = 1 行输入区 + chrome(3) = 4
        assert_eq!(multiline_viewport_height(30, None), 4);
        assert_eq!(multiline_viewport_height(30, Some("")), 4);
        // 有预填但内容短于 base：保持 base 大小
        assert_eq!(multiline_viewport_height(30, Some("one line")), 10);
        // 小终端：terminal=12, available=10，空输入仍保持 4 行紧凑 viewport
        assert_eq!(multiline_viewport_height(12, None), 4);
        // 大终端下空输入仍保持紧凑
        assert_eq!(multiline_viewport_height(40, None), 4);
    }

    #[test]
    fn multiline_viewport_height_expands_for_prefill_but_caps_to_available_rows() {
        let prefill = (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n");

        // terminal=40: available=38, base_textarea=8, content=20→clamp(8,8)=8, viewport=10
        assert_eq!(multiline_viewport_height(40, Some(&prefill)), 10);
        assert_eq!(multiline_viewport_height(10, Some(&prefill)), 8);
        assert_eq!(multiline_viewport_height(4, Some(&prefill)), 2);
        assert_eq!(multiline_viewport_height(4, None), 2); // available=2，仍受可用行数约束
    }

    #[test]
    fn completion_viewport_grows_with_candidates_without_shrinking_base() {
        // 无面板：保持 base 高度（4）。
        assert_eq!(viewport_height_with_completion(30, 4, None), 4);
        // 1 个候选：面板需要 1+2(边框)=3 + 2(chrome)=5，大于 base，撑到 5。
        assert_eq!(viewport_height_with_completion(30, 4, Some(1)), 5);
        // 3 个候选：3+2+2=7 > base，viewport 撑高到 7，多出的 3 行给面板。
        assert_eq!(viewport_height_with_completion(30, 4, Some(3)), 7);
        // 大量候选：补全态上限单独放宽到 16，可容纳 12 行候选 + 边框 + 压缩 chrome。
        assert_eq!(viewport_height_with_completion(30, 4, Some(50)), 16);
    }

    #[test]
    fn completion_viewport_capped_by_available_terminal_rows() {
        // 终端只有 12 行时 available=10，即便面板想要更高也不能超过 10。
        assert_eq!(viewport_height_with_completion(12, 4, Some(50)), 10);
        // base 本身也受 available 约束。
        assert_eq!(viewport_height_with_completion(6, 8, None), 4);
    }

    #[test]
    fn submitted_input_preview_formats_single_and_multi_line_content() {
        assert_eq!(
            submitted_input_preview_lines("hello"),
            vec!["\u{1b}[2m> hello\u{1b}[0m".to_string()]
        );
        assert_eq!(
            submitted_input_preview_lines("hello\nworld"),
            vec![
                "\u{1b}[2m> hello\u{1b}[0m".to_string(),
                "\u{1b}[2m  world\u{1b}[0m".to_string()
            ]
        );
    }
}
