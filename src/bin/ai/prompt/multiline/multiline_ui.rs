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

/// 空输入时的紧凑 viewport 高度：textarea(1) + model(1) + help(1) = 3。
/// 用户开始输入后会 resize 到 `ACTIVE_VIEWPORT_HEIGHT`，确保 textarea 有足够编辑空间。
const COMPACT_VIEWPORT_HEIGHT: u16 = 3;
/// 有内容时的 viewport 高度：top_margin(1) + textarea(4) + model(1) + help(2) = 8。
/// 空→非空首次输入时由事件循环触发 resize 从 COMPACT_VIEWPORT_HEIGHT 切换到此值。
const ACTIVE_VIEWPORT_HEIGHT: u16 = 8;
const MAX_VIEWPORT_HEIGHT: u16 = 20;
const VIEWPORT_CHROME_LINES: u16 = 5; // top margin + model line + spacer + help(2)
const MIN_TEXTAREA_LINES: u16 = 3;
const MAX_PREFILL_TEXTAREA_LINES: u16 = 12;

/// 补全面板激活时，除面板外需要保留的固定行数：
/// top_margin(1) + textarea 最小行(1) + model(1) + help(2) = 5。
/// 与 `render::popup_layout_config` 在 `has_completion_panel` 分支下的口径保持一致
/// （面板激活时 spacer=0、min_textarea_lines=1）。
const PANEL_CHROME_LINES: u16 = 5;
/// 补全面板一次最多显示的候选行数，与 `render::COMPLETION_WINDOW` 对齐。
const PANEL_COMPLETION_WINDOW: u16 = 12;

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
    if prefill.is_none_or(str::is_empty) {
        // 空输入先用紧凑 viewport（COMPACT_VIEWPORT_HEIGHT），用户首次输入时
        // 事件循环会 resize 到 ACTIVE_VIEWPORT_HEIGHT，确保 textarea 有足够编辑空间。
        return COMPACT_VIEWPORT_HEIGHT.min(available_rows);
    }
    let prefill_rows = prefill
        .map(|text| text.lines().count().max(1))
        .unwrap_or(1)
        .min(u16::MAX as usize) as u16;
    let prefill_viewport_height = prefill_rows
        .clamp(MIN_TEXTAREA_LINES, MAX_PREFILL_TEXTAREA_LINES)
        .saturating_add(VIEWPORT_CHROME_LINES);
    let desired = COMPACT_VIEWPORT_HEIGHT.max(prefill_viewport_height);
    desired.min(MAX_VIEWPORT_HEIGHT).min(available_rows)
}

/// 补全面板激活时所需的 viewport 高度：在保持输入框（textarea）行数不变的前提下，
/// 额外为面板腾出空间。面板期望行数 = min(候选数, PANEL_COMPLETION_WINDOW) + 上下边框(2)，
/// 再加上 PANEL_CHROME_LINES（top_margin + textarea 最小行 + model + help）。
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
        .min(MAX_VIEWPORT_HEIGHT)
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
            .unwrap_or(COMPACT_VIEWPORT_HEIGHT);

        let mut terminal = match build_inline_terminal(base_viewport_height) {
            Ok(terminal) => terminal,
            Err(err) => {
                let _ = disable_raw_mode();
                return Err(err);
            }
        };

        let viewport_top_pos = terminal.get_cursor_position().ok();

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
            // 空→非空首次输入时，将 viewport 从紧凑高度 resize 到编辑高度。
            let mut resized_for_content = false;

            loop {
                // 空→非空：首次输入时将 viewport 从 COMPACT 放大到 ACTIVE，
                // 给 textarea 腾出足够编辑空间；后续输入不再重复 resize。
                let has_content = !textarea.lines().join("").is_empty();
                if has_content && !resized_for_content {
                    let target = ACTIVE_VIEWPORT_HEIGHT
                        .min(terminal_size().map(|(_, h)| h.saturating_sub(2)).unwrap_or(ACTIVE_VIEWPORT_HEIGHT).max(1));
                    resize_inline_viewport(&mut terminal, target)?;
                    base_viewport_height = target;
                    resized_for_content = true;
                }

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

                terminal
                    .draw(|f| {
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

        // 退出 TUI：清除 ratatui 的 inline viewport 残留内容。
        // `terminal.clear()` 会把光标挪到 viewport 顶部再 FromCursorDown，
        // 但如果 resize 事件导致内部锚点漂移，clear() 可能未覆盖完整区域，
        // 旧内容残留在 scrollback 中——用户会看到输入被渲染两次。
        // 因此先调 terminal.clear()，再用创建 viewport 时保存的锚点坐标
        // 显式 MoveTo + FromCursorDown 做兜底清除。
        let _ = terminal.clear();
        if let Some(pos) = viewport_top_pos {
            let _ = execute!(
                io::stdout(),
                cursor::MoveTo(pos.x, pos.y),
                Clear(ClearType::FromCursorDown),
            );
        } else {
            let _ = execute!(io::stdout(), Clear(ClearType::FromCursorDown));
        }
        let _ = terminal.show_cursor();
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
            let mut lines = content.lines();
            if let Some(first) = lines.next() {
                println!("\x1b[2m> {first}\x1b[0m");
            }
            for line in lines {
                println!("\x1b[2m  {line}\x1b[0m");
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        clear_inline_viewport_preserving_cursor, multiline_viewport_height,
        viewport_height_with_completion,
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
    fn multiline_viewport_height_stays_compact_for_empty_prompt() {
        // 空输入返回 COMPACT_VIEWPORT_HEIGHT（3），有预填内容时扩大到能容纳编辑器的高度
        assert_eq!(multiline_viewport_height(30, None), 3);
        assert_eq!(multiline_viewport_height(30, Some("")), 3);
        assert_eq!(multiline_viewport_height(30, Some("one line")), 8);
    }

    #[test]
    fn multiline_viewport_height_expands_for_prefill_but_caps_to_available_rows() {
        let prefill = (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(multiline_viewport_height(40, Some(&prefill)), 17);
        assert_eq!(multiline_viewport_height(10, Some(&prefill)), 8);
        assert_eq!(multiline_viewport_height(4, Some(&prefill)), 2);
        assert_eq!(multiline_viewport_height(4, None), 2);
    }

    #[test]
    fn completion_viewport_grows_with_candidates_without_shrinking_base() {
        // 无面板：保持 base 高度（8）。
        assert_eq!(viewport_height_with_completion(30, 8, None), 8);
        // 1 个候选：面板需要 1+2(边框)+5(chrome)=8，不小于 base，仍是 8。
        assert_eq!(viewport_height_with_completion(30, 8, Some(1)), 8);
        // 3 个候选：3+2+5=10 > base，viewport 撑高到 10，多出的 2 行全给面板。
        assert_eq!(viewport_height_with_completion(30, 8, Some(3)), 10);
        // 大量候选：受 PANEL_COMPLETION_WINDOW(12) 与 MAX_VIEWPORT_HEIGHT(20) 双重封顶。
        // 12+2+5=19 <= 20，取 19。
        assert_eq!(viewport_height_with_completion(30, 8, Some(50)), 19);
    }

    #[test]
    fn completion_viewport_capped_by_available_terminal_rows() {
        // 终端只有 12 行时 available=10，即便面板想要更高也不能超过 10。
        assert_eq!(viewport_height_with_completion(12, 8, Some(50)), 10);
        // base 本身也受 available 约束。
        assert_eq!(viewport_height_with_completion(6, 8, None), 4);
    }
}
