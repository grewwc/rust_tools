use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    cursor,
    event::{self, DisableBracketedPaste, EnableBracketedPaste, Event},
    execute,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size as terminal_size},
};
use ratatui::{
    Terminal,
    backend::{Backend, ClearType as BackendClearType, CrosstermBackend},
    layout::Position,
};
use tui_textarea::TextArea;

use super::{
    MultilineHistoryState,
    completion_panel::{CompletionPanel, PendingTabCompletion},
    events::{EventLoopAction, RecentTextInput, handle_multiline_event},
    render::render_multiline_popup,
};
use crate::ai::prompt::{PromptEditor, interrupted_error};

/// viewport 最大高度（textarea + chrome），随终端尺寸动态缩放，上限 11 行。
const MAX_VIEWPORT_HEIGHT: u16 = 11;
/// textarea 最大行数上限（大终端下的舒适值）。
const MAX_TEXTAREA_LINES: u16 = 7;
/// 普通编辑态的 chrome 固定行数：model(1) + help(2)。
/// 不再绘制装饰性 divider，避免 terminal resize 后横线残影堆积。
const VIEWPORT_CHROME_LINES: u16 = 3;
/// textarea 最小行数，用于 clamp 计算。
const MIN_TEXTAREA_LINES: u16 = 2;
/// 空输入时保持更紧凑：只保留 1 行输入区 + 固定 chrome，减少上一轮输出与
/// model/help 区之间的空白。
const EMPTY_VIEWPORT_HEIGHT: u16 = 1 + VIEWPORT_CHROME_LINES;

/// 补全面板一次最多显示的候选行数，与 `render::COMPLETION_WINDOW` 对齐。
const PANEL_COMPLETION_WINDOW: u16 = 12;
/// 补全面板激活时的保底 chrome：textarea 最小行(1) + 压缩后的帮助行(1) = 2。
/// 补全态会隐藏 model/session 信息，优先把高度让给候选列表。
const PANEL_CHROME_LINES: u16 = 2;
/// 补全态允许比普通编辑态更高的 inline viewport，这样大终端里可以一次看到更多候选。
const MAX_COMPLETION_VIEWPORT_HEIGHT: u16 = PANEL_CHROME_LINES + PANEL_COMPLETION_WINDOW + 2;

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

/// 横向 resize 后终端会先对现有内容做 reflow，此时 ratatui 保存的 viewport 顶行
/// 还是 resize 前的坐标。根据真实 cursor 和上一帧的相对行偏移重新找到顶行，先擦除
/// 旧 viewport，避免它在下一次 `autoresize` 重锚时被推进 scrollback。
fn clear_reflowed_inline_viewport<B: Backend>(
    terminal: &mut Terminal<B>,
    cursor_offset_row: u16,
) -> Result<(), B::Error> {
    let cursor_position = terminal.backend_mut().get_cursor_position()?;
    let viewport_top = cursor_position.y.saturating_sub(cursor_offset_row);
    terminal
        .backend_mut()
        .set_cursor_position(Position::new(0, viewport_top))?;
    terminal
        .backend_mut()
        .clear_region(BackendClearType::CurrentLine)?;
    terminal
        .backend_mut()
        .clear_region(BackendClearType::AfterCursor)?;
    terminal
        .backend_mut()
        .set_cursor_position(cursor_position)?;
    terminal.backend_mut().flush()
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
        // resize reflow 后 viewport 的绝对坐标会变化，但 cursor 在 viewport 内的相对行
        // 不变；记录该偏移用于在下一次 autoresize 前清掉旧帧。
        let mut last_cursor_offset_row: Option<u16> = None;

        let result: io::Result<Option<String>> = (|| {
            // 预填内容（编辑已有 memo 场景）：按行载入 textarea，读取后清空。
            let mut textarea: TextArea = match self.pending_prefill.take() {
                Some(text) => TextArea::from(text.lines().map(|l| l.to_string())),
                None => TextArea::default(),
            };
            let mut history = MultilineHistoryState::new(self.multiline_history_entries());
            let mut status_msg: Option<String> = self.pending_status_msg.take();
            let mut pending_tab_completion: Option<PendingTabCompletion> = None;
            let mut completion_panel: Option<CompletionPanel> = None;
            let mut recent_text_input: Option<RecentTextInput> = None;
            let mut generated_title_seen = false;
            let mut next_title_refresh = Instant::now();
            // 记录当前 viewport 已适配的补全候选数：None 表示无面板（base 高度）。
            // 面板出现/消失/候选数变化时，据此重建 viewport 让面板获得足够高度，
            // 而输入框行数保持不变。
            let mut fitted_completion_items: Option<usize> = None;

            loop {
                // 首轮 user message 落盘后，session 标题会在后台生成。输入框已经打开时
                // 不能只依赖下一轮 prompt 初始化，否则底部会一直停留在首条消息摘要。
                if !generated_title_seen {
                    let now = Instant::now();
                    if now >= next_title_refresh {
                        if let Ok(Some(_)) = self.refresh_generated_session_topic() {
                            generated_title_seen = true;
                        }
                        next_title_refresh = now + Duration::from_millis(500);
                    }
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
                        let area = f.area();
                        last_viewport_top_row = Some(area.y);
                        last_cursor_offset_row = render_multiline_popup(
                            f,
                            &mut textarea,
                            status_msg.as_deref(),
                            completion_panel.as_ref(),
                            &self.current_model_label,
                            self.session_topic.as_deref(),
                        );
                    })
                    .map_err(|e| io::Error::other(e.to_string()))?;

                if !event::poll(Duration::from_millis(250))
                    .map_err(|e| io::Error::other(e.to_string()))?
                {
                    continue;
                }
                let event = event::read().map_err(|e| io::Error::other(e.to_string()))?;
                if matches!(event, Event::Resize(_, _)) {
                    // VS Code 已经完成横向 reflow；先按真实 cursor 重算并清理旧 viewport，
                    // 再让下一轮 `Terminal::draw` 调用 `autoresize` 完成重锚和重绘。
                    if let Some(cursor_offset_row) = last_cursor_offset_row {
                        clear_reflowed_inline_viewport(&mut terminal, cursor_offset_row)
                            .map_err(|e| io::Error::other(e.to_string()))?;
                    }
                    continue;
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
    use ratatui::{
        Terminal,
        backend::{Backend, TestBackend},
        layout::Position,
        widgets::Paragraph,
    };

    use super::{
        clear_reflowed_inline_viewport, multiline_viewport_height, submitted_input_preview_lines,
        viewport_height_with_completion,
    };

    #[test]
    fn clear_reflowed_viewport_uses_cursor_relative_top_and_restores_cursor() {
        let backend = TestBackend::new(8, 5);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(Paragraph::new("row0\nrow1\nrow2\nrow3\nrow4"), frame.area());
            })
            .unwrap();
        let cursor = Position::new(3, 4);
        terminal.backend_mut().set_cursor_position(cursor).unwrap();

        clear_reflowed_inline_viewport(&mut terminal, 2).unwrap();

        assert_eq!(terminal.backend_mut().get_cursor_position().unwrap(), cursor);
        assert_eq!(terminal.backend().buffer()[(0, 1)].symbol(), "r");
        assert_eq!(terminal.backend().buffer()[(0, 2)].symbol(), " ");
        assert_eq!(terminal.backend().buffer()[(0, 4)].symbol(), " ");
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

        // terminal=40: available=38, base_textarea=7, content=20→clamp(7,7)=7, viewport=10
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
