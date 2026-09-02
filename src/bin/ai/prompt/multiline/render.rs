use ratatui::{
    layout::Alignment,
    layout::Rect,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};
use tui_textarea::TextArea;
use unicode_width::UnicodeWidthChar;
use unicode_width::UnicodeWidthStr;

use super::completion_panel::CompletionPanel;
use crate::ai::prompt::MAX_INPUT_CHARS;

/// Maximum number of candidate lines the completion panel shows at once (overflow scrolls with the selection).
const COMPLETION_WINDOW: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PopupLayoutConfig {
    top_margin: u16,
    top_rule_lines: u16,
    help_lines: u16,
    model_header_lines: u16,
    min_textarea_lines: u16,
}

fn popup_layout_config(
    _area_height: u16,
    _current_content: &str,
    content_lines: usize,
    _trailing_blank_lines: usize,
    has_completion_panel: bool,
    _has_status_msg: bool,
    has_model_label: bool,
) -> PopupLayoutConfig {
    // top_margin always stays 0. Do not put stable decorative horizontal rules inside the prompt viewport:
    // on terminal resize / reflow / re-anchoring, such full-line decorations easily get pushed into scrollback,
    // and after the width is restored they keep stacking like "a few extra horizontal lines". The divider is
    // removed entirely here, keeping only the necessary model/help info to avoid ghost artifacts.
    let top_margin: u16 = 0;
    let top_rule_lines: u16 = 0;
    // While the completion panel is active, compress the bottom help to 1 line and hide model/session info,
    // giving the height to the candidate list first; on small terminals this significantly reduces the "only 1 candidate visible" case.
    let help_lines: u16 = 1;
    let model_header_lines = if has_completion_panel || !has_model_label {
        0
    } else {
        1
    };
    let min_textarea_lines = if has_completion_panel {
        1
    } else {
        (content_lines.max(1)).min(6) as u16
    };

    PopupLayoutConfig {
        top_margin,
        top_rule_lines,
        help_lines,
        model_header_lines,
        min_textarea_lines,
    }
}

pub(in crate::ai::prompt::multiline) fn render_multiline_popup(
    f: &mut ratatui::Frame<'_>,
    textarea: &mut TextArea<'_>,
    status_msg: Option<&str>,
    completion_panel: Option<&CompletionPanel>,
    model_label: &str,
    reasoning_effort_label: &str,
    session_topic: Option<&str>,
) -> Option<u16> {
    let area = f.area();
    let current_lines = textarea.lines().to_vec();
    let current_content = current_lines.join("\n");
    let trailing_blank_lines = count_trailing_blank_lines(&current_lines);
    let layout = popup_layout_config(
        area.height,
        &current_content,
        current_lines.len(),
        trailing_blank_lines,
        completion_panel.is_some(),
        status_msg.is_some(),
        !model_label.is_empty(),
    );

    // Compute the popup size: always fill the current prompt viewport. Whitespace for the empty-input case is
    // achieved via a smaller viewport height and removing the top gap, not by carving unused regions inside the viewport.
    let popup_height = area.height;
    let popup_width = area.width.saturating_sub(2).clamp(40, 180).min(area.width);

    // Compute the popup position (top-aligned, right below the previous output)
    let popup_x = area.x + area.width.saturating_sub(popup_width) / 2;
    let popup_y = area.y;
    let popup = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Compute the inner area: 1 column of horizontal margin on each side, no extra top/bottom padding,
    // to avoid creating extra blank lines inside the prompt viewport.
    let h_margin: u16 = 1;
    let top_margin = layout.top_margin;
    let inner = Rect::new(
        popup.x + h_margin,
        popup.y + top_margin,
        popup.width - h_margin * 2,
        popup.height - top_margin,
    );

    // Compute each region's height
    let top_rule_lines = layout.top_rule_lines;
    let help_lines = layout.help_lines;
    // Model/topic info line: keep it at the bottom (above the help line), where it
    // remains visually stable while the textarea grows and the viewport is
    // re-anchored after terminal reflow.
    let model_header_lines = layout.model_header_lines;
    // While the panel is active, prioritize filling the height: subtract the help line and the textarea's minimum
    // rows first, then give the rest to the panel (panel desired height = min(candidate count, COMPLETION_WINDOW) +
    // 2 for top/bottom borders, capped by available space). The textarea yields to its minimum 1 row (the user is
    // picking from a list and does not need a large editor). Without a panel, adapt to content and viewport height.
    let min_textarea_lines = layout.min_textarea_lines;
    let (textarea_lines, panel_lines) = match completion_panel {
        Some(panel) => {
            let desired_panel = (panel.items.len().min(COMPLETION_WINDOW) as u16).saturating_add(2);
            // Panel usable cap = total height - help - textarea minimum rows.
            let panel_cap = inner
                .height
                .saturating_sub(top_rule_lines)
                .saturating_sub(help_lines)
                .saturating_sub(model_header_lines)
                .saturating_sub(min_textarea_lines);
            let panel = desired_panel.min(panel_cap).max(1.min(panel_cap));
            let textarea = inner
                .height
                .saturating_sub(top_rule_lines)
                .saturating_sub(panel)
                .saturating_sub(model_header_lines)
                .saturating_sub(help_lines)
                .max(min_textarea_lines);
            (textarea, panel)
        }
        None => {
            let textarea = inner
                .height
                .saturating_sub(top_rule_lines)
                .saturating_sub(model_header_lines)
                .saturating_sub(help_lines)
                .max(min_textarea_lines);
            (textarea, 0)
        }
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(top_rule_lines),
            Constraint::Length(textarea_lines),
            Constraint::Length(panel_lines),
            Constraint::Length(model_header_lines),
            Constraint::Length(help_lines),
        ])
        .split(inner);

    // textarea render area
    let textarea_area = chunks[1];

    // Clear the popup area so old borders/text do not linger after a resize
    f.render_widget(Clear, popup);

    // Bottom model/topic info line: lets the user see the current model and session topic while typing.
    // Draw it in the dedicated bottom chunk (chunks[3], above the help line) so
    // textarea growth does not move it through the editing region.
    if model_header_lines > 0 {
        let header_area = chunks[3];
        let mut spans = vec![
            Span::styled(" model: ", Style::default().fg(Color::Rgb(148, 163, 184))),
            Span::styled(
                model_label,
                Style::default()
                    .fg(Color::Rgb(134, 194, 166))
                    .add_modifier(Modifier::BOLD),
            ),
        ];
        if !reasoning_effort_label.is_empty() {
            spans.push(Span::styled(
                "  |  reasoning: ",
                Style::default().fg(Color::Rgb(100, 116, 139)),
            ));
            spans.push(Span::styled(
                reasoning_effort_label,
                Style::default()
                    .fg(Color::Rgb(96, 165, 250))
                    .add_modifier(Modifier::BOLD),
            ));
        }
        // Show the session topic on the same line as the model
        let topic_text = match session_topic {
            Some(t) if !t.is_empty() => t,
            _ => "new session",
        };
        spans.push(Span::styled(
            "  |  ",
            Style::default().fg(Color::Rgb(100, 116, 139)),
        ));
        spans.push(Span::styled(
            topic_text,
            Style::default()
                .fg(Color::Rgb(251, 191, 36))
                .add_modifier(Modifier::ITALIC),
        ));
        let header = Line::from(spans);
        f.render_widget(Paragraph::new(header), header_area);
    }

    let char_count = current_content.chars().count();

    // Set alignment
    textarea.set_alignment(Alignment::Left);
    // User input uses a low-saturation warm gray; it does not compete for attention with blue/cyan status info
    // or yellow warnings, and reads better over long sessions than high-saturation colors on dark terminals.
    let (red, green, blue) = crate::ai::theme::ACCENT_INPUT_RGB;
    textarea.set_style(Style::default().fg(Color::Rgb(red, green, blue)));
    // tui-textarea draws the cursor into the buffer as a REVERSED space by default;
    // terminal reflow can preserve that drawn cell as a white cursor ghost. Disable
    // the buffer cursor here and use ratatui's real terminal cursor (see
    // set_cursor_position below), which is not persistable content.
    textarea.set_cursor_style(Style::default());

    f.render_widget(&*textarea, textarea_area);
    // Use the real terminal cursor position computed by tui-textarea's own rendering.
    // Previously this recomputed screen coordinates itself using `UnicodeWidthChar::width_cjk`, while
    // tui-textarea internally (screen_map.rs) uses the non-CJK `c.width()`. The two disagree on the width of
    // CJK/ambiguous-width characters, which under CJK input puts the real terminal cursor 1+ columns off from
    // the editor's buffer cursor (more visible for earlier CJK characters).
    // Reuse the position the library computed during its render plan so the two always agree.
    //
    // Empty-input frames park the real terminal cursor on the viewport's
    // top-left cell and paint the editing caret as a styled cell instead.
    // Reflowing terminals (VS Code's xterm.js, Terminal.app, tmux) keep the
    // cursor attached to its cell while re-wrapping rows, so after a resize a
    // cursor-position query reports the viewport's exact top row (offset 0).
    // Re-anchoring from that report can never drift, so a resize rebuild
    // clears exactly the stale panel rows below the anchor and leaves no
    // blank band above the next prompt. A remembered non-zero offset cannot
    // survive reflow — the panel's own painted rows are re-wrapped too —
    // which is what previously produced those bands during width resizes.
    let cursor_offset_row = match textarea.rendered_cursor_position() {
        // The styled cell mirrors tui-textarea's default reversed-space caret
        // (disabled above to avoid reflow ghosts inside content), so the idle
        // input line looks unchanged; the parked terminal cursor itself is
        // only visible at the left edge of the top rule row.
        Some(pos) if current_content.is_empty() => {
            if let Some(cell) = f.buffer_mut().cell_mut((pos.x, pos.y)) {
                cell.set_style(Style::default().add_modifier(Modifier::REVERSED));
            }
            f.set_cursor_position((area.x, area.y));
            Some(0)
        }
        Some(pos) => {
            f.set_cursor_position((pos.x, pos.y));
            Some(pos.y.saturating_sub(area.y))
        }
        None => None,
    };

    // Render the completion panel
    if let Some(panel) = completion_panel {
        // The scroll window must use the panel's **actually visible row count** (chunk height minus the top/bottom
        // borders), not a fixed COMPLETION_WINDOW: on short terminals the layout squeezes the panel to fewer rows
        // than COMPLETION_WINDOW, and if `start` is still computed from the fixed value, once the selection passes
        // the visible area it lands off-screen — the panel appears "stuck on the first items, unable to scroll".
        let visible_rows = (chunks[2].height as usize).saturating_sub(2).max(1);
        let window_size = visible_rows.min(panel.items.len()).max(1);
        let start = panel
            .selected_index
            .saturating_sub(window_size.saturating_sub(1))
            .min(panel.items.len().saturating_sub(window_size));

        let items: Vec<Line> = panel
            .items
            .iter()
            .enumerate()
            .skip(start)
            .take(window_size)
            .map(|(idx, item)| {
                let selected = idx == panel.selected_index;
                completion_item_line(&item.display, selected)
            })
            .collect();

        let panel_block = Block::default()
            .borders(Borders::ALL)
            .border_style(
                Style::default()
                    .fg(Color::Rgb(74, 92, 112))
                    .add_modifier(Modifier::BOLD),
            )
            .title(Span::styled(
                format!(" Completions {} ", panel.items.len()),
                Style::default()
                    .fg(Color::Rgb(140, 190, 220))
                    .add_modifier(Modifier::BOLD),
            ));

        f.render_widget(Paragraph::new(items).block(panel_block), chunks[2]);
    }

    // Get cursor position
    let (cursor_row, cursor_col) = textarea.cursor();

    // Status bar info: character count + cursor position
    let status_info = if char_count > MAX_INPUT_CHARS {
        format!(
            " Chars: {} (exceeded) | Ln {}, Col {} ",
            char_count,
            cursor_row + 1,
            cursor_col + 1
        )
    } else if char_count > MAX_INPUT_CHARS * 90 / 100 {
        format!(
            " Chars: {} (⚠) | Ln {}, Col {} ",
            char_count,
            cursor_row + 1,
            cursor_col + 1
        )
    } else {
        format!(
            " Ln {}, Col {} | Chars: {} ",
            cursor_row + 1,
            cursor_col + 1,
            char_count
        )
    };

    // Render the help line
    let help_lines = if completion_panel.is_some() {
        vec![Line::from(vec![
            Span::styled("移动：", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "↑↓",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("选择：", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "Enter",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("关闭：", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "Esc",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("取消：", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "Ctrl+C",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(status_info, Style::default().fg(Color::DarkGray)),
            Span::styled("刷新：", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "Tab",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("发送：", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "+Alt/F2",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
        ])]
    } else {
        vec![Line::from({
            let spans: Vec<Span> = vec![
                Span::styled("换行:", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "↵",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("发送:", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Alt+↵/F2",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("取消:", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Ctrl+C",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                Span::styled(status_info, Style::default().fg(Color::DarkGray)),
                Span::raw("  "),
                Span::styled("历史:", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "↑↓/Ctrl+P/N",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("删行:", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "⌘/Ctrl+U",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("粘贴:", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Ctrl+V",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("清空:", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "F8",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("复制回答:", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "F9",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("复制全部:", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "F10",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
            ];
            spans
        })]
    };
    f.render_widget(Paragraph::new(help_lines), chunks[4]);

    if let Some(msg) = status_msg {
        let c2 = chunks[4];
        if c2.height >= 1 && c2.width > 2 {
            let status_width = (c2.width - 2) as usize;
            let status_text = truncate_with_ellipsis(msg, status_width);
            let status_para = Paragraph::new(Line::from(Span::styled(
                status_text,
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )))
            .alignment(Alignment::Center);

            let status_area = Rect::new(c2.x + 1, c2.y, c2.width - 2, 1);
            f.render_widget(Clear, status_area);
            f.render_widget(status_para, status_area);
        }
    }

    cursor_offset_row
}

fn completion_item_line(display: &str, selected: bool) -> Line<'_> {
    let selected_bg = Color::Rgb(31, 45, 61);
    let marker_style = if selected {
        Style::default()
            .fg(Color::Rgb(119, 221, 255))
            .bg(selected_bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let selector_style = if selected {
        Style::default()
            .fg(Color::Rgb(235, 246, 255))
            .bg(selected_bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Rgb(218, 226, 235))
    };
    let meta_style = if selected {
        Style::default()
            .fg(Color::Rgb(170, 185, 198))
            .bg(selected_bg)
    } else {
        Style::default().fg(Color::Rgb(125, 137, 148))
    };
    let current_style = if selected {
        Style::default()
            .fg(Color::Rgb(138, 226, 168))
            .bg(selected_bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Rgb(115, 194, 145))
    };

    let mut spans = vec![Span::styled(
        if selected { "› " } else { "  " },
        marker_style,
    )];
    for (idx, part) in display.split(" · ").enumerate() {
        if idx > 0 {
            spans.push(Span::styled(" · ", meta_style));
        }
        let style = if part == "current" {
            current_style
        } else if idx == 0 {
            selector_style
        } else {
            meta_style
        };
        spans.push(Span::styled(part, style));
    }
    Line::from(spans)
}

fn count_trailing_blank_lines(lines: &[String]) -> usize {
    lines
        .iter()
        .rev()
        .take_while(|line| line.trim().is_empty())
        .count()
}

/// Truncate text to fit a display width.
/// Width is computed with the unicode-width crate; unrecognized characters are conservatively estimated at width 1.
fn truncate_with_ellipsis(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }

    let total = UnicodeWidthStr::width_cjk(text);
    if total <= max_width {
        return text.to_string();
    }

    let ellipsis_w = UnicodeWidthStr::width_cjk("...");
    if max_width <= ellipsis_w {
        return " ".repeat(max_width);
    }

    let target = max_width - ellipsis_w;
    let mut out = String::new();
    let mut width: usize = 0;

    for ch in text.chars() {
        // For characters where unicode-width returns 0, conservatively estimate width 1
        let ch_w = UnicodeWidthChar::width_cjk(ch).unwrap_or(1);

        if width + ch_w > target {
            break;
        }
        out.push(ch);
        width += ch_w;
    }

    out.push_str("...");
    out
}

#[cfg(test)]
mod tests {
    use super::{
        count_trailing_blank_lines, popup_layout_config, render_multiline_popup,
        truncate_with_ellipsis,
    };
    use ratatui::{
        Terminal, TerminalOptions, Viewport,
        backend::TestBackend,
        layout::{Position, Rect},
    };
    use tui_textarea::{CursorMove, TextArea};
    use unicode_width::UnicodeWidthStr;

    fn display_width(s: &str) -> usize {
        UnicodeWidthStr::width_cjk(s)
    }

    fn buffer_row(backend: &TestBackend, y: u16, x_start: u16, width: u16) -> String {
        (x_start..x_start.saturating_add(width))
            .map(|x| {
                backend
                    .buffer()
                    .cell((x, y))
                    .map(|cell| cell.symbol())
                    .unwrap_or(" ")
            })
            .collect()
    }

    #[test]
    fn test_truncate_cjk() {
        // Test CJK truncation
        let result = truncate_with_ellipsis("已补全为 /agent", 10);
        assert!(result.ends_with("..."));
        assert!(display_width(&result) <= 10);
    }

    #[test]
    fn test_truncate_ascii() {
        assert_eq!(truncate_with_ellipsis("Copied!", 20), "Copied!");
        assert_eq!(truncate_with_ellipsis("Hello World!", 8), "Hello...");
    }

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate_with_ellipsis("test", 0), "");
        assert_eq!(truncate_with_ellipsis("test", 2), "  ");
    }

    #[test]
    fn test_truncate_unicode() {
        // Test assorted Unicode characters
        let result = truncate_with_ellipsis("日本語テスト", 8);
        assert!(result.ends_with("..."));
        assert!(display_width(&result) <= 8);
    }

    #[test]
    fn test_count_trailing_blank_lines() {
        let lines = vec!["第一行".to_string(), String::new(), "   ".to_string()];
        assert_eq!(count_trailing_blank_lines(&lines), 2);
    }

    #[test]
    fn test_count_trailing_blank_lines_none() {
        let lines = vec!["第一行".to_string(), "第二行".to_string()];
        assert_eq!(count_trailing_blank_lines(&lines), 0);
    }

    #[test]
    fn empty_prompt_keeps_consistent_top_margin() {
        let layout = popup_layout_config(8, "", 1, 0, false, false, true);
        // Compact empty input: no decorative divider anymore, avoiding horizontal-line ghosts after resize.
        assert_eq!(layout.top_margin, 0);
        assert_eq!(layout.top_rule_lines, 0);
        assert_eq!(layout.help_lines, 1);
        assert_eq!(layout.model_header_lines, 1);
        assert_eq!(layout.min_textarea_lines, 1);
    }

    #[test]
    fn non_empty_prompt_keeps_full_editor_layout() {
        let layout = popup_layout_config(8, "hello", 1, 0, false, false, true);
        // Normal editing mode no longer draws a divider either, avoiding decorative rules piling into scrollback after resize/re-anchor.
        assert_eq!(layout.top_margin, 0);
        assert_eq!(layout.top_rule_lines, 0);
        assert_eq!(layout.help_lines, 1);
        assert_eq!(layout.model_header_lines, 1);
        assert_eq!(layout.min_textarea_lines, 1);
    }

    #[test]
    fn completion_panel_prioritizes_candidate_rows_over_chrome() {
        let layout = popup_layout_config(8, "/model", 1, 0, true, true, true);
        assert_eq!(layout.top_margin, 0);
        assert_eq!(layout.top_rule_lines, 0);
        assert_eq!(layout.help_lines, 1);
        assert_eq!(layout.model_header_lines, 0);
        assert_eq!(layout.min_textarea_lines, 1);
    }

    #[test]
    fn empty_prompt_parks_cursor_on_viewport_top_left() {
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(8),
            },
        )
        .unwrap();
        let mut textarea = TextArea::default();
        let mut viewport_area = Rect::ZERO;

        terminal
            .draw(|f| {
                viewport_area = f.area();
                render_multiline_popup(
                    f,
                    &mut textarea,
                    None,
                    None,
                    "glm-5.2-super-relay",
                    "max",
                    None,
                );
            })
            .unwrap();

        let rendered = (viewport_area.y..viewport_area.bottom())
            .map(|y| buffer_row(terminal.backend(), y, viewport_area.x, viewport_area.width))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("reasoning: max"));

        // Empty-input frames park the real terminal cursor on the viewport's
        // top-left cell so the position report after a reflow names the exact
        // viewport top (offset 0). The editing caret itself is drawn as a
        // styled cell at the textarea's first position, which stays on the
        // same row as the parked cursor for empty input.
        let expected = Position::new(viewport_area.x, viewport_area.y);
        terminal.backend_mut().assert_cursor_position(expected);
    }

    #[test]
    fn normal_editor_renders_without_decorative_divider_rows() {
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(8),
            },
        )
        .unwrap();
        let mut textarea = TextArea::from(vec!["hello".to_string()]);
        let mut viewport_area = Rect::ZERO;

        terminal
            .draw(|f| {
                viewport_area = f.area();
                render_multiline_popup(
                    f,
                    &mut textarea,
                    None,
                    None,
                    "glm-5.2-super-relay",
                    "max",
                    None,
                );
            })
            .unwrap();

        let popup_width = viewport_area
            .width
            .saturating_sub(2)
            .clamp(40, 180)
            .min(viewport_area.width);
        let popup_x = viewport_area.x + viewport_area.width.saturating_sub(popup_width) / 2;
        let inner_x = popup_x + 1;
        let inner_width = popup_width.saturating_sub(2);
        for y in viewport_area.y..viewport_area.bottom() {
            let row = buffer_row(terminal.backend(), y, inner_x, inner_width);
            assert!(
                !row.starts_with('╶'),
                "viewport row {y} still contains a decorative divider: {row:?}"
            );
        }
    }

    #[test]
    fn cjk_cursor_position_matches_popup_inner_left_plus_width() {
        // Under CJK input, the cursor should land at the popup's inner left edge on the textarea's first line plus the text's display width.
        // The previous hand-computation used width_cjk, which disagrees with tui-textarea's internal width,
        // shifting the cursor to the right (1 column per CJK character).
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(8),
            },
        )
        .unwrap();
        // 4 CJK characters, width 2 each, total display width 8
        let content = "你好世界";
        let mut textarea = TextArea::from(vec![content.to_string()]);
        // Move the cursor to the end of the text ("你好世界" → col 4)
        textarea.move_cursor(CursorMove::End);
        let mut viewport_area = Rect::ZERO;

        terminal
            .draw(|f| {
                viewport_area = f.area();
                render_multiline_popup(
                    f,
                    &mut textarea,
                    None,
                    None,
                    "glm-5.2-super-relay",
                    "max",
                    None,
                );
            })
            .unwrap();

        let popup_width = viewport_area
            .width
            .saturating_sub(2)
            .clamp(40, 180)
            .min(viewport_area.width);
        let popup_x = viewport_area.x + viewport_area.width.saturating_sub(popup_width) / 2;
        // Cursor at the end of "你好世界" → popup inner left(1) + display width of "你好世界"(8)
        let expected_x = popup_x + 1 + 8;
        let expected = Position::new(expected_x, viewport_area.y);
        terminal.backend_mut().assert_cursor_position(expected);
    }
}
