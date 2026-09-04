use ratatui::{
    buffer::Buffer,
    layout::Alignment,
    layout::{Constraint, Direction, Layout},
    layout::{Position, Rect, Size},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};
use tui_textarea::{CursorRenderMode, TextArea};
use unicode_width::UnicodeWidthChar;
use unicode_width::UnicodeWidthStr;

use super::completion_panel::CompletionPanel;
use crate::ai::prompt::MAX_INPUT_CHARS;

/// Maximum number of candidate lines the completion panel shows at once (overflow scrolls with the selection).
const COMPLETION_WINDOW: usize = 12;

/// Styles only cells occupied by input text.
///
/// Applying a foreground color through `TextArea::set_style` styles the whole
/// widget rectangle, including otherwise empty cells. Those invisible styled
/// cells become real terminal output and can be reflowed into extra rows when
/// an IDE terminal is repeatedly narrowed and widened.
fn style_input_text(textarea: &mut TextArea<'_>, lines: &[String]) {
    let (red, green, blue) = crate::ai::theme::ACCENT_INPUT_RGB;
    let input_style = Style::default().fg(Color::Rgb(red, green, blue));
    let cursor_row = textarea.cursor().0;

    textarea.set_style(Style::default());
    textarea.set_cursor_line_style(Style::default());
    let selection_style = input_style.patch(textarea.selection_style());
    textarea.set_selection_style(selection_style);
    textarea.clear_custom_highlight();

    for (row, line) in lines.iter().enumerate() {
        if line.is_empty() {
            continue;
        }
        let style = if row == cursor_row {
            input_style.add_modifier(Modifier::UNDERLINED)
        } else {
            input_style
        };
        // tui-textarea highlight offsets are UTF-8 byte offsets.
        textarea.custom_highlight(((row, 0), (row, line.len())), style, 1);
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PopupGeometry {
    popup: Rect,
    chunks: [Rect; 5],
    /// Content area inside the input box border (chunks[1] inset by one cell on
    /// every side); the textarea widget renders here.
    textarea_inner: Rect,
}

fn popup_geometry(
    area: Rect,
    current_content: &str,
    content_lines: usize,
    trailing_blank_lines: usize,
    completion_items: Option<usize>,
    has_status_msg: bool,
    has_model_label: bool,
) -> PopupGeometry {
    let layout = popup_layout_config(
        area.height,
        current_content,
        content_lines,
        trailing_blank_lines,
        completion_items.is_some(),
        has_status_msg,
        has_model_label,
    );
    let popup_width = area.width.saturating_sub(2).clamp(40, 180).min(area.width);
    let popup_x = area.x + area.width.saturating_sub(popup_width) / 2;
    let popup = Rect::new(popup_x, area.y, popup_width, area.height);
    let horizontal_margin = 1;
    let inner = Rect::new(
        popup.x + horizontal_margin,
        popup.y + layout.top_margin,
        popup.width - horizontal_margin * 2,
        popup.height - layout.top_margin,
    );
    let (textarea_lines, panel_lines) = match completion_items {
        Some(items) => {
            let desired_panel = (items.min(COMPLETION_WINDOW) as u16).saturating_add(2);
            let panel_cap = inner
                .height
                .saturating_sub(layout.top_rule_lines)
                .saturating_sub(layout.help_lines)
                .saturating_sub(layout.model_header_lines)
                .saturating_sub(layout.min_textarea_lines);
            let panel = desired_panel.min(panel_cap).max(1.min(panel_cap));
            let textarea = inner
                .height
                .saturating_sub(layout.top_rule_lines)
                .saturating_sub(panel)
                .saturating_sub(layout.model_header_lines)
                .saturating_sub(layout.help_lines)
                .max(layout.min_textarea_lines);
            (textarea, panel)
        }
        None => {
            let textarea = inner
                .height
                .saturating_sub(layout.top_rule_lines)
                .saturating_sub(layout.model_header_lines)
                .saturating_sub(layout.help_lines)
                .max(layout.min_textarea_lines);
            (textarea, 0)
        }
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(layout.top_rule_lines),
            // chunks[1] is the input box. The left-edge bar occupies one column
            // but no extra rows, so its height is exactly the textarea's line
            // count.
            Constraint::Length(textarea_lines),
            Constraint::Length(panel_lines),
            Constraint::Length(layout.model_header_lines),
            Constraint::Length(layout.help_lines),
        ])
        .split(inner);

    // Inset chunks[1] by the single left gutter column to get the area the
    // textarea draws into; the left bar itself is rendered separately in
    // `render_multiline_popup`. A LEFT border shifts content right by one column
    // and adds no rows.
    let textarea_inner = Block::default()
        .borders(Borders::LEFT)
        .inner(chunks[1]);

    PopupGeometry {
        popup,
        chunks: [chunks[0], chunks[1], chunks[2], chunks[3], chunks[4]],
        textarea_inner,
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
) -> Option<Position> {
    let area = f.area();
    let current_lines = textarea.lines().to_vec();
    let current_content = current_lines.join("\n");
    let trailing_blank_lines = count_trailing_blank_lines(&current_lines);
    let geometry = popup_geometry(
        area,
        &current_content,
        current_lines.len(),
        trailing_blank_lines,
        completion_panel.map(|panel| panel.items.len()),
        status_msg.is_some(),
        !model_label.is_empty(),
    );
    let popup = geometry.popup;
    let chunks = geometry.chunks;
    let textarea_area = chunks[1];

    // Clear the popup area so old borders/text do not linger after a resize
    f.render_widget(Clear, popup);

    // Bottom model/topic info line: lets the user see the current model and session topic while typing.
    // Draw it in the dedicated bottom chunk (chunks[3], above the help line) so
    // textarea growth does not move it through the editing region.
    if chunks[3].height > 0 {
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
    // Keep the existing input color and current-line underline, but apply them
    // only to actual characters so blank cells cannot participate in reflow.
    style_input_text(textarea, &current_lines);
    // Draw the editing caret into the buffer as a thin vertical bar instead of
    // a reversed block. The real terminal cursor is kept hidden and parked on
    // the viewport's final row by the input loop so that terminal width reflow
    // cannot split the long help line below the caret into persistent
    // scrollback rows. A drawn caret is required here: the hardware cursor is
    // the width-reflow anchor and cannot also sit at the editing position.
    textarea.set_cursor_render_mode(CursorRenderMode::Hidden);

    // Input box: a left-edge vertical bar marking the editing area, in the same
    // color as the completion panel frame. Only the LEFT border is drawn: a full
    // frame's top/bottom rules fill the popup width and wrap into permanent
    // scrollback rows when the terminal is narrowed (the same reflow hazard the
    // parked-cursor anchor exists to avoid). A single left-column glyph can never
    // reach the right margin, so it survives width changes without leaving
    // ghost boxes.
    let input_bar = Block::default()
        .borders(Borders::LEFT)
        .border_style(
            Style::default()
                .fg(Color::Rgb(74, 92, 112))
                .add_modifier(Modifier::BOLD),
        );
    f.render_widget(&input_bar, textarea_area);
    f.render_widget(&*textarea, geometry.textarea_inner);
    // Reuse tui-textarea's own rendering plan for the visual caret position.
    // This avoids a duplicate CJK-width calculation that can disagree with the
    // widget's internal screen map.
    let visual_cursor_position = textarea.rendered_cursor_position();
    if let Some(position) = visual_cursor_position {
        // LEFT ONE EIGHTH BLOCK (U+258F): a 1/8-width vertical bar at the
        // cell's left edge, matching a hardware bar cursor. Replacing the cell
        // hides any character underneath only while the caret sits on it.
        let (red, green, blue) = crate::ai::theme::ACCENT_INPUT_RGB;
        let caret_cell = &mut f.buffer_mut()[(position.x, position.y)];
        caret_cell.set_symbol("▏");
        caret_cell.set_style(Style::default().fg(Color::Rgb(red, green, blue)));
    }

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

    visual_cursor_position.map(|position| {
        Position::new(
            position.x.saturating_sub(area.x),
            position.y.saturating_sub(area.y),
        )
    })
}

pub(in crate::ai::prompt::multiline) fn measure_multiline_cursor_offset(
    textarea: &TextArea<'_>,
    viewport_size: Size,
    status_msg: Option<&str>,
    completion_panel: Option<&CompletionPanel>,
    model_label: &str,
) -> Option<Position> {
    let area = Rect::new(0, 0, viewport_size.width, viewport_size.height);
    let current_lines = textarea.lines().to_vec();
    let current_content = current_lines.join("\n");
    let geometry = popup_geometry(
        area,
        &current_content,
        current_lines.len(),
        count_trailing_blank_lines(&current_lines),
        completion_panel.map(|panel| panel.items.len()),
        status_msg.is_some(),
        !model_label.is_empty(),
    );
    let mut measured = textarea.clone();
    measured.set_alignment(Alignment::Left);
    style_input_text(&mut measured, &current_lines);
    measured.set_cursor_render_mode(CursorRenderMode::Hidden);
    let mut buffer = Buffer::empty(area);
    Widget::render(&measured, geometry.textarea_inner, &mut buffer);
    measured.rendered_cursor_position()
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
        style::{Color, Modifier},
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
    fn empty_prompt_draws_buffer_cursor_and_hides_hardware_cursor() {
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
        let mut cursor_position = None;

        terminal
            .draw(|f| {
                viewport_area = f.area();
                cursor_position = render_multiline_popup(
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

        let popup_width = viewport_area
            .width
            .saturating_sub(2)
            .clamp(40, 180)
            .min(viewport_area.width);
        let popup_x = viewport_area.x + viewport_area.width.saturating_sub(popup_width) / 2;
        // The caret sits one column right of the left-edge marker bar. The bar
        // occupies a column but no row, so the caret stays on the box's top row.
        let visual_caret = Position::new(popup_x + 2, viewport_area.y);
        assert_eq!(
            cursor_position,
            Some(Position::new(
                visual_caret.x.saturating_sub(viewport_area.x),
                visual_caret.y.saturating_sub(viewport_area.y),
            ))
        );
        assert!(!terminal.backend().cursor_visible());
        // The visible caret is a drawn vertical-bar cell; the input loop parks
        // the hidden hardware cursor at the viewport tail after the draw.
        let caret_cell = &terminal.backend().buffer()[(visual_caret.x, visual_caret.y)];
        assert_eq!(caret_cell.symbol(), "▏");
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
    fn input_style_does_not_fill_blank_textarea_cells() {
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
        // Textarea content renders one column right of the left-edge marker bar;
        // the bar adds no row, so content starts on the box's top row.
        let input_x = popup_x + 2;
        let input_y = viewport_area.y;
        // The first character is the reversed buffer cursor, so inspect the
        // next text cell when checking that input styling does not fill blanks.
        let text_cell = &terminal.backend().buffer()[(input_x + 1, input_y)];
        let blank_cell = &terminal.backend().buffer()[(input_x + 20, input_y)];
        let (red, green, blue) = crate::ai::theme::ACCENT_INPUT_RGB;

        assert_eq!(text_cell.symbol(), "e");
        assert_eq!(text_cell.fg, Color::Rgb(red, green, blue));
        assert!(text_cell.modifier.contains(Modifier::UNDERLINED));
        assert_eq!(blank_cell.fg, Color::Reset);
        assert!(!blank_cell.modifier.contains(Modifier::UNDERLINED));

        textarea.start_selection();
        textarea.move_cursor(CursorMove::End);
        terminal
            .draw(|f| {
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

        let selected_cell = &terminal.backend().buffer()[(input_x, input_y)];
        let blank_cell = &terminal.backend().buffer()[(input_x + 20, input_y)];
        assert_eq!(selected_cell.fg, Color::Rgb(red, green, blue));
        assert_eq!(selected_cell.bg, Color::LightBlue);
        assert_eq!(blank_cell.fg, Color::Reset);
        assert_eq!(blank_cell.bg, Color::Reset);
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
        // Move the cursor to the end of the four-character CJK text.
        textarea.move_cursor(CursorMove::End);
        let mut viewport_area = Rect::ZERO;
        let mut cursor_offset = None;

        terminal
            .draw(|f| {
                viewport_area = f.area();
                cursor_offset = render_multiline_popup(
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
        // The buffer caret at the end of the CJK text uses tui-textarea's
        // calculated position, avoiding an independent CJK-width calculation.
        // Content starts one column right of the left-edge marker bar; the bar
        // occupies no row, so the first text line sits on the box's top row.
        let expected_x = popup_x + 2 + 8;
        let expected = Position::new(expected_x, viewport_area.y);
        assert_eq!(
            cursor_offset,
            Some(Position::new(
                expected.x.saturating_sub(viewport_area.x),
                expected.y.saturating_sub(viewport_area.y),
            ))
        );
        assert!(!terminal.backend().cursor_visible());
        // The cell at the measured caret holds the drawn vertical-bar glyph.
        let caret_cell = &terminal.backend().buffer()[(expected.x, expected.y)];
        assert_eq!(caret_cell.symbol(), "▏");
    }
}
