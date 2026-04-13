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

pub(in crate::ai::prompt::multiline) fn render_multiline_popup(
    f: &mut ratatui::Frame<'_>,
    textarea: &mut TextArea<'_>,
    status_msg: Option<&str>,
    completion_panel: Option<&CompletionPanel>,
) {
    let area = f.area();

    // 计算 popup 尺寸
    let popup_height = area.height.saturating_sub(4).clamp(10, 18);
    let popup_width = area.width.saturating_sub(2).clamp(60, 180).min(area.width);

    // 计算 popup 位置（居中）
    let popup_x = area.x + area.width.saturating_sub(popup_width) / 2;
    let popup_y = area.y + area.height.saturating_sub(popup_height) / 2;
    let popup = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // 计算内区域 - popup 的边框占据 1 格，标题占据 1 行
    let border: u16 = 1;
    let title_lines: u16 = 0;
    let inner = Rect::new(
        popup.x + border,
        popup.y + border + title_lines,
        popup.width - border * 2,
        popup.height - border * 2,
    );

    // 计算各区域高度
    let panel_lines: u16 = completion_panel
        .map_or(0u16, |p| (p.items.len().min(5) as u16).saturating_add(2));
    let help_lines: u16 = 2;
    let textarea_lines = inner.height.saturating_sub(panel_lines).saturating_sub(help_lines);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(textarea_lines.max(3)),
            Constraint::Length(panel_lines),
            Constraint::Length(help_lines),
        ])
        .split(inner);

    // textarea 的渲染区域
    let textarea_area = chunks[0];

    f.render_widget(Clear, popup);

    let popup_block = Block::default()
        .borders(Borders::TOP.union(Borders::BOTTOM))
        .border_style(Style::default().fg(Color::DarkGray));
    f.render_widget(popup_block, popup);

    let _n = textarea.lines().len();
    let current_content = textarea.lines().join("\n");
    let char_count = current_content.chars().count();

    // 设置对齐方式
    textarea.set_alignment(Alignment::Left);

    f.render_widget(&*textarea, textarea_area);

    // 渲染 completion panel
    if let Some(panel) = completion_panel {
        let window_size = 5usize;
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
                let style = if selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::LightCyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Gray)
                };

                Line::from(vec![
                    Span::styled(
                        if selected { "▶ " } else { "  " },
                        if selected {
                            style
                        } else {
                            Style::default().fg(Color::DarkGray)
                        },
                    ),
                    Span::styled(item, style),
                ])
            })
            .collect();

        let panel_block = Block::default()
            .borders(Borders::ALL)
            .border_style(
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .title(Span::styled(
                " Completions ",
                Style::default()
                    .fg(Color::Gray)
                    .add_modifier(Modifier::BOLD),
            ));

        f.render_widget(Paragraph::new(items).block(panel_block), chunks[1]);
    }

    // 获取光标位置
    let (cursor_row, cursor_col) = textarea.cursor();

    // 状态栏信息：字符数 + 光标位置
    let status_info = if char_count > MAX_INPUT_CHARS {
        format!(" Chars: {} (exceeded) | Ln {}, Col {} ", char_count, cursor_row + 1, cursor_col + 1)
    } else if char_count > MAX_INPUT_CHARS * 90 / 100 {
        format!(" Chars: {} (⚠) | Ln {}, Col {} ", char_count, cursor_row + 1, cursor_col + 1)
    } else {
        format!(" Ln {}, Col {} | Chars: {} ", cursor_row + 1, cursor_col + 1, char_count)
    };

    // 渲染帮助行
    let help_lines = if completion_panel.is_some() {
        vec![
            Line::from(vec![
                Span::styled("↵", Style::default().fg(Color::Blue)),
                Span::raw("select "),
                Span::styled("Esc", Style::default().fg(Color::Green)),
                Span::raw("close "),
                Span::styled("⌨C+C", Style::default().fg(Color::Yellow)),
                Span::raw("cancel "),
                Span::styled(status_info, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("↑↓", Style::default().fg(Color::Blue)),
                Span::raw(" move "),
                Span::styled("Tab", Style::default().fg(Color::Blue)),
                Span::raw(" refresh "),
                Span::styled("↵+Alt/F2", Style::default().fg(Color::Green)),
                Span::raw(" send "),
            ]),
        ]
    } else {
        vec![
            Line::from(vec![
                Span::styled("↵", Style::default().fg(Color::Blue)),
                Span::raw("newline "),
                Span::styled("Alt+↵/F2", Style::default().fg(Color::Green)),
                Span::raw("send "),
                Span::styled("⌨C+C", Style::default().fg(Color::Yellow)),
                Span::raw("cancel "),
                Span::styled(status_info, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("↑↓/⌨P/N", Style::default().fg(Color::Blue)),
                Span::raw(" hist "),
                Span::styled("⌫", Style::default().fg(Color::Blue)),
                Span::raw(" edit "),
                Span::styled("⌨V", Style::default().fg(Color::Blue)),
                Span::raw(" paste "),
                Span::styled("F9", Style::default().fg(Color::Blue)),
                Span::raw(" last "),
                Span::styled("F10", Style::default().fg(Color::Blue)),
                Span::raw(" full "),
            ]),
        ]
    };
    f.render_widget(Paragraph::new(help_lines), chunks[2]);

    // 渲染状态消息
    if let Some(msg) = status_msg {
        let status_width = chunks[2].width.saturating_sub(2) as usize;
        let status_text = truncate_with_ellipsis(msg, status_width);
        let status_para = Paragraph::new(Line::from(Span::styled(
            status_text,
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center);

        let status_area = Rect::new(chunks[2].x + 1, chunks[2].y + 1, chunks[2].width - 2, 1);
        f.render_widget(Clear, status_area);
        f.render_widget(status_para, status_area);
    }
}

/// 截断文本以适应显示宽度
/// 使用 unicode-width 包计算宽度，对于未识别的字符保守估计为宽度 1
fn truncate_with_ellipsis(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }

    let total = UnicodeWidthStr::width(text);
    if total <= max_width {
        return text.to_string();
    }

    let ellipsis_w = UnicodeWidthStr::width("...");
    if max_width <= ellipsis_w {
        return " ".repeat(max_width);
    }

    let target = max_width - ellipsis_w;
    let mut out = String::new();
    let mut width: usize = 0;

    for ch in text.chars() {
        // 对于 unicode-width 返回 0 的字符，保守估计为宽度 1
        let ch_w = UnicodeWidthChar::width(ch).unwrap_or(1);

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
    use super::truncate_with_ellipsis;
    use unicode_width::UnicodeWidthStr;

    fn display_width(s: &str) -> usize {
        UnicodeWidthStr::width(s)
    }

    #[test]
    fn test_truncate_cjk() {
        // 测试中文截断
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
        // 测试各种 Unicode 字符
        let result = truncate_with_ellipsis("日本語テスト", 8);
        assert!(result.ends_with("..."));
        assert!(display_width(&result) <= 8);
    }
}
