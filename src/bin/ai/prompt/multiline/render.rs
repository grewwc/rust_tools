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

/// 补全面板一次最多显示的候选行数（超出部分随选中项滚动）。
const COMPLETION_WINDOW: usize = 8;

pub(in crate::ai::prompt::multiline) fn render_multiline_popup(
    f: &mut ratatui::Frame<'_>,
    textarea: &mut TextArea<'_>,
    status_msg: Option<&str>,
    completion_panel: Option<&CompletionPanel>,
) {
    let area = f.area();

    // 计算 popup 尺寸
    let popup_height = area.height.saturating_sub(2).clamp(8, 28).min(area.height);
    let popup_width = area.width.saturating_sub(2).clamp(40, 180).min(area.width);

    // 计算 popup 位置（顶部对齐，减少与上方日志的空白间隔）
    let popup_x = area.x + area.width.saturating_sub(popup_width) / 2;
    let popup_y = area.y;
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
    // 补全面板可视窗口：最多展示 COMPLETION_WINDOW 项（含上下边框 +2）。
    let panel_lines: u16 = completion_panel
        .map_or(0u16, |p| (p.items.len().min(COMPLETION_WINDOW) as u16).saturating_add(2));
    let help_lines: u16 = 2;
    // 补全面板激活时优先保证面板完整显示：textarea 退让到最小 1 行
    // （此时用户在选列表，不需要大编辑区），避免矮 viewport 下面板被挤成 1~2 行
    // 而"只看到一个候选"。无面板时 textarea 至少保留 3 行。
    let min_textarea_lines: u16 = if completion_panel.is_some() { 1 } else { 3 };
    let textarea_lines = inner
        .height
        .saturating_sub(panel_lines)
        .saturating_sub(help_lines)
        .max(min_textarea_lines);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(textarea_lines),
            Constraint::Length(panel_lines),
            Constraint::Length(help_lines),
        ])
        .split(inner);

    // textarea 的渲染区域
    let textarea_area = chunks[0];

    // 清除 popup 区域，确保 resize 后旧边框/文本不残留
    f.render_widget(Clear, popup);

    // 注意：这里**故意不画** `Borders::TOP/BOTTOM` 全宽横线。
    // 在 ratatui inline viewport 下，每当 viewport 重新锚定（resize 突发、
    // 上方有新输出、退出清屏）时，顶部/底部的全宽横线会被推入终端的
    // 永久 scrollback，表现为「输入框上方堆叠多条横线」的 bug——而退出时
    // 的 `Clear(FromCursorDown)` 无法擦除光标上方的历史行。去掉装饰性横线
    // 即可根除该污染；底部 help 行已提供足够的区域分隔。
    let _n = textarea.lines().len();
    let current_content = textarea.lines().join("\n");
    let char_count = current_content.chars().count();

    // 设置对齐方式
    textarea.set_alignment(Alignment::Left);

    f.render_widget(&*textarea, textarea_area);

    // 渲染 completion panel
    if let Some(panel) = completion_panel {
        let window_size = COMPLETION_WINDOW;
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
                    Span::styled(&item.display, style),
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

    // 渲染帮助行
    let help_lines = if completion_panel.is_some() {
        vec![
            Line::from(vec![
                Span::styled("选择：", Style::default().fg(Color::DarkGray)),
                Span::styled("", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("关闭：", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("取消：", Style::default().fg(Color::DarkGray)),
                Span::styled("Ctrl+C", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::styled(status_info, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("移动：", Style::default().fg(Color::DarkGray)),
                Span::styled("↑↓", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("刷新：", Style::default().fg(Color::DarkGray)),
                Span::styled("Tab", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("发送：", Style::default().fg(Color::DarkGray)),
                Span::styled("+Alt/F2", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            ]),
        ]
    } else {
        vec![
            Line::from(vec![
                Span::styled("换行：", Style::default().fg(Color::DarkGray)),
                Span::styled("↵", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("发送：", Style::default().fg(Color::DarkGray)),
                Span::styled("Alt+↵/F2", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("取消：", Style::default().fg(Color::DarkGray)),
                Span::styled("Ctrl+C", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::styled(status_info, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("历史：", Style::default().fg(Color::DarkGray)),
                Span::styled("↑↓/Ctrl+P/N", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("删行：", Style::default().fg(Color::DarkGray)),
                Span::styled("⌘/Ctrl+U", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("粘贴：", Style::default().fg(Color::DarkGray)),
                Span::styled("Ctrl+V", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("复制回答：", Style::default().fg(Color::DarkGray)),
                Span::styled("F9", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("复制全部：", Style::default().fg(Color::DarkGray)),
                Span::styled("F10", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            ]),
        ]
    };
    f.render_widget(Paragraph::new(help_lines), chunks[2]);

    if let Some(msg) = status_msg {
        let c2 = chunks[2];
        if c2.height >= 2 && c2.width > 2 {
            let status_width = (c2.width - 2) as usize;
            let status_text = truncate_with_ellipsis(msg, status_width);
            let status_para = Paragraph::new(Line::from(Span::styled(
                status_text,
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )))
            .alignment(Alignment::Center);

            let status_area = Rect::new(c2.x + 1, c2.y + 1, c2.width - 2, 1);
            f.render_widget(Clear, status_area);
            f.render_widget(status_para, status_area);
        }
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
