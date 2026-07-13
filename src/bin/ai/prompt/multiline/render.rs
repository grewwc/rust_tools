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
const COMPLETION_WINDOW: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PopupLayoutConfig {
    top_margin: u16,
    top_rule_lines: u16,
    help_lines: u16,
    model_header_lines: u16,
    spacer_lines: u16,
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
    // top_margin 始终保持 0。不要在 inline viewport 的**顶行**放稳定装饰元素：
    // 终端 resize / reflow 时，顶行内容会被推入 scrollback，恢复宽度后就会像
    // “多出一条横线”一样不断堆叠。把分隔线移到 footer 上方的 spacer 行里，
    // 既保留视觉收口，又避免顶行残影。
    let top_margin: u16 = 0;
    let top_rule_lines: u16 = 0;
    // 补全面板激活时，把底部帮助压缩为 1 行并隐藏 model/session 信息，
    // 优先把高度让给候选列表；小终端里这能显著减少"只能看到 1 个候选"的情况。
    let help_lines: u16 = if has_completion_panel { 1 } else { 2 };
    let model_header_lines = if has_completion_panel || !has_model_label {
        0
    } else {
        1
    };
    let spacer_lines = if has_completion_panel { 0 } else { 1 };
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
        spacer_lines,
        min_textarea_lines,
    }
}

pub(in crate::ai::prompt::multiline) fn render_multiline_popup(
    f: &mut ratatui::Frame<'_>,
    textarea: &mut TextArea<'_>,
    status_msg: Option<&str>,
    completion_panel: Option<&CompletionPanel>,
    model_label: &str,
    session_topic: Option<&str>,
) {
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

    // 计算 popup 尺寸：始终填满当前 inline viewport。空输入场景的留白通过更小的
    // viewport 高度和去掉顶部间距解决，而不是在 viewport 内再造未使用区域。
    let popup_height = area.height;
    let popup_width = area.width.saturating_sub(2).clamp(40, 180).min(area.width);

    // 计算 popup 位置（顶部对齐，紧贴上次输出）
    let popup_x = area.x + area.width.saturating_sub(popup_width) / 2;
    let popup_y = area.y;
    let popup = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // 计算内区域：左右各 1 列水平边距，不额外增加顶部/底部 padding，
    // 避免在 inline viewport 里制造多余空白行。
    let h_margin: u16 = 1;
    let top_margin = layout.top_margin;
    let inner = Rect::new(
        popup.x + h_margin,
        popup.y + top_margin,
        popup.width - h_margin * 2,
        popup.height - top_margin,
    );

    // 计算各区域高度
    let top_rule_lines = layout.top_rule_lines;
    let help_lines = layout.help_lines;
    // 模型/主题信息行：放在底部（help 行上方）而非 viewport 顶行。inline viewport 每次
    // 重新锚定（resize、上方有新输出、退出清屏）时，**顶行**会被推入终端永久 scrollback
    // 且退出时的 `Clear(FromCursorDown)` 无法擦除光标上方历史行——把这行画在顶部会像
    // 装饰性横线一样反复堆叠（表现为 `model: ... | ...` 在恢复/重绘时多次出现）。底部区域
    // 在光标下方，随每帧重绘、退出时被 FromCursorDown 清除，不会污染 scrollback。
    let model_header_lines = layout.model_header_lines;
    // 正文和底部帮助/状态栏之间预留 1 行视觉间距，但只在以下情况下启用：
    // 1. 没有补全面板（否则可用高度太紧张）；
    // 2. 正文末尾自己没有留空行。
    let spacer_lines = layout.spacer_lines;
    // 面板激活时优先占满高度：先扣掉 help 行与 textarea 最小行，余下的尽量给面板
    // （面板期望高度 = min(候选数, COMPLETION_WINDOW) + 上下边框 2，但不超过可用空间）。
    // textarea 退让到最小 1 行（此时用户在选列表，不需要大编辑区）。
    // 无面板时按当前内容与 viewport 高度自适应。
    let min_textarea_lines = layout.min_textarea_lines;
    let (textarea_lines, panel_lines) = match completion_panel {
        Some(panel) => {
            let desired_panel = (panel.items.len().min(COMPLETION_WINDOW) as u16).saturating_add(2);
            // 面板可用上限 = 总高度 - help - textarea 最小行。
            let panel_cap = inner
                .height
                .saturating_sub(top_rule_lines)
                .saturating_sub(help_lines)
                .saturating_sub(model_header_lines)
                .saturating_sub(spacer_lines)
                .saturating_sub(min_textarea_lines);
            let panel = desired_panel.min(panel_cap).max(1.min(panel_cap));
            let textarea = inner
                .height
                .saturating_sub(top_rule_lines)
                .saturating_sub(panel)
                .saturating_sub(spacer_lines)
                .saturating_sub(model_header_lines)
                .saturating_sub(help_lines)
                .max(min_textarea_lines);
            (textarea, panel)
        }
        None => {
            let textarea = inner
                .height
                .saturating_sub(top_rule_lines)
                .saturating_sub(spacer_lines)
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
            Constraint::Length(spacer_lines),
            Constraint::Length(model_header_lines),
            Constraint::Length(help_lines),
        ])
        .split(inner);

    // textarea 的渲染区域
    let textarea_area = chunks[1];

    // 清除 popup 区域，确保 resize 后旧边框/文本不残留
    f.render_widget(Clear, popup);

    // 底部模型/主题信息行：让用户在输入时看到当前模型与 session 主题。
    // **必须画在底部专属 chunk（chunks[4]，help 行上方）而非 viewport 顶行**：顶行会在
    // 每次 viewport 重新锚定时被推进 scrollback 反复堆叠（见上方 model_header_lines 注释）。
    if model_header_lines > 0 {
        let header_area = chunks[4];
        let mut spans = vec![
            Span::styled(" model: ", Style::default().fg(Color::Rgb(148, 163, 184))),
            Span::styled(
                model_label,
                Style::default()
                    .fg(Color::Rgb(134, 194, 166))
                    .add_modifier(Modifier::BOLD),
            ),
        ];
        // 在 model 同行展示 session 主题
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

    // 这里用单独 chunk 画“细分隔线”，而不是粗边框。分隔线放在 footer 上方的
    // spacer 行中，避免顶行在 resize/reflow 时被推进 scrollback 后留下横线残影。
    if spacer_lines > 0 {
        f.render_widget(Paragraph::new(divider_line(chunks[3].width)), chunks[3]);
    }

    let char_count = current_content.chars().count();

    // 设置对齐方式
    textarea.set_alignment(Alignment::Left);
    // tui-textarea 默认用 REVERSED 空格把 cursor 画进 buffer；在 ratatui inline
    // viewport 下，resize 重锚会把这块"画出来的 cursor"推进 scrollback，表现为
    // 每次侧栏展开/收回都多一个白色 cursor 残影。这里禁用 buffer cursor，改用
    // ratatui 的真实终端 cursor（见下方 set_cursor_position），它不会成为可持久化内容。
    textarea.set_cursor_style(Style::default());

    f.render_widget(&*textarea, textarea_area);
    if let Some((cursor_x, cursor_y)) = textarea_terminal_cursor(textarea, textarea_area) {
        f.set_cursor_position((cursor_x, cursor_y));
    }

    // 渲染 completion panel
    if let Some(panel) = completion_panel {
        // 滚动窗口必须用面板**实际可见行数**（chunk 高度减去上下边框），
        // 而不是固定 COMPLETION_WINDOW：在矮终端下 layout 会把面板挤压成
        // 比 COMPLETION_WINDOW 更少的行，若仍按固定值算 `start`，选中项一旦
        // 越过可见区就会落到屏幕外，表现为"卡在前几项、无法滚动"。
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
        vec![
            Line::from(vec![
                Span::styled("换行：", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "↵",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("发送：", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Alt+↵/F2",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("取消：", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Ctrl+C",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                Span::styled(status_info, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("历史：", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "↑↓/Ctrl+P/N",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("删行：", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "⌘/Ctrl+U",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("粘贴：", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Ctrl+V",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("清空：", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "F8",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("复制回答：", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "F9",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("复制全部：", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "F10",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
        ]
    };
    f.render_widget(Paragraph::new(help_lines), chunks[5]);

    if let Some(msg) = status_msg {
        let c2 = chunks[5];
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

fn divider_line(width: u16) -> Line<'static> {
    if width <= 2 {
        return Line::from(Span::styled(
            "─".repeat(width as usize),
            Style::default().fg(Color::Rgb(71, 85, 105)),
        ));
    }
    let body_width = width.saturating_sub(2) as usize;
    Line::from(vec![
        Span::styled("╶", Style::default().fg(Color::Rgb(100, 116, 139))),
        Span::styled(
            "─".repeat(body_width),
            Style::default().fg(Color::Rgb(71, 85, 105)),
        ),
        Span::styled("╴", Style::default().fg(Color::Rgb(100, 116, 139))),
    ])
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

/// 截断文本以适应显示宽度
/// 使用 unicode-width 包计算宽度，对于未识别的字符保守估计为宽度 1
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
        // 对于 unicode-width 返回 0 的字符，保守估计为宽度 1
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

fn textarea_terminal_cursor(textarea: &TextArea<'_>, area: Rect) -> Option<(u16, u16)> {
    if area.width == 0 || area.height == 0 {
        return None;
    }

    let (cursor_row, cursor_col) = textarea.cursor();
    let rows = textarea.lines().len();
    if rows == 0 {
        return Some((area.x, area.y));
    }

    let visible_rows = area.height as usize;
    let top_row = cursor_row.saturating_sub(visible_rows.saturating_sub(1));
    let y = area.y + cursor_row.saturating_sub(top_row).min(visible_rows - 1) as u16;

    let line = textarea
        .lines()
        .get(cursor_row)
        .map(|s| s.as_str())
        .unwrap_or("");
    let cursor_col_width = line
        .chars()
        .take(cursor_col)
        .map(|ch| unicode_width::UnicodeWidthChar::width_cjk(ch).unwrap_or(0))
        .sum::<usize>();
    let visible_cols = area.width as usize;
    let left_col = cursor_col_width.saturating_sub(visible_cols.saturating_sub(1));
    let x = area.x
        + cursor_col_width
            .saturating_sub(left_col)
            .min(visible_cols - 1) as u16;

    Some((x, y))
}

#[cfg(test)]
mod tests {
    use super::{
        count_trailing_blank_lines, divider_line, popup_layout_config, render_multiline_popup,
        truncate_with_ellipsis,
    };
    use ratatui::{
        Terminal, TerminalOptions, Viewport,
        backend::TestBackend,
        layout::{Position, Rect},
    };
    use tui_textarea::TextArea;
    use unicode_width::UnicodeWidthStr;

    fn display_width(s: &str) -> usize {
        UnicodeWidthStr::width_cjk(s)
    }

    fn buffer_row(
        backend: &TestBackend,
        y: u16,
        x_start: u16,
        width: u16,
    ) -> String {
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
        // 紧凑空输入：顶行不再放 divider，普通编辑态把收口线放到 footer 上方 spacer。
        assert_eq!(layout.top_margin, 0);
        assert_eq!(layout.top_rule_lines, 0);
        assert_eq!(layout.help_lines, 2);
        assert_eq!(layout.model_header_lines, 1);
        assert_eq!(layout.spacer_lines, 1);
        assert_eq!(layout.min_textarea_lines, 1);
    }

    #[test]
    fn non_empty_prompt_keeps_full_editor_layout() {
        let layout = popup_layout_config(8, "hello", 1, 0, false, false, true);
        // 普通编辑态统一 chrome 布局：divider 在 spacer 行，避免 resize 时顶行残影。
        assert_eq!(layout.top_margin, 0);
        assert_eq!(layout.top_rule_lines, 0);
        assert_eq!(layout.help_lines, 2);
        assert_eq!(layout.model_header_lines, 1);
        assert_eq!(layout.spacer_lines, 1);
        assert_eq!(layout.min_textarea_lines, 1);
    }

    #[test]
    fn completion_panel_prioritizes_candidate_rows_over_chrome() {
        let layout = popup_layout_config(8, "/model", 1, 0, true, true, true);
        assert_eq!(layout.top_margin, 0);
        assert_eq!(layout.top_rule_lines, 0);
        assert_eq!(layout.help_lines, 1);
        assert_eq!(layout.model_header_lines, 0);
        assert_eq!(layout.spacer_lines, 0);
        assert_eq!(layout.min_textarea_lines, 1);
    }

    #[test]
    fn empty_prompt_cursor_renders_below_top_margin() {
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
                render_multiline_popup(f, &mut textarea, None, None, "glm-5.2-super-relay", None);
            })
            .unwrap();

        let popup_width = viewport_area
            .width
            .saturating_sub(2)
            .clamp(40, 180)
            .min(viewport_area.width);
        let popup_x = viewport_area.x + viewport_area.width.saturating_sub(popup_width) / 2;
        // 顶部分隔线已移走，空输入时光标应直接落在 textarea 首行。
        let expected = Position::new(popup_x + 1, viewport_area.y);
        terminal.backend_mut().assert_cursor_position(expected);
    }

    #[test]
    fn normal_editor_divider_renders_in_footer_spacer_instead_of_top_row() {
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
                render_multiline_popup(f, &mut textarea, None, None, "glm-5.2-super-relay", None);
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

        let top_row = buffer_row(terminal.backend(), viewport_area.y, inner_x, inner_width);
        let spacer_row = buffer_row(terminal.backend(), viewport_area.y + 4, inner_x, inner_width);

        assert!(!top_row.starts_with('╶'));
        assert!(spacer_row.starts_with('╶'));
    }

    #[test]
    fn divider_line_spans_requested_width() {
        let rendered = divider_line(12);
        assert_eq!(rendered.width(), 12);
    }
}
