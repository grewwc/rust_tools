use tui_textarea::TextArea;

use super::super::completion::CommandCompleter;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::ai::prompt::multiline) struct PendingTabCompletion {
    row: usize,
    col: usize,
    token_start: usize,
    line: String,
    candidates: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::ai::prompt::multiline) struct CompletionPanel {
    pub(in crate::ai::prompt::multiline) items: Vec<String>,
    pub(in crate::ai::prompt::multiline) selected_index: usize,
}

fn replace_current_line_range(
    textarea: &mut TextArea<'_>,
    row: usize,
    start: usize,
    end: usize,
    replacement: &str,
) {
    let mut lines = textarea.lines().to_vec();
    if row >= lines.len() {
        return;
    }
    let line = &lines[row];
    let start = start.min(line.len());
    let end = end.min(line.len()).max(start);
    let mut new_line = String::with_capacity(line.len() + replacement.len());
    new_line.push_str(&line[..start]);
    new_line.push_str(replacement);
    new_line.push_str(&line[end..]);
    lines[row] = new_line;
    textarea.set_lines(lines, (row, start + replacement.len()));
}

fn build_multiline_completion_context(textarea: &TextArea<'_>) -> Option<PendingTabCompletion> {
    let (row, col) = textarea.cursor();
    let lines = textarea.lines();
    let line = lines.get(row)?.to_string();
    let (token_start, candidates) = CommandCompleter::complete_for_line(&line, col);
    Some(PendingTabCompletion {
        row,
        col,
        token_start,
        line,
        candidates,
    })
}

pub(in crate::ai::prompt::multiline) fn move_completion_selection(
    panel: &mut Option<CompletionPanel>,
    delta: isize,
) {
    let Some(panel) = panel.as_mut() else {
        return;
    };
    if panel.items.is_empty() {
        panel.selected_index = 0;
        return;
    }
    let last = panel.items.len().saturating_sub(1) as isize;
    panel.selected_index = (panel.selected_index as isize + delta).clamp(0, last) as usize;
}

pub(in crate::ai::prompt::multiline) fn confirm_completion_selection(
    textarea: &mut TextArea<'_>,
    pending: &mut Option<PendingTabCompletion>,
    completion_panel: &mut Option<CompletionPanel>,
) -> Option<String> {
    let Some((row, token_start, col)) = pending
        .as_ref()
        .map(|ctx| (ctx.row, ctx.token_start, ctx.col))
    else {
        *completion_panel = None;
        return None;
    };
    let Some(panel) = completion_panel.as_ref() else {
        return None;
    };
    let Some(selected) = panel.items.get(panel.selected_index).cloned() else {
        return None;
    };
    replace_current_line_range(textarea, row, token_start, col, &selected);
    *pending = None;
    *completion_panel = None;
    Some(format!("已选择 {}", selected))
}

pub(in crate::ai::prompt::multiline) fn dismiss_completion_panel(
    pending: &mut Option<PendingTabCompletion>,
    completion_panel: &mut Option<CompletionPanel>,
) {
    *pending = None;
    *completion_panel = None;
}

pub(in crate::ai::prompt::multiline) fn apply_multiline_completion(
    textarea: &mut TextArea<'_>,
    pending: &mut Option<PendingTabCompletion>,
    completion_panel: &mut Option<CompletionPanel>,
) -> Option<String> {
    let Some(ctx) = build_multiline_completion_context(textarea) else {
        *pending = None;
        *completion_panel = None;
        return Some("没有可补全的内容".to_string());
    };
    if ctx.candidates.is_empty() {
        *pending = None;
        *completion_panel = None;
        return Some("没有匹配的命令补全".to_string());
    }

    if ctx.candidates.len() == 1 {
        replace_current_line_range(
            textarea,
            ctx.row,
            ctx.token_start,
            ctx.col,
            &ctx.candidates[0],
        );
        *pending = None;
        *completion_panel = None;
        return Some(format!("已补全为 {}", ctx.candidates[0]));
    }

    let repeated_tab = pending.as_ref() == Some(&ctx);
    if !repeated_tab {
        *pending = Some(ctx.clone());
        *completion_panel = None;
        return None;
    }

    *pending = Some(ctx.clone());
    *completion_panel = Some(CompletionPanel {
        items: ctx.candidates.clone(),
        selected_index: 0,
    });
    Some(format!("发现 {} 个候选", ctx.candidates.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tui_textarea::CursorMove;

    #[test]
    fn multiline_completion_first_tab_is_silent_for_ambiguous_matches() {
        let mut textarea = TextArea::new(vec!["/agen".to_string()]);
        textarea.move_cursor(CursorMove::End);
        let mut pending = None;
        let mut panel = None;

        let status = apply_multiline_completion(&mut textarea, &mut pending, &mut panel);

        assert_eq!(textarea.lines(), vec!["/agen"]);
        assert!(pending.is_some());
        assert!(panel.is_none());
        assert!(status.is_none());
    }

    #[test]
    fn multiline_completion_second_tab_lists_candidates() {
        let mut textarea = TextArea::new(vec!["/agen".to_string()]);
        textarea.move_cursor(CursorMove::End);
        let mut pending = None;
        let mut panel = None;

        let _ = apply_multiline_completion(&mut textarea, &mut pending, &mut panel);
        let status = apply_multiline_completion(&mut textarea, &mut pending, &mut panel).unwrap();

        assert_eq!(textarea.lines(), vec!["/agen"]);
        assert!(pending.is_some());
        assert!(status.contains("发现 2 个候选"));
        let items = panel.as_ref().map(|panel| panel.items.clone()).unwrap();
        assert!(items.contains(&"/agent".to_string()));
        assert!(items.contains(&"/agents".to_string()));
    }

    #[test]
    fn multiline_completion_lists_candidates_for_subcommands() {
        let mut textarea = TextArea::new(vec!["/agents ".to_string()]);
        textarea.move_cursor(CursorMove::End);
        let mut pending = None;
        let mut panel = None;
        let first = apply_multiline_completion(&mut textarea, &mut pending, &mut panel);
        assert!(first.is_none());
        let status = apply_multiline_completion(&mut textarea, &mut pending, &mut panel).unwrap();

        assert!(status.contains("发现"));
        let items = panel.as_ref().map(|panel| panel.items.clone()).unwrap();
        assert!(items.contains(&"help".to_string()));
        assert!(items.contains(&"auto".to_string()));
    }

    #[test]
    fn completion_panel_selection_moves_with_arrow_navigation() {
        let mut panel = Some(CompletionPanel {
            items: vec!["/agents".to_string(), "/agent".to_string(), "/help".to_string()],
            selected_index: 0,
        });

        move_completion_selection(&mut panel, 1);
        assert_eq!(panel.as_ref().map(|panel| panel.selected_index), Some(1));

        move_completion_selection(&mut panel, 10);
        assert_eq!(panel.as_ref().map(|panel| panel.selected_index), Some(2));

        move_completion_selection(&mut panel, -10);
        assert_eq!(panel.as_ref().map(|panel| panel.selected_index), Some(0));
    }

    #[test]
    fn completion_panel_enter_confirms_selected_candidate() {
        let mut textarea = TextArea::new(vec!["/agen".to_string()]);
        textarea.move_cursor(CursorMove::End);
        let mut pending = None;
        let mut panel = None;

        let _ = apply_multiline_completion(&mut textarea, &mut pending, &mut panel);
        let _ = apply_multiline_completion(&mut textarea, &mut pending, &mut panel);
        move_completion_selection(&mut panel, 1);

        let status =
            confirm_completion_selection(&mut textarea, &mut pending, &mut panel).unwrap();

        assert_eq!(textarea.lines(), vec!["/agent"]);
        assert!(pending.is_none());
        assert!(panel.is_none());
        assert!(status.contains("已选择 /agent"));
    }

    #[test]
    fn dismiss_completion_panel_clears_panel_without_changing_input() {
        let mut textarea = TextArea::new(vec!["/agen".to_string()]);
        textarea.move_cursor(CursorMove::End);
        let mut pending = None;
        let mut panel = None;

        let _ = apply_multiline_completion(&mut textarea, &mut pending, &mut panel);
        let _ = apply_multiline_completion(&mut textarea, &mut pending, &mut panel);

        dismiss_completion_panel(&mut pending, &mut panel);

        assert_eq!(textarea.lines(), vec!["/agen"]);
        assert!(pending.is_none());
        assert!(panel.is_none());
    }
}
