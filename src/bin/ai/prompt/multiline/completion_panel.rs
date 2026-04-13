use tui_textarea::TextArea;

use super::super::completion::{CommandCompleter, CompletionCandidate};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::ai::prompt::multiline) struct PendingTabCompletion {
    row: usize,
    col: usize,
    token_start: usize,
    line: String,
    candidates: Vec<CompletionCandidate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::ai::prompt::multiline) struct CompletionPanel {
    pub(in crate::ai::prompt::multiline) items: Vec<CompletionCandidate>,
    pub(in crate::ai::prompt::multiline) selected_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::ai::prompt::multiline) struct CompletionConfirmResult {
    pub(in crate::ai::prompt::multiline) status: Option<String>,
    pub(in crate::ai::prompt::multiline) submit: Option<String>,
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

fn should_open_popup_on_first_tab(ctx: &PendingTabCompletion) -> bool {
    let trimmed = ctx.line[..ctx.col].trim();
    matches!(trimmed, "/model" | ":model")
        || trimmed.starts_with("/model ")
        || trimmed.starts_with(":model ")
}

fn should_submit_immediately(line: &str) -> bool {
    let trimmed = line.trim();
    (trimmed.starts_with("/model ") || trimmed.starts_with(":model "))
        && trimmed.split_whitespace().nth(1).is_some()
        && !matches!(
            trimmed.split_whitespace().nth(1),
            Some("help" | "current" | "list")
        )
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
) -> CompletionConfirmResult {
    let Some((row, token_start, col)) = pending
        .as_ref()
        .map(|ctx| (ctx.row, ctx.token_start, ctx.col))
    else {
        *completion_panel = None;
        return CompletionConfirmResult {
            status: None,
            submit: None,
        };
    };
    let Some(panel) = completion_panel.as_ref() else {
        return CompletionConfirmResult {
            status: None,
            submit: None,
        };
    };
    let Some(selected) = panel.items.get(panel.selected_index).cloned() else {
        return CompletionConfirmResult {
            status: None,
            submit: None,
        };
    };
    replace_current_line_range(textarea, row, token_start, col, &selected.replacement);
    *pending = None;
    *completion_panel = None;
    let final_line = textarea
        .lines()
        .get(row)
        .cloned()
        .unwrap_or_else(String::new);
    CompletionConfirmResult {
        status: Some(format!("已选择 {}", selected.replacement)),
        submit: should_submit_immediately(&final_line).then_some(final_line),
    }
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
            &ctx.candidates[0].replacement,
        );
        *pending = None;
        *completion_panel = None;
        return Some(format!("已补全为 {}", ctx.candidates[0].replacement));
    }

    let repeated_tab = pending.as_ref() == Some(&ctx);
    if !repeated_tab && !should_open_popup_on_first_tab(&ctx) {
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
        assert!(items.iter().any(|item| item.replacement == "/agent"));
        assert!(items.iter().any(|item| item.replacement == "/agents"));
    }

    #[test]
    fn model_completion_opens_popup_on_first_tab() {
        let current = crate::ai::model_names::all()
            .first()
            .expect("models.json is empty")
            .name
            .clone();
        CommandCompleter::set_current_model_hint(&current);

        let mut textarea = TextArea::new(vec!["/model".to_string()]);
        textarea.move_cursor(CursorMove::End);
        let mut pending = None;
        let mut panel = None;

        let status = apply_multiline_completion(&mut textarea, &mut pending, &mut panel).unwrap();

        assert!(status.contains("发现"));
        let panel = panel.expect("panel should open on first tab");
        assert!(!panel.items.is_empty());
        assert_eq!(panel.selected_index, 0);
        assert_eq!(panel.items[0].replacement, format!("/model {current}"));
        assert!(panel.items[0].display.contains("current"));
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
        assert!(items.iter().any(|item| item.replacement == "help"));
        assert!(items.iter().any(|item| item.replacement == "auto"));
    }

    #[test]
    fn completion_panel_selection_moves_with_arrow_navigation() {
        let mut panel = Some(CompletionPanel {
            items: vec![
                CompletionCandidate {
                    display: "/agents".to_string(),
                    replacement: "/agents".to_string(),
                },
                CompletionCandidate {
                    display: "/agent".to_string(),
                    replacement: "/agent".to_string(),
                },
                CompletionCandidate {
                    display: "/help".to_string(),
                    replacement: "/help".to_string(),
                },
            ],
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

        let result = confirm_completion_selection(&mut textarea, &mut pending, &mut panel);

        assert_eq!(textarea.lines(), vec!["/agent"]);
        assert!(pending.is_none());
        assert!(panel.is_none());
        assert_eq!(result.submit, None);
        assert_eq!(result.status.as_deref(), Some("已选择 /agent"));
    }

    #[test]
    fn model_panel_confirmation_submits_immediately() {
        let current = crate::ai::model_names::all()
            .first()
            .expect("models.json is empty")
            .name
            .clone();
        CommandCompleter::set_current_model_hint(&current);

        let mut textarea = TextArea::new(vec!["/model".to_string()]);
        textarea.move_cursor(CursorMove::End);
        let mut pending = None;
        let mut panel = None;
        let _ = apply_multiline_completion(&mut textarea, &mut pending, &mut panel);

        let result = confirm_completion_selection(&mut textarea, &mut pending, &mut panel);

        assert_eq!(result.submit, Some(format!("/model {current}")));
        assert_eq!(textarea.lines(), vec![format!("/model {current}")]);
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

    #[test]
    fn multiline_completion_lists_file_reference_candidates() {
        let dir = std::env::temp_dir().join(format!("ai-multi-complete-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let image = dir.join("screen.png");
        let other = dir.join("script.rs");
        std::fs::write(&image, b"fake").unwrap();
        std::fs::write(&other, b"fn main() {}").unwrap();

        let mut textarea = TextArea::new(vec![format!("@{}", dir.join("scr").display())]);
        textarea.move_cursor(CursorMove::End);
        let mut pending = None;
        let mut panel = None;

        let first = apply_multiline_completion(&mut textarea, &mut pending, &mut panel);

        assert!(first.is_none());
        assert!(pending.is_some());
        assert!(panel.is_none());

        let status = apply_multiline_completion(&mut textarea, &mut pending, &mut panel).unwrap();
        assert!(status.contains("发现"));
        let items = panel.as_ref().map(|panel| panel.items.clone()).unwrap();
        assert!(items
            .iter()
            .any(|item| item.replacement == format!("@{}", image.display())));
        assert!(items
            .iter()
            .any(|item| item.replacement == format!("@{}", other.display())));
    }
}
