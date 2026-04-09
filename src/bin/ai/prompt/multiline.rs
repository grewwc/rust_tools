#[path = "multiline/completion_panel.rs"]
mod completion_panel;
#[path = "multiline/events.rs"]
mod events;
#[path = "multiline/render.rs"]
mod render;
#[path = "multiline/multiline_ui.rs"]
mod multiline_ui;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(in super::super) struct MultilineHistoryState {
    entries: Vec<String>,
    index: Option<usize>,
    draft: Option<String>,
}

impl MultilineHistoryState {
    pub(in super::super) fn new(entries: Vec<String>) -> Self {
        Self {
            entries,
            index: None,
            draft: None,
        }
    }

    pub(in super::super) fn previous(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }

        let next_index = match self.index {
            Some(0) => return None,
            Some(index) => index - 1,
            None => {
                self.draft = Some(current.to_string());
                self.entries.len() - 1
            }
        };
        self.index = Some(next_index);
        self.entries.get(next_index).cloned()
    }

    pub(in super::super) fn next(&mut self) -> Option<String> {
        let index = self.index?;
        if index + 1 < self.entries.len() {
            self.index = Some(index + 1);
            return self.entries.get(index + 1).cloned();
        }

        self.index = None;
        Some(self.draft.take().unwrap_or_default())
    }
}
