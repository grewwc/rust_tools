//! Resolve the client's captured ANSI output for observer-row regressions.

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

struct CapturedOutput(Rc<RefCell<Vec<u8>>>);

impl Write for CapturedOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn render(
    resp: reqwest::blocking::Response,
    tty: bool,
    widths: &[usize],
) -> (String, Vec<String>) {
    let last_width = *widths.last().expect("width script");
    let captured = Rc::new(RefCell::new(Vec::new()));
    let mut out = CapturedOutput(Rc::clone(&captured));
    let mut widths = widths.iter().copied();
    let mut probes = Vec::new();
    super::render_turn_stream_to_with_width(resp, &mut out, tty, None, || {
        let width = widths.next().unwrap_or(last_width);
        probes.push((captured.borrow().len(), width));
        width
    })
    .expect("render canned SSE");
    let raw = String::from_utf8(std::mem::take(&mut *captured.borrow_mut())).expect("utf8");
    if !tty || probes.iter().all(|(_, width)| *width == 0) {
        return (raw, Vec::new());
    }
    let mut terminal = Terminal::new(probes[0].1);
    let mut consumed = 0;
    for (at, width) in probes {
        terminal.feed(&raw[consumed..at]);
        terminal.resize(width);
        consumed = at;
    }
    terminal.feed(&raw[consumed..]);
    (raw, terminal.screen())
}

/// Models only the writer's SGR, CR/LF, CUU and erase sequences. Resizes reflow
/// the active logical row; committed scrollback stays captured for assertions.
struct Terminal {
    width: usize,
    rows: Vec<Vec<char>>,
    row: usize,
    col: usize,
    active_start: usize,
}

impl Terminal {
    fn new(width: usize) -> Self {
        Self {
            width: width.max(1),
            rows: vec![Vec::new()],
            row: 0,
            col: 0,
            active_start: 0,
        }
    }

    fn ensure_row(&mut self) {
        self.rows
            .resize_with(self.rows.len().max(self.row + 1), Vec::new);
    }

    fn put(&mut self, ch: char) {
        let width = crate::ai::stream::terminal_cell_width(ch);
        if width == 0 {
            return;
        }
        if self.col + width > self.width {
            self.row += 1;
            self.col = 0;
        }
        self.ensure_row();
        let row = &mut self.rows[self.row];
        row.resize(row.len().max(self.col + width), ' ');
        row[self.col] = ch;
        for col in self.col + 1..self.col + width {
            row[col] = '\0';
        }
        self.col += width;
    }

    fn resize(&mut self, width: usize) {
        let width = width.max(1);
        if width == self.width {
            return;
        }
        let mut active = Vec::new();
        for index in self.active_start..=self.row {
            let row = &self.rows[index];
            let end = if index == self.row {
                self.col.min(row.len())
            } else {
                row.len()
            };
            active.extend(row[..end].iter().copied().filter(|ch| *ch != '\0'));
        }
        self.rows.truncate(self.active_start);
        self.row = self.active_start;
        self.col = 0;
        self.width = width;
        self.ensure_row();
        for ch in active {
            self.put(ch);
        }
    }

    fn feed(&mut self, text: &str) {
        let mut chars = text.chars();
        while let Some(ch) = chars.next() {
            match ch {
                '\r' => self.col = 0,
                '\n' => {
                    self.row += 1;
                    self.col = 0;
                    self.active_start = self.row;
                    self.ensure_row();
                }
                '\x1b' => {
                    assert_eq!(chars.next(), Some('['), "unexpected terminal escape");
                    let mut params = String::new();
                    for final_ch in chars.by_ref() {
                        if !('@'..='~').contains(&final_ch) {
                            params.push(final_ch);
                            continue;
                        }
                        let n = params.parse::<usize>().unwrap_or(0);
                        match final_ch {
                            'm' => {}
                            'A' => {
                                self.row = self.row.saturating_sub(n.max(1));
                                self.active_start = self.active_start.min(self.row);
                            }
                            'K' if n == 2 => self.rows[self.row].clear(),
                            'K' => self.rows[self.row].truncate(self.col),
                            'J' if n == 0 => {
                                self.rows[self.row].truncate(self.col);
                                self.rows.truncate(self.row + 1);
                            }
                            _ => panic!("unsupported terminal sequence: {params}{final_ch}"),
                        }
                        break;
                    }
                }
                ch if ch.is_control() => panic!("unexpected terminal control: {ch:?}"),
                ch => self.put(ch),
            }
        }
    }

    fn screen(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|row| {
                row.iter()
                    .filter(|ch| **ch != '\0')
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }
}
