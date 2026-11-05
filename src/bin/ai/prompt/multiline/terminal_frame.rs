use std::io::{self, Write};

/// Paint transient rows without terminal-owned wrapping, including while the
/// emulator has resized before the PTY reports its new width. Keep the frame
/// synchronized until its cursor group and anchor have also been restored.
pub(in crate::ai) struct SingleRowFrame<W: Write> {
    output: W,
    finished: bool,
}

impl<W: Write> SingleRowFrame<W> {
    /// Measure the emulator's edge without relying on a lagging PTY winsize.
    /// The caller owns terminal input and accounts for unanswered CPR replies.
    pub(in crate::ai) fn measure_columns(
        output: &mut W,
        read_cursor_column: impl FnOnce() -> io::Result<u16>,
    ) -> io::Result<u16> {
        let result = (|| {
            write!(output, "\x1b7\x1b[65535G")?;
            output.flush()?;
            read_cursor_column().map(|column| column.saturating_add(1).max(1))
        })();
        write!(output, "\x1b8")?;
        output.flush()?;
        result
    }

    pub(in crate::ai) fn begin(output: W) -> io::Result<Self> {
        let mut frame = Self {
            output,
            finished: false,
        };
        write!(frame.output, "\x1b[?2026h\x1b[?7l")?;
        frame.output.flush()?;
        Ok(frame)
    }

    pub(in crate::ai) fn output(&mut self) -> &mut W {
        &mut self.output
    }

    /// Only blank edge writes may wrap while linking the transient rows.
    pub(in crate::ai) fn finish_content(&mut self) -> io::Result<()> {
        write!(self.output, "\r\x1b[?7h")
    }

    pub(in crate::ai) fn start_content(&mut self) -> io::Result<()> {
        write!(self.output, "\r\x1b[?7l")
    }

    /// Erase the protected rows before a changed layout paints its new caret.
    /// ECH keeps their links intact, including across a resize during the wipe.
    pub(in crate::ai) fn clear_cursor_group(&mut self, rows: usize) -> io::Result<()> {
        for row in 0..rows {
            write!(self.output, "\r\x1b[65535X")?;
            if row + 1 < rows {
                write!(self.output, "\x1b[1B")?;
            }
        }
        if rows > 1 {
            write!(self.output, "\x1b[{}A\r", rows - 1)?;
        }
        Ok(())
    }

    /// Link reserved rows through their blank edge columns before painting. A saturating column
    /// address reaches the emulator's actual edge even before PTY winsize
    /// catches up. Save/restore avoids depending on whether an edge write
    /// advanced the cursor while a resize interrupted it.
    pub(in crate::ai) fn prepare_cursor_group(&mut self, rows: usize) -> io::Result<()> {
        self.finish_content()?;
        write!(self.output, "\x1b[0m")?;
        // The first pass establishes the links. Once linked, the group survives a
        // resize during the second pass; that pass also repairs an edge write
        // interrupted before its wrap in the first pass.
        for _ in 0..2 {
            for _ in 1..rows {
                write!(self.output, "\x1b7\x1b[65535G  \x1b8\x1b[1B")?;
            }
            if rows > 1 {
                write!(self.output, "\x1b[{}A\r", rows - 1)?;
            }
        }
        self.start_content()
    }

    pub(in crate::ai) fn finish(mut self) -> io::Result<()> {
        write!(self.output, "\x1b[?7h\x1b[?2026l")?;
        self.output.flush()?;
        self.finished = true;
        Ok(())
    }
}

impl<W: Write> Drop for SingleRowFrame<W> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = write!(self.output, "\x1b[?7h\x1b[?2026l");
            let _ = self.output.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    #[test]
    fn emulator_columns_restore_the_cursor_after_a_query_error() {
        let mut bytes = Vec::new();
        let error = SingleRowFrame::measure_columns(&mut bytes, || {
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "injected CPR timeout",
            ))
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert_eq!(bytes, b"\x1b7\x1b[65535G\x1b8");

        bytes.clear();
        assert_eq!(
            SingleRowFrame::measure_columns(&mut bytes, || Ok(11)).unwrap(),
            12
        );
        assert_eq!(bytes, b"\x1b7\x1b[65535G\x1b8");
    }

    #[test]
    fn single_row_frame_restores_terminal_modes_after_a_write_error() {
        struct FailingOutput(Rc<RefCell<Vec<u8>>>);
        impl Write for FailingOutput {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if bytes == b"task" {
                    return Err(io::Error::other("injected write failure"));
                }
                self.0.borrow_mut().extend_from_slice(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let bytes = Rc::new(RefCell::new(Vec::new()));
        {
            let mut frame = SingleRowFrame::begin(FailingOutput(bytes.clone())).unwrap();
            assert!(frame.output().write_all(b"task").is_err());
        }
        assert_eq!(
            bytes.borrow().as_slice(),
            b"\x1b[?2026h\x1b[?7l\x1b[?7h\x1b[?2026l"
        );
    }
}
