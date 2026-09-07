use std::collections::VecDeque;

const LIMIT: usize = 256 * 1024;

#[derive(Default)]
enum Escape {
    #[default]
    Text,
    Start,
    Csi(Vec<u8>),
    String { escaped: bool },
}

/// Replay cannot ask the terminal questions: stale cursor/device replies would
/// become input to the still-running worker. Live output bypasses this filter.
#[derive(Default)]
pub(super) struct Replay {
    bytes: VecDeque<u8>,
    escape: Escape,
    truncated: bool,
}

impl Replay {
    pub(super) fn push(&mut self, input: &[u8]) {
        for &byte in input {
            let mut output = Vec::new();
            self.escape = match std::mem::take(&mut self.escape) {
                Escape::Text => match byte {
                    0x1b => Escape::Start,
                    b'\t' | b'\n' | b'\r' | 0x08 | 0x20..=0x7e | 0x80..=0xff => {
                        output.push(byte);
                        Escape::Text
                    }
                    _ => Escape::Text,
                },
                Escape::Start => match byte {
                    b'[' => Escape::Csi(Vec::new()),
                    b']' | b'P' | b'_' | b'^' | b'X' => Escape::String { escaped: false },
                    b'7' | b'8' => { output.extend_from_slice(&[0x1b, byte]); Escape::Text }
                    _ => Escape::Text,
                },
                Escape::Csi(mut sequence) => {
                    if (0x40..=0x7e).contains(&byte) {
                        // Keep display edits, never queries, clipboard operations,
                        // window controls or keyboard/alternate-screen modes.
                        if b"mABCDEFGHJKLMPSTXdfsur".contains(&byte) {
                            output.extend_from_slice(b"\x1b[");
                            output.append(&mut sequence);
                            output.push(byte);
                        }
                        Escape::Text
                    } else if sequence.len() < 128 && (0x20..=0x3f).contains(&byte) {
                        sequence.push(byte);
                        Escape::Csi(sequence)
                    } else {
                        Escape::Text
                    }
                }
                Escape::String { escaped } => {
                    if byte == 0x07 || (escaped && byte == b'\\') { Escape::Text }
                    else { Escape::String { escaped: byte == 0x1b } }
                }
            };
            for byte in output {
                if self.bytes.len() == LIMIT { self.bytes.pop_front(); self.truncated = true; }
                self.bytes.push_back(byte);
            }
        }
    }

    pub(super) fn snapshot(&self) -> Vec<u8> {
        let bytes: Vec<u8> = self.bytes.iter().copied().collect();
        if !self.truncated { return bytes; }
        // Start at a line boundary rather than in a split UTF-8/CSI sequence.
        let start = bytes.iter().position(|b| *b == b'\n').map_or(bytes.len(), |i| i + 1);
        let mut output = b"[Earlier terminal output omitted; conversation history is unchanged.]\r\n".to_vec();
        output.extend_from_slice(&bytes[start..]);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_filters_split_queries_and_strings_but_keeps_display() {
        let mut replay = Replay::default();
        replay.push(b"hello\x1b[");
        replay.push(b"6n\x1b]52;c;secret\x1b");
        replay.push(b"\\\x1bPpayload\x1b\\\x1b[31mworld\x1b[0m\r\n");
        assert_eq!(replay.snapshot(), b"hello\x1b[31mworld\x1b[0m\r\n");
    }

    #[test]
    fn replay_is_bounded_and_starts_on_a_complete_line() {
        let mut replay = Replay::default();
        replay.push(&vec![b'x'; LIMIT + 300]);
        replay.push("\n中文\n".as_bytes());
        assert_eq!(replay.bytes.len(), LIMIT);
        let output = String::from_utf8(replay.snapshot()).unwrap();
        assert!(output.starts_with("[Earlier terminal output omitted"));
        assert!(output.ends_with("中文\n"));
    }
}