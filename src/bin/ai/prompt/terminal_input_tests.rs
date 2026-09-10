//! Real-PTY regressions for cursor replies delayed beyond the DSR timeout.
//! Each editor runs in a child test process to isolate crossterm's global reader.

use std::{
    fs::{self, File},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::process::CommandExt,
    },
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

const CHILD_TEST: &str = "ai::prompt::terminal_input_tests::prompt_editor_pty_child";
const ROOT_ENV: &str = "RUST_TOOLS_PROMPT_PTY_TEST_ROOT";

struct PtyEditor {
    child: Child,
    master: File,
    output: Vec<u8>,
    root: PathBuf,
}

impl PtyEditor {
    fn start(rounds: usize) -> Self {
        let root = std::env::temp_dir().join(format!("prompt-cpr-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let mut master = -1;
        let mut slave = -1;
        let mut size = libc::winsize {
            ws_row: 24,
            ws_col: 80,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // Both returned descriptors become uniquely owned Files immediately.
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut size,
                )
            },
            0,
            "openpty: {}",
            io::Error::last_os_error()
        );
        let master = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", CHILD_TEST, "--nocapture", "--test-threads=1"])
            .env(ROOT_ENV, &root)
            .env("RUST_TOOLS_PROMPT_PTY_TEST_ROUNDS", rounds.to_string())
            .env("TERM", "xterm-256color")
            .env_remove("SSH_CONNECTION")
            .env_remove("SSH_CLIENT")
            .env_remove("SSH_TTY")
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave));
        // Only async-signal-safe syscalls run between fork and exec.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() == -1 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().unwrap();
        Self {
            child,
            master,
            output: Vec::new(),
            root,
        }
    }

    fn pump(&mut self, duration: Duration) {
        let deadline = Instant::now() + duration;
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            let mut fd = libc::pollfd {
                fd: self.master.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let result = unsafe { libc::poll(&mut fd, 1, remaining.as_millis().max(1) as i32) };
            if result < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                panic!("poll: {error}");
            }
            if result == 0 {
                break;
            }
            let mut buffer = [0; 8192];
            match self.master.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => self.output.extend_from_slice(&buffer[..n]),
                Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => panic!("PTY read: {error}"),
            }
        }
    }

    fn contains(&self, text: &str) -> bool {
        self.output
            .windows(text.len())
            .any(|bytes| bytes == text.as_bytes())
    }

    fn wait_for(&mut self, text: &str) {
        let deadline = Instant::now() + Duration::from_secs(8);
        while !self.contains(text) && Instant::now() < deadline {
            self.pump(Duration::from_millis(20));
        }
        assert!(
            self.contains(text),
            "missing {text:?}: {}",
            String::from_utf8_lossy(&self.output)
        );
    }

    fn send(&mut self, bytes: &[u8]) {
        self.master.write_all(bytes).unwrap();
    }

    fn submit(&mut self) {
        self.send(b"\x1bOQ"); // F2 submits; an unmodified Enter inserts a newline.
    }

    fn ready(&mut self, round: usize) {
        self.wait_for(&format!("PTY_READY_{round}"));
    }

    fn result(&mut self, round: usize, text: &str) {
        self.wait_for(&format!(
            "PTY_RESULT_{round}={}",
            serde_json::to_string(text).unwrap()
        ));
    }

    fn still_editing(&mut self, round: usize) {
        self.pump(Duration::from_millis(80));
        assert!(
            !self.contains(&format!("PTY_RESULT_{round}=")),
            "reply submitted input: {}",
            String::from_utf8_lossy(&self.output)
        );
    }

    fn query_count(&self) -> usize {
        self.output
            .windows(4)
            .filter(|bytes| *bytes == b"\x1b[6n")
            .count()
    }

    fn assert_query_count(&self, expected: usize) {
        assert_eq!(
            self.query_count(),
            expected,
            "unexpected DSR query count: {}",
            String::from_utf8_lossy(&self.output)
        );
    }

    /// Verifies the child exited cleanly. `degraded` says whether the session
    /// ever fell back to the alternate screen: the recovery probe can have left
    /// it again, so only the enter/leave balance and that lower bound are
    /// asserted.
    fn finish(&mut self, degraded: bool) {
        self.wait_for("PTY_CHILD_DONE");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "child exited {status}: {}",
                    String::from_utf8_lossy(&self.output)
                );
                break;
            }
            assert!(Instant::now() < deadline, "child did not exit");
            self.pump(Duration::from_millis(20));
        }
        // Every prompt retries the main screen (recovery probe) while the
        // editor is degraded, so only the lower bound is stable.
        assert!(
            self.query_count() >= 1,
            "the editor never queried the cursor: {}",
            String::from_utf8_lossy(&self.output)
        );
        let enters = self
            .output
            .windows(8)
            .filter(|bytes| *bytes == b"\x1b[?1049h")
            .count();
        let leaves = self
            .output
            .windows(8)
            .filter(|bytes| *bytes == b"\x1b[?1049l")
            .count();
        assert_eq!(enters, leaves, "alternate screen was not restored");
        assert_eq!(enters > 0, degraded);
        assert!(
            !self.contains("^[[13;1R"),
            "cooked-mode echo leaked a cursor reply"
        );
    }
}

impl Drop for PtyEditor {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn prompt_editor_pty_child() {
    let Some(root) = std::env::var_os(ROOT_ENV) else {
        return;
    };
    let root = PathBuf::from(root);
    let rounds: usize = std::env::var("RUST_TOOLS_PROMPT_PTY_TEST_ROUNDS")
        .unwrap()
        .parse()
        .unwrap();
    let mut editor = super::PromptEditor::new("pty-test", &root.join("history"));
    // Never append the fixture input to the user's real REPL history.
    editor.history_path = root.join("repl-history");
    // Test text paste without reading or changing the system clipboard. A file
    // makes image-directory creation fail before the clipboard is accessed.
    editor.session_image_dir = root.join("disabled-image-directory");
    fs::write(&editor.session_image_dir, b"").unwrap();
    editor.set_prefill("draft");
    for round in 0..rounds {
        let (sender, receiver) = std::sync::mpsc::channel();
        editor.set_first_render_notifier(sender);
        let ready = std::thread::spawn(move || {
            receiver.recv_timeout(Duration::from_secs(6)).unwrap();
            let mut termios = std::mem::MaybeUninit::<libc::termios>::uninit();
            assert_eq!(unsafe { libc::tcgetattr(0, termios.as_mut_ptr()) }, 0);
            let termios = unsafe { termios.assume_init() };
            assert_eq!(
                termios.c_lflag & (libc::ECHO | libc::ICANON),
                0,
                "editor left raw mode"
            );
            println!("\r\nPTY_READY_{round}\r");
        });
        let result = editor.read_multi_line();
        ready.join().unwrap();
        assert!(
            !crossterm::terminal::is_raw_mode_enabled().unwrap(),
            "raw mode not restored"
        );
        match result {
            Ok(text) => println!(
                "PTY_RESULT_{round}={}",
                serde_json::to_string(&text).unwrap()
            ),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                println!("PTY_INTERRUPTED_{round}")
            }
            Err(error) => panic!("editor failed: {error}"),
        }
    }
    println!("PTY_CHILD_DONE");
}

#[test]
fn late_cursor_reply_keeps_typeahead_editing_paste_and_escape() {
    let mut pty = PtyEditor::start(5);
    pty.wait_for("\x1b[6n");
    pty.send(b"\x1b[FX"); // Move to End and type while the cursor query is waiting.
    pty.ready(0); // Deliberately withhold the CPR until the query times out.
    pty.send(b"\x1b[13;1R");
    pty.still_editing(0);
    pty.send(b"\x7fok");
    pty.submit();
    pty.result(0, "draftok");
    pty.ready(1);
    pty.send(b"[13;1R"); // Literal text without ESC must remain user input.
    pty.submit();
    pty.result(1, "[13;1R");
    pty.ready(2);
    pty.send(b"\x1b[200~hello\nworld\x1b[201~");
    pty.submit();
    pty.result(2, "hello\nworld");
    pty.ready(3);
    pty.send(b"last");
    pty.pump(Duration::from_millis(30));
    pty.send(b"\x1b"); // A real Escape must still submit after the bounded grace.
    pty.result(3, "last");
    pty.ready(4);
    pty.send(b"\x03");
    pty.wait_for("PTY_INTERRUPTED_4");
    pty.finish(true);
}

#[test]
fn late_cursor_reply_split_after_escape_does_not_submit_prefill() {
    let mut pty = PtyEditor::start(3);
    pty.wait_for("\x1b[6n");
    pty.ready(0);
    pty.send(b"\x1b");
    pty.pump(Duration::from_millis(60));
    assert!(
        !pty.contains("PTY_RESULT_0="),
        "isolated ESC prematurely submitted prefill"
    );
    pty.send(b"[13;1R");
    pty.still_editing(0);
    pty.submit();
    pty.result(0, "draft");
    pty.ready(1);
    pty.send(b"first");
    pty.pump(Duration::from_millis(30));
    pty.send(b"\x1b");
    pty.pump(Duration::from_millis(60));
    pty.send(b"next"); // Non-CPR lookahead must survive the Escape submission.
    pty.result(1, "first");
    pty.ready(2);
    pty.submit();
    pty.result(2, "next");
    pty.finish(true);
}

#[test]
fn late_cursor_reply_split_inside_csi_survives_long_fragment_delay() {
    let mut pty = PtyEditor::start(1);
    pty.wait_for("\x1b[6n");
    pty.ready(0);
    pty.send(b"\x1b[13;");
    pty.pump(Duration::from_millis(400));
    assert!(!pty.contains("PTY_RESULT_0="));
    pty.send(b"1R");
    pty.still_editing(0);
    pty.submit();
    pty.result(0, "draft");
    pty.finish(true);
}

#[test]
fn responsive_cursor_query_keeps_inline_editor() {
    let mut pty = PtyEditor::start(1);
    pty.wait_for("\x1b[6n");
    pty.send(b"\x1b[13;1R");
    pty.ready(0);
    pty.submit();
    pty.result(0, "draft");
    pty.finish(false);
        pty.assert_query_count(1);
    }

    #[test]
    fn cursor_query_retry_keeps_editor_inline_after_one_stalled_round_trip() {
        let mut pty = PtyEditor::start(1);
        pty.wait_for("\x1b[6n");
        // Withhold the first reply so the query times out, then answer the retry:
        // one stalled round-trip must not cost the whole session its transcript.
        let deadline = Instant::now() + Duration::from_secs(4);
        while pty.query_count() < 2 && Instant::now() < deadline {
            pty.pump(Duration::from_millis(20));
        }
        assert_eq!(pty.query_count(), 2, "the timed-out query was not retried");
        pty.send(b"\x1b[13;1R");
        pty.ready(0);
        assert!(
            !pty.contains("\x1b[?1049h"),
            "a retried query must keep the inline editor: {}",
            String::from_utf8_lossy(&pty.output)
        );
        pty.submit();
        pty.result(0, "draft");
        pty.finish(false);
        pty.assert_query_count(2);
}

    #[test]
    fn recovery_probe_restores_the_main_screen_after_the_link_recovers() {
        let mut pty = PtyEditor::start(1);
        pty.wait_for("\x1b[6n");
        // Withhold the reply so the query times out, then answer the recovery
        // probe that follows the fallback: the editor must leave the alternate
        // screen and put the inline box back under the main transcript.
        pty.ready(0);
        let deadline = Instant::now() + Duration::from_secs(8);
        while pty.query_count() < 3 && Instant::now() < deadline {
            pty.pump(Duration::from_millis(20));
        }
        assert_eq!(
            pty.query_count(),
            3,
            "no recovery probe followed the fallback: {}",
            String::from_utf8_lossy(&pty.output)
        );
        assert!(pty.contains("\x1b[?1049h"), "the editor never degraded");
        pty.send(b"\x1b[13;1R");
        pty.wait_for("\x1b[?1049l");
        pty.still_editing(0);
        pty.submit();
        pty.result(0, "draft");
        pty.finish(true);
    }
