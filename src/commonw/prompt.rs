use std::{
    io::{IsTerminal, Write},
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
};

static STDIN_OWNER: Mutex<()> = Mutex::new(());
static FOREGROUND_STDIN_REQUESTS: AtomicUsize = AtomicUsize::new(0);

/// stdin / termios 的进程内独占租约。前台确认先发布抢占请求，再等待后台监听器交还租约。
#[doc(hidden)]
pub struct StdinOwnerGuard {
    owner: Option<MutexGuard<'static, ()>>,
    foreground: bool,
}

impl Drop for StdinOwnerGuard {
    fn drop(&mut self) {
        // 必须先释放 stdin，再撤销抢占请求；否则后台监听器可能先恢复 cbreak，
        // 与尚未完全退出的前台 prompt 再次并发修改 termios。
        self.owner.take();
        if self.foreground {
            FOREGROUND_STDIN_REQUESTS.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

fn lock_stdin_owner() -> MutexGuard<'static, ()> {
    STDIN_OWNER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[doc(hidden)]
pub fn foreground_stdin_requested() -> bool {
    FOREGROUND_STDIN_REQUESTS.load(Ordering::SeqCst) != 0
}

/// Foreground interactions must call this before printing the prompt or reading
/// stdin. Publishing the preemption flag makes the streaming side-note listener
/// stop poll/read, restore termios, and release its stdin lease; this function
/// returns only after that handshake completes.
#[doc(hidden)]
pub fn acquire_foreground_stdin() -> StdinOwnerGuard {
    FOREGROUND_STDIN_REQUESTS.fetch_add(1, Ordering::SeqCst);
    StdinOwnerGuard {
        owner: Some(lock_stdin_owner()),
        foreground: true,
    }
}

/// 流式 side-note 监听器的后台租约。若已有前台交互请求则不再接管 stdin。
#[doc(hidden)]
pub fn acquire_background_stdin() -> Option<StdinOwnerGuard> {
    if foreground_stdin_requested() {
        return None;
    }
    let owner = lock_stdin_owner();
    if foreground_stdin_requested() {
        return None;
    }
    Some(StdinOwnerGuard {
        owner: Some(owner),
        foreground: false,
    })
}

pub fn read_line(prompt: &str) -> String {
    if !prompt.is_empty() {
        print!("{prompt}");
        let _ = std::io::stdout().flush();
    }
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).ok();
    buf.trim_end_matches(['\n', '\r']).to_string()
}

pub fn prompt_yes_or_no(prompt: &str) -> bool {
    loop {
        let s = read_line(prompt);
        match s.trim().to_lowercase().as_str() {
            "y" | "yes" => return true,
            "n" | "no" => return false,
            _ => {}
        }
    }
}

/// 高危操作确认：提示文字以红色显示，用户输入后颜色立即还原。
pub fn prompt_yes_or_no_danger(prompt: &str) -> Option<bool> {
    // \x1b[31m 红色，\x1b[0m 还原；prompt 内打印完颜色即归位，不影响后续输出。
    prompt_yes_or_no_interruptible(&format!("\x1b[31m{prompt}\x1b[0m"))
}

pub fn prompt_yes_or_no_interruptible(prompt: &str) -> Option<bool> {
    if !std::io::stdin().is_terminal() {
        return Some(prompt_yes_or_no(prompt));
    }

    // SideNoteInputGuard 与工具确认可能同时存在于一个 turn。必须先抢占 stdin 并等待
    // side-note 恢复 termios，之后才能打印提示、启用 raw mode 或读取确认键。
    let _stdin_owner = acquire_foreground_stdin();

    use crossterm::{
        event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
        terminal::{disable_raw_mode, enable_raw_mode},
    };

    if !prompt.is_empty() {
        print!("{prompt}");
        let _ = std::io::stdout().flush();
    }

    if enable_raw_mode().is_err() {
        return Some(prompt_yes_or_no(prompt));
    }

    struct RawModeGuard;
    impl Drop for RawModeGuard {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
        }
    }
    let _guard = RawModeGuard;

    loop {
        let evt = match event::read() {
            Ok(e) => e,
            Err(_) => {
                // raw mode 下 \n 不回车，必须 \r\n 才能正确换行
                let _ = write!(std::io::stdout(), "\r\n");
                let _ = std::io::stdout().flush();
                return None;
            }
        };

        let Event::Key(key) = evt else {
            continue;
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue;
        }

        match (key.code, key.modifiers) {
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                let _ = write!(std::io::stdout(), "\r\n");
                let _ = std::io::stdout().flush();
                return None;
            }
            (KeyCode::Esc, _) => {
                let _ = write!(std::io::stdout(), "\r\n");
                let _ = std::io::stdout().flush();
                return None;
            }
            (KeyCode::Char(ch), _) => match ch.to_ascii_lowercase() {
                'y' => {
                    let _ = write!(std::io::stdout(), "y\r\n");
                    let _ = std::io::stdout().flush();
                    return Some(true);
                }
                'n' => {
                    let _ = write!(std::io::stdout(), "n\r\n");
                    let _ = std::io::stdout().flush();
                    return Some(false);
                }
                _ => {}
            },
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    #[test]
    fn foreground_request_preempts_background_stdin_owner() {
        let background = acquire_background_stdin().expect("background should acquire stdin");
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let foreground = thread::spawn(move || {
            let _owner = acquire_foreground_stdin();
            let _ = acquired_tx.send(());
            let _ = release_rx.recv();
        });

        let deadline = Instant::now() + Duration::from_secs(1);
        while !foreground_stdin_requested() && Instant::now() < deadline {
            thread::yield_now();
        }
        let request_visible = foreground_stdin_requested();
        let background_reacquire_denied = request_visible && acquire_background_stdin().is_none();

        drop(background);
        let foreground_acquired = acquired_rx.recv_timeout(Duration::from_secs(1)).is_ok();
        let _ = release_tx.send(());
        let _ = foreground.join();

        assert!(request_visible, "foreground request was not published");
        assert!(background_reacquire_denied);
        assert!(
            foreground_acquired,
            "foreground did not receive stdin ownership"
        );
        assert!(!foreground_stdin_requested());
    }
}
