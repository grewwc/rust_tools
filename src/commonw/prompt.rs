use std::io::{IsTerminal, Write};

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
