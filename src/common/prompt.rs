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
                println!();
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
                println!();
                return None;
            }
            (KeyCode::Esc, _) => {
                println!();
                return None;
            }
            (KeyCode::Char(ch), _) => match ch.to_ascii_lowercase() {
                'y' => {
                    println!("y");
                    return Some(true);
                }
                'n' => {
                    println!("n");
                    return Some(false);
                }
                _ => {}
            },
            _ => {}
        }
    }
}
