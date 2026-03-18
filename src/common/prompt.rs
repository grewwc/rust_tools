use std::io::Write;

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

