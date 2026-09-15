use std::path::PathBuf;

use rust_tools::commonw::utils::expanduser;

const PREV_FILE: &str = "~/.go_tools_previous_op.txt";

pub fn write_previous_operation(op: &str) {
    let path = PathBuf::from(expanduser(PREV_FILE).as_ref());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(path, op).ok();
}

pub fn read_previous_operation() -> String {
    let path = PathBuf::from(expanduser(PREV_FILE).as_ref());
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

pub fn choose_from_list(items: &[String]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    if items.len() == 1 {
        return Some(items[0].clone());
    }
    for (i, item) in items.iter().enumerate() {
        println!("{:>3}. {}", i + 1, item);
    }
    print!("\ninput the number: ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).ok();
    let n = buf.trim().parse::<usize>().ok()?;
    if n == 0 || n > items.len() {
        return None;
    }
    Some(items[n - 1].clone())
}
