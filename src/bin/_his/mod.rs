use rust_tools::cmd::run::run_cmd;

pub fn current_branch() -> String {
    let s = match run_cmd("git branch | grep '*'") {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    let mut ss = s.trim();
    if let Some(idx) = ss.find(' ') {
        ss = &ss[idx + 1..];
        return ss.to_string();
    }
    s
}
