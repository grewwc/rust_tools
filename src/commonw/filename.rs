pub fn add_suffix(fname: &str, suffix: &str, cond: impl Fn() -> bool) -> String {
    let mut result = String::from(fname);
    if cond() {
        result.push_str(suffix);
    }
    result
}
