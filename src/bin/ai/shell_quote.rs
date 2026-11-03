/// Quote a value for use inside a POSIX shell single-quoted context.
pub(crate) fn shell_single_quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('\'');
    for ch in value.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

#[cfg(test)]
mod tests {
    use super::shell_single_quote;

    #[test]
    fn wraps_plain_value() {
        assert_eq!(shell_single_quote("on_turn_start"), "'on_turn_start'");
        assert_eq!(shell_single_quote(""), "''");
    }

    #[test]
    fn escapes_embedded_quote() {
        assert_eq!(shell_single_quote("a'b"), "'a'\\''b'");
    }

    #[test]
    fn neutralizes_injection() {
        assert_eq!(shell_single_quote("x; rm -rf /"), "'x; rm -rf /'");
    }
}
