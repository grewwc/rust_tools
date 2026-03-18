use crate::strw::split::split_no_empty;

pub fn wrap(s: &str, width: usize, indent: usize, delimiter: &str) -> String {
    let mut res = String::new();
    for line in split_no_empty(s, "\n") {
        res.push_str(&wrap_single_line(line, width, indent, delimiter));
        res.push('\n');
    }
    res
}

fn wrap_single_line(s: &str, width: usize, indent: usize, delimiter: &str) -> String {
    let words = split_no_empty(s, " ");

    let mut lines: Vec<String> = Vec::new();
    let mut cur_line: Vec<String> = Vec::new();
    let mut cursor: usize = 0;

    for word in words {
        let word = word.trim();
        if word.is_empty() {
            continue;
        }
        let word_len = get_num_chars(word);

        if cursor + word_len > width {
            let joined = cur_line.concat();
            let joined = joined.trim_end_matches(' ').to_string();
            if !joined.is_empty() {
                lines.push(joined);
            }
            cur_line.clear();

            if word_len > width {
                cursor = 0;
                lines.push(word.to_string());
                continue;
            }

            cursor = word_len + delimiter.len();
            cur_line.push(format!("{word}{delimiter}"));
        } else {
            cursor += word_len + delimiter.len();
            cur_line.push(format!("{word}{delimiter}"));
        }
    }

    if !cur_line.is_empty() {
        let joined = cur_line.concat();
        let joined = joined.trim_end_matches(' ').to_string();
        if !joined.is_empty() {
            lines.push(joined);
        }
    }

    let prefix = " ".repeat(indent);
    let mut out = String::new();
    for line in lines {
        out.push_str(&prefix);
        out.push_str(&line);
        out.push('\n');
    }
    out.trim_end_matches('\n').to_string()
}

fn get_num_chars(word: &str) -> usize {
    let mut n = 0usize;
    for ch in word.chars() {
        n += 1;
        if (ch as u32) > 128 {
            n += 1;
        }
    }
    n
}

pub fn format_int64(mut val: i64) -> String {
    let mut suffix = "";
    let mut decimal: i64 = 0;
    let _1k: i64 = 1 << 10;
    let _1m: i64 = _1k * _1k;
    let _1g: i64 = _1m * _1k;
    let _1t: i64 = _1g * _1k;
    let _1p: i64 = _1t * _1k;

    if val < _1k {
    } else if val < _1m {
        suffix = "K";
        decimal = val % _1k;
        val /= _1k;
    } else if val < _1g {
        suffix = "M";
        decimal = val % _1m;
        val /= _1m;
    } else if val < _1t {
        suffix = "G";
        decimal = val % _1g;
        val /= _1g;
    } else if val < _1p {
        suffix = "T";
        decimal = val % _1t;
        val /= _1t;
    } else {
        suffix = "P";
        decimal = val % _1p;
        val /= _1p;
    }

    if suffix.is_empty() {
        format!("{val}")
    } else {
        format!("{val}.{decimal}{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_basic() {
        let s = "hello world test";
        let wrapped = wrap(s, 6, 0, " ");
        assert_eq!(wrapped, "hello\nworld\ntest\n");
    }

    #[test]
    fn test_wrap_indent() {
        let s = "hello world";
        let wrapped = wrap(s, 5, 2, " ");
        assert_eq!(wrapped, "  hello\n  world\n");
    }

    #[test]
    fn test_format_int64() {
        assert_eq!(format_int64(1), "1");
        assert_eq!(format_int64(1024), "1.0K");
        assert_eq!(format_int64(1536), "1.512K");
    }
}
