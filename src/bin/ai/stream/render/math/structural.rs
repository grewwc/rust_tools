/// 结构性 TeX 命令：\frac、\sqrt、sizing、括号包裹等。
use super::symbols::is_control_word_boundary;

/// 跳过 `\left` / `\right` 等 sizing 命令前缀。
pub(super) fn strip_sizing_commands(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < s.len() {
        if s[i..].starts_with("\\left") && is_control_word_boundary(s, i + "\\left".len()) {
            i += "\\left".len();
            continue;
        }
        if s[i..].starts_with("\\right") && is_control_word_boundary(s, i + "\\right".len()) {
            i += "\\right".len();
            continue;
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// 读取从 `start` 位置开始的 `{...}` 分组内容（不含外层花括号）。
pub(super) fn read_group_braced(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    if start >= bytes.len() || bytes[start] != b'{' {
        return None;
    }
    let mut i = start + 1;
    let mut depth = 1usize;
    let mut out = String::new();
    while i < bytes.len() {
        let ch = match s.get(i..) {
            Some(rest) => match rest.chars().next() {
                Some(ch) => ch,
                None => break,
            },
            None => break,
        };
        i += ch.len_utf8();
        match ch {
            '{' => {
                depth += 1;
                out.push(ch);
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some((out, i));
                }
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    None
}

/// 读取从 `start` 位置开始的 `[...]` 分组内容（不含外层方括号）。
pub(super) fn read_group_bracketed(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    if start >= bytes.len() || bytes[start] != b'[' {
        return None;
    }
    let mut i = start + 1;
    let mut depth = 1usize;
    let mut out = String::new();
    while i < bytes.len() {
        let ch = match s.get(i..) {
            Some(rest) => match rest.chars().next() {
                Some(ch) => ch,
                None => break,
            },
            None => break,
        };
        i += ch.len_utf8();
        match ch {
            '[' => {
                depth += 1;
                out.push(ch);
            }
            ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some((out, i));
                }
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    None
}

/// 判断分组内容是否需要包裹括号（含运算符、负数、多字符等）。
pub(super) fn needs_parens(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    if s.starts_with('-') {
        return true;
    }
    if s.chars().count() <= 1 {
        return false;
    }
    for ch in s.chars() {
        if ch.is_whitespace() {
            return true;
        }
        if matches!(
            ch,
            '+' | '-' | '*' | '/' | '=' | '±' | '∓' | '×' | '·' | '÷' | '→' | '←' | '↔'
        ) {
            return true;
        }
    }
    false
}

/// 按需将内容包裹括号。
pub(super) fn wrap_parens(s: &str) -> String {
    let s = s.trim();
    if needs_parens(s) {
        format!("({s})")
    } else {
        s.to_string()
    }
}

/// 递归处理 \frac、\sqrt 等结构性命令。
pub(super) fn replace_structural_tex(mut s: String) -> String {
    let mut changed = true;
    while changed {
        changed = false;
        let bytes = s.as_bytes();
        let mut out = String::with_capacity(s.len());
        let mut i = 0usize;
        while i < bytes.len() {
            if s[i..].starts_with("\\frac") {
                let mut j = i + "\\frac".len();
                while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                    j += 1;
                }
                if let Some((num, j2)) = read_group_braced(&s, j) {
                    let mut k = j2;
                    while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t') {
                        k += 1;
                    }
                    if let Some((den, k2)) = read_group_braced(&s, k) {
                        let num = replace_structural_tex(num);
                        let den = replace_structural_tex(den);
                        let num = wrap_parens(&num);
                        let den = wrap_parens(&den);
                        out.push_str(&format!("{num}/{den}"));
                        i = k2;
                        changed = true;
                        continue;
                    }
                }
            }
            if s[i..].starts_with("\\sqrt") {
                let mut j = i + "\\sqrt".len();
                let mut root_index = None;
                while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                    j += 1;
                }
                if j < bytes.len()
                    && bytes[j] == b'['
                    && let Some((index, j2)) = read_group_bracketed(&s, j)
                {
                    root_index = Some(replace_structural_tex(index).trim().to_string());
                    j = j2;
                    while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                        j += 1;
                    }
                }
                if let Some((rad, j2)) = read_group_braced(&s, j) {
                    let rad = replace_structural_tex(rad);
                    let rad = rad.trim();
                    match root_index.as_deref() {
                        Some("3") => out.push_str(&format!("∛({rad})")),
                        Some("4") => out.push_str(&format!("∜({rad})")),
                        Some(index) if !index.is_empty() => {
                            out.push_str(&format!("√[{index}]({rad})"));
                        }
                        _ => out.push_str(&format!("√({rad})")),
                    }
                    i = j2;
                    changed = true;
                    continue;
                }
            }
            let ch = s[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
        s = out;
    }
    s
}
