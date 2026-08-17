/// 上标 / 下标 Unicode 转换。
use super::structural::read_group_braced;

pub(super) fn apply_super_subscripts(s: &str) -> String {
    fn map_sup(ch: char) -> Option<char> {
        match ch {
            '0' => Some('⁰'),
            '1' => Some('¹'),
            '2' => Some('²'),
            '3' => Some('³'),
            '4' => Some('⁴'),
            '5' => Some('⁵'),
            '6' => Some('⁶'),
            '7' => Some('⁷'),
            '8' => Some('⁸'),
            '9' => Some('⁹'),
            '+' => Some('⁺'),
            '-' => Some('⁻'),
            '=' => Some('⁼'),
            '(' => Some('⁽'),
            ')' => Some('⁾'),
            'n' => Some('ⁿ'),
            'i' => Some('ⁱ'),
            _ => None,
        }
    }

    fn map_sub(ch: char) -> Option<char> {
        match ch {
            '0' => Some('₀'),
            '1' => Some('₁'),
            '2' => Some('₂'),
            '3' => Some('₃'),
            '4' => Some('₄'),
            '5' => Some('₅'),
            '6' => Some('₆'),
            '7' => Some('₇'),
            '8' => Some('₈'),
            '9' => Some('₉'),
            '+' => Some('₊'),
            '-' => Some('₋'),
            '=' => Some('₌'),
            '(' => Some('₍'),
            ')' => Some('₎'),
            'a' => Some('ₐ'),
            'e' => Some('ₑ'),
            'h' => Some('ₕ'),
            'i' => Some('ᵢ'),
            'j' => Some('ⱼ'),
            'k' => Some('ₖ'),
            'l' => Some('ₗ'),
            'm' => Some('ₘ'),
            'n' => Some('ₙ'),
            'o' => Some('ₒ'),
            'p' => Some('ₚ'),
            'r' => Some('ᵣ'),
            's' => Some('ₛ'),
            't' => Some('ₜ'),
            'u' => Some('ᵤ'),
            'v' => Some('ᵥ'),
            'x' => Some('ₓ'),
            _ => None,
        }
    }

    fn convert_group(group: &str, sup: bool) -> Option<String> {
        let mut out = String::new();
        for ch in group.chars() {
            let mapped = if sup { map_sup(ch) } else { map_sub(ch) }?;
            out.push(mapped);
        }
        Some(out)
    }

    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let ch = s[i..].chars().next().unwrap();
        if ch == '^' || ch == '_' {
            let sup = ch == '^';
            i += ch.len_utf8();
            if i >= bytes.len() {
                out.push(ch);
                break;
            }
            if bytes[i] == b'{'
                && let Some((group, next)) = read_group_braced(s, i)
            {
                if let Some(converted) = convert_group(group.trim(), sup) {
                    out.push_str(&converted);
                } else {
                    out.push(if sup { '^' } else { '_' });
                    out.push('(');
                    out.push_str(group.trim());
                    out.push(')');
                }
                i = next;
                continue;
            }
            let next_ch = s[i..].chars().next().unwrap();
            if let Some(mapped) = if sup {
                map_sup(next_ch)
            } else {
                map_sub(next_ch)
            } {
                out.push(mapped);
            } else {
                out.push(if sup { '^' } else { '_' });
                out.push(next_ch);
            }
            i += next_ch.len_utf8();
            continue;
        }
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}
