/// Unicode superscript / subscript conversion.
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
            'a' => Some('ᵃ'),
            'b' => Some('ᵇ'),
            'c' => Some('ᶜ'),
            'd' => Some('ᵈ'),
            'e' => Some('ᵉ'),
            'f' => Some('ᶠ'),
            'g' => Some('ᵍ'),
            'h' => Some('ʰ'),
            'n' => Some('ⁿ'),
            'i' => Some('ⁱ'),
            'j' => Some('ʲ'),
            'k' => Some('ᵏ'),
            'l' => Some('ˡ'),
            'm' => Some('ᵐ'),
            'o' => Some('ᵒ'),
            'p' => Some('ᵖ'),
            'r' => Some('ʳ'),
            's' => Some('ˢ'),
            't' => Some('ᵗ'),
            'u' => Some('ᵘ'),
            'v' => Some('ᵛ'),
            'w' => Some('ʷ'),
            'x' => Some('ˣ'),
            'y' => Some('ʸ'),
            'z' => Some('ᶻ'),
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

    fn convert_group(group: &str, sup: bool) -> String {
        let mut out = String::new();
        let mut fully_mapped = true;
        for ch in group.chars() {
            if let Some(mapped) = if sup { map_sup(ch) } else { map_sub(ch) } {
                out.push(mapped);
            } else {
                fully_mapped = false;
                out.push(ch);
            }
        }
        if fully_mapped {
            return out;
        }
        let (open, close) = if sup { ('⁽', '⁾') } else { ('₍', '₎') };
        format!("{open}{out}{close}")
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
                out.push_str(&convert_group(group.trim(), sup));
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
                let (open, close) = if sup { ('⁽', '⁾') } else { ('₍', '₎') };
                out.push(open);
                out.push(next_ch);
                out.push(close);
            }
            i += next_ch.len_utf8();
            continue;
        }
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}
