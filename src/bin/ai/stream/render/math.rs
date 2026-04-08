const LITERAL_LBRACE_PLACEHOLDER: &str = "\u{E000}";
const LITERAL_RBRACE_PLACEHOLDER: &str = "\u{E001}";

macro_rules! lookup_match {
    ($key1:expr, $key2:expr, $guard:expr; $( (($pat1:pat, $pat2:pat), $name:literal) => $replacement:expr ),+ $(,)?) => {
        match ($key1, $key2) {
            $(
                ($pat1, $pat2) if $guard == $name => Some($replacement),
            )+
            _ => None,
        }
    };
    ($value:expr; $($pattern:pat => $replacement:expr),+ $(,)?) => {
        match $value {
            $(
                $pattern => Some($replacement),
            )+
            _ => None,
        }
    };
}


fn is_control_word_boundary(s: &str, index: usize) -> bool {
    match s.get(index..) {
        Some(rest) => match rest.chars().next() {
            Some(ch) => !ch.is_ascii_alphabetic(),
            None => true,
        },
        None => true,
    }
}

fn strip_sizing_commands(s: &str) -> String {
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

fn read_group_braced(s: &str, start: usize) -> Option<(String, usize)> {
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

fn read_group_bracketed(s: &str, start: usize) -> Option<(String, usize)> {
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

fn lookup_named_tex_command(cmd: &str) -> Option<&'static str> {
    let first = cmd.as_bytes().first().copied()?;
    let len = cmd.len();
    lookup_match!(first, len, cmd;
        ((b'a', 5), "alpha") => "α",
        ((b'a', 6), "approx") => "≈",
        ((b'b', 4), "beta") => "β",
        ((b'c', 3), "cap") => "∩",
        ((b'c', 3), "chi") => "χ",
        ((b'c', 3), "cup") => "∪",
        ((b'c', 4), "cdot") => "·",
        ((b'd', 3), "div") => "÷",
        ((b'd', 5), "delta") => "δ",
        ((b'e', 3), "eta") => "η",
        ((b'e', 5), "equiv") => "≡",
        ((b'e', 7), "epsilon") => "ε",
        ((b'g', 3), "geq") => "≥",
        ((b'g', 5), "gamma") => "γ",
        ((b'i', 2), "in") => "∈",
        ((b'i', 3), "int") => "∫",
        ((b'i', 4), "iota") => "ι",
        ((b'i', 5), "infty") => "∞",
        ((b'k', 5), "kappa") => "κ",
        ((b'l', 3), "leq") => "≤",
        ((b'l', 6), "lambda") => "λ",
        ((b'l', 9), "leftarrow") => "←",
        ((b'l', 14), "leftrightarrow") => "↔",
        ((b'm', 2), "mp") => "∓",
        ((b'm', 2), "mu") => "μ",
        ((b'n', 2), "nu") => "ν",
        ((b'n', 3), "neq") => "≠",
        ((b'n', 5), "notin") => "∉",
        ((b'o', 5), "omega") => "ω",
        ((b'p', 2), "pi") => "π",
        ((b'p', 2), "pm") => "±",
        ((b'p', 3), "phi") => "φ",
        ((b'p', 3), "psi") => "ψ",
        ((b'p', 4), "prod") => "∏",
        ((b'r', 3), "rho") => "ρ",
        ((b'r', 10), "rightarrow") => "→",
        ((b's', 3), "sum") => "∑",
        ((b's', 5), "sigma") => "σ",
        ((b's', 6), "subset") => "⊂",
        ((b's', 6), "supset") => "⊃",
        ((b's', 8), "subseteq") => "⊆",
        ((b's', 8), "supseteq") => "⊇",
        ((b't', 2), "to") => "→",
        ((b't', 3), "tau") => "τ",
        ((b't', 5), "theta") => "θ",
        ((b't', 5), "times") => "×",
        ((b'u', 7), "upsilon") => "υ",
        ((b'x', 2), "xi") => "ξ",
        ((b'z', 4), "zeta") => "ζ",
        ((b'D', 5), "Delta") => "Δ",
        ((b'G', 5), "Gamma") => "Γ",
        ((b'L', 6), "Lambda") => "Λ",
        ((b'O', 5), "Omega") => "Ω",
        ((b'P', 2), "Pi") => "Π",
        ((b'P', 3), "Phi") => "Φ",
        ((b'P', 3), "Psi") => "Ψ",
        ((b'S', 5), "Sigma") => "Σ",
        ((b'T', 5), "Theta") => "Θ",
        ((b'X', 2), "Xi") => "Ξ"
    )
}

fn lookup_escaped_tex_char(ch: char) -> Option<&'static str> {
    lookup_match!(ch;
        '_' => "_",
        '{' => LITERAL_LBRACE_PLACEHOLDER,
        '}' => LITERAL_RBRACE_PLACEHOLDER,
        ',' | ';' | ':' | ' ' => " ",
        '!' => ""
    )
}

fn lookup_mathbb_symbol(value: &str) -> Option<&'static str> {
    lookup_match!(value;
        "R" => "ℝ",
        "N" => "ℕ",
        "Z" => "ℤ",
        "Q" => "ℚ",
        "C" => "ℂ"
    )
}

fn replace_symbolic_tex_once(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] != b'\\' {
            let ch = s[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }

        let next = match s.get(i + 1..) {
            Some(rest) => match rest.chars().next() {
                Some(ch) => ch,
                None => {
                    out.push('\\');
                    break;
                }
            },
            None => {
                out.push('\\');
                break;
            }
        };

        if next.is_ascii_alphabetic() {
            let mut j = i + 1;
            while j < bytes.len() {
                let ch = s[j..].chars().next().unwrap();
                if !ch.is_ascii_alphabetic() {
                    break;
                }
                j += ch.len_utf8();
            }
            let cmd = &s[i + 1..j];

            if cmd == "mathbb"
                && let Some((group, next_index)) = read_group_braced(s, j)
            {
                let value = group.trim();
                out.push_str(lookup_mathbb_symbol(value).unwrap_or(value));
                i = next_index;
                continue;
            }

            if let Some(replacement) = lookup_named_tex_command(cmd) {
                out.push_str(replacement);
            } else {
                out.push('\\');
                out.push_str(cmd);
            }
            i = j;
            continue;
        }

        if let Some(replacement) = lookup_escaped_tex_char(next) {
            out.push_str(replacement);
            i += 1 + next.len_utf8();
            continue;
        }

        out.push('\\');
        out.push(next);
        i += 1 + next.len_utf8();
    }

    out
}

fn needs_parens(s: &str) -> bool {
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

fn wrap_parens(s: &str) -> String {
    let s = s.trim();
    if needs_parens(s) {
        format!("({s})")
    } else {
        s.to_string()
    }
}

fn replace_structural_tex(mut s: String) -> String {
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

pub(in crate::ai::stream) fn render_math_tex_to_unicode(s: &str) -> String {
    let mut t = strip_sizing_commands(s);

    t = replace_structural_tex(t);
    t = replace_symbolic_tex_once(&t);

    t = apply_super_subscripts(&t);
    t = t.replace('{', "");
    t = t.replace('}', "");
    t = t.replace(LITERAL_LBRACE_PLACEHOLDER, "{");
    t = t.replace(LITERAL_RBRACE_PLACEHOLDER, "}");

    t
}

fn apply_super_subscripts(s: &str) -> String {
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
