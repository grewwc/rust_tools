/// 字符级 TeX 符号 → Unicode 映射表及相关辅助函数。
use super::structural::read_group_braced;

pub(super) const LITERAL_LBRACE_PLACEHOLDER: &str = "\u{E000}";
pub(super) const LITERAL_RBRACE_PLACEHOLDER: &str = "\u{E001}";

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

/// 判断字符 `s[index..]` 是否为控制词边界（非 ASCII 字母）。
pub(super) fn is_control_word_boundary(s: &str, index: usize) -> bool {
    match s.get(index..) {
        Some(rest) => match rest.chars().next() {
            Some(ch) => !ch.is_ascii_alphabetic(),
            None => true,
        },
        None => true,
    }
}

/// 查找命名 TeX 命令的 Unicode 替换。
pub(super) fn lookup_named_tex_command(cmd: &str) -> Option<&'static str> {
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

/// 查找转义字符序列（`\,` `\{` 等）的 Unicode 替换。
pub(super) fn lookup_escaped_tex_char(ch: char) -> Option<&'static str> {
    lookup_match!(ch;
        '_' => "_",
        '{' => LITERAL_LBRACE_PLACEHOLDER,
        '}' => LITERAL_RBRACE_PLACEHOLDER,
        ',' | ';' | ':' | ' ' => " ",
        '!' => ""
    )
}

/// 查找 `\mathbb{X}` 中 X 对应的双线字母。
pub(super) fn lookup_mathbb_symbol(value: &str) -> Option<&'static str> {
    lookup_match!(value;
        "R" => "ℝ",
        "N" => "ℕ",
        "Z" => "ℤ",
        "Q" => "ℚ",
        "C" => "ℂ"
    )
}

/// 将 TeX 符号命令（`\alpha`、`\to`、`\mathbb{R}` 等）替换为 Unicode。
/// 仅做**单轮**替换，不做递归（结构性替换由 `structural.rs` 负责）。
pub(super) fn replace_symbolic_tex_once(s: &str) -> String {
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
