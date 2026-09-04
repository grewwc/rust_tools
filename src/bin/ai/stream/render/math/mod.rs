/// TeX → Unicode rendering pipeline.
///
/// Converts LaTeX math formulas into Unicode text displayable in a terminal.
/// Supports inline formulas (`$...$` / `\(...\)`) and block formulas (`$$...$$` / `\[...\]`).
mod environments;
mod scripts;
mod structural;
mod symbols;

use environments::{
    detect_begin_env, detect_end_env, render_aligned_block, replace_inline_environment_commands,
};
use scripts::apply_super_subscripts;
use structural::{replace_structural_tex, strip_sizing_commands};
use symbols::{
    LITERAL_LBRACE_PLACEHOLDER, LITERAL_RBRACE_PLACEHOLDER, is_control_word_boundary,
    replace_symbolic_tex_once,
};

/// Replaces named TeX spacing commands with one terminal space.
///
/// TeX normally ignores source whitespace around these commands, so consume it
/// here as well to avoid displaying an accidental run of spaces in the terminal.
fn replace_named_spacing_commands(s: &str) -> String {
    const SPACING_COMMANDS: [&str; 4] = ["\\qquad", "\\quad", "\\enspace", "\\enskip"];

    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < s.len() {
        let command = SPACING_COMMANDS.iter().copied().find(|command| {
            s[i..].starts_with(command) && is_control_word_boundary(s, i + command.len())
        });
        if let Some(command) = command {
            while out.ends_with(' ') {
                out.pop();
            }
            out.push(' ');
            i += command.len();
            while let Some(ch) = s.get(i..).and_then(|rest| rest.chars().next()) {
                if !ch.is_whitespace() {
                    break;
                }
                i += ch.len_utf8();
            }
            continue;
        }

        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Converts a single line of TeX math content into a Unicode string.
///
/// Processing steps:
/// 1. Remove `\left`/`\right` sizing prefixes
/// 2. Recursively handle structural commands such as `\frac`, `\sqrt`
/// 3. Replace symbol commands like Greek letters and operators
/// 4. Handle `^` superscripts / `_` subscripts
/// 5. Clean up braces, keeping escaped literals
/// 6. Handle the `\text{}` and `\boxed{}` environment commands
pub(in crate::ai::stream) fn render_math_tex_to_unicode(s: &str) -> String {
    let mut t = replace_named_spacing_commands(s);
    t = strip_sizing_commands(&t);

    // Structural transforms must run before symbol substitution, since \frac{...}{...} relies on brace grouping
    t = replace_structural_tex(t);
    t = replace_symbolic_tex_once(&t);

    t = apply_super_subscripts(&t);
    t = replace_inline_environment_commands(&t);
    t = t.replace('{', "");
    t = t.replace('}', "");
    t = t.replace(LITERAL_LBRACE_PLACEHOLDER, "{");
    t = t.replace(LITERAL_RBRACE_PLACEHOLDER, "}");
    t
}

/// Renders a complete multi-line math block (handling environments like `\begin{aligned}`).
///
/// The caller (`markdown.rs`) should accumulate all lines between `$$`/`\[` and
/// `$$`/`\]` and pass them into this function in one go.
pub(in crate::ai::stream) fn render_math_block(lines: &[String]) -> String {
    let mut out = Vec::new();
    let mut env_buf: Option<(String, Vec<String>)> = None; // (env_name, accumulated_lines)

    for line in lines {
        let trimmed = line.trim();

        // Detect environment start
        if let Some(env_name) = detect_begin_env(trimmed) {
            env_buf = Some((env_name, Vec::new()));
            continue;
        }

        // Detect environment end
        if let Some(env_name) = detect_end_env(trimmed) {
            if let Some((ref buf_name, ref buf_lines)) = env_buf {
                if *buf_name == env_name {
                    match buf_name.as_str() {
                        "aligned" => {
                            out.push(render_aligned_block(buf_lines));
                        }
                        "gathered" | "align" | "cases" => {
                            // Generic fallback: render line by line
                            for bl in buf_lines {
                                out.push(render_math_tex_to_unicode(bl));
                            }
                        }
                        _ => {
                            // Unknown environment: render line by line
                            for bl in buf_lines {
                                out.push(render_math_tex_to_unicode(bl));
                            }
                        }
                    }
                    env_buf = None;
                    continue;
                }
            }
        }

        // Inside an environment: buffer the line
        if let Some((_, ref mut buf)) = env_buf {
            buf.push(line.to_string());
            continue;
        }

        // Ordinary line: render directly
        if trimmed.is_empty() {
            continue;
        }
        out.push(render_math_tex_to_unicode(trimmed));
    }

    // Unclosed environment: render the buffered lines one by one
    if let Some((_, buf)) = env_buf {
        for bl in &buf {
            out.push(render_math_tex_to_unicode(bl));
        }
    }

    out.join("\n")
}

/// Renders a single line of math content (for block-level non-environment lines).
pub(in crate::ai::stream) fn render_math_line(s: &str) -> String {
    render_math_tex_to_unicode(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_basic_symbols() {
        let result = render_math_tex_to_unicode(r#"\alpha + \beta"#);
        assert!(result.contains("α"), "got: {result}");
        assert!(result.contains("+"), "got: {result}");
        assert!(result.contains("β"), "got: {result}");
    }

    #[test]
    fn render_frac() {
        let result = render_math_tex_to_unicode(r#"\frac{1}{2}"#);
        assert!(result.contains("1"), "got: {result}");
        assert!(result.contains("/"), "got: {result}");
        assert!(result.contains("2"), "got: {result}");
    }

    #[test]
    fn render_binom_as_combination_notation() {
        for command in [r"\binom", r"\dbinom", r"\tbinom"] {
            let result = render_math_tex_to_unicode(&format!("{command}{{n}}{{k}}"));
            assert_eq!(result, "C(n, k)", "command: {command}");
        }
    }

    #[test]
    fn render_subscript() {
        let result = render_math_tex_to_unicode(r#"NCF_7"#);
        assert!(result.contains("NCF"), "got: {result}");
        assert!(result.contains("₇"), "got: {result}");
    }

    #[test]
    fn render_boxed_inline() {
        let result = render_math_tex_to_unicode(r#"\boxed{3.37}"#);
        assert!(result.contains("⌈"), "got: {result}");
        assert!(result.contains("3.37"), "got: {result}");
        assert!(result.contains("⌉"), "got: {result}");
    }

    #[test]
    fn render_text_command() {
        let result = render_math_tex_to_unicode(r#"\text{年}"#);
        assert!(result.contains("年"), "got: {result}");
        assert!(!result.contains("\\text"), "got: {result}");
    }

    #[test]
    fn render_math_block_aligned() {
        let lines = vec![
            r"NCF_0 &= -8600".to_string(),
            r"NCF_1 &= -360".to_string(),
            r"NCF_2 &= 2345".to_string(),
        ];
        let result = render_math_block(&lines);
        assert!(result.contains("NCF"), "got: {result}");
        assert!(result.contains("-8600"), "got: {result}");
        assert!(result.contains("-360"), "got: {result}");
        assert!(result.contains("2345"), "got: {result}");
    }

    #[test]
    fn renders_common_spacing_and_partial_scripts_without_raw_tex() {
        let result =
            render_math_tex_to_unicode(r"\sum_{n=0}^{\infty} ar^n = \frac{a}{1-r}, \quad |r| < 1");

        assert_eq!(result, "∑ₙ₌₀⁽∞⁾ arⁿ = a/(1-r), |r| < 1");
        assert!(!result.contains("\\quad"), "got: {result}");
        assert!(!result.contains("^("), "got: {result}");
    }

    #[test]
    fn renders_integral_bounds_with_compact_scripts() {
        let result = render_math_tex_to_unicode(r"\int_{a}^{b} f(x)\,dx");

        assert_eq!(result, "∫ₐᵇ f(x) dx");
    }
}
