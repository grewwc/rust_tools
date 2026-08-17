/// TeX → Unicode 渲染管线。
///
/// 将 LaTeX 数学公式转换为终端可显示的 Unicode 文本。
/// 支持行内公式（`$...$` / `\(...\)`）和块级公式（`$$...$$` / `\[...\]`）。
mod environments;
mod scripts;
mod structural;
mod symbols;

use environments::{
    detect_begin_env, detect_end_env, render_aligned_block, replace_inline_environment_commands,
};
use scripts::apply_super_subscripts;
use structural::{replace_structural_tex, strip_sizing_commands};
use symbols::{LITERAL_LBRACE_PLACEHOLDER, LITERAL_RBRACE_PLACEHOLDER, replace_symbolic_tex_once};

/// 将单行 TeX 数学内容转换为 Unicode 字符串。
///
/// 处理流程：
/// 1. 移除 `\left`/`\right` sizing 前缀
/// 2. 递归处理 `\frac`、`\sqrt` 等结构性命令
/// 3. 替换希腊字母、运算符等符号命令
/// 4. 处理 `^` 上标 / `_` 下标
/// 5. 清理花括号，保留转义字面量
/// 6. 处理 `\text{}` 和 `\boxed{}` 环境命令
pub(in crate::ai::stream) fn render_math_tex_to_unicode(s: &str) -> String {
    let mut t = strip_sizing_commands(s);

    // 结构变换必须在符号替换之前，因为 \frac{...}{...} 依赖花括号分组
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

/// 渲染完整的多行数学块内容（处理 `\begin{aligned}` 等环境）。
///
/// 调用方（`markdown.rs`）应将 `$$`/`\[` 和 `$$`/`\]` 之间的所有行
/// 累积后一次性传入本函数。
pub(in crate::ai::stream) fn render_math_block(lines: &[String]) -> String {
    let mut out = Vec::new();
    let mut env_buf: Option<(String, Vec<String>)> = None; // (env_name, accumulated_lines)

    for line in lines {
        let trimmed = line.trim();

        // 检测环境开始
        if let Some(env_name) = detect_begin_env(trimmed) {
            env_buf = Some((env_name, Vec::new()));
            continue;
        }

        // 检测环境结束
        if let Some(env_name) = detect_end_env(trimmed) {
            if let Some((ref buf_name, ref buf_lines)) = env_buf {
                if *buf_name == env_name {
                    match buf_name.as_str() {
                        "aligned" => {
                            out.push(render_aligned_block(buf_lines));
                        }
                        "gathered" | "align" | "cases" => {
                            // 通用回退：逐行渲染
                            for bl in buf_lines {
                                out.push(render_math_tex_to_unicode(bl));
                            }
                        }
                        _ => {
                            // 未知环境：逐行渲染
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

        // 在环境内：缓冲行
        if let Some((_, ref mut buf)) = env_buf {
            buf.push(line.to_string());
            continue;
        }

        // 普通行：直接渲染
        if trimmed.is_empty() {
            continue;
        }
        out.push(render_math_tex_to_unicode(trimmed));
    }

    // 未闭合的环境：将缓冲区内容逐行渲染
    if let Some((_, buf)) = env_buf {
        for bl in &buf {
            out.push(render_math_tex_to_unicode(bl));
        }
    }

    out.join("\n")
}

/// 渲染单行数学内容（用于块级非环境行）。
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
}
