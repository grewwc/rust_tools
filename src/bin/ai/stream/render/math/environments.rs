use super::scripts::apply_super_subscripts;
/// LaTeX 环境渲染：`\begin{aligned}...\end{aligned}`、`\boxed{}`、`\text{}`。
///
/// 多行环境在块级（block-level）数学中使用，由 `render_math_block()` 调度。
/// 单行内联场景（`\boxed{...}\text{...}`）由 `replace_inline_environment_commands()` 处理。
use super::structural::read_group_braced;
use super::symbols::replace_symbolic_tex_once;

/// 处理单行内的环境级命令（`\boxed`、`\text`），返回处理后的字符串。
///
/// 在内联数学中，`\boxed{3.37}\text{年}` 应渲染为 `⌈3.37⌉ 年`。
/// 这些命令只能出现在 TeX 转 Unicode 的最后阶段——在结构变换和符号替换之后、
/// 清除花括号之前——否则花括号参数会被提前剥离导致命令失去参数。
pub(super) fn replace_inline_environment_commands(s: &str) -> String {
    let mut result = s.to_string();

    // 多轮：\boxed 和 \text 可能嵌套或交替出现
    let mut changed = true;
    while changed {
        changed = false;
        // \text{...}：提取花括号内容作为纯文本
        if let Some(replacement) = replace_text_command(&result) {
            result = replacement;
            changed = true;
        }
        // \boxed{...}：用 Unicode 方括号包裹
        if let Some(replacement) = replace_boxed_inline(&result) {
            result = replacement;
            changed = true;
        }
    }

    result
}

/// 将 `\text{...}` 替换为其花括号内的纯文本内容。
fn replace_text_command(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let needle = b"\\text";
    let mut i = 0usize;

    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] != needle {
            i += 1;
            continue;
        }

        // 确认 \text 后紧跟 {（而不是 \textbf 等）
        let after = i + needle.len();
        if after < bytes.len() && bytes[after] == b'{' {
            if let Some((_content, end)) = read_group_braced(s, after) {
                // content 包含花括号内的原始内容（含未处理的 TeX）
                let mut out = String::with_capacity(s.len());
                out.push_str(&s[..i]);
                // 对 text 内容做基本的 TeX→Unicode 转换（符号替换 + 脚本）
                out.push_str(&render_text_content(&s[i + needle.len() + 1..end - 1]));
                out.push_str(&s[end..]);
                return Some(out);
            }
        }

        i += 1;
    }

    None
}

/// 对 `\text{...}` 内部做轻量级 TeX→Unicode 转换。
///
/// `\text` 内容不应做 \frac/\sqrt 等结构变换，但需要处理下标/希腊字母等。
fn render_text_content(s: &str) -> String {
    let mut t = replace_symbolic_tex_once(s);
    t = apply_super_subscripts(&t);
    t
}

/// 内联模式下的 `\boxed{...}` → Unicode 方括号包裹。
///
/// 例如 `\boxed{3.37}` → `⌈3.37⌉`，用于行内公式。
fn replace_boxed_inline(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let needle = b"\\boxed";
    let mut i = 0usize;

    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] != needle {
            i += 1;
            continue;
        }

        let after = i + needle.len();
        if after < bytes.len() && bytes[after] == b'{' {
            if let Some((_content, end)) = read_group_braced(s, after) {
                let inner = &s[i + needle.len() + 1..end - 1];
                let rendered = render_math_single_line(inner);
                let mut out = String::with_capacity(s.len());
                out.push_str(&s[..i]);
                out.push_str(&format!("⌈{rendered}⌉"));
                out.push_str(&s[end..]);
                return Some(out);
            }
        }

        i += 1;
    }

    None
}

/// 对单行 TeX 做完整的 TeX→Unicode 转换（结构 + 符号 + 脚本）。
fn render_math_single_line(s: &str) -> String {
    let mut t = super::structural::strip_sizing_commands(s);
    t = super::structural::replace_structural_tex(t);
    t = replace_symbolic_tex_once(&t);
    t = apply_super_subscripts(&t);
    t
}

/// 检测 `\begin{<env>}` 开头，返回环境名（如 "aligned"）。
pub(super) fn detect_begin_env(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("\\begin{")?;
    let end = rest.find('}')?;
    Some(rest[..end].to_string())
}

/// 检测 `\end{<env>}` 开头，返回环境名。
pub(super) fn detect_end_env(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("\\end{")?;
    let end = rest.find('}')?;
    Some(rest[..end].to_string())
}

/// 渲染整个 `\begin{aligned}...\end{aligned}` 块为对齐的 Unicode 文本。
///
/// 协议：
/// - `lines` 包含 `\begin{aligned}` 到 `\end{aligned}` 之间的所有行
/// - 每行用 `\\` 结束（最后一行可能没有）
/// - `&` 标记对齐点：左列右对齐，右列左对齐
pub(super) fn render_aligned_block(lines: &[String]) -> String {
    let mut rows: Vec<Vec<String>> = Vec::new();

    for line in lines {
        // `\\` 是逻辑换行：既支持每个物理行末尾写法，也支持一行内多个公式行。
        for text in line.split("\\\\") {
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            rows.push(
                text.split('&')
                    .map(|c| render_math_single_line(c.trim()))
                    .collect(),
            );
        }
    }

    if rows.is_empty() {
        return String::new();
    }

    // 找到最大列数
    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(1);

    // 如果只有一列，不需要对齐
    if max_cols == 1 {
        return rows
            .iter()
            .map(|r| r[0].clone())
            .collect::<Vec<_>>()
            .join("\n");
    }

    // 对齐渲染：按列宽对齐
    // 对于 2 列情况：左列右对齐（右对齐到分隔符），右列左对齐
    render_aligned_columns(&rows, max_cols)
}

/// 按列渲染对齐的 Unicode 文本。
fn render_aligned_columns(rows: &[Vec<String>], max_cols: usize) -> String {
    // 计算每列最大宽度
    let mut col_widths = vec![0usize; max_cols];
    for row in rows {
        for (j, cell) in row.iter().enumerate() {
            if j < max_cols {
                col_widths[j] = col_widths[j].max(display_width(cell));
            }
        }
    }

    // 渲染每行
    let mut out = Vec::new();
    for row in rows {
        let mut line = String::new();
        for (j, cell) in row.iter().enumerate() {
            if j > 0 {
                line.push_str("  "); // 列间间距
            }
            let w = display_width(cell);
            if j < max_cols - 1 {
                // 左列右对齐
                let pad = col_widths[j].saturating_sub(w);
                line.push_str(&" ".repeat(pad));
                line.push_str(cell);
            } else {
                // 最后一列左对齐
                line.push_str(cell);
                let _ = w; // suppress unused
            }
        }
        out.push(line);
    }

    out.join("\n")
}

/// 计算字符串的终端显示宽度（复用 inline.rs 的 CJK-aware 实现）。
fn display_width(s: &str) -> usize {
    super::super::inline::terminal_display_width(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_text_command() {
        let input = "\\text{年}";
        let result = replace_text_command(input).unwrap();
        assert!(result.contains("年"), "got: {result}");
        assert!(!result.contains("\\text"), "got: {result}");
    }

    #[test]
    fn test_replace_boxed_inline() {
        let input = "\\boxed{3.37}";
        let result = replace_boxed_inline(input).unwrap();
        assert!(result.contains("⌈"), "got: {result}");
        assert!(result.contains("3.37"), "got: {result}");
        assert!(result.contains("⌉"), "got: {result}");
    }

    #[test]
    fn test_detect_begin_end_env() {
        assert_eq!(
            detect_begin_env("  \\begin{aligned}"),
            Some("aligned".into())
        );
        assert_eq!(detect_end_env("\\end{aligned}"), Some("aligned".into()));
        assert_eq!(detect_begin_env("plain text"), None);
        assert_eq!(detect_end_env("plain text"), None);
    }

    #[test]
    fn test_render_aligned_block() {
        let lines = vec![
            "NCF_0 = -8600".to_string(),
            "NCF_1 = -360".to_string(),
            "NCF_2 = 2345".to_string(),
        ];
        let result = render_aligned_block(&lines);
        assert!(result.contains("NCF"), "got: {result}");
        assert!(result.contains("-8600"), "got: {result}");
    }

    #[test]
    fn test_render_aligned_block_splits_logical_rows() {
        let lines = vec![r"x &= 1 \\ y &= 2".to_string()];
        let result = render_aligned_block(&lines);
        let rows = result.lines().collect::<Vec<_>>();
        assert_eq!(rows.len(), 2, "got: {result}");
        assert!(rows[0].contains("x"), "got: {result}");
        assert!(rows[1].contains("y"), "got: {result}");
    }
}
