use crate::ai::stream::render::code::{MONOKAI_BG, MONOKAI_FG};

/// 终端默认把 East-Asian **Ambiguous** 宽度字符（箭头 `→`、数学符号 `× ± ≤ ≥ ≠`、
/// box-drawing、braille 等）按单列渲染，只有真正的 Wide/全角字符（CJK 等）才占 2 列。
/// `width_cjk` 会把所有 ambiguous 字符算成 2 列，导致表格边框、cursor-up 高度整体漂移
/// （尤其是含 `→` 的单元格右边框被逐行拉偏）。因此这里统一用 `width`（ambiguous=1）。
pub(super) fn terminal_cell_width(ch: char) -> usize {
    // Emoji variation selector（U+FE0F）本身宽度 0，但它把前一个 base 符号从
    // 文本呈现（1 列）强制成 emoji 呈现（2 列）。真实终端据此渲染，例如 `⚠️`
    // （U+26A0 + U+FE0F）占 2 列。若按 unicode-width 的 0 计算，表格/预览会比
    // 终端实际显示窄 1 列，导致行被硬折行、cursor-up 擦除行数算少、旧表头残留
    // 堆叠。这里把 VS16 计为 1 列，等价于给 base 补上被撑开的那一列。
    if ch == '\u{fe0f}' {
        return 1;
    }
    if is_single_width_terminal_symbol(ch) {
        return 1;
    }
    // 现代 macOS 终端把 Miscellaneous Symbols（U+2600-U+26FF）、
    // Miscellaneous Technical（U+2300-U+23FF）、Dingbats（U+2700-U+27BF）
    // 以及部分 Geometric Shapes 里的评分/涨跌标记（△▽▲▼）等块中的
    // ambiguous-width 符号当作 emoji 渲染为 2 列。unicode-width
    // 对这些字符返回 1（ambiguous），但终端实际占 2 列。若不修正，含 ⚠ ☎ ✂
    // 或 `4.0→4.0 △` 这类涨跌标记的单元格右边框会被逐行拉偏。
    if is_ambiguous_emoji_block_char(ch) {
        return 2;
    }
    unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0)
}

pub(super) fn terminal_display_width(s: &str) -> usize {
    let mut total = 0;
    let mut prev_was_emoji_block = false;
    for ch in s.chars() {
        if ch == '\u{fe0f}' && prev_was_emoji_block {
            // 前一个字符已经是 emoji 块字符（按 2 列计算），VS16 不额外占宽
            prev_was_emoji_block = false;
            continue;
        }
        total += terminal_cell_width(ch);
        prev_was_emoji_block = is_ambiguous_emoji_block_char(ch);
    }
    total
}

/// Strip redundant U+FE0F (VS16) from visible text.
///
/// When VS16 follows an `is_ambiguous_emoji_block_char` (e.g. ⚠ U+26A0), the base
/// already renders as 2-column emoji without VS16. Keeping VS16 in the string causes
/// `render_and_pad_cell` to undercount by 1 column (VS16 takes 1 cell in the terminal
/// but `terminal_display_width` skips it), shifting table borders right.
pub(super) fn strip_redundant_vs16(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_was_emoji_block = false;
    for ch in s.chars() {
        if ch == '\u{fe0f}' && prev_was_emoji_block {
            prev_was_emoji_block = false;
            continue; // drop redundant VS16
        }
        out.push(ch);
        prev_was_emoji_block = is_ambiguous_emoji_block_char(ch);
    }
    out
}

fn is_single_width_terminal_symbol(ch: char) -> bool {
    matches!(
        ch,
        '\u{2500}'..='\u{259f}' // box drawing + block elements
            | '\u{2800}'..='\u{28ff}' // braille patterns
    )
}

/// 判断字符是否属于"终端按 emoji 渲染为 2 列"的 Unicode 块。
///
/// 这些块里的字符在 Unicode 里是 East Asian Ambiguous 宽度（unicode-width 返回 1），
/// 但现代 macOS 终端用 Apple Color Emoji 字体渲染它们，实际占 2 列。
/// 本身就是 Wide 的 emoji（✅ ❌ 等）已由 unicode-width 正确返回 2，无需处理。
fn is_ambiguous_emoji_block_char(ch: char) -> bool {
    let c = ch as u32;
    matches!(
        c,
        // Miscellaneous Technical: ⌚ ⌛ ⏰ 等
        0x2300..=0x23FF
            // Miscellaneous Symbols: ☀ ☁ ⚠ ☎ ⚡ 等
            | 0x2600..=0x26FF
            // Dingbats: ✂ ✆ ✈ ✉ ✌ ✍ ✎ ✏ ✓ ✔ ✨ 等
            | 0x2700..=0x27BF
            // Geometric Shapes 中常见的涨跌/评分三角标记：△ ▲ ▽ ▼
            | 0x25B2 | 0x25B3 | 0x25BC | 0x25BD
    )
}

/// 把紧贴文件名/链接的中文标点（`：` `，` `。`）转成英文版（`:` `,` `.`）。
///
/// Agent 常把中文标点和文件名/行号或链接连在一起输出，如 `src/foo.rs：42`、
/// `调用时机： app.py:334`、`https://x.com，详见`。全角标点会让终端无法识别
/// file:line / URL 边界，导致无法点击跳转。标点前是路径/链接常见字符时直接转换；
/// 标点前是普通中文时，仅在其后（允许空格及行内代码标记）确实跟着可点击目标时转换，
/// 避免误伤 `时间：12点` 这类纯中文语境。
fn normalize_cjk_punct_around_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev: Option<char> = None;
    let mut i = 0;
    while i < s.len() {
        // code / math span 是 Markdown 字面量，不能为了终端跳转而改写其内容。
        if let Some((span, next)) = take_atomic_markdown_span(s, i)
            && (span.starts_with('`') || span.starts_with('$'))
        {
            out.push_str(&span);
            prev = span.chars().last();
            i = next;
            continue;
        }

        let ch = s[i..].chars().next().expect("character boundary");
        let next_is_clickable_target = starts_clickable_terminal_target(s, i + ch.len_utf8());
        let replaced = match ch {
            '：' if prev.is_some_and(is_path_neighbor) || next_is_clickable_target => ':',
            '，' if prev.is_some_and(is_path_neighbor) || next_is_clickable_target => ',',
            '。' if prev.is_some_and(is_path_neighbor) || next_is_clickable_target => '.',
            _ => ch,
        };
        out.push(replaced);
        // 用原始字符判断前文，避免连续全角标点连锁转换（如 `：：` 只转第一个）。
        prev = Some(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_path_neighbor(ch: char) -> bool {
    matches!(
        ch,
        'a'..='z' | 'A'..='Z' | '0'..='9' | '/' | '_' | '.' | '-' | '~' | '+' | '@'
            | ':' | '#' | '?' | '=' | '&' | '%' | '`'
    )
}

/// 判断全角标点右侧是否紧跟终端通常可点击的路径 / URL。
///
/// Markdown 行内代码里的 `app.py:334` 仍是终端链接目标，因此识别其内容；数学 span
/// 则保持字面量，不参与判断。这里只接受带路径分隔符、文件扩展名或 URL scheme 的目标，
/// 不能把 `时间：12点` 之类普通中文后的数字误判为文件。
fn starts_clickable_terminal_target(s: &str, mut start: usize) -> bool {
    while let Some(ch) = s.get(start..).and_then(|rest| rest.chars().next()) {
        if !matches!(ch, ' ' | '\t') {
            break;
        }
        start += ch.len_utf8();
    }

    if start >= s.len() {
        return false;
    }

    if let Some((span, _)) = take_atomic_markdown_span(s, start) {
        if span.starts_with('`') {
            return is_clickable_terminal_target(&span[1..span.len() - 1]);
        }
        if span.starts_with('$') {
            return false;
        }
    }

    let end = s[start..]
        .char_indices()
        .find_map(|(offset, ch)| (!ch.is_ascii() || ch.is_ascii_whitespace()).then_some(start + offset))
        .unwrap_or(s.len());
    is_clickable_terminal_target(&s[start..end])
}

fn is_clickable_terminal_target(token: &str) -> bool {
    let token = token.trim_end_matches(['.', ',', ';', ':', ')', ']']);
    if token.starts_with("https://") || token.starts_with("http://") || token.starts_with("file://") {
        return true;
    }

    if token.starts_with('/')
        || token.starts_with("./")
        || token.starts_with("../")
        || token.starts_with("~/")
        || token.contains('/')
    {
        return true;
    }

    let Some((stem, extension)) = token.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty()
        && stem
            .chars()
            .any(|ch| ch.is_ascii_alphabetic() || matches!(ch, '_' | '-'))
        && extension.chars().any(|ch| ch.is_ascii_alphabetic())
}

pub(super) fn render_inline_md(s: &str, base: &str) -> String {
    let normalized = normalize_cjk_punct_around_path(s);
    let s = normalized.as_str();
    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    let mut bold = false;
    let mut italic = false;
    let mut code = false;
    // math 状态在新实现里只在配对成功的局部分支内短暂为 true，循环外恒为 false。
    let mut math = false;

    fn apply_style(out: &mut String, base: &str, bold: bool, italic: bool, code: bool, math: bool) {
        out.push_str("\x1b[0m");
        out.push_str(base);
        if bold {
            out.push_str("\x1b[1m");
        }
        if code {
            out.push_str(MONOKAI_BG);
            out.push_str(MONOKAI_FG);
        }
        if italic {
            out.push_str("\x1b[3m");
        }
        if math {
            out.push_str("\x1b[95m");
        }
    }

    fn is_url_start(bytes: &[u8], i: usize) -> bool {
        bytes
            .get(i..i + 8)
            .is_some_and(|s| s.eq_ignore_ascii_case(b"https://"))
            || bytes
                .get(i..i + 7)
                .is_some_and(|s| s.eq_ignore_ascii_case(b"http://"))
    }

    fn url_raw_end(bytes: &[u8], start: usize) -> usize {
        let mut end = start;
        while end < bytes.len() {
            let b = bytes[end];
            if b.is_ascii_whitespace()
                || b == b'<'
                || b == b'"'
                || b == b'\''
                || b == b'`'
                || b == b'\\'
            {
                break;
            }
            end += 1;
        }
        end
    }

    while i < bytes.len() {
        // 反引号 `code`：必须找到配对的闭合反引号才开启样式，否则按字面字符输出。
        // 旧实现是无脑切换 `code = !code`，模型若输出"use `cargo run to test"
        // （单个未闭合反引号），剩余整行都会被着上代码块背景色。
        if bytes[i] == b'`' && !math {
            if let Some(close) = find_unescaped_delim(s, i + 1, "`") {
                let content = &s[i + 1..close - 1];
                code = true;
                apply_style(&mut out, base, bold, italic, code, math);
                out.push_str(content);
                code = false;
                apply_style(&mut out, base, bold, italic, code, math);
                i = close;
                continue;
            }
            out.push('`');
            i += 1;
            continue;
        }

        // **bold**：同样要求配对。模型常输出未闭合的 "**Note:" 或 "5 ** 3"，
        // 旧实现会让其后整段都加粗。
        if !code && !math && bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            if let Some(close) = find_unescaped_delim(s, i + 2, "**") {
                let content = &s[i + 2..close - 2];
                bold = true;
                apply_style(&mut out, base, bold, italic, code, math);
                // bold 内部仍可能含 italic / code，递归处理保证嵌套样式正确。
                out.push_str(&render_inline_md(content, base));
                bold = false;
                apply_style(&mut out, base, bold, italic, code, math);
                i = close;
                continue;
            }
            out.push_str("**");
            i += 2;
            continue;
        }

        // *italic*：同上。"5 * 3 = 15" 不应被当成 italic 触发。
        if !code && !math && bytes[i] == b'*' {
            if let Some(close) = find_unescaped_delim(s, i + 1, "*") {
                let content = &s[i + 1..close - 1];
                italic = true;
                apply_style(&mut out, base, bold, italic, code, math);
                out.push_str(&render_inline_md(content, base));
                italic = false;
                apply_style(&mut out, base, bold, italic, code, math);
                i = close;
                continue;
            }
            out.push('*');
            i += 1;
            continue;
        }

        // ~~strikethrough~~
        if !code && !math && i + 1 < bytes.len() && bytes[i] == b'~' && bytes[i + 1] == b'~' {
            i += 2; // skip opening ~~
            let start = i;
            while i + 1 < bytes.len() && !(bytes[i] == b'~' && bytes[i + 1] == b'~') {
                let ch = s[i..].chars().next().unwrap();
                i += ch.len_utf8();
            }
            let content = &s[start..i];
            out.push_str("\x1b[0m");
            out.push_str(base);
            if bold {
                out.push_str("\x1b[1m");
            }
            if italic {
                out.push_str("\x1b[3m");
            }
            out.push_str("\x1b[9m"); // strikethrough
            out.push_str(content);
            out.push_str("\x1b[0m");
            out.push_str(base);
            if bold {
                out.push_str("\x1b[1m");
            }
            if italic {
                out.push_str("\x1b[3m");
            }
            if i + 1 < bytes.len() {
                i += 2; // skip closing ~~
            }
            continue;
        }

        // $math$ / $$display$$：要求配对，避免把单独的"$5"、"$PATH"等 $ 字符
        // 误识为公式起点，把行尾全部当成 LaTeX 渲染。
        if !code && bytes[i] == b'$' && !math {
            let is_double = i + 1 < bytes.len() && bytes[i + 1] == b'$';
            let delim = if is_double { "$$" } else { "$" };
            if let Some(close) = find_unescaped_delim(s, i + delim.len(), delim) {
                let content = &s[i + delim.len()..close - delim.len()];
                math = true;
                apply_style(&mut out, base, bold, italic, code, math);
                out.push_str(&crate::ai::stream::render_math_tex_to_unicode(
                    content.trim(),
                ));
                math = false;
                apply_style(&mut out, base, bold, italic, code, math);
                i = close;
                continue;
            }
            // 未配对：按字面输出
            out.push('$');
            i += 1;
            continue;
        }

        if !math && is_url_start(bytes, i) {
            let raw_end = url_raw_end(bytes, i);
            let mut end = raw_end;
            while end > i {
                match bytes[end - 1] {
                    b'.' | b',' | b';' | b':' | b')' | b']' => end -= 1,
                    _ => break,
                }
            }
            let url = &s[i..end];
            let trail = &s[end..raw_end];

            out.push_str("\x1b[0m");
            out.push_str(base);
            if bold {
                out.push_str("\x1b[1m");
            }
            if italic {
                out.push_str("\x1b[3m");
            }
            out.push_str("\x1b[4m\x1b[34m");
            out.push_str(url);
            apply_style(&mut out, base, bold, italic, code, math);
            out.push_str(trail);

            i = raw_end;
            continue;
        }

        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    out.push_str("\x1b[0m");
    out
}

fn strip_inline_md_markers(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    let mut code = false;
    let mut math = false;
    let mut math_delim = "$";
    let mut math_buf = String::new();
    while i < bytes.len() {
        if bytes[i] == b'`' {
            code = !code;
            i += 1;
            continue;
        }
        if !code && bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            continue;
        }
        if !code && bytes[i] == b'*' {
            i += 1;
            continue;
        }
        if !code && i + 1 < bytes.len() && bytes[i] == b'~' && bytes[i + 1] == b'~' {
            i += 2;
            continue;
        }
        if !code && bytes[i] == b'$' {
            let is_double = i + 1 < bytes.len() && bytes[i + 1] == b'$';
            let delim = if is_double { "$$" } else { "$" };
            if math {
                if delim == math_delim {
                    out.push_str(&crate::ai::stream::render_math_tex_to_unicode(
                        math_buf.trim(),
                    ));
                    math_buf.clear();
                    math = false;
                    i += delim.len();
                    continue;
                }
            } else {
                math = true;
                math_delim = delim;
                i += delim.len();
                continue;
            }
        }
        let ch = s[i..].chars().next().unwrap();
        if math && !code {
            math_buf.push(ch);
        } else {
            out.push(ch);
        }
        i += ch.len_utf8();
    }
    if math && !math_buf.is_empty() {
        out.push_str(&crate::ai::stream::render_math_tex_to_unicode(
            math_buf.trim(),
        ));
    }
    out
}

pub(super) fn visible_width(s: &str) -> usize {
    terminal_display_width(&strip_inline_md_markers(s))
}

pub(super) fn wrap_md_cell(s: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    if s.trim().is_empty() {
        return vec![String::new()];
    }

    let mut bold = false;
    let mut italic = false;
    let mut cur = String::new();
    let mut cur_w = 0usize;
    let mut lines: Vec<String> = Vec::new();

    fn style_prefix(bold: bool, italic: bool) -> String {
        let mut p = String::new();
        if bold {
            p.push_str("**");
        }
        if italic {
            p.push('*');
        }
        p
    }

    fn style_suffix(bold: bool, italic: bool) -> String {
        let mut s = String::new();
        if italic {
            s.push('*');
        }
        if bold {
            s.push_str("**");
        }
        s
    }

    let start_new_line = |cur: &mut String, cur_w: &mut usize, bold: bool, italic: bool| {
        *cur = style_prefix(bold, italic);
        *cur_w = 0;
    };

    let trim_trailing_spaces = |cur: &mut String, cur_w: &mut usize| {
        while cur.ends_with(' ') {
            cur.pop();
            *cur_w = cur_w.saturating_sub(1);
        }
    };

    let close_line = |lines: &mut Vec<String>, cur: &mut String, bold: bool, italic: bool| {
        cur.push_str(&style_suffix(bold, italic));
        lines.push(std::mem::take(cur));
    };

    let mut i = 0usize;
    start_new_line(&mut cur, &mut cur_w, bold, italic);

    while i < s.len() {
        let rest = &s[i..];

        if rest.starts_with("**") {
            bold = !bold;
            cur.push_str("**");
            i += 2;
            continue;
        }

        if rest.starts_with('*') && !rest.starts_with("**") {
            italic = !italic;
            cur.push('*');
            i += 1;
            continue;
        }

        if let Some((piece, next)) = take_atomic_markdown_span(s, i) {
            let piece_width = visible_width(&piece);
            if piece_width > width {
                if cur_w > 0 {
                    trim_trailing_spaces(&mut cur, &mut cur_w);
                    close_line(&mut lines, &mut cur, bold, italic);
                    start_new_line(&mut cur, &mut cur_w, bold, italic);
                }
                for wrapped_piece in wrap_overlong_atomic_markdown_span(&piece, width) {
                    if cur_w > 0 {
                        close_line(&mut lines, &mut cur, bold, italic);
                        start_new_line(&mut cur, &mut cur_w, bold, italic);
                    }
                    cur.push_str(&wrapped_piece);
                    cur_w = visible_width(&wrapped_piece);
                }
                i = next;
                continue;
            }
            if cur_w > 0 && cur_w + piece_width > width {
                trim_trailing_spaces(&mut cur, &mut cur_w);
                close_line(&mut lines, &mut cur, bold, italic);
                start_new_line(&mut cur, &mut cur_w, bold, italic);
            }
            cur.push_str(&piece);
            cur_w += piece_width;
            i = next;
            continue;
        }

        if let Some((piece, next)) = take_ascii_non_whitespace_run(s, i) {
            let piece_width = visible_width(&piece);
            if piece_width <= width {
                if cur_w > 0 && cur_w + piece_width > width {
                    trim_trailing_spaces(&mut cur, &mut cur_w);
                    close_line(&mut lines, &mut cur, bold, italic);
                    start_new_line(&mut cur, &mut cur_w, bold, italic);
                }
                cur.push_str(&piece);
                cur_w += piece_width;
                i = next;
                continue;
            }
        }

        let ch = rest.chars().next().unwrap();
        let w = terminal_cell_width(ch);
        if cur_w > 0 && cur_w + w > width {
            trim_trailing_spaces(&mut cur, &mut cur_w);
            close_line(&mut lines, &mut cur, bold, italic);
            start_new_line(&mut cur, &mut cur_w, bold, italic);
        }
        cur.push(ch);
        cur_w += w;
        i += ch.len_utf8();
    }

    close_line(&mut lines, &mut cur, bold, italic);
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn wrap_overlong_atomic_markdown_span(s: &str, width: usize) -> Vec<String> {
    let Some((prefix, inner, suffix)) = split_atomic_markdown_span(s) else {
        return wrap_plain_visible_text(s, width);
    };
    wrap_plain_visible_text(inner, width)
        .into_iter()
        .map(|chunk| format!("{prefix}{chunk}{suffix}"))
        .collect()
}

fn split_atomic_markdown_span(s: &str) -> Option<(&str, &str, &str)> {
    if s.starts_with("```") || s.starts_with("~~~") {
        return None;
    }
    for delim in ["~~", "$$", "`", "$", "*"] {
        if s.starts_with(delim) && s.ends_with(delim) && s.len() >= delim.len() * 2 {
            let inner_start = delim.len();
            let inner_end = s.len() - delim.len();
            return Some((
                &s[..inner_start],
                &s[inner_start..inner_end],
                &s[inner_end..],
            ));
        }
    }
    None
}

fn wrap_plain_visible_text(s: &str, width: usize) -> Vec<String> {
    if s.is_empty() {
        return vec![String::new()];
    }

    let width = width.max(1);
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0usize;
    for ch in s.chars() {
        let w = terminal_cell_width(ch);
        if cur_w > 0 && cur_w + w > width {
            out.push(std::mem::take(&mut cur));
            cur_w = 0;
        }
        cur.push(ch);
        cur_w += w;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn take_ascii_non_whitespace_run(s: &str, start: usize) -> Option<(String, usize)> {
    let rest = &s[start..];
    let first = rest.chars().next()?;
    if !first.is_ascii() || first.is_ascii_whitespace() {
        return None;
    }
    if matches!(first, '*' | '`' | '$' | '\\') {
        return None;
    }

    let mut end = start;
    for (offset, ch) in rest.char_indices() {
        if !ch.is_ascii() || ch.is_ascii_whitespace() || matches!(ch, '*' | '`' | '$' | '\\') {
            break;
        }
        end = start + offset + ch.len_utf8();
    }

    (end > start).then(|| (s[start..end].to_string(), end))
}

fn take_atomic_markdown_span(s: &str, start: usize) -> Option<(String, usize)> {
    let rest = &s[start..];

    if rest.starts_with('`') {
        let end = find_unescaped_delim(s, start + 1, "`")?;
        return Some((s[start..end].to_string(), end));
    }

    if rest.starts_with("~~") {
        let end = find_unescaped_delim(s, start + 2, "~~")?;
        return Some((s[start..end].to_string(), end));
    }

    if rest.starts_with("$$") {
        let end = find_unescaped_delim(s, start + 2, "$$")?;
        return Some((s[start..end].to_string(), end));
    }

    if rest.starts_with('$') {
        let end = find_unescaped_delim(s, start + 1, "$")?;
        return Some((s[start..end].to_string(), end));
    }

    // Single `*` for italic — grab until matching closing `*`
    if rest.starts_with('*') && !rest.starts_with("**") {
        let end = find_unescaped_delim(s, start + 1, "*")?;
        return Some((s[start..end].to_string(), end));
    }

    if let Some(stripped) = rest.strip_prefix('\\') {
        let next = stripped.chars().next()?;
        let end = start + 1 + next.len_utf8();
        return Some((s[start..end].to_string(), end));
    }

    None
}

fn find_unescaped_delim(s: &str, mut i: usize, delim: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    while i < bytes.len() {
        if s[i..].starts_with(delim) && !is_escaped_at(s, i) {
            return Some(i + delim.len());
        }
        let ch = s[i..].chars().next()?;
        i += ch.len_utf8();
    }
    None
}

fn is_escaped_at(s: &str, idx: usize) -> bool {
    if idx == 0 {
        return false;
    }

    let mut backslashes = 0usize;
    let mut i = idx;
    while i > 0 {
        let prev = s[..i].chars().next_back().unwrap();
        if prev != '\\' {
            break;
        }
        backslashes += 1;
        i -= prev.len_utf8();
    }
    backslashes % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::stream::render::code::{MONOKAI_BG, MONOKAI_FG};

    #[test]
    fn wrap_md_cell_uses_visible_width_for_math_and_code_spans() {
        let math = wrap_md_cell(r#"$\frac{1}{2}$"#, 5);
        assert_eq!(math, vec![r#"$\frac{1}{2}$"#]);

        let code = wrap_md_cell(r#"`a|b`"#, 3);
        assert_eq!(code, vec![r#"`a|b`"#]);
    }

    #[test]
    fn wrap_md_cell_prefers_ascii_word_boundaries() {
        let wrapped = wrap_md_cell("ANSI 失败 → clickhouse 成功", 16);
        assert_eq!(wrapped, vec!["ANSI 失败 →", "clickhouse 成功"]);
    }

    #[test]
    fn wrap_md_cell_splits_overlong_ascii_token_only_as_fallback() {
        let wrapped = wrap_md_cell("supercalifragilistic", 6);
        assert!(wrapped.len() > 1);
        assert_eq!(wrapped.join(""), "supercalifragilistic");
        for line in wrapped {
            assert!(visible_width(&line) <= 6);
        }
    }

    #[test]
    fn wrap_md_cell_splits_overlong_code_span() {
        let wrapped = wrap_md_cell(
            "`async processOrder(orderId: string, options?: { timeout?: number })`",
            18,
        );

        assert!(wrapped.len() > 1);
        for line in wrapped {
            assert!(
                line.starts_with('`') && line.ends_with('`'),
                "code styling should be preserved per wrapped line: {line:?}"
            );
            assert!(visible_width(&line) <= 18, "{line:?}");
        }
    }

    #[test]
    fn terminal_width_counts_box_drawing_as_single_width() {
        assert_eq!(terminal_display_width("┌────┬────┐"), 11);
        assert_eq!(visible_width("────"), 4);
        assert_eq!(wrap_md_cell("──────", 4), vec!["────", "──"]);
    }

    #[test]
    fn terminal_width_counts_ambiguous_symbols_as_single_width() {
        // 箭头/数学符号是 East-Asian Ambiguous 宽度：终端按 1 列渲染。
        // 若按 width_cjk 算成 2 列，含 `→` 的表格单元格右边框会被逐行拉偏。
        for ch in ['→', '←', '↔', '×', '±', '≤', '≥', '≠', '∈', '⊂'] {
            assert_eq!(
                terminal_cell_width(ch),
                1,
                "ambiguous-width symbol {ch:?} must render as a single terminal column"
            );
        }
        // 真正的全角/CJK 字符仍占 2 列，不受影响。
        for ch in ['中', '文', '你', '好'] {
            assert_eq!(
                terminal_cell_width(ch),
                2,
                "CJK char {ch:?} stays double-width"
            );
        }
        // 含箭头的结果单元格：3 个可见字符（→ 空格 x）应为 3 列，而非 4。
        assert_eq!(terminal_display_width("→ x"), 3);
        // emoji 块的 ambiguous 字符（⚠ ☎ ✂）现代终端按 2 列渲染。
        for ch in ['⚠', '☎', '✂', '☀', '✈'] {
            assert_eq!(
                terminal_cell_width(ch),
                2,
                "emoji-block symbol {ch:?} must render as double width"
            );
        }
        // emoji 块字符 + 数字 = 3 列（emoji 2 + 数字 1）。
        assert_eq!(terminal_display_width("⚠1"), 3);
    }

    #[test]
    fn terminal_width_counts_emoji_presentation_as_double_width() {
        // 带 emoji variation selector（U+FE0F）的符号，真实终端按 emoji 呈现占 2 列。
        // `⚠️` = U+26A0 + U+FE0F：base 是 ambiguous(1) + VS16(补 1) = 2 列。
        assert_eq!(terminal_display_width("⚠️"), 2);
        // 现代 macOS 终端把 Miscellaneous Symbols 块的字符当作 emoji 渲染为 2 列，
        // 即使没有 VS16。⚠（U+26A0）属于此块。
        assert_eq!(terminal_display_width("⚠"), 2);
        // 本身即 emoji-presentation 的字符（unicode-width 判为 2）不受影响。
        assert_eq!(terminal_display_width("✅"), 2);
        assert_eq!(terminal_display_width("❌"), 2);
        // macOS 终端里的涨跌/评分三角标记也会占 2 列。
        assert_eq!(terminal_display_width("△"), 2);
        assert_eq!(terminal_display_width("▲"), 2);
        assert_eq!(terminal_display_width("▽"), 2);
        assert_eq!(terminal_display_width("▼"), 2);
        // VS16 单独出现时贡献 1 列（等价于给紧邻 base 补足被撑开的那一列）。
        assert_eq!(terminal_cell_width('\u{fe0f}'), 1);
    }

    #[test]
    fn inline_code_uses_monokai_colors() {
        let rendered = render_inline_md("use `cargo test` please", "");
        assert!(rendered.contains(MONOKAI_BG));
        assert!(rendered.contains(MONOKAI_FG));
        assert!(rendered.contains("cargo test"));
    }

    #[test]
    fn italic_rendering() {
        let rendered = render_inline_md("*hello*", "");
        assert!(!rendered.contains("*hello*")); // markers consumed
        assert!(rendered.contains("hello"));
    }

    #[test]
    fn strikethrough_rendering() {
        let rendered = render_inline_md("~~deleted~~", "");
        assert!(!rendered.contains("~~deleted~~")); // markers consumed
        assert!(rendered.contains("deleted"));
    }

    #[test]
    fn bold_italic_combined() {
        let rendered = render_inline_md("***bold italic***", "");
        assert!(rendered.contains("bold italic"));
    }

    #[test]
    fn strip_markers_with_strikethrough() {
        let stripped = strip_inline_md_markers("~~text~~");
        assert_eq!(stripped, "text");
    }

    #[test]
    fn unclosed_backtick_is_not_styled() {
        // 未配对反引号：应原样输出为字面字符，绝不开启 code 背景色。
        let rendered = render_inline_md("use `cargo to test", "");
        assert!(!rendered.contains(MONOKAI_BG));
        assert!(!rendered.contains(MONOKAI_FG));
        assert!(rendered.contains("`cargo to test"));
    }

    #[test]
    fn unclosed_asterisk_does_not_italicize_rest_of_line() {
        // "5 * 3 = 15"：单 * 不应触发 italic，整段不应包含 \x1b[3m。
        let rendered = render_inline_md("5 * 3 = 15", "");
        assert!(!rendered.contains("\x1b[3m"));
        assert!(rendered.contains("5 * 3 = 15"));
    }

    #[test]
    fn unclosed_double_asterisk_does_not_bold_rest_of_line() {
        let rendered = render_inline_md("note: **important things to do later", "");
        assert!(!rendered.contains("\x1b[1m"));
        assert!(rendered.contains("**important things"));
    }

    #[test]
    fn standalone_dollar_sign_is_literal() {
        // "$5 USD" 不应被当成 math 起点，整段不应触发 math 颜色 \x1b[95m。
        let rendered = render_inline_md("price: $5 USD", "");
        assert!(!rendered.contains("\x1b[95m"));
        assert!(rendered.contains("$5 USD"));
    }

    #[test]
    fn code_and_math_spans_keep_cjk_punctuation_literal() {
        let rendered = render_inline_md("`src/main.rs：42` 和 $https://x.com，a$", "");
        assert!(rendered.contains("src/main.rs：42"));
        assert!(rendered.contains("https://x.com，a"));
    }

    #[test]
    fn cjk_punctuation_before_clickable_target_is_normalized() {
        assert_eq!(
            normalize_cjk_punct_around_path(
                "调用时机： `app.py:334`，文档：https://example.com/guide。"
            ),
            "调用时机: `app.py:334`,文档:https://example.com/guide."
        );
    }

    #[test]
    fn cjk_punctuation_without_clickable_target_stays_literal() {
        let text = "时间：12点，普通说明。公式：$https://x.com$。";
        assert_eq!(normalize_cjk_punct_around_path(text), text);
    }
}
