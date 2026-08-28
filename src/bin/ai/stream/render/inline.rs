use crate::ai::stream::render::code::{MONOKAI_BG, MONOKAI_FG};

/// Terminals by default render East-Asian **Ambiguous** width characters (arrows `→`, math symbols `× ± ≤ ≥ ≠`,
/// box-drawing, braille, etc.) as single-column; only true Wide/fullwidth characters (CJK, etc.) take 2 columns.
/// `width_cjk` counts every ambiguous character as 2 columns, shifting table borders and cursor-up heights overall
/// (especially dragging right borders of cells containing `→` off by a line). So this module consistently uses `width` (ambiguous=1).
pub(super) fn terminal_cell_width(ch: char) -> usize {
    // The emoji variation selector (U+FE0F) has width 0 itself, but it forces the preceding base symbol from
    // text presentation (1 column) into emoji presentation (2 columns). Real terminals render it that way; e.g. `⚠️`
    // (U+26A0 + U+FE0F) takes 2 columns. Counting it as unicode-width's 0 would make tables/previews one column
    // narrower than the terminal actually shows, causing hard line wraps, under-counted cursor-up erase lines, and stale
    // header residue stacking up. Here VS16 is counted as 1 column, giving the base the extra column it expands into.
    if ch == '\u{fe0f}' {
        return 1;
    }
    if is_single_width_terminal_symbol(ch) {
        return 1;
    }
    // Modern macOS terminals render Miscellaneous Symbols (U+2600-U+26FF),
    // Miscellaneous Technical (U+2300-U+23FF), Dingbats (U+2700-U+27BF),
    // and the ambiguous-width symbols in some blocks such as Geometric Shapes rating/up-down markers (△▽▲▼)
    // as emoji at 2 columns. unicode-width
    // returns 1 (ambiguous) for these characters, but the terminal actually uses 2. Without a fix, cells containing ⚠ ☎ ✂
    // or `4.0→4.0 △`-style markers get their right border dragged off line by line.
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
            // The previous char is already an emoji-block character (counted as 2 columns); VS16 adds no extra width
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

/// Whether the character belongs to a Unicode block the terminal renders as emoji at 2 columns.
///
/// Characters in these blocks are East Asian Ambiguous width in Unicode (unicode-width returns 1),
/// but modern macOS terminals render them with the Apple Color Emoji font, actually using 2 columns.
/// Inherently Wide emoji (✅ ❌, etc.) already return 2 from unicode-width and need no handling.
fn is_ambiguous_emoji_block_char(ch: char) -> bool {
    let c = ch as u32;
    matches!(
        c,
        // Miscellaneous Technical: ⌚ ⌛ ⏰ etc.
        0x2300..=0x23FF
            // Miscellaneous Symbols: ☀ ☁ ⚠ ☎ ⚡ etc.
            | 0x2600..=0x26FF
            // Dingbats: ✂ ✆ ✈ ✉ ✌ ✍ ✎ ✏ ✓ ✔ ✨ etc.
            | 0x2700..=0x27BF
            // Common rating/up-down triangle markers in Geometric Shapes: △ ▲ ▽ ▼
            | 0x25B2 | 0x25B3 | 0x25BC | 0x25BD
    )
}

/// Convert CJK punctuation (`：` `，` `。`) adjacent to file names / links into their ASCII forms (`:` `,` `.`).
///
/// Agents often emit CJK punctuation glued to file names / line numbers or links, e.g. `src/foo.rs：42`,
/// `调用时机： app.py:334`, `https://x.com，详见`. Fullwidth punctuation keeps the terminal from recognizing
/// file:line / URL boundaries, breaking click-to-jump. Convert directly when the char before the punctuation is common in paths/links;
/// when it follows ordinary Chinese text, convert only if a clickable target actually follows (allowing spaces and inline code markers),
/// avoiding collateral damage to purely Chinese contexts like `时间：12点`.
fn normalize_cjk_punct_around_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev: Option<char> = None;
    let mut i = 0;
    while i < s.len() {
        // code / math spans are Markdown literals; their content must not be rewritten for terminal jump links.
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
        // Inspect the preceding char in its original form so consecutive fullwidth punctuation does not chain-convert (`：：` converts only the first).
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

/// Whether the right side of a fullwidth punctuation mark is immediately followed by a path / URL the terminal can usually click.
///
/// `app.py:334` inside Markdown inline code is still a terminal link target, so its content is recognized; math spans
/// stay literal and do not participate. Only targets with a path separator, file extension, or URL scheme are accepted,
/// so a number after ordinary Chinese like `时间：12点` is never mistaken for a file.
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
        .find_map(|(offset, ch)| {
            (!ch.is_ascii() || ch.is_ascii_whitespace()).then_some(start + offset)
        })
        .unwrap_or(s.len());
    is_clickable_terminal_target(&s[start..end])
}

fn is_clickable_terminal_target(token: &str) -> bool {
    let token = token.trim_end_matches(['.', ',', ';', ':', ')', ']']);
    if token.starts_with("https://") || token.starts_with("http://") || token.starts_with("file://")
    {
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
    // In the new implementation the math state is briefly true only inside the local paired-success branch; it is always false outside the loop.
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
        // Backtick `code`: open the style only when the paired closing backtick is found, otherwise emit literal characters.
        // The old implementation blindly toggled `code = !code`; if the model emitted "use `cargo run to test"
        // (a single unclosed backtick), the whole rest of the line got the code-block background.
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

        // **bold**: also requires pairing. Models often emit unclosed "**Note:" or "5 ** 3",
        // which the old implementation bolded for the entire following span.
        if !code && !math && bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            if let Some(close) = find_unescaped_delim(s, i + 2, "**") {
                let content = &s[i + 2..close - 2];
                bold = true;
                apply_style(&mut out, base, bold, italic, code, math);
                // bold interiors may still contain italic / code; recursion keeps nested styles correct.
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

        // *italic*: same as above. "5 * 3 = 15" must not trigger italic.
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

        // \(math\) inline math: requires pairing, handled the same as $...$.
        if !code && !math && bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            if let Some(close) = find_unescaped_delim(s, i + 2, "\\)") {
                let content = &s[i + 2..close - 2];
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
            // Unpaired: emit literally
            out.push('\\');
            i += 1;
            continue;
        }

        // $math$ / $$display$$: requires pairing, so lone $ characters like "$5" or "$PATH" are not
        // mistaken for math starts that render the rest of the line as LaTeX.
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
            // Unpaired: emit literally
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
        // \(math\) inline math
        if !code && !math && bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            if let Some(end) = find_unescaped_delim(s, i + 2, "\\)") {
                out.push_str(&crate::ai::stream::render_math_tex_to_unicode(
                    &s[i + 2..end - 2],
                ));
                i = end;
                continue;
            }
            // Unpaired: emit the original characters
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
    // `\(...\)` may contain inseparable TeX control words like `\alpha`. Render the whole run to
    // Unicode first, then wrap by terminal width, so generic character splitting cannot cut a command into `\alp` / `ha`.
    let rendered_inner;
    let inner = if prefix == "\\(" {
        rendered_inner = crate::ai::stream::render_math_tex_to_unicode(inner);
        rendered_inner.as_str()
    } else {
        inner
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
    if s.starts_with("\\(") && s.ends_with("\\)") && s.len() >= 4 {
        return Some((&s[..2], &s[2..s.len() - 2], &s[s.len() - 2..]));
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

    // \(math\) inline math
    if rest.starts_with("\\(") {
        let end = find_unescaped_delim(s, start + 2, "\\)")?;
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
        // Arrows / math symbols are East-Asian Ambiguous width: terminals render them at 1 column.
        // Counting them as 2 columns via width_cjk would drag right borders of cells containing `→` off line by line.
        for ch in ['→', '←', '↔', '×', '±', '≤', '≥', '≠', '∈', '⊂'] {
            assert_eq!(
                terminal_cell_width(ch),
                1,
                "ambiguous-width symbol {ch:?} must render as a single terminal column"
            );
        }
        // True fullwidth / CJK characters still take 2 columns and are unaffected.
        for ch in ['中', '文', '你', '好'] {
            assert_eq!(
                terminal_cell_width(ch),
                2,
                "CJK char {ch:?} stays double-width"
            );
        }
        // A result cell containing an arrow: 3 visible chars (→ space x) should be 3 columns, not 4.
        assert_eq!(terminal_display_width("→ x"), 3);
        // Ambiguous emoji-block characters (⚠ ☎ ✂) render at 2 columns in modern terminals.
        for ch in ['⚠', '☎', '✂', '☀', '✈'] {
            assert_eq!(
                terminal_cell_width(ch),
                2,
                "emoji-block symbol {ch:?} must render as double width"
            );
        }
        // Emoji-block char + digit = 3 columns (emoji 2 + digit 1).
        assert_eq!(terminal_display_width("⚠1"), 3);
    }

    #[test]
    fn terminal_width_counts_emoji_presentation_as_double_width() {
        // A symbol with an emoji variation selector (U+FE0F) takes 2 columns via emoji presentation in real terminals.
        // `⚠️` = U+26A0 + U+FE0F: base is ambiguous(1) + VS16(adds 1) = 2 columns.
        assert_eq!(terminal_display_width("⚠️"), 2);
        // Modern macOS terminals render Miscellaneous Symbols block characters as emoji at 2 columns,
        // even without VS16. ⚠ (U+26A0) belongs to this block.
        assert_eq!(terminal_display_width("⚠"), 2);
        // Characters with inherent emoji presentation (unicode-width says 2) are unaffected.
        assert_eq!(terminal_display_width("✅"), 2);
        assert_eq!(terminal_display_width("❌"), 2);
        // Up-down / rating triangle markers also take 2 columns in macOS terminals.
        assert_eq!(terminal_display_width("△"), 2);
        assert_eq!(terminal_display_width("▲"), 2);
        assert_eq!(terminal_display_width("▽"), 2);
        assert_eq!(terminal_display_width("▼"), 2);
        // A lone VS16 contributes 1 column (equivalent to giving the adjacent base the column it expands into).
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
        // Unclosed backtick: must be emitted as a literal character, never opening the code background.
        let rendered = render_inline_md("use `cargo to test", "");
        assert!(!rendered.contains(MONOKAI_BG));
        assert!(!rendered.contains(MONOKAI_FG));
        assert!(rendered.contains("`cargo to test"));
    }

    #[test]
    fn unclosed_asterisk_does_not_italicize_rest_of_line() {
        // "5 * 3 = 15": a single * must not trigger italic; the output must not contain \x1b[3m.
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
        // "$5 USD" must not be taken as a math start; the output must not trigger the math color \x1b[95m.
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

    #[test]
    fn paren_math_renders_inline() {
        // \(x_1\) should be recognized as inline math with the subscript rendered
        let rendered = render_inline_md(r"其中 \(x_1\) 是变量", "");
        assert!(rendered.contains("x₁"), "got: {rendered}");
        assert!(
            !rendered.contains("\\("),
            "markers should be consumed: {rendered}"
        );
    }

    #[test]
    fn paren_math_in_markdown_table_cell() {
        // \(...\) inside a table cell is also handled by strip_inline_md_markers
        let stripped = strip_inline_md_markers(r"公式 \(NCF_0=-8600\)");
        assert!(stripped.contains("NCF₀=-8600"), "got: {stripped}");
        assert!(!stripped.contains("\\("), "got: {stripped}");
    }

    #[test]
    fn unpaired_paren_math_is_literal() {
        // An unpaired \( must be emitted as literal text
        let rendered = render_inline_md(r"半公式 \(x_1", "");
        assert!(rendered.contains(r"\(x_1"), "got: {rendered}");
    }

    #[test]
    fn long_paren_math_keeps_balanced_markers_when_wrapped() {
        let wrapped = wrap_md_cell(r"\(\alpha\alpha\alpha\alpha\alpha\)", 4);
        assert!(wrapped.len() > 1, "got: {wrapped:?}");
        assert!(
            wrapped
                .iter()
                .all(|line| line.starts_with(r"\(") && line.ends_with(r"\)")),
            "got: {wrapped:?}"
        );
        let rendered = wrapped
            .iter()
            .map(|line| render_inline_md(line, ""))
            .collect::<String>();
        assert_eq!(rendered.matches('α').count(), 5, "got: {rendered}");
        assert!(!rendered.contains("alp"), "got: {rendered}");
    }
}
