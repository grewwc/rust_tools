    use super::*;
    use crate::ai::stream::render::inline::terminal_display_width;
    use crate::ai::test_support::ENV_LOCK;

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn strip_ansi_for_test(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out = String::with_capacity(s.len());
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                i += 2;
                while i < bytes.len() {
                    let b = bytes[i];
                    i += 1;
                    if (b as char) >= '@' && (b as char) <= '~' {
                        break;
                    }
                }
                continue;
            }
            let Some(ch) = s[i..].chars().next() else {
                break;
            };
            if ch != '\r' {
                out.push(ch);
            }
            i += ch.len_utf8();
        }
        out
    }

    #[test]
    fn prose_palette_applies_after_list_and_quote_markers() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        for (source, expected) in [
            ("正文", "正文\n"),
            ("1. 正文", "1. 正文\n"),
            ("  - 正文", "  • 正文\n"),
            ("- [ ] 正文", "○ 正文\n"),
            ("- [x] 正文", "✓ 正文\n"),
        ] {
            let out = renderer.consume_line(source, false);
            assert_eq!(strip_ansi_for_test(&out), expected);
            assert!(out.ends_with(&format!("{}正文\x1b[0m\n", theme::current().markdown_body)));
        }
        let quote = renderer.consume_line("> 引用", false);
        assert_eq!(strip_ansi_for_test(&quote), "▍ 引用\n");
        assert!(quote.contains(&format!("{}引用", theme::current().accent_muted)));
    }

    #[test]
    fn headings_use_a_warm_hierarchy_without_changing_structure() {
        for level in 1..=6 {
            let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
            renderer.bol = true;
            let out = renderer.consume_line(&format!("{} 标题", "#".repeat(level)), false);
            assert!(out.contains(&format!(
                "\x1b[1m{}标题",
                theme::current().markdown_heading
            )));
            let expected = match level {
                1 => "标题\n━━━\n",
                2 => "标题\n───\n",
                _ => "标题\n",
            };
            assert_eq!(strip_ansi_for_test(&out), expected);
        }
    }

    #[test]
    fn streaming_preview_and_repaint_share_body_color_including_buffered_prefixes() {
        for dimmed in [false, true] {
            for source in ["正文", "  正文", "$PATH"] {
                let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
                let mut preview = String::new();
                for ch in source.chars() {
                    preview.push_str(
                        &renderer
                            .write_chunk_for_test(&ch.to_string(), dimmed)
                            .unwrap(),
                    );
                }
                let base = if dimmed {
                    theme::current().accent_muted
                } else {
                    theme::current().markdown_body
                };
                assert_eq!(preview, format!("{base}{source}"));
                let repaint = renderer.flush_pending_for_test().unwrap();
                assert!(repaint.contains(&base));
                assert!(repaint.ends_with("\x1b[0m\n"));
                assert!(renderer.flush_pending_for_test().unwrap().is_empty());
            }
        }
    }

    #[test]
    fn prose_palette_does_not_replace_code_highlighting_or_status_colors() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.consume_line("```rust", false);
        assert_eq!(
            renderer.consume_line("let x = 1;", false),
            format!(
                "{}{}\x1b[0m\n",
                theme::current().code_background,
                highlight_code_line("let x = 1;", Some("rust"))
            )
        );
        let status = renderer.consume_line(END_THINKING_TAG_TEXT, false);
        assert_eq!(status, format!("{}✓ thinking\x1b[0m\n", theme::current().accent_muted));
    }

    #[test]
    fn consume_line_move_up_matches_preview_height() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "6") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.set_line_preview_height(1);
        let out = renderer.consume_line("**hello**", true);
        assert!(out.contains("\x1b[1A\r\x1b[0J"));
        assert!(!out.contains("\x1b[2A\r\x1b[0J"));
    }

    #[test]
    fn test_write_chunk_preview_height() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "10") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.write_chunk_for_test("123456789012345", false);
        let first_height = renderer.line_preview_height();
        assert!(first_height >= 1);

        let _ = renderer.write_chunk_for_test("678901", false);
        assert!(renderer.line_preview_height() >= first_height);
    }

    #[test]
    fn test_pending_header_restore() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let _ = renderer.consume_line("| Header A | Header B |", false);
        let _ = renderer.consume_line("Not a separator", false);
    }

    #[test]
    fn plain_body_is_emitted_before_stream_finish_and_not_repeated_on_flush() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let streamed = renderer
            .write_chunk_for_test("正文实时输出\n", false)
            .unwrap();
        let mut grid = VtGrid::new(80);
        grid.feed(&streamed);
        assert_eq!(
            grid.screen()
                .into_iter()
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>(),
            vec!["正文实时输出".to_string()],
            "plain body must already be visible before stream finish"
        );

        let flushed = renderer.flush_pending_for_test().unwrap();
        grid.feed(&flushed);
        assert_eq!(
            grid.screen()
                .into_iter()
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>(),
            vec!["正文实时输出".to_string()],
            "plain body must not appear twice when the stream finishes"
        );
    }

    #[test]
    fn pending_header_flush_rewrites_to_plain_markdown_line() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "80") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.bol = true;

        // 新契约：疑似表头行进入缓冲并显示单行占位提示（不 echo 原始 markdown）。
        let first = renderer
            .write_chunk_for_test("| **Header** | Value |\n", false)
            .unwrap();
        assert!(
            strip_ansi_for_test(&first).contains("生成表格中"),
            "suspected header should emit the placeholder; got {first:?}"
        );

        // 流结束仍未等到分隔行——它只是含 `|` 的普通文本。flush 先用确定的 1 行
        // cursor-up 清掉占位，再按普通行落地；除占位清除外无任何 cursor-up 重写。
        let flushed = renderer.flush_pending_for_test().unwrap();
        assert!(
            flushed.starts_with("\x1b[1A\r\x1b[0J"),
            "flush must first clear the 1-line placeholder; got {flushed:?}"
        );
        let after_clear = &flushed["\x1b[1A\r\x1b[0J".len()..];
        assert!(
            !after_clear.contains("A\r\x1b[0J"),
            "buffered plain line must not use table-height cursor-up rewrite; got {flushed:?}"
        );

        let visible = strip_ansi_for_test(&flushed);
        assert_eq!(visible, "| Header | Value |\n");
    }

    #[test]
    fn trailing_blank_lines_are_dropped_on_flush() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let stream = render_full_stream("done\n\n\n", false);
        let mut grid = VtGrid::new(40);
        grid.feed(&stream);
        let non_empty: Vec<String> = grid
            .screen()
            .into_iter()
            .filter(|line| !line.is_empty())
            .collect();
        assert_eq!(non_empty, vec!["done".to_string()]);
    }

    #[test]
    fn interior_blank_lines_are_preserved_when_content_follows() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let stream = render_full_stream("first\n\nsecond\n", false);
        let mut grid = VtGrid::new(40);
        grid.feed(&stream);
        let screen = grid.screen();
        let trimmed: Vec<&String> = screen
            .iter()
            .rev()
            .skip_while(|line| line.is_empty())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let joined: Vec<&str> = trimmed.iter().map(|s| s.as_str()).collect();
        assert_eq!(joined, vec!["first", "", "second"]);
    }

    #[test]
    fn code_block_keeps_inner_indentation_without_visible_gutter() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```rust", false);
        let out = renderer.consume_line("    let x = 1;", false);

        let visible = strip_ansi_for_test(&out);
        assert_eq!(visible, "    let x = 1;\n");
    }

    #[test]
    fn code_block_nested_indent_is_stable() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("  ```rust", false);
        let out = renderer.consume_line("      let x = 1;", false);

        let visible = strip_ansi_for_test(&out);
        assert_eq!(visible, "      let x = 1;\n");
    }

    #[test]
    fn task_list_uses_minimal_markers_instead_of_emoji() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let checked = renderer.consume_line("- [x] done", false);
        let unchecked = renderer.consume_line("- [ ] todo", false);

        let checked_visible = strip_ansi_for_test(&checked);
        let unchecked_visible = strip_ansi_for_test(&unchecked);
        assert!(checked_visible.contains("✓ done"));
        assert!(unchecked_visible.contains("○ todo"));
        assert!(!checked_visible.contains("✅"));
        assert!(!unchecked_visible.contains("⬜"));
    }

    #[test]
    fn blockquote_and_rule_render_with_cleaner_structure() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let quote = renderer.consume_line("> note", false);
        let rule = renderer.consume_line("---", false);

        let quote_visible = strip_ansi_for_test(&quote);
        let rule_visible = strip_ansi_for_test(&rule);
        assert!(quote_visible.contains("▍ note"));
        assert!(rule_visible.contains(&"─".repeat(28)));
    }

    #[test]
    fn thinking_markers_render_cleanly_without_leaking_ansi_bytes() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let start = renderer.consume_line(THINKING_TAG_TEXT, false);
        let end = renderer.consume_line(END_THINKING_TAG_TEXT, false);

        let start_visible = strip_ansi_for_test(&start);
        let end_visible = strip_ansi_for_test(&end);
        assert_eq!(start_visible, "○ thinking\n");
        assert_eq!(end_visible, "✓ thinking\n");
    }

    #[test]
    fn inner_info_fence_is_content_not_closer() {
        // Mirrors a model dump that nests fences: an outer ```text block wrapping
        // inner ```think / ```dependency / ```python fences. CommonMark closers
        // carry no info string, so inner fence lines stay content of the open
        // block; only bare ``` lines are boundaries.
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        assert!(renderer.consume_line("```text", false).contains("╭─ text"));
        let inner = renderer.consume_line("```think", false);
        // Syntax highlighting injects ANSI codes inside the backtick run, so
        // assert on behavior (no border, block still open) not on the literal.
        assert!(!inner.contains("╰"), "inner info fence must not close: {inner}");
        assert_eq!(renderer.code_block_lang(), Some("text"));
        assert!(renderer.consume_line("```", false).contains("╰"));
        assert_eq!(renderer.code_block_lang(), None);
        // The next info-fence line opens its own block as usual.
        assert!(renderer.consume_line("```python", false).contains("╭─ python"));
    }

    #[test]
    fn backtick_fence_with_backticks_in_info_is_not_a_fence() {
        // A paragraph line like ````think`/```dependency`/```python` (4-backtick
        // run, backticks in the would-be info string) is not a fence per
        // CommonMark: it must render as ordinary text and never open a code box
        // that swallows the rest of the message.
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let out = renderer.consume_line("````think`/```dependency`/```python` 各 1 次", false);
        assert!(!out.contains("╭─"), "must not open a code block: {out}");
        assert_eq!(renderer.code_block_lang(), None);
        // Following markdown keeps rendering as markdown (not code content).
        let heading = renderer.consume_line("## 结论", false);
        assert!(heading.contains("结论"), "heading must stay a heading: {heading}");
        assert!(!heading.contains("╭─"), "heading must not enter a code box: {heading}");
        assert_eq!(renderer.code_block_lang(), None);
    }

    #[test]
    fn shorter_bare_fence_does_not_close_longer_fence() {
        // ```` opens a run-4 fence; a bare ``` inside is content (CommonMark:
        // the closer's run must be at least as long as the opener's).
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        assert!(renderer.consume_line("````text", false).contains("╭─ text"));
        let inner = renderer.consume_line("```", false);
        assert!(!inner.contains("╰"), "3-backtick run must not close: {inner}");
        assert_eq!(renderer.code_block_lang(), Some("text"));
        assert!(renderer.consume_line("````", false).contains("╰"));
        assert_eq!(renderer.code_block_lang(), None);
    }

    #[test]
    fn nested_fence_dump_keeps_content_inside_boxes() {
        // Serializes against tests that mutate COLUMNS: a narrow width wraps the
        // fixture lines and breaks the substring-position assertions below.
        let _guard = env_guard();
        // End-to-end shape of the misaligned replay: outer ```text wrapping
        // inner fences, each with content, closed by bare fences. Content must
        // stay between its block's ╭─ / ╰ borders instead of leaking outside.
        let text = concat!(
            "```text\n",
            "```think\n",
            "用户输入仅为问候\n",
            "```\n",
            "\n",
            "```dependency\n",
            "{}\n",
            "```\n",
            "\n",
            "```python\n",
            "return \"hi\"\n",
            "```\n",
        );
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let out = renderer.write_block_for_test(text, false).unwrap();
        let text_open = out.find("╭─ text").expect("text block opens");
        let para = out.find("用户输入仅为问候").expect("paragraph present");
        let first_close = out.find("╰").expect("text block closes");
        assert!(
            text_open < para && para < first_close,
            "paragraph must render inside the text box"
        );
        let dep_open = out.find("╭─ dependency").expect("dependency opens");
        let brace = out.find("{}").expect("content present");
        let dep_close = out[first_close + "╰".len()..]
            .find("╰")
            .map(|i| i + first_close + 1)
            .expect("dependency closes");
        assert!(dep_open < brace && brace < dep_close);
        let py_open = out.find("╭─ python").expect("python opens");
        // Keyword coloring may split `return "hi"` with ANSI codes; the token
        // itself stays contiguous.
        let ret = out.find("return").expect("return present");
        let last_close = out.rfind("╰").expect("python closes");
        assert!(py_open < ret && ret < last_close);
        assert_eq!(renderer.code_block_lang(), None);
    }

    #[test]
    fn unclosed_fence_swallows_rest_like_commonmark() {
        // A stray ``` that opens a block with no closer keeps everything after
        // it as code content — the same behavior as CommonMark renderers (e.g.
        // GitHub). The real fix for such source is the author's; the renderer
        // stays spec-faithful.
        let text = "done\n\n```\nstill code\n";
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let out = renderer.write_block_for_test(text, false).unwrap();
        let open = out.find("╭─ code").expect("stray fence opens");
        let body = out.find("still").expect("tail present");
        assert!(open < body);
        assert!(!out.contains("╰"), "no closer follows: {out}");
        // A bare ``` stores lang None; the "code" label is display-only, and the
        // open-marker/no-border ordering above already proves the block is open.
    }

    #[test]
    fn thinking_marker_breaks_out_of_code_block_rendering() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let _ = renderer.consume_line("```rust", false);
        let out = renderer.consume_line(END_THINKING_TAG_TEXT, false);

        let visible = strip_ansi_for_test(&out);
        assert_eq!(visible, "✓ thinking\n");
    }

    #[test]
    fn unfinished_line_state_tracks_pending_inline_content() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        renderer.write_chunk_for_test("hello", false).unwrap();
        assert!(renderer.has_unfinished_line());

        renderer.write_chunk_for_test("\n", false).unwrap();
        assert!(!renderer.has_unfinished_line());
    }

    #[test]
    fn html_code_block_preview_height_matches_plain_width_without_gutter() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "7") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);

        let _ = renderer.consume_line("```html", false);
        renderer.write_chunk_for_test("<x>", false).unwrap();

        assert!(
            renderer.line_preview_height() == live_preview_cursor_rows("<x>"),
            "code-block preview height should match the visible code width when gutter is hidden"
        );
    }

    #[test]
    fn html_code_block_line_remains_visible_after_rewrite() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "16") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```html", false);
        let out = renderer.consume_line(r#"<div class="input-area"></div>"#, true);

        let visible = strip_ansi_for_test(&out);
        let flattened = visible.replace('\n', "");
        assert_eq!(flattened, r#"<div class="input-area"></div>"#);
        assert!(visible.lines().count() > 1, "{visible:?}");
        assert!(!visible.contains('│'), "{visible:?}");
    }

    #[test]
    fn live_preview_cursor_rows_counts_exact_width_cjk_boundary() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "20") };
        assert_eq!(live_preview_cursor_rows("你好你好你好你好你好"), 1);
    }

    #[test]
    fn live_preview_cursor_rows_counts_box_drawing_as_single_width() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "20") };

        assert_eq!(live_preview_cursor_rows("─".repeat(20).as_str()), 1);
        assert_eq!(live_preview_cursor_rows("─".repeat(21).as_str()), 2);
    }

    #[test]
    fn exact_width_cjk_preview_updates_renderer_height_to_one_row() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "20") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer
            .write_chunk_for_test("你好你好你好你好你好", true)
            .unwrap();

        assert_eq!(renderer.line_preview_height(), 1);
    }

    #[test]
    fn exact_width_cjk_rewrite_moves_up_one_row() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "20") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.set_line_preview_height(1);

        let out = renderer.consume_line("你好你好你好你好你好", true);
        assert!(out.contains("\x1b[1A\r\x1b[0J"), "{out:?}");
        assert!(!out.contains("\x1b[2A\r\x1b[0J"), "{out:?}");
    }

    #[test]
    fn code_block_long_line_wraps_without_gutter_prefix() {
        let _guard = env_guard();
        // preview_terminal_width 引入 RIGHT_MARGIN=4 与 MIN_PREVIEW_WIDTH=20
        // 之后，最小有效宽度是 20。这里给 30 列模拟终端宽度，扣除右边距得到
        // 26 列内容宽度（block_indent="" + gutter 1 列），实际 wrap 阈值约 25。
        unsafe { std::env::set_var("COLUMNS", "30") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```text", false);

        // 30 个字符确保超过当前内容宽度，会被强制 wrap 至两行。
        let out = renderer.consume_line(&"a".repeat(30), false);
        let visible = strip_ansi_for_test(&out);
        let lines = visible.lines().collect::<Vec<_>>();

        assert!(
            lines.len() >= 2,
            "expected wrap to >=2 lines, got {lines:?}"
        );
        // 拼回所有行后字符总数仍应等于原输入（仅 wrap，不应丢字符）。
        let joined: String = lines.concat();
        assert_eq!(joined, "a".repeat(30));
    }

    #[test]
    fn code_block_streaming_preview_wraps_before_newline() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "30") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("```text", false);

        renderer
            .write_chunk_for_test(&"a".repeat(30), false)
            .unwrap();

        assert!(
            renderer.line_preview_height() >= 2,
            "expected preview height >=2, got {}",
            renderer.line_preview_height()
        );
    }

    #[test]
    fn optional_code_block_gutter_still_renders_line_numbers() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.set_show_line_gutter(true);
        let _ = renderer.consume_line("```rust", false);

        let out = renderer.consume_line("let x = 1;", false);
        let visible = strip_ansi_for_test(&out);

        assert!(visible.starts_with("  1 │let x = 1;"), "{visible:?}");
    }

    #[test]
    fn nested_code_block_streaming_preview_does_not_double_emit_block_indent() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "40") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("  ```rust", false);

        let streamed = renderer
            .write_chunk_for_test("      if len", false)
            .unwrap();
        let visible = strip_ansi_for_test(&streamed);

        assert!(
            visible.starts_with("      if len"),
            "streamed preview should render the source line verbatim once, got {visible:?}"
        );
        assert!(
            !visible.starts_with("            "),
            "block_indent must not be emitted twice during realtime preview, got {visible:?}"
        );
        assert_eq!(renderer.line_preview_height(), 1);
    }

    #[test]
    fn nested_code_block_preview_height_matches_final_render_height() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "14") };
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let _ = renderer.consume_line("  ```rust", false);

        renderer
            .write_chunk_for_test("      if len_is_long", false)
            .unwrap();
        let streamed_height = renderer.line_preview_height();

        let final_out = renderer.consume_line("      if len_is_long", true);
        let move_up = format!("\x1b[{streamed_height}A\r\x1b[0J");
        assert!(
            final_out.contains(&move_up),
            "final rewrite must move the cursor up by exactly the streamed preview height \
             ({streamed_height}); got {final_out:?}"
        );
    }

    /// Minimal VT100 grid for streaming previews and cursor-up repainting.
    /// Handles CJK widths, CRLF, cursor-up, erase-to-end, and skips complete SGR
    /// sequences including truecolor parameters. Wide characters wrap early if
    /// only one column remains, matching terminal auto-wrap behavior.
    struct VtGrid {
        width: usize,
        rows: Vec<Vec<char>>,
        row: usize,
        col: usize,
    }

    impl VtGrid {
        fn new(width: usize) -> Self {
            Self {
                width,
                rows: vec![vec![' '; width]],
                row: 0,
                col: 0,
            }
        }

        fn ensure_row(&mut self, r: usize) {
            while self.rows.len() <= r {
                self.rows.push(vec![' '; self.width]);
            }
        }

        fn newline(&mut self) {
            self.row += 1;
            self.col = 0;
            self.ensure_row(self.row);
        }

        fn put(&mut self, ch: char) {
            let w = terminal_cell_width(ch);
            if w == 0 {
                return;
            }
            if self.col + w > self.width {
                self.newline();
            }
            self.ensure_row(self.row);
            self.rows[self.row][self.col] = ch;
            for k in 1..w {
                if self.col + k < self.width {
                    self.rows[self.row][self.col + k] = '\0';
                }
            }
            self.col += w;
        }

        fn feed(&mut self, s: &str) {
            let mut chars = s.chars().peekable();
            while let Some(ch) = chars.next() {
                match ch {
                    '\n' => self.newline(),
                    '\r' => self.col = 0,
                    '\x1b' => {
                        if chars.peek() == Some(&'[') {
                            chars.next();
                            let mut num = String::new();
                            let mut final_byte = '\0';
                            for c in chars.by_ref() {
                                if ('@'..='~').contains(&c) {
                                    final_byte = c;
                                    break;
                                }
                                num.push(c);
                            }
                            let n: usize = num.parse().unwrap_or(0);
                            match final_byte {
                                'A' => self.row = self.row.saturating_sub(n.max(1)),
                                'J' => {
                                    // 0J: clear from the cursor to the end of the screen.
                                    for c in self.col..self.width {
                                        self.rows[self.row][c] = ' ';
                                    }
                                    let start = self.row + 1;
                                    self.rows.truncate(start.max(1));
                                    self.ensure_row(self.row);
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => self.put(ch),
                }
            }
        }

        fn screen(&self) -> Vec<String> {
            self.rows
                .iter()
                .map(|r| {
                    r.iter()
                        .filter(|c| **c != '\0')
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                })
                .collect()
        }

        fn border_columns_by_row(&self) -> Vec<Vec<usize>> {
            self.rows
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .filter_map(|(idx, ch)| {
                            matches!(
                                ch,
                                '┌' | '┬' | '┐' | '├' | '┼' | '┤' | '└' | '┴' | '┘' | '│'
                            )
                            .then_some(idx)
                        })
                        .collect()
                })
                .collect()
        }
    }

    fn render_full_stream(markdown: &str, dimmed: bool) -> String {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let mut bytes = String::new();
        for ch in markdown.chars() {
            let mut buf = [0u8; 4];
            bytes.push_str(
                &renderer
                    .write_chunk_for_test(ch.encode_utf8(&mut buf), dimmed)
                    .unwrap(),
            );
        }
        bytes.push_str(&renderer.flush_pending_for_test().unwrap());
        bytes
    }

    fn render_full_block(markdown: &str, dimmed: bool) -> String {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let mut bytes = renderer.write_block_for_test(markdown, dimmed).unwrap();
        bytes.push_str(&renderer.flush_pending_for_test().unwrap());
        bytes
    }

    #[test]
    fn long_review_keeps_visible_text_and_wrapping_in_stream_and_block_output() {
        let _guard = env_guard();
        let markdown = "1. **[P1] `read_file` 仍会读取整行，长内容需要限制内存。**\n\n   证据: `src/bin/ai/tools/service/file.rs:42`，建议分块读取并保留完整内容。\n\n2. **[P2] 嵌套样式** 后的正文不应变亮。";
        for cols in [24, 80] {
            unsafe { std::env::set_var("COLUMNS", cols.to_string()) };
            let mut streamed = VtGrid::new(cols);
            streamed.feed(&render_full_stream(markdown, false));
            let mut blocked = VtGrid::new(cols);
            blocked.feed(&render_full_block(markdown, false));
            // Streamed mode may defer paragraph spacing until it sees the next
            // content line. Compare nonempty rows so this test isolates the
            // invariant relevant to palette changes: ANSI colors cannot alter
            // visible text or terminal wrapping.
            let streamed_rows = streamed
                .screen()
                .into_iter()
                .filter(|row| !row.is_empty())
                .collect::<Vec<_>>();
            let blocked_rows = blocked
                .screen()
                .into_iter()
                .filter(|row| !row.is_empty())
                .collect::<Vec<_>>();
            assert_eq!(streamed_rows, blocked_rows, "terminal width: {cols}");
            let text = blocked_rows.join("");
            assert!(text.contains("read_file"));
            assert!(text.contains("src/bin/ai/tools/service/file.rs:42"));
            assert!(!text.contains("**"));
            assert!(!text.contains("38;2;"));
        }
    }

    #[test]
    fn streamed_cjk_table_has_no_residual_fragments_on_screen() {
        let _guard = env_guard();

        let md = "\
下面是一个对照表，用来说明不同协议的默认 Endpoint 与典型后端服务的对应关系：
| 协议 | 默认 Endpoint | 典型后端 |
| --- | --- | --- |
| Compatible | dashscope.aliyuncs.com/compatible-mode | 阿里云百炼（DashScope）的 OpenAI 兼容模式 |
| OpenAi | api.openai.com/v1/chat/completions | OpenAI 官方 API |
";
        let para = "下面是一个对照表，用来说明不同协议的默认 Endpoint 与典型后端服务的对应关系：";

        for cols in [80usize, 72, 64, 60, 56, 52, 48] {
            unsafe { std::env::set_var("COLUMNS", cols.to_string()) };

            let stream = render_full_stream(md, false);
            let mut grid = VtGrid::new(cols);
            grid.feed(&stream);
            let screen = grid.screen();
            let joined: String = screen.join("");

            // 表格上方的引导段落必须完整保留（不能被 cursor-up 越界清掉）。
            // 终端可能按列宽自动折行，所以拼接所有行后再比对去掉换行的原文。
            let para_joined: String = para.chars().filter(|c| !c.is_whitespace()).collect();
            let screen_joined: String = joined.chars().filter(|c| !c.is_whitespace()).collect();
            assert!(
                screen_joined.contains(&para_joined),
                "COLUMNS={cols}: leading paragraph was clobbered:\n{}",
                screen.join("\n")
            );

            // 表格区域不得残留原始 markdown 预览碎片（裸 `|`，区别于盒框 `│`）。
            // 残留说明 cursor-up 行数与实际渲染行数不一致，预览没被完全覆盖。
            for line in &screen {
                if line.is_empty() || para.contains(line.trim()) {
                    continue;
                }
                assert!(
                    !line.contains('|'),
                    "COLUMNS={cols}: residual raw markdown pipe: {line:?}\nfull screen:\n{}",
                    screen.join("\n")
                );
            }
        }
    }

    #[test]
    fn table_placeholder_shows_while_buffering_then_clears_on_final_screen() {
        let _guard = env_guard();

        let md = "\
前言。
| 协议 | 默认 Endpoint |
| --- | --- |
| OpenAi | api.openai.com |
后记。
";
        for cols in [80usize, 60, 48] {
            unsafe { std::env::set_var("COLUMNS", cols.to_string()) };

            let stream = render_full_stream(md, false);
            // 缓冲期占位提示必须出现在原始输出流里（表格生成期不空窗）。
            assert!(
                stream.contains("生成表格中"),
                "COLUMNS={cols}: placeholder should appear in the raw stream"
            );

            // 但经终端渲染后的最终屏幕上，占位必须被完全清除、不得残留。
            let mut grid = VtGrid::new(cols);
            grid.feed(&stream);
            let screen = grid.screen();
            for line in &screen {
                assert!(
                    !line.contains("生成表格中"),
                    "COLUMNS={cols}: placeholder leaked onto final screen: {line:?}\n{}",
                    screen.join("\n")
                );
            }
            // 成品盒框表与前后文都在。
            let joined = screen.join("\n");
            assert!(joined.contains('│'), "COLUMNS={cols}: missing box table");
            assert!(
                joined.contains("后记"),
                "COLUMNS={cols}: trailing text lost"
            );
        }
    }

    #[test]
    fn table_is_buffered_silently_and_flushed_once_without_cursor_up() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        renderer.bol = true;

        // 新契约：表头行返回单行占位提示；分隔行、数据行静默缓冲返回空。
        let header = renderer.consume_line("| Header | Value |", false);
        assert!(
            strip_ansi_for_test(&header).contains("生成表格中"),
            "header row should emit the single-line placeholder; got {header:?}"
        );
        assert_eq!(renderer.consume_line("| --- | --- |", false), "");
        assert_eq!(renderer.consume_line("| foo | bar |", false), "");

        // 流结束：先用确定的 1 行 cursor-up 清掉占位，再一次性画出成品盒框表。
        // 唯一允许的 cursor-up 是 `\x1b[1A`（占位清除），绝无按表高的多行重写。
        let flushed = renderer.flush_pending_for_test().unwrap();
        assert!(
            flushed.starts_with("\x1b[1A\r\x1b[0J"),
            "flush must first clear the 1-line placeholder deterministically; got {flushed:?}"
        );
        let after_clear = &flushed["\x1b[1A\r\x1b[0J".len()..];
        assert!(
            !after_clear.contains("A\r\x1b[0J"),
            "table itself must be rendered once, not via cursor-up rewrite; got {flushed:?}"
        );
        let visible = strip_ansi_for_test(&flushed);
        assert!(
            visible.contains("Header") && visible.contains("foo") && visible.contains('│'),
            "flushed output must contain the full box-drawn table; got {visible:?}"
        );
    }

    #[test]
    fn heading_followed_by_cjk_table_leaves_no_raw_header_fragment() {
        let _guard = env_guard();

        let md = "\
## 常用选项

| 选项 | 作用 |
| --- | --- |
| -b a | 对所有行编号（包括空行），等价于 `cat -n` |
| -b t | 仅对非空行编号（默认行为） |
";

        for cols in [96usize, 80, 72, 64] {
            unsafe { std::env::set_var("COLUMNS", cols.to_string()) };

            let stream = render_full_stream(md, false);
            let mut grid = VtGrid::new(cols);
            grid.feed(&stream);
            let screen = grid.screen();

            assert!(
                !screen.iter().any(|line| line.contains("| 选项 | 作用 |")),
                "COLUMNS={cols}: raw table header leaked onto final screen:\n{}",
                screen.join("\n")
            );
        }
    }

    #[test]
    fn streamed_single_column_table_renders_as_table() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "64") };

        let md = "\
| 函数签名 |
| --- |
| `processOrder(orderId: string)` |
";
        let stream = render_full_stream(md, false);
        let mut grid = VtGrid::new(64);
        grid.feed(&stream);
        let screen = grid.screen();
        let joined = screen.join("\n");

        assert!(joined.contains('┌'), "{joined}");
        assert!(joined.contains('│'), "{joined}");
        assert!(
            !joined.contains("| 函数签名 |"),
            "raw single-column markdown table leaked:\n{joined}"
        );
    }

    #[test]
    fn table_literal_inside_inline_code_does_not_swallow_the_following_markdown() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "96") };

        // Regression: `<table` inside an inline code span matched like an HTML table opening tag,
        // so every following line was buffered into `html_table_buf` and finally dumped as raw
        // source — the heading kept its `###` and the markdown table below stayed unrendered.
        let md = "\
`html.rs:30` `line.to_lowercase().contains(\"<table\")`：无条件整行小写拷贝分配。

### 子标题

| 优先级 | 位置 |
| --- | --- |
| P0 | normalize.rs:87 |
";

        for (path, stream) in [
            ("stream", render_full_stream(md, false)),
            ("block", render_full_block(md, false)),
        ] {
            let mut grid = VtGrid::new(96);
            grid.feed(&stream);
            let screen = grid.screen().join("\n");

            assert!(
                !screen.contains("### 子标题"),
                "{path}: raw heading leaked:\n{screen}"
            );
            assert!(
                !screen.contains("| 优先级 | 位置 |"),
                "{path}: raw markdown table leaked:\n{screen}"
            );
            assert!(screen.contains("子标题"), "{path}: heading missing:\n{screen}");
            assert!(screen.contains('│'), "{path}: table not rendered:\n{screen}");
        }
    }

    #[test]
    fn html_table_buffering_only_starts_on_a_real_table_tag() {
        for line in [
            "`contains(\"<table\")` 是裸子串匹配",
            "用 `<table>` 标签包裹",
            "table_start = \"<table\"",
        ] {
            let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
            let _ = renderer.consume_line(line, false);
            assert!(
                !renderer.in_html_table,
                "{line:?} must not open an HTML table buffer"
            );
        }

        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let _ = renderer.consume_line("  <table border=1>", false);
        assert!(
            renderer.in_html_table,
            "a real tag must still start buffering"
        );
    }

    #[test]
    fn unclosed_html_table_fallback_keeps_raw_source_and_ends_on_a_fresh_row() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let mut output = renderer
            .write_block_for_test("<table>\n<tr><td>1</td></tr>\n", false)
            .unwrap();
        output.push_str(&renderer.flush_pending_for_test().unwrap());

        let visible = strip_ansi_for_test(&output);
        assert!(
            visible.contains("<table>"),
            "raw source must survive:\n{visible:?}"
        );
        assert!(
            visible.contains("<tr><td>1</td></tr>"),
            "raw rows must survive:\n{visible:?}"
        );
        assert!(
            visible.ends_with('\n'),
            "fallback must end its line so the input box anchors below the output:\n{visible:?}"
        );
    }

    #[test]
    fn block_render_bare_table_without_leading_pipe_uses_box_table() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "96") };

        let md = "\
对，除了模型层面的差异，其他全部一致。我逐项确认过：

维度 | 是否一致 | 说明
--- | --- | ---
tracing/telemetry 上报 | 一致 | 同一批 telemetry publisher
make_llm_call_publisher | 一致 | 同一条 on_call complete callback
";

        let stream = render_full_block(md, false);
        let mut grid = VtGrid::new(96);
        grid.feed(&stream);
        let screen = grid.screen();
        let joined = screen.join("\n");

        assert!(joined.contains('┌'), "{joined}");
        assert!(joined.contains('│'), "{joined}");
        assert!(joined.contains("tracing/telemetry 上报"), "{joined}");
        assert!(
            !joined.contains("维度 | 是否一致 | 说明"),
            "raw bare-table header leaked:\n{joined}"
        );
        assert!(
            !joined.contains("--- | --- | ---"),
            "raw bare-table separator leaked:\n{joined}"
        );
    }

    #[test]
    fn streamed_bare_table_without_leading_pipe_leaves_no_raw_header() {
        // 复现截图 bug：无前导 `|` 的裸表头在流式路径下，行首（不含 `|` 的 token）会先被
        // 逐字 echo，等换行识别成表头时若不回收这些物理行，原始 markdown 会泄漏在成品表
        // 上方。修复后 consume_line 在切 PendingHeader 前用 cursor-up 回收已 echo 的行。
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "96") };

        let md = "\
对，除了模型层面的差异，其他全部一致。我逐项确认过：

维度 | 是否一致 | 说明
--- | --- | ---
tracing/telemetry 上报 | 一致 | 同一批 telemetry publisher
make_llm_call_publisher | 一致 | 同一条 on_call complete callback
";

        let stream = render_full_stream(md, false);
        let mut grid = VtGrid::new(96);
        grid.feed(&stream);
        let screen = grid.screen();
        let joined = screen.join("\n");

        assert!(joined.contains('┌'), "{joined}");
        assert!(joined.contains('│'), "{joined}");
        assert!(joined.contains("tracing/telemetry 上报"), "{joined}");
        // 引导段落必须保留。
        assert!(
            joined.contains("其他全部一致"),
            "leading paragraph clobbered:\n{joined}"
        );
        // 裸表头/分隔行的原始 markdown（裸 `|`）不得残留在屏幕上。
        for line in &screen {
            if line.is_empty() || line.contains('│') {
                continue;
            }
            assert!(
                !line.contains('|'),
                "residual raw markdown pipe in streamed bare table: {line:?}\nfull screen:\n{joined}"
            );
        }
    }

    #[test]
    fn streamed_table_rewrite_clears_terminal_wrapped_previous_render() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "24") };

        let md = "\
        | 时间 | 代码位置 | 动作 |
        | --- | --- | --- |
        | 45.059 | aida/tools/data/df_to_chart.py:620 SimpleDfToChartTool.run | jinja 触发，goal=render_to_chart |
        | 45.060-081 | aeolus_llm/service/ada/renderers/viz_data.py:300 | 推断列类型，自动划分 dimension/metric |
";

        let stream = render_full_stream(md, false);
        let mut grid = VtGrid::new(24);
        grid.feed(&stream);
        let screen = grid.screen();
        let joined = screen.join("\n");
        // 续接列块用 ├ 代替 ┌ 顶边框，所以用 └ 底边框来计数列块数
        let table_count = joined.matches('└').count();
        let expected_table_count = table_column_ranges("", 3).len();

        assert_eq!(
            table_count, expected_table_count,
            "table rewrite should leave exactly the expected split column blocks on screen, got {table_count}, expected {expected_table_count}:\n{joined}"
        );
    }

    #[test]
    fn streamed_table_with_emoji_presentation_does_not_duplicate_header() {
        let _guard = env_guard();

        let md = "\
## 与 master / 线上兼容性评估

| 改动 | 兼容性 | 风险 |
| --- | --- | --- |
| _escape_sql_identifier_inner (v2) | ✅ 向后兼容 | 极低。正常字段名/ID 输出不变，只有包含特殊字符时才转义 |
| parse_where_clause | ✅ 向后兼容 | 低。仅在解析失败时回退，不影响正常路径 |
| build_query 组装逻辑 | ⚠️ 需回归 | 中。列顺序变化可能影响依赖隐式顺序的调用方 |
";
        for cols in [
            120usize, 110, 100, 96, 90, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40,
        ] {
            unsafe { std::env::set_var("COLUMNS", cols.to_string()) };
            let stream = render_full_stream(md, false);
            let mut grid = VtGrid::new(cols);
            grid.feed(&stream);
            let screen = grid.screen();
            let joined = screen.join("\n");

            let header_count = screen
                .iter()
                .filter(|line| {
                    line.contains("改动") && line.contains("兼容性") && line.contains("风险")
                })
                .count();
            let top_border_count = joined.matches('┌').count();
            let expected_tables = table_column_ranges("", 3).len();
            assert_eq!(
                header_count, expected_tables,
                "COLUMNS={cols}: header should appear once per split table (expected {expected_tables}), got {header_count}:\n{joined}"
            );
            assert_eq!(
                top_border_count, expected_tables,
                "COLUMNS={cols}: expected {expected_tables} top borders, got {top_border_count}:\n{joined}"
            );
        }
    }

    #[test]
    fn streamed_comparison_table_keeps_border_columns_aligned() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "96") };

        let md = "\
| | Skill 路由（已移除） | Agent 路由（已移除） |
| --- | --- | --- |
| 代码机制 | skill_runtime.rs -> prepare_skill_for_turn 仅保留显式 activate_skill | agent_routing.rs 自动路由已整体移除，agent 由显式 agent 字段或默认路由选择 |
";

        let stream = render_full_stream(md, false);
        let mut grid = VtGrid::new(96);
        grid.feed(&stream);
        let screen = grid.screen();
        let table_border_cols = grid
            .border_columns_by_row()
            .into_iter()
            .filter(|cols| !cols.is_empty())
            .collect::<Vec<_>>();
        let expected_border_cols = table_border_cols.first().expect("table top border").clone();

        assert!(
            table_border_cols.len() >= 4,
            "expected rendered table lines:\n{}",
            screen.join("\n")
        );
        for border_cols in table_border_cols {
            assert_eq!(
                border_cols,
                expected_border_cols,
                "table border columns drifted:\nfull screen:\n{}",
                screen.join("\n")
            );
        }
    }

    #[test]
    fn streamed_single_column_long_json_row_rewrites_without_raw_preview_residue() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "72") };

        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let mut stream = String::new();
        let md = "\
| 事件载荷 |
| --- |
| {\"event\":\"order.completed\",\"payload\":{\"orderId\":\"ORD-20260703-000001\",\"status\":\"paid\",\"items\":[{\"sku\":\"long-sku-code-001\",\"quantity\":2}],\"traceId\":\"trace-abcdefghijklmnopqrstuvwxyz\"}} |
";

        for ch in md.chars() {
            let mut buf = [0u8; 4];
            stream.push_str(
                &renderer
                    .write_chunk_for_test(ch.encode_utf8(&mut buf), false)
                    .unwrap(),
            );
        }
        // 表格是最后一段内容，需 flush 才会一次性画出成品盒框表。
        stream.push_str(&renderer.flush_pending_for_test().unwrap());

        let mut grid = VtGrid::new(72);
        grid.feed(&stream);
        let screen = grid.screen();
        let joined = screen.join("\n");

        assert!(joined.contains('┌'), "{joined}");
        assert!(joined.contains("事件载荷"), "{joined}");
        for line in &screen {
            assert!(
                !line.contains('|'),
                "raw markdown table preview leaked after row rewrite: {line:?}\n{}",
                screen.join("\n")
            );
            if line.contains("event") || line.contains("payload") || line.contains("trace") {
                assert!(
                    line.starts_with('│'),
                    "JSON table content must stay inside the rendered table: {line:?}\n{}",
                    screen.join("\n")
                );
            }
        }
    }

    #[test]
    fn confirmed_table_row_is_buffered_not_streamed_before_newline() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "72") };

        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let mut stream = String::new();
        for chunk in ["| 事件载荷 |\n", "| --- |\n", "| {\"event\":\"order"] {
            stream.push_str(&renderer.write_chunk_for_test(chunk, false).unwrap());
        }

        // 新契约：表格行全程静默缓冲，未落定的分块不得出现在屏幕上——不再逐字实时预览。
        let mut grid = VtGrid::new(72);
        grid.feed(&stream);
        let screen = grid.screen();
        for line in &screen {
            assert!(
                !line.contains("event"),
                "partial table row must stay buffered, not streamed to screen: {line:?}"
            );
            assert!(
                !line.contains('|'),
                "no raw markdown pipe preview should reach the screen: {line:?}"
            );
        }
    }

    #[test]
    fn multi_column_table_with_overlong_code_span_stays_within_terminal_width() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "80") };

        let renderer = MarkdownStreamRenderer::new_with_tty(true);
        let header = vec![
            "函数签名".to_string(),
            "说明".to_string(),
            "返回值".to_string(),
        ];
        let align = vec![TableAlign::Left; 3];
        let rows = vec![vec![
            "`async processOrder(orderId: string, options?: { retry?: number; timeout?: number; callback?: (result: OrderResult) => void; metadata?: Record<string, string> }) => Promise<OrderResult>`".to_string(),
            "异步处理订单。该函数会依次执行：校验订单、锁定库存、调用支付网关、写入流水、释放锁、发送通知。".to_string(),
            "`{ success, data, error, traceId }`".to_string(),
        ]];

        let rendered = renderer.render_table_block("", &header, &align, &rows);
        assert!(!rendered.is_empty());
        for line in rendered.lines() {
            let visible = crate::ai::stream::extract::strip_ansi_codes(line);
            let width = terminal_display_width(visible.as_str());
            assert!(
                width <= 80,
                "rendered table line exceeds terminal width ({width}):\n{visible}"
            );
            assert!(
                visible.starts_with(['┌', '├', '└', '│']),
                "table line should not be a wrapped continuation:\n{visible}"
            );
        }
    }

    #[test]
    fn overwide_table_splits_into_column_blocks_that_fit_terminal_width() {
        let _guard = env_guard();
        unsafe { std::env::set_var("COLUMNS", "80") };

        let renderer = MarkdownStreamRenderer::new_with_tty(true);
        let header = (0..20).map(|idx| format!("列{idx}")).collect::<Vec<_>>();
        let align = vec![TableAlign::Left; 20];
        let rows = vec![vec![
            "alpha beta gamma".to_string(),
            "delta epsilon zeta".to_string(),
            "eta theta iota".to_string(),
            "kappa lambda mu".to_string(),
            "nu xi omicron".to_string(),
            "pi rho sigma".to_string(),
            "tau upsilon phi".to_string(),
            "chi psi omega".to_string(),
            "一二三四五六".to_string(),
            "七八九十十一".to_string(),
            "long-token-abcdefghij".to_string(),
            "long-token-klmnopqrst".to_string(),
            "long-token-uvwxyz".to_string(),
            "markdown `code span`".to_string(),
            "**bold value**".to_string(),
            "plain value".to_string(),
            "another value".to_string(),
            "more value".to_string(),
            "tail value".to_string(),
            "final value".to_string(),
        ]];

        let rendered = renderer.render_table_block("", &header, &align, &rows);
        // 续接列块用 ├ 代替 ┌ 顶边框，用 └ 底边框确认分列
        let bottom_count = rendered.matches('└').count();

        assert!(
            bottom_count > 1,
            "overwide table should be split into multiple column blocks:\n{rendered}"
        );
        for line in rendered.lines().filter(|line| !line.is_empty()) {
            let visible = crate::ai::stream::extract::strip_ansi_codes(line);
            let width = terminal_display_width(visible.as_str());
            assert!(
                width <= 80,
                "split table line exceeds terminal width ({width}):\n{visible}\n\n{rendered}"
            );
        }
    }

    #[test]
    fn display_math_is_buffered_and_flushed_without_raw_tex_preview() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let mut output = renderer
            .write_block_for_test(
                "  $$\n  \\begin{aligned}\n  x &= 1 | 2 \\\\ y &= 2\n  \\end{aligned}\n  $$\n",
                false,
            )
            .unwrap();
        output.push_str(&renderer.flush_pending_for_test().unwrap());

        let visible = crate::ai::stream::extract::strip_ansi_codes(&output);
        assert!(!visible.contains("$$"), "got: {visible}");
        assert!(!visible.contains("\\begin"), "got: {visible}");
        assert!(!visible.contains('&'), "got: {visible}");
        assert!(visible.contains('|'), "got: {visible}");
        assert_eq!(visible.matches('x').count(), 1, "got: {visible}");
        assert_eq!(visible.matches('y').count(), 1, "got: {visible}");
        for line in visible.lines().filter(|line| line.contains(['x', 'y'])) {
            assert!(line.starts_with("  "), "got: {visible}");
        }
    }

    #[test]
    fn unmatched_display_closer_does_not_open_a_math_block() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let _ = renderer.consume_line("\\]", false);
        assert!(renderer.math_block_delimiter.is_none());
    }

    #[test]
    fn flush_includes_unterminated_math_block_final_line() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let mut output = renderer.write_block_for_test("$$\nx^2", false).unwrap();
        output.push_str(&renderer.flush_pending_for_test().unwrap());

        let visible = crate::ai::stream::extract::strip_ansi_codes(&output);
        assert!(visible.contains("x²"), "got: {visible}");
        assert!(!visible.contains("x^2"), "got: {visible}");
    }

    #[test]
    fn latex_fenced_block_renders_as_math_without_code_box() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let mut output = renderer
            .write_block_for_test(
                "```latex\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n```\n",
                false,
            )
            .unwrap();
        output.push_str(&renderer.flush_pending_for_test().unwrap());

        let visible = crate::ai::stream::extract::strip_ansi_codes(&output);
        // Fenced latex blocks render as math, not as a code box.
        assert!(!visible.contains("```"), "got: {visible}");
        assert!(!visible.contains("latex"), "got: {visible}");
        assert!(!visible.contains('╭'), "got: {visible}");
        assert!(!visible.contains("\\frac"), "got: {visible}");
        assert!(visible.contains("(-b ± √(b² - 4ac))/2a"), "got: {visible}");
    }

    #[test]
    fn latex_fence_closes_only_on_matching_bare_fence() {
        // The math fenced path shares the code-block closer rule: a bare fence
        // of the wrong char or a shorter run is formula content, and only a
        // bare fence matching the opener (same char, run >= opener run)
        // terminates the block. Regression: any bare fence used to close it,
        // leaking the remaining lines out as markdown (and opening a bogus
        // code box from the trailing longer fence).
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let mut output = renderer
            .write_block_for_test("````latex\n\\frac{1}{2}\n```\n~~~\n````\n", false)
            .unwrap();
        output.push_str(&renderer.flush_pending_for_test().unwrap());

        let visible = crate::ai::stream::extract::strip_ansi_codes(&output);
        assert!(visible.contains("1/2"), "got: {visible}");
        // Neither inner fence line terminated the block, so no code box opens.
        assert!(!visible.contains('╭'), "got: {visible}");
        assert!(!visible.contains('╰'), "got: {visible}");
    }

    #[test]
    fn latex_fence_renders_binomial_and_chain_rule_without_raw_tex() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let mut output = renderer
            .write_block_for_test(
                "```latex\n\\frac{dy}{dx} = \\frac{dy}{du} \\cdot \\frac{du}{dx}\n```\n\n```latex\n(a + b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k\n```\n",
                false,
            )
            .unwrap();
        output.push_str(&renderer.flush_pending_for_test().unwrap());

        let visible = crate::ai::stream::extract::strip_ansi_codes(&output);
        assert!(visible.contains("dy/dx = dy/du · du/dx"), "got: {visible}");
        assert!(
            visible.contains("(a + b)ⁿ = ∑ₖ₌₀ⁿ C(n, k) aⁿ⁻ᵏ bᵏ"),
            "got: {visible}"
        );
        assert!(!visible.contains("\\binom"), "got: {visible}");
    }

    #[test]
    fn streamed_latex_fence_rewrites_to_compact_math_without_placeholder_rows() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(true);
        let mut stream = String::new();
        for chunk in [
            "1. 勾股定理\n",
            "```latex\n",
            "a^2 + b^2 = c^2\n",
            "```\n",
            "2. 定积分\n",
        ] {
            stream.push_str(&renderer.write_chunk_for_test(chunk, false).unwrap());
        }

        let mut grid = VtGrid::new(80);
        grid.feed(&stream);
        let screen = grid.screen();
        let title_row = screen
            .iter()
            .position(|line| line.contains("勾股定理"))
            .unwrap();
        let formula_row = screen
            .iter()
            .position(|line| line.contains("a² + b² = c²"))
            .unwrap();
        let next_title_row = screen
            .iter()
            .position(|line| line.contains("定积分"))
            .unwrap();

        assert_eq!(formula_row, title_row + 1, "screen: {}", screen.join("\n"));
        assert_eq!(
            next_title_row,
            formula_row + 1,
            "screen: {}",
            screen.join("\n")
        );
        let visible = screen.join("\n");
        assert!(!visible.contains("```"), "screen: {visible}");
    }

    #[test]
    fn tex_and_math_fence_aliases_render_as_math() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let mut output = renderer
            .write_block_for_test(
                "```tex\ne^{i\\pi} + 1 = 0\n```\n~~~math\nf(x) = \\frac{1}{2}\n~~~\n",
                false,
            )
            .unwrap();
        output.push_str(&renderer.flush_pending_for_test().unwrap());

        let visible = crate::ai::stream::extract::strip_ansi_codes(&output);
        assert!(!visible.contains("```"), "got: {visible}");
        assert!(!visible.contains("~~~"), "got: {visible}");
        assert!(!visible.contains('╭'), "got: {visible}");
        assert!(!visible.contains("\\frac"), "got: {visible}");
        assert!(visible.contains('π'), "got: {visible}");
        assert!(visible.contains("1/2"), "got: {visible}");
    }

    #[test]
    fn latex_fenced_aligned_environment_renders_as_block() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let mut output = renderer
            .write_block_for_test(
                "```latex\n\\begin{aligned}\nx &= 1 \\\\ y &= 2\n\\end{aligned}\n```\n",
                false,
            )
            .unwrap();
        output.push_str(&renderer.flush_pending_for_test().unwrap());

        let visible = crate::ai::stream::extract::strip_ansi_codes(&output);
        assert!(!visible.contains("```"), "got: {visible}");
        assert!(!visible.contains("\\begin"), "got: {visible}");
        assert!(visible.contains('x'), "got: {visible}");
        assert!(visible.contains('y'), "got: {visible}");
    }

    #[test]
    fn unterminated_latex_fence_flushes_rendered_math_at_end() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let mut output = renderer
            .write_block_for_test("```latex\nx^2\n", false)
            .unwrap();
        output.push_str(&renderer.flush_pending_for_test().unwrap());

        let visible = crate::ai::stream::extract::strip_ansi_codes(&output);
        assert!(visible.contains("x²"), "got: {visible}");
        assert!(!visible.contains("x^2"), "got: {visible}");
    }

    #[test]
    fn non_math_fence_still_renders_code_box() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let out = renderer
            .write_block_for_test("```rust\nfn main() {}\n```\n", false)
            .unwrap();
        let visible = crate::ai::stream::extract::strip_ansi_codes(&out);
        assert!(visible.contains('╭'), "got: {visible}");
        assert!(visible.contains("rust"), "got: {visible}");
    }
