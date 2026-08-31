//! Tool-result preparation: inline/offload decisions and the lossless
//! line-trim policy for evidence-bearing tool outputs.

use super::*;

/// Tools suited to imprecise overview where the middle can be trimmed line by line.
///
/// Every line of `read_file(_lines)` output can be exact evidence the agent may
/// need to cite in later judgments, so lossy middle sampling is not allowed; these
/// tools may only be offloaded to a session file after exceeding the inline limit,
/// keeping `path` + a stub in the model context.
pub(in crate::ai::driver::turn_runtime) fn supports_line_trim(tool_name: &str) -> bool {
    matches!(tool_name, "tree" | "ast_outline")
}

/// Fold "medium-sized" structured output (between MAX_TOOL_RESULT_LINE_TRIM_CHARS and
/// MAX_TOOL_RESULT_INLINE_CHARS) into: first N lines + a few keyword-matching lines +
/// last M lines + a middle marker. Nothing is written to disk and the overall semantics
/// are preserved; it only squeezes out the redundant middle.
pub(in crate::ai::driver::turn_runtime) fn line_trim_middle(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    if total_lines <= 80 {
        return content.to_string();
    }

    let head_lines = 40usize;
    let tail_lines = 20usize;

    let mut head = Vec::with_capacity(head_lines);
    for line in lines.iter().take(head_lines) {
        head.push(*line);
    }
    let tail_start = total_lines.saturating_sub(tail_lines);
    let mut tail = Vec::with_capacity(tail_lines);
    if tail_start > head_lines {
        for line in lines.iter().skip(tail_start) {
            tail.push(*line);
        }
    }

    // Sample up to 8 lines from the middle (head_lines..tail_start) by keyword
    let mut key_lines: Vec<(usize, &str)> = Vec::new();
    if tail_start > head_lines {
        for (i, line) in lines.iter().enumerate().take(tail_start).skip(head_lines) {
            let lower = line.to_ascii_lowercase();
            let important = lower.contains("error")
                || lower.contains("fail")
                || lower.contains("panic")
                || lower.contains("warn")
                || lower.contains("todo")
                || lower.contains("fixme")
                || lower.contains("//!")
                || lower.contains("///")
                || lower.starts_with("fn ")
                || lower.starts_with("pub fn ")
                || lower.starts_with("impl ")
                || lower.starts_with("struct ")
                || lower.starts_with("trait ")
                || lower.starts_with("enum ")
                || lower.starts_with("#[")
                || lower.contains(": error")
                || lower.contains(": warning");
            if important {
                key_lines.push((i, *line));
                if key_lines.len() >= 8 {
                    break;
                }
            }
        }
    }

    let omitted_count = total_lines.saturating_sub(head_lines + tail.len());
    let mut out = String::with_capacity(content.len() / 2);
    for line in &head {
        out.push_str(line);
        out.push('\n');
    }
    if !key_lines.is_empty() {
        out.push_str(&format!(
            "\n... [middle trimmed: {} lines folded; key-match samples below]\n",
            omitted_count.saturating_sub(key_lines.len())
        ));
        for (idx, line) in &key_lines {
            out.push_str(&format!("L{idx}: {line}\n"));
        }
        out.push_str("...\n");
    } else {
        out.push_str(&format!(
            "\n... [middle trimmed: {} lines folded]\n",
            omitted_count
        ));
    }
    for line in &tail {
        out.push_str(line);
        out.push('\n');
    }
    out
}

pub(in crate::ai::driver::turn_runtime) fn prepare_tool_result(
    app: &App,
    tool_name: &str,
    content: &str,
) -> PreparedToolResult {
    let inline_limit = max_tool_result_inline_chars(&app.current_model);
    let char_count = content.chars().count();
    if char_count <= MAX_TOOL_RESULT_LINE_TRIM_CHARS {
        return PreparedToolResult {
            content_for_model: content.to_string(),
            content_for_terminal: build_terminal_preview(tool_name, content),
        };
    }

    if char_count <= inline_limit && supports_line_trim(tool_name) {
        let trimmed = line_trim_middle(content);
        // Reuse the trimmed byte length as a cheap short-circuit: trimmed is
        // assembled from selected lines of content (possibly rewritten; ASCII/UTF-8
        // preserved), so if it is shorter in bytes it is necessarily shorter in
        // chars too — no need for a full chars().count() second scan.
        if trimmed.len() < content.len() && trimmed.chars().count() < char_count {
            return PreparedToolResult {
                content_for_model: trimmed,
                content_for_terminal: build_terminal_preview(tool_name, content),
            };
        }
    }

    if char_count <= inline_limit {
        return PreparedToolResult {
            content_for_model: content.to_string(),
            content_for_terminal: build_terminal_preview(tool_name, content),
        };
    }

    let summary = summarize_large_tool_output(content);
    let path = write_tool_overflow_file(app, tool_name, &summary.body).ok();
    let content_for_model = build_model_overflow_stub(path.as_ref(), &summary);
    let content_for_terminal = if let Some(path) = path {
        format!(
            "{}\n[Saved full output to {}]\n",
            build_terminal_preview(
                tool_name,
                &tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS)
            ),
            path.display()
        )
    } else {
        build_terminal_preview(
            tool_name,
            &tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS),
        )
    };

    PreparedToolResult {
        content_for_model,
        content_for_terminal,
    }
}

/// Tool results just produced in the current round must enter messages as raw content
/// first, so the “keep the last N tool results verbatim” protection holds from the
/// entry point, instead of being weakened here by stub/summary and then relying on
/// `KEEP_RECENT_TOOL_MESSAGES` to bail out later.
///
/// The terminal side keeps the existing preview / overflow-file logic, so oversized
/// results are not dumped wholesale to the screen.
pub(in crate::ai::driver::turn_runtime) fn prepare_recent_tool_result(
    app: &App,
    tool_name: &str,
    content: &str,
) -> PreparedToolResult {
    let content_for_terminal = prepare_tool_result(app, tool_name, content).content_for_terminal;
    PreparedToolResult {
        content_for_model: content.to_string(),
        content_for_terminal,
    }
}
