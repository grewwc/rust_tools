use super::*;

pub(super) fn should_prefer_docx_blocks_render(raw_content: &str, blocks_text: &str) -> bool {
    let raw = raw_content.trim();
    let rendered = blocks_text.trim();
    if rendered.is_empty() {
        return false;
    }
    if raw.is_empty() {
        return true;
    }
    if rendered == raw {
        return false;
    }

    if rendered.contains("```") {
        return true;
    }

    if rendered.contains("| --- |") {
        return true;
    }

    if rendered.contains("- [ ] ") || rendered.contains("- [x] ") {
        return true;
    }

    docx_blocks_text_has_non_text_placeholders(rendered)
}

pub(super) fn docx_blocks_text_has_non_text_placeholders(s: &str) -> bool {
    s.contains("[流程图]")
        || s.contains("[UML 图]")
        || s.contains("[文字绘图")
        || s.contains("[图片")
        || s.contains("[文件")
        || s.contains("[思维笔记")
        || s.contains("[电子表格")
        || s.contains("[多维表格")
        || s.contains("[嵌入内容")
        || s.contains("[会话卡片")
        || s.contains("[小组件")
}

pub(super) fn render_docx_blocks_as_text(items: &[Value], default_origin: Option<&str>) -> String {
    if items.is_empty() {
        return String::new();
    }

    let mut by_id: FastMap<String, &Value> = FastMap::default();
    let mut root_id = None::<String>;
    for item in items {
        if let Some(block_id) = item.get("block_id").and_then(|v| v.as_str()) {
            if item
                .get("parent_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .is_empty()
            {
                root_id = Some(block_id.to_string());
            }
            by_id.insert(block_id.to_string(), item);
        }
    }

    let Some(root_id) = root_id.or_else(|| {
        items
            .first()
            .and_then(|v| v.get("block_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }) else {
        return String::new();
    };

    let mut out = String::new();
    render_docx_block_text(&root_id, &by_id, default_origin, &mut out, 0, 0);
    normalize_rendered_docx_text(&out)
}

pub(super) fn render_docx_block_text(
    block_id: &str,
    by_id: &FastMap<String, &Value>,
    default_origin: Option<&str>,
    out: &mut String,
    list_depth: usize,
    quote_depth: usize,
) {
    let Some(block) = by_id.get(block_id).copied() else {
        return;
    };

    let block_type = block
        .get("block_type")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let line = render_docx_block_line(block, default_origin, list_depth);
    if !line.is_empty() {
        out.push_str(&prefix_rendered_line(&line, quote_depth));
        out.push('\n');
    }

    if block_type == 31 {
        render_docx_table_cells(block, by_id, default_origin, out);
        return;
    }

    let children = block
        .get("children")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    for child in children {
        if let Some(child_id) = child.as_str() {
            let next_depth = if matches!(block_type, 12 | 13 | 17) {
                list_depth + 1
            } else {
                list_depth
            };
            let next_quote_depth = if block_type == 15 {
                quote_depth + 1
            } else {
                quote_depth
            };
            render_docx_block_text(
                child_id,
                by_id,
                default_origin,
                out,
                next_depth,
                next_quote_depth,
            );
        }
    }
}

pub(super) fn render_docx_table_cells(
    block: &Value,
    by_id: &FastMap<String, &Value>,
    default_origin: Option<&str>,
    out: &mut String,
) {
    let cells = block
        .get("table")
        .and_then(|v| v.get("cells"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let col_size = block
        .get("table")
        .and_then(|v| v.get("property"))
        .and_then(|v| v.get("column_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let header_row = block
        .get("table")
        .and_then(|v| v.get("property"))
        .and_then(|v| v.get("header_row"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if cells.is_empty() || col_size == 0 {
        return;
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    for cell in cells {
        let Some(cell_id) = cell.as_str() else {
            continue;
        };
        let cell_text = render_docx_table_cell_text(cell_id, by_id, default_origin);
        let escaped = cell_text.replace('|', "\\|").replace('\n', "<br>");
        row.push(escaped);
        if row.len() >= col_size {
            rows.push(row);
            row = Vec::new();
        }
    }
    if !row.is_empty() {
        rows.push(row);
    }

    for (i, row) in rows.iter().enumerate() {
        let mut normalized_row = row.clone();
        while normalized_row.len() < col_size {
            normalized_row.push(String::new());
        }
        out.push_str("| ");
        out.push_str(&normalized_row.join(" | "));
        out.push_str(" |\n");
        if i == 0 && header_row {
            out.push_str("|");
            for _ in 0..col_size {
                out.push_str(" --- |");
            }
            out.push('\n');
        }
    }
}

pub(super) fn render_docx_table_cell_text(
    cell_id: &str,
    by_id: &FastMap<String, &Value>,
    default_origin: Option<&str>,
) -> String {
    let Some(block) = by_id.get(cell_id).copied() else {
        return String::new();
    };

    let direct = render_text_elements(
        block
            .get("table_cell")
            .and_then(|v| v.get("elements"))
            .or_else(|| block.get("text").and_then(|v| v.get("elements"))),
        default_origin,
    );
    if !direct.is_empty() {
        return direct;
    }

    let mut parts = Vec::new();
    let children = block
        .get("children")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    for child in children {
        let Some(child_id) = child.as_str() else {
            continue;
        };
        let Some(child_block) = by_id.get(child_id).copied() else {
            continue;
        };
        let line = render_docx_block_line(child_block, default_origin, 0);
        if !line.is_empty() {
            parts.push(line);
        }
    }
    parts.join("<br>")
}

pub(super) fn render_docx_block_line(
    block: &Value,
    default_origin: Option<&str>,
    list_depth: usize,
) -> String {
    let block_type = block
        .get("block_type")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    match block_type {
        1 => render_text_elements(
            block.get("page").and_then(|v| v.get("elements")),
            default_origin,
        ),
        2 => render_text_elements(
            block.get("text").and_then(|v| v.get("elements")),
            default_origin,
        ),
        3..=11 => {
            let level = (block_type - 2) as usize;
            let text = render_text_elements(
                block
                    .get(format!("heading{}", level).as_str())
                    .and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                String::new()
            } else {
                format!("{} {}", "#".repeat(level), text)
            }
        }
        12 => {
            let text = render_text_elements(
                block.get("bullet").and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                String::new()
            } else {
                format!("{}- {}", "  ".repeat(list_depth), text)
            }
        }
        13 => {
            let text = render_text_elements(
                block.get("ordered").and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                String::new()
            } else {
                format!(
                    "{}{}. {}",
                    "  ".repeat(list_depth),
                    ordered_list_sequence(block),
                    text
                )
            }
        }
        14 => {
            let text = render_code_block_elements(
                block.get("code").and_then(|v| v.get("elements")),
                default_origin,
            );
            let lang = code_block_language_tag(block.get("code").and_then(|v| v.get("style")));
            if text.is_empty() {
                String::new()
            } else {
                format!("```{}\n{}\n```", lang, text)
            }
        }
        15 => {
            let text = render_text_elements(
                block.get("quote").and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                String::new()
            } else {
                format!("> {}", text)
            }
        }
        17 => {
            let todo = block.get("todo");
            let text = render_text_elements(todo.and_then(|v| v.get("elements")), default_origin);
            let checked = todo
                .and_then(|v| v.get("style"))
                .and_then(|v| v.get("done"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if text.is_empty() {
                String::new()
            } else {
                format!(
                    "{}- [{}] {}",
                    "  ".repeat(list_depth),
                    if checked { "x" } else { " " },
                    text
                )
            }
        }
        18 => render_token_placeholder(block.get("bitable"), "token", "多维表格"),
        19 => {
            let text = render_text_elements(
                block.get("callout").and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                "[高亮块]".to_string()
            } else {
                format!("[高亮块] {}", text)
            }
        }
        20 => "[会话卡片]".to_string(),
        21 => render_diagram_placeholder(block),
        22 => "---".to_string(),
        23 => render_named_placeholder(block.get("file"), "name", "文件"),
        24 | 25 | 26 | 32 | 33 | 34 | 35 | 36 | 37 => String::new(),
        27 => render_token_placeholder(block.get("image"), "token", "图片"),
        28 => "[小组件]".to_string(),
        29 => render_token_placeholder(block.get("mindnote"), "token", "思维笔记"),
        30 => render_token_placeholder(block.get("sheet"), "token", "电子表格"),
        31 => String::new(),
        43 => render_token_placeholder(block.get("board"), "token", "文字绘图"),
        _ => String::new(),
    }
}

pub(super) fn prefix_rendered_line(line: &str, quote_depth: usize) -> String {
    if quote_depth == 0 || line.is_empty() {
        return line.to_string();
    }
    let prefix = format!("{} ", ">".repeat(quote_depth));
    line.lines()
        .map(|part| format!("{prefix}{part}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn ordered_list_sequence(block: &Value) -> String {
    block
        .get("ordered")
        .and_then(|v| v.get("style"))
        .and_then(|v| v.get("sequence"))
        .and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_i64().map(|n| n.to_string()))
        })
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "1".to_string())
}

pub(super) fn render_code_block_elements(elements: Option<&Value>, default_origin: Option<&str>) -> String {
    let Some(arr) = elements.and_then(|v| v.as_array()) else {
        return String::new();
    };
    let mut out = String::new();
    for el in arr {
        if let Some(text_run) = el.get("text_run") {
            let content = text_run
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            out.push_str(content);
            continue;
        }
        let fallback = render_text_elements(Some(el), default_origin);
        if !fallback.is_empty() {
            out.push_str(&fallback);
        }
    }
    out
}

pub(super) fn code_block_language_tag(style: Option<&Value>) -> String {
    let Some(style) = style else {
        return "text".to_string();
    };
    if let Some(lang) = style.get("language").and_then(|v| v.as_str()) {
        let normalized = lang.trim().to_ascii_lowercase();
        if !normalized.is_empty() {
            return normalized;
        }
    }
    match style.get("language").and_then(|v| v.as_i64()) {
        Some(7) => "bash".to_string(),
        Some(22) => "go".to_string(),
        Some(24) => "html".to_string(),
        Some(28) => "json".to_string(),
        Some(29) => "java".to_string(),
        Some(30) => "javascript".to_string(),
        Some(32) => "kotlin".to_string(),
        Some(39) => "markdown".to_string(),
        Some(43) => "php".to_string(),
        Some(49) => "python".to_string(),
        Some(50) => "r".to_string(),
        Some(53) => "rust".to_string(),
        Some(56) => "sql".to_string(),
        Some(60) => "shell".to_string(),
        Some(63) => "typescript".to_string(),
        Some(66) => "xml".to_string(),
        Some(67) => "yaml".to_string(),
        Some(68) => "cmake".to_string(),
        Some(74) => "solidity".to_string(),
        Some(75) => "toml".to_string(),
        _ => "text".to_string(),
    }
}

pub(super) fn render_diagram_placeholder(block: &Value) -> String {
    let diagram_type = block
        .get("diagram")
        .and_then(|v| v.get("diagram_type"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    match diagram_type {
        1 => "[流程图]".to_string(),
        2 => "[UML 图]".to_string(),
        _ => "[文字绘图]".to_string(),
    }
}

pub(super) fn render_named_placeholder(container: Option<&Value>, field: &str, label: &str) -> String {
    let name = container
        .and_then(|v| v.get(field))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if name.is_empty() {
        format!("[{}]", label)
    } else {
        format!("[{}: {}]", label, name)
    }
}

pub(super) fn render_token_placeholder(container: Option<&Value>, field: &str, label: &str) -> String {
    let token = container
        .and_then(|v| v.get(field))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if token.is_empty() {
        format!("[{}]", label)
    } else {
        format!("[{}: {}]", label, token)
    }
}

pub(super) fn extract_url_origin(url: &str) -> Option<String> {
    let raw = url.trim();
    if raw.is_empty() {
        return None;
    }
    let raw = raw.split('#').next().unwrap_or(raw);
    let raw = raw.split('?').next().unwrap_or(raw);
    let (scheme, rest) = if let Some((s, r)) = raw.split_once("://") {
        (s.trim(), r)
    } else {
        ("https", raw)
    };
    let host = rest.split('/').next().unwrap_or("").trim();
    if host.is_empty() {
        return None;
    }
    Some(format!("{}://{}", scheme, host))
}

pub(super) fn render_doc_mention_as_markdown(mention: &Value, default_origin: Option<&str>) -> Option<String> {
    let title = mention
        .get("title")
        .or_else(|| mention.get("name"))
        .or_else(|| mention.get("text"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let url = mention
        .get("url")
        .and_then(|v| v.as_str())
        .or_else(|| mention.get("href").and_then(|v| v.as_str()))
        .or_else(|| {
            mention
                .get("link")
                .and_then(|v| v.get("url"))
                .and_then(|v| v.as_str())
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let token = mention
        .get("token")
        .or_else(|| mention.get("obj_token"))
        .or_else(|| mention.get("docs_token"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let obj_type = mention
        .get("obj_type")
        .or_else(|| mention.get("docs_type"))
        .or_else(|| mention.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();

    let url = url.or_else(|| {
        if token.is_empty() {
            return None;
        }
        let kind = match obj_type.as_str() {
            "docx" => "docx",
            "doc" | "docs" => "doc",
            "sheet" | "sheets" => "sheets",
            "wiki" => "wiki",
            _ => "docx",
        };
        let origin = default_origin.unwrap_or("https://www.feishu.cn").trim();
        Some(format!(
            "{}/{}/{}",
            origin.trim_end_matches('/'),
            kind,
            token
        ))
    });

    let Some(url) = url else {
        if title.is_empty() {
            return None;
        }
        return Some(title);
    };

    if title.is_empty() {
        Some(url)
    } else {
        Some(format!("[{}]({})", title, url))
    }
}

pub(super) fn render_text_elements(elements: Option<&Value>, default_origin: Option<&str>) -> String {
    let Some(arr) = elements.and_then(|v| v.as_array()) else {
        return String::new();
    };

    let mut out = String::new();
    for el in arr {
        // 处理 text_run，包括链接
        if let Some(text_run) = el.get("text_run") {
            let content = text_run
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // 检查是否有链接
            let link_url = text_run
                .get("text_style")
                .and_then(|v| v.get("link"))
                .and_then(|v| v.get("url"))
                .and_then(|v| v.as_str());

            if let Some(url) = link_url {
                // 如果有链接，格式化为 [文本](URL) 的 Markdown 格式
                if !content.is_empty() {
                    out.push('[');
                    out.push_str(content);
                    out.push_str("](");
                    out.push_str(url);
                    out.push(')');
                } else {
                    // 如果内容为空但 URL 存在，直接显示 URL
                    out.push_str(url);
                }
            } else {
                // 没有链接，直接显示文本
                out.push_str(content);
            }
            continue;
        }

        if let Some(v) = el
            .get("equation")
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
        {
            out.push_str(v);
            continue;
        }
        if let Some(v) = el
            .get("mention_user")
            .and_then(|v| {
                v.get("user_name")
                    .or_else(|| v.get("name"))
                    .or_else(|| v.get("title"))
            })
            .and_then(|v| v.as_str())
        {
            out.push('@');
            out.push_str(v);
            continue;
        }
        if let Some(v) = el
            .get("mention_doc")
            .and_then(|v| render_doc_mention_as_markdown(v, default_origin))
        {
            out.push_str(&v);
            continue;
        }
        if let Some(v) = el
            .get("reminder")
            .and_then(|v| v.get("notify_time"))
            .and_then(|v| v.as_str())
        {
            out.push_str("[提醒:");
            out.push_str(v);
            out.push(']');
            continue;
        }
        if let Some(v) = el
            .get("file")
            .and_then(|v| {
                v.get("name")
                    .or_else(|| v.get("file_token"))
                    .or_else(|| v.get("token"))
            })
            .and_then(|v| v.as_str())
        {
            out.push_str("[附件:");
            out.push_str(v);
            out.push(']');
            continue;
        }
        if let Some(v) = el
            .get("inline_block")
            .and_then(|v| {
                v.get("block_id")
                    .or_else(|| v.get("token"))
                    .or_else(|| v.get("url"))
            })
            .and_then(|v| v.as_str())
        {
            out.push_str("[内联块:");
            out.push_str(v);
            out.push(']');
        }
    }
    out.trim().to_string()
}

pub(super) fn normalize_rendered_docx_text(s: &str) -> String {
    let mut out = String::new();
    let mut blank_run = 0usize;
    for line in s.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                out.push('\n');
            }
            continue;
        }
        blank_run = 0;
        out.push_str(trimmed);
        out.push('\n');
    }
    out.trim().to_string()
}
