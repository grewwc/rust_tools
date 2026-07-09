//! HTML 表格解析模块。
//!
//! 将 LLM 输出的 `<table>` HTML 片段解析为 header + rows 网格结构，
//! 复用 `table.rs` 中已有的渲染函数（`render_table_top/header/mid/row/bottom`）
//! 在终端中绘制对齐的表格。支持 `rowspan` / `colspan` 合并单元格。

use super::table::{
    TableAlign, compute_table_widths, render_table_bottom, render_table_header, render_table_mid,
    render_table_row, render_table_top,
};

/// 解析后的 HTML 表格。
pub(super) struct HtmlTable {
    /// 表头行（若第一行含 `<th>` 则提取为表头，否则为空）。
    pub header: Vec<String>,
    /// 数据行（已展开 rowspan/colspan）。
    pub rows: Vec<Vec<String>>,
}

/// 单个单元格的解析结果。
struct ParsedCell {
    text: String,
    rowspan: usize,
    colspan: usize,
    is_header: bool,
}

/// 判断一行文本是否包含 `<table` 开标签（大小写不敏感）。
pub(super) fn contains_open_table_tag(line: &str) -> bool {
    line.to_lowercase().contains("<table")
}

/// 判断一行文本是否包含 `</table>` 闭标签（大小写不敏感）。
pub(super) fn contains_close_table_tag(line: &str) -> bool {
    line.to_lowercase().contains("</table>")
}

/// 解析 HTML 表格文本，返回 `HtmlTable`。解析失败返回 `None`。
pub(super) fn parse_html_table(html: &str) -> Option<HtmlTable> {
    let table_content = extract_table_content(html)?;
    let rows = extract_rows(&table_content);
    if rows.is_empty() {
        return None;
    }

    // 展开 rowspan / colspan 为稠密网格。
    let grid = expand_grid(&rows);
    if grid.is_empty() || grid[0].is_empty() {
        return None;
    }

    // 第一行全部是 `<th>` → 表头；否则全部为数据行。
    let first_row_all_header = rows[0]
        .iter()
        .all(|c| c.is_header || c.text.trim().is_empty());
    let (header, data_rows) = if first_row_all_header && rows[0].iter().any(|c| c.is_header) {
        let header = grid[0]
            .iter()
            .map(|c| c.clone().unwrap_or_default())
            .collect();
        let rows: Vec<Vec<String>> = grid[1..]
            .iter()
            .map(|r| r.iter().map(|c| c.clone().unwrap_or_default()).collect())
            .collect();
        (header, rows)
    } else {
        let rows: Vec<Vec<String>> = grid
            .iter()
            .map(|r| r.iter().map(|c| c.clone().unwrap_or_default()).collect())
            .collect();
        (Vec::new(), rows)
    };

    if header.is_empty() && data_rows.is_empty() {
        return None;
    }

    Some(HtmlTable {
        header,
        rows: data_rows,
    })
}

/// 渲染 HTML 表格为终端文本（复用 table.rs 的渲染函数）。
pub(super) fn render_html_table(indent: &str, table: &HtmlTable) -> String {
    let cols = table
        .header
        .len()
        .max(table.rows.iter().map(|r| r.len()).max().unwrap_or(0));
    if cols == 0 {
        return String::new();
    }

    let align = vec![TableAlign::Left; cols];
    let widths = compute_table_widths(indent, &table.header, &table.rows);

    let mut out = String::new();
    out.push_str(&render_table_top(indent, &widths));
    if !table.header.is_empty() {
        out.push_str(&render_table_header(indent, &table.header, &align, &widths));
        out.push_str(&render_table_mid(indent, &widths));
    }
    for row in &table.rows {
        out.push_str(&render_table_row(indent, row, &align, &widths));
    }
    out.push_str(&render_table_bottom(indent, &widths));
    out
}

// ─── 内部解析函数 ───────────────────────────────────────────────

/// 提取 `<table...>` 与 `</table>` 之间的内容（不含 table 标签本身）。
fn extract_table_content(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<table")?;
    // 跳过开标签（到第一个 `>`）
    let tag_end = lower[start..].find('>')? + start + 1;
    let rest = &html[tag_end..];

    let end = rest.to_lowercase().find("</table>")?;
    Some(rest[..end].to_string())
}

/// 从 table 内容中提取所有 `<tr>` 行，每行解析出单元格列表。
fn extract_rows(table_content: &str) -> Vec<Vec<ParsedCell>> {
    let lower = table_content.to_lowercase();
    let mut rows = Vec::new();
    let mut search_pos = 0;

    while let Some(tr_start) = lower[search_pos..].find("<tr") {
        let abs_start = search_pos + tr_start;
        // 跳过 <tr ...> 开标签
        let tag_end = match lower[abs_start..].find('>') {
            Some(pos) => abs_start + pos + 1,
            None => break,
        };

        let close_pos = match lower[tag_end..].find("</tr>") {
            Some(pos) => tag_end + pos,
            None => break,
        };

        let row_html = &table_content[tag_end..close_pos];
        let cells = extract_cells(row_html);
        if !cells.is_empty() {
            rows.push(cells);
        }

        search_pos = close_pos + 5; // 跳过 "</tr>"
    }

    rows
}

/// 从一行 `<tr>` 内容中提取所有 `<td>` / `<th>` 单元格。
fn extract_cells(row_html: &str) -> Vec<ParsedCell> {
    let lower = row_html.to_lowercase();
    let mut cells = Vec::new();
    let mut search_pos = 0;

    while search_pos < row_html.len() {
        // 查找下一个 <td 或 <th
        let (tag_start, is_header) = {
            let td_pos = lower[search_pos..].find("<td");
            let th_pos = lower[search_pos..].find("<th");
            match (td_pos, th_pos) {
                (Some(td), Some(th)) => {
                    if td < th {
                        (search_pos + td, false)
                    } else {
                        (search_pos + th, true)
                    }
                }
                (Some(td), None) => (search_pos + td, false),
                (None, Some(th)) => (search_pos + th, true),
                (None, None) => break,
            }
        };

        // 跳过开标签
        let tag_end = match lower[tag_start..].find('>') {
            Some(pos) => tag_start + pos + 1,
            None => break,
        };

        // 查找对应的闭标签 </td> 或 </th>（大小写不敏感）
        let close_tag = if is_header { "</th>" } else { "</td>" };
        let close_pos = match lower[tag_end..].find(close_tag) {
            Some(pos) => tag_end + pos,
            None => break,
        };

        let cell_html = &row_html[tag_end..close_pos];
        let open_tag = &row_html[tag_start..tag_end];

        let rowspan = parse_attr_int(open_tag, "rowspan").unwrap_or(1).max(1);
        let colspan = parse_attr_int(open_tag, "colspan").unwrap_or(1).max(1);
        let text = strip_html_tags(cell_html);

        cells.push(ParsedCell {
            text,
            rowspan,
            colspan,
            is_header,
        });

        search_pos = close_pos + close_tag.len();
    }

    cells
}

/// 从 HTML 标签中提取整数属性值（如 `rowspan="2"` → 2）。
fn parse_attr_int(tag: &str, attr: &str) -> Option<usize> {
    let lower = tag.to_lowercase();
    let attr_pattern = format!("{attr}=");
    let idx = lower.find(&attr_pattern)?;
    let after = &tag[idx + attr_pattern.len()..];

    let value = if after.starts_with('"') {
        let end = after[1..].find('"')?;
        &after[1..1 + end]
    } else if after.starts_with('\'') {
        let end = after[1..].find('\'')?;
        &after[1..1 + end]
    } else {
        // 无引号属性：取到下一个空白或 >
        let end = after
            .find(|c: char| c.is_whitespace() || c == '>')
            .unwrap_or(after.len());
        &after[..end]
    };

    value.trim().parse::<usize>().ok()
}

/// 去除 HTML 标签，保留纯文本内容。`<br>` 转换为换行。
/// 解码常见 HTML 实体。
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let lower = html.to_lowercase();
    let mut i = 0;
    let bytes = html.as_bytes();

    while i < html.len() {
        if bytes[i] == b'<' {
            // 检查是否是 <br> 标签
            let rest_lower = &lower[i..];
            if rest_lower.starts_with("<br") {
                // 找到 > 结束
                if let Some(gt) = rest_lower.find('>') {
                    result.push('\n');
                    i += gt + 1;
                    continue;
                }
            }
            // 跳过其他标签
            if let Some(gt) = rest_lower.find('>') {
                i += gt + 1;
                continue;
            } else {
                // 未闭合的 < ，保留为字面量
                break;
            }
        }

        // HTML 实体解码
        if bytes[i] == b'&' {
            if let Some(semi) = html[i..].find(';') {
                let entity = &html[i..i + semi + 1];
                if let Some(decoded) = decode_entity(entity) {
                    result.push_str(&decoded);
                    i += semi + 1;
                    continue;
                }
            }
        }

        // 普通字符
        let ch_len = utf8_char_len(bytes[i]);
        result.push_str(&html[i..i + ch_len]);
        i += ch_len;
    }

    // 合并连续空白，去除首尾空白
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 解码常见 HTML 实体。
fn decode_entity(entity: &str) -> Option<String> {
    let lower = entity.to_lowercase();
    match lower.as_str() {
        "&amp;" => Some("&".to_string()),
        "&lt;" => Some("<".to_string()),
        "&gt;" => Some(">".to_string()),
        "&quot;" => Some("\"".to_string()),
        "&apos;" => Some("'".to_string()),
        "&#39;" => Some("'".to_string()),
        "&nbsp;" => Some(" ".to_string()),
        "&copy;" => Some("©".to_string()),
        "&reg;" => Some("®".to_string()),
        "&mdash;" => Some("—".to_string()),
        "&ndash;" => Some("–".to_string()),
        _ => {
            // &#数字; 格式
            if lower.starts_with("&#") && lower.ends_with(';') {
                let num_str = &lower[2..lower.len() - 1];
                if let Ok(hex) = u32::from_str_radix(num_str, 16) {
                    return char::from_u32(hex).map(|c| c.to_string());
                }
                if let Ok(dec) = num_str.parse::<u32>() {
                    return char::from_u32(dec).map(|c| c.to_string());
                }
            }
            None
        }
    }
}

/// 返回 UTF-8 首字节对应的字符长度。
fn utf8_char_len(first_byte: u8) -> usize {
    if first_byte < 0x80 {
        1
    } else if first_byte < 0xC0 {
        1 // 无效字节，按 1 处理
    } else if first_byte < 0xE0 {
        2
    } else if first_byte < 0xF0 {
        3
    } else {
        4
    }
}

/// 将带 rowspan/colspan 的稀疏行展开为稠密网格。
///
/// 算法：维护 `grid[row][col]`，对每个单元格先跳过已被上方 rowspan 预填的列，
/// 再放置内容并预填 rowspan 覆盖的下方行。
fn expand_grid(rows: &[Vec<ParsedCell>]) -> Vec<Vec<Option<String>>> {
    let mut grid: Vec<Vec<Option<String>>> = Vec::new();

    for (row_idx, row_cells) in rows.iter().enumerate() {
        // 确保网格有此行
        while grid.len() <= row_idx {
            grid.push(Vec::new());
        }

        let mut col = 0usize;
        for cell in row_cells {
            // 跳过已被 rowspan 预填的列
            while grid[row_idx].get(col).map_or(false, |c| c.is_some()) {
                col += 1;
            }

            // 放置单元格（含 colspan）
            for c in 0..cell.colspan {
                let actual_col = col + c;
                while grid[row_idx].len() <= actual_col {
                    grid[row_idx].push(None);
                }
                grid[row_idx][actual_col] = Some(cell.text.clone());

                // 预填 rowspan 覆盖的下方行
                if cell.rowspan > 1 {
                    for r in 1..cell.rowspan {
                        while grid.len() <= row_idx + r {
                            grid.push(Vec::new());
                        }
                        while grid[row_idx + r].len() <= actual_col {
                            grid[row_idx + r].push(None);
                        }
                        grid[row_idx + r][actual_col] = Some(cell.text.clone());
                    }
                }
            }
            col += cell.colspan;
        }
    }

    // 填充网格中可能残留的 None 为空字符串占位（渲染时会被 pad_cell 填充）
    for row in &mut grid {
        let max_len = row.len();
        for cell in row.iter_mut() {
            if cell.is_none() {
                *cell = Some(String::new());
            }
        }
        let _ = max_len;
    }

    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_html_table() {
        let html = r#"<table>
<tr><th>A</th><th>B</th></tr>
<tr><td>1</td><td>2</td></tr>
</table>"#;
        let table = parse_html_table(html).unwrap();
        assert_eq!(table.header, vec!["A", "B"]);
        assert_eq!(table.rows, vec![vec!["1", "2"]]);
    }

    #[test]
    fn parse_rowspan_table() {
        let html = r#"<table>
<tr><th>部门</th><th>姓名</th><th>职位</th></tr>
<tr><td rowspan="2">研发部</td><td>张三</td><td>前端工程师</td></tr>
<tr><td>王五</td><td>后端工程师</td></tr>
</table>"#;
        let table = parse_html_table(html).unwrap();
        assert_eq!(table.header, vec!["部门", "姓名", "职位"]);
        assert_eq!(
            table.rows,
            vec![
                vec!["研发部", "张三", "前端工程师"],
                vec!["研发部", "王五", "后端工程师"],
            ]
        );
    }

    #[test]
    fn parse_colspan_table() {
        let html = r#"<table>
<tr><th>Name</th><th colspan="2">Info</th></tr>
<tr><td>Alice</td><td>30</td><td>Engineer</td></tr>
</table>"#;
        let table = parse_html_table(html).unwrap();
        assert_eq!(table.header, vec!["Name", "Info", "Info"]);
        assert_eq!(table.rows, vec![vec!["Alice", "30", "Engineer"]]);
    }

    #[test]
    fn parse_table_no_header() {
        let html = r#"<table>
<tr><td>1</td><td>2</td></tr>
<tr><td>3</td><td>4</td></tr>
</table>"#;
        let table = parse_html_table(html).unwrap();
        assert!(table.header.is_empty());
        assert_eq!(table.rows, vec![vec!["1", "2"], vec!["3", "4"]]);
    }

    #[test]
    fn parse_table_with_html_entities() {
        let html = r#"<table>
<tr><th>Expr</th><th>Result</th></tr>
<tr><td>a &amp; b</td><td>x &lt; y</td></tr>
</table>"#;
        let table = parse_html_table(html).unwrap();
        assert_eq!(table.header, vec!["Expr", "Result"]);
        assert_eq!(table.rows, vec![vec!["a & b", "x < y"]]);
    }

    #[test]
    fn parse_table_with_br_in_cell() {
        let html = r#"<table>
<tr><th>Multi</th></tr>
<tr><td>line1<br>line2</td></tr>
</table>"#;
        let table = parse_html_table(html).unwrap();
        // <br> becomes space after split_whitespace join
        assert_eq!(table.rows[0][0], "line1 line2");
    }

    #[test]
    fn parse_table_with_inner_tags() {
        let html = r#"<table>
<tr><th>Item</th></tr>
<tr><td><strong>Bold</strong> text</td></tr>
</table>"#;
        let table = parse_html_table(html).unwrap();
        assert_eq!(table.rows[0][0], "Bold text");
    }

    #[test]
    fn parse_table_case_insensitive() {
        let html = r#"<TABLE>
<TR><TH>A</TH><TH>B</TH></TR>
<TR><TD>1</TD><TD>2</TD></TR>
</TABLE>"#;
        let table = parse_html_table(html).unwrap();
        assert_eq!(table.header, vec!["A", "B"]);
        assert_eq!(table.rows, vec![vec!["1", "2"]]);
    }

    #[test]
    fn parse_table_single_line() {
        let html = r#"<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"#;
        let table = parse_html_table(html).unwrap();
        assert_eq!(table.header, vec!["A", "B"]);
        assert_eq!(table.rows, vec![vec!["1", "2"]]);
    }

    #[test]
    fn render_html_table_produces_output() {
        let html = r#"<table>
<tr><th>部门</th><th>姓名</th><th>职位</th></tr>
<tr><td rowspan="2">研发部</td><td>张三</td><td>前端工程师</td></tr>
<tr><td>王五</td><td>后端工程师</td></tr>
</table>"#;
        let table = parse_html_table(html).unwrap();
        let rendered = render_html_table("", &table);
        // 应该包含边框和内容
        assert!(rendered.contains("部门"));
        assert!(rendered.contains("研发部"));
        assert!(rendered.contains("张三"));
        assert!(rendered.contains("前端工程师"));
        assert!(rendered.contains("王五"));
        assert!(rendered.contains("后端工程师"));
        // header 1 行，rowspan 展开后数据 2 行，再加 top/mid/bottom 共 6 行。
        let line_count = rendered.lines().count();
        assert_eq!(line_count, 6);
    }

    #[test]
    fn invalid_html_returns_none() {
        assert!(parse_html_table("not a table").is_none());
        assert!(parse_html_table("<div>hello</div>").is_none());
    }

    #[test]
    fn parse_table_with_thead_tbody() {
        let html = r#"<table>
<thead>
<tr><th>A</th><th>B</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>2</td></tr>
</tbody>
</table>"#;
        let table = parse_html_table(html).unwrap();
        assert_eq!(table.header, vec!["A", "B"]);
        assert_eq!(table.rows, vec![vec!["1", "2"]]);
    }
}
