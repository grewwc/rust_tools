use super::*;

#[derive(Clone)]
pub(super) enum BlockOp {
    Simple(Value),
    Descendant {
        children_id: Vec<String>,
        descendants: Vec<Value>,
    },
}

#[derive(Clone)]
pub(super) enum MdNode {
    Heading {
        level: u8,
        elements: Vec<Value>,
    },
    Paragraph {
        elements: Vec<Value>,
    },
    BulletList {
        items: Vec<ListItem>,
    },
    OrderedList {
        items: Vec<ListItem>,
    },
    TodoList {
        items: Vec<ListItem>,
    },
    CodeBlock {
        lang: Option<String>,
        content: String,
    },
    BlockQuote {
        children: Vec<MdNode>,
    },
    Table {
        rows: Vec<Vec<String>>,
    },
    Divider,
}

#[derive(Clone)]
pub(super) struct ListItem {
    pub(super) elements: Vec<Value>,
    pub(super) children: Vec<MdNode>,
    pub(super) done: Option<bool>,
}

pub(super) fn count_indent(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ').count()
}

pub(super) fn is_table_separator_line(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
        return false;
    }
    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return false;
    }
    inner.split('|').all(|cell| {
        let c = cell.trim();
        c.starts_with('-')
            && c.ends_with('-')
            && c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')
    })
}

pub(super) fn parse_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let inner = if trimmed.starts_with('|') && trimmed.ends_with('|') {
        &trimmed[1..trimmed.len() - 1]
    } else if trimmed.starts_with('|') {
        &trimmed[1..]
    } else if trimmed.ends_with('|') {
        &trimmed[..trimmed.len() - 1]
    } else {
        trimmed
    };
    inner.split('|').map(|c| c.trim().to_string()).collect()
}

pub(super) fn strip_ordered_list_prefix(s: &str) -> Option<&str> {
    let mut chars = s.char_indices().peekable();
    let mut found_digit = false;
    while let Some(&(idx, c)) = chars.peek() {
        if c.is_ascii_digit() {
            found_digit = true;
            chars.next();
        } else if found_digit && c == '.' {
            chars.next();
            let rest = &s[idx + 1..];
            return Some(rest.trim_start());
        } else {
            break;
        }
    }
    None
}

pub(super) fn parse_heading_node(trimmed: &str) -> Option<MdNode> {
    let (level, rest) = if let Some(r) = trimmed.strip_prefix("###### ") {
        (6u8, r)
    } else if let Some(r) = trimmed.strip_prefix("##### ") {
        (5, r)
    } else if let Some(r) = trimmed.strip_prefix("#### ") {
        (4, r)
    } else if let Some(r) = trimmed.strip_prefix("### ") {
        (3, r)
    } else if let Some(r) = trimmed.strip_prefix("## ") {
        (2, r)
    } else if let Some(r) = trimmed.strip_prefix("# ") {
        (1, r)
    } else {
        return None;
    };
    Some(MdNode::Heading {
        level,
        elements: parse_inline_elements(rest),
    })
}

pub(super) fn detect_list_marker(trimmed: &str) -> Option<(ListKind, &str)> {
    if let Some(rest) = trimmed.strip_prefix("- [x] ") {
        return Some((ListKind::Todo(true), rest));
    }
    if let Some(rest) = trimmed.strip_prefix("- [ ] ") {
        return Some((ListKind::Todo(false), rest));
    }
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
    {
        return Some((ListKind::Bullet, rest));
    }
    if let Some(rest) = strip_ordered_list_prefix(trimmed) {
        return Some((ListKind::Ordered, rest));
    }
    None
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum ListKind {
    Bullet,
    Ordered,
    Todo(bool),
}

pub(super) fn parse_markdown_ast(markdown: &str) -> Vec<MdNode> {
    let lines: Vec<&str> = markdown.lines().collect();
    let mut ctx = ParseCtx {
        lines: &lines,
        pos: 0,
    };
    let mut nodes = Vec::new();
    while ctx.pos < lines.len() {
        if let Some(node) = parse_next_node(&mut ctx) {
            nodes.push(node);
        }
    }
    nodes
}

pub(super) struct ParseCtx<'a> {
    lines: &'a [&'a str],
    pos: usize,
}

pub(super) fn parse_next_node(ctx: &mut ParseCtx) -> Option<MdNode> {
    while ctx.pos < ctx.lines.len() && ctx.lines[ctx.pos].trim().is_empty() {
        ctx.pos += 1;
    }
    if ctx.pos >= ctx.lines.len() {
        return None;
    }

    let line = ctx.lines[ctx.pos];
    let trimmed = line.trim();

    if trimmed.starts_with("```") {
        return Some(parse_code_block_node(ctx));
    }

    if is_table_start_at(ctx.lines, ctx.pos) {
        return Some(parse_table_node(ctx));
    }

    if let Some(node) = parse_heading_node(trimmed) {
        ctx.pos += 1;
        return Some(node);
    }

    if trimmed.starts_with('>') {
        return Some(parse_block_quote_node(ctx));
    }

    if trimmed == "---" || trimmed == "***" || trimmed == "___" {
        ctx.pos += 1;
        return Some(MdNode::Divider);
    }

    if detect_list_marker(trimmed).is_some() {
        return Some(parse_list_node(ctx, 0));
    }

    Some(parse_paragraph_node(ctx))
}

pub(super) fn parse_code_block_node(ctx: &mut ParseCtx) -> MdNode {
    parse_code_block_node_with_indent(ctx, 0)
}

pub(super) fn parse_code_block_node_with_indent(ctx: &mut ParseCtx, base_indent: usize) -> MdNode {
    let first = ctx.lines[ctx.pos].trim();
    let lang = first.strip_prefix("```").map(|s| s.trim().to_string());
    let lang = lang.filter(|s| !s.is_empty());
    ctx.pos += 1;

    let mut content = String::new();
    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        if line.trim().starts_with("```") {
            ctx.pos += 1;
            break;
        }
        if !content.is_empty() {
            content.push('\n');
        }
        let code_line =
            if line.len() > base_indent && line.chars().take(base_indent).all(|c| c == ' ') {
                &line[base_indent..]
            } else {
                line.trim_start()
            };
        content.push_str(code_line);
        ctx.pos += 1;
    }

    MdNode::CodeBlock {
        lang,
        content: content.trim_end().to_string(),
    }
}

pub(super) fn is_table_start_at(lines: &[&str], pos: usize) -> bool {
    let trimmed = lines[pos].trim();
    if !trimmed.contains('|') {
        return false;
    }
    if pos + 1 >= lines.len() {
        return false;
    }
    is_table_separator_line(lines[pos + 1].trim())
}

pub(super) fn parse_table_node(ctx: &mut ParseCtx) -> MdNode {
    let mut rows: Vec<Vec<String>> = Vec::new();

    while ctx.pos < ctx.lines.len() {
        let trimmed = ctx.lines[ctx.pos].trim();
        if trimmed.is_empty() {
            break;
        }
        if !trimmed.contains('|') {
            break;
        }
        if is_table_separator_line(trimmed) {
            ctx.pos += 1;
            continue;
        }
        rows.push(parse_table_row(trimmed));
        ctx.pos += 1;
    }

    MdNode::Table { rows }
}

pub(super) fn parse_block_quote_node(ctx: &mut ParseCtx) -> MdNode {
    let mut inner_lines: Vec<String> = Vec::new();

    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        let trimmed = line.trim();
        if trimmed.is_empty() {
            let mut j = ctx.pos + 1;
            while j < ctx.lines.len() && ctx.lines[j].trim().is_empty() {
                j += 1;
            }
            if j >= ctx.lines.len() || !ctx.lines[j].trim().starts_with('>') {
                break;
            }
            inner_lines.push(String::new());
            ctx.pos += 1;
            continue;
        }
        if !trimmed.starts_with('>') {
            break;
        }
        let content = if trimmed == ">" {
            String::new()
        } else if let Some(rest) = trimmed.strip_prefix("> ") {
            rest.to_string()
        } else {
            trimmed[1..].to_string()
        };
        inner_lines.push(content);
        ctx.pos += 1;
    }

    let inner_text = inner_lines.join("\n");
    let children = parse_markdown_ast(&inner_text);

    MdNode::BlockQuote { children }
}

pub(super) fn parse_list_node(ctx: &mut ParseCtx, base_indent: usize) -> MdNode {
    let first_trimmed = ctx.lines[ctx.pos].trim_start();
    let (first_kind, first_rest) =
        detect_list_marker(first_trimmed).expect("parse_list_node called on non-list line");

    let mut items: Vec<ListItem> = Vec::new();
    items.push(ListItem {
        elements: parse_inline_elements(first_rest),
        children: Vec::new(),
        done: match first_kind {
            ListKind::Todo(done) => Some(done),
            _ => None,
        },
    });
    ctx.pos += 1;

    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        if line.trim().is_empty() {
            let mut j = ctx.pos + 1;
            while j < ctx.lines.len() && ctx.lines[j].trim().is_empty() {
                j += 1;
            }
            if j >= ctx.lines.len() {
                break;
            }
            let next_indent = count_indent(ctx.lines[j]);
            let next_trimmed = ctx.lines[j].trim_start();
            if next_indent < base_indent {
                break;
            }
            if next_indent == base_indent && detect_list_marker(next_trimmed).is_none() {
                break;
            }
            ctx.pos += 1;
            continue;
        }

        let indent = count_indent(line);
        let trimmed = line.trim_start();

        if indent < base_indent {
            break;
        }

        if indent == base_indent {
            if let Some((kind, rest)) = detect_list_marker(trimmed) {
                let same_type = match (&first_kind, &kind) {
                    (ListKind::Bullet, ListKind::Bullet) => true,
                    (ListKind::Ordered, ListKind::Ordered) => true,
                    (ListKind::Todo(_), ListKind::Todo(_)) => true,
                    _ => false,
                };
                if !same_type {
                    break;
                }
                items.push(ListItem {
                    elements: parse_inline_elements(rest),
                    children: Vec::new(),
                    done: match kind {
                        ListKind::Todo(done) => Some(done),
                        _ => None,
                    },
                });
                ctx.pos += 1;
            } else {
                break;
            }
        } else {
            let nested = parse_nested_content(ctx, indent);
            if let Some(last) = items.last_mut() {
                last.children.extend(nested);
            }
        }
    }

    match first_kind {
        ListKind::Bullet => MdNode::BulletList { items },
        ListKind::Ordered => MdNode::OrderedList { items },
        ListKind::Todo(_) => MdNode::TodoList { items },
    }
}

pub(super) fn parse_nested_content(ctx: &mut ParseCtx, indent: usize) -> Vec<MdNode> {
    let mut nodes = Vec::new();

    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        if line.trim().is_empty() {
            let mut j = ctx.pos + 1;
            while j < ctx.lines.len() && ctx.lines[j].trim().is_empty() {
                j += 1;
            }
            if j >= ctx.lines.len() || count_indent(ctx.lines[j]) < indent {
                break;
            }
            ctx.pos += 1;
            continue;
        }

        let cur_indent = count_indent(line);
        if cur_indent < indent {
            break;
        }

        if cur_indent >= indent {
            let sub_indent = cur_indent;
            let trimmed = line.trim_start();

            if let Some((kind, rest)) = detect_list_marker(trimmed) {
                let item = ListItem {
                    elements: parse_inline_elements(rest),
                    children: Vec::new(),
                    done: match kind {
                        ListKind::Todo(done) => Some(done),
                        _ => None,
                    },
                };
                ctx.pos += 1;

                let mut sub_items = vec![item];
                while ctx.pos < ctx.lines.len() {
                    let sub_line = ctx.lines[ctx.pos];
                    if sub_line.trim().is_empty() {
                        let mut j = ctx.pos + 1;
                        while j < ctx.lines.len() && ctx.lines[j].trim().is_empty() {
                            j += 1;
                        }
                        if j >= ctx.lines.len() || count_indent(ctx.lines[j]) < sub_indent {
                            break;
                        }
                        ctx.pos += 1;
                        continue;
                    }
                    let si = count_indent(sub_line);
                    let st = sub_line.trim_start();
                    if si < sub_indent {
                        break;
                    }
                    if si == sub_indent {
                        if let Some((k2, r2)) = detect_list_marker(st) {
                            let same = matches!(
                                (&kind, &k2),
                                (ListKind::Bullet, ListKind::Bullet)
                                    | (ListKind::Ordered, ListKind::Ordered)
                                    | (ListKind::Todo(_), ListKind::Todo(_))
                            );
                            if !same {
                                break;
                            }
                            sub_items.push(ListItem {
                                elements: parse_inline_elements(r2),
                                children: Vec::new(),
                                done: match k2 {
                                    ListKind::Todo(done) => Some(done),
                                    _ => None,
                                },
                            });
                            ctx.pos += 1;
                        } else {
                            break;
                        }
                    } else {
                        let nested = parse_nested_content(ctx, si);
                        if let Some(last) = sub_items.last_mut() {
                            last.children.extend(nested);
                        }
                    }
                }

                let list_node = match kind {
                    ListKind::Bullet => MdNode::BulletList { items: sub_items },
                    ListKind::Ordered => MdNode::OrderedList { items: sub_items },
                    ListKind::Todo(_) => MdNode::TodoList { items: sub_items },
                };
                nodes.push(list_node);
            } else if trimmed.starts_with('>') {
                nodes.push(parse_block_quote_node(ctx));
            } else if trimmed.starts_with("```") {
                nodes.push(parse_code_block_node_with_indent(ctx, indent));
            } else if let Some(node) = parse_heading_node(trimmed) {
                ctx.pos += 1;
                nodes.push(node);
            } else if trimmed == "---" || trimmed == "***" || trimmed == "___" {
                ctx.pos += 1;
                nodes.push(MdNode::Divider);
            } else {
                let mut para = trimmed.to_string();
                ctx.pos += 1;
                while ctx.pos < ctx.lines.len() {
                    let pl = ctx.lines[ctx.pos];
                    if pl.trim().is_empty() {
                        break;
                    }
                    let pi = count_indent(pl);
                    if pi < indent {
                        break;
                    }
                    if detect_list_marker(pl.trim_start()).is_some()
                        || pl.trim_start().starts_with('>')
                        || pl.trim_start().starts_with("```")
                        || parse_heading_node(pl.trim()).is_some()
                    {
                        break;
                    }
                    para.push(' ');
                    para.push_str(pl.trim());
                    ctx.pos += 1;
                }
                nodes.push(MdNode::Paragraph {
                    elements: parse_inline_elements(&para),
                });
            }
        }
    }

    nodes
}

pub(super) fn parse_paragraph_node(ctx: &mut ParseCtx) -> MdNode {
    let mut text = ctx.lines[ctx.pos].trim().to_string();
    ctx.pos += 1;

    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if trimmed.starts_with("```")
            || parse_heading_node(trimmed).is_some()
            || trimmed.starts_with('>')
            || detect_list_marker(trimmed).is_some()
            || trimmed == "---"
            || trimmed == "***"
            || trimmed == "___"
            || is_table_start_at(ctx.lines, ctx.pos)
        {
            break;
        }
        text.push(' ');
        text.push_str(trimmed);
        ctx.pos += 1;
    }

    MdNode::Paragraph {
        elements: parse_inline_elements(text.trim()),
    }
}

pub(super) fn alloc_id(counter: &mut usize) -> String {
    let id = format!("blk_{}", counter);
    *counter += 1;
    id
}

pub(super) fn heading_block_type(level: u8) -> i64 {
    match level {
        1 => 3,
        2 => 4,
        3 => 5,
        4 => 6,
        5 => 7,
        6 => 8,
        _ => 2,
    }
}

pub(super) fn heading_block_key(level: u8) -> &'static str {
    match level {
        1 => "heading1",
        2 => "heading2",
        3 => "heading3",
        4 => "heading4",
        5 => "heading5",
        6 => "heading6",
        _ => "text",
    }
}

pub(super) fn md_node_to_block_ops(node: MdNode, id_counter: &mut usize) -> Vec<BlockOp> {
    match node {
        MdNode::Heading { level, elements } => {
            let bt = heading_block_type(level);
            let key = heading_block_key(level);
            let mut block = json!({
                "block_type": bt,
            });
            block[key] = json!({ "elements": elements });
            vec![BlockOp::Simple(block)]
        }
        MdNode::Paragraph { elements } => {
            vec![BlockOp::Simple(json!({
                "block_type": 2,
                "text": { "elements": elements }
            }))]
        }
        MdNode::CodeBlock { lang, content } => {
            let style = build_code_block_style(lang.as_deref());
            vec![BlockOp::Simple(json!({
                "block_type": 14,
                "code": {
                    "elements": [{ "text_run": { "content": content, "text_element_style": {} } }],
                    "style": style
                }
            }))]
        }
        MdNode::Divider => {
            vec![BlockOp::Simple(json!({
                "block_type": 22,
                "divider": {}
            }))]
        }
        MdNode::BulletList { items } => {
            let mut ops = Vec::new();
            for item in items {
                ops.extend(list_item_to_block_ops(item, 12, "bullet", id_counter));
            }
            ops
        }
        MdNode::OrderedList { items } => {
            let mut ops = Vec::new();
            for item in items {
                ops.extend(list_item_to_block_ops(item, 13, "ordered", id_counter));
            }
            ops
        }
        MdNode::TodoList { items } => {
            let mut ops = Vec::new();
            for item in items {
                ops.extend(list_item_to_block_ops(item, 17, "todo", id_counter));
            }
            ops
        }
        MdNode::BlockQuote { children } => {
            if children.is_empty() {
                return vec![BlockOp::Simple(json!({
                    "block_type": 15,
                    "quote": {}
                }))];
            }
            let quote_id = alloc_id(id_counter);
            let mut child_ids: Vec<String> = Vec::new();
            let mut all_descendants: Vec<Value> = Vec::new();

            for child in children {
                let (root_ids, descs) = md_node_to_descendant_blocks(child, id_counter);
                child_ids.extend(root_ids);
                all_descendants.extend(descs);
            }

            let quote_block = json!({
                "block_id": quote_id,
                "block_type": 15,
                "quote": {},
                "children": child_ids
            });
            let mut descendants = vec![quote_block];
            descendants.extend(all_descendants);
            vec![BlockOp::Descendant {
                children_id: vec![quote_id],
                descendants,
            }]
        }
        MdNode::Table { rows } => {
            vec![build_table_descendant(&rows, id_counter)]
        }
    }
}

pub(super) fn list_item_to_block_ops(
    item: ListItem,
    block_type: i64,
    key: &str,
    id_counter: &mut usize,
) -> Vec<BlockOp> {
    let content = build_list_block_content(&item, block_type);
    if item.children.is_empty() {
        let mut block = json!({ "block_type": block_type });
        block[key] = content;
        vec![BlockOp::Simple(block)]
    } else {
        let item_id = alloc_id(id_counter);
        let mut child_ids: Vec<String> = Vec::new();
        let mut all_descendants: Vec<Value> = Vec::new();

        for child in item.children {
            let (root_ids, descs) = md_node_to_descendant_blocks(child, id_counter);
            child_ids.extend(root_ids);
            all_descendants.extend(descs);
        }

        let mut item_block = json!({
            "block_id": item_id,
            "block_type": block_type,
            "children": child_ids
        });
        item_block[key] = content;

        let mut descendants = vec![item_block];
        descendants.extend(all_descendants);

        vec![BlockOp::Descendant {
            children_id: vec![item_id],
            descendants,
        }]
    }
}

pub(super) fn md_node_to_descendant_blocks(node: MdNode, id_counter: &mut usize) -> (Vec<String>, Vec<Value>) {
    match node {
        MdNode::Heading { level, elements } => {
            let id = alloc_id(id_counter);
            let bt = heading_block_type(level);
            let key = heading_block_key(level);
            let mut block = json!({
                "block_id": id,
                "block_type": bt,
                "children": []
            });
            block[key] = json!({ "elements": elements });
            (vec![id], vec![block])
        }
        MdNode::Paragraph { elements } => {
            let id = alloc_id(id_counter);
            let block = json!({
                "block_id": id,
                "block_type": 2,
                "text": { "elements": elements },
                "children": []
            });
            (vec![id], vec![block])
        }
        MdNode::CodeBlock { lang, content } => {
            let id = alloc_id(id_counter);
            let style = build_code_block_style(lang.as_deref());
            let block = json!({
                "block_id": id,
                "block_type": 14,
                "code": {
                    "elements": [{ "text_run": { "content": content, "text_element_style": {} } }],
                    "style": style
                },
                "children": []
            });
            (vec![id], vec![block])
        }
        MdNode::Divider => {
            let id = alloc_id(id_counter);
            let block = json!({
                "block_id": id,
                "block_type": 22,
                "divider": {},
                "children": []
            });
            (vec![id], vec![block])
        }
        MdNode::BulletList { items } => {
            let mut root_ids = Vec::new();
            let mut all_blocks = Vec::new();
            for item in items {
                let (ids, blocks) = list_item_to_descendant(item, 12, "bullet", id_counter);
                root_ids.extend(ids);
                all_blocks.extend(blocks);
            }
            (root_ids, all_blocks)
        }
        MdNode::OrderedList { items } => {
            let mut root_ids = Vec::new();
            let mut all_blocks = Vec::new();
            for item in items {
                let (ids, blocks) = list_item_to_descendant(item, 13, "ordered", id_counter);
                root_ids.extend(ids);
                all_blocks.extend(blocks);
            }
            (root_ids, all_blocks)
        }
        MdNode::TodoList { items } => {
            let mut root_ids = Vec::new();
            let mut all_blocks = Vec::new();
            for item in items {
                let (ids, blocks) = list_item_to_descendant(item, 17, "todo", id_counter);
                root_ids.extend(ids);
                all_blocks.extend(blocks);
            }
            (root_ids, all_blocks)
        }
        MdNode::BlockQuote { children } => {
            let quote_id = alloc_id(id_counter);
            let mut child_ids: Vec<String> = Vec::new();
            let mut all_blocks: Vec<Value> = Vec::new();
            for child in children {
                let (ids, blocks) = md_node_to_descendant_blocks(child, id_counter);
                child_ids.extend(ids);
                all_blocks.extend(blocks);
            }
            let quote_block = json!({
                "block_id": quote_id,
                "block_type": 15,
                "quote": {},
                "children": child_ids
            });
            let mut result = vec![quote_block];
            result.extend(all_blocks);
            (vec![quote_id], result)
        }
        MdNode::Table { rows } => {
            let (ids, blocks) = build_table_descendant_data(&rows, id_counter);
            (ids, blocks)
        }
    }
}

pub(super) fn list_item_to_descendant(
    item: ListItem,
    block_type: i64,
    key: &str,
    id_counter: &mut usize,
) -> (Vec<String>, Vec<Value>) {
    let item_id = alloc_id(id_counter);
    let mut child_ids: Vec<String> = Vec::new();
    let mut all_blocks: Vec<Value> = Vec::new();
    let content = build_list_block_content(&item, block_type);

    for child in item.children {
        let (ids, blocks) = md_node_to_descendant_blocks(child, id_counter);
        child_ids.extend(ids);
        all_blocks.extend(blocks);
    }

    let mut item_block = json!({
        "block_id": item_id,
        "block_type": block_type,
        "children": child_ids
    });
    item_block[key] = content;

    let mut result = vec![item_block];
    result.extend(all_blocks);
    (vec![item_id], result)
}

pub(super) fn build_list_block_content(item: &ListItem, block_type: i64) -> Value {
    if block_type == 17 {
        json!({
            "elements": item.elements.clone(),
            "style": { "done": item.done.unwrap_or(false) }
        })
    } else {
        json!({ "elements": item.elements.clone() })
    }
}

pub(super) fn build_code_block_style(lang: Option<&str>) -> Value {
    let mut style = json!({ "wrap": true });
    if let Some(language) = map_feishu_code_language(lang) {
        style["language"] = json!(language);
    }
    style
}

pub(super) fn map_feishu_code_language(lang: Option<&str>) -> Option<i64> {
    let normalized = lang?.trim().to_ascii_lowercase();
    let value = match normalized.as_str() {
        "" | "text" | "plaintext" | "plain" => 1,
        "bash" | "sh" | "zsh" | "shell" => 7,
        "csharp" | "cs" => 8,
        "cpp" | "c++" => 9,
        "c" => 10,
        "css" => 12,
        "dockerfile" => 18,
        "go" => 22,
        "html" => 24,
        "json" => 28,
        "java" => 29,
        "javascript" | "js" => 30,
        "kotlin" | "kt" => 32,
        "markdown" | "md" | "mermaid" => 39,
        "php" => 43,
        "python" | "py" => 49,
        "ruby" | "rb" => 52,
        "rust" | "rs" => 53,
        "sql" => 56,
        "swift" => 61,
        "typescript" | "ts" => 63,
        "xml" => 66,
        "yaml" | "yml" => 67,
        "cmake" => 68,
        "graphql" | "gql" => 71,
        "toml" => 75,
        _ => return None,
    };
    Some(value)
}

pub(super) fn build_table_descendant(rows: &[Vec<String>], id_counter: &mut usize) -> BlockOp {
    let (children_id, descendants) = build_table_descendant_data(rows, id_counter);
    BlockOp::Descendant {
        children_id,
        descendants,
    }
}

pub(super) fn build_table_descendant_data(
    rows: &[Vec<String>],
    id_counter: &mut usize,
) -> (Vec<String>, Vec<Value>) {
    let row_size = rows.len();
    let column_size = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if row_size == 0 || column_size == 0 {
        let id = alloc_id(id_counter);
        return (
            vec![id.clone()],
            vec![json!({
                "block_id": id,
                "block_type": 2,
                "text": { "elements": [make_text_element("", false, false, false)] },
                "children": []
            })],
        );
    }

    let table_id = alloc_id(id_counter);
    let mut cell_ids: Vec<String> = Vec::new();
    let mut descendants: Vec<Value> = Vec::new();

    for row in rows {
        for col_idx in 0..column_size {
            let cell_id = alloc_id(id_counter);
            cell_ids.push(cell_id.clone());

            let cell_text = row.get(col_idx).cloned().unwrap_or_default();
            let child_id = alloc_id(id_counter);

            let cell_block = json!({
                "block_id": cell_id,
                "block_type": 32,
                "table_cell": {},
                "children": [child_id]
            });

            let child_block = json!({
                "block_id": child_id,
                "block_type": 2,
                "text": { "elements": parse_inline_elements(&cell_text) },
                "children": []
            });

            descendants.push(cell_block);
            descendants.push(child_block);
        }
    }

    let table_block = json!({
        "block_id": table_id,
        "block_type": 31,
        "table": {
            "property": {
                "row_size": row_size,
                "column_size": column_size,
                "header_row": true
            }
        },
        "children": cell_ids
    });

    let mut all_descendants = vec![table_block];
    all_descendants.extend(descendants);
    (vec![table_id], all_descendants)
}

pub(super) fn convert_markdown_to_docx_blocks(markdown: &str) -> Vec<BlockOp> {
    let ast = parse_markdown_ast(markdown);
    let mut id_counter: usize = 1;
    let mut ops = Vec::new();
    for node in ast {
        ops.extend(md_node_to_block_ops(node, &mut id_counter));
    }
    ops
}
