use std::io::{self, Write};

use colored::Colorize;

use crate::features::core::*;
use crate::memo::{MemoBackend, ui};

pub fn list_tags_feature(
    db: &MemoBackend,
    cli: &Cli,
    record_limit: i64,
    reverse: bool,
    all: bool,
    list_special: bool,
    list_tags_and_order_by_time: bool,
) {
    let mut tags = if all || list_tags_and_order_by_time {
        let include_finished = !list_tags_and_order_by_time || all;
        let mut tags = compute_tags_from_records(db, include_finished, list_special);
        if list_tags_and_order_by_time {
            tags.sort_by_key(|a| a.modified_date);
        } else {
            tags.sort_by(|a, b| {
                b.count
                    .cmp(&a.count)
                    .then_with(|| b.modified_date.cmp(&a.modified_date))
            });
        }
        if reverse {
            tags.reverse();
        }
        tags
    } else {
        let mut tags = db.list_tags(None, None, record_limit).unwrap_or_default();
        if reverse {
            tags.reverse();
        }
        tags
    };

    if record_limit > 0 && tags.len() > record_limit as usize {
        tags.truncate(record_limit as usize);
    }
    tags = filter_tags_ex(tags, &cli.ex);

    if cli.verbose {
        for mut tag in tags {
            tag.name = tag.name.bright_green().to_string();
            ui::print_separator();
            println!("ID: {}", tag.id);
            println!("Name: {}", tag.name);
            println!("Count: {}", tag.count);
            println!("ModifiedDate: {}", tag.modified_date);
        }
        return;
    }

    let terminal_width = get_terminal_width();
    let terminal_indent = 2usize;
    let delimiter = "   ";
    let mut buf = String::new();
    for tag in &tags {
        buf.push_str(&format!("{}[{}]  ", tag.name, tag.count));
    }
    let raw = crate::strw::wrap(
        &buf,
        terminal_width.saturating_sub(terminal_indent),
        terminal_indent,
        delimiter,
    );

    for line in crate::strw::split_no_empty(&raw, "\n") {
        let arr = crate::strw::split_no_empty(line, " ");
        let mut changed = Vec::with_capacity(arr.len());
        for x in arr {
            if let Some(idx) = x.find('[') {
                let name = &x[..idx];
                let rest = &x[idx..];
                changed.push(format!("{}{}", name.bright_green(), rest));
            } else {
                changed.push(x.to_string());
            }
        }
        println!("{}{}", " ".repeat(terminal_indent), changed.join(delimiter));
    }
}

/// 为 shell 自动补全输出匹配前缀的 tag 名称（每行一个）。
/// 被 `--complete-tags <prefix>` 调用。
pub fn complete_tags_feature(db: &MemoBackend, prefix: &str) {
    let prefix = prefix.trim();
    let tags = db
        .list_tags(
            if prefix.is_empty() {
                None
            } else {
                Some(prefix)
            },
            None,
            i64::MAX,
        )
        .unwrap_or_default();
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    for tag in &tags {
        let _ = writeln!(handle, "{}", tag.name);
    }
    // 如果前缀非空且有匹配结果，打印第二遍（不带换行），
    // 让 shell 的 compgen 知道应该直接填充而不是等待更多字符。
}
