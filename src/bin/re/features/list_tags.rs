use colored::Colorize;

use crate::features::core::*;
use crate::memo::{ui, MemoBackend};

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
            tags.sort_by(|a, b| a.modified_date.cmp(&b.modified_date));
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
