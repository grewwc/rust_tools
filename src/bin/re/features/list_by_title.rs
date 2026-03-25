use colored::Colorize;
use regex::Regex;

use crate::features::core::*;
use crate::memo::{MemoBackend, history, ui};

#[allow(clippy::too_many_arguments)]
pub fn list_by_title_feature(
    db: &MemoBackend,
    query: &str,
    tag_query: &str,
    use_and: bool,
    prefix: bool,
    limit: i64,
    reverse: bool,
    include_finished: bool,
    out_name: &str,
    to_binary: bool,
    count_only: bool,
    verbose: bool,
    force: bool,
) {
    let tags = parse_tag_query(tag_query);

    let mut records = if tags.is_empty() {
        db.search_title(query, limit, include_finished)
            .unwrap_or_default()
    } else {
        let mut records = db
            .list_records_by_tags(&tags, use_and, prefix, -1, reverse, include_finished)
            .unwrap_or_default();
        let lower = query.to_lowercase();
        records.retain(|r| r.title.to_lowercase().contains(&lower));
        records.sort_by_key(|a| a.modified_date);
        if !reverse {
            records.reverse();
        }
        if limit > 0 && records.len() > limit as usize {
            records.truncate(limit as usize);
        }
        records
    };

    records.sort_by_key(|a| a.modified_date);
    if !reverse {
        records.reverse();
    }
    if limit > 0 && records.len() > limit as usize {
        records.truncate(limit as usize);
    }

    if count_only {
        println!("{} records found", records.len());
        return;
    }

    if to_binary {
        eprintln!("not supported");
        std::process::exit(1);
    }

    if !out_name.trim().is_empty() {
        let out = records_to_output_text(&records);
        write_text_output(out_name, &out, force);
        return;
    }

    let pattern = Regex::new(&format!("(?i){}", regex::escape(query))).ok();
    for record in records {
        ui::print_separator();
        let mut display = record.clone();
        let mut highlighted_title = None;

        if verbose {
            if is_probably_text(&record.title) {
                highlighted_title = Some(format!(
                    "\n{}",
                    highlight_search_text(&record.title, pattern.as_ref())
                ));
            } else {
                highlighted_title = Some(format!("\n{}", "<binary>".bright_yellow()));
            }
        } else {
            display.title = "<hidden>".to_string();
        }

        println!(
            "{}",
            ui::format_record_like_go(&display, verbose, highlighted_title)
        );
        println!("{}", ui::colorize_id(&record.id));
        history::write_previous_operation(&record.id);
    }
}

pub fn highlight_search_text(text: &str, pattern: Option<&Regex>) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }
    let Some(pattern) = pattern else {
        return text.bright_white().to_string();
    };
    let indices = pattern.find_iter(text).collect::<Vec<_>>();
    if indices.is_empty() {
        return text.bright_white().to_string();
    }

    let mut out = String::new();
    let mut last = 0;
    for m in indices {
        if m.start() > last {
            out.push_str(&text[last..m.start()].bright_white().to_string());
        }
        out.push_str(&text[m.start()..m.end()].red().to_string());
        last = m.end();
    }
    if last < text.len() {
        out.push_str(&text[last..].bright_white().to_string());
    }
    out
}
