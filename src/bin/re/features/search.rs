use colored::Colorize;
use regex::Regex;

use crate::features::core::*;
use crate::memo::{MemoBackend, history, search as memo_search, ui};

const SEARCH_PREVIEW_MAX_LINES: usize = 6;

pub fn search_feature(
    db: &MemoBackend,
    query: &str,
    tag_query: &str,
    use_and: bool,
    prefix: bool,
    limit: i64,
    include_finished: bool,
    list_special: bool,
    out_name: &str,
    to_binary: bool,
    count_only: bool,
    verbose: bool,
    force: bool,
) {
    let tags = parse_tag_query(tag_query);
    let mut records = if tags.is_empty() {
        db.list_records(-1, false, include_finished)
            .unwrap_or_default()
    } else {
        db.list_records_by_tags(&tags, use_and, prefix, -1, false, include_finished)
            .unwrap_or_default()
    };

    if !list_special {
        let patterns = load_special_patterns();
        records.retain(|r| !is_special_record(r, &patterns));
    }

    let mut results = Vec::new();
    for record in records {
        let score = memo_search::score_record(&record, query);
        if score <= 0.0 {
            continue;
        }
        let preview = memo_search::build_preview_lines(&record, query, SEARCH_PREVIEW_MAX_LINES);
        results.push(SearchHit {
            score,
            record,
            preview,
        });
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.record.modified_date.cmp(&a.record.modified_date))
    });
    if limit > 0 && results.len() > limit as usize {
        results.truncate(limit as usize);
    }

    if count_only {
        println!("{} records found", results.len());
        return;
    }

    if to_binary {
        eprintln!("not supported");
        std::process::exit(1);
    }

    if !out_name.trim().is_empty() {
        let mut out = String::new();
        for idx in (0..results.len()).rev() {
            let item = &results[idx];
            out.push_str(&format!(
                "{} score={:.3} {}\n",
                "-".repeat(10),
                item.score,
                "-".repeat(10)
            ));
            out.push_str(&item.record.title);
            out.push('\n');
        }
        write_text_output(out_name, &out, force);
        return;
    }

    let pattern = memo_search::highlight_pattern(query);
    for idx in (0..results.len()).rev() {
        if idx < results.len().saturating_sub(1) {
            println!();
        }
        print_search_result(idx, &results[idx], query, pattern.as_ref(), verbose);
        history::write_previous_operation(&results[idx].record.id);
    }
}

pub fn print_search_result(
    index: usize,
    result: &SearchHit,
    query: &str,
    pattern: Option<&Regex>,
    verbose: bool,
) {
    let mut header = vec![
        format!("[{}]", index + 1).bright_blue().to_string(),
        format!("score {:.3}", result.score)
            .bright_white()
            .to_string(),
    ];
    if result.record.finished {
        header.push("finished".yellow().to_string());
    }
    println!("{}", header.join("  "));

    if !result.record.tags.is_empty() {
        println!(
            "    {} {}",
            "tags:".bright_black(),
            ui::colorize_tags(&result.record.tags).join(", ")
        );
    }

    if verbose {
        let mut display = result.record.clone();
        if is_probably_text(&display.title) {
            display.title = format!("\n{}", highlight_search_text(&display.title, pattern));
        } else {
            display.title = format!("\n{}", "<binary>".bright_yellow());
        }
        let formatted = ui::format_record_like_go(&display, true, None);
        for line in formatted.lines() {
            if line.trim().is_empty() {
                continue;
            }
            println!("    {}", line);
        }
        println!(
            "    {} {}",
            "id:".bright_black(),
            ui::colorize_id(&result.record.id)
        );
        return;
    }

    if is_probably_text(&result.record.title) {
        let preview = if result.preview.is_empty() {
            memo_search::build_preview_lines(&result.record, query, SEARCH_PREVIEW_MAX_LINES)
        } else {
            result.preview.clone()
        };
        for line in preview {
            print_search_preview_line(&line, pattern);
        }
    } else {
        println!("    {} {}", "-".cyan(), "<binary>".bright_yellow());
    }
    println!(
        "    {} {}",
        "id:".bright_black(),
        ui::colorize_id(&result.record.id)
    );
}

pub fn print_search_preview_line(line: &str, pattern: Option<&Regex>) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    let (prefix, url, has_url) = split_search_preview_url(line);
    if has_url && !prefix.is_empty() {
        println!(
            "    {} {}",
            "-".cyan(),
            highlight_search_text(&prefix, pattern)
        );
        println!("      {}", highlight_search_text(&url, pattern));
        return;
    }
    println!(
        "    {} {}",
        "-".cyan(),
        highlight_search_text(line, pattern)
    );
}

pub fn split_search_preview_url(line: &str) -> (String, String, bool) {
    let lower = line.to_lowercase();
    let mut start = lower.find("https://");
    if start.is_none() {
        start = lower.find("http://");
    }
    let Some(idx) = start else {
        return (String::new(), String::new(), false);
    };
    let prefix = line[..idx].trim().to_string();
    let url = line[idx..].trim().to_string();
    let has_url = !url.is_empty();
    (prefix, url, has_url)
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
