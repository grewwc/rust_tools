use chrono::{Local, TimeZone};
use colored::Colorize;

use crate::memo::model::MemoRecord;

pub fn print_separator() {
    println!("{}", "~".repeat(20).blue());
}

fn format_epoch_secs_local(secs: i64) -> String {
    Local
        .timestamp_opt(secs, 0)
        .single()
        .map(|dt| dt.format("%Y/%m/%d %H:%M:%S").to_string())
        .unwrap_or_else(|| secs.to_string())
}

pub fn colorize_tags(tags: &[String]) -> Vec<String> {
    tags.iter().map(|t| t.bright_green().to_string()).collect()
}

pub fn colorize_title(title: &str) -> String {
    format!("\n{}", title.bright_white())
}

pub fn colorize_id(id: &str) -> String {
    id.bright_red().to_string()
}

pub fn format_record_like_go(
    record: &MemoRecord,
    verbose: bool,
    highlighted_title: Option<String>,
) -> String {
    let mut out = String::new();
    let indent = "  ";
    let title_indent = "    ";
    let tags = colorize_tags(&record.tags);
    let tags_display = if tags.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", tags.join(" "))
    };

    out.push_str(&format!("{indent}ID: {}\n", record.id));
    out.push_str(&format!("{indent}Tags: {}\n", tags_display));
    out.push_str(&format!("{indent}MyProblem: true\n"));
    out.push_str(&format!("{indent}Finished: {}\n", record.finished));
    out.push_str(&format!("{indent}Hold: {}\n", record.hold));

    if verbose {
        out.push_str(&format!(
            "{indent}AddDate: {}\n",
            format_epoch_secs_local(record.add_date)
        ));
        out.push_str(&format!(
            "{indent}ModifiedDate: {}\n",
            format_epoch_secs_local(record.modified_date)
        ));
    }

    let title = highlighted_title.unwrap_or_else(|| record.title.clone());
    if title.contains('\n') {
        out.push_str(&format!("{indent}Title:\n"));
        let mut lines = title.lines();
        if title.starts_with('\n') {
            let _ = lines.next();
        }
        for line in lines {
            if line.is_empty() {
                out.push('\n');
            } else {
                out.push_str(title_indent);
                out.push_str(line);
                out.push('\n');
            }
        }
    } else {
        out.push_str(&format!("{indent}Title: {title}\n"));
    }

    out.trim_end_matches('\n').to_string()
}
