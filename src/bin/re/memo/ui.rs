use colored::Colorize;

use crate::memo::model::MemoRecord;

pub fn print_separator() {
    println!("{}", "~".repeat(20).blue());
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

pub fn format_record_like_go(record: &MemoRecord, verbose: bool, highlighted_title: Option<String>) -> String {
    let mut out = String::new();
    let tags = colorize_tags(&record.tags);
    let tags_display = if tags.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", tags.join(" "))
    };

    out.push_str(&format!("ID: {}\n", record.id));
    out.push_str(&format!("Tags: {}\n", tags_display));
    out.push_str("MyProblem: true\n");
    out.push_str(&format!("Finished: {}\n", record.finished));
    out.push_str(&format!("Hold: {}\n", record.hold));

    if verbose {
        out.push_str(&format!("AddDate: {}\n", record.add_date));
        out.push_str(&format!("ModifiedDate: {}\n", record.modified_date));
    }

    let title = highlighted_title.unwrap_or_else(|| record.title.clone());
    out.push_str(&format!("Title: {title}\n"));

    out.trim_end_matches('\n').to_string()
}

