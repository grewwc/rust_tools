use colored::Colorize;

use crate::features::core::*;
use crate::memo::{MemoBackend, MemoRecord, history, ui};

fn list_default_records(db: &MemoBackend, limit: i64) -> Vec<MemoRecord> {
    let tags = parse_tag_query("todo urgent");
    db.list_records_by_tags(&tags, false, true, limit, false, false)
        .unwrap_or_default()
}

fn filter_special_records(records: &mut Vec<MemoRecord>, list_special: bool) {
    if list_special {
        return;
    }
    let patterns = load_special_patterns();
    records.retain(|r| !is_special_record(r, &patterns));
}

fn format_record_title(record: &MemoRecord) -> String {
    if is_probably_text(&record.title) {
        let normalized_title = normalize_title_for_display(&record.title);
        let wrapped_title = wrap_title_for_terminal(&normalized_title, get_terminal_width());
        ui::colorize_title(&print_title_with_colored_separator(&wrapped_title))
    } else {
        format!("\n{}", "<binary>".bright_yellow())
    }
}

fn print_record(record: &MemoRecord) {
    ui::print_separator();
    let title = format_record_title(record);
    println!("{}", ui::format_record_like_go(record, false, Some(title)));
    println!("{}", ui::colorize_id(&record.id));
    history::write_previous_operation(&record.id);
}

pub fn default_print(db: &MemoBackend, limit: i64, list_special: bool) {
    let mut records = list_default_records(db, limit);
    filter_special_records(&mut records, list_special);
    for record in records {
        print_record(&record);
    }
}
