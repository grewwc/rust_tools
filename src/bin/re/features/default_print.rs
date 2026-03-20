use colored::Colorize;

use crate::features::core::*;
use crate::memo::{MemoBackend, history, ui};

pub fn default_print(db: &MemoBackend, limit: i64, list_special: bool) {
    let mut records = db
        .list_records_by_tags(
            &vec!["todo".to_string(), "urgent".to_string()],
            false,
            true,
            limit,
            false,
            false,
        )
        .unwrap_or_default();

    if !list_special {
        let patterns = load_special_patterns();
        records.retain(|r| !is_special_record(r, &patterns));
    }

    for record in records {
        ui::print_separator();
        let title = if is_probably_text(&record.title) {
            format!("\n{}", record.title.bright_white())
        } else {
            format!("\n{}", "<binary>".bright_yellow())
        };
        println!("{}", ui::format_record_like_go(&record, false, Some(title)));
        println!("{}", ui::colorize_id(&record.id));
        history::write_previous_operation(&record.id);
    }
}
