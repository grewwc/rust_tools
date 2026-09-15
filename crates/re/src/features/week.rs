use crate::memo::{MemoBackend, time as memo_time};

pub fn handle_week(db: &MemoBackend) {
    let mut first = memo_time::first_day_of_week(memo_time::today_local_date());
    let now = memo_time::today_local_date();
    let week_tag = memo_time::format_week_tag(first);

    let records = db
        .list_records_by_tags(
            std::slice::from_ref(&week_tag),
            false,
            false,
            -1,
            true,
            true,
        )
        .unwrap_or_default();
    if records.len() > 1 {
        eprintln!("too many week tags");
        std::process::exit(1);
    }

    let mut title = String::new();
    while first < now {
        let day_tag = memo_time::format_log_tag(first);
        let day_records = db
            .list_records_by_tags(std::slice::from_ref(&day_tag), false, false, -1, true, true)
            .unwrap_or_default();
        if day_records.len() > 1 {
            eprintln!("log failed");
            std::process::exit(1);
        }
        if let Some(r) = day_records.first() {
            title.push_str(&format!("-- {} --\n", first.format("%Y-%m-%d")));
            title.push_str(&r.title);
            title.push_str("\n\n");
        }
        first = memo_time::add_days(first, 1);
    }

    if let Some(existing) = records.first() {
        let _ = db.update_title(&existing.id, &title);
        crate::memo::history::write_previous_operation(&existing.id);
    } else {
        let id = db
            .insert(&title, std::slice::from_ref(&week_tag))
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
        crate::memo::history::write_previous_operation(&id);
    }
}
