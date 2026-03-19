use crate::features::core::*;
use crate::memo::{history, MemoBackend};

pub fn open_urls(db: &MemoBackend, args: &[String], prefix: bool) {
    let tags = args
        .iter()
        .filter(|x| x.as_str() != "open" && x.as_str() != "o")
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect::<Vec<_>>();

    if tags.len() == 1 && is_object_id_like(&tags[0]) {
        if let Some(record) = db.get_record(&tags[0]).unwrap_or(None) {
            open_urls_from_title(&record.title);
        }
        return;
    }

    if tags.is_empty() {
        let prev_id = history::read_previous_operation();
        if is_object_id_like(&prev_id)
            && let Some(record) = db.get_record(&prev_id).unwrap_or(None)
        {
            open_urls_from_title(&record.title);
        }
        return;
    }

    let records = db
        .list_records_by_tags(&tags, false, prefix, -1, true, true)
        .unwrap_or_default();
    let mut urls = Vec::new();
    for r in records {
        urls.extend(extract_urls(&r.title));
    }

    if urls.is_empty() {
        println!(
            "there are NO urls associated with tags: {:?} (prefix: {})",
            tags, prefix
        );
        return;
    }
    open_choose_url(&urls);
}

pub fn open_urls_from_title(title: &str) {
    let urls = extract_urls(title);
    if urls.is_empty() {
        return;
    }
    open_choose_url(&urls);
}
