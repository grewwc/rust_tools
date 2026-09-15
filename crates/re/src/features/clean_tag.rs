use colored::Colorize;

use crate::features::core::*;
use crate::memo::MemoBackend;

pub fn clean_by_tags(db: &MemoBackend, tags_str: &str) {
    let tags = parse_tag_query(tags_str);
    if tags.is_empty() {
        println!("empty tags");
        return;
    }

    let colored_tags = tags
        .iter()
        .map(|x| x.bright_red().to_string())
        .collect::<Vec<_>>();
    println!("cleaning tags: {:?}", colored_tags);

    let records = db
        .list_records_by_tags(&tags, true, false, -1, false, true)
        .unwrap_or_default();
    for record in records {
        let _ = db.delete(&record.id);
    }
}
