use crate::features::core::*;
use crate::memo::{MemoBackend, history};

pub fn delete_feature(db: &MemoBackend, cli: &Cli) {
    let arg = cli.delete.as_deref().unwrap_or("").trim();
    if !arg.is_empty() && delete_by_tag_prefix(db, arg) {
        return;
    }

    let id = resolve_id_or_prev(arg, cli.prev);
    if id.trim().is_empty() {
        return;
    }
    if !is_object_id_like(&id) {
        eprintln!("invalid ObjectID: {id}");
        std::process::exit(1);
    }
    let _ = db.delete(&id);
    history::write_previous_operation(&id);
}

pub fn delete_by_tag_prefix(db: &MemoBackend, tag_ref: &str) -> bool {
    let tags = parse_tag_query(tag_ref);
    if tags.is_empty() {
        return false;
    }
    let records = db
        .list_records_by_tags(&tags, false, true, -1, false, true)
        .unwrap_or_default();
    if records.is_empty() {
        return false;
    }
    for r in records {
        println!("deleting record. id:{}, tag:{:?}", r.id, r.tags);
        let _ = db.delete(&r.id);
    }
    true
}
