use colored::Colorize;

use crate::features::core::*;
use crate::memo::{history, sync, MemoBackend};

pub fn push_single(db: &MemoBackend, push_ref: &str, cli_host: &str) {
    println!("pushing...");
    let host = resolve_host(cli_host);
    println!("target: {}", sync::remote_sqlite_target_display(&host));

    let mut reference = push_ref.trim().to_string();
    if reference.is_empty() {
        reference = crate::common::prompt::read_line("Input the ObjectID/title/tag: ")
            .trim()
            .to_string();
    }

    if !is_object_id_like(&reference) {
        let tags = parse_tag_query(&reference);
        if !tags.is_empty() {
            let records = db
                .list_records_by_tags(&tags, false, false, -1, true, true)
                .unwrap_or_default();
            if !records.is_empty() {
                let record_ids = records.iter().map(|record| record.id.clone()).collect::<Vec<_>>();
                println!("begin to sync {} records...", record_ids.len());
                let synced = sync::sync_records_to_host(db, &record_ids, &host).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
                for record_id in &record_ids {
                    history::write_previous_operation(record_id);
                }
                println!("finished push {} records", synced);
                return;
            }
        }
    }

    let id = if is_object_id_like(&reference) {
        reference.clone()
    } else {
        resolve_record_ref_local(db, &reference, true, true).unwrap_or_default()
    };
    if !is_object_id_like(&id) {
        eprintln!("invalid ObjectID: {id}");
        std::process::exit(1);
    }

    let title_preview = db
        .get_record(&id)
        .unwrap_or(None)
        .map(|r| primary_title(&r.title))
        .unwrap_or_else(|| id.clone());

    sync::sync_record_to_host(db, &id, &host).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    println!(
        "finished push {}:",
        crate::strw::substring_quiet(&title_preview, 0, 20).green()
    );
}
