use colored::Colorize;

use crate::features::core::*;
use crate::memo::MemoBackend;
use crate::memo::sync;

pub fn pull_single(db: &MemoBackend, pull_ref: &str, cli_host: &str) {
    println!("pulling...");
    let host = resolve_host(cli_host);
    println!("source: {}", sync::remote_sqlite_target_display(&host));

    let mut reference = pull_ref.trim().to_string();
    if reference.is_empty() {
        reference = crate::common::prompt::read_line("Input the ObjectID/title/tag: ")
            .trim()
            .to_string();
    }

    sync::sync_record_from_host(db, &reference, &host).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let label = if is_object_id_like(&reference) {
        db.get_record(&reference)
            .unwrap_or(None)
            .map(|r| primary_title(&r.title))
            .unwrap_or(reference)
    } else {
        reference
    };
    println!(
        "finished pull {}:",
        crate::strw::substring_quiet(&label, 0, 20).green()
    );
}
