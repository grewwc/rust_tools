use std::collections::BTreeSet;

use colored::Colorize;

use crate::features::core::*;
use crate::memo::{MemoBackend, history, sync, ui};

pub fn list_by_tag_name_feature(
    db: &MemoBackend,
    query: &str,
    cli: &Cli,
    prefix: bool,
    limit: i64,
    reverse: bool,
    include_finished: bool,
    list_special: bool,
    only_tags: bool,
    to_binary: bool,
    out_name: &str,
) {
    let tags = parse_tag_query(query);
    if tags.is_empty() {
        return;
    }

    let mut records = if tags.len() == 1 && is_object_id_like(&tags[0]) {
        db.get_record(&tags[0])
            .unwrap_or(None)
            .into_iter()
            .collect()
    } else {
        let effective_reverse = !reverse;
        db.list_records_by_tags(
            &tags,
            cli.r#and,
            prefix,
            limit,
            effective_reverse,
            include_finished,
        )
        .unwrap_or_default()
    };

    if !list_special {
        let patterns = load_special_patterns();
        records.retain(|r| !is_special_record(r, &patterns));
    }

    if cli.count {
        println!("{} records found", records.len());
        return;
    }

    if cli.pull.is_some() || cli.push.is_some() {
        let host = resolve_host(&cli.host);
        let push_mode = cli.push.is_some();
        if push_mode {
            let record_ids = records
                .iter()
                .map(|record| record.id.clone())
                .collect::<Vec<_>>();
            if !record_ids.is_empty() {
                println!("begin to sync {} records...", record_ids.len());
                let synced =
                    sync::sync_records_to_host(db, &record_ids, &host).unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(1);
                    });
                for record_id in &record_ids {
                    history::write_previous_operation(record_id);
                }
                println!("finished syncing {} records", synced);
            }
            return;
        }
        for record in &records {
            println!("begin to sync {}...", record.id);
            let res = if push_mode {
                sync::sync_record_to_host(db, &record.id, &host)
            } else {
                sync::sync_record_from_host(db, &record.id, &host)
            };
            if let Err(e) = res {
                eprintln!("{e}");
                std::process::exit(1);
            }
            println!("finished syncing");
        }
        return;
    }

    if only_tags {
        let mut set = BTreeSet::new();
        for r in &records {
            for t in &r.tags {
                set.insert(t.clone());
            }
        }
        for t in set {
            print!("{:?}  ", t);
        }
        println!();
        return;
    }

    if to_binary {
        for r in &records {
            if let Some((filename, content)) = split_binary_title(&r.title) {
                write_file_prompt(&filename, &content, cli.force);
            }
        }
        return;
    }

    if !out_name.trim().is_empty() {
        let out = records_to_output_text(&records);
        write_text_output(out_name, &out, cli.force);
        return;
    }

    for record in records {
        ui::print_separator();
        let mut display = record.clone();
        if !cli.verbose {
            display.title = "<hidden>".to_string();
        }

        if cli.verbose {
            let title = if is_probably_text(&record.title) {
                let normalized_title = normalize_title_for_display(&record.title);
                let wrapped_title =
                    wrap_title_for_terminal(&normalized_title, get_terminal_width());
                ui::colorize_title(&print_title_with_colored_separator(&wrapped_title))
            } else {
                format!("\n{}", "<binary>".bright_yellow())
            };
            println!("{}", ui::format_record_like_go(&display, true, Some(title)));
        } else {
            println!("{}", ui::format_record_like_go(&display, false, None));
            if is_probably_text(&record.title) {
                let normalized_title = normalize_title_for_display(&record.title);
                let wrapped_title =
                    wrap_title_for_terminal(&normalized_title, get_terminal_width());
                let title = ui::colorize_title(&print_title_with_colored_separator(&wrapped_title));
                let indented_title = title
                    .lines()
                    .map(|line| {
                        if line.is_empty() {
                            String::new()
                        } else {
                            format!("  {line}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                println!("{indented_title}");
            } else {
                println!("\n  {}", "<binary>".bright_yellow());
            }
        }
        println!("\n{}", ui::colorize_id(&record.id));
        history::write_previous_operation(&record.id);
    }
}
