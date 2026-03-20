use crate::features::core::*;
use crate::memo::{MemoBackend, history};
use colored::Colorize;

pub fn update_feature(db: &MemoBackend, cli: &Cli, prefix: bool, use_vscode: bool) {
    let from_file = cli.file_flag.is_some();
    let mut id = cli.update.as_deref().unwrap_or("").trim().to_string();

    if cli.prev && id.is_empty() {
        id = history::read_previous_operation();
    }

    if id.is_empty() {
        let positional = cli
            .args
            .iter()
            .filter(|x| x.as_str() != "u")
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect::<Vec<_>>();

        if positional.len() == 1 && is_object_id_like(&positional[0]) {
            id = positional[0].clone();
        } else if !positional.is_empty() {
            let prefix_lookup = prefix || pos_has(&cli.args, "log");
            let records = db
                .list_records_by_tags(&positional, false, prefix_lookup, -1, true, false)
                .unwrap_or_default();
            if records.is_empty() {
                println!(
                    "{}",
                    format!(
                        "no records associated with the tags ({:?}: prefix: {}) found",
                        positional, prefix
                    )
                    .yellow()
                );
                return;
            }
            if let Some(chosen_id) = choose_record_id(&records) {
                id = chosen_id;
            }
        }
    }

    if id.is_empty() {
        id = resolve_id_or_prev("", false);
    }
    if id.is_empty() {
        return;
    }

    update_record_impl(db, &id, from_file, true, use_vscode);
}

pub fn update_record_impl(
    db: &MemoBackend,
    id: &str,
    from_file: bool,
    from_editor: bool,
    use_vscode: bool,
) {
    if !is_object_id_like(id) {
        eprintln!("invalid ObjectID: {id}");
        std::process::exit(1);
    }
    let Some(mut record) = db.get_record(id).unwrap_or(None) else {
        eprintln!("record not found: {id}");
        std::process::exit(1);
    };

    let old_title = record.title.clone();
    let old_tags = record.tags.clone();
    let mut changed = false;

    print!("input the title: ");
    crate::common::editor::flush_stdout();
    if from_editor {
        let new_title = crate::common::editor::input_with_editor(&old_title, use_vscode)
            .unwrap_or(old_title.clone());
        if new_title != old_title {
            record.title = new_title;
            changed = true;
        }
        println!();
    } else {
        let mut title = crate::common::prompt::read_line("").trim().to_string();
        if from_file && !title.is_empty() {
            title = std::fs::read_to_string(&title)
                .unwrap_or_default()
                .trim_end()
                .to_string();
        }
        if !title.is_empty() && title != old_title {
            record.title = title;
            changed = true;
        }
    }

    let tags = crate::common::prompt::read_line("input the tags: ").replace(',', " ");
    let tags = tags.chars().filter(|c| !c.is_control()).collect::<String>();
    let tags = parse_tag_query(&tags);
    if !tags.is_empty() {
        record.tags = tags;
        changed = true;
    }

    if !changed {
        return;
    }

    if record.title != old_title {
        let _ = db.update_title(id, &record.title);
    }
    if record.tags != old_tags {
        let _ = db.remove_tags(id, &old_tags);
        let _ = db.add_tags(id, &record.tags);
    }
    crate::memo::history::write_previous_operation(id);
}
