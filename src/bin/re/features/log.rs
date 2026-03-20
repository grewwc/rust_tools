use crate::features::core::*;
use crate::memo::{MemoBackend, time as memo_time};

pub fn handle_log(db: &MemoBackend, cli: &Cli, use_vscode: bool) {
    let extra = cli
        .args
        .iter()
        .filter(|x| x.as_str() != "log")
        .cloned()
        .collect::<Vec<_>>();
    let mut next_day = 0_i64;
    if extra.len() == 1 {
        next_day = extra[0].parse::<i64>().unwrap_or(0);
    }

    let day = memo_time::add_days(memo_time::today_local_date(), next_day);
    let tag = memo_time::format_log_tag(day);
    let records = db
        .list_records_by_tags(&vec![tag.clone()], false, false, -1, true, true)
        .unwrap_or_default();

    if records.len() > 1 {
        eprintln!("log failed");
        std::process::exit(1);
    }
    if records.is_empty() {
        insert_records(db, true, "", "", &tag, use_vscode);
    } else {
        update_record_impl(db, &records[0].id, false, true, use_vscode);
    }
}

pub fn insert_records(
    db: &MemoBackend,
    from_editor: bool,
    filename: &str,
    tag_flag: &str,
    tag_name: &str,
    use_vscode: bool,
) {
    let mut title_list = Vec::new();
    let tag_name = tag_name.trim();

    if !filename.trim().is_empty() {
        let raw = filename.replace(',', " ");
        let mut parts = crate::strw::split_no_empty(&raw, " ")
            .into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>();
        if parts.len() == 1
            && let Ok(globbed) = glob::glob(&parts[0])
        {
            parts = globbed
                .filter_map(|x| x.ok())
                .map(|x| x.to_string_lossy().to_string())
                .collect();
        }
        for p in parts {
            let base = std::path::Path::new(&p)
                .file_name()
                .and_then(|x| x.to_str())
                .unwrap_or(&p)
                .to_string();
            let content = std::fs::read_to_string(&p).unwrap_or_default();
            title_list.push(format!("{base}\n{content}"));
        }
    } else if from_editor {
        print!("input the title: ");
        crate::common::editor::flush_stdout();
        let title = crate::common::editor::input_with_editor("", use_vscode).unwrap_or_default();
        println!();
        title_list.push(title);
    } else {
        title_list.push(
            crate::common::prompt::read_line("input the title: ")
                .trim()
                .to_string(),
        );
    }

    let tags_str = if !tag_name.is_empty() {
        tag_name.to_string()
    } else if !tag_flag.trim().is_empty() {
        tag_flag.to_string()
    } else {
        crate::common::prompt::read_line("input the tags: ")
    };
    let mut tags = parse_tag_query(&tags_str.replace(',', " "));
    if tags.is_empty() {
        tags.push("auto".to_string());
    }

    for title in title_list {
        let id = db.insert(&title, &tags).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        println!("Inserted: ");
        println!("\tTags: {:?}", tags);
        println!("\tTitle: {}", crate::strw::substring_quiet(&title, 0, 200));
        crate::memo::history::write_previous_operation(&id);
    }
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
