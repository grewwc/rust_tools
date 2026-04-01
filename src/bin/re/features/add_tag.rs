use rust_tools::commonw::types::FastSet;
use crate::features::core::*;
use crate::memo::{MemoBackend, history};

pub fn add_or_del_tags_feature(db: &MemoBackend, cli: &Cli, add: bool) {
    let id_arg = if add {
        cli.add_tag.as_deref().unwrap_or("")
    } else {
        cli.del_tag.as_deref().unwrap_or("")
    };
    let id = resolve_id_or_prev(id_arg, cli.prev);
    if id.trim().is_empty() {
        return;
    }
    let Some(record) = db.get_record(&id).unwrap_or(None) else {
        eprintln!("record not found: {id}");
        std::process::exit(1);
    };

    let tags =
        parse_tag_query(&rust_tools::commonw::prompt::read_line("input the Tag: ").replace(',', " "));
    if tags.is_empty() {
        return;
    }

    if add {
        let _ = db.add_tags(&id, &tags);
    } else {
        let mut remain = record.tags.iter().cloned().collect::<FastSet<_>>();
        for t in &tags {
            remain.remove(t);
        }
        if remain.is_empty() {
            eprintln!("You must have at least 1 tag");
            std::process::exit(1);
        }
        let _ = db.remove_tags(&id, &tags);
    }

    if let Some(new_record) = db.get_record(&id).unwrap_or(None) {
        println!("New Record: ");
        println!("\tTags: {:?}", new_record.tags);
        println!(
            "\tTitle: {}",
            crate::strw::substring_quiet(&new_record.title, 0, 200)
        );
    }
    history::write_previous_operation(&id);
}
