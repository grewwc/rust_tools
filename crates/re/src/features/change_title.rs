use crate::features::core::*;
use crate::memo::{MemoBackend, history};

pub fn change_title_feature(db: &MemoBackend, cli: &Cli, use_vscode: bool) {
    let id_arg = cli.change_title.as_deref().unwrap_or("");
    let id = resolve_id_or_prev(id_arg, cli.prev);
    if id.trim().is_empty() {
        return;
    }
    if !is_object_id_like(&id) {
        eprintln!("invalid ObjectID: {id}");
        std::process::exit(1);
    }
    let Some(record) = db.get_record(&id).unwrap_or(None) else {
        eprintln!("record not found: {id}");
        std::process::exit(1);
    };

    print!("input the New Title: ");
    rust_tools::commonw::editor::flush_stdout();
    let mut new_title = if cli.e {
        rust_tools::commonw::editor::input_with_editor(&record.title, use_vscode)
            .unwrap_or(record.title.clone())
    } else {
        rust_tools::commonw::prompt::read_line("")
    };
    new_title = new_title.trim().to_string();
    if cli.file_flag.is_some() && !cli.e && !new_title.is_empty() {
        new_title = std::fs::read_to_string(&new_title)
            .unwrap_or_default()
            .trim_end()
            .to_string();
    }
    if new_title == record.title {
        println!("content not changed");
        return;
    }
    let _ = db.update_title(&id, &new_title);
    println!("New Record: ");
    println!("\tTags: {:?}", record.tags);
    println!(
        "\tTitle: {}",
        crate::strw::substring_quiet(&new_title, 0, 200)
    );
    history::write_previous_operation(&id);
}
