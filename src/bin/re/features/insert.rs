use crate::features::core::*;
use crate::memo::{MemoBackend, history};

pub fn insert_feature(db: &MemoBackend, cli: &Cli, use_vscode: bool) {
    let from_editor = cli.e && (cli.insert || pos_has(&cli.args, "i"));
    let filename = cli.file_flag.as_deref().unwrap_or("");
    insert_records(db, from_editor, filename, &cli.tag, "", use_vscode);
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
        history::write_previous_operation(&id);
    }
}
