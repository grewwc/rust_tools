use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::LazyLock;

use clap::Parser;
use colored::Colorize;
use regex::Regex;
pub use rust_tools::cmd;
pub use rust_tools::common;
pub use rust_tools::strw;

#[path = "re/memo/mod.rs"]
mod memo;

use crate::common::configw;
use crate::common::{editor, prompt};
use crate::memo::{
    MemoBackend, MemoBackendMode, MemoMongo, MemoRecord, MemoTag, history, search as memo_search,
    sync, time as memo_time, ui,
};

const DEFAULT_TXT_OUTPUT_NAME: &str = "output.txt";

static NUMBERED_ITEM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+\.\s").expect("invalid numbered item regex"));
static WRAPPED_NUMBERED_ITEM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<head>.*?)[ ]{2,}(?P<tail>\d+\.\s.*)$")
        .expect("invalid wrapped numbered item regex")
});

#[derive(Parser, Debug)]
#[command(about = "Memo record tool (go_tools re compatible)")]
struct Cli {
    #[arg(
        long,
        default_value = "",
        help = "backend: auto|mongo|sqlite. '.configW:re.backend'"
    )]
    backend: String,

    #[arg(short = 'i', default_value_t = false, help = "insert a record")]
    insert: bool,

    #[arg(
        long = "ct",
        num_args = 0..=1,
        default_missing_value = "",
        help = "change a record title"
    )]
    change_title: Option<String>,

    #[arg(
        short = 'u',
        num_args = 0..=1,
        default_missing_value = "",
        help = "update a record"
    )]
    update: Option<String>,

    #[arg(
        short = 'd',
        num_args = 0..=1,
        default_missing_value = "",
        help = "delete a record"
    )]
    delete: Option<String>,

    #[arg(
        short = 'f',
        num_args = 0..=1,
        default_missing_value = "",
        help = "finish a record"
    )]
    finish: Option<String>,

    #[arg(
        long = "nf",
        num_args = 0..=1,
        default_missing_value = "",
        help = "set a record unfinished"
    )]
    unfinish: Option<String>,

    #[arg(short = 'n', default_value_t = 100, help = "# of records to list")]
    n: i64,

    #[arg(short = 'r', default_value_t = false, help = "reverse sort")]
    reverse: bool,

    #[arg(long = "all", default_value_t = false, help = "including all record")]
    all: bool,

    #[arg(short = 'a', default_value_t = false, help = "shortcut for -all")]
    a: bool,

    #[arg(
        short = 't',
        long = "t",
        num_args = 0..=1,
        default_missing_value = "",
        help = "search by tags"
    )]
    tag_query: Option<String>,

    #[arg(
        long = "ta",
        num_args = 0..=1,
        default_missing_value = "",
        hide = true
    )]
    tag_query_ta: Option<String>,

    #[arg(
        long = "at",
        num_args = 0..=1,
        default_missing_value = "",
        hide = true
    )]
    tag_query_at: Option<String>,

    #[arg(
        long = "and",
        default_value_t = false,
        help = "use and logic to match tags"
    )]
    r#and: bool,

    #[arg(long = "include-finished", default_value_t = false)]
    include_finished: bool,

    #[arg(long = "title", default_value = "", help = "search by title")]
    title: String,

    #[arg(long = "c", default_value = "", help = "content (alias for title)")]
    content: String,

    #[arg(
        long = "search",
        short = 'q',
        default_value = "",
        help = "fuzzy search records"
    )]
    search: String,

    #[arg(long = "out", num_args = 0..=1, default_missing_value = "")]
    out: Option<String>,

    #[arg(long = "remote", default_value_t = false)]
    remote: bool,

    #[arg(long = "prev", default_value_t = false)]
    prev: bool,

    #[arg(long = "count", default_value_t = false)]
    count: bool,

    #[arg(long = "prefix", default_value_t = false)]
    prefix: bool,

    #[arg(long = "pre", default_value_t = false)]
    pre: bool,

    #[arg(long = "binary", default_value_t = false)]
    binary: bool,

    #[arg(short = 'b', default_value_t = false)]
    b: bool,

    #[arg(long = "force", default_value_t = false)]
    force: bool,

    #[arg(long = "sp", default_value_t = false)]
    sp: bool,

    #[arg(
        long = "ex",
        default_value = "",
        help = "exclude tag prefix when list tags"
    )]
    ex: String,

    #[arg(long = "code", default_value_t = false)]
    code: bool,

    #[arg(
        short = 's',
        default_value_t = false,
        help = "short format, only print tags"
    )]
    short: bool,

    #[arg(short = 'l', default_value_t = false, help = "list tags")]
    list_tags_time: bool,

    #[arg(long = "tags", default_value_t = false, help = "list all tags")]
    list_tags: bool,

    #[arg(long = "v", short = 'v', default_value_t = false)]
    verbose: bool,

    #[arg(long = "file", num_args = 0..=1, default_missing_value = "")]
    file_flag: Option<String>,

    #[arg(long = "e", default_value_t = false, help = "read from editor")]
    e: bool,

    #[arg(long = "host", default_value = "")]
    host: String,

    #[arg(long = "push", num_args = 0..=1, default_missing_value = "")]
    push: Option<String>,

    #[arg(long = "pull", num_args = 0..=1, default_missing_value = "")]
    pull: Option<String>,

    #[arg(long = "add-tag", num_args = 0..=1, default_missing_value = "")]
    add_tag: Option<String>,

    #[arg(long = "del-tag", num_args = 0..=1, default_missing_value = "")]
    del_tag: Option<String>,

    #[arg(long = "clean-tag", num_args = 0..=1, default_missing_value = "")]
    clean_tag: Option<String>,

    #[arg(long = "tag", default_value = "")]
    tag: String,

    #[arg(long = "db", default_value = "")]
    db: String,

    #[arg(long = "it", default_value_t = false, hide = true)]
    it: bool,

    #[arg(long = "ti", default_value_t = false, hide = true)]
    ti: bool,

    #[arg(
        value_name = "ARGS",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    args: Vec<String>,
}

#[derive(Clone)]
struct SearchHit {
    score: f64,
    record: MemoRecord,
    preview: Vec<String>,
}

fn main() {
    let argv = normalize_legacy_single_dash_long_args(std::env::args());
    let mut cli = Cli::parse_from(argv);
    maybe_parse_positional_limit(&mut cli);

    let db = open_backend(&cli).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let to_binary = cli.binary || cli.b;
    let all = (cli.all
        || (cli.a && cli.add_tag.is_none() && cli.del_tag.is_none() && !cli.list_tags))
        && !to_binary;
    let prefix = cli.prefix || cli.pre || all || cli.a;
    let only_tags = cli.short || (cli.a && cli.short);
    let use_vscode = cli.code;
    let list_special = cli.sp || all;
    let reverse = cli.reverse && !(cli.prev || cli.remote || cli.prefix || cli.pre);
    let include_finished = cli.include_finished || all;
    let record_limit = if all { i64::MAX } else { cli.n };

    let out_name = cli
        .out
        .as_deref()
        .map(|s| {
            if s.trim().is_empty() {
                DEFAULT_TXT_OUTPUT_NAME
            } else {
                s.trim()
            }
        })
        .unwrap_or("");

    let tag_query_raw = get_tag_query_raw(&cli);
    let tag_query_non_empty = tag_query_raw
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty());
    let list_tags_and_order_by_time = order_by_time(&cli, tag_query_raw.as_deref());

    if cli.unfinish.is_some() || pos_has(&cli.args, "nf") {
        handle_finish_command(&db, &cli, cli.unfinish.as_deref(), false, prefix);
        return;
    }

    if cli.finish.is_some() {
        handle_finish_command(&db, &cli, cli.finish.as_deref(), true, prefix);
        return;
    }

    if pos_has(&cli.args, "open") || pos_has(&cli.args, "o") {
        open_urls(&db, &cli.args, prefix);
        return;
    }

    if cli.clean_tag.is_some() {
        clean_by_tags(&db, cli.clean_tag.as_deref().unwrap_or(""));
        return;
    }

    if should_log_command(&cli) {
        handle_log(&db, &cli, use_vscode);
        return;
    }

    if pos_has(&cli.args, "week") {
        handle_week(&db);
        return;
    }

    if should_list_tags_feature(&cli, list_tags_and_order_by_time) {
        list_tags_feature(
            &db,
            &cli,
            record_limit,
            reverse,
            all,
            list_special,
            list_tags_and_order_by_time,
        );
        return;
    }

    if should_update_feature(&cli) {
        update_feature(&db, &cli, prefix, use_vscode);
        return;
    }

    if should_insert_feature(&cli) {
        insert_feature(&db, &cli, use_vscode);
        return;
    }

    if cli.change_title.is_some() {
        change_title_feature(&db, &cli, use_vscode);
        return;
    }

    if cli.delete.is_some() || pos_has(&cli.args, "d") {
        delete_feature(&db, &cli);
        return;
    }

    if cli.add_tag.is_some() {
        add_or_del_tags_feature(&db, &cli, true);
        return;
    }

    if cli.del_tag.is_some() {
        add_or_del_tags_feature(&db, &cli, false);
        return;
    }

    if let Some(push_ref) = cli.push.as_deref()
        && !push_ref.trim().is_empty()
    {
        push_single(&db, push_ref, &cli.host);
        return;
    }

    let title_query = first_non_empty(&[cli.title.as_str(), cli.content.as_str()]);
    if !title_query.trim().is_empty() {
        list_by_title_feature(
            &db,
            title_query,
            tag_query_raw.as_deref().unwrap_or(""),
            cli.r#and,
            prefix,
            record_limit,
            reverse,
            include_finished,
            out_name,
            to_binary,
            cli.count,
            cli.verbose,
            cli.force,
        );
        return;
    }

    if !cli.search.trim().is_empty() {
        search_feature(
            &db,
            &cli.search,
            tag_query_raw.as_deref().unwrap_or(""),
            cli.r#and,
            prefix,
            record_limit,
            include_finished,
            list_special,
            out_name,
            to_binary,
            cli.count,
            cli.verbose,
            cli.force,
        );
        return;
    }

    if should_list_by_tag_name(
        tag_query_non_empty,
        list_tags_and_order_by_time,
        title_query,
        &cli.search,
    ) {
        list_by_tag_name_feature(
            &db,
            tag_query_raw.as_deref().unwrap_or(""),
            &cli,
            prefix,
            record_limit,
            reverse,
            include_finished,
            list_special,
            only_tags,
            to_binary,
            out_name,
        );
        return;
    }

    if let Some(pull_ref) = cli.pull.as_deref() {
        pull_single(&db, pull_ref, &cli.host);
        return;
    }

    if cli.push.is_some() {
        push_single(&db, "", &cli.host);
        return;
    }

    default_print(&db, record_limit, list_special);
}

fn normalize_legacy_single_dash_long_args(args: impl IntoIterator<Item = String>) -> Vec<String> {
    const LEGACY_LONG_FLAGS: &[&str] = &[
        "backend",
        "ct",
        "all",
        "nf",
        "t",
        "ta",
        "at",
        "and",
        "include-finished",
        "title",
        "c",
        "search",
        "out",
        "remote",
        "prev",
        "count",
        "prefix",
        "pre",
        "binary",
        "force",
        "sp",
        "ex",
        "code",
        "tags",
        "v",
        "file",
        "e",
        "host",
        "push",
        "pull",
        "add-tag",
        "del-tag",
        "clean-tag",
        "tag",
        "db",
        "it",
        "ti",
    ];

    let mut out = Vec::new();
    for (idx, arg) in args.into_iter().enumerate() {
        if idx == 0 {
            out.push(arg);
            continue;
        }

        let Some(stripped) = arg.strip_prefix('-') else {
            out.push(arg);
            continue;
        };

        if arg.starts_with("--") || stripped.is_empty() {
            out.push(arg);
            continue;
        }

        if stripped.chars().all(|c| c.is_ascii_digit()) {
            out.push(arg);
            continue;
        }

        let (name, value_suffix) = match stripped.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (stripped, None),
        };

        if LEGACY_LONG_FLAGS.contains(&name) {
            if let Some(v) = value_suffix {
                out.push(format!("--{name}={v}"));
            } else {
                out.push(format!("--{name}"));
            }
        } else {
            out.push(arg);
        }
    }

    out
}

fn maybe_parse_positional_limit(cli: &mut Cli) {
    if cli.args.len() != 1 || !cli.args[0].chars().all(|c| c.is_ascii_digit()) {
        return;
    }
    if cli.insert
        || cli.change_title.is_some()
        || cli.update.is_some()
        || cli.delete.is_some()
        || cli.finish.is_some()
        || cli.unfinish.is_some()
        || cli.add_tag.is_some()
        || cli.del_tag.is_some()
        || cli.clean_tag.is_some()
        || cli.tag_query.is_some()
        || !cli.title.trim().is_empty()
        || !cli.content.trim().is_empty()
        || !cli.search.trim().is_empty()
        || cli.push.is_some()
        || cli.pull.is_some()
    {
        return;
    }
    if let Ok(n) = cli.args[0].parse::<i64>() {
        cli.n = n;
        cli.args.clear();
    }
}

fn pos_has(args: &[String], needle: &str) -> bool {
    args.iter().any(|x| x == needle)
}

fn get_tag_query_raw(cli: &Cli) -> Option<String> {
    if let Some(v) = cli.tag_query.as_ref() {
        return Some(v.clone());
    }
    if let Some(v) = cli.tag_query_ta.as_ref() {
        return Some(v.clone());
    }
    if let Some(v) = cli.tag_query_at.as_ref() {
        return Some(v.clone());
    }
    None
}

fn should_update_feature(cli: &Cli) -> bool {
    cli.update.is_some() || pos_has(&cli.args, "u")
}

fn should_insert_feature(cli: &Cli) -> bool {
    cli.insert || pos_has(&cli.args, "i")
}

fn should_log_command(cli: &Cli) -> bool {
    pos_has(&cli.args, "log") && !pos_has(&cli.args, "u") && cli.update.is_none()
}

fn should_list_tags_feature(cli: &Cli, list_tags_and_order_by_time: bool) -> bool {
    list_tags_and_order_by_time
        || cli.list_tags
        || pos_has(&cli.args, "tags")
        || pos_has(&cli.args, "t")
        || pos_has(&cli.args, "i")
}

fn should_list_by_tag_name(
    tag_query_non_empty: bool,
    list_tags_and_order_by_time: bool,
    title_query: &str,
    search_query: &str,
) -> bool {
    tag_query_non_empty
        && !list_tags_and_order_by_time
        && title_query.trim().is_empty()
        && search_query.trim().is_empty()
}

fn order_by_time(cli: &Cli, tag_query_raw: Option<&str>) -> bool {
    if matches!(tag_query_raw, Some(v) if v.trim().is_empty()) {
        return true;
    }
    if cli.it || cli.ti {
        return true;
    }
    if pos_has(&cli.args, "it") || pos_has(&cli.args, "ti") {
        return true;
    }
    cli.list_tags_time
}

fn open_backend(cli: &Cli) -> Result<MemoBackend, String> {
    if cli.remote {
        let host = resolve_host(&cli.host);
        return MemoMongo::connect_remote_host(&host).map(MemoBackend::Mongo);
    }
    let mode = if cli.backend.trim().is_empty() {
        MemoBackendMode::parse(&configw::get_config("re.backend", "auto"))
    } else {
        MemoBackendMode::parse(&cli.backend)
    };
    let sqlite_override = if cli.db.trim().is_empty() {
        None
    } else {
        Some(cli.db.trim())
    };
    MemoBackend::open(mode, sqlite_override)
}

fn resolve_host(cli_host: &str) -> String {
    let host = if cli_host.trim().is_empty() {
        configw::get_config("re.remote.host", "")
    } else {
        cli_host.trim().to_string()
    };
    if host.trim().is_empty() {
        eprintln!("--host is required (or set .configW: re.remote.host=...)");
        std::process::exit(1);
    }
    host
}

fn first_non_empty<'a>(items: &[&'a str]) -> &'a str {
    for s in items {
        if !s.trim().is_empty() {
            return s;
        }
    }
    ""
}

fn parse_tag_query(s: &str) -> Vec<String> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

fn is_object_id_like(s: &str) -> bool {
    let s = s.trim();
    s.len() == 24 && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn resolve_id_or_prev(id: &str, prev: bool) -> String {
    if prev {
        let v = history::read_previous_operation();
        if !v.trim().is_empty() {
            return v;
        }
    }
    if !id.trim().is_empty() {
        return id.trim().to_string();
    }
    prompt::read_line("input the Object ID: ")
        .trim()
        .to_string()
}

fn toggle_finish(db: &MemoBackend, id: &str, finished: bool) {
    let id = id.trim();
    if id.is_empty() {
        return;
    }
    let _ = db.set_finished(id, finished);
    history::write_previous_operation(id);
}

fn collect_refs(value: Option<&str>, args: &[String], skip_tokens: &[&str]) -> Vec<String> {
    let mut refs = Vec::new();
    if let Some(v) = value
        && !v.trim().is_empty()
    {
        refs.push(v.trim().to_string());
    }
    for arg in args {
        let a = arg.trim();
        if a.is_empty() {
            continue;
        }
        if skip_tokens.iter().any(|x| *x == a) {
            continue;
        }
        refs.push(a.to_string());
    }
    refs
}

fn primary_title(title: &str) -> String {
    let s = title.replace('\r', "");
    for line in s.lines() {
        let line = line.trim();
        if !line.is_empty() {
            return line.to_string();
        }
    }
    s.trim().to_string()
}

fn choose_record_id(records: &[MemoRecord]) -> Option<String> {
    if records.is_empty() {
        return None;
    }
    if records.len() == 1 {
        return Some(records[0].id.clone());
    }
    let items = records
        .iter()
        .map(|r| format!("{}\t{}", r.id, primary_title(&r.title)))
        .collect::<Vec<_>>();
    let selected = history::choose_from_list(&items)?;
    Some(selected.split_whitespace().next().unwrap_or("").to_string())
}

fn resolve_record_ref_local(
    db: &MemoBackend,
    record_ref: &str,
    include_finished: bool,
    prefix: bool,
) -> Option<String> {
    let mut reference = record_ref.trim().to_string();
    if reference.is_empty() {
        reference = prompt::read_line("Input the ObjectID/title/tag: ")
            .trim()
            .to_string();
    }
    if reference.is_empty() {
        return None;
    }
    if is_object_id_like(&reference) {
        return Some(reference);
    }

    let records = db
        .list_records(-1, false, include_finished)
        .unwrap_or_default();

    let mut exact = Vec::new();
    let mut seen = HashSet::new();
    for r in &records {
        if (primary_title(&r.title).eq_ignore_ascii_case(&reference)
            || r.title.trim().eq_ignore_ascii_case(&reference))
            && seen.insert(r.id.clone())
        {
            exact.push(r.clone());
        }
    }
    for r in &records {
        if r.tags.iter().any(|t| t == &reference) && seen.insert(r.id.clone()) {
            exact.push(r.clone());
        }
    }
    if !exact.is_empty() {
        return choose_record_id(&exact);
    }

    let lower = reference.to_lowercase();
    let mut fuzzy = Vec::new();
    seen.clear();
    for r in &records {
        let title_hit = r.title.to_lowercase().contains(&lower);
        let tag_hit = if prefix {
            r.tags.iter().any(|t| t.starts_with(&reference))
        } else {
            r.tags.iter().any(|t| t.contains(&reference))
        };
        if (title_hit || tag_hit) && seen.insert(r.id.clone()) {
            fuzzy.push(r.clone());
        }
    }
    if fuzzy.is_empty() {
        return None;
    }
    choose_record_id(&fuzzy)
}

fn handle_finish_command(
    db: &MemoBackend,
    cli: &Cli,
    value: Option<&str>,
    finished: bool,
    prefix: bool,
) {
    let refs = collect_refs(value, &cli.args, &["f", "nf"]);
    if refs.is_empty() {
        let id = resolve_id_or_prev("", cli.prev);
        toggle_finish(db, &id, finished);
        return;
    }

    if finished && prefix {
        for r in refs {
            let tags = parse_tag_query(&r);
            if tags.is_empty() {
                continue;
            }
            let records = db
                .list_records_by_tags(&tags, false, true, -1, false, true)
                .unwrap_or_default();
            for rec in records {
                let _ = db.set_finished(&rec.id, true);
                history::write_previous_operation(&rec.id);
            }
        }
        return;
    }

    for r in refs {
        let id = if cli.prev {
            history::read_previous_operation()
        } else if is_object_id_like(&r) {
            r
        } else {
            resolve_record_ref_local(db, &r, !finished, false).unwrap_or_default()
        };
        if id.trim().is_empty() {
            continue;
        }
        toggle_finish(db, &id, finished);
    }
}

fn update_feature(db: &MemoBackend, cli: &Cli, prefix: bool, use_vscode: bool) {
    let from_file = cli.file_flag.is_some();
    let mut from_editor = cli.e;
    let mut id = cli.update.as_deref().unwrap_or("").trim().to_string();

    if cli.prev && id.is_empty() {
        id = history::read_previous_operation();
        if !from_file {
            from_editor = true;
        }
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
                from_editor = true;
            }
        }
    }

    if id.is_empty() {
        id = resolve_id_or_prev("", false);
    }
    if id.is_empty() {
        return;
    }

    update_record_impl(db, &id, from_file, from_editor, use_vscode);
}

fn update_record_impl(
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
    editor::flush_stdout();
    if from_editor {
        let new_title =
            editor::input_with_editor(&old_title, use_vscode).unwrap_or(old_title.clone());
        if new_title != old_title {
            record.title = new_title;
            changed = true;
        }
        println!();
    } else {
        let mut title = prompt::read_line("").trim().to_string();
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

    let tags = prompt::read_line("input the tags: ").replace(',', " ");
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
    history::write_previous_operation(id);
}

fn insert_feature(db: &MemoBackend, cli: &Cli, use_vscode: bool) {
    let from_editor = cli.e && (cli.insert || pos_has(&cli.args, "i"));
    let filename = cli.file_flag.as_deref().unwrap_or("");
    insert_records(db, from_editor, filename, &cli.tag, "", use_vscode);
}

fn insert_records(
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
        let mut parts = strw::split_no_empty(&raw, " ")
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
        editor::flush_stdout();
        let title = editor::input_with_editor("", use_vscode).unwrap_or_default();
        println!();
        title_list.push(title);
    } else {
        title_list.push(prompt::read_line("input the title: ").trim().to_string());
    }

    let tags_str = if !tag_name.is_empty() {
        tag_name.to_string()
    } else if !tag_flag.trim().is_empty() {
        tag_flag.to_string()
    } else {
        prompt::read_line("input the tags: ")
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
        println!("\tTitle: {}", strw::substring_quiet(&title, 0, 200));
        history::write_previous_operation(&id);
    }
}

fn change_title_feature(db: &MemoBackend, cli: &Cli, use_vscode: bool) {
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
    editor::flush_stdout();
    let mut new_title = if cli.e {
        editor::input_with_editor(&record.title, use_vscode).unwrap_or(record.title.clone())
    } else {
        prompt::read_line("")
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
    println!("\tTitle: {}", strw::substring_quiet(&new_title, 0, 200));
    history::write_previous_operation(&id);
}

fn delete_feature(db: &MemoBackend, cli: &Cli) {
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

fn delete_by_tag_prefix(db: &MemoBackend, tag_ref: &str) -> bool {
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

fn add_or_del_tags_feature(db: &MemoBackend, cli: &Cli, add: bool) {
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

    let tags = parse_tag_query(&prompt::read_line("input the Tag: ").replace(',', " "));
    if tags.is_empty() {
        return;
    }

    if add {
        let _ = db.add_tags(&id, &tags);
    } else {
        let mut remain = record.tags.iter().cloned().collect::<HashSet<_>>();
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
            strw::substring_quiet(&new_record.title, 0, 200)
        );
    }
    history::write_previous_operation(&id);
}

fn push_single(db: &MemoBackend, push_ref: &str, cli_host: &str) {
    println!("pushing...");
    let host = resolve_host(cli_host);

    let mut reference = push_ref.trim().to_string();
    if reference.is_empty() {
        reference = prompt::read_line("Input the ObjectID/title/tag: ")
            .trim()
            .to_string();
    }

    // Compatibility with go_tools behavior: if input looks like tag query,
    // push all matched records instead of resolving a single one.
    if !is_object_id_like(&reference) {
        let tags = parse_tag_query(&reference);
        if !tags.is_empty() {
            let records = db
                .list_records_by_tags(&tags, false, false, -1, true, true)
                .unwrap_or_default();
            if !records.is_empty() {
                for record in &records {
                    println!("begin to sync {}...", record.id);
                    sync::sync_record_to_host(db, &record.id, &host).unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(1);
                    });
                    history::write_previous_operation(&record.id);
                    println!("finished syncing");
                }
                println!("finished push {} records", records.len());
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
        strw::substring_quiet(&title_preview, 0, 20).green()
    );
}

fn pull_single(db: &MemoBackend, pull_ref: &str, cli_host: &str) {
    println!("pulling...");
    let host = resolve_host(cli_host);

    let mut reference = pull_ref.trim().to_string();
    if reference.is_empty() {
        reference = prompt::read_line("Input the ObjectID/title/tag: ")
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
        strw::substring_quiet(&label, 0, 20).green()
    );
}

fn handle_log(db: &MemoBackend, cli: &Cli, use_vscode: bool) {
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

fn handle_week(db: &MemoBackend) {
    let mut first = memo_time::first_day_of_week(memo_time::today_local_date());
    let now = memo_time::today_local_date();
    let week_tag = memo_time::format_week_tag(first);

    let records = db
        .list_records_by_tags(&vec![week_tag.clone()], false, false, -1, true, true)
        .unwrap_or_default();
    if records.len() > 1 {
        eprintln!("too many week tags");
        std::process::exit(1);
    }

    let mut title = String::new();
    while first < now {
        let day_tag = memo_time::format_log_tag(first);
        let day_records = db
            .list_records_by_tags(&vec![day_tag], false, false, -1, true, true)
            .unwrap_or_default();
        if day_records.len() > 1 {
            eprintln!("log failed");
            std::process::exit(1);
        }
        if let Some(r) = day_records.first() {
            title.push_str(&format!("-- {} --\n", first.format("%Y-%m-%d")));
            title.push_str(&r.title);
            title.push_str("\n\n");
        }
        first = memo_time::add_days(first, 1);
    }

    if let Some(existing) = records.first() {
        let _ = db.update_title(&existing.id, &title);
        history::write_previous_operation(&existing.id);
    } else {
        let id = db.insert(&title, &vec![week_tag]).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        history::write_previous_operation(&id);
    }
}

fn load_special_patterns() -> Vec<String> {
    let mut patterns = vec!["learn".to_string()];
    patterns.extend(parse_tag_query(&configw::get_config("special.tags", "")));
    patterns.retain(|x| !x.trim().is_empty());
    patterns.sort();
    patterns.dedup();
    patterns
}

fn is_special_record(record: &MemoRecord, patterns: &[String]) -> bool {
    record
        .tags
        .iter()
        .any(|t| patterns.iter().any(|p| t.starts_with(p)))
}

fn compute_tags_from_records(
    db: &MemoBackend,
    include_finished: bool,
    list_special: bool,
) -> Vec<MemoTag> {
    let mut records = db
        .list_records(-1, false, include_finished)
        .unwrap_or_default();
    if !list_special {
        let patterns = load_special_patterns();
        records.retain(|r| !is_special_record(r, &patterns));
    }

    let mut map: HashMap<String, (i64, i64)> = HashMap::new();
    for record in records {
        let modified = record.modified_date;
        for tag in record.tags {
            let e = map.entry(tag).or_insert((0, 0));
            e.0 += 1;
            if modified > e.1 {
                e.1 = modified;
            }
        }
    }

    map.into_iter()
        .map(|(name, (count, modified_date))| MemoTag {
            id: String::new(),
            name,
            count,
            modified_date,
        })
        .collect()
}

fn filter_tags_ex(tags: Vec<MemoTag>, ex: &str) -> Vec<MemoTag> {
    let ex_prefix = parse_tag_query(ex);
    if ex_prefix.is_empty() {
        return tags;
    }
    tags.into_iter()
        .filter(|t| !ex_prefix.iter().any(|p| t.name.starts_with(p)))
        .collect()
}

fn list_tags_feature(
    db: &MemoBackend,
    cli: &Cli,
    record_limit: i64,
    reverse: bool,
    all: bool,
    list_special: bool,
    list_tags_and_order_by_time: bool,
) {
    let mut tags = if all || list_tags_and_order_by_time {
        let include_finished = !list_tags_and_order_by_time || all;
        let mut tags = compute_tags_from_records(db, include_finished, list_special);
        if list_tags_and_order_by_time {
            tags.sort_by(|a, b| a.modified_date.cmp(&b.modified_date));
        } else {
            tags.sort_by(|a, b| {
                b.count
                    .cmp(&a.count)
                    .then_with(|| b.modified_date.cmp(&a.modified_date))
            });
        }
        if reverse {
            tags.reverse();
        }
        tags
    } else {
        let mut tags = db.list_tags(None, None, record_limit).unwrap_or_default();
        if reverse {
            tags.reverse();
        }
        tags
    };

    if record_limit > 0 && tags.len() > record_limit as usize {
        tags.truncate(record_limit as usize);
    }
    tags = filter_tags_ex(tags, &cli.ex);

    if cli.verbose {
        for mut tag in tags {
            tag.name = tag.name.bright_green().to_string();
            ui::print_separator();
            println!("ID: {}", tag.id);
            println!("Name: {}", tag.name);
            println!("Count: {}", tag.count);
            println!("ModifiedDate: {}", tag.modified_date);
        }
        return;
    }

    let terminal_width = get_terminal_width();
    let terminal_indent = 2usize;
    let delimiter = "   ";
    let mut buf = String::new();
    for tag in &tags {
        buf.push_str(&format!("{}[{}]  ", tag.name, tag.count));
    }
    let raw = strw::wrap(
        &buf,
        terminal_width.saturating_sub(terminal_indent),
        terminal_indent,
        delimiter,
    );

    for line in strw::split_no_empty(&raw, "\n") {
        let arr = strw::split_no_empty(line, " ");
        let mut changed = Vec::with_capacity(arr.len());
        for x in arr {
            if let Some(idx) = x.find('[') {
                let name = &x[..idx];
                let rest = &x[idx..];
                changed.push(format!("{}{}", name.bright_green(), rest));
            } else {
                changed.push(x.to_string());
            }
        }
        println!("{}{}", " ".repeat(terminal_indent), changed.join(delimiter));
    }
}

fn get_terminal_width() -> usize {
    if let Some(cols) = std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        && cols > 0
    {
        return cols;
    }

    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = std::io::stdout().as_raw_fd();
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) };
        if rc == 0 && ws.ws_col > 0 {
            return ws.ws_col as usize;
        }
    }

    80
}

fn list_by_tag_name_feature(
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
        db.list_records_by_tags(&tags, cli.r#and, prefix, limit, reverse, include_finished)
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
        println!("{}", ui::format_record_like_go(&display, cli.verbose, None));

        if is_probably_text(&record.title) {
            let normalized_title = normalize_title_for_display(&record.title);
            let wrapped_title = wrap_title_for_terminal(&normalized_title, get_terminal_width());
            println!(
                "{}",
                ui::colorize_title(&print_title_with_colored_separator(&wrapped_title))
            );
        } else {
            println!("\n{}", "<binary>".bright_yellow());
        }
        println!("{}", ui::colorize_id(&record.id));
        history::write_previous_operation(&record.id);
    }
}

fn list_by_title_feature(
    db: &MemoBackend,
    query: &str,
    tag_query: &str,
    use_and: bool,
    prefix: bool,
    limit: i64,
    reverse: bool,
    include_finished: bool,
    out_name: &str,
    to_binary: bool,
    count_only: bool,
    verbose: bool,
    force: bool,
) {
    let tags = parse_tag_query(tag_query);

    let mut records = if tags.is_empty() {
        db.search_title(query, limit, include_finished)
            .unwrap_or_default()
    } else {
        let mut records = db
            .list_records_by_tags(&tags, use_and, prefix, -1, reverse, include_finished)
            .unwrap_or_default();
        let lower = query.to_lowercase();
        records.retain(|r| r.title.to_lowercase().contains(&lower));
        records.sort_by(|a, b| a.modified_date.cmp(&b.modified_date));
        if !reverse {
            records.reverse();
        }
        if limit > 0 && records.len() > limit as usize {
            records.truncate(limit as usize);
        }
        records
    };

    records.sort_by(|a, b| a.modified_date.cmp(&b.modified_date));
    if !reverse {
        records.reverse();
    }
    if limit > 0 && records.len() > limit as usize {
        records.truncate(limit as usize);
    }

    if count_only {
        println!("{} records found", records.len());
        return;
    }

    if to_binary {
        eprintln!("not supported");
        std::process::exit(1);
    }

    if !out_name.trim().is_empty() {
        let out = records_to_output_text(&records);
        write_text_output(out_name, &out, force);
        return;
    }

    let pattern = Regex::new(&format!("(?i){}", regex::escape(query))).ok();
    for record in records {
        ui::print_separator();
        let mut display = record.clone();
        let mut highlighted_title = None;

        if verbose {
            if is_probably_text(&record.title) {
                highlighted_title = Some(format!(
                    "\n{}",
                    highlight_search_text(&record.title, pattern.as_ref())
                ));
            } else {
                highlighted_title = Some(format!("\n{}", "<binary>".bright_yellow()));
            }
        } else {
            display.title = "<hidden>".to_string();
        }

        println!(
            "{}",
            ui::format_record_like_go(&display, verbose, highlighted_title)
        );
        println!("{}", ui::colorize_id(&record.id));
        history::write_previous_operation(&record.id);
    }
}

fn search_feature(
    db: &MemoBackend,
    query: &str,
    tag_query: &str,
    use_and: bool,
    prefix: bool,
    limit: i64,
    include_finished: bool,
    list_special: bool,
    out_name: &str,
    to_binary: bool,
    count_only: bool,
    verbose: bool,
    force: bool,
) {
    let tags = parse_tag_query(tag_query);
    let mut records = if tags.is_empty() {
        db.list_records(-1, false, include_finished)
            .unwrap_or_default()
    } else {
        db.list_records_by_tags(&tags, use_and, prefix, -1, false, include_finished)
            .unwrap_or_default()
    };

    if !list_special {
        let patterns = load_special_patterns();
        records.retain(|r| !is_special_record(r, &patterns));
    }

    let mut results = Vec::new();
    for record in records {
        let score = memo_search::score_record(&record, query);
        if score <= 0.0 {
            continue;
        }
        let preview = memo_search::build_preview_lines(&record, query, 3);
        results.push(SearchHit {
            score,
            record,
            preview,
        });
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.record.modified_date.cmp(&a.record.modified_date))
    });
    if limit > 0 && results.len() > limit as usize {
        results.truncate(limit as usize);
    }

    if count_only {
        println!("{} records found", results.len());
        return;
    }

    if to_binary {
        eprintln!("not supported");
        std::process::exit(1);
    }

    if !out_name.trim().is_empty() {
        let mut out = String::new();
        for idx in (0..results.len()).rev() {
            let item = &results[idx];
            out.push_str(&format!(
                "{} score={:.3} {}\n",
                "-".repeat(10),
                item.score,
                "-".repeat(10)
            ));
            out.push_str(&item.record.title);
            out.push('\n');
        }
        write_text_output(out_name, &out, force);
        return;
    }

    let pattern = memo_search::highlight_pattern(query);
    for idx in (0..results.len()).rev() {
        if idx < results.len().saturating_sub(1) {
            println!();
        }
        print_search_result(idx, &results[idx], query, pattern.as_ref(), verbose);
        history::write_previous_operation(&results[idx].record.id);
    }
}

fn print_search_result(
    index: usize,
    result: &SearchHit,
    query: &str,
    pattern: Option<&Regex>,
    verbose: bool,
) {
    let mut header = vec![
        format!("[{}]", index + 1).bright_blue().to_string(),
        format!("score {:.3}", result.score)
            .bright_white()
            .to_string(),
    ];
    if result.record.finished {
        header.push("finished".yellow().to_string());
    }
    println!("{}", header.join("  "));

    if !result.record.tags.is_empty() {
        println!(
            "    {} {}",
            "tags:".bright_black(),
            ui::colorize_tags(&result.record.tags).join(", ")
        );
    }

    if verbose {
        let mut display = result.record.clone();
        if is_probably_text(&display.title) {
            display.title = format!("\n{}", highlight_search_text(&display.title, pattern));
        } else {
            display.title = format!("\n{}", "<binary>".bright_yellow());
        }
        let formatted = ui::format_record_like_go(&display, true, None);
        for line in formatted.lines() {
            if line.trim().is_empty() {
                continue;
            }
            println!("    {}", line);
        }
        println!(
            "    {} {}",
            "id:".bright_black(),
            ui::colorize_id(&result.record.id)
        );
        return;
    }

    if is_probably_text(&result.record.title) {
        let preview = if result.preview.is_empty() {
            memo_search::build_preview_lines(&result.record, query, 3)
        } else {
            result.preview.clone()
        };
        for line in preview {
            print_search_preview_line(&line, pattern);
        }
    } else {
        println!("    {} {}", "-".cyan(), "<binary>".bright_yellow());
    }
    println!(
        "    {} {}",
        "id:".bright_black(),
        ui::colorize_id(&result.record.id)
    );
}

fn print_search_preview_line(line: &str, pattern: Option<&Regex>) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    let (prefix, url, has_url) = split_search_preview_url(line);
    if has_url && !prefix.is_empty() {
        println!(
            "    {} {}",
            "-".cyan(),
            highlight_search_text(&prefix, pattern)
        );
        println!("      {}", highlight_search_text(&url, pattern));
        return;
    }
    println!(
        "    {} {}",
        "-".cyan(),
        highlight_search_text(line, pattern)
    );
}

fn split_search_preview_url(line: &str) -> (String, String, bool) {
    let lower = line.to_lowercase();
    let mut start = lower.find("https://");
    if start.is_none() {
        start = lower.find("http://");
    }
    let Some(idx) = start else {
        return (String::new(), String::new(), false);
    };
    let prefix = line[..idx].trim().to_string();
    let url = line[idx..].trim().to_string();
    let has_url = !url.is_empty();
    (prefix, url, has_url)
}

fn highlight_search_text(text: &str, pattern: Option<&Regex>) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }
    let Some(pattern) = pattern else {
        return text.bright_white().to_string();
    };
    let indices = pattern.find_iter(text).collect::<Vec<_>>();
    if indices.is_empty() {
        return text.bright_white().to_string();
    }

    let mut out = String::new();
    let mut last = 0;
    for m in indices {
        if m.start() > last {
            out.push_str(&text[last..m.start()].bright_white().to_string());
        }
        out.push_str(&text[m.start()..m.end()].red().to_string());
        last = m.end();
    }
    if last < text.len() {
        out.push_str(&text[last..].bright_white().to_string());
    }
    out
}

fn records_to_output_text(records: &[MemoRecord]) -> String {
    let mut out = String::new();
    for r in records {
        out.push_str(&format!(
            "{} {:?} {}\n",
            "-".repeat(10),
            r.tags,
            "-".repeat(10)
        ));
        out.push_str(&r.title);
        out.push('\n');
    }
    out
}

fn write_text_output(path: &str, content: &str, force: bool) {
    let path = path.trim();
    if path.is_empty() {
        return;
    }
    if std::path::Path::new(path).exists() && !force {
        if !prompt::prompt_yes_or_no(&format!("{path} exists, overwrite? (y/n): ")) {
            return;
        }
    }
    std::fs::write(path, content).unwrap_or_else(|e| {
        eprintln!("failed to write {path}: {e}");
        std::process::exit(1);
    });
    println!("write to {path}");
}

fn split_binary_title(title: &str) -> Option<(String, String)> {
    let s = title.replace('\r', "");
    let mut lines = s.lines();
    let filename = lines.next()?.trim().to_string();
    if filename.is_empty() {
        return None;
    }
    let content = lines.collect::<Vec<_>>().join("\n");
    Some((filename, content))
}

fn write_file_prompt(path: &str, content: &str, force: bool) {
    let path = path.trim();
    if path.is_empty() {
        return;
    }
    if std::path::Path::new(path).exists() && !force {
        if !prompt::prompt_yes_or_no(&format!("{path} exists, overwrite? (y/n): ")) {
            return;
        }
    }
    std::fs::write(path, content).unwrap_or_else(|e| {
        eprintln!("failed to write {path}: {e}");
        std::process::exit(1);
    });
}

fn print_title_with_colored_separator(title: &str) -> String {
    if !title.contains("<sep>") {
        return title.to_string();
    }
    let sep = "~~~~~~~~~~~~".repeat(10).cyan().to_string();
    title.replace("<sep>", &sep)
}

fn normalize_title_for_display(title: &str) -> String {
    let normalized_newline = title.replace("\r\n", "\n").replace('\r', "\n");
    let has_url = normalized_newline.contains("https://") || normalized_newline.contains("http://");
    if !has_url {
        return normalized_newline;
    }

    let mut split_lines = Vec::new();
    for line in normalized_newline.lines() {
        split_wrapped_numbered_line(line, &mut split_lines);
    }

    let mut merged: Vec<String> = Vec::new();
    for line in split_lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let is_numbered_item = NUMBERED_ITEM_RE.is_match(line);
        if !is_numbered_item
            && let Some(prev) = merged.last_mut()
            && (prev.contains("https://") || prev.contains("http://"))
        {
            prev.push_str(line);
            continue;
        }
        merged.push(line.to_string());
    }

    if merged.is_empty() {
        normalized_newline
    } else {
        merged.join("\n")
    }
}

fn split_wrapped_numbered_line(line: &str, out: &mut Vec<String>) {
    let mut rest = line.trim_end().to_string();
    loop {
        let Some(caps) = WRAPPED_NUMBERED_ITEM_RE.captures(&rest) else {
            break;
        };
        let head = caps
            .name("head")
            .map(|m| m.as_str().trim())
            .unwrap_or_default();
        if !head.is_empty() {
            out.push(head.to_string());
        }
        rest = caps
            .name("tail")
            .map(|m| m.as_str().trim_start().to_string())
            .unwrap_or_default();
    }

    let trimmed = rest.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
}

fn wrap_title_for_terminal(title: &str, terminal_width: usize) -> String {
    let max_width = terminal_width.max(24).saturating_sub(2);
    let mut out = Vec::new();

    for raw_line in title.lines() {
        let line = raw_line.trim_end();
        if line.is_empty() {
            out.push(String::new());
            continue;
        }

        if display_width_approx(line) <= max_width {
            out.push(line.to_string());
            continue;
        }

        if let Some((prefix, url)) = split_line_url(line) {
            let prefix = prefix.trim_end();
            if !prefix.is_empty() {
                out.push(prefix.to_string());
            }
            out.push(format!("    {url}"));
            continue;
        }

        out.push(line.to_string());
    }

    out.join("\n")
}

fn display_width_approx(s: &str) -> usize {
    s.chars().map(|ch| if ch.is_ascii() { 1 } else { 2 }).sum()
}

fn split_line_url(line: &str) -> Option<(&str, &str)> {
    let idx = line.find("https://").or_else(|| line.find("http://"))?;
    Some((&line[..idx], line[idx..].trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_title_repairs_wrapped_numbered_links() {
        let broken = "1. 第一条：https://example.com/abc                                      2. 第二条：https://example.com/def\nghi                                      3. 第三条：https://example.com/xyz";
        let normalized = normalize_title_for_display(broken);
        assert_eq!(
            normalized,
            "1. 第一条：https://example.com/abc\n2. 第二条：https://example.com/defghi\n3. 第三条：https://example.com/xyz"
        );
    }

    #[test]
    fn normalize_title_keeps_non_link_text() {
        let plain = "第一行\r\n第二行";
        let normalized = normalize_title_for_display(plain);
        assert_eq!(normalized, "第一行\n第二行");
    }

    #[test]
    fn wrap_title_moves_long_url_to_indented_lines() {
        let title = "1. 文档：https://example.com/very/long/path/for/test?foo=bar&baz=qux";
        let wrapped = wrap_title_for_terminal(title, 36);
        assert_eq!(
            wrapped,
            "1. 文档：\n    https://example.com/very/long/path/for/test?foo=bar&baz=qux"
        );
    }
}

fn is_probably_text(s: &str) -> bool {
    !s.chars()
        .any(|c| c == '\0' || (c.is_control() && c != '\n' && c != '\r' && c != '\t'))
}

fn clean_by_tags(db: &MemoBackend, tags_str: &str) {
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

fn open_urls(db: &MemoBackend, args: &[String], prefix: bool) {
    let tags = args
        .iter()
        .filter(|x| x.as_str() != "open" && x.as_str() != "o")
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect::<Vec<_>>();

    if tags.len() == 1 && is_object_id_like(&tags[0]) {
        if let Some(record) = db.get_record(&tags[0]).unwrap_or(None) {
            open_urls_from_title(&record.title);
        }
        return;
    }

    if tags.is_empty() {
        let prev_id = history::read_previous_operation();
        if is_object_id_like(&prev_id)
            && let Some(record) = db.get_record(&prev_id).unwrap_or(None)
        {
            open_urls_from_title(&record.title);
        }
        return;
    }

    let records = db
        .list_records_by_tags(&tags, false, prefix, -1, true, true)
        .unwrap_or_default();
    let mut urls = Vec::new();
    for r in records {
        urls.extend(extract_urls(&r.title));
    }

    if urls.is_empty() {
        println!(
            "there are NO urls associated with tags: {:?} (prefix: {})",
            tags, prefix
        );
        return;
    }
    open_choose_url(&urls);
}

fn open_urls_from_title(title: &str) {
    let urls = extract_urls(title);
    if urls.is_empty() {
        return;
    }
    open_choose_url(&urls);
}

fn extract_urls(s: &str) -> Vec<String> {
    let re = Regex::new(r#"https?://[^\s"'>)]+"#).unwrap();
    re.find_iter(s).map(|m| m.as_str().to_string()).collect()
}

fn open_choose_url(urls: &[String]) {
    let chosen = history::choose_from_list(&urls.to_vec());
    let Some(url) = chosen else {
        return;
    };
    let _ = std::process::Command::new("open").arg(&url).status();
}

fn default_print(db: &MemoBackend, limit: i64, list_special: bool) {
    let mut records = db
        .list_records_by_tags(
            &vec!["todo".to_string(), "urgent".to_string()],
            false,
            true,
            limit,
            false,
            false,
        )
        .unwrap_or_default();

    if !list_special {
        let patterns = load_special_patterns();
        records.retain(|r| !is_special_record(r, &patterns));
    }

    for record in records {
        ui::print_separator();
        let title = if is_probably_text(&record.title) {
            format!("\n{}", record.title.bright_white())
        } else {
            format!("\n{}", "<binary>".bright_yellow())
        };
        println!("{}", ui::format_record_like_go(&record, false, Some(title)));
        println!("{}", ui::colorize_id(&record.id));
        history::write_previous_operation(&record.id);
    }
}
