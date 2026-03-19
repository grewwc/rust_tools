use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use clap::Parser;
use colored::Colorize;
use regex::Regex;

use crate::common::configw;
use crate::common::prompt;
use crate::memo::{
    MemoBackend, MemoBackendMode, MemoMongo, MemoRecord, MemoTag, history,
};

pub static NUMBERED_ITEM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+\.\s").expect("invalid numbered item regex"));
pub static WRAPPED_NUMBERED_ITEM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<head>.*?)[ ]{2,}(?P<tail>\d+\.\s.*)$")
        .expect("invalid wrapped numbered item regex")
});

#[derive(Parser, Debug, Clone)]
#[command(about = "Memo record tool (go_tools re compatible)")]
pub struct Cli {
    #[arg(
        long,
        default_value = "",
        help = "backend: auto|mongo|sqlite. '.configW:re.backend'"
    )]
    pub backend: String,

    #[arg(short = 'i', default_value_t = false, help = "insert a record")]
    pub insert: bool,

    #[arg(
        long = "ct",
        num_args = 0..=1,
        default_missing_value = "",
        help = "change a record title"
    )]
    pub change_title: Option<String>,

    #[arg(
        short = 'u',
        num_args = 0..=1,
        default_missing_value = "",
        help = "update a record"
    )]
    pub update: Option<String>,

    #[arg(
        short = 'd',
        num_args = 0..=1,
        default_missing_value = "",
        help = "delete a record"
    )]
    pub delete: Option<String>,

    #[arg(
        short = 'f',
        num_args = 0..=1,
        default_missing_value = "",
        help = "finish a record"
    )]
    pub finish: Option<String>,

    #[arg(
        long = "nf",
        num_args = 0..=1,
        default_missing_value = "",
        help = "set a record unfinished"
    )]
    pub unfinish: Option<String>,

    #[arg(short = 'n', default_value_t = 100, help = "# of records to list")]
    pub n: i64,

    #[arg(short = 'r', default_value_t = false, help = "reverse sort")]
    pub reverse: bool,

    #[arg(long = "all", default_value_t = false, help = "including all record")]
    pub all: bool,

    #[arg(short = 'a', default_value_t = false, help = "shortcut for -all")]
    pub a: bool,

    #[arg(
        short = 't',
        long = "t",
        num_args = 0..=1,
        default_missing_value = "",
        help = "search by tags"
    )]
    pub tag_query: Option<String>,

    #[arg(
        long = "ta",
        num_args = 0..=1,
        default_missing_value = "",
        hide = true
    )]
    pub tag_query_ta: Option<String>,

    #[arg(
        long = "at",
        num_args = 0..=1,
        default_missing_value = "",
        hide = true
    )]
    pub tag_query_at: Option<String>,

    #[arg(
        long = "and",
        default_value_t = false,
        help = "use and logic to match tags"
    )]
    pub r#and: bool,

    #[arg(long = "include-finished", default_value_t = false)]
    pub include_finished: bool,

    #[arg(long = "title", default_value = "", help = "search by title")]
    pub title: String,

    #[arg(long = "c", default_value = "", help = "content (alias for title)")]
    pub content: String,

    #[arg(
        long = "search",
        short = 'q',
        default_value = "",
        help = "fuzzy search records"
    )]
    pub search: String,

    #[arg(long = "out", num_args = 0..=1, default_missing_value = "")]
    pub out: Option<String>,

    #[arg(long = "remote", default_value_t = false)]
    pub remote: bool,

    #[arg(long = "prev", default_value_t = false)]
    pub prev: bool,

    #[arg(long = "count", default_value_t = false)]
    pub count: bool,

    #[arg(long = "prefix", default_value_t = false)]
    pub prefix: bool,

    #[arg(long = "pre", default_value_t = false)]
    pub pre: bool,

    #[arg(long = "binary", default_value_t = false)]
    pub binary: bool,

    #[arg(short = 'b', default_value_t = false)]
    pub b: bool,

    #[arg(long = "force", default_value_t = false)]
    pub force: bool,

    #[arg(long = "sp", default_value_t = false)]
    pub sp: bool,

    #[arg(
        long = "ex",
        default_value = "",
        help = "exclude tag prefix when list tags"
    )]
    pub ex: String,

    #[arg(long = "code", default_value_t = false)]
    pub code: bool,

    #[arg(
        short = 's',
        default_value_t = false,
        help = "short format, only print tags"
    )]
    pub short: bool,

    #[arg(short = 'l', default_value_t = false, help = "list tags")]
    pub list_tags_time: bool,

    #[arg(long = "tags", default_value_t = false, help = "list all tags")]
    pub list_tags: bool,

    #[arg(long = "v", short = 'v', default_value_t = false)]
    pub verbose: bool,

    #[arg(long = "file", num_args = 0..=1, default_missing_value = "")]
    pub file_flag: Option<String>,

    #[arg(long = "e", default_value_t = false, help = "read from editor")]
    pub e: bool,

    #[arg(long = "host", default_value = "")]
    pub host: String,

    #[arg(long = "push", num_args = 0..=1, default_missing_value = "")]
    pub push: Option<String>,

    #[arg(long = "pull", num_args = 0..=1, default_missing_value = "")]
    pub pull: Option<String>,

    #[arg(long = "add-tag", num_args = 0..=1, default_missing_value = "")]
    pub add_tag: Option<String>,

    #[arg(long = "del-tag", num_args = 0..=1, default_missing_value = "")]
    pub del_tag: Option<String>,

    #[arg(long = "clean-tag", num_args = 0..=1, default_missing_value = "")]
    pub clean_tag: Option<String>,

    #[arg(long = "tag", default_value = "")]
    pub tag: String,

    #[arg(long = "db", default_value = "")]
    pub db: String,

    #[arg(long = "it", default_value_t = false, hide = true)]
    pub it: bool,

    #[arg(long = "ti", default_value_t = false, hide = true)]
    pub ti: bool,

    #[arg(
        value_name = "ARGS",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub args: Vec<String>,
}

#[derive(Clone)]
pub struct SearchHit {
    pub score: f64,
    pub record: MemoRecord,
    pub preview: Vec<String>,
}

pub fn normalize_legacy_single_dash_long_args(args: impl IntoIterator<Item = String>) -> Vec<String> {
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

pub fn maybe_parse_positional_limit(cli: &mut Cli) {
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

pub fn pos_has(args: &[String], needle: &str) -> bool {
    args.iter().any(|x| x == needle)
}

pub fn get_tag_query_raw(cli: &Cli) -> Option<String> {
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

pub fn should_update_feature(cli: &Cli) -> bool {
    cli.update.is_some() || pos_has(&cli.args, "u")
}

pub fn should_insert_feature(cli: &Cli) -> bool {
    cli.insert || pos_has(&cli.args, "i")
}

pub fn should_log_command(cli: &Cli) -> bool {
    pos_has(&cli.args, "log") && !pos_has(&cli.args, "u") && cli.update.is_none()
}

pub fn should_list_tags_feature(cli: &Cli, list_tags_and_order_by_time: bool) -> bool {
    list_tags_and_order_by_time
        || cli.list_tags
        || pos_has(&cli.args, "tags")
        || pos_has(&cli.args, "t")
        || pos_has(&cli.args, "i")
}

pub fn should_list_by_tag_name(
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

pub fn order_by_time(cli: &Cli, tag_query_raw: Option<&str>) -> bool {
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

pub fn open_backend(cli: &Cli) -> Result<MemoBackend, String> {
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

pub fn resolve_host(cli_host: &str) -> String {
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

pub fn first_non_empty<'a>(items: &[&'a str]) -> &'a str {
    for s in items {
        if !s.trim().is_empty() {
            return s;
        }
    }
    ""
}

pub fn parse_tag_query(s: &str) -> Vec<String> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

pub fn is_object_id_like(s: &str) -> bool {
    let s = s.trim();
    s.len() == 24 && s.chars().all(|c| c.is_ascii_hexdigit())
}

pub fn resolve_id_or_prev(id: &str, prev: bool) -> String {
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

pub fn primary_title(title: &str) -> String {
    let s = title.replace('\r', "");
    for line in s.lines() {
        let line = line.trim();
        if !line.is_empty() {
            return line.to_string();
        }
    }
    s.trim().to_string()
}

pub fn choose_record_id(records: &[MemoRecord]) -> Option<String> {
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

pub fn resolve_record_ref_local(
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

pub fn load_special_patterns() -> Vec<String> {
    let mut patterns = vec!["learn".to_string()];
    patterns.extend(parse_tag_query(&configw::get_config("special.tags", "")));
    patterns.retain(|x| !x.trim().is_empty());
    patterns.sort();
    patterns.dedup();
    patterns
}

pub fn is_special_record(record: &MemoRecord, patterns: &[String]) -> bool {
    record
        .tags
        .iter()
        .any(|t| patterns.iter().any(|p| t.starts_with(p)))
}

pub fn compute_tags_from_records(
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

pub fn filter_tags_ex(tags: Vec<MemoTag>, ex: &str) -> Vec<MemoTag> {
    let ex_prefix = parse_tag_query(ex);
    if ex_prefix.is_empty() {
        return tags;
    }
    tags.into_iter()
        .filter(|t| !ex_prefix.iter().any(|p| t.name.starts_with(p)))
        .collect()
}

pub fn get_terminal_width() -> usize {
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

pub fn records_to_output_text(records: &[MemoRecord]) -> String {
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

pub fn write_text_output(path: &str, content: &str, force: bool) {
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

pub fn split_binary_title(title: &str) -> Option<(String, String)> {
    let s = title.replace('\r', "");
    let mut lines = s.lines();
    let filename = lines.next()?.trim().to_string();
    if filename.is_empty() {
        return None;
    }
    let content = lines.collect::<Vec<_>>().join("\n");
    Some((filename, content))
}

pub fn write_file_prompt(path: &str, content: &str, force: bool) {
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

pub fn print_title_with_colored_separator(title: &str) -> String {
    if !title.contains("<sep>") {
        return title.to_string();
    }
    let sep = "~~~~~~~~~~~~".repeat(10).cyan().to_string();
    title.replace("<sep>", &sep)
}

pub fn normalize_title_for_display(title: &str) -> String {
    let normalized_newline = title.replace("\r\n", "\n").replace('\r', "\n");
    let mut split_lines = Vec::new();
    for line in normalized_newline.lines() {
        if line.trim().is_empty() {
            split_lines.push(String::new());
            continue;
        }
        split_wrapped_numbered_line(line, &mut split_lines);
    }

    let mut merged: Vec<String> = Vec::new();
    for line in split_lines {
        let line = line.trim().to_string();
        if line.is_empty() {
            merged.push(String::new());
            continue;
        }
        let is_numbered_item = NUMBERED_ITEM_RE.is_match(&line);
        if !is_numbered_item
            && let Some(prev) = merged.last_mut()
            && (prev.contains("https://") || prev.contains("http://"))
            && looks_like_url_continuation_fragment(&line)
        {
            prev.push_str(&line);
            continue;
        }
        merged.push(line);
    }

    if merged.is_empty() {
        normalized_newline
    } else {
        merged.join("\n")
    }
}

pub fn looks_like_url_continuation_fragment(line: &str) -> bool {
    let line = line.trim();
    if line.is_empty() {
        return false;
    }
    if line.contains("http://") || line.contains("https://") {
        return false;
    }
    if line.contains('：') {
        return false;
    }
    if line.starts_with('~') || line.starts_with('-') {
        return false;
    }
    if let Some(idx) = line.find(':')
        && idx <= 24
    {
        return false;
    }
    if line.chars().any(|ch| ch.is_whitespace()) {
        return false;
    }
    line.chars().all(|ch| {
        ch.is_ascii_alphanumeric()
            || matches!(
                ch,
                '/' | '?' | '#' | '[' | ']' | '@' | '!' | '$' | '&' | '\'' | '(' | ')' | '*'
                    | '+' | ',' | ';' | '=' | '%' | '-' | '.' | '_' | '~'
            )
    })
}

pub fn split_wrapped_numbered_line(line: &str, out: &mut Vec<String>) {
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

pub fn wrap_title_for_terminal(title: &str, terminal_width: usize) -> String {
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

pub fn display_width_approx(s: &str) -> usize {
    s.chars().map(|ch| if ch.is_ascii() { 1 } else { 2 }).sum()
}

pub fn split_line_url(line: &str) -> Option<(&str, &str)> {
    let idx = line.find("https://").or_else(|| line.find("http://"))?;
    Some((&line[..idx], line[idx..].trim()))
}

pub fn is_probably_text(s: &str) -> bool {
    !s.chars()
        .any(|c| c == '\0' || (c.is_control() && c != '\n' && c != '\r' && c != '\t'))
}

pub fn extract_urls(s: &str) -> Vec<String> {
    let re = Regex::new(r#"https?://[^\s"'>)]+"#).unwrap();
    re.find_iter(s).map(|m| m.as_str().to_string()).collect()
}

pub fn open_choose_url(urls: &[String]) {
    let chosen = history::choose_from_list(&urls.to_vec());
    let Some(url) = chosen else {
        return;
    };
    let _ = std::process::Command::new("open").arg(&url).status();
}
