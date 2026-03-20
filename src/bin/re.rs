pub use rust_tools::cmd;
pub use rust_tools::common;
pub use rust_tools::strw;

#[path = "re/memo/mod.rs"]
mod memo;

#[path = "re/features/mod.rs"]
mod features;

use features::*;
use std::sync::Arc;

const DEFAULT_TXT_OUTPUT_NAME: &str = "output.txt";

fn main() {
    let argv = normalize_legacy_single_dash_long_args(std::env::args());
    let (mut parser, cli) = parse_cli_and_parser(argv);

    let db = open_backend(&cli).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let db = Arc::new(db);
    let cli = Arc::new(cli);

    let to_binary = cli.binary || cli.b;
    let all = (cli.all
        || (cli.a && cli.add_tag.is_none() && cli.del_tag.is_none() && !cli.list_tags))
        && !to_binary;
    let prefix = cli.prefix || cli.pre || all || cli.a;
    let only_tags = cli.short;
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
                DEFAULT_TXT_OUTPUT_NAME.to_string()
            } else {
                s.trim().to_string()
            }
        })
        .unwrap_or_default();

    let tag_query_raw = get_tag_query_raw(&cli);
    let tag_query_non_empty = tag_query_raw
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty());
    let list_tags_and_order_by_time = order_by_time(&cli, tag_query_raw.as_deref());

    let title_query = first_non_empty(&[cli.title.as_str(), cli.content.as_str()]).to_string();

    let ctx = Arc::new(features::register::ReContext {
        db: Arc::clone(&db),
        cli: Arc::clone(&cli),
        prefix,
        use_vscode,
        list_special,
        reverse,
        all,
        include_finished,
        record_limit,
        out_name,
        to_binary,
        only_tags,
        tag_query_raw,
        tag_query_non_empty,
        list_tags_and_order_by_time,
        title_query,
    });

    features::register::register_all(&mut parser, ctx);
    let _ = parser.execute_first();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ie_sets_insert_and_editor() {
        let argv = normalize_legacy_single_dash_long_args(["re".to_string(), "-ie".to_string()]);
        let cli = parse_cli_and_parser(argv).1;
        assert!(cli.insert);
        assert!(cli.e);
    }

    #[test]
    fn normalize_single_dash_long_flags() {
        let argv = normalize_legacy_single_dash_long_args([
            "re".to_string(),
            "-include-finished".to_string(),
            "-backend=mongo".to_string(),
            "-add-tag=todo".to_string(),
        ]);
        assert!(argv.iter().any(|a| a == "--include-finished"));
        assert!(argv.iter().any(|a| a == "--backend=mongo"));
        assert!(argv.iter().any(|a| a == "--add-tag=todo"));
    }

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
    fn parse_backend_with_equals_value() {
        let argv = normalize_legacy_single_dash_long_args([
            "re".to_string(),
            "-backend=mongo".to_string(),
            "-n=12".to_string(),
        ]);
        let cli = parse_cli_and_parser(argv).1;
        assert_eq!(cli.backend, "mongo");
        assert_eq!(cli.n, 12);
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

    #[test]
    fn normalize_title_preserves_newlines_after_url_lines() {
        let original = "fabric数据集链接：\nhttps://example.com/a\n子数据集id：5130071\ntask_id：t#1770303001804771022\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nfabric数据集ID：4333589";
        let normalized = normalize_title_for_display(original);
        assert_eq!(normalized, original);
    }
}
