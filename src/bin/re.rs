use clap::Parser;
pub use rust_tools::cmd;
pub use rust_tools::common;
pub use rust_tools::strw;

#[path = "re/memo/mod.rs"]
mod memo;

#[path = "re/features/mod.rs"]
mod features;

use features::*;

const DEFAULT_TXT_OUTPUT_NAME: &str = "output.txt";

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
        finish::handle_finish_command(&db, &cli, cli.unfinish.as_deref(), false, prefix);
        return;
    }

    if cli.finish.is_some() {
        finish::handle_finish_command(&db, &cli, cli.finish.as_deref(), true, prefix);
        return;
    }

    if pos_has(&cli.args, "open") || pos_has(&cli.args, "o") {
        open::open_urls(&db, &cli.args, prefix);
        return;
    }

    if cli.clean_tag.is_some() {
        clean_tag::clean_by_tags(&db, cli.clean_tag.as_deref().unwrap_or(""));
        return;
    }

    if should_log_command(&cli) {
        log::handle_log(&db, &cli, use_vscode);
        return;
    }

    if pos_has(&cli.args, "week") {
        week::handle_week(&db);
        return;
    }

    if should_insert_feature(&cli) {
        insert::insert_feature(&db, &cli, use_vscode);
        return;
    }

    if should_list_tags_feature(&cli, list_tags_and_order_by_time) {
        list_tags::list_tags_feature(
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
        update::update_feature(&db, &cli, prefix, use_vscode);
        return;
    }

    if cli.change_title.is_some() {
        change_title::change_title_feature(&db, &cli, use_vscode);
        return;
    }

    if cli.delete.is_some() || pos_has(&cli.args, "d") {
        delete::delete_feature(&db, &cli);
        return;
    }

    if cli.add_tag.is_some() {
        add_tag::add_or_del_tags_feature(&db, &cli, true);
        return;
    }

    if cli.del_tag.is_some() {
        del_tag::add_or_del_tags_feature(&db, &cli, false);
        return;
    }

    if let Some(push_ref) = cli.push.as_deref()
        && !push_ref.trim().is_empty()
    {
        push::push_single(&db, push_ref, &cli.host);
        return;
    }

    let title_query = first_non_empty(&[cli.title.as_str(), cli.content.as_str()]);
    if !title_query.trim().is_empty() {
        list_by_title::list_by_title_feature(
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
        search::search_feature(
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
        list_by_tag_name::list_by_tag_name_feature(
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
        pull::pull_single(&db, pull_ref, &cli.host);
        return;
    }

    if cli.push.is_some() {
        push::push_single(&db, "", &cli.host);
        return;
    }

    default_print::default_print(&db, record_limit, list_special);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ie_sets_insert_and_editor() {
        let argv = normalize_legacy_single_dash_long_args(["re".to_string(), "-ie".to_string()]);
        let cli = Cli::parse_from(argv);
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
