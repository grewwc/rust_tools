use std::sync::Arc;

use rust_tools::terminalw;

use crate::features::core::*;
use crate::memo::MemoBackend;

use super::{
    add_tag, change_title, clean_tag, default_print, del_tag, delete, finish, insert,
    list_by_tag_name, list_by_title, list_tags, log, open, pull, push, search, update, week,
};

fn should_list_by_positional_object_id(
    cli: &Cli,
    list_tags_and_order_by_time: bool,
    title_query: &str,
    search_query: &str,
) -> bool {
    if list_tags_and_order_by_time {
        return false;
    }
    if !title_query.trim().is_empty() || !search_query.trim().is_empty() {
        return false;
    }
    if cli.args.len() != 1 {
        return false;
    }
    is_object_id_like(&cli.args[0])
}

pub struct ReContext {
    pub db: Arc<MemoBackend>,
    pub cli: Arc<Cli>,
    pub prefix: bool,
    pub use_vscode: bool,
    pub list_special: bool,
    pub reverse: bool,
    pub all: bool,
    pub include_finished: bool,
    pub record_limit: i64,
    pub out_name: String,
    pub clipboard: bool,
    pub to_binary: bool,
    pub only_tags: bool,
    pub tag_query_raw: Option<String>,
    pub tag_query_non_empty: bool,
    pub list_tags_and_order_by_time: bool,
    pub title_query: String,
}

pub fn register_all(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    register_nf(parser, Arc::clone(&ctx));
    register_f(parser, Arc::clone(&ctx));
    register_open(parser, Arc::clone(&ctx));
    register_clean_tag(parser, Arc::clone(&ctx));
    register_log(parser, Arc::clone(&ctx));
    register_week(parser, Arc::clone(&ctx));
    register_insert(parser, Arc::clone(&ctx));
    register_list_tags(parser, Arc::clone(&ctx));
    register_update(parser, Arc::clone(&ctx));
    register_change_title(parser, Arc::clone(&ctx));
    register_delete(parser, Arc::clone(&ctx));
    register_add_tag(parser, Arc::clone(&ctx));
    register_del_tag(parser, Arc::clone(&ctx));
    register_push(parser, Arc::clone(&ctx));
    register_list_by_title(parser, Arc::clone(&ctx));
    register_search(parser, Arc::clone(&ctx));
    register_list_by_tag_name(parser, Arc::clone(&ctx));
    register_pull(parser, Arc::clone(&ctx));
    register_push_empty(parser, Arc::clone(&ctx));
    register_default(parser, ctx);
}

fn register_nf(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| ctx.cli.unfinish.is_some() || pos_has(&ctx.cli.args, "nf")
        })
        .do_action(move || {
            finish::handle_finish_command(
                ctx.db.as_ref(),
                ctx.cli.as_ref(),
                ctx.cli.unfinish.as_deref(),
                false,
                ctx.prefix,
            );
        });
}

fn register_f(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| ctx.cli.finish.is_some()
        })
        .do_action(move || {
            finish::handle_finish_command(
                ctx.db.as_ref(),
                ctx.cli.as_ref(),
                ctx.cli.finish.as_deref(),
                true,
                ctx.prefix,
            );
        });
}

fn register_open(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| pos_has(&ctx.cli.args, "open") || pos_has(&ctx.cli.args, "o")
        })
        .do_action(move || {
            open::open_urls(ctx.db.as_ref(), &ctx.cli.args, ctx.prefix);
        });
}

fn register_clean_tag(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| ctx.cli.clean_tag.is_some()
        })
        .do_action(move || {
            clean_tag::clean_by_tags(ctx.db.as_ref(), ctx.cli.clean_tag.as_deref().unwrap_or(""));
        });
}

fn register_log(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| should_log_command(ctx.cli.as_ref())
        })
        .do_action(move || {
            log::handle_log(ctx.db.as_ref(), ctx.cli.as_ref(), ctx.use_vscode);
        });
}

fn register_week(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| pos_has(&ctx.cli.args, "week")
        })
        .do_action(move || {
            week::handle_week(ctx.db.as_ref());
        });
}

fn register_insert(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| should_insert_feature(ctx.cli.as_ref())
        })
        .do_action(move || {
            insert::insert_feature(ctx.db.as_ref(), ctx.cli.as_ref(), ctx.use_vscode);
        });
}

fn register_list_tags(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| should_list_tags_feature(ctx.cli.as_ref(), ctx.list_tags_and_order_by_time)
        })
        .do_action(move || {
            list_tags::list_tags_feature(
                ctx.db.as_ref(),
                ctx.cli.as_ref(),
                ctx.record_limit,
                ctx.reverse,
                ctx.all,
                ctx.list_special,
                ctx.list_tags_and_order_by_time,
            );
        });
}

fn register_update(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| should_update_feature(ctx.cli.as_ref())
        })
        .do_action(move || {
            update::update_feature(
                ctx.db.as_ref(),
                ctx.cli.as_ref(),
                ctx.prefix,
                ctx.use_vscode,
            );
        });
}

fn register_change_title(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| ctx.cli.change_title.is_some()
        })
        .do_action(move || {
            change_title::change_title_feature(ctx.db.as_ref(), ctx.cli.as_ref(), ctx.use_vscode);
        });
}

fn register_delete(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| ctx.cli.delete.is_some() || pos_has(&ctx.cli.args, "d")
        })
        .do_action(move || {
            delete::delete_feature(ctx.db.as_ref(), ctx.cli.as_ref());
        });
}

fn register_add_tag(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| ctx.cli.add_tag.is_some()
        })
        .do_action(move || {
            add_tag::add_or_del_tags_feature(ctx.db.as_ref(), ctx.cli.as_ref(), true);
        });
}

fn register_del_tag(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| ctx.cli.del_tag.is_some()
        })
        .do_action(move || {
            del_tag::add_or_del_tags_feature(ctx.db.as_ref(), ctx.cli.as_ref(), false);
        });
}

fn register_push(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| matches!(ctx.cli.push.as_deref(), Some(v) if !v.trim().is_empty())
        })
        .do_action(move || {
            let push_ref = ctx.cli.push.as_deref().unwrap_or("").trim();
            push::push_single(ctx.db.as_ref(), push_ref, &ctx.cli.host);
        });
}

fn register_list_by_title(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| !ctx.title_query.trim().is_empty()
        })
        .do_action(move || {
            list_by_title::list_by_title_feature(
                ctx.db.as_ref(),
                &ctx.title_query,
                ctx.tag_query_raw.as_deref().unwrap_or(""),
                ctx.cli.r#and,
                ctx.prefix,
                ctx.record_limit,
                ctx.reverse,
                ctx.include_finished,
                &ctx.out_name,
                ctx.clipboard,
                ctx.to_binary,
                ctx.cli.count,
                ctx.cli.verbose,
                ctx.cli.force,
            );
        });
}

fn register_search(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| !ctx.cli.search.trim().is_empty()
        })
        .do_action(move || {
            search::search_feature(
                ctx.db.as_ref(),
                &ctx.cli.search,
                ctx.tag_query_raw.as_deref().unwrap_or(""),
                ctx.cli.r#and,
                ctx.prefix,
                ctx.record_limit,
                ctx.include_finished,
                ctx.list_special,
                &ctx.out_name,
                ctx.clipboard,
                ctx.to_binary,
                ctx.cli.count,
                ctx.cli.verbose,
                ctx.cli.force,
            );
        });
}

fn register_list_by_tag_name(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| {
                should_list_by_tag_name(
                    ctx.tag_query_non_empty,
                    ctx.list_tags_and_order_by_time,
                    &ctx.title_query,
                    &ctx.cli.search,
                ) || should_list_by_positional_object_id(
                    ctx.cli.as_ref(),
                    ctx.list_tags_and_order_by_time,
                    &ctx.title_query,
                    &ctx.cli.search,
                )
            }
        })
        .do_action(move || {
            let by_tag_query = should_list_by_tag_name(
                ctx.tag_query_non_empty,
                ctx.list_tags_and_order_by_time,
                &ctx.title_query,
                &ctx.cli.search,
            );
            let query = if by_tag_query {
                ctx.tag_query_raw.as_deref().unwrap_or("")
            } else {
                ctx.cli.args.first().map(|s| s.as_str()).unwrap_or("")
            };
            list_by_tag_name::list_by_tag_name_feature(
                ctx.db.as_ref(),
                query,
                ctx.cli.as_ref(),
                ctx.prefix,
                ctx.record_limit,
                ctx.reverse,
                ctx.include_finished,
                ctx.list_special,
                ctx.only_tags,
                ctx.to_binary,
                &ctx.out_name,
                ctx.clipboard,
            );
        });
}

fn register_pull(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| ctx.cli.pull.is_some()
        })
        .do_action(move || {
            pull::pull_single(
                ctx.db.as_ref(),
                ctx.cli.pull.as_deref().unwrap_or(""),
                &ctx.cli.host,
            );
        });
}

fn register_push_empty(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser
        .on({
            let ctx = Arc::clone(&ctx);
            move |_| ctx.cli.push.is_some()
        })
        .do_action(move || {
            push::push_single(ctx.db.as_ref(), "", &ctx.cli.host);
        });
}

fn register_default(parser: &mut terminalw::Parser, ctx: Arc<ReContext>) {
    parser.on(|_| true).do_action(move || {
        default_print::default_print(ctx.db.as_ref(), ctx.record_limit, ctx.list_special);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_object_id_triggers_list_by_tag_name_path() {
        let argv = normalize_legacy_single_dash_long_args([
            "re".to_string(),
            "6938d88ca1f8f546540c3d68".to_string(),
        ]);
        let cli = parse_cli_and_parser(argv).1;
        let title_query = first_non_empty(&[cli.title.as_str(), cli.content.as_str()]).to_string();
        let list_tags_and_order_by_time = order_by_time(&cli, None);
        assert!(should_list_by_positional_object_id(
            &cli,
            list_tags_and_order_by_time,
            &title_query,
            &cli.search
        ));
    }

    #[test]
    fn positional_object_id_does_not_override_title_search() {
        let argv = normalize_legacy_single_dash_long_args([
            "re".to_string(),
            "--title=hello".to_string(),
            "6938d88ca1f8f546540c3d68".to_string(),
        ]);
        let cli = parse_cli_and_parser(argv).1;
        let title_query = first_non_empty(&[cli.title.as_str(), cli.content.as_str()]).to_string();
        let list_tags_and_order_by_time = order_by_time(&cli, None);
        assert!(!should_list_by_positional_object_id(
            &cli,
            list_tags_and_order_by_time,
            &title_query,
            &cli.search
        ));
    }
}
