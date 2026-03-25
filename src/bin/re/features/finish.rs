use crate::features::core::*;
use crate::memo::{MemoBackend, history};

pub fn handle_finish_command(
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

pub fn toggle_finish(db: &MemoBackend, id: &str, finished: bool) {
    let id = id.trim();
    if id.is_empty() {
        return;
    }
    let _ = db.set_finished(id, finished);
    history::write_previous_operation(id);
}

pub fn collect_refs(value: Option<&str>, args: &[String], skip_tokens: &[&str]) -> Vec<String> {
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
        if skip_tokens.contains(&a) {
            continue;
        }
        refs.push(a.to_string());
    }
    refs
}
