use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use crate::memo::{MemoBackend, db::MemoDb, history};

const REMOTE_SQLITE_PATH: &str = "~/.go_tools_memo.sqlite3";

pub fn sync_record_to_host(
    local_db: &MemoBackend,
    record_id: &str,
    host: &str,
) -> Result<(), String> {
    let record_ids = vec![record_id.to_string()];
    sync_records_to_host(local_db, &record_ids, host).map(|_| ())
}

pub fn remote_sqlite_target_display(host: &str) -> String {
    let (host_no_port, port) = split_host_port(host);
    match port {
        Some(port) => format!("{host_no_port}:{REMOTE_SQLITE_PATH} (port {port})"),
        None => format!("{host_no_port}:{REMOTE_SQLITE_PATH}"),
    }
}

pub fn sync_records_to_host(
    local_db: &MemoBackend,
    record_ids: &[String],
    host: &str,
) -> Result<usize, String> {
    let record_ids = unique_non_empty_record_ids(record_ids);
    if record_ids.is_empty() {
        return Ok(0);
    }

    let tmp = temp_sqlite_path();
    match pull_remote_sqlite_to_temp(host, &tmp, true) {
        Ok(()) => {}
        Err(err) if remote_sqlite_missing(&err) => {
            MemoDb::open(&tmp).map_err(|e| e.to_string())?;
        }
        Err(err) => return Err(err),
    }
    let remote_db = MemoDb::open(&tmp).map_err(|e| e.to_string())?;
    upsert_records_to_remote(local_db, &remote_db, &record_ids)?;
    remote_db
        .checkpoint_wal_truncate()
        .map_err(|e| e.to_string())?;
    push_temp_sqlite_to_remote(host, &tmp)?;
    Ok(record_ids.len())
}

fn unique_non_empty_record_ids(record_ids: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for record_id in record_ids {
        let record_id = record_id.trim();
        if record_id.is_empty() {
            continue;
        }
        if seen.insert(record_id.to_string()) {
            unique.push(record_id.to_string());
        }
    }
    unique
}

fn upsert_records_to_remote(
    local_db: &MemoBackend,
    remote_db: &MemoDb,
    record_ids: &[String],
) -> Result<(), String> {
    for record_id in record_ids {
        let Some(record) = local_db.get_record(record_id)? else {
            return Err(format!("record {record_id} not found"));
        };
        remote_db
            .upsert_record(&record)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn sync_record_from_host(
    local_db: &MemoBackend,
    record_ref: &str,
    host: &str,
) -> Result<(), String> {
    let tmp = temp_sqlite_path();
    pull_remote_sqlite_to_temp(host, &tmp, true)?;
    let remote_db = MemoDb::open(&tmp).map_err(|e| e.to_string())?;
    let record_id = resolve_record_ref(&remote_db, record_ref)?;
    let Some(record) = remote_db
        .get_record(&record_id)
        .map_err(|e| e.to_string())?
    else {
        return Err(format!("record {record_id} not found on remote"));
    };
    local_db.upsert_record(&record)?;
    Ok(())
}

fn resolve_record_ref(remote_db: &MemoDb, record_ref: &str) -> Result<String, String> {
    let mut ref_str = record_ref.trim().to_string();
    if ref_str.is_empty() {
        ref_str = crate::common::prompt::read_line("Input the ObjectID/title/tag: ")
            .trim()
            .to_string();
    }
    if is_object_id_like(&ref_str) {
        return Ok(ref_str);
    }

    let records = remote_db
        .list_records(-1, false, true)
        .map_err(|e| e.to_string())?;

    let exact = records
        .iter()
        .filter(|r| {
            primary_title(&r.title).eq_ignore_ascii_case(&ref_str)
                || r.title.trim().eq_ignore_ascii_case(&ref_str)
        })
        .map(|r| format!("{}\t{}", r.id, primary_title(&r.title)))
        .collect::<Vec<_>>();
    let exact_tag = records
        .iter()
        .filter(|r| r.tags.iter().any(|t| t == &ref_str))
        .map(|r| format!("{}\t{}", r.id, primary_title(&r.title)))
        .collect::<Vec<_>>();
    let mut merged = exact;
    for x in exact_tag {
        if !merged.contains(&x) {
            merged.push(x);
        }
    }
    if !merged.is_empty() {
        let chosen =
            history::choose_from_list(&merged).ok_or_else(|| "no selection".to_string())?;
        let id = chosen.split_whitespace().next().unwrap_or("").to_string();
        return Ok(id);
    }

    let lower = ref_str.to_lowercase();
    let fuzzy = records
        .iter()
        .filter(|r| {
            r.title.to_lowercase().contains(&lower)
                || r.tags.iter().any(|t| t.starts_with(&ref_str))
        })
        .map(|r| format!("{}\t{}", r.id, primary_title(&r.title)))
        .collect::<Vec<_>>();
    if fuzzy.is_empty() {
        return Err(format!("record {ref_str} not found on remote"));
    }
    let chosen = history::choose_from_list(&fuzzy).ok_or_else(|| "no selection".to_string())?;
    let id = chosen.split_whitespace().next().unwrap_or("").to_string();
    Ok(id)
}

fn primary_title(title: &str) -> String {
    let s = title.replace('\r', "");
    for line in s.lines() {
        let t = line.trim();
        if !t.is_empty() {
            return t.to_string();
        }
    }
    s.trim().to_string()
}

fn is_object_id_like(s: &str) -> bool {
    let s = s.trim();
    s.len() == 24 && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn pull_remote_sqlite_to_temp(
    host: &str,
    tmp_path: &Path,
    prepare_remote: bool,
) -> Result<(), String> {
    let host = host.trim();
    if host.is_empty() {
        return Err("host is required".to_string());
    }
    if prepare_remote {
        if let Err(err) = run_remote_checkpoint(host) {
            eprintln!("warning: failed to checkpoint remote sqlite before sync: {err}");
        }
    }
    let mut args = scp_base_args(host);
    args.push(remote_sqlite_spec(host));
    args.push(tmp_path.to_string_lossy().to_string());
    run_command_checked("scp", &args)?;
    Ok(())
}

fn push_temp_sqlite_to_remote(host: &str, tmp_path: &Path) -> Result<(), String> {
    let host = host.trim();
    if host.is_empty() {
        return Err("host is required".to_string());
    }
    let mut args = scp_base_args(host);
    args.push(tmp_path.to_string_lossy().to_string());
    args.push(remote_sqlite_spec(host));
    run_command_checked("scp", &args)?;
    Ok(())
}

fn run_remote_checkpoint(host: &str) -> Result<(), String> {
    let timeout = 8;
    let mut args = vec!["-o".to_string(), format!("ConnectTimeout={timeout}")];
    let (host_no_port, port) = split_host_port(host);
    if let Some(port) = port {
        args.push("-p".to_string());
        args.push(port.to_string());
    }
    args.push(host_no_port.to_string());
    args.push(format!(
        "sqlite3 {REMOTE_SQLITE_PATH} 'PRAGMA wal_checkpoint(TRUNCATE);'"
    ));
    run_command_checked("ssh", &args).map(|_| ())
}

fn temp_sqlite_path() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "rust_tools_memo_remote_{}.sqlite3",
        uuid::Uuid::new_v4()
    ));
    p
}

fn split_host_port(host: &str) -> (&str, Option<u16>) {
    let host = host.trim();
    let Some(idx) = host.rfind(':') else {
        return (host, None);
    };
    let (h, port_str) = host.split_at(idx);
    let port_str = &port_str[1..];
    if port_str.chars().all(|c| c.is_ascii_digit())
        && let Ok(port) = port_str.parse::<u16>()
    {
        return (h, Some(port));
    }
    (host, None)
}

fn scp_base_args(host: &str) -> Vec<String> {
    let (_, port) = split_host_port(host);
    let mut args = vec!["-q".to_string()];
    if let Some(port) = port {
        args.push("-P".to_string());
        args.push(port.to_string());
    }
    args
}

fn remote_sqlite_spec(host: &str) -> String {
    let (host_no_port, _) = split_host_port(host);
    format!("{host_no_port}:{REMOTE_SQLITE_PATH}")
}

fn remote_sqlite_missing(err: &str) -> bool {
    let lower = err.to_lowercase();
    lower.contains("no such file or directory")
        || lower.contains("not a regular file")
        || lower.contains("could not stat remote file")
}

fn run_command_checked(program: &str, args: &[String]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("failed to run {}: {}", format_command(program, args), e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        let details = if !stderr.trim().is_empty() {
            stderr.trim().to_string()
        } else if !stdout.trim().is_empty() {
            stdout.trim().to_string()
        } else {
            format!("exit status {}", output.status)
        };
        return Err(format!(
            "{} failed: {}",
            format_command(program, args),
            details,
        ));
    }

    let mut combined = stdout;
    if !stderr.is_empty() {
        combined.push_str(&stderr);
    }
    Ok(combined)
}

fn format_command(program: &str, args: &[String]) -> String {
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(program.to_string());
    parts.extend(args.iter().cloned());
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> MemoDb {
        let dir = std::env::temp_dir().join(format!(
            "rust_tools_memo_sync_test_{}",
            uuid::Uuid::new_v4()
        ));
        let path = dir.join("memo.sqlite3");
        MemoDb::open(path).unwrap()
    }

    #[test]
    fn upsert_records_to_remote_syncs_all_requested_records() {
        let local_db = temp_db();
        let remote_db = temp_db();

        let first_id = local_db
            .insert("first", &vec!["links".to_string()])
            .unwrap();
        let second_id = local_db
            .insert("second", &vec!["links".to_string(), "read".to_string()])
            .unwrap();
        let third_id = local_db
            .insert("third", &vec!["other".to_string()])
            .unwrap();

        let backend = MemoBackend::Sqlite(local_db);
        upsert_records_to_remote(
            &backend,
            &remote_db,
            &[first_id.clone(), second_id.clone(), first_id.clone()],
        )
        .unwrap();

        assert!(remote_db.get_record(&first_id).unwrap().is_some());
        assert!(remote_db.get_record(&second_id).unwrap().is_some());
        assert!(remote_db.get_record(&third_id).unwrap().is_none());
    }

    #[test]
    fn unique_non_empty_record_ids_preserves_first_occurrence_order() {
        let unique = unique_non_empty_record_ids(&[
            String::new(),
            "first".to_string(),
            "second".to_string(),
            "first".to_string(),
            "  second  ".to_string(),
        ]);

        assert_eq!(unique, vec!["first".to_string(), "second".to_string()]);
    }

    #[test]
    fn remote_sqlite_target_display_includes_port_when_present() {
        assert_eq!(
            remote_sqlite_target_display("user@example.com:2222"),
            "user@example.com:~/.go_tools_memo.sqlite3 (port 2222)"
        );
        assert_eq!(
            remote_sqlite_target_display("user@example.com"),
            "user@example.com:~/.go_tools_memo.sqlite3"
        );
    }

    #[test]
    fn remote_sqlite_spec_keeps_host_with_remote_path() {
        assert_eq!(
            remote_sqlite_spec("user@example.com:2222"),
            "user@example.com:~/.go_tools_memo.sqlite3"
        );
    }

    #[test]
    fn remote_sqlite_missing_matches_common_scp_errors() {
        assert!(remote_sqlite_missing(
            "scp failed: scp: ~/.go_tools_memo.sqlite3: No such file or directory"
        ));
        assert!(!remote_sqlite_missing("permission denied"));
    }
}
