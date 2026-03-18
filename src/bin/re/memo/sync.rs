use std::path::Path;

use crate::{
    cmd::run::run_cmd,
    memo::{MemoBackend, db::MemoDb, history},
};

const REMOTE_SQLITE_PATH: &str = "$HOME/.go_tools_memo.sqlite3";

pub fn sync_record_to_host(
    local_db: &MemoBackend,
    record_id: &str,
    host: &str,
) -> Result<(), String> {
    let tmp = temp_sqlite_path();
    pull_remote_sqlite_to_temp(host, &tmp, true)?;
    let remote_db = MemoDb::open(&tmp).map_err(|e| e.to_string())?;
    let Some(record) = local_db.get_record(record_id)? else {
        return Err(format!("record {record_id} not found"));
    };
    remote_db
        .upsert_record(&record)
        .map_err(|e| e.to_string())?;
    remote_db
        .checkpoint_wal_truncate()
        .map_err(|e| e.to_string())?;
    push_temp_sqlite_to_remote(host, &tmp)?;
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
        ref_str = crate::common::prompt::read_line("Input the ObjectID/title/tag: ").trim().to_string();
    }
    if is_object_id_like(&ref_str) {
        return Ok(ref_str);
    }

    let records = remote_db
        .list_records(-1, false, true)
        .map_err(|e| e.to_string())?;

    let exact = records
        .iter()
        .filter(|r| primary_title(&r.title).eq_ignore_ascii_case(&ref_str) || r.title.trim().eq_ignore_ascii_case(&ref_str))
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
        let chosen = history::choose_from_list(&merged)
            .ok_or_else(|| "no selection".to_string())?;
        let id = chosen.split_whitespace().next().unwrap_or("").to_string();
        return Ok(id);
    }

    let lower = ref_str.to_lowercase();
    let fuzzy = records
        .iter()
        .filter(|r| {
            r.title.to_lowercase().contains(&lower) || r.tags.iter().any(|t| t.starts_with(&ref_str))
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
        let _ = run_cmd(&remote_checkpoint_cmd(host)).map_err(|e| e.to_string());
    }
    let cmd = format!(
        "scp -q {}:{} {}",
        scp_host_port_flags(host),
        REMOTE_SQLITE_PATH,
        shell_escape(tmp_path.to_string_lossy().as_ref())
    );
    run_cmd(&cmd).map_err(|e| e.to_string())?;
    Ok(())
}

fn push_temp_sqlite_to_remote(host: &str, tmp_path: &Path) -> Result<(), String> {
    let host = host.trim();
    if host.is_empty() {
        return Err("host is required".to_string());
    }
    let cmd = format!(
        "scp -q {} {}:{}",
        scp_host_port_flags(host),
        shell_escape(tmp_path.to_string_lossy().as_ref()),
        REMOTE_SQLITE_PATH
    );
    run_cmd(&cmd).map_err(|e| e.to_string())?;
    Ok(())
}

fn remote_checkpoint_cmd(host: &str) -> String {
    let timeout = 8;
    let (host_no_port, port) = split_host_port(host);
    if let Some(port) = port {
        return format!(
            "ssh -o ConnectTimeout={timeout} -p {port} {host_no_port} \"sqlite3 {REMOTE_SQLITE_PATH} 'PRAGMA wal_checkpoint(TRUNCATE);'\"",
        );
    }
    format!(
        "ssh -o ConnectTimeout={timeout} {host_no_port} \"sqlite3 {REMOTE_SQLITE_PATH} 'PRAGMA wal_checkpoint(TRUNCATE);'\"",
    )
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

fn scp_host_port_flags(host: &str) -> String {
    let (h, port) = split_host_port(host);
    if let Some(port) = port {
        return format!("-P {port} {h}");
    }
    h.to_string()
}

fn shell_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}
