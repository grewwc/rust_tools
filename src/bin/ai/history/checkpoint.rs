//! 会话检查点（checkpoint）/ 回滚。
//!
//! 每个 session 的对话历史落在一个独立的 SQLite 文件里（append 操作短连接、
//! 用完即关，静止时只剩 `.sqlite` 主文件，默认 rollback-journal 模式下没有
//! 常驻的 `-wal`/`-shm`）。因此「保存检查点」= 复制该 sqlite 文件，「回滚」=
//! 把检查点文件复制回 session 文件即可，简单且健壮。
//!
//! context-history 缓存以文件 len/mtime/`PRAGMA data_version` 作为失效键，
//! 复制文件会改变这些值，从而自动让缓存失效，回滚后能读到正确历史。

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local};

use crate::ai::history::SessionStore;

#[derive(Debug, Clone)]
pub(in crate::ai) struct CheckpointInfo {
    pub(in crate::ai) name: String,
    pub(in crate::ai) modified_local: Option<DateTime<Local>>,
    pub(in crate::ai) size_bytes: u64,
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct CheckpointStore {
    /// 当前 session 的 live sqlite 文件。
    session_file: PathBuf,
    /// 该 session 的检查点目录。
    dir: PathBuf,
}

impl CheckpointStore {
    pub(in crate::ai) fn new(history_file: &Path, session_id: &str) -> Self {
        let store = SessionStore::new(history_file);
        Self {
            session_file: store.session_history_file(session_id),
            dir: store.checkpoints_dir(session_id),
        }
    }

    fn checkpoint_path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{}.sqlite", sanitize_name(name)))
    }

    /// 保存当前 session 历史为名为 `name` 的检查点。返回检查点文件路径。
    pub(in crate::ai) fn save(&self, name: &str) -> io::Result<PathBuf> {
        if !self.session_file.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "current session has no history yet; nothing to checkpoint",
            ));
        }
        fs::create_dir_all(&self.dir)?;
        let dest = self.checkpoint_path(name);
        fs::copy(&self.session_file, &dest)?;
        Ok(dest)
    }

    /// 列出该 session 的全部检查点（按修改时间从新到旧）。
    pub(in crate::ai) fn list(&self) -> io::Result<Vec<CheckpointInfo>> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("sqlite") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let metadata = match entry.metadata() {
                Ok(v) => v,
                Err(_) => continue,
            };
            out.push(CheckpointInfo {
                name: stem.to_string(),
                modified_local: metadata.modified().ok().map(DateTime::<Local>::from),
                size_bytes: metadata.len(),
            });
        }
        out.sort_by(|a, b| b.modified_local.cmp(&a.modified_local));
        Ok(out)
    }

    /// 把名为 `name` 的检查点回滚到当前 session（覆盖 live 历史）。
    pub(in crate::ai) fn rollback(&self, name: &str) -> io::Result<()> {
        let src = self.checkpoint_path(name);
        if !src.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("checkpoint '{}' not found", sanitize_name(name)),
            ));
        }
        if let Some(parent) = self.session_file.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &self.session_file)?;
        Ok(())
    }

    /// 删除名为 `name` 的检查点。返回是否确有文件被删除。
    pub(in crate::ai) fn delete(&self, name: &str) -> io::Result<bool> {
        let path = self.checkpoint_path(name);
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(err),
        }
    }
}

/// 把检查点名规整为安全的文件名（字母数字、`-`、`_`，其余转 `_`）。
fn sanitize_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() || ch == '.' || ch == '/' {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "checkpoint".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_history_file() -> PathBuf {
        let mut p = std::env::temp_dir();
        let unique = format!(
            "ai_ckpt_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        p.push(unique);
        p.push("history.sqlite");
        p
    }

    #[test]
    fn sanitize_name_handles_unsafe_chars() {
        assert_eq!(sanitize_name("my checkpoint"), "my_checkpoint");
        assert_eq!(sanitize_name("../../etc/passwd"), "etc_passwd");
        assert_eq!(sanitize_name(""), "checkpoint");
        assert_eq!(sanitize_name("v1.2"), "v1_2");
    }

    #[test]
    fn save_list_rollback_delete_roundtrip() {
        let history_file = temp_history_file();
        let session_id = "sess-abc";
        let store = CheckpointStore::new(&history_file, session_id);

        // 还没有 session 文件时 save 应失败。
        assert!(store.save("c1").is_err());

        // 造一个 live session 文件，内容 = "v1"。
        if let Some(parent) = store.session_file.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&store.session_file, b"v1").unwrap();

        // 保存检查点 c1。
        let ckpt = store.save("c1").unwrap();
        assert!(ckpt.exists());

        // 修改 live 历史为 "v2"。
        fs::write(&store.session_file, b"v2").unwrap();

        // list 能看到 c1。
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "c1");

        // 回滚到 c1，live 历史应恢复为 "v1"。
        store.rollback("c1").unwrap();
        assert_eq!(fs::read(&store.session_file).unwrap(), b"v1");

        // 回滚不存在的检查点应报错。
        assert!(store.rollback("nope").is_err());

        // 删除 c1。
        assert!(store.delete("c1").unwrap());
        assert!(!store.delete("c1").unwrap());
        assert!(store.list().unwrap().is_empty());

        // 清理。
        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }
}
