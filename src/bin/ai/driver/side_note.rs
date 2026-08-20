// side_note.rs — 实时 side-note 文件队列 + 内存通知
// 用户或 lead-agent 在 turn 执行过程中写入，执行中的 turn 在下一次迭代前 drain 并注入 LLM 上下文。
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::ai::history::{runtime_synthetic_user_message, Message};

const SIDE_NOTE_DIR: &str = "side_notes";
const FOREGROUND_FILE: &str = "foreground.jsonl";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideNote {
    pub from: String,              // "user" | "lead" | task_id
    pub content: String,
    pub ts: u64,
    #[serde(default)]
    pub target: Option<String>,    // None=foreground, Some(task_id)
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// 历史文件 -> session assets 目录：与 `plan_state` / `checkpoint` 共用同一 session assets 根。
// `history_file` 始终是 `SessionStore::session_history_file(id)`（即 `<sessions_root>/<id>.sqlite`），
// 因此 `parent` 即 sessions_root，`{stem}.assets` 即 `SessionStore::session_assets_dir(id)`。
// 旧实现用 `"{file_name}.assets"`（含 `.sqlite` 后缀）导致路径为 `"<id>.sqlite.assets"`，与 plan 的
// `"<id>.assets"` 分叉而丢注。新实现统一取 `file_stem`。
pub(crate) fn assets_dir_for_history(history_file: &Path) -> PathBuf {
    let parent = history_file.parent().unwrap_or_else(|| Path::new("."));
    let stem = history_file
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("default");
    parent.join(format!("{stem}.assets"))
}

pub fn side_note_dir(history_file: &Path) -> PathBuf {
    assets_dir_for_history(history_file).join(SIDE_NOTE_DIR)
}

pub fn side_note_file(history_file: &Path, target: Option<&str>) -> PathBuf {
    let dir = side_note_dir(history_file);
    let name = match target {
        None | Some("foreground") | Some("") => FOREGROUND_FILE.to_string(),
        Some(id) => format!("{}.jsonl", sanitize_task_id(id)),
    };
    dir.join(name)
}

fn sanitize_task_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

/// 写入一条 side-note（文件追加，原子性由 OS 保证单行追加）
///
/// `from` 建议取 "user" 或 lead 的 task_id / "lead"。
/// `target`: None 表示发给 foreground；Some(task_id) 表示发给某个 subagent。
pub fn push_side_note(
    history_file: &Path,
    content: &str,
    from: &str,
    target: Option<&str>,
) -> Result<PathBuf, String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err("side-note content is empty".to_string());
    }
    if trimmed.chars().count() > 8000 {
        return Err("side-note exceeds 8000 characters (single note limit)".to_string());
    }
    let file = side_note_file(history_file, target);
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create side-note dir: {e}"))?;
    }
    let note = SideNote {
        from: from.to_string(),
        content: trimmed.to_string(),
        ts: now_ts(),
        target: target.map(|s| s.to_string()),
    };
    let line = serde_json::to_string(&note).map_err(|e| format!("encode side-note: {e}"))?;
    // 追加写入，带文件锁避免并发截断
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file)
        .map_err(|e| format!("open side-note file: {e}"))?;
    writeln!(f, "{}", line).map_err(|e| format!("write side-note: {e}"))?;
    Ok(file)
}

/// 排出并清空目标队列的所有 pending notes。调用方在迭代边界调用，若有新 note 立即注入。
/// 使用 rename 原子化 drain：push 是 append，drain 侧 rename 旧文件到临时文件再读取，
/// 避免 "先读后截断" 窗口中推送的新 note 被截断丢弃。
pub fn drain_side_notes(history_file: &Path, target: Option<&str>) -> Vec<SideNote> {
    let file = side_note_file(history_file, target);
    // 原子 drain：尝试把现有队列文件 rename 到临时文件，失败则说明无 pending。
    let tmp = file.with_extension(format!("drain.{}.tmp", std::process::id()));
    let renamed = fs::rename(&file, &tmp).is_ok();
    let content = if renamed {
        match fs::read_to_string(&tmp) {
            Ok(c) => {
                let _ = fs::remove_file(&tmp);
                c
            }
            Err(_) => {
                let _ = fs::remove_file(&tmp);
                return Vec::new();
            }
        }
    } else {
        if !file.exists() {
            return Vec::new();
        }
        // 回退路径：跨盘等导致 rename 失败，降级为原子截断（仍有极小竞态，但仅在跨盘时触发）。
        match fs::read_to_string(&file) {
            Ok(c) => {
                // 尽力原子截断：先写空临时文件再 rename 覆盖
                let empty_tmp = file.with_extension(format!("empty.{}.tmp", std::process::id()));
                let _ = fs::write(&empty_tmp, "");
                let _ = fs::rename(&empty_tmp, &file);
                c
            }
            Err(_) => return Vec::new(),
        }
    };
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(note) = serde_json::from_str::<SideNote>(trimmed) {
            out.push(note);
        } else {
            // 兼容旧格式：纯文本一行即一条 note
            out.push(SideNote {
                from: "user".to_string(),
                content: trimmed.to_string(),
                ts: now_ts(),
                target: target.map(|s| s.to_string()),
            });
        }
    }
    out
}

/// 把 SideNote 转为可直接推入 LLM `messages` 的 user 消息
/// 使用 `runtime_synthetic_user_message` 以避免被误判为真实 user 轮次边界。
pub fn side_notes_to_messages(notes: Vec<SideNote>) -> Vec<Message> {
    notes
        .into_iter()
        .map(|n| {
            let header = match n.from.as_str() {
                "user" => "Live guidance (side-note) from user".to_string(),
                other => format!("Live guidance (side-note) from upstream agent `{}`", other),
            };
            let text = format!(
                "[side-note from={} ts={}] {}\n\n{}",
                n.from, n.ts, header, n.content
            );
            runtime_synthetic_user_message(serde_json::Value::String(text))
        })
        .collect()
}

/// 便捷：排出并直接得到 Messages（空则返回空 Vec）
pub fn drain_side_notes_as_messages(history_file: &Path, target: Option<&str>) -> Vec<Message> {
    let notes = drain_side_notes(history_file, target);
    if notes.is_empty() {
        return Vec::new();
    }
    side_notes_to_messages(notes)
}

/// 当前进程的 target 标识：foreground 为 None，subagent 通过 task_local 或环境变量获知 task_id
pub fn current_target_id() -> Option<String> {
    // 优先 task_local（in-process subagent 由 background_dispatch 经 SUBAGENT_TASK_ID 注入）
    if let Some(id) = crate::ai::driver::runtime_ctx::try_subagent_task_id() {
        let t = id.trim().to_string();
        if !t.is_empty() {
            return Some(t);
        }
    }
    // 回退到环境变量：兼容进程隔离或外部注入，支持两种命名
    for key in ["AIOS_SUBAGENT_TASK_ID", "SUBAGENT_TASK_ID", "AIOS_TASK_ID"] {
        if let Ok(v) = std::env::var(key) {
            let t = v.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    None
}

/// 供 turn 循环在每次模型请求前调用：排出当前 target 的 pending notes 并注入 messages。
/// 返回注入的条数，供调用方决定是否需要打印提示。
pub fn poll_and_inject(history_file: &Path, messages: &mut Vec<Message>) -> usize {
    let target = current_target_id();
    let target_ref = target.as_deref();
    let injected = drain_side_notes_as_messages(history_file, target_ref);
    let n = injected.len();
    if n > 0 {
        messages.extend(injected);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    // 项目无 tempfile 依赖，沿用 std::env::temp_dir + uuid 的既有测试模式
    fn temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("side_note_test_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn push_and_drain_roundtrip() {
        let dir = temp_dir();
        let hist = dir.join("sess.sqlite");
        push_side_note(&hist, "hello", "user", None).unwrap();
        push_side_note(&hist, "world", "lead", None).unwrap();
        let notes = drain_side_notes(&hist, None);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].content, "hello");
        // drain 后应为空
        assert!(drain_side_notes(&hist, None).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn target_isolation() {
        let dir = temp_dir();
        let hist = dir.join("sess.sqlite");
        push_side_note(&hist, "fg", "user", None).unwrap();
        push_side_note(&hist, "sub", "lead", Some("task_abc")).unwrap();
        assert_eq!(drain_side_notes(&hist, None).len(), 1);
        assert_eq!(drain_side_notes(&hist, Some("task_abc")).len(), 1);
        assert!(drain_side_notes(&hist, Some("task_abc")).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
