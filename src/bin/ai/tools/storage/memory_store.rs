use rustc_hash::{FxHashMap, FxHashSet};
/// Agent 记忆存储 — 底层持久化存储系统
///
/// ## 架构说明
/// `MemoryStore` 是所有知识/记忆的底层存储，被两个上层系统共用：
///
/// 1. **Knowledge (知识库)** - 面向用户的知识管理
///    - 通过 `knowledge_tools.rs` 暴露给用户
///    - 存储项目知识、决策记录、用户偏好等事实性知识
///    - 类别: `user_memory`, `project_info`, `architecture`, `decision_log`
///
/// 2. **Memory (记忆)** - Agent 内部自动学习
///    - 通过 `memory.rs` 服务层管理
///    - 存储安全规则、编码规范、自我反思等行为指导
///    - 类别: `safety_rules`, `coding_guideline`, `self_note`, `common_sense`
///
/// ## 类别区分
/// - **Guideline 类别** (用于 `build_persistent_guidelines`):
///   `safety_rules`, `user_preference`, `preference`, `coding_guideline`,
///   `best_practice`, `common_sense`, `self_note`
///
/// - **Knowledge 类别** (用于 `build_auto_recalled_knowledge`):
///   `user_memory`, `project_info`, `architecture`, `decision_log`
///   以及其他非 guideline 类别
///
/// ## 搜索机制
/// - BM25 关键词搜索 + 向量语义搜索 (通过 RAG store)
/// - 支持归档文件搜索 (可配置)
/// - 自动去重和 GC
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::memory_index::MemoryIndex;
use super::with_memory_file_lock;
use crate::ai::knowledge::indexing::{embedder, similarity};
use crate::ai::tools::service::memory::{execute_memory_dedup, execute_memory_gc};
use crate::commonw::configw;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::ffi::OsStr;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

/// 全局 (source_path, MemoryIndex) 注册表，懒加载、按 path 复用。
/// 第一次访问某条 source 路径时打开 / 重建 SQLite 索引；后续都拿到同一个
/// `Arc<MemoryIndex>`，跨调用共享 LFU 计数与 FTS 索引。
fn memory_index_for(source_path: &Path) -> Option<Arc<MemoryIndex>> {
    use std::sync::Mutex;
    static REG: OnceLock<Mutex<Vec<(PathBuf, Arc<MemoryIndex>)>>> = OnceLock::new();
    let reg = REG.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = reg.lock().ok()?;
    if let Some((_, idx)) = guard.iter().find(|(p, _)| p == source_path) {
        return Some(idx.clone());
    }
    let db_path = derive_db_path(source_path)?;
    match MemoryIndex::open_or_init(db_path.clone(), source_path.to_path_buf()) {
        Ok(idx) => {
            let arc = Arc::new(idx);
            guard.push((source_path.to_path_buf(), arc.clone()));
            Some(arc)
        }
        Err(e) => {
            // 索引不可用时降级到 BM25 路径，不阻断主存储。
            trace_memory_event(
                "memory.index.open_failed",
                "MemoryIndex unavailable; falling back to BM25",
                &[
                    ("source", source_path.display().to_string()),
                    ("db", db_path.display().to_string()),
                    ("error", e),
                ],
            );
            None
        }
    }
}

pub(crate) fn rebuild_index_for_path(path: &Path) {
    if let Some(idx) = memory_index_for(path)
        && let Err(err) = idx.rebuild_from_source()
    {
        trace_memory_event(
            "memory.index.rebuild_failed",
            "MemoryIndex rebuild failed after explicit rewrite; index may drift",
            &[("path", path.display().to_string()), ("error", err)],
        );
    }
}

/// 由 source jsonl 路径派生出对应的 sqlite 路径：
/// `agent_memory.jsonl` -> `agent_memory.db`
/// `agent_memory.subagent-xxx.jsonl` -> `agent_memory.subagent-xxx.db`
fn derive_db_path(source: &Path) -> Option<PathBuf> {
    let stem = source.file_stem()?.to_str()?;
    let parent = source.parent()?;
    Some(parent.join(format!("{stem}.db")))
}

/// 把 memory 子系统的关键运维事件镜像到 AIOS kernel trace ring，
/// 让 rotate / enforce / GC 等"会动数据"的动作在 AIOS 侧可观测。
/// 任何获取不到内核或锁失败的情况都静默返回——不能影响主流程。
pub(crate) fn trace_memory_event(location: &'static str, msg: &str, fields: &[(&str, String)]) {
    use aios_kernel::{FastMap, primitives::TraceLevel};

    let g = match crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let kernel = match g.as_ref() {
        Some(k) => k.clone(),
        None => return,
    };
    drop(g);

    let mut map: FastMap<String, String> = FastMap::default();
    for (k, v) in fields {
        map.insert((*k).to_string(), v.clone());
    }
    if let Ok(mut guard) = kernel.lock() {
        guard.trace_event(
            location.to_string(),
            TraceLevel::Info,
            None,
            map,
            Some(msg.to_string()),
        );
    }
}

/// 原子地把内容写入 `path`：先写到同目录下的 tmp 文件，fsync 后 rename。
/// 进程在中途崩溃也只会留下被 rename 替换前的旧主文件，不会写出半截 JSONL。
fn atomic_write_file(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("memory");
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = parent.join(format!(".{}.tmp.{}.{}", file_name, pid, nanos));
    {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(contents)?;
        f.flush()?;
        let _ = f.sync_all();
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        // rename 失败时尽力清理 tmp，避免遗留半成品
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AgentMemoryEntry {
    #[serde(default)]
    pub(crate) id: Option<String>,
    pub(crate) timestamp: String,
    pub(crate) category: String,
    pub(crate) note: String,
    pub(crate) tags: Vec<String>,
    pub(crate) source: Option<String>,
    /// Priority level: 0-255. Higher = more important. 255 = permanent (never delete).
    /// Default: 100 (normal priority). Low: 0-49, Normal: 50-99, High: 100-200, Permanent: 255
    #[serde(default = "default_priority")]
    pub(crate) priority: Option<u8>,
    #[serde(default)]
    pub(crate) owner_pid: Option<u64>,
    #[serde(default)]
    pub(crate) owner_pgid: Option<u64>,

    /// Optional image path (for memo entries that include screenshots/images).
    /// When set, OCR text is extracted and stored in `note` for search indexing.
    #[serde(default)]
    pub(crate) image_path: Option<String>,
}

fn default_priority() -> Option<u8> {
    Some(100)
}

impl Default for AgentMemoryEntry {
    fn default() -> Self {
        Self {
            id: None,
            timestamp: String::new(),
            category: String::new(),
            note: String::new(),
            tags: Vec::new(),
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        }
    }
}

pub(crate) struct MemoryStore {
    path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryBatchUpdateReport {
    pub(crate) deleted: usize,
    pub(crate) appended: usize,
}

impl MemoryStore {
    pub(crate) fn from_env_or_config() -> Self {
        Self {
            path: resolve_memory_file(),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn append(&self, entry: &AgentMemoryEntry) -> Result<(), String> {
        // 单条 note 字节 cap：避免 LLM 把巨型 tool 输出整段写进 long-term memory，
        // 导致下次 recall 时把 8KB+ 单条直接拉回上下文。
        const MAX_NOTE_BYTES: usize = 4_096;
        let mut entry_owned;
        let entry_ref: &AgentMemoryEntry = if entry.note.len() > MAX_NOTE_BYTES {
            entry_owned = entry.clone();
            // 按字符边界裁剪到大约 MAX_NOTE_BYTES，并加省略提示
            let mut truncated = String::with_capacity(MAX_NOTE_BYTES + 64);
            let mut used = 0usize;
            for ch in entry_owned.note.chars() {
                let extra = ch.len_utf8();
                if used + extra > MAX_NOTE_BYTES {
                    break;
                }
                truncated.push(ch);
                used += extra;
            }
            truncated.push_str("\n…[note truncated to fit memory store cap]");
            entry_owned.note = truncated;
            &entry_owned
        } else {
            entry
        };
        if should_dedup_learning_entry(entry_ref)
            && self.has_recent_duplicate(entry_ref, 200).unwrap_or(false)
        {
            return Ok(());
        }

        super::with_memory_file_lock(&self.path, || {
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create memory dir: {e}"))?;
            }
            let serialized = serde_json::to_string(entry_ref)
                .map_err(|e| format!("Failed to serialize memory entry: {e}"))?;

            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .read(true)
                .open(&self.path)
                .map_err(|e| format!("Failed to open memory file: {e}"))?;

            let needs_newline = file
                .metadata()
                .map_err(|e| format!("Failed to read memory file metadata: {e}"))?
                .len()
                > 0
                && {
                    file.seek(SeekFrom::End(-1))
                        .map_err(|e| format!("Failed to seek memory file: {e}"))?;
                    let mut last = [0u8; 1];
                    file.read_exact(&mut last)
                        .map_err(|e| format!("Failed to read memory file: {e}"))?;
                    last[0] != b'\n'
                };

            if needs_newline {
                file.write_all(b"\n")
                    .map_err(|e| format!("Failed to write memory file: {e}"))?;
            }
            file.write_all(serialized.as_bytes())
                .and_then(|_| file.write_all(b"\n"))
                .map_err(|e| format!("Failed to write memory file: {e}"))?;

            // JSONL 是 source of truth；下面的 SQLite 索引同步是 best-effort，
            // 失败只 trace 不冒泡。这样即使 rusqlite 出问题也不会阻断主存储。
            if let Some(idx) = memory_index_for(&self.path) {
                if let Err(e) = idx.upsert_entry(entry_ref) {
                    trace_memory_event(
                        "memory.index.upsert_failed",
                        "MemoryIndex upsert failed; index may drift",
                        &[
                            ("path", self.path.display().to_string()),
                            ("entry_id", entry_ref.id.clone().unwrap_or_default()),
                            ("error", e),
                        ],
                    );
                } else {
                    let _ = idx.refresh_signature();
                }
            }

            Ok(())
        })
    }

    /// 批量应用“删除 + 新增”变更并一次性原子重写 JSONL。
    /// JSONL 仍是 source of truth；SQLite 索引在成功写回后做 best-effort 全量重建。
    pub(crate) fn apply_batch_update(
        &self,
        delete_ids: &[&str],
        new_entries: &[AgentMemoryEntry],
    ) -> Result<MemoryBatchUpdateReport, String> {
        if delete_ids.is_empty() && new_entries.is_empty() {
            return Ok(MemoryBatchUpdateReport {
                deleted: 0,
                appended: 0,
            });
        }
        let id_set: FxHashSet<&str> = delete_ids.iter().copied().collect();
        super::with_memory_file_lock(&self.path, || {
            let content = match std::fs::read_to_string(&self.path) {
                Ok(content) => content,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
                Err(err) => return Err(format!("Failed to read memory file: {err}")),
            };

            let mut kept = Vec::new();
            let mut deleted = 0usize;
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) {
                    if entry.id.as_deref().map_or(false, |id| id_set.contains(id)) {
                        deleted += 1;
                        continue;
                    }
                    kept.push(entry);
                }
            }
            kept.extend(new_entries.iter().cloned());

            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("Failed to create memory dir: {err}"))?;
            }
            Self::write_all_entries(&self.path, &kept)?;

            if let Some(idx) = memory_index_for(&self.path)
                && let Err(err) = idx.rebuild_from_source()
            {
                trace_memory_event(
                    "memory.index.rebuild_failed",
                    "MemoryIndex rebuild failed after batch rewrite; index may drift",
                    &[
                        ("path", self.path.display().to_string()),
                        ("error", err),
                    ],
                );
            }

            Ok(MemoryBatchUpdateReport {
                deleted,
                appended: new_entries.len(),
            })
        })
    }

    fn has_recent_duplicate(
        &self,
        target: &AgentMemoryEntry,
        recent_limit: usize,
    ) -> Result<bool, String> {
        let target_norm = normalize_learning_note(&target.note);
        let target_source = target.source.as_deref().unwrap_or("");
        let recent = self.recent(recent_limit)?;
        Ok(recent.into_iter().any(|entry| {
            if entry.category != target.category {
                return false;
            }
            if entry.source.as_deref().unwrap_or("") != target_source {
                return false;
            }
            normalize_learning_note(&entry.note) == target_norm
        }))
    }

    fn memory_files_to_scan(&self) -> Result<Vec<PathBuf>, String> {
        let cfg = configw::get_all_config();
        let search_archives = cfg
            .get_opt("ai.memory.search_archives.enable")
            .unwrap_or_else(|| "false".to_string())
            .trim()
            .eq_ignore_ascii_case("true");
        let keep_last_archives = cfg
            .get_opt("ai.memory.search_archives.keep_last")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);

        let mut files: Vec<PathBuf> = Vec::new();
        if search_archives {
            if let Some(parent) = self.path.parent() {
                let base = self
                    .path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .unwrap_or("")
                    .to_string();
                let mut archives = Vec::new();
                for entry in fs::read_dir(parent).map_err(|e| format!("{}", e))? {
                    let entry = entry.map_err(|e| format!("{}", e))?;
                    let file_name = entry.file_name().to_str().unwrap_or("").to_string();
                    if !file_name.starts_with(&(base.clone() + ".")) {
                        continue;
                    }
                    let meta = entry.metadata().map_err(|e| format!("{}", e))?;
                    if !meta.is_file() {
                        continue;
                    }
                    let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    archives.push((entry.path(), modified));
                }
                archives.sort_by_key(|(_, modified)| *modified);
                let take_from = archives.len().saturating_sub(keep_last_archives);
                for (path, _) in archives.into_iter().skip(take_from) {
                    files.push(path);
                }
            }
        }
        files.push(self.path.clone());
        Ok(files)
    }

    pub(crate) fn entries_by_category(
        &self,
        category: &str,
        limit: usize,
    ) -> Result<Vec<AgentMemoryEntry>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut window: VecDeque<AgentMemoryEntry> = VecDeque::new();
        for path in self.memory_files_to_scan()? {
            if !path.exists() {
                continue;
            }
            let file =
                fs::File::open(&path).map_err(|e| format!("Failed to read memory file: {e}"))?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                    continue;
                };
                if entry.category != category {
                    continue;
                }
                window.push_back(entry);
                if window.len() > limit {
                    window.pop_front();
                }
            }
        }

        let mut entries: Vec<AgentMemoryEntry> = window.into_iter().collect();
        entries.reverse();
        Ok(entries)
    }

    pub(crate) fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(AgentMemoryEntry, f64)>, String> {
        let query_lc = query.to_lowercase();

        // Fast-path: 先用 SQLite FTS5 拿候选 id 集合（O(log N) MATCH），
        // 再回到 JSONL 把候选条目精确加载，跑现有 BM25 + embedding 计分。
        // 这样把 search 从 "全文件扫描 + 全文件 tokenize" 降为 "命中行 + tokenize"，
        // 输出格式 / 分数权重 / 排序逻辑全部不变。
        // FTS 不可用 / 候选过少时回到原扫描路径，行为完全等价。
        let fts_candidate_cap = limit.saturating_mul(20).max(60).min(400);
        let fts_ids: Option<std::collections::HashSet<String>> = memory_index_for(&self.path)
            .and_then(|idx| match idx.search_ids(&query_lc, fts_candidate_cap) {
                Ok(v) if v.len() >= limit => Some(v.into_iter().collect()),
                _ => None,
            });

        let mut docs: Vec<(AgentMemoryEntry, String, Vec<String>)> = Vec::new();
        for p in self.memory_files_to_scan()? {
            if !p.exists() {
                continue;
            }
            let file =
                fs::File::open(&p).map_err(|e| format!("Failed to read memory file: {e}"))?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                    continue;
                };
                if let Some(ids) = &fts_ids {
                    // fast-path：只保留 FTS 命中的条目
                    if let Some(id) = entry.id.as_deref() {
                        if !ids.contains(id) {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                let mut full = String::new();
                full.push_str(&entry.category);
                full.push(' ');
                full.push_str(&entry.note);
                if let Some(s) = &entry.source {
                    full.push(' ');
                    full.push_str(s);
                }
                if !entry.tags.is_empty() {
                    full.push(' ');
                    full.push_str(&entry.tags.join(" "));
                }
                let tokens = similarity::expand_tokens(&similarity::tokenize(&full.to_lowercase()));
                docs.push((entry, full, tokens));
            }
        }
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let nq_tokens = similarity::expand_tokens(&similarity::tokenize(&query_lc));
        let mut df: FxHashMap<String, usize> = FxHashMap::default();
        let mut avgdl = 0.0f64;
        for (_, _, toks) in &docs {
            avgdl += toks.len() as f64;
            let mut set: FxHashSet<&str> = FxHashSet::default();
            for t in toks {
                if set.insert(t.as_str()) {
                    *df.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }
        avgdl /= docs.len() as f64;
        let n_docs = docs.len() as f64;
        let k1 = 1.2f64;
        let b = 0.75f64;
        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(docs.len());
        let mut bm25_vals: Vec<f64> = Vec::with_capacity(docs.len());
        for (idx, (entry, _full, toks)) in docs.iter().enumerate() {
            let mut tf: FxHashMap<&str, usize> = FxHashMap::default();
            for t in toks {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
            let mut bm25 = 0.0f64;
            let dl = toks.len() as f64;
            let mut seenq: FxHashSet<&str> = FxHashSet::default();
            for qt in &nq_tokens {
                if !seenq.insert(qt.as_str()) {
                    continue;
                }
                let dfv = *df.get(qt.as_str()).unwrap_or(&0) as f64;
                if dfv <= 0.0 {
                    continue;
                }
                let idf = ((n_docs - dfv + 0.5) / (dfv + 0.5) + 1.0).ln();
                let tfv = *tf.get(qt.as_str()).unwrap_or(&0) as f64;
                if tfv <= 0.0 {
                    continue;
                }
                let denom = tfv + k1 * (1.0 - b + b * (dl / avgdl.max(1e-6)));
                bm25 += idf * (tfv * (k1 + 1.0)) / denom;
            }
            bm25_vals.push(bm25);
            let sim = compute_similarity(entry, &query_lc) as f64;
            let pre = 0.5 * sim + 0.5 * 0.0;
            scored.push((pre, idx));
        }
        let max_bm25 = bm25_vals.iter().cloned().fold(0.0f64, f64::max);
        for i in 0..scored.len() {
            let bm = if max_bm25 > 0.0 {
                bm25_vals[i] / max_bm25
            } else {
                0.0
            };
            let s = 0.55 * scored[i].0 + 0.45 * bm;
            scored[i].0 = s;
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let cap = limit.saturating_mul(10).min(200).max(limit);
        let mut top_idx: Vec<(f64, usize)> =
            scored.iter().take(cap).map(|(s, i)| (*s, *i)).collect();
        let qv = embedder::embed_text(&query_lc);
        if let Some(qv) = qv {
            let texts: Vec<String> = top_idx.iter().map(|&(_, i)| docs[i].1.clone()).collect();
            let batch = embedder::embed_texts(&texts);
            let mut rescored: Vec<(f64, usize)> = Vec::with_capacity(top_idx.len());
            for (idx, &(_s, i)) in top_idx.iter().enumerate() {
                let emb = batch
                    .as_ref()
                    .and_then(|v| v.get(idx))
                    .map(|v| similarity::cosine_similarity(&qv, v))
                    .unwrap_or(0.0);
                let base = _s;
                let final_s = 0.85 * base + 0.15 * emb as f64;
                rescored.push((final_s, i));
            }
            rescored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            top_idx = rescored.into_iter().take(limit).collect();
        } else {
            top_idx.truncate(limit);
        }
        let mut out = Vec::with_capacity(top_idx.len());
        for (s, i) in top_idx {
            out.push((docs[i].0.clone(), s));
        }
        // 给被命中的条目计 LFU；失败只 trace。注意只对 top-N（已截到 limit）计数，
        // 而不是 cap=200 的中间集合，避免 score 很低的边缘条目刷出 hits。
        if let Some(idx) = memory_index_for(&self.path) {
            let ids: Vec<String> = out.iter().filter_map(|(e, _)| e.id.clone()).collect();
            if !ids.is_empty() {
                if let Err(e) = idx.record_hits(&ids) {
                    trace_memory_event(
                        "memory.index.hits_failed",
                        "MemoryIndex record_hits failed",
                        &[("path", self.path.display().to_string()), ("error", e)],
                    );
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn recent(&self, limit: usize) -> Result<Vec<AgentMemoryEntry>, String> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file =
            fs::File::open(&self.path).map_err(|e| format!("Failed to read memory file: {e}"))?;
        let reader = BufReader::new(file);

        let mut window: VecDeque<AgentMemoryEntry> = VecDeque::with_capacity(limit + 1);
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                continue;
            };
            window.push_back(entry);
            if window.len() > limit {
                window.pop_front();
            }
        }

        let mut entries: Vec<AgentMemoryEntry> = window.into_iter().collect();
        entries.reverse();
        Ok(entries)
    }
}

fn should_dedup_learning_entry(entry: &AgentMemoryEntry) -> bool {
    matches!(
        entry.category.as_str(),
        "self_note"
            | "project_memory"
            | "coding_guideline"
            | "code_discovery"
            | "common_sense"
            | "best_practice"
            | "safety_rules"
    )
}

fn normalize_learning_note(note: &str) -> String {
    note.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(test)]
impl MemoryStore {
    pub(crate) fn for_tests_with_path(path: PathBuf) -> Self {
        Self { path }
    }
}

/// 直接基于一个明确路径构建 store，绕过 task_local override / env / config。
/// 用于父任务在 sub-agent finalize 后把白名单条目写回主 memory 文件。
pub(crate) fn store_for_path(path: PathBuf) -> MemoryStore {
    MemoryStore { path }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    #[test]
    fn test_search_recall_ngram() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e1 = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            category: "log".to_string(),
            note: "parsing login error occurred".to_string(),
            tags: vec!["auth".to_string()],
            source: Some("svc".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let e2 = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-02T00:00:00Z".to_string(),
            category: "info".to_string(),
            note: "user profile updated".to_string(),
            tags: vec!["user".to_string()],
            source: Some("svc".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        store.append(&e1).unwrap();
        store.append(&e2).unwrap();
        let out = store.search("parse login", 5).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|(x, _)| x.note.contains("parsing login")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_search_recall_synonym_login() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_syn_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-03T00:00:00Z".to_string(),
            category: "auth".to_string(),
            note: "user login failed due to authentication error".to_string(),
            tags: vec!["login".to_string()],
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        store.append(&e).unwrap();
        let out = store.search("signin failure", 3).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|(x, _)| x.note.contains("login failed")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_search_recall_chinese_login_variants() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_cn_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-04T00:00:00Z".to_string(),
            category: "auth".to_string(),
            note: "登录失败，密码错误".to_string(),
            tags: vec!["登录".to_string()],
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        store.append(&e).unwrap();
        let out = store.search("登陆失败", 3).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|(x, _)| x.note.contains("登录失败")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn learning_entries_deduplicate_recent_exact_writes() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_dedup_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());

        let entry = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-05T00:00:00Z".to_string(),
            category: "self_note".to_string(),
            note: "Do: verify before write".to_string(),
            tags: vec!["agent".to_string()],
            source: Some("session:test".to_string()),
            priority: Some(120),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };

        store.append(&entry).unwrap();
        store.append(&entry).unwrap();

        let recent = store.recent(10).unwrap();
        assert_eq!(
            recent
                .iter()
                .filter(|e| e.category == "self_note" && e.note == "Do: verify before write")
                .count(),
            1
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_batch_update_rewrites_delete_and_append_in_one_pass() {
        let path = std::env::temp_dir().join(format!(
            "rt_mem_batch_update_{}.jsonl",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let entry_with_id = |id: &str, note: &str, ts: &str| AgentMemoryEntry {
            id: Some(id.to_string()),
            timestamp: ts.to_string(),
            category: "user_memory".to_string(),
            note: note.to_string(),
            tags: Vec::new(),
            source: None,
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let write_lines = |entries: &[AgentMemoryEntry]| {
            let mut buf = String::new();
            for entry in entries {
                buf.push_str(&serde_json::to_string(entry).unwrap());
                buf.push('\n');
            }
            std::fs::write(&path, buf).unwrap();
        };
        let read_entries = || -> Vec<AgentMemoryEntry> {
            std::fs::read_to_string(&path)
                .unwrap_or_default()
                .lines()
                .filter_map(|line| serde_json::from_str::<AgentMemoryEntry>(line.trim()).ok())
                .collect()
        };
        write_lines(&[
            entry_with_id("mem_1", "keep me", "2025-01-01T00:00:00Z"),
            entry_with_id("mem_2", "drop me", "2025-01-01T00:00:01Z"),
            entry_with_id("mem_3", "merge me", "2025-01-01T00:00:02Z"),
        ]);

        let store = MemoryStore::for_tests_with_path(path.clone());
        let merged = entry_with_id("mem_merged", "merged note", "2025-01-02T00:00:00Z");
        let report = store
            .apply_batch_update(&["mem_2", "mem_3"], &[merged.clone()])
            .unwrap();

        assert_eq!(
            report,
            MemoryBatchUpdateReport {
                deleted: 2,
                appended: 1
            }
        );

        let kept = read_entries();
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|entry| entry.id.as_deref() == Some("mem_1")));
        assert!(kept.iter().any(|entry| entry.id.as_deref() == Some("mem_merged")));
        assert!(!kept.iter().any(|entry| entry.id.as_deref() == Some("mem_2")));
        assert!(!kept.iter().any(|entry| entry.id.as_deref() == Some("mem_3")));

        let _ = std::fs::remove_file(&path);
    }
}

fn compute_similarity(entry: &AgentMemoryEntry, query_lc: &str) -> f64 {
    let base_contains = if entry.note.to_lowercase().contains(query_lc)
        || entry.category.to_lowercase().contains(query_lc)
        || entry
            .tags
            .iter()
            .any(|t| t.to_lowercase().contains(query_lc))
        || entry
            .source
            .as_ref()
            .is_some_and(|s| s.to_lowercase().contains(query_lc))
    {
        0.35
    } else {
        0.0
    };
    let mut full = String::new();
    full.push_str(&entry.category);
    full.push(' ');
    full.push_str(&entry.note);
    if let Some(s) = &entry.source {
        full.push(' ');
        full.push_str(s);
    }
    if !entry.tags.is_empty() {
        full.push(' ');
        full.push_str(&entry.tags.join(" "));
    }
    let nq = similarity::norm_text(query_lc);
    let ne = similarity::norm_text(&full);
    let d = similarity::dice_coefficient(&similarity::bigrams(&nq), &similarity::bigrams(&ne));
    let tq = similarity::expand_tokens(&similarity::tokenize(query_lc));
    let te = similarity::expand_tokens(&similarity::tokenize(&full.to_lowercase()));
    let j = similarity::jaccard(&tq, &te);
    let co = similarity::char_overlap(&nq, &ne);
    let s = 0.5 * d + 0.3 * j + 0.15 * co + base_contains;
    if s < 0.0 { 0.0 } else { s.min(1.0) }
}

fn resolve_memory_file() -> PathBuf {
    if let Some(path) = crate::ai::driver::runtime_ctx::override_memory_path() {
        return path;
    }
    if let Ok(path) = std::env::var("RUST_TOOLS_MEMORY_FILE") {
        let path = path.trim();
        if !path.is_empty() {
            return PathBuf::from(crate::commonw::utils::expanduser(path).as_ref());
        }
    }
    let cfg = crate::commonw::configw::get_all_config();
    let raw = cfg
        .get_opt("ai.memory.file")
        .unwrap_or_else(|| "~/.config/rust_tools/agent_memory.jsonl".to_string());
    PathBuf::from(crate::commonw::utils::expanduser(&raw).as_ref())
}

impl MemoryStore {
    pub(crate) fn rotate_if_exceeds(&self, max_bytes: u64) -> Result<bool, String> {
        let path = self.path().to_path_buf();
        with_memory_file_lock(&path, || {
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => return Ok(false),
            };
            if meta.len() <= max_bytes {
                return Ok(false);
            }

            // 修复点 P0-2：原实现直接 `rename` + `File::create`，category 白名单内的
            // 永久条目（safety_rules / reflection self_note / coding_guideline /
            // user_preference / project_memory ...）也会被一起冻进归档，之后默认
            // 不参与召回——等于把"长期记忆"的核心规则丢掉。
            //
            // 现在先把所有条目读出来，把白名单条目留在新主文件，其余写入归档：
            //   - 新主文件 = 原内容 ∩ {is_permanent_memory}
            //   - 归档文件 = 原内容（保持不变，与旧实现一致）
            // 这样无论召回器是否启用 search_archives，长期资产都不会丢。
            let content = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read memory file before rotate: {}", e))?;

            let entries: Vec<AgentMemoryEntry> = content
                .lines()
                .filter_map(|line| {
                    let line = line.trim();
                    if line.is_empty() {
                        return None;
                    }
                    serde_json::from_str::<AgentMemoryEntry>(line).ok()
                })
                .collect();
            let permanent: Vec<&AgentMemoryEntry> = entries
                .iter()
                .filter(|e| crate::ai::tools::service::memory::is_permanent_memory(e))
                .collect();
            let preserved_total = permanent.len();
            let archived_total = entries.len();

            let ts = chrono::Local::now().format("%Y%m%d%H%M%S").to_string();
            let mut new_name = path.clone();
            let ext = new_name
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("jsonl")
                .to_string();
            new_name.set_extension(format!("{ext}.{}", ts));
            std::fs::rename(&path, &new_name)
                .map_err(|e| format!("Failed to rotate file: {}", e))?;

            // 重建主文件：写回所有 priority=255 的条目（保留原 timestamp 顺序）
            let mut head = String::new();
            for entry in &permanent {
                if let Ok(s) = serde_json::to_string(*entry) {
                    head.push_str(&s);
                    head.push('\n');
                }
            }
            atomic_write_file(&path, head.as_bytes()).map_err(|e| {
                format!(
                    "Failed to recreate memory file with permanent entries after rotate: {}",
                    e
                )
            })?;

            trace_memory_event(
                "memory.rotate",
                "memory file rotated; permanent entries preserved in head",
                &[
                    ("path", path.display().to_string()),
                    ("archive", new_name.display().to_string()),
                    ("archived_total", archived_total.to_string()),
                    ("preserved_permanent", preserved_total.to_string()),
                    ("max_bytes", max_bytes.to_string()),
                    ("file_size", meta.len().to_string()),
                ],
            );
            // rotate 把绝大多数条目搬到归档，索引内容已经严重失效。
            // 这里直接触发一次 rebuild —— 现在主文件只剩永久条目，rebuild 很轻。
            if let Some(idx) = memory_index_for(&path) {
                let _ = idx.rebuild_from_source();
            }
            Ok(true)
        })
    }

    pub(crate) fn maintain_after_append(&self) {
        let cfg = configw::get_all_config();
        let max_bytes = cfg
            .get_opt("ai.memory.auto_rotate.max_bytes")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(8 * 1024 * 1024);
        let gc_days = cfg
            .get_opt("ai.memory.auto_gc.days")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(30);
        let min_keep = cfg
            .get_opt("ai.memory.auto_gc.min_keep")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(200);
        let prob = cfg
            .get_opt("ai.memory.auto_maintain.probability")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.05);
        let max_entries = cfg
            .get_opt("ai.memory.quota.max_entries")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10000);
        let rotated = self.rotate_if_exceeds(max_bytes).unwrap_or(false);
        let _ = if rotated {
            self.cleanup_archives_auto()
        } else {
            Ok(())
        };
        let _ = self.enforce_max_entries(max_entries, min_keep);
        let roll = rand::random::<f64>();
        if roll < prob {
            let _ = execute_memory_dedup(&json!({}));
            let _ = execute_memory_gc(&json!({ "max_days": gc_days, "min_keep": min_keep }));
            let _ = self.cleanup_archives_auto();
        }
    }

    fn enforce_max_entries(&self, max_entries: usize, min_keep: usize) -> Result<(), String> {
        super::with_memory_file_lock(&self.path, || {
            let content = std::fs::read_to_string(&self.path)
                .map_err(|e| format!("Failed to read memory file: {}", e))?;

            let mut entries: Vec<AgentMemoryEntry> = content
                .lines()
                .filter_map(|line| {
                    let line = line.trim();
                    if line.is_empty() {
                        return None;
                    }
                    serde_json::from_str::<AgentMemoryEntry>(line).ok()
                })
                .collect();

            let original_total = entries.len();
            if original_total <= max_entries {
                return Ok(());
            }

            // 排序键：永久条目（白名单：safety/preference/coding_guideline/
            // project_memory/...）永远靠后，其余按 priority 升序、ts 升序。
            // 这样删除的时候从头开始砍最低优先级、最旧的条目。
            entries.sort_by(|a, b| {
                let perm_a = crate::ai::tools::service::memory::is_permanent_memory(a);
                let perm_b = crate::ai::tools::service::memory::is_permanent_memory(b);
                if perm_a && !perm_b {
                    return std::cmp::Ordering::Greater;
                }
                if perm_b && !perm_a {
                    return std::cmp::Ordering::Less;
                }
                let pa = a.priority.unwrap_or(100);
                let pb = b.priority.unwrap_or(100);
                pa.cmp(&pb).then_with(|| a.timestamp.cmp(&b.timestamp))
            });

            // 修复点 P0-1：原实现用 `while … { remove(i); if … { remove(i); } }`
            // 在同一个下标位置删了两次——第二次删的其实是已经"前移"上来的下一条，
            // 而且没有跳过永久条目检查，可能误伤白名单条目。这里改成单删 +
            // i 保持不动（remove(i) 后下一条自动落到 i），同时白名单条目跳过。
            //
            // target 字段保留只是为了"达到配额尽快收手"，不再用来触发第二次删除。
            let target = max_entries.saturating_sub(min_keep);
            let mut removed = 0usize;
            let mut skipped_permanent = 0usize;
            let mut i = 0usize;
            while i < entries.len() && entries.len() > max_entries {
                if crate::ai::tools::service::memory::is_permanent_memory(&entries[i]) {
                    // 永久条目永远跳过，不能 remove。
                    skipped_permanent += 1;
                    i += 1;
                    continue;
                }
                entries.remove(i);
                removed += 1;
                if target > 0 && removed >= target && entries.len() <= max_entries {
                    break;
                }
            }

            let mut output = String::new();
            for entry in &entries {
                if let Ok(s) = serde_json::to_string(entry) {
                    output.push_str(&s);
                    output.push('\n');
                }
            }

            // 修复点 P1-1：原实现 `fs::write(&path, output)` 是 truncate-then-write，
            // 中途崩溃会留下不完整的主文件。改成 tmp+rename，文件系统层面原子。
            atomic_write_file(&self.path, output.as_bytes()).map_err(|e| {
                format!("Failed to write memory file after quota enforcement: {}", e)
            })?;

            // 文件已被整体重写，触发索引重建以保持一致；rebuild 内部包事务，失败只 trace。
            if let Some(idx) = memory_index_for(&self.path) {
                let _ = idx.rebuild_from_source();
            }

            trace_memory_event(
                "memory.enforce_max_entries",
                "memory quota enforced",
                &[
                    ("path", self.path.display().to_string()),
                    ("before", original_total.to_string()),
                    ("after", entries.len().to_string()),
                    ("removed", removed.to_string()),
                    ("skipped_permanent", skipped_permanent.to_string()),
                    ("max_entries", max_entries.to_string()),
                    ("min_keep", min_keep.to_string()),
                ],
            );
            Ok(())
        })
    }

    /// 批量删除多条记忆（原子操作：一次性读 → 过滤 → 写回）。
    /// 返回实际删除的条数。
    pub(crate) fn delete_by_ids(&self, ids: &[&str]) -> Result<usize, String> {
        if ids.is_empty() {
            return Ok(0);
        }
        self.apply_batch_update(ids, &[])
            .map(|report| report.deleted)
    }

    /// 根据 id 删除条目（返回被删除的条目）
    pub(crate) fn delete_by_id(&self, id: &str) -> Result<Option<AgentMemoryEntry>, String> {
        super::with_memory_file_lock(&self.path, || {
            let content = std::fs::read_to_string(&self.path)
                .map_err(|e| format!("Failed to read memory file: {}", e))?;

            let mut entries: Vec<AgentMemoryEntry> = Vec::new();
            let mut deleted_entry: Option<AgentMemoryEntry> = None;

            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) {
                    let entry_id = entry.id.as_deref().unwrap_or("");
                    if entry_id == id {
                        deleted_entry = Some(entry);
                        continue;
                    }
                    entries.push(entry);
                }
            }

            if deleted_entry.is_none() {
                return Ok(None);
            }

            let mut output = String::new();
            for entry in &entries {
                if let Ok(s) = serde_json::to_string(entry) {
                    output.push_str(&s);
                    output.push('\n');
                }
            }

            // 与 enforce_max_entries 保持一致：tmp + rename 原子写，避免崩溃后留下不完整主文件。
            atomic_write_file(&self.path, output.as_bytes())
                .map_err(|e| format!("Failed to write memory file: {}", e))?;

            Ok(deleted_entry)
        })
    }

    /// 批量追加多条记忆；内部复用 `apply_batch_update([], entries)`，
    /// 通过一次原子重写避免“先追加一半”这类中间态。
    pub(crate) fn append_batch(&self, entries: &[AgentMemoryEntry]) -> Result<usize, String> {
        if entries.is_empty() {
            return Ok(0);
        }
        self.apply_batch_update(&[], entries)
            .map(|report| report.appended)
    }

    fn cleanup_archives_auto(&self) -> Result<(), String> {
        let cfg = configw::get_all_config();
        let retain_days = cfg
            .get_opt("ai.memory.archives.retain_days")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(60);
        let keep_last = cfg
            .get_opt("ai.memory.archives.keep_last")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10);
        let max_total = cfg
            .get_opt("ai.memory.archives.max_bytes")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(64 * 1024 * 1024);
        self.cleanup_archives(retain_days, keep_last, max_total)
    }

    pub(crate) fn cleanup_archives(
        &self,
        retain_days: i64,
        keep_last: usize,
        max_total_bytes: u64,
    ) -> Result<(), String> {
        let path = self.path().to_path_buf();
        let parent = match path.parent() {
            Some(p) => p.to_path_buf(),
            None => return Ok(()),
        };
        let base = match path.file_name().and_then(OsStr::to_str) {
            Some(s) => s.to_string(),
            None => return Ok(()),
        };

        let mut archives = Vec::new();
        for entry in std::fs::read_dir(&parent).map_err(|e| format!("{}", e))? {
            let entry = entry.map_err(|e| format!("{}", e))?;
            let file_name = entry.file_name().to_str().unwrap_or("").to_string();
            if !file_name.starts_with(&(base.clone() + ".")) {
                continue;
            }
            let meta = entry.metadata().map_err(|e| format!("{}", e))?;
            if !meta.is_file() {
                continue;
            }
            let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let size = meta.len();
            archives.push((entry.path(), modified, size));
        }

        if archives.is_empty() {
            return Ok(());
        }

        archives.sort_by_key(|(_, modified, _)| *modified);

        // Age-based cleanup
        if retain_days > 0 {
            let cutoff = SystemTime::now()
                .checked_sub(Duration::from_secs((retain_days as u64) * 86400))
                .unwrap_or(SystemTime::UNIX_EPOCH);
            for (p, m, _) in archives.clone() {
                if m < cutoff {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }

        // Refresh list after potential deletions
        let mut archives2 = Vec::new();
        for (p, m, s) in archives.into_iter() {
            if p.exists() {
                archives2.push((p, m, s));
            }
        }
        if archives2.is_empty() {
            return Ok(());
        }
        archives2.sort_by_key(|(_, modified, _)| *modified);

        // Keep last N
        if archives2.len() > keep_last {
            let to_delete = archives2.len() - keep_last;
            for i in 0..to_delete {
                let (p, _, _) = &archives2[i];
                let _ = std::fs::remove_file(p);
            }
        }

        // Size cap
        let mut archives3: Vec<(std::path::PathBuf, SystemTime, u64)> = archives2
            .into_iter()
            .filter(|(p, _, _)| p.exists())
            .collect();
        archives3.sort_by_key(|(_, m, _)| *m);
        let mut total: u64 = archives3.iter().map(|(_, _, s)| *s).sum();
        let mut idx = 0usize;
        while total > max_total_bytes && idx < archives3.len() {
            let (p, _, s) = &archives3[idx];
            if std::fs::remove_file(p).is_ok() {
                total = total.saturating_sub(*s);
            }
            idx += 1;
        }
        Ok(())
    }
}

/// 记忆重要性评分 - 用于主动学习和自动遗忘
#[derive(Debug, Clone)]
pub struct MemoryImportance {
    /// 被引用次数
    pub frequency: u32,
    /// 时间衰减因子 (0.0 - 1.0, 越近越高)
    pub recency: f64,
    /// 适用范围广度 (0.0 - 1.0)
    pub generality: f64,
    /// 用户是否确认过
    pub user_validated: bool,
}

impl MemoryImportance {
    pub fn new() -> Self {
        Self {
            frequency: 0,
            recency: 1.0,
            generality: 0.5,
            user_validated: false,
        }
    }

    /// 计算综合重要性分数 (0.0 - 1.0)
    pub fn score(&self) -> f64 {
        let freq_score = (self.frequency as f64).min(10.0) / 10.0; // 0-1
        let recency_score = self.recency.clamp(0.0, 1.0);
        let generality_score = self.generality.clamp(0.0, 1.0);
        let validation_bonus = if self.user_validated { 0.2 } else { 0.0 };

        // 权重：频率 30%, 时效性 30%, 通用性 20%, 用户确认 20%
        (freq_score * 0.3 + recency_score * 0.3 + generality_score * 0.2 + validation_bonus)
            .min(1.0)
    }

    /// 增加引用次数
    pub fn increment_frequency(&mut self) {
        self.frequency += 1;
    }

    /// 更新时间衰减
    pub fn update_recency(&mut self, created_at: &str) {
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(created_at) {
            let now = chrono::Utc::now();
            let age_days = (now - dt.with_timezone(&chrono::Utc)).num_seconds() as f64 / 86400.0;
            // 指数衰减：半衰期 30 天
            self.recency = (-age_days * std::f64::consts::LN_2 / 30.0).exp();
        }
    }

    /// 评估通用性（基于类别和标签）
    pub fn evaluate_generality(&mut self, category: &str, tags: &[String]) {
        let general_categories = [
            "common_sense",
            "best_practice",
            "coding_guideline",
            "safety_rules",
        ];

        let general_tags = ["general", "universal", "fundamental", "core"];

        let category_score = if general_categories.contains(&category.as_ref()) {
            1.0
        } else {
            0.5
        };

        let tag_score = tags
            .iter()
            .filter(|t| general_tags.contains(&t.as_str()))
            .count() as f64
            / tags.len().max(1) as f64;

        self.generality = (category_score * 0.7 + tag_score * 0.3).clamp(0.0, 1.0);
    }

    /// 标记为用户确认
    pub fn mark_user_validated(&mut self) {
        self.user_validated = true;
    }

    /// 判断是否应该被遗忘（低价值记忆）
    pub fn should_prune(&self, min_score: f64) -> bool {
        self.score() < min_score
    }
}

impl Default for MemoryImportance {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStore {
    /// 修剪低价值记忆
    ///
    /// 删除满足以下条件的记忆：
    /// - 重要性分数 < min_score
    /// - 优先级 < 200（非高优先级）
    /// - 90 天未使用
    pub fn prune_low_value_memories(
        &self,
        min_score: f64,
        max_age_days: i64,
    ) -> Result<usize, String> {
        let entries = self.all()?;
        let mut to_remove = Vec::new();
        let now = chrono::Utc::now();

        for entry in entries {
            // 永久记忆和高优先级记忆不删除
            if entry.priority.unwrap_or(100) >= 200 {
                continue;
            }

            // 计算重要性
            let mut importance = MemoryImportance::new();
            importance.update_recency(&entry.timestamp);
            importance.evaluate_generality(&entry.category, &entry.tags);

            // 检查年龄
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&entry.timestamp) {
                let age_days = (now - dt.with_timezone(&chrono::Utc)).num_days();
                if age_days > max_age_days && importance.should_prune(min_score) {
                    to_remove.push(entry.id.clone());
                }
            }
        }

        let removed_count = to_remove.len();

        // 执行删除
        for id in to_remove {
            if let Some(id) = id {
                self.remove_by_id(&id)?;
            }
        }

        Ok(removed_count)
    }

    /// 根据 ID 删除记忆
    fn remove_by_id(&self, id: &str) -> Result<(), String> {
        let entries = self.all()?;
        let new_entries: Vec<AgentMemoryEntry> = entries
            .into_iter()
            .filter(|e| e.id.as_deref() != Some(id))
            .collect();

        // 重写文件
        super::with_memory_file_lock(&self.path, || {
            let mut file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.path)
                .map_err(|e| format!("Failed to open memory file: {e}"))?;

            for entry in new_entries {
                let serialized = serde_json::to_string(&entry)
                    .map_err(|e| format!("Failed to serialize memory entry: {e}"))?;
                writeln!(file, "{}", serialized)
                    .map_err(|e| format!("Failed to write memory entry: {e}"))?;
            }

            // 同步删除 SQLite 索引；失败只 trace，下次 search 启动时漂移检测会重建。
            if let Some(idx) = memory_index_for(&self.path) {
                let _ = idx.delete_id(id);
                let _ = idx.refresh_signature();
            }

            Ok(())
        })
    }

    /// 重写整个 JSONL 文件（原子写：tmp → rename）。
    fn write_all_entries(path: &Path, entries: &[AgentMemoryEntry]) -> Result<(), String> {
        let mut output = String::new();
        for entry in entries {
            if let Ok(s) = serde_json::to_string(entry) {
                output.push_str(&s);
                output.push('\n');
            }
        }
        atomic_write_file(path, output.as_bytes())
            .map_err(|e| format!("Failed to write memory file: {}", e))
    }

    /// 获取所有记忆
    pub fn all(&self) -> Result<Vec<AgentMemoryEntry>, String> {
        self.search("", usize::MAX)
            .map(|results| results.into_iter().map(|(e, _score)| e).collect())
    }

    /// 记录记忆被使用（增加引用次数）。
    /// 真正落地点是 SQLite 索引的 `hits` 列；JSONL 没法原地 update。
    /// 索引不可用时静默返回 Ok，与历史行为兼容。
    pub fn record_usage(&self, entry_id: &str) -> Result<(), String> {
        if entry_id.is_empty() {
            return Ok(());
        }
        if let Some(idx) = memory_index_for(&self.path) {
            let _ = idx.record_hits(&[entry_id.to_string()]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod importance_tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_memory_importance_score() {
        let mut importance = MemoryImportance::new();
        assert_eq!(importance.frequency, 0);
        assert_eq!(importance.recency, 1.0);
        assert_eq!(importance.generality, 0.5);
        assert!(!importance.user_validated);

        // 初始分数
        let initial_score = importance.score();
        assert!(initial_score > 0.0 && initial_score < 1.0);

        // 增加引用
        for _ in 0..10 {
            importance.increment_frequency();
        }
        assert_eq!(importance.frequency, 10);

        // 用户确认
        importance.mark_user_validated();
        assert!(importance.user_validated);

        // 分数应该提高
        let new_score = importance.score();
        assert!(new_score > initial_score);
    }

    #[test]
    fn test_memory_importance_recency_decay() {
        let mut importance = MemoryImportance::new();

        // 30 天前的记忆
        let old_timestamp = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();
        importance.update_recency(&old_timestamp);

        // 时效性应该衰减到约 0.5（半衰期）
        assert!(importance.recency > 0.4 && importance.recency < 0.6);

        // 90 天前的记忆
        let very_old_timestamp = (Utc::now() - chrono::Duration::days(90)).to_rfc3339();
        importance.update_recency(&very_old_timestamp);

        // 时效性应该很低
        assert!(importance.recency < 0.2);
    }

    #[test]
    fn test_memory_importance_generality() {
        let mut importance = MemoryImportance::new();

        // 通用类别
        importance.evaluate_generality("common_sense", &vec![]);
        assert!(importance.generality >= 0.7);

        // 特定类别
        importance.evaluate_generality("user_specific", &vec![]);
        assert!(importance.generality <= 0.5);

        // 带有通用标签
        importance.evaluate_generality(
            "user_specific",
            &vec!["general".to_string(), "core".to_string()],
        );
        assert!(importance.generality >= 0.5);
    }

    #[test]
    fn test_should_prune() {
        let mut importance = MemoryImportance::new();

        // 高价值记忆不应该被修剪
        importance.frequency = 10;
        importance.user_validated = true;
        assert!(!importance.should_prune(0.3));

        // 低价值记忆应该被修剪
        let mut low_importance = MemoryImportance::new();
        low_importance.frequency = 0;
        low_importance.recency = 0.1;
        low_importance.generality = 0.2;
        assert!(low_importance.should_prune(0.3));
    }
}

#[cfg(test)]
mod retention_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_path(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rt_mem_retention_{tag}_{nanos}.jsonl"))
    }

    fn entry(category: &str, note: &str, ts: &str, priority: u8) -> AgentMemoryEntry {
        AgentMemoryEntry {
            id: None,
            timestamp: ts.to_string(),
            category: category.to_string(),
            note: note.to_string(),
            tags: vec![],
            source: None,
            priority: Some(priority),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        }
    }

    fn entry_with_id(
        id: &str,
        category: &str,
        note: &str,
        ts: &str,
        priority: u8,
    ) -> AgentMemoryEntry {
        let mut entry = entry(category, note, ts, priority);
        entry.id = Some(id.to_string());
        entry
    }

    fn write_lines(path: &Path, entries: &[AgentMemoryEntry]) {
        let mut buf = String::new();
        for e in entries {
            buf.push_str(&serde_json::to_string(e).unwrap());
            buf.push('\n');
        }
        std::fs::write(path, buf).unwrap();
    }

    fn read_entries(path: &Path) -> Vec<AgentMemoryEntry> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<AgentMemoryEntry>(l.trim()).ok())
            .collect()
    }

    /// P0-1 回归：原实现的双删 bug 会在配额满后误删紧邻 priority=255 的条目，
    /// 这里制造一份"低优先级 + 永久"混合，断言 enforce 后所有 priority=255
    /// 都还在，且配额被压回 max_entries 之内。
    #[test]
    fn enforce_max_entries_keeps_all_permanent_entries() {
        let path = unique_path("enforce_perm");
        let mut all = Vec::new();
        // 100 条普通低优先级，按时间从旧到新
        for i in 0..100 {
            all.push(entry(
                "tool_stat",
                &format!("note-{i}"),
                &format!("2025-01-01T00:00:{:02}Z", i % 60),
                50,
            ));
        }
        // 5 条永久条目（safety_rules）
        for i in 0..5 {
            all.push(entry(
                "safety_rules",
                &format!("perm-{i}"),
                &format!("2025-02-01T00:00:{:02}Z", i),
                255,
            ));
        }
        write_lines(&path, &all);

        let store = MemoryStore::for_tests_with_path(path.clone());
        store.enforce_max_entries(50, 10).unwrap();

        let kept = read_entries(&path);
        assert!(
            kept.len() <= 50,
            "expected <=50 entries, got {}",
            kept.len()
        );
        let perm_kept = kept.iter().filter(|e| e.priority == Some(255)).count();
        assert_eq!(perm_kept, 5, "all permanent entries must survive");

        let _ = std::fs::remove_file(&path);
    }

    /// P0-2 回归：rotate 后新主文件只能含 priority=255 的条目，
    /// 归档文件包含全部原条目。
    #[test]
    fn rotate_preserves_permanent_entries_in_main_file() {
        let path = unique_path("rotate_perm");
        let mut all = Vec::new();
        // 用足够大的 note 把文件膨胀到几 KB
        let big = "x".repeat(2048);
        for i in 0..20 {
            all.push(entry(
                "tool_cache",
                &format!("{}-{}", big, i),
                &format!("2025-01-01T00:00:{:02}Z", i),
                80,
            ));
        }
        all.push(entry(
            "safety_rules",
            "do not run rm -rf /",
            "2025-02-02T00:00:00Z",
            255,
        ));
        all.push(entry(
            "self_note",
            "always read before edit",
            "2025-02-02T00:00:01Z",
            255,
        ));
        write_lines(&path, &all);

        let store = MemoryStore::for_tests_with_path(path.clone());
        // 阈值故意比当前 size 小，强制 rotate
        let rotated = store.rotate_if_exceeds(1024).unwrap();
        assert!(rotated, "expected rotate to happen");

        // 主文件只剩永久
        let head = read_entries(&path);
        assert_eq!(head.len(), 2);
        assert!(head.iter().all(|e| e.priority == Some(255)));

        // 归档文件存在，且包含所有原条目
        let parent = path.parent().unwrap();
        let archives: Vec<_> = std::fs::read_dir(parent)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let head_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.starts_with(head_name) && p != &path
            })
            .collect();
        assert_eq!(archives.len(), 1, "expected exactly one archive");
        let archived = read_entries(&archives[0]);
        assert_eq!(archived.len(), all.len());

        let _ = std::fs::remove_file(&path);
        for a in archives {
            let _ = std::fs::remove_file(a);
        }
    }
}
