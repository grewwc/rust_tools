use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use crate::commonw::configw;
use super::with_memory_file_lock;
use crate::ai::tools::service::memory::{execute_memory_dedup, execute_memory_gc};
use serde_json::json;
use std::time::{SystemTime, Duration};
use std::ffi::OsStr;
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AgentMemoryEntry {
    pub(crate) timestamp: String,
    pub(crate) category: String,
    pub(crate) note: String,
    pub(crate) tags: Vec<String>,
    pub(crate) source: Option<String>,
}

pub(crate) struct MemoryStore {
    path: PathBuf,
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
        super::with_memory_file_lock(&self.path, || {
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create memory dir: {e}"))?;
            }
            let serialized = serde_json::to_string(entry)
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

            Ok(())
        })
    }

    pub(crate) fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<AgentMemoryEntry>, String> {
        let query_lc = query.to_lowercase();
        let mut docs: Vec<(AgentMemoryEntry, String, Vec<String>)> = Vec::new();
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
                archives.sort_by_key(|(_, m)| *m);
                let take_from = archives.len().saturating_sub(keep_last_archives);
                for (p, _) in archives.into_iter().skip(take_from) {
                    files.push(p);
                }
            }
        }
        files.push(self.path.clone());

        for p in files {
            if !p.exists() {
                continue;
            }
            let file = fs::File::open(&p).map_err(|e| format!("Failed to read memory file: {e}"))?;
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
                let tokens = expand_tokens(&tokenize(&full.to_lowercase()));
                docs.push((entry, full, tokens));
            }
        }
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let nq_tokens = expand_tokens(&tokenize(&query_lc));
        use std::collections::{HashMap, HashSet};
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut avgdl = 0.0f64;
        for (_, _, toks) in &docs {
            avgdl += toks.len() as f64;
            let mut set = HashSet::new();
            for t in toks {
                if set.insert(t) {
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
            let mut tf: HashMap<&str, usize> = HashMap::new();
            for t in toks {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
            let mut bm25 = 0.0f64;
            let dl = toks.len() as f64;
            let mut seenq = HashSet::new();
            for qt in &nq_tokens {
                if !seenq.insert(qt) {
                    continue;
                }
                let dfv = *df.get(qt).unwrap_or(&0) as f64;
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
            let bm = if max_bm25 > 0.0 { bm25_vals[i] / max_bm25 } else { 0.0 };
            let s = 0.55 * scored[i].0 + 0.45 * bm;
            scored[i].0 = s;
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let cap = limit.saturating_mul(10).min(200).max(limit);
        let mut top_idx: Vec<usize> = scored.iter().take(cap).map(|(_, i)| *i).collect();
        let qv = embed_text(&query_lc);
        if let Some(qv) = qv {
            let mut rescored: Vec<(f64, usize)> = Vec::with_capacity(top_idx.len());
            for &i in &top_idx {
                let (_, full, _) = &docs[i];
                let ev = embed_text(full);
                let emb = ev.as_ref().map(|v| cosine_similarity(&qv, v)).unwrap_or(0.0);
                let base = scored[i].0;
                let final_s = 0.85 * base + 0.15 * emb as f64;
                rescored.push((final_s, i));
            }
            rescored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            top_idx = rescored.into_iter().take(limit).map(|(_, i)| i).collect();
        } else {
            top_idx.truncate(limit);
        }
        let mut out = Vec::with_capacity(top_idx.len());
        for i in top_idx {
            out.push(docs[i].0.clone());
        }
        Ok(out)
    }

    pub(crate) fn recent(&self, limit: usize) -> Result<Vec<AgentMemoryEntry>, String> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.path).map_err(|e| format!("Failed to read memory file: {e}"))?;
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

#[cfg(test)]
impl MemoryStore {
    pub(crate) fn for_tests_with_path(path: PathBuf) -> Self {
        Self { path }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    #[test]
    fn test_search_recall_ngram() {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e1 = AgentMemoryEntry {
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            category: "log".to_string(),
            note: "parsing login error occurred".to_string(),
            tags: vec!["auth".to_string()],
            source: Some("svc".to_string()),
        };
        let e2 = AgentMemoryEntry {
            timestamp: "2025-01-02T00:00:00Z".to_string(),
            category: "info".to_string(),
            note: "user profile updated".to_string(),
            tags: vec!["user".to_string()],
            source: Some("svc".to_string()),
        };
        store.append(&e1).unwrap();
        store.append(&e2).unwrap();
        let out = store.search("parse login", 5).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|x| x.note.contains("parsing login")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_search_recall_synonym_login() {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_syn_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e = AgentMemoryEntry {
            timestamp: "2025-01-03T00:00:00Z".to_string(),
            category: "auth".to_string(),
            note: "user login failed due to authentication error".to_string(),
            tags: vec!["login".to_string()],
            source: None,
        };
        store.append(&e).unwrap();
        let out = store.search("signin failure", 3).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|x| x.note.contains("login failed")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_search_recall_chinese_login_variants() {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_cn_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e = AgentMemoryEntry {
            timestamp: "2025-01-04T00:00:00Z".to_string(),
            category: "auth".to_string(),
            note: "登录失败，密码错误".to_string(),
            tags: vec!["登录".to_string()],
            source: None,
        };
        store.append(&e).unwrap();
        let out = store.search("登陆失败", 3).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|x| x.note.contains("登录失败")));
        let _ = std::fs::remove_file(&path);
    }
}

fn norm_text(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).flat_map(|c| c.to_lowercase()).collect()
}

fn bigrams(s: &str) -> Vec<(char, char)> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return Vec::new();
    }
    let mut v = Vec::with_capacity(chars.len() - 1);
    for i in 0..(chars.len() - 1) {
        v.push((chars[i], chars[i + 1]));
    }
    v
}

fn dice_coefficient(a: &[(char, char)], b: &[(char, char)]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    use std::collections::HashMap;
    let mut count = 0usize;
    let mut map: HashMap<(char, char), usize> = HashMap::new();
    for x in a {
        *map.entry(*x).or_insert(0) += 1;
    }
    for y in b {
        if let Some(c) = map.get_mut(y) {
            if *c > 0 {
                count += 1;
                *c -= 1;
            }
        }
    }
    (2.0 * count as f64) / ((a.len() + b.len()) as f64)
}

fn is_han_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x30000..=0x3134F
    )
}

fn tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            buf.push(ch.to_ascii_lowercase());
        } else {
            if !buf.is_empty() {
                tokens.push(buf.clone());
                buf.clear();
            }
            if is_han_char(ch) {
                tokens.push(ch.to_string());
            }
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

fn synonyms_for(token: &str) -> Vec<&'static str> {
    match token {
        "login" => vec!["signin", "sign-in", "logon"],
        "signin" => vec!["login", "sign-in", "logon"],
        "sign-in" => vec!["login", "signin", "logon"],
        "auth" => vec!["authentication", "authorize", "authorization"],
        "authentication" => vec!["auth"],
        "解析" => vec!["parse", "parsing"],
        "parse" => vec!["解析", "parsing"],
        "parsing" => vec!["parse", "解析"],
        "登录" => vec!["登陆", "login", "signin", "sign-in", "logon"],
        "登陆" => vec!["登录", "login", "signin", "sign-in", "logon"],
        "失败" => vec!["错误", "error", "failed"],
        "错误" => vec!["失败", "error", "bug"],
        "error" => vec!["failed", "failure"],
        "配置" => vec!["config", "configuration"],
        "config" => vec!["configuration", "配置"],
        "代码" => vec!["code", "源码"],
        "code" => vec!["源码", "代码"],
        _ => vec![],
    }
}

fn expand_tokens(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len() * 2);
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    for t in tokens {
        let tnorm = t.to_lowercase();
        if seen.insert(tnorm.clone()) {
            out.push(tnorm.clone());
        }
        for syn in synonyms_for(&tnorm) {
            let s = syn.to_string();
            if seen.insert(s.clone()) {
                out.push(s);
            }
        }
    }
    out
}

fn jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    use std::collections::HashSet;
    let sa: HashSet<&String> = a.iter().collect();
    let sb: HashSet<&String> = b.iter().collect();
    let inter = sa.intersection(&sb).count() as f64;
    let uni = sa.union(&sb).count() as f64;
    if uni == 0.0 {
        0.0
    } else {
        inter / uni
    }
}

fn char_overlap(a: &str, b: &str) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    use std::collections::HashSet;
    let sa: HashSet<char> = a.chars().collect();
    let sb: HashSet<char> = b.chars().collect();
    let inter = sa.intersection(&sb).count() as f64;
    let denom = sa.len().min(sb.len()) as f64;
    if denom == 0.0 {
        0.0
    } else {
        inter / denom
    }
}

fn compute_similarity(entry: &AgentMemoryEntry, query_lc: &str) -> f64 {
    let base_contains = if entry.note.to_lowercase().contains(query_lc)
        || entry.category.to_lowercase().contains(query_lc)
        || entry.tags.iter().any(|t| t.to_lowercase().contains(query_lc))
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
    let nq = norm_text(query_lc);
    let ne = norm_text(&full);
    let d = dice_coefficient(&bigrams(&nq), &bigrams(&ne));
    let tq = expand_tokens(&tokenize(query_lc));
    let te = expand_tokens(&tokenize(&full.to_lowercase()));
    let j = jaccard(&tq, &te);
    let co = char_overlap(&nq, &ne);
    let s = 0.5 * d + 0.3 * j + 0.15 * co + base_contains;
    if s < 0.0 { 0.0 } else { s.min(1.0) }
}

pub trait EmbeddingProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
}

struct NoopEmbeddingProvider;
impl EmbeddingProvider for NoopEmbeddingProvider {
    fn embed(&self, _text: &str) -> Option<Vec<f32>> { None }
}

static EMBEDDING_PROVIDER: OnceLock<Box<dyn EmbeddingProvider + Sync + Send>> = OnceLock::new();

pub fn embed_text(text: &str) -> Option<Vec<f32>> {
    if let Some(p) = EMBEDDING_PROVIDER.get() {
        p.embed(text)
    } else {
        None
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    for i in 0..len {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 0.0 } else { (dot / denom).max(-1.0).min(1.0) }
}

fn resolve_memory_file() -> PathBuf {
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
            let meta = std::fs::metadata(&path).ok();
            if let Some(meta) = meta {
                if meta.len() > max_bytes {
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
                    std::fs::File::create(&path)
                        .map_err(|e| format!("Failed to create new memory file: {}", e))?;
                    return Ok(true);
                }
            }
            Ok(false)
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
        let rotated = self.rotate_if_exceeds(max_bytes).unwrap_or(false);
        let _ = if rotated {
            self.cleanup_archives_auto()
        } else {
            Ok(())
        };
        let roll = rand::random::<f64>();
        if roll < prob {
            let _ = execute_memory_dedup(&json!({}));
            let _ = execute_memory_gc(&json!({ "max_days": gc_days, "min_keep": min_keep }));
            let _ = self.cleanup_archives_auto();
        }
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
            let file_name = entry
                .file_name()
                .to_str()
                .unwrap_or("")
                .to_string();
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
