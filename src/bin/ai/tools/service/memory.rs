use chrono::{DateTime, Local, Utc};
use rust_tools::cw::SkipSet;

use serde_json::Value;
use std::io::{BufRead, Write};
use uuid::Uuid;

use crate::ai::tools::os_tools::GLOBAL_OS;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};

fn current_owner_tags() -> (Option<u64>, Option<u64>) {
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os_arc) = guard.as_ref() {
            if let Ok(os) = os_arc.lock() {
                if let Some(pid) = os.current_process_id() {
                    let pgid = os.get_process(pid).and_then(|p| p.process_group);
                    return (Some(pid), pgid);
                }
            }
        }
    }
    (None, None)
}

fn is_memory_visible_to(entry: &AgentMemoryEntry, viewer_pid: Option<u64>) -> bool {
    let Some(viewer) = viewer_pid else {
        return true;
    };
    let Some(owner) = entry.owner_pid else {
        return true;
    };
    if owner == viewer {
        return true;
    }
    let Ok(guard) = GLOBAL_OS.lock() else {
        return false;
    };
    let Some(os_arc) = guard.as_ref() else {
        return false;
    };
    let Ok(os) = os_arc.lock() else {
        return false;
    };
    if let Some(entry_pgid) = entry.owner_pgid {
        if let Some(vpgid) = os.get_process(viewer).and_then(|p| p.process_group) {
            if entry_pgid == vpgid {
                return true;
            }
        }
    }
    let mut cursor = owner;
    while let Some(proc) = os.get_process(cursor) {
        if proc.parent_pid == Some(viewer) {
            return true;
        }
        match proc.parent_pid {
            Some(parent) => cursor = parent,
            None => break,
        }
    }
    false
}

struct ViewerContext {
    viewer_pid: Option<u64>,
    viewer_pgid: Option<u64>,
}

impl ViewerContext {
    fn current() -> Self {
        let (pid, pgid) = current_owner_tags();
        Self {
            viewer_pid: pid,
            viewer_pgid: pgid,
        }
    }

    fn can_see(&self, entry: &AgentMemoryEntry) -> bool {
        let Some(viewer) = self.viewer_pid else {
            return true;
        };
        let Some(owner) = entry.owner_pid else {
            return true;
        };
        if owner == viewer {
            return true;
        }
        if let (Some(entry_pgid), Some(vpgid)) = (entry.owner_pgid, self.viewer_pgid) {
            if entry_pgid == vpgid {
                return true;
            }
        }
        let Ok(guard) = GLOBAL_OS.lock() else {
            return false;
        };
        let Some(os_arc) = guard.as_ref() else {
            return false;
        };
        let Ok(os) = os_arc.lock() else {
            return false;
        };
        let mut cursor = owner;
        while let Some(proc) = os.get_process(cursor) {
            if proc.parent_pid == Some(viewer) {
                return true;
            }
            match proc.parent_pid {
                Some(parent) => cursor = parent,
                None => break,
            }
        }
        false
    }
}

fn parse_string_array(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn render_memory_entries(entries: &[AgentMemoryEntry]) -> String {
    if entries.is_empty() {
        return "No memory entries found.".to_string();
    }
    let mut output = String::new();
    for (idx, entry) in entries.iter().enumerate() {
        if idx > 0 {
            output.push('\n');
        }
        output.push_str(&format!(
            "{}. [{}] {}\n{}",
            idx + 1,
            entry.timestamp,
            entry.category,
            entry.note
        ));
        if let Some(id) = &entry.id
            && !id.trim().is_empty()
        {
            output.push_str(&format!("\nID: {}", id));
        }
        if !entry.tags.is_empty() {
            output.push_str(&format!("\nTags: {}", entry.tags.join(", ")));
        }
        if let Some(source) = &entry.source
            && !source.trim().is_empty()
        {
            output.push_str(&format!("\nSource: {}", source));
        }
    }
    output
}

fn next_memory_id() -> String {
    format!("mem_{}", Uuid::new_v4().simple())
}

fn normalized_category(raw: Option<&str>, fallback: &str) -> String {
    let value = raw.unwrap_or(fallback).trim().to_lowercase();
    if value.is_empty() {
        fallback.to_string()
    } else {
        value
    }
}

fn default_priority_for_category(category: &str) -> u8 {
    match category {
        "common_sense" | "coding_guideline" | "best_practice" | "user_preference"
        | "preference" => 210,
        "safety_rules" => 255,
        _ => 150,
    }
}

fn is_long_term_learning_category(category: &str) -> bool {
    matches!(
        category,
        "common_sense"
            | "coding_guideline"
            | "best_practice"
            | "safety_rules"
            | "user_preference"
            | "preference"
            | "project_memory"
    )
}

fn maybe_downgrade_memory_save(
    category: &str,
    note: &str,
    tags: &mut Vec<String>,
    source: &mut Option<String>,
    priority: Option<u8>,
) -> (
    String,
    Option<u8>,
    crate::ai::driver::reflection::LearningNoteAssessment,
) {
    if !is_long_term_learning_category(category) {
        let assessment = crate::ai::driver::reflection::assess_learning_note_quality(note);
        return (category.to_string(), priority, assessment);
    }

    let assessment = crate::ai::driver::reflection::assess_learning_note_quality(note);
    if assessment.high_quality {
        return (category.to_string(), priority, assessment);
    }

    if !tags.iter().any(|tag| tag == "auto_downgraded") {
        tags.push("auto_downgraded".to_string());
    }
    if !tags.iter().any(|tag| tag == "low_signal") {
        tags.push("low_signal".to_string());
    }
    if let Some(src) = source.as_mut() {
        if !src.contains("memory_save_downgraded") {
            src.push_str(":memory_save_downgraded");
        }
    } else {
        *source = Some("agent_memory_save:memory_save_downgraded".to_string());
    }
    (
        "self_note".to_string(),
        Some(priority.unwrap_or(120).min(120)),
        assessment,
    )
}

/// 判断一条记忆是否豁免 30d 时间窗 GC / 配额淘汰。
///
/// 历史实现只看 `priority == 255`：
///   - 用户偏好 / coding_guideline / project_memory 等长期资产 priority 是 210/180，
///     30 天没刷新就有概率被 GC 掉，与"长期记忆应该保留"的直觉不符。
///
/// 新策略：priority==255 仍然豁免（safety_rules 等显式声明的永久记忆），
/// 同时把以下 category 视为长期资产，无论 priority 多少都豁免：
///   - 多数 guideline 类（safety/preference/user_preference/coding_guideline/
///     best_practice/common_sense）
///   - `project_memory`：项目级事实，writeback 路径会主动 upsert，不应被时间淘汰
///
/// 注意 self_note 是会话期反思，已从 `guideline_categories()` 中移除，
/// 不再被全局召回，也不在此处豁免 GC（自然落入正常时间淘汰）。
pub(crate) fn is_permanent_memory(entry: &AgentMemoryEntry) -> bool {
    if entry.priority.unwrap_or(100) == 255 {
        return true;
    }
    if crate::ai::knowledge::retrieval::recall::is_guideline_category(&entry.category) {
        return true;
    }
    matches!(entry.category.as_str(), "project_memory")
}

fn parse_priority_arg(args: &Value, field: &str) -> Result<Option<u8>, String> {
    match args.get(field).and_then(|value| value.as_u64()) {
        Some(priority) if priority > u8::MAX as u64 => Err("priority out of range".to_string()),
        Some(priority) => Ok(Some(priority as u8)),
        None => Ok(None),
    }
}

fn load_memory_entries(path: &std::path::Path) -> Result<Vec<AgentMemoryEntry>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = std::io::BufReader::new(file);
    let mut entries = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read memory file: {}", e))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(&line) {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn write_memory_entries(
    path: &std::path::Path,
    entries: &[AgentMemoryEntry],
) -> Result<(), String> {
    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut f =
            std::fs::File::create(&tmp).map_err(|e| format!("Failed to create tmp: {}", e))?;
        for entry in entries {
            let line = serde_json::to_string(entry).map_err(|e| format!("{}", e))?;
            f.write_all(line.as_bytes())
                .and_then(|_| f.write_all(b"\n"))
                .map_err(|e| format!("Failed to write tmp: {}", e))?;
        }
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("Failed to replace memory file: {}", e))?;
    Ok(())
}

pub(crate) fn execute_memory_append(args: &Value) -> Result<String, String> {
    let note = args["note"].as_str().ok_or("Missing note")?.trim();
    if note.is_empty() {
        return Err("note is empty".to_string());
    }
    let category = normalized_category(args["category"].as_str(), "general");
    let tags = parse_string_array(&args["tags"]);
    let source = args["source"]
        .as_str()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let priority = parse_priority_arg(args, "priority")?
        .or_else(|| Some(default_priority_for_category(&category)));
    let (owner_pid, owner_pgid) = current_owner_tags();
    let entry = AgentMemoryEntry {
        id: Some(next_memory_id()),
        timestamp: Local::now().to_rfc3339(),
        category,
        note: note.to_string(),
        tags,
        source,
        priority,
        owner_pid,
        owner_pgid,
        image_path: None,
    };

    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;
    Ok(format!(
        "Memory appended: {} (id: {})",
        store.path().display(),
        entry.id.as_deref().unwrap_or("")
    ))
}

pub(crate) fn execute_memory_search(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().ok_or("Missing query")?.trim();
    if query.is_empty() {
        return Err("query is empty".to_string());
    }
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 50) as usize;
    let category_filter = args["category"].as_str().map(|s| s.trim().to_lowercase());
    let tags_any = parse_string_array(&args["tags_any"])
        .into_iter()
        .map(|s| s.to_lowercase())
        .collect::<Vec<_>>();
    let tags_all = parse_string_array(&args["tags_all"])
        .into_iter()
        .map(|s| s.to_lowercase())
        .collect::<Vec<_>>();
    let source_sub = args["source_substring"]
        .as_str()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty());
    let debug_score = args["debug_score"].as_bool().unwrap_or(false);
    let store = MemoryStore::from_env_or_config();
    let results = store.search(query, 10_000)?;
    let viewer = ViewerContext::current();

    let mut scored = Vec::with_capacity(results.len());
    for (e, _search_score) in results {
        if !viewer.can_see(&e) {
            continue;
        }
        if let Some(cat) = category_filter.as_ref() {
            if e.category.to_lowercase() != *cat {
                continue;
            }
        }
        if !tags_any.is_empty()
            && !e
                .tags
                .iter()
                .any(|t| tags_any.iter().any(|x| t.to_lowercase() == *x))
        {
            continue;
        }
        if !tags_all.is_empty()
            && !tags_all
                .iter()
                .all(|x| e.tags.iter().any(|t| t.to_lowercase() == *x))
        {
            continue;
        }
        if let Some(sub) = source_sub.as_ref() {
            if !e
                .source
                .as_ref()
                .map(|s| s.to_lowercase().contains(sub))
                .unwrap_or(false)
            {
                continue;
            }
        }
        let mut score = 0.0_f64;
        let qlc = query.to_lowercase();
        if e.note.to_lowercase().contains(&qlc) {
            score += 3.0;
            score += (qlc.len() as f64).min(20.0) * 0.05;
        }
        if e.category.to_lowercase().contains(&qlc) {
            score += 1.5;
        }
        if e.tags.iter().any(|t| t.to_lowercase().contains(&qlc)) {
            score += 1.2;
        }
        if e.source
            .as_ref()
            .map(|s| s.to_lowercase().contains(&qlc))
            .unwrap_or(false)
        {
            score += 0.8;
        }
        let recency_bonus = parse_rfc3339_ts(&e.timestamp)
            .map(|ts| {
                let age_secs = (Utc::now() - ts).num_seconds().max(0) as f64;
                if age_secs <= 7.0 * 86400.0 {
                    1.0
                } else if age_secs <= 30.0 * 86400.0 {
                    0.3
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        score += recency_bonus;
        scored.push((score, e));
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut top_scored = scored;
    top_scored.truncate(limit);
    let top: Vec<AgentMemoryEntry> = top_scored.iter().map(|(_, e)| e.clone()).collect();
    let mut out = render_memory_entries(&top);
    if debug_score {
        out.push_str("\n");
        out.push_str("--- scores ---\n");
        for (idx, (s, e)) in top_scored.iter().enumerate() {
            out.push_str(&format!(
                "{}. score={:.2} [{}] {}\n",
                idx + 1,
                s,
                e.category,
                e.note
            ));
        }
    }
    Ok(out)
}

/// 带分数的 memo 检索结果。`semantic` 表示本次是否用到了语义（embedding）打分——
/// 上层据此决定能否按语义分数做上下文收紧。
pub(crate) struct ScoredMemo {
    pub entry: AgentMemoryEntry,
    pub score: f64,
    pub semantic: bool,
}

/// 根据查询文本检索 memo 候选条目，返回结构化条目（按相关度排序）。
/// 用于 `-nd` 删除流程：让上层用模型挑选最匹配的条目再确认删除。
pub(crate) fn search_memo_candidates(
    query: &str,
    limit: usize,
) -> Result<Vec<AgentMemoryEntry>, String> {
    Ok(search_memo_candidates_scored(query, limit)?
        .into_iter()
        .map(|s| s.entry)
        .collect())
}

/// 与 `search_memo_candidates` 同源，但保留分数与"是否用了语义"标记。
pub(crate) fn search_memo_candidates_scored(
    query: &str,
    limit: usize,
) -> Result<Vec<ScoredMemo>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Err("query is empty".to_string());
    }
    let limit = limit.clamp(1, 50);
    let store = MemoryStore::from_env_or_config();
    // 只加载 memo 类别条目即可：memo 是用户手工记录的类别，量很小。
    // category 写入恒为小写 "memo"。
    let results = store.entries_by_category("memo", 100_000)?;
    let viewer = ViewerContext::current();
    let qlc = query.to_lowercase();

    let visible: Vec<AgentMemoryEntry> =
        results.into_iter().filter(|e| viewer.can_see(e)).collect();

    // 字面打分（子串命中），任何情况下都计算——作为基础信号 / embedding 不可用时的唯一信号。
    let lexical = |e: &AgentMemoryEntry| -> f64 {
        let mut score = 0.0_f64;
        if e.note.to_lowercase().contains(&qlc) {
            score += 3.0;
            score += (qlc.len() as f64).min(20.0) * 0.05;
        }
        if e.tags.iter().any(|t| t.to_lowercase().contains(&qlc)) {
            score += 1.2;
        }
        score
    };

    // 语义重排：仅当远程 embedding 可用时启用。任意一步拿不到向量都整体回退到
    // 纯字面打分（即历史行为），绝不因为 embedding 故障而改变/中断检索结果。
    let semantic: Option<Vec<f64>> = if crate::ai::knowledge::indexing::embedder::is_ready() {
        let qv = crate::ai::knowledge::indexing::embedder::embed_text(&qlc);
        match qv {
            Some(qv) => {
                let texts: Vec<String> = visible.iter().map(|e| e.note.clone()).collect();
                crate::ai::knowledge::indexing::embedder::embed_texts(&texts).map(|batch| {
                    batch
                        .iter()
                        .map(|v| {
                            crate::ai::knowledge::indexing::similarity::cosine_similarity(&qv, v)
                                as f64
                        })
                        .collect()
                })
            }
            None => None,
        }
    } else {
        None
    };

    let used_semantic = semantic.is_some();
    let mut scored: Vec<ScoredMemo> = Vec::with_capacity(visible.len());
    for (i, e) in visible.into_iter().enumerate() {
        let lex = lexical(&e);
        // 有 embedding 时语义为主、字面为辅；无 embedding 时退回纯字面。
        let score = match &semantic {
            Some(sims) => sims.get(i).copied().unwrap_or(0.0) * 10.0 + lex,
            None => lex,
        };
        scored.push(ScoredMemo {
            entry: e,
            score,
            semantic: used_semantic,
        });
    }
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    scored.truncate(limit);
    Ok(scored)
}

pub(crate) fn execute_memory_recent(args: &Value) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 50) as usize;
    let store = MemoryStore::from_env_or_config();
    let entries = store.recent(limit)?;
    let viewer = ViewerContext::current();
    let visible: Vec<AgentMemoryEntry> =
        entries.into_iter().filter(|e| viewer.can_see(e)).collect();
    Ok(render_memory_entries(&visible))
}

pub(crate) fn execute_memory_list_json(args: &Value) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(50).clamp(1, 200) as usize;
    let offset = args["offset"].as_u64().unwrap_or(0) as usize;
    let store = MemoryStore::from_env_or_config();
    let entries = store.recent(limit + offset)?;
    let viewer = ViewerContext::current();
    let visible: Vec<AgentMemoryEntry> =
        entries.into_iter().filter(|e| viewer.can_see(e)).collect();
    let sliced = if offset >= visible.len() {
        Vec::new()
    } else {
        visible.into_iter().skip(offset).collect::<Vec<_>>()
    };
    serde_json::to_string(&sliced).map_err(|e| format!("{}", e))
}

pub(crate) fn execute_memory_rotate(args: &Value) -> Result<String, String> {
    let max_bytes = args["max_bytes"].as_u64().ok_or("Missing max_bytes")? as u64;
    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        let meta = std::fs::metadata(&path).ok();
        if let Some(meta) = meta {
            if meta.len() > max_bytes {
                let ts = Local::now().format("%Y%m%d%H%M%S").to_string();
                let mut new_name = path.clone();
                new_name.set_extension(format!("jsonl.{}", ts));
                std::fs::rename(&path, &new_name)
                    .map_err(|e| format!("Failed to rotate file: {}", e))?;
                std::fs::File::create(&path)
                    .map_err(|e| format!("Failed to create new memory file: {}", e))?;
                return Ok(format!(
                    "Rotated: {} -> {}",
                    path.display(),
                    new_name.display()
                ));
            }
        }
        Ok("Rotate skipped: size within limit".to_string())
    })
}

pub(crate) fn execute_memory_gc(args: &Value) -> Result<String, String> {
    let max_days = args["max_days"].as_u64().ok_or("Missing max_days")? as i64;
    let min_keep = args["min_keep"].as_u64().unwrap_or(200) as usize;
    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Ok("No memory file".to_string());
        }
        let file = std::fs::File::open(&path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = std::io::BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read memory file: {}", e))?;
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            if let Ok(e) = serde_json::from_str::<AgentMemoryEntry>(&line) {
                entries.push(e);
            }
        }
        if entries.is_empty() {
            return Ok("No entries".to_string());
        }

        // Separate permanent entries (whitelist) - never deleted by GC
        let mut permanent: Vec<AgentMemoryEntry> = entries
            .iter()
            .filter(|e| is_permanent_memory(e))
            .cloned()
            .collect();
        let mut deletable: Vec<AgentMemoryEntry> = entries
            .into_iter()
            .filter(|e| !is_permanent_memory(&e))
            .collect();

        let now = Utc::now();
        let cutoff_secs = max_days * 86400;

        // 摘要回写（writeback）：原始实现把"过期的 deletable"直接丢掉，丢失了
        // 与项目相关但不是永久白名单的事实（例如 30 天前的 task_event）。
        // 现在把过期项按 (category, source) 分组，每组合成 1 条 summary
        // 写回 permanent 区——以"sum:"前缀标记，priority 沿用该组最高，
        // 后续不会被时间窗 GC（summary 自身永远是新创建，时间 = now）。
        let mut evicted: Vec<AgentMemoryEntry> = Vec::new();
        deletable.retain(|e| {
            let keep = parse_rfc3339_ts(&e.timestamp)
                .map(|ts| (now - ts).num_seconds() <= cutoff_secs)
                .unwrap_or(true);
            if !keep {
                evicted.push(e.clone());
            }
            keep
        });
        let summaries = if evicted.is_empty() {
            Vec::new()
        } else {
            build_gc_summaries(&evicted, max_days)
        };
        let summary_count = summaries.len();

        // Sort deletable entries by priority (ascending) then by timestamp (ascending)
        // This ensures low priority and old entries are deleted first
        deletable.sort_by(|a, b| {
            let prio_a = a.priority.unwrap_or(100);
            let prio_b = b.priority.unwrap_or(100);
            prio_a.cmp(&prio_b).then_with(|| {
                let ts_a = parse_rfc3339_ts(&a.timestamp);
                let ts_b = parse_rfc3339_ts(&b.timestamp);
                ts_a.cmp(&ts_b)
            })
        });

        // Ensure minimum keep count (but never delete permanent entries)
        let total_permanent = permanent.len();
        if deletable.len() + total_permanent + summary_count < min_keep {
            // Need to restore some entries, but prefer higher priority ones
            let all_entries = store.recent(min_keep)?;
            let new_deletable: Vec<AgentMemoryEntry> = all_entries
                .iter()
                .filter(|e| !is_permanent_memory(e))
                .cloned()
                .collect();
            let new_permanent: Vec<AgentMemoryEntry> = all_entries
                .iter()
                .filter(|e| is_permanent_memory(e))
                .cloned()
                .collect();
            permanent = new_permanent;
            deletable = new_deletable;
        }

        // Combine permanent + summaries (新生成) + deletable
        let permanent_count = permanent.len();
        let mut final_entries = permanent;
        final_entries.extend(summaries);
        final_entries.append(&mut deletable);

        let tmp = path.with_extension("jsonl.tmp");
        {
            let mut f =
                std::fs::File::create(&tmp).map_err(|e| format!("Failed to create tmp: {}", e))?;
            for e in &final_entries {
                let line = serde_json::to_string(e).map_err(|e| format!("{}", e))?;
                f.write_all(line.as_bytes())
                    .and_then(|_| f.write_all(b"\n"))
                    .map_err(|e| format!("Failed to write tmp: {}", e))?;
            }
        }
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("Failed to replace memory file: {}", e))?;
        crate::ai::tools::storage::memory_store::trace_memory_event(
            "memory.gc",
            "GC pass with summary writeback",
            &[
                ("kept", final_entries.len().to_string()),
                ("permanent", permanent_count.to_string()),
                ("summaries", summary_count.to_string()),
                ("max_days", max_days.to_string()),
            ],
        );
        Ok(format!(
            "GC done: {} entries kept (including {} permanent, {} summary writeback)",
            final_entries.len(),
            permanent_count,
            summary_count
        ))
    })
}

/// 把过期 evicted 条目按 (category, source) 分组，每组聚合成一条 summary。
///
/// summary 设计：
///   - category 沿用原 category（这样跟原条目处于同一个召回桶里）
///   - tags 取首个被合并条目的 tags + 加 "summary"
///   - source 沿用原 source
///   - priority = 该组最高 priority（保留原本的重要性信号）
///   - note = "[summary of N entries from <ts1> to <ts2>] " + 截断的代表性内容
///   - timestamp = 当前时间（让它进入"最新区域"，不会立即又被时间窗 GC）
///
/// 不调外部模型——以本地拼接为主，避免 GC 路径阻塞。
fn build_gc_summaries(evicted: &[AgentMemoryEntry], max_days: i64) -> Vec<AgentMemoryEntry> {
    use std::collections::BTreeMap;

    // (category, source) -> Vec<&entry>
    let mut groups: BTreeMap<(String, Option<String>), Vec<&AgentMemoryEntry>> = BTreeMap::new();
    for e in evicted {
        groups
            .entry((e.category.clone(), e.source.clone()))
            .or_default()
            .push(e);
    }

    let mut out: Vec<AgentMemoryEntry> = Vec::with_capacity(groups.len());
    let now_iso = Local::now().to_rfc3339();
    for ((category, source), items) in groups.into_iter() {
        if items.is_empty() {
            continue;
        }
        let count = items.len();
        // 时间范围
        let mut ts_min = items[0].timestamp.as_str();
        let mut ts_max = items[0].timestamp.as_str();
        let mut max_prio: u8 = 0;
        let mut sample_tags: Vec<String> = Vec::new();
        for it in &items {
            if it.timestamp.as_str() < ts_min {
                ts_min = it.timestamp.as_str();
            }
            if it.timestamp.as_str() > ts_max {
                ts_max = it.timestamp.as_str();
            }
            let p = it.priority.unwrap_or(100);
            if p > max_prio {
                max_prio = p;
            }
            if sample_tags.is_empty() && !it.tags.is_empty() {
                sample_tags = it.tags.clone();
            }
        }

        // 摘要正文：取每条 note 的前 80 字符，最多拼 5 条，用 "; " 连接
        const PER_ITEM_CHARS: usize = 80;
        const MAX_SAMPLES: usize = 5;
        let mut samples: Vec<String> = Vec::new();
        for it in items.iter().take(MAX_SAMPLES) {
            let snippet: String = it.note.chars().take(PER_ITEM_CHARS).collect();
            samples.push(snippet);
        }
        let extra = count.saturating_sub(samples.len());
        let body = if extra > 0 {
            format!("{}; …(+{} more)", samples.join("; "), extra)
        } else {
            samples.join("; ")
        };

        let header = format!(
            "[summary] {} entries in '{}' aged out of {}d window ({}..{}): ",
            count, category, max_days, ts_min, ts_max
        );
        let note = format!("{}{}", header, body);

        let mut tags = sample_tags;
        if !tags.iter().any(|t| t == "summary") {
            tags.push("summary".to_string());
        }

        out.push(AgentMemoryEntry {
            id: Some(next_memory_id()),
            timestamp: now_iso.clone(),
            category,
            note,
            tags,
            source,
            priority: Some(max_prio.max(150)),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        });
    }
    out
}

fn parse_rfc3339_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// 把 sub-agent 的私有 memory 文件按白名单 merge 回主 memory 文件。
///
/// - `private_path`：sub-agent 的 jsonl（make_subagent_memory_path 生成）
/// - `main_path`：主 agent 的 memory jsonl（resolve 时不能再走 task_local override，
///   所以由调用方传入实际目标路径）
///
/// 仅 `is_permanent_memory` 命中的条目（safety/preference/coding_guideline/
/// project_memory/...）会被 append；普通对话级 task_event 留在私有文件，不污染主记忆。
/// 写入复用 MemoryStore::append（自带 lock + index upsert）。
pub(crate) fn merge_subagent_whitelist(
    private_path: &std::path::Path,
    main_path: &std::path::Path,
) -> Result<usize, String> {
    if !private_path.exists() {
        return Ok(0);
    }
    let entries = load_memory_entries(private_path)?;
    if entries.is_empty() {
        return Ok(0);
    }
    let main_store =
        crate::ai::tools::storage::memory_store::store_for_path(main_path.to_path_buf());
    let mut merged = 0usize;
    for entry in entries {
        if !is_permanent_memory(&entry) {
            continue;
        }
        // 把 owner pid/pgid 重置：sub-agent 可能已经退出，
        // 主 store 重新打 owner tag 没意义，留 None 即可。
        let mut e = entry;
        e.owner_pid = None;
        e.owner_pgid = None;
        if main_store.append(&e).is_ok() {
            merged += 1;
        }
    }
    crate::ai::tools::storage::memory_store::trace_memory_event(
        "memory.subagent_merge",
        "merged sub-agent whitelist entries back to main memory",
        &[
            ("private", private_path.display().to_string()),
            ("main", main_path.display().to_string()),
            ("merged", merged.to_string()),
        ],
    );
    Ok(merged)
}

pub(crate) fn execute_memory_dedup(_args: &Value) -> Result<String, String> {
    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Ok("No memory file".to_string());
        }
        let entries = load_memory_entries(&path)?;
        if entries.is_empty() {
            return Ok("No entries".to_string());
        }
        let total_before = entries.len();

        // Step 1: 严格去重——note + category + tags + source 完全一致的，
        // 只保留时间最新（reverse 遍历后写回）。
        let mut seen: SkipSet<(String, String, Vec<String>, Option<String>)> = SkipSet::default();
        let mut deduped: Vec<AgentMemoryEntry> = Vec::with_capacity(entries.len());
        for e in entries.into_iter().rev() {
            let key = (
                e.note.clone(),
                e.category.clone(),
                {
                    let mut t = e.tags.clone();
                    t.sort();
                    t
                },
                e.source.clone(),
            );
            if seen.insert(key) {
                deduped.push(e);
            }
        }
        deduped.reverse();
        let exact_dedup_removed = total_before.saturating_sub(deduped.len());

        // Step 2: 同 category 内 cosine ≥ 0.85 的语义重复合并保留较新一条。
        // 复用 ai::knowledge::indexing::embedder + similarity::cosine_similarity，
        // embedding 不可用时（fastembed 模型未就绪）静默退化为只做严格去重。
        let cosine_threshold = 0.85f32;
        let texts: Vec<String> = deduped
            .iter()
            .map(|e| format!("[{}] {}", e.category, e.note))
            .collect();
        let mut cosine_removed = 0usize;
        let final_entries =
            if let Some(vectors) = crate::ai::knowledge::indexing::embedder::embed_texts(&texts) {
                // 时间排序：保留较新的优先策略 = 先按 timestamp 降序遍历，
                // 若与已保留集合中同 category 的某条 cosine ≥ 阈值，则丢弃当前。
                use crate::ai::knowledge::indexing::similarity::cosine_similarity;
                let mut indexed: Vec<usize> = (0..deduped.len()).collect();
                indexed.sort_by(|&a, &b| deduped[b].timestamp.cmp(&deduped[a].timestamp));

                // kept: Vec<(原 idx, 同 cat 标记)>；用 idx 引用 deduped/vectors 避免 clone embedding
                let mut kept_idx: Vec<usize> = Vec::with_capacity(deduped.len());
                for &i in &indexed {
                    let cat_i = deduped[i].category.as_str();
                    let v_i = &vectors[i];
                    let mut merged = false;
                    for &j in &kept_idx {
                        if deduped[j].category != cat_i {
                            continue;
                        }
                        let sim = cosine_similarity(v_i, &vectors[j]);
                        if sim >= cosine_threshold {
                            merged = true;
                            break;
                        }
                    }
                    if merged {
                        cosine_removed += 1;
                    } else {
                        kept_idx.push(i);
                    }
                }
                // 还原原始顺序（按原 idx 升序）
                kept_idx.sort();
                let mut out: Vec<AgentMemoryEntry> = Vec::with_capacity(kept_idx.len());
                // 用 swap_remove 思路不行（会乱序），直接 clone：
                for i in kept_idx {
                    out.push(deduped[i].clone());
                }
                out
            } else {
                deduped
            };

        write_memory_entries(&path, &final_entries)?;

        crate::ai::tools::storage::memory_store::trace_memory_event(
            "memory.dedup",
            "dedup pass completed (exact + cosine)",
            &[
                ("total_before", total_before.to_string()),
                ("exact_removed", exact_dedup_removed.to_string()),
                ("cosine_removed", cosine_removed.to_string()),
                ("kept", final_entries.len().to_string()),
                ("threshold", cosine_threshold.to_string()),
            ],
        );

        Ok(format!(
            "Dedup done: {} -> {} (exact: {}, cosine≥{}: {})",
            total_before,
            final_entries.len(),
            exact_dedup_removed,
            cosine_threshold,
            cosine_removed
        ))
    })
}

pub(crate) fn execute_memory_update(args: &Value) -> Result<String, String> {
    let id = args["id"].as_str().ok_or("Missing id")?.trim();
    if id.is_empty() {
        return Err("id is empty".to_string());
    }

    let has_content = args.get("content").is_some();
    let has_category = args.get("category").is_some();
    let has_tags = args.get("tags").is_some();
    let has_source = args.get("source").is_some();
    let has_priority = args.get("priority").is_some();
    if !has_content && !has_category && !has_tags && !has_source && !has_priority {
        return Err("no fields to update".to_string());
    }

    let new_content = if has_content {
        let value = args["content"]
            .as_str()
            .ok_or("content must be a string")?
            .trim();
        if value.is_empty() {
            return Err("content is empty".to_string());
        }
        Some(value.to_string())
    } else {
        None
    };
    let new_category = if has_category {
        let value = args["category"]
            .as_str()
            .ok_or("category must be a string")?
            .trim();
        if value.is_empty() {
            return Err("category is empty".to_string());
        }
        Some(value.to_string())
    } else {
        None
    };
    let new_tags = if has_tags {
        Some(parse_string_array(&args["tags"]))
    } else {
        None
    };
    let new_source = if has_source {
        match args["source"].as_str() {
            Some(value) => {
                let value = value.trim();
                if value.is_empty() {
                    Some(None)
                } else {
                    Some(Some(value.to_string()))
                }
            }
            None if args["source"].is_null() => Some(None),
            None => return Err("source must be a string or null".to_string()),
        }
    } else {
        None
    };
    let new_priority = if has_priority {
        let priority = args["priority"]
            .as_u64()
            .ok_or("priority must be an integer")?;
        if priority > u8::MAX as u64 {
            return Err("priority out of range".to_string());
        }
        Some(priority as u8)
    } else {
        None
    };

    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Err("No memory file".to_string());
        }

        let mut entries = load_memory_entries(&path)?;

        let Some(entry) = entries
            .iter_mut()
            .rev()
            .find(|entry| entry.id.as_deref() == Some(id))
        else {
            return Err(format!("memory id not found: {id}"));
        };

        if let Some(content) = new_content.as_ref() {
            entry.note = content.clone();
        }
        if let Some(category) = new_category.as_ref() {
            entry.category = category.clone();
        }
        if let Some(tags) = new_tags.as_ref() {
            entry.tags = tags.clone();
        }
        if let Some(source) = new_source.as_ref() {
            entry.source = source.clone();
        }
        if let Some(priority) = new_priority {
            entry.priority = Some(priority);
        }
        entry.timestamp = Local::now().to_rfc3339();

        write_memory_entries(&path, &entries)?;

        Ok(format!("Memory updated: {} (id: {})", path.display(), id))
    })
}

pub(crate) fn execute_memory_delete(args: &Value) -> Result<String, String> {
    let id = args["id"].as_str().ok_or("Missing id")?.trim();
    if id.is_empty() {
        return Err("id is empty".to_string());
    }

    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Err("No memory file".to_string());
        }

        let mut entries = load_memory_entries(&path)?;
        let before_len = entries.len();
        entries.retain(|entry| entry.id.as_deref() != Some(id));
        if entries.len() == before_len {
            return Err(format!("memory id not found: {id}"));
        }

        write_memory_entries(&path, &entries)?;
        Ok(format!("Memory deleted: {} (id: {})", path.display(), id))
    })
}

/// 修改一条已存在的 memo：定位 target（优先 id，否则时间戳+原文精确匹配），
/// 把内容替换为 `new_note`，保留原 id，timestamp 更新为当前时间（体现最近被编辑）。
/// 只改第一条匹配项，避免误改重复内容。
pub(crate) fn update_memo_entry(
    target: &AgentMemoryEntry,
    new_note: &str,
) -> Result<String, String> {
    let new_note = new_note.trim();
    if new_note.is_empty() {
        return Err("new note is empty".to_string());
    }
    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    let target_id = target.id.clone().filter(|s| !s.is_empty());
    let target_ts = target.timestamp.clone();
    let target_note = target.note.clone();

    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Err("No memory file".to_string());
        }
        let mut entries = load_memory_entries(&path)?;
        let mut updated = false;
        for e in entries.iter_mut() {
            let hit = if let Some(id) = target_id.as_deref() {
                e.id.as_deref() == Some(id)
            } else {
                e.id.as_deref().map(|s| s.is_empty()).unwrap_or(true)
                    && e.timestamp == target_ts
                    && e.note == target_note
            };
            if hit {
                e.note = new_note.to_string();
                e.timestamp = Local::now().to_rfc3339();
                updated = true;
                break;
            }
        }
        if !updated {
            return Err("matching memo entry not found".to_string());
        }
        write_memory_entries(&path, &entries)?;
        Ok(format!("Memory updated: {}", path.display()))
    })
}

/// 删除一条 memo 条目：优先按 id 匹配；若条目没有 id（历史数据），
/// 则按 (timestamp, note) 精确匹配删除。返回删除结果描述。
pub(crate) fn delete_memo_entry(target: &AgentMemoryEntry) -> Result<String, String> {
    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    let target_id = target.id.clone().filter(|s| !s.is_empty());
    let target_ts = target.timestamp.clone();
    let target_note = target.note.clone();

    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Err("No memory file".to_string());
        }
        let mut entries = load_memory_entries(&path)?;
        let before_len = entries.len();

        if let Some(id) = target_id.as_deref() {
            entries.retain(|e| e.id.as_deref() != Some(id));
        } else {
            // 无 id：按时间戳 + 内容精确匹配，只删第一条匹配项，避免误删重复内容。
            let mut removed = false;
            entries.retain(|e| {
                if !removed
                    && e.id.as_deref().map(|s| s.is_empty()).unwrap_or(true)
                    && e.timestamp == target_ts
                    && e.note == target_note
                {
                    removed = true;
                    false
                } else {
                    true
                }
            });
        }

        if entries.len() == before_len {
            return Err("matching memo entry not found".to_string());
        }
        write_memory_entries(&path, &entries)?;
        Ok(format!("Memory deleted: {}", path.display()))
    })
}

/// 用户主动保存记忆到全局 memory store
pub(crate) fn execute_memory_save(args: &Value) -> Result<String, String> {
    let content = args["content"].as_str().ok_or("Missing content")?.trim();
    if content.is_empty() {
        return Err("content is empty".to_string());
    }

    let requested_category = normalized_category(args["category"].as_str(), "self_note");
    let mut tags = args["tags"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_else(|| vec!["agent".to_string(), "memory_save".to_string()]);

    let mut source = args["source"]
        .as_str()
        .map(|s| s.trim().to_string())
        .or(Some("agent_memory_save".to_string()));

    let requested_priority = parse_priority_arg(args, "priority")?
        .or_else(|| Some(default_priority_for_category(&requested_category)));
    let (category, priority, assessment) = maybe_downgrade_memory_save(
        &requested_category,
        content,
        &mut tags,
        &mut source,
        requested_priority,
    );
    let downgraded = category != requested_category;
    let (owner_pid, owner_pgid) = current_owner_tags();
    let entry = AgentMemoryEntry {
        id: Some(next_memory_id()),
        timestamp: Local::now().to_rfc3339(),
        category,
        note: content.to_string(),
        tags,
        source,
        priority,
        owner_pid,
        owner_pgid,
        image_path: None,
    };

    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;
    crate::ai::driver::decision_log::log_memory_save_assessment(
        crate::ai::driver::decision_log::get_decision_log_store(),
        &crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
        crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
        &requested_category,
        &entry.category,
        content,
        &assessment,
        downgraded,
    );
    Ok(format!(
        "Memory saved: {} (category: {}, id: {})",
        store.path().display(),
        entry.category,
        entry.id.as_deref().unwrap_or("")
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        execute_memory_delete, execute_memory_list_json, execute_memory_save,
        execute_memory_update, is_memory_visible_to, update_memo_entry,
    };
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
    use aios_kernel::kernel::{KernelInternal, Syscall};
    use chrono::Local;

    #[test]
    fn memory_save_assigns_id_and_update_rewrites_entry() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_memory_update_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_args = serde_json::json!({
            "content": "delete carefully",
            "category": "safety_rules",
            "tags": ["safety"],
            "source": "test",
            "priority": 255
        });
        let save_msg = execute_memory_save(&save_args).unwrap();
        assert!(save_msg.contains("id: mem_"));

        let list_before = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let before: serde_json::Value = serde_json::from_str(&list_before).unwrap();
        let id = before[0]["id"].as_str().unwrap().to_string();
        assert_eq!(before[0]["note"].as_str().unwrap(), "delete carefully");

        let update_args = serde_json::json!({
            "id": id,
            "content": "delete carefully after confirmation",
            "priority": 200,
            "tags": ["safety", "confirmed"]
        });
        let update_msg = execute_memory_update(&update_args).unwrap();
        assert!(update_msg.contains("Memory updated"));

        let list_after = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let after: serde_json::Value = serde_json::from_str(&list_after).unwrap();
        assert_eq!(
            after[0]["note"].as_str().unwrap(),
            "delete carefully after confirmation"
        );
        assert_eq!(after[0]["priority"].as_u64().unwrap(), 200);
        assert_eq!(after[0]["tags"].as_array().unwrap().len(), 2);

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn memory_save_common_sense_defaults_to_persistent_priority_and_can_delete() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_memory_delete_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_args = serde_json::json!({
            "content": "Files should end with a trailing newline",
            "category": "common_sense",
            "tags": ["editing"]
        });
        execute_memory_save(&save_args).unwrap();

        let list_before = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let before: serde_json::Value = serde_json::from_str(&list_before).unwrap();
        let id = before[0]["id"].as_str().unwrap().to_string();
        assert_eq!(before[0]["category"].as_str().unwrap(), "common_sense");
        assert_eq!(before[0]["priority"].as_u64().unwrap(), 210);

        let delete_args = serde_json::json!({ "id": id });
        let delete_msg = execute_memory_delete(&delete_args).unwrap();
        assert!(delete_msg.contains("Memory deleted"));

        let list_after = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let after: serde_json::Value = serde_json::from_str(&list_after).unwrap();
        assert!(after.as_array().unwrap().is_empty());

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn memo_update_preserves_id_and_replaces_note() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_memory_update_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_args = serde_json::json!({
            "content": "ida 交接文档 原始内容",
            "category": "memo",
            "tags": ["handoff"]
        });
        execute_memory_save(&save_args).unwrap();

        let list_before = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let before: serde_json::Value = serde_json::from_str(&list_before).unwrap();
        let id = before[0]["id"].as_str().unwrap().to_string();

        let target = AgentMemoryEntry {
            id: Some(id.clone()),
            timestamp: before[0]["timestamp"].as_str().unwrap().to_string(),
            category: "memo".to_string(),
            note: "ida 交接文档 原始内容".to_string(),
            tags: vec!["handoff".to_string()],
            source: None,
            priority: None,
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let msg = update_memo_entry(&target, "ida 交接文档 更新后的内容").unwrap();
        assert!(msg.contains("Memory updated"));

        let list_after = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let after: serde_json::Value = serde_json::from_str(&list_after).unwrap();
        // id 保持不变；内容被替换；条目数不变。
        assert_eq!(after.as_array().unwrap().len(), 1);
        assert_eq!(after[0]["id"].as_str().unwrap(), id);
        assert_eq!(
            after[0]["note"].as_str().unwrap(),
            "ida 交接文档 更新后的内容"
        );

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn memory_save_defaults_to_short_term_self_note() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_memory_save_default_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_args = serde_json::json!({
            "content": "Prefer code_search before repeated raw reads"
        });
        execute_memory_save(&save_args).unwrap();

        let list = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let items: serde_json::Value = serde_json::from_str(&list).unwrap();
        assert_eq!(items[0]["category"].as_str().unwrap(), "self_note");

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn memory_save_downgrades_low_signal_long_term_entries_to_self_note() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_memory_save_downgrade_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_args = serde_json::json!({
            "content": "be careful",
            "category": "common_sense"
        });
        execute_memory_save(&save_args).unwrap();

        let list = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let items: serde_json::Value = serde_json::from_str(&list).unwrap();
        assert_eq!(items[0]["category"].as_str().unwrap(), "self_note");
        assert!(
            items[0]["tags"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_str() == Some("auto_downgraded"))
        );
        assert_eq!(items[0]["priority"].as_u64().unwrap(), 120);

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn memory_visibility_unowned_entries_visible_to_all() {
        let entry = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "public note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        assert!(is_memory_visible_to(&entry, None));
        assert!(is_memory_visible_to(&entry, Some(1)));
        assert!(is_memory_visible_to(&entry, Some(999)));
    }

    #[test]
    fn memory_visibility_owner_sees_own_entries() {
        let entry = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "my note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(42),
            owner_pgid: None,
            image_path: None,
        };
        assert!(is_memory_visible_to(&entry, Some(42)));
        assert!(!is_memory_visible_to(&entry, Some(99)));
    }

    #[test]
    fn memory_visibility_foreground_sees_all() {
        let entry = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "tagged note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(42),
            owner_pgid: Some(10),
            image_path: None,
        };
        assert!(is_memory_visible_to(&entry, None));
    }

    #[test]
    fn memory_visibility_same_process_group() {
        let _guard = ENV_LOCK.lock().unwrap();
        let kernel = crate::ai::driver::new_local_kernel();
        let (root, child_a, child_b) = {
            let mut os = kernel.lock().unwrap();
            let root =
                os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
            let child_a = os
                .spawn(
                    Some(root),
                    "a".to_string(),
                    "goal a".to_string(),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();
            let child_b = os
                .spawn(
                    Some(root),
                    "b".to_string(),
                    "goal b".to_string(),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();
            os.set_process_group(child_a, 100).unwrap();
            os.set_process_group(child_b, 100).unwrap();
            (root, child_a, child_b)
        };
        crate::ai::tools::os_tools::init_os_tools_globals(kernel.clone());

        let entry_a = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "a's note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(child_a),
            owner_pgid: Some(100),
            image_path: None,
        };

        assert!(is_memory_visible_to(&entry_a, Some(child_b)));

        {
            if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
                *guard = None;
            }
        }
    }

    #[test]
    fn memory_visibility_ancestor_sees_descendant() {
        let _guard = ENV_LOCK.lock().unwrap();
        let kernel = crate::ai::driver::new_local_kernel();
        let (root, child) = {
            let mut os = kernel.lock().unwrap();
            let root =
                os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
            let child = os
                .spawn(
                    Some(root),
                    "child".to_string(),
                    "goal child".to_string(),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();
            (root, child)
        };
        crate::ai::tools::os_tools::init_os_tools_globals(kernel.clone());

        let child_entry = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "child's secret".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(child),
            owner_pgid: None,
            image_path: None,
        };

        assert!(is_memory_visible_to(&child_entry, Some(root)));

        {
            if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
                *guard = None;
            }
        }
    }

    #[test]
    fn memory_visibility_unrelated_process_blocked() {
        let _guard = ENV_LOCK.lock().unwrap();
        let kernel = crate::ai::driver::new_local_kernel();
        let (child_a, child_b) = {
            let mut os = kernel.lock().unwrap();
            let root1 =
                os.begin_foreground("fg1".to_string(), "goal1".to_string(), 10, usize::MAX, None);
            let child_a = os
                .spawn(
                    Some(root1),
                    "a".to_string(),
                    "goal a".to_string(),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();

            let root2 =
                os.begin_foreground("fg2".to_string(), "goal2".to_string(), 10, usize::MAX, None);
            let child_b = os
                .spawn(
                    Some(root2),
                    "b".to_string(),
                    "goal b".to_string(),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();
            (child_a, child_b)
        };
        crate::ai::tools::os_tools::init_os_tools_globals(kernel.clone());

        let entry_a = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "a's private note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(child_a),
            owner_pgid: None,
            image_path: None,
        };

        assert!(!is_memory_visible_to(&entry_a, Some(child_b)));

        {
            if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
                *guard = None;
            }
        }
    }
}
