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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryOwnerScope {
    Scoped,
    Global,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedMemorySave {
    pub(crate) requested_category: String,
    pub(crate) downgraded: bool,
    pub(crate) assessment: crate::ai::driver::reflection::LearningNoteAssessment,
    pub(crate) entry: AgentMemoryEntry,
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

pub(crate) struct ViewerContext {
    viewer_pid: Option<u64>,
    viewer_pgid: Option<u64>,
}

impl ViewerContext {
    pub(crate) fn current() -> Self {
        let (pid, pgid) = current_owner_tags();
        Self {
            viewer_pid: pid,
            viewer_pgid: pgid,
        }
    }

    pub(crate) fn can_see(&self, entry: &AgentMemoryEntry) -> bool {
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

pub(crate) fn next_memory_id() -> String {
    format!("mem_{}", Uuid::new_v4().simple())
}

/// Model attribution for a newly written memory entry: the model driving the
/// current turn, or `None` outside a turn (CLI helpers, tests) where no model
/// produced the content.
pub(crate) fn current_write_model() -> Option<String> {
    crate::ai::driver::runtime_ctx::try_current()
        .map(|ctx| ctx.app_proto.current_model.clone())
        .filter(|model| !model.trim().is_empty())
}

/// Provenance header for a losslessly merged source entry: keeps the source's
/// id/category/timestamp/origin/model with its full text, so consolidation
/// never drops the when/who behind the content it replaces.
pub(crate) fn format_preserved_source(src: &AgentMemoryEntry, trimmed_note: &str) -> String {
    let mut header = format!(
        "[id: {} | category: {} | timestamp: {}",
        src.id.as_deref().unwrap_or("unknown"),
        src.category,
        src.timestamp,
    );
    if let Some(source) = src.source.as_deref().filter(|s| !s.trim().is_empty()) {
        header.push_str(&format!(" | source: {source}"));
    }
    if let Some(model) = src.model.as_deref().filter(|s| !s.trim().is_empty()) {
        header.push_str(&format!(" | model: {model}"));
    }
    header.push(']');
    format!("{header}\n{trimmed_note}")
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

fn maybe_downgrade_long_term_save(
    category: &str,
    note: &str,
    tags: &mut Vec<String>,
    source: &mut Option<String>,
    priority: Option<u8>,
    downgrade_marker: &str,
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
        if !src.contains(downgrade_marker) {
            src.push(':');
            src.push_str(downgrade_marker);
        }
    } else {
        *source = Some(downgrade_marker.to_string());
    }
    (
        "self_note".to_string(),
        Some(priority.unwrap_or(120).min(120)),
        assessment,
    )
}

/// Classify entries exempt from age/quota filtering. Guideline categories and `project_memory`
/// supplement explicit permanent priority so durable facts do not require timestamp refresh.
/// `self_note` has no category-based exemption.
pub(crate) fn is_permanent_memory(entry: &AgentMemoryEntry) -> bool {
    if entry.priority.unwrap_or(100) == 255 {
        return true;
    }
    if crate::ai::knowledge::types::Category::from_str(&entry.category).is_guideline() {
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

pub(crate) fn prepare_memory_save_entry(
    args: &Value,
    fallback_category: &str,
    default_tags: &[&str],
    default_source: &str,
    owner_scope: MemoryOwnerScope,
    downgrade_marker: &str,
) -> Result<PreparedMemorySave, String> {
    let content = args["content"].as_str().ok_or("Missing content")?.trim();
    if content.is_empty() {
        return Err("content is empty".to_string());
    }

    let requested_category = normalized_category(args["category"].as_str(), fallback_category);
    let mut tags = parse_string_array(&args["tags"]);
    if tags.is_empty() {
        tags = default_tags.iter().map(|tag| (*tag).to_string()).collect();
    }

    let mut source = args["source"]
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| (!default_source.trim().is_empty()).then(|| default_source.trim().to_string()));

    let requested_priority = parse_priority_arg(args, "priority")?
        .or_else(|| Some(default_priority_for_category(&requested_category)));
    let (category, priority, assessment) = maybe_downgrade_long_term_save(
        &requested_category,
        content,
        &mut tags,
        &mut source,
        requested_priority,
        downgrade_marker,
    );
    let downgraded = category != requested_category;
    let (owner_pid, owner_pgid) = match owner_scope {
        MemoryOwnerScope::Scoped => current_owner_tags(),
        MemoryOwnerScope::Global => (None, None),
    };

    Ok(PreparedMemorySave {
        requested_category,
        downgraded,
        assessment,
        entry: AgentMemoryEntry {
            id: Some(next_memory_id()),
            timestamp: Local::now().to_rfc3339(),
            category,
            note: content.to_string(),
            tags,
            source,
            distilled: None,
            priority,
            owner_pid,
            owner_pgid,
            image_path: None,
            model: current_write_model(),
        },
    })
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

/// A scored memo search result.
pub(crate) struct ScoredMemo {
    pub entry: AgentMemoryEntry,
    pub score: f64,
}

/// Searches memo candidate entries by query text and returns structured entries (sorted by relevance).
/// Used by the `-nd` delete flow: lets the upper layer pick the best-matching entry with the model before confirming deletion.
pub(crate) fn search_memo_candidates(
    query: &str,
    limit: usize,
    include_archives: bool,
) -> Result<Vec<AgentMemoryEntry>, String> {
    Ok(
        search_memo_candidates_scored(query, limit, include_archives)?
            .into_iter()
            .map(|s| s.entry)
            .collect(),
    )
}

/// Same source as `search_memo_candidates`, but keeps the scores.
pub(crate) fn search_memo_candidates_scored(
    query: &str,
    limit: usize,
    include_archives: bool,
) -> Result<Vec<ScoredMemo>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Err("query is empty".to_string());
    }
    let limit = limit.clamp(1, 50);
    let store = MemoryStore::from_env_or_config();
    // Notes saved via `a -n` are always memo entries; -ns / -nd / -ne only handle this kind.
    // With include_archives=false, only the current memory file is read: delete/edit can only rewrite the current file,
    // and archive entries are excluded from candidates even when ai.memory.search_archives.enable is on globally,
    // otherwise deleting/updating a selected archive entry would fail with "matching memo entry not found".
    let results: Vec<AgentMemoryEntry> = if include_archives {
        store.entries_by_category("memo", 100_000, true)?
    } else {
        load_memory_entries(store.path())?
            .into_iter()
            .filter(|e| e.category == "memo")
            .collect()
    };
    let viewer = ViewerContext::current();
    let qlc = query.to_lowercase();
    let query_tokens = crate::ai::knowledge::indexing::similarity::expand_tokens(
        &crate::ai::knowledge::indexing::similarity::tokenize(&qlc),
    );

    let visible: Vec<AgentMemoryEntry> =
        results.into_iter().filter(|e| viewer.can_see(e)).collect();

    // Score note/tag phrase matches plus expanded-token overlap across note, tags, and source.
    let lexical = |e: &AgentMemoryEntry| -> f64 {
        let mut score = 0.0_f64;
        if e.note.to_lowercase().contains(&qlc) {
            score += 3.0;
            score += (qlc.len() as f64).min(20.0) * 0.05;
        }
        if e.tags.iter().any(|t| t.to_lowercase().contains(&qlc)) {
            score += 1.2;
        }

        // Cannot rely only on contiguous substrings of the whole query: users often add project names or
        // descriptive words to search terms, while old notes cover only part of them. Add score by token coverage,
        // while full phrase hits keep the highest priority.
        if !query_tokens.is_empty() {
            let mut searchable = e.note.to_lowercase();
            if !e.tags.is_empty() {
                searchable.push(' ');
                searchable.push_str(&e.tags.join(" ").to_lowercase());
            }
            if let Some(source) = &e.source {
                searchable.push(' ');
                searchable.push_str(&source.to_lowercase());
            }
            let entry_tokens = crate::ai::knowledge::indexing::similarity::expand_tokens(
                &crate::ai::knowledge::indexing::similarity::tokenize(&searchable),
            );
            let overlap = query_tokens
                .iter()
                .filter(|token| entry_tokens.contains(*token))
                .count();
            score += 2.5 * overlap as f64 / query_tokens.len() as f64;
        }
        score
    };

    let mut scored: Vec<ScoredMemo> = Vec::with_capacity(visible.len());
    for e in visible {
        let score = lexical(&e);
        scored.push(ScoredMemo { entry: e, score });
    }
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    scored.truncate(limit);
    Ok(scored)
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

        // Separate entries exempt from age filtering.
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

        // Summarize entries removed by the age filter so their category/source retain a bounded sample.
        // Summaries get fresh timestamps but keep their categories, so later GC passes may expire them again.
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

        deletable.sort_by(|a, b| {
            let prio_a = a.priority.unwrap_or(100);
            let prio_b = b.priority.unwrap_or(100);
            prio_a.cmp(&prio_b).then_with(|| {
                let ts_a = parse_rfc3339_ts(&a.timestamp);
                let ts_b = parse_rfc3339_ts(&b.timestamp);
                ts_a.cmp(&ts_b)
            })
        });

        // If age filtering leaves too few entries, reload a recent window from the store.
        let total_permanent = permanent.len();
        if deletable.len() + total_permanent + summary_count < min_keep {
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

/// Group expired entries by (category, source) and concatenate bounded note samples.
/// Preserve category/source and the first nonempty tag set, add the summary tag, and
/// raise the group's highest priority to the summary minimum. Fresh timestamps defer
/// later age filtering; they do not make the summaries permanent.
///
/// Use local concatenation so GC never waits for an external model.
fn build_gc_summaries(evicted: &[AgentMemoryEntry], max_days: i64) -> Vec<AgentMemoryEntry> {
    use std::collections::BTreeMap;

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
            distilled: None,
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
            model: None,
        });
    }
    out
}

fn parse_rfc3339_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Merges a sub-agent's private memory file back into the main memory file according to the whitelist.
///
/// - `private_path`: the sub-agent's jsonl (generated by make_subagent_memory_path)
/// - `main_path`: the main agent's memory jsonl (resolution must not go through the task_local override again,
///   so the caller passes in the actual target path)
///
/// Only entries matching `is_permanent_memory` (safety/preference/coding_guideline/
/// project_memory/...) are appended; ordinary conversational task_event entries stay in the private file and do not pollute main memory.
/// Writes reuse MemoryStore::append (which brings its own lock + index upsert).
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
        // Clear child ownership so merged long-term entries remain visible after the child exits.
        let mut e = entry;
        e.owner_pid = None;
        e.owner_pgid = None;
        main_store.append(&e).map_err(|error| {
            format!(
                "failed to merge subagent memory entry {} into {}: {error}",
                e.id.as_deref().unwrap_or("<without-id>"),
                main_path.display()
            )
        })?;
        merged += 1;
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

        // Deduplicate by note, category, source, and sorted tags, retaining the last stored occurrence.
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

        let final_entries = deduped;

        write_memory_entries(&path, &final_entries)?;

        crate::ai::tools::storage::memory_store::trace_memory_event(
            "memory.dedup",
            "dedup pass completed (exact)",
            &[
                ("total_before", total_before.to_string()),
                ("exact_removed", exact_dedup_removed.to_string()),
                ("kept", final_entries.len().to_string()),
            ],
        );

        Ok(format!(
            "Dedup done: {} -> {} (exact: {})",
            total_before,
            final_entries.len(),
            exact_dedup_removed,
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

        let viewer = ViewerContext::current();
        let idx = match entries
            .iter()
            .rposition(|entry| entry.id.as_deref() == Some(id))
        {
            Some(idx) => idx,
            None => return Err(format!("memory id not found: {id}")),
        };
        if !viewer.can_see(&entries[idx]) {
            return Err("Permission denied: entry owned by another agent".to_string());
        }
        let entry = &mut entries[idx];

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
        let viewer = ViewerContext::current();
        let denied = match entries.iter().find(|entry| entry.id.as_deref() == Some(id)) {
            Some(entry) => !viewer.can_see(entry),
            None => return Err(format!("memory id not found: {id}")),
        };
        if denied {
            return Err("Permission denied: entry owned by another agent".to_string());
        }

        let before_len = entries.len();
        entries.retain(|entry| entry.id.as_deref() != Some(id));
        if entries.len() == before_len {
            return Err(format!("memory id not found: {id}"));
        }

        write_memory_entries(&path, &entries)?;
        Ok(format!("Memory deleted: {} (id: {})", path.display(), id))
    })
}

/// Edits an existing memo: locates the target (by id first, otherwise exact match on timestamp + original text),
/// replaces the content with `new_note`, keeps the original id, and updates the timestamp to now (reflecting the recent edit).
/// Only the first match is modified, to avoid touching duplicated content by mistake.
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

/// Delete entries matching the target id; without an id (legacy data), remove only the
/// first exact (timestamp, note) match. Returns a description of the deletion result.
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
            // No id: exact match on timestamp + content; delete only the first match to avoid removing duplicates by mistake.
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

/// Prepare and save a scoped memory entry with a learning-quality assessment.
pub(crate) fn execute_memory_save(args: &Value) -> Result<String, String> {
    let prepared = prepare_memory_save_entry(
        args,
        "self_note",
        &["agent", "memory_save"],
        "agent_memory_save",
        MemoryOwnerScope::Scoped,
        "memory_save_downgraded",
    )?;
    let crate::ai::tools::service::memory::PreparedMemorySave {
        requested_category,
        downgraded,
        assessment,
        entry,
    } = prepared;
    let content = entry.note.clone();
    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;
    crate::ai::driver::decision_log::log_memory_save_assessment(
        crate::ai::driver::decision_log::get_decision_log_store(),
        &crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
        crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
        &requested_category,
        &entry.category,
        &content,
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
        delete_memo_entry, execute_memory_delete, execute_memory_list_json, execute_memory_save,
        execute_memory_update, is_memory_visible_to, search_memo_candidates,
        search_memo_candidates_scored, update_memo_entry,
    };
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
    use chrono::Local;
    use std::path::Path;
    use std::sync::MutexGuard;

    fn env_lock_guard() -> MutexGuard<'static, ()> {
        // This lock only serializes env-var / GLOBAL_OS related tests.
        // Even if a previous case poisons it via an assertion failure, later cases must not
        // be distorted into a PoisonError, which would mask the real first failure.
        ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    fn cleanup_memory_artifacts(path: &Path) {
        let db_path = path.with_extension("db");
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
        let _ = std::fs::remove_file(db_path.with_extension("db-shm"));
    }

    #[test]
    fn memory_save_assigns_id_and_update_rewrites_entry() {
        let _guard = env_lock_guard();
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
        let _guard = env_lock_guard();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_memory_delete_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_args = serde_json::json!({
            "content": "Files should end with a trailing newline before write_file commits the edit",
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
        let _guard = env_lock_guard();
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
            distilled: None,
            priority: None,
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
            model: None,
        };
        let msg = update_memo_entry(&target, "ida 交接文档 更新后的内容").unwrap();
        assert!(msg.contains("Memory updated"));

        let list_after = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let after: serde_json::Value = serde_json::from_str(&list_after).unwrap();
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
    fn memo_search_finds_archived_memo_and_excludes_non_memo_entries() {
        let _guard = env_lock_guard();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_notebook_search_{ts}.jsonl"));
        let archive = path.with_extension("jsonl.20260101000000");
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let non_memo_decoy = AgentMemoryEntry {
            id: Some("non-memo-decoy".to_string()),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            category: "project_memory".to_string(),
            note: "AeolusLLM Copilot 二次分析问题排查：项目自动写回记录。".to_string(),
            tags: vec!["aeolusllm".to_string(), "copilot".to_string()],
            source: Some("project_writeback:aeolus".to_string()),
            distilled: None,
            priority: Some(180),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
            model: None,
        };
        let archived_memo = AgentMemoryEntry {
            id: Some("archived-memo".to_string()),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            category: "memo".to_string(),
            note: "AeolusLLM Copilot 二次分析：通过 trace_id 在数据库检索原始问题。".to_string(),
            tags: vec!["aeolusllm".to_string(), "copilot".to_string()],
            source: Some("cli_note".to_string()),
            distilled: None,
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
            model: None,
        };
        let decoy_line = serde_json::to_string(&non_memo_decoy).unwrap();
        let archived_line = serde_json::to_string(&archived_memo).unwrap();
        std::fs::write(&archive, format!("{decoy_line}\n{archived_line}\n")).unwrap();

        let store = MemoryStore::from_env_or_config();
        store
            .append(&AgentMemoryEntry {
                id: Some("current-memo".to_string()),
                timestamp: "2026-01-02T00:00:00Z".to_string(),
                category: "memo".to_string(),
                note: "无关的手工笔记".to_string(),
                tags: vec![],
                source: Some("cli_note".to_string()),
                distilled: None,
                priority: Some(150),
                owner_pid: None,
                owner_pgid: None,
                image_path: None,
                model: None,
            })
            .unwrap();

        let memo_only =
            search_memo_candidates_scored("aeolusllm copilot 二次分析问题排查", 10, true).unwrap();
        assert_eq!(memo_only[0].entry.id.as_deref(), Some("archived-memo"));
        assert!(
            memo_only
                .iter()
                .all(|candidate| candidate.entry.category == "memo")
        );
        assert!(
            memo_only
                .iter()
                .all(|candidate| candidate.entry.id.as_deref() != Some("non-memo-decoy"))
        );

        cleanup_memory_artifacts(&path);
        let _ = std::fs::remove_file(&archive);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    /// Regression: the -nd/-ne candidate functions read only the current memory file when include_archives=false,
    /// excluding archive entries (delete/update can only rewrite the current file; otherwise they fail with "matching memo entry not found").
    #[test]
    fn memo_delete_candidates_exclude_archives_and_delete_fails_on_archived() {
        let _guard = env_lock_guard();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_notebook_delete_{ts}.jsonl"));
        let archive = path.with_extension("jsonl.20260101000000");
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let archived_memo = AgentMemoryEntry {
            id: Some("archived-del-memo".to_string()),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            category: "memo".to_string(),
            note: "归档中的待删笔记 二次分析问题排查".to_string(),
            tags: vec!["aeolusllm".to_string()],
            source: Some("cli_note".to_string()),
            distilled: None,
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
            model: None,
        };
        std::fs::write(
            &archive,
            format!("{}\n", serde_json::to_string(&archived_memo).unwrap()),
        )
        .unwrap();

        let store = MemoryStore::from_env_or_config();
        store
            .append(&AgentMemoryEntry {
                id: Some("current-del-memo".to_string()),
                timestamp: "2026-01-02T00:00:00Z".to_string(),
                category: "memo".to_string(),
                note: "当前文件中的笔记 二次分析问题排查".to_string(),
                tags: vec!["aeolusllm".to_string()],
                source: Some("cli_note".to_string()),
                distilled: None,
                priority: Some(150),
                owner_pid: None,
                owner_pgid: None,
                image_path: None,
                model: None,
            })
            .unwrap();

        // include_archives=false: candidates must not include archive entries,
        // leaving only "current-del-memo" from the current file.
        let candidates = search_memo_candidates("二次分析问题排查", 10, false).unwrap();
        assert!(
            candidates
                .iter()
                .all(|c| c.id.as_deref() != Some("archived-del-memo"))
        );
        assert!(
            candidates
                .iter()
                .any(|c| c.id.as_deref() == Some("current-del-memo"))
        );

        // Even when an archive entry is passed in externally, delete_memo_entry must fail because it cannot find it in the current file,
        // rather than silently deleting the wrong thing or falling through to a success path.
        let err = delete_memo_entry(&archived_memo).unwrap_err();
        assert_eq!(err, "matching memo entry not found");

        cleanup_memory_artifacts(&path);
        let _ = std::fs::remove_file(&archive);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn memory_save_defaults_to_short_term_self_note() {
        let _guard = env_lock_guard();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_memory_save_default_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_args = serde_json::json!({
            "content": "Prefer targeted reads over repeated raw reads"
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
        let _guard = env_lock_guard();
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
            distilled: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
            model: None,
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
            distilled: None,
            priority: Some(100),
            owner_pid: Some(42),
            owner_pgid: None,
            image_path: None,
            model: None,
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
            distilled: None,
            priority: Some(100),
            owner_pid: Some(42),
            owner_pgid: Some(10),
            image_path: None,
            model: None,
        };
        assert!(is_memory_visible_to(&entry, None));
    }

    #[test]
    fn memory_visibility_same_process_group() {
        let _guard = env_lock_guard();
        let kernel = crate::ai::driver::new_local_kernel();
        let (_root, child_a, child_b) = {
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
            distilled: None,
            priority: Some(100),
            owner_pid: Some(child_a),
            owner_pgid: Some(100),
            image_path: None,
            model: None,
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
        let _guard = env_lock_guard();
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
            distilled: None,
            priority: Some(100),
            owner_pid: Some(child),
            owner_pgid: None,
            image_path: None,
            model: None,
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
        let _guard = env_lock_guard();
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
            distilled: None,
            priority: Some(100),
            owner_pid: Some(child_a),
            owner_pgid: None,
            image_path: None,
            model: None,
        };

        assert!(!is_memory_visible_to(&entry_a, Some(child_b)));

        {
            if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
                *guard = None;
            }
        }
    }
}
