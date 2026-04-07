/// Knowledge recall — build guidelines and auto-recalled knowledge for prompts.
/// Uses the retrieval APIs instead of directly accessing stores.
use super::super::config::KnowledgeConfig;
use super::super::entry::KnowledgeEntry;
use super::super::indexing::similarity;
use super::super::storage::jsonl_store::JsonlStore;
use super::super::types::GuidelineGroup;
use super::keyword_search;

/// Result of auto-recalled knowledge.
pub struct AutoRecalledKnowledge {
    pub content: String,
    pub high_confidence_project_memory: bool,
    pub entry_count: usize,
    pub project_hint: Option<String>,
    pub categories: Vec<String>,
}

/// Build persistent guidelines for the system prompt.
pub fn build_persistent_guidelines(
    store: &JsonlStore,
    question: &str,
    max_chars: usize,
    config: &KnowledgeConfig,
) -> Option<String> {
    let max_days = config.maintenance.guidelines_max_days;

    let mut entries: Vec<KnowledgeEntry> = Vec::new();

    // Search for guideline categories
    for cat in guideline_categories() {
        if let Ok(results) = keyword_search::keyword_search(store, cat, 80, config) {
            for (e, score) in results {
                if score < config.thresholds.min_score_guideline {
                    continue;
                }
                if !e.category_enum().is_guideline() {
                    continue;
                }
                entries.push(e);
            }
        }
    }

    // Search for question-relevant guidelines
    if !question.trim().is_empty() {
        if let Ok(results) = keyword_search::keyword_search(store, question, 60, config) {
            for (e, score) in results {
                if score < 0.2 {
                    continue;
                }
                if !e.category_enum().is_guideline() {
                    continue;
                }
                entries.push(e);
            }
        }
    }

    // Fallback to recent high-priority guidelines
    if entries.is_empty() {
        if let Ok(recent) = store.recent(100) {
            entries = recent
                .into_iter()
                .filter(|e| e.category_enum().is_guideline())
                .filter(|e| e.priority_value() >= 150)
                .collect();
        }
    }

    if entries.is_empty() {
        return None;
    }

    // Deduplicate
    deduplicate_entries(&mut entries, config.thresholds.dedup_similarity_guideline);

    // Rank and group
    let mut ranked: Vec<(u8, u8, i64, KnowledgeEntry)> = Vec::with_capacity(entries.len());
    for e in entries {
        let cat = e.category_enum();
        let priority = e.priority_value();
        let group = GuidelineGroup::from_category(&cat).as_u8();

        if group >= 3 && priority < 200 {
            continue;
        }

        if max_days > 0 && group >= 2 {
            if let Some(dt) = parse_ts_utc(&e.timestamp) {
                let age_days = (chrono::Utc::now() - dt).num_seconds().max(0) as i64 / 86400;
                if age_days > max_days {
                    continue;
                }
            }
        }

        if e.note.trim().is_empty() {
            continue;
        }

        let ts_rank = parse_ts_utc(&e.timestamp)
            .map(|dt| dt.timestamp())
            .unwrap_or(0);
        ranked.push((group, priority, ts_rank, e));
    }

    ranked.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
    });

    let mut seen = rust_tools::cw::SkipSet::new(16);
    let mut by_group: [Vec<String>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (group, _priority, _ts, entry) in ranked {
        let note = entry.note.trim().to_string();
        if seen.insert(note.clone()) {
            let g = (group.min(3)) as usize;
            by_group[g].push(note);
        }
    }

    if by_group.iter().all(|v| v.is_empty()) {
        return None;
    }

    // Format output
    let mut out = String::from("Persistent Guidelines:\n");
    let mut used = out.len();
    if used >= max_chars {
        return Some(out);
    }

    let weights = config.maintenance.guidelines_group_weights;
    for group_idx in 0..4 {
        if by_group[group_idx].is_empty() {
            continue;
        }
        let mut budget = group_budget(max_chars, used, weights, group_idx);
        budget = budget.max(120).min(max_chars.saturating_sub(used));
        append_notes(&mut out, &mut used, max_chars, budget, &by_group[group_idx]);
        if used >= max_chars {
            break;
        }
    }

    if used <= "Persistent Guidelines:\n".len() {
        return None;
    }
    Some(out)
}

/// Build auto-recalled knowledge for the system prompt.
pub fn build_auto_recalled_knowledge(
    store: &JsonlStore,
    question: &str,
    max_chars: usize,
    config: &KnowledgeConfig,
) -> Option<AutoRecalledKnowledge> {
    let project_hint = current_project_hint();
    build_auto_recalled_knowledge_with_project(
        store,
        question,
        max_chars,
        project_hint.as_deref(),
        config,
    )
}

/// Build auto-recalled knowledge with explicit project hint.
pub fn build_auto_recalled_knowledge_with_project(
    store: &JsonlStore,
    question: &str,
    max_chars: usize,
    project_hint: Option<&str>,
    config: &KnowledgeConfig,
) -> Option<AutoRecalledKnowledge> {
    let project_hint = project_hint
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let mut entries: Vec<KnowledgeEntry> = Vec::new();
    let mut seen = rust_tools::cw::SkipSet::new(32);

    // Build query variants
    let mut query_variants = Vec::new();
    let q = question.trim();
    if !q.is_empty() {
        query_variants.push(q.to_string());
        if let Some(project) = project_hint.as_deref() {
            query_variants.push(format!("{q} {project}"));
        }
    }
    if let Some(project) = project_hint.as_deref() {
        query_variants.push(project.to_string());
    }

    // Search by query variants
    for query in &query_variants {
        if query.trim().is_empty() {
            continue;
        }
        if let Ok(results) = keyword_search::keyword_search(store, query, 30, config) {
            for (entry, score) in results {
                if score < config.thresholds.min_score_knowledge {
                    continue;
                }
                if entry.category == "tool_cache" {
                    continue;
                }
                if entry.category_enum().is_guideline() {
                    continue;
                }
                let key = entry.dedup_key();
                if seen.insert(key) {
                    entries.push(entry);
                }
            }
        }
    }

    // Add project-matching recent entries
    if let Some(project) = project_hint.as_deref() {
        if let Ok(recent) = store.recent(60) {
            for entry in recent {
                if entry.category == "tool_cache" {
                    continue;
                }
                if entry.category_enum().is_guideline() {
                    continue;
                }
                if !entry.mentions_project(project) {
                    continue;
                }
                let key = entry.dedup_key();
                if seen.insert(key) {
                    entries.push(entry);
                }
            }
        }
    }

    if entries.is_empty() {
        return None;
    }

    // Deduplicate
    deduplicate_entries(&mut entries, config.thresholds.dedup_similarity_knowledge);

    // Sort: project match > priority > recency
    let project_hint_lc = project_hint.as_deref().map(|s| s.to_lowercase());
    entries.sort_by(|a, b| {
        let a_project = project_hint_lc
            .as_deref()
            .map(|hint| a.mentions_project(hint))
            .unwrap_or(false);
        let b_project = project_hint_lc
            .as_deref()
            .map(|hint| b.mentions_project(hint))
            .unwrap_or(false);
        b_project
            .cmp(&a_project)
            .then_with(|| b.priority_value().cmp(&a.priority_value()))
            .then_with(|| {
                let b_ts = parse_ts_utc(&b.timestamp)
                    .map(|dt| dt.timestamp())
                    .unwrap_or(0);
                let a_ts = parse_ts_utc(&a.timestamp)
                    .map(|dt| dt.timestamp())
                    .unwrap_or(0);
                b_ts.cmp(&a_ts)
            })
    });

    // Format output
    let max_entries = if project_hint.is_some() {
        config.thresholds.max_entries_with_project
    } else {
        config.thresholds.max_entries_without_project
    };

    let mut out = String::from("Auto-Recalled Knowledge:\n");
    let mut used = out.len();
    let mut appended = 0usize;
    let mut appended_project_matches = 0usize;
    let mut first_entry_project_match = false;
    let mut strongest_project_priority = 0u8;
    let mut categories = Vec::new();

    for (idx, entry) in entries.into_iter().take(max_entries).enumerate() {
        let is_project_match = project_hint_lc
            .as_deref()
            .map(|hint| entry.mentions_project(hint))
            .unwrap_or(false);
        let used_before = used;

        if !push_entry_lines(&mut out, &mut used, max_chars, &entry) {
            if used > used_before {
                appended += 1;
                if !categories.iter().any(|cat| cat == &entry.category) {
                    categories.push(entry.category.clone());
                }
                if is_project_match {
                    appended_project_matches += 1;
                    strongest_project_priority =
                        strongest_project_priority.max(entry.priority_value());
                    if idx == 0 {
                        first_entry_project_match = true;
                    }
                }
            }
            break;
        }

        appended += 1;
        if !categories.iter().any(|cat| cat == &entry.category) {
            categories.push(entry.category.clone());
        }
        if is_project_match {
            appended_project_matches += 1;
            strongest_project_priority = strongest_project_priority.max(entry.priority_value());
            if idx == 0 {
                first_entry_project_match = true;
            }
        }
        if used >= max_chars {
            break;
        }
    }

    if appended == 0 {
        return None;
    }

    let high_confidence_project_memory = project_hint.is_some()
        && appended_project_matches > 0
        && (first_entry_project_match
            || appended_project_matches >= config.thresholds.high_confidence_min_matches
            || strongest_project_priority >= config.thresholds.high_confidence_min_priority);

    Some(AutoRecalledKnowledge {
        content: out,
        high_confidence_project_memory,
        entry_count: appended,
        project_hint,
        categories,
    })
}

/// Get all guideline categories.
pub fn guideline_categories() -> &'static [&'static str] {
    &[
        "safety_rules",
        "user_preference",
        "preference",
        "coding_guideline",
        "best_practice",
        "common_sense",
        "self_note",
    ]
}

/// Check if a category is a guideline category.
pub fn is_guideline_category(category: &str) -> bool {
    guideline_categories()
        .iter()
        .any(|c| c.eq_ignore_ascii_case(category))
}

// --- Helper functions ---

fn parse_ts_utc(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&chrono::Utc))
}

fn current_project_hint() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let name = cwd.file_name()?.to_str()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn deduplicate_entries(entries: &mut Vec<KnowledgeEntry>, similarity_threshold: f64) {
    if entries.len() <= 1 {
        return;
    }
    let mut keep = vec![true; entries.len()];
    for i in 0..entries.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..entries.len() {
            if !keep[j] {
                continue;
            }
            let sim = similarity::note_similarity(&entries[i].note, &entries[j].note);
            if sim >= similarity_threshold {
                let pri_i = entries[i].priority_value();
                let pri_j = entries[j].priority_value();
                if pri_j > pri_i {
                    keep[i] = false;
                    break;
                } else {
                    keep[j] = false;
                }
            }
        }
    }
    let mut filtered = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        if keep[i] {
            filtered.push(entry.clone());
        }
    }
    *entries = filtered;
}

fn push_entry_lines(
    out: &mut String,
    used: &mut usize,
    max_chars: usize,
    entry: &KnowledgeEntry,
) -> bool {
    let note = entry.note.trim();
    if note.is_empty() {
        return false;
    }
    let mut wrote_any = false;
    for line in note.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let bullet = if line.starts_with('-') {
            format!("{line}\n")
        } else {
            format!("- {line}\n")
        };
        if *used + bullet.len() > max_chars {
            if !wrote_any && *used + 40 <= max_chars {
                let summary = note
                    .chars()
                    .take(max_chars.saturating_sub(*used).saturating_sub(40))
                    .collect::<String>();
                out.push_str(&format!(
                    "- [summary, truncated]: {}\n- ... ({} chars total)\n",
                    summary,
                    note.len()
                ));
                *used += out.len();
                return true;
            }
            out.push_str("- ... (truncated)\n");
            *used += 19;
            return true;
        }
        out.push_str(&bullet);
        *used += bullet.len();
        wrote_any = true;
    }
    true
}

fn group_budget(max_chars: usize, used: usize, weights: [usize; 4], idx: usize) -> usize {
    let remaining = max_chars.saturating_sub(used);
    if remaining == 0 {
        return 0;
    }
    let total: usize = weights.iter().sum();
    let w = weights[idx];
    remaining.saturating_mul(w) / total.max(1)
}

fn append_notes(
    out: &mut String,
    used: &mut usize,
    max_chars: usize,
    budget: usize,
    notes: &[String],
) {
    if budget == 0 || *used >= max_chars {
        return;
    }
    let start_used = *used;
    for note in notes {
        for line in note.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let bullet = if line.starts_with('-') {
                format!("{line}\n")
            } else {
                format!("- {line}\n")
            };
            if *used + bullet.len() > max_chars {
                return;
            }
            if *used - start_used + bullet.len() > budget {
                return;
            }
            out.push_str(&bullet);
            *used += bullet.len();
            if *used >= max_chars {
                return;
            }
        }
    }
}
