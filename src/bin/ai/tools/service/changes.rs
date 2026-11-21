//! Change viewing service for `show_changes` and `open_diff`.

use chrono::{DateTime, Duration, NaiveDate, Utc};
use rust_tools::commonw::FastMap;
use serde_json::{Value, json};

use crate::ai::tools::registry::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::storage::changes::{self, EditorKind};
use crate::ai::tools::storage::mutation_log::MutationEntry;

fn execute_show_changes(args: &Value) -> Result<String, String> {
    let v = handle_show_changes_result(args)?;
    serde_json::to_string_pretty(&v).map_err(|e| format!("failed to serialize: {e}"))
}

fn execute_open_diff(args: &Value) -> Result<String, String> {
    let v = handle_open_diff(args);
    serde_json::to_string_pretty(&v).map_err(|e| format!("failed to serialize: {e}"))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "show_changes",
        description: "",
        execute: execute_show_changes,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "open_diff",
        description: "",
        execute: execute_open_diff,
    }
});

const HISTORY_DEFAULT_LIMIT: usize = 20;
const HISTORY_MAX_LIMIT: usize = 100;
const HISTORY_DEFAULT_DIFF_LINES: usize = 20;
const HISTORY_MAX_DIFF_LINES: usize = 100;
const HISTORY_MAX_BYTES: usize = 32_768;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShowMode {
    Summary,
    History,
}

#[derive(Debug, Clone, Default)]
struct ChangeFilter {
    path: Option<String>,
    since_seq: Option<u64>,
    until_seq: Option<u64>,
    since: Option<DateTime<Utc>>,
    until: Option<DateTime<Utc>>,
    since_raw: Option<String>,
    until_raw: Option<String>,
    turn_id: Option<usize>,
    task_id: Option<String>,
}

impl ChangeFilter {
    fn is_active(&self) -> bool {
        self.path.is_some()
            || self.since_seq.is_some()
            || self.until_seq.is_some()
            || self.since.is_some()
            || self.until.is_some()
            || self.turn_id.is_some()
            || self.task_id.is_some()
    }

    fn only_path(&self) -> bool {
        self.path.is_some()
            && self.since_seq.is_none()
            && self.until_seq.is_none()
            && self.since.is_none()
            && self.until.is_none()
            && self.turn_id.is_none()
            && self.task_id.is_none()
    }

    fn matches(&self, entry: &MutationEntry) -> bool {
        if let Some(path) = &self.path {
            if !entry.path.contains(path.as_str()) {
                return false;
            }
        }
        if self.since_seq.is_some_and(|since| entry.seq < since) {
            return false;
        }
        if self.until_seq.is_some_and(|until| entry.seq > until) {
            return false;
        }
        if self.since.is_some() || self.until.is_some() {
            let Ok(parsed) = DateTime::parse_from_rfc3339(&entry.ts) else {
                return false;
            };
            let timestamp = parsed.with_timezone(&Utc);
            if self.since.is_some_and(|since| timestamp < since) {
                return false;
            }
            if self.until.is_some_and(|until| timestamp > until) {
                return false;
            }
        }
        if self.turn_id.is_some_and(|turn| entry.turn_id != Some(turn)) {
            return false;
        }
        if let Some(task) = &self.task_id {
            if entry.task_id.as_deref() != Some(task.as_str()) {
                return false;
            }
        }
        true
    }

    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(path) = &self.path {
            parts.push(format!("path contains {path:?}"));
        }
        if let Some(since) = self.since_seq {
            parts.push(format!("seq >= {since}"));
        }
        if let Some(until) = self.until_seq {
            parts.push(format!("seq <= {until}"));
        }
        if let Some(since) = &self.since_raw {
            parts.push(format!("ts >= {since}"));
        }
        if let Some(until) = &self.until_raw {
            parts.push(format!("ts <= {until}"));
        }
        if let Some(turn) = self.turn_id {
            parts.push(format!("turn_id = {turn}"));
        }
        if let Some(task) = &self.task_id {
            parts.push(format!("task_id = {task:?}"));
        }
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join(", ")
        }
    }

    fn as_json(&self) -> Value {
        json!({
            "path": self.path,
            "since_seq": self.since_seq,
            "until_seq": self.until_seq,
            "since": self.since_raw,
            "until": self.until_raw,
            "since_resolved": self.since.as_ref().map(|value| value.to_rfc3339()),
            "until_resolved": self.until.as_ref().map(|value| value.to_rfc3339()),
            "turn_id": self.turn_id,
            "task_id": self.task_id,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct HistoryOptions {
    after_seq: Option<u64>,
    limit: usize,
    diff_lines: usize,
}

impl Default for HistoryOptions {
    fn default() -> Self {
        Self {
            after_seq: None,
            limit: HISTORY_DEFAULT_LIMIT,
            diff_lines: HISTORY_DEFAULT_DIFF_LINES,
        }
    }
}

#[derive(serde::Serialize)]
struct HistoryItem {
    seq: u64,
    ts: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    turn_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    op: String,
    path: String,
    rel: String,
    version: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    diff: Option<String>,
    /// The entry's diff was omitted because it alone exceeded the page byte budget.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    diff_truncated: bool,
}

struct HistoryOutput {
    items: Vec<HistoryItem>,
    page: Value,
    text: String,
}

fn read_opt_string(args: &Value, name: &str) -> Result<Option<String>, String> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => {
            let value = value.trim();
            Ok((!value.is_empty()).then(|| value.to_string()))
        }
        Some(_) => Err(format!("`{name}` must be a string")),
    }
}

fn read_opt_u64(args: &Value, name: &str) -> Result<Option<u64>, String> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("`{name}` must be a non-negative integer")),
    }
}

fn read_opt_usize(args: &Value, name: &str) -> Result<Option<usize>, String> {
    match read_opt_u64(args, name)? {
        None => Ok(None),
        Some(value) => usize::try_from(value)
            .map(Some)
            .map_err(|_| format!("`{name}` is too large")),
    }
}

fn read_bounded_usize(
    args: &Value,
    name: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, String> {
    match read_opt_u64(args, name)? {
        None => Ok(default),
        Some(value) => {
            let value =
                usize::try_from(value).map_err(|_| format!("`{name}` is too large"))?;
            Ok(value.clamp(min, max))
        }
    }
}

fn parse_mode(args: &Value) -> Result<ShowMode, String> {
    match args.get("mode") {
        None | Some(Value::Null) => Ok(ShowMode::Summary),
        Some(Value::String(mode)) => match mode.as_str() {
            "summary" => Ok(ShowMode::Summary),
            "history" => Ok(ShowMode::History),
            other => Err(format!(
                "`mode` must be \"summary\" or \"history\", got {other:?}"
            )),
        },
        Some(_) => Err("`mode` must be a string".to_string()),
    }
}

fn parse_time_bound(raw: &str, now: DateTime<Utc>) -> Result<DateTime<Utc>, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("time bound must not be empty".to_string());
    }
    if let Ok(parsed) = DateTime::parse_from_rfc3339(raw) {
        return Ok(parsed.with_timezone(&Utc));
    }
    if let Ok(date) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return date
            .and_hms_opt(0, 0, 0)
            .map(|naive| naive.and_utc())
            .ok_or_else(|| format!("invalid date `{raw}`"));
    }
    let split = raw.find(|c: char| !c.is_ascii_digit()).ok_or_else(|| {
        format!(
            "invalid time bound `{raw}`: expected RFC3339, YYYY-MM-DD, or <n><s|m|h|d|w>"
        )
    })?;
    let (number, unit) = raw.split_at(split);
    if number.is_empty() {
        return Err(format!(
            "invalid time bound `{raw}`: expected a positive number and unit"
        ));
    }
    let value: i64 = number
        .parse()
        .map_err(|_| format!("invalid time bound `{raw}`"))?;
    if value <= 0 || value > 1_000_000 {
        return Err(format!("time bound `{raw}` is out of range"));
    }
    let duration = match unit.trim().to_ascii_lowercase().as_str() {
        "s" | "sec" | "secs" | "second" | "seconds" => Duration::seconds(value),
        "m" | "min" | "mins" | "minute" | "minutes" => Duration::minutes(value),
        "h" | "hr" | "hrs" | "hour" | "hours" => Duration::hours(value),
        "d" | "day" | "days" => Duration::days(value),
        "w" | "week" | "weeks" => Duration::weeks(value),
        _ => {
            return Err(format!(
                "invalid time bound `{raw}`: unknown unit `{unit}`"
            ));
        }
    };
    now.checked_sub_signed(duration)
        .ok_or_else(|| format!("time bound `{raw}` is out of range"))
}

fn change_filter_from_args(args: &Value) -> Result<ChangeFilter, String> {
    let path = read_opt_string(args, "path")?
        .map(|path| path.trim_start_matches("./").to_string())
        .filter(|path| !path.is_empty());
    let since_raw = read_opt_string(args, "since")?;
    let until_raw = read_opt_string(args, "until")?;
    let now = Utc::now();
    let since = since_raw
        .as_deref()
        .map(|raw| parse_time_bound(raw, now))
        .transpose()?;
    let until = until_raw
        .as_deref()
        .map(|raw| parse_time_bound(raw, now))
        .transpose()?;
    if let (Some(since), Some(until)) = (since.as_ref(), until.as_ref()) {
        if since > until {
            return Err("`since` must not be later than `until`".to_string());
        }
    }
    let since_seq = read_opt_u64(args, "since_seq")?;
    let until_seq = read_opt_u64(args, "until_seq")?;
    if let (Some(since), Some(until)) = (since_seq, until_seq) {
        if since > until {
            return Err("`since_seq` must not be greater than `until_seq`".to_string());
        }
    }
    Ok(ChangeFilter {
        path,
        since_seq,
        until_seq,
        since,
        until,
        since_raw,
        until_raw,
        turn_id: read_opt_usize(args, "turn_id")?,
        task_id: read_opt_string(args, "task_id")?,
    })
}

fn history_options_from_args(args: &Value) -> Result<HistoryOptions, String> {
    Ok(HistoryOptions {
        after_seq: read_opt_u64(args, "after_seq")?,
        limit: read_bounded_usize(
            args,
            "history_limit",
            HISTORY_DEFAULT_LIMIT,
            1,
            HISTORY_MAX_LIMIT,
        )?,
        diff_lines: read_bounded_usize(
            args,
            "history_diff_lines",
            HISTORY_DEFAULT_DIFF_LINES,
            0,
            HISTORY_MAX_DIFF_LINES,
        )?,
    })
}

fn history_versions(entries: &[MutationEntry]) -> Vec<usize> {
    let mut counts: FastMap<&str, usize> = FastMap::default();
    let mut versions = Vec::with_capacity(entries.len());
    for entry in entries {
        let count = counts.entry(entry.path.as_str()).or_insert(0);
        *count += 1;
        versions.push(*count);
    }
    versions
}

fn build_history(
    entries: &[MutationEntry],
    versions: &[usize],
    matched: &[usize],
    options: HistoryOptions,
    filter: &ChangeFilter,
) -> HistoryOutput {
    let start = options.after_seq.map_or(0, |after| {
        matched
            .iter()
            .position(|&index| entries[index].seq > after)
            .unwrap_or(matched.len())
    });
    let mut items = Vec::new();
    let mut bytes = 2usize;
    let mut stopped_by_bytes = false;
    let mut omitted_diffs = 0usize;
    for &index in &matched[start..] {
        if items.len() >= options.limit {
            break;
        }
        let entry = &entries[index];
        let diff = (options.diff_lines > 0)
            .then(|| changes::entry_snippet(entry, options.diff_lines))
            .flatten();
        let mut item = HistoryItem {
            seq: entry.seq,
            ts: entry.ts.clone(),
            turn_id: entry.turn_id,
            task_id: entry.task_id.clone(),
            tool_call_id: entry.tool_call_id.clone(),
            op: entry.op.clone(),
            path: entry.path.clone(),
            rel: changes::display_rel(&entry.path),
            version: versions[index],
            diff,
            diff_truncated: false,
        };
        let mut item_bytes = serde_json::to_vec(&item)
            .map(|serialized| serialized.len())
            .unwrap_or(0)
            + 2;
        // The first item cannot be skipped without losing the cursor, so drop its
        // diff when that alone would exceed the page budget. The flag tells the
        // caller the version is present but its diff is not.
        if items.is_empty() && bytes + item_bytes > HISTORY_MAX_BYTES && item.diff.is_some() {
            item.diff = None;
            item.diff_truncated = true;
            item_bytes = serde_json::to_vec(&item)
                .map(|serialized| serialized.len())
                .unwrap_or(0)
                + 2;
            omitted_diffs += 1;
        }
        if !items.is_empty() && bytes + item_bytes > HISTORY_MAX_BYTES {
            stopped_by_bytes = true;
            break;
        }
        // Metadata alone is far smaller than the page budget; dropping the diff
        // above keeps the first item within it. Keep a defensive truncation flag
        // if a pathological path/ts makes the metadata itself exceed the budget.
        if items.is_empty() && bytes + item_bytes > HISTORY_MAX_BYTES {
            stopped_by_bytes = true;
        }
        bytes += item_bytes;
        items.push(item);
    }
    let complete = start + items.len() >= matched.len();
    let next_after_seq = if complete {
        None
    } else {
        items.last().map(|item| item.seq)
    };
    let page = json!({
        "after_seq": options.after_seq,
        "total": matched.len(),
        "offset": start,
        "shown": items.len(),
        "limit": options.limit,
        "diff_lines": options.diff_lines,
        "max_bytes": HISTORY_MAX_BYTES,
        "shown_bytes": bytes,
        "truncated_by_bytes": stopped_by_bytes || omitted_diffs > 0,
        "diff_omitted": omitted_diffs,
        "complete": complete,
        "next_after_seq": next_after_seq,
    });
    let text = history_text(&items, &page, filter);
    HistoryOutput { items, page, text }
}

fn history_text(items: &[HistoryItem], page: &Value, filter: &ChangeFilter) -> String {
    let total = page["total"].as_u64().unwrap_or(0);
    let shown = page["shown"].as_u64().unwrap_or(0);
    let mut out = format!(
        "mutation 历史（窗口 {total} 条，按 seq 升序；本页展示 {shown} 条；过滤器：{}）\n",
        filter.describe()
    );
    for (index, item) in items.iter().enumerate() {
        let turn = item
            .turn_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let task = item.task_id.as_deref().unwrap_or("-");
        out.push_str(&format!(
            "{}. seq={} ts={} turn={} task={} op={} path={} version={}\n",
            index + 1,
            item.seq,
            item.ts,
            turn,
            task,
            item.op,
            item.rel,
            item.version
        ));
    }
    if items.iter().any(|item| item.diff_truncated) {
        out.push_str(
            "（部分条目的 diff 因单页字节上限已省略；可直接读取 mutation log 查看该条记录。）\n",
        );
    }
    if let Some(next) = page["next_after_seq"].as_u64() {
        out.push_str(&format!(
            "下一批：show_changes(mode=\"history\", after_seq={next}, history_limit={})\n",
            page["limit"].as_u64().unwrap_or(0)
        ));
    }
    out
}

fn build_window(
    entries: &[MutationEntry],
    filter: &ChangeFilter,
    matched: &[usize],
) -> Value {
    let first = matched.first().map(|&index| &entries[index]);
    let last = matched.last().map(|&index| &entries[index]);
    let time_active = filter.since.is_some() || filter.until.is_some();
    let empty_reason = if entries.is_empty() {
        Some("no_session_mutations")
    } else if matched.is_empty() {
        Some("no_matches_in_window")
    } else {
        None
    };
    let unparsed_ts = if time_active {
        entries
            .iter()
            .filter(|entry| DateTime::parse_from_rfc3339(&entry.ts).is_err())
            .count()
    } else {
        0
    };
    json!({
        "filters": filter.as_json(),
        "session_mutations": entries.len(),
        "matched_mutations": matched.len(),
        "first_seq": first.map(|entry| entry.seq),
        "last_seq": last.map(|entry| entry.seq),
        "first_ts": first.map(|entry| entry.ts.clone()),
        "last_ts": last.map(|entry| entry.ts.clone()),
        "empty_reason": empty_reason,
        "unparsed_ts": unparsed_ts,
        "time_filter_scope": if time_active { "session_mutations_only" } else { "none" },
        "verification_scope": "session",
    })
}

fn window_header(filter: &ChangeFilter, entries: &[MutationEntry], matched: &[usize]) -> String {
    let first = matched.first().map(|&index| entries[index].seq);
    let last = matched.last().map(|&index| entries[index].seq);
    let range = match (first, last) {
        (Some(first), Some(last)) => format!("seq {first}..{last}"),
        _ => "空窗口".to_string(),
    };
    format!(
        "以下为 mutation 窗口内的改动（{range}；匹配 {} / 会话共 {} 条；过滤器：{}；按窗口内首次改动顺序）。\n",
        matched.len(),
        entries.len(),
        filter.describe()
    )
}

fn next_seq(entries: &[MutationEntry]) -> Option<u64> {
    match entries.iter().map(|entry| entry.seq).max() {
        Some(seq) => seq.checked_add(1),
        None => Some(1),
    }
}

fn truncate_text(text: &mut String, limit_snippet: usize, patch_bytes: usize) {
    if limit_snippet == 0 {
        return;
    }
    let cap = limit_snippet.saturating_mul(80);
    let Some((byte_index, _)) = text.char_indices().nth(cap) else {
        return;
    };
    let truncated = &text[..byte_index];
    *text = format!(
        "{}…\n[truncated to {} chars, patch {} bytes]",
        truncated, cap, patch_bytes
    );
}

pub(crate) fn handle_show_changes(args: &Value) -> Value {
    handle_show_changes_result(args).unwrap_or_else(|error| json!({ "error": error }))
}

pub(crate) fn handle_show_changes_result(args: &Value) -> Result<Value, String> {
    let include_git = args
        .get("include_git")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            crate::commonw::configw::get_all_config()
                .get_opt(crate::ai::config_schema::AiConfig::DIFF_INCLUDE_GIT)
                .filter(|v| !v.trim().is_empty())
                .map(|v| {
                    let low = v.trim().to_ascii_lowercase();
                    !(low == "false" || low == "0" || low == "no" || low == "off")
                })
                .unwrap_or(true)
        });
    let limit_snippet = args
        .get("limit_snippet")
        .and_then(Value::as_i64)
        .map(|v| v.max(0) as usize)
        .unwrap_or(120);
    let mode = parse_mode(args)?;
    let filter = change_filter_from_args(args)?;
    let history_options = history_options_from_args(args)?;

    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let is_git_repo = changes::is_inside_git_work_tree(&cwd);
    let snapshot = crate::ai::tools::storage::mutation_log::read_committed_snapshot();
    let entries = &snapshot.mutations;
    let has_session_mutations = !entries.is_empty();
    let matched: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| filter.matches(entry).then_some(index))
        .collect();
    let grouped = changes::grouped_changes_entries(matched.iter().map(|&index| &entries[index]));
    let versions = if mode == ShowMode::History {
        history_versions(entries)
    } else {
        Vec::new()
    };

    let patch_opt = if entries.is_empty() {
        if include_git && !filter.is_active() {
            changes::combined_patch()
        } else {
            None
        }
    } else if matched.is_empty() {
        None
    } else {
        changes::mutation_patch_entries(matched.iter().map(|&index| &entries[index]))
    };
    let patch_bytes = patch_opt.as_ref().map_or(0, |patch| patch.len());

    let history_output = (mode == ShowMode::History).then(|| {
        build_history(entries, &versions, &matched, history_options, &filter)
    });
    let mut text = if entries.is_empty() {
        if include_git {
            if !filter.is_active() {
                changes::combined_summary()
            } else if filter.only_path() {
                changes::fallback_git_summary(filter.path.as_deref())
            } else {
                "当前会话无 mutation 记录；时间/序号/provenance 过滤无法应用到 Git 变更（Git 没有记录的 seq/ts），已跳过 Git 回退。".to_string()
            }
        } else {
            "当前会话无文件变更（include_git=false，仅统计 mutation log）。".to_string()
        }
    } else if matched.is_empty() {
        let mut text = format!(
            "mutation 窗口内无变更（会话共 {} 条 mutation；过滤器：{}）。",
            entries.len(),
            filter.describe()
        );
        if include_git {
            text.push_str("\nGit 变更没有记录的 seq/时间，不参与本次窗口过滤。");
        }
        text
    } else if let Some(history) = &history_output {
        history.text.clone()
    } else {
        let header = filter
            .is_active()
            .then(|| window_header(&filter, entries, &matched));
        changes::format_grouped_summary(
            &grouped,
            include_git && !filter.is_active(),
            header.as_deref(),
        )
    };
    truncate_text(&mut text, limit_snippet, patch_bytes);

    let changed_files = grouped.len();
    let mut result = json!({
        "text": text,
        "changed_files": changed_files,
        "is_git_repo": is_git_repo,
        "has_session_mutations": has_session_mutations,
        "patch_bytes": patch_bytes,
        "window": build_window(entries, &filter, &matched),
        "next_seq": next_seq(entries),
    });
    if let Some(history) = history_output {
        result["history"] = json!(history.items);
        result["history_page"] = history.page;
    }
    if include_git {
        let mut inventory = changes::collect_current_changes(true);
        if let Some(path) = &filter.path {
            inventory
                .files
                .retain(|file| file.path.contains(path.as_str()));
            inventory.session_changes.retain(|change| {
                change.path.contains(path.as_str()) || change.rel.contains(path.as_str())
            });
        }
        let git_scope = if !filter.is_active() {
            "all_workspace_changes"
        } else if filter.only_path() {
            "path_filtered"
        } else if filter.path.is_some() {
            "path_filtered_time_unfiltered"
        } else {
            "unfiltered"
        };
        result["coverage"] = json!({
            "observed_paths": inventory.files.len(),
            "session_changed_files": if filter.is_active() {
                changed_files
            } else {
                inventory.session_changes.len()
            },
            "collection_complete": inventory.collection_complete(),
            "collection_errors": inventory.collection_errors,
            "coverage_gaps": inventory.coverage_gaps,
            "git_scope": git_scope,
        });
        result["changed_paths"] = json!(inventory.files);
    }
    use crate::ai::tools::storage::mutation_log::{self, VerificationDetailsPage};
    let full = args
        .get("verification_full")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let details = full
        || args
            .get("verification_details")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let mut page = VerificationDetailsPage::default();
    if full {
        page.limit = 0;
        page.max_bytes = 0;
    } else {
        let read_usize = |name: &str, default: usize| {
            args.get(name)
                .and_then(Value::as_u64)
                .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
                .unwrap_or(default)
        };
        page.offset = read_usize("verification_offset", page.offset);
        page.limit = read_usize("verification_limit", page.limit);
        page.max_bytes = read_usize("verification_max_bytes", page.max_bytes);
    }
    let verification = mutation_log::verification_summary_page(
        snapshot.revision(),
        &snapshot.checks,
        details.then_some(&page),
    );
    for (key, value) in verification.as_object().unwrap() {
        result[key] = value.clone();
    }
    Ok(result)
}

pub(crate) fn handle_open_diff(args: &Value) -> Value {
    let editor_str = args.get("editor").and_then(Value::as_str).unwrap_or("auto");
    let patch_file_arg = args.get("patch_file").and_then(Value::as_str);
    let as_patch_opt = args.get("as_patch").and_then(Value::as_bool).or_else(|| {
        crate::commonw::configw::get_all_config()
            .get_opt(crate::ai::config_schema::AiConfig::DIFF_OPEN_PATCH_FILE)
            .filter(|v| !v.trim().is_empty())
            .map(|v| {
                let low = v.trim().to_ascii_lowercase();
                !(low == "false" || low == "0" || low == "no" || low == "off")
            })
    });

    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let is_git_repo = changes::is_inside_git_work_tree(&cwd);
    if changes::session_grouped_changes().is_empty() && !is_git_repo {
        return json!({
            "ok": false,
            "error": "no changes to open (no session mutations and not a git repo)",
            "is_git_repo": is_git_repo,
        });
    }
    let patch_opt = changes::combined_patch();
    let patch = match patch_opt {
        Some(p) if !p.trim().is_empty() => p,
        _ => {
            return json!({
                "ok": false,
                "error": "generated patch is empty",
                "is_git_repo": is_git_repo,
            });
        }
    };

    // Prefer before/after comparison for a single file in VS Code or Cursor.
    if as_patch_opt == Some(false) {
        let grouped = changes::session_grouped_changes();
        if grouped.len() == 1 {
            let normalized = if editor_str.eq_ignore_ascii_case("vscode") {
                "code"
            } else {
                editor_str
            };
            let kind = if editor_str.eq_ignore_ascii_case("auto") {
                // Automatic selection allows comparisons only in VS Code or Cursor.
                changes::configured_editor()
            } else {
                EditorKind::from_str(normalized).unwrap_or_else(changes::configured_editor)
            };
            if matches!(kind, EditorKind::Vscode | EditorKind::Cursor) {
                let g = &grouped[0];
                if let Some(res) = try_open_single_file_diff(g, &kind) {
                    return match res {
                        Ok(cmd) => json!({
                            "ok": true,
                            "patch_file": null,
                            "mode": "diff",
                            "file": g.rel,
                            "editor": kind.label(),
                            "requested_editor": editor_str,
                            "command": cmd,
                            "is_git_repo": is_git_repo,
                        }),
                        Err(e) => json!({
                            "ok": false,
                            "error": e,
                            "mode": "diff",
                            "is_git_repo": is_git_repo,
                        }),
                    };
                }
            }
        }
        // Fall back to a patch when per-file comparison is unavailable.
    }

    // An explicit patch path overrides the session-assets destination.
    if let Some(custom) = patch_file_arg {
        let dest = std::path::PathBuf::from(custom);
        // Require a writable root or registered isolated temporary file,
        // matching FileStore::validate_write_access.
        let in_allowed = crate::ai::tools::storage::file_store::path_within_allowed_roots(&dest);
        let is_registered =
            crate::ai::tools::storage::temp_registry::is_registered(&dest.display().to_string());
        // Root validation resolves relative paths through effective_cwd. Absolute
        // paths within the runtime temp directory are also authorized scratch.
        let in_session_tmp = dest.is_absolute()
            && dest.starts_with(
                crate::ai::driver::runtime_ctx::temp_dir().unwrap_or_else(|_| std::env::temp_dir()),
            );
        if !(in_allowed || is_registered || in_session_tmp) {
            return json!({
                "ok": false,
                "error": format!("patch_file 被沙箱拦截：{} 不在可写根（effective_cwd / allowed_roots / session temp）内", dest.display()),
                "is_git_repo": is_git_repo,
            });
        }
        if let Some(parent) = dest.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return json!({
                        "ok": false,
                        "error": format!("failed to create parent dir {}: {e}", parent.display()),
                        "is_git_repo": is_git_repo,
                    });
                }
            }
        }
        if let Err(e) = std::fs::write(&dest, patch.as_bytes()) {
            return json!({
                "ok": false,
                "error": format!("failed to write patch to {}: {e}", dest.display()),
                "is_git_repo": is_git_repo,
            });
        }
        let kind = if editor_str.eq_ignore_ascii_case("auto") {
            changes::configured_editor()
        } else {
            let normalized = if editor_str.eq_ignore_ascii_case("vscode") {
                "code"
            } else {
                editor_str
            };
            EditorKind::from_str(normalized).unwrap_or_else(changes::configured_editor)
        };
        // Open directly because open_changes would ignore the explicit path.
        let open_res = open_patch_path_with_editor(&dest, &kind);
        return match open_res {
            Ok(cmd) => json!({
                "ok": true,
                "patch_file": dest.display().to_string(),
                "editor": kind.label(),
                "requested_editor": editor_str,
                "command": cmd,
                "is_git_repo": is_git_repo,
            }),
            Err(e) => json!({
                "ok": false,
                "error": e,
                "patch_file": dest.display().to_string(),
                "editor": kind.label(),
                "is_git_repo": is_git_repo,
            }),
        };
    }

    // Use the storage entry point for the default patch destination.
    let requested_kind = if editor_str.eq_ignore_ascii_case("auto") {
        None
    } else {
        let normalized = if editor_str.eq_ignore_ascii_case("vscode") {
            "code"
        } else {
            editor_str
        };
        EditorKind::from_str(normalized)
    };
    match changes::open_changes(requested_kind.clone()) {
        Ok(cmd) => {
            // open_changes already wrote the patch; retain the resolved viewer
            // without writing the patch again.
            let resolved = requested_kind.unwrap_or_else(changes::configured_editor);
            // Infer the patch destination from session assets or the temp fallback.
            let patch_path = crate::ai::tools::storage::file_store::current_session_assets_dir()
                .map(|d| d.join("changes.patch"))
                .unwrap_or_else(|| {
                    crate::ai::driver::runtime_ctx::temp_dir()
                        .unwrap_or_else(|_| std::path::PathBuf::from("."))
                        .join(format!("changes.{}.patch", std::process::id()))
                });
            json!({
                "ok": true,
                "patch_file": patch_path.display().to_string(),
                "editor": resolved.label(),
                "requested_editor": editor_str,
                "command": cmd,
                "is_git_repo": is_git_repo,
            })
        }
        Err(e) => json!({
            "ok": false,
            "error": e,
            "is_git_repo": is_git_repo,
        }),
    }
}

fn try_open_single_file_diff(
    g: &crate::ai::tools::storage::changes::FileChange,
    kind: &EditorKind,
) -> Option<Result<String, String>> {
    // A comparison needs at least one captured side.
    if g.before_first.is_none() && g.after_last.is_none() {
        return None;
    }
    // Unavailable or truncated snapshots cannot be represented by empty files
    // in code --diff; use the patch path with its explicit limitations instead.
    if !g.snapshots_full() {
        return None;
    }
    // Write comparison files within the session temp directory sandbox.
    let base_tmp =
        crate::ai::driver::runtime_ctx::temp_dir().unwrap_or_else(|_| std::env::temp_dir());
    let tmp_dir = base_tmp.join(format!("rust_tools_diff_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp_dir);
    let safe = g.rel.replace('/', "_").replace('\\', "_");
    let before_path = tmp_dir.join(format!("{}.before", safe));
    let after_path = tmp_dir.join(format!("{}.after", safe));
    if let Some(b) = g.before_first.as_deref() {
        if std::fs::write(&before_path, b).is_err() {
            return None;
        }
    } else {
        let _ = std::fs::write(&before_path, "");
    }
    if let Some(a) = g.after_last.as_deref() {
        if std::fs::write(&after_path, a).is_err() {
            return None;
        }
    } else {
        let _ = std::fs::write(&after_path, "");
    }
    let prog = match kind {
        EditorKind::Vscode => "code",
        EditorKind::Cursor => "cursor",
        _ => return None,
    };
    let cmd_display = format!(
        "{} --diff {} {}",
        prog,
        before_path.display(),
        after_path.display()
    );
    let res = std::process::Command::new(prog)
        .args([
            "--diff",
            &before_path.display().to_string(),
            &after_path.display().to_string(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| cmd_display.clone())
        .map_err(|e| format!("failed to launch editor '{}': {e}", prog));
    Some(res)
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use crate::ai::tools::storage::mutation_log::{BeforeState, MutationEntry};

    #[test]
    fn audit_default_show_changes_reports_session_text_and_window() {
        let result = handle_show_changes(&json!({"include_git": false, "limit_snippet": 0}));
        let mut keys = result
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "changed_files",
                "has_session_mutations",
                "is_git_repo",
                "next_seq",
                "patch_bytes",
                "revision",
                "text",
                "verification",
                "verification_gaps",
                "window",
            ]
        );
        assert_eq!(
            result["text"],
            "当前会话无文件变更（include_git=false，仅统计 mutation log）。"
        );
        assert_eq!(result["changed_files"], 0);
        assert_eq!(result["has_session_mutations"], false);
        assert_eq!(result["patch_bytes"], 0);
        assert_eq!(result["revision"], 0);
        assert_eq!(result["next_seq"], 1);
        assert_eq!(result["window"]["session_mutations"], 0);
        assert_eq!(result["window"]["matched_mutations"], 0);
        assert_eq!(result["window"]["empty_reason"], "no_session_mutations");
        assert_eq!(result["verification"]["status"], "no_changes");
        assert_eq!(result["verification_gaps"]["count"], 0);
        assert!(result["verification"].get("checks").is_none());
    }

    #[test]
    fn show_changes_verification_paging_arguments_are_observable() {
        use crate::ai::driver::runtime_ctx;
        use crate::ai::tools::storage::mutation_log;
        let root =
            std::env::temp_dir().join(format!("show-changes-pages-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join("history.jsonl");
        app.session_id = "show-changes-pages".into();
        let context = runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || runtime_ctx::DRIVER_CTX.sync_scope(context, || {
            for index in 0..10 {
                assert!(mutation_log::begin_check(&format!("check-{index}"), "execute_command", &json!({"command": format!("cargo check --package project-{index}")})).is_some());
            }
            let path = mutation_log::log_path().unwrap();
            let original = std::fs::read(&path).unwrap();
            let first = handle_show_changes(&json!({"include_git": false, "verification_details": true}));
            assert_eq!(first["verification"]["checks_page"]["shown"], 8);
            assert_eq!(first["verification"]["checks_page"]["total"], 10);
            let next = handle_show_changes(&json!({"include_git": false, "verification_details": true, "verification_offset": 8, "verification_limit": 1, "verification_max_bytes": 0}));
            assert_eq!(next["verification"]["checks_page"]["shown"], 1);
            assert_eq!(next["verification"]["checks_page"]["next_offset"], 9);
            assert_eq!(next["verification"]["checks"][0]["observation"]["originating_command_call_id"], "check-8");
            for max_bytes in [1, 2] {
                let tiny = handle_show_changes(&json!({"include_git": false, "verification_details": true, "verification_max_bytes": max_bytes}));
                assert_eq!(tiny["verification"]["checks"], json!([]));
                let page = &tiny["verification"]["checks_page"];
                assert_eq!(page["max_bytes"], 2);
                assert_eq!(page["shown_bytes"], 2);
                assert!(page["shown_bytes"].as_u64().unwrap() <= page["max_bytes"].as_u64().unwrap());
                assert_eq!(page["next_offset"], 0);
                assert_eq!(page["complete"], false);
            }
            let full = handle_show_changes(&json!({"include_git": false, "verification_full": true, "verification_offset": 9, "verification_limit": 1, "verification_max_bytes": 2}));
            assert_eq!(full["verification"]["checks_page"]["offset"], 0);
            assert_eq!(full["verification"]["checks_page"]["shown"], 10);
            assert_eq!(full["verification"]["checks_page"]["complete"], true);
            assert_eq!(full["verification_gaps"], first["verification_gaps"]);
            assert_eq!(std::fs::read(&path).unwrap(), original);
        }));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn show_changes_verification_uses_committed_log_snapshot() {
        use crate::ai::tools::storage::mutation_log;
        use std::io::Write;

        let root = std::env::temp_dir().join(format!(
            "show-changes-committed-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join("history.jsonl");
        app.session_id = "show-changes-committed".into();
        let context = crate::ai::driver::runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || {
            crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(context, || {
                let path = mutation_log::log_path().unwrap();
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                let entry = MutationEntry {
                    entry_id: None,
                    turn_id: None,
                    task_id: None,
                    tool_call_id: None,
                    seq: 1,
                    ts: "t".into(),
                    path: root.join("file.rs").display().to_string(),
                    op: "write".into(),
                    before: None,
                    before_state: Some(BeforeState::Absent),
                    after: Some("changed\n".into()),
                    diff: None,
                };
                std::fs::write(&path, serde_json::to_string(&entry).unwrap() + "\n").unwrap();
                let pending = mutation_log::begin_check(
                    "check",
                    "execute_command",
                    &json!({"command": "cargo check"}),
                );
                assert!(pending.is_some());
                let mut terminal = mutation_log::read_checks().last().unwrap().clone();
                terminal.status = mutation_log::CheckStatus::Success;
                terminal.observed_success = true;
                let before = mutation_log::recovered_records_at(&path);
                let mut writer = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .unwrap();
                writer
                    .write_all(&serde_json::to_vec(&terminal).unwrap())
                    .unwrap();
                let args = json!({
                    "include_git": false,
                    "limit_snippet": 0,
                    "verification_details": true,
                });
                for _ in 0..3 {
                    let result = handle_show_changes(&args);
                    assert_eq!(result["revision"], 1);
                    assert_eq!(result["changed_files"], 1);
                    assert_eq!(result["verification"]["status"], "running");
                    assert_eq!(result["verification_gaps"]["count"], 1);
                    assert_eq!(
                        result["verification"]["checks"][0]["observation"]["status"],
                        "running"
                    );
                    assert_eq!(mutation_log::verification_reminder_summary().gaps, 1);
                    assert_eq!(mutation_log::recovered_records_at(&path), before);
                }
                writer.write_all(b"\n").unwrap();
                for _ in 0..3 {
                    let result = handle_show_changes(&args);
                    assert_eq!(result["revision"], 1);
                    assert_eq!(result["verification"]["status"], "success");
                    assert_eq!(result["verification_gaps"]["count"], 0);
                    assert_eq!(
                        result["verification"]["checks"][0]["observation"],
                        json!(terminal)
                    );
                    assert_eq!(mutation_log::verification_reminder_summary().gaps, 0);
                    assert_eq!(mutation_log::recovered_records_at(&path), before + 1);
                }
            });
        });
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn show_changes_with_git_path_filter_applies_to_inventory_and_window() {
        let root = std::env::temp_dir().join(format!(
            "show-changes-inventory-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let root = std::fs::canonicalize(root).unwrap();
        let git = crate::fork_guard::output(
            std::process::Command::new("git")
                .args(["init", "-q"])
                .current_dir(&root),
        )
        .unwrap();
        assert!(git.status.success(), "{:?}", git.stderr);
        let git_path = root.join("git-only\npath.rs");
        std::fs::write(&git_path, "untracked\n").unwrap();
        let entry = MutationEntry {
            entry_id: None,
            turn_id: None,
            task_id: None,
            tool_call_id: None,
            seq: 1,
            ts: "t".into(),
            path: root.join("mutation-only.rs").display().to_string(),
            op: "delete".into(),
            before: Some("removed\n".into()),
            before_state: Some(BeforeState::Present),
            after: None,
            diff: None,
        };
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join(".git/audit-test/history.jsonl");
        app.session_id = "show-changes-test".into();
        let context = crate::ai::driver::runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || {
            crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(context, || {
                let log_path = crate::ai::tools::storage::mutation_log::log_path().unwrap();
                assert!(log_path.starts_with(root.join(".git")));
                std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
                let log_entry = serde_json::to_string(&entry).unwrap() + "\n";
                std::fs::write(&log_path, &log_entry).unwrap();
                let patch_bytes = changes::combined_patch().unwrap().len();
                let full = handle_show_changes(&json!({"include_git": true, "limit_snippet": 0}));
                assert_eq!(full["text"], changes::combined_summary());
                let result = handle_show_changes(&json!({
                    "include_git": true,
                    "path": "mutation-only",
                    "limit_snippet": 1,
                }));
                assert!(
                    result["text"]
                        .as_str()
                        .unwrap()
                        .contains("[truncated to 80 chars")
                );
                assert_eq!(result["changed_files"], 1);
                assert_eq!(result["is_git_repo"], true);
                assert_eq!(result["has_session_mutations"], true);
                assert_eq!(result["patch_bytes"], patch_bytes);
                let mut filtered_inventory = changes::collect_current_changes(true);
                filtered_inventory
                    .files
                    .retain(|file| file.path.contains("mutation-only"));
                filtered_inventory.session_changes.retain(|change| {
                    change.path.contains("mutation-only")
                        || change.rel.contains("mutation-only")
                });
                assert_eq!(
                    result["changed_paths"],
                    serde_json::to_value(&filtered_inventory.files).unwrap()
                );
                assert_eq!(result["coverage"]["observed_paths"], 1);
                assert_eq!(result["coverage"]["session_changed_files"], 1);
                assert_eq!(result["coverage"]["collection_complete"], true);
                assert_eq!(result["coverage"]["collection_errors"], json!([]));
                assert_eq!(result["coverage"]["git_scope"], "path_filtered");
                assert_eq!(
                    result["coverage"]["coverage_gaps"],
                    json!(filtered_inventory.coverage_gaps)
                );
                assert!(
                    !result["coverage"]["coverage_gaps"]
                        .as_array()
                        .unwrap()
                        .is_empty()
                );
                assert!(
                    result["changed_paths"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|file| {
                            file["path"] == entry.path.clone()
                                && file["ownership"] == "unknown"
                        })
                );
                assert!(
                    !result["changed_paths"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|file| file["path"] == git_path.display().to_string())
                );

                let windowed = handle_show_changes(&json!({
                    "include_git": true,
                    "since_seq": 99,
                    "limit_snippet": 0,
                }));
                assert_eq!(windowed["changed_files"], 0);
                assert_eq!(
                    windowed["window"]["empty_reason"],
                    "no_matches_in_window"
                );
                assert_eq!(windowed["coverage"]["git_scope"], "unfiltered");
                assert!(
                    !windowed["text"]
                        .as_str()
                        .unwrap()
                        .contains("git-only")
                );
                assert_eq!(
                    windowed["changed_paths"].as_array().unwrap().len(),
                    2
                );

                std::fs::write(&log_path, log_entry + "invalid mutation entry\n").unwrap();
                let incomplete =
                    handle_show_changes(&json!({"include_git": true, "limit_snippet": 1}));
                let inventory = changes::collect_current_changes(true);
                assert_eq!(incomplete["coverage"]["collection_complete"], false);
                assert!(!inventory.collection_errors.is_empty());
                assert_eq!(
                    incomplete["coverage"]["collection_errors"],
                    json!(inventory.collection_errors)
                );
                assert_eq!(
                    incomplete["changed_paths"],
                    serde_json::to_value(&inventory.files).unwrap()
                );
            });
        });
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn show_changes_path_filter_uses_substring_for_git_fallback() {
        let root = std::env::temp_dir().join(format!(
            "show-changes-git-substring-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let root = std::fs::canonicalize(root).unwrap();
        let git = |args: &[&str]| {
            let output = crate::fork_guard::output(
                std::process::Command::new("git")
                    .args(args)
                    .current_dir(&root),
            )
            .unwrap();
            assert!(
                output.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        };
        git(&["init", "-q"]);
        std::fs::write(root.join("target.rs"), "one\n").unwrap();
        git(&["add", "target.rs"]);
        git(&[
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "base",
        ]);
        std::fs::write(root.join("target.rs"), "two\n").unwrap();

        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join(".git/audit-fallback/history.jsonl");
        app.session_id = format!("show-changes-git-{}", uuid::Uuid::new_v4().simple());
        let context = crate::ai::driver::runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || {
            crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(context, || {
                let result = handle_show_changes(&json!({
                    "include_git": true,
                    "path": "target",
                    "limit_snippet": 0,
                }));
                let text = result["text"].as_str().unwrap();
                assert!(text.contains("target.rs"), "{text}");
                assert!(!text.contains("无匹配"), "{text}");
                // `changed_files` counts session mutation groups; the Git-only
                // fallback is represented by `changed_paths`.
                assert_eq!(result["changed_files"], 0);
                assert_eq!(result["coverage"]["git_scope"], "path_filtered");
                assert!(
                    result["changed_paths"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|file| file["path"]
                            .as_str()
                            .is_some_and(|path| path.contains("target.rs")))
                );
            });
        });
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn show_changes_history_omits_a_first_diff_that_exceeds_the_page_budget() {
        use crate::ai::driver::runtime_ctx;
        use crate::ai::tools::storage::mutation_log;

        let root = std::env::temp_dir().join(format!(
            "show-changes-history-big-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join("history.jsonl");
        app.session_id = "show-changes-history-big".into();
        let context = runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        runtime_ctx::DRIVER_CTX.sync_scope(context, || {
            let log_path = mutation_log::log_path().unwrap();
            std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
            let entry = MutationEntry {
                entry_id: None,
                turn_id: Some(1),
                task_id: None,
                tool_call_id: None,
                seq: 1,
                ts: "2026-11-20T00:00:00Z".into(),
                path: "/proj/big.rs".into(),
                op: "write".into(),
                // Capped snapshots make `entry_snippet` use the recorded diff.
                before: Some("[truncated 100 bytes]\nold\n".into()),
                before_state: Some(BeforeState::Present),
                after: Some("[truncated 100 bytes]\nnew\n".into()),
                diff: Some(format!("+{}\n", "x".repeat(40_000))),
            };
            std::fs::write(&log_path, format!("{}\n", serde_json::to_string(&entry).unwrap()))
                .unwrap();
            let result = handle_show_changes(&json!({
                "include_git": false,
                "mode": "history",
                "history_limit": 1,
                "history_diff_lines": 100,
                "limit_snippet": 0,
            }));
            assert_eq!(result["history_page"]["truncated_by_bytes"], true);
            assert_eq!(result["history_page"]["diff_omitted"], 1);
            assert_eq!(result["history"][0]["diff_truncated"], true);
            assert!(result["history"][0].get("diff").is_none());
            let shown = result["history_page"]["shown_bytes"].as_u64().unwrap();
            let max = result["history_page"]["max_bytes"].as_u64().unwrap();
            assert!(shown <= max, "shown={shown} max={max}");
        });
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn external_diff_rejects_unavailable_initial_snapshot_before_spawning() {
        for state in [BeforeState::Present, BeforeState::Unknown] {
            let entries = [MutationEntry {
                entry_id: None,
                turn_id: None,
                task_id: None,
                tool_call_id: None,
                seq: 0,
                ts: "t".into(),
                path: "/proj/file.rs".into(),
                op: "write".into(),
                before: None,
                before_state: Some(state),
                after: Some("replacement\n".into()),
                diff: None,
            }];
            let grouped = changes::grouped_changes(&entries);
            for editor in [EditorKind::Vscode, EditorKind::Cursor] {
                assert!(try_open_single_file_diff(&grouped[0], &editor).is_none());
            }
        }
    }

    fn with_change_session<R>(name: &str, test: impl FnOnce(&std::path::Path) -> R) -> R {
        use crate::ai::driver::runtime_ctx;
        let root = std::env::temp_dir().join(format!(
            "show-changes-{name}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join("history.jsonl");
        app.session_id = format!("show-changes-{name}");
        let context = runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        let result = runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || {
            runtime_ctx::DRIVER_CTX.sync_scope(context, || test(&root))
        });
        std::fs::remove_dir_all(root).unwrap();
        result
    }

    fn test_entry(
        seq: u64,
        ts: &str,
        path: &str,
        turn_id: Option<usize>,
        task_id: Option<&str>,
    ) -> MutationEntry {
        MutationEntry {
            entry_id: Some(format!("entry-{seq}")),
            turn_id,
            task_id: task_id.map(str::to_string),
            tool_call_id: None,
            seq,
            ts: ts.to_string(),
            path: path.to_string(),
            op: "write".to_string(),
            before: Some(format!("old-{seq}\n")),
            before_state: Some(BeforeState::Present),
            after: Some(format!("new-{seq}\n")),
            diff: Some(format!("- old-{seq}\n+ new-{seq}\n")),
        }
    }

    fn write_mutation_log(entries: &[MutationEntry]) {
        use crate::ai::tools::storage::mutation_log;
        let path = mutation_log::log_path().expect("session mutation log path");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut content = String::new();
        for entry in entries {
            content.push_str(&serde_json::to_string(entry).unwrap());
            content.push('\n');
        }
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn show_changes_window_filters_by_seq_time_path_and_provenance() {
        with_change_session("window", |root| {
            let now = chrono::Utc::now();
            let old = (now - chrono::Duration::hours(2)).to_rfc3339();
            let recent = now.to_rfc3339();
            let a = root.join("a.rs").display().to_string();
            let b = root.join("b.rs").display().to_string();
            write_mutation_log(&[
                test_entry(1, &old, &a, Some(1), None),
                test_entry(2, &recent, &b, Some(2), Some("task-1")),
                test_entry(3, &recent, &a, Some(2), None),
            ]);

            let by_seq = handle_show_changes(&json!({
                "include_git": false,
                "since_seq": 2,
                "limit_snippet": 0,
            }));
            assert_eq!(by_seq["window"]["session_mutations"], 3);
            assert_eq!(by_seq["window"]["matched_mutations"], 2);
            assert_eq!(by_seq["window"]["first_seq"], 2);
            assert_eq!(by_seq["window"]["last_seq"], 3);
            assert_eq!(by_seq["next_seq"], 4);
            assert_eq!(by_seq["changed_files"], 2);
            assert_eq!(by_seq["has_session_mutations"], true);
            assert!(by_seq["patch_bytes"].as_u64().unwrap() > 0);

            let by_time = handle_show_changes(&json!({
                "include_git": false,
                "since": "1h",
                "limit_snippet": 0,
            }));
            assert_eq!(by_time["window"]["matched_mutations"], 2);
            assert_eq!(by_time["window"]["first_seq"], 2);

            let by_turn = handle_show_changes(&json!({
                "include_git": false,
                "turn_id": 2,
                "limit_snippet": 0,
            }));
            assert_eq!(by_turn["window"]["matched_mutations"], 2);

            let by_task = handle_show_changes(&json!({
                "include_git": false,
                "task_id": "task-1",
                "limit_snippet": 0,
            }));
            assert_eq!(by_task["window"]["matched_mutations"], 1);
            assert_eq!(by_task["window"]["first_seq"], 2);
            assert_eq!(by_task["changed_files"], 1);

            let by_path = handle_show_changes(&json!({
                "include_git": false,
                "path": "a.rs",
                "limit_snippet": 0,
            }));
            assert_eq!(by_path["window"]["matched_mutations"], 2);
            assert_eq!(by_path["changed_files"], 1);
            let text = by_path["text"].as_str().unwrap();
            assert!(text.contains("a.rs"));
            assert!(!text.contains("b.rs"));

            let empty = handle_show_changes(&json!({
                "include_git": false,
                "since_seq": 99,
                "limit_snippet": 0,
            }));
            assert_eq!(empty["changed_files"], 0);
            assert_eq!(empty["has_session_mutations"], true);
            assert_eq!(empty["window"]["empty_reason"], "no_matches_in_window");
            assert_eq!(empty["window"]["matched_mutations"], 0);
            assert_eq!(empty["next_seq"], 4);
            assert_eq!(empty["patch_bytes"], 0);
        });
    }

    #[test]
    fn show_changes_history_mode_paginates_and_numbers_versions() {
        with_change_session("history", |root| {
            let now = chrono::Utc::now().to_rfc3339();
            let a = root.join("a.rs").display().to_string();
            let b = root.join("b.rs").display().to_string();
            write_mutation_log(&[
                test_entry(1, &now, &a, Some(1), None),
                test_entry(2, &now, &b, Some(1), None),
                test_entry(3, &now, &a, Some(2), None),
            ]);

            let first = handle_show_changes(&json!({
                "include_git": false,
                "mode": "history",
                "path": "a.rs",
                "history_limit": 1,
                "history_diff_lines": 5,
                "limit_snippet": 0,
            }));
            assert_eq!(first["history"].as_array().unwrap().len(), 1);
            assert_eq!(first["history"][0]["seq"], 1);
            assert_eq!(first["history"][0]["version"], 1);
            assert_eq!(first["history"][0]["turn_id"], 1);
            assert!(
                first["history"][0]["diff"]
                    .as_str()
                    .unwrap()
                    .contains("- old-1")
            );
            assert_eq!(first["history_page"]["total"], 2);
            assert_eq!(first["history_page"]["shown"], 1);
            assert_eq!(first["history_page"]["complete"], false);
            assert_eq!(first["history_page"]["next_after_seq"], 1);
            let text = first["text"].as_str().unwrap();
            assert!(text.contains("seq=1"));
            assert!(text.contains("version=1"));

            let second = handle_show_changes(&json!({
                "include_git": false,
                "mode": "history",
                "path": "a.rs",
                "after_seq": 1,
                "history_limit": 1,
                "history_diff_lines": 0,
                "limit_snippet": 0,
            }));
            assert_eq!(second["history"].as_array().unwrap().len(), 1);
            assert_eq!(second["history"][0]["seq"], 3);
            assert_eq!(second["history"][0]["version"], 2);
            assert!(second["history"][0].get("diff").is_none());
            assert_eq!(second["history_page"]["total"], 2);
            assert_eq!(second["history_page"]["complete"], true);
            assert_eq!(second["history_page"]["next_after_seq"], Value::Null);
        });
    }

    #[test]
    fn show_changes_rejects_invalid_window_arguments() {
        with_change_session("invalid", |_root| {
            let error = handle_show_changes_result(&json!({"since": "banana"})).unwrap_err();
            assert!(error.contains("invalid time bound"));
            let error =
                handle_show_changes_result(&json!({"since_seq": 5, "until_seq": 2})).unwrap_err();
            assert!(error.contains("since_seq"));
            let error =
                handle_show_changes_result(&json!({"since": "1h", "until": "2h"})).unwrap_err();
            assert!(error.contains("since"));
            let error = handle_show_changes_result(&json!({"mode": "bogus"})).unwrap_err();
            assert!(error.contains("mode"));
            let error = handle_show_changes_result(&json!({"path": 42})).unwrap_err();
            assert!(error.contains("path"));
            let wrapped = handle_show_changes(&json!({"since": "banana"}));
            assert!(
                wrapped["error"]
                    .as_str()
                    .unwrap()
                    .contains("invalid time bound")
            );
        });
    }

    #[test]
    fn show_changes_time_bound_parser_accepts_supported_forms() {
        let now = chrono::Utc::now();
        for raw in ["2026-11-20T10:00:00Z", "2026-11-20", "30m", "2h", "1d", "1w"] {
            assert!(parse_time_bound(raw, now).is_ok(), "{raw}");
        }
        assert_eq!(
            parse_time_bound("2026-11-20", now).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2026, 11, 20)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
        );
        for raw in ["banana", "-5m", "0m", "a lot", "10x"] {
            assert!(parse_time_bound(raw, now).is_err(), "{raw}");
        }
    }
}

fn open_patch_path_with_editor(
    patch_path: &std::path::Path,
    kind: &EditorKind,
) -> Result<String, String> {
    // Match storage::changes::open_patch while allowing an explicit path.
    let patch_str = patch_path.display().to_string();
    let (prog, args): (&str, Vec<String>) = match kind {
        EditorKind::Vscode => ("code", vec![patch_str.clone()]),
        EditorKind::Cursor => ("cursor", vec![patch_str.clone()]),
        EditorKind::Idea => ("idea", vec!["diff".to_string(), patch_str.clone()]),
        EditorKind::Git => {
            return Err("git difftool requires a git work tree; use code/cursor/open".to_string());
        }
        EditorKind::SystemOpen => {
            #[cfg(target_os = "macos")]
            {
                ("open", vec![patch_str.clone()])
            }
            #[cfg(not(target_os = "macos"))]
            {
                ("xdg-open", vec![patch_str.clone()])
            }
        }
        EditorKind::Auto => {
            #[cfg(target_os = "macos")]
            {
                ("open", vec![patch_str.clone()])
            }
            #[cfg(not(target_os = "macos"))]
            {
                ("xdg-open", vec![patch_str.clone()])
            }
        }
    };
    let cmd_display = format!("{} {}", prog, args.join(" "));
    std::process::Command::new(prog)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to launch editor '{}': {e}", prog))?;
    Ok(cmd_display)
}
