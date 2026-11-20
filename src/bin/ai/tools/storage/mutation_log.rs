//! Session file mutations and command-check observations share one append-only log.
//! Recording is best effort, excludes runtime artifacts, and never changes tool results.

use rust_tools::commonw::FastMap;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{File, Metadata, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::SystemTime;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use super::file_store::current_session_assets_dir;

/// The registry lock only selects a path's state; recovery and I/O use its own lock.
static LOG_STATES: LazyLock<Mutex<FastMap<PathBuf, Arc<Mutex<LogState>>>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

#[derive(PartialEq, Eq)]
struct FileIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    created: Option<SystemTime>,
}

impl FileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Self {
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            created: metadata.created().ok(),
        }
    }
}

#[derive(Default)]
struct LogState {
    identity: Option<FileIdentity>,
    modified: Option<SystemTime>,
    observed_len: u64,
    /// Bytes through the last newline; an interrupted final line stays uncommitted.
    offset: u64,
    prefix: Vec<u8>,
    boundary: Vec<u8>,
    mutations: u64,
    origins: FastMap<String, CheckObservation>,
    sessions: FastMap<String, CheckObservation>,
    origin_order: FastMap<String, usize>,
    /// Ordered starts preserve latest-check selection across delayed old polls.
    command_checks: FastMap<(String, String), BTreeMap<usize, ReminderCheck>>,
    latest_check_counts: FastMap<u64, ReminderCheckCounts>,
    latest_check_total: usize,
    #[cfg(test)]
    recovered_records: usize,
    #[cfg(test)]
    full_reads: usize,
    #[cfg(test)]
    recovery_pause: Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>,
}

#[derive(Clone, Copy)]
struct ReminderCheck {
    revision: u64,
    pending: bool,
}

#[derive(Default)]
struct ReminderCheckCounts {
    total: usize,
    pending: usize,
}

/// Small request-preparation state, without historical observations or snapshots.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VerificationReminderSummary {
    pub revision: u64,
    pub stale_checks: usize,
    pub gaps: usize,
}

impl LogState {
    fn invalidate(&mut self) {
        #[cfg(test)]
        let recovered_records = self.recovered_records;
        #[cfg(test)]
        let full_reads = self.full_reads;
        *self = Self::default();
        #[cfg(test)]
        {
            self.recovered_records = recovered_records;
            self.full_reads = full_reads;
        }
    }

    fn recover(&mut self, file: &mut File) -> io::Result<()> {
        let result = self.recover_inner(file);
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    fn recover_inner(&mut self, file: &mut File) -> io::Result<()> {
        #[cfg(test)]
        if let Some((entered, resume)) = self.recovery_pause.take() {
            let _ = entered.send(());
            let _ = resume.recv_timeout(std::time::Duration::from_secs(30));
        }
        let metadata = file.metadata()?;
        let identity = FileIdentity::from_metadata(&metadata);
        let modified = metadata.modified().ok();
        let len = metadata.len();
        let replaced = self.identity.as_ref() != Some(&identity);
        let truncated = len < self.observed_len;
        let rewritten = len == self.observed_len && modified != self.modified;
        if replaced || truncated || rewritten {
            self.invalidate();
        } else if len == self.observed_len && modified == self.modified {
            return Ok(());
        } else if self.offset > 0
            && (read_window(file, 0, self.prefix.len())? != self.prefix
                || read_window(
                    file,
                    self.offset - self.boundary.len() as u64,
                    self.boundary.len(),
                )? != self.boundary)
        {
            // Detect truncate-and-regrow between calls without rereading the history.
            self.invalidate();
        }

        file.seek(SeekFrom::Start(self.offset))?;
        let mut reader = BufReader::new((&mut *file).take(len - self.offset));
        let mut line = Vec::new();
        loop {
            let bytes = read_committed_line(&mut reader, &mut line)?;
            if bytes == 0 {
                break;
            }
            self.offset += bytes as u64;
            #[cfg(test)]
            {
                self.recovered_records += 1;
            }
            self.observe_record(&line);
        }
        drop(reader);
        let window = self.offset.min(64) as usize;
        self.prefix = read_window(file, 0, window)?;
        self.boundary = read_window(file, self.offset - window as u64, window)?;
        self.identity = Some(identity);
        self.modified = modified;
        self.observed_len = len;
        Ok(())
    }

    fn observe_record(&mut self, line: &[u8]) {
        let Ok(kind) = serde_json::from_slice::<RecordKind>(line) else {
            return;
        };
        if kind.is_check() {
            if let Ok(check) = serde_json::from_slice::<CheckObservation>(line) {
                let next_order = self.origin_order.len();
                let order = *self
                    .origin_order
                    .entry(check.originating_command_call_id.clone())
                    .or_insert(next_order);
                let key = (check.command.clone(), check.cwd.clone());
                if let Some(prior) = self.origins.get(&check.originating_command_call_id) {
                    let prior_key = (prior.command.clone(), prior.cwd.clone());
                    if prior_key != key {
                        self.update_command_check(prior_key, order, None);
                    }
                }
                self.update_command_check(
                    key,
                    order,
                    Some(ReminderCheck {
                        revision: check.covers_revision,
                        pending: check.status != CheckStatus::Success || check.reported_failure,
                    }),
                );
                if let Some(session) = &check.command_session_id {
                    self.sessions.insert(session.clone(), check.clone());
                }
                self.origins
                    .insert(check.originating_command_call_id.clone(), check);
            }
        } else if serde_json::from_slice::<MutationEntry>(line).is_ok() {
            self.mutations += 1;
        }
    }

    fn update_command_check(
        &mut self,
        key: (String, String),
        order: usize,
        check: Option<ReminderCheck>,
    ) {
        let commands = self.command_checks.entry(key.clone()).or_default();
        let previous = commands.last_key_value().map(|(_, check)| *check);
        if let Some(check) = check {
            commands.insert(order, check);
        } else {
            commands.remove(&order);
        }
        let latest = commands.last_key_value().map(|(_, check)| *check);
        if commands.is_empty() {
            self.command_checks.remove(&key);
        }
        if let Some(previous) = previous {
            self.latest_check_total -= 1;
            let counts = self
                .latest_check_counts
                .get_mut(&previous.revision)
                .unwrap();
            counts.total -= 1;
            counts.pending -= usize::from(previous.pending);
            if counts.total == 0 {
                self.latest_check_counts.remove(&previous.revision);
            }
        }
        if let Some(latest) = latest {
            self.latest_check_total += 1;
            let counts = self.latest_check_counts.entry(latest.revision).or_default();
            counts.total += 1;
            counts.pending += usize::from(latest.pending);
        }
    }

    fn reminder_summary(&self) -> VerificationReminderSummary {
        let current = self.latest_check_counts.get(&self.mutations);
        let stale_checks = self.latest_check_total - current.map_or(0, |counts| counts.total);
        VerificationReminderSummary {
            revision: self.mutations,
            stale_checks,
            gaps: stale_checks
                + current.map_or(0, |counts| counts.pending)
                + usize::from(self.mutations > 0 && self.latest_check_total == 0),
        }
    }

    fn prepare_append(&mut self, file: &mut File) -> io::Result<()> {
        self.recover(file)?;
        if self.offset < self.observed_len {
            // Preserve an interrupted tail and separate it from the next valid record.
            if let Err(error) = file.write_all(b"\n") {
                self.invalidate();
                return Err(error);
            }
            self.recover(file)?;
        }
        Ok(())
    }

    fn append(&mut self, file: &mut File, record: &impl Serialize) -> io::Result<()> {
        let mut line = serde_json::to_vec(record)?;
        line.push(b'\n');
        if let Err(error) = file.write_all(&line) {
            // A failed write may have left a partial record; never cache its revision.
            self.invalidate();
            return Err(error);
        }
        // Recover only the successfully appended bytes, including external additions.
        self.recover(file)
    }
}

/// A newline commits a JSONL record; an interrupted tail is invisible to readers.
fn read_committed_line(reader: &mut impl BufRead, line: &mut Vec<u8>) -> io::Result<usize> {
    line.clear();
    let bytes = reader.read_until(b'\n', line)?;
    Ok(if line.last() == Some(&b'\n') {
        bytes
    } else {
        0
    })
}

fn read_window(file: &mut File, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0; len];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn state_at(path: &Path) -> Arc<Mutex<LogState>> {
    let key = std::fs::canonicalize(path).unwrap_or_else(|_| {
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
        let parent = parent.unwrap_or_else(|| Path::new("."));
        let parent = std::fs::canonicalize(parent)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default().join(parent));
        parent.join(path.file_name().unwrap_or_default())
    });
    LOG_STATES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .entry(key)
        .or_default()
        .clone()
}

fn with_log_writer<T>(
    path: &Path,
    write: impl FnOnce(&mut LogState, &mut File) -> io::Result<T>,
) -> io::Result<T> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let shared = state_at(path);
    let mut state = shared.lock().unwrap_or_else(|poison| poison.into_inner());
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)?;
    // Cooperating processes also recover and assign revisions under the file lock.
    file.lock()?;
    state.prepare_append(&mut file)?;
    write(&mut state, &mut file)
}

fn with_log_reader<T>(path: &Path, read: impl FnOnce(&LogState) -> T) -> io::Result<T> {
    let shared = state_at(path);
    let mut state = shared.lock().unwrap_or_else(|poison| poison.into_inner());
    match File::open(path) {
        Ok(mut file) => {
            file.lock_shared()?;
            state.recover(&mut file)?;
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => state.invalidate(),
        Err(error) => return Err(error),
    }
    Ok(read(&state))
}

#[derive(Deserialize)]
struct RecordKind {
    record_type: Option<Value>,
}

impl RecordKind {
    fn is_check(&self) -> bool {
        self.record_type.as_ref().and_then(Value::as_str) == Some("check_observation")
    }
}

thread_local! {
    static TOOL_CALL_ID: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// The synchronous tool executor owns this scope, including panic recovery.
pub(crate) fn with_tool_call<T>(id: &str, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TOOL_CALL_ID.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(TOOL_CALL_ID.with(|slot| slot.replace(Some(id.to_string()))));
    run()
}

/// Bound file snapshots; the authoritative diff has its own input/output caps.
const MAX_CONTENT_BYTES: usize = 16 * 1024;

/// Recognizable prefix of the truncation marker. `is_capped` and the rendering
/// side use it to tell whether a before/after snapshot is complete.
const TRUNCATED_MARKER: &str = "[truncated ";

/// Per-entry cap for the authoritative diff. The diff normally contains only the
/// changed lines, far smaller than the file itself; in the extreme case (full
/// rewrite) it is capped here. Capping happens at a line boundary, so the
/// rendered output is still a truthful partial diff — unlike truncated
/// snapshots, it never misjudges the truncation edge as a deletion.
const MAX_DIFF_BYTES: usize = 64 * 1024;

/// Bound each input to `entry_diff` before allocating line vectors and diff text.
/// Capping only the output would still materialize the whole diff for a large rewrite.
const MAX_DIFF_INPUT_BYTES: usize = 1024 * 1024;

/// Initial file presence, independent of whether its contents were captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BeforeState {
    Absent,
    Present,
    Unknown,
}

/// One recorded file mutation.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct MutationEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Persistent mutation order, independent of command observations.
    pub seq: u64,
    /// ISO8601 UTC timestamp.
    pub ts: String,
    /// Absolute file path.
    pub path: String,
    /// Operation: `"write"` or `"delete"`.
    pub op: String,
    /// Captured preimage. None can mean absence or unavailable contents; consult
    /// `effective_before_state` before treating it as an empty file.
    pub before: Option<String>,
    /// Missing in legacy logs, whose None preimages retain their old meaning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_state: Option<BeforeState>,
    /// Captured postimage; None for deletion.
    pub after: Option<String>,
    /// Line-level diff of this write (`- ` old lines / `+ ` new lines), computed
    /// from the before/after content within `MAX_DIFF_INPUT_BYTES`. Once a
    /// snapshot is truncated a reliable diff cannot be rebuilt (the truncation
    /// edge reads as a deletion), so this diff is the authoritative source for
    /// audit and display; old logs that lack the field deserialize to None and
    /// rendering falls back to snapshot diffs.
    pub diff: Option<String>,
}

impl MutationEntry {
    pub(crate) fn effective_before_state(&self) -> BeforeState {
        self.before_state.unwrap_or(if self.before.is_some() {
            BeforeState::Present
        } else {
            BeforeState::Absent
        })
    }
}

/// Mutation-log path of the active session.
pub(crate) fn log_path() -> Option<PathBuf> {
    current_session_assets_dir().map(|d| d.join("mutation_log.jsonl"))
}

/// Legacy entry point for callers whose None preimage means known absence.
pub(crate) fn record(path: &Path, op: &str, before: Option<&str>, after: Option<&str>) {
    let before_state = if before.is_some() {
        BeforeState::Present
    } else {
        BeforeState::Absent
    };
    record_with_before_state(path, op, before, after, before_state);
}

/// Best-effort recording with file presence separate from snapshot availability.
/// Runtime artifacts are skipped and recording failures never affect the write.
pub(crate) fn record_with_before_state(
    path: &Path,
    op: &str,
    before: Option<&str>,
    after: Option<&str>,
    before_state: BeforeState,
) {
    let Some(assets_dir) = current_session_assets_dir() else {
        // Tests and one-shot callers without a driver context do not record.
        return;
    };
    if should_skip(path, &assets_dir) {
        return;
    }

    let entry = make_entry(path, op, before, after, before_state);
    append_entry(&assets_dir.join("mutation_log.jsonl"), &entry);
}

fn make_entry(
    path: &Path,
    op: &str,
    before: Option<&str>,
    after: Option<&str>,
    before_state: BeforeState,
) -> MutationEntry {
    MutationEntry {
        entry_id: Some(uuid::Uuid::new_v4().to_string()),
        turn_id: crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .try_with(|(_, id)| *id)
            .ok(),
        task_id: crate::ai::driver::runtime_ctx::try_subagent_task_id(),
        tool_call_id: TOOL_CALL_ID.with(|slot| slot.borrow().clone()),
        seq: 0,
        ts: Utc::now().to_rfc3339(),
        path: path.to_string_lossy().into_owned(),
        op: op.to_string(),
        before: before.map(cap_content),
        before_state: Some(before_state),
        after: after.map(cap_content),
        diff: if before.is_none() && before_state != BeforeState::Absent {
            Some("…[before snapshot unavailable; contents not compared]\n".to_string())
        } else {
            entry_diff(before, after)
        },
    }
}

/// Cap content on a UTF-8 boundary and identify omitted bytes.
pub(crate) fn cap_content(content: &str) -> String {
    if content.len() <= MAX_CONTENT_BYTES {
        return content.to_string();
    }
    let mut end = MAX_CONTENT_BYTES;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n…{TRUNCATED_MARKER}{} more bytes; read the file for full content]",
        &content[..end],
        content.len() - end
    )
}

/// Whether `cap_content` truncated the content. Rendering uses this to decide
/// whether a snapshot can be diffed directly.
pub(crate) fn is_capped(s: &str) -> bool {
    s.contains(TRUNCATED_MARKER)
}

/// Truncates `s` to at most `MAX_DIFF_INPUT_BYTES` bytes at a char boundary.
/// Returns a prefix slice (no allocation) plus whether truncation happened.
fn cap_diff_input(s: &str) -> (&str, bool) {
    if s.len() <= MAX_DIFF_INPUT_BYTES {
        return (s, false);
    }
    let mut end = MAX_DIFF_INPUT_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    (&s[..end], true)
}

/// Computes the line-level diff of this write from the before/after content.
/// For files within `MAX_DIFF_INPUT_BYTES` the diff is computed on the full
/// content and is always reliable; rendering prefers it over snapshot diffs to
/// avoid false deletions at the truncation edge. Content beyond the cap is never
/// scanned, so a large rewrite cannot materialize an O(file) diff; the result is
/// then a truthful leading partial and a truncation line is appended so a
/// partial diff is never mistaken for a complete one — or for "unchanged".
/// Identical uncapped line sequences produce no diff. Identical capped prefixes
/// still report that the remaining content was not compared.
pub(crate) fn entry_diff(before: Option<&str>, after: Option<&str>) -> Option<String> {
    let (b, b_capped) = match before {
        Some(s) => {
            let (t, c) = cap_diff_input(s);
            (Some(t), c)
        }
        None => (None, false),
    };
    let (a, a_capped) = match after {
        Some(s) => {
            let (t, c) = cap_diff_input(s);
            (Some(t), c)
        }
        None => (None, false),
    };
    let capped = b_capped || a_capped;

    let mut out = match (b, a) {
        (None, None) => return None,
        (None, Some(a)) => {
            let lines: Vec<&str> = a.lines().collect();
            if lines.is_empty() {
                return None;
            }
            let mut out = String::new();
            for l in &lines {
                out.push_str(&format!("+ {l}\n"));
            }
            out
        }
        (Some(b), None) => {
            let lines: Vec<&str> = b.lines().collect();
            if lines.is_empty() {
                return None;
            }
            let mut out = String::new();
            for l in &lines {
                out.push_str(&format!("- {l}\n"));
            }
            out
        }
        (Some(b), Some(a)) => {
            let bv: Vec<&str> = b.lines().collect();
            let av: Vec<&str> = a.lines().collect();
            let mut prefix = 0;
            while prefix < bv.len() && prefix < av.len() && bv[prefix] == av[prefix] {
                prefix += 1;
            }
            let mut suffix = 0;
            while suffix < bv.len() - prefix
                && suffix < av.len() - prefix
                && bv[bv.len() - 1 - suffix] == av[av.len() - 1 - suffix]
            {
                suffix += 1;
            }
            let mut out = String::new();
            for l in &bv[prefix..bv.len() - suffix] {
                out.push_str(&format!("- {l}\n"));
            }
            for l in &av[prefix..av.len() - suffix] {
                out.push_str(&format!("+ {l}\n"));
            }
            out
        }
    };

    if out.is_empty() {
        if capped {
            // Matching prefixes prove neither equality nor a difference in the
            // unscanned tail, including when the full inputs happen to be equal.
            out.push_str(&format!(
                "…[no differing lines within the first {MAX_DIFF_INPUT_BYTES} bytes; \
                 remaining content not compared; leading portion only]\n"
            ));
        } else {
            return None;
        }
    } else if capped {
        out.push_str(&format!(
            "…[diff input truncated at {MAX_DIFF_INPUT_BYTES} bytes; leading portion only — \
             read the file for the full change]\n"
        ));
    }
    Some(cap_diff(out))
}

/// Caps the diff text at `MAX_DIFF_BYTES`, cutting at a line boundary when
/// possible (never leaves a partial line). The capped diff is the leading
/// truthful portion, so rendering stays a correct partial view — unlike the
/// false deletions caused by diffing truncated snapshots.
fn cap_diff(mut out: String) -> String {
    if out.len() <= MAX_DIFF_BYTES {
        return out;
    }
    let mut end = MAX_DIFF_BYTES;
    while end > 0 && !out.is_char_boundary(end) {
        end -= 1;
    }
    while end > 0 && out.as_bytes()[end - 1] != b'\n' {
        end -= 1;
    }
    out.truncate(end);
    out.push_str(&format!(
        "…[diff truncated at {MAX_DIFF_BYTES} bytes; see the file for the rest]\n"
    ));
    out
}

/// Runtime session artifacts are outside the project mutation inventory.
fn should_skip(path: &std::path::Path, assets_dir: &std::path::Path) -> bool {
    let sessions_root = assets_dir.parent().unwrap_or(assets_dir);
    path.starts_with(sessions_root)
}

/// Assign order from the recovered count while holding this log's append lock.
fn append_entry(log_path: &Path, entry: &MutationEntry) {
    let _ = with_log_writer(log_path, |state, file| {
        let mut entry = entry.clone();
        entry.seq = state.mutations + 1;
        state.append(file, &entry)
    });
}

/// Read mutations in persistent order; absent or unreadable logs return empty.
pub(crate) fn read_all() -> Vec<MutationEntry> {
    let Some(log_path) = log_path() else {
        return Vec::new();
    };
    read_entries(&log_path)
}

/// Mutations and checks from one committed log prefix, without updating the cache.
#[derive(Default)]
pub(crate) struct CommittedSnapshot {
    pub mutations: Vec<MutationEntry>,
    pub checks: Vec<CheckObservation>,
}

impl CommittedSnapshot {
    pub(crate) fn revision(&self) -> u64 {
        self.mutations.len() as u64
    }
}

pub(crate) fn read_committed_snapshot() -> CommittedSnapshot {
    log_path()
        .and_then(|path| read_committed_snapshot_at(&path).ok())
        .unwrap_or_default()
}

/// Hold the shared file lock through the complete scan, matching append locking.
fn read_committed_snapshot_at(path: &Path) -> io::Result<CommittedSnapshot> {
    #[cfg(test)]
    {
        state_at(path).lock().unwrap().full_reads += 1;
    }
    let file = File::open(path)?;
    file.lock_shared()?;
    let mut reader = BufReader::new((&file).take(file.metadata()?.len()));
    let mut snapshot = CommittedSnapshot::default();
    let mut line = Vec::new();
    while read_committed_line(&mut reader, &mut line)? > 0 {
        let Ok(kind) = serde_json::from_slice::<RecordKind>(&line) else {
            continue;
        };
        if kind.is_check() {
            if let Ok(check) = serde_json::from_slice::<CheckObservation>(&line) {
                snapshot.checks.push(check);
            }
        } else if let Ok(entry) = serde_json::from_slice::<MutationEntry>(&line) {
            snapshot.mutations.push(entry);
        }
    }
    Ok(snapshot)
}

/// Read valid committed mutations only, skipping malformed lines and observations.
fn read_entries(log_path: &std::path::Path) -> Vec<MutationEntry> {
    read_committed_snapshot_at(log_path)
        .unwrap_or_default()
        .mutations
}

/// Check observations carry command provenance, never file snapshots or inventory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct CheckObservation {
    pub record_type: String,
    pub entry_id: String,
    pub ts: String,
    pub session_id: Option<String>,
    pub turn_id: Option<usize>,
    pub task_id: Option<String>,
    pub command: String,
    pub cwd: String,
    pub originating_command_call_id: String,
    pub tool_call_id: String,
    pub command_session_id: Option<String>,
    pub covers_revision: u64,
    pub status: CheckStatus,
    pub reported_failure: bool,
    pub observed_success: bool,
    #[serde(default)]
    pub partial_success: bool,
    pub success_guaranteed: bool,
    pub evidence_locator: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CheckStatus {
    Success,
    Failure,
    Running,
    Unknown,
}

impl From<crate::ai::driver::turn_runtime::progress::ToolResultProgressStatus> for CheckStatus {
    fn from(status: crate::ai::driver::turn_runtime::progress::ToolResultProgressStatus) -> Self {
        use crate::ai::driver::turn_runtime::progress::ToolResultProgressStatus as Status;
        match status {
            Status::Success => Self::Success,
            Status::Failure | Status::BlockedOutsideWorkspace(_) => Self::Failure,
            Status::Running => Self::Running,
            _ => Self::Unknown,
        }
    }
}

pub(crate) fn parse_mutation(line: &str) -> Result<Option<MutationEntry>, serde_json::Error> {
    if serde_json::from_str::<RecordKind>(line)?.is_check() {
        return Ok(None);
    }
    serde_json::from_str(line).map(Some)
}

#[cfg(test)]
fn revision_at(path: &Path) -> u64 {
    with_log_reader(path, |state| state.mutations).unwrap_or_default()
}

pub(crate) fn read_checks() -> Vec<CheckObservation> {
    read_committed_snapshot().checks
}

/// Recover newly completed lines under the shared lock and return cached counts.
pub(crate) fn verification_reminder_summary() -> VerificationReminderSummary {
    log_path()
        .map(|path| verification_reminder_summary_at(&path))
        .unwrap_or_default()
}

fn verification_reminder_summary_at(path: &Path) -> VerificationReminderSummary {
    with_log_reader(path, LogState::reminder_summary).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn recovered_records_at(path: &Path) -> usize {
    state_at(path).lock().unwrap().recovered_records
}

#[cfg(test)]
pub(crate) fn full_reads_at(path: &Path) -> usize {
    state_at(path).lock().unwrap().full_reads
}

#[cfg(test)]
fn read_checks_at(path: &Path) -> Vec<CheckObservation> {
    read_committed_snapshot_at(path).unwrap_or_default().checks
}

pub(crate) struct PendingCheck {
    path: PathBuf,
    observation: CheckObservation,
}

/// Capture the revision before dispatch, including commands that return a session.
pub(crate) fn begin_check(id: &str, name: &str, args: &Value) -> Option<PendingCheck> {
    let path = log_path()?;
    begin_check_at(&path, id, name, args)
}

fn begin_check_at(path: &Path, id: &str, name: &str, args: &Value) -> Option<PendingCheck> {
    let mut observation = if name == "execute_command" {
        let evidence = crate::ai::driver::turn_runtime::command_check_evidence(args, "")?;
        CheckObservation {
            record_type: "check_observation".into(),
            entry_id: uuid::Uuid::new_v4().to_string(),
            ts: Utc::now().to_rfc3339(),
            session_id: Some(crate::ai::driver::runtime_ctx::current_session_id_or_empty())
                .filter(|id| !id.is_empty()),
            turn_id: crate::ai::driver::runtime_ctx::TURN_IDENTITY
                .try_with(|(_, id)| *id)
                .ok(),
            task_id: crate::ai::driver::runtime_ctx::try_subagent_task_id(),
            command: crate::ai::tools::service::execute_command_arg(args)?.to_string(),
            cwd: args
                .get("cwd")
                .and_then(Value::as_str)
                .filter(|cwd| !cwd.trim().is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| {
                    crate::ai::driver::runtime_ctx::effective_cwd()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                }),
            originating_command_call_id: id.to_string(),
            tool_call_id: id.to_string(),
            command_session_id: None,
            covers_revision: 0,
            status: CheckStatus::Running,
            reported_failure: false,
            observed_success: false,
            partial_success: false,
            success_guaranteed: evidence.success_guaranteed,
            evidence_locator: format!("tool_call_id:{id}"),
        }
    } else if name == "command_session"
        && matches!(
            args.get("action").and_then(Value::as_str),
            Some("poll" | "write" | "close")
        )
    {
        let session = args.get("session_id")?.as_str()?;
        let mut origin = with_log_reader(path, |state| state.sessions.get(session).cloned())
            .ok()
            .flatten()?;
        origin.tool_call_id = id.to_string();
        origin.evidence_locator = format!("tool_call_id:{id}");
        if args.get("action").and_then(Value::as_str) == Some("close") {
            origin.status = CheckStatus::Unknown;
        }
        origin
    } else {
        return None;
    };
    observation.entry_id = uuid::Uuid::new_v4().to_string();
    observation.ts = Utc::now().to_rfc3339();
    if name == "execute_command" {
        // Snapshot and start observation share the same append serialization point.
        let _ = with_log_writer(path, |state, file| {
            observation.covers_revision = state.mutations;
            state.append(file, &observation)
        });
    }
    Some(PendingCheck {
        path: path.to_path_buf(),
        observation,
    })
}

/// Polls reuse the original command/revision and accumulate failures across chunks.
pub(crate) fn finish_check(pending: Option<PendingCheck>, name: &str, args: &Value, text: &str) {
    let Some(PendingCheck {
        path,
        mut observation,
    }) = pending
    else {
        return;
    };
    let _ = with_log_writer(&path, |state, file| {
        finish_check_at(state, file, &mut observation, name, args, text)
    });
}

fn finish_check_at(
    state: &mut LogState,
    file: &mut File,
    observation: &mut CheckObservation,
    name: &str,
    args: &Value,
    text: &str,
) -> io::Result<()> {
    if let Some(prior) = state.origins.get(&observation.originating_command_call_id) {
        observation.reported_failure |=
            prior.reported_failure || prior.status == CheckStatus::Failure;
        observation.observed_success |= prior.observed_success;
        observation.partial_success |= prior.partial_success;
    }
    if name == "command_session"
        && crate::ai::driver::turn_runtime::command_session_observation(text).is_some_and(
            |(session, _)| {
                session.as_deref().is_some_and(|session| {
                    observation.command_session_id.as_deref() != Some(session)
                })
            },
        )
    {
        observation.status = CheckStatus::Unknown;
        observation.entry_id = uuid::Uuid::new_v4().to_string();
        observation.ts = Utc::now().to_rfc3339();
        return state.append(file, observation);
    }
    let command_args = json!({"command": observation.command});
    if let Some(evidence) =
        crate::ai::driver::turn_runtime::command_check_evidence(&command_args, text)
    {
        observation.reported_failure |= evidence.reported_failure
            || CheckStatus::from(evidence.status.clone()) == CheckStatus::Failure;
        observation.observed_success |= evidence.observed_success;
        observation.partial_success |= evidence.partial_success;
        observation.status = evidence.status.into();
        if name == "execute_command" {
            observation.command_session_id = evidence.session_id;
        }
    }
    if name == "command_session" {
        let envelope = crate::ai::driver::turn_runtime::command_session_observation(text);
        observation.status = match envelope {
            Some((session, status))
                if session.as_deref().is_none_or(|session| {
                    observation.command_session_id.as_deref() == Some(session)
                }) =>
            {
                status.into()
            }
            _ => CheckStatus::Unknown,
        };
        if args.get("action").and_then(Value::as_str) == Some("close") {
            observation.status = CheckStatus::Unknown;
        }
    }
    observation.partial_success |= observation.observed_success && observation.reported_failure;
    if observation.status == CheckStatus::Success {
        observation.status = if observation.reported_failure {
            CheckStatus::Failure
        } else if !observation.success_guaranteed {
            CheckStatus::Unknown
        } else {
            CheckStatus::Success
        };
    }
    observation.entry_id = uuid::Uuid::new_v4().to_string();
    observation.ts = Utc::now().to_rfc3339();
    state.append(file, observation)
}

/// Collapse polls by origin, then select the latest start for each command/cwd.
/// A delayed old poll cannot override a newer recheck of the same command.
fn latest_checks(checks: &[CheckObservation]) -> Vec<CheckObservation> {
    let mut origins = Vec::<CheckObservation>::new();
    let mut indices = FastMap::default();
    for check in checks {
        if let Some(&index) = indices.get(&check.originating_command_call_id) {
            origins[index] = check.clone();
        } else {
            indices.insert(check.originating_command_call_id.clone(), origins.len());
            origins.push(check.clone());
        }
    }
    let mut commands = FastMap::default();
    for (index, check) in origins.iter().enumerate() {
        commands.insert((&check.command, &check.cwd), index);
    }
    origins
        .iter()
        .enumerate()
        .filter(|(index, check)| commands.get(&(&check.command, &check.cwd)) == Some(index))
        .map(|(_, check)| check.clone())
        .collect()
}

/// Page whole observations; commands and evidence locators are never cut mid-record.
pub(crate) struct VerificationDetailsPage {
    pub offset: usize,
    pub limit: usize,
    /// Zero disables the cap; nonzero budgets are normalized to fit an empty array.
    pub max_bytes: usize,
}

impl Default for VerificationDetailsPage {
    fn default() -> Self {
        Self {
            offset: 0,
            limit: 8,
            max_bytes: 8 * 1024,
        }
    }
}

pub(crate) fn verification_summary(
    revision: u64,
    checks: &[CheckObservation],
    details: bool,
) -> Value {
    verification_summary_page(
        revision,
        checks,
        details.then_some(&VerificationDetailsPage::default()),
    )
}

/// Size of a pretty-printed observation inside `verification.checks`, including
/// its indentation and array separator. The non-empty array adds six more bytes.
fn verification_detail_bytes(detail: &Value) -> usize {
    let text = serde_json::to_string_pretty(detail).expect("observation is a JSON value");
    text.len() + 6 * (text.bytes().filter(|byte| *byte == b'\n').count() + 1) + 2
}

pub(crate) fn verification_summary_page(
    revision: u64,
    checks: &[CheckObservation],
    page: Option<&VerificationDetailsPage>,
) -> Value {
    let latest = latest_checks(checks);
    let stale = latest
        .iter()
        .filter(|check| check.covers_revision != revision)
        .count();
    let pending = latest
        .iter()
        .filter(|check| {
            check.covers_revision == revision
                && (check.status != CheckStatus::Success || check.reported_failure)
        })
        .count();
    let missing = revision > 0 && latest.is_empty();
    let gaps = stale + pending + usize::from(missing);
    let current = latest
        .iter()
        .filter(|check| check.covers_revision == revision)
        .collect::<Vec<_>>();
    let statuses = |status| {
        current
            .iter()
            .filter(|check| check.status == status)
            .count()
    };
    let status = if statuses(CheckStatus::Failure) > 0 {
        "failure"
    } else if statuses(CheckStatus::Running) > 0 {
        "running"
    } else if missing || statuses(CheckStatus::Unknown) > 0 {
        "unknown"
    } else if stale > 0 {
        "stale"
    } else if latest.is_empty() {
        "no_changes"
    } else {
        "success"
    };
    let mut result = json!({
        "revision": revision,
        "verification": {"current": current.len(), "stale": stale, "pending": pending,
            "status": status,
            "current_statuses": {"success": statuses(CheckStatus::Success), "failure": statuses(CheckStatus::Failure), "running": statuses(CheckStatus::Running), "unknown": statuses(CheckStatus::Unknown)},
            "scope": "recorded_commands_only", "query_hint": "show_changes(verification_details=true)"},
        "verification_gaps": {"count": gaps, "missing_checks": missing, "stale_checks": stale, "pending_checks": pending},
    });
    if let Some(page) = page {
        let total = latest.len();
        let offset = page.offset.min(total);
        let limit = if page.limit == 0 {
            usize::MAX
        } else {
            page.limit
        };
        let max_bytes = if page.max_bytes == 0 {
            0
        } else {
            page.max_bytes.max(2)
        };
        let mut shown = Vec::new();
        let mut shown_bytes = 2;
        let mut total_bytes = if total == 0 { 2 } else { 6 };
        let mut next_check_bytes = None;
        let mut stopped = false;
        for (index, check) in latest.into_iter().enumerate() {
            let detail = json!({
                "freshness": if check.covers_revision == revision { "current" } else { "stale" },
                "observation": check,
            });
            let bytes = verification_detail_bytes(&detail);
            total_bytes += bytes;
            if index < offset || stopped {
                continue;
            }
            let candidate_bytes = if shown.is_empty() { 6 } else { shown_bytes } + bytes;
            if shown.len() >= limit || (max_bytes != 0 && candidate_bytes > max_bytes) {
                stopped = true;
                next_check_bytes = Some(6 + bytes);
                continue;
            }
            shown_bytes = candidate_bytes;
            shown.push(detail);
        }
        let next_offset = offset + shown.len();
        let complete = next_offset == total;
        let query_hint = if complete {
            "All remaining latest checks shown.".to_string()
        } else if shown.is_empty() {
            format!(
                "show_changes(verification_details=true, verification_offset={next_offset}, verification_max_bytes=0) for the next whole observation; verification_full=true returns all latest checks."
            )
        } else {
            format!(
                "show_changes(verification_details=true, verification_offset={next_offset}) for the next page; verification_full=true returns all latest checks."
            )
        };
        result["verification"]["checks"] = json!(shown);
        result["verification"]["checks_page"] = json!({
            "offset": offset,
            "shown": next_offset - offset,
            "total": total,
            "shown_bytes": shown_bytes,
            "total_bytes": total_bytes,
            "limit": page.limit,
            "max_bytes": max_bytes,
            "complete": complete,
            "next_offset": if complete { None } else { Some(next_offset) },
            "query_hint": query_hint,
        });
        if !complete && shown_bytes == 2 {
            result["verification"]["checks_page"]["next_check_bytes"] = json!(next_check_bytes);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Record {
        id: String,
        name: String,
        args: Value,
        text: String,
    }
    impl Record {
        fn new(id: &str, name: &str, args: Value, text: &str) -> Self {
            Self {
                id: id.into(),
                name: name.into(),
                args,
                text: text.into(),
            }
        }
        fn mutation(id: &str, path: &str) -> Self {
            Self::new(id, "write_file", json!({"file_path": path}), "written")
        }
    }
    struct ObservationFixture {
        root: PathBuf,
        revision: u64,
        checks: Vec<CheckObservation>,
        known_check_failure: bool,
    }
    impl ObservationFixture {
        fn new() -> Self {
            let root =
                std::env::temp_dir().join(format!("mutation-observation-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            Self {
                root,
                revision: 0,
                checks: Vec::new(),
                known_check_failure: false,
            }
        }
        fn path(&self) -> PathBuf {
            self.root.join("mutation_log.jsonl")
        }
        fn mutate(&self, id: &str, path: &str) {
            let entry = with_tool_call(id, || {
                make_entry(
                    Path::new(path),
                    "write",
                    None,
                    Some("changed"),
                    BeforeState::Absent,
                )
            });
            append_entry(&self.path(), &entry);
        }
        fn observe(&self, id: &str, name: &str, args: Value, text: &str) {
            let pending = begin_check_at(&self.path(), id, name, &args);
            finish_check(pending, name, &args, text);
        }
    }
    impl Drop for ObservationFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn cmd_alias_check_observation_preserves_original_command() {
        let fixture = ObservationFixture::new();
        fixture.observe(
            "alias-check",
            "execute_command",
            json!({"cmd": "  cargo check --bin a  "}),
            "Finished dev profile",
        );
        let checks = read_checks_at(&fixture.path());
        assert_eq!(checks.len(), 2);
        for check in checks {
            assert_eq!(check.command, "  cargo check --bin a  ");
        }
    }

    fn external_append(path: &Path, bytes: &[u8]) {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
    }

    fn mutation_line(seq: u64, path: &str) -> Vec<u8> {
        let mut entry = make_entry(
            Path::new(path),
            "write",
            None,
            Some("new"),
            BeforeState::Absent,
        );
        entry.seq = seq;
        let mut bytes = serde_json::to_vec(&entry).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn assert_reminder_summary_matches_log(path: &Path) {
        let snapshot = read_committed_snapshot_at(path).unwrap_or_default();
        let revision = snapshot.revision();
        assert_eq!(read_entries(path).len() as u64, revision);
        assert_eq!(read_checks_at(path), snapshot.checks);
        let full = verification_summary(revision, &snapshot.checks, false);
        assert_eq!(
            verification_reminder_summary_at(path),
            VerificationReminderSummary {
                revision,
                stale_checks: full["verification_gaps"]["stale_checks"].as_u64().unwrap() as usize,
                gaps: full["verification_gaps"]["count"].as_u64().unwrap() as usize,
            }
        );
    }

    fn verification_page_fixture(count: usize, command_bytes: usize) -> Vec<CheckObservation> {
        let fixture = ObservationFixture::new();
        fixture.observe(
            "check",
            "execute_command",
            json!({"command": "cargo check"}),
            "Finished dev target(s)",
        );
        let base = read_checks_at(&fixture.path()).last().unwrap().clone();
        (0..count)
            .map(|index| {
                let mut check = base.clone();
                check.command = format!(
                    "cargo check --package project-{index} {}",
                    "x".repeat(command_bytes)
                );
                check.originating_command_call_id = format!("check-{index}");
                check.tool_call_id = format!("poll-{index}");
                check.evidence_locator = format!("tool_call_id:poll-{index}");
                check.covers_revision = u64::from(index != 0);
                if index == 1 {
                    check.status = CheckStatus::Failure;
                }
                if index == 2 {
                    check.status = CheckStatus::Running;
                }
                check
            })
            .collect()
    }

    fn rendered_verification_array_bytes(result: &Value) -> usize {
        let array = serde_json::to_string_pretty(&result["verification"]["checks"]).unwrap();
        array.len() + 4 * array.bytes().filter(|byte| *byte == b'\n').count()
    }

    #[test]
    fn show_changes_verification_pages_normalize_minimum_byte_budget() {
        let checks = verification_page_fixture(1, 0);
        for records in [&checks[..], &[][..]] {
            for offset in [0, records.len()] {
                for requested_max_bytes in [1, 2] {
                    let result = verification_summary_page(
                        1,
                        records,
                        Some(&VerificationDetailsPage {
                            offset,
                            max_bytes: requested_max_bytes,
                            ..Default::default()
                        }),
                    );
                    let page = &result["verification"]["checks_page"];
                    let shown_bytes = page["shown_bytes"].as_u64().unwrap();
                    let max_bytes = page["max_bytes"].as_u64().unwrap();
                    assert_eq!(max_bytes, 2);
                    assert!(shown_bytes <= max_bytes);
                    assert_eq!(
                        shown_bytes as usize,
                        rendered_verification_array_bytes(&result)
                    );
                    assert_eq!(shown_bytes, 2);
                    assert_eq!(result["verification"]["checks"], json!([]));
                    assert_eq!(page["shown"], 0);
                    assert_eq!(page["total"], records.len());
                    assert_eq!(page["complete"], offset == records.len());
                    if offset < records.len() {
                        assert_eq!(page["next_offset"], offset);
                        assert!(page["next_check_bytes"].as_u64().unwrap() > max_bytes);
                    } else {
                        assert!(page["next_offset"].is_null());
                    }
                }
            }
        }
    }

    #[test]
    fn show_changes_verification_pages_preserve_complete_observations_and_status() {
        let checks = verification_page_fixture(12, 0);
        let full = verification_summary_page(
            1,
            &checks,
            Some(&VerificationDetailsPage {
                offset: 0,
                limit: 0,
                max_bytes: 0,
            }),
        );
        let first = verification_summary(1, &checks, true);
        let second = verification_summary_page(
            1,
            &checks,
            Some(&VerificationDetailsPage {
                offset: 8,
                ..Default::default()
            }),
        );
        assert_eq!(first["verification"]["checks_page"]["shown"], 8);
        assert_eq!(first["verification"]["checks_page"]["total"], 12);
        assert_eq!(first["verification"]["checks_page"]["next_offset"], 8);
        assert_eq!(first["verification"]["checks_page"]["complete"], false);
        assert_eq!(second["verification"]["checks_page"]["shown"], 4);
        assert_eq!(second["verification"]["checks_page"]["complete"], true);
        assert!(second["verification"]["checks_page"]["next_offset"].is_null());
        let reconstructed = first["verification"]["checks"]
            .as_array()
            .unwrap()
            .iter()
            .chain(second["verification"]["checks"].as_array().unwrap())
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(json!(reconstructed), full["verification"]["checks"]);
        for page in [&first, &second, &full] {
            assert_eq!(page["revision"], 1);
            assert_eq!(page["verification"]["status"], "failure");
            assert_eq!(page["verification_gaps"], full["verification_gaps"]);
            assert_eq!(
                page["verification"]["current_statuses"],
                full["verification"]["current_statuses"]
            );
            assert_eq!(
                page["verification"]["checks_page"]["shown_bytes"],
                rendered_verification_array_bytes(page)
            );
        }
        let past_end = verification_summary_page(
            1,
            &checks,
            Some(&VerificationDetailsPage {
                offset: usize::MAX,
                ..Default::default()
            }),
        );
        assert_eq!(past_end["verification"]["checks_page"]["offset"], 12);
        assert_eq!(past_end["verification"]["checks_page"]["complete"], true);
        assert_eq!(past_end["verification"]["checks"], json!([]));
    }

    #[test]
    fn show_changes_verification_pages_bound_long_commands_and_recover_oversized_checks() {
        let checks = verification_page_fixture(16, 3000);
        let first = verification_summary(1, &checks, true);
        let full = verification_summary_page(
            1,
            &checks,
            Some(&VerificationDetailsPage {
                offset: 0,
                limit: 0,
                max_bytes: 0,
            }),
        );
        let shown_bytes = rendered_verification_array_bytes(&first);
        let full_bytes = rendered_verification_array_bytes(&full);
        assert!(shown_bytes <= 8192);
        assert!(shown_bytes * 100 <= full_bytes * 16);
        let mut offset = 0;
        let mut reconstructed = Vec::new();
        loop {
            let page = verification_summary_page(
                1,
                &checks,
                Some(&VerificationDetailsPage {
                    offset,
                    ..Default::default()
                }),
            );
            assert!(rendered_verification_array_bytes(&page) <= 8192);
            assert_eq!(page["verification_gaps"], full["verification_gaps"]);
            reconstructed.extend(
                page["verification"]["checks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .cloned(),
            );
            match page["verification"]["checks_page"]["next_offset"].as_u64() {
                Some(next) => {
                    assert!(next as usize > offset);
                    offset = next as usize;
                }
                None => break,
            }
        }
        assert_eq!(json!(reconstructed), full["verification"]["checks"]);
        println!(
            "verification detail bytes: default={shown_bytes}, full={full_bytes}, saved={}",
            full_bytes - shown_bytes
        );

        let oversized = verification_page_fixture(1, 9000);
        let omitted = verification_summary(1, &oversized, true);
        assert_eq!(omitted["verification"]["checks"], json!([]));
        assert_eq!(omitted["verification"]["checks_page"]["next_offset"], 0);
        assert_eq!(omitted["verification"]["checks_page"]["complete"], false);
        assert!(
            omitted["verification"]["checks_page"]["next_check_bytes"]
                .as_u64()
                .unwrap()
                > 8192
        );
        assert!(
            omitted["verification"]["checks_page"]["query_hint"]
                .as_str()
                .unwrap()
                .contains("verification_max_bytes=0")
        );
        let recovered = verification_summary_page(
            1,
            &oversized,
            Some(&VerificationDetailsPage {
                max_bytes: 0,
                ..Default::default()
            }),
        );
        assert_eq!(
            recovered["verification"]["checks"][0]["observation"],
            json!(oversized[0])
        );
        assert_eq!(recovered["verification"]["checks_page"]["complete"], true);
    }

    #[test]
    fn mutation_observation_incremental_reminder_reads_only_new_log_records() {
        let fixture = ObservationFixture::new();
        let path = fixture.path();
        for seq in 1..=64 {
            external_append(&path, &mutation_line(seq, "/project/history.rs"));
        }
        let missing = VerificationReminderSummary {
            revision: 64,
            stale_checks: 0,
            gaps: 1,
        };
        for _ in 0..8 {
            assert_eq!(verification_reminder_summary_at(&path), missing);
            assert_eq!(recovered_records_at(&path), 64);
        }
        let source = ObservationFixture::new();
        source.observe(
            "check",
            "execute_command",
            json!({"command":"cargo check"}),
            "Finished dev target(s)",
        );
        let mut check = read_checks_at(&source.path()).last().unwrap().clone();
        check.covers_revision = 64;
        let mut line = serde_json::to_vec(&check).unwrap();
        line.push(b'\n');
        external_append(&path, &line);
        for _ in 0..8 {
            assert_eq!(verification_reminder_summary_at(&path).gaps, 0);
            assert_eq!(recovered_records_at(&path), 65);
        }
        external_append(&path, &mutation_line(65, "/project/new.rs"));
        assert_eq!(
            verification_reminder_summary_at(&path),
            VerificationReminderSummary {
                revision: 65,
                stale_checks: 1,
                gaps: 1,
            }
        );
        check.covers_revision = 65;
        check.originating_command_call_id = "recheck".into();
        line = serde_json::to_vec(&check).unwrap();
        line.push(b'\n');
        external_append(&path, &line);
        assert_eq!(verification_reminder_summary_at(&path).gaps, 0);
        assert_eq!(recovered_records_at(&path), 67);
        let evidence = std::fs::read(&path).unwrap();
        for _ in 0..8 {
            assert_reminder_summary_matches_log(&path);
            assert_eq!(recovered_records_at(&path), 67);
        }
        assert_eq!(std::fs::read(&path).unwrap(), evidence);
    }

    #[test]
    fn mutation_observation_reminder_recovers_complete_tails_and_replaced_logs() {
        let fixture = ObservationFixture::new();
        let path = fixture.path();
        assert_eq!(verification_reminder_summary_at(&path), Default::default());
        assert!(!path.exists());
        fixture.mutate("first", "/project/a.rs");
        fixture.observe(
            "old",
            "execute_command",
            json!({"command":"cargo check"}),
            "Finished dev target(s)",
        );
        assert_reminder_summary_matches_log(&path);
        let line = mutation_line(2, "/project/incomplete.rs");
        let before = recovered_records_at(&path);
        external_append(&path, &line[..line.len() - 1]);
        let evidence = std::fs::read(&path).unwrap();
        for _ in 0..3 {
            assert_eq!(verification_reminder_summary_at(&path).revision, 1);
            assert_reminder_summary_matches_log(&path);
            assert_eq!(recovered_records_at(&path), before);
        }
        assert_eq!(std::fs::read(&path).unwrap(), evidence);
        external_append(&path, b"\n");
        for _ in 0..3 {
            assert_eq!(verification_reminder_summary_at(&path).revision, 2);
            assert_eq!(verification_reminder_summary_at(&path).stale_checks, 1);
            assert_reminder_summary_matches_log(&path);
            assert_eq!(recovered_records_at(&path), before + 1);
        }
        LOG_STATES
            .lock()
            .unwrap()
            .remove(&std::fs::canonicalize(&path).unwrap());
        assert_reminder_summary_matches_log(&path);
        assert_eq!(recovered_records_at(&path), 4);

        let replacement = fixture.root.join("replacement.jsonl");
        std::fs::write(&replacement, mutation_line(1, "/project/replaced.rs")).unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        assert_reminder_summary_matches_log(&path);
        assert_eq!(verification_reminder_summary_at(&path).stale_checks, 0);
        std::fs::write(&path, b"").unwrap();
        assert_eq!(verification_reminder_summary_at(&path), Default::default());
        external_append(&path, &line);
        assert_reminder_summary_matches_log(&path);
        let regrown = (1..=4)
            .flat_map(|seq| mutation_line(seq, "/project/regrown.rs"))
            .collect::<Vec<_>>();
        std::fs::write(&path, regrown).unwrap();
        assert_reminder_summary_matches_log(&path);
        assert_eq!(verification_reminder_summary_at(&path).revision, 4);
    }

    #[test]
    fn mutation_observation_terminal_check_commits_only_after_newline() {
        let fixture = ObservationFixture::new();
        let path = fixture.path();
        fixture.mutate("first", "/project/a.rs");
        fixture.observe(
            "check",
            "execute_command",
            json!({"command":"cargo check"}),
            "[session external]\n[command session running]\n",
        );
        let mut terminal = read_checks_at(&path).last().unwrap().clone();
        assert_eq!(terminal.status, CheckStatus::Running);
        terminal.status = CheckStatus::Success;
        terminal.observed_success = true;
        let before = recovered_records_at(&path);
        external_append(&path, &serde_json::to_vec(&terminal).unwrap());
        let evidence = std::fs::read(&path).unwrap();
        for _ in 0..3 {
            assert_reminder_summary_matches_log(&path);
            let snapshot = read_committed_snapshot_at(&path).unwrap();
            assert_eq!(snapshot.revision(), 1);
            assert_eq!(snapshot.checks.last().unwrap().status, CheckStatus::Running);
            assert_eq!(verification_reminder_summary_at(&path).gaps, 1);
            assert_eq!(recovered_records_at(&path), before);
        }
        assert_eq!(std::fs::read(&path).unwrap(), evidence);
        external_append(&path, b"\n");
        for _ in 0..3 {
            assert_reminder_summary_matches_log(&path);
            let snapshot = read_committed_snapshot_at(&path).unwrap();
            assert_eq!(snapshot.revision(), 1);
            assert_eq!(snapshot.checks.last(), Some(&terminal));
            assert_eq!(verification_reminder_summary_at(&path).gaps, 0);
            assert_eq!(recovered_records_at(&path), before + 1);
        }
    }

    #[test]
    fn mutation_observation_snapshot_waits_for_writer_commit() {
        use std::sync::mpsc;
        use std::time::Duration;

        let fixture = ObservationFixture::new();
        let path = fixture.path();
        fixture.mutate("first", "/project/a.rs");
        fixture.observe(
            "check",
            "execute_command",
            json!({"command":"cargo check"}),
            "Finished dev target(s)",
        );
        let mut terminal = read_checks_at(&path).last().unwrap().clone();
        terminal.covers_revision = 2;
        let mut writer = OpenOptions::new().append(true).open(&path).unwrap();
        writer.lock().unwrap();
        writer
            .write_all(&mutation_line(2, "/project/b.rs"))
            .unwrap();
        let reader_path = path.clone();
        let (started_sender, started_receiver) = mpsc::channel();
        let (sender, receiver) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            started_sender.send(()).unwrap();
            sender
                .send(read_committed_snapshot_at(&reader_path))
                .unwrap();
        });
        started_receiver
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        let while_locked = receiver.recv_timeout(Duration::from_millis(100));
        let mut line = serde_json::to_vec(&terminal).unwrap();
        line.push(b'\n');
        writer.write_all(&line).unwrap();
        writer.unlock().unwrap();
        reader.join().unwrap();
        assert!(matches!(while_locked, Err(mpsc::RecvTimeoutError::Timeout)));
        let snapshot = receiver
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.revision(), 2);
        assert_eq!(snapshot.mutations.last().unwrap().seq, 2);
        assert_eq!(snapshot.checks.last(), Some(&terminal));
        assert_eq!(
            verification_summary(snapshot.revision(), &snapshot.checks, false)["verification"]["status"],
            "success"
        );
        assert_reminder_summary_matches_log(&path);
    }

    #[test]
    fn mutation_observation_incremental_append_reads_only_new_log_records() {
        let fixture = ObservationFixture::new();
        let path = fixture.path();
        // Seed a cold log without warming its in-process state.
        for seq in 1..=64 {
            let mut entry = make_entry(
                Path::new("/project/history.rs"),
                "write",
                None,
                Some(&"x".repeat(MAX_CONTENT_BYTES)),
                BeforeState::Absent,
            );
            entry.seq = seq;
            let mut line = serde_json::to_vec(&entry).unwrap();
            line.push(b'\n');
            external_append(&path, &line);
        }
        fixture.mutate("first", "/project/a.rs");
        let shared = state_at(&path);
        assert_eq!(shared.lock().unwrap().recovered_records, 65);
        for index in 0..8 {
            fixture.mutate(&format!("write-{index}"), "/project/a.rs");
        }
        fixture.observe(
            "check",
            "execute_command",
            json!({"command":"cargo check"}),
            "[session external]\n[command session running]\n",
        );
        assert_eq!(shared.lock().unwrap().recovered_records, 75);
        assert_eq!(revision_at(&path), 73);
        // An external completion updates both indexes without scanning old snapshots.
        let mut check = read_checks_at(&path).last().unwrap().clone();
        check.reported_failure = true;
        check.status = CheckStatus::Failure;
        let mut line = serde_json::to_vec(&check).unwrap();
        line.extend_from_slice(b"\nnot valid json\n");
        external_append(&path, &line);
        let pending = begin_check_at(
            &path,
            "poll",
            "command_session",
            &json!({"action":"poll","session_id":"external"}),
        )
        .unwrap();
        assert!(pending.observation.reported_failure);
        assert_eq!(shared.lock().unwrap().recovered_records, 77);
        fixture.mutate("after-external", "/project/b.rs");
        assert_eq!(read_entries(&path).last().unwrap().seq, 74);
        assert_eq!(shared.lock().unwrap().recovered_records, 78);
        // Only complete records count; a split external append is recovered once.
        let line = mutation_line(75, "/project/split.rs");
        let split = line.len() / 2;
        external_append(&path, &line[..split]);
        assert_eq!(revision_at(&path), 74);
        assert_eq!(shared.lock().unwrap().recovered_records, 78);
        external_append(&path, &line[split..]);
        fixture.mutate("after-split", "/project/c.rs");
        assert_eq!(read_entries(&path).last().unwrap().seq, 76);
        assert_eq!(shared.lock().unwrap().recovered_records, 80);
        assert_eq!(revision_at(&path), 76);
        assert_eq!(shared.lock().unwrap().recovered_records, 80);
    }

    #[test]
    fn mutation_observation_append_on_other_path_does_not_wait_for_busy_log() {
        use std::sync::mpsc;
        use std::time::Duration;

        let busy = ObservationFixture::new();
        let other = ObservationFixture::new();
        busy.mutate("busy", "/project/a.rs");
        let shared = state_at(&busy.path());
        assert!(!Arc::ptr_eq(&shared, &state_at(&other.path())));
        let (entered_sender, entered_receiver) = mpsc::channel();
        let (resume_sender, resume_receiver) = mpsc::channel();
        shared.lock().unwrap().recovery_pause = Some((entered_sender, resume_receiver));
        let busy_path = busy.path();
        let busy_worker = std::thread::spawn(move || {
            let entry = make_entry(
                Path::new("/project/busy.rs"),
                "write",
                None,
                Some("busy"),
                BeforeState::Absent,
            );
            append_entry(&busy_path, &entry);
        });
        // Pause an actual append inside recovery, holding its path and file locks.
        let entered = entered_receiver.recv_timeout(Duration::from_secs(10));
        let other_path = other.path();
        let (sender, receiver) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let entry = make_entry(
                Path::new("/project/b.rs"),
                "write",
                None,
                Some("b"),
                BeforeState::Absent,
            );
            append_entry(&other_path, &entry);
            sender.send(read_entries(&other_path).len()).unwrap();
        });
        let result = receiver.recv_timeout(Duration::from_secs(10));
        // Release before joining even on failure, so a lock regression cannot hang.
        let _ = resume_sender.send(());
        busy_worker.join().unwrap();
        worker.join().unwrap();
        entered.unwrap();
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn mutation_observation_concurrent_mixed_records_keep_sequence_after_restart() {
        let fixture = Arc::new(ObservationFixture::new());
        let path = fixture.path();
        external_append(&path, &mutation_line(1, "/project/preexisting.rs"));
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let workers = (0..4)
            .map(|worker| {
                let fixture = fixture.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    for index in 0..8 {
                        let id = format!("{worker}-{index}");
                        fixture.mutate(&id, "/project/concurrent.rs");
                        let args = json!({"command":"cargo check"});
                        let pending =
                            begin_check_at(&fixture.path(), &id, "execute_command", &args);
                        std::thread::yield_now();
                        finish_check(pending, "execute_command", &args, "Finished dev target(s)");
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
        let mut revision = 0;
        let mut starts = FastMap::default();
        for line in std::fs::read_to_string(&path).unwrap().lines() {
            if let Some(entry) = parse_mutation(line).unwrap() {
                revision += 1;
                assert_eq!(entry.seq, revision);
            } else {
                let check: CheckObservation = serde_json::from_str(line).unwrap();
                if check.status == CheckStatus::Running {
                    assert_eq!(check.covers_revision, revision);
                    starts.insert(check.originating_command_call_id, revision);
                } else {
                    assert_eq!(check.status, CheckStatus::Success);
                    assert_eq!(
                        starts[&check.originating_command_call_id],
                        check.covers_revision
                    );
                }
            }
        }
        assert_eq!(revision, 33);
        assert_eq!(starts.len(), 32);
        // Drop the cache to model a process restart on the same durable path.
        LOG_STATES
            .lock()
            .unwrap()
            .remove(&std::fs::canonicalize(&path).unwrap());
        fixture.mutate("restarted", "/project/restarted.rs");
        assert_eq!(read_entries(&path).last().unwrap().seq, 34);
        assert_eq!(revision_at(&path), 34);
        assert_reminder_summary_matches_log(&path);
    }

    #[test]
    fn mutation_observation_rebuilds_after_truncation_replacement_and_regrowth() {
        let fixture = ObservationFixture::new();
        let path = fixture.path();
        fixture.mutate("first", "/project/a.rs");
        fixture.mutate("second", "/project/b.rs");
        fixture.observe(
            "old",
            "execute_command",
            json!({"command":"cargo check"}),
            "[session removed]\n[command session running]\n",
        );
        std::fs::write(&path, mutation_line(1, "/project/truncated.rs")).unwrap();
        fixture.mutate("after-truncate", "/project/c.rs");
        assert_eq!(read_entries(&path).last().unwrap().seq, 2);
        assert!(
            begin_check_at(
                &path,
                "missing",
                "command_session",
                &json!({"action":"poll","session_id":"removed"})
            )
            .is_none()
        );

        let replacement = fixture.root.join("replacement.jsonl");
        std::fs::write(&replacement, mutation_line(1, "/project/replaced.rs")).unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        fixture.mutate("after-replace", "/project/d.rs");
        assert_eq!(read_entries(&path).last().unwrap().seq, 2);

        let regrown = (1..=4)
            .flat_map(|seq| mutation_line(seq, "/project/regrown.rs"))
            .collect::<Vec<_>>();
        assert!(regrown.len() as u64 > std::fs::metadata(&path).unwrap().len());
        std::fs::write(&path, &regrown).unwrap();
        fixture.mutate("after-regrow", "/project/e.rs");
        assert_eq!(read_entries(&path).last().unwrap().seq, 5);
        assert_eq!(revision_at(&path), 5);
    }

    #[test]
    fn mutation_observation_failed_append_and_interrupted_tail_do_not_lose_records() {
        let fixture = ObservationFixture::new();
        let path = fixture.path();
        fixture.mutate("first", "/project/a.rs");
        let shared = state_at(&path);
        let entry = make_entry(
            Path::new("/project/failed.rs"),
            "write",
            None,
            Some("failed"),
            BeforeState::Absent,
        );
        let mut read_only = File::open(&path).unwrap();
        assert!(
            shared
                .lock()
                .unwrap()
                .append(&mut read_only, &entry)
                .is_err()
        );
        fixture.mutate("second", "/project/b.rs");
        assert_eq!(read_entries(&path).last().unwrap().seq, 2);

        external_append(&path, b"{\"interrupted\":");
        fixture.mutate("after-interruption", "/project/c.rs");
        assert_eq!(read_entries(&path).last().unwrap().seq, 3);
        // An otherwise valid record lacking its newline becomes complete before seq assignment.
        let line = mutation_line(4, "/project/unterminated.rs");
        external_append(&path, &line[..line.len() - 1]);
        assert_eq!(revision_at(&path), 3);
        fixture.mutate("after-unterminated", "/project/d.rs");
        assert_eq!(read_entries(&path).last().unwrap().seq, 5);
        assert_eq!(revision_at(&path), 5);
    }

    fn collect(prior: Option<ObservationFixture>, records: &[Record]) -> ObservationFixture {
        let mut fixture = prior.unwrap_or_else(ObservationFixture::new);
        for record in records {
            if record.name == "write_file" {
                fixture.mutate(&record.id, record.args["file_path"].as_str().unwrap());
            } else {
                let before = read_checks_at(&fixture.path()).len();
                fixture.observe(&record.id, &record.name, record.args.clone(), &record.text);
                let checks = read_checks_at(&fixture.path());
                if checks.len() > before {
                    fixture.checks.push(checks.last().unwrap().clone());
                }
            }
        }
        fixture.revision = revision_at(&fixture.path());
        fixture.known_check_failure = latest_checks(&read_checks_at(&fixture.path()))
            .iter()
            .any(|check| check.reported_failure || check.status == CheckStatus::Failure);
        assert_reminder_summary_matches_log(&fixture.path());
        fixture
    }

    #[test]
    fn mutation_observation_sleep_wrapped_running_session_preserves_poll_origin() {
        let output = format!(
            "{SLEEP_WARNING}\n[session sleep-check]\n[command session running]\nstdout:\nFinished dev target(s)\n"
        );
        let checkpoint = collect(
            None,
            &[
                Record::mutation("write", "/project/a.rs"),
                Record::new(
                    "start",
                    "execute_command",
                    json!({"command": "sleep 30 && cargo check --bin a"}),
                    &output,
                ),
            ],
        );
        let check = &checkpoint.checks[0];
        assert_eq!(check.status, CheckStatus::Running);
        assert_eq!(check.command_session_id.as_deref(), Some("sleep-check"));
        assert_eq!(
            crate::ai::driver::turn_runtime::progress::classify_tool_result_progress(&output),
            crate::ai::driver::turn_runtime::progress::ToolResultProgressStatus::Running
        );
        let checkpoint = collect(
            Some(checkpoint),
            &[Record::new(
                "poll",
                "command_session",
                json!({"action": "poll", "session_id": "sleep-check"}),
                "[session sleep-check]\n[command session exited: exit code 0]\n",
            )],
        );
        let terminal = checkpoint.checks.last().unwrap();
        assert_eq!(terminal.status, CheckStatus::Success);
        assert_eq!(terminal.originating_command_call_id.as_str(), "start");
        assert_eq!(terminal.command.as_str(), "sleep 30 && cargo check --bin a");
        assert_eq!(terminal.covers_revision, 1);
    }

    const SLEEP_WARNING: &str = "Note: this command runs `sleep`, which blocks the whole turn on \
dead time — it is expensive. Use it only when truly necessary (e.g. waiting for a server to \
accept connections); prefer starting the work with `yield_time_ms` and polling it with \
`command_session`, or `task_wait` for subagent tasks.";

    #[test]
    fn command_result_wrapped_terminal_statuses_are_authoritative() {
        use crate::ai::driver::turn_runtime::progress::{
            ToolResultProgressStatus, classify_tool_result_progress,
        };
        for (body, expected, progress, id) in [
            (
                "[session terminal]\n[command session exited: exit code 0]\n",
                CheckStatus::Success,
                ToolResultProgressStatus::Success,
                Some("terminal"),
            ),
            (
                "Exit code: 1\n[session terminal]\n[command session exited: exit code 1]\nFinished dev target(s)\n",
                CheckStatus::Failure,
                ToolResultProgressStatus::Failure,
                Some("terminal"),
            ),
            (
                "[command session exited: exit code 0]\n",
                CheckStatus::Success,
                ToolResultProgressStatus::Success,
                None,
            ),
            (
                "Exit code: 1\n[command session exited: exit code 1]\n",
                CheckStatus::Failure,
                ToolResultProgressStatus::Failure,
                None,
            ),
            (
                "[session terminal]\n",
                CheckStatus::Unknown,
                ToolResultProgressStatus::Unknown,
                Some("terminal"),
            ),
            (
                "Exit code: 1\nFinished dev target(s)\n",
                CheckStatus::Failure,
                ToolResultProgressStatus::Failure,
                None,
            ),
        ] {
            let output = format!("{SLEEP_WARNING}\n\nNote: another tool note\n{body}");
            let checkpoint = collect(
                None,
                &[
                    Record::mutation("write", "/project/a.rs"),
                    Record::new(
                        "check",
                        "execute_command",
                        json!({"command": "sleep 1 && cargo check --bin a"}),
                        &output,
                    ),
                ],
            );
            assert_eq!(checkpoint.checks[0].status, expected, "{body}");
            assert_eq!(
                checkpoint.checks[0].command_session_id.as_deref(),
                id,
                "{body}"
            );
            assert_eq!(classify_tool_result_progress(&output), progress, "{body}");
        }
    }

    #[test]
    fn mutation_observation_compound_pytest_failure_cannot_be_overridden_by_cargo_success() {
        for join in [";", "||", "|", "&&"] {
            for failure in [
                "FAILED tests/test_example.py::test_assertion - AssertionError",
                "1 failed in 0.01s",
                "================ FAILURES ================",
            ] {
                let command = format!("pytest -q {join} cargo check --bin a");
                let output = format!("{failure}\nFinished dev target(s)\n");
                let checkpoint = collect(
                    None,
                    &[
                        Record::mutation("write", "/project/a.rs"),
                        Record::new(
                            "check",
                            "execute_command",
                            json!({"command": command}),
                            &output,
                        ),
                    ],
                );
                assert_eq!(
                    checkpoint.checks[0].status,
                    CheckStatus::Failure,
                    "{join}: {failure}"
                );
                assert!(checkpoint.checks[0].reported_failure);
                assert!(checkpoint.checks[0].partial_success);
                assert!(checkpoint.known_check_failure);
            }
        }
        for join in [";", "||", "|"] {
            let checkpoint = collect(
                None,
                &[
                    Record::mutation("write", "/project/a.rs"),
                    Record::new(
                        "check",
                        "execute_command",
                        json!({"command": format!("pytest -q {join} cargo check --bin a")}),
                        "unrecognized check output\nFinished dev target(s)\n",
                    ),
                ],
            );
            assert_eq!(checkpoint.checks[0].status, CheckStatus::Unknown, "{join}");
            assert!(!checkpoint.known_check_failure);
        }
    }

    #[test]
    fn mutation_observation_compound_async_poll_preserves_failures_and_unknown_scope() {
        for (early, terminal, expected) in [
            (
                "FAILED tests/test_example.py::test_assertion",
                "Finished dev target(s)",
                CheckStatus::Failure,
            ),
            (
                "",
                "1 failed in 0.01s\nFinished dev target(s)",
                CheckStatus::Failure,
            ),
            (
                "unrecognized check output",
                "Finished dev target(s)",
                CheckStatus::Unknown,
            ),
            ("Finished dev target(s)", "", CheckStatus::Unknown),
        ] {
            let checkpoint = collect(
                None,
                &[
                    Record::mutation("write", "/project/a.rs"),
                    Record::new(
                        "start",
                        "execute_command",
                        json!({"command": "pytest -q; cargo check --bin a"}),
                        &format!(
                            "{SLEEP_WARNING}\n[session compound]\n[command session running]\nstdout:\n{early}\n"
                        ),
                    ),
                ],
            );
            assert_eq!(checkpoint.checks[0].status, CheckStatus::Running);
            let checkpoint = collect(
                Some(checkpoint),
                &[Record::new(
                    "poll",
                    "command_session",
                    json!({"action": "poll", "session_id": "compound"}),
                    &format!(
                        "[session compound]\n[command session exited: exit code 0]\nstdout:\n{terminal}\n"
                    ),
                )],
            );
            let check = checkpoint.checks.last().unwrap();
            assert_eq!(check.status, expected, "{early:?}, {terminal:?}");
            assert_eq!(check.originating_command_call_id.as_str(), "start");
            assert_eq!(check.command.as_str(), "pytest -q; cargo check --bin a");
            assert_eq!(check.covers_revision, 1);
        }
        let checkpoint = collect(
            None,
            &[
                Record::mutation("write", "/project/a.rs"),
                Record::new(
                    "start",
                    "execute_command",
                    json!({"command": "cargo check"}),
                    "[session failed]\n[command session running]\n",
                ),
                Record::new(
                    "failed",
                    "command_session",
                    json!({"action": "poll", "session_id": "failed"}),
                    "Exit code: 1\n[session failed]\n[command session exited: exit code 1]\n",
                ),
                Record::new(
                    "later",
                    "command_session",
                    json!({"action": "poll", "session_id": "failed"}),
                    "[session failed]\n[command session exited: exit code 0]\nFinished dev target(s)\n",
                ),
            ],
        );
        assert_eq!(
            checkpoint.checks.last().unwrap().status,
            CheckStatus::Failure
        );
    }

    #[test]
    fn mutation_observation_roundtrip_and_checks_excluded_from_mutations() {
        let fixture = ObservationFixture::new();
        crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
            ("source-label".into(), 17),
            || {
                crate::ai::driver::runtime_ctx::SUBAGENT_TASK_ID.sync_scope(
                    "task-origin".into(),
                    || {
                        fixture.mutate("write-id", "/project/a.rs");
                        fixture.observe(
                            "check-id",
                            "execute_command",
                            json!({"command":"cargo check","cwd":"/project"}),
                            "Finished dev target(s)",
                        );
                    },
                );
            },
        );
        let entries = read_entries(&fixture.path());
        assert_eq!(entries.len(), 1);
        assert!(entries[0].entry_id.is_some());
        assert_eq!(entries[0].turn_id, Some(17));
        assert_eq!(entries[0].task_id.as_deref(), Some("task-origin"));
        assert_eq!(entries[0].tool_call_id.as_deref(), Some("write-id"));
        let checks = read_checks_at(&fixture.path());
        assert_eq!(checks.len(), 2);
        let check = checks.last().unwrap();
        assert_eq!(check.covers_revision, 1);
        assert_eq!(check.turn_id, Some(17));
        assert_eq!(check.task_id.as_deref(), Some("task-origin"));
        assert_eq!(check.session_id.as_deref(), Some("source-label"));
        assert_eq!(check.status, CheckStatus::Success);
        let value = serde_json::to_value(check).unwrap();
        for field in ["path", "before", "after", "files", "inventory"] {
            assert!(value.get(field).is_none());
        }
        assert_eq!(
            serde_json::from_value::<CheckObservation>(value).unwrap(),
            *check
        );
        assert!(
            parse_mutation(&serde_json::to_string(check).unwrap())
                .unwrap()
                .is_none()
        );
        assert_eq!(
            verification_summary(1, &checks, false)["verification"]["current"],
            1
        );
        assert!(
            verification_summary(1, &checks, false)["verification"]
                .get("checks")
                .is_none()
        );
        assert_eq!(
            verification_summary(1, &checks, true)["verification"]["checks"][0]["observation"]["evidence_locator"],
            "tool_call_id:check-id"
        );
    }

    #[test]
    fn mutation_observation_later_mutation_stales_without_history_rewrite() {
        let fixture = ObservationFixture::new();
        fixture.mutate("first", "/project/a.rs");
        fixture.observe(
            "check",
            "execute_command",
            json!({"command":"cargo check"}),
            "Finished dev target(s)",
        );
        let history = std::fs::read(&fixture.path()).unwrap();
        let checks = read_checks_at(&fixture.path());
        fixture.mutate("second", "/project/b.rs");
        assert!(
            std::fs::read(&fixture.path())
                .unwrap()
                .starts_with(&history)
        );
        assert_eq!(read_checks_at(&fixture.path()), checks);
        assert_eq!(
            verification_summary(revision_at(&fixture.path()), &checks, false)["verification_gaps"]
                ["stale_checks"],
            1
        );
        fixture.observe(
            "unrelated",
            "execute_command",
            json!({"command":"git diff -- /project/b.rs"}),
            "diff",
        );
        assert_eq!(
            verification_summary(2, &read_checks_at(&fixture.path()), false)["verification_gaps"]["stale_checks"],
            1
        );
        fixture.observe(
            "recheck",
            "execute_command",
            json!({"command":"cargo check"}),
            "Finished dev target(s)",
        );
        assert_eq!(
            verification_summary(2, &read_checks_at(&fixture.path()), false)["verification_gaps"]["count"],
            0
        );
        assert_eq!(
            &read_checks_at(&fixture.path())[..checks.len()],
            checks.as_slice()
        );
    }

    #[test]
    fn mutation_observation_restart_persistence_and_start_revision() {
        let fixture = ObservationFixture::new();
        fixture.mutate("first", "/project/a.rs");
        let args = json!({"command":"cargo check"});
        let pending = begin_check_at(&fixture.path(), "origin", "execute_command", &args);
        fixture.mutate("concurrent", "/project/b.rs");
        finish_check(
            pending,
            "execute_command",
            &args,
            "Note: wrapped\n[session command]\n[command session running]\nerror: could not compile\n",
        );
        let disk = std::fs::read(&fixture.path()).unwrap();
        let reopened = fixture.root.join("reopened.jsonl");
        std::fs::write(&reopened, &disk).unwrap();
        let poll_args = json!({"action":"poll","session_id":"command"});
        let pending = begin_check_at(&reopened, "poll", "command_session", &poll_args);
        finish_check(
            pending,
            "command_session",
            &poll_args,
            "[session command]\n[command session exited: exit code 0]\n",
        );
        let checks = read_checks_at(&reopened);
        let poll = checks.last().unwrap();
        assert_eq!(poll.originating_command_call_id, "origin");
        assert_eq!(poll.tool_call_id, "poll");
        assert_eq!(poll.covers_revision, 1);
        assert_eq!(poll.status, CheckStatus::Failure);
        assert!(poll.reported_failure);
        let entry = make_entry(
            Path::new("/project/c.rs"),
            "write",
            None,
            Some("c"),
            BeforeState::Absent,
        );
        append_entry(&reopened, &entry);
        assert_eq!(read_entries(&reopened).last().unwrap().seq, 3);
        assert_eq!(revision_at(&reopened), 3);
        assert!(std::fs::read(&reopened).unwrap().starts_with(&disk));
    }

    #[test]
    fn mutation_observation_delayed_old_poll_does_not_override_new_recheck() {
        let fixture = ObservationFixture::new();
        fixture.mutate("first", "/project/a.rs");
        fixture.observe(
            "old",
            "execute_command",
            json!({"command":"cargo check"}),
            "[session old]\n[command session running]\n",
        );
        fixture.mutate("second", "/project/b.rs");
        fixture.observe(
            "new",
            "execute_command",
            json!({"command":"cargo check"}),
            "Finished dev target(s)",
        );
        fixture.observe(
            "old-poll",
            "command_session",
            json!({"action":"poll","session_id":"old"}),
            "[session old]\n[command session exited: exit code 1]\n",
        );
        let checks = read_checks_at(&fixture.path());
        assert_eq!(latest_checks(&checks).len(), 1);
        assert_eq!(latest_checks(&checks)[0].originating_command_call_id, "new");
        assert_eq!(
            verification_summary(2, &checks, false)["verification_gaps"]["count"],
            0
        );
        assert_reminder_summary_matches_log(&fixture.path());
    }

    #[test]
    fn mutation_observation_real_write_and_patch_bind_tool_call_ids() {
        use crate::ai::driver::runtime_ctx;
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let fixture = ObservationFixture::new();
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = fixture.root.join(".git/runtime/history.sqlite");
        app.session_id = "mutation-tool-scope".into();
        let context = runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        runtime_ctx::SUBAGENT_CWD.sync_scope(fixture.root.clone(), || runtime_ctx::DRIVER_CTX.sync_scope(context, || {
            let write = crate::ai::tools::registry::common::execute_tool_call_with_args("actual-write", "write_file", &json!({"file_path":"a.rs","content":"before\n"})).unwrap();
            assert!(write.content.contains("Successfully wrote"), "{}", write.content);
            let patch = crate::ai::tools::registry::common::execute_tool_call_with_args("actual-patch", "apply_patch", &json!({"patch":"*** Begin Patch\n*** Update File: a.rs\n@@\n-before\n+after\n*** Add File: b.rs\n+new\n*** End Patch\n"})).unwrap();
            assert!(patch.content.starts_with("Successfully patched"), "{}", patch.content);
            let entries = read_all();
            assert_eq!(entries.len(), 3);
            assert_eq!(entries[0].tool_call_id.as_deref(), Some("actual-write"));
            assert!(entries[1..].iter().all(|entry| entry.tool_call_id.as_deref() == Some("actual-patch")));
            assert_eq!(entries.iter().map(|entry| entry.seq).collect::<Vec<_>>(), [1, 2, 3]);
            assert_eq!(std::fs::read_to_string(fixture.root.join("a.rs")).unwrap(), "after\n");
            assert_eq!(TOOL_CALL_ID.with(|slot| slot.borrow().clone()), None);
        }));
    }

    #[test]
    fn mutation_observation_wrong_session_poll_is_unknown_and_cannot_erase_origin_failure() {
        let fixture = ObservationFixture::new();
        fixture.mutate("write", "/project/a.rs");
        fixture.observe(
            "start",
            "execute_command",
            json!({"command":"cargo check"}),
            "[session expected]\n[command session running]\nerror: could not compile\n",
        );
        fixture.observe(
            "wrong-poll",
            "command_session",
            json!({"action":"poll","session_id":"expected"}),
            "[session different]\n[command session exited: exit code 0]\nFinished dev target(s)\n",
        );
        let checks = read_checks_at(&fixture.path());
        assert_eq!(checks.last().unwrap().status, CheckStatus::Unknown);
        assert!(checks.last().unwrap().reported_failure);
        assert!(!checks.last().unwrap().observed_success);
        fixture.observe(
            "poll",
            "command_session",
            json!({"action":"poll","session_id":"expected"}),
            "[session expected]\n[command session exited: exit code 0]\n",
        );
        assert_eq!(
            read_checks_at(&fixture.path()).last().unwrap().status,
            CheckStatus::Failure
        );
    }
    #[test]
    fn record_is_safe_noop_without_driver_context() {
        record(
            std::path::Path::new("/tmp/nonexistent_audit_test.rs"),
            "write",
            None,
            Some("x"),
        );
        assert!(read_all().is_empty());
        assert!(log_path().is_none());
    }

    #[test]
    fn should_skip_paths_under_assets_dir() {
        let assets = std::path::Path::new("/home/u/.history_file.sessions/abc.assets");
        assert!(should_skip(&assets.join("tmp/scratch.rs"), assets));
        assert!(should_skip(&assets.join("mutation_log.jsonl"), assets));
        assert!(!should_skip(
            std::path::Path::new("/proj/src/main.rs"),
            assets
        ));
    }

    #[test]
    fn should_skip_subagent_scratch_sibling_dirs() {
        let assets = std::path::Path::new("/home/u/.history_file.sessions/abc.assets");
        let root = std::path::Path::new("/home/u/.history_file.sessions");
        assert!(should_skip(&root.join("subagent-cwd-t1/foo.rs"), assets));
        assert!(should_skip(
            &root.join("checkpoints/abc/gen-1/x.sqlite"),
            assets
        ));
        assert!(should_skip(&root.join("def.assets/tmp/y.rs"), assets));
        assert!(!should_skip(
            std::path::Path::new("/proj/src/main.rs"),
            assets
        ));
    }

    #[test]
    fn cap_content_truncates_oversized_payload_on_char_boundary() {
        let small = "hello";
        assert_eq!(cap_content(small), small);

        let big = "€".repeat(MAX_CONTENT_BYTES);
        let capped = cap_content(&big);
        assert!(capped.len() < big.len());
        assert!(capped.contains("truncated"));
        // The truncation marker must be recognized by is_capped (rendering uses
        // it to decide whether a snapshot can be diffed directly).
        assert!(is_capped(&capped));
        assert!(!is_capped(small));
    }

    #[test]
    fn entry_diff_modify_emits_only_changed_lines() {
        let before = "a\nb\nc\nd\ne\n";
        let after = "a\nb\nX\nd\ne\n";
        assert_eq!(
            entry_diff(Some(before), Some(after)).as_deref(),
            Some("- c\n+ X\n")
        );
    }

    #[test]
    fn entry_diff_created_and_deleted_emit_single_sided_lines() {
        assert_eq!(
            entry_diff(None, Some("x\ny\n")).as_deref(),
            Some("+ x\n+ y\n")
        );
        assert_eq!(
            entry_diff(Some("x\ny\n"), None).as_deref(),
            Some("- x\n- y\n")
        );
    }

    #[test]
    fn entry_diff_none_for_unchanged_or_empty() {
        assert!(entry_diff(Some("same"), Some("same")).is_none());
        assert!(entry_diff(None, None).is_none());
        assert!(entry_diff(None, Some("")).is_none());
        assert!(entry_diff(Some(""), None).is_none());
    }

    #[test]
    fn entry_diff_caps_huge_change_at_line_boundary() {
        // Full-file rewrite: when the diff exceeds MAX_DIFF_BYTES it must be cut
        // at a line boundary with a marker, and must never panic.
        let before = format!("{}\n", "b".repeat(MAX_DIFF_BYTES + 100));
        let after = format!("{}\n", "a".repeat(MAX_DIFF_BYTES + 100));
        let d = entry_diff(Some(&before), Some(&after)).unwrap();
        assert!(d.contains("truncated"));
        assert!(d.ends_with('\n'));
    }

    #[test]
    fn entry_diff_caps_huge_change_with_multibyte_chars() {
        // Multi-byte characters straddling the 64KiB cap must not panic and the
        // cut must stay on a line boundary.
        let before = format!("{}\n", "€€€".repeat(MAX_DIFF_BYTES));
        let after = format!("{}\n", "¥¥¥".repeat(MAX_DIFF_BYTES));
        let d = entry_diff(Some(&before), Some(&after)).unwrap();
        assert!(d.contains("truncated"));
        assert!(d.ends_with('\n'));
    }

    #[test]
    fn append_and_read_entries_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            ".agent_mutation_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let log_path = dir.join("mutation_log.jsonl");
        append_entry(
            &log_path,
            &MutationEntry {
                entry_id: None,
                turn_id: None,
                task_id: None,
                tool_call_id: None,
                seq: 1,
                ts: "t1".into(),
                path: "/proj/a.rs".into(),
                op: "write".into(),
                before: Some("old".into()),
                before_state: None,
                after: Some("new".into()),
                diff: Some("- old\n+ new\n".into()),
            },
        );
        append_entry(
            &log_path,
            &MutationEntry {
                entry_id: None,
                turn_id: None,
                task_id: None,
                tool_call_id: None,
                seq: 2,
                ts: "t2".into(),
                path: "/proj/b.rs".into(),
                op: "delete".into(),
                before: Some("gone".into()),
                before_state: None,
                after: None,
                diff: Some("- gone\n".into()),
            },
        );
        let entries = read_entries(&log_path);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].path, "/proj/a.rs");
        assert_eq!(entries[0].before.as_deref(), Some("old"));
        assert_eq!(entries[0].after.as_deref(), Some("new"));
        assert_eq!(entries[0].diff.as_deref(), Some("- old\n+ new\n"));
        assert_eq!(entries[1].op, "delete");
        assert_eq!(entries[1].after, None);
        assert_eq!(entries[1].diff.as_deref(), Some("- gone\n"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_entries_returns_empty_for_missing_file() {
        let entries = read_entries(std::path::Path::new(
            "/nonexistent/.agent/mutation_log.jsonl",
        ));
        assert!(entries.is_empty());
    }

    #[test]
    fn read_entries_skips_malformed_lines() {
        let dir = std::env::temp_dir().join(format!(
            ".agent_mutation_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let log_path = dir.join("mutation_log.jsonl");
        append_entry(
            &log_path,
            &MutationEntry {
                entry_id: None,
                turn_id: None,
                task_id: None,
                tool_call_id: None,
                seq: 1,
                ts: "t1".into(),
                path: "/proj/a.rs".into(),
                op: "write".into(),
                before: None,
                before_state: None,
                after: Some("x".into()),
                diff: None,
            },
        );
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&log_path) {
            let _ = writeln!(f, "not valid json");
        }
        let entries = read_entries(&log_path);
        assert_eq!(entries.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_entries_defaults_missing_diff_field_to_none() {
        // Old-format logs have no diff field: deserialization must yield None
        // (backward compatible) rather than failing to parse.
        let dir = std::env::temp_dir().join(format!(
            ".agent_mutation_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Bare OpenOptions does not create parent dirs (append_entry does), so
        // create it explicitly here.
        let _ = std::fs::create_dir_all(&dir);
        let log_path = dir.join("mutation_log.jsonl");
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = writeln!(
                f,
                r#"{{"seq":1,"ts":"t","path":"/proj/a.rs","op":"write","before":"old","after":"new"}}"#
            );
        }
        let entries = read_entries(&log_path);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].before.as_deref(), Some("old"));
        assert_eq!(entries[0].diff, None);
        assert_eq!(entries[0].before_state, None);
        assert_eq!(entries[0].effective_before_state(), BeforeState::Present);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_missing_before_state_retains_create_delete_semantics() {
        for (before, after, op, expected) in [
            (None, Some("new"), "write", BeforeState::Absent),
            (Some("old"), None, "delete", BeforeState::Present),
        ] {
            let entry: MutationEntry = serde_json::from_value(serde_json::json!({
                "seq": 1, "ts": "t", "path": "/proj/a.rs", "op": op,
                "before": before, "after": after,
            }))
            .unwrap();
            assert_eq!(entry.before_state, None);
            assert_eq!(entry.effective_before_state(), expected);
            assert_eq!(entry.diff, None);
        }
    }

    #[test]
    fn unavailable_preimage_records_state_without_fabricating_additions() {
        for state in [BeforeState::Present, BeforeState::Unknown] {
            for after in [Some("replacement\n"), Some(""), None] {
                let entry = make_entry(Path::new("/proj/a.rs"), "write", None, after, state);
                let json = serde_json::to_string(&entry).unwrap();
                let decoded: MutationEntry = serde_json::from_str(&json).unwrap();
                assert_eq!(decoded.before_state, Some(state));
                assert_eq!(decoded.effective_before_state(), state);
                assert!(decoded.before.is_none());
                let diff = decoded.diff.unwrap();
                assert!(diff.contains("before snapshot unavailable"));
                assert!(diff.contains("not compared"));
                assert!(
                    !diff
                        .lines()
                        .any(|line| line.starts_with("+ ") || line.starts_with("- "))
                );
            }
        }
        let created = make_entry(
            Path::new("/proj/new.rs"),
            "write",
            None,
            Some("new\n"),
            BeforeState::Absent,
        );
        assert_eq!(created.diff.as_deref(), Some("+ new\n"));
    }

    #[test]
    fn identical_large_inputs_do_not_claim_an_unobserved_tail_difference() {
        let input = "same line\n".repeat(MAX_DIFF_INPUT_BYTES / 5);
        assert!(input.len() > MAX_DIFF_INPUT_BYTES);
        let diff = entry_diff(Some(&input), Some(&input)).unwrap();
        assert!(diff.contains("remaining content not compared"));
        assert!(!diff.contains("differ beyond"));
        assert!(
            !diff
                .lines()
                .any(|line| line.starts_with("+ ") || line.starts_with("- "))
        );
        assert!(diff.len() < 512);
    }

    #[test]
    fn entry_diff_bounds_huge_content_instead_of_building_full_diff() {
        // An output cap alone would still allocate a diff for the entire write.
        let mut big = String::new();
        for i in 0..200_000 {
            big.push_str(&format!("line {i}\n"));
        }
        assert!(big.len() > MAX_DIFF_INPUT_BYTES);
        let d = entry_diff(None, Some(&big)).expect("diff must exist");
        // Leading additions only; the diff must stay far below the file size.
        assert!(d.starts_with("+ line 0\n"), "diff: {d}");
        assert!(
            d.len() < MAX_DIFF_INPUT_BYTES,
            "diff must be bounded, got {} bytes",
            d.len()
        );
        // The reader must see a truncation notice (input cap or output cap
        // marker), never silently believe the diff is complete.
        assert!(d.contains("truncated"), "diff: {d}");
    }

    #[test]
    fn entry_diff_capped_window_never_reports_unchanged() {
        // Regression: before/after share the first 1 MiB but differ beyond it.
        // Returning None would read as "unchanged"; the truncation must surface.
        let mut a = String::new();
        let mut b = String::new();
        for i in 0..300_000 {
            let line = format!("same line {i}\n");
            a.push_str(&line);
            b.push_str(&line);
        }
        a.push_str("DIFFERENT TAIL\n");
        let d = entry_diff(Some(&b), Some(&a)).expect("must not be None for capped input");
        assert!(
            d.contains("leading portion only"),
            "must surface truncation, got: {d}"
        );
    }

    #[test]
    fn entry_diff_single_line_huge_rewrite_is_bounded() {
        // A large single line exercises truncation without a complete line to retain.
        let line = "x".repeat(3 * 1024 * 1024);
        let d = entry_diff(None, Some(&line)).unwrap();
        assert!(d.len() < 200 * 1024, "bounded: {} bytes", d.len());
        // cap_diff walks back to the nearest newline; a single-line diff reduces
        // to the truncation notice, which is truthful and still bounded.
        assert!(d.contains("truncated"), "diff: {d}");
    }

    #[test]
    fn entry_diff_cap_input_handles_multibyte_char_at_window_boundary() {
        // A 3-byte UTF-8 char straddling the 1 MiB cap must not panic and must
        // produce a valid bounded diff (cap_diff_input walks back to a char
        // boundary before slicing).
        let mut s = String::new();
        s.push_str(&"a".repeat(MAX_DIFF_INPUT_BYTES - 1));
        s.push('中');
        s.push_str(&"b".repeat(100));
        assert!(s.len() > MAX_DIFF_INPUT_BYTES);
        let d = entry_diff(None, Some(&s)).unwrap();
        assert!(d.len() < MAX_DIFF_INPUT_BYTES, "bounded: {} bytes", d.len());
        assert!(d.contains("truncated"), "diff: {d}");
    }

    #[test]
    fn entry_diff_small_content_behavior_is_unchanged() {
        assert_eq!(entry_diff(Some("a\nb\n"), Some("a\nb\n")), None);
        assert_eq!(entry_diff(None, Some("")), None);
        assert_eq!(
            entry_diff(Some("a\nb\nc\n"), Some("a\nb\nx\n")),
            Some("- c\n+ x\n".into())
        );
    }
}
