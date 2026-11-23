//! Append-only, history-file-scoped decision evidence.

use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::ai::driver::{runtime_ctx, side_note::assets_dir_for_history};

const SCHEMA_VERSION: u32 = 1;
const MAX_RECEIPT_BYTES: u64 = 64 * 1024;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "paths", rename_all = "snake_case")]
pub(in crate::ai) enum Sources {
    File(PathBuf),
    // A registered resolver may include absent entries: their appearance would change the action.
    ToolTargets(Vec<PathBuf>),
}

impl Sources {
    pub(in crate::ai::tools) fn paths(&self) -> &[PathBuf] {
        match self {
            Self::File(path) => std::slice::from_ref(path),
            Self::ToolTargets(paths) => paths,
        }
    }

    pub(in crate::ai) fn fingerprint(&self) -> io::Result<String> {
        Ok(self.snapshot()?.fingerprint)
    }

    pub(in crate::ai::tools) fn tool_targets_missing(&self) -> io::Result<bool> {
        let Self::ToolTargets(paths) = self else {
            return Ok(false);
        };
        for path in paths {
            if metadata_if_present(path)?.is_some() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(in crate::ai::tools) fn snapshot(&self) -> io::Result<SourceSnapshot> {
        let mut digest = Sha256::new();
        let mut present = false;
        let mut targets = Vec::new();
        for path in self.paths() {
            put(&mut digest, path.as_os_str().as_encoded_bytes());
            match self {
                Self::File(path) => {
                    let real = fs::canonicalize(path)?;
                    validate_file_source(&real)?;
                    let links = fingerprint_path_links(path, &mut digest)?;
                    let meta = fs::symlink_metadata(&real)?;
                    if !meta.is_file() {
                        return Err(io::Error::other("the source must be a regular file"));
                    }
                    let observed = hash_entry(&mut digest, &real, &meta)?;
                    for link in links {
                        link.checked_metadata()?;
                    }
                    observed.checked_metadata()?;
                    if fs::canonicalize(path)? != real {
                        return Err(io::Error::other(
                            "the source path changed during verification",
                        ));
                    }
                    present = true;
                }
                Self::ToolTargets(_) => {
                    let (parent, leaf) = path.parent().zip(path.file_name()).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "invalid tool source target")
                    })?;
                    // Bind aliases and the directory's own version: moving it
                    // away and back can leave every target inode unchanged.
                    let real_parent = fs::canonicalize(parent)?;
                    let mut links = fingerprint_path_links(parent, &mut digest)?;
                    put(&mut digest, b"source-parent");
                    let parent_entry = fingerprint_path_version(&mut digest, &real_parent)?;
                    let root = real_parent.join(leaf);
                    let mut referent = None;
                    let mut pending = vec![root.clone()];
                    let mut entries = Vec::new();
                    while let Some(entry) = pending.pop() {
                        let Some(meta) = metadata_if_present(&entry)? else {
                            if entry != root {
                                return Err(io::Error::other(
                                    "a source entry disappeared during verification",
                                ));
                            }
                            put(&mut digest, b"missing");
                            entries.push(SourceEntry {
                                path: entry,
                                version: None,
                                content: None,
                            });
                            continue;
                        };
                        present = true;
                        if entry == root && meta.is_symlink() {
                            // The resolver relies on this chain and its containment.
                            // These are verification dependencies, not deletion entries.
                            let resolved = fs::canonicalize(&entry)?;
                            if resolved == real_parent || !resolved.starts_with(&real_parent) {
                                return Err(io::Error::other(
                                    "tool source link resolves outside its parent directory",
                                ));
                            }
                            links.extend(fingerprint_path_links(&entry, &mut digest)?);
                            put(&mut digest, b"source-referent");
                            referent = Some(fingerprint_path_version(&mut digest, &resolved)?);
                        }
                        entries.push(hash_entry(&mut digest, &entry, &meta)?);
                        if meta.is_dir() {
                            let mut children = fs::read_dir(&entry)?
                                .map(|child| child.map(|child| child.path()))
                                .collect::<io::Result<Vec<_>>>()?;
                            children.sort_unstable();
                            pending.extend(children.into_iter().rev());
                        }
                    }
                    targets.push(SourceTarget {
                        parent: parent.to_path_buf(),
                        parent_entry,
                        links,
                        referent,
                        entries,
                    });
                }
            }
        }
        if !present {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no source target exists",
            ));
        }
        for target in &targets {
            target.check_tree()?;
        }
        Ok(SourceSnapshot {
            fingerprint: hex_digest(digest),
            targets,
        })
    }
}

/// Observe each symlink in the input path and in every expanded link target.
/// `canonicalize` alone loses this chain and cannot detect a retarget-and-restore.
fn fingerprint_path_links(path: &Path, digest: &mut Sha256) -> io::Result<Vec<SourceEntry>> {
    use std::path::Component;

    let components = |path: &Path| {
        path.components()
            .rev()
            .map(|part| part.as_os_str().to_owned())
            .collect::<Vec<_>>()
    };
    let mut pending = components(path);
    let mut resolved = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir()?
    };
    let mut links = Vec::new();
    while let Some(part) = pending.pop() {
        let component = Path::new(&part)
            .components()
            .next()
            .expect("path component");
        match component {
            Component::Prefix(_) | Component::RootDir => resolved.push(&part),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                let candidate = resolved.join(name);
                let meta = fs::symlink_metadata(&candidate)?;
                if meta.is_symlink() {
                    // Bound cycles even if a link changes after the kernel's first lookup.
                    if links.len() == 64 {
                        return Err(io::Error::other("too many symlinks in the source path"));
                    }
                    let target = fs::read_link(&candidate)?;
                    put(digest, b"source-link");
                    links.push(hash_path_version(digest, &candidate, &meta)?);
                    put(digest, target.as_os_str().as_encoded_bytes());
                    pending.extend(components(&target));
                } else {
                    if !pending.is_empty() && !meta.is_dir() {
                        return Err(io::Error::new(
                            io::ErrorKind::NotADirectory,
                            "invalid source path component",
                        ));
                    }
                    resolved = candidate;
                }
            }
        }
    }
    Ok(links)
}

pub(in crate::ai::tools) struct SourceSnapshot {
    fingerprint: String,
    pub(in crate::ai::tools) targets: Vec<SourceTarget>,
}

/// Dependencies on resolution plus the complete, pre-order tree (or an absent root).
pub(in crate::ai::tools) struct SourceTarget {
    parent: PathBuf,
    parent_entry: SourceEntry,
    links: Vec<SourceEntry>,
    referent: Option<SourceEntry>,
    entries: Vec<SourceEntry>,
}

impl SourceTarget {
    pub(in crate::ai::tools) fn check_tree(&self) -> io::Result<()> {
        self.parent_entry.checked_metadata()?;
        for entry in self.links.iter().chain(&self.entries) {
            entry.checked_metadata()?;
        }
        if fs::canonicalize(&self.parent)? != self.parent_entry.path {
            return Err(io::Error::other(
                "the source parent changed during verification",
            ));
        }
        if let Some(referent) = &self.referent {
            referent.checked_metadata()?;
            if fs::canonicalize(self.entries[0].path())? != referent.path {
                return Err(io::Error::other(
                    "the source link referent changed during verification",
                ));
            }
        }
        Ok(())
    }

    pub(in crate::ai::tools) fn entries(&self) -> &[SourceEntry] {
        &self.entries
    }
}

pub(in crate::ai::tools) struct SourceEntry {
    path: PathBuf,
    version: Option<Version>,
    content: Option<[u8; 32]>,
}

impl SourceEntry {
    pub(in crate::ai::tools) fn path(&self) -> &Path {
        &self.path
    }

    fn checked_metadata(&self) -> io::Result<Option<Metadata>> {
        check_entry(&self.path, self.version.as_ref())
    }

    pub(in crate::ai::tools) fn is_removal_candidate(&self) -> io::Result<bool> {
        if self.version.is_none() {
            self.checked_metadata()?;
            return Ok(false);
        }
        self.check_removal_identity(&self.path)?;
        Ok(true)
    }

    pub(in crate::ai::tools) fn checked_staged_metadata(
        &self,
        staged: &Path,
    ) -> io::Result<Metadata> {
        let (meta, staged_version) = self.check_removal_identity(staged)?;
        if self.content != fingerprint_content(staged, &meta, &staged_version)? {
            return Err(io::Error::other("the staged entry's content changed"));
        }
        check_entry(staged, Some(&staged_version))?;
        Ok(meta)
    }

    fn check_removal_identity(&self, path: &Path) -> io::Result<(Metadata, Version)> {
        let expected = self
            .version
            .ok_or_else(|| io::Error::other("no source entry was verified for removal"))?;
        let meta = fs::symlink_metadata(path)?;
        let current = version(&meta)?;
        // Renaming changes ctime, as can removing children or hard links. The
        // staged leaf's content is checked separately against the snapshot.
        if current.kind != expected.kind
            || !expected.same_object(&current)
            || (expected.kind != "directory"
                && (expected.len, expected.modified) != (current.len, current.modified))
        {
            return Err(io::Error::other(
                "a verified source entry was replaced or changed",
            ));
        }
        Ok((meta, current))
    }
}

fn check_entry(path: &Path, expected: Option<&Version>) -> io::Result<Option<Metadata>> {
    let metadata = metadata_if_present(path)?;
    let current = metadata.as_ref().map(version).transpose()?;
    if current.as_ref() != expected {
        return Err(io::Error::other(
            "a source entry changed during verification",
        ));
    }
    Ok(metadata)
}

fn validate_file_source(real: &Path) -> io::Result<()> {
    let Some(context) = runtime_ctx::try_current() else {
        return Ok(());
    };
    let app = &context.app_proto;
    let store = crate::ai::history::SessionStore::new(&app.config.history_file);
    let protected_files = [
        &app.session_history_file,
        &app.config.history_file,
        &app.config.base_history_file,
    ];
    let in_session_store = fs::canonicalize(store.sessions_root())
        .is_ok_and(|root| real.starts_with(root))
        || protected_files
            .iter()
            .any(|path| fs::canonicalize(path).is_ok_and(|path| path == real));
    if in_session_store {
        // Scratch files are valid evidence; session history and archived evidence are not.
        let scratch = store.session_assets_dir(&app.session_id).join("tmp");
        if !fs::canonicalize(scratch).is_ok_and(|root| real.starts_with(root)) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "use a source data file, not session history or archives",
            ));
        }
    }
    Ok(())
}

fn metadata_if_present(path: &Path) -> io::Result<Option<Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(meta) => Ok(Some(meta)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct Version {
    kind: &'static str,
    len: u64,
    modified: i128,
    // On Unix, backdating mtime or restoring the old bytes still changes ctime.
    #[cfg(unix)]
    identity_and_change_time: (u64, u64, i64, i64),
    #[cfg(not(unix))]
    created: Option<std::time::SystemTime>,
}

impl Version {
    fn same_object(&self, other: &Self) -> bool {
        #[cfg(unix)]
        {
            let (dev, ino, ..) = self.identity_and_change_time;
            let (other_dev, other_ino, ..) = other.identity_and_change_time;
            (dev, ino) == (other_dev, other_ino)
        }
        #[cfg(not(unix))]
        {
            self.created == other.created
        }
    }
}

fn version(meta: &Metadata) -> io::Result<Version> {
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    use std::time::UNIX_EPOCH;

    let kind = if meta.is_file() {
        "file"
    } else if meta.is_dir() {
        "directory"
    } else if meta.is_symlink() {
        "symlink"
    } else {
        "special"
    };
    let modified = meta.modified()?.duration_since(UNIX_EPOCH);
    Ok(Version {
        kind,
        len: meta.len(),
        modified: modified.map_or_else(
            |error| -(error.duration().as_nanos() as i128),
            |time| time.as_nanos() as i128,
        ),
        #[cfg(unix)]
        identity_and_change_time: (meta.dev(), meta.ino(), meta.ctime(), meta.ctime_nsec()),
        #[cfg(not(unix))]
        created: meta.created().ok(),
    })
}

fn put(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}

fn hex_digest(digest: Sha256) -> String {
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn fingerprint_path_version(digest: &mut Sha256, path: &Path) -> io::Result<SourceEntry> {
    hash_path_version(digest, path, &fs::symlink_metadata(path)?)
}

fn hash_path_version(digest: &mut Sha256, path: &Path, meta: &Metadata) -> io::Result<SourceEntry> {
    put(digest, path.as_os_str().as_encoded_bytes());
    let expected = version(meta)?;
    put(digest, &serde_json::to_vec(&expected)?);
    Ok(SourceEntry {
        path: path.to_path_buf(),
        version: Some(expected),
        content: None,
    })
}

fn hash_entry(digest: &mut Sha256, path: &Path, meta: &Metadata) -> io::Result<SourceEntry> {
    let mut entry = hash_path_version(digest, path, meta)?;
    entry.content = fingerprint_content(path, meta, &entry.version.expect("observed version"))?;
    if let Some(content) = &entry.content {
        put(digest, content);
    }
    Ok(entry)
}

fn fingerprint_content(
    path: &Path,
    meta: &Metadata,
    expected: &Version,
) -> io::Result<Option<[u8; 32]>> {
    if meta.is_file() {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // A regular file replaced by a FIFO must not block the pre-dispatch check.
            options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
        }
        let mut file = options.open(path)?;
        if version(&file.metadata()?)? != *expected {
            return Err(io::Error::other(
                "a source file was replaced during verification",
            ));
        }
        let mut content = Sha256::new();
        let mut input = (&mut file).take(expected.len);
        let mut buffer = [0_u8; 16 * 1024];
        loop {
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            content.update(&buffer[..read]);
        }
        if input.limit() != 0 || version(&file.metadata()?)? != *expected {
            return Err(io::Error::other(
                "a source file changed during verification",
            ));
        }
        return Ok(Some(content.finalize().into()));
    }
    if meta.is_symlink() {
        // Tool-target symlinks are removed as links. Never traverse their trees.
        let link = fs::read_link(path)?;
        return Ok(Some(
            Sha256::digest(link.as_os_str().as_encoded_bytes()).into(),
        ));
    }
    Ok(None)
}

#[derive(Debug, Serialize, Deserialize)]
pub(in crate::ai) struct Decision {
    pub(in crate::ai) kind: String,
    pub(in crate::ai) target_tool: String,
    pub(in crate::ai) target_args: Value,
    pub(in crate::ai) cwd: PathBuf,
    pub(in crate::ai) source: Sources,
    pub(in crate::ai) source_fingerprint: String,
    pub(in crate::ai) numeric_params: Value,
    pub(in crate::ai) selected_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(in crate::ai) struct Receipt {
    schema_version: u32,
    pub(in crate::ai) id: String,
    session_id: String,
    pub(in crate::ai) created_at: String,
    #[serde(flatten)]
    pub(in crate::ai) decision: Decision,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::ai) enum Freshness {
    Fresh,
    Changed,
    Stale,
}

pub(in crate::ai) struct Ledger {
    dir: PathBuf,
    session_id: String,
}

fn valid_id(id: &str) -> bool {
    id.split_once('-').is_some_and(|(time, uuid)| {
        time.len() == 19
            && time.bytes().all(|c| c.is_ascii_digit())
            && uuid.len() == 32
            && uuid.bytes().all(|c| c.is_ascii_hexdigit())
    })
}

impl Ledger {
    pub(in crate::ai) fn current() -> Option<Self> {
        let context = runtime_ctx::try_current()?;
        let app = &context.app_proto;
        (!app.session_id.is_empty() && !app.session_history_file.as_os_str().is_empty())
            .then(|| Self::new(&app.session_history_file, &app.session_id))
    }

    pub(in crate::ai) fn new(history: &Path, session_id: &str) -> Self {
        Self {
            dir: assets_dir_for_history(history).join("decision-receipts"),
            session_id: session_id.to_owned(),
        }
    }

    fn path(&self, id: &str, extension: &str) -> PathBuf {
        self.dir.join(format!("{id}.{extension}"))
    }

    pub(in crate::ai) fn write(&self, decision: Decision) -> io::Result<Receipt> {
        if decision.source.fingerprint()? != decision.source_fingerprint {
            return Err(io::Error::other(
                "the source changed since snapshot; snapshot and re-read the source before recording",
            ));
        }
        let now = Utc::now();
        let receipt = Receipt {
            schema_version: SCHEMA_VERSION,
            id: format!(
                "{:019}-{}",
                now.timestamp_nanos_opt().unwrap_or_default(),
                Uuid::new_v4().simple()
            ),
            session_id: self.session_id.clone(),
            created_at: now.to_rfc3339(),
            decision,
        };
        let bytes = serde_json::to_vec(&receipt)?;
        if bytes.len() as u64 > MAX_RECEIPT_BYTES {
            return Err(io::Error::other("receipt exceeds 64 KiB"));
        }
        fs::create_dir_all(&self.dir)?;
        let temp = self.path(&receipt.id, "tmp");
        let mut file = File::create_new(&temp)?;
        let result = (|| {
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(&temp, self.path(&receipt.id, "json"))
        })();
        if result.is_err() {
            let _ = fs::remove_file(temp);
        }
        result?;
        Ok(receipt)
    }

    pub(in crate::ai) fn read(&self, id: &str) -> io::Result<Receipt> {
        if !valid_id(id) {
            return Err(io::Error::other("invalid receipt ID"));
        }
        let mut bytes = Vec::new();
        File::open(self.path(id, "json"))?
            .take(MAX_RECEIPT_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_RECEIPT_BYTES {
            return Err(io::Error::other("receipt exceeds 64 KiB"));
        }
        let receipt: Receipt = serde_json::from_slice(&bytes)?;
        if receipt.schema_version != SCHEMA_VERSION
            || receipt.id != id
            || receipt.session_id != self.session_id
        {
            return Err(io::Error::other(
                "receipt schema, ID, or session does not match",
            ));
        }
        Ok(receipt)
    }

    /// Page by on-disk IDs so an unreadable record still advances the cursor.
    pub(in crate::ai::tools) fn recent_ids(
        &self,
        before: Option<&str>,
        limit: usize,
    ) -> io::Result<Vec<String>> {
        if before.is_some_and(|id| !valid_id(id)) {
            return Err(io::Error::other("invalid receipt cursor"));
        }
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut newest = Vec::<String>::new();
        for entry in entries {
            let Ok(name) = entry?.file_name().into_string() else {
                continue;
            };
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            if !valid_id(id) || before.is_some_and(|cursor| id >= cursor) {
                continue;
            }
            let index = newest.partition_point(|other| other.as_str() > id);
            if index < limit {
                newest.insert(index, id.to_owned());
                newest.truncate(limit);
            }
        }
        Ok(newest)
    }

    pub(in crate::ai::tools) fn known_invalidation(&self, receipt: &Receipt) -> Option<Freshness> {
        match metadata_if_present(&self.path(&receipt.id, "verifying")) {
            Ok(None) => self.published_invalidation(receipt),
            _ => Some(Freshness::Stale),
        }
    }

    fn published_invalidation(&self, receipt: &Receipt) -> Option<Freshness> {
        match self.path(&receipt.id, "changed").try_exists() {
            Ok(true) => Some(Freshness::Changed),
            Err(_) => Some(Freshness::Stale),
            Ok(false) => match self.path(&receipt.id, "stale").try_exists() {
                Ok(false) => None,
                _ => Some(Freshness::Stale),
            },
        }
    }

    fn persist_marker(&self, path: &Path) -> io::Result<()> {
        File::create_new(path)?.sync_all()?;
        File::open(&self.dir)?.sync_all()
    }

    pub(in crate::ai::tools) fn verify_source_with(
        &self,
        receipt: &Receipt,
        observe: impl FnOnce() -> io::Result<SourceSnapshot>,
    ) -> Result<SourceSnapshot, Freshness> {
        if let Some(verdict) = self.known_invalidation(receipt) {
            return Err(verdict);
        }

        // Reserve a durable stale marker before observing the source. If we cannot
        // publish an invalidation afterward, the reservation remains and a new
        // opener cannot mistake a restored source for one that was never invalidated.
        let reservation = self.path(&receipt.id, "verifying");
        let verify = || -> io::Result<Result<SourceSnapshot, Freshness>> {
            self.persist_marker(&reservation)?;
            // A different verifier may have published its verdict just before
            // we acquired the reservation. Never replace that verdict.
            let result = if let Some(verdict) = self.published_invalidation(receipt) {
                Err(verdict)
            } else {
                let (verdict, extension) = match observe() {
                    Ok(snapshot) if snapshot.fingerprint == receipt.decision.source_fingerprint => {
                        fs::remove_file(&reservation)?;
                        return Ok(Ok(snapshot));
                    }
                    Err(error) if error.kind() != io::ErrorKind::NotFound => {
                        (Freshness::Stale, "stale")
                    }
                    _ => (Freshness::Changed, "changed"),
                };
                self.persist_marker(&self.path(&receipt.id, extension))?;
                Err(verdict)
            };
            fs::remove_file(&reservation)?;
            Ok(result)
        };
        match verify() {
            Ok(result) => result,
            Err(error) => {
                if runtime_ctx::terminal_output_enabled() {
                    eprintln!(
                        "[Warning] Decision receipt {} cannot record a verification verdict: {error}",
                        receipt.id
                    );
                }
                Err(Freshness::Stale)
            }
        }
    }
}
