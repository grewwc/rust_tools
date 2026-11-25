//! One-shot profile cloning. No source files are opened for writing.

use crate::profile::{self, Keychain};
use serde_json::Value;
use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, Metadata};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

const LOCAL_STATE: &str = "Local State";
const ATTEMPTS: usize = 3;
const RETRY_DELAY: Duration = Duration::from_millis(15);
const SQLITE_SUFFIXES: [&str; 3] = ["-wal", "-shm", "-journal"];

#[derive(Debug, PartialEq, Eq)]
pub(super) struct SyncOptions {
    pub(super) destination: PathBuf,
    pub(super) from: Option<PathBuf>,
    pub(super) profile: Option<String>,
    pub(super) force: bool,
    pub(super) keychain: Keychain,
}

pub(super) fn sync(options: SyncOptions) -> Result<(), String> {
    let (source, direct_profile) = resolve_source(options.from.as_deref())?;
    let dest = profile::resolve_directory(&options.destination)?;
    let dest_text = dest
        .to_str()
        .ok_or("the destination Chrome profile path must be valid Unicode")?;
    if profile::paths_overlap(&source, &dest) {
        return Err(
            "source and destination Chrome directories must not be equal or contain each other"
                .to_string(),
        );
    }
    profile::require_dedicated_directory(&dest)?;
    let parent = dest
        .parent()
        .ok_or("destination Chrome profile requires a parent directory")?;
    let _leases = profile::lease_profile_tree(&dest)?;
    validate_destination(&dest, options.force)?;

    let staging = Staging::new(parent, "stage")
        .map_err(|e| format!("cannot create Chrome profile staging directory: {e}"))?;
    let mut stats = CopyStats::default();
    let state_files = [OsString::from(LOCAL_STATE)];
    let state_copy = copy_files(&source, staging.path(), &state_files)
        .map_err(|e| copy_error(&source.join(LOCAL_STATE), e))?;
    if state_copy[0].is_none() {
        return Err("Chrome source Local State must be a regular file".to_string());
    }
    stats.record(state_copy);
    let state: Value = serde_json::from_reader(
        File::open(staging.path().join(LOCAL_STATE))
            .map_err(|_| "cannot open copied Chrome Local State")?,
    )
    .map_err(|e| {
        format!(
            "invalid Chrome Local State JSON (line {}, column {})",
            e.line(),
            e.column()
        )
    })?;
    if !state.is_object() {
        return Err("Chrome Local State must contain a JSON object".to_string());
    }
    let chosen = choose_profile(
        &source,
        &state,
        direct_profile.as_deref(),
        options.profile.as_deref(),
    )?;
    let profile_dir = source.join(&chosen.directory);
    let dest_profile = staging.path().join(&chosen.directory);
    profile::private_directory(&dest_profile, false)
        .map_err(|e| format!("cannot create selected profile in staging: {e}"))?;
    copy_tree(&profile_dir, &dest_profile, false, &mut stats)?;
    profile::write_marker(
        staging.path(),
        &chosen.directory,
        options.keychain,
        &source.to_string_lossy(),
    )?;

    // Recheck after copying: a Chrome instance not using the agent lease may have started.
    let destination = validate_destination(&dest, options.force)?;
    let portable_swap = staging.install(&dest, destination)?;
    if portable_swap {
        stats.warnings.push("this filesystem does not support atomic directory exchange; replacement used a backup with rollback".to_string());
    }
    if let Some(warning) = keychain_diagnostic(options.keychain, std::env::consts::OS, &state) {
        stats.warnings.push(warning.to_string());
    }

    println!(
        "Source: {:?}\nProfile: {:?} ({:?})\nDestination: {:?}\nCopied: {} files, {} bytes\nKeychain: {}",
        source,
        chosen.name,
        chosen.directory,
        dest,
        stats.files,
        stats.bytes,
        options.keychain.as_str()
    );
    println!(
        "Skipped entries: caches={}, locks={}, private non-auth data={}, links/special files={}",
        stats.skipped[Skip::Cache as usize],
        stats.skipped[Skip::Lock as usize],
        stats.skipped[Skip::Private as usize],
        stats.skipped[Skip::Link as usize]
    );
    println!("Profile files were copied; cookie decryption has not been verified.");
    for warning in &stats.warnings {
        eprintln!("Warning: {warning}");
    }
    println!(
        "Set mcpServers.browser.env in ~/.config/mcp.json (remove MCP_BROWSER_WS_URL):\n{}",
        serde_json::json!({profile::DATA_DIR_ENV: dest_text})
    );
    Ok(())
}

fn keychain_diagnostic(keychain: Keychain, os: &str, state: &Value) -> Option<&'static str> {
    match keychain {
        Keychain::Mock => Some(
            "--keychain mock selects launch flags only; it does not convert existing native cookies",
        ),
        Keychain::Native
            if os == "windows"
                && [
                    "/os_crypt/encrypted_key",
                    "/os_crypt/app_bound_encrypted_key",
                ]
                .iter()
                .all(|key| {
                    state
                        .pointer(key)
                        .and_then(Value::as_str)
                        .is_none_or(str::is_empty)
                }) =>
        {
            Some(
                "Local State lacks a Windows OS Crypt key; existing encrypted cookies may not be reusable",
            )
        }
        Keychain::Native => None,
    }
}

fn resolve_source(from: Option<&Path>) -> Result<(PathBuf, Option<String>), String> {
    let path = match from {
        Some(path) => path.to_path_buf(),
        None => profile::default_browser_roots()
            .into_iter()
            .find(|path| path.join(LOCAL_STATE).is_file())
            .ok_or("no default Chrome, Chromium, or Edge profile was found; specify --from")?,
    };
    let path = path
        .canonicalize()
        .map_err(|_| "cannot resolve Chrome source directory")?;
    if !path.is_dir() {
        return Err("Chrome source path is not a directory".to_string());
    }
    let has_state = |root: &Path| fs::symlink_metadata(root.join(LOCAL_STATE)).is_ok();
    if has_state(&path) {
        return Ok((path, None));
    }
    if let Some(parent) = path.parent().filter(|path| has_state(path)) {
        let directory = path
            .file_name()
            .and_then(OsStr::to_str)
            .ok_or("Chrome source profile name must be valid Unicode")?;
        return Ok((parent.to_path_buf(), Some(directory.to_string())));
    }
    Err("Chrome source must be a user-data root containing Local State or a profile directory directly inside it".to_string())
}

#[derive(Debug)]
struct ChromeProfile {
    directory: String,
    name: String,
}

fn choose_profile(
    root: &Path,
    state: &Value,
    direct: Option<&str>,
    selector: Option<&str>,
) -> Result<ChromeProfile, String> {
    let last = state
        .pointer("/profile/last_used")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty());
    let cache = state
        .pointer("/profile/info_cache")
        .and_then(Value::as_object);
    let mut profiles = Vec::new();
    for entry in fs::read_dir(root).map_err(|_| "cannot list Chrome source profiles")? {
        let entry = entry.map_err(|_| "cannot list Chrome source profiles")?;
        let Ok(directory) = entry.file_name().into_string() else {
            continue;
        };
        if !profile::valid_profile_directory(&directory)
            || !fs::symlink_metadata(entry.path())
                .is_ok_and(|m| m.is_dir() && !profile::is_link(&m))
        {
            continue;
        }
        let info = cache.and_then(|cache| cache.get(&directory));
        if info.is_none()
            && last != Some(&directory)
            && direct != Some(&directory)
            && directory != "Default"
            && !directory.starts_with("Profile ")
            && !entry.path().join("Preferences").is_file()
        {
            continue;
        }
        let name = info
            .and_then(|info| info["name"].as_str())
            .unwrap_or(&directory)
            .to_owned();
        profiles.push(ChromeProfile { directory, name });
    }
    profiles.sort_by(|left, right| left.directory.cmp(&right.directory));
    let needle = selector.or(direct).or(last).unwrap_or("Default");
    let exact = profiles
        .iter()
        .position(|profile| profile.directory == needle);
    let mut display_matches = profiles
        .iter()
        .enumerate()
        .filter(|(_, profile)| profile.name == needle);
    let first_display = display_matches.next().map(|(index, _)| index);
    let ambiguous = exact.is_none() && selector.is_some() && display_matches.next().is_some();
    let selected = exact
        .or_else(|| selector.and(first_display))
        .filter(|_| !ambiguous);
    if let Some(index) =
        selected.filter(|index| direct.is_none_or(|dir| dir == profiles[*index].directory))
    {
        return Ok(profiles.swap_remove(index));
    }
    let reason = if ambiguous {
        "matches multiple profiles; specify a directory name"
    } else if selected.is_some() {
        "does not match the directory supplied in --from"
    } else {
        "was not found"
    };
    let available = profiles
        .iter()
        .map(|p| {
            format!(
                "{:?} ({:?}{})",
                p.name,
                p.directory,
                if last == Some(&p.directory) {
                    ", last_used"
                } else {
                    ""
                }
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "Chrome profile {needle:?} {reason}; available profiles: {}; last_used: {last:?}",
        if available.is_empty() {
            "(none)"
        } else {
            &available
        }
    ))
}

enum Destination {
    Missing,
    Empty,
    Managed,
}

fn validate_destination(dest: &Path, force: bool) -> Result<Destination, String> {
    profile::require_dedicated_directory(dest)?;
    match fs::symlink_metadata(dest) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Destination::Missing),
        Ok(m) if m.is_dir() && !profile::is_link(&m) => {}
        _ => {
            return Err(
                "destination Chrome profile is not an accessible regular directory".to_string(),
            );
        }
    }
    if fs::read_dir(dest)
        .map_err(|_| "cannot list destination Chrome profile")?
        .next()
        .is_none()
    {
        return Ok(Destination::Empty);
    }
    if profile::read_marker(dest)?.is_none() {
        return Err("destination is nonempty and does not have an MCP profile marker; choose an empty or managed destination".to_string());
    }
    if !force {
        return Err(
            "destination Chrome profile is not empty; use --force to replace it".to_string(),
        );
    }
    if !fs::symlink_metadata(dest).is_ok_and(|m| profile::is_private_directory(&m)) {
        return Err(
            "the managed destination must be private and owned by the current user".to_string(),
        );
    }
    if profile::contains_persistent_marker(dest, false)
        .map_err(|_| "cannot check the destination for nested MCP profiles")?
    {
        return Err(
            "destination contains another managed Chrome profile; it cannot be replaced"
                .to_string(),
        );
    }
    profile::require_idle(dest, true)?;
    Ok(Destination::Managed)
}

#[derive(Clone, Copy)]
enum Skip {
    Cache,
    Lock,
    Private,
    Link,
}

fn skip(name: &OsStr, in_service_worker: bool) -> Option<Skip> {
    let name = name.to_string_lossy().to_ascii_lowercase();
    let prefix = |prefixes: &[&str]| prefixes.iter().any(|p| name.starts_with(p));
    if prefix(&[
        "cache",
        "code cache",
        "gpucache",
        "shadercache",
        "grshadercache",
        "graphitedawncache",
        "crashpad",
        "browsermetrics",
        "safe browsing",
        "component_crx_cache",
        "extensions_crx_cache",
        "optimization_guide_model_store",
        "gpupersistentcache",
    ]) || (name.starts_with("dawn") && name.contains("cache"))
        || (in_service_worker && name == "cachestorage")
    {
        Some(Skip::Cache)
    } else if name.starts_with("singleton")
        || ["lockfile", "devtoolsactiveport", "lock"].contains(&name.as_str())
    {
        Some(Skip::Lock)
    } else if prefix(&[
        "login data",
        "web data",
        "history",
        "bookmarks",
        "top sites",
        "favicons",
        "shortcuts",
        "thumbnails",
    ]) || [
        "visited links",
        "sessions",
        "current session",
        "last session",
        "current tabs",
        "last tabs",
    ]
    .contains(&name.as_str())
        || name == profile::MARKER
    {
        Some(Skip::Private)
    } else {
        None
    }
}

#[derive(Default)]
struct CopyStats {
    files: u64,
    bytes: u64,
    skipped: [u64; 4],
    warnings: Vec<String>,
}

impl CopyStats {
    fn record(&mut self, copies: Vec<Option<u64>>) {
        for bytes in copies.into_iter().flatten() {
            self.files += 1;
            self.bytes += bytes;
        }
    }
}

fn directory_names(path: &Path) -> io::Result<Vec<OsString>> {
    if path.canonicalize()? != path {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "source directory changed identity",
        ));
    }
    let mut names = fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<io::Result<Vec<_>>>()?;
    names.sort_unstable();
    Ok(names)
}

fn copy_tree(
    source: &Path,
    dest: &Path,
    in_service_worker: bool,
    stats: &mut CopyStats,
) -> Result<(), String> {
    let mut names = directory_names(source).map_err(|e| copy_error(source, e))?;
    let leveldb = source
        .file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| name.ends_with("leveldb"))
        || names.iter().any(|name| name == "CURRENT")
            && names
                .iter()
                .any(|name| name.to_string_lossy().starts_with("MANIFEST-"));
    if leveldb {
        let mut stable = false;
        for _ in 1..ATTEMPTS {
            std::thread::sleep(RETRY_DELAY);
            let next = directory_names(source).map_err(|e| copy_error(source, e))?;
            stable = names == next;
            names = next;
            if stable {
                break;
            }
        }
        if !stable {
            stats.warnings.push("a LevelDB file list did not stabilize; copied the latest listing (close Chrome for a consistent snapshot)".to_string());
        }
    }
    let sqlite: BTreeSet<_> = names
        .iter()
        .filter_map(|name| {
            SQLITE_SUFFIXES
                .iter()
                .find_map(|suffix| name.to_str()?.strip_suffix(suffix))
        })
        .chain(["Cookies"])
        .collect();
    let mut processed = BTreeSet::new();
    for name in &names {
        if let Some(category) = skip(name, in_service_worker) {
            stats.skipped[category as usize] += 1;
            continue;
        }
        let sqlite_base = name.to_str().and_then(|name| {
            SQLITE_SUFFIXES
                .iter()
                .find_map(|suffix| name.strip_suffix(suffix))
                .or_else(|| sqlite.contains(name).then_some(name))
        });
        if let Some(base) = sqlite_base {
            if let Some(category) = skip(OsStr::new(base), in_service_worker) {
                stats.skipped[category as usize] += 1;
                continue;
            }
            if processed.contains(base) {
                continue;
            }
        }
        let path = source.join(name);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(m) => m,
            // A SQLite sidecar might disappear before its group is reached.
            Err(e)
                if e.kind() == io::ErrorKind::NotFound
                    && sqlite_base.is_some_and(|base| name != base) =>
            {
                continue;
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound && leveldb => {
                stats.warnings.push("a LevelDB entry disappeared during the copy; close Chrome for a consistent snapshot".to_string());
                continue;
            }
            Err(e) => return Err(copy_error(&path, e)),
        };
        if profile::is_link(&metadata) {
            stats.skipped[Skip::Link as usize] += 1;
        } else if metadata.is_dir() {
            let target = dest.join(name);
            profile::private_directory(&target, false).map_err(|e| copy_error(&path, e))?;
            copy_tree(
                &path,
                &target,
                in_service_worker || name.eq_ignore_ascii_case("Service Worker"),
                stats,
            )?;
        } else if metadata.is_file() {
            let names = if let Some(base) = sqlite_base {
                processed.insert(base);
                std::iter::once(OsString::from(base))
                    .chain(SQLITE_SUFFIXES.map(|suffix| OsString::from(format!("{base}{suffix}"))))
                    .collect()
            } else {
                vec![name.clone()]
            };
            let copies = copy_files(source, dest, &names).map_err(|e| copy_error(&path, e))?;
            if copies[0].is_none() {
                if !leveldb {
                    return Err(copy_error(
                        &path,
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "source data file disappeared or is not a regular file",
                        ),
                    ));
                }
                stats.warnings.push("a LevelDB entry disappeared during the copy; close Chrome for a consistent snapshot".to_string());
            }
            stats.record(copies);
        } else {
            stats.skipped[Skip::Link as usize] += 1;
        }
    }
    Ok(())
}

#[derive(PartialEq, Eq)]
struct FileVersion {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
    #[cfg(windows)]
    created: u64,
}

impl FileVersion {
    fn new(metadata: &Metadata) -> io::Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        #[cfg(windows)]
        use std::os::windows::fs::MetadataExt;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            identity: (
                metadata.dev(),
                metadata.ino(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            ),
            #[cfg(windows)]
            created: metadata.creation_time(),
        })
    }

    fn at(path: &Path) -> io::Result<Option<Self>> {
        match fs::symlink_metadata(path) {
            Ok(m) if m.is_file() && !profile::is_link(&m) => Self::new(&m).map(Some),
            Ok(_) => Ok(None),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// Read SQLite files (including optional WAL/SHM) as a group and compare the entire
/// group's versions before and after. A standalone file is a group of one.
fn copy_files(source: &Path, dest: &Path, names: &[OsString]) -> io::Result<Vec<Option<u64>>> {
    let inputs = names
        .iter()
        .map(|name| source.join(name))
        .collect::<Vec<_>>();
    let outputs = names.iter().map(|name| dest.join(name)).collect::<Vec<_>>();
    let versions = || {
        inputs
            .iter()
            .map(|input| FileVersion::at(input))
            .collect::<io::Result<Vec<_>>>()
    };
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            std::thread::sleep(RETRY_DELAY);
            for output in &outputs {
                match fs::remove_file(output) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e),
                }
            }
        }
        let before = versions()?;
        let mut changed = false;
        for ((input, output), version) in inputs.iter().zip(&outputs).zip(&before) {
            let Some(version) = version else {
                continue;
            };
            let mut file = match open_source(input) {
                Ok(file) => file,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    changed = true;
                    break;
                }
                Err(e) => return Err(e),
            };
            if FileVersion::new(&file.metadata()?)? != *version {
                changed = true;
                break;
            }
            let bytes = io::copy(&mut file, &mut profile::private_file(output)?)?;
            if bytes != version.len || FileVersion::new(&file.metadata()?)? != *version {
                changed = true;
                break;
            }
        }
        if !changed && before == versions()? {
            return Ok(before
                .into_iter()
                .map(|version| version.map(|v| v.len))
                .collect());
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "source data changed repeatedly during the copy",
    ))
}

#[cfg(unix)]
fn open_source(path: &Path) -> io::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Component;

    // Walk from a directory handle. O_NOFOLLOW on only the final component would
    // still let a concurrently replaced parent directory redirect the read.
    let mut current = File::open("/")?;
    let mut components = path.components().peekable();
    if components.next() != Some(Component::RootDir) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "source path is not absolute",
        ));
    }
    while let Some(Component::Normal(name)) = components.next() {
        let name = CString::new(name.as_bytes())?;
        let kind = if components.peek().is_some() {
            libc::O_DIRECTORY
        } else {
            libc::O_NONBLOCK
        };
        // SAFETY: current is live and name is a NUL-terminated single path component.
        let fd = unsafe {
            libc::openat(
                current.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | kind,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: openat returned a newly owned descriptor.
        current = unsafe { File::from_raw_fd(fd) };
        if components.peek().is_none() && current.metadata()?.is_file() {
            return Ok(current);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "source path is not a regular file",
    ))
}

#[cfg(not(unix))]
fn open_source(path: &Path) -> io::Result<File> {
    if path.canonicalize()? != path {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "source file changed identity",
        ));
    }
    profile::read_regular(path)
}

fn copy_error(path: &Path, error: io::Error) -> String {
    #[cfg(windows)]
    if matches!(error.raw_os_error(), Some(32 | 33)) {
        return format!(
            "Chrome profile file {path:?} is locked on Windows; close Chrome before retrying, or use --login"
        );
    }
    format!(
        "cannot copy Chrome profile entry {path:?}: {error}; close Chrome before retrying, or use --login"
    )
}

struct Staging(Option<PathBuf>);

impl Staging {
    fn new(parent: &Path, kind: &str) -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = parent.join(format!(
                ".mcp-browser-{kind}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match profile::private_directory(&path, false) {
                Ok(()) => return Ok(Self(Some(path))),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    fn path(&self) -> &Path {
        self.0.as_deref().expect("staging path is owned")
    }

    fn install(mut self, dest: &Path, destination: Destination) -> Result<bool, String> {
        let install_error = |error| format!("cannot install the staged Chrome profile: {error}");
        if !matches!(destination, Destination::Managed) {
            // Do not exchange (and then recursively delete) an unmarked tree.
            // Unix rename replaces only an empty directory; elsewhere remove it
            // non-recursively, so a concurrent write cannot lose files.
            #[cfg(not(unix))]
            if matches!(destination, Destination::Empty) {
                fs::remove_dir(dest).map_err(install_error)?;
            }
            fs::rename(self.path(), dest).map_err(install_error)?;
            self.0.take();
            return Ok(false);
        }
        if exchange(self.path(), dest).map_err(install_error)? {
            // The previous destination is now in the staging location; the guard removes it.
            return Ok(false);
        }
        // Some filesystems and Windows cannot exchange directories. Keep the old data in a
        // private sibling and restore it if the atomic publication of the new tree fails.
        let mut backup = Self::new(dest.parent().expect("destination has a parent"), "backup")
            .map_err(install_error)?;
        let old = backup.path().join("previous-profile");
        fs::rename(dest, &old).map_err(install_error)?;
        if let Err(error) = fs::rename(self.path(), dest) {
            if fs::rename(&old, dest).is_err() {
                backup.0.take();
                return Err(format!(
                    "cannot install or restore the Chrome profile; previous profile preserved at {old:?}: {error}"
                ));
            }
            return Err(install_error(error));
        }
        self.0.take();
        Ok(true)
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if let Some(path) = self.0.take()
            && let Err(error) = fs::remove_dir_all(&path)
            && error.kind() != io::ErrorKind::NotFound
        {
            eprintln!(
                "Warning: cannot remove private Chrome profile staging or backup at {path:?}: {error}"
            );
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn exchange(staging: &Path, dest: &Path) -> io::Result<bool> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let left = CString::new(staging.as_os_str().as_bytes())?;
    let right = CString::new(dest.as_os_str().as_bytes())?;
    // SAFETY: both paths are valid NUL-terminated strings in the same parent directory.
    #[cfg(target_os = "linux")]
    let result = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            left.as_ptr(),
            libc::AT_FDCWD,
            right.as_ptr(),
            libc::RENAME_EXCHANGE,
        )
    };
    // SAFETY: both paths are valid NUL-terminated strings in the same parent directory.
    #[cfg(target_os = "macos")]
    let result = unsafe { libc::renamex_np(left.as_ptr(), right.as_ptr(), libc::RENAME_SWAP) };
    if result == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ENOSYS | libc::EOPNOTSUPP | libc::EINVAL) => Ok(false),
        _ => Err(error),
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn exchange(_staging: &Path, _dest: &Path) -> io::Result<bool> {
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::tests::TestDir;
    use serde_json::json;

    fn fixture(root: &Path) -> PathBuf {
        let source = root.join("source");
        for directory in ["Default", "Profile 1"] {
            fs::create_dir_all(source.join(directory)).unwrap();
        }
        let state = json!({"profile": { "last_used": "Profile 1", "info_cache": {
            "Default": {"name": "Personal"}, "Profile 1": {"name": "Team Desktop"}
        }}, "os_crypt": {"encrypted_key": "fixture"}});
        fs::write(source.join(LOCAL_STATE), state.to_string()).unwrap();
        source
    }

    fn options(source: &Path, destination: &Path, force: bool) -> SyncOptions {
        SyncOptions {
            destination: destination.to_path_buf(),
            from: Some(source.to_path_buf()),
            profile: None,
            force,
            keychain: Keychain::Mock,
        }
    }

    #[test]
    fn resolves_profile_by_exact_name_directory_last_used_and_direct_path() {
        let workspace = TestDir::new();
        let source = fixture(&workspace.0);
        let state = serde_json::from_reader(File::open(source.join(LOCAL_STATE)).unwrap()).unwrap();
        for selector in [None, Some("Team Desktop"), Some("Profile 1")] {
            assert_eq!(
                choose_profile(&source, &state, None, selector)
                    .unwrap()
                    .directory,
                "Profile 1"
            );
        }
        let (root, direct) = resolve_source(Some(&source.join("Default"))).unwrap();
        assert_eq!(root, source);
        assert_eq!(direct.as_deref(), Some("Default"));
        assert_eq!(
            choose_profile(&root, &state, direct.as_deref(), Some("Personal"))
                .unwrap()
                .directory,
            "Default"
        );
        assert!(
            choose_profile(&root, &state, direct.as_deref(), Some("Profile 1"))
                .unwrap_err()
                .contains("--from")
        );
        assert_eq!(
            choose_profile(&source, &json!({}), None, None)
                .unwrap()
                .directory,
            "Default"
        );
        let missing = choose_profile(&source, &state, None, Some("missing")).unwrap_err();
        for value in [
            "Personal",
            "Default",
            "Team Desktop",
            "Profile 1",
            "last_used",
        ] {
            assert!(missing.contains(value), "{missing}");
        }
        let duplicate = json!({"profile": {"info_cache": {"Default": {"name": "Same"}, "Profile 1": {"name": "Same"}}}});
        assert!(
            choose_profile(&source, &duplicate, None, Some("Same"))
                .unwrap_err()
                .contains("multiple profiles")
        );
    }

    #[test]
    fn rejects_canonical_source_destination_overlap() {
        let workspace = TestDir::new();
        let source = fixture(&workspace.0);
        for dest in [&source, &source.join("not-created/child"), &workspace.0] {
            assert!(
                sync(options(&source, dest, true))
                    .unwrap_err()
                    .contains("must not be equal or contain")
            );
        }
        assert!(!source.join("not-created").exists());
        #[cfg(unix)]
        {
            let alias = workspace.0.join("source-alias");
            std::os::unix::fs::symlink(&source, &alias).unwrap();
            assert!(
                sync(options(&source, &alias.join("child"), true))
                    .unwrap_err()
                    .contains("must not be equal or contain")
            );
        }
    }

    #[test]
    fn skip_list_excludes_non_auth_data_not_web_sessions() {
        for name in [
            "Cache",
            "Cache2",
            "CacheStorage",
            "Code Cache-old",
            "GPUCache",
            "ShaderCache3",
            "GrShaderCache",
            "DawnGraphiteCache",
            "GraphiteDawnCache",
            "CrashpadMetrics",
            "BrowserMetrics-old",
            "Safe Browsing Cookies",
            "component_crx_cache",
            "extensions_crx_cache2",
            "optimization_guide_model_store",
            "SingletonLock",
            "SingletonSocket",
            "SingletonCookie",
            "lockfile",
            "DevToolsActivePort",
            "LOCK",
            "Login Data-wal",
            "Web Data-journal",
            "History",
            "Bookmarks",
            "Top Sites",
            "Favicons",
            "Visited Links",
            "Shortcuts",
            "Thumbnails",
            "Sessions",
        ] {
            assert!(skip(OsStr::new(name), false).is_some(), "{name}");
        }
        for name in [
            "Cookies",
            "Cookies-wal",
            "Cookies-shm",
            "Network",
            "Local Storage",
            "Session Storage",
            "IndexedDB",
            "Service Worker",
        ] {
            assert!(skip(OsStr::new(name), false).is_none(), "{name}");
        }
    }

    #[test]
    fn clones_only_the_selected_profile_and_preserves_raw_local_state() {
        let workspace = TestDir::new();
        let source = fixture(&workspace.0);
        let dest = workspace.0.join("dedicated");
        let selected = source.join("Profile 1");
        for (path, data) in [
            ("Network/Cookies", "db"),
            ("Network/Cookies-wal", "log"),
            ("Network/Cookies-shm", "index"),
            ("Local Storage/leveldb/CURRENT", "MANIFEST-000001"),
            ("Local Storage/leveldb/LOCK", "not copied"),
            ("Session Storage/fixture", "session"),
            ("IndexedDB/fixture", "indexed"),
            ("Service Worker/CacheStorage/fixture", "not copied"),
            ("Login Data", "not copied"),
        ] {
            let file = selected.join(path);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(file, data).unwrap();
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let outside = workspace.0.join("external");
            fs::create_dir(&outside).unwrap();
            fs::write(outside.join("fixture"), b"outside file").unwrap();
            symlink(&outside, selected.join("outside-dir")).unwrap();
            symlink(outside.join("fixture"), selected.join("outside-file")).unwrap();
        }
        sync(options(&source, &dest, false)).unwrap();
        assert_eq!(
            fs::read(dest.join(LOCAL_STATE)).unwrap(),
            fs::read(source.join(LOCAL_STATE)).unwrap()
        );
        assert!(!dest.join("Default").exists());
        let copied = dest.join("Profile 1");
        for (file, data) in [
            ("Network/Cookies", "db"),
            ("Network/Cookies-wal", "log"),
            ("Network/Cookies-shm", "index"),
            ("Session Storage/fixture", "session"),
            ("IndexedDB/fixture", "indexed"),
        ] {
            assert_eq!(fs::read_to_string(copied.join(file)).unwrap(), data);
        }
        for excluded in [
            "Local Storage/leveldb/LOCK",
            "Service Worker/CacheStorage",
            "Login Data",
            "outside-dir",
            "outside-file",
        ] {
            assert!(
                fs::symlink_metadata(copied.join(excluded)).is_err(),
                "{excluded}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for dir in [&dest, &copied, &copied.join("Network")] {
                assert_eq!(
                    fs::metadata(dir).unwrap().permissions().mode() & 0o777,
                    0o700
                );
            }
            for file in [
                dest.join(LOCAL_STATE),
                dest.join(profile::MARKER),
                copied.join("Network/Cookies"),
            ] {
                assert_eq!(
                    fs::metadata(file).unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
        }
    }

    #[test]
    fn refuses_nonempty_destination_and_preserves_it_on_staging_failure() {
        let workspace = TestDir::new();
        let source = fixture(&workspace.0);
        let dest = workspace.0.join("dedicated");
        profile::private_directory(&dest, false).unwrap();
        fs::write(dest.join("old"), "keep this").unwrap();
        for force in [false, true] {
            assert!(
                sync(options(&source, &dest, force))
                    .unwrap_err()
                    .contains("does not have an MCP profile marker")
            );
            assert_eq!(fs::read_to_string(dest.join("old")).unwrap(), "keep this");
        }
        profile::write_marker(&dest, "Default", Keychain::Mock, "fixture").unwrap();
        assert!(
            sync(options(&source, &dest, false))
                .unwrap_err()
                .contains("not empty")
        );
        let child = dest.join("nested");
        profile::private_directory(&child, false).unwrap();
        profile::write_marker(&child, "Default", Keychain::Mock, "fixture").unwrap();
        assert!(
            sync(options(&source, &dest, true))
                .unwrap_err()
                .contains("contains another managed")
        );
        assert!(child.join(profile::MARKER).is_file());
        assert!(
            sync(options(&source, &child, true))
                .unwrap_err()
                .contains("inside another managed")
        );
        fs::remove_file(child.join(profile::MARKER)).unwrap();
        fs::remove_dir(child).unwrap();
        fs::write(dest.join("lockfile"), "unknown owner").unwrap();
        assert!(
            sync(options(&source, &dest, true))
                .unwrap_err()
                .contains("lockfile")
        );
        fs::remove_file(dest.join("lockfile")).unwrap();
        fs::write(source.join(LOCAL_STATE), "invalid staged JSON").unwrap();
        assert!(
            sync(options(&source, &dest, true))
                .unwrap_err()
                .contains("Local State JSON")
        );
        assert_eq!(fs::read_to_string(dest.join("old")).unwrap(), "keep this");
        assert!(!fs::read_dir(&workspace.0).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".mcp-browser-stage-")
        }));
        fixture_overwrite_state(&source);
        sync(options(&source, &dest, true)).unwrap();
        assert!(dest.join(profile::MARKER).is_file());
        assert!(!dest.join("old").exists());
        let empty = workspace.0.join("empty");
        fs::create_dir(&empty).unwrap();
        sync(options(&source, &empty, false)).unwrap();
        assert!(empty.join(profile::MARKER).is_file());
    }

    #[test]
    fn keychain_diagnostics_distinguish_encryption_contexts() {
        let missing = json!({});
        for os in ["linux", "macos", "windows"] {
            assert!(
                keychain_diagnostic(Keychain::Mock, os, &missing)
                    .unwrap()
                    .contains("does not convert")
            );
            assert_eq!(
                keychain_diagnostic(Keychain::Native, os, &missing).is_some(),
                os == "windows"
            );
        }
        for key in ["encrypted_key", "app_bound_encrypted_key"] {
            assert!(
                keychain_diagnostic(
                    Keychain::Native,
                    "windows",
                    &json!({"os_crypt": { (key): "fixture" }})
                )
                .is_none()
            );
        }
    }

    fn fixture_overwrite_state(source: &Path) {
        fs::write(source.join(LOCAL_STATE), b"{}").unwrap();
    }
}
