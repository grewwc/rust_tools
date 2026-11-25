//! Settings and ownership checks for explicitly persistent Chrome profiles.

use serde_json::{Value, json};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

pub(super) const MARKER: &str = ".mcp_browser_profile.json";
pub(super) const TEMP_PROFILE_PREFIX: &str = "mcp_browser-profile-";
pub(super) const DATA_DIR_ENV: &str = "MCP_BROWSER_USER_DATA_DIR";
const DIRECTORY_ENV: &str = "MCP_BROWSER_PROFILE_DIRECTORY";
const KEYSTORE_ENV: &str = "MCP_BROWSER_PROFILE_KEYSTORE";
const MARKER_VERSION: u64 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Keychain {
    #[default]
    Mock,
    Native,
}

impl Keychain {
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "mock" => Some(Self::Mock),
            "native" => Some(Self::Native),
            _ => None,
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::Native => "native",
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct ProfileSettings {
    pub(super) directory: Option<String>,
    pub(super) keychain: Keychain,
}

impl ProfileSettings {
    pub(super) fn with_env(self) -> Result<Self, String> {
        let read_override = |key| match std::env::var(key) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(format!("{key} is not valid Unicode")),
        };
        self.with_overrides(
            read_override(DIRECTORY_ENV)?.as_deref(),
            read_override(KEYSTORE_ENV)?.as_deref(),
        )
    }

    fn with_overrides(
        mut self,
        directory: Option<&str>,
        keychain: Option<&str>,
    ) -> Result<Self, String> {
        if let Some(directory) = directory.filter(|name| !name.trim().is_empty()) {
            if !valid_profile_directory(directory) {
                return Err(format!(
                    "{DIRECTORY_ENV} must be a single Chrome profile directory name"
                ));
            }
            self.directory = Some(directory.to_owned());
        }
        if let Some(keychain) = keychain {
            self.keychain = Keychain::parse(keychain)
                .ok_or_else(|| format!("{KEYSTORE_ENV} must be mock or native"))?;
        }
        Ok(self)
    }
}

pub(super) fn explicit_data_dir() -> Option<PathBuf> {
    std::env::var_os(DATA_DIR_ENV)
        .filter(|path| !path.to_str().is_some_and(|text| text.trim().is_empty()))
        .map(PathBuf::from)
}

#[derive(Clone, Copy)]
pub(super) enum ProfileUse {
    Serve,
    Login,
}

pub(super) struct DedicatedProfile {
    path: PathBuf,
    marker: Option<ProfileSettings>,
}

impl DedicatedProfile {
    /// Read-only preflight for settings and safety. An invalid explicit profile must never fall back.
    pub(super) fn inspect(path: &Path) -> Result<Self, String> {
        let path = resolve_directory(path)?;
        if path.to_str().is_none() {
            return Err("the dedicated Chrome profile path must be valid Unicode".to_string());
        }
        require_dedicated_directory(&path)?;
        let marker = match fs::symlink_metadata(&path) {
            Ok(metadata) => {
                if !metadata.is_dir() || is_link(&metadata) {
                    return Err("the dedicated Chrome profile must be a real directory".to_string());
                }
                if !is_private_directory(&metadata) {
                    return Err("the dedicated Chrome profile must be owned by the current user and have Unix mode 0700".to_string());
                }
                let marker = read_marker(&path)?;
                if marker.is_none()
                    && fs::read_dir(&path)
                        .map_err(|_| "cannot list the dedicated Chrome profile")?
                        .next()
                        .is_some()
                {
                    // An arbitrary custom Chrome directory is not proof of a dedicated agent profile.
                    require_idle(&path, false)?;
                    return Err("the Chrome profile is not empty and has no MCP marker; use --sync-profile or --login with an empty dedicated directory".to_string());
                }
                marker
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(_) => return Err("cannot inspect the dedicated Chrome profile".to_string()),
        };
        Ok(Self { path, marker })
    }

    pub(super) fn require_idle(&self) -> Result<(), String> {
        require_idle(&self.path, true)
    }

    fn resolve_settings(
        marker: Option<ProfileSettings>,
        usage: ProfileUse,
    ) -> Result<ProfileSettings, String> {
        let settings = marker.unwrap_or_default();
        match usage {
            ProfileUse::Serve => settings.with_env(),
            ProfileUse::Login => Ok(settings),
        }
    }

    pub(super) fn settings(self, usage: ProfileUse) -> Result<ProfileSettings, String> {
        let settings = Self::resolve_settings(self.marker, usage)?;
        if let Some(selected) = validate_chrome_profile_paths(
            &self.path,
            settings.directory.as_deref().unwrap_or("Default"),
        )? {
            validate_chrome_profile(&selected)?;
        }
        Ok(settings)
    }

    pub(super) fn prepare(
        self,
        usage: ProfileUse,
    ) -> Result<(PathBuf, Vec<ProfileLease>, ProfileSettings), String> {
        let path = self.path;
        let leases = lease_profile_tree(&path)?;
        // Revalidate while the leases exclude both directory replacement and scratch GC.
        let checked = Self::inspect(&path)?;
        if checked.path != path {
            return Err(
                "the dedicated Chrome profile path changed during initialization".to_string(),
            );
        }
        let unmarked = checked.marker.is_none();
        let settings = Self::resolve_settings(checked.marker, usage)?;
        let directory = settings.directory.as_deref().unwrap_or("Default");
        let _ = validate_chrome_profile_paths(&path, directory)?;
        if unmarked {
            private_directory(&path, true)
                .map_err(|_| "cannot create the dedicated Chrome profile directory")?;
            let inspected = Self::inspect(&path)?;
            if inspected.path != path || inspected.marker.is_some() {
                return Err(
                    "the dedicated Chrome profile changed during initialization".to_string()
                );
            }
            let source = match usage {
                ProfileUse::Serve => "mcp_browser (stdio)",
                ProfileUse::Login => "mcp_browser --login",
            };
            write_marker(&path, directory, settings.keychain, source)?;
        } else {
            clear_stale_chrome_locks(&path)?;
        }
        // When a selected profile does not yet exist, initialize that exact
        // directory. An existing link is never accepted as the directory.
        let selected = path.join(directory);
        if let Err(error) = private_directory(&selected, false)
            && error.kind() != io::ErrorKind::AlreadyExists
        {
            return Err("cannot create the selected Chrome profile directory".to_string());
        }
        let canonical = validate_chrome_profile_paths(&path, directory)?
            .ok_or("the selected Chrome profile disappeared during initialization")?;
        validate_chrome_profile(&canonical)?;
        Ok((path, leases, settings))
    }
}

fn validate_chrome_profile_paths(root: &Path, directory: &str) -> Result<Option<PathBuf>, String> {
    let local_state = root.join("Local State");
    match fs::symlink_metadata(&local_state) {
        Ok(m) if m.is_file() && !is_link(&m) => require_single_file_link(&local_state, &m)
            .map_err(|reason| format!("Chrome Local State {reason}"))?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        _ => return Err("Chrome Local State must be a regular file, not a link".to_string()),
    }
    let selected = root.join(directory);
    match fs::symlink_metadata(&selected) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Ok(m) if m.is_dir() && !is_link(&m) => {}
        _ => {
            return Err(
                "the selected Chrome profile must be a real directory, not a link".to_string(),
            );
        }
    }
    let canonical = selected
        .canonicalize()
        .map_err(|_| "cannot resolve the selected Chrome profile")?;
    if canonical == root || !path_contains(root, &canonical) {
        return Err(
            "the selected Chrome profile escapes the dedicated user-data directory".to_string(),
        );
    }
    Ok(Some(canonical))
}

#[cfg(test)]
std::thread_local! {
    static PROFILE_TREE_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn validate_chrome_profile(selected: &Path) -> Result<(), String> {
    #[cfg(test)]
    PROFILE_TREE_SCANS.with(|count| count.set(count.get() + 1));
    let unsafe_file = find_tree_entry(selected, |path, metadata| {
        if is_link(metadata) {
            return Some(
                "the selected Chrome profile contains a symbolic link or reparse point".to_string(),
            );
        }
        if metadata.is_file() {
            return require_single_file_link(path, metadata)
                .err()
                .map(|reason| format!("a regular file in the selected Chrome profile {reason}"));
        }
        None
    })
    .map_err(|_| "cannot inspect the selected Chrome profile for links")?;
    match unsafe_file {
        Some(reason) => Err(reason),
        None => Ok(()),
    }
}

fn require_single_file_link(path: &Path, metadata: &fs::Metadata) -> Result<(), &'static str> {
    match file_link_count(path, metadata) {
        Some(1) => Ok(()),
        Some(2..) => Err("has multiple hard links; use an independent copy of the file"),
        // An unavailable or zero count does not establish private ownership.
        _ => Err("has an unverifiable hard link count"),
    }
}

#[cfg(unix)]
fn file_link_count(_path: &Path, metadata: &fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(metadata.nlink())
}

#[cfg(windows)]
fn file_link_count(path: &Path, _metadata: &fs::Metadata) -> Option<u64> {
    use std::ffi::c_void;
    use std::mem::{MaybeUninit, size_of};
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::{AsRawHandle, RawHandle};

    // MetadataExt::number_of_links is unstable. Query FILE_STANDARD_INFO on
    // a handle opened for attributes only, without following reparse points.
    const FILE_READ_ATTRIBUTES: u32 = 0x0080;
    const FILE_STANDARD_INFO_CLASS: i32 = 1;
    #[repr(C)]
    struct FileStandardInfo {
        allocation_size: i64,
        end_of_file: i64,
        number_of_links: u32,
        delete_pending: u8,
        directory: u8,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFileInformationByHandleEx(
            file: RawHandle,
            class: i32,
            info: *mut c_void,
            size: u32,
        ) -> i32;
    }

    let mut options = OpenOptions::new();
    no_follow(options.access_mode(FILE_READ_ATTRIBUTES));
    let file = require_regular(options.open(path).ok()?).ok()?;
    let mut info = MaybeUninit::<FileStandardInfo>::uninit();
    // SAFETY: the open handle is live and the output buffer matches the requested class.
    let found = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FILE_STANDARD_INFO_CLASS,
            info.as_mut_ptr().cast(),
            size_of::<FileStandardInfo>() as u32,
        )
    };
    if found == 0 {
        return None;
    }
    // SAFETY: the successful query initialized all fields of FILE_STANDARD_INFO.
    let info = unsafe { info.assume_init() };
    (info.directory == 0 && info.delete_pending == 0).then_some(u64::from(info.number_of_links))
}

#[cfg(not(any(unix, windows)))]
fn file_link_count(_path: &Path, _metadata: &fs::Metadata) -> Option<u64> {
    None
}

pub(super) fn valid_profile_directory(name: &str) -> bool {
    !name.trim().is_empty()
        && !name.ends_with(['.', ' '])
        && !name
            .chars()
            .any(|c| c.is_control() || matches!(c, '/' | '\\' | ':'))
        && matches!(
            Path::new(name).components().next(),
            Some(Component::Normal(_))
        )
}

pub(super) fn read_marker(dir: &Path) -> Result<Option<ProfileSettings>, String> {
    let path = dir.join(MARKER);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(format!("cannot inspect Chrome profile marker {MARKER}")),
    };
    if is_link(&metadata) || !metadata.is_file() {
        return Err(format!(
            "Chrome profile marker {MARKER} must be a regular file"
        ));
    }
    let file =
        read_regular(&path).map_err(|_| format!("cannot open Chrome profile marker {MARKER}"))?;
    let value: Value = serde_json::from_reader(file).map_err(|error| {
        format!(
            "invalid JSON in Chrome profile marker {MARKER} (line {}, column {})",
            error.line(),
            error.column()
        )
    })?;
    let invalid =
        |field| format!("Chrome profile marker {MARKER} has a missing or invalid {field} field");
    if value["version"].as_u64() != Some(MARKER_VERSION) {
        return Err(format!(
            "Chrome profile marker {MARKER} requires version {MARKER_VERSION}"
        ));
    }
    let directory = value["profile_directory"]
        .as_str()
        .filter(|name| valid_profile_directory(name))
        .ok_or_else(|| invalid("profile_directory"))?;
    let keychain = value["keychain"]
        .as_str()
        .and_then(Keychain::parse)
        .ok_or_else(|| invalid("keychain (expected mock or native)"))?;
    value["source"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid("source"))?;
    value["synced_at_unix"]
        .as_u64()
        .ok_or_else(|| invalid("synced_at_unix"))?;
    Ok(Some(ProfileSettings {
        directory: Some(directory.to_owned()),
        keychain,
    }))
}

pub(super) fn write_marker(
    dir: &Path,
    directory: &str,
    keychain: Keychain,
    source: &str,
) -> Result<(), String> {
    let path = dir.join(MARKER);
    let synced_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| "the system clock predates the Unix epoch")?;
    let marker = json!({
        "version": MARKER_VERSION,
        "profile_directory": directory,
        "keychain": keychain.as_str(),
        "source": source,
        "synced_at_unix": synced_at.as_secs(),
    });
    let file =
        private_file(&path).map_err(|_| format!("cannot create Chrome profile marker {MARKER}"))?;
    if serde_json::to_writer_pretty(file, &marker).is_err() {
        let _ = fs::remove_file(&path);
        return Err(format!("cannot write Chrome profile marker {MARKER}"));
    }
    Ok(())
}

pub(super) fn is_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        // A junction or other reparse point can also redirect traversal.
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    metadata.is_symlink()
}

fn find_tree_entry<T>(
    root: &Path,
    mut matches: impl FnMut(&Path, &fs::Metadata) -> Option<T>,
) -> io::Result<Option<T>> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        match fs::symlink_metadata(&dir) {
            Ok(m) if m.is_dir() && !is_link(&m) => {}
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "directory changed during traversal",
                ));
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        }
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        };
        for entry in entries {
            let path = entry?.path();
            let metadata = match fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            };
            if let Some(found) = matches(&path, &metadata) {
                return Ok(Some(found));
            }
            if metadata.is_dir() && !is_link(&metadata) {
                pending.push(path);
            }
        }
    }
    Ok(None)
}

pub(super) fn contains_persistent_marker(root: &Path, include_root: bool) -> io::Result<bool> {
    find_tree_entry(root, |path, _| {
        (path.file_name().is_some_and(|name| name == MARKER)
            && (include_root || path.parent() != Some(root)))
        .then_some(())
    })
    .map(|found| found.is_some())
}

fn no_follow(options: &mut OpenOptions) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
}

fn require_regular(file: File) -> io::Result<File> {
    let metadata = file.metadata()?;
    if is_link(&metadata) || !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a regular file",
        ));
    }
    Ok(file)
}

pub(super) fn read_regular(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    no_follow(options.read(true));
    require_regular(options.open(path)?)
}

pub(super) fn private_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.write(true).create_new(true).open(path)
}

pub(super) fn private_directory(path: &Path, recursive: bool) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.recursive(recursive).create(path)
}

pub(super) fn is_private_directory(metadata: &fs::Metadata) -> bool {
    if !metadata.is_dir() || is_link(metadata) {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // SAFETY: geteuid only reads the process credentials.
        metadata.mode() & 0o777 == 0o700 && metadata.uid() == unsafe { libc::geteuid() }
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Resolve an existing directory or a new path under its nearest existing ancestor.
pub(super) fn resolve_directory(path: &Path) -> Result<PathBuf, String> {
    let path = std::path::absolute(path).map_err(|_| "cannot resolve Chrome profile directory")?;
    let mut current = path.as_path();
    let mut missing = Vec::new();
    loop {
        match fs::symlink_metadata(current) {
            Ok(_) => {
                let mut base = current
                    .canonicalize()
                    .map_err(|_| "cannot resolve Chrome profile directory")?;
                if !base.is_dir() {
                    return Err("Chrome profile path is not a directory".to_string());
                }
                base.extend(missing.into_iter().rev());
                return Ok(base);
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                missing.push(
                    current
                        .file_name()
                        .ok_or("cannot resolve Chrome profile directory")?,
                );
                current = current
                    .parent()
                    .ok_or("cannot resolve Chrome profile directory")?;
            }
            Err(_) => return Err("cannot inspect Chrome profile directory".to_string()),
        }
    }
}

pub(super) fn path_contains(parent: &Path, child: &Path) -> bool {
    #[cfg(windows)]
    {
        let mut components = child.components();
        parent.components().all(|expected| {
            components.next().is_some_and(|actual| {
                actual
                    .as_os_str()
                    .eq_ignore_ascii_case(expected.as_os_str())
            })
        })
    }
    #[cfg(not(windows))]
    child.starts_with(parent)
}

pub(super) fn paths_overlap(first: &Path, second: &Path) -> bool {
    path_contains(first, second) || path_contains(second, first)
}

pub(super) fn default_browser_roots() -> Vec<PathBuf> {
    let env_path = |key| {
        std::env::var_os(key)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    #[cfg(target_os = "windows")]
    let roots = env_path("LOCALAPPDATA")
        .into_iter()
        .flat_map(|base| {
            [
                base.join("Google/Chrome/User Data"),
                base.join("Chromium/User Data"),
                base.join("Microsoft/Edge/User Data"),
                base.join("Google/Chrome Beta/User Data"),
                base.join("Google/Chrome SxS/User Data"),
            ]
        })
        .collect();
    #[cfg(target_os = "macos")]
    let roots = env_path("HOME")
        .into_iter()
        .flat_map(|home| {
            let base = home.join("Library/Application Support");
            [
                base.join("Google/Chrome"),
                base.join("Chromium"),
                base.join("Microsoft Edge"),
                base.join("Google/Chrome Beta"),
                base.join("Google/Chrome Canary"),
            ]
        })
        .collect();
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let roots = {
        let home = env_path("HOME");
        let standard = home.as_ref().map(|home| home.join(".config"));
        let config = env_path("XDG_CONFIG_HOME");
        let chrome = env_path("CHROME_CONFIG_HOME");
        chrome
            .clone()
            .into_iter()
            .chain(config.clone())
            .chain(standard.clone())
            .map(|base| base.join("google-chrome"))
            .chain(
                config
                    .into_iter()
                    .chain(standard)
                    .chain(chrome)
                    .flat_map(|base| {
                        [
                            base.join("chromium"),
                            base.join("microsoft-edge"),
                            base.join("google-chrome-beta"),
                            base.join("google-chrome-unstable"),
                        ]
                    }),
            )
            .chain(home.into_iter().flat_map(|home| {
                [
                    home.join("snap/chromium/common/chromium"),
                    home.join(".var/app/org.chromium.Chromium/config/chromium"),
                ]
            }))
            .collect()
    };
    roots
}

pub(super) fn require_dedicated_directory(dir: &Path) -> Result<(), String> {
    let configured = std::env::var_os("CHROME_USER_DATA_DIR")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from);
    if default_browser_roots()
        .into_iter()
        .chain(configured)
        .any(|root| {
            // Check the spelling as well as the resolved target: a symlink to a
            // browser profile does not make its containing config directory safe.
            std::path::absolute(&root).is_ok_and(|root| paths_overlap(dir, &root))
                || root
                    .canonicalize()
                    .or_else(|_| resolve_directory(&root))
                    .is_ok_and(|root| paths_overlap(dir, &root))
        })
    {
        return Err(
            "choose a dedicated Chrome profile directory outside your ordinary browser data"
                .to_string(),
        );
    }
    for parent in dir.ancestors().skip(1) {
        match fs::symlink_metadata(parent.join(MARKER)) {
            Ok(_) => {
                return Err("the destination is inside another managed Chrome profile".to_string());
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(_) => {
                return Err(
                    "cannot verify that the dedicated profile has no managed ancestor".to_string(),
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct ProfileLease {
    file: File,
    path: PathBuf,
}

impl ProfileLease {
    fn acquire(file: File, path: PathBuf) -> Result<Self, String> {
        file.try_lock().map_err(
            |_| "agent Chrome profile is in use or cannot be locked; close the agent browser and retry",
        )?;
        let lease = Self { file, path };
        // An opener paused before try_lock may have opened a scratch lease
        // that was later unlinked. Only a lease still named by the path is valid.
        #[cfg(any(unix, windows))]
        if !lease_file_is_current(&lease.file, &lease.path) {
            return Err("agent Chrome profile lease changed; retry the operation".to_string());
        }
        Ok(lease)
    }

    fn reserve_for_profile_tree(&self) -> Result<(), String> {
        // Keep a reservation on this inode even if its profile has not been created yet.
        let error = "cannot reserve the agent Chrome profile lease file";
        if self.file.metadata().map_err(|_| error)?.len() == 0 {
            self.file.set_len(1).map_err(|_| error)?;
        }
        Ok(())
    }

    pub(super) fn remove_unreserved_scratch_lock(&self, dir: &Path) {
        if self.file.metadata().is_ok_and(|metadata| metadata.len() == 0) {
            self.remove_scratch_lock(dir);
        }
    }

    /// The caller must have verified that this path was an owned scratch profile.
    pub(super) fn remove_scratch_lock(&self, dir: &Path) {
        if dir.file_name().and_then(scratch_pid).is_some()
            && profile_lock_path(dir).is_ok_and(|path| path == self.path)
            && fs::symlink_metadata(dir).is_err_and(|err| err.kind() == io::ErrorKind::NotFound)
            && lease_file_is_current(&self.file, &self.path)
        {
            let _ = fs::remove_file(&self.path);
        }
    }
}

impl Drop for ProfileLease {
    fn drop(&mut self) {
        // On Unix a fork can retain the file description until exec. Unlock it
        // explicitly instead of waiting for every copy of the handle to close.
        let _ = self.file.unlock();
    }
}

fn profile_lock_path(dir: &Path) -> Result<PathBuf, &'static str> {
    let name = dir
        .file_name()
        .ok_or("Chrome profile directory requires a parent")?;
    let mut lock = OsString::from(".");
    lock.push(name);
    lock.push(".mcp_browser.lock");
    Ok(dir.with_file_name(lock))
}

#[cfg(unix)]
fn lease_file_is_current(file: &File, path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;

    let (Ok(locked), Ok(named)) = (file.metadata(), fs::symlink_metadata(path)) else {
        return false;
    };
    named.is_file() && locked.dev() == named.dev() && locked.ino() == named.ino()
}

#[cfg(windows)]
fn lease_file_is_current(file: &File, path: &Path) -> bool {
    use std::ffi::c_void;
    use std::mem::{MaybeUninit, size_of};
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::{AsRawHandle, RawHandle};

    const FILE_READ_ATTRIBUTES: u32 = 0x0080;
    const FILE_ID_INFO_CLASS: i32 = 18;
    #[derive(PartialEq, Eq)]
    #[repr(C)]
    struct FileIdInfo {
        volume_serial_number: u64,
        file_id: [u8; 16],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFileInformationByHandleEx(
            file: RawHandle,
            class: i32,
            info: *mut c_void,
            size: u32,
        ) -> i32;
    }
    fn file_id(file: &File) -> Option<FileIdInfo> {
        let mut info = MaybeUninit::<FileIdInfo>::uninit();
        // SAFETY: the open handle is live and the buffer matches FILE_ID_INFO.
        let found = unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle(),
                FILE_ID_INFO_CLASS,
                info.as_mut_ptr().cast(),
                size_of::<FileIdInfo>() as u32,
            )
        };
        if found == 0 {
            return None;
        }
        // SAFETY: a successful query initialized the entire output structure.
        let info = unsafe { info.assume_init() };
        (info.file_id != [0; 16]).then_some(info)
    }

    let mut options = OpenOptions::new();
    no_follow(options.access_mode(FILE_READ_ATTRIBUTES));
    let Ok(named) = options.open(path).and_then(require_regular) else {
        return false;
    };
    file_id(file).is_some_and(|locked| file_id(&named) == Some(locked))
}

#[cfg(not(any(unix, windows)))]
fn lease_file_is_current(_file: &File, _path: &Path) -> bool {
    false
}

/// A persistent profile keeps its sibling lock across directory replacement.
pub(super) fn lease_profile(dir: &Path, create: bool) -> Result<Option<ProfileLease>, String> {
    let path = profile_lock_path(dir)?;
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    no_follow(&mut options);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = if create {
        match options.create_new(true).open(&path) {
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                options.create_new(false).open(&path)
            }
            opened => opened,
        }
    } else {
        options.open(&path)
    };
    let file = match file {
        Ok(file) => require_regular(file).map_err(|_| "invalid agent Chrome profile lease file")?,
        Err(e) if !create && e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("cannot open the agent Chrome profile lease file".to_string()),
    };
    ProfileLease::acquire(file, path).map(Some)
}

pub(super) fn scratch_pid(name: &OsStr) -> Option<u32> {
    let name = name.to_str()?.strip_prefix(TEMP_PROFILE_PREFIX)?;
    let pid: u32 = name.parse().ok()?;
    (pid > 0 && name == pid.to_string()).then_some(pid)
}

/// Acquire outer scratch leases before making any directories within them, and
/// return the target's own lease last. GC takes the same lease on the candidate
/// scratch root before scanning it.
pub(super) fn lease_profile_tree(dir: &Path) -> Result<Vec<ProfileLease>, String> {
    let mut roots: Vec<_> = dir
        .ancestors()
        .skip(1)
        .filter(|path| path.file_name().and_then(scratch_pid).is_some())
        .collect();
    roots.reverse();
    roots.push(dir);
    roots
        .into_iter()
        .map(|root| {
            private_directory(
                root.parent()
                    .ok_or("Chrome profile directory requires a parent")?,
                true,
            )
            .map_err(|_| "cannot create the dedicated Chrome profile parent directory")?;
            let lease = lease_profile(root, true)?.expect("creation requested");
            lease.reserve_for_profile_tree()?;
            Ok(lease)
        })
        .collect()
}

const LOCK_COMPANIONS: [&str; 4] = [
    "lockfile",
    "DevToolsActivePort",
    "SingletonSocket",
    "SingletonCookie",
];

fn inspect_idle(dir: &Path, allow_stale: bool) -> Result<bool, String> {
    let active = |name| {
        format!(
            "Chrome profile has an active or unverified {name}; close the agent browser and retry or choose another directory"
        )
    };
    let singleton = dir.join("SingletonLock");
    let stale = match fs::symlink_metadata(&singleton) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => false,
        Ok(m) if allow_stale && is_link(&m) && stale_singleton(&singleton) => true,
        _ => return Err(active("SingletonLock")),
    };
    for name in LOCK_COMPANIONS {
        match fs::symlink_metadata(dir.join(name)) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Ok(m) if stale && (m.is_file() || is_link(&m)) => {}
            _ => return Err(active(name)),
        }
    }
    Ok(stale)
}

pub(super) fn require_idle(dir: &Path, allow_stale: bool) -> Result<(), String> {
    inspect_idle(dir, allow_stale).map(|_| ())
}

/// For a managed profile held under lease, a dead local singleton proves that
/// the companion files are leftovers. Keep that proof until cleanup completes.
fn clear_stale_chrome_locks(dir: &Path) -> Result<(), String> {
    if inspect_idle(dir, true)? {
        for name in LOCK_COMPANIONS.into_iter().chain(["SingletonLock"]) {
            match fs::remove_file(dir.join(name)) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(_) => return Err(format!("cannot remove the stale Chrome {name}")),
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn stale_singleton(path: &Path) -> bool {
    let Ok(target) = fs::read_link(path) else {
        return false;
    };
    let Some((host, pid)) = target.to_str().and_then(|s| s.rsplit_once('-')) else {
        return false;
    };
    let mut buffer = [0u8; 256];
    // SAFETY: the buffer is writable for the length passed to gethostname.
    if unsafe { libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len()) } != 0 {
        return false;
    }
    let Some(end) = buffer.iter().position(|byte| *byte == 0) else {
        return false;
    };
    host.as_bytes() == &buffer[..end] && pid.parse::<u32>().is_ok_and(process_is_dead)
}

#[cfg(not(unix))]
fn stale_singleton(_path: &Path) -> bool {
    false
}

#[cfg(unix)]
pub(super) fn process_is_dead(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // SAFETY: for a positive PID, signal 0 only checks existence and permissions.
    pid > 0
        && unsafe { libc::kill(pid, 0) } != 0
        && io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

#[cfg(windows)]
pub(super) fn process_is_dead(pid: u32) -> bool {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut c_void;
        fn WaitForSingleObject(handle: *mut c_void, timeout_ms: u32) -> u32;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }
    if pid == 0 {
        return false;
    }
    // SAFETY: open a non-inheritable, synchronization-only handle to the given PID.
    let handle = unsafe { OpenProcess(0x0010_0000, 0, pid) };
    if handle.is_null() {
        return io::Error::last_os_error().raw_os_error() == Some(87); // ERROR_INVALID_PARAMETER
    }
    // SAFETY: the successful open returned a live handle. Zero timeout does not block.
    unsafe {
        let exited = WaitForSingleObject(handle, 0) == 0; // WAIT_OBJECT_0
        CloseHandle(handle);
        exited
    }
}

#[cfg(not(any(unix, windows)))]
pub(super) fn process_is_dead(_pid: u32) -> bool {
    false
}

/// Chrome can leave this file after a clean exit. Remove only our own endpoint.
pub(super) fn clear_closed_debug_port(dir: &Path, ws: &str) {
    let Some((authority, endpoint)) = ws.strip_prefix("ws://").and_then(|s| s.split_once('/'))
    else {
        return;
    };
    let Some((_, port)) = authority.rsplit_once(':') else {
        return;
    };
    let file = dir.join("DevToolsActivePort");
    let mut content = String::new();
    if read_regular(&file)
        .and_then(|f| f.take(512).read_to_string(&mut content))
        .is_ok()
        && content.lines().eq([port, &format!("/{endpoint}")])
    {
        let _ = fs::remove_file(file);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(crate) struct TestDir(pub(crate) PathBuf);

    impl TestDir {
        pub(crate) fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "mcp-browser-test-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match private_directory(&path, false) {
                    Ok(()) => return Self(path.canonicalize().unwrap()),
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("cannot create test directory: {e}"),
                }
            }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn marker_defaults_and_env_precedence() {
        let dir = TestDir::new();
        assert!(read_marker(&dir.0).unwrap().is_none());
        write_marker(&dir.0, "Profile 2", Keychain::Native, "/fixture").unwrap();
        let marked = || read_marker(&dir.0).unwrap().unwrap();
        let settings = marked().with_overrides(Some(" \t"), None).unwrap();
        assert_eq!(settings.directory.as_deref(), Some("Profile 2"));
        assert_eq!(settings.keychain, Keychain::Native);
        let settings = marked()
            .with_overrides(Some("Override"), Some("mock"))
            .unwrap();
        assert_eq!(settings.directory.as_deref(), Some("Override"));
        assert_eq!(settings.keychain, Keychain::Mock);
        assert_eq!(
            ProfileSettings::default()
                .with_overrides(None, Some("native"))
                .unwrap()
                .keychain,
            Keychain::Native
        );
        for value in ["", "unknown"] {
            assert!(
                marked()
                    .with_overrides(None, Some(value))
                    .unwrap_err()
                    .contains(KEYSTORE_ENV)
            );
        }
        assert!(
            marked()
                .with_overrides(Some("../other"), None)
                .unwrap_err()
                .contains(DIRECTORY_ENV)
        );
    }

    #[test]
    fn malformed_marker_cannot_be_overridden() {
        let dir = TestDir::new();
        let path = dir.0.join(MARKER);
        fs::write(&path, "not JSON").unwrap();
        assert!(
            DedicatedProfile::inspect(&dir.0)
                .err()
                .unwrap()
                .contains("invalid JSON")
        );
        let complete = json!({
            "version": 1, "profile_directory": "Default", "keychain": "mock", "source": "/fixture", "synced_at_unix": 1,
        });
        for key in complete.as_object().unwrap().keys() {
            let mut incomplete = complete.clone();
            incomplete.as_object_mut().unwrap().remove(key);
            fs::write(&path, incomplete.to_string()).unwrap();
            assert!(read_marker(&dir.0).unwrap_err().contains(key));
        }
    }

    #[test]
    fn dedicated_profile_rechecks_before_using_an_existing_directory() {
        let parent = TestDir::new();
        let dir = parent.0.join("dedicated");
        let pending = DedicatedProfile::inspect(&dir).unwrap();
        assert!(!dir.exists(), "preflight must not create the directory");
        private_directory(&dir, false).unwrap();
        let sentinel = dir.join("fixture");
        fs::write(&sentinel, "pre-existing data").unwrap();
        assert!(
            pending
                .prepare(ProfileUse::Login)
                .unwrap_err()
                .contains("has no MCP marker")
        );
        assert!(!dir.join(MARKER).exists());
        assert_eq!(fs::read_to_string(&sentinel).unwrap(), "pre-existing data");
        fs::remove_file(sentinel).unwrap();
        let (_, lease, settings) = DedicatedProfile::inspect(&dir)
            .unwrap()
            .prepare(ProfileUse::Login)
            .unwrap();
        assert_eq!(settings, ProfileSettings::default());
        assert_eq!(
            read_marker(&dir).unwrap().unwrap().directory.as_deref(),
            Some("Default")
        );
        drop(lease);
    }

    #[test]
    fn managed_profile_scans_once_before_tools_and_once_after_prepare() {
        let parent = TestDir::new();
        let root = parent.0.join("managed");
        let usage = ProfileUse::Serve;
        let settings = ProfileSettings {
            directory: Some("Default".to_string()),
            keychain: Keychain::Mock,
        }
        .with_env()
        .unwrap();
        let nested = root.join(settings.directory.as_deref().unwrap()).join("Network");
        private_directory(&nested, true).unwrap();
        fs::write(nested.join("Cookies"), "synthetic data").unwrap();
        fs::write(root.join("Local State"), "{}").unwrap();
        write_marker(&root, "Default", Keychain::Mock, "fixture").unwrap();

        PROFILE_TREE_SCANS.set(0);
        let preflight = DedicatedProfile::inspect(&root).unwrap().settings(usage).unwrap();
        assert_eq!(PROFILE_TREE_SCANS.get(), 1);
        // The selected directory already exists, so mkdir must take the AlreadyExists path.
        let (prepared, _leases, settings) = DedicatedProfile::inspect(&root).unwrap().prepare(usage).unwrap();
        assert_eq!(prepared, root);
        assert_eq!(settings, preflight);
        assert_eq!(PROFILE_TREE_SCANS.get(), 2);
    }

    #[test]
    fn dedicated_profile_initializes_fresh_roots_but_rejects_a_conflicting_marker() {
        let parent = TestDir::new();
        let empty = parent.0.join("empty");
        private_directory(&empty, false).unwrap();
        for root in [empty, parent.0.join("missing")] {
            PROFILE_TREE_SCANS.set(0);
            let (prepared, _leases, _) = DedicatedProfile::inspect(&root)
                .unwrap()
                .prepare(ProfileUse::Login)
                .unwrap();
            assert_eq!(prepared, root);
            assert!(root.join("Default").is_dir());
            assert!(read_marker(&root).unwrap().is_some());
            assert_eq!(PROFILE_TREE_SCANS.get(), 1);
        }

        let root = parent.0.join("conflict");
        private_directory(&root, false).unwrap();
        let pending = DedicatedProfile::inspect(&root).unwrap();
        let marker = root.join(MARKER);
        fs::write(&marker, "not JSON").unwrap();
        assert!(pending.prepare(ProfileUse::Login).unwrap_err().contains("invalid JSON"));
        assert_eq!(fs::read_to_string(marker).unwrap(), "not JSON");
        assert!(!root.join("Default").exists());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_linked_profile_and_storage_but_accepts_root_alias() {
        use std::os::unix::fs::symlink;
        let dir = TestDir::new();
        let root = dir.0.join("managed");
        let outside = dir.0.join("outside");
        private_directory(&root, false).unwrap();
        private_directory(&outside, false).unwrap();
        write_marker(&root, "Default", Keychain::Mock, "fixture").unwrap();
        let alias = dir.0.join("alias");
        symlink(&root, &alias).unwrap();
        let settings = || {
            DedicatedProfile::inspect(&alias)
                .unwrap()
                .settings(ProfileUse::Login)
        };
        assert!(settings().is_ok());
        let selected = root.join("Default");
        symlink(&outside, &selected).unwrap();
        assert!(settings().unwrap_err().contains("real directory"));
        fs::remove_file(&selected).unwrap();
        private_directory(&selected.join("Network"), true).unwrap();
        let outside_file = outside.join("fixture");
        fs::write(&outside_file, "unchanged").unwrap();
        let cookie = selected.join("Network/Cookies");
        symlink(&outside_file, &cookie).unwrap();
        assert!(settings().unwrap_err().contains("symbolic link"));
        fs::remove_file(cookie).unwrap();
        symlink(&outside_file, root.join("Local State")).unwrap();
        assert!(settings().unwrap_err().contains("Local State"));
        assert_eq!(fs::read_to_string(outside_file).unwrap(), "unchanged");
    }

    #[test]
    fn cooperative_profile_lease_and_debug_port_ownership() {
        let dir = TestDir::new();
        let profile = dir.0.join("profile");
        private_directory(&profile, false).unwrap();
        assert!(lease_profile(&profile, false).unwrap().is_none());
        let lease = lease_profile(&profile, true).unwrap().unwrap();
        assert!(lease_profile(&profile, true).is_err());
        assert!(lease_profile(&profile, false).is_err());
        #[cfg(unix)]
        let duplicate = lease.file.try_clone().unwrap();
        drop(lease);
        assert!(lease_profile(&profile, false).unwrap().is_some());
        assert!(lease_profile(&profile, true).unwrap().is_some());
        #[cfg(unix)]
        {
            let lock = profile_lock_path(&profile).unwrap();
            assert!(lease_file_is_current(&duplicate, &lock));
        }

        let port_file = profile.join("DevToolsActivePort");
        fs::write(&port_file, "12000\n/devtools/browser/agent\n").unwrap();
        assert!(
            require_idle(&profile, true)
                .unwrap_err()
                .contains("DevToolsActivePort")
        );
        clear_closed_debug_port(&profile, "ws://127.0.0.1:12000/devtools/browser/other");
        assert!(port_file.exists());
        clear_closed_debug_port(&profile, "ws://127.0.0.1:12000/devtools/browser/agent");
        assert!(!port_file.exists());
    }

    #[cfg(unix)]
    #[test]
    fn scratch_lock_cleanup_rejects_a_delayed_opener() {
        let dir = TestDir::new();
        let scratch = dir.0.join("mcp_browser-profile-42");
        let lock = dir.0.join(".mcp_browser-profile-42.mcp_browser.lock");
        assert_eq!(profile_lock_path(&scratch).unwrap(), lock);
        private_directory(&scratch, false).unwrap();
        let lease = lease_profile(&scratch, true).unwrap().unwrap();
        // These are the two steps of a competing lease acquisition, with GC in between.
        let pending = OpenOptions::new().read(true).write(true).open(&lock).unwrap();
        lease.remove_scratch_lock(&scratch);
        assert!(lock.is_file());
        fs::remove_dir(&scratch).unwrap();
        lease.remove_scratch_lock(&dir.0.join("mcp_browser-profile-43"));
        assert!(lock.is_file());
        lease.remove_scratch_lock(&scratch);
        assert!(!lock.exists());
        drop(lease);
        let _replacement = lease_profile(&scratch, true).unwrap().unwrap();
        assert!(
            ProfileLease::acquire(pending, lock)
                .unwrap_err()
                .contains("lease changed")
        );
    }

    #[cfg(unix)]
    #[test]
    fn force_only_allows_a_provably_stale_local_singleton() {
        use std::os::unix::fs::symlink;
        let dir = TestDir::new();
        let lock = dir.0.join("SingletonLock");
        let mut host = [0u8; 256];
        // SAFETY: host has room for the provided length.
        assert_eq!(
            unsafe { libc::gethostname(host.as_mut_ptr().cast(), host.len()) },
            0
        );
        let host =
            std::str::from_utf8(&host[..host.iter().position(|&b| b == 0).unwrap()]).unwrap();
        symlink(format!("{host}-{}", std::process::id()), &lock).unwrap();
        assert!(require_idle(&dir.0, true).is_err());
        for name in ["DevToolsActivePort", "lockfile"] {
            fs::write(dir.0.join(name), b"fixture").unwrap();
        }
        assert!(clear_stale_chrome_locks(&dir.0).is_err());
        assert!(dir.0.join("DevToolsActivePort").is_file());
        fs::remove_file(&lock).unwrap();
        assert!(process_is_dead(i32::MAX as u32));
        symlink(format!("{host}-{}", i32::MAX), &lock).unwrap();
        assert!(require_idle(&dir.0, false).is_err());
        assert!(require_idle(&dir.0, true).is_ok());
        clear_stale_chrome_locks(&dir.0).unwrap();
        for name in ["SingletonLock", "DevToolsActivePort", "lockfile"] {
            assert!(fs::symlink_metadata(dir.0.join(name)).is_err(), "{name}");
        }
    }
}
