//! Browser session lifecycle: lazily launch a controlled Chrome, reuse a single Page,
//! poll the CDP Handler in the background, and shut down cleanly on exit.
//!
//! Invariants:
//! - **The Handler must be polled continuously**, otherwise no CDP call (goto/click/...)
//!   makes progress. `launch()` solves this with a `tokio::spawn`'d
//!   `while handler.next().await` loop.
//! - Reuse a **single** Page; login state / multi-step flows rely on it.
//! - `launch` starts a controlled Chrome with a separate profile. Attaching a
//!   user-started debugging instance requires MCP_BROWSER_WS_URL.

use crate::profile::{
    self, DedicatedProfile, Keychain, ProfileLease, ProfileSettings, ProfileUse,
    TEMP_PROFILE_PREFIX,
};
use chromiumoxide::browser::{Browser, BrowserConfig, BrowserConfigBuilder};
use chromiumoxide::handler::Handler;
use chromiumoxide::page::Page;
use futures_util::StreamExt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::time::Duration;
use tokio::task::JoinHandle;

const TEMP_PROFILE_MARKER: &str = ".mcp_browser_temp_profile";

/// Server cap stays below the host's default request deadline. Overrides can
/// reduce it; callers with a shorter host deadline must configure both sides.
const HARD_CAP_MS: u64 = 25_000;

pub(crate) fn parse_timeout(value: Option<&str>) -> u64 {
    value
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(|v| v.min(HARD_CAP_MS))
        .unwrap_or(HARD_CAP_MS)
}

pub(crate) fn op_timeout_ms() -> u64 {
    parse_timeout(std::env::var("MCP_BROWSER_OP_TIMEOUT_MS").ok().as_deref())
}

/// Reserve time for returning the final human-wait result, even with small caps.
pub(crate) fn human_wait_budget_ms(cap: u64) -> u64 {
    cap.saturating_sub(1_000).min(20_000)
}

/// Reclaim owned scratch profiles after abrupt MCP process exits. A name alone is
/// not ownership evidence, and neither the configured path nor its ancestors may be scanned.
pub fn gc_stale_profiles() {
    let explicit = match profile::explicit_data_dir() {
        Some(dir) => match profile::resolve_directory(&dir) {
            Ok(path) => Some(path),
            Err(_) => return, // Without a resolved explicit path, disjointness is not proven.
        },
        None => None,
    };
    gc_profiles_in(
        &std::env::temp_dir(),
        explicit.as_deref(),
        profile::process_is_dead,
    );
}

fn gc_profiles_in(base: &Path, explicit: Option<&Path>, is_dead: impl Fn(u32) -> bool) {
    let Ok(base) = base.canonicalize() else {
        return;
    };
    if explicit.is_some_and(|dir| profile::path_contains(dir, &base)) {
        return;
    }
    let Ok(entries) = fs::read_dir(&base) else {
        return;
    };
    let me = std::process::id();
    for entry in entries.flatten() {
        let Some(pid) = profile::scratch_pid(&entry.file_name()) else {
            continue;
        };
        let path = entry.path();
        if pid == me
            || !is_dead(pid)
            || explicit.is_some_and(|dir| profile::paths_overlap(dir, &path))
        {
            continue;
        }
        if !owned_temp_profile(&path, pid) {
            continue;
        }
        gc_owned_temp_profile(&path, pid, &is_dead);
    }
}

fn gc_owned_temp_profile(path: &Path, pid: u32, is_dead: impl Fn(u32) -> bool) {
    // Never scan or delete while another process is creating a profile
    // beneath this root. Recheck ownership and markers under the lease.
    let Ok(Some(lease)) = profile::lease_profile(path, true) else {
        return;
    };
    if !is_dead(pid) {
        return;
    }
    if fs::symlink_metadata(path).is_err_and(|err| err.kind() == io::ErrorKind::NotFound) {
        // A competing GC may have removed this tree after our ownership check.
        // The lease holder cleans up even if the creator of the new inode lost try_lock.
        lease.remove_unreserved_scratch_lock(path);
    } else {
        remove_temp_profile(path, pid, &lease);
    }
}

fn mark_temp_profile(dir: &Path, pid: u32) -> io::Result<()> {
    writeln!(
        profile::private_file(&dir.join(TEMP_PROFILE_MARKER))?,
        "{pid}"
    )
}

fn owned_temp_profile(dir: &Path, pid: u32) -> bool {
    if !fs::symlink_metadata(dir).is_ok_and(|metadata| profile::is_private_directory(&metadata)) {
        return false;
    }
    let mut owner = String::new();
    profile::read_regular(&dir.join(TEMP_PROFILE_MARKER))
        .and_then(|marker| marker.take(32).read_to_string(&mut owner))
        .is_ok()
        && owner == format!("{pid}\n")
}

fn removable_temp_profile(dir: &Path, pid: u32) -> bool {
    owned_temp_profile(dir, pid)
        && profile::contains_persistent_marker(dir, true).is_ok_and(|marked| !marked)
}

fn remove_temp_profile(dir: &Path, pid: u32, lease: &ProfileLease) {
    if dir.file_name().and_then(profile::scratch_pid) == Some(pid)
        && removable_temp_profile(dir, pid)
        && fs::remove_dir_all(dir).is_ok()
    {
        lease.remove_scratch_lock(dir);
    }
}

/// Clean up Chrome singleton locks in the profile dir (best-effort).
///
/// `SingletonLock`/`SingletonSocket`/`SingletonCookie` are Chrome's multi-instance
/// locks; if the previous controlled Chrome was killed abnormally (e.g. the MCP client
/// killed the subprocess on timeout), the locks linger and the next launch aborts with
/// `Failed to create ... SingletonLock: File exists (17)`. They are symlinks, and
/// `remove_file` removes the link itself without following it.
fn purge_singleton_locks(dir: &Path, owned: bool) {
    if !owned {
        eprintln!(
            "mcp_browser: preserving singleton locks in user profile {}",
            dir.display()
        );
        return;
    }
    for name in ["SingletonLock", "SingletonSocket", "SingletonCookie"] {
        let _ = std::fs::remove_file(dir.join(name));
    }
}

/// A live browser session: controlled Browser + a single reused Page + the Handler polling task.
pub struct BrowserSession {
    pub browser: Browser,
    pub page: Page,
    handler_task: JoinHandle<()>,
    attach_guard: Option<crate::attach::AttachGuard>,
    pub(crate) owned_pages: Vec<String>,
    pub(crate) snapshot_refs: Vec<(
        String,
        chromiumoxide::cdp::browser_protocol::dom::BackendNodeId,
    )>,
    /// A locally launched browser may be closed whole. An attached browser may not.
    local_profile: Option<LocalProfile>,
    /// Pending-human-action marker: set (with its category) when a human verification
    /// is detected, cleared once wait_for_human confirms resolution, detection is empty,
    /// or navigate reaches a new page. While set, mutating tools (click/type/press_key)
    /// prepend [HUMAN_ACTION_PENDING] to their output so the model stops and hands the
    /// action to the user.
    pub pending_human: Option<String>,
}

/// Top-level session abstraction: controlled Chrome (CDP) vs. driving the user's already-open Chrome (AppleScript).
pub enum Session {
    Cdp(BrowserSession),
    AppleScript(crate::applescript::ApplescriptSession),
}

/// CDP self-launch is the isolated default on every platform. AppleScript and
/// external WebSocket attachment require explicit configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverMode {
    AppleScript,
    Cdp,
}

impl DriverMode {
    pub fn from_env() -> Self {
        Self::parse(
            std::env::var("MCP_BROWSER_DRIVER").ok().as_deref(),
            std::env::var("MCP_BROWSER_WS_URL").ok().as_deref(),
        )
    }

    fn parse(driver: Option<&str>, ws: Option<&str>) -> Self {
        if nonempty_ws_url(ws).is_some() {
            return Self::Cdp;
        }
        if driver.is_some_and(|v| v.eq_ignore_ascii_case("applescript")) {
            Self::AppleScript
        } else {
            Self::Cdp
        }
    }

    pub fn is_applescript(&self) -> bool {
        matches!(self, DriverMode::AppleScript)
    }
}

fn nonempty_ws_url(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub(super) fn configured_ws_url() -> Option<String> {
    nonempty_ws_url(std::env::var("MCP_BROWSER_WS_URL").ok().as_deref()).map(str::to_owned)
}

/// Verify that the configured driver can be used before publishing browser
/// tools. The probe may inspect executables or establish a CDP connection, but
/// it never launches a browser instance or opens a tab.
pub async fn capability_preflight(mode: DriverMode) -> Result<(), String> {
    match mode {
        DriverMode::AppleScript => applescript_capability_preflight().await,
        DriverMode::Cdp => {
            if let Some(configured_url) = configured_ws_url() {
                remote_cdp_capability_preflight(&configured_url).await
            } else {
                match profile::explicit_data_dir() {
                    Some(path) => DedicatedProfile::inspect(&path)?.settings(ProfileUse::Serve)?,
                    None => ProfileSettings::default().with_env()?,
                };
                let chrome = configured_chrome_executable()?;
                local_cdp_capability_preflight(&chrome).await
            }
        }
    }
}

fn configured_chrome_executable() -> Result<PathBuf, String> {
    match std::env::var("MCP_BROWSER_CHROME") {
        Ok(path) if !path.trim().is_empty() => Ok(PathBuf::from(path)),
        Ok(_) => Err("MCP_BROWSER_CHROME is empty".to_string()),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("MCP_BROWSER_CHROME is not valid Unicode".to_string())
        }
        Err(std::env::VarError::NotPresent) => chromiumoxide::detection::default_executable(
            chromiumoxide::detection::DetectionOptions::default(),
        )
        .map_err(|_| {
            "no usable Chrome/Chromium executable was found; set MCP_BROWSER_CHROME".to_string()
        }),
    }
}

async fn local_cdp_capability_preflight(chrome: &Path) -> Result<(), String> {
    chrome_version_preflight(chrome).await?;
    if !headless_from_env() {
        require_gui().map_err(|hint| format!("{hint}; set MCP_BROWSER_HEADLESS=1 on a server"))?;
    }
    Ok(())
}

async fn chrome_version_preflight(chrome: &Path) -> Result<(), String> {
    if !is_executable_file(chrome) {
        return Err(format!(
            "configured Chrome executable is missing or not executable: {}",
            chrome.display()
        ));
    }

    let probe = tokio::process::Command::new(chrome)
        .arg("--version")
        .kill_on_drop(true)
        .output();
    match tokio::time::timeout(std::time::Duration::from_secs(3), probe).await {
        Ok(Ok(output)) if output.status.success() => Ok(()),
        Ok(Ok(_)) => Err(format!(
            "configured Chrome executable could not report its version: {}",
            chrome.display()
        )),
        Ok(Err(_)) => Err(format!(
            "configured Chrome executable could not be started: {}",
            chrome.display()
        )),
        Err(_) => Err(format!(
            "configured Chrome executable did not respond to a version probe: {}",
            chrome.display()
        )),
    }
}

fn require_gui() -> Result<(), &'static str> {
    #[cfg(target_os = "macos")]
    {
        if current_user_has_gui_session() {
            Ok(())
        } else {
            Err("headed CDP driver requires an active macOS GUI login session for the current user")
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let has_display = ["DISPLAY", "WAYLAND_DISPLAY"]
            .into_iter()
            .any(|name| std::env::var(name).is_ok_and(|value| !value.trim().is_empty()));
        if has_display {
            Ok(())
        } else {
            Err("headed CDP driver requires DISPLAY or WAYLAND_DISPLAY")
        }
    }
    #[cfg(not(unix))]
    {
        Ok(())
    }
}

pub(crate) fn headless_from_env() -> bool {
    parse_headless(std::env::var("MCP_BROWSER_HEADLESS").ok().as_deref())
}

fn parse_headless(value: Option<&str>) -> bool {
    !value.is_some_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
}

async fn remote_cdp_capability_preflight(configured_url: &str) -> Result<(), String> {
    let ws = resolve_ws_url(configured_url)
        .await
        .map_err(|err| format!("remote CDP endpoint is unavailable: {err}"))?;
    match tokio::time::timeout(std::time::Duration::from_secs(5), Browser::connect(ws)).await {
        Ok(Ok((_browser, _handler))) => Ok(()),
        Ok(Err(_)) => Err("remote CDP endpoint rejected the browser connection".to_string()),
        Err(_) => Err("remote CDP endpoint did not respond to the browser connection".to_string()),
    }
}

#[cfg(target_os = "macos")]
async fn applescript_capability_preflight() -> Result<(), String> {
    let osascript = Path::new("/usr/bin/osascript");
    if !is_executable_file(osascript) {
        return Err("AppleScript driver requires /usr/bin/osascript".to_string());
    }
    if !current_user_has_gui_session() {
        return Err(
            "AppleScript driver requires an active macOS GUI login session for the current user"
                .to_string(),
        );
    }

    let chrome = find_applescript_chrome()
        .await
        .ok_or_else(|| "AppleScript driver requires Google Chrome to be installed".to_string())?;
    local_cdp_capability_preflight(&chrome).await
}

#[cfg(not(target_os = "macos"))]
async fn applescript_capability_preflight() -> Result<(), String> {
    Err("AppleScript driver is only available on macOS".to_string())
}

#[cfg(target_os = "macos")]
fn current_user_has_gui_session() -> bool {
    use std::os::unix::fs::MetadataExt;

    let Ok(console) = std::fs::metadata("/dev/console") else {
        return false;
    };
    // SAFETY: geteuid has no preconditions and only reads process credentials.
    let current_uid = unsafe { libc::geteuid() };
    current_uid != 0 && console.uid() == current_uid
}

#[cfg(target_os = "macos")]
async fn find_applescript_chrome() -> Option<PathBuf> {
    let mut candidates = vec![PathBuf::from(
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )];
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(
            PathBuf::from(home).join("Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        );
    }
    if let Some(path) = candidates.into_iter().find(|path| is_executable_file(path)) {
        return Some(path);
    }

    let query = "kMDItemCFBundleIdentifier == 'com.google.Chrome'";
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        tokio::process::Command::new("/usr/bin/mdfind")
            .arg(query)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    if !output.status.success() {
        return None;
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|app| PathBuf::from(app).join("Contents/MacOS/Google Chrome"))
        .find(|path| is_executable_file(path))
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

// chromiumoxide 0.9.1 browser/config.rs DEFAULT_ARGS, except for the two flags
// that substitute the platform's password store and keychain. Its builder still
// owns the browser mode, sandbox, user-data-dir, and debugging port in both modes.
const NATIVE_DEFAULT_ARGS: &[(&str, &[&str])] = &[
    ("disable-background-networking", &[]),
    (
        "enable-features",
        &["NetworkService", "NetworkServiceInProcess"],
    ),
    ("disable-background-timer-throttling", &[]),
    ("disable-backgrounding-occluded-windows", &[]),
    ("disable-breakpad", &[]),
    ("disable-client-side-phishing-detection", &[]),
    ("disable-component-extensions-with-background-pages", &[]),
    ("disable-default-apps", &[]),
    ("disable-dev-shm-usage", &[]),
    ("disable-features", &["TranslateUI"]),
    ("disable-hang-monitor", &[]),
    ("disable-ipc-flooding-protection", &[]),
    ("disable-popup-blocking", &[]),
    ("disable-prompt-on-repost", &[]),
    ("disable-renderer-backgrounding", &[]),
    ("disable-sync", &[]),
    ("force-color-profile", &["srgb"]),
    ("metrics-recording-only", &[]),
    ("no-first-run", &[]),
    ("enable-automation", &[]),
    ("enable-blink-features", &["IdleDetection"]),
    ("lang", &["en_US"]),
];

fn chrome_config(
    chrome: &Path,
    data_dir: &Path,
    settings: &ProfileSettings,
    headless: bool,
    interactive: bool,
) -> Result<BrowserConfig, String> {
    chrome_builder(chrome, data_dir, settings, headless, interactive)
        .build()
        .map_err(|_| "cannot build the Chrome launch configuration".to_string())
}

fn chrome_builder(
    chrome: &Path,
    data_dir: &Path,
    settings: &ProfileSettings,
    headless: bool,
    interactive: bool,
) -> BrowserConfigBuilder {
    let mut builder = BrowserConfig::builder()
        .chrome_executable(chrome)
        .user_data_dir(data_dir)
        .request_timeout(Duration::from_millis(op_timeout_ms()));
    if settings.keychain == Keychain::Native {
        builder = builder
            .disable_default_args()
            .args(NATIVE_DEFAULT_ARGS.iter().copied());
    }
    if let Some(directory) = &settings.directory {
        builder = builder.arg(("profile-directory", directory.as_str()));
    }
    if interactive {
        builder = builder.arg("no-startup-window").viewport(None);
    }
    if headless {
        builder.new_headless_mode()
    } else {
        builder.with_head()
    }
}

struct LocalProfile {
    data_dir: PathBuf,
    temporary: bool,
    leases: Vec<ProfileLease>,
}

impl LocalProfile {
    fn for_server(explicit: Option<PathBuf>) -> Result<(Self, ProfileSettings), String> {
        if let Some(path) = explicit {
            return Self::persistent(DedicatedProfile::inspect(&path)?, ProfileUse::Serve);
        }
        let settings = ProfileSettings::default().with_env()?;
        let base = profile::resolve_directory(&std::env::temp_dir())?;
        let pid = std::process::id();
        let data_dir = base.join(format!("{TEMP_PROFILE_PREFIX}{pid}"));
        let leases = profile::lease_profile_tree(&data_dir)?;
        match profile::private_directory(&data_dir, false) {
            Ok(()) => {
                if mark_temp_profile(&data_dir, pid).is_err() {
                    let _ = fs::remove_file(data_dir.join(TEMP_PROFILE_MARKER));
                    if fs::remove_dir(&data_dir).is_ok()
                        && let Some(lease) = leases.last()
                    {
                        lease.remove_scratch_lock(&data_dir);
                    }
                    return Err("cannot mark the Chrome temporary profile as owned".to_string());
                }
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                if !removable_temp_profile(&data_dir, pid) {
                    return Err("the Chrome temporary profile path already exists and is not a safe scratch directory".to_string());
                }
            }
            Err(_) => {
                return Err("cannot create the Chrome temporary profile directory".to_string());
            }
        }
        purge_singleton_locks(&data_dir, true);
        Ok((
            Self {
                data_dir,
                temporary: true,
                leases,
            },
            settings,
        ))
    }

    fn persistent(
        dedicated: DedicatedProfile,
        usage: ProfileUse,
    ) -> Result<(Self, ProfileSettings), String> {
        let (data_dir, leases, settings) = dedicated.prepare(usage)?;
        Ok((
            Self {
                data_dir,
                temporary: false,
                leases,
            },
            settings,
        ))
    }

    async fn close(&self, browser: &mut Browser) -> Result<(), CloseError> {
        const GRACE: Duration = Duration::from_secs(5);
        let _ = tokio::time::timeout(GRACE, browser.close()).await;
        // A CDP acknowledgement is not process exit: wait for persistent data to
        // flush, and do not remove a scratch tree while Chrome is still writing it.
        let exit = confirm_browser_exit(tokio::time::timeout(GRACE, browser.wait()).await.ok());
        if self.temporary && matches!(exit, Err(CloseError::TimedOut | CloseError::Unknown)) {
            let _ = tokio::time::timeout(GRACE, browser.kill()).await;
        }
        if exit.is_ok() && !self.temporary {
            profile::clear_closed_debug_port(&self.data_dir, browser.websocket_address());
        }
        exit
    }
}

#[derive(Debug, PartialEq, Eq)]
enum CloseError {
    TimedOut,
    Unsuccessful(ExitStatus),
    Unknown,
}

impl std::fmt::Display for CloseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TimedOut => write!(
                f,
                "Chrome shutdown exceeded its deadline; profile writes were not confirmed"
            ),
            Self::Unsuccessful(status) => write!(
                f,
                "Chrome did not exit cleanly ({status}); profile writes were not confirmed"
            ),
            Self::Unknown => write!(
                f,
                "Chrome exit could not be confirmed; profile writes were not confirmed"
            ),
        }
    }
}

fn confirm_browser_exit(wait: Option<io::Result<Option<ExitStatus>>>) -> Result<(), CloseError> {
    match wait {
        Some(Ok(Some(status))) if status.success() => Ok(()),
        Some(Ok(Some(status))) => Err(CloseError::Unsuccessful(status)),
        None => Err(CloseError::TimedOut),
        _ => Err(CloseError::Unknown),
    }
}

fn poll_handler(mut handler: Handler) -> JoinHandle<()> {
    tokio::spawn(async move { while handler.next().await.is_some() {} })
}

pub(super) async fn login(destination: &Path, url: &str) -> Result<(), String> {
    use chromiumoxide::cdp::browser_protocol::target::CreateTargetParams;

    let dedicated = DedicatedProfile::inspect(destination)?;
    dedicated.require_idle()?;
    require_gui()
        .map_err(|hint| format!("{hint}; run --login on a desktop or use --sync-profile"))?;
    let chrome = configured_chrome_executable()?;
    chrome_version_preflight(&chrome).await?;
    let interrupt = crate::login_interrupt::LoginInterrupt::install()?;
    let (local, settings) = LocalProfile::persistent(dedicated, ProfileUse::Login)?;
    let builder = chrome_builder(&chrome, &local.data_dir, &settings, false, true);
    #[cfg(unix)]
    let builder = crate::login_interrupt::isolated_chrome(builder, &chrome)?;
    let config = builder
        .build()
        .map_err(|_| "cannot build the Chrome login configuration")?;
    let (mut browser, handler) = Browser::launch(config).await
        .map_err(|_| "cannot start the dedicated Chrome window; check the desktop session and close any agent browser using this profile")?;
    let handler_task = poll_handler(handler);
    let target = CreateTargetParams::builder()
        .url(url)
        .new_window(true)
        .build()
        .expect("initial URL supplied");
    let opened = tokio::select! {
        biased;
        _ = interrupt.wait() => None,
        page = browser.new_page(target) => Some(page),
    };
    match opened {
        None => {
            let result = local
                .close(&mut browser)
                .await
                .map_err(|err| err.to_string());
            handler_task.abort();
            return result;
        }
        Some(Err(_)) => {
            let _ = local.close(&mut browser).await;
            handler_task.abort();
            return Err("cannot open the login page in the dedicated Chrome window".to_string());
        }
        Some(Ok(_)) => {}
    }
    eprintln!(
        "Log in in the separate Chrome window, then return to this terminal and press Enter. Your everyday browser window is not used."
    );
    let (send, receive) = tokio::sync::oneshot::channel();
    // Tokio stdin's blocking reader would keep the runtime alive after Chrome exits.
    let reader = std::thread::Builder::new().spawn(move || {
        let mut line = String::new();
        let entered = std::io::stdin().read_line(&mut line).is_ok() && line.ends_with('\n');
        let _ = send.send(entered);
    });
    let result = if reader.is_err() {
        local
            .close(&mut browser)
            .await
            .map_err(|err| err.to_string())
            .and(Err("cannot start the terminal input reader".to_string()))
    } else {
        tokio::select! {
            biased;
            _ = interrupt.wait() => local.close(&mut browser).await.map_err(|err| err.to_string()),
            status = browser.wait() => {
                confirm_browser_exit(Some(status)).map_err(|err| err.to_string()).map(|()| {
                    profile::clear_closed_debug_port(&local.data_dir, browser.websocket_address());
                })
            }
            entered = receive => {
                local.close(&mut browser).await.map_err(|err| err.to_string()).and_then(|()| {
                    if entered == Ok(true) { Ok(()) } else {
                        Err("terminal input ended before Enter; the dedicated Chrome window was closed".into())
                    }
                })
            }
        }
    };
    handler_task.abort();
    result
}

impl BrowserSession {
    /// Lazily launch a controlled Chrome and open a blank page.
    ///
    /// Environment variables:
    /// - `MCP_BROWSER_WS_URL`: if set, attach an existing instance instead (`Browser::connect`).
    /// - `MCP_BROWSER_CHROME`: explicit Chrome executable path (otherwise auto-detected per OS).
    /// - `MCP_BROWSER_HEADLESS`: `1` (default) headless; `0` explicitly opens a visible window.
    /// - `MCP_BROWSER_USER_DATA_DIR`: explicit profile dir (persists login state); not
    ///   cleaned up, and sharing it across processes causes conflicts. When unset, each
    ///   process gets a unique temp dir that is removed on exit.
    ///
    /// The whole launch is bounded by the per-operation cap ([`op_timeout_ms`]):
    /// callers invoke it outside `with_timeout` (tools.rs wraps only the per-tool
    /// work), so without this outer bound a hung launch could outlive the host's
    /// request_timeout_ms and get the subprocess killed mid-launch.
    pub async fn launch() -> Result<Self, String> {
        let cap_ms = op_timeout_ms();
        match tokio::time::timeout(
            std::time::Duration::from_millis(cap_ms),
            Self::launch_inner(),
        )
        .await
        {
            Ok(Ok(session)) => Ok(session),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(format!(
                "browser session launch exceeded the {cap_ms} ms server operation cap"
            )),
        }
    }

    /// The actual launch work, bounded by [`launch`]'s per-operation cap.
    async fn launch_inner() -> Result<Self, String> {
        let (mut browser, handler, local_profile, attach_guard) =
            if let Some(ws) = configured_ws_url() {
                let ws = resolve_ws_url(&ws)
                    .await
                    .map_err(|e| format!("failed to resolve MCP_BROWSER_WS_URL ({ws}): {e}"))?;
                let guard = crate::attach::AttachGuard::connect(&ws).await?;
                let (browser, handler) = Browser::connect_with_config(
                    guard.url.clone(),
                    chromiumoxide::handler::HandlerConfig {
                        request_timeout: std::time::Duration::from_millis(op_timeout_ms()),
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| format!("failed to connect to browser at MCP_BROWSER_WS_URL: {e}"))?;
                (browser, handler, None, Some(guard))
            } else {
                let chrome = configured_chrome_executable()?;
                let (local, settings) = LocalProfile::for_server(profile::explicit_data_dir())?;
                let config = chrome_config(
                    &chrome,
                    &local.data_dir,
                    &settings,
                    headless_from_env(),
                    false,
                )?;
                let (browser, handler) = Browser::launch(config).await.map_err(|e| {
                    format!("failed to launch Chrome at '{}': {e}", chrome.display())
                })?;
                (browser, handler, Some(local), None)
            };
        let handler_task = poll_handler(handler);

        let page = match browser.new_page("about:blank").await {
            Ok(page) => page,
            Err(e) => {
                // Do not leak a half-started session on failure: in controlled mode
                // close the whole instance so the Chrome process exits (an orphan
                // would keep holding the per-pid profile dir and collide with the
                // next launch, which reuses that same dir); in attach mode dropping
                // `browser` closes the connection. Always stop the polling task.
                if let Some(local) = &local_profile {
                    let _ = local.close(&mut browser).await;
                }
                handler_task.abort();
                return Err(format!("failed to open initial page: {e}"));
            }
        };

        Ok(BrowserSession {
            owned_pages: vec![page.target_id().as_ref().to_string()],
            snapshot_refs: Vec::new(),
            browser,
            page,
            handler_task,
            attach_guard,
            local_profile,
            pending_human: None,
        })
    }

    pub(crate) fn may_close_tab(&self, id: &str) -> bool {
        self.local_profile.is_some() || self.owned_pages.iter().any(|owned| owned == id)
    }

    /// Close the session and stop the Handler polling task. Best-effort.
    ///
    /// Key protection: in attach mode (MCP_BROWSER_WS_URL, i.e. the user's own Chrome)
    /// never `close` the whole browser (that would quit the user's browser); only close
    /// the session tab we created. Only a self-launched controlled instance is closed whole.
    pub async fn shutdown(mut self) -> Result<&'static str, String> {
        let result = if let Some(local) = &self.local_profile {
            local
                .close(&mut self.browser)
                .await
                .map(|()| "Browser closed")
                .map_err(|err| err.to_string())
        } else {
            // Attach mode: never close the whole browser (that would quit the
            // user's Chrome); only close the session tab we created.
            let mut confirmed = true;
            for id in &self.owned_pages {
                let closed = tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    self.browser.execute(
                        chromiumoxide::cdp::browser_protocol::target::CloseTargetParams::new(
                            id.clone(),
                        ),
                    ),
                )
                .await;
                confirmed &= matches!(closed, Ok(Ok(_)));
            }
            if confirmed {
                Ok("Controlled tabs closed; the attached browser was left running")
            } else {
                Err("could not confirm closure of all controlled tabs; the attached browser was left running".into())
            }
        };
        self.handler_task.abort();
        self.attach_guard.take();
        if let Some(local) = self.local_profile.as_ref().filter(|local| local.temporary)
            && self.browser.try_wait().is_ok_and(|status| status.is_some())
            && let Some(lease) = local.leases.last()
        {
            remove_temp_profile(&local.data_dir, std::process::id(), lease);
        }
        result
    }
}

/// Normalize MCP_BROWSER_WS_URL into the ws:// address chromiumoxide needs:
/// - `ws://` / `wss://` returned as-is;
/// - `http://host:port` or bare `host:port` automatically fetches `/json/version` for
///   `webSocketDebuggerUrl`, avoiding manual copying of a long ws address (used with
///   `--remote-debugging-port` on Windows/Linux, equivalent to "reusing the user's
///   already-open Chrome").
async fn resolve_ws_url(input: &str) -> Result<String, String> {
    let input = input.trim();
    if input.starts_with("ws://") || input.starts_with("wss://") {
        return Ok(input.to_string());
    }
    let (host, port) = {
        let base = input
            .strip_prefix("http://")
            .or_else(|| input.strip_prefix("https://"))
            .unwrap_or(input);
        // Take only the "host:port" part, ignoring any /devtools/... path the user may have pasted.
        let hp = base.split('/').next().unwrap_or(base);
        match hp
            .strip_prefix('[')
            .and_then(|h| h.strip_suffix(']'))
            .unwrap_or(hp)
            .rsplit_once(':')
        {
            Some((h, p))
                if !h.is_empty() && !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) =>
            {
                (h.to_string(), p.to_string())
            }
            _ => {
                return Err(format!(
                    "无法解析主机端口（应为 host:port 或 http://host:port）: {input}"
                ));
            }
        }
    };
    let resp = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        http_get(&host, &port, "/json/version"),
    )
    .await
    .map_err(|_| {
        format!("连接 {host}:{port} 超时——Chrome 是否以 --remote-debugging-port={port} 启动？")
    })??;
    let (status, body) = resp;
    if !status.starts_with("200") {
        return Err(format!("{host}:{port}/json/version 返回 {status}"));
    }
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("解析调试端点响应失败: {e}"))?;
    let ws = v["webSocketDebuggerUrl"]
        .as_str()
        .ok_or_else(|| "调试端点响应里没有 webSocketDebuggerUrl".to_string())?;
    Ok(ws.to_string())
}

/// Minimal HTTP GET (only for local debugging endpoints like 127.0.0.1; zero new dependencies).
async fn http_get(host: &str, port: &str, path: &str) -> Result<(String, String), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let port: u16 = port.parse().map_err(|_| format!("端口不合法: {port}"))?;
    let mut s = tokio::net::TcpStream::connect((host, port))
        .await
        .map_err(|e| format!("连接 {host}:{port} 失败: {e}"))?;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nUser-Agent: mcp_browser\r\n\r\n"
    );
    s.write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).await.map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&buf).to_string();
    let status = text.split_whitespace().nth(1).unwrap_or("").to_string();
    let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    Ok((status, body))
}

/// Lazily launch a session if none exists, then return a mutable reference to it.
pub async fn ensure_session(
    session: &mut Option<BrowserSession>,
) -> Result<&mut BrowserSession, String> {
    if session.is_none() {
        *session = Some(BrowserSession::launch().await?);
    }
    Ok(session.as_mut().expect("session just initialized"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_fixes_mode_and_timeout_defaults() {
        assert_eq!(DriverMode::parse(None, None), DriverMode::Cdp);
        assert_eq!(DriverMode::parse(Some("unknown"), None), DriverMode::Cdp);
        assert_eq!(
            DriverMode::parse(Some("AppleScript"), None),
            DriverMode::AppleScript
        );
        assert_eq!(
            DriverMode::parse(Some("applescript"), Some("ws://example")),
            DriverMode::Cdp
        );
        assert!(parse_headless(None));
        assert!(!parse_headless(Some("0")));
        assert!(!parse_headless(Some("FALSE")));
        for value in [None, Some("0"), Some("-1"), Some("junk"), Some("90000")] {
            assert_eq!(parse_timeout(value), 25_000);
        }
        assert_eq!(parse_timeout(Some("1000")), 1000);
        for cap in [1, 500, 1000, 5000, 25_000] {
            assert!(human_wait_budget_ms(cap) < cap);
        }
    }

    #[test]
    fn blank_websocket_does_not_select_attach() {
        for ws in [None, Some(""), Some(" \t\r\n")] {
            assert_eq!(nonempty_ws_url(ws), None);
            assert_eq!(DriverMode::parse(None, ws), DriverMode::Cdp);
            assert_eq!(
                DriverMode::parse(Some("applescript"), ws),
                DriverMode::AppleScript
            );
        }
        assert_eq!(
            nonempty_ws_url(Some(" ws://localhost:9222 ")),
            Some("ws://localhost:9222")
        );
    }

    #[test]
    fn gc_keeps_marked_and_configured_profiles() {
        let dir = profile::tests::TestDir::new();
        let make_dir = |pid| {
            let path = dir.0.join(format!("{TEMP_PROFILE_PREFIX}{pid}"));
            profile::private_directory(&path, false).unwrap();
            mark_temp_profile(&path, pid).unwrap();
            path
        };
        let marked = make_dir(4_294_967_290);
        let explicit_ancestor = make_dir(4_294_967_291);
        let stale = make_dir(4_294_967_292);
        let live = make_dir(4_294_967_293);
        let unrelated = dir.0.join(format!("{TEMP_PROFILE_PREFIX}2147483644"));
        profile::private_directory(&unrelated, false).unwrap();
        let explicit = explicit_ancestor.join("nested/dedicated");
        profile::private_directory(&explicit, true).unwrap();
        // Even a corrupt marker nested elsewhere protects the scratch candidate.
        profile::private_directory(&marked.join("nested"), false).unwrap();
        fs::write(
            marked.join("nested").join(profile::MARKER),
            "incomplete marker",
        )
        .unwrap();
        #[cfg(unix)]
        let explicit = {
            use std::os::unix::fs::symlink;
            let alias = dir.0.join("alias");
            symlink(&explicit_ancestor, &alias).unwrap();
            let resolved = profile::resolve_directory(&alias.join("nested/dedicated")).unwrap();
            assert_eq!(resolved, explicit);
            resolved
        };
        gc_profiles_in(&dir.0, Some(&explicit), |pid| pid != 4_294_967_293);
        assert!(marked.exists() && explicit.exists() && live.exists() && unrelated.exists());
        assert!(!stale.exists());
    }

    #[test]
    fn gc_preserves_profile_tree_reservation_if_the_observed_scratch_disappears() {
        let base = profile::tests::TestDir::new();
        let pid = 2_147_483_644;
        let name = format!("{TEMP_PROFILE_PREFIX}{pid}");
        let scratch = base.0.join(&name);
        let lock = base.0.join(format!(".{name}.mcp_browser.lock"));
        let dead = |candidate| candidate == pid;
        profile::private_directory(&scratch, false).unwrap();
        mark_temp_profile(&scratch, pid).unwrap();
        assert!(owned_temp_profile(&scratch, pid));

        // A second GC reclaims the scratch after the first one observed ownership.
        gc_profiles_in(&base.0, None, dead);
        assert!(!scratch.exists() && !lock.exists());
        // Resume the first GC at lease acquisition without repeating its preflight.
        gc_owned_temp_profile(&scratch, pid, dead);
        assert!(!lock.exists(), "GC leaked the new lease for a missing tree");

        // A persistent setup can claim a GC-created lock without initializing the directory.
        drop(profile::lease_profile(&scratch, true).unwrap().unwrap());
        let leases = profile::lease_profile_tree(&scratch).unwrap();
        assert!(!scratch.exists());
        gc_owned_temp_profile(&scratch, pid, dead);
        assert!(lock.is_file(), "GC removed an active lease");
        drop(leases);
        gc_owned_temp_profile(&scratch, pid, dead);
        assert!(lock.is_file(), "GC removed a reserved profile-tree lease");
    }

    #[test]
    fn gc_cleans_orphan_lock_when_creator_loses_try_lock() {
        let base = profile::tests::TestDir::new();
        let pid = 2_147_483_644;
        let name = format!("{TEMP_PROFILE_PREFIX}{pid}");
        let scratch = base.0.join(&name);
        let lock = base.0.join(format!(".{name}.mcp_browser.lock"));
        profile::private_directory(&scratch, false).unwrap();
        mark_temp_profile(&scratch, pid).unwrap();
        assert!(owned_temp_profile(&scratch, pid));
        gc_profiles_in(&base.0, None, |candidate| candidate == pid);
        assert!(!scratch.exists() && !lock.exists());

        // One delayed GC creates the lock after reclamation but pauses before try_lock.
        let creator = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&lock)
            .unwrap();
        let creator_lost = std::cell::Cell::new(false);
        // The other delayed GC reaches the PID recheck only after acquiring that lock.
        gc_owned_temp_profile(&scratch, pid, |candidate| {
            assert_eq!(candidate, pid);
            assert!(matches!(creator.try_lock(), Err(fs::TryLockError::WouldBlock)));
            creator_lost.set(true);
            true
        });
        assert!(creator_lost.get());
        drop(creator);
        assert!(!scratch.exists());
        assert!(!lock.exists(), "an orphan lock remains after its creator lost try_lock");
    }

    #[test]
    fn gc_waits_for_a_nested_initialization_lease() {
        use std::sync::Barrier;

        let base = profile::tests::TestDir::new();
        let pid = 2_147_483_644;
        let scratch = base.0.join(format!("{TEMP_PROFILE_PREFIX}{pid}"));
        profile::private_directory(&scratch, false).unwrap();
        mark_temp_profile(&scratch, pid).unwrap();
        let barrier = Barrier::new(2);
        let kept = std::thread::scope(|scope| {
            scope.spawn(|| {
                let leases = profile::lease_profile_tree(&scratch.join("nested/agent")).unwrap();
                barrier.wait();
                barrier.wait();
                drop(leases);
            });
            barrier.wait();
            gc_profiles_in(&base.0, None, |candidate| candidate == pid);
            let kept = scratch.is_dir();
            barrier.wait();
            kept
        });
        assert!(kept);
        gc_profiles_in(&base.0, None, |candidate| candidate == pid);
        assert!(!scratch.exists());
    }

    #[cfg(unix)]
    #[test]
    fn shutdown_reports_only_confirmed_clean_exit_as_success() {
        use std::os::unix::process::ExitStatusExt;

        assert_eq!(confirm_browser_exit(None), Err(CloseError::TimedOut));
        assert_eq!(
            confirm_browser_exit(Some(Ok(None))),
            Err(CloseError::Unknown)
        );
        assert_eq!(
            confirm_browser_exit(Some(Err(io::Error::other("fixture")))),
            Err(CloseError::Unknown)
        );
        assert_eq!(
            confirm_browser_exit(Some(Ok(Some(ExitStatus::from_raw(0))))),
            Ok(())
        );
        let unsuccessful = ExitStatus::from_raw(37 << 8);
        assert_eq!(
            confirm_browser_exit(Some(Ok(Some(unsuccessful)))),
            Err(CloseError::Unsuccessful(unsuccessful))
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn chrome_argv_changes_only_keychain_flags_for_native_mode() {
        use std::os::unix::fs::PermissionsExt;
        use tokio::io::AsyncReadExt;

        let dir = profile::tests::TestDir::new();
        let exe = dir.0.join("print-argv");
        std::fs::write(&exe, "#!/bin/sh\nprintf '%s\\n' \"$@\" >&2\n").unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o700)).unwrap();
        let launch = |settings: ProfileSettings, headless, interactive| {
            chrome_config(&exe, &dir.0, &settings, headless, interactive)
                .unwrap()
                .launch()
                .unwrap()
        };
        async fn argv(mut child: chromiumoxide::async_process::Child) -> Vec<String> {
            let mut text = String::new();
            child
                .stderr
                .take()
                .unwrap()
                .into_inner()
                .read_to_string(&mut text)
                .await
                .unwrap();
            assert!(child.wait().await.unwrap().success());
            let mut args = text.lines().map(str::to_owned).collect::<Vec<_>>();
            args.sort_unstable();
            args
        }
        let mock = argv(launch(ProfileSettings::default(), true, false)).await;
        let native = argv(launch(
            ProfileSettings {
                keychain: Keychain::Native,
                ..Default::default()
            },
            true,
            false,
        ))
        .await;
        assert!(mock.iter().any(|arg| arg == "--password-store=basic"));
        assert!(mock.iter().any(|arg| arg == "--use-mock-keychain"));
        assert_eq!(
            mock.iter()
                .filter(|arg| !matches!(
                    arg.as_str(),
                    "--password-store=basic" | "--use-mock-keychain"
                ))
                .cloned()
                .collect::<Vec<_>>(),
            native
        );
        for args in [&mock, &native] {
            for prefix in [
                "--user-data-dir=",
                "--headless=new",
                "--remote-debugging-port=",
            ] {
                assert_eq!(
                    args.iter().filter(|arg| arg.starts_with(prefix)).count(),
                    1,
                    "{prefix}"
                );
            }
            assert!(
                !args
                    .iter()
                    .any(|arg| arg.starts_with("--profile-directory"))
            );
        }
        let login = argv(launch(
            ProfileSettings {
                directory: Some("Profile 3".to_string()),
                keychain: Keychain::Native,
            },
            false,
            true,
        ))
        .await;
        assert!(!login.iter().any(|arg| arg.starts_with("--headless")));
        assert!(
            login
                .iter()
                .any(|arg| arg == "--profile-directory=Profile 3")
        );
        assert!(login.iter().any(|arg| arg == "--no-startup-window"));
    }

    #[test]
    fn browser_fixes_preserves_user_singleton_files() {
        let dir =
            std::env::temp_dir().join(format!("mcp-browser-lock-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["SingletonLock", "SingletonSocket", "SingletonCookie"] {
            std::fs::write(dir.join(name), "user lock").unwrap();
        }
        purge_singleton_locks(&dir, false);
        assert!(dir.join("SingletonLock").exists());
        assert!(dir.join("SingletonSocket").exists());
        assert!(dir.join("SingletonCookie").exists());
        purge_singleton_locks(&dir, true);
        assert!(!dir.join("SingletonLock").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
