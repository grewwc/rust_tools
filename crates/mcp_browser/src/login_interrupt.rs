//! Interrupt handling for the standalone login command only.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
const CHILD_ARG: &str = "mcp-browser-login-child";
#[cfg(unix)]
const CHILD_EXE: &str = "MCP_BROWSER_INTERNAL_LOGIN_EXECUTABLE";

#[cfg(unix)]
pub(super) fn isolated_chrome(
    builder: chromiumoxide::browser::BrowserConfigBuilder,
    chrome: &std::path::Path,
) -> Result<chromiumoxide::browser::BrowserConfigBuilder, String> {
    // Chromiumoxide owns the child handle and does not expose a pre-exec hook.
    // Re-exec through this binary so the Chrome group can differ from the
    // terminal's foreground group without interposing another live process.
    let launcher =
        std::env::current_exe().map_err(|_| "cannot resolve the Chrome login launcher")?;
    let chrome = std::path::absolute(chrome).map_err(|_| "cannot resolve the Chrome executable")?;
    let chrome = chrome
        .to_str()
        .ok_or("the Chrome executable path is not valid Unicode")?;
    Ok(builder
        .chrome_executable(launcher)
        .arg(CHILD_ARG)
        .env(CHILD_EXE, chrome))
}

#[cfg(unix)]
pub(super) fn run_chrome_child() -> Option<std::process::ExitCode> {
    use std::os::unix::process::CommandExt;

    let chrome = std::env::var_os(CHILD_EXE)?;
    let flag = format!("--{CHILD_ARG}");
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !args.iter().any(|arg| arg == flag.as_str()) {
        return None;
    }
    // SAFETY: setpgid with zero arguments affects only this process. Chrome
    // must not receive the terminal's Ctrl-C alongside the login command.
    if unsafe { libc::setpgid(0, 0) } != 0 {
        eprintln!("mcp_browser: cannot isolate the Chrome login process group");
        return Some(std::process::ExitCode::FAILURE);
    }
    let _ = std::process::Command::new(chrome)
        .args(args.into_iter().filter(|arg| arg != flag.as_str()))
        .env_remove(CHILD_EXE)
        .exec();
    eprintln!("mcp_browser: cannot start the isolated Chrome executable");
    Some(std::process::ExitCode::FAILURE)
}

pub(super) struct LoginInterrupt {
    #[cfg(unix)]
    previous: Vec<(libc::c_int, libc::sigaction)>,
}

impl LoginInterrupt {
    pub(super) fn install() -> Result<Self, String> {
        INTERRUPTED.store(false, Ordering::Relaxed);
        #[cfg(unix)]
        {
            let mut guard = Self {
                previous: Vec::new(),
            };
            for signal in [libc::SIGINT, libc::SIGTERM] {
                // SAFETY: the action and old-action buffer are valid for sigaction;
                // the handler only sets a lock-free flag.
                let previous = unsafe {
                    let mut action: libc::sigaction = std::mem::zeroed();
                    action.sa_sigaction = on_signal as *const () as libc::sighandler_t;
                    action.sa_flags = libc::SA_RESTART;
                    libc::sigemptyset(&mut action.sa_mask);
                    let mut previous = std::mem::MaybeUninit::uninit();
                    if libc::sigaction(signal, &action, previous.as_mut_ptr()) != 0 {
                        return Err("cannot install a Chrome login signal handler".into());
                    }
                    previous.assume_init()
                };
                guard.previous.push((signal, previous));
            }
            Ok(guard)
        }
        #[cfg(windows)]
        {
            // SAFETY: on_console_event has the console-handler ABI and static lifetime.
            if unsafe { SetConsoleCtrlHandler(on_console_event, 1) } == 0 {
                return Err("cannot install the Chrome login console handler".into());
            }
            Ok(Self {})
        }
        #[cfg(not(any(unix, windows)))]
        Ok(Self {})
    }

    pub(super) async fn wait(&self) {
        while !INTERRUPTED.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

impl Drop for LoginInterrupt {
    fn drop(&mut self) {
        #[cfg(unix)]
        for (signal, action) in &self.previous {
            // SAFETY: both values were saved by a successful sigaction call.
            unsafe { libc::sigaction(*signal, action, std::ptr::null_mut()) };
        }
        #[cfg(windows)]
        // SAFETY: unregister the same static handler installed above.
        unsafe {
            SetConsoleCtrlHandler(on_console_event, 0);
        }
    }
}

#[cfg(unix)]
extern "C" fn on_signal(_signal: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetConsoleCtrlHandler(handler: extern "system" fn(u32) -> i32, add: i32) -> i32;
}

#[cfg(windows)]
extern "system" fn on_console_event(event: u32) -> i32 {
    if matches!(event, 0 | 1) {
        // CTRL_C_EVENT and CTRL_BREAK_EVENT.
        INTERRUPTED.store(true, Ordering::Relaxed);
        1
    } else {
        0
    }
}
