//! `/cd` — move the session to another working directory without restarting.
//!
//! The session working directory *is* the process working directory: file
//! tools, instruction discovery, sandbox write roots and the next turn's
//! system prompt all resolve paths through `runtime_ctx::effective_cwd()`
//! (which falls back to `std::env::current_dir()`), and `execute_command`
//! children inherit it. Switching therefore only has to move the process —
//! nothing else is captured at startup, so the change is live from this turn's
//! output on.
//!
//! Sub-agents spawned with `inherit.cwd == false` stay isolated: their
//! `SUBAGENT_CWD` scratch scope wins over the process directory, so `/cd`
//! never moves them.

use std::{
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
};

use crate::ai::driver::runtime_ctx;
use crate::commonw::utils::expanduser;

/// Destination of the shell-like `/cd -` form: the directory the session was in
/// before the last successful `/cd`. Session-scoped in the strictest sense — it
/// lives in the process and disappears when `a` exits.
static PREVIOUS_CWD: LazyLock<Mutex<Option<PathBuf>>> = LazyLock::new(|| Mutex::new(None));

/// A parsed `/cd`/`:cd` command, before any directory state is touched.
/// `parse_cwd_command` produces it; the local process switch and serve's
/// per-session switch (`serve/session_cwd.rs`) both consume it, so the two
/// paths interpret the same grammar and produce the same message text.
pub(crate) enum CwdCommand {
    /// Bare `/cd`: print the current working directory.
    Show,
    /// `/cd -`: return to the previously recorded directory.
    GoBack,
    /// `/cd <dir>`: switch to `<dir>`, kept verbatim so directory names
    /// containing spaces survive.
    Switch(String),
}

/// Parse a `/cd [dir]` / `:cd [dir]` line without touching any state.
/// Returns `None` when the line is not a `cd` command (e.g. `/model foo`).
pub(crate) fn parse_cwd_command(input: &str) -> Option<CwdCommand> {
    let trimmed = input.trim();
    let Some(rest) = trimmed
        .strip_prefix('/')
        .or_else(|| trimmed.strip_prefix(':'))
    else {
        return None;
    };
    if rest.split_whitespace().next() != Some("cd") {
        return None;
    }
    // Take the path verbatim (not word-by-word) so directory names containing
    // spaces work; the prefix check above guarantees `rest` starts with "cd".
    let arg = rest.trim_start()[2..].trim();
    if arg.is_empty() {
        Some(CwdCommand::Show)
    } else if arg == "-" {
        Some(CwdCommand::GoBack)
    } else {
        Some(CwdCommand::Switch(arg.to_string()))
    }
}

/// Handle `/cd [dir]` / `:cd [dir]`.
///
/// Bare `/cd` prints the directory tools currently resolve against; `-` goes
/// back to the previous one. Returns `true` when the input was a `/cd` command
/// (including a switch that failed), `false` otherwise.
pub fn try_handle_cwd_command(input: &str) -> bool {
    let Some(outcome) = run_cwd_command(input) else {
        return false;
    };
    if !outcome.stdout.is_empty() {
        print!("{}", outcome.stdout);
    }
    if !outcome.stderr.is_empty() {
        eprint!("{}", outcome.stderr);
    }
    true
}

/// Message lines a `/cd` switch produces, kept separate from the process's
/// stdout/stderr so a caller can return them over a channel instead of
/// printing into the caller's own terminal.
pub(crate) struct CwdOutcome {
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

/// Execute a `/cd [dir]` / `:cd [dir]` command against this process and return
/// the message lines it would print, without printing anything.
///
/// Serve mode uses this: the session working directory is the serve process's
/// working directory (turn children inherit it at spawn), so the switch must
/// happen in the serve process, but its output must come back over HTTP
/// instead of leaking into the server's terminal. Returns `None` when `input`
/// is not a `/cd` command.
pub(crate) fn run_cwd_command(input: &str) -> Option<CwdOutcome> {
    let outcome = match parse_cwd_command(input)? {
        CwdCommand::Show => current_dir_outcome(),
        CwdCommand::GoBack => return_to_previous_dir_outcome(),
        CwdCommand::Switch(arg) => change_dir_outcome(&arg),
    };
    Some(outcome)
}

fn current_dir_outcome() -> CwdOutcome {
    match runtime_ctx::effective_cwd() {
        Ok(dir) => CwdOutcome {
            stdout: format!("Working directory: {}\n", dir.display()),
            stderr: String::new(),
        },
        Err(err) => CwdOutcome {
            stdout: String::new(),
            stderr: format!("[cd] cannot resolve the working directory: {err}\n"),
        },
    }
}

/// Usage line shared by the local and per-session switch error messages.
pub(crate) fn cd_usage_line() -> String {
    "Usage: /cd [dir]   (no argument shows the current directory, \"-\" goes back)\n".to_string()
}

/// Expand `~`, then move the process to the resolved directory.
fn change_dir_outcome(target: &str) -> CwdOutcome {
    let expanded = expanduser(target);
    apply_dir_outcome(Path::new(expanded.as_ref()))
}

/// Move the process to `dir` and remember the old directory for `/cd -`.
fn apply_dir_outcome(dir: &Path) -> CwdOutcome {
    if !dir.is_dir() {
        return CwdOutcome {
            stdout: String::new(),
            stderr: format!("[cd] not a directory: {}\n{}", dir.display(), cd_usage_line()),
        };
    }
    let previous = match std::env::current_dir() {
        Ok(previous) => previous,
        Err(err) => return CwdOutcome {
            stdout: String::new(),
            stderr: format!("[cd] cannot read the current working directory: {err}\n"),
        },
    };
    if let Err(err) = std::env::set_current_dir(dir) {
        return CwdOutcome {
            stdout: String::new(),
            stderr: format!(
                "[cd] cannot switch to {}: {err}\n{}",
                dir.display(),
                cd_usage_line()
            ),
        };
    }
    if let Ok(mut guard) = PREVIOUS_CWD.lock() {
        *guard = Some(previous.clone());
    }
    // Read the directory back from the OS so the printed path is the one tools
    // will resolve against (absolute, symlinks resolved by getcwd).
    let now = runtime_ctx::effective_cwd().unwrap_or_else(|_| dir.to_path_buf());
    CwdOutcome {
        stdout: format!(
            "Working directory: {} (was {})\n",
            now.display(),
            previous.display()
        ),
        stderr: String::new(),
    }
}

fn return_to_previous_dir_outcome() -> CwdOutcome {
    let previous = PREVIOUS_CWD.lock().ok().and_then(|guard| guard.clone());
    match previous {
        Some(dir) => apply_dir_outcome(&dir),
        None => CwdOutcome {
            stdout: "No previous working directory recorded yet.\n".to_string(),
            stderr: String::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// Test that performs the real directory switch; it is only meaningful when
    /// re-executed with `CHILD_MARKER_ENV` set.
    const CHILD_SWITCH_TEST: &str =
        "ai::driver::commands::cwd::tests::directory_switch_moves_the_process";
    /// Marks the re-executed child run of [`CHILD_SWITCH_TEST`].
    const CHILD_MARKER_ENV: &str = "A_CD_CHILD_SWITCH";
    /// Test that exercises the printer-free `run_cwd_command` in a re-executed
    /// child (same isolation rationale as [`CHILD_SWITCH_TEST`]).
    const RUN_CWD_COMMAND_TEST: &str =
        "ai::driver::commands::cwd::tests::run_cwd_command_captures_outcome_in_a_child";
    /// Marks the re-executed child run of [`RUN_CWD_COMMAND_TEST`].
    const RUN_CWD_COMMAND_MARKER_ENV: &str = "A_CD_RUN_CWD_COMMAND";

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("a-cd-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `current_dir()` returns the physical path while `temp_dir()` keeps the
    /// `TMPDIR` spelling (`/var/...` vs `/private/var/...` on macOS), so both
    /// sides are canonicalized before comparing.
    fn cwd() -> PathBuf {
        std::fs::canonicalize(std::env::current_dir().unwrap()).unwrap()
    }

    #[test]
    fn ignores_other_commands() {
        assert!(!try_handle_cwd_command(""));
        assert!(!try_handle_cwd_command("/model foo"));
        assert!(!try_handle_cwd_command("cd /tmp"));
        assert!(!try_handle_cwd_command("/cdx /tmp"));
    }

    #[test]
    fn reporting_the_directory_never_moves_it() {
        let before = cwd();
        assert!(try_handle_cwd_command("/cd"));
        assert!(try_handle_cwd_command(":cd"));
        assert_eq!(cwd(), before);
    }

    #[test]
    fn switching_happens_in_an_isolated_process() -> std::io::Result<()> {
        // `set_current_dir` is process-wide, and the harness runs tests on
        // threads: switching here would move the directory under unrelated
        // tests that read the process cwd. Re-exec the same binary so the
        // switch stays contained.
        let child = Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                CHILD_SWITCH_TEST,
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_MARKER_ENV, "1")
            .output()?;
        assert!(
            child.status.success(),
            "isolated cwd switch failed:\n{}{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        Ok(())
    }

    #[test]
    fn run_cwd_command_captures_outcome_without_printing() -> std::io::Result<()> {
        let child = Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                RUN_CWD_COMMAND_TEST,
                "--nocapture",
                "--test-threads=1",
            ])
            .env(RUN_CWD_COMMAND_MARKER_ENV, "1")
            .output()?;
        assert!(
            child.status.success(),
            "isolated run_cwd_command failed:\n{}{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        Ok(())
    }

    /// Child-process body of
    /// [`run_cwd_command_captures_outcome_without_printing`]; a plain no-op
    /// when the harness runs it directly.
    #[test]
    fn run_cwd_command_captures_outcome_in_a_child() {
        if std::env::var_os(RUN_CWD_COMMAND_MARKER_ENV).is_none() {
            return;
        }
        // Non-`/cd` input: no outcome.
        assert!(run_cwd_command("plain prompt").is_none());
        assert!(run_cwd_command("/cdx /tmp").is_none());

        // The outcome carries the switch message instead of printing it; the
        // process cwd still moves exactly like the printing form.
        let start = temp_dir("outcome-start");
        std::env::set_current_dir(&start).unwrap();
        let target = temp_dir("outcome-target");
        let outcome = run_cwd_command(&format!("/cd {}", target.display())).expect("cd outcome");
        assert!(
            outcome
                .stdout
                .contains(&std::fs::canonicalize(&target).unwrap().display().to_string()),
            "stdout: {}",
            outcome.stdout
        );
        assert!(outcome.stderr.is_empty());
        assert_eq!(cwd(), std::fs::canonicalize(&target).unwrap());

        // A rejected target reports through stderr and leaves the process
        // where it was.
        let outcome =
            run_cwd_command("/cd /definitely/not/a/real/directory").expect("cd outcome");
        assert!(outcome.stderr.contains("not a directory"));
        assert!(outcome.stdout.is_empty());
        assert_eq!(cwd(), std::fs::canonicalize(&target).unwrap());
    }

    /// Child-process body of [`switching_happens_in_an_isolated_process`]; a
    /// plain no-op when the harness runs it directly.
    #[test]
    fn directory_switch_moves_the_process() {
        if std::env::var_os(CHILD_MARKER_ENV).is_none() {
            return;
        }
        let start = temp_dir("switch-start");
        std::env::set_current_dir(&start).unwrap();
        let target = temp_dir("switch-target");

        assert!(try_handle_cwd_command(&format!("/cd {}", target.display())));
        assert_eq!(cwd(), std::fs::canonicalize(&target).unwrap());

        assert!(try_handle_cwd_command(":cd -"));
        assert_eq!(cwd(), std::fs::canonicalize(&start).unwrap());

        // A relative target resolves against the directory the session is in.
        let nested = start.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        assert!(try_handle_cwd_command("/cd nested"));
        assert_eq!(cwd(), std::fs::canonicalize(&nested).unwrap());

        // `~` expands through the same helper the command tools use.
        if let Some(home) = std::env::var_os("HOME") {
            assert!(try_handle_cwd_command("/cd ~"));
            assert_eq!(cwd(), std::fs::canonicalize(Path::new(&home)).unwrap());
        }

        // A rejected target never moves the process.
        let before = cwd();
        assert!(try_handle_cwd_command(
            "/cd /definitely/not/a/real/directory"
        ));
        assert_eq!(cwd(), before);
    }
}
