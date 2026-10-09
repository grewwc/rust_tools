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

/// Handle `/cd [dir]` / `:cd [dir]`.
///
/// Bare `/cd` prints the directory tools currently resolve against; `-` goes
/// back to the previous one. Returns `true` when the input was a `/cd` command
/// (including a switch that failed), `false` otherwise.
pub fn try_handle_cwd_command(input: &str) -> bool {
    let trimmed = input.trim();
    let Some(rest) = trimmed
        .strip_prefix('/')
        .or_else(|| trimmed.strip_prefix(':'))
    else {
        return false;
    };
    if rest.split_whitespace().next() != Some("cd") {
        return false;
    }
    // Take the path verbatim (not word-by-word) so directory names containing
    // spaces work; the prefix check above guarantees `rest` starts with "cd".
    let arg = rest.trim_start()[2..].trim();
    if arg.is_empty() {
        print_current_dir();
    } else if arg == "-" {
        return_to_previous_dir();
    } else {
        change_dir(arg);
    }
    true
}

fn print_current_dir() {
    match runtime_ctx::effective_cwd() {
        Ok(dir) => println!("Working directory: {}", dir.display()),
        Err(err) => eprintln!("[cd] cannot resolve the working directory: {err}"),
    }
}

fn print_usage() {
    println!("Usage: /cd [dir]   (no argument shows the current directory, \"-\" goes back)");
}

/// Expand `~`, then move the process to the resolved directory.
fn change_dir(target: &str) {
    let expanded = expanduser(target);
    apply_dir(Path::new(expanded.as_ref()));
}

/// Move the process to `dir` and remember the old directory for `/cd -`.
fn apply_dir(dir: &Path) {
    if !dir.is_dir() {
        eprintln!("[cd] not a directory: {}", dir.display());
        print_usage();
        return;
    }
    let previous = match std::env::current_dir() {
        Ok(previous) => previous,
        Err(err) => {
            eprintln!("[cd] cannot read the current working directory: {err}");
            return;
        }
    };
    if let Err(err) = std::env::set_current_dir(dir) {
        eprintln!("[cd] cannot switch to {}: {err}", dir.display());
        print_usage();
        return;
    }
    if let Ok(mut guard) = PREVIOUS_CWD.lock() {
        *guard = Some(previous.clone());
    }
    // Read the directory back from the OS so the printed path is the one tools
    // will resolve against (absolute, symlinks resolved by getcwd).
    let now = runtime_ctx::effective_cwd().unwrap_or_else(|_| dir.to_path_buf());
    println!(
        "Working directory: {} (was {})",
        now.display(),
        previous.display()
    );
}

fn return_to_previous_dir() {
    let previous = PREVIOUS_CWD.lock().ok().and_then(|guard| guard.clone());
    match previous {
        Some(dir) => apply_dir(&dir),
        None => println!("No previous working directory recorded yet."),
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
