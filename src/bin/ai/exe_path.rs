//! Resolution of the executable path used whenever the runtime re-execs
//! itself.
//!
//! Detached helpers, `-bg` daemon children, terminal hosts/workers and every
//! serve turn start a fresh `a` from `std::env::current_exe()`. On Linux that
//! path comes from `/proc/self/exe`, and the kernel appends `" (deleted)"`
//! once the running file has been unlinked - which is what a rebuild does when
//! it replaces `bin/a` with a new inode while older processes keep running.
//! The suffixed name never exists on disk, so a blind exec fails with `ENOENT`
//! even though the install path holds a valid binary again; [`runtime_exe`]
//! maps the suffix back to that live file and fails with an actionable message
//! when the file is really gone.
//!
//! Callers that only want the location for display or for a directory lookup
//! keep using `std::env::current_exe()` directly.

use std::io;
use std::path::{Path, PathBuf};

/// Suffix the Linux kernel appends to `/proc/self/exe` for an unlinked file.
const DELETED_SUFFIX: &str = " (deleted)";

/// Path to exec for a re-executed helper or turn child.
pub(in crate::ai) fn runtime_exe() -> io::Result<PathBuf> {
    resolve_executable(std::env::current_exe()?)
}

/// [`runtime_exe`] without the syscall, so tests can drive the suffix logic.
///
/// The path as reported wins while it exists: a file name may itself end with
/// the marker, and only a missing file can be a deleted-binary path.
fn resolve_executable(exe: PathBuf) -> io::Result<PathBuf> {
    if exe.is_file() {
        return Ok(exe);
    }
    let Some(markerless) = strip_deleted_suffix(&exe) else {
        // Not a deleted-binary path: keep the previous behavior and let the
        // exec report whatever is wrong with it.
        return Ok(exe);
    };
    if markerless.is_file() {
        return Ok(markerless);
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "{} was deleted or replaced and no executable exists at that path; \
             restart this process (a serve daemon: `a --serve-restart`)",
            markerless.display()
        ),
    ))
}

/// Drop the `" (deleted)"` marker that `/proc/self/exe` carries for an
/// unlinked file. Byte-level on unix, so install paths that are not valid
/// UTF-8 still recover; elsewhere the kernel never appends the marker, but the
/// same shape is honored.
#[cfg(unix)]
fn strip_deleted_suffix(path: &Path) -> Option<PathBuf> {
    use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};

    let bytes = path.as_os_str().as_bytes();
    let stem = bytes.strip_suffix(DELETED_SUFFIX.as_bytes())?;
    (!stem.is_empty()).then(|| PathBuf::from(std::ffi::OsString::from_vec(stem.to_vec())))
}

#[cfg(not(unix))]
fn strip_deleted_suffix(path: &Path) -> Option<PathBuf> {
    let stem = path.to_str()?.strip_suffix(DELETED_SUFFIX)?;
    (!stem.is_empty()).then(|| PathBuf::from(stem))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("a-exe-path-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn deleted_suffix_resolves_to_the_live_file() {
        let dir = scratch_dir("live");
        let live = dir.join("probe");
        std::fs::write(&live, b"#!/bin/true\n").expect("write probe");
        let resolved = resolve_executable(dir.join("probe (deleted)")).expect("resolve");
        assert_eq!(resolved, live);
    }

    #[test]
    fn deleted_suffix_without_a_live_file_names_the_way_out() {
        let dir = scratch_dir("gone");
        let err = resolve_executable(dir.join("gone (deleted)")).expect_err("must fail");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("--serve-restart"), "unexpected: {err}");
    }

    #[test]
    fn unsuffixed_paths_pass_through_untouched() {
        let plain = PathBuf::from("/nonexistent/plain");
        assert_eq!(resolve_executable(plain.clone()).expect("resolve"), plain);
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_paths_pass_through_untouched() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let raw = PathBuf::from(OsStr::from_bytes(b"/nonexistent/\xff"));
        assert_eq!(resolve_executable(raw.clone()).expect("resolve"), raw);
    }

    #[test]
    fn an_existing_name_ending_in_the_marker_wins() {
        let dir = scratch_dir("literal");
        let literal = dir.join("probe (deleted)");
        std::fs::write(&literal, b"#!/bin/true\n").expect("write literal");
        assert_eq!(resolve_executable(literal.clone()).expect("resolve"), literal);
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_deleted_path_still_recovers() {
        use std::ffi::OsString;
        use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};

        let mut raw = scratch_dir("non-utf8").into_os_string().into_vec();
        raw.extend_from_slice(b"/pro\xffbe");
        let live = PathBuf::from(OsString::from_vec(raw.clone()));
        std::fs::write(&live, b"#!/bin/true\n").expect("write live");
        raw.extend_from_slice(DELETED_SUFFIX.as_bytes());
        let suffixed = PathBuf::from(OsString::from_vec(raw));
        assert_eq!(resolve_executable(suffixed).expect("resolve"), live);
    }
}