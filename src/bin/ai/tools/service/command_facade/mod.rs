//! Internal command fallbacks behind `execute_command`.
//!
//! `execute_command` is a facade: a simple command whose program is not
//! installed can be satisfied by the agent itself, without the local shell.
//! The model sees the normal tool result either way. The fallback only runs
//! when the real program is absent from `PATH`, so an installed binary keeps
//! its full CLI semantics, and it never runs for commands with pipes,
//! redirections, substitutions, or interactive PTY/stdin requests.
//!
//! Current handlers: `zip` and `unzip`, implemented with the `zip` crate
//! (deflate through flate2).

mod unzip_impl;
mod zip_impl;

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::audit::tokenize_shell_words;

/// Cap for raw bytes accumulated by handlers such as `unzip -p`; the command
/// formatter truncates the rendered text further.
const MAX_INTERNAL_OUTPUT_BYTES: usize = 1024 * 1024;
/// Buffer size for streaming file contents into or out of an archive.
const COPY_BUFFER_BYTES: usize = 64 * 1024;

/// Dispatch a command to an internal handler when the named program is
/// missing. `None` means "run it through the normal shell path".
pub(crate) fn try_run(
    command: &str,
    cwd: Option<&str>,
    timeout_secs: u64,
) -> Option<Result<String, String>> {
    if !is_simple_command(command) {
        return None;
    }
    let tokens = tokenize_shell_words(command);
    let program = tokens.first()?;
    let basename = Path::new(program).file_name().and_then(|name| name.to_str())?;
    let kind = match basename {
        "zip" => CommandKind::Zip,
        "unzip" => CommandKind::Unzip,
        _ => return None,
    };
    // An explicit path is the caller's choice; only a bare program name may be
    // satisfied internally.
    if program.contains('/') || program.contains('\\') {
        return None;
    }
    let base = resolve_base_dir(cwd)?;
    if program_available(program) {
        return None;
    }
    let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1));
    Some(match run_internal(kind, &tokens[1..], &base, deadline) {
        InternalOutcome::Success(output) => Ok(render_success(output)),
        InternalOutcome::Failed {
            code,
            stdout,
            stderr,
        } => Ok(render_failure(code, &stdout, &stderr)),
        InternalOutcome::TimedOut => Err(format!(
            "Command timed out after {timeout_secs}s and was terminated."
        )),
    })
}

#[derive(Clone, Copy)]
enum CommandKind {
    Zip,
    Unzip,
}

fn run_internal(
    kind: CommandKind,
    args: &[String],
    base: &Path,
    deadline: Instant,
) -> InternalOutcome {
    match kind {
        CommandKind::Zip => zip_impl::run(args, base, deadline),
        CommandKind::Unzip => unzip_impl::run(args, base, deadline),
    }
}

#[derive(Debug)]
enum InternalOutcome {
    Success(String),
    Failed {
        code: i32,
        stdout: String,
        stderr: String,
    },
    TimedOut,
}

fn failure(message: impl Into<String>) -> InternalOutcome {
    InternalOutcome::Failed {
        code: 1,
        stdout: String::new(),
        stderr: message.into(),
    }
}

#[derive(Debug)]
enum CopyError {
    Io(io::Error),
    TimedOut,
}

/// A command the shell would run as one plain program invocation: no pipes,
/// redirections, substitutions, globbing, subshells, or comments. Quotes and
/// backslash escapes are allowed because the tokenizer understands them.
fn is_simple_command(command: &str) -> bool {
    let bytes = command.as_bytes();
    let mut index = 0;
    let mut in_single = false;
    let mut in_double = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_single {
            if byte == b'\'' {
                in_single = false;
            }
            index += 1;
            continue;
        }
        if in_double {
            match byte {
                b'\\' | b'`' | b'$' => return false,
                b'"' => in_double = false,
                _ => {}
            }
            index += 1;
            continue;
        }
        match byte {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            // Skip the escaped byte; a trailing backslash simply ends the scan.
            b'\\' => index += 1,
            b'|' | b'&' | b';' | b'<' | b'>' | b'$' | b'`' | b'*' | b'?' | b'(' | b')'
            | b'[' | b']' | b'{' | b'}' | b'\n' | b'#' => return false,
            _ => {}
        }
        index += 1;
    }
    !in_single && !in_double
}

fn resolve_base_dir(cwd: Option<&str>) -> Option<PathBuf> {
    match cwd {
        Some(dir) => {
            let path = Path::new(dir);
            if path.is_absolute() {
                Some(path.to_path_buf())
            } else {
                crate::ai::driver::runtime_ctx::effective_cwd()
                    .ok()
                    .map(|base| base.join(path))
            }
        }
        None => crate::ai::driver::runtime_ctx::effective_cwd().ok(),
    }
}

fn resolve_path(base: &Path, raw: &str) -> PathBuf {
    let expanded = crate::commonw::utils::expanduser(raw);
    let path = Path::new(expanded.as_ref());
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn program_available(program: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut names = vec![program.to_string()];
    #[cfg(windows)]
    names.push(format!("{program}.exe"));
    std::env::split_paths(&path).any(|dir| {
        names
            .iter()
            .any(|name| is_executable_file(&dir.join(name)))
    })
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
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

fn deadline_passed(deadline: Instant) -> bool {
    Instant::now() >= deadline
}

fn copy_with_deadline<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    deadline: Instant,
) -> Result<u64, CopyError> {
    let mut buffer = [0u8; COPY_BUFFER_BYTES];
    let mut total = 0u64;
    loop {
        if deadline_passed(deadline) {
            return Err(CopyError::TimedOut);
        }
        let read = reader.read(&mut buffer).map_err(CopyError::Io)?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read]).map_err(CopyError::Io)?;
        total += read as u64;
    }
    Ok(total)
}

#[cfg(unix)]
fn unix_mode(metadata: &fs::Metadata, default_mode: u32) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    let mode = metadata.permissions().mode() & 0o777;
    if mode == 0 { default_mode } else { mode }
}

#[cfg(not(unix))]
fn unix_mode(_metadata: &fs::Metadata, default_mode: u32) -> u32 {
    default_mode
}

#[cfg(unix)]
fn apply_permissions(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode & 0o777));
}

#[cfg(not(unix))]
fn apply_permissions(_path: &Path, _mode: u32) {}

fn temp_path_for(archive: &Path) -> PathBuf {
    let file_name = archive
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("archive.zip");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    archive.with_file_name(format!(
        ".{file_name}.tmp-{}-{nonce}",
        std::process::id()
    ))
}

fn replace_file(temp: &Path, target: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        if target.exists() {
            fs::remove_file(target)?;
        }
    }
    fs::rename(temp, target)
}

/// Shell-style wildcard match (`*` any run, `?` one character). Patterns are
/// short, so the straightforward dynamic program is enough.
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let mut matched = vec![false; text.len() + 1];
    matched[0] = true;
    for &token in &pattern {
        if token == '*' {
            for index in 1..=text.len() {
                matched[index] = matched[index] || matched[index - 1];
            }
        } else {
            for index in (1..=text.len()).rev() {
                matched[index] = matched[index - 1] && (token == '?' || token == text[index - 1]);
            }
            matched[0] = false;
        }
    }
    matched[text.len()]
}

/// A member pattern selects the named entry and, when the pattern names a
/// directory, everything below it.
fn member_matches(pattern: &str, name: &str) -> bool {
    wildcard_match(pattern, name)
        || name
            .strip_prefix(pattern)
            .is_some_and(|rest| rest.starts_with('/'))
        || (pattern.ends_with('/') && name.starts_with(pattern))
}

fn render_success(output: String) -> String {
    if output.trim().is_empty() {
        "(command succeeded with exit code 0 and produced no output)".to_string()
    } else {
        super::command::truncate_chars(&output, super::command::MAX_COMMAND_OUTPUT_CHARS)
    }
}

fn render_failure(code: i32, stdout: &str, stderr: &str) -> String {
    super::command::truncate_chars(
        &format!("Exit code: {code}\n{}\n{}", stdout.trim(), stderr.trim()),
        super::command::MAX_COMMAND_OUTPUT_CHARS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn temp_dir(label: &str) -> PathBuf {
        let counter = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rust_tools_command_facade_{label}_{}_{counter}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    pub(super) fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }

    #[test]
    fn rejects_commands_the_shell_must_interpret() {
        for command in [
            "unzip archive.zip | head",
            "unzip archive.zip > out.txt",
            "unzip $(echo archive.zip)",
            "unzip archive*.zip",
            "unzip [ab].zip",
            "zip -r out.zip {a,b}",
            "unzip archive.zip && echo done",
            "unzip archive.zip # comment",
        ] {
            assert!(
                !is_simple_command(command),
                "should not intercept shell-interpreted command: {command}"
            );
        }
        assert!(is_simple_command("unzip 'my archive.zip' -d out"));
        assert!(is_simple_command(r"unzip my\ archive.zip"));
    }

    #[test]
    fn wildcard_matching_covers_directory_patterns() {
        assert!(wildcard_match("*.txt", "notes.txt"));
        assert!(!wildcard_match("*.txt", "notes.md"));
        assert!(member_matches("dir", "dir/file.txt"));
        assert!(member_matches("dir/", "dir/file.txt"));
        assert!(member_matches("dir/*", "dir/file.txt"));
        assert!(!member_matches("dir", "directory/file.txt"));
    }

    #[test]
    fn unknown_program_is_not_on_path() {
        assert!(!program_available(
            "rust-tools-command-facade-definitely-missing"
        ));
    }

    #[test]
    fn dispatch_uses_internal_zip_when_binary_is_absent() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = temp_dir("dispatch_missing_zip");
        fs::write(dir.join("a.txt"), b"payload").unwrap();
        let empty_path = dir.join("empty-bin");
        fs::create_dir_all(&empty_path).unwrap();
        let old_path = std::env::var_os("PATH");
        unsafe { std::env::set_var("PATH", &empty_path) };
        let command = "zip -r bundle.zip a.txt".to_string();
        let result = try_run(&command, Some(dir.to_str().unwrap()), 30);
        match old_path {
            Some(value) => unsafe { std::env::set_var("PATH", value) },
            None => unsafe { std::env::remove_var("PATH") },
        }
        let rendered = result
            .expect("missing zip should dispatch to the internal handler")
            .expect("internal zip should succeed");
        assert!(rendered.contains("adding: a.txt"), "{rendered}");
        assert!(dir.join("bundle.zip").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn zip_and_unzip_round_trip() {
        let dir = temp_dir("round_trip");
        fs::create_dir_all(dir.join("src/nested")).unwrap();
        fs::write(dir.join("src/a.txt"), b"hello").unwrap();
        fs::write(dir.join("src/nested/b.txt"), b"world").unwrap();
        let archive = dir.join("bundle.zip");
        let zip_args = vec![
            "-r".to_string(),
            archive.to_string_lossy().into_owned(),
            "src".to_string(),
        ];
        let outcome = zip_impl::run(&zip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");

        let out_dir = dir.join("out");
        fs::create_dir_all(&out_dir).unwrap();
        let unzip_args = vec![
            archive.to_string_lossy().into_owned(),
            "-d".to_string(),
            out_dir.to_string_lossy().into_owned(),
        ];
        let outcome = unzip_impl::run(&unzip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");
        assert_eq!(fs::read(out_dir.join("src/a.txt")).unwrap(), b"hello");
        assert_eq!(fs::read(out_dir.join("src/nested/b.txt")).unwrap(), b"world");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn zip_dot_input_uses_relative_entries_and_skips_its_own_archive() {
        let dir = temp_dir("dot_input");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/a.txt"), b"hello").unwrap();
        let archive = dir.join("bundle.zip");
        let zip_args = vec![
            "-r".to_string(),
            archive.to_string_lossy().into_owned(),
            ".".to_string(),
        ];
        // Run twice so the update pass walks a directory that already holds
        // the archive; the archive itself must not become an entry.
        for _ in 0..2 {
            let outcome = zip_impl::run(&zip_args, &dir, deadline());
            assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");
        }

        let file = fs::File::open(&archive).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|index| archive.by_index(index).unwrap().name().to_string())
            .collect();
        assert!(names.iter().any(|name| name == "src/a.txt"), "{names:?}");
        assert!(!names.iter().any(|name| name == "bundle.zip"), "{names:?}");
        assert!(
            !names.iter().any(|name| name.starts_with("Users/")),
            "{names:?}"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn zip_keeps_an_explicit_non_zip_extension() {
        let dir = temp_dir("explicit_extension");
        fs::write(dir.join("a.txt"), b"payload").unwrap();
        let archive = dir.join("backup.tar");
        let zip_args = vec![
            archive.to_string_lossy().into_owned(),
            "a.txt".to_string(),
        ];
        let outcome = zip_impl::run(&zip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");
        assert!(archive.exists());
        assert!(!dir.join("backup.tar.zip").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn zip_excludes_patterns() {
        let dir = temp_dir("exclude");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/keep.txt"), b"keep").unwrap();
        fs::write(dir.join("src/skip.tmp"), b"skip").unwrap();
        let archive = dir.join("bundle.zip");
        let zip_args = vec![
            "-r".to_string(),
            archive.to_string_lossy().into_owned(),
            "src".to_string(),
            "-x".to_string(),
            "*.tmp".to_string(),
        ];
        let outcome = zip_impl::run(&zip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");

        let out_dir = dir.join("out");
        fs::create_dir_all(&out_dir).unwrap();
        let unzip_args = vec![
            archive.to_string_lossy().into_owned(),
            "-d".to_string(),
            out_dir.to_string_lossy().into_owned(),
        ];
        let outcome = unzip_impl::run(&unzip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");
        assert!(out_dir.join("src/keep.txt").exists());
        assert!(!out_dir.join("src/skip.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn zip_follows_symlinked_directories() {
        use std::os::unix::fs::symlink;

        let dir = temp_dir("symlink_dir");
        fs::create_dir_all(dir.join("real")).unwrap();
        fs::write(dir.join("real/file.txt"), b"content").unwrap();
        symlink(dir.join("real"), dir.join("link")).unwrap();
        let archive = dir.join("bundle.zip");
        let zip_args = vec![
            "-r".to_string(),
            archive.to_string_lossy().into_owned(),
            "link".to_string(),
        ];
        let outcome = zip_impl::run(&zip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");

        let out_dir = dir.join("out");
        fs::create_dir_all(&out_dir).unwrap();
        let unzip_args = vec![
            archive.to_string_lossy().into_owned(),
            "-d".to_string(),
            out_dir.to_string_lossy().into_owned(),
        ];
        let outcome = unzip_impl::run(&unzip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");
        assert_eq!(
            fs::read(out_dir.join("link/file.txt")).unwrap(),
            b"content"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unzip_refuses_to_overwrite_without_a_flag() {
        let dir = temp_dir("overwrite");
        fs::write(dir.join("a.txt"), b"first").unwrap();
        let archive = dir.join("bundle.zip");
        let zip_args = vec![
            archive.to_string_lossy().into_owned(),
            "a.txt".to_string(),
        ];
        assert!(matches!(
            zip_impl::run(&zip_args, &dir, deadline()),
            InternalOutcome::Success(_)
        ));

        let out_dir = dir.join("out");
        fs::create_dir_all(&out_dir).unwrap();
        fs::write(out_dir.join("a.txt"), b"old").unwrap();
        let unzip_args = vec![
            archive.to_string_lossy().into_owned(),
            "-d".to_string(),
            out_dir.to_string_lossy().into_owned(),
        ];
        let outcome = unzip_impl::run(&unzip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Failed { .. }), "{outcome:?}");
        assert_eq!(fs::read(out_dir.join("a.txt")).unwrap(), b"old");

        let mut overwrite_args = unzip_args.clone();
        overwrite_args.insert(1, "-o".to_string());
        let outcome = unzip_impl::run(&overwrite_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");
        assert_eq!(fs::read(out_dir.join("a.txt")).unwrap(), b"first");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unzip_skips_path_traversal_entries() {
        use std::fs::File;
        use std::io::Write;
        use zip::write::SimpleFileOptions;

        let dir = temp_dir("traversal");
        let archive = dir.join("evil.zip");
        let file = File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        writer
            .start_file("../escape.txt", SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"escaped").unwrap();
        writer.finish().unwrap();

        let out_dir = dir.join("out");
        fs::create_dir_all(&out_dir).unwrap();
        let unzip_args = vec![
            archive.to_string_lossy().into_owned(),
            "-d".to_string(),
            out_dir.to_string_lossy().into_owned(),
        ];
        let outcome = unzip_impl::run(&unzip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");
        assert!(!dir.join("escape.txt").exists());
        assert!(!out_dir.join("../escape.txt").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn unzip_does_not_follow_pre_existing_symlinks() {
        use std::fs::File;
        use std::io::Write;
        use std::os::unix::fs::symlink;
        use zip::write::SimpleFileOptions;

        let dir = temp_dir("unzip_symlink");
        let outside = dir.join("outside");
        fs::create_dir_all(&outside).unwrap();
        let out_dir = dir.join("out");
        fs::create_dir_all(&out_dir).unwrap();
        symlink(&outside, out_dir.join("link")).unwrap();
        symlink(outside.join("target.txt"), out_dir.join("file.txt")).unwrap();

        let archive = dir.join("bundle.zip");
        let file = File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        writer
            .start_file("link/escape.txt", SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"escaped").unwrap();
        writer
            .start_file("file.txt", SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"overwritten").unwrap();
        writer.finish().unwrap();

        let unzip_args = vec![
            "-o".to_string(),
            archive.to_string_lossy().into_owned(),
            "-d".to_string(),
            out_dir.to_string_lossy().into_owned(),
        ];
        let outcome = unzip_impl::run(&unzip_args, &dir, deadline());
        assert!(matches!(outcome, InternalOutcome::Success(_)), "{outcome:?}");
        assert!(!outside.join("escape.txt").exists());
        assert!(!outside.join("target.txt").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unzip_list_and_pipe_modes() {
        let dir = temp_dir("list_pipe");
        fs::write(dir.join("a.txt"), b"payload").unwrap();
        let archive = dir.join("bundle.zip");
        let zip_args = vec![
            archive.to_string_lossy().into_owned(),
            "a.txt".to_string(),
        ];
        assert!(matches!(
            zip_impl::run(&zip_args, &dir, deadline()),
            InternalOutcome::Success(_)
        ));

        let list_args = vec!["-l".to_string(), archive.to_string_lossy().into_owned()];
        match unzip_impl::run(&list_args, &dir, deadline()) {
            InternalOutcome::Success(output) => assert!(output.contains("a.txt")),
            other => panic!("list failed: {other:?}"),
        }

        let pipe_args = vec!["-p".to_string(), archive.to_string_lossy().into_owned()];
        match unzip_impl::run(&pipe_args, &dir, deadline()) {
            InternalOutcome::Success(output) => assert_eq!(output, "payload"),
            other => panic!("pipe failed: {other:?}"),
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
