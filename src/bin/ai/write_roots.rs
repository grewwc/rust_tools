use std::path::{Path, PathBuf};

use crate::ai::history::SessionStore;

pub(in crate::ai) const MAX_ADDITIONAL_WRITE_ROOTS: usize = 64;
pub(in crate::ai) const MAX_WRITE_ROOTS_BYTES: usize = 64 * 1024;
pub(in crate::ai) const AGENT_COMMAND_ENV: &str = "AI_AGENT_COMMAND";

pub(in crate::ai) fn require_user_grant_origin() -> Result<(), String> {
    if std::env::var_os(AGENT_COMMAND_ENV).is_some() {
        return Err("directory grants are user commands; --add-dir, /dirs add, and /dirs del are unavailable inside execute_command subprocesses".to_string());
    }
    Ok(())
}

pub(in crate::ai) fn validate_root_limits(roots: &[PathBuf]) -> Result<(), String> {
    if roots.len() > MAX_ADDITIONAL_WRITE_ROOTS {
        return Err(format!(
            "writable directory limit exceeded (maximum {MAX_ADDITIONAL_WRITE_ROOTS})"
        ));
    }
    let bytes = serde_json::to_vec(roots)
        .map_err(|error| error.to_string())?
        .len();
    if bytes > MAX_WRITE_ROOTS_BYTES {
        return Err(format!(
            "writable directory list exceeds {MAX_WRITE_ROOTS_BYTES} bytes"
        ));
    }
    Ok(())
}

/// Reads retain a bounded prefix and surface the skipped suffix instead of granting it.
pub(in crate::ai) fn bounded_roots(roots: &[PathBuf]) -> (Vec<PathBuf>, Vec<String>) {
    let mut bounded = Vec::new();
    for root in roots {
        if bounded.contains(root) {
            continue;
        }
        bounded.push(root.clone());
        if let Err(error) = validate_root_limits(&bounded) {
            bounded.pop();
            return (
                bounded,
                vec![format!(
                    "Truncated writable directory list: {error}; remaining roots were ignored"
                )],
            );
        }
    }
    (bounded, Vec::new())
}

/// Resolve a user directory without creating it; stored grants use physical absolute paths.
pub(in crate::ai) fn resolve_directory(directory: &Path, cwd: &Path) -> Result<PathBuf, String> {
    if directory.as_os_str().is_empty() {
        return Err("requires a non-empty directory path".to_string());
    }
    let expanded = directory
        .to_str()
        .map(|value| PathBuf::from(crate::commonw::utils::expanduser(value).as_ref()))
        .unwrap_or_else(|| directory.to_path_buf());
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    };
    let root = std::fs::canonicalize(&absolute).map_err(|error| {
        format!(
            "'{}' must be an existing accessible directory: {error}",
            directory.display()
        )
    })?;
    if !root.is_dir() {
        return Err(format!("'{}' is not a directory", directory.display()));
    }
    if root.to_str().is_none() {
        return Err(
            "directory paths must be valid UTF-8 so session grants can be persisted".to_string(),
        );
    }
    Ok(root)
}

/// Parsed `/dirs` command family. `/add-dir <DIR>` remains an alias for adding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) enum DirsCommand {
    Add(String),
    List,
    Delete(String),
}

/// A directory command consumes the complete message, never a following model task.
pub(in crate::ai) fn parse_dirs_command(input: &str) -> Option<Result<DirsCommand, String>> {
    const USAGE: &str = "Usage: /dirs [ls] | /dirs add <DIR> | /dirs del <DIR>";
    let input = input.trim();
    let head = input.split_whitespace().next()?;
    let alias = head == "/add-dir" || head == ":add-dir";
    if !alias && head != "/dirs" && head != ":dirs" {
        return None;
    }
    if input.contains(['\n', '\r']) {
        return Some(Err(
            "/dirs must be submitted as a single command on its own line".to_string(),
        ));
    }
    let rest = input[head.len()..].trim();
    if alias {
        let usage = "Usage: /add-dir <DIR> (alias for /dirs add)";
        return Some(parse_directory_argument(rest, usage).map(DirsCommand::Add));
    }
    if rest.is_empty() {
        return Some(Ok(DirsCommand::List));
    }
    let (subcommand, argument) = rest
        .split_once(char::is_whitespace)
        .map(|(sub, argument)| (sub, argument.trim()))
        .unwrap_or((rest, ""));
    match subcommand {
        "ls" if argument.is_empty() => Some(Ok(DirsCommand::List)),
        "ls" => Some(Err("Usage: /dirs [ls]".to_string())),
        "add" => Some(parse_directory_argument(argument, USAGE).map(DirsCommand::Add)),
        "del" => Some(parse_directory_argument(argument, USAGE).map(DirsCommand::Delete)),
        _ => Some(Err(USAGE.to_string())),
    }
}

fn parse_directory_argument(value: &str, usage: &str) -> Result<String, String> {
    if value.is_empty() {
        return Err(usage.to_string());
    }
    let value = if value.starts_with(['\'', '"']) {
        let quote = value.chars().next().unwrap();
        if value.len() < 2 || !value.ends_with(quote) {
            return Err(format!("{usage} (matching quotes required)"));
        }
        &value[1..value.len() - 1]
    } else {
        value
    };
    if value.is_empty() {
        return Err(usage.to_string());
    }
    Ok(value.to_string())
}

/// Narrow compatibility entry point for callers that only add one grant.
pub(in crate::ai) fn parse_add_dir_command(input: &str) -> Option<Result<String, String>> {
    match parse_dirs_command(input)? {
        Ok(DirsCommand::Add(value)) => Some(Ok(value)),
        Ok(_) => None,
        Err(error) => Some(Err(error)),
    }
}

pub(in crate::ai) fn add_session_directory(
    store: &SessionStore,
    session_id: &str,
    value: &str,
    cwd: &Path,
    defaults: &[PathBuf],
) -> Result<(String, Vec<PathBuf>), String> {
    let root = resolve_directory(Path::new(value), cwd)
        .map_err(|error| format!("cannot add writable directory: {error}"))?;
    let (roots, added) = store
        .add_session_write_root_with_defaults(session_id, &root, defaults)
        .map_err(|error| format!("cannot save the session writable directory: {error}"))?;
    let mut message = if added {
        format!("Added session writable directory: {}", root.display())
    } else {
        format!(
            "Session writable directory already added: {}",
            root.display()
        )
    };
    let (effective, warnings) = available_roots(defaults.iter().chain(&roots));
    message.push_str(&format!(
        "\nSession writable directory grants: {}",
        effective
            .iter()
            .map(|root| format!("'{}'", root.display()))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    for warning in warnings {
        message.push('\n');
        message.push_str(&warning);
    }
    Ok((message, roots))
}

/// One report shared by `/dirs` and `/dirs ls` in local and serve sessions.
pub(in crate::ai) fn writable_directory_report(
    cwd: &Path,
    startup_roots: &[PathBuf],
    session_roots: &[PathBuf],
) -> String {
    fn render(roots: &[PathBuf]) -> String {
        let (bounded, _) = bounded_roots(roots);
        if bounded.is_empty() {
            return "(none)".to_string();
        }
        bounded
            .iter()
            .map(|root| {
                if root.is_dir() {
                    format!("'{}'", root.display())
                } else {
                    format!("'{}' (unavailable)", root.display())
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
    format!(
        "Writable directories:\n  working directory: '{}'\n  session grants: {}\n  startup grants: {}\nUse /dirs add <DIR> to add or /dirs del <DIR> to remove a session grant.",
        cwd.display(),
        render(session_roots),
        render(startup_roots)
    )
}

/// Remove one session grant. Startup (`--add-dir`) grants need a restart instead.
pub(in crate::ai) fn remove_session_directory(
    store: &SessionStore,
    session_id: &str,
    value: &str,
    cwd: &Path,
    defaults: &[PathBuf],
) -> Result<(String, Vec<PathBuf>), String> {
    let target = resolve_for_removal(Path::new(value), cwd)
        .map_err(|error| format!("cannot remove writable directory: {error}"))?;
    let (roots, removed) = store
        .remove_session_write_root(session_id, &target)
        .map_err(|error| format!("cannot remove the session writable directory: {error}"))?;
    if !removed {
        if defaults.iter().any(|root| root == &target) {
            return Err(format!(
                "'{}' is a startup grant (--add-dir); restart without it to revoke",
                target.display()
            ));
        }
        return Err(format!(
            "'{}' is not a session writable directory; run /dirs to list grants",
            target.display()
        ));
    }
    let mut message = format!("Removed session writable directory: {}", target.display());
    message.push_str(&format!("\nSession writable directory grants: {}", {
        let (bounded, _) = bounded_roots(&roots);
        if bounded.is_empty() {
            "(none)".to_string()
        } else {
            bounded
                .iter()
                .map(|root| format!("'{}'", root.display()))
                .collect::<Vec<_>>()
                .join(", ")
        }
    }));
    Ok((message, roots))
}

/// Removal targets may be temporarily unavailable mounts, so canonicalization falls
/// back to lexical normalization: stored grants are absolute and already canonical.
fn resolve_for_removal(directory: &Path, cwd: &Path) -> Result<PathBuf, String> {
    if directory.as_os_str().is_empty() {
        return Err("requires a non-empty directory path".to_string());
    }
    let expanded = directory
        .to_str()
        .map(|value| PathBuf::from(crate::commonw::utils::expanduser(value).as_ref()))
        .unwrap_or_else(|| directory.to_path_buf());
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    };
    Ok(std::fs::canonicalize(&absolute)
        .unwrap_or_else(|_| crate::ai::path_utils::normalize_lexical(&absolute)))
}

/// Keep absent grants stored so a temporarily unavailable mount can return later.
pub(in crate::ai) fn available_roots<'a>(
    roots: impl IntoIterator<Item = &'a PathBuf>,
) -> (Vec<PathBuf>, Vec<String>) {
    let configured: Vec<PathBuf> = roots.into_iter().cloned().collect();
    let (bounded, mut warnings) = bounded_roots(&configured);
    let mut available = Vec::new();
    for root in &bounded {
        if !root.is_absolute() || !root.is_dir() {
            warnings.push(format!(
                "Skipping unavailable writable directory: {}",
                root.display()
            ));
        } else if !available.contains(root) {
            available.push(root.clone());
        }
    }
    (available, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempRoot(PathBuf);
    impl TempRoot {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("add-dir-session-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path.canonicalize().unwrap())
        }
    }
    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    #[cfg(unix)]
    fn add_dir_resolves_symlink_parent_physically() {
        let root = TempRoot::new();
        let physical = root.0.join("physical");
        let child = physical.join("child");
        std::fs::create_dir_all(&child).unwrap();
        let alias = root.0.join("alias");
        std::os::unix::fs::symlink(&child, &alias).unwrap();
        assert_eq!(
            resolve_directory(&alias.join(".."), &root.0).unwrap(),
            physical
        );
        assert_eq!(
            resolve_directory(Path::new("alias/.."), &root.0).unwrap(),
            physical
        );
        assert_ne!(physical, Path::new("/"));
    }

    #[test]
    fn add_dir_session_grants_survive_config_writes_fork_and_clear_history() {
        let root = TempRoot::new();
        let first = root.0.join("first");
        let second = root.0.join("second");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let store = SessionStore::new(&root.0.join("history.sqlite"));
        let (_, roots) = add_session_directory(&store, "a", "first", &root.0, &[]).unwrap();
        assert_eq!(roots, [first.clone()]);
        assert!(
            add_session_directory(&store, "a", "first/.", &root.0, &[])
                .unwrap()
                .0
                .contains("already added")
        );
        store
            .write_session_serve_config("a", &crate::ai::history::SessionServeConfig::default())
            .unwrap();
        store.add_session_write_root("a", &second).unwrap();
        assert_eq!(
            store
                .read_session_serve_config("a")
                .unwrap()
                .additional_write_roots,
            [first.clone(), second.clone()]
        );
        store.fork_session("a", "fork").unwrap();
        assert_eq!(
            store
                .read_session_serve_config("fork")
                .unwrap()
                .additional_write_roots,
            [first.clone(), second.clone()]
        );
        store.clear_session_history("a").unwrap();
        let reopened = SessionStore::new(&root.0.join("history.sqlite"));
        assert_eq!(
            reopened
                .read_session_serve_config("a")
                .unwrap()
                .additional_write_roots,
            [first, second]
        );
        assert!(
            reopened
                .read_session_serve_config("b")
                .unwrap()
                .additional_write_roots
                .is_empty()
        );
    }

    #[test]
    fn add_dir_archive_import_drops_grants_even_after_checkpoint_restore() {
        let root = TempRoot::new();
        let foreign = root.0.join("foreign");
        let local = root.0.join("local");
        std::fs::create_dir_all(&foreign).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        let history = root.0.join("history.sqlite");
        let store = SessionStore::new(&history);
        store.add_session_write_root("source", &foreign).unwrap();
        crate::ai::history::append_history_messages(
            &store.session_history_file("source"),
            &[crate::ai::history::Message {
                role: "user".into(),
                content: serde_json::json!("initial task"),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let checkpoints = crate::ai::history::CheckpointStore::new(&history, "source");
        checkpoints.save("authorized").unwrap();
        let archive = root.0.join("session.zip");
        store.export_session_archive("source", &archive).unwrap();
        store.fork_session("source", "fork").unwrap();
        assert_eq!(
            store
                .read_session_serve_config("fork")
                .unwrap()
                .additional_write_roots,
            [foreign]
        );
        store.import_session_archive(&archive, "imported").unwrap();
        assert!(
            store
                .read_session_serve_config("imported")
                .unwrap()
                .additional_write_roots
                .is_empty()
        );
        let imported_checkpoints = crate::ai::history::CheckpointStore::new(&history, "imported");
        imported_checkpoints.save("after-import").unwrap();
        store.add_session_write_root("imported", &local).unwrap();
        imported_checkpoints.rollback("after-import").unwrap();
        assert_eq!(
            store
                .read_session_serve_config("imported")
                .unwrap()
                .additional_write_roots,
            [local]
        );
    }

    #[test]
    fn add_dir_rollback_keeps_legacy_missing_meta_behavior() {
        let root = TempRoot::new();
        let history = root.0.join("history.sqlite");
        let store = SessionStore::new(&history);
        let live = store.session_history_file("legacy");
        crate::ai::history::append_history_messages(
            &live,
            &[crate::ai::history::Message {
                role: "user".into(),
                content: serde_json::json!("initial task"),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let checkpoints = crate::ai::history::CheckpointStore::new(&history, "legacy");
        checkpoints.save("before").unwrap();
        rusqlite::Connection::open(&live)
            .unwrap()
            .execute("DROP TABLE meta", [])
            .unwrap();
        checkpoints.rollback("before").unwrap();
        assert!(
            store
                .read_session_serve_config("legacy")
                .unwrap()
                .additional_write_roots
                .is_empty()
        );
    }

    #[test]
    fn add_dir_grant_only_session_is_not_empty() {
        let root = TempRoot::new();
        let store = SessionStore::new(&root.0.join("history.sqlite"));
        assert!(store.is_empty_session("empty").unwrap());
        store.add_session_write_root("granted", &root.0).unwrap();
        assert!(!store.is_empty_session("granted").unwrap());
        assert_eq!(store.read_all_messages("granted").unwrap().len(), 0);
        let reopened = SessionStore::new(&root.0.join("history.sqlite"));
        assert!(!reopened.is_empty_session("granted").unwrap());
    }

    #[test]
    fn add_dir_success_and_duplicate_messages_list_all_session_grants() {
        let root = TempRoot::new();
        let first = root.0.join("first");
        let second = root.0.join("second");
        let shared = root.0.join("shared");
        for path in [&first, &second, &shared] {
            std::fs::create_dir_all(path).unwrap();
        }
        let store = SessionStore::new(&root.0.join("history.sqlite"));
        store.add_session_write_root("a", &first).unwrap();
        for duplicate in [false, true] {
            let (message, roots) = add_session_directory(
                &store,
                "a",
                "second",
                &root.0,
                std::slice::from_ref(&shared),
            )
            .unwrap();
            assert_eq!(message.contains("already added"), duplicate);
            assert_eq!(roots, [first.clone(), second.clone()]);
            for path in [&first, &second, &shared] {
                assert!(message.contains(path.to_str().unwrap()), "{message}");
            }
        }
    }

    #[test]
    fn add_dir_limits_reject_appends_and_bound_persisted_reads() {
        let root = TempRoot::new();
        let store = SessionStore::new(&root.0.join("history.sqlite"));
        let mut roots = Vec::new();
        for i in 0..=MAX_ADDITIONAL_WRITE_ROOTS {
            let path = root.0.join(format!("project-{i}"));
            std::fs::create_dir(&path).unwrap();
            if i < MAX_ADDITIONAL_WRITE_ROOTS {
                store.add_session_write_root("a", &path).unwrap();
            }
            roots.push(path);
        }
        assert!(
            store
                .add_session_write_root("a", roots.last().unwrap())
                .unwrap_err()
                .to_string()
                .contains("limit")
        );
        assert_eq!(
            store
                .read_session_serve_config("a")
                .unwrap()
                .additional_write_roots
                .len(),
            MAX_ADDITIONAL_WRITE_ROOTS
        );
        assert!(
            add_session_directory(
                &store,
                "new",
                roots.last().unwrap().to_str().unwrap(),
                &root.0,
                &roots[..MAX_ADDITIONAL_WRITE_ROOTS]
            )
            .unwrap_err()
            .contains("limit")
        );
        let conn = rusqlite::Connection::open(store.session_history_file("a")).unwrap();
        let write_raw = |roots: &[PathBuf]| {
            conn.execute("INSERT OR REPLACE INTO meta(key,value,created_at) VALUES ('session_write_roots',?1,unixepoch())", [serde_json::to_string(roots).unwrap()]).unwrap();
        };
        write_raw(&roots);
        let stored = store.read_session_serve_config("a").unwrap();
        assert_eq!(
            stored.additional_write_roots,
            roots[..MAX_ADDITIONAL_WRITE_ROOTS]
        );
        assert_eq!(stored.additional_write_root_warnings.len(), 1);
        let oversized = vec![PathBuf::from(format!(
            "/{}",
            "x".repeat(MAX_WRITE_ROOTS_BYTES)
        ))];
        assert!(
            validate_root_limits(&oversized)
                .unwrap_err()
                .contains("bytes")
        );
        write_raw(&oversized);
        let stored = store.read_session_serve_config("a").unwrap();
        assert!(stored.additional_write_roots.is_empty());
        assert!(stored.additional_write_root_warnings[0].contains("bytes"));
        assert!(!store.is_empty_session("a").unwrap());
    }

    #[test]
    fn add_dir_tool_subprocess_origin_is_rejected() {
        if std::env::var_os(AGENT_COMMAND_ENV).is_some() {
            let root = TempRoot::new();
            let store = SessionStore::new(&root.0.join("history.sqlite"));
            assert!(
                crate::ai::cli::try_parse_cli_args(
                    ["a", "--add-dir", root.0.to_str().unwrap()]
                        .into_iter()
                        .map(str::to_string)
                )
                .unwrap_err()
                .contains("execute_command")
            );
            let mut app = crate::ai::tests::test_app_with_cancel_stream(std::sync::Arc::new(
                std::sync::atomic::AtomicBool::new(false),
            ));
            app.config.history_file = root.0.join("history.sqlite");
            app.session_id = "a".into();
            assert!(
                crate::ai::driver::commands::try_handle_local_command(
                    &mut app,
                    &format!("/add-dir {}", root.0.display())
                )
                .unwrap()
            );
            assert!(app.cli.session_add_dirs.is_empty());
            assert!(!store.session_history_file("a").exists());
            return;
        }
        let name = "ai::write_roots::tests::add_dir_tool_subprocess_origin_is_rejected";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env(AGENT_COMMAND_ENV, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        assert!(String::from_utf8_lossy(&output.stderr).contains("execute_command"));
    }

    #[test]
    fn add_dir_restore_reports_missing_grants_once() {
        const PROBE: &str = "ADD_DIR_RESTORE_WARNING_PROBE";
        if std::env::var_os(PROBE).is_some() {
            let root = TempRoot::new();
            let project = root.0.join("missing");
            std::fs::create_dir(&project).unwrap();
            let store = SessionStore::new(&root.0.join("history.sqlite"));
            store.add_session_write_root("a", &project).unwrap();
            std::fs::remove_dir(&project).unwrap();
            let mut app = crate::ai::tests::test_app_with_cancel_stream(std::sync::Arc::new(
                std::sync::atomic::AtomicBool::new(false),
            ));
            app.config.history_file = root.0.join("history.sqlite");
            app.session_id = "a".into();
            crate::ai::driver::commands::session::refresh_session_write_roots(&mut app).unwrap();
            assert_eq!(app.cli.session_add_dirs, [project]);
            return;
        }
        let name = "ai::write_roots::tests::add_dir_restore_reports_missing_grants_once";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env(PROBE, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stderr)
                .matches("Skipping unavailable writable directory")
                .count(),
            1
        );
    }

    #[test]
    fn removing_an_absent_session_grant_does_not_create_a_session_database() {
        let root = TempRoot::new();
        let store = SessionStore::new(&root.0.join("history.sqlite"));
        let (roots, removed) = store
            .remove_session_write_root("a", &root.0.join("project"))
            .unwrap();
        assert!(!removed);
        assert!(roots.is_empty());
        assert!(!store.session_history_file("a").exists());
    }

    #[test]
    fn add_dir_command_parsing_and_local_runtime_use_the_shared_session_store() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let root = TempRoot::new();
        let project = root.0.join("project with spaces");
        std::fs::create_dir_all(&project).unwrap();
        assert_eq!(
            parse_add_dir_command(" /add-dir 'project with spaces' ")
                .unwrap()
                .unwrap(),
            "project with spaces"
        );
        for input in [
            "/add-dir",
            "/add-dir ''",
            "/add-dir 'broken",
            "/add-dir a\nedit a",
        ] {
            assert!(parse_add_dir_command(input).unwrap().is_err());
        }
        assert!(parse_add_dir_command("/add-directory a").is_none());
        assert_eq!(
            parse_dirs_command("/dirs").unwrap().unwrap(),
            DirsCommand::List
        );
        assert_eq!(
            parse_dirs_command("/dirs ls").unwrap().unwrap(),
            DirsCommand::List
        );
        assert_eq!(
            parse_dirs_command("/dirs add 'project with spaces'")
                .unwrap()
                .unwrap(),
            DirsCommand::Add("project with spaces".to_string())
        );
        assert_eq!(
            parse_dirs_command("/dirs del 'project with spaces'")
                .unwrap()
                .unwrap(),
            DirsCommand::Delete("project with spaces".to_string())
        );
        for input in ["/dirs rm x", "/dirs add", "/dirs del", "/dirs ls extra"] {
            assert!(parse_dirs_command(input).unwrap().is_err(), "{input}");
        }
        assert!(parse_dirs_command("/dirsfoo").is_none());
        let mut app = crate::ai::tests::test_app_with_cancel_stream(std::sync::Arc::new(
            std::sync::atomic::AtomicBool::new(false),
        ));
        app.config.history_file = root.0.join("history.sqlite");
        app.session_id = "local".into();
        let command = format!("/add-dir \"{}\"", project.display());
        assert!(crate::ai::driver::commands::try_handle_local_command(&mut app, &command).unwrap());
        assert_eq!(app.cli.session_add_dirs, [project.clone()]);
        let report = writable_directory_report(&root.0, &[], &app.cli.session_add_dirs);
        assert!(report.contains("session grants:"), "{report}");
        assert!(report.contains(&project.display().to_string()), "{report}");
        let child = app.fork_for_subagent();
        assert!(child.cli.add_dirs.contains(&project));
        assert!(child.cli.session_add_dirs.is_empty());
        let delete = format!("/dirs del \"{}\"", project.display());
        assert!(crate::ai::driver::commands::try_handle_local_command(&mut app, &delete).unwrap());
        assert!(app.cli.session_add_dirs.is_empty());
        let stored = SessionStore::new(&app.config.history_file)
            .read_session_serve_config("local")
            .unwrap();
        assert!(stored.additional_write_roots.is_empty());
        let startup = root.0.join("startup");
        std::fs::create_dir(&startup).unwrap();
        assert!(
            remove_session_directory(
                &SessionStore::new(&app.config.history_file),
                "local",
                startup.to_str().unwrap(),
                &root.0,
                &[startup.clone()],
            )
            .is_err()
        );
        assert!(crate::ai::driver::commands::is_local_command_start(
            "/add-dir"
        ));
        assert!(crate::ai::driver::commands::is_local_command_start("/dirs"));
        assert_eq!(
            crate::ai::driver::commands::classify_serve_input(&command),
            crate::ai::driver::commands::ServeInputRoute::ServerCommand
        );
    }
}
