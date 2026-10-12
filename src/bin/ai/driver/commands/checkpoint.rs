//! Interactive `/checkpoint` command (alias `/cp`): save / list / roll back /
//! delete conversation-history checkpoints for the current session.

use crate::ai::{
    history::{
        CheckpointStore, SessionStore, MAX_CHECKPOINTS_PER_SESSION,
        MAX_CHECKPOINT_STORAGE_BYTES,
    },
    types::App,
};

pub fn try_handle_checkpoint_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if cmd != "checkpoint" && cmd != "cp" {
        return Ok(false);
    }

    let action = parts.next().unwrap_or("list");
    let store = CheckpointStore::new(app.config.history_file.as_path(), &app.session_id);

    match action {
        "help" | "h" => {
            print_help();
        }
        "save" | "s" => {
            let Some(name) = parts.next() else {
                println!("Usage: /checkpoint save <name>");
                return Ok(true);
            };
            match store.save(name) {
                Ok(_) => {
                    if let Err(err) = store.set_head(name) {
                        eprintln!("[checkpoint] saved '{name}' but failed to move HEAD: {err}");
                    }
                    println!("Saved checkpoint '{name}'.");
                }
                Err(err) => println!("Failed to save checkpoint: {err}"),
            }
        }
        "list" | "ls" | "tree" | "log" | "" => {
            print_tree(app, &store)?;
        }
        "rollback" | "restore" | "rb" => {
            let args: Vec<String> = parts.map(|s| s.to_string()).collect();
            let (target, no_save, explicit_save) = parse_rollback_args(&args);
            let Some(raw_target) = target else {
                println!("Usage: /checkpoint rollback <name|HEAD> [--save <name> | --no-save]");
                return Ok(true);
            };
            if no_save && explicit_save.is_some() {
                println!("Usage: --save <name> and --no-save cannot be combined.");
                return Ok(true);
            }
            // Git-style HEAD: a pointer to the most recently preserved state,
            // so the head can move back and forth without remembering auto names.
            let head_requested = is_head_alias(&raw_target);
            let name = if head_requested {
                match store.head_target() {
                    Ok(Some(pointed)) => pointed,
                    Ok(None) => {
                        println!(
                            "No HEAD yet: HEAD is set by the first auto-preserving rollback or save."
                        );
                        return Ok(true);
                    }
                    Err(err) => {
                        println!("Failed to rollback: cannot read HEAD: {err}");
                        return Ok(true);
                    }
                }
            } else {
                raw_target
            };
            // Read the live history first so a rollback never silently discards
            // un-checkpointed turns. An empty live history has nothing to preserve.
            let session_store =
                SessionStore::new(app.config.history_file.as_path());
            let live_len = session_store
                .read_all_messages(&app.session_id)
                .map(|messages| messages.len())
                .unwrap_or(0);
            let mut preserved_as: Option<String> = None;
            if live_len > 0 && !no_save {
                let preserve_name = match explicit_save {
                    Some(requested) => requested,
                    None => match unique_auto_checkpoint_name(&store) {
                        Ok(generated) => generated,
                        Err(err) => {
                            println!("Failed to rollback: cannot pick auto-save name: {err}");
                            return Ok(true);
                        }
                    },
                };
                if preserve_name == name {
                    println!(
                        "Refusing to preserve current history as '{preserve_name}': \
                         that would overwrite the rollback target. Pick another --save name."
                    );
                    return Ok(true);
                }
                if let Err(err) = save_preserving_current(&store, &preserve_name) {
                    println!("Failed to rollback: current history was NOT overwritten. {err}");
                    return Ok(true);
                }
                preserved_as = Some(preserve_name);
            }
            match store.rollback(&name) {
                Ok(()) => {
                    refresh_runtime_after_rollback(app);
                    match preserved_as {
                        Some(saved) => {
                            if let Err(err) = store.set_head(&saved) {
                                eprintln!(
                                    "[checkpoint] rolled back but failed to move HEAD: {err}"
                                );
                            }
                            println!(
                                "Rolled back to '{name}'. \
                                 Previous history preserved as '{saved}' (HEAD now points there)."
                            );
                        }
                        None if live_len > 0 => println!(
                            "Rolled back to '{name}' (unsaved turns discarded by --no-save)."
                        ),
                        None => println!(
                            "Rolled back to '{name}'."
                        ),
                    }
                }
                Err(err) => {
                    let head_hint = if head_requested {
                        " (HEAD points to a checkpoint that no longer exists)"
                    } else {
                        ""
                    };
                    match preserved_as {
                        Some(saved) => {
                            if let Err(head_err) = store.set_head(&saved) {
                                eprintln!(
                                    "[checkpoint] rollback failed but history preserved as '{saved}'; HEAD move failed: {head_err}"
                                );
                            }
                            println!(
                                "Failed to rollback: {err}{head_hint} \
                                 (current history preserved as '{saved}', HEAD points there)."
                            );
                        }
                        None => println!("Failed to rollback: {err}{head_hint}"),
                    }
                }
            }
        }
        "delete" | "del" | "rm" => {
            let Some(name) = parts.next() else {
                println!("Usage: /checkpoint delete <name>");
                return Ok(true);
            };
            if is_head_alias(&name) {
                println!(
                    "HEAD is a pointer, not a checkpoint: delete the checkpoint it points to \
                     instead (see `/checkpoint tree`)."
                );
                return Ok(true);
            }
            let head_before = store.head_target().unwrap_or(None);
            match store.delete(name) {
                Ok(true) => {
                    let head_after = store.head_target().unwrap_or(None);
                    if head_before.is_some() && head_after.is_none() {
                        println!("Deleted checkpoint '{name}' (HEAD pointer cleared).");
                    } else {
                        println!("Deleted checkpoint '{name}'.");
                    }
                }
                Ok(false) => println!("Checkpoint '{name}' does not exist."),
                Err(err) => println!("Failed to delete checkpoint: {err}"),
            }
        }
        other => {
            println!("Unknown checkpoint action: '{other}'");
            print_help();
        }
    }
    Ok(true)
}

fn print_help() {
    println!("Checkpoint commands (current session):");
    println!();
    println!("  /checkpoint save <name>       save current history as a checkpoint");
    println!("  /checkpoint tree              show checkpoints as a tree (default; `list` works too)");
    println!("  /checkpoint rollback <name|HEAD> [--save <name> | --no-save]");
    println!("                                restore history; by default the current");
    println!("                                history is auto-saved first and HEAD points");
    println!("                                there, so `rollback HEAD` moves back like git");
    println!("  /checkpoint delete <name>     delete a checkpoint");
    println!();
    println!("  HEAD / @ / - all mean the HEAD pointer (most recently preserved state).");
    println!("  History is never dropped except by explicit `delete` or `--no-save`.");
    println!(
        "  Limit: {} checkpoints / {} MiB per session; saving an existing name replaces it.",
        MAX_CHECKPOINTS_PER_SESSION,
        MAX_CHECKPOINT_STORAGE_BYTES / (1024 * 1024)
    );
    println!("  Alias: /cp");
}

/// Tree preview of the session checkpoints: the live history on top, the HEAD
/// pointer (if set), then every checkpoint newest-first with the HEAD target
/// marked. History counts stay out on purpose: counting messages inside every
/// checkpoint would mean opening every snapshot on each listing.
fn print_tree(
    app: &App,
    store: &CheckpointStore,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoints = store.list()?;
    let head = store.head_target().unwrap_or(None);
    let live_len = SessionStore::new(app.config.history_file.as_path())
        .read_all_messages(&app.session_id)
        .map(|messages| messages.len())
        .unwrap_or(0);
    println!("Checkpoints (current session) — tree:");
    println!("@ live (current session, {live_len} messages)");
    match &head {
        Some(pointed) => println!("o HEAD -> {pointed} (`rollback HEAD` moves back here)"),
        None => println!("o HEAD (unset; set by the first auto-preserving rollback or save)"),
    }
    if checkpoints.is_empty() {
        println!("(no checkpoints yet)");
    } else {
        let mut total_size = 0u64;
        for cp in &checkpoints {
            let when = cp
                .modified_local
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "-".to_string());
            let marker = if head.as_deref() == Some(cp.name.as_str()) {
                "  [HEAD]"
            } else if cp.name.starts_with("auto-") {
                "  [auto]"
            } else {
                ""
            };
            println!("o {:<24} {:>8} bytes  {}{}", cp.name, cp.size_bytes, when, marker);
            total_size = total_size.saturating_add(cp.size_bytes);
        }
        println!(
            "Total: {total_size} bytes (limit: {} checkpoints, {} MiB).",
            MAX_CHECKPOINTS_PER_SESSION,
            MAX_CHECKPOINT_STORAGE_BYTES / (1024 * 1024)
        );
        if head
            .as_deref()
            .is_some_and(|pointed| !checkpoints.iter().any(|cp| cp.name == pointed))
        {
            println!("Warning: HEAD points to a checkpoint that no longer exists.");
        }
    }
    Ok(())
}

/// Git-style aliases for the HEAD pointer. `HEAD` matches case-insensitively
/// because `head` would collide with it on case-insensitive filesystems.
fn is_head_alias(name: &str) -> bool {
    let trimmed = name.trim();
    trimmed.eq_ignore_ascii_case("head") || trimmed == "@" || trimmed == "-"
}

/// Parse `rollback` arguments: the first non-flag token is the target name.
/// Flags: `--no-save` / `--discard` / `--hard` skip preservation (destructive,
/// like `git reset --hard`); `--save <name>` / `--as <name>` / `--save=<name>`
/// preserve the current history under an explicit name instead of an auto name.
fn parse_rollback_args(args: &[String]) -> (Option<String>, bool, Option<String>) {
    let mut target: Option<String> = None;
    let mut no_save = false;
    let mut explicit_save: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--no-save" || arg == "--discard" || arg == "--hard" {
            no_save = true;
        } else if arg == "--save" || arg == "--as" {
            if i + 1 < args.len() {
                explicit_save = Some(args[i + 1].clone());
                i += 1;
            }
        } else if let Some(value) = arg.strip_prefix("--save=") {
            if !value.is_empty() {
                explicit_save = Some(value.to_string());
            }
        } else if let Some(value) = arg.strip_prefix("--as=") {
            if !value.is_empty() {
                explicit_save = Some(value.to_string());
            }
        } else if arg.len() > 1 && arg.starts_with('-') {
            // Unknown flag: leave it unconsumed for future use, do not treat as
            // target. A lone `-` is the git-style previous-state alias for HEAD.
        } else if target.is_none() {
            target = Some(args[i].clone());
        } else if explicit_save.is_none() {
            // A second positional token acts as the preserve name for users coming
            // from `git stash`-style habits: `rollback <target> <save-as>`.
            explicit_save = Some(args[i].clone());
        }
        i += 1;
    }
    (target, no_save, explicit_save)
}

/// Generate a collision-free `auto-YYYYMMDD-HHMMSS` checkpoint name.
fn unique_auto_checkpoint_name(store: &CheckpointStore) -> std::io::Result<String> {
    let existing: std::collections::HashSet<String> =
        store.list()?.into_iter().map(|cp| cp.name).collect();
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let base = format!("auto-{stamp}");
    if !existing.contains(&base) {
        return Ok(base);
    }
    for n in 2..1000 {
        let candidate = format!("{base}-{n}");
        if !existing.contains(&candidate) {
            return Ok(candidate);
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::StorageFull,
        "too many auto checkpoints with the same timestamp",
    ))
}

/// Save the live history under `preserve_name`. Never evicts: dropping history
/// is only allowed through explicit `delete` or `--no-save`, so a full budget
/// aborts the rollback instead and tells the user how to make room. Returns an
/// explanatory message on failure so the caller can abort without data loss.
fn save_preserving_current(
    store: &CheckpointStore,
    preserve_name: &str,
) -> Result<(), String> {
    match store.save(preserve_name) {
        Ok(_) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::StorageFull => Err(format!(
            "auto-save '{preserve_name}' failed: checkpoint budget full ({err}). \
             Delete a checkpoint first, or rerun with --no-save to discard."
        )),
        Err(err) => Err(format!("auto-save '{preserve_name}' failed: {err}")),
    }
}

/// Rebuild history-derived runtime state after an in-place rollback so the
/// next turn does not act on deleted messages. User-pinned state (attached
/// images, forced skills, enabled tools) is intentionally left untouched;
/// only the ledger rebuilt from the restored messages plus prune marks is reset.
fn refresh_runtime_after_rollback(app: &mut App) {
    crate::ai::history::invalidate_context_history_cache_for(&app.session_history_file);
    let session_store = SessionStore::new(app.config.history_file.as_path());
    match session_store.read_all_messages(&app.session_id) {
        Ok(messages) => {
            if let Err(err) =
                super::session::reset_stale_patch_targets_from_messages(app, &messages)
            {
                eprintln!("[checkpoint] failed to rebuild patch ledger after rollback: {err}");
            }
        }
        Err(err) => {
            eprintln!("[checkpoint] failed to read restored history after rollback: {err}");
        }
    }
    if let Err(err) = super::session::restore_prune_marks_for_history(app) {
        eprintln!("[checkpoint] failed to restore prune marks after rollback: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::{is_head_alias, parse_rollback_args};

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn rollback_args_support_flags_and_positional_save_as() {
        let (target, no_save, explicit) = parse_rollback_args(&args(&["c1"]));
        assert_eq!(target.as_deref(), Some("c1"));
        assert!(!no_save);
        assert!(explicit.is_none());

        let (target, no_save, explicit) = parse_rollback_args(&args(&["c1", "--no-save"]));
        assert_eq!(target.as_deref(), Some("c1"));
        assert!(no_save);
        assert!(explicit.is_none());

        let (target, no_save, explicit) =
            parse_rollback_args(&args(&["c1", "--save", "mywork"]));
        assert_eq!(target.as_deref(), Some("c1"));
        assert!(!no_save);
        assert_eq!(explicit.as_deref(), Some("mywork"));

        // Second positional token acts as the preserve name.
        let (target, no_save, explicit) = parse_rollback_args(&args(&["c1", "mywork"]));
        assert_eq!(target.as_deref(), Some("c1"));
        assert!(!no_save);
        assert_eq!(explicit.as_deref(), Some("mywork"));
    }

    #[test]
    fn rollback_args_accept_head_aliases_as_target() {
        for alias in ["HEAD", "head", "@", "-"] {
            let (target, _, _) = parse_rollback_args(&args(&[alias]));
            assert_eq!(target.as_deref(), Some(alias));
            assert!(is_head_alias(&target.unwrap()));
        }
        assert!(is_head_alias("HEAD"));
        assert!(is_head_alias("@"));
        assert!(is_head_alias("-"));
        assert!(!is_head_alias("c1"));
        assert!(!is_head_alias("headspace"));
    }
}
