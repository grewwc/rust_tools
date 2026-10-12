//! Interactive `/checkpoint` command (alias `/cp`): immutable commits,
//! movable branches, checkout, tree, and deletion for the current session.

use crate::ai::{
    history::{
        CheckpointHead, CheckpointInfo, CheckpointStore, MAX_CHECKPOINT_STORAGE_BYTES,
        MAX_CHECKPOINTS_PER_SESSION, SessionStore,
    },
    types::App,
};
use chrono::Local;
use colored::Colorize;

/// Internal bridge used by `serve-chat` checkpoint requests. The serve
/// process runs the same one-shot CLI command as local mode, but interactive
/// confirmation must be surfaced to the client instead of reading server
/// stdin.
pub(crate) const SERVE_CHECKPOINT_CONFIRM_ENV: &str = "RUST_TOOLS_SERVE_CHECKPOINT_CONFIRM";
pub(crate) const SERVE_CHECKPOINT_CONFIRM_PROBE: &str = "probe";
pub(crate) const SERVE_CHECKPOINT_CONFIRM_YES: &str = "yes";
pub(crate) const SERVE_CHECKPOINT_CONFIRM_MARKER: &str = "__RUST_TOOLS_CHECKPOINT_CONFIRM__:";

fn prompt_checkpoint_confirmation(prompt: &str) -> Option<bool> {
    match std::env::var(SERVE_CHECKPOINT_CONFIRM_ENV).as_deref() {
        Ok(SERVE_CHECKPOINT_CONFIRM_PROBE) => {
            println!("{SERVE_CHECKPOINT_CONFIRM_MARKER}{prompt}");
            Some(false)
        }
        Ok(SERVE_CHECKPOINT_CONFIRM_YES) => Some(true),
        _ => crate::commonw::prompt::prompt_yes_or_no_danger(prompt),
    }
}

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
        "save" | "commit" | "s" => {
            let name = parts
                .next()
                .map(str::to_string)
                .unwrap_or_else(|| Local::now().format("commit-%Y%m%d-%H%M%S-%3f").to_string());
            match store.save(&name) {
                Ok(_) => {
                    println!("Saved checkpoint '{name}'.");
                    publish_checkpoint_names(&store);
                }
                Err(err) => println!("Failed to save checkpoint: {err}"),
            }
        }
        "list" | "ls" | "tree" | "log" | "" => {
            print_tree(app, &store)?;
        }
        "branch" | "br" => match parts.next() {
            None => {
                let active = match store.head_state() {
                    Ok(CheckpointHead::Attached { branch, .. }) => Some(branch),
                    Ok(CheckpointHead::Detached(_)) => None,
                    Err(err) => {
                        println!("Failed to read HEAD: {err}");
                        return Ok(true);
                    }
                };
                match store.list_branches() {
                    Ok(branches) if branches.is_empty() => println!("No branches yet."),
                    Ok(branches) => {
                        for branch in branches {
                            let marker = if active.as_deref() == Some(branch.name.as_str()) {
                                "*"
                            } else {
                                " "
                            };
                            let integrity = if branch.is_dangling {
                                " (missing checkpoint)"
                            } else {
                                ""
                            };
                            println!(
                                "{marker} {} -> {}{integrity}",
                                branch.name,
                                branch.target.as_deref().unwrap_or("(unborn)")
                            );
                        }
                    }
                    Err(err) => println!("Failed to list branches: {err}"),
                }
            }
            Some("-d" | "--delete" | "delete") => {
                let Some(name) = parts.next() else {
                    println!("Usage: /checkpoint branch --delete <name>");
                    return Ok(true);
                };
                match store.delete_branch(name) {
                    Ok(true) => {
                        publish_checkpoint_names(&store);
                        println!("Deleted branch '{name}'. Checkpoints were kept.");
                    }
                    Ok(false) => println!("Branch '{name}' not found."),
                    Err(err) => println!("Failed to delete branch: {err}"),
                }
            }
            Some(name) => {
                let start = parts.next();
                match store.create_branch(name, start) {
                    Ok(()) => {
                        publish_checkpoint_names(&store);
                        match store.head_state() {
                            Ok(CheckpointHead::Attached {
                                branch,
                                target: None,
                            }) => println!(
                                "Created and switched to unborn branch '{branch}'. The next save will create its first checkpoint."
                            ),
                            _ => println!(
                                "Created branch '{name}'. Switch with `/cp switch {name}`."
                            ),
                        }
                    }
                    Err(err) => println!("Failed to create branch: {err}"),
                }
            }
        },
        "checkout" | "co" | "switch" => {
            // Navigation requires a clean live tree by default; callers must
            // explicitly save or discard un-checkpointed work.
            let is_switch = action == "switch";
            let args: Vec<String> = parts.map(|s| s.to_string()).collect();
            let parsed = match parse_navigation_args(&args, if is_switch { "-c" } else { "-b" }) {
                Ok(parsed) => parsed,
                Err(err) => {
                    println!("Invalid {action} arguments: {err}");
                    return Ok(true);
                }
            };
            let NavigationArgs {
                target,
                create_branch,
                discard: discard_requested,
                explicit_save,
            } = parsed;
            if create_branch.is_none() && target.is_none() {
                if is_switch {
                    println!(
                        "Usage: /checkpoint switch <branch> | /checkpoint switch -c <new-branch> [start]"
                    );
                } else {
                    println!(
                        "Usage: /checkpoint checkout <branch|checkpoint> | /checkpoint checkout -b <new-branch> [start]"
                    );
                }
                return Ok(true);
            }
            if discard_requested && explicit_save.is_some() {
                println!("Usage: --save <name> and --discard cannot be combined.");
                return Ok(true);
            }
            let raw_target = match (target, create_branch.as_deref()) {
                (Some(target), _) => target,
                (None, Some(new_branch)) => match store.prepare_branch(new_branch, None) {
                    Ok(Some(target)) => target,
                    Ok(None) => match store.create_branch(new_branch, None) {
                        Ok(()) => {
                            publish_checkpoint_names(&store);
                            println!(
                                "Switched to a new unborn branch '{new_branch}'. The next save will create its first checkpoint."
                            );
                            return Ok(true);
                        }
                        Err(err) => {
                            println!("Failed to create branch '{new_branch}': {err}");
                            return Ok(true);
                        }
                    },
                    Err(err) => {
                        println!("Failed to create branch '{new_branch}': {err}");
                        return Ok(true);
                    }
                },
                (None, None) => unreachable!("missing navigation target handled above"),
            };
            // Git-style HEAD: the current commit. Every save and every
            // successful switch moves it, so `HEAD` always means "where live
            // is checked out", never "where we came from".
            let head_requested = is_head_alias(&raw_target);
            // `latest` resolves to the newest checkpoint (list order is
            // newest-first, so it matches the `[latest]` tag in the tree).
            let latest_requested = is_latest_alias(&raw_target);
            let matching_branch = (!head_requested && !latest_requested)
                .then(|| store.list_branches().ok())
                .flatten()
                .and_then(|branches| {
                    branches
                        .into_iter()
                        .find(|branch| branch.name == raw_target)
                });
            if is_switch && create_branch.is_none() && matching_branch.is_none() {
                println!("Failed to switch: branch '{raw_target}' not found.");
                return Ok(true);
            }
            let branch_name = create_branch
                .clone()
                .or_else(|| matching_branch.as_ref().map(|branch| branch.name.clone()));
            let name = if let Some(branch) = matching_branch {
                if branch.is_dangling {
                    println!(
                        "Failed to checkout: branch '{}' points to missing checkpoint '{}' (nothing was changed).",
                        branch.name,
                        branch.target.as_deref().unwrap_or_default()
                    );
                    return Ok(true);
                }
                match branch.target {
                    Some(target) => target,
                    None if create_branch.is_some() => {
                        println!(
                            "Failed to create branch: start branch '{}' has no checkpoint yet.",
                            branch.name
                        );
                        return Ok(true);
                    }
                    None => {
                        return switch_to_unborn_branch(
                            app,
                            &store,
                            &branch.name,
                            explicit_save.clone(),
                            discard_requested,
                        );
                    }
                }
            } else if head_requested {
                let listed = match store.list() {
                    Ok(checkpoints) => checkpoints,
                    Err(err) => {
                        println!("Failed to checkout: cannot list checkpoints: {err}");
                        return Ok(true);
                    }
                };
                if listed.is_empty() {
                    println!("No checkpoints yet: save one first (`/checkpoint save <name>`).");
                    return Ok(true);
                }
                let stored_head = match store.head_target() {
                    Ok(pointed) => pointed,
                    Err(err) => {
                        println!("Failed to checkout: cannot read HEAD: {err}");
                        return Ok(true);
                    }
                };
                let stored_base = match store.base_target() {
                    Ok(pointed) => pointed,
                    Err(err) => {
                        println!("Failed to checkout: cannot read BASE: {err}");
                        return Ok(true);
                    }
                };
                // Resolved straight from a fresh listing (stored pointer, else
                // live base, else newest), so the target trivially exists; the
                // check below is skipped for it, like `latest`.
                match effective_head(&listed, stored_head.as_deref(), stored_base.as_deref()) {
                    Some(pointed) => pointed.to_string(),
                    None => {
                        println!("No checkpoints yet: save one first (`/checkpoint save <name>`).");
                        return Ok(true);
                    }
                }
            } else if latest_requested {
                match store.list() {
                    Ok(checkpoints) => match checkpoints.into_iter().next() {
                        // Resolved straight from a fresh listing, so the target
                        // trivially exists; the check below is skipped for it.
                        Some(newest) => newest.name,
                        None => {
                            println!(
                                "No checkpoints yet: save one first (`/checkpoint save <name>`)."
                            );
                            return Ok(true);
                        }
                    },
                    Err(err) => {
                        println!("Failed to checkout: cannot list checkpoints: {err}");
                        return Ok(true);
                    }
                }
            } else {
                raw_target
            };
            if let Some(new_branch) = create_branch.as_deref()
                && let Err(err) = store.prepare_branch(new_branch, Some(&name))
            {
                println!("Failed to create branch '{new_branch}': {err}");
                return Ok(true);
            }
            // Validate the target before preserving anything: a failed switch
            // must have zero side effects (no save and no HEAD move).
            // `latest` and `HEAD` resolve straight from a fresh listing above,
            // so their targets trivially exist and skip this check.
            if !latest_requested && !head_requested {
                let names = match store.list() {
                    Ok(checkpoints) => checkpoints,
                    Err(err) => {
                        println!("Failed to checkout: cannot list checkpoints: {err}");
                        return Ok(true);
                    }
                };
                if !names.iter().any(|candidate| candidate.name == name) {
                    println!(
                        "Failed to checkout: checkpoint '{name}' not found (nothing was changed)."
                    );
                    return Ok(true);
                }
            }
            // Read the live history first so a switch never silently discards
            // un-checkpointed turns. An empty live history has nothing to preserve.
            let session_store = SessionStore::new(app.config.history_file.as_path());
            let live_messages = match session_store.read_all_messages(&app.session_id) {
                Ok(messages) => messages,
                Err(err) => {
                    println!("Failed to switch checkpoints: cannot read live history: {err}");
                    return Ok(true);
                }
            };
            let live_len = live_messages.len();
            // `--discard` drops un-checkpointed turns. Dropping anything the
            // tree cannot rebuild needs an explicit yes; clean switches never prompt.
            let discard = discard_requested && explicit_save.is_none();
            let mut preserved_as: Option<String> = None;
            let mut already_on_target = false;
            let mut clean_base: Option<String> = None;
            let live_is_clean = match store.live_matches_base(&live_messages) {
                Ok(matches) => matches,
                Err(err) => {
                    println!(
                        "Failed to switch checkpoints: cannot verify whether live history is clean: {err}"
                    );
                    return Ok(true);
                }
            };
            if live_is_clean {
                let base = match store.base_target() {
                    Ok(base) => base,
                    Err(err) => {
                        println!("Failed to switch checkpoints: cannot read BASE: {err}");
                        return Ok(true);
                    }
                };
                if let Some(base) = base {
                    if base == name {
                        already_on_target = true;
                    } else {
                        clean_base = Some(base);
                    }
                }
            }
            let preparation = switch_preparation(live_len, live_is_clean, discard, explicit_save);
            if preparation == SwitchPreparation::RefuseDirty {
                let operation = if is_switch { "switch" } else { "checkout" };
                println!(
                    "Cannot {operation} '{name}': live history has un-checkpointed changes. \
                     Save them first with `/cp save <name>`, or add `--discard` to discard them."
                );
                return Ok(true);
            }
            if let SwitchPreparation::Preserve(preserve_name) = preparation {
                if preserve_name == name {
                    println!(
                        "Refusing to preserve current history as '{preserve_name}': \
                         that would overwrite the checkout target. Pick another --save name."
                    );
                    return Ok(true);
                }
                if let Err(err) = save_preserving_current(&store, &preserve_name) {
                    println!("Failed to checkout: current history was NOT overwritten. {err}");
                    return Ok(true);
                }
                preserved_as = Some(preserve_name);
            }
            if already_on_target && preserved_as.is_none() {
                // Live already equals the target: restoring would be a no-op,
                // but HEAD still moves onto the current commit when needed.
                let attach_result = match (create_branch.as_deref(), branch_name.as_deref()) {
                    (Some(branch), _) => store
                        .create_branch(branch, Some(&name))
                        .and_then(|()| store.attach_branch(branch)),
                    (None, Some(branch)) => store.attach_branch(branch),
                    (None, None) => store.set_head(&name),
                };
                if let Err(err) = attach_result {
                    if let Some(branch) = create_branch.as_deref() {
                        let _ = store.delete_branch(branch);
                    }
                    println!("Failed to switch: {err}");
                    return Ok(true);
                }
                match (create_branch.as_deref(), branch_name.as_deref()) {
                    (Some(branch), _) => println!("Switched to a new branch '{branch}'."),
                    (None, Some(branch)) => println!("Already on branch '{branch}'."),
                    (None, None) => println!(
                        "Already on '{name}': live history matches it, so nothing was preserved or restored."
                    ),
                }
                return Ok(true);
            }
            if discard && live_len > 0 {
                let base_matches = match store.base_target().unwrap_or(None) {
                    Some(base) => matches!(
                        store.live_matches_checkpoint(&live_messages, &base),
                        Ok(true)
                    ),
                    None => false,
                };
                if !base_matches {
                    let confirm = prompt_checkpoint_confirmation(&format!(
                        "Discard {live_len} un-checkpointed live message(s) and switch to '{name}'? (y/n): ",
                    ));
                    if confirm != Some(true) {
                        println!("Canceled: nothing was changed.");
                        return Ok(true);
                    }
                }
            }
            if let Some(branch) = create_branch.as_deref()
                && let Err(err) = store.create_branch(branch, Some(&name))
            {
                println!("Failed to create branch '{branch}': {err}");
                return Ok(true);
            }
            let switch_result = match branch_name.as_deref() {
                Some(branch) => store.switch_branch(branch),
                None => store.rollback(&name),
            };
            match switch_result {
                Ok(()) => {
                    refresh_runtime_after_rollback(app);
                    let (verb, destination) =
                        match (create_branch.as_deref(), branch_name.as_deref()) {
                            (Some(branch), _) => ("Switched to a new branch", branch),
                            (None, Some(branch)) => ("Switched to branch", branch),
                            (None, None) => ("Checked out", name.as_str()),
                        };
                    match (preserved_as, clean_base) {
                        (Some(saved), _) => println!(
                            "{verb} '{destination}'. Previous history preserved as '{saved}'."
                        ),
                        (None, Some(base)) => println!(
                            "{verb} '{destination}'. No new turns since '{base}', so nothing was preserved."
                        ),
                        (None, None) if discard && live_len > 0 => {
                            println!("{verb} '{destination}' (un-checkpointed turns discarded).")
                        }
                        (None, None) => println!("{verb} '{destination}'."),
                    }
                }
                Err(err) => {
                    if let Some(branch) = create_branch.as_deref()
                        && let Err(cleanup_err) = store.delete_branch(branch)
                    {
                        eprintln!(
                            "[checkpoint] failed to clean up branch '{branch}' after switch failure: {cleanup_err}"
                        );
                    }
                    let head_hint = if head_requested {
                        " (HEAD points to a checkpoint that no longer exists)"
                    } else {
                        ""
                    };
                    match preserved_as {
                        Some(saved) => {
                            println!(
                                "Failed to switch: {err}{head_hint} \
                                 (current history preserved as '{saved}', HEAD points there)."
                            );
                        }
                        None => println!("Failed to switch: {err}{head_hint}"),
                    }
                }
            }
            // An explicit --save may have changed the listing, so refresh the
            // completion cache even on failure.
            publish_checkpoint_names(&store);
        }
        "delete" | "del" | "rm" => {
            let Some(name) = parts.next() else {
                println!("Usage: /checkpoint delete <name>");
                return Ok(true);
            };
            if is_head_alias(&name) || is_latest_alias(&name) {
                println!(
                    "'{name}' is a pointer alias, not a checkpoint: delete the checkpoint it points to \
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
                    publish_checkpoint_names(&store);
                }
                Ok(false) => println!("Checkpoint '{name}' does not exist."),
                Err(err) => println!("Failed to delete checkpoint: {err}"),
            }
        }
        "clear" => {
            let checkpoints = match store.list() {
                Ok(list) => list,
                Err(err) => {
                    println!("Failed to clear checkpoints: cannot list checkpoints: {err}");
                    return Ok(true);
                }
            };
            if checkpoints.is_empty() {
                println!("No checkpoints to clear.");
                return Ok(true);
            }
            let total_bytes: u64 = checkpoints.iter().map(|cp| cp.size_bytes).sum();
            let confirm = prompt_checkpoint_confirmation(&format!(
                "Permanently delete all {} checkpoint(s) ({})? Live history is untouched. (y/n): ",
                checkpoints.len(),
                format_human_bytes(total_bytes),
            ));
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }
            match store.clear() {
                Ok(removed) => {
                    println!("Cleared {removed} checkpoint(s).");
                    publish_checkpoint_names(&store);
                }
                Err(err) => println!("Failed to clear checkpoints: {err}"),
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
    println!(
        "  /checkpoint save [name]       create an immutable commit; advances the current branch"
    );
    println!(
        "  /checkpoint tree              show checkpoints as a tree, newest first (default; `list` works too)"
    );
    println!("  /checkpoint checkout <branch|checkpoint> [--save <name> | --discard]");
    println!("                                switch to a branch or detach at a checkpoint; dirty");
    println!(
        "                                history is refused unless explicitly saved or discarded"
    );
    println!("  /checkpoint checkout -b <new-branch> [start]");
    println!(
        "                                create and switch to a branch (Git-style checkout -b)"
    );
    println!("  /checkpoint branch <name> [start]");
    println!("                                create a branch at start (or current HEAD)");
    println!("  /checkpoint branch            list branches (* marks the current branch)");
    println!("  /checkpoint branch -d <name>  delete only the branch pointer");
    println!("  /checkpoint switch <branch>   restore and attach HEAD to a branch");
    println!("  /checkpoint switch -c <new-branch> [start]");
    println!("                                create and switch to a branch (Git-style switch -c)");
    println!("  /checkpoint delete <name>     delete a checkpoint");
    println!(
        "  /checkpoint clear             delete ALL checkpoints (asks for confirmation first)"
    );
    println!();
    println!("  HEAD / @ / - all mean the HEAD pointer (the current commit);");
    println!("  `latest` means the newest checkpoint (the `[latest]` row in the tree).");
    println!("  History is never dropped except by explicit `delete`, `clear`, or `--discard`.");
    println!(
        "  Limit: {} commits + branches / {} MiB per session; commit names cannot be overwritten.",
        MAX_CHECKPOINTS_PER_SESSION,
        MAX_CHECKPOINT_STORAGE_BYTES / (1024 * 1024)
    );
    println!("  Alias: /cp");
}

/// One display row of the checkpoint forest: the index into the newest-first
/// `checkpoints` slice plus the already-resolved tree-drawing prefix placed
/// before the HEAD glyph (empty for roots, e.g. `"│   └── "` for nested rows).
struct ForestRow {
    index: usize,
    prefix: String,
}

/// Order `checkpoints` (newest-first) depth-first for tree display: roots keep
/// their relative order, each parent is immediately followed by its subtree,
/// children stay newest-first. Lone-child links render flat with no indent and no connector,
/// so a straight save-after-save sequence reads as peers. If live history remains based on a
/// parent with one saved child, live is the implicit second branch and the saved child gets a
/// connector. Dangling parents (deleted after the
/// child was saved) and self-links count as roots, and anything still
/// unvisited after the root pass (only possible with a corrupt parent cycle)
/// is emitted as a root too, so the walk always terminates with every entry
/// shown exactly once.
fn forest_order(checkpoints: &[CheckpointInfo], live_base: Option<&str>) -> Vec<ForestRow> {
    fn is_root(checkpoints: &[CheckpointInfo], idx: usize) -> bool {
        match &checkpoints[idx].parent {
            None => true,
            Some(parent) if parent == &checkpoints[idx].name => true,
            Some(parent) => !checkpoints.iter().any(|cp| &cp.name == parent),
        }
    }
    let mut rows = Vec::with_capacity(checkpoints.len());
    let mut visited = vec![false; checkpoints.len()];
    for idx in 0..checkpoints.len() {
        if !visited[idx] && is_root(checkpoints, idx) {
            push_subtree(
                checkpoints,
                idx,
                String::new(),
                String::new(),
                live_base,
                &mut rows,
                &mut visited,
            );
        }
    }
    // Cycle guard: anything unreachable from a root is shown as its own root.
    for idx in 0..checkpoints.len() {
        if !visited[idx] {
            push_subtree(
                checkpoints,
                idx,
                String::new(),
                String::new(),
                live_base,
                &mut rows,
                &mut visited,
            );
        }
    }
    rows
}

/// Push one subtree depth-first: the row itself with `prefix`, then each
/// child. Branch connectors (`├── ` / `└── `) draw real forks: a parent with
/// two or more saved children, or one saved child while live history remains
/// based on the parent. Other lone children render flat. The base never accumulates:
/// the child inherits `child_base` unchanged, so a long chain keeps every
/// entry at the same indent, while a continuation inside a fork keeps that
/// branch's indent. `child_base` carries the ancestors' continuations
/// (`│   ` / `    `) for deeper levels.
fn push_subtree(
    checkpoints: &[CheckpointInfo],
    idx: usize,
    prefix: String,
    child_base: String,
    live_base: Option<&str>,
    rows: &mut Vec<ForestRow>,
    visited: &mut [bool],
) {
    if visited[idx] {
        return;
    }
    visited[idx] = true;
    rows.push(ForestRow { index: idx, prefix });
    let name = checkpoints[idx].name.clone();
    let kids: Vec<usize> = checkpoints
        .iter()
        .enumerate()
        .filter(|(child, cp)| !visited[*child] && cp.parent.as_deref() == Some(name.as_str()))
        .map(|(child, _)| child)
        .collect();
    // A lone saved child is normally a straight-line continuation. If live
    // history remains on its parent, live is the other side of the fork.
    if kids.len() == 1 && live_base != Some(name.as_str()) {
        push_subtree(
            checkpoints,
            kids[0],
            child_base.clone(),
            child_base,
            live_base,
            rows,
            visited,
        );
        return;
    }
    for (pos, child) in kids.iter().enumerate() {
        let last = pos + 1 == kids.len();
        let (connector, continuation) = if last {
            ("└── ", "    ")
        } else {
            ("├── ", "│   ")
        };
        push_subtree(
            checkpoints,
            *child,
            format!("{child_base}{connector}"),
            format!("{child_base}{continuation}"),
            live_base,
            rows,
            visited,
        );
    }
}

/// Resolve the HEAD row to show and to mark inline: the stored pointer when
/// it still names a listed checkpoint, else the live base (the commit live
/// is checked out on), else the newest checkpoint. HEAD therefore always
/// lands on a row whenever any checkpoint exists and never dangles.
fn effective_head<'a>(
    checkpoints: &'a [CheckpointInfo],
    head: Option<&'a str>,
    base: Option<&'a str>,
) -> Option<&'a str> {
    if head.is_some_and(|pointed| checkpoints.iter().any(|cp| cp.name == pointed)) {
        return head;
    }
    if base.is_some_and(|parent| checkpoints.iter().any(|cp| cp.name == parent)) {
        return base;
    }
    checkpoints.first().map(|cp| cp.name.as_str())
}

/// Tree preview of the session checkpoints: the live history on top, the HEAD
/// pointer (always resolved, never dangling), then every checkpoint depth-first
/// with the HEAD target marked inline like git decoration. Linear
/// save-after-save links render flat with no indent and no connector;
/// branch lines appear only at real forks. The newest row also carries
/// `[latest]`, so it stays identifiable when several saves share the same
/// displayed second. History counts stay out on purpose: counting messages
/// inside every checkpoint would mean opening every snapshot on each listing.
fn print_tree(app: &App, store: &CheckpointStore) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoints = store.list()?;
    let branches = store.list_branches()?;
    let (active_branch, stored_head) = match store.head_state()? {
        CheckpointHead::Attached { branch, target } => (Some(branch), target),
        CheckpointHead::Detached(target) => (None, target),
    };
    let base = store.base_target().unwrap_or(None);
    let live_len = SessionStore::new(app.config.history_file.as_path())
        .read_all_messages(&app.session_id)
        .map(|messages| messages.len())
        .unwrap_or(0);
    let total_size: u64 = checkpoints
        .iter()
        .map(|cp| cp.size_bytes)
        .fold(0u64, |acc, size| acc.saturating_add(size));
    println!(
        "{} ({} / {} checkpoints, {} / {} MiB)",
        "Checkpoints".bold(),
        checkpoints.len(),
        MAX_CHECKPOINTS_PER_SESSION,
        format_human_bytes(total_size).bold(),
        MAX_CHECKPOINT_STORAGE_BYTES / (1024 * 1024)
    );
    // The live history is the implicit child of its base checkpoint: naming
    // the base here connects the working copy to the tree below.
    let live_detail = match &base {
        Some(parent) => format!("{live_len} messages (current session, based on '{parent}')"),
        None => format!("{live_len} messages (current session)"),
    };
    println!("  {}  {}", "@ live".green().bold(), live_detail.dimmed());
    // The table row below shows HEAD with `→` + `[HEAD]`; a second pointer
    // line would repeat the same name twice. HEAD never dangles: it falls
    // back to the live base, then to the newest checkpoint, so the extra
    // line only exists when there is no row to mark (no checkpoints yet).
    // A best-effort write heals a missing or dangling stored pointer so the
    // next command reads it back directly.
    let head = if active_branch.is_some() && stored_head.is_none() {
        None
    } else {
        match effective_head(&checkpoints, stored_head.as_deref(), base.as_deref()) {
            Some(resolved) => {
                let resolved = resolved.to_string();
                if active_branch.is_none() && stored_head.as_deref() != Some(resolved.as_str()) {
                    let _ = store.set_head(&resolved);
                }
                Some(resolved)
            }
            None => None,
        }
    };
    if let Some(branch) = active_branch.as_deref().filter(|_| stored_head.is_none()) {
        println!(
            "  {}  {}",
            "○ HEAD".dimmed(),
            format!("branch:{branch} (unborn — next save creates its first checkpoint)").dimmed()
        );
    } else if checkpoints.is_empty() {
        println!(
            "  {}  {}",
            "○ HEAD".dimmed(),
            "unset (no checkpoints yet)".dimmed()
        );
    }
    if checkpoints.is_empty() {
        println!(
            "  {}",
            "(no checkpoints yet — save one with /checkpoint save <name>)".dimmed()
        );
        return Ok(());
    }
    // Branch lines widen rows with depth, so align the columns on the full
    // `prefix + glyph + name` label instead of the bare name. The name itself
    // is still capped so one long entry cannot push the table off screen.
    // Padding is manual spaces (not `format!` width) so box-drawing glyphs
    // never disturb the column math.
    let order = forest_order(&checkpoints, base.as_deref());
    let short_names: Vec<String> = checkpoints
        .iter()
        .map(|cp| truncate_middle(&cp.name, 48))
        .collect();
    let label_width = order
        .iter()
        .map(|row| row.prefix.chars().count() + 2 + short_names[row.index].chars().count())
        .max()
        .unwrap_or(12)
        .max(12);
    // Rows print as `  {prefix}{glyph} {name} ...`, so the header needs the
    // same 2-column prefix to keep NAME/SIZE/SAVED aligned with the body.
    println!(
        "  {}",
        format!(
            "{}{}  {:>9}  {:<19}  {}",
            "NAME",
            " ".repeat(label_width - "NAME".len()),
            "SIZE",
            "SAVED",
            "TAGS",
        )
        .dimmed()
    );
    // `checkpoints` arrives newest-first (see `CheckpointStore::list`), so the
    // first row is the newest save even when rows share a displayed second.
    for row in &order {
        let cp = &checkpoints[row.index];
        let when = cp
            .modified_local
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "-".to_string());
        let short_name = &short_names[row.index];
        let is_head = head.as_deref() == Some(cp.name.as_str());
        let is_latest = row.index == 0;
        let is_orphan = cp
            .parent
            .as_deref()
            .is_some_and(|parent| !checkpoints.iter().any(|other| other.name == parent));
        let glyph = if is_head {
            "→".cyan().bold()
        } else {
            "○".dimmed()
        };
        let name_pad =
            " ".repeat(label_width - row.prefix.chars().count() - 2 - short_name.chars().count());
        let name_cell = if is_head {
            short_name.cyan().bold()
        } else {
            short_name.bold()
        };
        let size_cell = format!("{:>9}", format_human_bytes(cp.size_bytes));
        let mut tags = String::new();
        if is_latest {
            tags.push_str(&"[latest] ".yellow().bold().to_string());
        }
        if is_head {
            tags.push_str(&"[HEAD] ".cyan().bold().to_string());
        }
        for branch in branches
            .iter()
            .filter(|branch| branch.target.as_deref() == Some(cp.name.as_str()))
        {
            let label = if active_branch.as_deref() == Some(branch.name.as_str()) {
                format!("[branch:{}*] ", branch.name)
                    .cyan()
                    .bold()
                    .to_string()
            } else {
                format!("[branch:{}] ", branch.name).green().to_string()
            };
            tags.push_str(&label);
        }
        if is_orphan {
            tags.push_str(&"[orphan] ".dimmed().to_string());
        }
        if cp.name.starts_with("auto-") {
            tags.push_str(&"[auto] ".dimmed().to_string());
        }
        println!(
            "  {}{} {}{}  {size_cell}  {when:<19}  {tags}",
            row.prefix.dimmed(),
            glyph,
            name_cell,
            name_pad,
        );
    }
    println!(
        "  {}",
        "linear commits stay flat · → marks HEAD · branch:<name>* marks the active branch · switch <branch> attaches · checkout <commit> detaches · save advances".dimmed()
    );
    // Tree is the most common precursor to `checkout <TAB>`: publish the
    // names just listed so third-token completion has fresh candidates.
    crate::ai::prompt::completion::CommandCompleter::set_checkpoint_name_candidates(
        checkpoints.into_iter().map(|cp| cp.name).collect(),
    );
    Ok(())
}

/// Format bytes as `1.9 MiB` instead of a raw count so the tree column stays
/// aligned for large snapshots.
fn format_human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Shorten very long checkpoint names with a middle ellipsis so one entry
/// cannot push the size/time columns off screen.
fn truncate_middle(name: &str, max_chars: usize) -> String {
    let count = name.chars().count();
    if count <= max_chars || max_chars < 5 {
        return name.to_string();
    }
    let keep = max_chars - 1;
    let head_len = keep / 2;
    let tail_len = keep - head_len;
    let head: String = name.chars().take(head_len).collect();
    let tail: String = name
        .chars()
        .rev()
        .take(tail_len)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}…{tail}")
}

/// Git-style aliases for the HEAD pointer. `HEAD` matches case-insensitively
/// because `head` would collide with it on case-insensitive filesystems.
fn is_head_alias(name: &str) -> bool {
    let trimmed = name.trim();
    trimmed.eq_ignore_ascii_case("head") || trimmed == "@" || trimmed == "-"
}

/// Alias for the newest checkpoint (matches the `[latest]` tag in the tree).
/// Case-insensitive, like `HEAD`. Keep in sync with
/// `history::checkpoint::is_reserved_checkpoint_name`, which rejects it as a
/// real checkpoint name so the alias can never collide with one.
fn is_latest_alias(name: &str) -> bool {
    name.trim().eq_ignore_ascii_case("latest")
}

/// Refresh the checkpoint and branch completion caches (best-effort):
/// completion sits on the hot path and cannot rescan the disk on every Tab,
/// so the driver publishes fresh names before each prompt and after mutations.
pub(super) fn publish_checkpoint_names(store: &CheckpointStore) {
    let names = store
        .list()
        .map(|checkpoints| checkpoints.into_iter().map(|cp| cp.name).collect())
        .unwrap_or_default();
    crate::ai::prompt::completion::CommandCompleter::set_checkpoint_name_candidates(names);
    let branches = store
        .list_branches()
        .map(|branches| branches.into_iter().map(|branch| branch.name).collect())
        .unwrap_or_default();
    crate::ai::prompt::completion::CommandCompleter::set_checkpoint_branch_candidates(branches);
}

#[derive(Debug, Default, PartialEq, Eq)]
struct NavigationArgs {
    target: Option<String>,
    create_branch: Option<String>,
    discard: bool,
    explicit_save: Option<String>,
}

/// Parse Git-style `checkout` / `switch` arguments while retaining the
/// checkpoint-specific dirty-history preservation flags.
fn parse_navigation_args(args: &[String], create_flag: &str) -> Result<NavigationArgs, String> {
    let mut parsed = NavigationArgs::default();
    let mut target: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--discard" {
            parsed.discard = true;
        } else if arg == create_flag || arg == "--create" {
            let Some(name) = args.get(i + 1).filter(|name| !name.starts_with('-')) else {
                return Err(format!("{arg} requires a branch name"));
            };
            parsed.create_branch = Some(name.clone());
            i += 1;
        } else if arg == "--save" || arg == "--as" {
            if i + 1 < args.len() {
                parsed.explicit_save = Some(args[i + 1].clone());
                i += 1;
            }
        } else if let Some(value) = arg.strip_prefix("--save=") {
            if !value.is_empty() {
                parsed.explicit_save = Some(value.to_string());
            }
        } else if let Some(value) = arg.strip_prefix("--as=") {
            if !value.is_empty() {
                parsed.explicit_save = Some(value.to_string());
            }
        } else if arg.len() > 1 && arg.starts_with('-') {
            return Err(format!("unknown option '{arg}'"));
        } else if target.is_none() {
            target = Some(args[i].clone());
        } else if parsed.explicit_save.is_none() {
            // A second positional token acts as the preserve name for users coming
            // from `git stash`-style habits: `checkout <target> <save-as>`.
            parsed.explicit_save = Some(args[i].clone());
        }
        i += 1;
    }
    parsed.target = target;
    Ok(parsed)
}

#[derive(Debug, PartialEq, Eq)]
enum SwitchPreparation {
    Proceed,
    Preserve(String),
    Discard,
    RefuseDirty,
}

fn switch_preparation(
    live_len: usize,
    live_is_clean: bool,
    discard: bool,
    explicit_save: Option<String>,
) -> SwitchPreparation {
    if let Some(name) = explicit_save.filter(|_| live_len > 0) {
        return SwitchPreparation::Preserve(name);
    }
    if live_len == 0 || live_is_clean {
        SwitchPreparation::Proceed
    } else if discard {
        SwitchPreparation::Discard
    } else {
        SwitchPreparation::RefuseDirty
    }
}

fn switch_to_unborn_branch(
    app: &mut App,
    store: &CheckpointStore,
    branch: &str,
    explicit_save: Option<String>,
    discard_requested: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    if matches!(
        store.head_state(),
        Ok(CheckpointHead::Attached {
            branch: current,
            target: None,
        }) if current == branch
    ) {
        println!("Already on unborn branch '{branch}': nothing was changed.");
        return Ok(true);
    }

    let session_store = SessionStore::new(app.config.history_file.as_path());
    let live_messages = match session_store.read_all_messages(&app.session_id) {
        Ok(messages) => messages,
        Err(err) => {
            println!("Failed to switch branches: cannot read live history: {err}");
            return Ok(true);
        }
    };
    let live_len = live_messages.len();
    let live_is_clean = match store.live_matches_base(&live_messages) {
        Ok(matches) => matches,
        Err(err) => {
            println!(
                "Failed to switch branches: cannot verify whether live history is clean: {err}"
            );
            return Ok(true);
        }
    };
    let preparation = switch_preparation(
        live_len,
        live_is_clean,
        discard_requested && explicit_save.is_none(),
        explicit_save,
    );
    let discarding = matches!(preparation, SwitchPreparation::Discard);
    let preserved_as = match preparation {
        SwitchPreparation::Proceed | SwitchPreparation::Discard => None,
        SwitchPreparation::Preserve(name) => {
            if let Err(err) = save_preserving_current(store, &name) {
                println!("Failed to checkout: current history was NOT overwritten. {err}");
                return Ok(true);
            }
            Some(name)
        }
        SwitchPreparation::RefuseDirty => {
            println!(
                "Refusing to switch: {live_len} un-checkpointed live message(s) would be lost. Use `--save <name>` to preserve them or `--discard` to overwrite intentionally."
            );
            return Ok(true);
        }
    };
    if discarding && live_len > 0 {
        let confirm = prompt_checkpoint_confirmation(&format!(
            "Discard {live_len} un-checkpointed live message(s) and switch to unborn branch '{branch}'? (y/n): "
        ));
        if confirm != Some(true) {
            println!("Canceled: nothing was changed.");
            return Ok(true);
        }
    }

    match store.switch_branch(branch) {
        Ok(()) => {
            refresh_runtime_after_rollback(app);
            publish_checkpoint_names(store);
            match preserved_as {
                Some(saved) => println!(
                    "Switched to unborn branch '{branch}'. Previous history preserved as '{saved}'."
                ),
                None => println!(
                    "Switched to unborn branch '{branch}'. The next save will create its first checkpoint."
                ),
            }
        }
        Err(err) => println!("Failed to switch to branch '{branch}': {err}"),
    }
    Ok(true)
}

/// Save the live history under `preserve_name`. Never evicts: dropping history
/// is only allowed through explicit `delete` or `--discard`, so a full budget
/// aborts the checkout instead and tells the user how to make room. Returns an
/// explanatory message on failure so the caller can abort without data loss.
fn save_preserving_current(store: &CheckpointStore, preserve_name: &str) -> Result<(), String> {
    match store.save(preserve_name) {
        Ok(_) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::StorageFull => Err(format!(
            "save '{preserve_name}' failed: checkpoint budget full ({err}). \
             Delete a checkpoint first, or rerun with --discard."
        )),
        Err(err) => Err(format!("save '{preserve_name}' failed: {err}")),
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
                eprintln!("[checkpoint] failed to rebuild patch ledger after checkout: {err}");
            }
        }
        Err(err) => {
            eprintln!("[checkpoint] failed to read restored history after checkout: {err}");
        }
    }
    if let Err(err) = super::session::restore_prune_marks_for_history(app) {
        eprintln!("[checkpoint] failed to restore prune marks after checkout: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SwitchPreparation, effective_head, forest_order, format_human_bytes, is_head_alias,
        is_latest_alias, parse_navigation_args, switch_preparation, truncate_middle,
    };
    use crate::ai::history::CheckpointInfo;

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    fn forest_info(name: &str, parent: Option<&str>) -> CheckpointInfo {
        CheckpointInfo {
            name: name.to_string(),
            modified_local: None,
            size_bytes: 0,
            parent: parent.map(str::to_string),
        }
    }

    fn forest_plan_at(
        checkpoints: &[CheckpointInfo],
        live_base: Option<&str>,
    ) -> Vec<(String, String)> {
        forest_order(checkpoints, live_base)
            .iter()
            .map(|row| (checkpoints[row.index].name.clone(), row.prefix.clone()))
            .collect()
    }

    fn forest_plan(checkpoints: &[CheckpointInfo]) -> Vec<(String, String)> {
        forest_plan_at(checkpoints, None)
    }

    #[test]
    fn checkout_args_support_flags_and_positional_save_as() {
        let parsed = parse_navigation_args(&args(&["c1"]), "-b").unwrap();
        assert_eq!(parsed.target.as_deref(), Some("c1"));
        assert!(!parsed.discard);
        assert!(parsed.explicit_save.is_none());

        let parsed = parse_navigation_args(&args(&["c1", "--discard"]), "-b").unwrap();
        assert_eq!(parsed.target.as_deref(), Some("c1"));
        assert!(parsed.discard);
        assert!(parsed.explicit_save.is_none());

        let parsed = parse_navigation_args(&args(&["c1", "--save", "mywork"]), "-b").unwrap();
        assert_eq!(parsed.target.as_deref(), Some("c1"));
        assert!(!parsed.discard);
        assert_eq!(parsed.explicit_save.as_deref(), Some("mywork"));

        // Second positional token acts as the preserve name.
        let parsed = parse_navigation_args(&args(&["c1", "mywork"]), "-b").unwrap();
        assert_eq!(parsed.target.as_deref(), Some("c1"));
        assert!(!parsed.discard);
        assert_eq!(parsed.explicit_save.as_deref(), Some("mywork"));

        for removed_alias in ["--no-save", "--hard"] {
            assert!(
                parse_navigation_args(&args(&["c1", removed_alias]), "-b").is_err(),
                "removed flag {removed_alias} must be rejected"
            );
        }
    }

    #[test]
    fn navigation_args_support_git_style_branch_creation() {
        let checkout = parse_navigation_args(&args(&["-b", "feature", "main"]), "-b").unwrap();
        assert_eq!(checkout.create_branch.as_deref(), Some("feature"));
        assert_eq!(checkout.target.as_deref(), Some("main"));

        let switch = parse_navigation_args(&args(&["-c", "feature"]), "-c").unwrap();
        assert_eq!(switch.create_branch.as_deref(), Some("feature"));
        assert!(switch.target.is_none());

        assert!(parse_navigation_args(&args(&["-c"]), "-c").is_err());
        assert!(parse_navigation_args(&args(&["-b", "feature"]), "-c").is_err());
    }

    #[test]
    fn dirty_switch_requires_an_explicit_save_or_discard() {
        assert_eq!(
            switch_preparation(2, false, false, None),
            SwitchPreparation::RefuseDirty
        );
        assert_eq!(
            switch_preparation(2, false, false, Some("work-in-progress".to_string())),
            SwitchPreparation::Preserve("work-in-progress".to_string())
        );
        assert_eq!(
            switch_preparation(2, false, true, None),
            SwitchPreparation::Discard
        );
        assert_eq!(
            switch_preparation(2, true, false, None),
            SwitchPreparation::Proceed
        );
    }

    #[test]
    fn checkout_args_accept_head_aliases_as_target() {
        for alias in ["HEAD", "head", "@", "-"] {
            let parsed = parse_navigation_args(&args(&[alias]), "-b").unwrap();
            assert_eq!(parsed.target.as_deref(), Some(alias));
            assert!(is_head_alias(&parsed.target.unwrap()));
        }
        assert!(is_head_alias("HEAD"));
        assert!(is_head_alias("@"));
        assert!(is_head_alias("-"));
        assert!(!is_head_alias("c1"));
        assert!(!is_head_alias("headspace"));
        assert!(!is_head_alias("latest"));
    }

    #[test]
    fn latest_alias_matches_newest_marker_case_insensitively() {
        for alias in ["latest", "LATEST", "Latest", "  latest  "] {
            let parsed = parse_navigation_args(&args(&[alias]), "-b").unwrap();
            assert!(is_latest_alias(&parsed.target.unwrap()));
        }
        assert!(is_latest_alias("latest"));
        assert!(!is_latest_alias("c1"));
        assert!(!is_latest_alias("latest-x"));
        // Reserved so it can never collide with a real checkpoint name.
        assert!(crate::ai::history::is_reserved_checkpoint_name("latest"));
        assert!(crate::ai::history::is_reserved_checkpoint_name("LATEST"));
    }

    #[test]
    fn human_bytes_use_mib_units() {
        assert_eq!(format_human_bytes(0), "0 B");
        assert_eq!(format_human_bytes(512), "512 B");
        assert_eq!(format_human_bytes(2041746), "1.9 MiB");
    }

    #[test]
    fn long_names_truncate_in_the_middle() {
        let name = "before_check_wangwenchao_129_fix_pop-date-shift-online-2_108_0";
        let short = truncate_middle(name, 48);
        assert!(short.chars().count() <= 48);
        assert!(short.contains('…'));
        assert!(short.ends_with("2_108_0"));
    }

    #[test]
    fn forest_linear_chain_stays_flat() {
        // Newest-first input: a child of b child of root c. A straight
        // save-after-save sequence reads as flat peers with no indent and
        // no connector, not nested levels.
        let checkpoints = vec![
            forest_info("a", Some("b")),
            forest_info("b", Some("c")),
            forest_info("c", None),
        ];
        assert_eq!(
            forest_plan(&checkpoints),
            vec![
                ("c".to_string(), "".to_string()),
                ("b".to_string(), "".to_string()),
                ("a".to_string(), "".to_string()),
            ]
        );
    }

    #[test]
    fn forest_live_base_exposes_single_saved_child_as_a_fork() {
        // `fork before_edit --save other_edit` preserves the edited line as
        // one saved child, then leaves live history on the parent as the
        // other side of the fork.
        let checkpoints = vec![
            forest_info("other_edit", Some("before_edit")),
            forest_info("before_edit", None),
        ];
        assert_eq!(
            forest_plan_at(&checkpoints, Some("before_edit")),
            vec![
                ("before_edit".to_string(), "".to_string()),
                ("other_edit".to_string(), "└── ".to_string()),
            ]
        );
    }

    #[test]
    fn forest_branch_keeps_children_newest_first() {
        // Newest-first input: x and y both branch from r.
        let checkpoints = vec![
            forest_info("x", Some("r")),
            forest_info("y", Some("r")),
            forest_info("r", None),
        ];
        assert_eq!(
            forest_plan(&checkpoints),
            vec![
                ("r".to_string(), "".to_string()),
                ("x".to_string(), "├── ".to_string()),
                ("y".to_string(), "└── ".to_string()),
            ]
        );
    }

    #[test]
    fn forest_fork_child_continues_inside_its_branch() {
        // Newest-first input: c and b both branch from a, d continues b.
        // The lone b -> d link inherits b's branch indent with no extra
        // connector of its own.
        let checkpoints = vec![
            forest_info("d", Some("b")),
            forest_info("c", Some("a")),
            forest_info("b", Some("a")),
            forest_info("a", None),
        ];
        assert_eq!(
            forest_plan(&checkpoints),
            vec![
                ("a".to_string(), "".to_string()),
                ("c".to_string(), "├── ".to_string()),
                ("b".to_string(), "└── ".to_string()),
                ("d".to_string(), "    ".to_string()),
            ]
        );
    }

    #[test]
    fn forest_treats_orphan_and_self_link_as_roots() {
        let checkpoints = vec![
            forest_info("orphan", Some("deleted")),
            forest_info("loop", Some("loop")),
            forest_info("root", None),
        ];
        // Every entry shows exactly once, all as roots.
        assert_eq!(
            forest_plan(&checkpoints),
            vec![
                ("orphan".to_string(), "".to_string()),
                ("loop".to_string(), "".to_string()),
                ("root".to_string(), "".to_string()),
            ]
        );
    }

    #[test]
    fn forest_breaks_parent_cycles_without_dropping_entries() {
        // Corrupt cycle with no root: both must still show exactly once.
        let checkpoints = vec![forest_info("a", Some("b")), forest_info("b", Some("a"))];
        let plan = forest_plan(&checkpoints);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].0, "a");
        assert_eq!(plan[0].1, "");
        assert_eq!(plan[1].0, "b");
    }

    #[test]
    fn effective_head_never_dangles_when_checkpoints_exist() {
        let checkpoints = vec![forest_info("new", None), forest_info("old", None)];
        // Stored pointer wins when it still names a row.
        assert_eq!(
            effective_head(&checkpoints, Some("old"), Some("new")),
            Some("old")
        );
        // Missing pointer falls back to the live base.
        assert_eq!(effective_head(&checkpoints, None, Some("old")), Some("old"));
        // Dangling pointer and dangling base fall back to newest (index 0).
        assert_eq!(
            effective_head(&checkpoints, Some("gone"), Some("gone")),
            Some("new")
        );
        assert_eq!(effective_head(&checkpoints, None, None), Some("new"));
        // No checkpoints: genuinely unset.
        let empty: Vec<CheckpointInfo> = vec![];
        assert_eq!(effective_head(&empty, None, None), None);
    }
}
