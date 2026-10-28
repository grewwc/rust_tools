//! Interactive `/git` command: immutable commits,
//! movable branches, checkout, tree, and deletion for the current session.

use crate::ai::{
    history::{
        CheckpointHead, CheckpointInfo, CheckpointStore, MAX_CHECKPOINT_STORAGE_BYTES,
        MAX_CHECKPOINTS_PER_SESSION, Message, SessionStore,
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

#[derive(Debug, PartialEq, Eq)]
struct SaveArgs {
    name: Option<String>,
    message: Option<String>,
}

fn parse_save_args(input: &str) -> Result<SaveArgs, String> {
    let uses_message_option = input
        .split_whitespace()
        .any(|arg| matches!(arg, "-m" | "--message") || arg.starts_with("--message="));
    if !uses_message_option {
        return Ok(SaveArgs {
            // Preserve the legacy contract: the first whitespace-delimited
            // token is the name and any later tokens are ignored.
            name: input.split_whitespace().next().map(str::to_string),
            message: None,
        });
    }

    let words = split_quoted_args(input, "commit arguments")?;
    let mut name = None;
    let mut messages = Vec::new();
    let mut index = 0;
    while index < words.len() {
        let word = &words[index];
        if matches!(word.as_str(), "-m" | "--message") {
            index += 1;
            let Some(message) = words.get(index) else {
                return Err(format!("option '{word}' requires a message"));
            };
            messages.push(message.clone());
        } else if let Some(message) = word.strip_prefix("--message=") {
            messages.push(message.to_string());
        } else if name.is_none() {
            name = Some(word.clone());
        }
        index += 1;
    }
    if messages.iter().any(|message| message.trim().is_empty()) {
        return Err("commit message cannot be empty".to_string());
    }
    Ok(SaveArgs {
        name,
        message: Some(messages.join("\n\n")),
    })
}

/// Split command arguments on whitespace, honoring single/double quotes
/// and `\` escapes (same rules everywhere it is used). Inputs without
/// quote or backslash characters split exactly like `split_whitespace`.
/// `args_what` names the argument list in error messages (e.g.
/// "commit arguments").
fn split_quoted_args(input: &str, args_what: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quote = None;
    let mut in_word = false;
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if let Some(delimiter) = quote {
            if ch == delimiter {
                quote = None;
            } else if ch == '\\' && delimiter == '"' {
                word.push(
                    chars
                        .next()
                        .ok_or_else(|| format!("trailing escape in {args_what}"))?,
                );
            } else {
                word.push(ch);
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                in_word = true;
            }
            '\\' => {
                word.push(
                    chars
                        .next()
                        .ok_or_else(|| format!("trailing escape in {args_what}"))?,
                );
                in_word = true;
            }
            ch if ch.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            _ => {
                word.push(ch);
                in_word = true;
            }
        }
    }
    if quote.is_some() {
        return Err(format!("unterminated quote in {args_what}"));
    }
    if in_word {
        words.push(word);
    }
    Ok(words)
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
    if cmd != "git" {
        return Ok(false);
    }

    let action = parts.next().unwrap_or("list");
    let store = CheckpointStore::new(app.config.history_file.as_path(), &app.session_id);

    match action {
        "help" | "h" => {
            print_help();
        }
        "save" | "commit" | "s" => {
            let raw_args = normalized
                .split_once(action)
                .map(|(_, rest)| rest.trim_start())
                .unwrap_or_default();
            let parsed = match parse_save_args(raw_args) {
                Ok(parsed) => parsed,
                Err(err) => {
                    println!("Failed to save checkpoint: {err}");
                    return Ok(true);
                }
            };
            let name = parsed
                .name
                .unwrap_or_else(|| Local::now().format("commit-%Y%m%d-%H%M%S-%3f").to_string());
            match store.save_with_message(&name, parsed.message.as_deref()) {
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
            Some("-b" | "-c" | "--create") => {
                let rest: Vec<&str> = parts.collect();
                if rest.is_empty() {
                    println!("Usage: /git branch -b <new-branch> [start]");
                    return Ok(true);
                }
                let mut rewritten = String::from("/git checkout -b");
                for token in rest {
                    rewritten.push(' ');
                    rewritten.push_str(token);
                }
                return try_handle_checkpoint_command(app, &rewritten);
            }
            Some("-d" | "--delete" | "delete") => {
                let Some(name) = parts.next() else {
                    println!("Usage: /git branch --delete <name>");
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
                        println!("Created branch '{name}'. Switch with `/git switch {name}`.")
                    }
                    Err(err) => println!("Failed to create branch: {err}"),
                }
            }
        },
        "checkout" | "co" | "switch" => {
            // Navigation requires a clean live tree by default; callers must
            // explicitly save or discard un-checkpointed work.
            let is_switch = action == "switch";
            // Quote-aware splitting (same splitter `/git commit` uses, same
            // `split_once` remainder trick as the save arm above):
            // `--save "my work"` is one name, not three tokens.
            let raw_args = normalized
                .split_once(action)
                .map(|(_, rest)| rest.trim_start())
                .unwrap_or_default();
            let args = match split_quoted_args(
                raw_args,
                if is_switch {
                    "switch arguments"
                } else {
                    "checkout arguments"
                },
            ) {
                Ok(args) => args,
                Err(err) => {
                    println!("Invalid {action} arguments: {err}");
                    return Ok(true);
                }
            };
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
                    println!("Usage: /git switch <branch> | /git switch -c <new-branch> [start]");
                } else {
                    println!(
                        "Usage: /git checkout <branch|checkpoint> | /git checkout -b <new-branch> [start]"
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
                            let result = switch_to_unborn_branch(
                                app,
                                &store,
                                new_branch,
                                explicit_save,
                                discard_requested,
                            );
                            let switched = matches!(
                                store.head_state(),
                                Ok(CheckpointHead::Attached { branch, .. }) if branch == new_branch
                            );
                            if !switched {
                                if let Err(err) = store.delete_branch(new_branch) {
                                    eprintln!(
                                        "[checkpoint] failed to clean up branch '{new_branch}' after switch failure: {err}"
                                    );
                                }
                                publish_checkpoint_names(&store);
                            }
                            return result;
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
            // `-` resolves to the previously checked-out branch or checkpoint
            // (Git-style `switch -` / `checkout -`).
            let previous_requested = is_previous_alias(&raw_target);
            let matching_branch = (!head_requested && !latest_requested && !previous_requested)
                .then(|| store.list_branches().ok())
                .flatten()
                .and_then(|branches| {
                    branches
                        .into_iter()
                        .find(|branch| branch.name == raw_target)
                });
            // A `-` attached to a branch reuses the normal branch path below;
            // a `-` attached to a detached checkpoint only `checkout` (and
            // branch creation from it) can follow it back.
            let mut previous_branch = None;
            let mut previous_detached: Option<String> = None;
            if previous_requested {
                match store.previous_head_state() {
                    Err(err) => {
                        println!("Failed to checkout: cannot read previous checkout: {err}");
                        return Ok(true);
                    }
                    Ok(None) => {
                        println!("No previous branch to switch to yet.");
                        return Ok(true);
                    }
                    Ok(Some(CheckpointHead::Attached { branch, .. })) => {
                        match store
                            .list_branches()
                            .map(|branches| branches.into_iter().find(|item| item.name == branch))
                        {
                            Ok(Some(found)) => previous_branch = Some(found),
                            _ => {
                                println!(
                                    "Previous branch '{branch}' no longer exists (nothing was changed)."
                                );
                                return Ok(true);
                            }
                        }
                    }
                    Ok(Some(CheckpointHead::Detached(target))) => match target {
                        Some(target) => {
                            if is_switch && create_branch.is_none() {
                                println!(
                                    "Previous checkout was detached at '{target}': `switch` only moves between branches; use `/git checkout -` to go back to it."
                                );
                                return Ok(true);
                            }
                            previous_detached = Some(target);
                        }
                        None => {
                            println!("No previous branch to switch to yet.");
                            return Ok(true);
                        }
                    },
                }
            }
            if is_switch
                && create_branch.is_none()
                && matching_branch.is_none()
                && previous_branch.is_none()
            {
                println!("Failed to switch: branch '{raw_target}' not found.");
                return Ok(true);
            }
            let branch_name = create_branch
                .clone()
                .or_else(|| matching_branch.as_ref().map(|branch| branch.name.clone()))
                .or_else(|| previous_branch.as_ref().map(|branch| branch.name.clone()));
            let name = if let Some(branch) = matching_branch.or(previous_branch) {
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
                    println!("No checkpoints yet: save one first (`/git save <name>`).");
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
                        println!("No checkpoints yet: save one first (`/git save <name>`).");
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
                            println!("No checkpoints yet: save one first (`/git save <name>`).");
                            return Ok(true);
                        }
                    },
                    Err(err) => {
                        println!("Failed to checkout: cannot list checkpoints: {err}");
                        return Ok(true);
                    }
                }
            } else if let Some(target) = previous_detached {
                // Resolved straight from the stored previous pointer, so the
                // target trivially exists; the check below is skipped for it.
                target
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
            // `latest`, `HEAD`, and `-` resolve straight from stored state
            // above, so their targets trivially exist and skip this check.
            if !latest_requested && !head_requested && !previous_requested {
                let exists = match store.checkpoint_exists(&name) {
                    Ok(exists) => exists,
                    Err(err) => {
                        println!("Failed to checkout: cannot list checkpoints: {err}");
                        return Ok(true);
                    }
                };
                if !exists {
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
                match checkpoint_switch_diff(&store, &live_messages, &name) {
                    Some((target_len, common)) => {
                        let dropped = live_len.saturating_sub(common);
                        println!(
                            "Cannot {operation} '{name}': live history has un-checkpointed changes \
                             (live {live_len}, target {target_len}, {common} shared; {dropped} would be dropped). \
                             Save them first with `/git save <name>`, or add `--discard` to discard them."
                        );
                    }
                    None => println!(
                        "Cannot {operation} '{name}': live history has un-checkpointed changes. \
                         Save them first with `/git save <name>`, or add `--discard` to discard them."
                    ),
                }
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
                    let prompt = match checkpoint_switch_diff(&store, &live_messages, &name) {
                        Some((target_len, common)) => {
                            let dropped = live_len.saturating_sub(common);
                            let restored = target_len.saturating_sub(common);
                            format!(
                                "Live history has un-checkpointed changes \
                                 (live {live_len}, target {target_len}, {common} shared). \
                                 Switch to '{name}'? This drops {dropped} live message(s) \
                                 not in '{name}' and restores {restored} from '{name}'? (y/n): "
                            )
                        }
                        None => format!(
                            "Live history has un-checkpointed changes. Replace all {live_len} \
                             live message(s) with '{name}'? (y/n): "
                        ),
                    };
                    let confirm = prompt_checkpoint_confirmation(&prompt);
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
                println!("Usage: /git delete <name>");
                return Ok(true);
            };
            if is_head_alias(&name) || is_latest_alias(&name) {
                println!(
                    "'{name}' is a pointer alias, not a checkpoint: delete the checkpoint it points to \
                     instead (see `/git tree`)."
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
            let total_bytes: u64 = checkpoints.iter().map(|cp| cp.storage_bytes).sum();
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
    println!("  /git commit [name] [-m <message>]");
    println!(
        "                                create an immutable commit; repeat -m for paragraphs"
    );
    println!("  /git save [name] [-m <message>]");
    println!("                                compatible alias; name and message are optional");
    println!(
        "  /git tree              show commit ancestry and branch pointers (default; `list` works too)"
    );
    println!("  /git checkout <branch|checkpoint> [--save <name> | --discard]");
    println!("                                switch to a branch or detach at a checkpoint; dirty");
    println!(
        "                                history is refused unless explicitly saved or discarded"
    );
    println!("  /git checkout -b <new-branch> [start]");
    println!(
        "                                create and switch to a branch (Git-style checkout -b)"
    );
    println!("  /git branch <name> [start]");
    println!("                                create a branch at start (or current HEAD)");
    println!("  /git branch -b <new-branch> [start]");
    println!("                                create and switch to a branch (same as checkout -b)");
    println!("  /git branch            list branches (* marks the current branch)");
    println!("  /git branch -d <name>  delete only the branch pointer");
    println!("  /git switch <branch>   restore and attach HEAD to a branch");
    println!("  /git switch -c <new-branch> [start]");
    println!("                                create and switch to a branch (Git-style switch -c)");
    println!("  /git delete <name>     delete a checkpoint");
    println!("  /git clear             delete ALL checkpoints (asks for confirmation first)");
    println!();
    println!("  HEAD / @ both mean the HEAD pointer (the current commit);");
    println!("  `-` means the previous checkout (`switch -` / `checkout -`, Git-style);");
    println!("  `latest` means the newest checkpoint (the `[latest]` row in the tree).");
    println!("  History is never dropped except by explicit `delete`, `clear`, or `--discard`.");
    println!(
        "  Limit: {} commits + branches / {} MiB per session; commit names cannot be overwritten.",
        MAX_CHECKPOINTS_PER_SESSION,
        MAX_CHECKPOINT_STORAGE_BYTES / (1024 * 1024)
    );
}

/// One commit and its outgoing parent edge in a newest-first topology graph.
struct CommitGraphRow {
    index: usize,
    prefix: String,
    suffix: String,
    connector: String,
}

/// Children precede parents, even when legacy timestamps disagree. Lanes
/// represent commit edges only: branch pointers, HEAD and unsaved live edits
/// cannot change the graph. Missing parents remain visible in row annotations.
fn commit_graph(checkpoints: &[CheckpointInfo]) -> Vec<CommitGraphRow> {
    let parents: Vec<_> = checkpoints
        .iter()
        .enumerate()
        .map(|(i, cp)| {
            checkpoints.iter().enumerate().find_map(|(j, parent)| {
                (i != j && Some(parent.name.as_str()) == cp.parent.as_deref()).then_some(j)
            })
        })
        .collect();
    let mut rows = Vec::with_capacity(checkpoints.len());
    let mut visited = vec![false; checkpoints.len()];
    let mut lanes = Vec::new();
    while rows.len() < checkpoints.len() {
        let ready = (0..checkpoints.len()).find(|&i| {
            !visited[i]
                && !parents
                    .iter()
                    .enumerate()
                    .any(|(child, parent)| !visited[child] && *parent == Some(i))
        });
        // Corrupt cycles must not hide commits or prevent the listing.
        let index = ready.unwrap_or_else(|| visited.iter().position(|seen| !seen).unwrap());
        visited[index] = true;
        let parent = parents[index].filter(|&i| !visited[i]);
        rows.push(commit_graph_row(index, parent, &mut lanes));
    }
    rows
}

/// Keep each pending parent in a stable lane; siblings converge onto its lane.
fn commit_graph_row(
    index: usize,
    parent: Option<usize>,
    lanes: &mut Vec<Option<usize>>,
) -> CommitGraphRow {
    let lane = lanes
        .iter()
        .position(|entry| *entry == Some(index))
        .or_else(|| lanes.iter().position(Option::is_none))
        .unwrap_or_else(|| {
            lanes.push(None);
            lanes.len() - 1
        });
    lanes[lane] = Some(index);
    let strokes = |entries: &[Option<usize>]| -> String {
        entries
            .iter()
            .map(|entry| if entry.is_some() { "│ " } else { "  " })
            .collect()
    };
    let prefix = strokes(&lanes[..lane]);
    let suffix = strokes(&lanes[lane + 1..]);
    lanes[lane] = None;
    let destination = parent.and_then(|p| lanes.iter().position(|entry| *entry == Some(p)));
    if destination.is_none() {
        lanes[lane] = parent;
    }
    let mut connector: Vec<char> = strokes(lanes).chars().collect();
    if let Some(destination) = destination {
        let left = lane.min(destination) * 2;
        let right = lane.max(destination) * 2;
        for cell in &mut connector[left + 1..right] {
            *cell = if *cell == '│' { '┼' } else { '─' };
        }
        connector[left] = if lane < destination { '╰' } else { '├' };
        connector[right] = if lane < destination { '┤' } else { '╯' };
    }
    while lanes.last() == Some(&None) {
        lanes.pop();
    }
    CommitGraphRow {
        index,
        prefix,
        suffix,
        connector: connector
            .into_iter()
            .collect::<String>()
            .trim_end()
            .to_string(),
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
    let live_messages = SessionStore::new(app.config.history_file.as_path())
        .read_all_messages(&app.session_id)
        .unwrap_or_default();
    let live_len = live_messages.len();
    let total_size: u64 = checkpoints
        .iter()
        .map(|cp| cp.storage_bytes)
        .fold(0u64, |acc, size| acc.saturating_add(size));
    println!(
        "{} ({} / {} checkpoints, {} stored / {} MiB)",
        "Checkpoints".bold(),
        checkpoints.len(),
        MAX_CHECKPOINTS_PER_SESSION,
        format_human_bytes(total_size).bold(),
        MAX_CHECKPOINT_STORAGE_BYTES / (1024 * 1024)
    );
    // Live is working state, not a commit or an extra edge in the graph.
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
    // Every branch pointer gets one line — including unborn branches (no
    // checkpoint yet) and branches whose target was deleted — because the
    // per-row `[branch:…]` tags below only cover branches that still point
    // at a listed checkpoint.
    if !branches.is_empty() {
        println!(
            "  {}",
            format!("Branch pointers ({})", branches.len()).dimmed()
        );
        let rows = branch_rows(&branches, active_branch.as_deref());
        let name_width = rows
            .iter()
            .map(|row| truncate_middle(&row.name, 32).chars().count())
            .max()
            .unwrap_or(4)
            .max(4);
        for row in &rows {
            let short = truncate_middle(&row.name, 32);
            let pad = " ".repeat(name_width - short.chars().count());
            let marker = if row.active {
                "*".cyan().bold().to_string()
            } else {
                " ".to_string()
            };
            let name_cell = if row.active {
                short.cyan().bold().to_string()
            } else {
                short.bold().to_string()
            };
            let target_cell = if row.target_gone {
                format!("{} {}", row.target, "(missing checkpoint)".yellow())
            } else {
                row.target.dimmed().to_string()
            };
            println!("    {marker} {name_cell}{pad} -> {target_cell}");
        }
    }
    if checkpoints.is_empty() {
        println!(
            "  {}",
            "(no checkpoints yet — save one with /git save <name>)".dimmed()
        );
        return Ok(());
    }
    // Commit topology is independent of the live state and branch pointers.
    let order = commit_graph(&checkpoints);
    let short_names: Vec<String> = checkpoints
        .iter()
        .map(|cp| truncate_middle(&cp.name, 48))
        .collect();
    let label_width = order
        .iter()
        .map(|row| {
            row.prefix.chars().count()
                + row.suffix.chars().count()
                + 2
                + short_names[row.index].chars().count()
        })
        .max()
        .unwrap_or(12)
        .max(12);
    println!("  {}", "Commits (children above parents)".dimmed());
    println!(
        "  {}",
        format!(
            "{}{}  {:>9}  {:<19}  {}",
            "COMMIT",
            " ".repeat(label_width - "COMMIT".len()),
            "SIZE",
            "SAVED",
            "TAGS",
        )
        .dimmed()
    );
    // Latest follows save time, not the topological display position.
    for row in &order {
        let cp = &checkpoints[row.index];
        let when = cp
            .modified_local
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "-".to_string());
        let short_name = &short_names[row.index];
        let is_head = head.as_deref() == Some(cp.name.as_str());
        let is_latest = row.index == 0;
        let glyph = if is_head {
            "→".cyan().bold()
        } else {
            "○".dimmed()
        };
        let name_pad = " ".repeat(
            label_width
                - row.prefix.chars().count()
                - row.suffix.chars().count()
                - 2
                - short_name.chars().count(),
        );
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
        // Every row names the commit it builds on, so linear chains and
        // forks read the same way. A parent whose object is gone (deleted
        // after the child was saved) keeps its name with `(missing)` instead
        // of rendering the child as a false root.
        if let Some(parent) = cp.parent.as_deref() {
            let label = if checkpoints.iter().any(|other| other.name == parent) {
                format!("[parent:{}] ", truncate_middle(parent, 24))
            } else {
                format!("[parent:{} (missing)] ", truncate_middle(parent, 24))
            };
            tags.push_str(&label.dimmed().to_string());
        }
        if cp.name.starts_with("auto-") {
            tags.push_str(&"[auto] ".dimmed().to_string());
        }
        if let Some(subject) = checkpoint_message_subject(cp.message.as_deref()) {
            tags.push_str(&format!("— {subject}").bold().to_string());
        }
        println!(
            "  {}{} {}{}{}  {size_cell}  {when:<19}  {tags}",
            row.prefix.dimmed(),
            glyph,
            row.suffix.dimmed(),
            name_cell,
            name_pad,
        );
        if !row.connector.is_empty() {
            println!("  {}", row.connector.dimmed());
        }
    }
    println!(
        "  {}",
        "Lines link commits, not branch ownership; branch labels are movable pointers.".dimmed()
    );
    println!(
        "  {}",
        "→ HEAD · * active branch · [latest] newest save · [parent:x] direct parent".dimmed()
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

fn checkpoint_message_subject(message: Option<&str>) -> Option<String> {
    let line = message?.lines().find(|line| !line.trim().is_empty())?;
    // Persisted metadata is untrusted terminal text. Make controls visible
    // instead of letting ANSI escapes or other control bytes affect output.
    let safe_line: String = line
        .chars()
        .map(|ch| if ch.is_control() { '�' } else { ch })
        .collect();
    let subject = safe_line.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX_SUBJECT_CHARS: usize = 72;
    if subject.chars().count() <= MAX_SUBJECT_CHARS {
        return Some(subject);
    }
    let mut short: String = subject.chars().take(MAX_SUBJECT_CHARS - 1).collect();
    short.push('…');
    Some(short)
}

/// One printable branch-pointer row for the `Branches` block in `print_tree`.
/// `target` is the checkpoint name, or `(unborn)` when the branch has no
/// checkpoint yet; `target_gone` means the named checkpoint object is gone.
struct BranchRow {
    active: bool,
    name: String,
    target: String,
    target_gone: bool,
}

/// Flatten every branch pointer — active, idle, unborn, and dangling — so
/// the display never hides a branch just because its target is not a listed
/// checkpoint row.
fn branch_rows(
    branches: &[crate::ai::history::CheckpointBranch],
    active_branch: Option<&str>,
) -> Vec<BranchRow> {
    branches
        .iter()
        .map(|branch| {
            let (target, target_gone) = match branch.target.as_deref() {
                None => ("(unborn)".to_string(), false),
                Some(name) => (name.to_string(), branch.is_dangling),
            };
            BranchRow {
                active: active_branch == Some(branch.name.as_str()),
                name: branch.name.clone(),
                target,
                target_gone,
            }
        })
        .collect()
}

/// Git-style aliases for the HEAD pointer. `HEAD` matches case-insensitively
/// because `head` would collide with it on case-insensitive filesystems.
/// (`-` is the previous checkout, handled separately, so it is not a HEAD
/// alias even though it can never be a real checkpoint name either.)
fn is_head_alias(name: &str) -> bool {
    let trimmed = name.trim();
    trimmed.eq_ignore_ascii_case("head") || trimmed == "@"
}

/// Git-style `-`: the previously checked-out branch or checkpoint, for
/// `switch -` / `checkout -`. Keep in sync with
/// `history::checkpoint::is_reserved_checkpoint_name`, which rejects it as a
/// real checkpoint name so the alias can never collide with one.
fn is_previous_alias(name: &str) -> bool {
    name.trim() == "-"
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

/// Detail for checkout prompts: `(target_len, common_prefix_len)` of the
/// target checkpoint versus live. `None` when the target is unreadable so
/// callers fall back to the live total.
fn checkpoint_switch_diff(
    store: &CheckpointStore,
    live: &[Message],
    name: &str,
) -> Option<(usize, usize)> {
    store.checkpoint_live_diff(live, name).unwrap_or(None)
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
                "Refusing to switch: live history has un-checkpointed changes; all {live_len} live message(s) would be replaced. Use `--save <name>` to preserve them or `--discard` to overwrite intentionally."
            );
            return Ok(true);
        }
    };
    if discarding && live_len > 0 {
        let confirm = prompt_checkpoint_confirmation(&format!(
            "Live history has un-checkpointed changes. Replace all {live_len} live message(s) and switch to unborn branch '{branch}'? (y/n): "
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
        BranchRow, SwitchPreparation, branch_rows, checkpoint_message_subject, commit_graph,
        effective_head, format_human_bytes, is_head_alias, is_latest_alias, is_previous_alias,
        parse_navigation_args, parse_save_args, split_quoted_args, switch_preparation,
        truncate_middle,
    };
    use crate::ai::history::{CheckpointBranch, CheckpointInfo};

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn save_args_preserve_legacy_name_behavior_without_message_flags() {
        let parsed = parse_save_args("release ignored words").unwrap();
        assert_eq!(parsed.name.as_deref(), Some("release"));
        assert_eq!(parsed.message, None);
    }

    #[test]
    fn save_args_parse_git_style_messages_and_optional_names() {
        let parsed = parse_save_args(r#"-m "fix login flow""#).unwrap();
        assert_eq!(parsed.name, None);
        assert_eq!(parsed.message.as_deref(), Some("fix login flow"));

        let parsed = parse_save_args(r#"release -m "subject" -m 'details'"#).unwrap();
        assert_eq!(parsed.name.as_deref(), Some("release"));
        assert_eq!(parsed.message.as_deref(), Some("subject\n\ndetails"));

        let parsed = parse_save_args("--message=subject release").unwrap();
        assert_eq!(parsed.name.as_deref(), Some("release"));
        assert_eq!(parsed.message.as_deref(), Some("subject"));
    }

    /// The CLI folds a multi-word inline value back as `--message="..."`, so
    /// that shape must identify the option instead of becoming the name.
    #[test]
    fn save_args_recognize_quoted_inline_message() {
        let parsed = parse_save_args("--message=\"before split feature\"").unwrap();
        assert_eq!(parsed.name, None);
        assert_eq!(parsed.message.as_deref(), Some("before split feature"));

        let parsed = parse_save_args("release --message=\"before split feature\"").unwrap();
        assert_eq!(parsed.name.as_deref(), Some("release"));
        assert_eq!(parsed.message.as_deref(), Some("before split feature"));
    }

    #[test]
    fn save_args_reject_missing_empty_or_unterminated_messages() {
        assert!(parse_save_args("-m").is_err());
        assert!(parse_save_args(r#"-m """#).is_err());
        assert!(parse_save_args(r#"-m "unterminated"#).is_err());
    }

    #[test]
    fn commit_subject_uses_first_nonempty_line_and_truncates() {
        assert_eq!(
            checkpoint_message_subject(Some("\n  subject  with   spaces\n\nbody")).as_deref(),
            Some("subject with spaces")
        );
        let long = "x".repeat(100);
        let subject = checkpoint_message_subject(Some(&long)).unwrap();
        assert_eq!(subject.chars().count(), 72);
        assert!(subject.ends_with('…'));
        let escaped = checkpoint_message_subject(Some("\u{1b}[2Jspoof")).unwrap();
        assert_eq!(escaped, "�[2Jspoof");
        assert!(!escaped.chars().any(char::is_control));
    }

    fn forest_info(name: &str, parent: Option<&str>) -> CheckpointInfo {
        CheckpointInfo {
            name: name.to_string(),
            message: None,
            modified_local: None,
            size_bytes: 0,
            storage_bytes: 0,
            parent: parent.map(str::to_string),
        }
    }

    fn graph_text(checkpoints: &[CheckpointInfo]) -> String {
        let mut lines = Vec::new();
        for row in commit_graph(checkpoints) {
            lines.push(format!(
                "{}● {}{}",
                row.prefix, row.suffix, checkpoints[row.index].name
            ));
            if !row.connector.is_empty() {
                lines.push(row.connector);
            }
        }
        lines.join("\n")
    }

    fn test_branch(name: &str, target: Option<&str>, is_dangling: bool) -> CheckpointBranch {
        CheckpointBranch {
            name: name.to_string(),
            target: target.map(str::to_string),
            is_dangling,
        }
    }

    fn branch_plan(branches: &[CheckpointBranch], active: Option<&str>) -> Vec<BranchRow> {
        branch_rows(branches, active)
    }

    #[test]
    fn branch_rows_show_every_pointer_including_unborn_and_missing() {
        let branches = vec![
            test_branch("check", None, false),
            test_branch("check_1", Some("c2"), false),
            test_branch("list", Some("next_1"), true),
            test_branch("main", Some("f2"), false),
        ];
        let rows = branch_plan(&branches, Some("main"));
        assert_eq!(rows.len(), 4);
        // Unborn branch: no checkpoint yet, never reported as gone.
        assert_eq!(rows[0].target, "(unborn)");
        assert!(!rows[0].target_gone);
        assert!(!rows[0].active);
        // Idle branch on a live checkpoint.
        assert_eq!(rows[1].target, "c2");
        assert!(!rows[1].target_gone);
        assert!(!rows[1].active);
        // Dangling branch keeps the deleted target's name for diagnosis.
        assert_eq!(rows[2].target, "next_1");
        assert!(rows[2].target_gone);
        assert!(!rows[2].active);
        // Active branch is marked, not duplicated.
        assert_eq!(rows[3].target, "f2");
        assert!(!rows[3].target_gone);
        assert!(rows[3].active);
        assert_eq!(rows.iter().filter(|row| row.active).count(), 1);
    }

    #[test]
    fn branch_rows_without_active_branch_marks_nothing() {
        let branches = vec![test_branch("main", Some("f2"), false)];
        let rows = branch_plan(&branches, None);
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].active);
        assert_eq!(rows[0].target, "f2");
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
    fn split_quoted_args_groups_quoted_navigation_names() {
        // A quoted save name survives as one word end to end.
        let words = split_quoted_args(r#"main --save "my work""#, "checkout arguments").unwrap();
        assert_eq!(words, args(&["main", "--save", "my work"]));
        let parsed = parse_navigation_args(&words, "-b").unwrap();
        assert_eq!(parsed.target.as_deref(), Some("main"));
        assert_eq!(parsed.explicit_save.as_deref(), Some("my work"));

        // Single quotes and backslash escapes follow the commit rules.
        let words = split_quoted_args(r#"-b 'my branch' main"#, "checkout arguments").unwrap();
        assert_eq!(words, args(&["-b", "my branch", "main"]));
        let words = split_quoted_args(r"main --save my\ work", "switch arguments").unwrap();
        assert_eq!(words, args(&["main", "--save", "my work"]));

        // Quoteless input splits exactly like whitespace splitting.
        for plain in ["", "  ", "main", "  main   --discard  ", "-b feature main"] {
            let words = split_quoted_args(plain, "checkout arguments").unwrap();
            let expected: Vec<String> =
                plain.split_whitespace().map(str::to_string).collect();
            assert_eq!(words, expected, "plain input must split like before: {plain:?}");
        }

        // Malformed quoting is a clear error naming the argument list,
        // not a silently wrong name.
        let err = split_quoted_args("main --save \"oops", "checkout arguments").unwrap_err();
        assert!(err.contains("checkout arguments"), "unexpected error: {err}");
        assert!(split_quoted_args("main\\", "switch arguments").is_err());
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
        for alias in ["HEAD", "head", "@"] {
            let parsed = parse_navigation_args(&args(&[alias]), "-b").unwrap();
            assert_eq!(parsed.target.as_deref(), Some(alias));
            assert!(is_head_alias(&parsed.target.unwrap()));
        }
        assert!(is_head_alias("HEAD"));
        assert!(is_head_alias("@"));
        assert!(!is_head_alias("-"));
        assert!(!is_head_alias("c1"));
        assert!(!is_head_alias("headspace"));
        assert!(!is_head_alias("latest"));
    }

    #[test]
    fn dash_is_the_previous_checkout_alias_not_a_head_alias() {
        let parsed = parse_navigation_args(&args(&["-"]), "-b").unwrap();
        assert_eq!(parsed.target.as_deref(), Some("-"));
        assert!(is_previous_alias("-"));
        assert!(!is_previous_alias("HEAD"));
        assert!(!is_previous_alias("c1"));
        // Still reserved so it can never collide with a real checkpoint name.
        assert!(crate::ai::history::is_reserved_checkpoint_name("-"));
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
    fn commit_graph_linear_chain_connects_in_one_lane() {
        let checkpoints = vec![
            forest_info("a", Some("b")),
            forest_info("b", Some("c")),
            forest_info("c", None),
        ];
        assert_eq!(graph_text(&checkpoints), "● a\n│\n● b\n│\n● c");
    }

    #[test]
    fn commit_graph_orders_parents_after_children_despite_timestamp_order() {
        // Legacy save times can place a parent before its child in list().
        let checkpoints = vec![
            forest_info("before_edit", None),
            forest_info("other_edit", Some("before_edit")),
        ];
        assert_eq!(graph_text(&checkpoints), "● other_edit\n│\n● before_edit");
    }

    #[test]
    fn commit_graph_branch_pointers_do_not_create_forks() {
        let checkpoints = vec![forest_info("c1", Some("main1")), forest_info("main1", None)];
        let branches = vec![
            test_branch("check", Some("c1"), false),
            test_branch("main", Some("main1"), false),
        ];
        for active in [Some("main"), Some("check"), None] {
            let refs = branch_rows(&branches, active);
            assert_eq!(refs[0].target, "c1");
            assert_eq!(refs[1].target, "main1");
            assert_eq!(graph_text(&checkpoints), "● c1\n│\n● main1");
        }
    }

    #[test]
    fn commit_graph_siblings_converge_on_shared_parent() {
        // Newest-first input: x and y both branch from r.
        let checkpoints = vec![
            forest_info("x", Some("r")),
            forest_info("y", Some("r")),
            forest_info("r", None),
        ];
        assert_eq!(graph_text(&checkpoints), "● x\n│\n│ ● y\n├─╯\n● r");
    }

    #[test]
    fn commit_graph_interleaved_descendants_keep_their_lanes() {
        // Newest-first input: c and b both branch from a, d continues b.
        let checkpoints = vec![
            forest_info("d", Some("b")),
            forest_info("c", Some("a")),
            forest_info("b", Some("a")),
            forest_info("a", None),
        ];
        assert_eq!(
            graph_text(&checkpoints),
            "● d\n│\n│ ● c\n│ │\n● │ b\n╰─┤\n  ● a"
        );
    }

    #[test]
    fn commit_graph_keeps_orphans_and_self_links_visible() {
        let checkpoints = vec![
            forest_info("orphan", Some("deleted")),
            forest_info("loop", Some("loop")),
            forest_info("root", None),
        ];
        // Every entry shows exactly once, all as roots.
        assert_eq!(graph_text(&checkpoints), "● orphan\n● loop\n● root");
    }

    #[test]
    fn commit_graph_breaks_cycles_without_dropping_entries() {
        // Corrupt cycle with no root: both must still show exactly once.
        let checkpoints = vec![forest_info("a", Some("b")), forest_info("b", Some("a"))];
        let plan = commit_graph(&checkpoints);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].index, 0);
        assert_eq!(plan[1].index, 1);
        assert!(plan[1].connector.is_empty());
        assert!(commit_graph(&[]).is_empty());
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
