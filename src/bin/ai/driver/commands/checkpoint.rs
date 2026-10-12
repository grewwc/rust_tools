//! Interactive `/checkpoint` command (alias `/cp`): save / checkout / fork /
//! list / delete conversation-history checkpoints for the current session.

use crate::ai::{
    history::{
        CheckpointInfo, CheckpointStore, MAX_CHECKPOINT_STORAGE_BYTES, MAX_CHECKPOINTS_PER_SESSION,
        SessionStore,
    },
    types::App,
};
use colored::Colorize;

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
            let Some(name) = parts.next() else {
                println!("Usage: /checkpoint save <name>");
                return Ok(true);
            };
            match store.save(name) {
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
        "checkout" | "co" | "switch" | "fork" => {
            // `checkout` and `fork` require a clean live tree by default;
            // callers must explicitly save or discard un-checkpointed work.
            // `fork` is an intention-revealing switch whose next `save`
            // records the target as parent.
            let is_fork = action == "fork";
            let args: Vec<String> = parts.map(|s| s.to_string()).collect();
            let (target, discard_requested, explicit_save) = parse_checkout_args(&args);
            if is_fork && discard_requested {
                println!("Usage: /checkpoint fork <name|HEAD|latest> [--save <name>]");
                return Ok(true);
            }
            let Some(raw_target) = target else {
                if is_fork {
                    println!("Usage: /checkpoint fork <name|HEAD|latest> [--save <name>]");
                } else {
                    println!(
                        "Usage: /checkpoint checkout <name|HEAD|latest> [--save <name> | --discard]"
                    );
                }
                return Ok(true);
            };
            if discard_requested && explicit_save.is_some() {
                println!("Usage: --save <name> and --discard cannot be combined.");
                return Ok(true);
            }
            // Git-style HEAD: the current commit. Every save and every
            // successful switch moves it, so `HEAD` always means "where live
            // is checked out", never "where we came from".
            let head_requested = is_head_alias(&raw_target);
            // `latest` resolves to the newest checkpoint (list order is
            // newest-first, so it matches the `[latest]` tag in the tree).
            let latest_requested = is_latest_alias(&raw_target);
            let name = if head_requested {
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
            let mut live_is_clean = live_len == 0;
            if live_len > 0 {
                let base = match store.base_target() {
                    Ok(base) => base,
                    Err(err) => {
                        println!("Failed to switch checkpoints: cannot read BASE: {err}");
                        return Ok(true);
                    }
                };
                if let Some(base) = base {
                    match store.live_matches_checkpoint(&live_messages, &base) {
                        Ok(true) => {
                            live_is_clean = true;
                            if base == name {
                                already_on_target = true;
                            } else {
                                clean_base = Some(base);
                            }
                        }
                        Ok(false) => {}
                        Err(err) => {
                            println!(
                                "Failed to switch checkpoints: cannot verify whether live history is clean: {err}"
                            );
                            return Ok(true);
                        }
                    }
                }
            }
            let preparation = switch_preparation(live_len, live_is_clean, discard, explicit_save);
            if preparation == SwitchPreparation::RefuseDirty {
                let operation = if is_fork { "fork" } else { "checkout" };
                println!(
                    "Cannot {operation} '{name}': live history has un-checkpointed changes. \
                     Save them first with `/cp save <name>`, or use `/cp checkout {name} --discard` to discard them."
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
                if let Err(err) = store.set_head(&name) {
                    eprintln!("[checkpoint] already on '{name}' but failed to move HEAD: {err}");
                }
                match is_fork {
                    true => println!(
                        "Fork base is already '{name}'. The next save will create a child checkpoint."
                    ),
                    false => println!(
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
                    let confirm = crate::commonw::prompt::prompt_yes_or_no_danger(&format!(
                        "Discard {live_len} un-checkpointed live message(s) and switch to '{name}'? (y/n): ",
                    ));
                    if confirm != Some(true) {
                        println!("Canceled: nothing was changed.");
                        return Ok(true);
                    }
                }
            }
            let verb = if is_fork {
                "Forked from"
            } else {
                "Checked out"
            };
            let fork_hint = if is_fork {
                " The next save will create a child checkpoint."
            } else {
                ""
            };
            match store.rollback(&name) {
                Ok(()) => {
                    refresh_runtime_after_rollback(app);
                    match (preserved_as, clean_base) {
                        (Some(saved), _) => println!(
                            "{verb} '{name}'. Previous history preserved as '{saved}'.{fork_hint}"
                        ),
                        (None, Some(base)) => println!(
                            "{verb} '{name}'. No new turns since '{base}', so nothing was preserved.{fork_hint}"
                        ),
                        (None, None) if discard && live_len > 0 => println!(
                            "{verb} '{name}' (un-checkpointed turns discarded).{fork_hint}"
                        ),
                        (None, None) => println!("{verb} '{name}'.{fork_hint}"),
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
                            println!(
                                "Failed to checkout: {err}{head_hint} \
                                 (current history preserved as '{saved}', HEAD points there)."
                            );
                        }
                        None => println!("Failed to checkout: {err}{head_hint}"),
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
            let confirm = crate::commonw::prompt::prompt_yes_or_no_danger(&format!(
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
        "  /checkpoint save <name>       save current history as a checkpoint (alias: commit)"
    );
    println!(
        "  /checkpoint tree              show checkpoints as a tree, newest first (default; `list` works too)"
    );
    println!("  /checkpoint checkout <name|HEAD|latest> [--save <name> | --discard]");
    println!("                                switch to a checkpoint (HEAD moves onto it); dirty");
    println!(
        "                                history is refused unless explicitly saved or discarded"
    );
    println!("  /checkpoint fork <name|HEAD|latest> [--save <name>]");
    println!(
        "                                start from a clean checkpoint; the next save becomes its child"
    );
    println!("  /checkpoint delete <name>     delete a checkpoint");
    println!(
        "  /checkpoint clear             delete ALL checkpoints (asks for confirmation first)"
    );
    println!();
    println!("  HEAD / @ / - all mean the HEAD pointer (the current commit);");
    println!("  `latest` means the newest checkpoint (the `[latest]` row in the tree).");
    println!("  History is never dropped except by explicit `delete`, `clear`, or `--discard`.");
    println!(
        "  Limit: {} checkpoints / {} MiB per session; saving an existing name replaces it.",
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
    let head = store.head_target().unwrap_or(None);
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
    let head = match effective_head(&checkpoints, head.as_deref(), base.as_deref()) {
        Some(resolved) => {
            let resolved = resolved.to_string();
            if head.as_deref() != Some(resolved.as_str()) {
                let _ = store.set_head(&resolved);
            }
            Some(resolved)
        }
        None => None,
    };
    if checkpoints.is_empty() {
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
        "linear saves stay flat · fork lines mark branches · → marks HEAD · [latest] marks newest · checkout <name> restores · fork <name> branches · save <name> snapshots · delete <name> drops".dimmed()
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

/// Refresh the `/checkpoint checkout|fork|delete <TAB>` name cache (best-effort):
/// completion sits on the hot path and cannot rescan the disk on every Tab,
/// so the driver publishes fresh names before each prompt and after mutations.
pub(super) fn publish_checkpoint_names(store: &CheckpointStore) {
    let names = store
        .list()
        .map(|checkpoints| checkpoints.into_iter().map(|cp| cp.name).collect())
        .unwrap_or_default();
    crate::ai::prompt::completion::CommandCompleter::set_checkpoint_name_candidates(names);
}

/// Parse `checkout` arguments: the first non-flag token is the target name.
/// `--discard` skips preservation (destructive); `--save <name>` / `--as <name>` /
/// `--save=<name>`
/// preserve the current history under an explicit name before switching.
fn parse_checkout_args(args: &[String]) -> (Option<String>, bool, Option<String>) {
    let mut target: Option<String> = None;
    let mut discard = false;
    let mut explicit_save: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--discard" {
            discard = true;
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
            // from `git stash`-style habits: `checkout <target> <save-as>`.
            explicit_save = Some(args[i].clone());
        }
        i += 1;
    }
    (target, discard, explicit_save)
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
        is_latest_alias, parse_checkout_args, switch_preparation, truncate_middle,
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
        let (target, discard, explicit) = parse_checkout_args(&args(&["c1"]));
        assert_eq!(target.as_deref(), Some("c1"));
        assert!(!discard);
        assert!(explicit.is_none());

        let (target, discard, explicit) = parse_checkout_args(&args(&["c1", "--discard"]));
        assert_eq!(target.as_deref(), Some("c1"));
        assert!(discard);
        assert!(explicit.is_none());

        let (target, discard, explicit) =
            parse_checkout_args(&args(&["c1", "--save", "mywork"]));
        assert_eq!(target.as_deref(), Some("c1"));
        assert!(!discard);
        assert_eq!(explicit.as_deref(), Some("mywork"));

        // Second positional token acts as the preserve name.
        let (target, discard, explicit) = parse_checkout_args(&args(&["c1", "mywork"]));
        assert_eq!(target.as_deref(), Some("c1"));
        assert!(!discard);
        assert_eq!(explicit.as_deref(), Some("mywork"));

        for removed_alias in ["--no-save", "--hard"] {
            let (_, discard, _) = parse_checkout_args(&args(&["c1", removed_alias]));
            assert!(!discard, "removed flag {removed_alias} must not discard history");
        }
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
            let (target, _, _) = parse_checkout_args(&args(&[alias]));
            assert_eq!(target.as_deref(), Some(alias));
            assert!(is_head_alias(&target.unwrap()));
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
            let (target, _, _) = parse_checkout_args(&args(&[alias]));
            assert!(is_latest_alias(&target.unwrap()));
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
