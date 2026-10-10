// ---------------------------------------------------------------------------
// Git subcommand blocking
// ---------------------------------------------------------------------------

/// Parse the `git` subcommand: skip git global options appearing before the
/// subcommand and return the index of the subcommand token in `command_tokens`
/// (whose first element is `git` itself).
///
/// Some git global options consume the argument right after them: `-C <path>`,
/// `-c <name>=<value>`,
/// `--git-dir <path>`、`--work-tree <path>`、`--namespace <name>`、`--exec-path <path>`。
/// `=`-attached forms (e.g. `--git-dir=/repo`, `-C/repo`) consume no extra token.
/// The first token after `--` is treated as the subcommand.
fn git_subcommand_index(command_tokens: &[String]) -> Option<usize> {
    const VALUE_CONSUMING_LONG: &[&str] =
        &["--git-dir", "--work-tree", "--namespace", "--exec-path"];
    let mut i = 1usize;
    while i < command_tokens.len() {
        let tok = command_tokens[i].as_str();
        if tok == "--" {
            return command_tokens.get(i + 1).map(|_| i + 1);
        }
        // The first non-option token is the subcommand.
        if !tok.starts_with('-') || tok == "-" {
            return Some(i);
        }
        // `=`-attached forms carry their own value; no need to consume the next
        // token.
        if tok.contains('=') {
            i += 1;
            continue;
        }
        let lower = tok.to_ascii_lowercase();
        // Both `-C` and `-c` consume the next argument (after lowercasing both
        // are `-c`).
        if lower == "-c" || VALUE_CONSUMING_LONG.contains(&lower.as_str()) {
            i += 2;
            continue;
        }
        i += 1;
    }
    None
}

fn cargo_subcommand_index(command_tokens: &[String]) -> Option<usize> {
    const VALUE_CONSUMING: &[&str] = &[
        "--color",
        "--config",
        "--manifest-path",
        "--target-dir",
        "-C",
        "-Z",
    ];
    let mut index = 1usize;
    while index < command_tokens.len() {
        let token = command_tokens[index].as_str();
        if token == "--" {
            return command_tokens.get(index + 1).map(|_| index + 1);
        }
        if !token.starts_with('-') || token == "-" {
            return Some(index);
        }
        if token.contains('=')
            || (token.starts_with("-C") && token.len() > 2)
            || (token.starts_with("-Z") && token.len() > 2)
        {
            index += 1;
            continue;
        }
        index += if VALUE_CONSUMING.contains(&token) {
            2
        } else {
            1
        };
    }
    None
}

pub(crate) fn command_subcommand_index(command_tokens: &[String]) -> Option<usize> {
    let program = command_tokens.first().and_then(|token| {
        std::path::Path::new(token)
            .file_name()
            .and_then(|name| name.to_str())
    })?;
    match program {
        "git" => git_subcommand_index(command_tokens),
        "cargo" => cargo_subcommand_index(command_tokens),
        _ => command_tokens
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, token)| !token.starts_with('-') && !token.contains('='))
            .map(|(index, _)| index),
    }
}

/// `git` subcommands hard-blocked by the safety policy and their rejection
/// reasons.
///
/// `git` itself is not in `denied_programs` (subcommands like status/log/diff are
/// harmless and necessary); only the subcommands below are hard-blocked. Global
/// option variants (e.g. `git -C /repo push`) hit too.
const BLOCKED_GIT_SUBCOMMANDS: &[(&str, &str)] = &[
    // Prevent pushing local commits to a remote repository.
    ("push", "git push is blocked by sandbox policy"),
    // `git rm` physically deletes files from the working tree, unrecoverable;
    // `git rm --cached` only removes the index entry, but blocking outright is
    // safer than per-argument analysis. Use safe tools like `trash` to delete
    // files.
    (
        "rm",
        "git rm is blocked by sandbox policy; delete project files with apply_patch `*** Delete File:`",
    ),
];

/// If a `git` subcommand hits the block list, return its rejection reason.
pub(crate) fn blocked_git_subcommand(command_tokens: &[String]) -> Option<&'static str> {
    let idx = git_subcommand_index(command_tokens)?;
    let sub = command_tokens[idx].to_ascii_lowercase();
    BLOCKED_GIT_SUBCOMMANDS
        .iter()
        .find(|(name, _)| *name == sub)
        .map(|(_, reason)| *reason)
}

/// Decide which `git` subcommands would irreversibly discard or delete
/// uncommitted work.
///
/// Unlike `BLOCKED_GIT_SUBCOMMANDS` (globally banned ones like push), the
/// subcommands below are harmless under some argument combinations (e.g. `git
/// switch` branch switching, `git restore --staged` unstaging), so block only
/// when they would truly destroy uncommitted changes (working-tree/staged
/// changes, untracked files), avoiding collateral damage to normal workflows.
/// Returns the rejection reason on a hit.
///
/// Covers the user requirement "ban `git checkout --` and any command that
/// deletes currently uncommitted files irreversibly".
pub(crate) fn blocked_git_destructive(command_tokens: &[String]) -> Option<&'static str> {
    let idx = git_subcommand_index(command_tokens)?;
    // Subcommand names are case-insensitive; match after lowercasing.
    let sub = command_tokens[idx].to_ascii_lowercase();
    let rest = &command_tokens[idx + 1..];
    match sub.as_str() {
        // `git checkout <branch>` (no `--`, no `--force`/`-B`) lets git itself
        // protect uncommitted changes and error out on conflict — allow; other
        // forms that discard working-tree changes are blocked. Note: short options
        // are case-sensitive; `-B` (force-create/reset branch) differs from `-b`
        // (create branch) and must be distinguished; `-f`/`--force`
        // force-switching also discards changes.
        "checkout" => {
            if rest.iter().any(|t| t == "--") {
                return Some("git checkout -- <path> discards uncommitted working-tree changes");
            }
            if rest.iter().any(|t| {
                t == "-f"
                    || t.eq_ignore_ascii_case("--force")
                    || t == "-B"
                    || t.eq_ignore_ascii_case("--force-create")
            }) {
                return Some(
                    "git checkout --force/-B discards uncommitted changes when switching branches",
                );
            }
            // With no `--` and no force, use heuristics to detect file paths:
            // 1. `.`/`..`/`./`/`../` are obviously path shapes — block directly.
            // 2. An argument ending in a file extension (e.g. `src/main.rs`,
            //    `package.json`) is most likely a file path, not a branch name;
            //    the `.` suffix of branch/tag names is usually numeric (e.g.
            //    `v1.2.3`), not all letters, so no false block.
            let looks_like_path = rest.iter().any(|t| {
                if t.starts_with('-') {
                    return false;
                }
                t == "."
                    || t == ".."
                    || t.starts_with("./")
                    || t.starts_with("../")
                    || t.rfind('.').map_or(false, |pos| {
                        // Skip dotfiles (.gitignore etc.); already covered above.
                        pos > 0 && {
                            let ext = &t[pos + 1..];
                            !ext.is_empty()
                                && ext.len() <= 12
                                && ext.chars().all(|c| c.is_ascii_alphabetic())
                        }
                    })
            });
            if looks_like_path {
                return Some("git checkout <path> discards uncommitted working-tree changes");
            }
            None
        }
        // `git switch -f`/`--force`/`--discard-changes` force-switches and discards
        // uncommitted changes; `-C`/`--force-create` force-resets and switches
        // when the branch exists, also discarding. Creating a new branch (`-c`/
        // `--create`, without force) is safe — allow. Short options are
        // case-sensitive: `-C` ≠ `-c`.
        "switch" => {
            if rest.iter().any(|t| {
                t == "-f"
                    || t.eq_ignore_ascii_case("--force")
                    || t.eq_ignore_ascii_case("--discard-changes")
                    || t == "-C"
                    || t.eq_ignore_ascii_case("--force-create")
            }) {
                return Some(
                    "git switch --force/-C discards uncommitted changes when switching branches",
                );
            }
            None
        }
        // `git restore` defaults to restoring the working tree, discarding
        // uncommitted working-tree changes; only "--staged alone" is a safe
        // unstage (working tree untouched, reversible).
        "restore" => {
            if rest.iter().any(|t| t == "--worktree") {
                return Some("git restore --worktree discards uncommitted working-tree changes");
            }
            let has_staged = rest.iter().any(|t| t == "--staged");
            let has_source = rest
                .iter()
                .any(|t| t == "--source" || t.starts_with("--source="));
            if has_source && !has_staged {
                return Some("git restore --source=... discards uncommitted working-tree changes");
            }
            if has_staged {
                // Unstage only, working tree untouched, reversible — allow.
                return None;
            }
            Some("git restore discards uncommitted working-tree changes")
        }
        // `git reset --hard`/`--merge`/`--keep` discard working-tree/staged
        // changes; `--soft` and the default (mixed) keep the working tree — allow.
        "reset" => {
            if rest
                .iter()
                .any(|t| matches!(t.as_str(), "--hard" | "--merge" | "--keep"))
            {
                return Some("git reset --hard/--merge/--keep discards uncommitted changes");
            }
            None
        }
        // `git clean -f` deletes untracked files, unrecoverable; `-n` (dry-run)
        // and the like do not actually delete — allow.
        "clean" => {
            if rest.iter().any(|t| {
                t == "-f"
                    || t == "--force"
                    // A clustered short option (e.g. `-fd` = `-f -d`) containing
                    // `-f` also truly deletes files.
                    || (t.starts_with('-') && !t.starts_with("--") && t.contains('f'))
            }) {
                return Some("git clean -f deletes untracked files irreversibly");
            }
            None
        }
        _ => None,
    }
}
