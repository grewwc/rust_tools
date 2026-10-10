use super::git::{blocked_git_destructive, blocked_git_subcommand};
use super::lexer::{
    command_word_index, effective_chain_uses_xargs, effective_command_tokens, indirect_command_index,
    tokenize_shell_words,
};
use super::config_blocked_commands;
use super::substitute::{
    expand_tilde_and_home, normalize_path, validate_substitution_positions,
};
use super::wrapper::{
    find_has_blocked_exec_semantics, is_interpreter_program, is_python_program, is_shell_program,
    python_c_argument, shell_c_option_present, validate_python_code,
};
use crate::ai::config_schema::AiConfig;
use crate::ai::tools::storage::file_store::path_within_allowed_roots;
use crate::ai::tools::storage::process_registry;

pub(crate) fn validate_single_segment(command: &str) -> Result<(), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("empty command".to_string());
    }

    let tokens = tokenize_shell_words(command);
    if tokens.is_empty() {
        return Err("empty command".to_string());
    }

    let lower_tokens = tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let shell_context = crate::cmd::run::command_requires_shell(command);
    let Some(command_idx) = command_word_index(&tokens, shell_context) else {
        // Pure-assignment segments (`FOO=bar`) carry no program, but an
        // assignment value can still smuggle `$(...)` into a shell variable
        // (`cmd=$(printf rm)`), so the substitution position guard must still
        // run before accepting the segment.
        validate_substitution_positions(command, "")?;
        return Ok(());
    };
    let command_tokens = &lower_tokens[command_idx..];
    let raw_command_tokens = &tokens[command_idx..];
    let program = command_tokens[0].as_str();
    // Take the basename of the program path so `/bin/rm`, `./rm` and other
    // absolute/relative paths cannot bypass the blacklist.
    let program_basename = std::path::Path::new(program)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(program);
    // All later comparisons use the basename uniformly, so `/bin/rm` and `rm`
    // are treated alike.
    let program = program_basename;
    let extra_blocked = config_blocked_commands();
    // ---- tilde / $HOME escape detection ----
    // home and its subpaths are normally accessible; only `~/..` / `$HOME/..`
    // escapes outward are rejected.
    {
        for token in raw_command_tokens.iter().skip(1) {
            if token.starts_with('-') {
                continue;
            }
            expand_tilde_and_home(token)?;
        }
    }

    // Fail-closed guard for command-substitution positions: `$(...)` output is
    // unverifiable runtime data, so it may not occupy any argument position the
    // audit semantically checks. Runs before the `mv` branch because that
    // branch returns early once its own path checks pass.
    validate_substitution_positions(command, program)?;

    if program == "mv" {
        let base_dir = crate::ai::driver::runtime_ctx::effective_cwd()
            .map_err(|err| format!("failed to resolve current directory: {err}"))?;
        let base_dir = normalize_path(&base_dir);
        let mut path_args: Vec<String> = Vec::new();
        let mut iter = command_tokens
            .iter()
            .zip(raw_command_tokens.iter())
            .skip(1)
            .peekable();
        let mut end_of_options = false;

        while let Some(token) = iter.next() {
            let (lower_token, raw_token) = token;
            if !end_of_options {
                if lower_token == "--" {
                    end_of_options = true;
                    continue;
                }

                if lower_token.starts_with('-') {
                    if program == "mv" {
                        let option = lower_token.as_str();
                        if option == "-t" || option == "--target-directory" {
                            let dir = iter.next().ok_or_else(|| {
                                format!("missing target directory for '{raw_token}'")
                            })?;
                            path_args.push(dir.1.to_string());
                            continue;
                        }

                        if let Some(dir) = raw_token.strip_prefix("--target-directory=") {
                            if dir.is_empty() {
                                return Err(format!("missing target directory for '{raw_token}'"));
                            }
                            path_args.push(dir.to_string());
                            continue;
                        }

                        if raw_token.starts_with("-t") && raw_token.len() > 2 {
                            path_args.push(raw_token[2..].to_string());
                            continue;
                        }
                    }

                    continue;
                }
            }

            path_args.push(raw_token.to_string());
        }

        if path_args.is_empty() {
            return Err(format!("program '{program}' requires path arguments"));
        }

        for raw_path in path_args {
            let raw_path = raw_path.trim();
            if raw_path.is_empty() {
                return Err(format!("program '{program}' contains an empty path"));
            }

            let resolved = if std::path::Path::new(raw_path).is_absolute() {
                normalize_path(std::path::Path::new(raw_path))
            } else {
                normalize_path(&base_dir.join(raw_path))
            };

            if !resolved.starts_with(&base_dir) {
                return Err(format!(
                    "path '{raw_path}' is outside the current directory"
                ));
            }
        }

        return Ok(());
    }

    let denied_programs = [
        "fish",
        "jshell",
        "dd",
        "chmod",
        "chown",
        "chgrp",
        "sudo",
        "su",
        "passwd",
        "shutdown",
        "reboot",
        "launchctl",
        "systemctl",
        // "service",
        // "diskutil",
        "mount",
        "umount",
        "ln",
        "truncate",
        "ssh",
        "scp",
        "rsync",
        // Bypass vector: `eval` / `source` / `.` re-interpret the following string
        // as shell code, completely bypassing validation.
        "eval",
        "source",
        ".",
        // Reverse-shell / network-listening tools: legitimate dev workflows almost
        // never need them; the risk outweighs the benefit.
        "nc",
        "ncat",
        "netcat",
        "telnet",
        "socat",
    ];
    if denied_programs.contains(&program) {
        return Err(format!("program '{program}' is blocked"));
    }

    // Users can add custom blacklist programs via `ai.sandbox.blocked_commands`.
    if extra_blocked.iter().any(|p| p == program) {
        return Err(format!(
            "program '{program}' is blocked by sandbox policy (ai.sandbox.blocked_commands)"
        ));
    }

    // `kill` / `pkill` / `killall` are allowed only against processes this
    // agent session started itself (background process groups registered by
    // `execute_command`); signaling any external process stays blocked. See
    // `validate_kill_targets` below for the per-target verification.
    if is_kill_program(program) {
        return validate_kill_targets(program, raw_command_tokens);
    }

    // `rm` is allowed only against files inside this session's private temp
    // dir (everything there is session-owned by construction); deleting
    // anything else — project or user files — stays blocked. See
    // `validate_rm_targets` below for the per-target verification.
    if program == "rm" {
        return validate_rm_targets(raw_command_tokens);
    }

    // Safety policy: block destructive/privilege-escalating `git` subcommands
    // (see `BLOCKED_GIT_SUBCOMMANDS`).
    // `git` itself is not in denied_programs (subcommands like status/log/diff are
    // harmless and necessary); only the listed subcommands are hard-blocked.
    // Global option variants (`git -C /repo push`) hit too.
    if program == "git" {
        if let Some(reason) = blocked_git_subcommand(command_tokens) {
            return Err(reason.to_string());
        }
        // Call with the original-case tokens to distinguish case-sensitive short
        // options like `-B`/`-b` and `-C`/`-c`.
        if let Some(reason) = blocked_git_destructive(raw_command_tokens) {
            return Err(reason.to_string());
        }
    }

    // "Second interpretation" like `bash -c "..."` / `sh -c` / `zsh -c` executes
    // the string as shell code, bypassing the segment blacklist — block outright.
    // Running scripts directly (`bash script.sh`) is still allowed.
    if is_shell_program(program) && shell_c_option_present(program, command_tokens) {
        return Err(format!(
            "shell `{program} -c ...` re-interprets a string as shell code; \
             run the literal command directly instead"
        ));
    }
    // The code string of `python -c '...'` is likewise executed as a program, but
    // it can be validated statically (validate_python_code): clean strings pass,
    // hits on dangerous primitives are blocked — more usable than blanket
    // blocking and no weaker than the original guarantee. fail-closed when the
    // code string cannot be extracted (missing / shell variable expansion).
    // Other interpreters (perl / ruby / node / php / awk / lua) have no matching
    // scanner and stay blocked as before.
    if is_python_program(program) {
        match python_c_argument(command_tokens) {
            Ok(Some(code)) => validate_python_code(&code)?,
            // No `-c`: `python3 script.py` / `python3 -m mod`, same tier as
            // `bash run.sh`; script file contents are not audited.
            Ok(None) => {}
            Err(reason) => {
                return Err(format!(
                    "python `{program} -c` code cannot be verified ({reason}); \
                     pass a literal quoted code string or write a script file instead"
                ));
            }
        }
    } else if is_interpreter_program(program) && shell_c_option_present(program, command_tokens) {
        return Err(format!(
            "interpreter `{program} -c` re-interprets a string as code and is blocked; \
             write a script file and run `{program} script` instead"
        ));
    }

    // `find`'s `-delete` / `-exec*` / `-ok*` are dangerous only when they act as a
    // real primary. When they are merely pattern arguments like `-name
    // '-delete'`, they must not be falsely blocked.
    if program == "find" {
        if let Some(flag) = find_has_blocked_exec_semantics(command_tokens) {
            return Err(format!(
                "find primary '{flag}' mutates files or executes commands and is blocked"
            ));
        }
    }

    // Global-search scope confinement: `find` / `grep -r` / `locate` and
    // absolute glob patterns must stay inside the allowed search roots
    // (effective cwd, or `ai.sandbox.allowed_roots` when configured). Without
    // this, a bare-name hunt like `find /Users/bytedance -name request.txt`
    // can hit several unrelated directories that each contain a same-named
    // file, and the agent picks the wrong (e.g. stale) copy. See the section
    // before `validate_find_scope` for the full rationale.
    {
        let base_dir = crate::ai::driver::runtime_ctx::effective_cwd()
            .map_err(|err| format!("failed to resolve current directory: {err}"))?;
        let base_dir = normalize_path(&base_dir);
        validate_glob_scope(program, raw_command_tokens, &base_dir)?;
        if program == "find" {
            validate_find_scope(command_tokens, raw_command_tokens, &base_dir)?;
        }
        if matches!(program, "grep" | "egrep" | "fgrep" | "rg" | "ag" | "ack") {
            validate_grep_scope(program, command_tokens, raw_command_tokens, &base_dir)?;
        }
        if program == "locate" {
            validate_locate_scope(raw_command_tokens, &base_dir)?;
        }
    }

    // Common wrappers treat later tokens as the program that will actually run;
    // check only "the program name that will be executed", avoiding misjudging
    // ordinary content arguments (like the `rm` inside `printf '%s' rm`) as
    // dangerous commands.
    const DANGEROUS_PROGRAM_NAMES: &[&str] = &[
        "mv",
        "chmod",
        "chown",
        "chgrp",
        "sudo",
        "su",
        "ssh",
        "scp",
        "rsync",
        "dd",
        "shutdown",
        "reboot",
        "eval",
        "mount",
        "umount",
        "ln",
        "truncate",
        "passwd",
        "launchctl",
        "systemctl",
    ];
    if let Some(idx) = indirect_command_index(program, command_tokens, raw_command_tokens) {
        let nested = command_tokens[idx].as_str();
        if DANGEROUS_PROGRAM_NAMES.contains(&nested) || extra_blocked.iter().any(|p| p == nested) {
            return Err(format!(
                "indirect execution of '{nested}' via '{program}' is blocked"
            ));
        }
        // Indirectly executing a blocked `git` subcommand (e.g. `env git push`,
        // `xargs git stash`) must be blocked too, otherwise wrappers bypass the
        // direct check.
        if nested == "git" {
            if let Some(reason) = blocked_git_subcommand(&command_tokens[idx..]) {
                return Err(reason.to_string());
            }
            if let Some(reason) = blocked_git_destructive(&raw_command_tokens[idx..]) {
                return Err(reason.to_string());
            }
        }
        // Kill tools behind wrappers (`env kill 123`, `timeout 5 pkill -f x`)
        // get the same per-target verification as the direct path, otherwise
        // a wrapper would bypass the sandbox gate.
        if is_kill_program(nested) {
            // `xargs` appends stdin items as extra runtime arguments, so the
            // kill target set is never fully visible on the command line.
            if program == "xargs" {
                return Err(xargs_kill_blocked_message());
            }
            validate_kill_targets(nested, &raw_command_tokens[idx..])?;
        }
        // `rm` behind wrappers gets the same temp-dir scoping as the direct
        // path, otherwise `env rm ...` would bypass the sandbox gate. `xargs
        // rm` stays blocked: xargs appends stdin items as extra deletion
        // targets invisible on the command line.
        if nested == "rm" {
            if program == "xargs" {
                return Err(xargs_rm_blocked_message());
            }
            validate_rm_targets(&raw_command_tokens[idx..])?;
        }
        // Interpreter `-c` / `-e` behind wrappers needs the same validation,
        // otherwise `env bash -c '...'` / `env perl -e '...'` /
        // `xargs python3 -c '...'` bypass the direct-path block via the wrapper.
        if is_python_program(nested) {
            match python_c_argument(&command_tokens[idx..]) {
                Ok(Some(code)) => validate_python_code(&code)?,
                Ok(None) => {}
                Err(reason) => {
                    return Err(format!(
                        "python `{nested} -c` code cannot be verified via '{program}' ({reason})"
                    ));
                }
            }
        } else if (is_shell_program(nested) || is_interpreter_program(nested))
            && shell_c_option_present(nested, &command_tokens[idx..])
        {
            return Err(format!(
                "indirect `{nested} -c` re-interpretation via '{program}' is blocked"
            ));
        }
    }

    // Layered wrappers (`nohup env python3 -c '...'`, `env env bash -c '...'`)
    // peel past the single-level indirect checks above: deep-unwrap to the
    // innermost command with effective_command_tokens and validate once more.
    let effective = effective_command_tokens(command);
    // Command substitution in the *program-name* slot is banned even when the
    // inner command is itself harmless: the substitution output is word-split
    // into command words, so `$(echo r)m -rf /` would execute `rm -rf /` at
    // runtime and bypass the program blacklist. Only materializable whole-word
    // `"$(...)"` data arguments and literal-`seq` loop lists (see
    // `check_substitution_at`) may carry a substitution; anything else in
    // argument position is rejected there. Check the deep-unwrapped program so wrappers (`env`, `exec`,
    // `command`, `timeout`, ...) cannot smuggle in a generated program name.
    if let Some(program_word) = effective.first() {
        if program_word.contains("$(") || program_word.contains('`') {
            return Err(format!(
                "command substitution cannot generate the program name; pass a literal \
                 program instead (`{program_word}`)"
            ));
        }
    }
    if let Some(eff_program) = effective.first() {
        if is_python_program(eff_program) {
            match python_c_argument(&effective) {
                Ok(Some(code)) => validate_python_code(&code)?,
                Ok(None) => {}
                Err(reason) => {
                    return Err(format!(
                        "python `{eff_program} -c` code cannot be verified inside '{command}' \
                         ({reason})"
                    ));
                }
            }
        } else if (is_shell_program(eff_program) || is_interpreter_program(eff_program))
            && shell_c_option_present(eff_program, &effective)
        {
            return Err(format!(
                "nested `{eff_program} -c` re-interpretation inside '{command}' is blocked"
            ));
        } else if is_kill_program(eff_program) {
            // `xargs` (possibly behind further wrappers, e.g.
            // `timeout 5 xargs kill 123`) appends stdin items as extra
            // arguments at runtime, so the kill target set is not fully
            // visible on the command line: such kills stay blocked.
            if effective_chain_uses_xargs(command) {
                return Err(xargs_kill_blocked_message());
            }
            validate_kill_targets(eff_program, &effective)?;
        } else if eff_program == "rm" {
            // `xargs` (possibly behind further wrappers, e.g.
            // `timeout 5 env xargs rm <tmp>/x`) appends stdin items as extra
            // deletion targets at runtime, so the removal set is not fully
            // visible on the command line: such removals stay blocked.
            if effective_chain_uses_xargs(command) {
                return Err(xargs_rm_blocked_message());
            }
            validate_rm_targets(&effective)?;
        }
    }

    Ok(())
}

// =========================================================================
// Kill-target verification (kill / pkill / killall)
// =========================================================================
//
// The kill family is not blanket-blocked: the agent routinely needs to stop
// services it started itself (`python app.py &` -> `kill <pid>` /
// `pkill -f app.py`). The guarantee the sandbox can give instead: every
// process the invocation would signal must belong to a background process
// group this session registered via `execute_command` (see
// `tools::storage::process_registry`). Targets are resolved with `pgrep` /
// `ps` at validation time, so only live, verifiably-ours processes pass;
// anything unverifiable fails closed.

fn is_kill_program(program: &str) -> bool {
    matches!(
        program.to_ascii_lowercase().as_str(),
        "kill" | "pkill" | "killall"
    )
}

fn xargs_kill_blocked_message() -> String {
    "kill tools behind 'xargs' are blocked: xargs appends stdin items as \
     additional targets the sandbox cannot verify"
        .to_string()
}

/// Largest plausible signal number. `kill -<n>` reads `n` as a signal when it
/// is a valid signal number and as a process group otherwise; real pids are
/// far larger than any signal, so values up to this bound are parsed as
/// signal options and larger negative values as process-group targets.
const MAX_SIGNAL_NUMBER: i64 = 128;

fn validate_kill_targets(program: &str, raw_tokens: &[String]) -> Result<(), String> {
    let session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    match program.to_ascii_lowercase().as_str() {
        "kill" => validate_kill_pids(raw_tokens, &session_id),
        "pkill" => validate_pattern_kill(raw_tokens, &session_id, "pkill", PgrepMode::Pkill),
        "killall" => validate_pattern_kill(raw_tokens, &session_id, "killall", PgrepMode::Killall),
        _ => Ok(()),
    }
}

/// `kill [options] pid...`: options (`-<signal>`, `-s`/`-n <signal>`,
/// `--signal=...`, `-l`, `-L`, `-t`, `--`) are skipped; every remaining token
/// must be a literal integer pid (negative = process group).
pub(crate) fn validate_kill_pids(raw_tokens: &[String], session_id: &str) -> Result<(), String> {
    let mut targets: Vec<i64> = Vec::new();
    let mut options_ended = false;
    let mut i = 1usize;
    while i < raw_tokens.len() {
        let token = raw_tokens[i].as_str();
        if !options_ended && token == "--" {
            options_ended = true;
        } else if !options_ended && token.starts_with('-') && token.len() > 1 {
            let rest = &token[1..];
            if let Ok(value) = rest.parse::<i64>() {
                if value > MAX_SIGNAL_NUMBER {
                    // Negative integer beyond any signal: a process-group kill.
                    targets.push(-value);
                }
                // `-9` (signal) and `-0` (liveness probe) carry no target.
            } else if rest == "s" || rest == "n" {
                // `-s` / `-n` take the signal name as their operand.
                i += 1;
            }
            // Other options (`-l`, `-L`, `-t`, `-KILL`, `--signal=...`):
            // no operand, no target.
        } else {
            match token.parse::<i64>() {
                Ok(pid) if pid > 0 => targets.push(pid),
                Ok(0) => {
                    return Err(
                        "kill target '0' (the command's own process group) cannot be verified"
                            .to_string(),
                    )
                }
                Ok(_) => {
                    return Err(format!("kill target '{token}' is not a valid positive pid"))
                }
                Err(_) => {
                    return Err(format!(
                        "kill target '{token}' is not a literal pid; the sandbox can only \
                         verify literal numeric targets (run `pgrep` first and pass the pid)"
                    ))
                }
            }
        }
        i += 1;
    }
    if targets.is_empty() {
        return Err("kill: no verifiable target process given".to_string());
    }
    for target in targets {
        if !kill_target_verified(target, session_id) {
            return Err(kill_target_denied_message(target));
        }
    }
    Ok(())
}

pub(crate) enum PgrepMode {
    /// `pkill`: pattern matches process name (or full command line with `-f`).
    Pkill,
    /// `killall`: process names only (mirrored as `pgrep -x`).
    Killall,
}

/// `pkill [options] pattern` / `killall [options] name...`: accept the
/// pattern-narrowing flags that map 1:1 onto `pgrep` plus signal options;
/// reject every other option, because the killed set could then differ from
/// what `pgrep` reports and the verification would be unsound.
pub(crate) fn validate_pattern_kill(
    raw_tokens: &[String],
    session_id: &str,
    program: &str,
    mode: PgrepMode,
) -> Result<(), String> {
    let mut filters: Vec<String> = Vec::new();
    let mut patterns: Vec<String> = Vec::new();
    let mut options_ended = false;
    let mut i = 1usize;
    while i < raw_tokens.len() {
        let token = raw_tokens[i].as_str();
        if !options_ended && token == "--" {
            options_ended = true;
        } else if !options_ended && token.starts_with('-') && token.len() > 1 {
            let supported = match mode {
                PgrepMode::Pkill => matches!(token, "-f" | "-x" | "-i" | "-n" | "-o"),
                PgrepMode::Killall => token == "-e",
            };
            if supported {
                // `killall -e` (exact name) maps to `pgrep -x`.
                filters.push(if token == "-e" { "-x" } else { token }.to_string());
            } else if is_signal_option(token) {
                // Signal specification: does not change the target set.
            } else {
                return Err(format!(
                    "{program} option '{token}' cannot be verified by the sandbox; use a \
                     plain name{} pattern instead",
                    if matches!(mode, PgrepMode::Pkill) {
                        " / -f / -x / -i"
                    } else {
                        " (or `pkill -f` for pattern matching)"
                    }
                ));
            }
        } else {
            patterns.push(token.to_string());
        }
        i += 1;
    }
    if patterns.is_empty() {
        return Err(match mode {
            PgrepMode::Pkill => "pkill: no pattern given".to_string(),
            PgrepMode::Killall => "killall: no process name given".to_string(),
        });
    }
    // Resolve each pattern to the exact pid set the kill would signal (pgrep
    // and pkill share their matching engine) and verify every pid.
    for pattern in &patterns {
        for pid in resolve_pattern_pids(&filters, pattern, program)? {
            if !kill_target_verified(pid as i64, session_id) {
                return Err(format!(
                    "{program} pattern '{pattern}' matches pid {pid}, which is not a process \
                     started by this agent session; refusing to signal external processes"
                ));
            }
        }
    }
    Ok(())
}

/// Run `pgrep [filters] <pattern>` and return the matched pids. Exit status
/// 0 = matched, 1 = no match (empty set, nothing to kill), anything else is a
/// resolution failure that fails closed.
fn resolve_pattern_pids(
    filters: &[String],
    pattern: &str,
    program: &str,
) -> Result<Vec<u32>, String> {
    let mut command = std::process::Command::new("pgrep");
    command.args(filters).arg(pattern);
    let output = command
        .output()
        .map_err(|err| format!("{program}: failed to resolve pattern '{pattern}': {err}"))?;
    match output.status.code() {
        Some(0) => {}
        Some(1) => return Ok(Vec::new()),
        _ => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = stderr.trim();
            return Err(format!(
                "{program}: cannot verify pattern '{pattern}' (pgrep exited with {}); \
                 refusing to run an unverifiable kill{}",
                output.status,
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(": {detail}")
                }
            ));
        }
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .collect())
}

/// True for `kill`-family signal specifications (`-9`, `-SIGTERM`,
/// `--signal=9`). Bare alphabetic options like `-v` are deliberately NOT
/// treated as signals: they are `pkill`/`killall` match modifiers that change
/// the target set in ways `pgrep` cannot mirror.
fn is_signal_option(token: &str) -> bool {
    if let Some(rest) = token.strip_prefix("--signal=") {
        return !rest.is_empty();
    }
    let Some(rest) = token.strip_prefix('-') else {
        return false;
    };
    rest.parse::<u64>().is_ok() || rest.starts_with("SIG")
}

/// A target is killable iff it is itself a registered pgid of this session,
/// or (for a pid) its process group is one. `ps` fails when the process no
/// longer exists, which fails closed (nothing to signal anyway).
fn kill_target_verified(target: i64, session_id: &str) -> bool {
    if session_id.is_empty() {
        return false;
    }
    let pgid = target.unsigned_abs() as u32;
    if process_registry::is_registered_pgid(session_id, pgid) {
        return true;
    }
    if target > 0 {
        if let Some(actual_pgid) = process_group_of(target as u32) {
            return process_registry::is_registered_pgid(session_id, actual_pgid);
        }
    }
    false
}

/// Resolve a pid's process group id via `ps -o pgid= -p <pid>`.
pub(crate) fn process_group_of(pid: u32) -> Option<u32> {
    let output = std::process::Command::new("ps")
        .args(["-o", "pgid=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u32>().ok().filter(|&pgid| pgid > 0)
}

fn kill_target_denied_message(target: i64) -> String {
    let subject = if target > 0 {
        format!("pid {target}")
    } else {
        format!("process group {}", -target)
    };
    format!(
        "cannot signal {subject}: not a process started by this agent session \
         (only background processes launched via execute_command are killable)"
    )
}

// =========================================================================
// rm target verification
// =========================================================================
//
// `rm` is not blanket-blocked: the agent routinely needs to clean up files it
// created (build outputs, downloaded archives, scratch files). The guarantee
// the sandbox can give instead: every path the invocation would delete must
// resolve inside the session's private temp dir (`runtime_ctx::temp_dir()`),
// where every file is session-owned by construction. Anything else — project
// files, user files, system paths — stays blocked; delete project files with
// `apply_patch`'s `*** Delete File:` envelope instead.

/// Message for `xargs rm`: stdin items would append extra deletion targets the
/// sandbox cannot see on the command line.
fn xargs_rm_blocked_message() -> String {
    "'rm' behind 'xargs' is blocked: xargs appends stdin items as extra \
     deletion targets the sandbox cannot verify"
        .to_string()
}

/// `rm [options] path...`: options (`-f`, `-r`, `-R`, `-d`, `-i`, `-v`,
/// `--force`, `--recursive`, ...) and the `--` separator are skipped (none of
/// them takes an operand, so the skip is unambiguous); every remaining token
/// must resolve inside the session temp dir. Glob metacharacters in a target
/// are safe only because the lexical prefix is verified first:
/// `normalize_path` collapses `..` before the prefix check, and shell globbing
/// cannot escape the directory the pattern is anchored in.
pub(crate) fn validate_rm_targets(raw_tokens: &[String]) -> Result<(), String> {
    let temp_root = crate::ai::driver::runtime_ctx::temp_dir()
        .map_err(|err| format!("cannot resolve session temp dir: {err}"))?;
    let temp_root = normalize_path(&temp_root);
    let base_dir = crate::ai::driver::runtime_ctx::effective_cwd()
        .map_err(|err| format!("failed to resolve current directory: {err}"))?;
    let base_dir = normalize_path(&base_dir);

    let mut options_ended = false;
    let mut target_count = 0usize;
    for token in &raw_tokens[1..] {
        if !options_ended && token == "--" {
            options_ended = true;
            continue;
        }
        if !options_ended && token.starts_with('-') && token.len() > 1 {
            // rm option; skip.
            continue;
        }
        let raw_path = strip_quotes(token.trim());
        if raw_path.is_empty() {
            return Err("rm contains an empty path".to_string());
        }
        // `~` / `$HOME` prefix: expand for resolution (`expand_tilde_and_home`
        // already rejected `~/..` escapes earlier in validation).
        let expanded = expand_tilde_and_home(raw_path)?;
        let resolved = if std::path::Path::new(&expanded).is_absolute() {
            normalize_path(std::path::Path::new(&expanded))
        } else {
            normalize_path(&base_dir.join(expanded))
        };
        if !resolved.starts_with(&temp_root) {
            return Err(format!(
                "rm target '{raw_path}' is outside the session temp dir: the sandbox only \
                 allows deleting files this session created (project files: use apply_patch \
                 `*** Delete File:`)"
            ));
        }
        target_count += 1;
    }
    if target_count == 0 {
        return Err("rm: no verifiable target path given".to_string());
    }
    Ok(())
}

// =========================================================================
// Search-scope confinement
// =========================================================================
//
// Why this exists: an agent searching for a *relative* file name (e.g. the
// user said "the request is in request.txt") can escalate to a whole-disk hunt
// (`find /Users/bytedance -name request.txt`) and then pick the wrong copy
// when several directories contain a same-named file (a real incident: a stale
// `self-dev/test_llm/request.txt` was picked over the workspace copy). The
// checks below keep name searches inside the allowed roots:
//
// - `find`: every search root must lie inside the allowed roots; a bare
//   relative `-name` / `-path` target (no glob metacharacters) must exist as a
//   direct child of one of the search roots, otherwise the search would roam
//   unrelated directories.
// - `grep -r` / `rg` / `ag` / `ack`: recursive search paths must lie inside
//   the allowed roots. Non-recursive `grep file` single-file reads are not
//   affected.
// - `locate`: a whole-disk name index; bare names must exist in the cwd, and
//   pattern searches are rejected outright.
// - glob patterns (`/Users/*/request.txt`): the literal path prefix before the
//   first metacharacter must lie inside the allowed roots. Bare relative globs
//   (`*.rs`, `src/*.rs`) can only match inside the cwd and stay allowed.
//
// The allowed roots are the same as for file writes
// (`file_store::path_within_allowed_roots`): `ai.sandbox.allowed_roots` when
// configured, else `effective_cwd()`, always plus the session temp dir, the
// skills dir and the rust_tools config dir. Like the rest of this module this
// is static best-effort: shell variable expansion inside paths is not tracked.

/// True when `arg` contains a glob / brace-expansion metacharacter.
pub(crate) fn has_glob_metachar(arg: &str) -> bool {
    arg.contains(['*', '?', '[', '{', '}'])
}

/// Strip one matching layer of single/double quotes from a shell token.
pub(crate) fn strip_quotes(token: &str) -> &str {
    let bytes = token.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
            || (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"'))
    {
        &token[1..token.len() - 1]
    } else {
        token
    }
}

/// Resolve a path argument to an absolute, lexically normalized path: expands
/// `~` / `$HOME`, joins relative paths against `base_dir`.
fn resolve_path_arg(arg: &str, base_dir: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let expanded = expand_tilde_and_home(arg)?;
    let path = std::path::Path::new(&expanded);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    };
    Ok(normalize_path(&resolved))
}

/// Reject glob patterns whose literal prefix escapes the allowed search roots.
fn validate_glob_scope(
    program: &str,
    raw_command_tokens: &[String],
    base_dir: &std::path::Path,
) -> Result<(), String> {
    // For the grep family, option values (`--include '*.h'`, `--glob '*.rs'`)
    // are filter patterns, not search paths; skip them so they are not
    // misread as path globs.
    let value_options = if matches!(program, "grep" | "egrep" | "fgrep" | "rg" | "ag" | "ack") {
        Some(grep_value_options(program))
    } else {
        None
    };
    let mut i = 1usize;
    while i < raw_command_tokens.len() {
        let raw = &raw_command_tokens[i];
        if raw.starts_with('-') {
            if let Some(opts) = value_options {
                if opts.contains(&raw.as_str()) {
                    i += 2; // skip the option and its value
                    continue;
                }
            }
            i += 1;
            continue;
        }
        // Quoted tokens are not globs (quotes make the shell treat the token
        // as a literal file name).
        if raw.starts_with('\'') || raw.starts_with('"') {
            i += 1;
            continue;
        }
        let Some(meta_idx) = raw.find(['*', '?', '[', '{', '}']) else {
            i += 1;
            continue;
        };
        // A glob with an empty literal prefix (`*.txt`, `**/*.rs`) can only
        // match inside the current directory (recursively for `**`).
        if meta_idx == 0 {
            i += 1;
            continue;
        }
        let prefix = &raw[..meta_idx];
        let resolved = resolve_path_arg(prefix, base_dir)?;
        if !path_within_allowed_roots(&resolved) {
            return Err(format!(
                "glob pattern '{raw}' would expand outside the current directory; \
                 use explicit paths inside the current directory (or add the directory \
                 to ai.sandbox.allowed_roots)"
            ));
        }
        i += 1;
    }
    Ok(())
}

/// Confine `find` search roots to the allowed roots and reject bare-name
/// searches that cannot resolve under those roots.
fn validate_find_scope(
    command_tokens: &[String],
    raw_command_tokens: &[String],
    base_dir: &std::path::Path,
) -> Result<(), String> {
    // Search roots are the leading positional arguments before the first
    // expression token (`-flag`, `(`, `)`, `!`); with no roots `find` searches
    // `.` (the current directory).
    let mut root_args: Vec<&str> = Vec::new();
    for raw in raw_command_tokens.iter().skip(1) {
        if raw.starts_with('-') || matches!(raw.as_str(), "(" | ")" | "!") {
            break;
        }
        root_args.push(raw);
    }
    let root_args: Vec<&str> = if root_args.is_empty() {
        vec!["."]
    } else {
        root_args
    };

    let mut resolved_roots = Vec::with_capacity(root_args.len());
    for raw_root in &root_args {
        // Strip quotes before resolving: the shell removes them at execution
        // time, so `find '..'` really searches the parent directory. Leaving
        // them on would make the quoted string lexically land under base_dir
        // and pass the confinement check while the real search escapes it.
        let resolved = resolve_path_arg(strip_quotes(raw_root), base_dir)?;
        if !path_within_allowed_roots(&resolved) {
            return Err(format!(
                "find search root '{raw_root}' is outside the current directory; \
                 only search inside the current directory (or add the root to \
                 ai.sandbox.allowed_roots)"
            ));
        }
        resolved_roots.push(resolved);
    }

    // A bare relative target (`-name request.txt`, `-path ./sub/request.txt`,
    // no glob metacharacters) must exist as a direct child of one of the search
    // roots. When it does not, the search would roam unrelated directories that
    // happen to contain a same-named file, and the agent can pick the wrong
    // (e.g. stale) copy.
    let mut i = 1usize;
    while i < command_tokens.len() {
        let tok = command_tokens[i].as_str();
        if matches!(tok, "-name" | "-iname" | "-path" | "-ipath" | "-wholename") {
            if let Some(pattern) = raw_command_tokens.get(i + 1) {
                let pattern = strip_quotes(pattern);
                if !pattern.is_empty()
                    && !has_glob_metachar(pattern)
                    // A pattern starting with `-` is find's own option-looking
                    // argument (e.g. `-name "-delete"`), not a user file name;
                    // the existence check would falsely reject it.
                    && !pattern.starts_with('-')
                    && !std::path::Path::new(pattern).is_absolute()
                {
                    let exists = resolved_roots
                        .iter()
                        .any(|root| normalize_path(&root.join(pattern)).exists());
                    if !exists {
                        return Err(format!(
                            "find searched for relative name '{pattern}', which does not exist \
                             under the current directory; a bare-name search across unrelated \
                             directories can pick a wrong (e.g. stale) copy of the file — use a \
                             more specific search root (`find <subdir> -name ...`), an absolute \
                             path, or ask the user where the file is"
                        ));
                    }
                }
            }
        }
        i += 1;
    }
    Ok(())
}

/// Options of `grep` / `rg` / `ag` / `ack` whose next token is a value, not a
/// path. Attached forms (`--include=*.rs`, `-C3`) are single tokens and need
/// no value skip.
fn grep_value_options(program: &str) -> &'static [&'static str] {
    if matches!(program, "rg" | "ag" | "ack") {
        // Note `rg -r` is `--replace` (takes a value), unlike `grep -r`.
        &[
            "-e", "--regexp", "-f", "--file", "-g", "--glob", "--iglob", "--match", "-t",
            "--type", "-T", "--type-not", "-r", "--replace",
        ]
    } else {
        &[
            "-e", "--regexp", "-f", "--file", "--include", "--exclude", "--exclude-from",
            "--include-dir", "--exclude-dir", "-d", "--directories", "-A", "--after-context",
            "-B", "--before-context", "-C", "--context", "-m", "--max-count", "--label",
        ]
    }
}

/// Confine recursive search paths (`grep -r`, `rg`, `ag`, `ack`) to the
/// allowed roots. Non-recursive `grep file` single-file reads stay unrestricted.
fn validate_grep_scope(
    program: &str,
    command_tokens: &[String],
    raw_command_tokens: &[String],
    base_dir: &std::path::Path,
) -> Result<(), String> {
    let always_recursive = matches!(program, "rg" | "ag" | "ack");
    let value_options = grep_value_options(program);
    let mut recursive = always_recursive;
    let mut pattern_from_option = false;
    let mut end_of_options = false;
    let mut positional: Vec<&str> = Vec::new();
    let mut i = 1usize;
    while i < command_tokens.len() {
        let tok = command_tokens[i].as_str();
        if !end_of_options && tok.starts_with('-') && tok != "-" {
            if tok == "--" {
                end_of_options = true;
                i += 1;
                continue;
            }
            // Recursion flags: `-r` / `-R` / `--recursive`, including clusters
            // like `-rn`. Long options other than `--recursive` are excluded by
            // the `!starts_with("--")` guard.
            if tok == "--recursive"
                || (tok.starts_with('-') && !tok.starts_with("--") && tok.contains('r'))
            {
                recursive = true;
            }
            if matches!(tok, "-e" | "--regexp" | "-f" | "--file")
                // Pattern-less modes: `rg --files` lists paths only, and
                // `ag`/`ack` `-g`/`--match` filter file names, so with these
                // every positional argument is a search path, not a pattern.
                || tok == "--files"
                || (matches!(tok, "-g" | "--match") && matches!(program, "ag" | "ack"))
            {
                pattern_from_option = true;
            }
            if value_options.contains(&tok) {
                i += 2; // skip the option and its value
                continue;
            }
            i += 1;
            continue;
        }
        positional.push(strip_quotes(&raw_command_tokens[i]));
        i += 1;
    }
    if !recursive {
        return Ok(());
    }
    // Without `-e` / `-f` the first positional is the pattern, not a path.
    let path_args = if pattern_from_option {
        &positional[..]
    } else {
        positional.get(1..).unwrap_or(&[])
    };
    for raw_path in path_args {
        // Relative paths without `..` cannot escape the cwd; metacharacter
        // paths are handled by `validate_glob_scope`.
        if raw_path.is_empty() || has_glob_metachar(raw_path) {
            continue;
        }
        let path = std::path::Path::new(raw_path);
        if path.is_relative() && !raw_path.split('/').any(|c| c == "..") {
            continue;
        }
        let resolved = resolve_path_arg(raw_path, base_dir)?;
        if !path_within_allowed_roots(&resolved) {
            return Err(format!(
                "`{program}` search path '{raw_path}' is outside the current directory; \
                 only search inside the current directory (or add the root to \
                 ai.sandbox.allowed_roots)"
            ));
        }
    }
    // Recursive `grep` reads every file it walks and has no ignore support, so
    // a root above build output costs minutes and buries the answer in matches
    // from object files and binaries. `rg` / `ag` / `ack` honour ignore files
    // and stay unrestricted.
    if matches!(program, "grep" | "egrep" | "fgrep") {
        validate_grep_heavy_dirs(program, command_tokens, path_args, base_dir)?;
    }
    Ok(())
}

/// Directories a recursive `grep` should never walk: build output and
/// dependency trees hold the object files, archives and binaries that dominate
/// the walk while containing no source. The names cover the common ecosystems
/// (Rust, JS/TS, Python, JVM, .NET, Apple, Dart, Zig, vendored Go/PHP
/// dependencies, Haskell, OCaml/Elixir) and stay tunable through
/// `AiConfig::SANDBOX_SEARCH_SKIP_DIRS`. VCS internals are left out on purpose
/// (they exist in every repository, so listing `.git` would reject every
/// bare-`grep` search); the navigation tools carry their own, wider display
/// skip list (`tools/tree_tools.rs::SKIP_DIRS`).
const HEAVY_SEARCH_DIRS: &[&str] = &[
    ".dart_tool",
    ".gradle",
    ".mypy_cache",
    ".next",
    ".nuxt",
    ".parcel-cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svelte-kit",
    ".terraform",
    ".tox",
    ".turbo",
    ".venv",
    "__pycache__",
    "_build",
    "bower_components",
    "build",
    "coverage",
    "DerivedData",
    "dist",
    "dist-newstyle",
    "node_modules",
    "obj",
    "Pods",
    "target",
    "vendor",
    "venv",
    "zig-cache",
    "zig-out",
];

/// How deep below the searched root the pre-flight scan looks for heavy
/// directories (monorepos nest them: `packages/web/node_modules`) and how many
/// directory entries it may visit. Both bounds keep the check itself far
/// cheaper than the walk it prevents.
const HEAVY_SEARCH_SCAN_DEPTH: usize = 3;
const HEAVY_SEARCH_SCAN_BUDGET: usize = 4096;

/// Built-in heavy directory names merged with `ai.sandbox.search_skip_dirs`:
/// an entry adds a name, a `-`-prefixed entry removes a built-in one.
fn heavy_search_dir_names() -> Vec<String> {
    let configured =
        crate::commonw::configw::get_all_config().get(AiConfig::SANDBOX_SEARCH_SKIP_DIRS, "");
    merge_heavy_search_dirs(&configured)
}

/// Pure half of `heavy_search_dir_names`, so the merge rules are testable
/// without touching configuration.
pub(crate) fn merge_heavy_search_dirs(configured: &str) -> Vec<String> {
    let mut names: Vec<String> = HEAVY_SEARCH_DIRS.iter().map(|name| name.to_string()).collect();
    for raw in configured.split(',') {
        let entry = raw.trim();
        if entry.is_empty() {
            continue;
        }
        if let Some(removed) = entry.strip_prefix('-') {
            names.retain(|name| name != removed);
        } else if !names.iter().any(|name| name == entry) {
            names.push(entry.to_string());
        }
    }
    names
}

/// Heavy directory names present at or below `root`, sorted and deduplicated.
/// A matched directory is never descended into, and symlinked directories are
/// skipped because a `grep -r` walk does not follow them either.
fn heavy_dirs_under(root: &std::path::Path, names: &[String]) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    let mut pending = vec![(root.to_path_buf(), 0usize)];
    let mut budget = HEAVY_SEARCH_SCAN_BUDGET;
    'scan: while let Some((dir, depth)) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if budget == 0 {
                break 'scan;
            }
            budget -= 1;
            if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if names.iter().any(|heavy| heavy == &name) {
                if !found.iter().any(|seen| seen == &name) {
                    found.push(name);
                }
                continue;
            }
            if depth + 1 < HEAVY_SEARCH_SCAN_DEPTH && !name.starts_with('.') {
                pending.push((entry.path(), depth + 1));
            }
        }
    }
    found.sort();
    found
}

/// True when the command already excludes `dir` from the search, either
/// attached (`--exclude-dir=target`) or spaced (`--exclude-dir target`).
fn grep_excludes_dir(command_tokens: &[String], dir: &str) -> bool {
    let excluding_options = ["--exclude", "--exclude-dir", "--exclude-from"];
    let mut spaced_value = false;
    for tok in command_tokens.iter().skip(1) {
        let attached = excluding_options
            .iter()
            .any(|opt| tok.strip_prefix(opt).is_some_and(|rest| rest.starts_with('=')));
        if (spaced_value || attached) && tok.contains(dir) {
            return true;
        }
        spaced_value = excluding_options.contains(&tok.as_str());
    }
    false
}

/// Reject a recursive `grep` whose root still contains, within
/// `HEAVY_SEARCH_SCAN_DEPTH` levels, a build-output directory the command does
/// not exclude: `grep` has no ignore-file support, so it reads the whole tree.
/// An explicit `--exclude-dir`, a narrower root, or a root at or inside a build
/// directory (searching it on purpose) all pass; `rg` / `ag` / `ack` are not
/// inspected because they honour ignore files by default.
fn validate_grep_heavy_dirs(
    program: &str,
    command_tokens: &[String],
    path_args: &[&str],
    base_dir: &std::path::Path,
) -> Result<(), String> {
    let names = heavy_search_dir_names();
    // Without a path argument the search starts at the current directory.
    let implicit_root = ["."];
    let roots: &[&str] = if path_args.is_empty() {
        &implicit_root
    } else {
        path_args
    };
    for raw_root in roots {
        // Shell-expanded roots are handled by `validate_glob_scope`.
        if raw_root.is_empty() || has_glob_metachar(raw_root) {
            continue;
        }
        let Ok(root) = resolve_path_arg(raw_root, base_dir) else {
            continue;
        };
        // Searching a heavy directory on purpose is fine, so a root that names
        // one anywhere below the base (`target`, `target/debug`) passes, while
        // `.` does not.
        let relative = root.strip_prefix(base_dir).unwrap_or(root.as_path());
        if relative.components().any(|part| {
            let part = part.as_os_str().to_string_lossy();
            names.iter().any(|name| name == part.as_ref())
        }) {
            continue;
        }
        let walked: Vec<String> = heavy_dirs_under(&root, &names)
            .into_iter()
            .filter(|name| !grep_excludes_dir(command_tokens, name))
            .collect();
        if walked.is_empty() {
            continue;
        }
        let fix = walked
            .iter()
            .map(|name| format!("`--exclude-dir={name}`"))
            .collect::<Vec<_>>()
            .join(" ");
        return Err(format!(
            "recursive `{program}` from '{raw_root}' would walk {} (build output or \
             dependencies); grep has no ignore support — retry with {fix}, a narrower \
             root, or `rg` / `git grep`. To search inside build output on purpose, \
             root the search there; `rg` needs `--no-ignore` for it",
            summarize_dirs(&walked)
        ));
    }
    Ok(())
}

/// Up to three quoted directory names plus the remainder count, so the message
/// stays one line even when a root holds many heavy directories.
pub(crate) fn summarize_dirs(names: &[String]) -> String {
    let mut listed: Vec<String> = names.iter().take(3).map(|name| format!("'{name}/'")).collect();
    if names.len() > listed.len() {
        listed.push(format!("and {} more", names.len() - listed.len()));
    }
    listed.join(", ")
}

/// `locate` searches a whole-disk name index; reject it unless the bare name
/// already exists in the current directory (in which case `find` would work
/// too and needs no index).
fn validate_locate_scope(
    raw_command_tokens: &[String],
    base_dir: &std::path::Path,
) -> Result<(), String> {
    let Some(raw) = raw_command_tokens
        .iter()
        .skip(1)
        .find(|t| !t.starts_with('-'))
    else {
        return Ok(());
    };
    let pattern = strip_quotes(raw);
    if pattern.is_empty() || std::path::Path::new(pattern).is_absolute() {
        return Ok(());
    }
    if has_glob_metachar(pattern) {
        return Err(format!(
            "locate pattern '{pattern}' is a whole-disk search; use `find` inside the \
             current directory, or ask the user for the absolute path"
        ));
    }
    if !base_dir.join(pattern).exists() {
        return Err(format!(
            "locate searched for relative name '{pattern}', which does not exist under the \
             current directory; ask the user for the absolute path instead"
        ));
    }
    Ok(())
}
