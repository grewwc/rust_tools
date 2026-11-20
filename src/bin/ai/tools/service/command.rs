use serde_json::Value;
use std::{
    fs::File,
    io::{IsTerminal, Read},
    path::Path,
};

use crate::ai::config_schema::AiConfig;
use crate::ai::shell_quote::shell_single_quote;
use crate::ai::tools::storage::command_runner;
use crate::cmd::run::CommandRunResult;

use super::command_session::{non_empty_str_arg, resolve_session_yield};

pub(super) const MAX_COMMAND_OUTPUT_CHARS: usize = 16_000;
/// Maximum one-shot stdin payload accepted by `execute_command`.
const MAX_COMMAND_INPUT_BYTES: usize = 16 * 1024;
/// Read cap when materializing `$(cat /absolute/literal/path)` into a plain
/// shell argument. The limit stops the tool from accidentally reading an
/// unbounded stream or an oversized file before validation; regular JSON/DSL
/// arguments are far smaller.
const MAX_LITERAL_FILE_SUBSTITUTION_BYTES: usize = 64 * 1024;

/// Built-in default timeout and ceiling (seconds), overridable via sandbox
/// config.
const DEFAULT_COMMAND_TIMEOUT_SECS: u64 = 60;
const DEFAULT_COMMAND_TIMEOUT_MAX_SECS: u64 = 300;

/// Returns `execute_command`'s (default timeout, timeout ceiling), overridden
/// by sandbox config. Invalid/missing values fall back to the built-in
/// constants; the ceiling is at least 1s and never below the default.
fn config_command_timeout_bounds() -> (u64, u64) {
    let cfg = crate::commonw::configw::get_all_config();
    let default_timeout = cfg
        .get(AiConfig::SANDBOX_COMMAND_TIMEOUT_DEFAULT, "")
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|v| *v >= 1)
        .unwrap_or(DEFAULT_COMMAND_TIMEOUT_SECS);
    let max_timeout = cfg
        .get(AiConfig::SANDBOX_COMMAND_TIMEOUT_MAX, "")
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|v| *v >= 1)
        .unwrap_or(DEFAULT_COMMAND_TIMEOUT_MAX_SECS)
        .max(default_timeout);
    (default_timeout, max_timeout)
}

/// Pure function: clamp the requested timeout seconds into `[1, max]`, using
/// `default` when unset.
fn resolve_command_timeout(requested: Option<u64>, default: u64, max: u64) -> u64 {
    requested.unwrap_or(default).clamp(1, max)
}

fn read_literal_file_substitution(path: &str) -> Result<String, String> {
    let file = File::open(path)
        .map_err(|err| format!("cannot open literal file substitution '{path}': {err}"))?;
    let metadata = file
        .metadata()
        .map_err(|err| format!("cannot stat literal file substitution '{path}': {err}"))?;
    if !metadata.is_file() {
        return Err(format!(
            "literal file substitution '{path}' must name a regular file"
        ));
    }
    if metadata.len() > MAX_LITERAL_FILE_SUBSTITUTION_BYTES as u64 {
        return Err(format!(
            "literal file substitution '{path}' exceeds the {}-byte limit",
            MAX_LITERAL_FILE_SUBSTITUTION_BYTES
        ));
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    let mut limited = file.take((MAX_LITERAL_FILE_SUBSTITUTION_BYTES + 1) as u64);
    limited
        .read_to_end(&mut bytes)
        .map_err(|err| format!("cannot read literal file substitution '{path}': {err}"))?;
    if bytes.len() > MAX_LITERAL_FILE_SUBSTITUTION_BYTES {
        return Err(format!(
            "literal file substitution '{path}' exceeds the {}-byte limit",
            MAX_LITERAL_FILE_SUBSTITUTION_BYTES
        ));
    }
    let contents = String::from_utf8(bytes)
        .map_err(|_| format!("literal file substitution '{path}' must be valid UTF-8"))?;
    if contents.contains('\0') {
        return Err(format!(
            "literal file substitution '{path}' must not contain NUL bytes"
        ));
    }
    Ok(contents)
}

/// Run the inner command of a harmless command substitution and capture its
/// output, used to materialize `"$(harmless_cmd)"`.
fn execute_inner_shell_command(inner: &str, cwd: Option<&str>) -> Result<String, String> {
    // Reuse the existing runner; a 10s timeout covers short commands like
    // date/echo/git while avoiding long blocks.
    let output = crate::ai::tools::storage::command_runner::run_command(inner, cwd, 10)
        .map_err(|err| format!("failed to execute substitution '{inner}': {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "substitution command '{inner}' failed with exit code {}",
            output.status.code().unwrap_or(-1)
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| format!("substitution '{inner}' produced non-UTF-8 output"))?;
    if stdout.len() > MAX_LITERAL_FILE_SUBSTITUTION_BYTES {
        return Err(format!(
            "substitution '{inner}' output exceeds the {}-byte limit",
            MAX_LITERAL_FILE_SUBSTITUTION_BYTES
        ));
    }
    if stdout.contains('\0') {
        return Err(format!("substitution '{inner}' output contains NUL bytes"));
    }
    // Remove trailing line endings before quoting stdout for the outer command.
    Ok(stdout
        .trim_end_matches(|c| c == '\n' || c == '\r')
        .to_string())
}

/// Materialize harmless `"$(...)"` substitutions proven safe by the audit
/// layer as plain data, supporting `cat` literals and generic harmless
/// commands. The substituted command is still fully audited afterwards, so a
/// replacement that turns into a blocked program name or dangerous argument is
/// still rejected.
///
/// A substitution that follows a command separator (`;`, `|`, `&&`, `||`) is
/// rejected before any side effect: eager evaluation would run it before the
/// shell decides whether the earlier segment even runs, so
/// `echo fresh > f && cat "$(cat f)"` must not read `f` early.
///
/// A simple `NAME=$(...)` assignment becomes `NAME='<literal>'`, and bindable
/// `$NAME` / `${NAME}` word references expand to that literal, so a value that
/// names a blocked program still reaches the blacklist.
fn materialize_safe_shell_substitutions(
    command: &str,
    cwd: Option<&str>,
) -> Result<String, String> {
    let substitutions = super::audit::safe_shell_substitutions(command);
    if substitutions.is_empty() {
        return Ok(command.to_string());
    }
    if substitutions
        .iter()
        .any(|substitution| substitution.after_command_separator)
    {
        return Err(
            "command substitution in a chained command requires deferred evaluation; \
             run each segment as a separate command"
                .to_string(),
        );
    }
    // Plan reference expansion before running anything: a reference the plan
    // cannot bind must reject the command before a substituted command has any
    // side effect.
    let spans: Vec<SubstitutionSpan<'_>> = substitutions
        .iter()
        .map(|substitution| SubstitutionSpan {
            start: substitution.start,
            end: substitution.end,
            assignment: substitution.assignment.as_deref(),
        })
        .collect();
    let references = plan_assignment_references(command, &spans)?;
    let mut edits: Vec<(usize, usize, String)> = Vec::with_capacity(substitutions.len());
    // (name, value) for materialized assignments, used to expand references.
    let mut assigned: Vec<(String, String)> = Vec::new();
    for substitution in &substitutions {
        let contents = match &substitution.kind {
            super::audit::SafeShellSubstitutionKind::FileRead { path } => {
                read_literal_file_substitution(path)?
            }
            super::audit::SafeShellSubstitutionKind::Command { inner } => {
                // The inner command must pass the full gate too: re-run
                // validate first (guards against relaxed classification), then
                // the Git confirmation gate. The
                // audit classifier only proves the inner command passed
                // validate; it does not intercept confirm-gated git commands.
                // Without this, `echo "$(git commit -am x)"` would commit
                // before the outer confirmation, leaving only `echo '...'`
                // after materialization and bypassing the gate.
                super::audit::validate_execute_command(inner)
                    .map_err(|reason| format!("inner substitution blocked: {reason}"))?;
                confirm_git_confirm_gated_if_needed(inner)?;
                execute_inner_shell_command(inner, cwd)?
            }
        };
        let replacement = if let Some(name) = substitution.assignment.as_deref() {
            // Assignment values are never word-split by the shell, so one
            // quoted literal preserves the exact value.
            let replacement = format!("{name}={}", shell_single_quote(&contents));
            assigned.push((name.to_string(), contents));
            replacement
        } else if substitution.bare {
            // Split bare substitution output into quoted words so values cannot
            // become shell syntax during materialization. Empty output removes the word.
            let pieces: Vec<String> = contents
                .split_whitespace()
                .map(shell_single_quote)
                .collect();
            if pieces.is_empty() {
                String::new()
            } else {
                pieces.join(" ")
            }
        } else {
            shell_single_quote(&contents)
        };
        edits.push((substitution.start, substitution.end, replacement));
    }
    for reference in &references {
        let Some((_, value)) = assigned.iter().find(|(name, _)| name == reference.name) else {
            // The plan only emits references it proved bindable; reaching here
            // means the shapes diverged, so refuse rather than let the shell
            // expand an audited-as-opaque `$NAME`.
            return Err(format!(
                "cannot bind `${}` to its materialized assignment; use a literal value",
                reference.name
            ));
        };
        let replacement = if reference.in_double {
            let mut escaped = String::with_capacity(value.len());
            push_double_quoted_value(&mut escaped, value);
            escaped
        } else {
            shell_single_quote(value)
        };
        edits.push((reference.start, reference.end, replacement));
    }
    // Apply every edit once, in order. Overlapping spans would mean an offset
    // plan bug, so fail closed instead of building a command nobody audited.
    edits.sort_by_key(|(start, _, _)| *start);
    let mut materialized = String::with_capacity(command.len());
    let mut cursor = 0;
    for (start, end, replacement) in edits {
        if start < cursor || end > command.len() {
            return Err(
                "materialized command substitutions overlap; run each segment as a \
                 separate command"
                    .to_string(),
            );
        }
        materialized.push_str(&command[cursor..start]);
        materialized.push_str(&replacement);
        cursor = end;
    }
    materialized.push_str(&command[cursor..]);
    Ok(materialized)
}

/// A `$NAME` / `${NAME}` reference whose value a materialized assignment fixes.
struct PlannedReference<'a> {
    start: usize,
    end: usize,
    in_double: bool,
    name: &'a str,
}

/// One classified substitution span, as the reference planner sees it.
struct SubstitutionSpan<'a> {
    start: usize,
    end: usize,
    assignment: Option<&'a str>,
}

/// Builtins that write a variable, so a later reference no longer reads the
/// materialized assignment value.
const ASSIGNING_BUILTINS: [&str; 8] = [
    "declare", "export", "let", "local", "read", "readonly", "typeset", "unset",
];

/// Builtins that run shell code the audit cannot read. When a command
/// materializes a variable, such code could reassign it behind the expanded
/// literal's back, so the pair is rejected instead of expanded.
const DYNAMIC_CODE_BUILTINS: [&str; 3] = ["eval", "source", "."];

/// How a shell word participates in its segment.
enum ShellWordRole {
    /// `NAME=...` before the command word: an assignment, not the command.
    Assignment,
    /// The segment's command word.
    Command,
    /// Any other argument.
    Argument,
}

/// Plan literal expansion for every reference to a variable this command
/// materializes.
///
/// Shell variables are command-global and assigned in order, so the plan is
/// all-or-nothing: a reference reads the materialized value only when nothing
/// can change the variable first. A reassignment in assignment position, an
/// assigning builtin, `printf -v NAME`, a reference before the assignment, or a
/// reference inside another substitution's text all make the value unprovable,
/// so they reject the command instead of leaving `$NAME` for the shell to
/// expand after the audit has already passed.
fn plan_assignment_references<'a>(
    command: &'a str,
    spans: &'a [SubstitutionSpan<'a>],
) -> Result<Vec<PlannedReference<'a>>, String> {
    let mut assignments: Vec<(&'a str, usize, usize)> = Vec::new();
    for span in spans {
        let Some(name) = span.assignment else {
            continue;
        };
        if assignments.iter().any(|(seen, _, _)| *seen == name) {
            return Err(format!(
                "`{name}` is assigned by more than one command substitution here; its \
                 value would depend on evaluation order"
            ));
        }
        assignments.push((name, span.start, span.end));
    }
    if assignments.is_empty() {
        return Ok(Vec::new());
    }
    // Substitutions run separately and cannot observe earlier materialized
    // assignments from this command. Reject references that would depend on
    // the outer shell's assignment order.
    for span in spans {
        let text = &command[span.start..span.end];
        for (name, _, _) in &assignments {
            if contains_variable_reference(text, name) {
                return Err(format!(
                    "command substitution text reads `${name}`, which this command \
                     materializes separately; use a literal value"
                ));
            }
            // Arithmetic contexts can read names without `$`; the check above
            // would miss dependencies on another materialized assignment.
            if arithmetic_mentions(text, name) {
                return Err(format!(
                    "command substitution text reads `{name}` inside an arithmetic \
                     expression, where this command materializes it separately; use a \
                     literal value"
                ));
            }
        }
    }

    let bytes = command.as_bytes();
    let mut references = Vec::new();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut word_start: Option<usize> = None;
    let mut command_word_seen = false;
    let mut segment_assigns_materialized = false;
    let mut assigning_builtin = false;
    let mut printf_command = false;
    let mut printf_v = false;
    let mut awaiting_wrapped_command = false;
    let mut span = 0;
    while i < bytes.len() {
        if span < spans.len() && i == spans[span].start {
            // A materialized assignment must be a standalone assignment segment.
            // Reject it after a command word; the command-word branch below also
            // rejects prefix assignments whose values would be temporary.
            let assigned_name = spans[span].assignment;
            if let Some(name) = assigned_name {
                if command_word_seen {
                    return Err(format!(
                        "`{name}` is materialized in front of a command word, which makes it \
                         a temporary environment assignment; its value does not persist"
                    ));
                }
                segment_assigns_materialized = true;
            }
            i = spans[span].end;
            span += 1;
            word_start = None;
            continue;
        }
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                in_single = false;
            }
            i += 1;
            continue;
        }
        if b == b'\\' {
            i += 2;
            continue;
        }
        if b == b'\'' && !in_double {
            in_single = true;
            word_start.get_or_insert(i);
            i += 1;
            continue;
        }
        if b == b'"' {
            in_double = !in_double;
            word_start.get_or_insert(i);
            i += 1;
            continue;
        }
        if b == b'$' {
            word_start.get_or_insert(i);
            let rest = &command[i..];
            if rest.starts_with("$((") {
                let end = skip_shell_arithmetic(command, i)
                    .ok_or_else(|| "unbalanced arithmetic expansion in command".to_string())?;
                let body = &command[i + 3..end - 2];
                for (name, _, _) in &assignments {
                    if arithmetic_writes(body, name) {
                        return Err(format!(
                            "`$(( ... ))` updates `{name}` with arithmetic, and this command \
                             materializes that variable; the audit cannot prove which value a \
                             later reference reads"
                        ));
                    }
                }
                i = end;
                continue;
            }
            if rest.starts_with("$(") {
                // Every active substitution is either classified above or the
                // audit already rejected the command; reaching here means the
                // shape changed, so refuse instead of leaving it to the shell.
                return Err("unproven command substitution in a materialized command".to_string());
            }
            if let Some((name, len, plain)) = variable_reference(rest) {
                if let Some((_, _, assignment_end)) =
                    assignments.iter().find(|(seen, _, _)| *seen == name)
                {
                    if !plain {
                        return Err(format!(
                            "`${{{name}...}}` applies modifiers the materializer cannot bind; \
                             use a literal value"
                        ));
                    }
                    if i < *assignment_end {
                        return Err(format!(
                            "`${name}` is read before this command assigns it; the audit \
                             cannot tell which value the shell uses"
                        ));
                    }
                    references.push(PlannedReference {
                        start: i,
                        end: i + len,
                        in_double,
                        name,
                    });
                }
                i += len;
                continue;
            }
            i += 1;
            continue;
        }
        if !in_double && (b.is_ascii_whitespace() || is_shell_word_separator(b)) {
            if let Some(start) = word_start.take() {
                let word = &command[start..i];
                check_arithmetic_write(word, &assignments)?;
                match shell_word_role(word, command_word_seen) {
                    ShellWordRole::Assignment => {
                        let reused = assignment_word_name(word).is_some_and(|name| {
                            assignments.iter().any(|(seen, _, _)| *seen == name)
                        });
                        if reused {
                            return Err(format!(
                                "`{word}` reassigns a variable this command materializes; the \
                                 audit cannot tell which value a later reference reads"
                            ));
                        }
                    }
                    ShellWordRole::Command => {
                        if segment_assigns_materialized {
                            return Err(
                                "a materialized assignment in front of a command word becomes \
                                 a temporary environment assignment; its value does not persist"
                                    .to_string(),
                            );
                        }
                        command_word_seen = true;
                        if let Some(name) = word_literal(word) {
                            if DYNAMIC_CODE_BUILTINS.contains(&name) {
                                return Err(format!(
                                    "`{name}` runs code this audit cannot read; a command that \
                                     materializes a variable cannot prove the value stays fixed"
                                ));
                            }
                            // `command`/`builtin`/`exec` run the next word as the
                            // command, so an assigning builtin can be one word later.
                            awaiting_wrapped_command =
                                matches!(name, "command" | "builtin" | "exec");
                            assigning_builtin = ASSIGNING_BUILTINS.contains(&name);
                            printf_command = name == "printf";
                        } else {
                            awaiting_wrapped_command = false;
                            assigning_builtin = false;
                            printf_command = false;
                        }
                    }
                    ShellWordRole::Argument => {
                        // The first non-option word after a wrapper is the real
                        // command word: `command export L=2` still assigns `L`.
                        if awaiting_wrapped_command && !word.starts_with('-') {
                            awaiting_wrapped_command = false;
                            if let Some(name) = word_literal(word) {
                                assigning_builtin = ASSIGNING_BUILTINS.contains(&name);
                                printf_command = name == "printf";
                            }
                        }
                        let literal = word_literal(word).unwrap_or(word);
                        let overwrites = assignments.iter().any(|(seen, _, _)| *seen == literal)
                            || assignment_word_name(literal).is_some_and(|name| {
                                assignments.iter().any(|(seen, _, _)| *seen == name)
                            });
                        if overwrites && (assigning_builtin || printf_v) {
                            return Err(format!(
                                "`{word}` overwrites a variable this command materializes; the \
                                 audit cannot tell which value a later reference reads"
                            ));
                        }
                        if printf_command && literal == "-v" {
                            printf_v = true;
                        } else if printf_v {
                            printf_v = false;
                        }
                    }
                }
            }
            if is_shell_word_separator(b) {
                command_word_seen = false;
                segment_assigns_materialized = false;
                assigning_builtin = false;
                printf_command = false;
                printf_v = false;
                awaiting_wrapped_command = false;
            }
            i += 1;
            continue;
        }
        word_start.get_or_insert(i);
        i += 1;
    }
    if let Some(start) = word_start {
        let word = &command[start..];
        check_arithmetic_write(word, &assignments)?;
        if shell_word_role(word, command_word_seen).is_assignment()
            && assignment_word_name(word)
                .is_some_and(|name| assignments.iter().any(|(seen, _, _)| *seen == name))
        {
            return Err(format!(
                "`{word}` reassigns a variable this command materializes; the audit cannot \
                 tell which value a later reference reads"
            ));
        }
    }
    Ok(references)
}

impl ShellWordRole {
    fn is_assignment(&self) -> bool {
        matches!(self, Self::Assignment)
    }
}

fn shell_word_role(word: &str, command_word_seen: bool) -> ShellWordRole {
    if !command_word_seen && assignment_word_name(word).is_some() {
        ShellWordRole::Assignment
    } else if command_word_seen {
        ShellWordRole::Argument
    } else {
        ShellWordRole::Command
    }
}

/// `NAME` when `word` is an `NAME=value` / `NAME+=value` assignment word.
fn assignment_word_name(word: &str) -> Option<&str> {
    if word.starts_with(['"', '\'']) {
        return None;
    }
    let (name, _) = word.split_once('=')?;
    let name = name.strip_suffix('+').unwrap_or(name);
    is_shell_name(name).then_some(name)
}

/// The text the shell passes through unchanged: an unquoted word, or one
/// whole-word quote pair whose body expands and escapes nothing. Returns `None`
/// when the word needs `$`/backtick expansion or escape processing, because the
/// audit cannot then know what the shell reads or writes.
fn word_literal(word: &str) -> Option<&str> {
    if let Some(inner) = word
        .strip_prefix('\'')
        .and_then(|rest| rest.strip_suffix('\''))
    {
        return (!inner.contains(['\'', '\\'])).then_some(inner);
    }
    if let Some(inner) = word
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
    {
        return (!inner.contains(['"', '\\', '$', '`'])).then_some(inner);
    }
    (!word.contains(['\'', '"', '\\', '$', '`'])).then_some(word)
}

/// The variable an arithmetic word updates: `L++`, `L--`, `++L`, or `--L`.
fn arithmetic_write_target(word: &str) -> Option<&str> {
    let name = word
        .strip_suffix("++")
        .or_else(|| word.strip_suffix("--"))
        .or_else(|| word.strip_prefix("++"))
        .or_else(|| word.strip_prefix("--"))?;
    is_shell_name(name).then_some(name)
}

/// Reject arithmetic-shaped updates of materialized variables in any word
/// position. Forms such as `let L++` can mutate the value that later references
/// were planned to read, so the materializer rejects them before expansion.
fn check_arithmetic_write(word: &str, assignments: &[(&str, usize, usize)]) -> Result<(), String> {
    let literal = word_literal(word).unwrap_or(word);
    let Some(name) = arithmetic_write_target(literal) else {
        return Ok(());
    };
    if assignments.iter().any(|(seen, _, _)| *seen == name) {
        return Err(format!(
            "`{word}` updates a variable this command materializes with arithmetic; the audit \
             cannot prove which value a later reference reads"
        ));
    }
    Ok(())
}

fn contains_variable_reference(text: &str, name: &str) -> bool {
    let mut from = 0;
    while let Some(found) = text[from..].find('$') {
        let at = from + found;
        match variable_reference(&text[at..]) {
            Some((seen, len, _)) => {
                if seen == name {
                    return true;
                }
                from = at + len;
            }
            None => from = at + 1,
        }
    }
    false
}

fn is_shell_word_separator(b: u8) -> bool {
    matches!(b, b';' | b'|' | b'&' | b'<' | b'>' | b'(' | b')')
}

/// Split a `$NAME` or `${NAME}` reference off the front of `rest` (which starts
/// with `$`), returning the name, the number of bytes it spans, and whether the
/// form is plain. Braced forms with modifiers (`${NAME:-x}`) are reported as
/// non-plain because their expansion rules differ. Returns `None` for other `$`
/// forms (`$1`, `$?`, `$(`, ...).
fn variable_reference(rest: &str) -> Option<(&str, usize, bool)> {
    let after = rest.strip_prefix('$')?;
    if let Some(body) = after.strip_prefix('{') {
        let close = body.find('}')?;
        let inner = &body[..close];
        let name_end = inner
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(inner.len());
        let name = &inner[..name_end];
        if !is_shell_name(name) {
            return None;
        }
        return Some((name, close + 3, name_end == inner.len()));
    }
    let end = after
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(after.len());
    let name = after.get(..end)?;
    if !is_shell_name(name) {
        return None;
    }
    Some((name, end + 1, true))
}

fn is_shell_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Append a literal value inside an existing double-quoted shell string.
/// Escape `\`, `"`, `` ` ``, and `$` to prevent additional shell expansion.
fn push_double_quoted_value(out: &mut String, value: &str) {
    for ch in value.chars() {
        if matches!(ch, '\\' | '"' | '`' | '$') {
            out.push('\\');
        }
        out.push(ch);
    }
}

/// Index just past the `))` closing the arithmetic expansion at `dollar_at`.
fn skip_shell_arithmetic(command: &str, dollar_at: usize) -> Option<usize> {
    let bytes = command.as_bytes();
    let mut depth = 2usize;
    let mut i = dollar_at + 3;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Whether `b` can be part of a shell identifier, so a name match inside a
/// longer identifier is not a variable reference.
fn is_identifier_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Whether `expr` mentions `name` as a standalone identifier.
fn mentions_identifier(expr: &str, name: &str) -> bool {
    let bytes = expr.as_bytes();
    expr.match_indices(name).any(|(at, _)| {
        (at == 0 || !is_identifier_byte(bytes[at - 1]))
            && (at + name.len() == bytes.len() || !is_identifier_byte(bytes[at + name.len()]))
    })
}

/// Whether an arithmetic expression writes `name` via `++`, `--`, or an
/// assignment operator. A `$`-prefixed mention reads the expanded value rather
/// than writing the variable, and comparisons (`==`, `>=`, `<=`, `!=`) are not
/// assignments.
fn arithmetic_writes(expr: &str, name: &str) -> bool {
    let bytes = expr.as_bytes();
    for (at, _) in expr.match_indices(name) {
        let before_ok = at == 0 || (!is_identifier_byte(bytes[at - 1]) && bytes[at - 1] != b'$');
        let after = at + name.len();
        if !before_ok || (after < bytes.len() && is_identifier_byte(bytes[after])) {
            continue;
        }
        let previous = expr[..at].trim_end();
        if previous.ends_with("++") || previous.ends_with("--") {
            return true;
        }
        let next = expr[after..].trim_start();
        if next.starts_with("++") || next.starts_with("--") {
            return true;
        }
        if ["+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^="]
            .iter()
            .any(|op| next.starts_with(op))
        {
            return true;
        }
        if next
            .strip_prefix('=')
            .is_some_and(|rest| !rest.starts_with('='))
        {
            return true;
        }
    }
    false
}

/// Whether `text` references `name` through an arithmetic context, where the
/// shell substitutes the variable without a `$` prefix: `$(( ... ))` bodies and
/// `let` arguments.
fn arithmetic_mentions(text: &str, name: &str) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // The cursor advances in bytes; only slice text after locating the
        // ASCII delimiters, which are necessarily UTF-8 character boundaries.
        if bytes[i..].starts_with(b"$((") {
            if let Some(end) = skip_shell_arithmetic(text, i) {
                if mentions_identifier(&text[i + 3..end - 2], name) {
                    return true;
                }
                i = end;
                continue;
            }
        }
        i += 1;
    }
    // `let` evaluates its arguments as arithmetic, but only in command
    // position; the leading `$(` of an enclosing substitution is trimmed so the
    // first word of the inner command still reads as that command.
    for segment in text.split([';', '|', '&']) {
        let mut words = segment.split_whitespace();
        let Some(first) = words.next() else {
            continue;
        };
        if first.trim_start_matches(['$', '(', '`']) == "let" {
            return words.any(|argument| mentions_identifier(argument, name));
        }
    }
    false
}

/// Truncate over-long output keeping head and tail, with **actionable
/// metadata** in the middle: total count, shown count, and an explicit warning
/// that the omitted middle may contain the lines the caller is looking for —
/// "not seen" does not mean "not present".
pub(super) fn truncate_chars(content: &str, max_chars: usize) -> String {
    let total_chars = content.chars().count();
    if total_chars <= max_chars {
        return content.to_string();
    }
    let total_lines = content.lines().count();
    let head_chars = (max_chars * 3 / 4).max(1);
    let tail_chars = max_chars.saturating_sub(head_chars);
    let head: String = content.chars().take(head_chars).collect();
    let tail: String = content
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let head_lines = head.lines().count();
    let tail_lines = tail.lines().count();
    let mut output = String::with_capacity(max_chars + 384);
    output.push_str(&head);
    output.push_str(&format!(
        "\n... [truncated: omitted middle; showing first {head_chars} and last {tail_chars} of {total_chars} chars \
(~{head_lines} + {tail_lines} of {total_lines} lines); expected matches may be there, not absent. \
Do not re-run near-identical variants; narrow the query instead (e.g. `grep -c` or a more specific pattern). To page a local file, prefer `read_file` with offset/limit; `sed -n 'START,ENDp'` only for line paging, `head -c`/`tail -c +N` for byte windows or non-text files.]\n"
    ));
    output.push_str(&tail);
    output
}

// =========================================================================
// Execution logic (command validation lives in the audit module)
// =========================================================================

/// `git stash` subactions that are read-only — they touch neither the working
/// tree nor stash entries, so they need no confirmation. A word only counts as
/// the subaction when it directly follows `stash` (first positional): in
/// `git stash push list` the `list` is a path argument and still requires
/// confirmation.
const READ_ONLY_GIT_STASH_SUBACTIONS: &[&str] = &["list", "show"];

/// Git subcommands that must be confirmed by the user before running
/// (`git commit`; `git push` in every form — pushing reaches the remote, so it
/// is always confirmed; `git stash` and its subactions that change the working
/// tree or stash entries, e.g. pop/drop/clear — read-only `git stash list`/
/// `show` need no confirmation; `git checkout` / `git switch` in every form —
/// the destructive forms (`-- <path>`, `-f`, `-B`, `--discard-changes`) are
/// additionally hard-blocked by the audit layer before this gate runs). Reuses
/// the audit layer's lexical parsing so
/// materialized quoted arguments cannot bypass confirmation, and data like
/// `echo 'git commit'` is not mistaken for a real command. Returns the matched
/// subcommand name (`commit` / `push` / `stash` / `checkout` / `switch`) for
/// prompt and error wording.
fn confirm_gated_git_subcommand(command: &str) -> Option<&'static str> {
    for segment in super::audit::split_unquoted_segments(command) {
        let tokens = super::audit::effective_command_tokens(&segment);
        let Some(program) = tokens
            .first()
            .and_then(|token| Path::new(token).file_name().and_then(|name| name.to_str()))
        else {
            continue;
        };
        if program != "git" {
            continue;
        }

        // Reuse the audit layer's single subcommand parser instead of a local
        // option list: it skips every git global option form (`-C`/`-c`
        // with value, `--git-dir=...`, plain flags like `--no-pager`/`--bare`,
        // `--` before the subcommand), so no option variant can escape the
        // confirmation gate the way it could not escape the audit blocklist.
        let Some(j) = super::audit::git_subcommand_index(&tokens) else {
            continue;
        };
        if let Some(sub) = tokens.get(j).map(String::as_str) {
            // `sub` borrows the local `tokens`; return the matching static
            // instead so the signature (`Option<&'static str>`) holds.
            if sub == "commit" {
                return Some("commit");
            }
            // Case-insensitive like the audit layer's block list, so `git PUSH`
            // cannot escape the gate the way it could not escape the old
            // hard block.
            if sub.eq_ignore_ascii_case("push") {
                return Some("push");
            }
            if sub == "checkout" {
                return Some("checkout");
            }
            if sub == "switch" {
                // Every `git checkout` / `git switch` form requires
                // confirmation. Destructive forms (`-- <path>`, `-f`/`-B`,
                // `--discard-changes`) never reach this gate: the audit layer
                // hard-blocks them first, so they cannot be approved; plain
                // branch switching (and other safe forms) lands here.
                return Some("switch");
            }
            if sub == "stash" {
                // Read-only subactions (list/show) directly after `stash`
                // need no confirmation; everything else (bare `git stash`=push,
                // `git stash -- <path>`, or stash options like `git stash -q
                // list`) still goes through the confirmation gate (fail-closed).
                if matches!(
                    tokens.get(j + 1).map(String::as_str),
                    Some("list") | Some("show")
                ) {
                    continue;
                }
                return Some("stash");
            }
        }
    }
    None
}

/// Ask the user to confirm confirm-gated git subcommands (`git commit` /
/// `git push` / `git stash` / `git checkout` / `git switch`) before running
/// them.
/// - Interactive terminal: red-highlighted prompt; y proceeds, n / Ctrl+C /
///   Esc cancels.
/// - Serve turn child with a remote confirmation channel: the question travels
///   to the serve client (phone/browser) and the answer arrives on stdin.
/// - Other non-interactive environments: rejected outright (fail-closed), both
///   to avoid background processes hanging on input and to avoid silent
///   execution.
fn confirm_git_confirm_gated_if_needed(command: &str) -> Result<(), String> {
    let Some(subcommand) = confirm_gated_git_subcommand(command) else {
        return Ok(());
    };
    let prompt = format!("\nConfirm git {subcommand}:\n{command}\nProceed? (y/n): ");
    let confirmed = if crate::ai::serve_confirm::active() {
        crate::ai::serve_confirm::confirm(&prompt)
    } else {
        if !std::io::stdin().is_terminal() {
            return Err(format!(
                "Command blocked: git {subcommand} requires user confirmation, but stdin is not \
                 an interactive terminal. Do not retry the command; report to the user and wait \
                 for explicit confirmation (or have them run it in an interactive session)."
            ));
        }
        crate::commonw::prompt::prompt_yes_or_no_danger(&prompt)
    };
    match confirmed {
        Some(true) => Ok(()),
        Some(false) => Err(format!("git {subcommand} canceled by user")),
        None => Err(format!("git {subcommand} canceled by user (Ctrl+C)")),
    }
}

fn format_command_result(output: CommandRunResult, timeout_secs: u64) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let stdout_trimmed = stdout.trim();
    let stderr_trimmed = stderr.trim();
    let combined = if stdout_trimmed.is_empty() {
        stderr_trimmed.to_string()
    } else if stderr_trimmed.is_empty() {
        stdout_trimmed.to_string()
    } else {
        format!("{stdout_trimmed}\n{stderr_trimmed}")
    };

    if output.stalled {
        // A no-output stall may be waiting for human input or buffering output.
        // Include captured output so the caller can distinguish it from an
        // ordinary timeout before retrying.
        let partial = if combined.trim().is_empty() {
            "(no output was captured before termination)".to_string()
        } else {
            format!(
                "Partial output captured before termination:\n{}",
                combined.trim()
            )
        };
        let msg = "Command appears to be waiting for interactive input and was terminated: it kept running without producing output for a sustained period. This usually means the command is interactive — e.g. a QR-code login, a password prompt, or a menu — and cannot proceed without human input; or its output is buffered by a pipe (like `| tail`), which only flushes when the command exits, so nothing was visible. If the command is a long-running server or daemon, start it with `yield_time_ms` and poll it with `command_session`. For login flows, prefer the CLI's non-blocking options (e.g. `--begin`/`--complete`, `--qr-image`, `--no-terminal-qr`, `-y`).";
        return truncate_chars(&format!("{msg}\n{partial}"), MAX_COMMAND_OUTPUT_CHARS);
    }

    if output.timed_out || output.cancelled {
        let reason = if output.timed_out {
            format!("Command timed out after {timeout_secs}s and was terminated.")
        } else {
            "Command was cancelled and terminated.".to_string()
        };
        let partial = if combined.trim().is_empty() {
            "(no output was captured before termination)".to_string()
        } else {
            format!(
                "Partial output captured before termination:\n{}",
                combined.trim()
            )
        };
        return truncate_chars(&format!("{reason}\n{partial}"), MAX_COMMAND_OUTPUT_CHARS);
    }

    let status = output
        .status
        .expect("completed command must carry a status");
    if status.success() {
        let combined = combined.trim();
        // Make a successful empty result explicit so the caller can distinguish
        // it from a command that did not run before deciding whether to retry.
        if combined.is_empty() {
            "(command succeeded with exit code 0 and produced no output)".to_string()
        } else {
            truncate_chars(combined, MAX_COMMAND_OUTPUT_CHARS)
        }
    } else {
        truncate_chars(
            &format!(
                "Exit code: {}\n{}\n{}",
                status.code().unwrap_or(-1),
                stdout_trimmed,
                stderr_trimmed
            ),
            MAX_COMMAND_OUTPUT_CHARS,
        )
    }
}

/// Detect direct or wrapped `sleep` programs, including simple shell `-c` bodies.
/// Synchronous execution occupies the tool call until completion; commands
/// started with `yield_time_ms` can return a session to poll. The audit splitter
/// and wrapper parser distinguish programs from data arguments such as `grep sleep file`.
fn contains_sleep(command: &str) -> bool {
    super::audit::split_unquoted_segments(command)
        .iter()
        .any(|segment| {
            let tokens = super::audit::effective_command_tokens(segment);
            let program = tokens
                .first()
                .and_then(|token| Path::new(token).file_name())
                .and_then(|name| name.to_str());
            if program == Some("sleep") {
                return true;
            }
            // Shells forwarding a script body via `-c` execute that body directly
            // (`bash -c "sleep 5"`, `sh -c 'sleep 3'`), so re-parse the body to
            // find `sleep` instead of treating it as data. `timeout 5 sleep 3`
            // is already unwrapped by the audit wrapper logic.
            if matches!(program, Some("bash" | "sh" | "zsh" | "ksh" | "dash")) {
                let Some(dash_c) = tokens.iter().position(|token| token == "-c") else {
                    return false;
                };
                let body = tokens[dash_c + 1..].join(" ");
                return super::audit::effective_command_tokens(&body)
                    .first()
                    .and_then(|token| Path::new(token).file_name())
                    .and_then(|name| name.to_str())
                    == Some("sleep");
            }
            false
        })
}

/// Guidance prepended to a `sleep` command's result: the hint lives in the tool
/// description too, but showing it at execution time makes it hard to miss.
const SLEEP_WARNING: &str = "Note: this command runs `sleep`, which blocks the whole turn on \
dead time — it is expensive. Use it only when truly necessary (e.g. waiting for a server to \
accept connections); prefer starting the work with `yield_time_ms` and polling it with \
`command_session`, or `task_wait` for subagent tasks.";

fn execute_command_inner<F>(args: &Value, mut on_chunk: F) -> Result<String, String>
where
    F: FnMut(&[u8]),
{
    // Session control is a separate tool. Reject meaningful legacy fields when
    // the service is called directly; the published schema already omits them.
    if non_empty_str_arg(args, "session_id").is_some()
        || non_empty_str_arg(args, "write_stdin").is_some()
        || args["close_session"].as_bool() == Some(true)
    {
        return Err(
            "Invalid arguments: session control moved to `command_session`; `execute_command` only starts commands"
                .to_string(),
        );
    }
    let yield_ms = args["yield_time_ms"].as_u64().map(resolve_session_yield);
    let raw_command = super::execute_command_arg(args).ok_or("Missing command")?;
    let cwd = args["cwd"].as_str().filter(|dir| !dir.trim().is_empty());
    // Materialize harmless "$(...)" first (cat literals and generic harmless
    // commands); the result is still fully audited afterwards.
    let command = materialize_safe_shell_substitutions(raw_command, cwd)
        .map_err(|reason| format!("Command blocked: {reason}"))?;
    let pseudo_terminal = args["pty"].as_bool().unwrap_or(false);
    let input = non_empty_str_arg(args, "input");
    if let Some(input) = input
        && input.len() > MAX_COMMAND_INPUT_BYTES
    {
        return Err(format!(
            "Command blocked: input exceeds the {MAX_COMMAND_INPUT_BYTES}-byte limit"
        ));
    }
    let (default_timeout, max_timeout) = config_command_timeout_bounds();
    let timeout = resolve_command_timeout(args["timeout"].as_u64(), default_timeout, max_timeout);

    // Command safety validation is delegated to the audit module.
    super::audit::validate_execute_command(&command)
        .map_err(|reason| format!("Command blocked: {reason}"))?;

    // Confirm-gated Git subcommands ask the user first; without a terminal
    // or a serve confirmation channel, the command is rejected.
    confirm_git_confirm_gated_if_needed(&command)?;

    // `execute_command` is a facade: when a simple command names a program that
    // is not installed, an internal implementation can satisfy it. Interactive
    // requests keep the shell path so a missing binary fails visibly.
    if !pseudo_terminal && input.is_none() {
        if let Some(result) = super::command_facade::try_run(&command, cwd, timeout) {
            let rendered = result?;
            on_chunk(rendered.as_bytes());
            return Ok(rendered);
        }
    }

    if let Some(wait) = yield_ms {
        let mut output = super::super::storage::command_session::start(
            &command,
            cwd,
            pseudo_terminal,
            input,
            wait,
        )?;
        if contains_sleep(&command) {
            output = format!("{SLEEP_WARNING}\n{output}");
        }
        return Ok(output);
    }

    let output = command_runner::run_command_streaming(
        &command,
        cwd,
        timeout,
        pseudo_terminal,
        input,
        on_chunk,
    )?;
    let interrupted = output.timed_out || output.cancelled || output.stalled;
    let mut formatted = format_command_result(output, timeout);
    if contains_sleep(&command) {
        formatted = format!("{SLEEP_WARNING}\n{formatted}");
    }
    if interrupted {
        Err(formatted)
    } else {
        Ok(formatted)
    }
}

pub(crate) fn execute_command(args: &Value) -> Result<String, String> {
    execute_command_inner(args, |_| {})
}

pub(crate) fn execute_command_streaming<F>(args: &Value, on_chunk: F) -> Result<String, String>
where
    F: FnMut(&[u8]),
{
    execute_command_inner(args, on_chunk)
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_COMMAND_INPUT_BYTES, MAX_COMMAND_OUTPUT_CHARS, SLEEP_WARNING,
        confirm_gated_git_subcommand, confirm_git_confirm_gated_if_needed, contains_sleep,
        execute_command, format_command_result, materialize_safe_shell_substitutions,
        resolve_command_timeout, truncate_chars,
    };
    use crate::cmd::run::CommandRunResult;
    use serde_json::json;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temporary_file_path(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rust_tools_{label}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    // ---- confirm-gated git subcommands (commit / push / stash / checkout / switch) ----

    #[test]
    fn materializes_bare_substitution_as_split_quoted_words() {
        let path = temporary_file_path("bare-substitution");
        fs::write(&path, "alpha beta\ngamma").unwrap();
        // Bare whole-word `$(...)`: output is word-split, each piece quoted.
        let command = format!("echo $(cat {})", path.display());
        let materialized = materialize_safe_shell_substitutions(&command, None).unwrap();
        assert_eq!(materialized, "echo 'alpha' 'beta' 'gamma'");
        // Quoted whole-word `"$(...)"`: output stays one quoted word.
        let command = format!("echo \"$(cat {})\"", path.display());
        let materialized = materialize_safe_shell_substitutions(&command, None).unwrap();
        assert_eq!(materialized, "echo 'alpha beta\ngamma'");
        // Empty bare output removes the word entirely (shell semantics).
        let empty = temporary_file_path("bare-substitution-empty");
        fs::write(&empty, "").unwrap();
        let command = format!("echo $(cat {})", empty.display());
        let materialized = materialize_safe_shell_substitutions(&command, None).unwrap();
        assert_eq!(materialized, "echo ");
    }

    #[test]
    fn commit_detection_matches_plain_git_commit() {
        assert_eq!(
            confirm_gated_git_subcommand("git commit -m \"fix x\""),
            Some("commit")
        );
        assert_eq!(confirm_gated_git_subcommand("git commit"), Some("commit"));
        assert_eq!(
            confirm_gated_git_subcommand("git commit --amend"),
            Some("commit")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git 'commit' -m message"),
            Some("commit")
        );
    }

    #[test]
    fn commit_detection_skips_global_options() {
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo commit -m x"),
            Some("commit")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git --git-dir=/repo/.git commit"),
            Some("commit")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -c user.name=X commit"),
            Some("commit")
        );
    }

    #[test]
    fn push_detection_matches_all_forms() {
        assert_eq!(confirm_gated_git_subcommand("git push"), Some("push"));
        assert_eq!(
            confirm_gated_git_subcommand("git push origin main"),
            Some("push")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git push --force-with-lease origin"),
            Some("push")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo push origin main"),
            Some("push")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git --git-dir=/repo push"),
            Some("push")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git --no-pager push origin main"),
            Some("push")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git 'push' origin main"),
            Some("push")
        );
        // Case-insensitive: the audit layer's old hard block matched `PUSH`,
        // so the gate must too, otherwise `git PUSH` bypasses confirmation.
        assert_eq!(
            confirm_gated_git_subcommand("git PUSH origin"),
            Some("push")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git status && git push -u origin main"),
            Some("push")
        );
    }

    #[test]
    fn commit_detection_finds_commit_in_command_chains() {
        assert_eq!(
            confirm_gated_git_subcommand("git add -A && git commit -m x"),
            Some("commit")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo add . && git -C /repo commit"),
            Some("commit")
        );
    }

    #[test]
    fn stash_detection_matches_all_stash_forms() {
        assert_eq!(confirm_gated_git_subcommand("git stash"), Some("stash"));
        assert_eq!(confirm_gated_git_subcommand("git stash pop"), Some("stash"));
        assert_eq!(
            confirm_gated_git_subcommand("git stash drop stash@{0}"),
            Some("stash")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git stash clear"),
            Some("stash")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo stash push -m wip"),
            Some("stash")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git stash && echo done"),
            Some("stash")
        );
        assert_eq!(confirm_gated_git_subcommand("env git stash"), Some("stash"));
        assert_eq!(
            confirm_gated_git_subcommand("xargs git stash"),
            Some("stash")
        );
        assert_eq!(
            confirm_gated_git_subcommand("nohup git stash"),
            Some("stash")
        );
        assert_eq!(
            confirm_gated_git_subcommand("command git stash"),
            Some("stash")
        );
        // Read-only subactions (list/show) directly after `stash` need no
        // confirmation.
        assert_eq!(confirm_gated_git_subcommand("git stash list"), None);
        assert_eq!(confirm_gated_git_subcommand("git stash show"), None);
        assert_eq!(
            confirm_gated_git_subcommand("git stash show stash@{0}"),
            None
        );
        assert_eq!(
            confirm_gated_git_subcommand("git stash list --format='%gd'"),
            None
        );
        // `list`/`show` outside the subaction position (`--` followed by a
        // path, `-q` a stash option, or coexisting with a confirm-gated
        // subaction) still require confirmation.
        assert_eq!(
            confirm_gated_git_subcommand("git stash -- list"),
            Some("stash")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git stash -q list"),
            Some("stash")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git stash list && git stash drop"),
            Some("stash")
        );
    }

    #[test]
    fn checkout_detection_matches_all_checkout_forms() {
        // Every `git checkout` form is confirm-gated, exactly like commit and
        // stash. Destructive forms (`-- <path>`, `-f`, `-B`, path-like args)
        // are additionally hard-blocked by the audit layer before this gate;
        // detection here still covers them so a fail-closed error is the
        // fallback wherever the audit layer is bypassed.
        assert_eq!(
            confirm_gated_git_subcommand("git checkout"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git checkout main"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git checkout -b feature/x"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git checkout -- src/main.rs"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git checkout -f main"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo checkout main"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git --git-dir=/repo/.git checkout main"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -c user.name=X checkout main"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git add -A && git checkout main"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo add . && git -C /repo checkout main"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("env git checkout main"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("nohup git checkout main"),
            Some("checkout")
        );
        assert_eq!(
            confirm_gated_git_subcommand("command git checkout main"),
            Some("checkout")
        );
    }

    #[test]
    fn switch_detection_matches_all_switch_forms() {
        // `git switch` is checkout's branch-switching sibling and follows the
        // exact same rule: every form is confirm-gated, destructive forms
        // (`-- <path>`, `-f`, `--discard-changes`) are hard-blocked by the
        // audit layer first.
        assert_eq!(confirm_gated_git_subcommand("git switch"), Some("switch"));
        assert_eq!(
            confirm_gated_git_subcommand("git switch main"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git switch -c feature/x"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git switch --detach"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git switch -f main"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo switch main"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git --git-dir=/repo/.git switch main"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -c user.name=X switch main"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git add -A && git switch main"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("git -C /repo add . && git -C /repo switch main"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("env git switch main"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("nohup git switch main"),
            Some("switch")
        );
        assert_eq!(
            confirm_gated_git_subcommand("command git switch main"),
            Some("switch")
        );
    }

    #[test]
    fn confirm_gated_detection_ignores_non_gated_commands() {
        assert_eq!(confirm_gated_git_subcommand("git status"), None);
        // Read-only branch listing is not confirm-gated.
        assert_eq!(confirm_gated_git_subcommand("git branch"), None);
        assert_eq!(confirm_gated_git_subcommand("git switchmessage"), None);
        assert_eq!(
            confirm_gated_git_subcommand("git log --oneline | grep commit"),
            None
        );
        assert_eq!(confirm_gated_git_subcommand("git svn commit"), None);
        assert_eq!(confirm_gated_git_subcommand("git svn checkout"), None);
        assert_eq!(
            confirm_gated_git_subcommand("echo 'git commit' > note.txt"),
            None
        );
        assert_eq!(confirm_gated_git_subcommand("git commitmessage"), None);
        assert_eq!(confirm_gated_git_subcommand("git stashlist"), None);
        assert_eq!(confirm_gated_git_subcommand("git checkoutmessage"), None);
        assert_eq!(
            confirm_gated_git_subcommand("echo 'git stash' > note.txt"),
            None
        );
        assert_eq!(
            confirm_gated_git_subcommand("echo 'git checkout' > note.txt"),
            None
        );
    }

    #[test]
    fn confirm_gated_fails_closed_without_terminal() {
        // Test stdin is not a terminal: confirm-gated git subcommands must be
        // rejected without hanging.
        for (command, sub) in [
            ("git commit -m x", "commit"),
            ("git stash", "stash"),
            ("git stash drop stash@{0}", "stash"),
            ("git -C /repo stash clear", "stash"),
            ("git checkout main", "checkout"),
            ("git -C /repo checkout -b feature/x", "checkout"),
            ("git switch main", "switch"),
            ("git -C /repo switch feature/x", "switch"),
        ] {
            let err = confirm_git_confirm_gated_if_needed(command).unwrap_err();
            assert!(err.contains("blocked"), "{command}: {err}");
            assert!(err.contains("confirmation"), "{command}: {err}");
            assert!(err.contains(sub), "{command}: {err}");
        }
    }

    #[test]
    fn execute_command_blocks_git_commit_inside_substitution_without_terminal() {
        // Without a TTY, `echo "$(git commit ...)"` must be
        // rejected fail-closed; command substitution must not silently bypass
        // the confirmation gate.
        let err = execute_command(&json!({
            "command": r#"echo "$(git commit -am x)""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap_err();
        assert!(err.contains("blocked"), "err: {err}");
        assert!(err.contains("confirmation"), "err: {err}");
    }

    #[test]
    fn execute_command_blocks_git_stash_inside_substitution_without_terminal() {
        // Same as commit: without a TTY, `echo "$(git stash)"` must fail
        // closed; command substitution must not silently bypass the
        // confirmation gate.
        let err = execute_command(&json!({
            "command": r#"echo "$(git stash)""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap_err();
        assert!(err.contains("blocked"), "err: {err}");
        assert!(err.contains("confirmation"), "err: {err}");
    }

    #[test]
    fn execute_command_blocks_git_checkout_inside_substitution_without_terminal() {
        // Same as commit/stash: without a TTY, `echo "$(git checkout)"` must
        // fail closed; command substitution must not silently bypass the
        // confirmation gate.
        let err = execute_command(&json!({
            "command": r#"echo "$(git checkout main)""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap_err();
        assert!(err.contains("blocked"), "err: {err}");
        assert!(err.contains("confirmation"), "err: {err}");
    }

    #[test]
    fn execute_command_blocks_git_switch_inside_substitution_without_terminal() {
        // Same as checkout: without a TTY, `echo "$(git switch)"` must fail
        // closed; command substitution must not silently bypass the
        // confirmation gate.
        let err = execute_command(&json!({
            "command": r#"echo "$(git switch main)""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap_err();
        assert!(err.contains("blocked"), "err: {err}");
        assert!(err.contains("confirmation"), "err: {err}");
    }

    #[test]
    fn confirm_gated_passes_through_non_gated_commands() {
        assert!(confirm_git_confirm_gated_if_needed("git status").is_ok());
        assert!(confirm_git_confirm_gated_if_needed("echo hello").is_ok());
        // Read-only stash subactions pass through even in non-interactive
        // environments without triggering confirmation.
        assert!(confirm_git_confirm_gated_if_needed("git stash list").is_ok());
        assert!(confirm_git_confirm_gated_if_needed("git stash show stash@{0}").is_ok());
    }

    // ---- contains_sleep ----

    #[test]
    fn sleep_detection_finds_effective_sleep_program() {
        assert!(contains_sleep("sleep 5"));
        assert!(contains_sleep("sleep 0.5"));
        assert!(contains_sleep("sleep 5 && cargo check"));
        assert!(contains_sleep("nohup sleep 30 &"));
        assert!(contains_sleep("env sleep 3"));
        assert!(contains_sleep("command sleep 1"));
        assert!(contains_sleep("/usr/bin/sleep 2"));
        // Programs forwarded through a shell `-c` body execute `sleep` directly.
        assert!(contains_sleep("bash -c \"sleep 5\""));
        assert!(contains_sleep("sh -c 'sleep 0.5'"));
        assert!(contains_sleep("timeout 5 sleep 3"));
    }

    #[test]
    fn sleep_detection_ignores_data_arguments() {
        assert!(!contains_sleep("grep sleep notes.txt"));
        assert!(!contains_sleep("man sleep"));
        assert!(!contains_sleep("echo sleep"));
        assert!(!contains_sleep("ls /usr/bin/sleep"));
        assert!(!contains_sleep("bash -c \"echo sleep\""));
        assert!(!contains_sleep("bash -c 'grep sleep notes.txt'"));
        assert!(!contains_sleep("timeout 5 echo 5"));
    }

    #[test]
    fn execute_command_prepends_sleep_warning() {
        let out = execute_command(&json!({
            "command": "sleep 0",
            "pty": false,
            "timeout": 5,
        }))
        .unwrap();
        assert!(
            out.starts_with(SLEEP_WARNING),
            "sleep result must carry the warning, got: {out}"
        );
    }

    #[test]
    fn execute_command_materializes_file_data_for_any_simple_outer_command() {
        let path = temporary_file_path("safe_file_read_substitution");
        let contents = "literal $(whoami); 'quoted'";
        fs::write(&path, contents).expect("write test file");

        let result = execute_command(&json!({
            "command": format!(r#"printf '%s' "$(cat {})""#, path.display()),
            "pty": false,
            "timeout": 5,
        }));
        let _ = fs::remove_file(&path);

        assert_eq!(result.unwrap(), contents);
    }

    #[test]
    fn execute_command_materializes_assignment_value_and_expands_references() {
        // The assignment becomes a literal and ordinary variable references expand
        // before auditing. Read-only arithmetic still evaluates in the outer shell
        // against that assigned value.
        let result = execute_command(&json!({
            "command": r#"L=$(printf 'x:42\n' | grep 'x:42' | cut -d: -f2); echo "line=$L"; echo "range=$((L-14)),$(($L+2))""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap();
        assert_eq!(result, "line=42\nrange=28,44");
    }

    #[test]
    fn execute_command_materializes_unicode_assignment_text() {
        for value in ["你好", "café", "😀"] {
            let result = execute_command(&json!({
                "command": format!(
                    r#"L=$(printf '%s' '{value}'); printf '<%s><%s>' "$L" "${{L}}""#
                ),
                "pty": false,
                "timeout": 5,
            }))
            .unwrap();
            assert_eq!(result, format!("<{value}><{value}>"));
        }
    }

    #[test]
    fn execute_command_blocks_assignment_value_that_names_blocked_program() {
        let target = temporary_file_path("assignment_program_target");
        fs::write(&target, "keep").expect("write target");
        let result = execute_command(&json!({
            "command": format!("cmd=$(printf rm); $cmd -rf {}", target.display()),
            "pty": false,
            "timeout": 5,
        }));
        let kept = fs::read_to_string(&target).ok();
        let _ = fs::remove_file(&target);
        assert_eq!(kept.as_deref(), Some("keep"), "materialized value ran `rm`");
        let err = result.unwrap_err();
        assert!(err.contains("rm"), "err: {err}");
    }

    #[test]
    fn execute_command_keeps_assignment_bindings_after_shorter_first_value() {
        // The first assignment materializes much shorter than its source text.
        // Offsets recorded against the pre-edit string would go stale and leave
        // `$B` unexpanded, so the audit would pass a program the shell resolves
        // from the variable; every edit is applied once against the original
        // text instead.
        let target = temporary_file_path("shifted_assignment_target");
        fs::write(&target, "keep").expect("write target");
        let result = execute_command(&json!({
            "command": format!(
                r#"A=$(printf 'yyyyyyyyyyyyyyyyyyyyyyyy') B=$(printf rm); $B -rf {}"#,
                target.display(),
            ),
            "pty": false,
            "timeout": 5,
        }));
        let kept = fs::read_to_string(&target).ok();
        let _ = fs::remove_file(&target);
        assert_eq!(kept.as_deref(), Some("keep"), "materialized value ran `rm`");
        let err = result.unwrap_err();
        assert!(err.contains("rm"), "err: {err}");
    }

    #[test]
    fn execute_command_keeps_assignment_bindings_after_longer_first_value() {
        // Same binding, with the first assignment growing: the materialized
        // literal is longer than the substitution text it replaces.
        let source = temporary_file_path("growing_assignment_source");
        fs::write(&source, "z".repeat(200)).expect("write source");
        let target = temporary_file_path("growing_assignment_target");
        fs::write(&target, "keep").expect("write target");
        let result = execute_command(&json!({
            "command": format!(
                r#"A=$(cat {}) B=$(printf rm); $B -rf {}"#,
                source.display(),
                target.display(),
            ),
            "pty": false,
            "timeout": 5,
        }));
        let kept = fs::read_to_string(&target).ok();
        let _ = fs::remove_file(&source);
        let _ = fs::remove_file(&target);
        assert_eq!(kept.as_deref(), Some("keep"), "materialized value ran `rm`");
        let err = result.unwrap_err();
        assert!(err.contains("rm"), "err: {err}");
    }

    #[test]
    fn execute_command_expands_multiple_assignments_in_one_segment() {
        let result = execute_command(&json!({
            "command": r#"A=$(printf 'x') B=$(printf 'y'); echo "$A$B""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap();
        assert_eq!(result, "xy");
    }

    #[test]
    fn execute_command_rejects_reference_the_materialized_value_cannot_cover() {
        // Each shape makes the value a later reference reads unprovable: a
        // reassignment, a temporary environment assignment, an assigning
        // builtin, `printf -v`, a read before the assignment, or a reference
        // inside another substitution's text. Silent literal expansion would
        // execute a command the user did not write, so they must be rejected.
        for (command, fragment) in [
            ("L=$(printf 1); L=2; echo \"$L\"", "reassigns"),
            ("L=$(printf 1) echo \"$L\"", "temporary environment"),
            ("L=$(printf 1); unset L; echo \"$L\"", "overwrites"),
            ("L=$(printf 1); export L=2; echo \"$L\"", "overwrites"),
            (
                "L=$(printf 1); printf -v L '%s' 2; echo \"$L\"",
                "overwrites",
            ),
            ("echo \"$L\" L=$(printf 1)", "read before"),
            (
                "N=$(printf a) M=$(printf '%s' \"$N\"); echo \"$M\"",
                "reads",
            ),
            ("L=$(printf 1); echo \"${L:-fallback}\"", "modifiers"),
        ] {
            let err = execute_command(&json!({
                "command": command,
                "pty": false,
                "timeout": 5,
            }))
            .unwrap_err();
            assert!(
                err.contains(fragment),
                "command {command:?} should mention {fragment:?}, got: {err}"
            );
        }
    }

    #[test]
    fn execute_command_rejects_arithmetic_write_to_materialized_variable() {
        // Arithmetic-shaped commands are rejected conservatively, since forms
        // such as `let L++` could invalidate later literal reference expansion.
        for command in [
            r#"L=$(printf 1); L++; echo "$L""#,
            r#"L=$(printf 1); L--; echo "$L""#,
            r#"L=$(printf 1); ++L; echo "$L""#,
            r#"L=$(printf 1); let L++; echo "$L""#,
        ] {
            let err = execute_command(&json!({
                "command": command,
                "pty": false,
                "timeout": 5,
            }))
            .unwrap_err();
            assert!(
                err.contains("arithmetic"),
                "command {command:?} should mention arithmetic, got: {err}"
            );
        }
    }

    #[test]
    fn execute_command_rejects_arithmetic_expansion_write_to_materialized_variable() {
        // `$(( ... ))` can reassign its variables, so a later `$L` expanded to
        // the materialized literal would disagree with the shell's own value.
        for command in [
            r#"L=$(printf 1); echo "$((L++))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((++L))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((L--))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((L+=1))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((L=2))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((L<<=2))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((L ++))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((++ L))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((L --))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((-- L))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((L += 1))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((L = 2))"; echo "$L""#,
            r#"L=$(printf 1); echo "$((L <<= 2))"; echo "$L""#,
        ] {
            let err = execute_command(&json!({
                "command": command,
                "pty": false,
                "timeout": 5,
            }))
            .unwrap_err();
            assert!(
                err.contains("arithmetic"),
                "command {command:?} should mention arithmetic, got: {err}"
            );
        }

        // Pure reads keep working: the shell evaluates the untouched
        // expression against the materialized value.
        let out = execute_command(&json!({
            "command": r#"L=$(printf 5); echo "$((L+1))""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap();
        assert_eq!(out, "6");
        let comparisons = execute_command(&json!({
            "command": r#"L=$(printf 5); printf '%s,%s,%s,%s' "$((L == 5))" "$((L != 5))" "$((L <= 5))" "$((L >= 5))""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap();
        assert_eq!(comparisons, "1,0,1,1");
    }

    #[test]
    fn execute_command_rejects_bare_arithmetic_reference_in_substitution_text() {
        // Separate substitution shells cannot observe this command's other
        // materialized assignments. Both substitutions sit in the first
        // segment, where materialization is allowed.
        for command in [
            r#"L=$(printf 5) B=$(echo $((L+1))); echo "$B""#,
            r#"L=$(printf 5) B=$(echo $((L++))); echo "$B""#,
            r#"L=$(printf 5) B=$(let L+=5; echo $((L))); echo "$B""#,
        ] {
            let err = execute_command(&json!({
                "command": command,
                "pty": false,
                "timeout": 5,
            }))
            .unwrap_err();
            assert!(
                err.contains("arithmetic"),
                "command {command:?} should mention arithmetic, got: {err}"
            );
        }
    }

    #[test]
    fn execute_command_follows_wrapped_and_quoted_assigning_command_words() {
        for command in [
            r#"L=$(printf 1); command export L=2; echo "$L""#,
            r#"L=$(printf 1); builtin export L=2; echo "$L""#,
            r#"L=$(printf 1); 'export' L=2; echo "$L""#,
            r#"L=$(printf 1); command unset L; echo "$L""#,
            r#"L=$(printf 1); eval 'L=2'; echo "$L""#,
        ] {
            let err = execute_command(&json!({
                "command": command,
                "pty": false,
                "timeout": 5,
            }))
            .unwrap_err();
            assert!(
                err.contains("overwrites") || err.contains("cannot read"),
                "command {command:?} should be rejected, got: {err}"
            );
        }

        // Non-writing forms keep working: a wrapper around a read, and an `env`
        // prefix, which does not change the shell's own variable.
        let output = execute_command(&json!({
            "command": r#"L=$(printf 1); command echo "$L"; env L=2 echo "$L""#,
            "pty": false,
            "timeout": 5,
        }))
        .unwrap();
        assert_eq!(output, "1\n1");
    }

    #[test]
    fn execute_command_materializes_substitution_with_force_overwrite_redirection() {
        let source = temporary_file_path("force_overwrite_source");
        let destination = temporary_file_path("force_overwrite_destination");
        let contents = "redirected file data";
        fs::write(&source, contents).expect("write test source");

        for backtick in [false, true] {
            let word = if backtick {
                format!("\"`cat {}`\"", source.display())
            } else {
                format!("\"$(cat {})\"", source.display())
            };
            for target in ["/dev/null".to_string(), destination.display().to_string()] {
                fs::write(&destination, "stale output").expect("write existing destination");
                let command = format!("printf '%s' {word} >|{target}");
                let materialized = materialize_safe_shell_substitutions(&command, None);
                let result = execute_command(&json!({
                    "command": command,
                    "pty": false,
                    "timeout": 5,
                }));
                let output = fs::read_to_string(&destination).expect("read destination");

                assert_eq!(
                    materialized.unwrap(),
                    format!("printf '%s' '{contents}' >|{target}"),
                );
                assert_eq!(
                    result.unwrap(),
                    "(command succeeded with exit code 0 and produced no output)",
                );
                assert_eq!(
                    output,
                    if target == "/dev/null" {
                        "stale output"
                    } else {
                        contents
                    },
                );
                assert_eq!(fs::read_to_string(&source).unwrap(), contents);
            }
        }

        let _ = fs::remove_file(&source);
        let _ = fs::remove_file(&destination);
    }

    #[test]
    fn execute_command_rejects_chained_substitution_before_short_circuited_side_effects() {
        for (prefix, substitution) in [
            ("false &&", "dollar"),
            ("true ||", "dollar"),
            ("false &&", "backtick"),
        ] {
            let probe = temporary_file_path("chained_substitution_probe");
            let word = if substitution == "backtick" {
                format!("\"`touch {}`\"", probe.display())
            } else {
                format!("\"$(touch {})\"", probe.display())
            };
            let result = execute_command(&json!({
                "command": format!("{prefix} echo {word}"),
                "pty": false,
                "timeout": 5,
            }));
            let created = probe.exists();
            let _ = fs::remove_file(&probe);

            assert!(!created, "short-circuited substitution created {probe:?}");
            let err = result.unwrap_err();
            assert!(err.contains("chained command"), "err: {err}");
        }
    }

    #[test]
    fn execute_command_rejects_chained_substitution_before_dependent_file_read() {
        for connector in ["&&", "||", ";", "|", "&"] {
            let path = temporary_file_path("chained_substitution_source");
            fs::write(&path, "stale").expect("write test source");
            let result = execute_command(&json!({
                "command": format!(
                    r#"echo fresh > {} {connector} cat "$(cat {})""#,
                    path.display(), path.display(),
                ),
                "pty": false,
                "timeout": 5,
            }));
            let contents = fs::read_to_string(&path).unwrap();
            let _ = fs::remove_file(&path);

            assert_eq!(contents, "stale", "blocked chain wrote its first segment");
            let err = result.unwrap_err();
            assert!(err.contains("chained command"), "err: {err}");
        }

        // A nonexistent source must be rejected as unsupported control flow,
        // without attempting the later read before the first segment writes it.
        let path = temporary_file_path("chained_substitution_new_source");
        let result = execute_command(&json!({
            "command": format!(
                r#"echo fresh > {} && echo "$(cat {})""#,
                path.display(), path.display(),
            ),
            "pty": false,
            "timeout": 5,
        }));
        let created = path.exists();
        let _ = fs::remove_file(&path);
        assert!(!created, "blocked chain created its source");
        let err = result.unwrap_err();
        assert!(err.contains("chained command"), "err: {err}");
    }

    #[test]
    fn execute_command_keeps_non_chained_substitution_control_characters() {
        let result = execute_command(&json!({
            "command": r#"printf '%s%s' "$(printf alpha; printf beta)" ';|&'"#,
            "pty": false,
            "timeout": 5,
        }));
        assert_eq!(result.unwrap(), "alphabeta;|&");
    }

    #[test]
    fn file_read_substitution_content_still_passes_command_audit() {
        let path = temporary_file_path("unsafe_file_read_substitution");
        fs::write(&path, "rm").expect("write test file");

        let result = execute_command(&json!({
            "command": format!(r#""$(cat {})" -rf /tmp/rust_tools_audit_test"#, path.display()),
            "pty": false,
            "timeout": 5,
        }));
        let _ = fs::remove_file(&path);

        let err = result.unwrap_err();
        assert!(err.contains("rm"), "err: {err}");
    }

    // ---- truncate_chars ----

    #[test]
    fn truncate_passthrough_when_within_limit() {
        let s = "short output";
        assert_eq!(truncate_chars(s, MAX_COMMAND_OUTPUT_CHARS), s);
    }

    #[test]
    fn truncate_emits_actionable_metadata_when_over_limit() {
        // 1000 short lines total far more than the small cap, triggering
        // truncation.
        let content: String = (0..1000).map(|i| format!("line{i}\n")).collect();
        let out = truncate_chars(&content, 100);
        // Not a bare "... (truncated)" anymore: the output carries
        // totals/shown counts and a paging hint.
        assert!(out.contains("truncated: omitted middle"), "out: {out}");
        assert!(out.contains("first 75 and last 25"), "out: {out}");
        assert!(out.contains("of 1000 lines"), "out: {out}");
        assert!(out.ends_with("line999\n"), "must preserve tail: {out}");
        assert!(
            out.contains("expected matches may be there, not absent"),
            "must warn that missing matches may be omitted, not absent"
        );
        assert!(
            out.contains("Do not re-run near-identical variants"),
            "must steer the model away from blind retries"
        );
        assert!(
            out.contains("`read_file` with offset/limit"),
            "must steer file paging toward read_file over sed: {out}"
        );
    }

    #[test]
    fn timeout_result_keeps_partial_output_and_clear_reason() {
        let out = format_command_result(
            CommandRunResult {
                status: None,
                stdout: b"progress before timeout\n".to_vec(),
                stderr: b"last diagnostic\n".to_vec(),
                timed_out: true,
                cancelled: false,
                stalled: false,
            },
            30,
        );
        assert!(out.contains("timed out after 30s"), "out: {out}");
        assert!(out.contains("Partial output captured"), "out: {out}");
        assert!(out.contains("progress before timeout"), "out: {out}");
        assert!(out.contains("last diagnostic"), "out: {out}");
    }

    #[test]
    fn stalled_result_explains_interactive_wait_and_keeps_partial_output() {
        let out = format_command_result(
            CommandRunResult {
                status: None,
                stdout: b"scan this QR: QR-CONTENT\n".to_vec(),
                stderr: Vec::new(),
                timed_out: false,
                cancelled: false,
                stalled: true,
            },
            60,
        );
        assert!(out.contains("waiting for interactive input"), "out: {out}");
        assert!(out.contains("QR-CONTENT"), "out: {out}");
        assert!(out.contains("`| tail`"), "out: {out}");
        assert!(
            !out.contains("timed out"),
            "must not read as a timeout: {out}"
        );
    }

    #[test]
    fn stalled_result_without_output_stays_informative() {
        let out = format_command_result(
            CommandRunResult {
                status: None,
                stdout: Vec::new(),
                stderr: Vec::new(),
                timed_out: false,
                cancelled: false,
                stalled: true,
            },
            60,
        );
        assert!(out.contains("no output was captured"), "out: {out}");
        assert!(out.contains("waiting for interactive input"), "out: {out}");
    }

    // ---- resolve_command_timeout ----

    #[test]
    fn timeout_uses_default_when_unset() {
        assert_eq!(resolve_command_timeout(None, 60, 300), 60);
    }

    #[test]
    fn timeout_clamps_to_max_and_floor() {
        assert_eq!(resolve_command_timeout(Some(10_000), 60, 300), 300);
        assert_eq!(resolve_command_timeout(Some(0), 60, 300), 1);
        assert_eq!(resolve_command_timeout(Some(120), 60, 300), 120);
    }

    #[test]
    fn execute_command_forwards_one_shot_stdin() {
        let output = execute_command(&json!({
            "command": "cat",
            "pty": false,
            "input": "service-stdin\n",
        }))
        .expect("cat with stdin should run");
        assert!(output.contains("service-stdin"), "out: {output}");
    }

    #[test]
    fn execute_command_accepts_cmd_alias() {
        let output = execute_command(&json!({
            "cmd": "printf alias-ok",
            "pty": false,
        }))
        .expect("cmd alias should run");
        assert!(output.contains("alias-ok"), "out: {output}");
    }

    #[test]
    fn execute_command_preserves_cmd_alias_whitespace() {
        for key in ["command", "cmd"] {
            let args = json!({
                key: "  printf ws  ",
                "pty": false,
            });
            // The service passes the original text through materialization to audit
            // and execution; downstream runner normalization keeps its existing behavior.
            let raw_command = super::super::execute_command_arg(&args).unwrap();
            assert_eq!(raw_command, "  printf ws  ");
            assert_eq!(
                materialize_safe_shell_substitutions(raw_command, None).unwrap(),
                "  printf ws  "
            );
            assert_eq!(execute_command(&args).unwrap(), "ws");
        }
    }

    #[test]
    fn execute_command_prefers_canonical_command_over_cmd_alias() {
        let output = execute_command(&json!({
            "command": "printf canonical",
            "cmd": "printf alias",
            "pty": false,
        }))
        .expect("canonical command should run");
        assert!(output.contains("canonical"), "out: {output}");
        assert!(!output.contains("alias"), "out: {output}");
    }

    #[test]
    fn execute_command_rejects_oversized_stdin() {
        let input = "x".repeat(MAX_COMMAND_INPUT_BYTES + 1);
        let err = execute_command(&json!({
            "command": "cat",
            "pty": false,
            "input": input,
        }))
        .expect_err("oversized input must be rejected");
        assert!(err.contains("input exceeds"), "err: {err}");
    }

    #[test]
    fn execute_command_rejects_legacy_session_control_fields() {
        for args in [
            json!({"command": "printf ok", "pty": false, "session_id": "cmd-1"}),
            json!({"command": "printf ok", "pty": false, "write_stdin": "x\n"}),
            json!({"command": "printf ok", "pty": false, "close_session": true}),
        ] {
            let err =
                execute_command(&args).expect_err("session control belongs to command_session");
            assert!(err.contains("command_session"), "err: {err}");
        }
    }

    #[test]
    fn execute_command_ignores_empty_optional_placeholders() {
        // Bridges sometimes materialize every optional schema field. Empty
        // placeholders must stay equivalent to omission rather than selecting
        // session control that no longer lives in this tool.
        let output = execute_command(&json!({
            "command": "printf ok",
            "pty": false,
            "session_id": "",
            "write_stdin": "",
            "input": "",
            "close_session": false,
            "yield_time_ms": null,
        }))
        .expect("empty optional placeholders should be ignored");
        assert!(output.contains("ok"), "out: {output}");
    }
}
