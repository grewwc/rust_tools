use super::lexer::{
    effective_command_tokens, matches_heredoc_terminator, parse_heredoc_at,
    validate_and_skip_heredoc_bodies, HereDocSpec,
};
use super::validate_execute_command;

// ---------------------------------------------------------------------------
// Shell injection-surface checks
// ---------------------------------------------------------------------------

/// Whether `(` immediately follows an escaped `$` (`\$(`). Once `\$` escapes the
/// `$` as a literal, `$(...)` no longer forms command substitution; the leftover
/// `(` only triggers a syntax error in bash (no subshell executes), so it is not
/// treated as an injection surface. Detection: the char before `(` is `$`, and
/// the run of backslashes directly before that `$` is odd (odd ⇒ `$` is escaped).
fn paren_follows_escaped_dollar(bytes: &[u8], i: usize) -> bool {
    // bytes[i] == b'('; the previous char must be `$`.
    if i < 2 || bytes[i - 1] != b'$' {
        return false;
    }
    let mut k = i - 2;
    let mut backslashes = 0u32;
    loop {
        match bytes.get(k) {
            Some(&b'\\') => {
                backslashes += 1;
                if k == 0 {
                    break;
                }
                k -= 1;
            }
            _ => break,
        }
    }
    backslashes % 2 == 1
}

/// Find the closing bracket pairing the left bracket at `open_idx` in the shell
/// structure. Brackets inside quotes or after backslash escapes are treated as
/// literals.
fn find_matching_shell_paren(command: &str, open_idx: usize) -> Option<usize> {
    let bytes = command.as_bytes();
    if bytes.get(open_idx) != Some(&b'(') {
        return None;
    }

    let mut depth = 1_u32;
    let mut i = open_idx + 1;
    let mut in_single = false;
    let mut in_double = false;
    let mut in_backtick = false;
    let mut escaped = false;
    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if b == b'\\' && !in_single {
            escaped = true;
            i += 1;
            continue;
        }
        if in_single {
            if b == b'\'' {
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if b == b'"' {
                in_double = false;
            }
            i += 1;
            continue;
        }
        if in_backtick {
            if b == b'`' {
                in_backtick = false;
            }
            i += 1;
            continue;
        }
        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'`' => in_backtick = true,
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct ShellWordSpan {
    start: usize,
    end: usize,
}

/// Parse the outer shell word used only for restricted file-read substitution.
/// Deliberately does not reuse `tokenize_shell_words`: that would erase quote
/// provenance, making it impossible to prove the `$()` really sits inside one
/// complete double-quoted word. Only simple command lines without control
/// operators, escapes, or extra active expansions are accepted.
fn restricted_outer_shell_words(command: &str) -> Option<(Vec<ShellWordSpan>, Vec<usize>)> {
    if command.bytes().any(|b| matches!(b, b'\n' | b'\r')) {
        return None;
    }

    let bytes = command.as_bytes();
    let mut words = Vec::new();
    let mut active_dollars = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i == bytes.len() {
            break;
        }

        let start = i;
        let mut in_single = false;
        let mut in_double = false;
        while i < bytes.len() {
            let b = bytes[i];
            if in_single {
                if b == b'\'' {
                    in_single = false;
                }
                i += 1;
                continue;
            }
            if in_double {
                match b {
                    b'"' => in_double = false,
                    // The restricted grammar rejects backslashes and backticks,
                    // so they cannot alter quoting or expansion semantics.
                    b'\\' | b'`' => return None,
                    b'$' => {
                        if bytes.get(i + 1) != Some(&b'(') {
                            return None;
                        }
                        active_dollars.push(i);
                    }
                    _ => {}
                }
                i += 1;
                continue;
            }
            if b.is_ascii_whitespace() {
                break;
            }
            match b {
                b'\'' => in_single = true,
                b'"' => in_double = true,
                // The outer word must not contain chars that change command
                // structure or trigger expansion or globbing.
                b'\\' | b'`' | b'$' | b'(' | b')' | b'{' | b'}' | b';' | b'&' | b'|' | b'<'
                | b'>' | b'*' | b'?' | b'[' => return None,
                _ => {}
            }
            i += 1;
        }
        if in_single || in_double {
            return None;
        }
        words.push(ShellWordSpan { start, end: i });
    }
    Some((words, active_dollars))
}

/// Allow only ASCII absolute paths and forbid empty, `.`, and `..` components;
/// this keeps `cat`'s argument free of any shell expansion, options, or extra
/// command fragments.
fn is_literal_absolute_path(path: &str) -> bool {
    if !path.starts_with('/')
        || !path
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'_' | b'-'))
    {
        return false;
    }

    let mut components = path.split('/');
    components.next() == Some("")
        && components
            .all(|component| !component.is_empty() && component != "." && component != "..")
}

/// Kinds of harmless shell substitutions: prefer literal file reads (no shell
/// runs), otherwise run a harmless command and capture its output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SafeShellSubstitutionKind {
    FileRead { path: String },
    Command { inner: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SafeShellSubstitution {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) kind: SafeShellSubstitutionKind,
}

/// Recognize and classify harmless `"$(...)"` substitutions in a simple outer
/// command.
///
/// - The outer command must use the restricted grammar: no control operators, no
///   escapes, no extra active expansions, and every `$()` must be a complete
///   double-quoted word `"$(inner)"`.
/// - If `inner` is `cat /absolute/path/literal`, take the file-read path (no
///   shell execution).
/// - Otherwise, if `inner` itself passes `validate_execute_command` (i.e. is
///   judged harmless by the existing blacklist), treat it as an executable
///   harmless command substitution.
/// - If any `$()` fails both categories above, the whole thing is unsafe; return
///   empty so the upper injection validation can block it.
pub(crate) fn safe_shell_substitutions(command: &str) -> Vec<SafeShellSubstitution> {
    let Some((words, active_dollars)) = restricted_outer_shell_words(command) else {
        return Vec::new();
    };
    if active_dollars.is_empty() {
        return Vec::new();
    }
    let mut substitutions = Vec::with_capacity(active_dollars.len());
    for dollar_idx in active_dollars {
        let Some(word) = words
            .iter()
            .find(|word| word.start <= dollar_idx && dollar_idx < word.end)
            .copied()
        else {
            return Vec::new();
        };
        let raw_word = &command[word.start..word.end];
        if dollar_idx != word.start + 1
            || raw_word.len() < 5
            || !raw_word.starts_with("\"$(")
            || !raw_word.ends_with(")\"")
        {
            return Vec::new();
        }
        let inner = &command[word.start + 3..word.end - 2];
        let inner_trim = inner.trim();
        if inner_trim.is_empty() || inner_trim.contains('\0') {
            return Vec::new();
        }
        if let Some(path) = inner_trim.strip_prefix("cat ") {
            if is_literal_absolute_path(path) && inner_trim == format!("cat {path}") {
                substitutions.push(SafeShellSubstitution {
                    start: word.start,
                    end: word.end,
                    kind: SafeShellSubstitutionKind::FileRead {
                        path: path.to_string(),
                    },
                });
                continue;
            }
        }
        if validate_execute_command(inner_trim).is_ok() {
            substitutions.push(SafeShellSubstitution {
                start: word.start,
                end: word.end,
                kind: SafeShellSubstitutionKind::Command {
                    inner: inner_trim.to_string(),
                },
            });
        } else {
            return Vec::new();
        }
    }
    substitutions
}

/// Check whether the command string contains an unsafe shell injection surface.
///
/// This function is a **shell-specific** safety check and should only be called
/// for commands executed through a shell (i.e. the `execute_command` tool). For
/// non-shell tools (pure string operations like `write_file`, `apply_patch`),
/// do not apply this check — they write the filesystem or do text replacement
/// directly and never feed arguments to a shell, so `<<` / `$()` are just plain
/// text.
///
    /// Command substitution `$(...)` is allowed after recursively validating the
    /// inner command (same trust model as process substitution), so quoted data
    /// usages like `echo "$(date)"` and `for i in $(seq 1 40)` loops work while
    /// `$(rm ...)` stays blocked. The inner check alone cannot see where the
    /// output lands, so `validate_substitution_positions` additionally restricts
    /// every substitution to a materializable whole-word `"$(...)"` or a
    /// literal-`seq` loop list. Generating the *program name* from a
    /// substitution remains banned — its output is word-split into command
    /// words, so `$(echo r)m -rf /` could run `rm` at runtime; that guard lives
    /// in `validate_single_segment`. Backticks stay banned (legacy form).
    /// Process substitution `<(...)` / `>(...)` is allowed after recursively
    /// validating the inner command, avoiding false blocks on common usages
    /// like diff/sort.
pub(crate) fn validate_no_injection_surface(command: &str) -> Result<(), String> {
    let bytes = command.as_bytes();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut pending_heredocs: Vec<HereDocSpec> = Vec::new();
    let mut arith_depth: u32 = 0;
    let mut literal_paren_depth: u32 = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        // Everything inside single quotes is a literal; the shell does not parse
        // $() or backticks.
        if in_single {
            if b == b'\'' {
                in_single = false;
            }
            i += 1;
            continue;
        }
        // Inside double quotes `<(` / `>(` are plain text, but `$()` / `` `...` ``
        // can still take effect, so blocking continues below.
        if in_double {
            match b {
                b'\\' => {
                    escaped = true;
                    i += 1;
                    continue;
                }
                b'`' => {
                    return Err(
                        "backtick command substitution is not allowed; pass a literal command instead"
                            .to_string(),
                    );
                }
                b'"' => {
                    in_double = false;
                    i += 1;
                    continue;
                }
                _ => {}
            }
        }
        if !in_double && b == b'\\' {
            escaped = true;
            i += 1;
            continue;
        }
        if b == b'\'' {
            in_single = true;
            i += 1;
            continue;
        }
        if b == b'"' {
            in_double = true;
            i += 1;
            continue;
        }
        if b == b'`' {
            return Err(
                "backtick command substitution is not allowed; pass a literal command instead"
                    .to_string(),
            );
        }
        if !in_double && b == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'<' {
            if let Some((end, spec)) = parse_heredoc_at(command, i) {
                pending_heredocs.push(spec);
                i = end;
                continue;
            }
        }
        // Command substitution `$(`
        if b == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            // Arithmetic expansion `$(( ... ))` executes no commands and is
            // harmless (typical: `echo $((RANDOM % 20))`); it must not be falsely
            // killed by the command-substitution rule. Push the arithmetic depth
            // and keep scanning inward — genuinely nested command substitutions
            // (like the `$(` inside `$(( $(whoami) ))`) still get caught in later
            // iterations, while the trailing `))` and inner grouping parens are
            // correctly allowed by the arith_depth branch below.
            if i + 2 < bytes.len() && bytes[i + 2] == b'(' {
                arith_depth += 1;
                i += 3;
                continue;
            }
            // Nested `$(` inside arithmetic expansion stays blocked.
            if arith_depth > 0 {
                return Err(
                    "command substitution `$(...)` inside arithmetic expansion is not allowed"
                        .to_string(),
                );
            }
            // Same trust model as process substitution below: the inner command
            // itself must pass the whole safety check, so `echo "$(date)"` and
            // `for i in $(seq 1 40)` work while `$(rm ...)` stays blocked. Where
            // the output lands is checked separately by
            // `validate_substitution_positions`. Generating the *program name*
            // from a substitution is still banned (its output is word-split
            // into command words) — that guard lives in `validate_single_segment`.
            let close = find_matching_shell_paren(command, i + 1).ok_or_else(|| {
                "unterminated command substitution `$(...)`".to_string()
            })?;
            let inner = command[i + 2..close].trim();
            validate_execute_command(inner)
                .map_err(|reason| format!("unsafe command substitution `$(...)`: {reason}"))?;
            i = close + 1;
            continue;
        }
        // Process substitution `<(...)` / `>(...)` has shell semantics only
        // outside quotes. Recursively validate the full inner command instead of
        // banning indiscriminately; safe commands stay usable while `<(rm ...)`
        // etc. are still caught by the existing rules.
        if !in_double && (b == b'<' || b == b'>') && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            let close = find_matching_shell_paren(command, i + 1).ok_or_else(|| {
                "unterminated process substitution `<(...)` / `>(...)`".to_string()
            })?;
            let inner = command[i + 2..close].trim();
            validate_execute_command(inner)
                .map_err(|reason| format!("unsafe process substitution: {reason}"))?;
            i = close + 1;
            continue;
        }
        // Unquoted `(` / `)` / `{` / `}` open a subshell or command grouping
        // (e.g. `(rm -rf /tmp)`, `{ rm -rf /tmp; }`), bypassing segment-blacklist
        // validation.
        // `$(` / `$((` / `<(` / `>(` are handled separately above; block bare
        // `(` / `)` / `{` / `}` here.
        // But the `(` / `)` inside arithmetic expansion `$(( ... ))` are just
        // grouping parens and `))` closes the expansion — neither forms a
        // subshell; in `\$(` the `$` is escaped as a literal and the leftover `(`
        // is only a bash syntax error, which likewise executes no subshell — allow
        // both cases.
        if !in_double && matches!(b, b'(' | b')' | b'{' | b'}') {
            if arith_depth > 0 && matches!(b, b'(' | b')') {
                if b == b')' && i + 1 < bytes.len() && bytes[i + 1] == b')' {
                    arith_depth -= 1;
                    i += 2;
                    continue;
                }
                i += 1;
                continue;
            }
            if b == b'(' && paren_follows_escaped_dollar(bytes, i) {
                literal_paren_depth = 1;
                i += 1;
                continue;
            }
            if literal_paren_depth > 0 && matches!(b, b'(' | b')') {
                if b == b'(' {
                    literal_paren_depth += 1;
                } else {
                    literal_paren_depth -= 1;
                }
                i += 1;
                continue;
            }
            return Err(
                "unquoted shell metacharacters `(` `)` `{` `}` start a subshell or command group and bypass command validation; run the command directly instead".to_string(),
            );
        }
        if !in_double && b == b'\n' && !pending_heredocs.is_empty() {
            i += 1;
            i = validate_and_skip_heredoc_bodies(command, i, &pending_heredocs)?;
            pending_heredocs.clear();
            continue;
        }
        i += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Segment-level validation entry
// ---------------------------------------------------------------------------

pub(crate) fn normalize_path(path: &std::path::Path) -> std::path::PathBuf {
    let mut normalized = std::path::PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            std::path::Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

/// Expand a leading `~` / `$HOME` in the argument. The home directory itself and
/// its subpaths are normal development access; only escaping outward from home
/// via `..` is rejected. `~other` is outside the shell's current-user home
/// semantics and is left to the shell itself.
pub(crate) fn expand_tilde_and_home(arg: &str) -> Result<String, String> {
    let home = if arg == "~" || arg.starts_with("~/") {
        std::env::var("HOME")
            .map_err(|_| "cannot expand ~: HOME environment variable not set".to_string())?
    } else if arg == "$HOME" || arg.starts_with("$HOME/") {
        std::env::var("HOME")
            .map_err(|_| "cannot expand $HOME: HOME environment variable not set".to_string())?
    } else {
        return Ok(arg.to_string());
    };
    let rest = arg
        .strip_prefix("~/")
        .or_else(|| arg.strip_prefix("$HOME/"));
    let expanded = rest.map_or_else(|| home.clone(), |rest| format!("{home}/{rest}"));
    let home = normalize_path(std::path::Path::new(&home));
    let resolved = normalize_path(std::path::Path::new(&expanded));
    if resolved.starts_with(&home) {
        Ok(expanded)
    } else {
        Err(format!(
            "command references path {arg} (resolves to {}) which escapes the home directory",
            resolved.display()
        ))
    }
}

/// Validate a single command segment against the program/argument blacklist.
/// True when `b` terminates a shell word for the purpose of locating which word
/// contains a command substitution (control operators and unquoted whitespace;
/// quote, `$`, `=`, and pathname-expansion characters stay part of the word).
fn is_substitution_word_boundary(b: u8) -> bool {
    b.is_ascii_whitespace() || matches!(b, b';' | b'|' | b'&' | b'<' | b'>')
}

/// For `for VAR in ...` prefixes, return the byte index just past the `in`
/// keyword, so list elements (where `$(...)` is allowed) can be distinguished
/// from every other argument position.
fn for_in_list_prefix_end(command: &str) -> Option<usize> {
    let bytes = command.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if !command[i..].starts_with("for") {
        return None;
    }
    i += 3;
    if !bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
        return None;
    }
    // Variable name (one word).
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    // The `in` keyword.
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if !command[i..].starts_with("in") {
        return None;
    }
    i += 2;
    if !bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
        return None;
    }
    Some(i)
}

/// Whether the inner command of a `$(...)` is exactly `seq` with one to three
/// literal integer bounds (`seq 40`, `seq 1 40`, `seq 1 2 40`). Such output is
/// provably numeric, so storing it in a loop variable cannot smuggle a program
/// name or an audited argument value past the checks below.
fn is_literal_seq_substitution(command: &str, dollar_at: usize, close: usize) -> bool {
    let inner = command[dollar_at + 2..close].trim();
    let mut words = inner.split_whitespace();
    if words.next() != Some("seq") {
        return false;
    }
    let mut count = 0u32;
    for word in words {
        let digits = word
            .strip_prefix('+')
            .or_else(|| word.strip_prefix('-'))
            .unwrap_or(word);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        count += 1;
    }
    (1..=3).contains(&count)
}

/// Reject a single active `$(...)` whose runtime output could reach a position
/// the audit semantically checks (the program name, a `git` subcommand, an
/// interpreter `-c` flag, a path or glob subject to scope confinement). The
/// output is unverifiable runtime data, so `git "pu$(printf sh)"` would execute
/// `git push` at runtime and bypass the literal-token checks below. Worse, any
/// substitution stored in a shell variable escapes per-segment analysis
/// entirely: `for cmd in $(printf rm); do $cmd -rf target; done` and
/// `cmd=$(printf rm); $cmd -rf target` split into individually innocent
/// segments while the shell executes the banned program. Shell variables are
/// not tracked, so only two positions pass: a `for ... in` list word holding
/// exactly `$(seq FIRST LAST)` with literal integer bounds (provably numeric
/// output), and a complete double-quoted word `"$(...)"` in an argument of a
/// pure-data program (`echo`/`printf`/`seq`/`date`, which command.rs
/// materializes before execution). Every other form is rejected.
pub(crate) fn check_substitution_at(
    command: &str,
    dollar_at: usize,
    program: &str,
    eff_program: &str,
) -> Result<(), String> {
    let bytes = command.as_bytes();
    let mut word_start = dollar_at;
    while word_start > 0 && !is_substitution_word_boundary(bytes[word_start - 1]) {
        word_start -= 1;
    }
    // Assignment values (including `VAR=...` env prefixes) flow into shell
    // variables the audit cannot track: `cmd=$(printf rm); $cmd ...` would
    // execute `rm` without any segment seeing it. Banned in every form.
    if let Some((name, _)) = command[word_start..dollar_at].split_once('=') {
        let mut chars = name.chars();
        let is_name = chars
            .next()
            .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
            && chars.all(|c| c == '_' || c.is_ascii_alphanumeric());
        if is_name {
            return Err(
                "command substitution `$(...)` may not appear in an assignment value; its \
                 output flows into a shell variable, which the audit cannot track. Use a \
                 literal value or a `for i in $(seq FIRST LAST)` loop instead"
                    .to_string(),
            );
        }
    }
    // The substitution sits inside the very first word, so it can generate the
    // program name at runtime (`$(echo rm) -rf /`).
    let first_word_start = bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(0);
    if word_start == first_word_start {
        return Err(
            "command substitution `$(...)` cannot generate the program name; pass a literal \
             program instead"
                .to_string(),
        );
    }
    let Some(close) = find_matching_shell_paren(command, dollar_at + 1) else {
        return Err("unbalanced command substitution `$(...)`".to_string());
    };
    let mut word_end = close + 1;
    while word_end < bytes.len() && !is_substitution_word_boundary(bytes[word_end]) {
        word_end += 1;
    }
    // Exact-word shapes. Only a complete double-quoted word `"$(...)"` is
    // materialized by command.rs; any other form reaches the shell raw, where
    // its output undergoes word-splitting and glob expansion.
    let bare_exact = word_start == dollar_at && word_end == close + 1;
    let quoted_exact = dollar_at == word_start + 1
        && word_end == close + 2
        && bytes.get(word_start) == Some(&b'"')
        && bytes.get(close + 1) == Some(&b'"');
    // `for ... in` list elements feed the loop variable, which the audit
    // cannot track — only provably numeric `seq` output is safe there, as an
    // exact list word (so digit output cannot fuse with literal affixes).
    if eff_program == "for" {
        if let Some(prefix_end) = for_in_list_prefix_end(command) {
            if word_start >= prefix_end {
                if (bare_exact || quoted_exact)
                    && is_literal_seq_substitution(command, dollar_at, close)
                {
                    return Ok(());
                }
                return Err(
                    "command substitution `$(...)` in a `for ... in` list must be exactly \
                     `$(seq FIRST LAST)` with literal integer bounds; its output flows into \
                     the loop variable, which the audit cannot track"
                        .to_string(),
                );
            }
        }
    }
    // Pure-data programs whose arguments the audit never inspects — but only
    // for the materializable whole-word form. An unquoted or partial-word
    // substitution would execute its inner command raw (e.g. `echo $(git
    // commit ...)` runs the commit without the confirmation gate) or let the
    // output split into extra words.
    if quoted_exact && matches!(eff_program, "echo" | "printf" | "seq" | "date") {
        return Ok(());
    }
    Err(format!(
        "command substitution `$(...)` may not generate an argument for `{program}`: its \
         runtime output cannot be verified, so it could bypass the sandbox checks (git \
         subcommand blocks, interpreter `-c`, path/glob scope). Use a literal argument, a \
         complete double-quoted word (`\"$(...)\"`) with `echo`/`printf`, or a \
         `for i in $(seq FIRST LAST)` loop instead"
    ))
}

/// Fail-closed position guard for command substitutions in a segment: every
/// active `$(...)` must occupy an allowed position (see `check_substitution_at`).
/// `validate_no_injection_surface` only proves the inner command is safe; this
/// closes the remaining gap where the substitution's *output* lands in a
/// checked slot. Heredoc bodies are skipped so literal `$(` inside quoted
/// bodies is not mistaken for an active substitution, and herestrings (`<<<`)
/// are scanned (their content is subject to expansion too).
pub(crate) fn validate_substitution_positions(command: &str, program: &str) -> Result<(), String> {
    let eff_program = effective_command_tokens(command)
        .first()
        .and_then(|token| std::path::Path::new(token).file_name())
        .and_then(|name| name.to_str())
        .unwrap_or(program)
        .to_ascii_lowercase();

    let bytes = command.as_bytes();
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut pending_heredocs: Vec<HereDocSpec> = Vec::new();
    let mut active_body: Option<HereDocSpec> = None;

    while i < bytes.len() {
        if let Some(spec) = &active_body {
            let line_end = command[i..]
                .find('\n')
                .map(|offset| i + offset)
                .unwrap_or(bytes.len());
            let line = &command[i..line_end];
            if matches_heredoc_terminator(line, spec) {
                active_body = None;
                i = line_end;
            } else {
                i = if line_end < bytes.len() {
                    line_end + 1
                } else {
                    line_end
                };
            }
            continue;
        }

        let b = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if in_single {
            if b == b'\'' {
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            match b {
                b'\\' => {
                    escaped = true;
                }
                b'"' => {
                    in_double = false;
                }
                // `$(( ... ))` is arithmetic expansion, not a substitution.
                b'$'
                    if bytes.get(i + 1) == Some(&b'(') && bytes.get(i + 2) != Some(&b'(') =>
                {
                    check_substitution_at(command, i, program, &eff_program)?;
                }
                _ => {}
            }
            i += 1;
            continue;
        }
        match b {
            b'\\' => {
                escaped = true;
            }
            b'\'' => {
                in_single = true;
            }
            b'"' => {
                in_double = true;
            }
            b'$'
                if bytes.get(i + 1) == Some(&b'(') && bytes.get(i + 2) != Some(&b'(') =>
            {
                check_substitution_at(command, i, program, &eff_program)?;
            }
            // `<<<` herestrings are not heredocs; their content still undergoes
            // expansion, so it must be scanned rather than skipped.
            b'<'
                if bytes.get(i + 1) == Some(&b'<') && bytes.get(i + 2) != Some(&b'<') =>
            {
                if let Some((end, spec)) = parse_heredoc_at(command, i) {
                    pending_heredocs.push(spec);
                    i = end;
                    continue;
                }
            }
            b'\n' => {
                if let Some(spec) = pending_heredocs.pop() {
                    active_body = Some(spec);
                }
            }
            _ => {}
        }
        i += 1;
    }
    Ok(())
}
