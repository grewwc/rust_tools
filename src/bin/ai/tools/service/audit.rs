/// Audit module: command safety validation (injection-surface checks + segment
/// blacklists).
///
/// Separation of concerns:
/// - This module only "validates"; it never "executes".
/// - `execute_command` just calls the `validate_execute_command()` entry point.
/// - Easy to test and evolve the safety policy independently, decoupled from
///   execution logic.
use crate::ai::config_schema::AiConfig;

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

/// Read the user-configured list of denied programs.
fn config_blocked_commands() -> Vec<String> {
    let raw = crate::commonw::configw::get_all_config().get(AiConfig::SANDBOX_BLOCKED_COMMANDS, "");
    raw.split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Shell segmentation (split chained commands on `&&` / `||` / `;` / `|` / `\n`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShellJoin {
    Start,
    And,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShellSegment {
    pub(crate) command: String,
    pub(crate) join: ShellJoin,
}

fn ampersand_is_redirection_operator(bytes: &[u8], index: usize) -> bool {
    (index > 0 && matches!(bytes[index - 1], b'>' | b'<'))
        || (index + 1 < bytes.len() && bytes[index + 1] == b'>')
}

/// Split the whole command into independent segments using unquoted
/// `&&` / `||` / `;` / `|` / `\n` as separators. Separators inside single/double
/// quotes do not trigger a split; newlines inside a single-quoted heredoc body
/// are skipped too (heredoc body content is literal and must not be consumed by
/// the splitting logic).
pub(crate) fn split_unquoted_command_segments(command: &str) -> Vec<ShellSegment> {
    let bytes = command.as_bytes();
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut current_join = ShellJoin::Start;
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut pending_heredocs: Vec<HereDocSpec> = Vec::new();

    let push_current = |segments: &mut Vec<ShellSegment>, current: &mut String, join: ShellJoin| {
        let command = std::mem::take(current).trim().to_string();
        if !command.is_empty() {
            segments.push(ShellSegment { command, join });
        }
    };

    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            current.push(b as char);
            escaped = false;
            i += 1;
            continue;
        }
        if in_single {
            if b == b'\'' {
                in_single = false;
            }
            current.push(b as char);
            i += 1;
            continue;
        }
        if in_double {
            current.push(b as char);
            // Escape chars inside double quotes are only valid for a few
            // characters; coarsely skipping the next byte here is enough
            if b == b'\\' && i + 1 < bytes.len() {
                current.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if b == b'"' {
                in_double = false;
            }
            i += 1;
            continue;
        }
        match b {
            b'\'' => {
                in_single = true;
                current.push('\'');
                i += 1;
            }
            b'"' => {
                in_double = true;
                current.push('"');
                i += 1;
            }
            b'\\' if i + 1 < bytes.len() => {
                // Backslash escape outside quotes: keep both bytes
                current.push(b as char);
                current.push(bytes[i + 1] as char);
                i += 2;
            }
            b'<' if i + 1 < bytes.len() && bytes[i + 1] == b'<' => {
                if let Some((end, spec)) = parse_heredoc_at(command, i) {
                    current.push_str(&command[i..end]);
                    pending_heredocs.push(spec);
                    i = end;
                } else {
                    current.push('<');
                    i += 1;
                }
            }
            // Two-char operators `&&` / `||`
            b'&' if i + 1 < bytes.len() && bytes[i + 1] == b'&' => {
                push_current(&mut segments, &mut current, current_join);
                current_join = ShellJoin::And;
                i += 2;
            }
            b'|' if i + 1 < bytes.len() && bytes[i + 1] == b'|' => {
                push_current(&mut segments, &mut current, current_join);
                current_join = ShellJoin::Other;
                i += 2;
            }
            b'&' if ampersand_is_redirection_operator(bytes, i) => {
                current.push('&');
                i += 1;
            }
            // Single-char separators
            b';' | b'|' | b'&' => {
                push_current(&mut segments, &mut current, current_join);
                current_join = ShellJoin::Other;
                i += 1;
            }
            b'\n' => {
                push_current(&mut segments, &mut current, current_join);
                current_join = ShellJoin::Other;
                i += 1;
                if !pending_heredocs.is_empty() {
                    i = skip_heredoc_bodies(command, i, &pending_heredocs);
                    pending_heredocs.clear();
                }
            }
            _ => {
                current.push(b as char);
                i += 1;
            }
        }
    }
    let trailing_non_success_join = current.trim().is_empty() && current_join == ShellJoin::Other;
    push_current(&mut segments, &mut current, current_join);
    if trailing_non_success_join && !segments.is_empty() {
        segments.push(ShellSegment {
            command: String::new(),
            join: ShellJoin::Other,
        });
    }
    segments
}

pub(crate) fn split_unquoted_segments(command: &str) -> Vec<String> {
    split_unquoted_command_segments(command)
        .into_iter()
        .map(|segment| segment.command)
        .filter(|command| !command.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Heredoc parsing helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct HereDocSpec {
    delimiter: String,
    strip_tabs: bool,
    literal_body: bool,
}

fn parse_heredoc_at(command: &str, start: usize) -> Option<(usize, HereDocSpec)> {
    let bytes = command.as_bytes();
    if bytes.get(start) != Some(&b'<') || bytes.get(start + 1) != Some(&b'<') {
        return None;
    }

    let mut i = start + 2;
    let mut strip_tabs = false;
    if bytes.get(i) == Some(&b'-') {
        strip_tabs = true;
        i += 1;
    }
    while matches!(bytes.get(i), Some(b' ' | b'\t')) {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] == b'\n' {
        return None;
    }

    let mut delimiter = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut saw_any = false;
    let mut literal_body = false;

    while i < bytes.len() {
        let Some(ch) = command[i..].chars().next() else {
            break;
        };
        let next_i = i + ch.len_utf8();

        if escaped {
            delimiter.push(ch);
            saw_any = true;
            literal_body = true;
            escaped = false;
            i = next_i;
            continue;
        }
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                delimiter.push(ch);
            }
            saw_any = true;
            literal_body = true;
            i = next_i;
            continue;
        }
        if in_double {
            match ch {
                '"' => {
                    in_double = false;
                }
                '\\' => {
                    escaped = true;
                }
                _ => delimiter.push(ch),
            }
            saw_any = true;
            literal_body = true;
            i = next_i;
            continue;
        }

        if ch.is_whitespace() || matches!(ch, ';' | '|' | '&' | '<' | '>' | '\n') {
            break;
        }
        match ch {
            '\'' => {
                in_single = true;
                saw_any = true;
                literal_body = true;
            }
            '"' => {
                in_double = true;
                saw_any = true;
                literal_body = true;
            }
            '\\' => {
                escaped = true;
                saw_any = true;
                literal_body = true;
            }
            _ => {
                delimiter.push(ch);
                saw_any = true;
            }
        }
        i = next_i;
    }

    if !saw_any || delimiter.is_empty() {
        return None;
    }
    Some((
        i,
        HereDocSpec {
            delimiter,
            strip_tabs,
            literal_body,
        },
    ))
}

fn matches_heredoc_terminator(line: &str, spec: &HereDocSpec) -> bool {
    let candidate = if spec.strip_tabs {
        line.trim_start_matches('\t')
    } else {
        line
    };
    candidate == spec.delimiter
}

fn skip_heredoc_bodies(command: &str, mut start: usize, pending: &[HereDocSpec]) -> usize {
    for spec in pending {
        while start < command.len() {
            let line_end = command[start..]
                .find('\n')
                .map(|offset| start + offset)
                .unwrap_or(command.len());
            let line = &command[start..line_end];
            let next_start = if line_end < command.len() {
                line_end + 1
            } else {
                line_end
            };
            start = next_start;
            if matches_heredoc_terminator(line, spec) {
                break;
            }
        }
    }
    start
}

fn validate_unquoted_heredoc_line(line: &str) -> Result<(), String> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    let mut escaped = false;
    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if b == b'\\' {
            escaped = true;
            i += 1;
            continue;
        }
        if b == b'`' {
            return Err(
                "backtick command substitution is not allowed; pass a literal command instead"
                    .to_string(),
            );
        }
        if b == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            if i + 2 < bytes.len() && bytes[i + 2] == b'(' {
                i += 3;
                continue;
            }
            return Err(
                "command substitution `$(...)` is not allowed; pass a literal command instead"
                    .to_string(),
            );
        }
        i += 1;
    }
    Ok(())
}

fn validate_and_skip_heredoc_bodies(
    command: &str,
    mut start: usize,
    pending: &[HereDocSpec],
) -> Result<usize, String> {
    for spec in pending {
        while start < command.len() {
            let line_end = command[start..]
                .find('\n')
                .map(|offset| start + offset)
                .unwrap_or(command.len());
            let line = &command[start..line_end];
            let next_start = if line_end < command.len() {
                line_end + 1
            } else {
                line_end
            };
            start = next_start;
            if matches_heredoc_terminator(line, spec) {
                break;
            }
            if !spec.literal_body {
                validate_unquoted_heredoc_line(line)?;
            }
        }
    }
    Ok(start)
}

// ---------------------------------------------------------------------------
// Shell lexical analysis (used for per-segment validation)
// ---------------------------------------------------------------------------

pub(crate) fn tokenize_shell_words(command: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut token_started = false;

    for ch in command.chars() {
        if escaped {
            current.push(ch);
            token_started = true;
            escaped = false;
            continue;
        }
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                current.push(ch);
            }
            token_started = true;
            continue;
        }
        if in_double {
            match ch {
                '"' => in_double = false,
                '\\' => escaped = true,
                _ => current.push(ch),
            }
            token_started = true;
            continue;
        }

        if ch.is_whitespace() {
            if token_started {
                tokens.push(std::mem::take(&mut current));
                token_started = false;
            }
            continue;
        }

        match ch {
            '\'' => {
                in_single = true;
                token_started = true;
            }
            '"' => {
                in_double = true;
                token_started = true;
            }
            '\\' => {
                escaped = true;
                token_started = true;
            }
            _ => {
                current.push(ch);
                token_started = true;
            }
        }
    }

    if escaped {
        current.push('\\');
    }
    if token_started {
        tokens.push(current);
    }
    tokens
}

// ---------------------------------------------------------------------------
// Command index resolution (skip options and locate the program that will
// actually run)
// ---------------------------------------------------------------------------

fn is_env_assignment_word(token: &str) -> bool {
    let Some((name, _)) = token.split_once('=') else {
        return false;
    };
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn command_word_index(tokens: &[String], shell_context: bool) -> Option<usize> {
    if !shell_context {
        return (!tokens.is_empty()).then_some(0);
    }

    let mut i = 0usize;
    while i < tokens.len() && is_env_assignment_word(&tokens[i]) {
        i += 1;
    }
    (i < tokens.len()).then_some(i)
}

fn xargs_command_index(tokens: &[String]) -> Option<usize> {
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            return (i + 1 < tokens.len()).then_some(i + 1);
        }
        if !tok.starts_with('-') || tok == "-" {
            return Some(i);
        }
        let attached_value = tok.starts_with("--arg-file=")
            || tok.starts_with("--delimiter=")
            || tok.starts_with("--eof=")
            || tok.starts_with("--replace=")
            || tok.starts_with("--max-lines=")
            || tok.starts_with("--max-args=")
            || tok.starts_with("--max-procs=")
            || tok.starts_with("--max-chars=")
            || matches!(
                tok.chars().nth(1),
                Some('a' | 'd' | 'E' | 'e' | 'I' | 'i' | 'L' | 'l' | 'n' | 'P' | 's')
            ) && tok.len() > 2
                && !tok.starts_with("--");
        if attached_value {
            i += 1;
            continue;
        }
        let takes_value = matches!(
            tok,
            "-a" | "--arg-file"
                | "-d"
                | "--delimiter"
                | "-E"
                | "-e"
                | "--eof"
                | "-I"
                | "-i"
                | "--replace"
                | "-L"
                | "-l"
                | "--max-lines"
                | "-n"
                | "--max-args"
                | "-P"
                | "--max-procs"
                | "-s"
                | "--max-chars"
        );
        i += if takes_value { 2 } else { 1 };
    }
    None
}

fn env_command_index(tokens: &[String], raw_tokens: &[String]) -> Option<usize> {
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            return (i + 1 < tokens.len()).then_some(i + 1);
        }
        if matches!(
            tok,
            "-u" | "--unset" | "-c" | "--chdir" | "-s" | "--split-string"
        ) || tok == "-a"
        {
            i += 2;
            continue;
        }
        if tok.starts_with("--unset=")
            || tok.starts_with("--chdir=")
            || tok.starts_with("--split-string=")
            || tok.starts_with("--argv0=")
        {
            i += 1;
            continue;
        }
        if tok.starts_with('-') && tok != "-" {
            i += 1;
            continue;
        }
        if is_env_assignment_word(&raw_tokens[i]) {
            i += 1;
            continue;
        }
        return Some(i);
    }
    None
}

fn command_builtin_index(tokens: &[String]) -> Option<usize> {
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            return (i + 1 < tokens.len()).then_some(i + 1);
        }
        if !tok.starts_with('-') || tok == "-" {
            return Some(i);
        }
        if matches!(tok, "-p") {
            i += 1;
            continue;
        }
        if matches!(tok, "-v" | "-V") {
            return None;
        }
        i += 1;
    }
    None
}

fn exec_builtin_index(tokens: &[String]) -> Option<usize> {
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            return (i + 1 < tokens.len()).then_some(i + 1);
        }
        if !tok.starts_with('-') || tok == "-" {
            return Some(i);
        }
        if matches!(tok, "-a" | "-c" | "-l") {
            i += if tok == "-a" { 2 } else { 1 };
            continue;
        }
        i += 1;
    }
    None
}

fn first_non_option_index(
    tokens: &[String],
    start: usize,
    options_with_value: &[&str],
) -> Option<usize> {
    let mut i = start;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            return (i + 1 < tokens.len()).then_some(i + 1);
        }
        if !tok.starts_with('-') || tok == "-" {
            return Some(i);
        }
        let takes_value = options_with_value.contains(&tok);
        i += if takes_value { 2 } else { 1 };
    }
    None
}

fn nice_command_index(tokens: &[String]) -> Option<usize> {
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            return (i + 1 < tokens.len()).then_some(i + 1);
        }
        if !tok.starts_with('-') || tok == "-" {
            return Some(i);
        }
        if tok == "-n" || tok == "--adjustment" {
            i += 2;
            continue;
        }
        if tok.starts_with("--adjustment=")
            || tok[1..]
                .chars()
                .all(|ch| ch == '+' || ch == '-' || ch.is_ascii_digit())
        {
            i += 1;
            continue;
        }
        i += 1;
    }
    None
}

fn time_command_index(tokens: &[String]) -> Option<usize> {
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            return (i + 1 < tokens.len()).then_some(i + 1);
        }
        if !tok.starts_with('-') || tok == "-" {
            return Some(i);
        }
        if matches!(tok, "-f" | "--format" | "-o" | "--output") {
            i += 2;
            continue;
        }
        if tok.starts_with("--format=") || tok.starts_with("--output=") {
            i += 1;
            continue;
        }
        i += 1;
    }
    None
}

fn timeout_command_index(tokens: &[String]) -> Option<usize> {
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            i += 1;
            break;
        }
        if !tok.starts_with('-') || tok == "-" {
            break;
        }
        if matches!(tok, "-k" | "--kill-after" | "-s" | "--signal") {
            i += 2;
            continue;
        }
        if tok.starts_with("--kill-after=") || tok.starts_with("--signal=") {
            i += 1;
            continue;
        }
        i += 1;
    }
    if i >= tokens.len() {
        return None;
    }
    let command_idx = i + 1;
    (command_idx < tokens.len()).then_some(command_idx)
}

fn indirect_command_index(
    program: &str,
    tokens: &[String],
    raw_tokens: &[String],
) -> Option<usize> {
    match program {
        "xargs" => xargs_command_index(tokens),
        "env" => env_command_index(tokens, raw_tokens),
        "nohup" | "setsid" => first_non_option_index(tokens, 1, &[]),
        "nice" => nice_command_index(tokens),
        "time" => time_command_index(tokens),
        "timeout" => timeout_command_index(tokens),
        "stdbuf" => first_non_option_index(tokens, 1, &["-i", "-o", "-e"]),
        "command" => command_builtin_index(tokens),
        "exec" => exec_builtin_index(tokens),
        _ => None,
    }
}

pub(crate) fn effective_command_tokens(segment: &str) -> Vec<String> {
    let tokens = tokenize_shell_words(segment);
    let Some(start) = command_word_index(&tokens, true) else {
        return Vec::new();
    };
    let mut current = tokens[start..].to_vec();
    for _ in 0..4 {
        let Some(program) = current.first().and_then(|token| {
            std::path::Path::new(token)
                .file_name()
                .and_then(|name| name.to_str())
        }) else {
            break;
        };
        let lower = current
            .iter()
            .map(|token| token.to_ascii_lowercase())
            .collect::<Vec<_>>();
        let Some(index) = indirect_command_index(&program.to_ascii_lowercase(), &lower, &current)
        else {
            break;
        };
        current = current[index..].to_vec();
    }
    current
}

fn is_shell_program(program: &str) -> bool {
    matches!(program, "bash" | "sh" | "zsh" | "ksh" | "dash")
}

// Script interpreters also accept `-c` / `-e` to pass and execute a code string directly,
// which would bypass the per-segment blacklist validation.
fn is_interpreter_program(program: &str) -> bool {
    matches!(
        program,
        "python" | "python3" | "perl" | "ruby" | "node" | "php" | "awk" | "lua"
    )
}

fn is_python_program(program: &str) -> bool {
    matches!(program, "python" | "python3")
}

/// Presence of "second-interpretation" options: `-c` / `--command` (shells),
/// `-c` / `-e` (interpreters).
fn shell_c_option_present(program: &str, tokens: &[String]) -> bool {
    let is_shell = is_shell_program(program);
    let is_interpreter = is_interpreter_program(program);
    if !is_shell && !is_interpreter {
        return false;
    }
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" {
            return false;
        }
        if !tok.starts_with('-') || tok == "-" {
            return false;
        }
        // Clustered short options can also carry `-c` / `-e` (e.g. `bash -lc`,
        // `perl -le`, `node -pe`). Match only single-dash tokens longer than 2
        // chars: long options like `--norc`, and `-c` / `-e` already exactly
        // matched, are not hit (note tokens are lowercased, so `-C` noclobber gets
        // conflated with `-c` — a pre-existing false positive).
        let grouped = !tok.starts_with("--") && tok.len() > 2;
        if is_shell && (tok == "-c" || tok == "--command" || (grouped && tok.contains('c'))) {
            return true;
        }
        if is_interpreter
            && (tok == "-c" || tok == "-e" || (grouped && (tok.contains('c') || tok.contains('e'))))
        {
            return true;
        }
        i += 1;
    }
    false
}

/// Extract the code string of `python -c <code>` (the tokenizer already removed
/// shell quotes).
/// - `Ok(None)`: no `-c` option (e.g. `python3 script.py` / `python3 -m mod`), no
///   code string involved.
/// - `Ok(Some(code))`: a literal code string was extracted.
/// - `Err`: `-c` is present but the code string cannot be statically obtained
///   (missing / empty / from shell variable expansion), fail-closed.
fn python_c_argument(tokens: &[String]) -> Result<Option<String>, String> {
    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "--" || !tok.starts_with('-') || tok == "-" {
            // Option region ended without `-c` -> ordinary script / module
            // execution.
            return Ok(None);
        }
        // `-W` / `-X` / `-m` consume one value argument (attached `-Wfoo` or
        // separate `-W foo`); the value itself is not `-c`, so skip and keep
        // looking.
        if matches!(tok, "-W" | "-X" | "-m")
            || tok.starts_with("-W")
            || tok.starts_with("-X")
            || tok.starts_with("-m")
        {
            if matches!(tok, "-W" | "-X" | "-m") {
                i += 1; // skip the value argument token
            }
            i += 1;
            continue;
        }
        if tok == "-c" {
            return match tokens.get(i + 1) {
                Some(code) => Ok(Some(code.clone())),
                None => Err("`-c` requires a code argument".to_string()),
            };
        }
        if let Some(code) = tok.strip_prefix("-c") {
            // Attached form `-cCODE`.
            return if code.is_empty() {
                Err("`-c` requires a non-empty code argument".to_string())
            } else {
                Ok(Some(code.to_string()))
            };
        }
        // Clustered short options may contain `-c` (e.g. `-uc` equals `-u -c`,
        // `-Oc` equals `-O -c`).
        if tok.contains('c') {
            return match tokens.get(i + 1) {
                Some(code) => Ok(Some(code.clone())),
                None => Err("`-c` requires a code argument".to_string()),
            };
        }
        i += 1;
    }
    Ok(None)
}

/// Validate the code string passed to `python -c` (static, best-effort): strip
/// whitespace, lowercase-flatten, then scan for dangerous primitives; a hit is
/// rejected (fail-closed). Occurrences inside comments/strings are matched too —
/// an accepted false positive (safety first).
///
/// Note: this is a static defense at the same level as the whole command audit,
/// not a sandbox — deliberately obfuscated code can in theory always find blind
/// spots. But compared with blanket blocking, it turns python `-c` from
/// "unauditable" into "auditable", covering direct calls and common obfuscation
/// entry points (getattr / __import__ / exec / eval / dunder escape chains, etc.).
fn validate_python_code(code: &str) -> Result<(), String> {
    // `$` or backticks in the code string suggest the content may come from shell
    // variable expansion (e.g. `python3 -c $CODE`); the audit cannot see the
    // expanded content -> fail-closed.
    if code.contains('$') || code.contains('`') {
        return Err(
            "python -c code must be a literal quoted string without shell expansion (`$`)"
                .to_string(),
        );
    }
    let compact: String = code
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect();
    if compact.is_empty() {
        return Err("python -c requires a non-empty code string".to_string());
    }

    const DANGEROUS_PYTHON_PATTERNS: &[&str] = &[
        // —— External command execution / process control ——
        "os.system",
        "os.popen",
        "os.spawn",
        "os.exec",
        "os.fork",
        "os.kill",
        "os.killpg",
        "subprocess.",
        "importsubprocess",
        "fromsubprocess",
        // Block every import form of dangerous modules (including `import X as` /
        // `from X import *`): otherwise `from os import system; system("...")` /
        // `import os as o; o.system(...)` / `import os; x = os; x.system(...)` all
        // bypass the direct `os.system` match above.
        "importos",
        "fromos",
        "importposix",
        "fromposix",
        "importshutil",
        "fromshutil",
        "importsocket",
        "fromsocket",
        "importctypes",
        "fromctypes",
        "importpty",
        "frompty",
        "importmarshal",
        "frommarshal",
        "importpickle",
        "frompickle",
        "importtelnetlib",
        "fromtelnetlib",
        "importftplib",
        "fromftplib",
        "importsmtplib",
        "fromsmtplib",
        "importpwn",
        "frompwn",
        "importcommands",
        "fromcommands",
        "importimportlib",
        "fromimportlib",
        // Fetch the already-loaded os via `sys.modules` and call through it (e.g.
        // `import json` loads os internally).
        "sys.modules",
        "commands.getoutput",
        "signal.kill",
        "pty.",
        // —— File destruction / permissions / ownership / links / renaming ——
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.removedirs",
        "os.chmod",
        "os.chown",
        "os.chflags",
        "os.rename",
        "os.replace",
        "os.link",
        "os.symlink",
        "os.truncate",
        "os.mkfifo",
        "os.mknod",
        "os.setuid",
        "os.setgid",
        "shutil.rmtree",
        "shutil.move",
        "shutil.chown",
        // Path(...) method-call forms (`.unlink()` etc.); `).replace(` also covers
        // os.replace.
        ").unlink(",
        ").rmdir(",
        ").rename(",
        ").replace(",
        ").chmod(",
        ").chown(",
        ").symlink_to(",
        ").write_text(",
        ").write_bytes(",
        ").truncate(",
        // —— Dynamic execution / dynamic import (obfuscation and escape entry
        // points) ——
        "eval(",
        "exec(",
        "execfile(",
        "compile(",
        "__import__",
        "importlib.",
        "getattr(",
        "setattr(",
        "__builtins__",
        "__globals__",
        "__subclasses__",
        "ctypes.",
        "marshal.",
        "pickle.loads",
        // —— Network / listening (mirrors the shell-side nc / telnet / socat
        // blacklist) ——
        "socket.",
        "http.server",
        "baseserver",
        "socketserver",
        "telnetlib.",
        "ftplib.",
        "smtplib.",
        "asyncio.start_server",
        "pwn.",
    ];

    for pattern in DANGEROUS_PYTHON_PATTERNS {
        if compact.contains(pattern) {
            return Err(format!(
                "python -c code contains blocked primitive '{pattern}'"
            ));
        }
    }
    Ok(())
}

fn find_has_blocked_exec_semantics(tokens: &[String]) -> Option<&str> {
    const BLOCKED_FIND_FLAGS: &[&str] = &["-delete", "-exec", "-execdir", "-ok", "-okdir"];
    fn find_primary_arg_count(tok: &str) -> usize {
        match tok {
            "-amin" | "-anewer" | "-atime" | "-cmin" | "-cnewer" | "-context" | "-ctime"
            | "-files0-from" | "-fls" | "-fprint" | "-fprint0" | "-fstype" | "-gid" | "-group"
            | "-ilname" | "-iname" | "-inum" | "-ipath" | "-iregex" | "-iwholename" | "-links"
            | "-lname" | "-maxdepth" | "-mindepth" | "-mmin" | "-mtime" | "-name" | "-newer"
            | "-newerxy" | "-path" | "-perm" | "-printf" | "-regex" | "-samefile" | "-size"
            | "-since" | "-type" | "-uid" | "-used" | "-user" | "-wholename" | "-xtype" => 1,
            "-fprintf" => 2,
            _ => 0,
        }
    }

    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok.starts_with('-') || matches!(tok, "!" | "(" | ")" | ",") {
            break;
        }
        i += 1;
    }
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if BLOCKED_FIND_FLAGS.contains(&tok) {
            return Some(tok);
        }
        if tok == "--" || matches!(tok, "!" | "(" | ")" | "," | "-a" | "-and" | "-o" | "-or") {
            i += 1;
            continue;
        }
        let arg_count = find_primary_arg_count(tok);
        if arg_count > 0 {
            i += 1 + arg_count;
            continue;
        }
        i += 1;
    }
    None
}

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
    // `git stash` (including pop/drop/clear subactions) stashes or even discards
    // working-tree changes and can easily lose uncommitted work; the agent may
    // not invoke it on its own.
    ("stash", "git stash is blocked by sandbox policy"),
    // `git rm` physically deletes files from the working tree, unrecoverable;
    // `git rm --cached` only removes the index entry, but blocking outright is
    // safer than per-argument analysis. Use safe tools like `trash` to delete
    // files.
    (
        "rm",
        "git rm is blocked by sandbox policy; use trash or similar safe-delete tool",
    ),
];

/// If a `git` subcommand hits the block list, return its rejection reason.
fn blocked_git_subcommand(command_tokens: &[String]) -> Option<&'static str> {
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
/// Unlike `BLOCKED_GIT_SUBCOMMANDS` (globally banned ones like push/stash), the
/// subcommands below are harmless under some argument combinations (e.g. `git
/// switch` branch switching, `git restore --staged` unstaging), so block only
/// when they would truly destroy uncommitted changes (working-tree/staged
/// changes, untracked files), avoiding collateral damage to normal workflows.
/// Returns the rejection reason on a hit.
///
/// Covers the user requirement "ban `git checkout --` and any command that
/// deletes currently uncommitted files irreversibly".
fn blocked_git_destructive(command_tokens: &[String]) -> Option<&'static str> {
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
/// Command substitution (`$(...)` / `` `...` ``) can generate the program name at
/// runtime and stays banned. Process substitution `<(...)` / `>(...)` is allowed
/// after recursively validating the inner command, avoiding false blocks on
/// common usages like diff/sort.
fn validate_no_injection_surface(command: &str) -> Result<(), String> {
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
            return Err(
                "command substitution `$(...)` is not allowed; pass a literal command instead"
                    .to_string(),
            );
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

fn normalize_path(path: &std::path::Path) -> std::path::PathBuf {
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
fn expand_tilde_and_home(arg: &str) -> Result<String, String> {
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
fn validate_single_segment(command: &str) -> Result<(), String> {
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
        "rm",
        "dd",
        "chmod",
        "chown",
        "chgrp",
        "kill",
        "pkill",
        "killall",
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

    // Common wrappers treat later tokens as the program that will actually run;
    // check only "the program name that will be executed", avoiding misjudging
    // ordinary content arguments (like the `rm` inside `printf '%s' rm`) as
    // dangerous commands.
    const DANGEROUS_PROGRAM_NAMES: &[&str] = &[
        "rm",
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
        "kill",
        "pkill",
        "killall",
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
        }
    }

    Ok(())
}

// =========================================================================
// Public entry point
// =========================================================================

/// Validate the safety of one complete command (including chained `&&` / `||`).
///
/// This is the audit module's single public entry point; `execute_command` just
/// calls it.
pub(crate) fn validate_execute_command(command: &str) -> Result<(), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("empty command".to_string());
    }

    // First line of defense: block shell injection surfaces (command substitution
    // / process substitution). Letting these through renders the segment
    // blacklist pointless.
    validate_no_injection_surface(command)?;

    // Second line of defense: split the chained command into segments and run the
    // program/argument blacklist on each one. That way `echo ok && rm -rf /` is
    // caught by the `rm` blacklist in the second segment.
    let segments = split_unquoted_segments(command);
    if segments.is_empty() {
        return Err("empty command".to_string());
    }
    if segments.len() > 1 {
        for seg in &segments {
            validate_single_segment(seg)?;
        }
        return Ok(());
    }
    validate_single_segment(&segments[0])
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::{
        SafeShellSubstitutionKind, ShellJoin, command_subcommand_index, effective_command_tokens,
        safe_shell_substitutions, split_unquoted_command_segments, split_unquoted_segments,
        tokenize_shell_words, validate_no_injection_surface,
    };

    // ---- split_unquoted_segments ----

    #[test]
    fn split_handles_chained_operators() {
        let segs = split_unquoted_segments("echo ok && rm -rf /tmp/foo");
        assert_eq!(
            segs,
            vec!["echo ok".to_string(), "rm -rf /tmp/foo".to_string()]
        );
    }

    #[test]
    fn split_handles_pipe_and_semicolon() {
        let segs = split_unquoted_segments("a | b ; c || d");
        assert_eq!(
            segs,
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }

    #[test]
    fn split_preserves_fd_duplication_redirections() {
        let segments =
            split_unquoted_command_segments("cargo test --bin a 2>&1 | tail -6 && echo done");
        assert_eq!(
            segments
                .iter()
                .map(|segment| (segment.command.as_str(), segment.join))
                .collect::<Vec<_>>(),
            vec![
                ("cargo test --bin a 2>&1", ShellJoin::Start),
                ("tail -6", ShellJoin::Other),
                ("echo done", ShellJoin::And),
            ]
        );

        let segs = split_unquoted_segments("cmd &>out && cat out");
        assert_eq!(segs, vec!["cmd &>out".to_string(), "cat out".to_string()]);
    }

    #[test]
    fn split_preserves_success_chain_semantics() {
        let segments = split_unquoted_command_segments("a && b; c");
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.join)
                .collect::<Vec<_>>(),
            vec![ShellJoin::Start, ShellJoin::And, ShellJoin::Other]
        );
    }

    #[test]
    fn split_does_not_break_inside_single_quotes() {
        let segs = split_unquoted_segments("echo 'a && b' ; echo done");
        assert_eq!(
            segs,
            vec!["echo 'a && b'".to_string(), "echo done".to_string()]
        );
    }

    #[test]
    fn split_does_not_break_inside_double_quotes() {
        let segs = split_unquoted_segments("echo \"a | b\" && true");
        assert_eq!(segs, vec!["echo \"a | b\"".to_string(), "true".to_string()]);
    }

    #[test]
    fn split_ignores_quoted_heredoc_body_content() {
        let segs = split_unquoted_segments("cat <<'EOF'\nrm -rf /\nEOF\nls");
        assert_eq!(segs, vec!["cat <<'EOF'".to_string(), "ls".to_string()]);
    }

    // ---- tokenize_shell_words ----

    #[test]
    fn tokenize_shell_words_respects_single_and_double_quotes() {
        let tokens = tokenize_shell_words(r#"printf '%s\n' "a b" '\$(literal)'"#);
        assert_eq!(
            tokens,
            vec![
                "printf".to_string(),
                "%s\\n".to_string(),
                "a b".to_string(),
                "\\$(literal)".to_string()
            ]
        );
    }

    #[test]
    fn command_analysis_handles_wrappers_and_value_options() {
        let git = effective_command_tokens("env -C /tmp FOO=1 git -C /repo status");
        let git_index = command_subcommand_index(&git).unwrap();
        assert_eq!(git[git_index], "status");

        let cargo = effective_command_tokens("cargo --manifest-path Cargo.toml check");
        let cargo_index = command_subcommand_index(&cargo).unwrap();
        assert_eq!(cargo[cargo_index], "check");
    }

    // ---- injection surface ----

    #[test]
    fn injection_blocks_dollar_paren() {
        assert!(validate_no_injection_surface("echo $(whoami)").is_err());
    }

    #[test]
    fn injection_blocks_backtick_command_substitution() {
        assert!(validate_no_injection_surface("echo `whoami`").is_err());
    }

    #[test]
    fn injection_allows_heredoc_and_herestring() {
        assert!(validate_no_injection_surface("cat <<EOF").is_ok());
        assert!(validate_no_injection_surface("cat <<<\"hi\"").is_ok());
    }

    #[test]
    fn injection_allows_command_substitution_text_inside_quoted_heredoc() {
        assert!(validate_no_injection_surface("cat <<'EOF'\n$(whoami)\nEOF").is_ok());
        assert!(validate_no_injection_surface("cat <<'EOF'\n`whoami`\nEOF").is_ok());
    }

    #[test]
    fn injection_blocks_command_substitution_inside_unquoted_heredoc() {
        assert!(validate_no_injection_surface("cat <<EOF\n$(whoami)\nEOF").is_err());
        assert!(validate_no_injection_surface("cat <<EOF\n`whoami`\nEOF").is_err());
    }

    #[test]
    fn injection_allows_validated_process_substitution() {
        assert!(validate_no_injection_surface("diff <(echo a) <(echo b)").is_ok());
        assert!(validate_no_injection_surface("cat <(printf '%s' ok)").is_ok());
    }

    #[test]
    fn injection_blocks_unsafe_or_unterminated_process_substitution() {
        assert!(validate_no_injection_surface("cat <(rm -rf target)").is_err());
        assert!(validate_no_injection_surface("cat <(echo missing").is_err());
    }

    #[test]
    fn injection_allows_clean_command() {
        assert!(validate_no_injection_surface("cargo build --release").is_ok());
    }

    #[test]
    fn injection_treats_single_quoted_as_literal() {
        // A `$()` entirely inside single quotes is a literal; bash does not
        // expand it.
        assert!(validate_no_injection_surface("echo 'price: $(100)'").is_ok());
        assert!(validate_no_injection_surface("echo '`whoami`'").is_ok());
    }

    #[test]
    fn injection_treats_double_quoted_process_substitution_like_text_as_literal() {
        assert!(validate_no_injection_surface(r#"echo "<(literal)""#).is_ok());
        assert!(validate_no_injection_surface(r#"echo ">(literal)""#).is_ok());
    }

    #[test]
    fn injection_treats_escaped_substitution_markers_as_literal() {
        assert!(validate_no_injection_surface(r#"echo \$(whoami)"#).is_ok());
        assert!(validate_no_injection_surface(r#"echo "\$(whoami)""#).is_ok());
        assert!(validate_no_injection_surface(r#"echo "\`whoami\`""#).is_ok());
    }

    #[test]
    fn injection_still_blocks_substitution_inside_double_quotes() {
        assert!(validate_no_injection_surface(r#"echo "user=$(whoami)""#).is_err());
    }

    // ---- end-to-end validate_execute_command ----

    fn validate(cmd: &str) -> Result<(), String> {
        super::validate_execute_command(cmd)
    }

    #[test]
    fn blocks_chained_rm_after_safe_prefix() {
        let err = validate("echo ok && rm -rf /").unwrap_err();
        assert!(err.contains("rm"), "expected rm blocked, got: {err}");
    }

    #[test]
    fn blocks_rm_even_within_current_directory() {
        let err = validate("rm -rf ./target").unwrap_err();
        assert!(err.contains("rm"), "expected rm blocked, got: {err}");
    }

    #[test]
    fn blocks_shell_rm_with_glob_expansion() {
        let err = validate("rm -rf *.zcompdump").unwrap_err();
        assert!(err.contains("rm"), "expected rm blocked, got: {err}");
    }

    #[test]
    fn blocks_sudo_anywhere_in_chain() {
        let err = validate("true ; sudo reboot").unwrap_err();
        assert!(
            err.contains("sudo") || err.contains("reboot"),
            "expected sudo/reboot to be blocked, got: {err}"
        );
    }

    #[test]
    fn blocks_eval_segment() {
        let err = validate("eval \"echo hi\"").unwrap_err();
        assert!(err.contains("eval"), "expected eval blocked, got: {err}");
    }

    #[test]
    fn blocks_bash_dash_c() {
        let err = validate("bash -c \"echo ok\"").unwrap_err();
        assert!(err.contains("-c"), "expected `bash -c` blocked, got: {err}");
    }

    #[test]
    fn allows_bash_script_arg_named_dash_c() {
        assert!(validate("bash script.sh -c literal").is_ok());
    }

    #[test]
    fn allows_bash_running_a_script_file() {
        assert!(validate("bash run.sh").is_ok());
    }

    #[test]
    fn blocks_command_substitution() {
        let err = validate("echo $(whoami)").unwrap_err();
        assert!(
            err.contains("command substitution"),
            "expected $(...) blocked, got: {err}"
        );
    }

    #[test]
    fn detects_literal_file_read_substitution_for_any_simple_outer_command() {
        // The FileRead branch of safe_shell_substitutions (replaced the removed
        // safe_file_read_substitutions)
        let substitutions = safe_shell_substitutions(
            r#"curl --data "$(cat /tmp/request.json)" https://example.test/api"#,
        );
        assert_eq!(substitutions.len(), 1);
        assert_eq!(
            substitutions[0].kind,
            SafeShellSubstitutionKind::FileRead {
                path: "/tmp/request.json".to_string()
            }
        );
    }

    #[test]
    fn literal_file_read_substitution_requires_a_complete_simple_shell_word() {
        let kinds: Vec<_> = safe_shell_substitutions(r#"echo "$(cat /tmp/dsl.json)""#)
            .into_iter()
            .map(|s| s.kind)
            .collect();
        assert_eq!(
            kinds,
            vec![SafeShellSubstitutionKind::FileRead {
                path: "/tmp/dsl.json".to_string()
            }]
        );
        assert_eq!(
            safe_shell_substitutions(r#"bytedcli --dsl "$(cat /tmp/dsl.json)""#).len(),
            1
        );
        assert_eq!(
            safe_shell_substitutions(r#"echo "$(cat /tmp/a)" "$(cat /tmp/b)""#).len(),
            2
        );
        // Outer pipes/connectors -> the whole command is not recognized as a safe
        // substitution
        assert!(safe_shell_substitutions(r#"echo "$(cat /tmp/dsl.json)" | jq ."#).is_empty());
        assert!(safe_shell_substitutions(r#"echo "$(cat /tmp/dsl.json)" && id"#).is_empty());
        // A $() that is not a complete word is still an injection surface
        assert!(validate(r#"echo "$(cat /tmp/dsl.json)""#).is_err());
        assert!(validate(r#"bytedcli --dsl "prefix$(cat /tmp/dsl.json)""#).is_err());
        assert!(validate(r#"bytedcli --dsl "$(cat /tmp/dsl.json)suffix""#).is_err());
    }

    #[test]
    fn file_read_substitution_requires_one_literal_absolute_path() {
        // Non-absolute, non-literal paths must not materialize as FileRead (may
        // fall back to a validate-passed harmless Command)
        for command in [
            r#"echo "$(cat /tmp/a /tmp/b)""#,
            r#"echo "$(cat $HOME/a)""#,
            r#"echo "$(cat /tmp/a; id)""#,
            r#"echo "$(cat /tmp/a$(id))""#,
            r#"echo "$(cat /tmp/../secret)""#,
            r#"echo "$(cat /tmp/*.json)""#,
        ] {
            let substitutions = safe_shell_substitutions(command);
            assert!(
                substitutions
                    .iter()
                    .all(|s| !matches!(s.kind, SafeShellSubstitutionKind::FileRead { .. })),
                "unsafe cat path must not be materialized as FileRead: {command}"
            );
        }
        // Nested $() rejected as a whole
        assert!(safe_shell_substitutions(r#"echo "$(cat /tmp/a$(id))""#).is_empty());
    }

    #[test]
    fn allows_arithmetic_expansion() {
        assert!(validate("echo $((RANDOM % 20 + 1))").is_ok());
        assert!(validate("echo $((1 + 2 * 3))").is_ok());
    }

    #[test]
    fn blocks_command_substitution_nested_in_arithmetic() {
        let err = validate("echo $(( $(whoami) + 1 ))").unwrap_err();
        assert!(
            err.contains("command substitution"),
            "expected nested $(...) blocked, got: {err}"
        );
    }

    #[test]
    fn allows_subcommand_patterns_that_resemble_blocked_programs() {
        // `git rm` is now blocked by BLOCKED_GIT_SUBCOMMANDS (unrecoverable
        // deletion).
        assert!(validate("git rm file.txt").is_err());
        assert!(validate("git mv old.txt new.txt").is_ok());
        assert!(validate("docker rm my_container").is_ok());
        assert!(validate("docker rmi my_image").is_ok());
        assert!(validate("npm rm some-package").is_ok());
        assert!(validate("pip install rsync").is_ok());
    }

    #[test]
    fn blocks_git_push_in_all_common_forms() {
        assert!(validate("git push").is_err());
        assert!(validate("git push origin main").is_err());
        assert!(validate("git push --force").is_err());
        assert!(validate("git push --force-with-lease origin").is_err());
        assert!(validate("git push -u origin main").is_err());
        assert!(validate("git push origin --tags").is_err());
        assert!(validate("git -C /repo push").is_err());
        assert!(validate("git -C /repo push origin main").is_err());
        assert!(validate("git -c user.email=a@b.c push").is_err());
        assert!(validate("git --git-dir=/repo push").is_err());
        assert!(validate("git --git-dir /repo push").is_err());
        assert!(validate("git --no-pager push").is_err());
        assert!(validate("git PUSH origin").is_err());
        assert!(validate("/usr/bin/git push").is_err());
        assert!(validate("git status && git push").is_err());
        assert!(validate("git push && echo done").is_err());
        assert!(validate("env git push").is_err());
        assert!(validate("env FOO=1 git push origin main").is_err());
        assert!(validate("xargs git push").is_err());
        assert!(validate("nohup git push").is_err());
        assert!(validate("command git push").is_err());
    }

    #[test]
    fn git_non_push_subcommands_remain_allowed() {
        assert!(validate("git status").is_ok());
        assert!(validate("git log --oneline -5").is_ok());
        assert!(validate("git diff").is_ok());
        assert!(validate("git diff --cached").is_ok());
        assert!(validate("git -C /repo status").is_ok());
        assert!(validate("git -C /repo log --oneline").is_ok());
        assert!(validate("git add -A").is_ok());
        assert!(validate("git commit -m msg").is_ok());
        assert!(validate("echo git push").is_ok());
        assert!(validate("printf '%s' push").is_ok());
    }

    #[test]
    fn blocks_git_stash_in_all_common_forms() {
        assert!(validate("git stash").is_err());
        assert!(validate("git stash list").is_err());
        assert!(validate("git stash pop").is_err());
        assert!(validate("git stash drop").is_err());
        assert!(validate("git stash clear").is_err());
        assert!(validate("git stash push -m wip").is_err());
        assert!(validate("git -C /repo stash").is_err());
        assert!(validate("git -c user.email=a@b.c stash").is_err());
        assert!(validate("git --git-dir=/repo stash").is_err());
        assert!(validate("git --no-pager stash").is_err());
        assert!(validate("git STASH").is_err());
        assert!(validate("/usr/bin/git stash").is_err());
        assert!(validate("git status && git stash").is_err());
        assert!(validate("git stash && echo done").is_err());
        assert!(validate("env git stash").is_err());
        assert!(validate("xargs git stash").is_err());
        assert!(validate("nohup git stash").is_err());
        assert!(validate("command git stash").is_err());
        assert!(validate("echo git stash").is_ok());
        assert!(validate("printf '%s' stash").is_ok());
    }

    #[test]
    fn shell_literal_rm_text_remains_allowed() {
        assert!(validate("echo 'rm -rf ~/.zcompdump*'").is_ok());
    }

    #[test]
    fn blocks_exec_flags_that_run_subsequent_args_as_commands() {
        assert!(validate("find . -exec rm {} +").is_err());
        assert!(validate("find . -execdir chmod 777 {} \\;").is_err());
        assert!(validate("find /tmp -ok rm {} \\;").is_err());
        assert!(validate("find . -okdir mv {} /tmp/ \\;").is_err());
        assert!(validate("find . -name '*.rs' -type f").is_ok());
        assert!(validate("find . -delete").is_err());
        assert!(validate("find . -empty -delete").is_err());
        assert!(validate(r#"find . "-exec" rm {} +"#).is_err());
        assert!(validate(r#"find . -name "-delete" -print"#).is_ok());
        assert!(validate(r#"find . -name "-exec" -print"#).is_ok());
        assert!(validate(r#"find . -printf "-delete\n""#).is_ok());
        // `git rm` is now blocked by BLOCKED_GIT_SUBCOMMANDS.
        assert!(validate("git rm file.txt").is_err());
        assert!(validate("docker rm container").is_ok());
        assert!(validate("npm rm pkg").is_ok());
        assert!(validate("pip install rsync").is_ok());
    }

    #[test]
    fn blocks_common_indirect_wrappers_but_allows_safe_payload_args() {
        assert!(validate("xargs rm").is_err());
        assert!(validate("env FOO=1 sudo whoami").is_err());
        assert!(validate("env FOO=1 rm -rf target").is_err());
        assert!(validate("nohup ssh user@host").is_err());
        assert!(validate("nice -n 5 chmod 777 file").is_err());
        assert!(validate("timeout --signal=KILL 10 dd if=/dev/zero of=foo").is_err());
        assert!(validate("command rm -rf *").is_err());
        assert!(validate("exec rm -rf *").is_err());

        assert!(validate(r#"xargs printf "%s\n" rm"#).is_ok());
        assert!(validate(r#"env FOO=1 cargo test"#).is_ok());
        assert!(validate(r#"nice -n 5 cargo check"#).is_ok());
        assert!(validate(r#"timeout 10 cargo test"#).is_ok());
    }

    #[test]
    fn leading_env_assignment_only_has_shell_meaning_when_shell_is_used() {
        assert!(validate("FOO=1 rm -rf target").is_ok());
        assert!(validate("FOO=1 rm -rf *.tmp").is_err());
    }

    #[test]
    fn allows_literal_dangerous_text_when_writing_files() {
        assert!(validate(r#"printf "%s\n" "-exec" "-delete" "rm -rf /""#).is_ok());
        assert!(validate("cat <<'EOF' > out.txt\n$(whoami)\n-exec\n-delete\nEOF").is_ok());
        assert!(validate("cat <<'EOF' > out.txt\n`whoami`\nEOF").is_ok());
        assert!(validate("printf '%s\n' '`whoami`'").is_ok());
    }

    #[test]
    fn allows_normal_dev_commands() {
        assert!(validate("cargo check --bin a").is_ok());
        assert!(validate("git status").is_ok());
        assert!(validate("ls -la").is_ok());
        assert!(validate("echo 'literal $(x)'").is_ok());
    }

    // ---- tilde / $HOME escape detection ----

    #[test]
    fn home_paths_are_allowed() {
        assert!(validate("ls ~").is_ok());
        assert!(validate("cat ~/.gitconfig").is_ok());
        assert!(validate("cat $HOME/.cargo/config.toml").is_ok());
    }

    #[test]
    fn tilde_escape_to_parent_dir_blocked() {
        // cwd=/Users/bytedance/rust_tools -> ~/.. walks up to /Users/bytedance ->
        // /Users -> /
        assert!(validate("cp foo.txt ~/../..").is_err());
        assert!(validate("cp foo.txt ~/..").is_err());
    }

    #[test]
    fn tilde_to_parent_blocked() {
        assert!(validate("ls ~/..").is_err());
    }

    #[test]
    fn home_env_var_escape_blocked() {
        assert!(validate("cp foo.txt $HOME/../../..").is_err());
    }

    // ---- python -c code-string audit ----

    #[test]
    fn python_dash_c_clean_code_allowed() {
        assert!(validate("python3 -c 'print(1 + 1)'").is_ok());
        assert!(validate(r#"python3 -c "import json; print(json.load(open('x.json')))""#).is_ok());
        assert!(validate("python -c 'print(sum(i*i for i in range(10)))'").is_ok());
        assert!(validate("python3 -u -c 'print(\"hi\")'").is_ok());
        assert!(validate("python3 -c'print(1)'").is_ok());
        assert!(validate("python3 -W ignore -c 'print(1)'").is_ok());
        assert!(validate("python3 -c 'print(len(\"abc\"))'").is_ok());
        assert!(validate("python3 -c 'import re; print(re.findall(r\"\\d+\", \"a1b2\"))'").is_ok());
    }

    #[test]
    fn python_dash_c_dangerous_code_blocked() {
        let err = validate("python3 -c 'import os; os.system(\"rm -rf /\")'").unwrap_err();
        assert!(err.contains("blocked primitive"), "got: {err}");
        assert!(validate("python3 -c 'os.remove(\"x\")'").is_err());
        assert!(validate("python3 -c 'import subprocess; subprocess.run([\"ls\"])'").is_err());
        assert!(validate("python3 -c 'from subprocess import call; call(\"ls\")'").is_err());
        assert!(validate("python3 -c 'eval(\"1+1\")'").is_err());
        assert!(validate("python3 -c 'exec(\"x=1\")'").is_err());
        assert!(validate("python3 -c 'getattr(os, \"system\")(\"rm -rf /\")'").is_err());
        assert!(validate("python3 -c '__import__(\"os\").system(\"id\")'").is_err());
        assert!(validate("python3 -c 'shutil.rmtree(\"d\")'").is_err());
        assert!(validate("python3 -c 'import socket; socket.socket()'").is_err());
        assert!(validate("python3 -c 'ctypes.CDLL(None).system(\"id\")'").is_err());
        assert!(validate("python3 -c 'Path(\"x\").unlink()'").is_err());
        // Every import form of dangerous modules (from-import / aliasing /
        // variable copy / sys.modules).
        assert!(validate("python3 -c 'from os import system; system(\"rm -rf /\")'").is_err());
        assert!(validate("python3 -c 'import os as o; o.system(\"id\")'").is_err());
        assert!(validate("python3 -c 'import os; x = os; x.system(\"id\")'").is_err());
        assert!(validate("python3 -c 'import sys; sys.modules[\"os\"].system(\"id\")'").is_err());
        assert!(validate("python3 -c 'import posix; posix.system(\"id\")'").is_err());
        assert!(validate("python3 -c 'import signal; signal.kill(1, 9)'").is_err());
        // Common obfuscations: still hit after stripping whitespace / changing
        // case.
        assert!(validate("python3 -c 'os . system(\"id\")'").is_err());
        assert!(validate("python3 -c 'OS.SYSTEM(\"id\")'").is_err());
        // Clustered short option `-uc` equals `-u -c`.
        assert!(validate("python3 -uc 'os.system(\"id\")'").is_err());
        // The `__subclasses__` sandbox escape chain.
        assert!(validate("python3 -c '().__class__.__bases__[0].__subclasses__()'").is_err());
    }

    #[test]
    fn python_dash_c_unverifiable_code_blocked() {
        // Code comes from shell variable expansion, statically unverifiable ->
        // fail-closed.
        assert!(validate("python3 -c $CODE").is_err());
        assert!(validate("CODE=x python3 -c $CODE").is_err());
        assert!(validate("python3 -c \"$CODE\"").is_err());
        // Missing / empty code.
        assert!(validate("python3 -c").is_err());
        assert!(validate("python3 -c ''").is_err());
        // Clustered short options carrying `-c` without code.
        assert!(validate("python3 -uc").is_err());
    }

    #[test]
    fn python_without_dash_c_unchanged() {
        assert!(validate("python3 script.py").is_ok());
        assert!(validate("python3 -m json.tool < data.json").is_ok());
        assert!(validate("python3 --version").is_ok());
    }

    #[test]
    fn grouped_short_options_caught() {
        // Clustered short options can smuggle `-c` / `-e` too: `bash -lc` /
        // `perl -le` / `node -pe` / `ruby -ne` are equivalent to `-c` / `-e` and
        // must not pass.
        assert!(validate("bash -lc 'rm -rf /'").is_err());
        assert!(validate("perl -le 'system(\"rm -rf /\")'").is_err());
        assert!(validate("node -pe 'require(\"child_process\").execSync(\"id\")'").is_err());
        assert!(validate("ruby -ne 'puts 1'").is_err());
        // Legitimate short options without `-c` / `-e` are unaffected (`-e` on a
        // shell is errexit, not code; `--norc` is a long option).
        assert!(validate("bash -e script.sh").is_ok());
        assert!(validate("bash --norc script.sh").is_ok());
        assert!(validate("perl -w script.pl").is_ok());
    }

    #[test]
    fn indirect_interpreter_dash_c_audited() {
        // Clean indirect python -c still passes.
        assert!(validate("env python3 -c 'print(1)'").is_ok());
        assert!(validate("nohup env python3 -c 'print(1)'").is_ok());
        assert!(validate("env env python3 -c 'print(1)'").is_ok());
        // Wrappers can no longer bypass validation via `-c`.
        let err = validate("env python3 -c 'os.system(\"id\")'").unwrap_err();
        assert!(err.contains("blocked primitive"), "got: {err}");
        assert!(validate("xargs python3 -c 'os.system(\"id\")'").is_err());
        assert!(validate("nohup python3 -c 'os.system(\"id\")'").is_err());
        // Layered wrappers (`nohup env python3 -c ...`) are likewise caught by
        // deep unwrapping.
        assert!(validate("nohup env python3 -c 'os.system(\"id\")'").is_err());
        assert!(validate("env bash -c 'echo ok && rm -rf /'").is_err());
        assert!(validate("timeout 10 bash -c 'rm -rf /'").is_err());
        assert!(validate("env perl -e 'system(\"rm -rf /\")'").is_err());
        assert!(validate("env env bash -c 'rm -rf /'").is_err());
    }
}
