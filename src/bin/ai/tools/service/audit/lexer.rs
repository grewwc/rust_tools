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
pub(crate) struct HereDocSpec {
    delimiter: String,
    strip_tabs: bool,
    literal_body: bool,
}

pub(crate) fn parse_heredoc_at(command: &str, start: usize) -> Option<(usize, HereDocSpec)> {
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

pub(crate) fn matches_heredoc_terminator(line: &str, spec: &HereDocSpec) -> bool {
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

pub(crate) fn validate_and_skip_heredoc_bodies(
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

pub(crate) fn command_word_index(tokens: &[String], shell_context: bool) -> Option<usize> {
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

pub(crate) fn indirect_command_index(
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
    // Env-assignment prefixes (`FOO=1 rm ...`) only have shell meaning when
    // the command is shell-executed; a no-shell segment execs `FOO=1` as the
    // literal program and fails, so it must not be skipped here.
    let shell_context = crate::cmd::run::command_requires_shell(segment);
    let Some(start) = command_word_index(&tokens, shell_context) else {
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

/// True when the wrapper chain of `segment` (peeled like
/// `effective_command_tokens`) contains `xargs`, directly or behind further
/// wrappers (`timeout 5 xargs kill 123`). `xargs` appends stdin items as
/// extra arguments at runtime, so the final command's argument set is never
/// fully visible on the command line; callers use this to fail closed for
/// commands whose runtime arguments cannot be audited.
pub(crate) fn effective_chain_uses_xargs(segment: &str) -> bool {
    let tokens = tokenize_shell_words(segment);
    let shell_context = crate::cmd::run::command_requires_shell(segment);
    let Some(start) = command_word_index(&tokens, shell_context) else {
        return false;
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
        if program.eq_ignore_ascii_case("xargs") {
            return true;
        }
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
    false
}
