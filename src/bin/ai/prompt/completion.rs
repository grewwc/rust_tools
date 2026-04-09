use std::{cmp::Ordering, fs, path::PathBuf};

use rustyline::{
    Context, Editor, Helper,
    completion::{Completer, Pair},
    highlight::Highlighter,
    hint::Hinter,
    history::DefaultHistory,
    validate::Validator,
};

use crate::commonw::utils::expanduser;

pub(super) type LineEditor = Editor<CommandCompleter, DefaultHistory>;

#[derive(Clone, Default)]
pub(super) struct CommandCompleter;

impl CommandCompleter {
    fn top_level_commands() -> &'static [&'static str] {
        &[
            "/help",
            ":help",
            "/h",
            ":h",
            "/history",
            ":history",
            "/feishu-auth",
            ":feishu-auth",
            "/share",
            ":share",
            "/agents",
            ":agents",
            "/agent",
            ":agent",
            "/sessions",
            ":sessions",
        ]
    }

    fn agent_subcommands() -> &'static [&'static str] {
        &["help", "list", "current", "use", "auto"]
    }

    fn session_subcommands() -> &'static [&'static str] {
        &[
            "list",
            "current",
            "new",
            "use",
            "delete",
            "clear-all",
            "export",
            "export-current",
            "export-last",
        ]
    }

    fn history_subcommands() -> &'static [&'static str] {
        &[
            "full",
            "user",
            "assistant",
            "tool",
            "system",
            "grep",
            "export",
            "copy",
            "3",
            "6",
            "10",
            "20",
        ]
    }

    pub(super) fn complete_for_line(line: &str, pos: usize) -> (usize, Vec<String>) {
        let pos = pos.min(line.len());
        let before = &line[..pos];
        if let Some((token_start, candidates)) = Self::complete_file_reference(before) {
            return (token_start, candidates);
        }
        let token_start = before
            .rfind(char::is_whitespace)
            .map(|idx| idx + 1)
            .unwrap_or(0);
        let token = &before[token_start..];
        if token.is_empty() && token_start == 0 && !before.is_empty() {
            return (pos, Vec::new());
        }

        let candidates = if token_start == 0 {
            Self::top_level_commands()
                .iter()
                .filter(|candidate| candidate.starts_with(token))
                .map(|candidate| (*candidate).to_string())
                .collect()
        } else {
            let mut words = before[..token_start].split_whitespace();
            let Some(first) = words.next() else {
                return (token_start, Vec::new());
            };
            let source = match first {
                "/agents" | ":agents" | "/agent" | ":agent" => Self::agent_subcommands(),
                "/sessions" | ":sessions" => Self::session_subcommands(),
                "/history" | ":history" => Self::history_subcommands(),
                _ => &[],
            };
            source
                .iter()
                .filter(|candidate| candidate.starts_with(token))
                .map(|candidate| (*candidate).to_string())
                .collect()
        };

        (token_start, candidates)
    }

    fn complete_file_reference(before: &str) -> Option<(usize, Vec<String>)> {
        let (token_start, raw_token, quote) = find_file_reference_token(before)?;
        let fragment = raw_token.strip_prefix('@')?;
        let fragment = if let Some(quote) = quote {
            fragment.strip_prefix(quote)?
        } else {
            fragment
        };
        let candidates = complete_path_fragment(fragment, quote);
        Some((token_start, candidates))
    }
}

fn find_file_reference_token(before: &str) -> Option<(usize, &str, Option<char>)> {
    let mut last_at = None;
    for (idx, ch) in before.char_indices() {
        if ch == '@' {
            last_at = Some(idx);
        }
    }
    let at_index = last_at?;
    let prev = before[..at_index].chars().next_back();
    if prev.is_some_and(|ch| !(ch.is_whitespace() || matches!(ch, '(' | '[' | '{' | '"' | '\'')))
    {
        return None;
    }

    let token = &before[at_index..];
    if token.len() <= 1 {
        return Some((at_index, token, None));
    }

    let mut chars = token.chars();
    let _ = chars.next();
    let next = chars.next()?;
    if next == '"' || next == '\'' {
        let closing_count = token[2..].chars().filter(|ch| *ch == next).count();
        if closing_count > 0 {
            return None;
        }
        return Some((at_index, token, Some(next)));
    }

    if token.chars().skip(1).any(char::is_whitespace) {
        return None;
    }
    Some((at_index, token, None))
}

fn complete_path_fragment(fragment: &str, quote: Option<char>) -> Vec<String> {
    let (dir_part, file_prefix) = split_fragment(fragment);
    let base_dir = resolve_completion_base_dir(dir_part);
    let Ok(entries) = fs::read_dir(&base_dir) else {
        return relative_navigation_candidates(fragment, quote);
    };

    let show_hidden = file_prefix.starts_with('.');
    let mut matches: Vec<FileCompletionCandidate> = relative_navigation_candidates(fragment, quote)
        .into_iter()
        .map(|replacement| FileCompletionCandidate::synthetic(replacement))
        .collect();

    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        if !name.starts_with(file_prefix) {
            continue;
        }
        let is_hidden = name.starts_with('.');
        if !show_hidden && is_hidden {
            continue;
        }

        let is_dir = path.is_dir();
        let mut replacement_path = String::new();
        replacement_path.push_str(dir_part);
        replacement_path.push_str(name);
        if is_dir {
            replacement_path.push('/');
        }
        matches.push(FileCompletionCandidate::path(
            format_file_completion(&replacement_path, quote, is_dir),
            is_dir,
            is_hidden,
        ));
    }

    if let Some(toggle) = hidden_toggle_candidate(dir_part, file_prefix, quote, show_hidden) {
        matches.push(FileCompletionCandidate::synthetic(toggle));
    }

    matches.sort_by(compare_file_completion_candidates);
    matches.dedup_by(|left, right| left.replacement == right.replacement);
    matches.into_iter().map(|candidate| candidate.replacement).collect()
}

fn split_fragment(fragment: &str) -> (&str, &str) {
    if fragment.is_empty() {
        return ("", "");
    }
    if fragment.ends_with('/') {
        return (fragment, "");
    }
    if let Some(idx) = fragment.rfind('/') {
        return (&fragment[..idx + 1], &fragment[idx + 1..]);
    }
    ("", fragment)
}

fn resolve_completion_base_dir(dir_part: &str) -> PathBuf {
    if dir_part.is_empty() {
        return std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    }

    let expanded = expanduser(dir_part).to_string();
    let path = PathBuf::from(&expanded);
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn format_file_completion(path: &str, quote: Option<char>, is_dir: bool) -> String {
    let needs_quotes = quote.is_some() || path.contains(' ');
    if needs_quotes {
        let quote = quote.unwrap_or('"');
        if is_dir {
            format!("@{quote}{path}")
        } else {
            format!("@{quote}{path}{quote}")
        }
    } else {
        format!("@{path}")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileCompletionCandidate {
    replacement: String,
    is_dir: bool,
    is_hidden: bool,
    is_synthetic: bool,
}

impl FileCompletionCandidate {
    fn path(replacement: String, is_dir: bool, is_hidden: bool) -> Self {
        Self {
            replacement,
            is_dir,
            is_hidden,
            is_synthetic: false,
        }
    }

    fn synthetic(replacement: String) -> Self {
        Self {
            replacement,
            is_dir: true,
            is_hidden: false,
            is_synthetic: true,
        }
    }
}

fn compare_file_completion_candidates(
    left: &FileCompletionCandidate,
    right: &FileCompletionCandidate,
) -> Ordering {
    file_completion_rank(left)
        .cmp(&file_completion_rank(right))
        .then_with(|| left.replacement.to_ascii_lowercase().cmp(&right.replacement.to_ascii_lowercase()))
        .then_with(|| left.replacement.cmp(&right.replacement))
}

fn file_completion_rank(candidate: &FileCompletionCandidate) -> (u8, u8, u8) {
    let nav_rank = match candidate.replacement.as_str() {
        "@./" | "@\"./" | "@'./" => 0,
        "@../" | "@\"../" | "@'../" => 1,
        s if s.ends_with("/.") || s.ends_with("/.\"") || s.ends_with("/.'") => 3,
        _ => 2,
    };
    let kind_rank = if candidate.is_dir { 0 } else { 1 };
    let hidden_rank = if candidate.is_hidden { 1 } else { 0 };
    (nav_rank, kind_rank, hidden_rank)
}

fn relative_navigation_candidates(fragment: &str, quote: Option<char>) -> Vec<String> {
    let mut candidates = Vec::new();
    for candidate in ["./", "../"] {
        if candidate.starts_with(fragment) || fragment.is_empty() || fragment == "." || fragment == ".." {
            candidates.push(format_file_completion(candidate, quote, true));
        }
    }
    candidates
}

fn hidden_toggle_candidate(
    dir_part: &str,
    file_prefix: &str,
    quote: Option<char>,
    show_hidden: bool,
) -> Option<String> {
    if show_hidden {
        return None;
    }
    let toggle_path = if dir_part.is_empty() {
        if file_prefix.is_empty() || ".".starts_with(file_prefix) {
            "./.".to_string()
        } else {
            return None;
        }
    } else if file_prefix.is_empty() || ".".starts_with(file_prefix) {
        format!("{dir_part}.")
    } else {
        return None;
    };

    Some(format_file_completion(&toggle_path, quote, true))
}

impl Helper for CommandCompleter {}
impl Hinter for CommandCompleter {
    type Hint = String;
}
impl Highlighter for CommandCompleter {}
impl Validator for CommandCompleter {}

impl Completer for CommandCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (token_start, candidates) = Self::complete_for_line(line, pos);
        let candidates = candidates
            .into_iter()
            .map(|candidate| Pair {
                display: candidate.clone(),
                replacement: candidate,
            })
            .collect();
        Ok((token_start, candidates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_completion_expands_top_level_agent_command() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (_, pairs) = completer
            .complete("/agen", 5, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == "/agents"));
    }

    #[test]
    fn command_completion_lists_agent_subcommands() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (start, pairs) = completer
            .complete("/agents a", 9, &Context::new(&history))
            .unwrap();
        assert_eq!(start, 8);
        assert!(pairs.iter().any(|pair| pair.replacement == "auto"));
    }

    #[test]
    fn history_command_completion_is_suggested() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (_, pairs) = completer
            .complete("/his", 4, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == "/history"));
    }

    #[test]
    fn history_command_completion_lists_subcommands() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (start, pairs) = completer
            .complete("/history a", 10, &Context::new(&history))
            .unwrap();
        assert_eq!(start, 9);
        assert!(pairs.iter().any(|pair| pair.replacement == "assistant"));
    }

    #[test]
    fn history_command_completion_lists_extended_subcommands() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (_, pairs) = completer
            .complete("/history ", 9, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == "tool"));
        assert!(pairs.iter().any(|pair| pair.replacement == "system"));
        assert!(pairs.iter().any(|pair| pair.replacement == "grep"));
        assert!(pairs.iter().any(|pair| pair.replacement == "export"));
        assert!(pairs.iter().any(|pair| pair.replacement == "copy"));
    }

    #[test]
    fn file_completion_suggests_absolute_image_path() {
        let dir = std::env::temp_dir().join(format!("ai-complete-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let image = dir.join("shot.png");
        std::fs::write(&image, b"fake").unwrap();
        let line = format!("@{}", dir.join("sh").display());

        let (start, candidates) = CommandCompleter::complete_for_line(&line, line.len());

        assert_eq!(start, 0);
        assert!(candidates.iter().any(|candidate| candidate == &format!("@{}", image.display())));
    }

    #[test]
    fn file_completion_quotes_paths_with_spaces() {
        let dir = std::env::temp_dir().join(format!("ai complete {}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let image = dir.join("error shot.png");
        std::fs::write(&image, b"fake").unwrap();
        let line = format!("@\"{}/err", dir.display());

        let (_, candidates) = CommandCompleter::complete_for_line(&line, line.len());

        assert!(candidates.iter().any(|candidate| {
            candidate == &format!("@\"{}\"", image.display())
        }));
    }

    #[test]
    fn relative_navigation_candidates_are_prioritized() {
        let candidates = complete_path_fragment(".", None);

        assert!(candidates.len() >= 2);
        assert_eq!(candidates[0], "@./");
        assert_eq!(candidates[1], "@../");
    }

    #[test]
    fn hidden_toggle_candidate_is_offered_for_current_directory() {
        let candidates = complete_path_fragment("./", None);

        assert!(candidates.iter().any(|candidate| candidate == "@./."));
    }
}
