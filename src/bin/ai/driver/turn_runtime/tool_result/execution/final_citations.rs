//! Final-response citation gate: extracts `path:line` citations from
//! final answers and validates them against local files before the answer is
//! accepted.
//! A token named only to deny or to question it is exempt (see
//! `citation_in_non_assertive_context`): a correct refusal that names the path
//! it declines to cite is not an evidence claim, and revalidating it would
//! force the model to rewrite an already-correct answer.

use super::*;

pub(in crate::ai::driver::turn_runtime) const FINAL_CITATION_RETRY_MARKER: &str =
    "[final-citation-retry]";
pub(in crate::ai::driver::turn_runtime) const FINAL_CITATION_UNVERIFIED_NOTE: &str = "runtime:final_citation_unverified\nA final response contained one or more file/line citations that could not be validated locally.";
pub(in crate::ai::driver::turn_runtime) const FINAL_CITATION_WARNING: &str = "[Runtime warning] One or more file/line citations in this answer could not be validated locally; treat the cited details as unverified.";
pub(in crate::ai::driver::turn_runtime) const MAX_FINAL_RESPONSE_CITATIONS: usize = 64;
pub(in crate::ai::driver::turn_runtime) const MAX_FINAL_CITATION_FILE_BYTES: u64 = 16 * 1024 * 1024;
pub(in crate::ai::driver::turn_runtime) const MAX_FINAL_CITATION_LINE_SCAN: u64 = 1_000_000;

/// This recognizes only conventional, file-looking `path:line` references. A final-response
/// gate must prefer false negatives over false positives: prose such as `phase: 2` must never
/// force the model to repeat an otherwise valid answer.
pub(in crate::ai::driver::turn_runtime) static PATH_LINE_CITATION_RE: LazyLock<Regex> =
    LazyLock::new(|| {
        Regex::new(
            r"(?x)
        (?P<path>
            (?:/|\./|\.\./|~/)?
            [A-Za-z0-9_.@%+=,-]+
            (?:/[A-Za-z0-9_.@%+=,-]+)*
        )
        :
        (?P<start>[1-9][0-9]*)
        (?:-(?P<end>[1-9][0-9]*))?
        (?::[0-9]+)?
        ",
        )
        .expect("path:line citation regular expression must compile")
    });

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum FinalCitationGateAction {
    Allow,
    Reopen,
    Warn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) struct FinalCitation {
    pub(super) text: String,
    path: String,
    start_line: u64,
    end_line: u64,
}

/// Byte ranges of fenced code blocks (``` or ~~~) in `text`, used by the citation
/// scanner to skip example/diff code: paths mentioned inside a fence are
/// illustrative, not evidence-bearing citations, and flagging them would attach a
/// false warning to an otherwise correct answer. A fence opens on a line whose
/// non-whitespace content starts with 3+ backticks or tildes, and closes on a
/// line whose non-whitespace content consists only of the same marker repeated
/// at least as many times; an unclosed fence covers the rest of the text, which
/// errs toward skipping.
/// Inline code spans are intentionally NOT skipped — real citations are usually
/// written as `src/lib.rs:42` in prose, so skipping them would lose true positives.
pub(in crate::ai::driver::turn_runtime) fn fenced_code_block_byte_ranges(
    text: &str,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    // (marker char, minimum closing marker count, range start byte)
    let mut open_fence: Option<(char, usize, usize)> = None;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if let Some((marker, open_count, start)) = open_fence {
            let marker_count = trimmed.chars().filter(|c| *c == marker).count();
            let closes_fence = marker_count >= open_count && trimmed.chars().all(|c| c == marker);
            if closes_fence {
                ranges.push((start, offset + line.len()));
                open_fence = None;
            }
        } else {
            for (marker, prefix) in [('`', "```"), ('~', "~~~")] {
                if trimmed.starts_with(prefix) {
                    let open_count = trimmed.chars().take_while(|c| *c == marker).count();
                    open_fence = Some((marker, open_count, offset));
                    break;
                }
            }
        }
        offset += line.len();
    }
    if let Some((_, _, start)) = open_fence {
        ranges.push((start, text.len()));
    }
    ranges
}

/// How far around a citation token the non-assertive context is inspected, and how close an
/// immediately preceding negation must sit to qualify that token. The wide window catches
/// "the file does not exist ... `src/x.rs:3`"; the short one catches "There is no `src/x.rs:3` to
/// cite" without swallowing a citation that a negation in an earlier clause is not about
/// ("There is no test for this, but see src/lib.rs:12").
const NON_ASSERTIVE_CITATION_WINDOW_BEFORE_CHARS: usize = 160;
const NON_ASSERTIVE_CITATION_WINDOW_AFTER_CHARS: usize = 80;
const NON_ASSERTIVE_CITATION_ADJACENT_CHARS: usize = 20;

/// Phrases that turn a nearby `path:line` token into something other than an evidence claim: an
/// existence check ("the file does not exist", `read_file src/x.rs` -> "File not found"), or a
/// question back to the user ("if you meant ..."). A refusal necessarily names the path it declines
/// to cite, so without this exemption the gate reopens and then warns on an answer that is already
/// correct. Chinese variants are listed because the refusal prose may be translated while the path
/// stays Latin.
const NON_ASSERTIVE_CITATION_PHRASES: &[&str] = &[
    "does not exist",
    "doesn't exist",
    "no such file",
    "no file",
    "not found",
    "no matches",
    "no such path",
    "isn't a file",
    "if you meant",
    "meant a different",
    "tell me the intended",
    "did you mean",
    "不存在",
    "找不到",
    "无法找到",
    "没有这个文件",
];

/// Negations that count only immediately before the token, because the same words further away
/// usually qualify a different path.
const NON_ASSERTIVE_ADJACENT_PHRASES: &[&str] = &[
    "there is no",
    "there's no",
    "no such",
    "won't invent",
    "will not invent",
    "不存在",
];

fn char_window_before(text: &str, start: usize, max_chars: usize) -> &str {
    let mut boundary = start;
    for _ in 0..max_chars {
        match text[..boundary].chars().next_back() {
            Some(character) => boundary -= character.len_utf8(),
            None => break,
        }
    }
    &text[boundary..start]
}

fn char_window_after(text: &str, end: usize, max_chars: usize) -> &str {
    match text[end..].char_indices().nth(max_chars) {
        Some((offset, _)) => &text[end..end + offset],
        None => &text[end..],
    }
}

/// Whether the citation token at `start..end` is named only to deny or to question it.
fn citation_in_non_assertive_context(text: &str, start: usize, end: usize) -> bool {
    let window = format!(
        "{} {}",
        char_window_before(text, start, NON_ASSERTIVE_CITATION_WINDOW_BEFORE_CHARS),
        char_window_after(text, end, NON_ASSERTIVE_CITATION_WINDOW_AFTER_CHARS)
    )
    .to_lowercase();
    if NON_ASSERTIVE_CITATION_PHRASES
        .iter()
        .any(|phrase| window.contains(phrase))
    {
        return true;
    }
    let adjacent =
        char_window_before(text, start, NON_ASSERTIVE_CITATION_ADJACENT_CHARS).to_lowercase();
    NON_ASSERTIVE_ADJACENT_PHRASES
        .iter()
        .any(|phrase| adjacent.contains(phrase))
}

pub(in crate::ai::driver::turn_runtime) fn final_response_citations(
    final_text: &str,
) -> Vec<FinalCitation> {
    let mut citations = Vec::new();
    let fenced_ranges = fenced_code_block_byte_ranges(final_text);
    for captures in PATH_LINE_CITATION_RE.captures_iter(final_text) {
        let (Some(full), Some(path), Some(start)) = (
            captures.get(0),
            captures.name("path"),
            captures.name("start"),
        ) else {
            continue;
        };
        if fenced_ranges
            .iter()
            .any(|(start_byte, end_byte)| full.start() >= *start_byte && full.start() < *end_byte)
        {
            continue;
        }
        if citations.len() == MAX_FINAL_RESPONSE_CITATIONS {
            break;
        }
        if !citation_has_token_boundaries(final_text, full.start(), full.end())
            || !looks_like_final_citation_path(path.as_str())
        {
            continue;
        }
        // Only asserted citations are validated: a path named to say it is missing, or to ask which
        // file was meant, carries no evidence claim to check.
        if citation_in_non_assertive_context(final_text, full.start(), full.end()) {
            continue;
        }
        let Ok(start_line) = start.as_str().parse::<u64>() else {
            continue;
        };
        let end_line = match captures.name("end") {
            Some(end) => match end.as_str().parse::<u64>() {
                Ok(line) => line,
                Err(_) => continue,
            },
            None => start_line,
        };
        let citation = FinalCitation {
            text: full.as_str().to_string(),
            path: path.as_str().to_string(),
            start_line,
            end_line,
        };
        if !citations.iter().any(|existing| existing == &citation) {
            citations.push(citation);
        }
    }
    citations
}

pub(in crate::ai::driver::turn_runtime) fn citation_has_token_boundaries(
    text: &str,
    start: usize,
    end: usize,
) -> bool {
    let preceding = text[..start].chars().next_back();
    let following = text[end..].chars().next();
    !preceding.is_some_and(is_citation_path_character)
        && !following.is_some_and(|character| {
            character.is_ascii_alphanumeric()
                || matches!(character, '_' | '/' | ':' | '-' | '@' | '%' | '+' | '=')
        })
}

pub(in crate::ai::driver::turn_runtime) fn is_citation_path_character(character: char) -> bool {
    character.is_ascii_alphanumeric()
        || matches!(
            character,
            '_' | '.' | '-' | '/' | '@' | '%' | '+' | '=' | ',' | ':'
        )
}

/// Extensions that appear in prose mainly as version/phase qualifiers rather than
/// real file extensions (e.g. `phase.alpha:2`, `build.release:3`). Treating them
/// as citation paths would probe phantom files like `phase.alpha` and attach a
/// false warning; real source/config extensions practically never collide with
/// these. This only narrows detection — the gate still prefers false negatives
/// over false positives, so tokens with other unknown extensions stay candidates.
pub(in crate::ai::driver::turn_runtime) const PROSE_QUALIFIER_EXTENSIONS: &[&str] = &[
    "alpha", "beta", "rc", "dev", "debug", "release", "final", "snapshot", "nightly", "canary",
    "preview", "draft", "wip", "test", "prod", "stage", "staging",
];

pub(in crate::ai::driver::turn_runtime) fn looks_like_final_citation_path(path: &str) -> bool {
    let file_name = path.rsplit('/').next().unwrap_or(path);
    if matches!(
        file_name,
        "Makefile" | "Dockerfile" | "LICENSE" | "README" | "AGENTS"
    ) {
        return true;
    }
    let Some((_, extension)) = file_name.rsplit_once('.') else {
        return false;
    };
    extension
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic())
        && extension
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
        && !PROSE_QUALIFIER_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str())
}

pub(in crate::ai::driver::turn_runtime) fn resolve_final_citation_path(
    path: &str,
    effective_cwd: Option<&Path>,
    home: Option<&std::ffi::OsStr>,
) -> Option<PathBuf> {
    if let Some(home_relative_path) = path.strip_prefix("~/") {
        return home.map(|home| PathBuf::from(home).join(home_relative_path));
    }
    let path = Path::new(path);
    if path.is_absolute() {
        Some(path.to_path_buf())
    } else {
        effective_cwd.map(|cwd| cwd.join(path))
    }
}

fn push_citation_base_dir(base_dirs: &mut Vec<PathBuf>, path: PathBuf) {
    if !base_dirs.contains(&path) {
        base_dirs.push(path);
    }
}

/// Extract the directory from a simple leading `cd <path> && ...` shell prefix.
/// This deliberately accepts only an unquoted, shell-metacharacter-free operand:
/// unsupported shell syntax remains unknown rather than adding an invented
/// citation root. A relative operand is resolved from the command's explicit cwd
/// when present, otherwise from the turn's effective cwd.
fn inline_execute_command_cd_dir(command: &str, base: Option<&Path>) -> Option<PathBuf> {
    let command = command.trim_start_matches(|character| matches!(character, ' ' | '\t' | '\n'));
    let after_cd = command.strip_prefix("cd")?;
    if !matches!(after_cd.as_bytes().first(), Some(b' ' | b'\t')) {
        return None;
    }
    let after_cd = after_cd.trim_start_matches(|character| matches!(character, ' ' | '\t'));
    let (operand, trailing_command) = after_cd.split_once("&&")?;
    if trailing_command.trim().is_empty() {
        return None;
    }
    let operand = operand.trim_matches(|character| matches!(character, ' ' | '\t'));
    if operand.is_empty()
        || operand.chars().any(|character| {
            character.is_whitespace()
                || matches!(
                    character,
                    '\'' | '"'
                        | '\\'
                        | '$'
                        | '`'
                        | '~'
                        | '#'
                        | '('
                        | ')'
                        | '{'
                        | '}'
                        | '['
                        | ']'
                        | '*'
                        | '?'
                        | ';'
                        | '|'
                        | '&'
                        | '<'
                        | '>'
                )
        })
    {
        return None;
    }
    let path = Path::new(operand);
    if path.is_absolute() {
        Some(path.to_path_buf())
    } else {
        base.map(|base| base.join(path))
    }
}

/// Build conservative resolution roots from the current turn's observed tool paths. Models
/// often cite a basename after reading an absolute file path or running a command from a
/// subdirectory; validating only against the process cwd incorrectly rejects those citations.
/// No recursive search is performed: every extra root must come from an explicit tool path,
/// cwd, or a conservatively parsed leading `cd <path> && ...` command prefix.
pub(in crate::ai::driver::turn_runtime) fn final_citation_base_dirs(
    messages: &[Message],
    effective_cwd: Option<&Path>,
) -> Vec<PathBuf> {
    let mut base_dirs = Vec::new();
    if let Some(cwd) = effective_cwd {
        push_citation_base_dir(&mut base_dirs, cwd.to_path_buf());
    }
    let turn_start = crate::ai::history::last_real_user_index(messages).unwrap_or(0);
    for message in messages.iter().skip(turn_start) {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            let Ok(args) = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
            else {
                continue;
            };
            let tool_cwd = args
                .get("cwd")
                .and_then(serde_json::Value::as_str)
                .filter(|cwd| !cwd.trim().is_empty())
                .map(PathBuf::from)
                .map(|cwd| {
                    if cwd.is_absolute() {
                        cwd
                    } else {
                        effective_cwd.map_or(cwd.clone(), |base| base.join(cwd))
                    }
                });
            if let Some(cwd) = &tool_cwd {
                push_citation_base_dir(&mut base_dirs, cwd.clone());
            }
            if tool_call.function.name == "execute_command" {
                let command_cwd = args
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|command| {
                        inline_execute_command_cd_dir(
                            command,
                            tool_cwd.as_deref().or(effective_cwd),
                        )
                    });
                if let Some(command_cwd) = command_cwd {
                    push_citation_base_dir(&mut base_dirs, command_cwd);
                }
            }
            for raw_path in ["file_path", "path"]
                .into_iter()
                .filter_map(|key| args.get(key).and_then(serde_json::Value::as_str))
            {
                let path = Path::new(raw_path);
                let resolved = if path.is_absolute() {
                    path.to_path_buf()
                } else if let Some(cwd) = &tool_cwd {
                    cwd.join(path)
                } else if let Some(cwd) = effective_cwd {
                    cwd.join(path)
                } else {
                    continue;
                };
                if let Some(parent) = resolved.parent() {
                    push_citation_base_dir(&mut base_dirs, parent.to_path_buf());
                }
                push_citation_base_dir(&mut base_dirs, resolved);
            }
        }
    }
    base_dirs
}

/// `Some(false)` is reserved for a locally provable bad citation. I/O failures and oversized
/// files stay unknown so this gate never claims a citation is invalid without direct evidence.
pub(in crate::ai::driver::turn_runtime) fn citation_file_contains_line(
    path: &Path,
    line: u64,
) -> Option<bool> {
    if line > MAX_FINAL_CITATION_LINE_SCAN {
        // Cheap falsification before giving up: a file of S bytes has at most S
        // lines (every line needs at least one byte), so a line number beyond
        // size + 1 is provably past EOF even above the scan cap. Anything else
        // stays unknown here; only the bounded scan below can verify smaller
        // line numbers.
        return match std::fs::metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Some(false),
            Ok(metadata) if line > metadata.len().saturating_add(1) => Some(false),
            _ => None,
        };
    }
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Some(false),
        Err(_) => return None,
    };
    if !metadata.is_file() {
        return Some(false);
    }
    if metadata.len() > MAX_FINAL_CITATION_FILE_BYTES {
        return None;
    }
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return None,
    };
    let mut reader = BufReader::new(file);
    let mut buffer = String::new();
    for _ in 0..line {
        buffer.clear();
        match reader.read_line(&mut buffer) {
            Ok(0) => return Some(false),
            Ok(_) => {}
            Err(_) => return None,
        }
    }
    Some(true)
}

pub(in crate::ai::driver::turn_runtime) fn unvalidated_final_response_citations(
    final_text: &str,
    effective_cwd: Option<&Path>,
) -> Vec<String> {
    let base_dirs = effective_cwd
        .into_iter()
        .map(Path::to_path_buf)
        .collect::<Vec<_>>();
    unvalidated_final_response_citations_with_bases(final_text, &base_dirs)
}

pub(in crate::ai::driver::turn_runtime) fn unvalidated_final_response_citations_with_bases(
    final_text: &str,
    base_dirs: &[PathBuf],
) -> Vec<String> {
    let home = std::env::var_os("HOME");
    final_response_citations(final_text)
        .into_iter()
        .filter_map(|citation| {
            if citation.end_line < citation.start_line {
                return Some(citation.text);
            }
            // Resolution failure (no cwd / no HOME) means "cannot validate", not
            // "valid": skip without flagging, exactly like the other unknown
            // verdicts. Only provably bad citations may trigger the retry/warning
            // path.
            if citation.path.starts_with("~/") || Path::new(&citation.path).is_absolute() {
                let path = resolve_final_citation_path(&citation.path, None, home.as_deref())?;
                return match citation_file_contains_line(&path, citation.end_line) {
                    Some(true) | None => None,
                    Some(false) => Some(citation.text),
                };
            }
            let verdicts = base_dirs
                .iter()
                .map(|base| {
                    citation_file_contains_line(&base.join(&citation.path), citation.end_line)
                })
                .collect::<Vec<_>>();
            if verdicts.iter().any(|verdict| matches!(verdict, Some(true)))
                || verdicts.iter().all(Option::is_none)
            {
                None
            } else {
                Some(citation.text)
            }
        })
        .collect()
}

pub(in crate::ai::driver::turn_runtime) fn final_response_citation_gate_action(
    messages: &mut Vec<Message>,
    final_text: &str,
    effective_cwd: Option<&Path>,
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> FinalCitationGateAction {
    let base_dirs = final_citation_base_dirs(messages, effective_cwd);
    let unvalidated = unvalidated_final_response_citations_with_bases(final_text, &base_dirs);
    if unvalidated.is_empty() {
        return FinalCitationGateAction::Allow;
    }
    let already_retried = current_turn_has_internal_marker(messages, FINAL_CITATION_RETRY_MARKER);
    if already_retried || force_final_response || iteration >= max_iterations {
        return FinalCitationGateAction::Warn;
    }

    let listed = unvalidated
        .iter()
        .take(8)
        .map(|citation| format!("`{citation}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let omitted = unvalidated.len().saturating_sub(8);
    let suffix = (omitted > 0).then(|| format!(" and {omitted} more"));
    let note = format!(
        "{FINAL_CITATION_RETRY_MARKER}\n\
         The draft final response contains file/line citations that could not be validated locally: {listed}{}.\n\
         This is not a final answer. Recheck the cited paths and line numbers using existing evidence or focused reads, then give a corrected answer.\n\
         Do not retain, invent, or replace a citation unless the path and line are supported by observed evidence.",
        suffix.as_deref().unwrap_or_default(),
    );
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    FinalCitationGateAction::Reopen
}
