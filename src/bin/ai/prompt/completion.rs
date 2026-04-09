use rustyline::{
    Context, Editor, Helper,
    completion::{Completer, Pair},
    highlight::Highlighter,
    hint::Hinter,
    history::DefaultHistory,
    validate::Validator,
};

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

    pub(super) fn complete_for_line(line: &str, pos: usize) -> (usize, Vec<String>) {
        let pos = pos.min(line.len());
        let before = &line[..pos];
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
}
