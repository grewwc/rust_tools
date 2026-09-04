use std::{
    cmp::Ordering,
    fs,
    path::PathBuf,
    sync::{LazyLock, RwLock},
};

use rustyline::{
    Context, Editor, Helper,
    completion::{Completer, Pair},
    highlight::Highlighter,
    hint::Hinter,
    history::DefaultHistory,
    validate::Validator,
};

use crate::{commonw::utils::expanduser, cw::Trie};

pub(super) type LineEditor = Editor<CommandCompleter, DefaultHistory>;

static CURRENT_MODEL_HINT: LazyLock<RwLock<String>> = LazyLock::new(|| RwLock::new(String::new()));
/// Written by the driver after loading manifests. Completion sits on the hot path and must not rescan the disk on every Tab.
/// `None` means not yet initialized, kept as the compatibility fallback for unit tests and non-interactive calls.
static SKILL_NAME_CANDIDATES: LazyLock<RwLock<Option<Vec<CompletionCandidate>>>> =
    LazyLock::new(|| RwLock::new(None));
/// Same semantics as SKILL_NAME_CANDIDATES: `None` means the driver has not populated it yet,
/// and agent-name completion then falls back synchronously to a disk scan (otherwise Tab before the first input of a new session has no candidates).
static AGENT_NAME_CANDIDATES: LazyLock<RwLock<Option<Vec<CompletionCandidate>>>> =
    LazyLock::new(|| RwLock::new(None));

/// Trie holding all top-level commands starting with "/" and ":", replacing the previous linear starts_with filtering.
static COMMANDS_TRIE: LazyLock<Trie> = LazyLock::new(|| {
    let mut trie = Trie::new();
    for &cmd in &[
        "/help",
        ":help",
        "/h",
        ":h",
        "/history",
        ":history",
        "/usage",
        ":usage",
        "/feishu-auth",
        ":feishu-auth",
        "/share",
        ":share",
        "/checkpoint",
        ":checkpoint",
        "/cp",
        ":cp",
        "/model",
        "/memo",
        ":memo",
        "/export",
        ":export",
        ":model",
        "/effort",
        ":effort",
        "/audit",
        ":audit",
        "/changes",
        ":changes",
        "/diff",
        ":diff",
        "/agent",
        ":agent",
        "/personas",
        ":personas",
        "/sessions",
        ":sessions",
        "/ss",
        ":ss",
        "/close",
        ":close",
        "/fork",
        ":fork",
        "/proc",
        ":proc",
        "/skills",
        ":skills",
        "/mark",
        ":mark",
        "/unmark",
        ":unmark",
    ] {
        trie.insert(cmd);
    }
    trie
});

/// Trie holding all CLI options starting with "--" and "-" (short forms included) to support option completion.
static FLAGS_TRIE: LazyLock<Trie> = LazyLock::new(|| {
    let mut trie = Trie::new();
    for flag in &[
        // bool options
        "--clear",
        "--new-session",
        "--new",
        "--resume",
        "-r",
        "--list-tools",
        "--list-mcp-tools",
        "--list-skills",
        "--list-agents",
        "--no-skills",
        "--help",
        "-h",
        "--interactive",
        "-i",
        "--consolidate-knowledge",
        "--note-search",
        "-ns",
        "--generate-completions",
        // string/int options
        "--model",
        "-m",
        "--agent",
        "-a",
        "--session",
        "-ss",
        "--files",
        "-f",
        "--mcp-config",
        "--reasoning-effort",
        "-re",
        "--note",
        "-n",
        "--note-delete",
        "-nd",
        "--note-edit",
        "-ne",
    ] {
        trie.insert(flag);
    }
    trie
});

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::ai) struct CompletionCandidate {
    pub(in crate::ai) display: String,
    pub(in crate::ai) replacement: String,
}

#[derive(Clone, Default)]
pub(in crate::ai) struct CommandCompleter;

impl CommandCompleter {
    pub(in crate::ai) fn set_current_model_hint(model: &str) {
        if let Ok(mut guard) = CURRENT_MODEL_HINT.write() {
            *guard = model.trim().to_string();
        }
    }

    pub(crate) fn current_model_hint() -> Option<String> {
        CURRENT_MODEL_HINT
            .read()
            .ok()
            .map(|guard| guard.trim().to_string())
            .filter(|model| !model.is_empty())
    }

    /// Update the skill completion cache from the runtime manifest snapshot.
    pub(in crate::ai) fn set_skill_manifests(manifests: &[crate::ai::skills::SkillManifest]) {
        if let Ok(mut guard) = SKILL_NAME_CANDIDATES.write() {
            *guard = Some(Self::skill_candidates_from_manifests(manifests));
        }
    }

    /// Update the switchable-agent completion cache from the runtime manifest snapshot.
    pub(in crate::ai) fn set_agent_manifests(manifests: &[crate::ai::agents::AgentManifest]) {
        if let Ok(mut guard) = AGENT_NAME_CANDIDATES.write() {
            *guard = Some(Self::agent_candidates_from_manifests(manifests));
        }
    }

    fn top_level_commands() -> &'static [&'static str] {
        &[
            "/help",
            ":help",
            "/h",
            ":h",
            "/history",
            ":history",
            "/usage",
            ":usage",
            "/feishu-auth",
            ":feishu-auth",
            "/share",
            ":share",
            "/checkpoint",
            ":checkpoint",
            "/cp",
            ":cp",
            "/model",
            "/memo",
            ":memo",
            "/export",
            ":export",
            ":model",
            "/effort",
            ":effort",
            "/audit",
            ":audit",
            "/changes",
            ":changes",
            "/diff",
            ":diff",
            "/agent",
            ":agent",
            "/personas",
            ":personas",
            "/sessions",
            ":sessions",
            "/ss",
            ":ss",
            "/close",
            ":close",
            "/fork",
            ":fork",
            "/proc",
            ":proc",
            "/skills",
            ":skills",
            "/mark",
            ":mark",
            "/unmark",
            ":unmark",
        ]
    }

    fn is_model_command(token: &str) -> bool {
        matches!(token, "/model" | ":model")
    }

    fn plain_candidates(values: impl IntoIterator<Item = String>) -> Vec<CompletionCandidate> {
        values
            .into_iter()
            .map(|value| CompletionCandidate {
                display: value.clone(),
                replacement: value,
            })
            .collect()
    }

    fn model_handle(model: &crate::ai::model_names::ModelDef) -> String {
        crate::ai::model_names::model_handle(model)
    }

    fn model_replacement(model: &crate::ai::model_names::ModelDef) -> String {
        // replacement uses model.key so find_by_identifier can resolve it correctly by key.
        // When key is empty, fall back to handle (compatible with older models).
        let key = model.key.trim();
        if key.is_empty() {
            Self::model_handle(model)
        } else {
            key.to_string()
        }
    }

    fn current_model_matches(
        current: Option<&str>,
        model: &crate::ai::model_names::ModelDef,
    ) -> bool {
        let Some(current) = current else {
            return false;
        };
        let handle = Self::model_handle(model);
        crate::ai::model_names::find_by_identifier(current)
            .map(|def| Self::model_handle(def).eq_ignore_ascii_case(&handle))
            .unwrap_or_else(|| {
                current.eq_ignore_ascii_case(&model.name) || current.eq_ignore_ascii_case(&handle)
            })
    }

    fn ordered_model_names() -> Vec<String> {
        let current = Self::current_model_hint();
        let mut current_first = Vec::new();
        let mut rest = Vec::new();
        for model in crate::ai::model_names::all() {
            let replacement = Self::model_replacement(model);
            if Self::current_model_matches(current.as_deref(), model) {
                current_first.push(replacement);
            } else {
                rest.push(replacement);
            }
        }
        current_first.extend(rest);
        current_first
    }

    fn model_candidate_detail(model: &crate::ai::model_names::ModelDef) -> String {
        let mut flags = Vec::new();
        if model.enable_thinking {
            flags.push("thinking");
        }
        if model.tools_default_enabled {
            flags.push("tools");
        }
        if model.is_vl {
            flags.push("vl");
        }
        let flags = if flags.is_empty() {
            "plain".to_string()
        } else {
            flags.join("/")
        };
        let handle = Self::model_handle(model);
        format!(
            "{} · {}/{} · {}",
            handle,
            crate::ai::model_names::platform_label(model),
            crate::ai::model_names::adapter_slug(model.adapter),
            flags
        )
    }

    fn model_command_candidates(prefix: &str) -> Vec<CompletionCandidate> {
        let current = Self::current_model_hint();
        let mut candidates = Vec::new();
        for model in crate::ai::model_names::all() {
            let replacement = Self::model_replacement(model);
            let display = if Self::current_model_matches(current.as_deref(), model) {
                format!("{} · current", Self::model_candidate_detail(model))
            } else {
                Self::model_candidate_detail(model)
            };
            candidates.push(CompletionCandidate {
                display,
                replacement: format!("{prefix} {}", replacement),
            });
        }
        candidates
    }

    fn model_name_candidates() -> Vec<CompletionCandidate> {
        let current = Self::current_model_hint();
        let mut candidates = Vec::new();
        for replacement in Self::ordered_model_names() {
            let model = crate::ai::model_names::find_by_identifier(&replacement)
                .expect("ordered model handle must exist");
            let display = if Self::current_model_matches(current.as_deref(), model) {
                format!("{} · current", Self::model_candidate_detail(model))
            } else {
                Self::model_candidate_detail(model)
            };
            candidates.push(CompletionCandidate {
                display,
                replacement,
            });
        }
        candidates
    }

    /// Model-name completion match ranks:
    /// - 0: prefix match on replacement (original behavior, case-insensitive)
    /// - 1: two-segment "name + platform" match, e.g. `deep-v` → `deepseek-v4-flash-volcano`
    /// - 2: per-segment prefix match, e.g. `deep-v` → all `deepseek-v4-*`
    /// Returns None when nothing matches.
    fn model_token_match_rank(token: &str, model: &crate::ai::model_names::ModelDef) -> Option<u8> {
        let token = token.trim().to_ascii_lowercase();
        // With an empty token (e.g. Tab right after `/model `), an empty prefix matches all models (original behavior).
        let replacement = Self::model_replacement(model).to_ascii_lowercase();
        if replacement.starts_with(&token) {
            return Some(0);
        }
        // Two-segment: treat the last segment as the platform prefix, e.g. `deep-v` → `deep` + platform `v`.
        if let Some((head, tail)) = token.rsplit_once(['-', '.', '_', '/', ':', ' ']) {
            if !head.is_empty() && !tail.is_empty() {
                let key = model.key.to_ascii_lowercase();
                let name = model.name.to_ascii_lowercase();
                let platform = crate::ai::model_names::platform_slug(model).to_ascii_lowercase();
                let head_matches = key.starts_with(&head) || name.starts_with(&head);
                if head_matches && platform.starts_with(tail) {
                    return Some(1);
                }
            }
        }
        // Per-segment prefix matching: each query segment prefix-hits candidate segments in order (skipping segments allowed).
        if Self::segments_prefix_match(
            &token,
            &Self::model_searchable_text(model).to_ascii_lowercase(),
        ) {
            return Some(2);
        }
        None
    }

    /// Search text used for per-segment matching: key + name + platform + aliases.
    fn model_searchable_text(model: &crate::ai::model_names::ModelDef) -> String {
        let mut text = format!("{} {}", model.key, model.name);
        let platform = crate::ai::model_names::platform_slug(model);
        if !platform.is_empty() {
            text.push(' ');
            text.push_str(&platform);
        }
        for alias in &model.aliases {
            text.push(' ');
            text.push_str(alias);
        }
        text
    }

    /// Split query and candidate on `- . _ / : whitespace`, requiring every query segment
    /// to prefix-hit some candidate segment in order (skipping segments allowed).
    fn segments_prefix_match(query: &str, candidate: &str) -> bool {
        let q_segments: Vec<&str> = query
            .split(['-', '.', '_', '/', ':', ' '])
            .filter(|seg| !seg.is_empty())
            .collect();
        if q_segments.is_empty() {
            return false;
        }
        let c_segments: Vec<&str> = candidate
            .split(['-', '.', '_', '/', ':', ' '])
            .filter(|seg| !seg.is_empty())
            .collect();
        let mut qi = 0;
        for seg in &c_segments {
            if seg.starts_with(q_segments[qi]) {
                qi += 1;
                if qi == q_segments.len() {
                    return true;
                }
            }
        }
        false
    }

    /// Generic name completion match ranks (agent / skill etc.):
    /// - 0: prefix match on replacement (original behavior, case-insensitive)
    /// - 1: per-segment prefix match on replacement, e.g. `fast` → `audit-fast`, `own` → `audit_own_changes`
    /// Only the name itself participates: descriptions are display-only and never matched (preventing `audit_o`
    /// from falsely hitting unrelated skills whose descriptions contain words like "audits of").
    /// Returns None when nothing matches.
    fn name_token_match_rank(token: &str, replacement: &str) -> Option<u8> {
        let token = token.trim().to_ascii_lowercase();
        if replacement.to_ascii_lowercase().starts_with(&token) {
            return Some(0);
        }
        if Self::segments_prefix_match(&token, &replacement.to_ascii_lowercase()) {
            return Some(1);
        }
        None
    }

    fn agent_subcommands() -> &'static [&'static str] {
        &["help", "list", "current", "use", "auto", "reload"]
    }

    fn agent_candidates_from_manifests(
        manifests: &[crate::ai::agents::AgentManifest],
    ) -> Vec<CompletionCandidate> {
        crate::ai::agents::get_primary_agents(manifests)
            .into_iter()
            .map(|agent| CompletionCandidate {
                display: format!("{} · {}", agent.name, agent.description),
                replacement: agent.name.clone(),
            })
            .collect()
    }

    /// Name candidates of all loaded agents (with display).
    fn agent_name_candidates(token: &str) -> Vec<CompletionCandidate> {
        let candidates = if let Ok(guard) = AGENT_NAME_CANDIDATES.read()
            && let Some(candidates) = guard.as_ref()
        {
            candidates.clone()
        } else {
            // When the cache is not yet populated (before the first input is submitted / non-interactive calls / unit tests),
            // fall back synchronously to a disk scan, mirroring the skill completion fallback when SKILL_NAME_CANDIDATES=None.
            Self::agent_candidates_from_manifests(&crate::ai::agents::load_all_agents())
        };
        // Smart matching: prefix (rank 0) > per-segment prefix (rank 1, e.g. `fast` → `audit-fast`),
        // consistent with model completion. Only names match; descriptions are display-only.
        let mut matched: Vec<(u8, CompletionCandidate)> = candidates
            .into_iter()
            .filter_map(|candidate| {
                Self::name_token_match_rank(token, &candidate.replacement)
                    .map(|rank| (rank, candidate))
            })
            .collect();
        matched.sort_by_key(|(rank, _)| *rank);
        matched
            .into_iter()
            .map(|(_, candidate)| candidate)
            .collect()
    }

    /// Second-level subcommand literals of `/model`. Note: these subcommands and "model names" occupy
    /// the second token mutually exclusively; so Tab can complete both subcommands and model names,
    /// `complete_for_line` merges them into one candidate list (prefix-filtered).
    fn model_subcommands() -> &'static [&'static str] {
        &["current", "list", "help", "effort"]
    }

    /// Third-token candidates for `/model effort`: reasoning-effort levels + auto/off.
    fn model_effort_levels() -> &'static [&'static str] {
        &[
            "minimal", "low", "medium", "high", "xhigh", "max", "auto", "off",
        ]
    }

    fn session_subcommands() -> &'static [&'static str] {
        crate::ai::driver::commands::session::CANONICAL_SESSION_SUBCOMMANDS
    }

    fn persona_subcommands() -> &'static [&'static str] {
        &["list", "current", "create", "new", "use", "delete", "help"]
    }

    /// Subcommands of `/usage`.
    fn usage_subcommands() -> &'static [&'static str] {
        &[
            "today", "7d", "30d", "all", "models", "daily", "trend", "days", "help",
        ]
    }

    /// Subcommands of `/checkpoint` / `/cp`.
    fn checkpoint_subcommands() -> &'static [&'static str] {
        &["save", "list", "rollback", "delete", "help"]
    }

    /// Subcommands of `/changes` / `/diff` (-- long options are primary, matching the command docs;
    /// the bare-word forms `stat`/`json`/`patch`/`open`/`help` are accepted by the parser too).
    fn changes_subcommands() -> &'static [&'static str] {
        &["--help", "--stat", "--json", "--patch", "--open"]
    }

    /// Editor candidates for `/changes --open [editor]` (same aliases `changes::EditorKind::from_str`
    /// accepts, using the canonical names).
    fn changes_open_editors() -> &'static [&'static str] {
        &["auto", "code", "vscode", "cursor", "idea", "git", "open"]
    }

    fn history_subcommands() -> &'static [&'static str] {
        &[
            "full",
            "user",
            "assistant",
            "tool",
            "system",
            "grep",
            "rewind",
            "export",
            "copy",
            "last",
            "replay",
            "help",
            "3",
            "6",
            "10",
            "20",
        ]
    }

    /// Subcommand literals of `/skills` / `/skill`.
    fn skills_subcommands() -> &'static [&'static str] {
        &["list", "current", "use", "help"]
    }

    fn skill_candidates_from_manifests(
        manifests: &[crate::ai::skills::SkillManifest],
    ) -> Vec<CompletionCandidate> {
        manifests
            .iter()
            .map(|skill| {
                let display = if skill.description.trim().is_empty() {
                    skill.name.clone()
                } else {
                    format!("{} · {}", skill.name, skill.description.trim())
                };
                CompletionCandidate {
                    display,
                    replacement: skill.name.clone(),
                }
            })
            .collect()
    }

    /// Name candidates of all loaded skills (with display).
    fn skill_name_candidates() -> Vec<CompletionCandidate> {
        if let Ok(guard) = SKILL_NAME_CANDIDATES.read()
            && let Some(candidates) = guard.as_ref()
        {
            return candidates.clone();
        }

        // Non-interactive calls and unit tests without a runtime snapshot keep the original behavior.
        Self::skill_candidates_from_manifests(&crate::ai::skills::load_all_skills())
    }

    pub(super) fn complete_for_line(line: &str, pos: usize) -> (usize, Vec<CompletionCandidate>) {
        // `pos` is a byte-offset cursor position that may land inside a multi-byte UTF-8 character (e.g. Chinese),
        // where a direct `&line[..pos]` slice panics. Align down to the nearest character boundary.
        let mut pos = pos.min(line.len());
        while pos > 0 && !line.is_char_boundary(pos) {
            pos -= 1;
        }
        let before = &line[..pos];
        // `@skills` / `@skill[:prefix]` trigger skill completion and must be handled before plain `@file` completion,
        // otherwise `complete_file_reference` would treat `@skills` as a file-path fragment.
        if let Some((token_start, candidates)) = complete_skill_reference(before) {
            return (token_start, candidates);
        }
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

        if token_start == 0 && Self::is_model_command(token) {
            return (0, Self::model_command_candidates(token));
        }

        let candidates = if token_start == 0 {
            // Prefix-match with Tries: "/" / ":" go through the command Trie, "--" / "-" through the option Trie;
            // sort the results for determinism (HashMap iteration order is unordered).
            if token.starts_with('/') || token.starts_with(':') {
                let mut words = COMMANDS_TRIE.words_with_prefix(token);
                words.sort();
                Self::plain_candidates(words)
            } else if token.starts_with('-') {
                let mut words = FLAGS_TRIE.words_with_prefix(token);
                words.sort();
                Self::plain_candidates(words)
            } else {
                Vec::new()
            }
        } else {
            let mut words = before[..token_start].split_whitespace();
            let Some(first) = words.next() else {
                return (token_start, Vec::new());
            };
            if Self::is_model_command(first) {
                // Deeper completions exist beyond the second token (currently only `/model effort <level>`).
                // Here we check "how many non-empty tokens already follow `/model`".
                let second = words.next();
                match second {
                    None => {
                        // Second token: model names (with current pinned on top) + `/model` subcommand literals.
                        // Model names use smart matching (prefix > name+platform two-segment > per-segment prefix),
                        // so `deep-v` also completes to `deepseek-v4-flash-volcano`.
                        let mut matched: Vec<(u8, CompletionCandidate)> =
                            Self::model_name_candidates()
                                .into_iter()
                                .filter_map(|candidate| {
                                    let model = crate::ai::model_names::find_by_identifier(
                                        &candidate.replacement,
                                    )?;
                                    Self::model_token_match_rank(token, model)
                                        .map(|rank| (rank, candidate))
                                })
                                .collect();
                        matched.sort_by_key(|(rank, _)| *rank);
                        let mut merged: Vec<CompletionCandidate> = matched
                            .into_iter()
                            .map(|(_, candidate)| candidate)
                            .collect();
                        merged.extend(
                            Self::model_subcommands()
                                .iter()
                                .filter(|candidate| candidate.starts_with(token))
                                .map(|candidate| CompletionCandidate {
                                    display: format!("{} · subcommand", candidate),
                                    replacement: (*candidate).to_string(),
                                }),
                        );
                        merged
                    }
                    Some("effort") => {
                        // `/model effort <TAB>` -> list the level literals.
                        Self::plain_candidates(
                            Self::model_effort_levels()
                                .iter()
                                .filter(|candidate| candidate.starts_with(token))
                                .map(|candidate| (*candidate).to_string()),
                        )
                    }
                    _ => Vec::new(),
                }
            } else if matches!(first, "/agent" | ":agent" | "/agents" | ":agents") {
                match words.next() {
                    None => {
                        let mut candidates = Self::agent_name_candidates(token);
                        candidates.extend(Self::plain_candidates(
                            Self::agent_subcommands()
                                .iter()
                                .filter(|c| c.starts_with(token))
                                .map(|c| c.to_string()),
                        ));
                        candidates
                    }
                    Some("use") if words.next().is_none() => Self::agent_name_candidates(token),
                    _ => Vec::new(),
                }
            } else if matches!(first, "/changes" | ":changes" | "/diff" | ":diff") {
                match words.next() {
                    None => {
                        // `/changes --open=<prefix>` glued form: complete the editor name and backfill the whole prefix;
                        // otherwise prefix-filter on the subcommand literals.
                        if let Some(prefix) = token.strip_prefix("--open=") {
                            let mut candidates = Self::plain_candidates(
                                Self::changes_open_editors()
                                    .iter()
                                    .filter(|c| c.starts_with(prefix))
                                    .map(|c| c.to_string()),
                            );
                            for c in &mut candidates {
                                c.replacement = format!("--open={}", c.replacement);
                                c.display = format!("--open={}", c.display);
                            }
                            candidates
                        } else {
                            Self::plain_candidates(
                                Self::changes_subcommands()
                                    .iter()
                                    .filter(|c| c.starts_with(token))
                                    .map(|c| c.to_string()),
                            )
                        }
                    }
                    Some("--open") | Some("open") if words.next().is_none() => {
                        Self::plain_candidates(
                            Self::changes_open_editors()
                                .iter()
                                .filter(|c| c.starts_with(token))
                                .map(|c| c.to_string()),
                        )
                    }
                    _ => Vec::new(),
                }
            } else {
                let sources: &[&str] = match first {
                    "/skills" | ":skills" | "/skill" | ":skill" => {
                        // Argument tokens already typed (before the cursor, excluding the command itself) — for multi-skill
                        // completion exclude already-picked names so `/skills a <TAB>` can pick the next one.
                        let consumed: Vec<&str> =
                            before[..token_start].split_whitespace().skip(1).collect();
                        let mut matched: Vec<(u8, CompletionCandidate)> = Vec::new();
                        if consumed.is_empty() {
                            // First argument position: subcommand literals join the hints too (`/skills us<TAB>` → use)
                            for sub in Self::skills_subcommands() {
                                if sub.starts_with(token) {
                                    matched.push((
                                        0,
                                        CompletionCandidate {
                                            display: format!("{sub} · subcommand"),
                                            replacement: sub.to_string(),
                                        },
                                    ));
                                }
                            }
                        }
                        for c in Self::skill_name_candidates() {
                            if consumed
                                .iter()
                                .any(|t| t.eq_ignore_ascii_case(&c.replacement))
                            {
                                continue;
                            }
                            if let Some(rank) = Self::name_token_match_rank(token, &c.replacement) {
                                matched.push((rank, c));
                            }
                        }
                        // Stable sort: at equal rank keep subcommands first and skills in manifest order.
                        matched.sort_by_key(|(rank, _)| *rank);
                        let candidates: Vec<CompletionCandidate> = matched
                            .into_iter()
                            .map(|(_, candidate)| candidate)
                            .collect();
                        return (token_start, candidates);
                    }
                    "/sessions" | ":sessions" | "/ss" | ":ss" => Self::session_subcommands(),
                    "/history" | ":history" => Self::history_subcommands(),
                    "/personas" | ":personas" => Self::persona_subcommands(),
                    "/usage" | ":usage" => Self::usage_subcommands(),
                    "/checkpoint" | ":checkpoint" | "/cp" | ":cp" => Self::checkpoint_subcommands(),
                    "/model" | ":model" => Self::model_subcommands(),
                    // `/effort <TAB>` -> reasoning-effort levels (same set as `/model effort <level>`).
                    "/effort" | ":effort" => Self::model_effort_levels(),
                    _ => &[],
                };
                Self::plain_candidates(
                    sources
                        .iter()
                        .filter(|c| c.starts_with(token))
                        .map(|c| c.to_string()),
                )
            }
        };

        (token_start, candidates)
    }

    fn complete_file_reference(before: &str) -> Option<(usize, Vec<CompletionCandidate>)> {
        let (token_start, raw_token, quote) = find_file_reference_token(before)?;
        let fragment = raw_token.strip_prefix('@')?;
        let fragment = if let Some(quote) = quote {
            fragment.strip_prefix(quote)?
        } else {
            fragment
        };
        let candidates = Self::plain_candidates(complete_path_fragment(fragment, quote));
        Some((token_start, candidates))
    }
}

/// Skill completion. Trigger and filtering rules (`<filter>` case-insensitive):
/// - `@ski` / `@skil` / `@skill` / `@skills` (a prefix of "skills", ≥3 chars): list all skills;
/// - `@skill<filter>` / `@skills<filter>`: after the keyword, keep typing letters to filter by name,
///   e.g. `@skillhum` → matches skills starting with `hum`;
/// - `@skill:<filter>` / `@skills:<filter>`: the colon-equivalent form (the canonical form inserted when a completion is picked).
///
/// Matching uses [`CommandCompleter::name_token_match_rank`] smart matching (prefix > per-segment prefix),
/// e.g. `@skills:own` → `audit_own_changes`. Once picked the line becomes `@skills:<name>`,
/// and the skill is force-injected for this turn. Returns `(token_start, candidates)` where token_start is the byte offset of `@`.
fn complete_skill_reference(before: &str) -> Option<(usize, Vec<CompletionCandidate>)> {
    let (token_start, token) = find_skill_reference_token(before)?;
    let rest = token.strip_prefix('@')?;
    let filters = skill_token_filters(rest)?;

    let skills = CommandCompleter::skill_name_candidates();

    // Smart-match each filter independently and union the multiple splits (any hit is kept).
    // An empty filter (still typing `@skill`/`@skills`) matches everything with an empty prefix, equivalent to "list all".
    let mut candidates = Vec::new();
    for skill in &skills {
        if !filters.iter().any(|filter| {
            CommandCompleter::name_token_match_rank(filter, &skill.replacement).is_some()
        }) {
            continue;
        }
        candidates.push(CompletionCandidate {
            display: skill.display.clone(),
            replacement: format!("@skills:{}", skill.replacement),
        });
    }
    Some((token_start, candidates))
}

/// Parse what follows `@`, decide whether it is a skill reference, and return all possible filter prefixes (lowercased).
/// `None` means it is not a skill reference token; an empty string inside the returned Vec means "list all".
///
/// Colon-less forms like `@skillsec` are ambiguous to split against the `skill`/`skills` keyword, so both
/// interpretations (e.g. `["ec", "sec"]`) are returned for a union, avoiding missing candidates the user wants.
pub(in crate::ai::prompt) fn skill_token_filters(rest: &str) -> Option<Vec<String>> {
    const MIN_TRIGGER_LEN: usize = 3;
    // Colon form: `<keyword>:<filter>`, where keyword must be a non-empty prefix of "skills" (≥3 chars).
    if let Some((keyword, filter)) = rest.split_once(':') {
        let keyword_lower = keyword.to_ascii_lowercase();
        if keyword_lower.len() >= MIN_TRIGGER_LEN && "skills".starts_with(&keyword_lower) {
            return Some(vec![filter.to_ascii_lowercase()]);
        }
        return None;
    }

    let rest_lower = rest.to_ascii_lowercase();
    // Still typing the keyword (`ski`/`skil`/`skill`/`skills`) ⇒ list all.
    if rest_lower.len() >= MIN_TRIGGER_LEN && "skills".starts_with(&rest_lower) {
        return Some(vec![String::new()]);
    }

    // Keyword fully typed; the letters after it are the filter prefix. Collect both splits and take their union.
    let mut filters = Vec::new();
    if let Some(filter) = rest_lower.strip_prefix("skills") {
        filters.push(filter.to_string());
    }
    if let Some(filter) = rest_lower.strip_prefix("skill") {
        filters.push(filter.to_string());
    }
    if filters.is_empty() {
        None
    } else {
        Some(filters)
    }
}

/// Locate the skill-reference token at end of line. Requires whitespace or line start before `@`, no
/// whitespace inside the token (same boundary rules as `@file`), and content after `@` recognized by [`skill_token_filters`] as a skill reference.
fn find_skill_reference_token(before: &str) -> Option<(usize, &str)> {
    let mut last_at = None;
    for (idx, ch) in before.char_indices() {
        if ch == '@' {
            last_at = Some(idx);
        }
    }
    let at_index = last_at?;
    let prev = before[..at_index].chars().next_back();
    if prev.is_some_and(|ch| !(ch.is_whitespace() || matches!(ch, '(' | '[' | '{' | '"' | '\''))) {
        return None;
    }
    let token = &before[at_index..];
    if token.chars().skip(1).any(char::is_whitespace) {
        return None;
    }
    skill_token_filters(&token[1..])?;
    Some((at_index, token))
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
    if prev.is_some_and(|ch| !(ch.is_whitespace() || matches!(ch, '(' | '[' | '{' | '"' | '\''))) {
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
    matches
        .into_iter()
        .map(|candidate| candidate.replacement)
        .collect()
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
        .then_with(|| {
            left.replacement
                .to_ascii_lowercase()
                .cmp(&right.replacement.to_ascii_lowercase())
        })
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
        if candidate.starts_with(fragment)
            || fragment.is_empty()
            || fragment == "."
            || fragment == ".."
        {
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
                display: candidate.display,
                replacement: candidate.replacement,
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
        assert!(pairs.iter().any(|pair| pair.replacement == "/agent"));
        assert!(!pairs.iter().any(|pair| pair.replacement == "/agents"));
    }

    #[test]
    fn command_completion_suggests_agent_names_for_direct_switch() {
        let manifests = crate::ai::agents::load_all_agents();
        CommandCompleter::set_agent_manifests(&manifests);

        let (_, direct) = CommandCompleter::complete_for_line("/agent au", 9);
        assert!(
            direct
                .iter()
                .any(|candidate| candidate.replacement == "audit")
        );

        let (_, legacy) = CommandCompleter::complete_for_line("/agent use au", 13);
        assert!(
            legacy
                .iter()
                .any(|candidate| candidate.replacement == "audit")
        );
    }

    #[test]
    fn command_completion_suggests_agent_names_for_plural_alias() {
        let manifests = crate::ai::agents::load_all_agents();
        CommandCompleter::set_agent_manifests(&manifests);

        // `/agents <name>` aligns with the runtime alias: the second token should also complete by agent name.
        let (_, direct) = CommandCompleter::complete_for_line("/agents au", 10);
        assert!(
            direct
                .iter()
                .any(|candidate| candidate.replacement == "audit")
        );

        let (_, legacy) = CommandCompleter::complete_for_line("/agents use au", 14);
        assert!(
            legacy
                .iter()
                .any(|candidate| candidate.replacement == "audit")
        );
    }

    #[test]
    fn command_completion_agent_name_matches_segments() {
        let manifests = crate::ai::agents::load_all_agents();
        CommandCompleter::set_agent_manifests(&manifests);
        if !crate::ai::agents::get_primary_agents(&manifests)
            .iter()
            .any(|a| a.name == "audit-fast")
        {
            return; // skip when the agent is absent
        }
        // Segment matching: `fast` is not a prefix of `audit-fast` but hits its second segment.
        let (_, direct) = CommandCompleter::complete_for_line("/agent fast", 11);
        assert!(
            direct
                .iter()
                .any(|candidate| candidate.replacement == "audit-fast"),
            "expected audit-fast for `/agent fast`: {:?}",
            direct.iter().map(|c| &c.replacement).collect::<Vec<_>>()
        );
        let (_, use_cmd) = CommandCompleter::complete_for_line("/agent use fast", 15);
        assert!(
            use_cmd
                .iter()
                .any(|candidate| candidate.replacement == "audit-fast"),
            "expected audit-fast for `/agent use fast`: {:?}",
            use_cmd.iter().map(|c| &c.replacement).collect::<Vec<_>>()
        );
    }

    #[test]
    fn command_completion_skill_name_matches_segments() {
        let skills = crate::ai::skills::load_all_skills();
        if !skills.iter().any(|s| s.name == "audit_own_changes") {
            return; // skip when the skill is absent
        }
        // Segment matching: `own` is not a prefix of `audit_own_changes` but hits its second segment.
        let (_, candidates) = CommandCompleter::complete_for_line("/skills own", 11);
        assert!(
            candidates
                .iter()
                .any(|c| c.replacement == "audit_own_changes"),
            "expected audit_own_changes for `/skills own`: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn agent_candidates_fall_back_to_disk_scan_before_cache_is_filled() {
        // On a fresh start where the driver has not called set_agent_manifests yet (cache is None),
        // agent-name completion must synchronously scan out candidates, otherwise
        // `/agent <prefix><Tab>` / `/agents <prefix><Tab>` before the first input of a new session hangs.
        let manifests = crate::ai::agents::load_all_agents();
        let candidates = CommandCompleter::agent_candidates_from_manifests(&manifests);
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.replacement == "audit")
        );
    }

    #[test]
    fn command_completion_expands_top_level_close_command() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (_, pairs) = completer
            .complete("/clo", 4, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == "/close"));
        // the ":" prefix works the same way
        let (_, pairs) = completer
            .complete(":clo", 4, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == ":close"));
    }

    #[test]
    fn command_completion_lists_agent_subcommands() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (start, pairs) = completer
            .complete("/agent a", 8, &Context::new(&history))
            .unwrap();
        assert_eq!(start, 7);
        assert!(pairs.iter().any(|pair| pair.replacement == "auto"));
    }

    #[test]
    fn command_completion_expands_top_level_persona_command() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (_, pairs) = completer
            .complete("/pers", 5, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == "/personas"));
    }

    #[test]
    fn command_completion_suggests_mark_and_unmark() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (_, pairs) = completer
            .complete("/ma", 3, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == "/mark"));
        let (_, pairs) = completer
            .complete("/un", 3, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == "/unmark"));
        let (_, pairs) = completer
            .complete(":ma", 3, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == ":mark"));
    }

    #[test]
    fn command_completion_lists_persona_subcommands() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (start, pairs) = completer
            .complete("/personas c", 11, &Context::new(&history))
            .unwrap();
        assert_eq!(start, 10);
        assert!(pairs.iter().any(|pair| pair.replacement == "create"));
        assert!(pairs.iter().any(|pair| pair.replacement == "current"));
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
    fn history_command_completion_includes_rewind_and_last_shortcut() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (_, pairs) = completer
            .complete("/history ", 9, &Context::new(&history))
            .unwrap();

        assert!(pairs.iter().any(|pair| pair.replacement == "rewind"));
        assert!(!pairs.iter().any(|pair| pair.replacement == "undo"));
        assert!(pairs.iter().any(|pair| pair.replacement == "last"));
    }

    #[test]
    fn session_command_completion_tracks_real_subcommands() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (_, pairs) = completer
            .complete("/sessions ", 10, &Context::new(&history))
            .unwrap();

        assert!(pairs.iter().any(|pair| pair.replacement == "clear-history"));
        assert!(pairs.iter().any(|pair| pair.replacement == "fork"));
        assert!(pairs.iter().any(|pair| pair.replacement == "branch"));
        assert!(!pairs.iter().any(|pair| pair.replacement == "rewind"));
    }

    #[test]
    fn model_command_completion_lists_full_command_candidates() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let model = crate::ai::model_names::all()
            .first()
            .map(|m| crate::ai::model_names::model_handle(m))
            .expect("model registry is empty");

        let (_, pairs) = completer
            .complete("/model", 6, &Context::new(&history))
            .unwrap();

        assert!(
            pairs
                .iter()
                .any(|pair| pair.replacement == format!("/model {model}"))
        );
    }

    #[test]
    fn model_command_completion_prefers_current_model_first() {
        let current = crate::ai::model_names::all()
            .first()
            .map(|m| crate::ai::model_names::model_handle(m))
            .expect("model registry is empty");
        CommandCompleter::set_current_model_hint(&current);

        let (_, candidates) = CommandCompleter::complete_for_line("/model ", 7);

        let first = candidates
            .first()
            .expect("model candidates should not be empty");
        assert_eq!(first.replacement, current);
        assert!(first.display.contains("current"));
    }

    #[test]
    fn trie_command_completion_expands_usage_prefix() {
        // /usa → /usage (Trie prefix match)
        let (_, candidates) = CommandCompleter::complete_for_line("/usa", 4);
        assert!(
            candidates.iter().any(|c| c.replacement == "/usage"),
            "expected /usage for /usa, got: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn trie_command_completion_expands_changes_prefix() {
        // /cha → /changes (Trie prefix match)
        let (_, candidates) = CommandCompleter::complete_for_line("/cha", 4);
        assert!(
            candidates.iter().any(|c| c.replacement == "/changes"),
            "expected /changes for /cha, got: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
        // /diff alias exact match
        let (_, candidates) = CommandCompleter::complete_for_line("/diff", 5);
        assert!(
            candidates.iter().any(|c| c.replacement == "/diff"),
            "expected /diff for /diff, got: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
        // :changes completes in the colon form too
        let (_, candidates) = CommandCompleter::complete_for_line(":ch", 3);
        assert!(candidates.iter().any(|c| c.replacement == ":changes"));
    }

    #[test]
    fn changes_subcommand_completion_lists_flags_and_editors() {
        // `/changes --st` → --stat
        let (_, candidates) =
            CommandCompleter::complete_for_line("/changes --st", "/changes --st".len());
        assert!(
            candidates.iter().any(|c| c.replacement == "--stat"),
            "expected --stat for `/changes --st`, got: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
        // `/changes --open c` → code / cursor
        let (_, candidates) =
            CommandCompleter::complete_for_line("/changes --open c", "/changes --open c".len());
        assert!(candidates.iter().any(|c| c.replacement == "code"));
        assert!(candidates.iter().any(|c| c.replacement == "cursor"));
        assert!(!candidates.iter().any(|c| c.replacement == "idea"));
        // `/changes --open=co` glued form backfills the whole prefix
        let (_, candidates) =
            CommandCompleter::complete_for_line("/changes --open=co", "/changes --open=co".len());
        assert!(
            candidates.iter().any(|c| c.replacement == "--open=code"),
            "expected --open=code for `/changes --open=co`, got: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
        // `/diff --j` alias goes through subcommand completion too
        let (_, candidates) = CommandCompleter::complete_for_line("/diff --j", "/diff --j".len());
        assert!(
            candidates.iter().any(|c| c.replacement == "--json"),
            "expected --json for `/diff --j`, got: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn trie_flag_completion_expands_model_prefix() {
        // --mod → --model (option Trie prefix match)
        let (_, candidates) = CommandCompleter::complete_for_line("--mod", 5);
        assert!(
            candidates.iter().any(|c| c.replacement == "--model"),
            "expected --model for --mod, got: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn trie_flag_completion_expands_h_flag() {
        // -h → -h (short option exact match)
        let (_, candidates) = CommandCompleter::complete_for_line("-h", 2);
        assert!(candidates.iter().any(|c| c.replacement == "-h"));
        // --h → --help matches too
        let (_, candidates) = CommandCompleter::complete_for_line("--h", 3);
        assert!(candidates.iter().any(|c| c.replacement == "--help"));
    }

    #[test]
    fn completion_pos_inside_multibyte_char_does_not_panic() {
        // When the cursor byte offset lands inside a multi-byte UTF-8 character (e.g. a CJK char),
        // a direct slice panics. After aligning down to a character boundary it should return safely.
        let line = "帮我给a.rs 这个 agent 增加一个dump 功能";
        for pos in 0..=line.len() {
            let _ = CommandCompleter::complete_for_line(line, pos);
        }
    }

    #[test]
    fn skill_reference_completion_lists_skills() {
        // This test compares two independent load_all_skills() snapshots (one inside complete_for_line,
        // one for expected), while other cases rewrite the global HOME under ENV_LOCK. Without holding
        // the lock, HOME could flip between the two snapshots and desync the candidate sets, so serialize with them.
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let (start, candidates) = CommandCompleter::complete_for_line("@skills", 7);
        assert_eq!(start, 0);
        let expected: Vec<String> = crate::ai::skills::load_all_skills()
            .into_iter()
            .map(|s| format!("@skills:{}", s.name))
            .collect();
        assert!(!expected.is_empty(), "no skills available to complete");
        for replacement in expected {
            assert!(
                candidates.iter().any(|c| c.replacement == replacement),
                "missing candidate {replacement}"
            );
        }
    }

    #[test]
    fn skill_reference_completion_triggers_on_short_prefix() {
        // Same as above: comparing candidate count against two snapshots of load_all_skills().len() requires serializing with the HOME-rewriting cases.
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // Typing `@ski` (a prefix of "skills") should already trigger and list all skills.
        let (start, candidates) = CommandCompleter::complete_for_line("@ski", 4);
        assert_eq!(start, 0);
        let total = crate::ai::skills::load_all_skills().len();
        assert_eq!(candidates.len(), total);
        // `@sk` (<3 chars) does not trigger, avoiding hijacking plain file-path completion.
        assert!(complete_skill_reference("@sk").is_none());
        // `@skq` is not a prefix of "skills" and does not trigger.
        assert!(complete_skill_reference("@skq").is_none());
    }

    #[test]
    fn skill_reference_completion_filters_by_prefix() {
        let skills = crate::ai::skills::load_all_skills();
        let Some(first) = skills.first() else {
            return;
        };
        // Use the first letter as the filter; every candidate name should pass smart matching (prefix or segment).
        let ch = first.name.chars().next().unwrap();
        let line = format!("@skills:{ch}");
        let (_, candidates) = CommandCompleter::complete_for_line(&line, line.len());
        assert!(!candidates.is_empty());
        for c in &candidates {
            let name = c.replacement.strip_prefix("@skills:").unwrap();
            assert!(
                CommandCompleter::name_token_match_rank(
                    &ch.to_ascii_lowercase().to_string(),
                    name,
                )
                .is_some(),
                "candidate {name} should match filter {ch}"
            );
        }
    }

    #[test]
    fn skill_reference_completion_ignores_midword_at() {
        // In `foo@skills` the `@` is not preceded by a boundary character, so skill completion must not trigger.
        let result = complete_skill_reference("foo@skills");
        assert!(result.is_none());
    }

    #[test]
    fn skill_reference_completion_filters_without_colon() {
        let skills = crate::ai::skills::load_all_skills();
        let Some(target) = skills.first() else {
            return;
        };
        // Take the first 3 characters of a real skill name as the filter prefix, using the colon-less form `@skill<prefix>`.
        let name_lower = target.name.to_ascii_lowercase();
        let take = name_lower.chars().take(3).collect::<String>();
        if take.chars().count() < 3 {
            return;
        }
        let line = format!("@skill{take}");
        let (_, candidates) = CommandCompleter::complete_for_line(&line, line.len());
        // The target skill must be among the candidates, and all candidate names must pass smart matching (both unioned splits comply).
        assert!(
            candidates
                .iter()
                .any(|c| c.replacement == format!("@skills:{}", target.name)),
            "expected {} in candidates for line {line}",
            target.name
        );
        for c in &candidates {
            let name = c.replacement.strip_prefix("@skills:").unwrap();
            assert!(
                CommandCompleter::name_token_match_rank(&take, name).is_some(),
                "candidate {name} should match filter {take}"
            );
        }
    }

    #[test]
    fn skill_reference_completion_matches_segments() {
        let skills = crate::ai::skills::load_all_skills();
        if !skills.iter().any(|s| s.name == "audit_own_changes") {
            return; // skip when the skill is absent
        }
        // Segment matching: `own` is not a prefix of `audit_own_changes` but hits its second segment.
        let (_, candidates) = CommandCompleter::complete_for_line("@skills:own", 11);
        assert!(
            candidates
                .iter()
                .any(|c| c.replacement == "@skills:audit_own_changes"),
            "expected audit_own_changes for `@skills:own`: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn skill_completion_matches_name_not_description() {
        // Regression: `/skills audit_o` must only hit the name-matched skill (audit_own_changes),
        // never a false match because some skill's description contains words like "audits of" (e.g. TRAE-security-review).
        let line = "/skills audit_o";
        let (_, candidates) = CommandCompleter::complete_for_line(line, line.len());
        assert!(
            candidates
                .iter()
                .any(|c| c.replacement == "audit_own_changes"),
            "expected audit_own_changes for `{line}`: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
        for c in &candidates {
            // Every candidate must match `audit_o` by name alone (descriptions must not participate in matching).
            assert!(
                CommandCompleter::name_token_match_rank("audit_o", &c.replacement).is_some(),
                "candidate {} matched via description only",
                c.replacement
            );
        }
    }

    #[test]
    fn skill_token_filters_parses_variants() {
        // Mid-keyword: list all (empty filter).
        assert_eq!(skill_token_filters("ski"), Some(vec![String::new()]));
        assert_eq!(skill_token_filters("skill"), Some(vec![String::new()]));
        assert_eq!(skill_token_filters("skills"), Some(vec![String::new()]));
        // Colon-less filter: `skillhum` → keyword `skill` + prefix `hum`.
        assert_eq!(
            skill_token_filters("skillhum"),
            Some(vec!["hum".to_string()])
        );
        // Colon filter.
        assert_eq!(
            skill_token_filters("skills:deb"),
            Some(vec!["deb".to_string()])
        );
        // Too short or not a prefix: not recognized.
        assert_eq!(skill_token_filters("sk"), None);
        assert_eq!(skill_token_filters("skq"), None);
    }

    #[test]
    fn direct_model_completion_lists_models() {
        let current = crate::ai::model_names::all()
            .first()
            .map(|m| crate::ai::model_names::model_handle(m))
            .expect("model registry is empty");
        CommandCompleter::set_current_model_hint(&current);

        let (_, candidates) = CommandCompleter::complete_for_line("/model ", 7);

        assert_eq!(
            candidates.first().map(|c| c.replacement.as_str()),
            Some(current.as_str())
        );
    }

    #[test]
    fn model_completion_deep_keeps_prefix_behavior() {
        let (_, candidates) = CommandCompleter::complete_for_line("/model deep", 11);
        let repls: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        assert!(
            repls.iter().any(|r| r.starts_with("deepseek-")),
            "expected deepseek models for `/model deep`: {:?}",
            repls
        );
    }

    #[test]
    fn model_completion_deep_v_completes_volcano_deepseek() {
        if crate::ai::model_names::find_by_identifier("deepseek-v4-flash-volcano").is_none() {
            return; // skip when the model registry (models/) lacks this model
        }
        let (_, candidates) = CommandCompleter::complete_for_line("/model deep-v", 13);
        let repls: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        assert!(
            repls.iter().any(|r| r == "deepseek-v4-flash-volcano"),
            "expected volcano deepseek for `/model deep-v`: {:?}",
            repls
        );
        // The volcano deepseek should rank ahead of other deepseek-v4 models (two-segment matching wins).
        let volcano_idx = repls
            .iter()
            .position(|r| r == "deepseek-v4-flash-volcano")
            .expect("volcano deepseek should be a candidate");
        for (idx, r) in repls.iter().enumerate() {
            if r.starts_with("deepseek-v4-") && r.as_str() != "deepseek-v4-flash-volcano" {
                assert!(
                    idx > volcano_idx,
                    "volcano deepseek should rank before other deepseek-v4 models: {:?}",
                    repls
                );
            }
        }
    }

    #[test]
    fn model_completion_glm_v_completes_glm_volcano() {
        if crate::ai::model_names::find_by_identifier("glm-5.2-volcano").is_none() {
            return; // skip when the model registry (models/) lacks this model
        }
        let (_, candidates) = CommandCompleter::complete_for_line("/model glm-v", 12);
        let repls: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        assert!(
            repls.iter().any(|r| r == "glm-5.2-volcano"),
            "expected glm volcano for `/model glm-v`: {:?}",
            repls
        );
    }

    #[test]
    fn model_completion_deepseek_v4_f_keeps_prefix_behavior() {
        let (_, candidates) = CommandCompleter::complete_for_line("/model deepseek-v4-f", 20);
        let repls: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        assert!(
            repls.iter().any(|r| r.starts_with("deepseek-v4-flash-")),
            "expected deepseek-v4-flash models for `/model deepseek-v4-f`: {:?}",
            repls
        );
    }

    #[test]
    fn model_completion_includes_effort_subcommand() {
        let (_, candidates) = CommandCompleter::complete_for_line("/model ef", 9);
        assert!(
            candidates.iter().any(|c| c.replacement == "effort"),
            "expected `effort` in candidates: {:?}",
            candidates
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn model_completion_includes_current_help_list_subcommands() {
        let (_, candidates) = CommandCompleter::complete_for_line("/model ", 7);
        let labels: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        for sub in ["current", "list", "help", "effort"] {
            assert!(
                labels.iter().any(|x| x == sub),
                "expected subcommand `{}` in candidates: {:?}",
                sub,
                labels
            );
        }
        for removed in ["use", "select", "switch"] {
            assert!(
                !labels.iter().any(|x| x == removed),
                "did not expect removed model alias `{}` in candidates: {:?}",
                removed,
                labels
            );
        }
    }

    #[test]
    fn model_effort_completion_lists_levels() {
        let (start, candidates) = CommandCompleter::complete_for_line("/model effort ", 14);
        assert_eq!(start, 14);
        let labels: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        for level in [
            "minimal", "low", "medium", "high", "xhigh", "max", "auto", "off",
        ] {
            assert!(
                labels.iter().any(|x| x == level),
                "expected level `{}` in candidates: {:?}",
                level,
                labels
            );
        }
    }

    #[test]
    fn model_effort_completion_filters_by_prefix() {
        let (_, candidates) = CommandCompleter::complete_for_line("/model effort m", 15);
        let labels: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        assert!(labels.iter().any(|x| x == "minimal"));
        assert!(labels.iter().any(|x| x == "medium"));
        assert!(!labels.iter().any(|x| x == "high"));
        assert!(!labels.iter().any(|x| x == "low"));
    }

    #[test]
    fn command_completion_expands_effort_shortcut() {
        let (_, candidates) = CommandCompleter::complete_for_line("/effor", 6);
        let labels: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        assert!(
            labels.iter().any(|x| x == "/effort"),
            "expected `/effort` for `/effor`: {:?}",
            labels
        );
    }

    #[test]
    fn effort_shortcut_completion_lists_levels() {
        let (start, candidates) = CommandCompleter::complete_for_line("/effort ", 8);
        assert_eq!(start, 8);
        let labels: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        for level in [
            "minimal", "low", "medium", "high", "xhigh", "max", "auto", "off",
        ] {
            assert!(
                labels.iter().any(|x| x == level),
                "expected level `{}` for `/effort `: {:?}",
                level,
                labels
            );
        }
    }

    #[test]
    fn effort_shortcut_completion_filters_by_prefix() {
        let (_, candidates) = CommandCompleter::complete_for_line("/effort h", 9);
        let labels: Vec<_> = candidates.iter().map(|c| c.replacement.clone()).collect();
        assert!(labels.iter().any(|x| x == "high"));
        assert!(!labels.iter().any(|x| x == "low"));
        assert!(!labels.iter().any(|x| x == "auto"));
    }

    #[test]
    fn model_removed_alias_completion_lists_no_models() {
        let (_, candidates) = CommandCompleter::complete_for_line("/model use ", 11);
        assert!(
            candidates.is_empty(),
            "did not expect model candidates after removed `/model use` alias"
        );
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
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.replacement == format!("@{}", image.display()))
        );
    }

    #[test]
    fn file_completion_quotes_paths_with_spaces() {
        let dir = std::env::temp_dir().join(format!("ai complete {}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let image = dir.join("error shot.png");
        std::fs::write(&image, b"fake").unwrap();
        let line = format!("@\"{}/err", dir.display());

        let (_, candidates) = CommandCompleter::complete_for_line(&line, line.len());

        assert!(
            candidates
                .iter()
                .any(|candidate| { candidate.replacement == format!("@\"{}\"", image.display()) })
        );
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
