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

/// Trie 存储所有 "/" 和 ":" 开头的顶层命令，替换原先的线性 starts_with 过滤。
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
        ":model",
        "/agents",
        ":agents",
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
        "/proc",
        ":proc",
        "/skills",
        ":skills",
    ] {
        trie.insert(cmd);
    }
    trie
});

/// Trie 存储所有 "--" 和 "-" 开头的 CLI 选项（含简写），支持选项补全。
static FLAGS_TRIE: LazyLock<Trie> = LazyLock::new(|| {
    let mut trie = Trie::new();
    for flag in &[
        // bool 选项
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
        // string/int 选项
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
            ":model",
            "/agents",
            ":agents",
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
            "/proc",
            ":proc",
            "/skills",
            ":skills",
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
        // replacement 使用 model.key，这样 find_by_identifier 能通过 key 正确解析。
        // 如果 key 为空，fallback 到 handle（兼容旧模型）。
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

    fn agent_subcommands() -> &'static [&'static str] {
        &["help", "list", "current", "use", "auto"]
    }

    /// `/model` 的二级子命令字面量。注意：这些子命令与"模型名"互斥地占据
    /// 第二个 token；为了让 Tab 既能补出子命令也能补出模型名，
    /// `complete_for_line` 会把它们合并到候选列表中（前缀过滤）。
    fn model_subcommands() -> &'static [&'static str] {
        &["current", "list", "help", "effort"]
    }

    /// `/model effort` 的第三个 token 候选：推理强度档位 + auto/off。
    fn model_effort_levels() -> &'static [&'static str] {
        &["minimal", "low", "medium", "high", "xhigh", "auto", "off"]
    }

    fn session_subcommands() -> &'static [&'static str] {
        crate::ai::driver::commands::session::CANONICAL_SESSION_SUBCOMMANDS
    }

    fn persona_subcommands() -> &'static [&'static str] {
        &["list", "current", "create", "new", "use", "delete", "help"]
    }

    /// `/usage` 的子命令。
    fn usage_subcommands() -> &'static [&'static str] {
        &[
            "today", "7d", "30d", "all", "models", "daily", "trend", "days", "help",
        ]
    }

    /// `/checkpoint` / `/cp` 的子命令。
    fn checkpoint_subcommands() -> &'static [&'static str] {
        &["save", "list", "rollback", "delete", "help"]
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

    /// `/skills` / `/skill` 的子命令字面量。
    fn skills_subcommands() -> &'static [&'static str] {
        &["list", "current", "use", "help"]
    }

    /// 所有已加载 skill 的名称候选（带 display）。
    fn skill_name_candidates() -> Vec<CompletionCandidate> {
        let skills = crate::ai::skills::load_all_skills();
        skills
            .into_iter()
            .map(|s| {
                let display = if s.description.trim().is_empty() {
                    s.name.clone()
                } else {
                    format!("{} · {}", s.name, s.description.trim())
                };
                CompletionCandidate {
                    display,
                    replacement: s.name,
                }
            })
            .collect()
    }

    pub(super) fn complete_for_line(line: &str, pos: usize) -> (usize, Vec<CompletionCandidate>) {
        // `pos` 是字节偏移的光标位置，可能落在多字节 UTF-8 字符内部（如中文），
        // 直接 `&line[..pos]` 切片会 panic。向下对齐到最近的字符边界。
        let mut pos = pos.min(line.len());
        while pos > 0 && !line.is_char_boundary(pos) {
            pos -= 1;
        }
        let before = &line[..pos];
        // `@skills` / `@skill[:prefix]` 触发技能补全，必须先于普通 `@file` 补全，
        // 否则 `complete_file_reference` 会把 `@skills` 当成文件路径片段处理。
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
            // 用 Trie 做前缀匹配："/" / ":" 走命令 Trie，"--" / "-" 走选项 Trie；
            // 排序结果以保证确定性（HashMap 迭代无序）。
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
                // 第二个 token 之后还有更深层补全（目前只有 `/model effort <level>`）。
                // 这里检查"`/model` 后面已经有几个非空 token"。
                let second = words.next();
                match second {
                    None => {
                        // 第二个 token：模型名（带 current 置顶）+ `/model` 子命令字面量。
                        // 模型名放在前面以保留"当前模型 == 第一项"的体验。
                        let mut merged: Vec<CompletionCandidate> = Self::model_name_candidates()
                            .into_iter()
                            .filter(|candidate| candidate.replacement.starts_with(token))
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
                        // `/model effort <TAB>` -> 列出档位字面量。
                        Self::plain_candidates(
                            Self::model_effort_levels()
                                .iter()
                                .filter(|candidate| candidate.starts_with(token))
                                .map(|candidate| (*candidate).to_string()),
                        )
                    }
                    _ => Vec::new(),
                }
            } else {
                let candidates = match first {
                    "/skills" | ":skills" | "/skill" | ":skill" => Self::skill_name_candidates()
                        .into_iter()
                        .filter(|c| c.replacement.starts_with(token))
                        .collect(),
                    _ => {
                        let sources: &[&str] = match first {
                            "/agents" | ":agents" | "/agent" | ":agent" => {
                                Self::agent_subcommands()
                            }
                            "/sessions" | ":sessions" | "/ss" | ":ss" => {
                                Self::session_subcommands()
                            }
                            "/history" | ":history" => Self::history_subcommands(),
                            "/personas" | ":personas" => Self::persona_subcommands(),
                            "/usage" | ":usage" => Self::usage_subcommands(),
                            "/checkpoint" | ":checkpoint" | "/cp" | ":cp" => {
                                Self::checkpoint_subcommands()
                            }
                            "/model" | ":model" => Self::model_subcommands(),
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
                candidates
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

/// 技能补全。触发与过滤规则（`<filter>` 大小写不敏感）：
/// - `@ski` / `@skil` / `@skill` / `@skills`（"skills" 的前缀，≥3 字符）：列出全部 skill；
/// - `@skill<filter>` / `@skills<filter>`：输完关键字后直接续打字母即按前缀过滤，
///   例如 `@skillhum` → 匹配以 `hum` 开头的 skill；
/// - `@skill:<filter>` / `@skills:<filter>`：带冒号的等价写法（补全选中后插入的规范形式）。
///
/// 前缀匹配用项目内的 [`Trie`](rust_tools::cw::Trie) 实现：把全部 skill 名（小写）插入
/// 字典树，再用 `words_with_prefix` 取出命中集合。选中后行内变成 `@skills:<name>`，
/// 本轮对话将强制注入该 skill。返回 `(token_start, candidates)`，token_start 是 `@` 的字节偏移。
fn complete_skill_reference(before: &str) -> Option<(usize, Vec<CompletionCandidate>)> {
    let (token_start, token) = find_skill_reference_token(before)?;
    let rest = token.strip_prefix('@')?;
    let filters = skill_token_filters(rest)?;

    let skills = crate::ai::skills::load_all_skills();

    // 任一切分得到空过滤词 ⇒ 仍在输入关键字（如 `@skill`/`@skills`），列出全部。
    let list_all = filters.iter().any(|f| f.is_empty());

    // 否则用 Trie 做前缀匹配：小写 skill 名 → 命中集合（多种切分取并集）。
    let matched: Option<rust_tools::commonw::FastSet<String>> = if list_all {
        None
    } else {
        let mut trie = rust_tools::cw::Trie::new();
        for skill in &skills {
            trie.insert(&skill.name.to_ascii_lowercase());
        }
        let mut set = rust_tools::commonw::FastSet::default();
        for filter in &filters {
            for word in trie.words_with_prefix(filter) {
                set.insert(word);
            }
        }
        Some(set)
    };

    let mut candidates = Vec::new();
    for skill in &skills {
        if let Some(set) = &matched {
            if !set.contains(&skill.name.to_ascii_lowercase()) {
                continue;
            }
        }
        let display = if skill.description.trim().is_empty() {
            skill.name.clone()
        } else {
            format!("{} · {}", skill.name, skill.description.trim())
        };
        candidates.push(CompletionCandidate {
            display,
            replacement: format!("@skills:{}", skill.name),
        });
    }
    Some((token_start, candidates))
}

/// 解析 `@` 之后的内容，判断是否为技能引用并返回所有可能的过滤前缀（小写）。
/// 返回 `None` 表示不是技能引用 token；返回的 Vec 中含空串表示"列出全部"。
///
/// `@skillsec` 这类无冒号写法对关键字 `skill`/`skills` 存在切分歧义，故同时返回
/// 两种解释（如 `["ec", "sec"]`）取并集，避免漏掉用户想要的候选。
pub(in crate::ai::prompt) fn skill_token_filters(rest: &str) -> Option<Vec<String>> {
    const MIN_TRIGGER_LEN: usize = 3;
    // 冒号写法：`<keyword>:<filter>`，keyword 须为 "skills" 的非空前缀（≥3 字符）。
    if let Some((keyword, filter)) = rest.split_once(':') {
        let keyword_lower = keyword.to_ascii_lowercase();
        if keyword_lower.len() >= MIN_TRIGGER_LEN && "skills".starts_with(&keyword_lower) {
            return Some(vec![filter.to_ascii_lowercase()]);
        }
        return None;
    }

    let rest_lower = rest.to_ascii_lowercase();
    // 仍在输入关键字途中（`ski`/`skil`/`skill`/`skills`）⇒ 列出全部。
    if rest_lower.len() >= MIN_TRIGGER_LEN && "skills".starts_with(&rest_lower) {
        return Some(vec![String::new()]);
    }

    // 关键字已输完，其后字母即过滤前缀。两种切分都收集，取并集。
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

/// 定位行尾的技能引用 token。要求 `@` 前是空白或行首、token 内无空白（与 `@file`
/// 边界规则一致），且 `@` 之后内容能被 [`skill_token_filters`] 识别为技能引用。
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
        assert!(pairs.iter().any(|pair| pair.replacement == "/agents"));
    }
    #[test]
    fn command_completion_expands_top_level_close_command() {
        let completer = CommandCompleter;
        let history = DefaultHistory::new();
        let (_, pairs) = completer
            .complete("/clo", 4, &Context::new(&history))
            .unwrap();
        assert!(pairs.iter().any(|pair| pair.replacement == "/close"));
        // ":" 前缀同样可用
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
            .complete("/agents a", 9, &Context::new(&history))
            .unwrap();
        assert_eq!(start, 8);
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
            .expect("models.json is empty");

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
            .expect("models.json is empty");
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
        // /usa → /usage（Trie 前缀匹配）
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
    fn trie_flag_completion_expands_model_prefix() {
        // --mod → --model（选项 Trie 前缀匹配）
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
        // -h → -h（短选项精确匹配）
        let (_, candidates) = CommandCompleter::complete_for_line("-h", 2);
        assert!(candidates.iter().any(|c| c.replacement == "-h"));
        // --h → --help 也匹配
        let (_, candidates) = CommandCompleter::complete_for_line("--h", 3);
        assert!(candidates.iter().any(|c| c.replacement == "--help"));
    }

    #[test]
    fn completion_pos_inside_multibyte_char_does_not_panic() {
        // 光标字节偏移落在多字节 UTF-8 字符内部（如中文'能'）时，
        // 直接切片会 panic。向下对齐到字符边界后应安全返回。
        let line = "帮我给a.rs 这个 agent 增加一个dump 功能";
        for pos in 0..=line.len() {
            let _ = CommandCompleter::complete_for_line(line, pos);
        }
    }

    #[test]
    fn skill_reference_completion_lists_skills() {
        // 该测试比较两次独立的 load_all_skills() 快照（complete_for_line 内部一次、
        // expected 一次），而其他用例会在 ENV_LOCK 下改写全局 HOME。不持锁则 HOME
        // 可能在两次快照之间翻转导致候选集不一致，故与之串行。
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
        // 同上：比较候选数量与 load_all_skills().len() 两次快照，需与改写 HOME 的用例串行。
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // 输入 `@ski`（"skills" 的前缀）即应触发，列出全部 skill。
        let (start, candidates) = CommandCompleter::complete_for_line("@ski", 4);
        assert_eq!(start, 0);
        let total = crate::ai::skills::load_all_skills().len();
        assert_eq!(candidates.len(), total);
        // `@sk`（<3 字符）不触发，避免劫持普通文件路径补全。
        assert!(complete_skill_reference("@sk").is_none());
        // `@skq` 不是 "skills" 的前缀，不触发。
        assert!(complete_skill_reference("@skq").is_none());
    }

    #[test]
    fn skill_reference_completion_filters_by_prefix() {
        let skills = crate::ai::skills::load_all_skills();
        let Some(first) = skills.first() else {
            return;
        };
        // 取首字母作为前缀，结果里所有候选名都应以该前缀开头。
        let ch = first.name.chars().next().unwrap();
        let line = format!("@skills:{ch}");
        let (_, candidates) = CommandCompleter::complete_for_line(&line, line.len());
        assert!(!candidates.is_empty());
        for c in &candidates {
            let name = c.replacement.strip_prefix("@skills:").unwrap();
            assert!(
                name.to_ascii_lowercase()
                    .starts_with(&ch.to_ascii_lowercase().to_string())
            );
        }
    }

    #[test]
    fn skill_reference_completion_ignores_midword_at() {
        // `foo@skills` 中的 `@` 前不是边界字符，不应触发技能补全。
        let result = complete_skill_reference("foo@skills");
        assert!(result.is_none());
    }

    #[test]
    fn skill_reference_completion_filters_without_colon() {
        let skills = crate::ai::skills::load_all_skills();
        let Some(target) = skills.first() else {
            return;
        };
        // 取某个真实 skill 名的前 3 个字符作为过滤前缀，用无冒号写法 `@skill<prefix>`。
        let name_lower = target.name.to_ascii_lowercase();
        let take = name_lower.chars().take(3).collect::<String>();
        if take.chars().count() < 3 {
            return;
        }
        let line = format!("@skill{take}");
        let (_, candidates) = CommandCompleter::complete_for_line(&line, line.len());
        // 目标 skill 必须在候选里，且所有候选名都以该前缀开头（取并集的两种切分均符合）。
        assert!(
            candidates
                .iter()
                .any(|c| c.replacement == format!("@skills:{}", target.name)),
            "expected {} in candidates for line {line}",
            target.name
        );
        for c in &candidates {
            let name = c.replacement.strip_prefix("@skills:").unwrap();
            assert!(name.to_ascii_lowercase().starts_with(&take));
        }
    }

    #[test]
    fn skill_token_filters_parses_variants() {
        // 关键字途中：列出全部（空过滤词）。
        assert_eq!(skill_token_filters("ski"), Some(vec![String::new()]));
        assert_eq!(skill_token_filters("skill"), Some(vec![String::new()]));
        assert_eq!(skill_token_filters("skills"), Some(vec![String::new()]));
        // 无冒号过滤：`skillhum` → 关键字 `skill` + 前缀 `hum`。
        assert_eq!(
            skill_token_filters("skillhum"),
            Some(vec!["hum".to_string()])
        );
        // 冒号过滤。
        assert_eq!(
            skill_token_filters("skills:deb"),
            Some(vec!["deb".to_string()])
        );
        // 太短或非前缀：不识别。
        assert_eq!(skill_token_filters("sk"), None);
        assert_eq!(skill_token_filters("skq"), None);
    }

    #[test]
    fn direct_model_completion_lists_models() {
        let current = crate::ai::model_names::all()
            .first()
            .map(|m| crate::ai::model_names::model_handle(m))
            .expect("models.json is empty");
        CommandCompleter::set_current_model_hint(&current);

        let (_, candidates) = CommandCompleter::complete_for_line("/model ", 7);

        assert_eq!(
            candidates.first().map(|c| c.replacement.as_str()),
            Some(current.as_str())
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
        for level in ["minimal", "low", "medium", "high", "xhigh", "auto", "off"] {
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
