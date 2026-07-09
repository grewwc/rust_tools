use serde_json::Value;
use std::process::Output;

use crate::ai::config_schema::AiConfig;
use crate::ai::tools::storage::command_runner;

const MAX_COMMAND_OUTPUT_CHARS: usize = 16_000;

/// 内置默认超时与上限（秒），可被 sandbox 配置覆盖。
const DEFAULT_COMMAND_TIMEOUT_SECS: u64 = 60;
const DEFAULT_COMMAND_TIMEOUT_MAX_SECS: u64 = 300;

/// 读取用户在 `ai.sandbox.blocked_commands` 中追加的禁用程序名（小写、去空白）。
fn config_blocked_commands() -> Vec<String> {
    let raw = crate::commonw::configw::get_all_config().get(AiConfig::SANDBOX_BLOCKED_COMMANDS, "");
    raw.split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// 返回 `execute_command` 的 (默认超时, 超时上限)，由 sandbox 配置覆盖。
/// 非法/缺省值回退到内置常量；上限至少为 1 秒且不小于默认值。
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

/// 纯函数：把请求的超时秒数夹在 `[1, max]` 范围内，缺省时用 `default`。
fn resolve_command_timeout(requested: Option<u64>, default: u64, max: u64) -> u64 {
    requested.unwrap_or(default).clamp(1, max)
}

fn truncate_chars(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let mut output = String::with_capacity(max_chars + 32);
    for (idx, ch) in content.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        output.push(ch);
    }
    output.push_str("\n... (truncated)");
    output
}

/// 把命令按 shell 控制操作符 (`;`, `&&`, `||`, `|`, `&`) 在引号外切分。
/// 引号识别仅做最小判断：单引号内的字符全部视为字面量；双引号同样视为字面量
/// （注意：bash 中双引号内 `$()`/`\`` 仍可生效，所以命令替换的检测需要在
/// **未拆分的整段命令**上独立完成，而不是依赖本函数）。
///
/// 这里的目的是把 `validate_execute_command` 已有的 program/参数黑名单
/// 套到链式命令的每一段上，例如 `echo ok && rm -rf /` 在原实现中只验
/// `echo`，分段后会再验 `rm`，从而拦下尾段。
fn split_unquoted_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let bytes = command.as_bytes();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut pending_heredocs: Vec<HereDocSpec> = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            current.push(b as char);
            if b == b'\'' {
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            current.push(b as char);
            // 转义字符在双引号内仅对少数字符有效；这里粗粒度跳过下一个字节即可
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
                // 引号外的反斜杠转义：保留两个字节
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
            // 双字符操作符 `&&` / `||`
            b'&' if i + 1 < bytes.len() && bytes[i + 1] == b'&' => {
                segments.push(std::mem::take(&mut current));
                i += 2;
            }
            b'|' if i + 1 < bytes.len() && bytes[i + 1] == b'|' => {
                segments.push(std::mem::take(&mut current));
                i += 2;
            }
            // 单字符分隔符
            b';' | b'|' | b'&' => {
                segments.push(std::mem::take(&mut current));
                i += 1;
            }
            b'\n' => {
                segments.push(std::mem::take(&mut current));
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
    segments.push(current);
    segments
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

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

fn tokenize_shell_words(command: &str) -> Vec<String> {
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

fn shell_c_option_present(program: &str, tokens: &[String]) -> bool {
    let is_shell = matches!(program, "bash" | "sh" | "zsh" | "ksh" | "dash");
    // 脚本解释器同样支持 `-c` / `-e` 直接传入并执行代码字符串，会绕过分段黑名单验证。
    let is_interpreter = matches!(
        program,
        "python" | "python3" | "perl" | "ruby" | "node" | "php" | "awk" | "lua"
    );
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
        if is_shell && (tok == "-c" || tok == "--command") {
            return true;
        }
        if is_interpreter && (tok == "-c" || tok == "-e") {
            return true;
        }
        i += 1;
    }
    false
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

/// 解析 `git` 子命令：跳过出现在子命令之前的 git 全局选项，返回子命令 token
/// 在 `command_tokens`（首元素为 `git` 本身）中的索引。
///
/// 部分 git 全局选项会消费紧随其后的一个参数：`-C <path>`、`-c <name>=<value>`、
/// `--git-dir <path>`、`--work-tree <path>`、`--namespace <name>`、`--exec-path <path>`。
/// `=` 附着形式（如 `--git-dir=/repo`、`-C/repo`）不额外消费 token。
/// `--` 之后的第一个 token 视为子命令。
fn git_subcommand_index(command_tokens: &[String]) -> Option<usize> {
    const VALUE_CONSUMING_LONG: &[&str] =
        &["--git-dir", "--work-tree", "--namespace", "--exec-path"];
    let mut i = 1usize;
    while i < command_tokens.len() {
        let tok = command_tokens[i].as_str();
        if tok == "--" {
            return command_tokens.get(i + 1).map(|_| i + 1);
        }
        // 非 option token 即为子命令。
        if !tok.starts_with('-') || tok == "-" {
            return Some(i);
        }
        // `=` 附着形式自带值，无需额外消费下一 token。
        if tok.contains('=') {
            i += 1;
            continue;
        }
        let lower = tok.to_ascii_lowercase();
        // `-C` / `-c` 都消费下一参数（lower 后二者均为 `-c`）。
        if lower == "-c" || VALUE_CONSUMING_LONG.contains(&lower.as_str()) {
            i += 2;
            continue;
        }
        i += 1;
    }
    None
}

/// 被安全策略硬拦的 `git` 子命令及其拒绝原因。
///
/// `git` 本身不在 `denied_programs` 里（status/log/diff 等子命令无害且必要），
/// 只对下列子命令硬拦。全局选项变体（如 `git -C /repo push`）同样命中。
const BLOCKED_GIT_SUBCOMMANDS: &[(&str, &str)] = &[
    // 避免把本地提交推送到远端仓库。
    ("push", "git push is blocked by sandbox policy"),
    // `git stash`（含 pop/drop/clear 等子动作）会暂存甚至丢弃工作区改动，
    // 容易丢失未提交的工作，禁止 agent 自主调用。
    ("stash", "git stash is blocked by sandbox policy"),
];

/// 若 `git` 子命令命中拦截名单，返回对应的拒绝原因。
fn blocked_git_subcommand(command_tokens: &[String]) -> Option<&'static str> {
    let idx = git_subcommand_index(command_tokens)?;
    let sub = command_tokens[idx].to_ascii_lowercase();
    BLOCKED_GIT_SUBCOMMANDS
        .iter()
        .find(|(name, _)| *name == sub)
        .map(|(_, reason)| *reason)
}

/// =========================================================================
/// Shell 注入面检查
/// =========================================================================
///
/// 本函数是 **shell-specific** 的安全检查，只应对经 shell 解释执行的命令
///（即 `execute_command` 工具）调用。对于非 shell 的工具（如 `write_file`、
/// `apply_patch` 等纯字符串操作），不应应用本检查——它们是直接写入文件系统
/// 或做文本替换，不会把参数喂给 shell 解释，`<<` / `$()` 只是普通文本。
///
/// 当前本函数仅在 `validate_execute_command` 内部被调用，天然只作用在 shell
/// 执行路径上。
///
/// 拦截那些不被分段化策略覆盖的注入面：命令替换
/// （`$(...)` / `` `...` ``）、进程替换。它们在正当 dev 工作流里几乎不出现，但可以一举绕过任何
/// program/参数级黑名单（典型样例：`$(echo rm) -rf /tmp/foo`）。
/// `(` 是否紧跟在被转义的 `$` 之后（`\$(`）。`\$` 把 `$` 转义为字面量后，
/// `$(...)` 不再构成命令替换；残留的 `(` 在 bash 里只会触发语法错误（不会执行子
/// shell），因此不视为注入面。判断方式：`(` 前一字符是 `$`，且紧贴 `$` 之前的
/// 连续反斜杠数为奇数（奇数 ⇒ `$` 被转义）。
fn paren_follows_escaped_dollar(bytes: &[u8], i: usize) -> bool {
    // bytes[i] == b'('；需要前一字符是 `$`。
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
        // 单引号内的所有内容都是字面量，shell 不会解析 $() / `。
        if in_single {
            if b == b'\'' {
                in_single = false;
            }
            i += 1;
            continue;
        }
        // 双引号内 `<(` / `>(` 只是普通文本，但 `$()` / `` `...` `` 仍可能生效，因此后面继续做拦截。
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
        // 命令替换 `$(`
        if b == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            // 算术展开 `$(( ... ))` 不执行任何命令，是无害的（典型：`echo $((RANDOM % 20))`），
            // 不应被命令替换规则误杀。压入算术深度后继续向内扫描——这样真正内嵌的
            // 命令替换（如 `$(( $(whoami) ))` 里的 `$(`）仍会在后续迭代被命中拦下，
            // 而结尾的 `))` 与内部分组括号由下方的 arith_depth 分支正确放行。
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
        // 进程替换 `<(...)` / `>(...)` 只在引号外有 shell 语义。
        if !in_double && (b == b'<' || b == b'>') && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            return Err("process substitution `<(...)` / `>(...)` is not allowed".to_string());
        }
        // 未引用的 `(` / `)` / `{` / `}` 开启子 shell 或命令分组（如 `(rm -rf /tmp)`、
        // `{ rm -rf /tmp; }`），会绕过分段黑名单验证。
        // `$(` / `$((` / `<(` / `>(` 已在上方单独处理，此处拦截裸 `(` / `)` / `{` / `}`。
        // 但算术展开 `$(( ... ))` 内部的 `(` / `)` 只是分组括号、`))` 用于闭合算术展开，
        // 都不构成子 shell；`\$(` 中 `$` 已被转义为字面量，残留 `(` 只是 bash 语法错误，
        // 同样不执行子 shell——这两种情况都要放行。
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

pub fn validate_execute_command(command: &str) -> Result<(), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("empty command".to_string());
    }

    // 第一道防线：阻断 shell 注入面（命令替换 / 进程替换）。
    // 这些放过去，分段黑名单就是摆设。
    validate_no_injection_surface(command)?;

    // 第二道防线：把链式命令拆段，对每一段都跑一次 program/参数黑名单。
    // 这样 `echo ok && rm -rf /` 会在第二段被 `rm` 黑名单命中。
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
    // 对程序路径取 basename，防止 `/bin/rm`、`./rm` 等绝对/相对路径绕过黑名单。
    let program_basename = std::path::Path::new(program)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(program);
    // 后续所有比较统一使用 basename，确保 `/bin/rm` 与 `rm` 被同等对待。
    let program = program_basename;
    let extra_blocked = config_blocked_commands();
    let normalize_path = |path: &std::path::Path| {
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
    };

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
        // "scp",
        // 绕过手段：`eval` / `source` / `.` 会把后续字符串当 shell 代码再次
        // 解释，等于把验证完全 bypass 掉。
        "eval",
        "source",
        ".",
        // 反向 shell / 网络监听工具，正当 dev 流程几乎不会用，留着风险大于收益。
        "nc",
        "ncat",
        "netcat",
        "telnet",
        "socat",
    ];
    if denied_programs.contains(&program) {
        return Err(format!("program '{program}' is blocked"));
    }

    // 用户可通过 `ai.sandbox.blocked_commands` 追加自定义黑名单程序。
    if extra_blocked.iter().any(|p| p == program) {
        return Err(format!(
            "program '{program}' is blocked by sandbox policy (ai.sandbox.blocked_commands)"
        ));
    }

    // 安全策略：拦截有破坏性/越权风险的 `git` 子命令（见 `BLOCKED_GIT_SUBCOMMANDS`）。
    // `git` 本身不在 denied_programs 里（status/log/diff 等子命令无害且必要），
    // 只对名单内的子命令硬拦。全局选项变体（`git -C /repo push`）同样命中。
    if program == "git" {
        if let Some(reason) = blocked_git_subcommand(command_tokens) {
            return Err(reason.to_string());
        }
    }

    // 拦下 `bash -c "..."` / `sh -c` / `zsh -c` 这种"二次解释"形式。
    // 直接执行脚本（`bash script.sh`）仍然允许，避免把脚本参数里的 `-c` 误判为 shell 选项。
    if shell_c_option_present(program, command_tokens) {
        return Err(format!(
            "shell `{program} -c ...` re-interprets a string as shell code; \
             run the literal command directly instead"
        ));
    }

    // `find` 的 `-delete` / `-exec*` / `-ok*` 只有在作为真正 primary 时才有危险语义。
    // 若它们只是 `-name '-delete'` 之类 pattern 参数，不应误拦。
    if program == "find" {
        if let Some(flag) = find_has_blocked_exec_semantics(command_tokens) {
            return Err(format!(
                "find primary '{flag}' mutates files or executes commands and is blocked"
            ));
        }
    }

    // 常见包装器会把后续 token 当作真正要执行的程序；只检查"将被执行的那个程序名"，
    // 避免把普通内容参数（如 `printf '%s' rm` 里的 `rm`）误判为危险命令。
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
        // 间接执行被拦的 `git` 子命令（如 `env git push`、`xargs git stash`）同样需要拦截，
        // 否则可借包装器绕过直接的检查。
        if nested == "git" {
            if let Some(reason) = blocked_git_subcommand(&command_tokens[idx..]) {
                return Err(reason.to_string());
            }
        }
    }

    Ok(())
}

fn format_command_output(output: Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let stdout_trimmed = stdout.trim();
    let stderr_trimmed = stderr.trim();

    if output.status.success() {
        let combined = if stdout_trimmed.is_empty() {
            stderr_trimmed.to_string()
        } else if stderr_trimmed.is_empty() {
            stdout_trimmed.to_string()
        } else {
            format!("{stdout_trimmed}\n{stderr_trimmed}")
        };
        truncate_chars(combined.trim(), MAX_COMMAND_OUTPUT_CHARS)
    } else {
        truncate_chars(
            &format!(
                "Exit code: {}\n{}\n{}",
                output.status.code().unwrap_or(-1),
                stdout_trimmed,
                stderr_trimmed
            ),
            MAX_COMMAND_OUTPUT_CHARS,
        )
    }
}

fn execute_command_inner<F>(args: &Value, on_chunk: F) -> Result<String, String>
where
    F: FnMut(&[u8]),
{
    let command = args["command"].as_str().ok_or("Missing command")?;
    let cwd = args["cwd"].as_str().filter(|dir| !dir.trim().is_empty());
    let (default_timeout, max_timeout) = config_command_timeout_bounds();
    let timeout = resolve_command_timeout(args["timeout"].as_u64(), default_timeout, max_timeout);

    if let Err(reason) = validate_execute_command(command) {
        return Err(format!("Command blocked: {reason}"));
    }

    let output = command_runner::run_command_streaming(command, cwd, timeout, on_chunk)?;
    Ok(format_command_output(output))
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
        resolve_command_timeout, split_unquoted_segments, tokenize_shell_words,
        validate_no_injection_surface,
    };

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
    fn injection_blocks_process_substitution() {
        assert!(validate_no_injection_surface("diff <(echo a) <(echo b)").is_err());
    }

    #[test]
    fn injection_allows_clean_command() {
        assert!(validate_no_injection_surface("cargo build --release").is_ok());
    }

    #[test]
    fn injection_treats_single_quoted_as_literal() {
        // 整段在单引号内的 `$()` 是字面量，bash 不会展开。
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
        // 修复前：仅验 `echo`，整体放行。
        // 修复后：第二段会命中 `rm` 默认拦截。
        let err = validate("echo ok && rm -rf /").unwrap_err();
        assert!(err.contains("rm"), "expected rm blocked, got: {err}");
    }

    #[test]
    fn blocks_rm_even_within_current_directory() {
        let err = validate("rm -rf ./target").unwrap_err();
        assert!(err.contains("rm"), "expected rm blocked, got: {err}");
    }

    #[test]
    fn blocks_shell_rm_with_home_and_glob_expansion() {
        let err = validate("rm -rf ~/.zcompdump*").unwrap_err();
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
        // `bash run.sh` 不是二次解释，正常工作流应继续允许。
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
    fn allows_arithmetic_expansion() {
        // 算术展开 `$(( ... ))` 不执行命令，是无害的，应放行。
        assert!(validate("echo $((RANDOM % 20 + 1))").is_ok());
        assert!(validate("echo $((1 + 2 * 3))").is_ok());
    }

    #[test]
    fn blocks_command_substitution_nested_in_arithmetic() {
        // 算术展开内部仍内嵌真正的命令替换 `$(...)`，必须被拦下。
        let err = validate("echo $(( $(whoami) + 1 ))").unwrap_err();
        assert!(
            err.contains("command substitution"),
            "expected nested $(...) blocked, got: {err}"
        );
    }

    #[test]
    fn allows_subcommand_patterns_that_resemble_blocked_programs() {
        // `git rm` / `docker rm` / `git mv` 等：rm/mv 是子命令，不是直接调用 /bin/rm。
        // 不应被参数级黑名单误杀。
        assert!(validate("git rm file.txt").is_ok());
        assert!(validate("git mv old.txt new.txt").is_ok());
        assert!(validate("docker rm my_container").is_ok());
        assert!(validate("docker rmi my_image").is_ok());
        assert!(validate("npm rm some-package").is_ok());
        assert!(validate("pip install rsync").is_ok());
    }

    #[test]
    fn blocks_git_push_in_all_common_forms() {
        // 直接调用
        assert!(validate("git push").is_err());
        assert!(validate("git push origin main").is_err());
        assert!(validate("git push --force").is_err());
        assert!(validate("git push --force-with-lease origin").is_err());
        assert!(validate("git push -u origin main").is_err());
        assert!(validate("git push origin --tags").is_err());
        // 全局选项变体（-C / -c / --git-dir 等）不应绕过检查
        assert!(validate("git -C /repo push").is_err());
        assert!(validate("git -C /repo push origin main").is_err());
        assert!(validate("git -c user.email=a@b.c push").is_err());
        assert!(validate("git --git-dir=/repo push").is_err());
        assert!(validate("git --git-dir /repo push").is_err());
        assert!(validate("git --no-pager push").is_err());
        // 大小写不敏感
        assert!(validate("git PUSH origin").is_err());
        // 绝对/相对路径仍按 basename 识别
        assert!(validate("/usr/bin/git push").is_err());
        // 链式命令中的 push 段也要被拦
        assert!(validate("git status && git push").is_err());
        assert!(validate("git push && echo done").is_err());
        // 间接包装器执行 git push 同样拦截
        assert!(validate("env git push").is_err());
        assert!(validate("env FOO=1 git push origin main").is_err());
        assert!(validate("xargs git push").is_err());
        assert!(validate("nohup git push").is_err());
        assert!(validate("command git push").is_err());
    }

    #[test]
    fn git_non_push_subcommands_remain_allowed() {
        // 只有 push 被拦，其余 git 子命令保持可用
        assert!(validate("git status").is_ok());
        assert!(validate("git log --oneline -5").is_ok());
        assert!(validate("git diff").is_ok());
        assert!(validate("git diff --cached").is_ok());
        assert!(validate("git -C /repo status").is_ok());
        assert!(validate("git -C /repo log --oneline").is_ok());
        assert!(validate("git add -A").is_ok());
        assert!(validate("git commit -m msg").is_ok());
        // `push` 作为普通文本参数（非子命令）不应误拦
        assert!(validate("echo git push").is_ok());
        assert!(validate("printf '%s' push").is_ok());
    }

    #[test]
    fn blocks_git_stash_in_all_common_forms() {
        // 直接调用（含 stash 的各种子动作）
        assert!(validate("git stash").is_err());
        assert!(validate("git stash list").is_err());
        assert!(validate("git stash pop").is_err());
        assert!(validate("git stash drop").is_err());
        assert!(validate("git stash clear").is_err());
        assert!(validate("git stash push -m wip").is_err());
        // 全局选项变体（-C / -c / --git-dir 等）不应绕过检查
        assert!(validate("git -C /repo stash").is_err());
        assert!(validate("git -c user.email=a@b.c stash").is_err());
        assert!(validate("git --git-dir=/repo stash").is_err());
        assert!(validate("git --no-pager stash").is_err());
        // 大小写不敏感
        assert!(validate("git STASH").is_err());
        // 绝对/相对路径仍按 basename 识别
        assert!(validate("/usr/bin/git stash").is_err());
        // 链式命令中的 stash 段也要被拦
        assert!(validate("git status && git stash").is_err());
        assert!(validate("git stash && echo done").is_err());
        // 间接包装器执行 git stash 同样拦截
        assert!(validate("env git stash").is_err());
        assert!(validate("xargs git stash").is_err());
        assert!(validate("nohup git stash").is_err());
        assert!(validate("command git stash").is_err());
        // `stash` 作为普通文本参数（非子命令）不应误拦
        assert!(validate("echo git stash").is_ok());
        assert!(validate("printf '%s' stash").is_ok());
    }

    #[test]
    fn shell_literal_rm_text_remains_allowed() {
        assert!(validate("echo 'rm -rf ~/.zcompdump*'").is_ok());
    }

    #[test]
    fn blocks_exec_flags_that_run_subsequent_args_as_commands() {
        // find -exec/-execdir/-ok/-okdir 会将后续参数当命令执行，必须拦截
        assert!(validate("find . -exec rm {} +").is_err());
        assert!(validate("find . -execdir chmod 777 {} \\;").is_err());
        assert!(validate("find /tmp -ok rm {} \\;").is_err());
        assert!(validate("find . -okdir mv {} /tmp/ \\;").is_err());
        // 无害的 find 用法不受影响
        assert!(validate("find . -name '*.rs' -type f").is_ok());
        assert!(validate("find . -delete").is_err());
        assert!(validate("find . -empty -delete").is_err());
        assert!(validate(r#"find . "-exec" rm {} +"#).is_err());
        assert!(validate(r#"find . -name "-delete" -print"#).is_ok());
        assert!(validate(r#"find . -name "-exec" -print"#).is_ok());
        assert!(validate(r#"find . -printf "-delete\n""#).is_ok());
        // 子命令/包名场景不受影响（这些不含危险 primary）
        assert!(validate("git rm file.txt").is_ok());
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
        // 使用单引号包裹的字面量 `$()` 应放行（不会被 shell 展开）
        assert!(validate("echo 'literal $(x)'").is_ok());
    }
}
