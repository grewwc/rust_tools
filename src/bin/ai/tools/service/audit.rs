/// 审计模块：负责命令安全校验（注入面检查 + 分段黑名单）。
///
/// 职责分离：
/// - 本模块只做「校验」，不做「执行」。
/// - `execute_command` 调用 `validate_execute_command()` 入口即可。
/// - 便于独立测试、独立演进安全策略，不与执行逻辑耦合。
use crate::ai::config_schema::AiConfig;

// ---------------------------------------------------------------------------
// 配置辅助
// ---------------------------------------------------------------------------

/// 从配置读取用户自定义的被禁程序列表。
fn config_blocked_commands() -> Vec<String> {
    let raw = crate::commonw::configw::get_all_config().get(AiConfig::SANDBOX_BLOCKED_COMMANDS, "");
    raw.split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Shell 分段（按 `&&` / `||` / `;` / `|` / `\n` 拆分链式命令）
// ---------------------------------------------------------------------------

/// 以 unquoted 的 `&&` / `||` / `;` / `|` / `\n` 为分隔符，把整条命令拆成
/// 独立段。单/双引号内的分隔符不会触发拆分；单引号 heredoc body 内的换行也会
/// 被跳过（heredoc body 内容是字面量，不应被拆分逻辑消费）。
pub(super) fn split_unquoted_segments(command: &str) -> Vec<String> {
    let bytes = command.as_bytes();
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut pending_heredocs: Vec<HereDocSpec> = Vec::new();

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

// ---------------------------------------------------------------------------
// Heredoc 解析辅助
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
// Shell 词法分析（用于单段校验）
// ---------------------------------------------------------------------------

pub(super) fn tokenize_shell_words(command: &str) -> Vec<String> {
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
// 命令索引解析（跳过选项，定位到真正要执行的程序）
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

// ---------------------------------------------------------------------------
// Git 子命令拦截
// ---------------------------------------------------------------------------

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
    // `git rm` 从工作树物理删除文件，不可恢复；`git rm --cached` 仅移除 index，
    // 但直接拦截比逐参数分析更安全。需要删除文件时请用 `trash` 等安全工具。
    (
        "rm",
        "git rm is blocked by sandbox policy; use trash or similar safe-delete tool",
    ),
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

/// `git` 子命令中「会不可逆地丢弃或删除未提交工作」的判定。
///
/// 与 `BLOCKED_GIT_SUBCOMMANDS`（push/stash 这类全局禁止）不同，下面这些子命令在
/// 部分参数组合下是无害的（如 `git switch` 分支切换、`git restore --staged` 取消暂存），
/// 因此只在「确实会销毁未提交改动（工作树/暂存区改动、未跟踪文件）」时才拦截，避免误伤
/// 正常流程。命中返回拒绝原因。
///
/// 覆盖用户要求的「禁止 `git checkout --` 以及任何会把当前未提交文件删掉且无法回滚的命令」。
fn blocked_git_destructive(command_tokens: &[String]) -> Option<&'static str> {
    let idx = git_subcommand_index(command_tokens)?;
    // 子命令名大小写不敏感，统一小写后匹配。
    let sub = command_tokens[idx].to_ascii_lowercase();
    let rest = &command_tokens[idx + 1..];
    match sub.as_str() {
        // `git checkout <branch>`（无 `--`、无 `--force`/`-B`）由 git 自身保护未提交改动，
        // 冲突时直接报错，放行；其余会丢弃工作树改动的形态一律拦截。
        // 注意：短选项区分大小写，`-B`(force-create/重置分支) 与 `-b`(创建分支) 不同，
        // 必须区分；`-f`/`--force` 强切分支同样会丢改动。
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
            // 无 `--`、无 force 的情况下，用启发式判断是否为文件路径：
            // 1. `.`/`..`/`./`/`../` 明显是路径形态，直接拦截。
            // 2. 参数末尾含有文件扩展名（如 `src/main.rs`、`package.json`），
            //    大概率是文件路径而非分支名；分支名/标签的 `.` 后缀通常是数字
            //   （如 `v1.2.3`），不会全是字母，不会误拦。
            let looks_like_path = rest.iter().any(|t| {
                if t.starts_with('-') {
                    return false;
                }
                t == "."
                    || t == ".."
                    || t.starts_with("./")
                    || t.starts_with("../")
                    || t.rfind('.').map_or(false, |pos| {
                        // 跳过 `.` 开头的隐藏文件（.gitignore 等），已在上面覆盖。
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
        // `git switch -f`/`--force`/`--discard-changes` 强制切分支会丢弃未提交改动；
        // `-C`/`--force-create` 在分支已存在时强制重置并切分支，同样丢弃。创建新分支
        //（`-c`/`--create`，不带 force）是安全操作，放行。短选项区分大小写：`-C` ≠ `-c`。
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
        // `git restore` 默认恢复工作树，会丢弃未提交的工作树改动；仅「只 `--staged`」是
        // 安全的取消暂存（工作树不动、可回滚）。
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
                // 仅取消暂存、工作树不动，可回滚，放行。
                return None;
            }
            Some("git restore discards uncommitted working-tree changes")
        }
        // `git reset --hard`/`--merge`/`--keep` 会丢弃工作树/暂存区改动；
        // `--soft` 与默认（mixed）保留工作树，放行。
        "reset" => {
            if rest
                .iter()
                .any(|t| matches!(t.as_str(), "--hard" | "--merge" | "--keep"))
            {
                return Some("git reset --hard/--merge/--keep discards uncommitted changes");
            }
            None
        }
        // `git clean -f` 删除未跟踪文件，不可回滚；`-n`(dry-run) 等不实际删除，放行。
        "clean" => {
            if rest.iter().any(|t| {
                t == "-f"
                    || t == "--force"
                    // 合并短选项（如 `-fd` 即 `-f -d`）里包含 `-f`，同样会真正删除文件。
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
// Shell 注入面检查
// ---------------------------------------------------------------------------

/// 判断 `(` 是否紧跟在被转义的 `$` 之后（`\$(`）。`\$` 把 `$` 转义为字面量后，
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

/// 查找 shell 结构中与 `open_idx` 处左括号配对的右括号。
/// 引号内和反斜杠转义后的括号按字面量处理。
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

/// 检查命令字符串中是否存在不安全的 shell 注入面。
///
/// 本函数是 **shell-specific** 的安全检查，只应对经 shell 解释执行的命令
///（即 `execute_command` 工具）调用。对于非 shell 的工具（如 `write_file`、
/// `apply_patch` 等纯字符串操作），不应应用本检查——它们是直接写入文件系统
/// 或做文本替换，不会把参数喂给 shell 解释，`<<` / `$()` 只是普通文本。
///
/// 命令替换（`$(...)` / `` `...` ``）可能在运行时生成 program 名，继续禁止。
/// 进程替换 `<(...)` / `>(...)` 则递归校验内部命令后放行，避免误伤 diff/sort 等常见用法。
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
        // 进程替换 `<(...)` / `>(...)` 只在引号外有 shell 语义。递归校验其中的完整
        // 命令，而不是无差别禁止；这样安全命令可用，`<(rm ...)` 等仍会被原有规则拦截。
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

// ---------------------------------------------------------------------------
// 分段级校验入口
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

/// 展开参数开头的 `~` / `$HOME`。home 本身及其子路径属于正常开发访问；仅拒绝
/// 通过 `..` 从 home 目录向外逃逸。`~other` 不在 shell 的当前用户 home 语义内，
/// 留给 shell 自身处理。
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

/// 对单段命令做 program/参数级黑名单校验。
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
    // ---- tilde / $HOME 逃逸检测 ----
    // home 与其子路径可正常访问；仅拒绝 `~/..` / `$HOME/..` 向外逃逸。
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
        // 用原始大小写 token 调用，以区分 `-B`/`-b`、`-C`/`-c` 这类大小写敏感的短选项。
        if let Some(reason) = blocked_git_destructive(raw_command_tokens) {
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
            if let Some(reason) = blocked_git_destructive(&raw_command_tokens[idx..]) {
                return Err(reason.to_string());
            }
        }
    }

    Ok(())
}

// =========================================================================
// 公开入口
// =========================================================================

/// 校验一条完整命令（含链式 `&&` / `||`）的安全性。
///
/// 这是审计模块的唯一公开入口，`execute_command` 调用此函数即可。
pub(crate) fn validate_execute_command(command: &str) -> Result<(), String> {
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

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::{split_unquoted_segments, tokenize_shell_words, validate_no_injection_surface};

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
        // `git rm` 现在被 BLOCKED_GIT_SUBCOMMANDS 拦截（不可恢复删除）。
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
        // `git rm` 现在被 BLOCKED_GIT_SUBCOMMANDS 拦截。
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

    // ---- tilde / $HOME 逃逸检测 ----

    #[test]
    fn home_paths_are_allowed() {
        assert!(validate("ls ~").is_ok());
        assert!(validate("cat ~/.gitconfig").is_ok());
        assert!(validate("cat $HOME/.cargo/config.toml").is_ok());
    }

    #[test]
    fn tilde_escape_to_parent_dir_blocked() {
        // cwd=/Users/bytedance/rust_tools → ~/.. 遍历到 /Users/bytedance → /Users → /
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
}
