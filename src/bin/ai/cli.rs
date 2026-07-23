use std::path::Path;

use crate::ai::provider::ReasoningEffort;
use crate::commonw::utils::expanduser;
use crate::terminalw::parser::Parser as TermParser;

/// 解析后的 CLI 参数结构体
#[derive(Debug, Clone)]
pub(super) struct ParsedCli {
    pub(super) model: Option<String>,
    pub(super) agent: Option<String>,
    pub(super) clear: bool,
    pub(super) new_session: bool,
    pub(super) resume: bool,
    pub(super) session: Option<String>,
    pub(super) files: String,
    pub(super) args: Vec<String>,
    pub(super) list_tools: bool,
    pub(super) list_mcp_tools: bool,
    pub(super) list_skills: bool,
    pub(super) list_agents: bool,
    pub(super) no_skills: bool,
    pub(super) mcp_config: String,
    pub(super) help: bool,
    /// 是否在消费完 CLI prompt 后继续停留在交互式 REPL。
    /// 通过 `--interactive` / `-i` 开启；与 `-ns` 联用时，后续每轮都会继续走 notebook 检索问答。
    pub(super) interactive: bool,
    /// 用户对推理强度档位的会话级覆盖。语义说明：
    /// - `None`：未设置，遵循 [models.json](../../../models.json) 的模型默认值；
    /// - `Some(Some(level))`：强制使用该档位（minimal/low/medium/high）；
    /// - `Some(None)`：用户显式关闭，请求里不带 `reasoning_effort` 字段。
    ///
    /// `/model effort <x>` 与 `--reasoning-effort` 都写入此字段。
    pub(super) reasoning_effort_override: Option<Option<ReasoningEffort>>,
    /// 截断重试的兜底开关：为 `true` 时本轮请求强制关闭 thinking，忽略模型默认与
    /// 自动判定。对 always-thinking 模型（如 GLM 走 `enable_thinking`），单纯降
    /// `reasoning_effort` 无法抑制思考链占满输出预算；连续多次截断后置位此字段，
    /// 把整个思考预算让给可见内容。仅在 turn 内临时生效，turn 末统一恢复。
    pub(super) thinking_disabled_override: bool,
    /// 截断重试时的 `max_tokens` 自适应覆盖。当检测到「零输出截断」
    /// （`completion=0` + `finish_reason=length`）时，说明服务端拒绝了当前
    /// `max_tokens` 值（典型：relay/兼容层对超大 max_tokens 返回空响应）。
    /// 此时将 max_tokens 减半写入此字段，下一轮请求使用更小的值重试，
    /// 直到服务端接受。仅在 turn 内临时生效，turn 末统一恢复。
    /// - `None`：未设置，使用 `clamp_max_tokens_for_prompt` 的正常计算值；
    /// - `Some(n)`：用 `n` 作为 max_tokens 上限（仍受 clamp 的剩余窗口约束）。
    pub(super) max_tokens_override: Option<u32>,
    /// 是否只搜索 memo 类别的记录。
    /// 通过 `--note-search` / `-ns` 开启，用于快速查找用户手动记录的内容（如截图、笔记等）。
    /// 默认 false，即走正常的知识召回流程。
    pub(super) note_search: bool,
    /// 快速保存 memo 到知识库。
    /// 通过 `--note` 或 `-n` 指定内容，保存后直接退出。
    pub(super) note: Option<String>,
    /// 是否传入了 `--note` / `-n`（即便没有文本，例如只想保存剪贴板图片）。
    pub(super) note_flag: bool,
    /// 通过 `--note-delete` / `-nd <id>` 指定要删除的 memo 条目 ID。
    pub(super) note_delete: Option<String>,
    /// 通过 `--note-edit` / `-ne <描述>` 指定要修改的 memo：AI 匹配后在编辑器中改写。
    pub(super) note_edit: Option<String>,
    /// AI 驱动的知识库整理：读取全部条目 → 模型分析 → 执行整理。
    pub(super) consolidate_knowledge: bool,
    /// --generate-completions
    pub(super) generate_completions: bool,
    /// 是否以后台模式运行（`--background` / `-bg`）。
    /// 后台模式下会 detach 终端、把完整输出写入当前目录下 `<sessionid>.log`，
    /// 并提示 agent 在任务完成前不要停止。
    /// 可搭配位置参数（任务描述）使用；若未提供位置参数，则会在 daemonize 之前
    /// 交互式读取多行输入作为任务描述。
    pub(super) background: bool,
    /// --stop <session-id>：向后台任务的进程发送 SIGTERM 停止它。
    /// 后台模式会在当前目录下写入 <sessionid>.pid 文件，--stop 读取它并 kill。
    pub(super) stop_session: Option<String>,
}

/// `a` 内部 "/" / ":" 命令列表，用于 shell 补全。
const INTERNAL_COMMANDS: &[&str] = &[
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
    "/proc",
    ":proc",
];

const FILES_USAGE: &str = "input file names (repeat -f or use comma-separated list)";
const NOTE_SEARCH_USAGE: &str =
    "search knowledge base (memo category) and answer using positional prompt";
const GENERATE_COMPLETIONS_USAGE: &str =
    "generate shell completion script (bash/zsh/fish) and exit";
const REASONING_EFFORT_USAGE: &str = "reasoning effort: minimal | low | medium | high | xhigh | off (clears default; only effective on OpenAI/OpenRouter/OpenCode providers)";

fn build_cli_parser() -> TermParser {
    let mut parser = TermParser::new();
    register_cli_flags(&mut parser);
    parser
}

fn register_cli_flags(parser: &mut TermParser) {
    parser.add_bool(
        "clear",
        false,
        "clear specified session history (use with --session)",
    );
    parser.add_bool(
        "new-session",
        false,
        "force creating a new session and skip suspended-session auto resume",
    );
    parser.add_bool(
        "resume",
        false,
        "resume the suspended session bound to the current terminal",
    );
    parser.add_bool("list-tools", false, "list builtin tools and exit");
    parser.add_bool("list-mcp-tools", false, "list mcp tools and exit");
    parser.alias("list-mcp-servers", "list-mcp-tools");
    parser.add_bool("list-skills", false, "list skills and exit");
    parser.add_bool("list-agents", false, "list available agents and exit");
    parser.add_bool("no-skills", false, "disable loading all skills");
    parser.add_bool("help", false, "print help");
    parser.add_bool(
        "interactive",
        false,
        "stay in REPL after the initial CLI prompt",
    );
    parser.add_bool(
        "consolidate-knowledge",
        false,
        "AI-driven consolidation of all knowledge entries",
    );
    parser.add_bool("note-search", false, NOTE_SEARCH_USAGE);
    parser.add_bool("generate-completions", false, GENERATE_COMPLETIONS_USAGE);
    parser.add_bool(
        "background",
        false,
        "run in background: detach from terminal, log output to <sessionid>.log, and keep running after the shell exits (alias: -bg)",
    );
    parser.alias("bg", "background");
    parser.alias("i", "interactive");
    parser.alias("new", "new-session");
    parser.alias("r", "resume");
    parser.alias("ns", "note-search");
    parser.alias("h", "help");

    parser.add_string(
        "stop",
        "",
        "stop a background session by session id (e.g. a --stop <sessionid>)",
    );
    parser.add_string("model", "", "model name");
    parser.alias("m", "model");
    parser.add_string("agent", "", "agent name");
    parser.alias("a", "agent");
    parser.add_string("session", "", "session id");
    parser.alias("ss", "session");
    parser.add_string("files", "", FILES_USAGE);
    parser.alias("f", "files");
    parser.add_string("mcp-config", "", "mcp config json path override");
    parser.add_string("reasoning-effort", "", REASONING_EFFORT_USAGE);
    parser.alias("re", "reasoning-effort");

    parser.add_string("note", "", "save text as memo to knowledge base and exit");
    parser.alias("n", "note");
    parser.add_string(
        "note-delete",
        "",
        "describe a memo to delete; AI matches it, confirm to delete",
    );
    parser.alias("nd", "note-delete");
    parser.add_string(
        "note-edit",
        "",
        "describe a memo to edit; AI matches it, edit in editor and save",
    );
    parser.alias("ne", "note-edit");
}

fn rewrite_legacy_session_aliases(argv: &mut [String]) {
    for arg in argv {
        if arg == "--ss" || arg.starts_with("--ss=") {
            *arg = arg.replace("--ss", "--session");
        }
        if arg == "-ss" || arg.starts_with("-ss=") {
            *arg = arg.replace("-ss", "--session");
        }
    }
}

fn file_spec_exists(raw: &str) -> bool {
    let raw = raw.trim();
    if raw.is_empty() || raw.starts_with('-') {
        return false;
    }
    if raw.contains(',') {
        let mut saw_any = false;
        for part in raw.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            saw_any = true;
            let expanded = expanduser(part);
            if !Path::new(expanded.as_ref()).exists() {
                return false;
            }
        }
        return saw_any;
    }
    let expanded = expanduser(raw);
    Path::new(expanded.as_ref()).exists()
}

fn normalize_files_flags(argv: Vec<String>) -> Vec<String> {
    let mut normalized = Vec::with_capacity(argv.len());
    let mut collected_files: Vec<String> = Vec::new();
    let mut idx = 0usize;

    while idx < argv.len() {
        let arg = &argv[idx];
        if let Some(value) = arg.strip_prefix("--files=") {
            if !value.trim().is_empty() {
                collected_files.push(value.to_string());
            }
            idx += 1;
            while idx < argv.len() && file_spec_exists(&argv[idx]) {
                collected_files.push(argv[idx].clone());
                idx += 1;
            }
            continue;
        }
        if let Some(value) = arg.strip_prefix("-f=") {
            if !value.trim().is_empty() {
                collected_files.push(value.to_string());
            }
            idx += 1;
            while idx < argv.len() && file_spec_exists(&argv[idx]) {
                collected_files.push(argv[idx].clone());
                idx += 1;
            }
            continue;
        }
        if arg == "--files" || arg == "-f" {
            if let Some(value) = argv.get(idx + 1) {
                collected_files.push(value.clone());
                idx += 2;
                while idx < argv.len() && file_spec_exists(&argv[idx]) {
                    collected_files.push(argv[idx].clone());
                    idx += 1;
                }
            } else {
                normalized.push("--files".to_string());
                idx += 1;
            }
            continue;
        }
        normalized.push(arg.clone());
        idx += 1;
    }

    if !collected_files.is_empty() {
        normalized.push("--files".to_string());
        normalized.push(collected_files.join(","));
    }
    normalized
}

fn normalize_cli_argv(raw: &[String]) -> Vec<String> {
    let mut argv = if raw.len() > 1 {
        raw[1..].to_vec()
    } else {
        Vec::new()
    };
    rewrite_legacy_session_aliases(&mut argv);
    normalize_files_flags(argv)
}

impl Default for ParsedCli {
    fn default() -> Self {
        Self {
            model: None,
            agent: None,
            clear: false,
            new_session: false,
            resume: false,
            session: None,
            files: String::new(),
            args: Vec::new(),
            list_tools: false,
            list_mcp_tools: false,
            list_skills: false,
            list_agents: false,
            no_skills: false,
            mcp_config: String::new(),
            help: false,
            interactive: false,
            reasoning_effort_override: None,
            thinking_disabled_override: false,
            max_tokens_override: None,
            note_search: false,
            note: None,
            note_flag: false,
            note_delete: None,
            note_edit: None,
            consolidate_knowledge: false,
            generate_completions: false,
            background: false,
            stop_session: None,
        }
    }
}

/// 使用 terminalw::Parser 解析 CLI 参数
pub(super) fn parse_cli_args(args: impl Iterator<Item = String>) -> ParsedCli {
    let raw: Vec<String> = args.collect();
    if raw.is_empty() {
        return ParsedCli::default();
    }

    let mut parser = build_cli_parser();
    let argv = normalize_cli_argv(&raw);

    // 使用 terminalw 解析参数
    parser.parse_argv(&argv, &[]);

    // 构建 ParsedCli 结构体
    let mut cli = ParsedCli::default();

    // 处理 help（需要特殊处理，因为它是别名）
    cli.help = parser.contains_flag_strict("help") || parser.contains_flag_strict("h");
    cli.interactive = parser.contains_flag_strict("interactive");

    // 处理 model
    if parser.contains_flag_strict("model") {
        let val = parser.flag_value_or_default("model");
        if !val.trim().is_empty() {
            cli.model = Some(val);
        }
    }

    // 处理 agent
    if parser.contains_flag_strict("agent") {
        let val = parser.flag_value_or_default("agent");
        if !val.trim().is_empty() {
            cli.agent = Some(val);
        }
    }

    // 处理 clear（与 --session 联用，清空对应 session 的 history）
    cli.clear = parser.contains_flag_strict("clear");
    cli.new_session = parser.contains_flag_strict("new-session");
    cli.resume = parser.contains_flag_strict("resume");

    // 处理 session
    if parser.contains_flag_strict("session") {
        let val = parser.flag_value_or_default("session");
        cli.session = Some(val);
    }

    // 处理 files
    if parser.contains_flag_strict("files") {
        cli.files = parser.flag_value_or_default("files");
    }

    // 处理 consolidate-knowledge
    cli.consolidate_knowledge = parser.contains_flag_strict("consolidate-knowledge");

    // 处理 generate-completions
    cli.generate_completions = parser.contains_flag_strict("generate-completions");

    // 处理 background / -bg
    cli.background = parser.contains_flag_strict("background");

    // 处理 --stop <session-id>
    if parser.contains_flag_strict("stop") {
        let val = parser.flag_value_or_default("stop");
        cli.stop_session = Some(val.trim().to_string());
    }

    // 处理 list-tools
    cli.list_tools = parser.contains_flag_strict("list-tools");

    // 处理 list-mcp-tools
    cli.list_mcp_tools = parser.contains_flag_strict("list-mcp-tools");

    // 处理 list-skills
    cli.list_skills = parser.contains_flag_strict("list-skills");

    // 处理 list-agents
    cli.list_agents = parser.contains_flag_strict("list-agents");

    // 处理 no-skills
    cli.no_skills = parser.contains_flag_strict("no-skills");

    // 处理 note-search
    cli.note_search = parser.contains_flag_strict("note-search");

    // 处理 note
    if parser.contains_flag_strict("note") {
        cli.note_flag = true;
        let val = parser.flag_value_or_default("note");
        if !val.trim().is_empty() {
            cli.note = Some(val);
        }
    }

    // 处理 note-delete
    if parser.contains_flag_strict("note-delete") {
        let val = parser.flag_value_or_default("note-delete");
        cli.note_delete = Some(val.trim().to_string());
    }

    // 处理 note-edit
    if parser.contains_flag_strict("note-edit") {
        let val = parser.flag_value_or_default("note-edit");
        cli.note_edit = Some(val.trim().to_string());
    }

    // 处理 mcp-config
    if parser.contains_flag_strict("mcp-config") {
        cli.mcp_config = parser.flag_value_or_default("mcp-config");
    }

    // 处理 reasoning-effort
    if parser.contains_flag_strict("reasoning-effort") {
        let raw = parser.flag_value_or_default("reasoning-effort");
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            cli.reasoning_effort_override = Some(None);
        } else if matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "off" | "none" | "no" | "false" | "disable" | "disabled"
        ) {
            cli.reasoning_effort_override = Some(None);
        } else if let Some(level) = ReasoningEffort::parse(trimmed) {
            cli.reasoning_effort_override = Some(Some(level));
        } else {
            eprintln!(
                "[Warn] unknown --reasoning-effort value '{}'. Expected: minimal | low | medium | high | xhigh | off",
                trimmed
            );
        }
    }

    // 处理位置参数（prompt args）
    cli.args = parser.positional_args(false);

    cli
}

/// 打印帮助信息
pub(super) fn print_help() {
    let parser = build_cli_parser();
    println!("AI CLI - Interactive AI Assistant\n");

    // ── Quick Start ──────────────────────────────────────────────
    println!("USAGE:");
    println!("  a [OPTIONS] <prompt>          Run a one-shot prompt and exit");
    println!("  a [OPTIONS]                   Start interactive REPL\n");

    println!("QUICK START:");
    println!("  a fix the bug in main.rs      One-shot prompt");
    println!("  a -i \"explain this code\"      Start REPL after prompt");
    println!("  a -bg refactor the auth       Run in background (logs to <id>.log)");
    println!("  a --stop <session-id>         Stop a background session");
    println!("  a -n \"TODO: remember this\"    Save a memo and exit");
    println!("  a -ns \"meeting notes\"         Search memos with AI\n");

    // ── Options ──────────────────────────────────────────────────
    parser.print_defaults();

    // ── Session Behavior ─────────────────────────────────────────
    println!("\nSESSION BEHAVIOR:");
    println!("  Each process auto-creates a dedicated session (no shared history).");
    println!("  Launching `a` interactively resumes the sole suspended session, or");
    println!("  lets you choose when multiple are available.");
    println!("  Use --resume / --new-session / --session to control this.\n");

    // ── REPL ─────────────────────────────────────────────────────
    println!("REPL COMMANDS:");
    println!("  In interactive mode, type /help to see all available commands.\n");
}
/// 生成 shell 补全脚本并打印到 stdout。
/// `shell` 取值 "bash" | "zsh" | "fish"，不区分大小写。
/// 通过 --generate-completions 触发。
pub fn generate_completion_script(shell: &str) {
    let parser = build_cli_parser();
    let info = parser.collect_completion_info();

    let is_bool = |ty: &str| ty == "bool";
    let has_value = |ty: &str| ty == "string" || ty == "int" || ty == "float";

    match shell.to_ascii_lowercase().as_str() {
        "bash" => generate_bash(&info, is_bool, has_value),
        "zsh" => generate_zsh(&info, is_bool, has_value),
        "fish" => generate_fish(&info, is_bool, has_value),
        _ => {
            eprintln!("Unsupported shell: {shell}. Use: bash, zsh, or fish.");
            std::process::exit(1);
        }
    }
}

fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn model_selector_words() -> String {
    crate::ai::model_names::all()
        .into_iter()
        .map(crate::ai::model_names::model_handle)
        .collect::<Vec<_>>()
        .join(" ")
}

fn generate_bash(
    info: &[(String, String, String, Vec<String>)],
    _is_bool: fn(&str) -> bool,
    _has_value: fn(&str) -> bool,
) {
    let session_subcommands =
        crate::ai::driver::commands::session::CANONICAL_SESSION_SUBCOMMANDS.join(" ");
    println!("_a_completions() {{");
    println!("  local cur prev words cword");
    println!("  _get_comp_words_by_ref -n = cur prev words cword 2>/dev/null || true");
    println!();
    println!("  cur=\"${{COMP_WORDS[COMP_CWORD]}}\"");
    println!("  prev=\"${{COMP_WORDS[COMP_CWORD-1]}}\"");
    let flag_name = |name: &str| -> String {
        if name.len() > 1 {
            format!("--{}", name)
        } else {
            format!("-{}", name)
        }
    };
    let mut opts = String::new();
    for (name, _ty, _usage, aliases) in info {
        opts.push_str(&flag_name(name));
        opts.push(' ');
        for a in aliases {
            opts.push_str(&flag_name(a));
            opts.push(' ');
        }
    }
    // 追加 "/" / ":" 内部命令
    let mut all = opts;
    for cmd in INTERNAL_COMMANDS {
        all.push_str(cmd);
        all.push(' ');
    }
    // 子命令映射（与 zsh 分支保持一致）。当第一个参数是内部命令时，
    // 第二个参数补全对应的子命令而不是顶层 flags/命令列表。
    println!("  local usage_sub='today 7d 30d all daily trend days models help'");
    println!("  local checkpoint_sub='save list rollback delete help'");
    println!(
        "  local history_sub='full user assistant tool system grep rewind export copy last replay help 3 6 10 20'"
    );
    println!(
        "  local persona_sub='help list ls current cur create new use select switch delete del rm'"
    );
    println!(
        "  local session_sub={}",
        shell_single_quote(&session_subcommands)
    );
    println!("  local agent_sub='help list current use auto'");
    println!("  local model_sub='current list help effort'");
    println!(
        "  local model_selectors={}",
        shell_single_quote(&model_selector_words())
    );
    println!("  local effort_levels='minimal low medium high xhigh auto off'");
    println!();
    // COMP_WORDS[0] 是命令名 a，内部命令位于 COMP_WORDS[1]。
    println!("  if [ \"$COMP_CWORD\" -ge 2 ]; then");
    println!("    case \"${{COMP_WORDS[1]}}\" in");
    println!("      /usage|:usage)");
    println!("        COMPREPLY=($(compgen -W \"$usage_sub\" -- \"$cur\")); return 0 ;;");
    println!("      /checkpoint|:checkpoint|/cp|:cp)");
    println!("        COMPREPLY=($(compgen -W \"$checkpoint_sub\" -- \"$cur\")); return 0 ;;");
    println!("      /history|:history)");
    println!("        COMPREPLY=($(compgen -W \"$history_sub\" -- \"$cur\")); return 0 ;;");
    println!("      /sessions|:sessions|/ss|:ss)");
    println!("        COMPREPLY=($(compgen -W \"$session_sub\" -- \"$cur\")); return 0 ;;");
    println!("      /agent|:agent|/agents|:agents)");
    println!("        COMPREPLY=($(compgen -W \"$agent_sub\" -- \"$cur\")); return 0 ;;");
    println!("      /personas|:personas)");
    println!("        COMPREPLY=($(compgen -W \"$persona_sub\" -- \"$cur\")); return 0 ;;");
    println!("      /model|:model)");
    println!("        if [ \"$COMP_CWORD\" -eq 2 ]; then");
    println!(
        "          COMPREPLY=($(compgen -W \"$model_selectors $model_sub\" -- \"$cur\")); return 0"
    );
    println!("        fi");
    println!(
        "        if [ \"$COMP_CWORD\" -eq 3 ] && [ \"${{COMP_WORDS[2]}}\" = \"effort\" ]; then"
    );
    println!("          COMPREPLY=($(compgen -W \"$effort_levels\" -- \"$cur\")); return 0");
    println!("        fi");
    println!("        COMPREPLY=(); return 0 ;;");
    println!("    esac");
    println!("  fi");
    println!();
    println!("  COMPREPLY=($(compgen -W \"{}\" -- \"$cur\"))", all.trim());
    println!("  return 0");
    println!("}}");
    println!("complete -F _a_completions a");
}

fn generate_zsh(
    info: &[(String, String, String, Vec<String>)],
    is_bool: fn(&str) -> bool,
    _has_value: fn(&str) -> bool,
) {
    let session_subcommands =
        crate::ai::driver::commands::session::CANONICAL_SESSION_SUBCOMMANDS.join(" ");
    println!("#compdef a");
    println!();
    println!("_a() {{");
    println!("  local -a _a_args");
    println!();
    let emit_flag = |flag: &str, ty: &str, usage: &str| {
        let escaped = usage.replace('\'', "'\\''");
        if is_bool(ty) {
            format!("'{}[{}]'", flag, escaped)
        } else {
            format!("'{}:{}: '", flag, escaped)
        }
    };
    for (name, ty, usage, aliases) in info {
        let prefix = if name.len() > 1 { "--" } else { "-" };
        println!(
            "  _a_args+=({})",
            emit_flag(&format!("{}{}", prefix, name), ty, usage)
        );
        for a in aliases {
            let a_prefix = if a.len() > 1 { "--" } else { "-" };
            println!(
                "  _a_args+=({})",
                emit_flag(&format!("{}{}", a_prefix, a), ty, usage)
            );
        }
    }
    // 内部命令作为第一层 position args
    println!(
        "  local -a _a_internal_cmds=({})",
        INTERNAL_COMMANDS.join(" ")
    );
    println!();
    // 子命令映射
    println!("  local -a _a_usage_subcmds=(today 7d 30d all daily trend days models help)");
    println!("  local -a _a_checkpoint_subcmds=(save list rollback delete help)");
    println!(
        "  local -a _a_history_subcmds=(full user assistant tool system grep rewind export copy last replay help 3 6 10 20)"
    );
    println!("  local -a _a_session_subcmds=({session_subcommands})");
    println!("  local -a _a_agent_subcmds=(help list current use auto)");
    println!(
        "  local -a _a_persona_subcmds=(help list ls current cur create new use select switch delete del rm)"
    );
    println!("  local -a _a_model_subcmds=(current list help effort)");
    println!("  local -a _a_model_selectors=({})", model_selector_words());
    println!("  local -a _a_effort_levels=(minimal low medium high xhigh auto off)");
    println!("  local -a _a_model_entries");
    println!("  _a_model_entries=($_a_model_selectors $_a_model_subcmds)");
    println!();
    // 若正在补全内部命令的子命令，先按子命令处理并 return，
    // 避免回落到 flags / 顶层命令补全。
    //
    // zsh 在 `a /personas <TAB>` 这种“一级命令后刚输入一个空格”的场景里，
    // CURRENT 有时仍是 2，因此不能只依赖 `CURRENT >= 3`。这里同时兼容：
    // - CURRENT >= 3：已经进入第三个词；
    // - CURRENT == 2 且 LBUFFER 以空白结尾：刚输入完一级命令并跟了空格。
    // 注意：zsh 补全里 $words[1] 是命令名 a 本身，内部命令位于 $words[2]。
    println!("  local _a_subcmd_owner=''");
    println!("  if (( CURRENT >= 3 )); then");
    println!("    _a_subcmd_owner=\"$words[2]\"");
    println!("  elif (( CURRENT == 2 )) && [[ \"$LBUFFER\" == *[[:space:]] ]]; then");
    println!("    _a_subcmd_owner=\"$words[2]\"");
    println!("  fi");
    println!("  case \"$_a_subcmd_owner\" in");
    println!("      /usage|:usage)");
    println!("        _describe 'usage subcommand' _a_usage_subcmds && return");
    println!("        ;;");
    println!("      /checkpoint|:checkpoint|/cp|:cp)");
    println!("        _describe 'checkpoint subcommand' _a_checkpoint_subcmds && return");
    println!("        ;;");
    println!("      /history|:history)");
    println!("        _describe 'history subcommand' _a_history_subcmds && return");
    println!("        ;;");
    println!("      /sessions|:sessions|/ss|:ss)");
    println!("        _describe 'session subcommand' _a_session_subcmds && return");
    println!("        ;;");
    println!("      /agent|:agent|/agents|:agents)");
    println!("        _describe 'agent subcommand' _a_agent_subcmds && return");
    println!("        ;;");
    println!("      /personas|:personas)");
    println!("        _describe 'persona subcommand' _a_persona_subcmds && return");
    println!("        ;;");
    println!("      /model|:model)");
    println!("        if (( CURRENT >= 4 )) && [[ \"$words[3]\" == \"effort\" ]]; then");
    println!("          _describe 'reasoning effort' _a_effort_levels && return");
    println!("        fi");
    println!("        if (( CURRENT <= 3 )); then");
    println!("          _describe 'model selector or subcommand' _a_model_entries && return");
    println!("        fi");
    println!("        return");
    println!("        ;;");
    println!("  esac");
    println!();
    // _arguments: flags + 第一个 position arg 是内部命令。
    // 用 ($_a_internal_cmds) 展开数组成员作为候选；早期写成 (_a_internal_cmds)
    // 会把字面量字符串 "_a_internal_cmds" 当成唯一候选，导致 /usa<tab> 无反应。
    println!("  _arguments $_a_args ':first command:(($_a_internal_cmds))'");
    println!("}}");
    println!();
    println!("compdef _a a");
}

fn generate_fish(
    info: &[(String, String, String, Vec<String>)],
    is_bool: fn(&str) -> bool,
    _has_value: fn(&str) -> bool,
) {
    for (name, ty, usage, aliases) in info {
        let escaped = usage.replace('\'', "'\\''");
        if is_bool(ty) {
            println!("complete -c a -l '{name}' -d '{escaped}'");
            for a in aliases {
                if a.len() > 1 {
                    println!("complete -c a -l {a} -d '{escaped}'");
                }
            }
        } else {
            println!("complete -c a -l {name} -d '{escaped}' -r");
            for a in aliases {
                if a.len() > 1 {
                    println!("complete -c a -l {a} -d '{escaped}' -r");
                }
            }
        }
    }
    // 追加 "/" / ":" 内部命令
    for cmd in INTERNAL_COMMANDS {
        println!("complete -c a -a '{cmd}' -d 'internal command'");
    }
    println!(
        "complete -c a -n '__fish_seen_subcommand_from /model :model' -a '{}' -d 'model selector'",
        model_selector_words().replace('\'', "\\'")
    );
    println!(
        "complete -c a -n '__fish_seen_subcommand_from /model :model' -a 'current list help effort' -d 'model command'"
    );
    println!(
        "complete -c a -n '__fish_seen_subcommand_from effort' -a 'minimal low medium high xhigh auto off' -d 'reasoning effort'"
    );
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
