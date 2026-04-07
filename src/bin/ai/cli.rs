use crate::terminalw::parser::Parser as TermParser;

pub(super) const DEFAULT_NUM_HISTORY: usize = 256;

/// 解析后的 CLI 参数结构体
#[derive(Debug, Clone)]
pub(super) struct ParsedCli {
    pub(super) history: usize,
    pub(super) model: Option<String>,
    pub(super) agent: Option<String>,
    pub(super) multi_line: bool,
    pub(super) clear: bool,
    pub(super) session: Option<String>,
    pub(super) clipboard: bool,
    pub(super) files: String,
    pub(super) out: Option<String>,
    pub(super) thinking: bool,
    pub(super) short_output: bool,
    pub(super) args: Vec<String>,
    pub(super) list_tools: bool,
    pub(super) list_mcp_tools: bool,
    pub(super) list_skills: bool,
    pub(super) list_agents: bool,
    pub(super) no_skills: bool,
    pub(super) mcp_config: String,
    pub(super) help: bool,
}

impl Default for ParsedCli {
    fn default() -> Self {
        Self {
            history: DEFAULT_NUM_HISTORY,
            model: None,
            agent: None,
            multi_line: false,
            clear: false,
            session: None,
            clipboard: false,
            files: String::new(),
            out: None,
            thinking: false,
            short_output: false,
            args: Vec::new(),
            list_tools: false,
            list_mcp_tools: false,
            list_skills: false,
            list_agents: false,
            no_skills: false,
            mcp_config: String::new(),
            help: false,
        }
    }
}

/// 使用 terminalw::Parser 解析 CLI 参数
pub(super) fn parse_cli_args(args: impl Iterator<Item = String>) -> ParsedCli {
    let raw: Vec<String> = args.collect();
    if raw.is_empty() {
        return ParsedCli::default();
    }

    // 创建 terminalw parser
    let mut parser = TermParser::new();

    // 定义所有 bool 选项
    parser.add_bool("clear", false, "clear history");
    parser.add_bool("multi-line", false, "input with multline");
    parser.alias("mul", "multi-line");
    parser.add_bool("clipboard", false, "prepend content in clipboard");
    parser.add_bool("thinking", false, "use thinking model");
    parser.alias("t", "thinking");
    parser.add_bool("short-output", false, "short output");
    parser.alias("s", "short-output");
    parser.add_bool("list-tools", false, "list builtin tools and exit");
    parser.add_bool("list-mcp-tools", false, "list mcp tools and exit");
    parser.alias("list-mcp-servers", "list-mcp-tools");
    parser.add_bool("list-skills", false, "list skills and exit");
    parser.add_bool("list-agents", false, "list available agents and exit");
    parser.add_bool("no-skills", false, "disable loading all skills");
    parser.add_bool("help", false, "print help");
    parser.alias("h", "help");

    // 定义所有 string/int 选项
    parser.add_int("history", DEFAULT_NUM_HISTORY as i32, "number of history");
    parser.add_string("model", "", "model name");
    parser.alias("m", "model");
    parser.add_string("agent", "", "agent name");
    parser.alias("a", "agent");
    parser.add_string("session", "", "session id");
    parser.alias("ss", "session");
    parser.add_string("files", "", "input file names");
    parser.alias("f", "files");
    parser.add_string("out", "", "write output to file");
    parser.alias("o", "out");
    parser.add_string("mcp-config", "", "mcp config json path override");

    // 解析 argv（跳过 program name）
    let mut argv: Vec<String> = if raw.len() > 1 {
        raw[1..].to_vec()
    } else {
        Vec::new()
    };

    // 预处理：将 --ss 转换为 --session，避免与 -s 冲突
    // 这是必要的，因为 terminalw::Parser 的布尔簇检测会将 -ss 分解为 -s + -s
    for arg in &mut argv {
        if arg == "--ss" || arg.starts_with("--ss=") {
            *arg = arg.replace("--ss", "--session");
        }
        if arg == "-ss" || arg.starts_with("-ss=") {
            *arg = arg.replace("-ss", "--session");
        }
    }

    // 使用 terminalw 解析参数
    parser.parse_argv(&argv, &[]);

    // 构建 ParsedCli 结构体
    let mut cli = ParsedCli::default();

    // 处理 help（需要特殊处理，因为它是别名）
    cli.help = parser.contains_flag_strict("help") || parser.contains_flag_strict("h");

    // 处理 history
    if parser.contains_flag_strict("history") {
        cli.history = parser.flag_value_i32("history") as usize;
    }

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

    // 处理 multi-line
    cli.multi_line = parser.contains_flag_strict("multi-line");

    // 处理 clear
    cli.clear = parser.contains_flag_strict("clear");

    // 处理 session
    if parser.contains_flag_strict("session") {
        let val = parser.flag_value_or_default("session");
        cli.session = Some(val);
    }

    // 处理 clipboard
    cli.clipboard = parser.contains_flag_strict("clipboard");

    // 处理 files
    if parser.contains_flag_strict("files") {
        cli.files = parser.flag_value_or_default("files");
    }

    // 处理 out
    if parser.contains_flag_strict("out") {
        let val = parser.flag_value_or_default("out");
        if !val.trim().is_empty() {
            cli.out = Some(val);
        }
    }

    // 处理 thinking
    cli.thinking = parser.contains_flag_strict("thinking");

    // 处理 short-output
    cli.short_output = parser.contains_flag_strict("short-output");

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

    // 处理 mcp-config
    if parser.contains_flag_strict("mcp-config") {
        cli.mcp_config = parser.flag_value_or_default("mcp-config");
    }

    // 处理位置参数（prompt args）
    cli.args = parser.positional_args(false);

    cli
}

/// 打印帮助信息
pub(super) fn print_help() {
    let mut parser = TermParser::new();

    parser.add_bool("clear", false, "clear history");
    parser.add_bool("multi-line", false, "input with multline");
    parser.alias("mul", "multi-line");
    parser.add_bool("clipboard", false, "prepend content in clipboard");
    parser.add_bool("thinking", false, "use thinking model");
    parser.alias("t", "thinking");
    parser.add_bool("short-output", false, "short output");
    parser.alias("s", "short-output");
    parser.add_bool("list-tools", false, "list builtin tools and exit");
    parser.add_bool("list-mcp-tools", false, "list mcp tools and exit");
    parser.alias("list-mcp-servers", "list-mcp-tools");
    parser.add_bool("list-skills", false, "list skills and exit");
    parser.add_bool("list-agents", false, "list available agents and exit");
    parser.add_bool("no-skills", false, "disable loading all skills");
    parser.add_bool("help", false, "print help");
    parser.alias("h", "help");

    parser.add_int("history", DEFAULT_NUM_HISTORY as i32, "number of history");
    parser.add_string("model", "", "model name");
    parser.alias("m", "model");
    parser.add_string("agent", "", "agent name");
    parser.alias("a", "agent");
    parser.add_string("session", "", "session id");
    parser.alias("ss", "session");
    parser.add_string("files", "", "input file names");
    parser.alias("f", "files");
    parser.add_string("out", "", "write output to file");
    parser.alias("o", "out");
    parser.add_string("mcp-config", "", "mcp config json path override");

    println!("AI CLI compatible with go_tools executable/ai/a.go");
    println!();
    parser.print_defaults();
    println!();
    println!("Agent");
    println!("  --agent <name>            start with specified agent (build/plan/explore)");
    println!("  --list-agents             list available agents and exit");
    println!();
    println!("Session");
    println!("  默认每个进程自动创建独立 session（不会和其它窗口串 history）");
    println!("  --session <id>            指定 session id");
    println!("  --session                 不指定 id，等同于自动创建新 session");
    println!("  --clear --session <id>    清空指定 session 的 history");
    println!();
    println!("Interactive");
    println!("  /help                     打印交互命令帮助");
    println!("  /agents                   list available agents");
    println!("  /agents use <name>        switch to an agent");
    println!("  /sessions                 列出所有 sessions");
    println!("  /sessions current         查看当前 session");
    println!("  /sessions new             新建并切换到新 session");
    println!("  /sessions use <id>        切换 session");
    println!(
        "  /sessions delete <id>     删除 session（若删除当前 session，会自动切到新 session）"
    );
    println!("  /sessions clear-all       删除全部 sessions");
    println!("  /sessions export <id> [output.md]    导出指定 session 为 Markdown");
    println!("  /sessions export-current [output.md] 导出当前 session 为 Markdown");
    println!("  /sessions export-last [output.md]    导出最新 session 为 Markdown");
}
