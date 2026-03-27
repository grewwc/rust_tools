use clap::{ArgAction, Parser};

pub(super) const DEFAULT_NUM_HISTORY: usize = 4;

#[derive(Parser, Debug)]
#[command(
    about = "AI CLI compatible with go_tools executable/ai/a.go",
    after_help = "Session\n  默认每个进程自动创建独立 session（不会和其它窗口串 history）\n  --session <id>            指定 session id\n  --session                 不指定 id，等同于自动创建新 session\n  --clear --session <id>    清空指定 session 的 history\n\nInteractive\n  /help                     打印交互命令帮助\n  /sessions                 列出所有 sessions\n  /sessions current         查看当前 session\n  /sessions new             新建并切换到新 session\n  /sessions use <id>        切换 session\n  /sessions delete <id>     删除 session（若删除当前 session，会自动切到新 session）\n  /sessions clear-all       删除全部 sessions\n"
)]
pub(super) struct Cli {
    #[arg(long, default_value_t = DEFAULT_NUM_HISTORY, help = "number of history")]
    pub(super) history: usize,

    #[arg(
        short = 'm',
        long = "model",
        default_value = "",
        help = "model name. kimi-k2.5[0], qwen3.5-plus[1], qwen3-max[2], qwen3-max[3], qwen3-coder-plus[4], deepseek-v3.2[5], qwen3.5-flash[6] (thinking mode uses glm-5 for [5])"
    )]
    pub(super) model: String,

    #[arg(
        long = "multi-line",
        visible_alias = "mul",
        action = ArgAction::SetTrue,
        help = "input with multline"
    )]
    pub(super) multi_line: bool,

    #[arg(long, action = ArgAction::SetTrue, help = "clear history")]
    pub(super) clear: bool,

    #[arg(
        long,
        visible_alias = "ss",
        num_args = 0..=1,
        default_missing_value = "",
        help = "session id. empty means create a new session for this process."
    )]
    pub(super) session: Option<String>,

    #[arg(short = 'c', action = ArgAction::SetTrue, help = "prepend content in clipboard")]
    pub(super) clipboard: bool,

    #[arg(
        short = 'f',
        default_value = "",
        help = "input file names. seprated by comma."
    )]
    pub(super) files: String,

    #[arg(
        short = 'o',
        long = "out",
        num_args = 0..=1,
        default_missing_value = "output.md",
        help = "write output to file. default is output.md"
    )]
    pub(super) out: Option<String>,

    #[arg(short = 't', action = ArgAction::SetTrue, help = "use thinking model. default: false.")]
    pub(super) thinking: bool,

    #[arg(short = 's', action = ArgAction::SetTrue, help = "short output")]
    pub(super) short_output: bool,

    // Model selectors are single-digit shortcuts that mirror the trailing " -<digit>" syntax.
    #[arg(short = '0', action = ArgAction::SetTrue, help = "select kimi-k2.5")]
    pub(super) model_0: bool,

    #[arg(short = '1', action = ArgAction::SetTrue, help = "select qwen3.5-plus")]
    pub(super) model_1: bool,

    #[arg(short = '2', action = ArgAction::SetTrue, help = "select qwen3-max")]
    pub(super) model_2: bool,

    #[arg(short = '3', action = ArgAction::SetTrue, help = "select qwen3-max")]
    pub(super) model_3: bool,

    #[arg(short = '4', action = ArgAction::SetTrue, help = "select qwen3-coder-plus")]
    pub(super) model_4: bool,

    #[arg(short = '5', action = ArgAction::SetTrue, help = "select deepseek-v3.2 / glm-5")]
    pub(super) model_5: bool,

    #[arg(short = '6', action = ArgAction::SetTrue, help = "select qwen3.5-flash")]
    pub(super) model_6: bool,

    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub(super) args: Vec<String>,

    #[arg(long, action = ArgAction::SetTrue, help = "list builtin tools and exit")]
    pub(super) list_tools: bool,

    #[arg(
        long,
        visible_alias = "list-mcp-servers",
        action = ArgAction::SetTrue,
        help = "list mcp tools and exit"
    )]
    pub(super) list_mcp_tools: bool,

    #[arg(long, action = ArgAction::SetTrue, help = "list skills and exit")]
    pub(super) list_skills: bool,

    #[arg(long, default_value = "", help = "mcp config json path override")]
    pub(super) mcp_config: String,
}

pub(super) fn normalize_single_dash_long_opts(args: impl Iterator<Item = String>) -> Vec<String> {
    args.map(|arg| {
        let bytes = arg.as_bytes();
        if bytes.len() > 2 && bytes[0] == b'-' && bytes[1] != b'-' && bytes[1].is_ascii_alphabetic()
        {
            format!("-{arg}")
        } else {
            arg
        }
    })
    .collect()
}
