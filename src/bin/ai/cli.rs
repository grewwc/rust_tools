use std::path::Path;

use crate::ai::provider::ReasoningEffort;
use crate::commonw::utils::expanduser;
use crate::terminalw::parser::Parser as TermParser;

/// Parsed CLI argument struct.
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
    /// Whether to stay in the interactive REPL after consuming the CLI prompt.
    /// Enabled via `--interactive` / `-i`; when combined with `-ns`, every later
    /// turn continues with notebook retrieval Q&A.
    pub(super) interactive: bool,
    /// Session-level override of the reasoning effort tier. Semantics:
    /// - `None`: not set; follow the model default from the model registry
    ///   ([models/](../../../../models));
    /// - `Some(Some(level))`: force this tier (minimal/low/medium/high);
    /// - `Some(None)`: user explicitly disabled it; requests omit the
    ///   `reasoning_effort` field.
    ///
    /// Both `/model effort <x>` and `--reasoning-effort` write to this field.
    pub(super) reasoning_effort_override: Option<Option<ReasoningEffort>>,
    /// Fallback switch for truncation retries: when `true`, the current request
    /// force-disables thinking, ignoring the model default and automatic detection.
    /// For always-thinking models (e.g. GLM via `enable_thinking`), merely lowering
    /// `reasoning_effort` cannot stop the chain of thought from filling the output
    /// budget; after several consecutive truncations this flag is set to give the
    /// whole thinking budget to visible content. Takes effect only within the turn
    /// and is uniformly restored at turn end.
    pub(super) thinking_disabled_override: bool,
    /// Adaptive `max_tokens` override for truncation retries. On detecting a
    /// "zero-output truncation" (`completion=0` + `finish_reason=length`), the
    /// server rejected the current `max_tokens` value (typically: relay/compat
    /// layers return empty responses for very large max_tokens).
    /// This field then records max_tokens halved, and the next request retries
    /// with the smaller value until the server accepts it. Takes effect only
    /// within the turn and is uniformly restored at turn end.
    /// - `None`: not set; use the normal value computed by
    ///   `clamp_max_tokens_for_prompt`;
    /// - `Some(n)`: use `n` as the max_tokens cap (still bounded by the clamp's
    ///   remaining window).
    pub(super) max_tokens_override: Option<u32>,
    /// Whether to search only memo-category records.
    /// Enabled via `--note-search` / `-ns`, for quickly finding content the user
    /// recorded manually (screenshots, notes, etc.).
    /// Defaults to false, i.e. the normal knowledge recall flow.
    pub(super) note_search: bool,
    /// Quickly save a memo to the knowledge base.
    /// Content given via `--note` or `-n`; exits right after saving.
    pub(super) note: Option<String>,
    /// Whether `--note` / `-n` was passed (even without text, e.g. to save only a
    /// clipboard image).
    pub(super) note_flag: bool,
    /// Memo entry ID to delete, via `--note-delete` / `-nd <id>`.
    pub(super) note_delete: Option<String>,
    /// Memo to edit, via `--note-edit` / `-ne <description>`: AI matches it, then
    /// it is rewritten in an editor.
    pub(super) note_edit: Option<String>,
    /// AI-driven knowledge base consolidation: read all entries → model analysis
    /// → perform consolidation.
    pub(super) consolidate_knowledge: bool,
    /// --generate-completions
    pub(super) generate_completions: bool,
    /// Whether to run in background mode (`--background` / `-bg`).
    /// In background mode the terminal is detached, full output is written to
    /// `<sessionid>.log` in the current directory, and the agent is instructed
    /// not to stop until the task completes.
    /// Can be combined with a positional argument (task description); if none is
    /// given, multi-line input is read interactively before daemonize as the
    /// task description.
    pub(super) background: bool,
    /// --stop <session-id>: send SIGTERM to the background task's process to stop
    /// it. Background mode writes a <sessionid>.pid file in the current directory;
    /// --stop reads it and kills the process.
    pub(super) stop_session: Option<String>,
}

/// List of `a` internal "/" / ":" commands, used for shell completion.
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
    "/memo",
    ":memo",
    "/export",
    ":export",
    "/model",
    ":model",
    "/audit",
    ":audit",
    "/agent",
    ":agent",
    "/personas",
    ":personas",
    "/sessions",
    ":sessions",
    "/ss",
    ":ss",
    "/mark",
    ":mark",
    "/unmark",
    ":unmark",
    "/proc",
    ":proc",
];

const FILES_USAGE: &str = "input file names (repeat -f or use comma-separated list)";
const NOTE_SEARCH_USAGE: &str =
    "search knowledge base (memo category); with a positional query answer once, without it enter interactive memo search";
const GENERATE_COMPLETIONS_USAGE: &str =
    "generate shell completion script (bash/zsh/fish) and exit";
const REASONING_EFFORT_USAGE: &str = "reasoning effort: minimal | low | medium | high | xhigh | max | off (clears default; support depends on the selected model)";

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

/// Parse CLI arguments with terminalw::Parser.
pub(super) fn parse_cli_args(args: impl Iterator<Item = String>) -> ParsedCli {
    let raw: Vec<String> = args.collect();
    if raw.is_empty() {
        return ParsedCli::default();
    }

    let mut parser = build_cli_parser();
    let argv = normalize_cli_argv(&raw);

    // Parse arguments with terminalw.
    parser.parse_argv(&argv, &[]);

    // Build the ParsedCli struct.
    let mut cli = ParsedCli::default();

    // Handle help (needs special handling because it is an alias).
    cli.help = parser.contains_flag_strict("help") || parser.contains_flag_strict("h");
    cli.interactive = parser.contains_flag_strict("interactive");

    // Handle model.
    if parser.contains_flag_strict("model") {
        let val = parser.flag_value_or_default("model");
        if !val.trim().is_empty() {
            cli.model = Some(val);
        }
    }

    // Handle agent.
    if parser.contains_flag_strict("agent") {
        let val = parser.flag_value_or_default("agent");
        if !val.trim().is_empty() {
            cli.agent = Some(val);
        }
    }

    // Handle clear (combined with --session, clears the given session's history).
    cli.clear = parser.contains_flag_strict("clear");
    cli.new_session = parser.contains_flag_strict("new-session");
    cli.resume = parser.contains_flag_strict("resume");

    // Handle session.
    if parser.contains_flag_strict("session") {
        let val = parser.flag_value_or_default("session");
        cli.session = Some(val);
    }

    // Handle files.
    if parser.contains_flag_strict("files") {
        cli.files = parser.flag_value_or_default("files");
    }

    // Handle consolidate-knowledge.
    cli.consolidate_knowledge = parser.contains_flag_strict("consolidate-knowledge");

    // Handle generate-completions.
    cli.generate_completions = parser.contains_flag_strict("generate-completions");

    // Handle background / -bg.
    cli.background = parser.contains_flag_strict("background");

    // Handle --stop <session-id>.
    if parser.contains_flag_strict("stop") {
        let val = parser.flag_value_or_default("stop");
        cli.stop_session = Some(val.trim().to_string());
    }

    // Handle list-tools.
    cli.list_tools = parser.contains_flag_strict("list-tools");

    // Handle list-mcp-tools.
    cli.list_mcp_tools = parser.contains_flag_strict("list-mcp-tools");

    // Handle list-skills.
    cli.list_skills = parser.contains_flag_strict("list-skills");

    // Handle list-agents.
    cli.list_agents = parser.contains_flag_strict("list-agents");

    // Handle no-skills.
    cli.no_skills = parser.contains_flag_strict("no-skills");

    // Handle note-search.
    cli.note_search = parser.contains_flag_strict("note-search");

    // Handle note.
    if parser.contains_flag_strict("note") {
        cli.note_flag = true;
        let val = parser.flag_value_or_default("note");
        if !val.trim().is_empty() {
            cli.note = Some(val);
        }
    }

    // Handle note-delete.
    if parser.contains_flag_strict("note-delete") {
        let val = parser.flag_value_or_default("note-delete");
        cli.note_delete = Some(val.trim().to_string());
    }

    // Handle note-edit.
    if parser.contains_flag_strict("note-edit") {
        let val = parser.flag_value_or_default("note-edit");
        cli.note_edit = Some(val.trim().to_string());
    }

    // Handle mcp-config.
    if parser.contains_flag_strict("mcp-config") {
        cli.mcp_config = parser.flag_value_or_default("mcp-config");
    }

    // Handle reasoning-effort.
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
                "[Warn] unknown --reasoning-effort value '{}'. Expected: minimal | low | medium | high | xhigh | max | off",
                trimmed
            );
        }
    }

    // Handle positional arguments (prompt args).
    cli.args = parser.positional_args(false);

    cli
}

/// Print help information.
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
    println!("  a -ns \"meeting notes\"         Search memos with AI (one-shot)");
    println!("  a -ns                        Enter interactive memo search\n");

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
/// Generate the shell completion script and print it to stdout.
/// `shell` is "bash" | "zsh" | "fish", case-insensitive.
/// Triggered via --generate-completions.
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

/// List of `'selector|platform'` entries for model selectors + platform slugs,
/// used by bash/zsh to build "name + platform" two-stage smart completion.
fn model_meta_words() -> String {
    crate::ai::model_names::all()
        .into_iter()
        .map(|m| {
            let handle = crate::ai::model_names::model_handle(m);
            let platform = crate::ai::model_names::platform_slug(m);
            format!("'{}|{}'", handle, platform)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Skill name list for bash/zsh completion. Only names are fed, no descriptions:
/// the rank-1 two-stage branch of `_a_name_rank` treats the secondary field after
/// `|` as a match target, so putting a description in would cause "description
/// false hits" (interactive completion matches by name only; see
/// name_token_match_rank in prompt/completion.rs). Entries without `|` have an
/// empty secondary field, so rank-1 naturally never matches; rank-0 prefix and
/// rank-2 segment-wise matching still work by name.
fn skill_meta_words() -> String {
    crate::ai::skills::load_all_skills()
        .into_iter()
        .map(|s| shell_single_quote(&s.name))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Primary agent name list for bash/zsh completion. Like skill_meta_words, only
/// names are fed, no descriptions, to avoid rank-1 two-stage matching treating a
/// description as the secondary field ("description false hits").
fn agent_meta_words() -> String {
    crate::ai::agents::get_primary_agents(&crate::ai::agents::load_all_agents())
        .into_iter()
        .map(|a| shell_single_quote(&a.name))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Skill names (space-separated), used for fish prefix completion.
fn skill_names() -> String {
    crate::ai::skills::load_all_skills()
        .into_iter()
        .map(|s| s.name)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Primary agent names (space-separated), used for fish prefix completion.
fn agent_names() -> String {
    crate::ai::agents::get_primary_agents(&crate::ai::agents::load_all_agents())
        .into_iter()
        .map(|a| a.name.clone())
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
    // Append "/" / ":" internal commands.
    let mut all = opts;
    for cmd in INTERNAL_COMMANDS {
        all.push_str(cmd);
        all.push(' ');
    }
    // Subcommand mapping (kept in sync with the zsh branch). When the first
    // argument is an internal command, the second argument completes its
    // subcommands instead of the top-level flags/command list.
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
    println!("  local skill_sub='help list current use'");
    println!("  local -a _a_model_meta=({})", model_meta_words());
    println!("  local -a _a_skill_meta=({})", skill_meta_words());
    println!("  local -a _a_agent_meta=({})", agent_meta_words());
    println!("  local effort_levels='minimal low medium high xhigh max auto off'");
    println!();
    // Values of `--model`/`-m`: smart-match model names.
    println!("  if [[ \"$prev\" == \"--model\" || \"$prev\" == \"-m\" ]]; then");
    println!("    COMPREPLY=($(_a_name_matches \"$cur\" \"${{_a_model_meta[@]}}\"))");
    println!("    return 0");
    println!("  fi");
    println!("  if [[ \"$cur\" == --model=* || \"$cur\" == -m=* ]]; then");
    println!("    local _mdl_ip='--model='");
    println!("    [[ \"$cur\" == -m=* ]] && _mdl_ip='-m='");
    println!("    local -a _vals");
    println!("    _vals=($(_a_name_matches \"${{cur#$_mdl_ip}}\" \"${{_a_model_meta[@]}}\"))");
    println!("    COMPREPLY=(\"${{_vals[@]/#/$_mdl_ip}}\")");
    println!("    return 0");
    println!("  fi");
    // COMP_WORDS[0] is the command name a; internal commands live in COMP_WORDS[1].
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
    println!("        if [ \"$COMP_CWORD\" -ge 3 ] && [ \"${{COMP_WORDS[2]}}\" = \"use\" ]; then");
    println!("          COMPREPLY=($(_a_name_matches \"$cur\" \"${{_a_agent_meta[@]}}\"))");
    println!("        else");
    println!("          COMPREPLY=($(_a_name_matches \"$cur\" \"${{_a_agent_meta[@]}}\") $(compgen -W \"$agent_sub\" -- \"$cur\"))");
    println!("        fi");
    println!("        return 0 ;;");
    println!("      /skills|:skills|/skill|:skill)");
    println!("        COMPREPLY=($(_a_name_matches \"$cur\" \"${{_a_skill_meta[@]}}\") $(compgen -W \"$skill_sub\" -- \"$cur\")); return 0 ;;");
    println!("      /personas|:personas)");
    println!("        COMPREPLY=($(compgen -W \"$persona_sub\" -- \"$cur\")); return 0 ;;");
    println!("      /model|:model)");
    println!("        if [ \"$COMP_CWORD\" -eq 2 ]; then");
    println!(
        "          COMPREPLY=($(_a_name_matches \"$cur\" \"${{_a_model_meta[@]}}\") $(compgen -W \"$model_sub\" -- \"$cur\")); return 0"
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
    print!(
        "{}",
        r#"# 智能名称匹配：0=前缀, 1=名称+次字段两段式, 2=逐段前缀（与交互式补全一致）
_a_name_rank() {
  local query="$1" word="$2" platform="$3"
  case "$word" in
    "$query"*) printf '0\n'; return 0 ;;
  esac
  local qseg="${query##*[._/-]}"
  if [ -n "$qseg" ] && [ "$qseg" != "$query" ]; then
    local qhead="${query%[._/-]*}"
    if [ -n "$qhead" ]; then
      case "$word" in
        "$qhead"*)
          case "$platform" in
            "$qseg"*) printf '1\n'; return 0 ;;
          esac
          ;;
      esac
    fi
  fi
  local qs="${query//[._\/-]/ }" ws="${word//[._\/-]/ }"
  local -a qa wa
  read -r -a qa <<< "$qs"
  read -r -a wa <<< "$ws"
  local qi=0 w
  for w in "${wa[@]}"; do
    if (( qi < ${#qa[@]} )) && [[ "$w" == "${qa[$qi]}"* ]]; then
      qi=$((qi + 1))
      (( qi == ${#qa[@]} )) && { printf '2\n'; return 0; }
    fi
  done
  return 1
}
_a_name_matches() {
  local query="$1" entry w p r
  local -a r0 r1 r2
  shift
  for entry in "$@"; do
    # 无 `|` 的条目（skill/agent 只喂名字）没有次字段；若直接取 `${entry#*|}`，
    # 会返回整个名字，导致 rank-1 两段式误把名字当次字段命中。次字段只属于
    # 带 `|` 的 model 条目（rank-1 仅用于 model 的 name|platform）。
    if [[ "$entry" == *"|"* ]]; then
      w="${entry%%|*}"
      p="${entry#*|}"
    else
      w="$entry"
      p=""
    fi
    r="$(_a_name_rank "$query" "$w" "$p")"
    case "$r" in
      0) r0+=("$w") ;;
      1) r1+=("$w") ;;
      2) r2+=("$w") ;;
    esac
  done
  # 空数组在 bash<4.4 + set -u 下展开会报 unbound variable，逐组判空输出。
  (( ${#r0[@]} )) && printf '%s\n' "${r0[@]}"
  (( ${#r1[@]} )) && printf '%s\n' "${r1[@]}"
  (( ${#r2[@]} )) && printf '%s\n' "${r2[@]}"
}
"#
    );
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
    // Internal commands as first-level position args.
    println!(
        "  local -a _a_internal_cmds=({})",
        INTERNAL_COMMANDS.join(" ")
    );
    println!();
    // Subcommand mapping.
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
    println!("  local -a _a_skill_subcmds=(help list current use)");
    println!("  local -a _a_model_meta=({})", model_meta_words());
    println!("  local -a _a_skill_meta=({})", skill_meta_words());
    println!("  local -a _a_agent_meta=({})", agent_meta_words());
    println!("  local -a _a_effort_levels=(minimal low medium high xhigh max auto off)");
    print!(
        "{}",
        r#"  # 智能名称匹配：0=前缀, 1=名称+次字段两段式, 2=逐段前缀（与交互式补全一致）
  _a_name_rank() {
    local query="$1" word="$2" platform="$3"
    case "$word" in
      "$query"*) print -r -- 0; return 0 ;;
    esac
    local qseg="${query##*[._/-]}"
    if [[ -n "$qseg" && "$qseg" != "$query" ]]; then
      local qhead="${query%[._/-]*}"
      if [[ -n "$qhead" ]]; then
        case "$word" in
          "$qhead"*)
            case "$platform" in
              "$qseg"*) print -r -- 1; return 0 ;;
            esac
            ;;
        esac
      fi
    fi
    local qs="${query//[._\/-]/ }" ws="${word//[._\/-]/ }"
    local -a qa wa
    qa=(${(s: :)qs})
    wa=(${(s: :)ws})
    local qi=1 w
    for w in $wa; do
      if (( qi <= ${#qa} )) && [[ "$w" == "${qa[$qi]}"* ]]; then
        (( qi++ ))
        (( qi > ${#qa} )) && { print -r -- 2; return 0; }
      fi
    done
    return 1
  }
  # 通用候选生成：第一个参数是 `'name|secondary'` 条目数组的名字，
  # 按 rank 0/1/2 分组后 compadd（与交互式补全一致）。
  _a_name_candidates() {
    local -a _r0 _r1 _r2
    local entry w p r
    local -a _meta
    _meta=(${(@P)1})
    for entry in "${_meta[@]}"; do
      # 无 `|` 的条目（skill/agent 只喂名字）没有次字段；若直接取 `${entry#*|}`，
      # 会返回整个名字，导致 rank-1 两段式误把名字当次字段命中。次字段只属于
      # 带 `|` 的 model 条目（rank-1 仅用于 model 的 name|platform）。
      if [[ "$entry" == *"|"* ]]; then
        w="${entry%%|*}"
        p="${entry#*|}"
      else
        w="$entry"
        p=""
      fi
      r="$(_a_name_rank "$PREFIX" "$w" "$p")"
      case "$r" in
        0) _r0+=("$w") ;;
        1) _r1+=("$w") ;;
        2) _r2+=("$w") ;;
      esac
    done
    compadd -S ' ' -- $_r0 $_r1 $_r2
  }
"#
    );
    // Values of `--model`/`-m`: smart-match model names.
    println!("  if [[ \"$words[CURRENT-1]\" == --model* || \"$words[CURRENT-1]\" == -m* ]]; then");
    println!("    _a_name_candidates _a_model_meta");
    println!("    return");
    println!("  fi");
    // `--model=VALUE` / `-m=VALUE` (equals form): set `--model=` as IPREFIX so
    // only the part after the equals sign is matched, keeping the prefix on
    // candidate insertion.
    println!("  if [[ \"$words[CURRENT]\" == --model=* || \"$words[CURRENT]\" == -m=* ]]; then");
    println!("    local _mdl_ip='--model='");
    println!("    [[ \"$words[CURRENT]\" == -m=* ]] && _mdl_ip='-m='");
    println!("    PREFIX=\"${{PREFIX#$_mdl_ip}}\"");
    println!("    IPREFIX=\"$_mdl_ip\"");
    println!("    _a_name_candidates _a_model_meta");
    println!("    return");
    println!("  fi");
    println!();
    // If a subcommand of an internal command is being completed, handle it as a
    // subcommand first and return, instead of falling back to flags / top-level
    // command completion.
    //
    // In zsh, in scenarios like `a /personas <TAB>` ("a single space typed right
    // after a first-level command"), CURRENT can still be 2, so we cannot rely on
    // `CURRENT >= 3` alone. Accept both:
    // - CURRENT >= 3: already inside the third word;
    // - CURRENT == 2 with LBUFFER ending in whitespace: a first-level command was
    //   just typed and followed by a space.
    // Note: in zsh completion $words[1] is the command name a itself; internal
    // commands live in $words[2].
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
    println!("        if (( CURRENT >= 4 )) && [[ \"$words[3]\" == \"use\" ]]; then");
    println!("          _a_name_candidates _a_agent_meta && return");
    println!("        fi");
    println!("        if (( CURRENT <= 3 )); then");
    println!("          _a_name_candidates _a_agent_meta");
    println!("          local _ag_sub");
    println!("          for _ag_sub in \"${{_a_agent_subcmds[@]}}\"; do");
    println!("            [[ \"$_ag_sub\" == \"$PREFIX\"* ]] && compadd -S ' ' -- \"$_ag_sub\"");
    println!("          done");
    println!("          return");
    println!("        fi");
    println!("        return");
    println!("        ;;");
    println!("      /skills|:skills|/skill|:skill)");
    println!("        if (( CURRENT <= 3 )); then");
    println!("          _a_name_candidates _a_skill_meta");
    println!("          local _sk_sub");
    println!("          for _sk_sub in \"${{_a_skill_subcmds[@]}}\"; do");
    println!("            [[ \"$_sk_sub\" == \"$PREFIX\"* ]] && compadd -S ' ' -- \"$_sk_sub\"");
    println!("          done");
    println!("          return");
    println!("        fi");
    println!("        return");
    println!("        ;;");
    println!("      /personas|:personas)");
    println!("        _describe 'persona subcommand' _a_persona_subcmds && return");
    println!("        ;;");
    println!("      /model|:model)");
    println!("        if (( CURRENT >= 4 )) && [[ \"$words[3]\" == \"effort\" ]]; then");
    println!("          _describe 'reasoning effort' _a_effort_levels && return");
    println!("        fi");
    println!("        if (( CURRENT <= 3 )); then");
    println!("          _a_name_candidates _a_model_meta");
    println!("          local _m_sub");
    println!("          for _m_sub in \"${{_a_model_subcmds[@]}}\"; do");
    println!("            [[ \"$_m_sub\" == \"$PREFIX\"* ]] && compadd -S ' ' -- \"$_m_sub\"");
    println!("          done");
    println!("          return");
    println!("        fi");
    println!("        return");
    println!("        ;;");
    println!("  esac");
    println!();
    // _arguments: flags + the first position arg is an internal command.
    // Expand the array members as candidates via ($_a_internal_cmds); an early
    // version wrote (_a_internal_cmds) instead, which made the literal string
    // "_a_internal_cmds" the only candidate, leaving /usa<tab> unresponsive.
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
    // Append "/" / ":" internal commands.
    for cmd in INTERNAL_COMMANDS {
        println!("complete -c a -a '{cmd}' -d 'internal command'");
    }
    println!(
        "complete -c a -n '__fish_seen_subcommand_from /model :model' -a '{}' -d 'model selector'",
        model_selector_words().replace('\'', "\\'")
    );
    // Values of `--model`/`-m`: model selector. fish's `-a` already prefix-filters,
    // so segmented matching like bash/zsh is not possible, but prefix scenarios
    // such as `a --model dee<TAB>` work.
    println!(
        "complete -c a -l model -r -a '{}' -d 'model selector'",
        model_selector_words().replace('\'', "\\'")
    );
    println!(
        "complete -c a -s m -r -a '{}' -d 'model selector'",
        model_selector_words().replace('\'', "\\'")
    );
    println!(
        "complete -c a -n '__fish_seen_subcommand_from /model :model' -a 'current list help effort' -d 'model command'"
    );
    println!(
        "complete -c a -n '__fish_seen_subcommand_from effort' -a 'minimal low medium high xhigh max auto off' -d 'reasoning effort'"
    );
    // skill/agent name completion: fish's `-a` can only match by prefix and cannot
    // do the two-stage smart matching of bash/zsh (platform limitation), so only
    // the names themselves are completed here.
    println!(
        "complete -c a -n '__fish_seen_subcommand_from /skills :skills /skill :skill' -a '{}' -d 'skill name'",
        skill_names().replace('\'', "\\'")
    );
    println!(
        "complete -c a -n '__fish_seen_subcommand_from /agent :agent /agents :agents' -a '{}' -d 'agent name'",
        agent_names().replace('\'', "\\'")
    );
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
