pub use rust_tools::cmd;
pub use rust_tools::commonw;
pub use rust_tools::strw;
pub use rust_tools::{cw::Trie, terminalw};
use std::io::{self, Write};

#[path = "re/memo/mod.rs"]
mod memo;

#[path = "re/features/mod.rs"]
mod features;

use features::*;
use std::sync::Arc;

const DEFAULT_TXT_OUTPUT_NAME: &str = "output.txt";

fn main() {
    let argv = normalize_legacy_single_dash_long_args(std::env::args());

    // intercept --generate-completions (meta-operation, requires only the parser)
    if let Some(idx) = argv.iter().position(|a| a == "--generate-completions") {
        let shell = if let Some(val) = argv.get(idx + 1) {
            val.clone()
        } else if let Ok(s) = std::env::var("SHELL") {
            s.rsplit('/').next().unwrap_or("bash").to_string()
        } else {
            "bash".to_string()
        };
        generate_completion_script(&shell);
        return;
    }

    let (mut parser, cli) = parse_cli_and_parser(argv);

    let db = open_backend(&cli).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let db = Arc::new(db);
    let cli = Arc::new(cli);

    let to_binary = cli.binary || cli.b;
    let all = (cli.all
        || (cli.a && cli.add_tag.is_none() && cli.del_tag.is_none() && !cli.list_tags))
        && !to_binary;
    let prefix = cli.prefix || cli.pre;
    let only_tags = cli.short;
    let use_vscode = cli.code;
    let list_special = cli.sp || all;
    let reverse = cli.reverse && !(cli.prev || cli.remote || cli.prefix || cli.pre);
    let include_finished = cli.include_finished || all;
    let record_limit = if all { i64::MAX } else { cli.n };

    let out_name = cli
        .out
        .as_deref()
        .map(|s| {
            if s.trim().is_empty() {
                DEFAULT_TXT_OUTPUT_NAME.to_string()
            } else {
                s.trim().to_string()
            }
        })
        .unwrap_or_default();

    let tag_query_raw = get_tag_query_raw(&cli);
    let tag_query_non_empty = tag_query_raw
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty());
    let list_tags_and_order_by_time = order_by_time(&cli, tag_query_raw.as_deref());

    let title_query = first_non_empty(&[cli.title.as_str(), cli.content.as_str()]).to_string();

    let clipboard = cli.clipboard;
    let ctx = Arc::new(features::register::ReContext {
        db: Arc::clone(&db),
        cli: Arc::clone(&cli),
        prefix,
        use_vscode,
        list_special,
        reverse,
        all,
        include_finished,
        record_limit,
        out_name,
        clipboard,
        to_binary,
        only_tags,
        tag_query_raw,
        tag_query_non_empty,
        list_tags_and_order_by_time,
        title_query,
    });
    features::register::register_all(&mut parser, ctx);
    let _ = parser.execute_first();
}

/// Generate shell completion script for `re`.
fn generate_completion_script(shell: &str) {
    match shell.to_ascii_lowercase().as_str() {
        "bash" => generate_bash_completion(),
        "zsh" => generate_zsh_completion(),
        _ => {
            eprintln!("Unsupported shell: {shell}. Supported: bash, zsh.");
            std::process::exit(1);
        }
    }
}

/// Build a word list of all flag names from the re parser for shell completion.
fn build_flag_words(p: &terminalw::Parser) -> Vec<String> {
    let info = p.collect_completion_info();
    let mut trie = Trie::new();
    for (name, _ty, _usage, aliases) in &info {
        if name.len() > 1 {
            trie.insert(&format!("--{name}"));
        } else {
            trie.insert(&format!("-{name}"));
        }
        for alias in aliases {
            if alias.len() == 1 {
                trie.insert(&format!("-{alias}"));
            } else {
                trie.insert(&format!("--{alias}"));
            }
        }
    }
    trie.words_with_prefix("")
}

/// Generate bash completion script.
fn generate_bash_completion() {
    let parser = features::core::build_re_parser();
    let flags = build_flag_words(&parser);
    let flag_list = flags.join(" ");

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, r#"# bash completion for re"#);
    let _ = writeln!(out, r#"_re() {{"#);
    let _ = writeln!(out, r#"  local cur prev words cword"#);
    let _ = writeln!(
        out,
        r#"  _get_comp_words_by_ref -n = cur prev words cword 2>/dev/null || true"#
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        r#"  if [ "$prev" = "-t" ] || [ "$prev" = "--tag" ]; then"#
    );
    let _ = writeln!(
        out,
        r#"    COMPREPLY=( $(compgen -W "$("${{COMP_WORDS[0]}}" --complete-tags "$cur" 2>/dev/null)" -- "$cur") )"#
    );
    let _ = writeln!(out, r#"    return 0"#);
    let _ = writeln!(out, r#"  fi"#);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        r#"  COMPREPLY=( $(compgen -W "{flag_list}" -- "$cur") )"#,
        flag_list = flag_list
    );
    let _ = writeln!(out, r#"  return 0"#);
    let _ = writeln!(out, r#"}}"#);
    let _ = writeln!(out, r#"complete -F _re re"#);
}

/// Generate zsh completion script.
fn generate_zsh_completion() {
    let parser = features::core::build_re_parser();
    let info = parser.collect_completion_info();

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "#compdef re\n");
    let _ = writeln!(out, "_re() {{");
    let _ = writeln!(out, "  local -a _re_args");
    for (name, ty, usage, aliases) in &info {
        let flag_name = if name.len() > 1 {
            format!("--{name}")
        } else {
            format!("-{name}")
        };
        let mut flags = vec![flag_name];
        flags.extend(aliases.iter().map(|a| {
            if a.len() > 1 {
                format!("--{a}")
            } else {
                format!("-{a}")
            }
        }));
        let flags = flags.join(", ");
        let desc = usage.replace('\'', "'\\''");
        let value = if ty == "string" {
            format!(":{name}: ")
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "  _re_args+=('{flags}[{desc}]{value}')",
            flags = flags,
            desc = desc,
            value = value
        );
    }
    let _ = writeln!(
        out,
        "  if [[ $words[CURRENT-1] == -t || $words[CURRENT-1] == --tag ]]; then"
    );
    let _ = writeln!(out, "    local -a tags");
    let _ = writeln!(
        out,
        "    tags=(${{(f)\"$(${{words[1]}} --complete-tags \"${{words[CURRENT]}}\" 2>/dev/null)\"}})"
    );
    let _ = writeln!(out, "    compadd -a tags");
    let _ = writeln!(out, "    return");
    let _ = writeln!(out, "  fi");
    let _ = writeln!(out, "  _arguments -s : \"$_re_args[@]\"");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out, "\ncompdef _re re");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ie_sets_insert_and_editor() {
        let argv = normalize_legacy_single_dash_long_args(["re".to_string(), "-ie".to_string()]);
        let cli = parse_cli_and_parser(argv).1;
        assert!(cli.insert);
        assert!(cli.e);
    }

    #[test]
    fn normalize_single_dash_long_flags() {
        let argv = normalize_legacy_single_dash_long_args([
            "re".to_string(),
            "-include-finished".to_string(),
            "-backend=mongo".to_string(),
            "-add-tag=todo".to_string(),
        ]);
        assert!(argv.iter().any(|a| a == "--include-finished"));
        assert!(argv.iter().any(|a| a == "--backend=mongo"));
        assert!(argv.iter().any(|a| a == "--add-tag=todo"));
    }

    #[test]
    fn normalize_title_repairs_wrapped_numbered_links() {
        let broken = "1. 第一条：https://example.com/abc                                      2. 第二条：https://example.com/def\nghi                                      3. 第三条：https://example.com/xyz";
        let normalized = normalize_title_for_display(broken);
        assert_eq!(
            normalized,
            "1. 第一条：https://example.com/abc\n2. 第二条：https://example.com/def\nghi\n3. 第三条：https://example.com/xyz"
        );
    }

    #[test]
    fn parse_backend_with_equals_value() {
        let argv = normalize_legacy_single_dash_long_args([
            "re".to_string(),
            "-backend=mongo".to_string(),
            "-n=12".to_string(),
        ]);
        let cli = parse_cli_and_parser(argv).1;
        assert_eq!(cli.backend, "mongo");
        assert_eq!(cli.n, 12);
    }

    #[test]
    fn tag_query_t_uses_exact_match_by_default() {
        let argv = normalize_legacy_single_dash_long_args([
            "re".to_string(),
            "-t".to_string(),
            "l".to_string(),
        ]);
        let cli = parse_cli_and_parser(argv).1;
        assert_eq!(get_tag_query_raw(&cli).as_deref(), Some("=l"));
    }

    #[test]
    fn tag_query_ta_uses_contains_match() {
        let argv = normalize_legacy_single_dash_long_args([
            "re".to_string(),
            "-ta".to_string(),
            "l".to_string(),
        ]);
        let cli = parse_cli_and_parser(argv).1;
        assert_eq!(get_tag_query_raw(&cli).as_deref(), Some("~l"));
    }

    #[test]
    fn tag_query_t_with_a_uses_contains_match() {
        let argv = normalize_legacy_single_dash_long_args([
            "re".to_string(),
            "-t".to_string(),
            "db".to_string(),
            "-a".to_string(),
        ]);
        let cli = parse_cli_and_parser(argv).1;
        assert_eq!(get_tag_query_raw(&cli).as_deref(), Some("~db"));
    }

    #[test]
    fn normalize_title_keeps_non_link_text() {
        let plain = "第一行\r\n第二行";
        let normalized = normalize_title_for_display(plain);
        assert_eq!(normalized, "第一行\n第二行");
    }

    #[test]
    fn wrap_title_moves_long_url_to_indented_lines() {
        let title = "1. 文档：https://example.com/very/long/path/for/test?foo=bar&baz=qux";
        let wrapped = wrap_title_for_terminal(title, 36);
        assert_eq!(
            wrapped,
            "1. 文档：\n    https://example.com/very/long/path/for/test?foo=bar&baz=qux"
        );
    }

    #[test]
    fn normalize_title_preserves_newlines_after_url_lines() {
        let original = "fabric数据集链接：\nhttps://example.com/a\n子数据集id：5130071\ntask_id：t#1770303001804771022\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nfabric数据集ID：4333589";
        let normalized = normalize_title_for_display(original);
        assert_eq!(normalized, original);
    }
}
