use super::exclude;
use rust_tools::{
    common::{configw, utils::expanduser},
    terminalw,
};
use std::{fs, path::PathBuf};

const ABS_CONFIG_KEY: &str = "ff.abs";

#[derive(Clone)]
pub struct Options {
    pub verbose: bool,
    pub only_dir: bool,
    pub print_md5: bool,
    pub glob_mode: bool,
    pub case_insensitive: bool,
    pub relative: bool,
    pub num_print: i64,
    pub thread_count: usize,
    pub wd: PathBuf,
    pub root_pat: String,
    pub targets: Vec<String>,
    pub excludes: Vec<regex::Regex>,
}

pub fn build_parser() -> terminalw::Parser {
    let mut p = terminalw::new_parser(&[]);
    p.add_i64("n", 10, "number of found results to print, -10 for short");
    p.add_bool("v", false, "if print error");
    p.add_string("d", ".", "root directory for searching");
    p.add_bool("i", false, "ignore case (only for -glob)");
    p.add_string("ex", "", "exclude file patterns (glob )");
    p.add_bool("a", false, "list all matches (has the highest priority)");
    p.add_int("p", 4, "how many threads to use");
    p.add_bool("dir", false, "only search directories");
    p.add_bool("h", false, "print this help");
    p.add_bool("md5", false, "print md5 value");
    p.add_bool("abs", false, "print absolute path. (ff.abs in ~/.configW)");
    p.add_bool("rel", false, "print relative path.");
    p.add_bool("glob", false, "use glob to match");
    p.add_bool("g", false, "shortcut for -glob");
    p
}

pub fn parse_from_env() -> Option<Options> {
    let argv = std::env::args().skip(1).collect::<Vec<_>>();
    let mut parser = build_parser();
    parser.parse_argv(&argv, &[]);

    if parser.is_empty() || parser.contains_flag_strict("h") {
        parser.print_defaults();
        return None;
    }

    let verbose = parser.contains_flag_strict("v");
    let only_dir = parser.contains_flag_strict("dir");
    let print_md5 = parser.contains_flag_strict("md5");
    let glob_mode = parser.contains_any_flag_strict(&["glob", "g"]);
    let case_insensitive = parser.contains_flag_strict("i");
    let thread_count = parser.flag_value_i32("p").max(1) as usize;

    let mut num_print = if parser.num_args() == -1 {
        parser.flag_value_i64("n")
    } else {
        parser.num_args() as i64
    };
    if num_print <= 0 {
        num_print = 10;
    }
    if parser.contains_flag_strict("a") {
        num_print = i64::MAX;
    }

    let cfg = configw::get_all_config();
    let default_abs = cfg.get(ABS_CONFIG_KEY, "") == "true";
    let mut relative = !parser.contains_flag_strict("abs") && !default_abs;
    relative = relative || parser.contains_flag_strict("rel");

    let wd = std::env::current_dir()
        .ok()
        .and_then(|p| fs::canonicalize(&p).ok())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let root_pat = expanduser(parser.flag_value_or_default("d").trim()).to_string();
    let root_pat = if root_pat.trim().is_empty() {
        ".".to_string()
    } else {
        root_pat
    };

    let excludes = exclude::globs_to_regexes(parser.flag_value_or_default("ex").trim(), verbose);

    let targets = parser.positional_args(true);
    if targets.is_empty() {
        parser.print_defaults();
        return None;
    }

    Some(Options {
        verbose,
        only_dir,
        print_md5,
        glob_mode,
        case_insensitive,
        relative,
        num_print,
        thread_count,
        wd,
        root_pat,
        targets,
        excludes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_parses_flags_after_positional() {
        let argv = vec![
            "main.go".to_string(),
            "-d".to_string(),
            "SOME_ROOT_DIR".to_string(),
        ];
        let mut p = build_parser();
        p.parse_argv(&argv, &[]);
        assert_eq!(p.positional_args(true), vec!["main.go".to_string()]);
        assert_eq!(p.flag_value_or_default("d"), "SOME_ROOT_DIR");
    }
}
