use rust_tools::{common::utils::expanduser, terminalw};

#[derive(Clone)]
pub struct Options {
    pub target: String,
    pub root_dir: String,
    pub ext: String,
    pub ext_exclude: String,
    pub verbose: bool,
    pub is_regex: bool,
    pub ignore_case: bool,
    pub strict: bool,
    pub word: bool,
    pub max_level: i32,
    pub file_names: Vec<String>,
    pub not_file_names: Vec<String>,
    pub num_lines: usize,
    pub thread_count: usize,
    pub max_len: usize,
    pub num_print: i64,
}

pub fn build_parser() -> terminalw::Parser {
    let mut p = terminalw::new_parser(&[]);
    p.add_i64("n", 5, "number of found results to print");
    p.add_string("t", "", "what type of file to search");
    p.add_bool("v", false, "if print error");
    p.add_string("d", ".", "root directory for searching");
    p.add_bool(
        "re",
        false,
        r#"turn on regular expression (use "\" instead of "\\")"#,
    );
    p.add_bool("ignore", false, "ignore upper/lower case");
    p.add_bool("i", false, "ignore upper/lower case (shortcut for -ignore)");
    p.add_int(
        "level",
        i32::MAX,
        "number of directory levels to search. current directory's level is 0",
    );
    p.add_bool(
        "strict",
        false,
        "find exact the same matches (after triming space)",
    );
    p.add_string("nt", "", "check files which are not some types");
    p.add_bool(
        "w",
        false,
        "only match the concrete word, is a shortcut for -re",
    );
    p.add_bool("all", false, "shortcut for -n=-1");
    p.add_bool("a", false, "shortcut for -all");
    p.add_string("f", "", "check only these files/directories");
    p.add_string("nf", "", "don't check these files/directories");
    p.add_int("l", 1, "how many lines more read to match");
    p.add_int("p", 4, "how many threads to use");
    p.add_bool("h", false, "print help info");
    p.add_int("maxlen", 128, "maxlen of one line");
    p
}

fn split_list(s: &str) -> Vec<String> {
    let normalized = s.replace(',', " ");
    normalized
        .split_whitespace()
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

pub fn parse_from_env() -> Option<Options> {
    println!();

    let argv = std::env::args().skip(1).collect::<Vec<_>>();
    let mut p = build_parser();
    p.parse_argv(&argv, &[]);

    if p.is_empty() || p.contains_flag_strict("h") {
        p.print_defaults();
        return None;
    }

    let args = p.positional_args(true);
    if args.len() != 1 {
        p.print_defaults();
        return None;
    }

    let mut num_print = p.flag_value_i64("n");
    if p.num_args() != -1 {
        num_print = p.num_args() as i64;
    }
    if num_print <= 0 {
        num_print = 5;
    }
    if p.contains_any_flag_strict(&["all", "a"]) || num_print < 0 {
        num_print = i64::MAX;
    }

    let target = args[0].replace(r"\\", r"\");
    let root_dir_raw = p.flag_value_or_default("d").replace(r"\\", r"\");
    let root_dir = expanduser(root_dir_raw.trim()).to_string();

    let ext = p.flag_value_or_default("t");
    let ext_exclude = p.flag_value_or_default("nt");
    let verbose = p.contains_flag_strict("v");
    let mut is_regex = p.contains_flag_strict("re");
    let ignore_case = p.contains_any_flag_strict(&["ignore", "i"]);
    let strict = p.contains_flag_strict("strict");
    let word = p.contains_flag_strict("w");
    if word {
        is_regex = true;
    }
    let max_level = p.flag_value_i32("level");
    let num_lines = p.flag_value_i32("l").max(1) as usize;
    let thread_count = p.flag_value_i32("p").max(1) as usize;
    let max_len = p.flag_value_i32("maxlen").max(1) as usize;

    let file_names = split_list(&p.flag_value_or_default("f"));
    let not_file_names = split_list(&p.flag_value_or_default("nf"));

    Some(Options {
        target,
        root_dir,
        ext,
        ext_exclude,
        verbose,
        is_regex,
        ignore_case,
        strict,
        word,
        max_level,
        file_names,
        not_file_names,
        num_lines,
        thread_count,
        max_len,
        num_print,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_list() {
        assert_eq!(
            split_list("a,b  c,,d"),
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }
}
