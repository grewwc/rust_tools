use std::usize;

use clap::{ArgAction, Parser};
use regex::Regex;
use rust_tools::{cmd::run::run_cmd, strw::split::split_space_keep_symbol};

use colored::*;
use once_cell::sync::Lazy;
const LOG_HISTORY_CMD: &'static str =
    r#"git log $branch$ --oneline --format="%h %an %ad %s" --date=short"#;
const BRANCH_CMD: &'static str = r#"git for-each-ref --sort=-committerdate --format="%(refname:short) %(committerdate:short) %(subject)" refs/heads/ "#;
const DEFAULT_N: usize = 5;

static MERGE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"\w+\s.*\s\d{4}-\d{2}-\d{2}\sMerge.*"#).unwrap());

static DIGIT_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^\-?(\d+)$"#).unwrap());

#[derive(Parser)]
struct UserInput {
    #[arg(help = "if get all histories", short, long, action = ArgAction::SetTrue)]
    all: bool,

    #[arg(help = "number of histories to show", value_parser = clap::value_parser!(i32), allow_negative_numbers = true)]
    n: Option<i32>,

    #[arg(
        help = "branch name (default=current branch)",
        short,
        long,
        default_missing_value = "",
        num_args = 0..=1
    )]
    branch: Option<String>,

    #[arg(help="verbose print", short, long, action=ArgAction::SetTrue)]
    v: bool,
}

impl UserInput {
    fn is_verbose(&self) -> bool {
        self.v
    }

    fn get_branch(&self) -> (Option<&str>, usize) {
        let branch = self.branch.as_deref();
        if branch.is_none() {
            return (None, 0);
        }
        let branch = branch.unwrap();
        if let Some(cap) = DIGIT_PATTERN.captures(branch)
            && cap.len() >= 2
        {
            if let Some(cap) = cap.get(1) {
                let str = cap.as_str();
                if let Ok(num) = str.parse::<usize>() {
                    return (None, num as usize);
                }
            }
            return (None, 0);
        }
        (Some(branch), 0)
    }

    fn is_print_all(&self) -> bool {
        self.all
    }

    fn get_print_history(&self) -> usize {
        if self.is_print_all() {
            return usize::MAX;
        }
        if let Some(mut n) = self.n {
            if n < 0 {
                n *= -1;
            }
            n as usize
        } else {
            DEFAULT_N
        }
    }

    fn is_print_branch(&self) -> bool {
        self.branch.is_some()
    }
}

trait GitPrint {
    fn print(&self, content: &str) -> bool;
}

struct BranchHistory;

impl GitPrint for BranchHistory {
    fn print(&self, content: &str) -> bool {
        if let Some((name, date, msg)) = self.parse(content) {
            println!("{} ({}) {}", name.cyan(), date.bright_yellow(), msg);
        }
        true
    }
}

impl BranchHistory {
    fn parse(&self, content: &str) -> Option<(String, String, String)> {
        let mut content = content.trim();
        content = content.trim_matches('"');
        // println!("|{}|", content);
        let mut iter = split_space_keep_symbol(content, "\"");
        // let mut iter = content.split("\"");
        let name = iter.next();
        if name.is_none() {
            return None;
        }
        let name = name.unwrap().to_owned();
        let date = iter.next();
        if date.is_none() {
            return None;
        }
        let date = date.unwrap().to_owned();
        let msg: String = iter.fold(String::new(), |mut acc, item| {
            if !acc.is_empty() {
                acc.push_str(" ");
            }
            acc.push_str(item);
            acc
        });

        Some((name, date, msg))
    }
    fn new() -> Self {
        BranchHistory
    }
}

struct LogHistory {
    verbose: bool,
}
impl LogHistory {
    fn new(verbose: bool) -> Self {
        LogHistory { verbose }
    }
}

impl GitPrint for LogHistory {
    fn print(&self, content: &str) -> bool {
        let content = content.trim_matches('"');
        let is_merge_commit = MERGE_PATTERN.is_match(content);
        if is_merge_commit && !self.verbose {
            return false;
        }
        if is_merge_commit {
            println!("{}", content.bright_black());
        } else {
            println!("{}", content);
        }
        true
    }
}

struct Handler {
    handler: Box<dyn GitPrint>,
    cmd: Box<dyn AsRef<str>>,
    n: usize,
}

impl Handler {
    fn new(user_input: &UserInput) -> Self {
        // take mutable borrow only to parse and possibly mutate input, then drop it
        let mut n_print = user_input.get_print_history();
        let (branch, n_print_by_branch_arg) = user_input.get_branch();
        if n_print_by_branch_arg != 0 {
            n_print = n_print_by_branch_arg;
        }
        let branch = branch.unwrap_or("");

        let mut cmd = LOG_HISTORY_CMD;
        if user_input.is_print_branch() && branch.is_empty() {
            cmd = BRANCH_CMD;
            // replace $branch$
            let handler = Box::new(BranchHistory::new());
            Handler {
                handler,
                cmd: Box::new(cmd),
                n: n_print,
            }
        } else {
            let handler = Box::new(LogHistory::new(user_input.is_verbose()));
            let cmd = cmd.replace("$branch$", branch);
            Handler {
                handler,
                cmd: Box::new(cmd),
                n: n_print,
            }
        }
    }

    fn handle(&self) {
        let mut cnt: usize = 0;
        match run_cmd(self.cmd.as_ref().as_ref()) {
            Ok(output) => {
                let lines = output.split("\n");
                let mut iter = lines.map(|line| line.trim());
                while let Some(line) = iter.next() {
                    if cnt >= self.n {
                        break;
                    }
                    if self.handler.print(line) {
                        cnt += 1;
                    }
                }
            }
            Err(err) => {
                println!("Failed. err: {err:?}");
            }
        }
    }
}

fn main() {
    let input = UserInput::parse();
    let handler = Handler::new(&input);
    handler.handle();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse() {
        let l = BranchHistory::new();
        l.parse("feature/20251030_27098361_fix_security_issues_1 2025-10-30 fix:校验是否是合法的ossendpoint");
    }

    #[test]
    fn test_match() {
        let result = DIGIT_PATTERN.captures("-23");

        println!("{}", result.unwrap().get(1).unwrap().as_str());
    }
}

