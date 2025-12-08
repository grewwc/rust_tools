use std::{cell::LazyCell, usize};

use clap::{ArgAction, Parser};
use regex::Regex;
use rust_tools::{cmd::run::run_cmd, strw::split::split_space_keep_symbol};

use colored::*;

use _his::current_branch;
const LOG_HISTORY_CMD: &'static str =
    r#"git log $branch$ --oneline --format="%h %an %ad %s" --date=short"#;
const BRANCH_CMD: &'static str = r#"git for-each-ref --sort=-committerdate --format="%(refname:short) %(committerdate:short) %(subject)" refs/heads/ "#;
const DEFAULT_N: usize = 5;

const MERGE_PATTERN: LazyCell<Regex> =
    LazyCell::new(|| Regex::new(r#"\w+\s.*\s\d{4}-\d{2}-\d{2}\sMerge.*"#).unwrap());

const PURE_DIGITAL_PATTERN: LazyCell<Regex> =
    LazyCell::new(|| Regex::new(r#"^\-(\d+)$"#).unwrap());

const DIGITAL_PATTERN: LazyCell<Regex> = LazyCell::new(|| Regex::new(r#"\s-(\d+)"#).unwrap());

mod _his;

#[derive(Parser)]
struct UserInput {
    #[arg(help = "if get all histories", short, long, action = ArgAction::SetTrue)]
    all: bool,

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

    #[arg(
        help = "number of histories to show or other arguments",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    args: Vec<String>,

    #[arg(skip)]
    n: i32,
}

impl UserInput {
    fn is_verbose(&self) -> bool {
        self.v
    }

    fn modify_by_positional_args(&mut self) {
        if self.args.is_empty() && self.branch.is_none() {
            return;
        }
        let mut args = self.args.join(" ");
        let mut branch_is_modified = false;
        if let Some(cap) = DIGITAL_PATTERN.captures(args.as_str())
            && let Some(m) = cap.get(1)
        {
            self.n = match m.as_str().parse::<usize>() {
                Ok(n) => n as i32,
                _ => DEFAULT_N as i32,
            };
            if self.branch.is_none() {
                args.replace_range(cap.get(0).unwrap().range(), "");
                let args = args.trim();
                if !args.is_empty() {
                    self.branch = Some(args.to_string());
                    branch_is_modified = true;
                }
            }
        }
        if !branch_is_modified && self.branch.is_none() && !args.trim().is_empty() {
            self.branch = Some(args);
        }
        if let Some(b) = self.branch.as_deref()
            && let Some(caps) = PURE_DIGITAL_PATTERN.captures(b)
            && let Some(m) = caps.get(1) 
            && let Ok(n)  =m.as_str().parse::<i32>() 
        {
            self.branch = Some("".to_string());
            self.n = n;
        }
        // println!("branch:|{:?}|",self.branch);
    }

    fn get_branch(&self) -> String {
        if self.branch.is_none() {
            return current_branch();
        }
        self.branch.as_deref().unwrap_or("").to_string()
    }

    fn is_print_all(&self) -> bool {
        self.all
    }

    fn get_print_history(&self) -> usize {
        if self.is_print_all() {
            return usize::MAX;
        }
        if self.n <= 0 {
            return DEFAULT_N;
        }
        self.n as usize
    }

    fn is_print_branch(&self) -> bool {
        self.branch.is_some()
    }
}

trait GitPrint {
    fn before_print(&self) {}
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
    branch: String,
}
impl LogHistory {
    fn new(verbose: bool, branch: String) -> Self {
        LogHistory { verbose, branch }
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

    fn before_print(&self) {
        let branch: &str = self.branch.as_ref();
        if branch.is_empty() {
            let b = current_branch();
            println!("{}", b.trim().green());
            return;
        }
        println!("{}\n{}", branch.green(), "--".repeat(8));
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
        let n_print = user_input.get_print_history();
        let branch = user_input.get_branch();
        // println!("|{}|", branch);

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
            let handler = Box::new(LogHistory::new(user_input.is_verbose(), branch.to_string()));
            let cmd = cmd.replace("$branch$", &branch);
            // println!("cmd: {}", cmd);
            Handler {
                handler,
                cmd: Box::new(cmd),
                n: n_print,
            }
        }
    }

    fn handle(&self) {
        let mut cnt: usize = 0;
        self.handler.before_print();
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
    let mut input = UserInput::parse();
    input.modify_by_positional_args();
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
        let result = DIGITAL_PATTERN.captures("-b -2");

        println!("{}", result.unwrap().get(1).unwrap().as_str());
    }

    #[test]
    fn test_get_current_branch() {
        let branch = current_branch();
        println!("==> branch: {}", branch);
    }
}

