use rust_tools::cmd::run::run_cmd;

pub fn current_branch() -> String {
    run_cmd("git branch | grep '*'").unwrap()
}

