//! Execute-command validation / execution tests.

use super::super::types::{FunctionCall, ToolCall};
use super::super::*;

#[test]
fn execute_command_blocks_dangerous_programs() {
    assert!(tools::validate_execute_command("rm -rf /").is_err());
    // assert!(tools::validate_execute_command("mv a b").is_err());
    assert!(tools::validate_execute_command("sudo ls").is_err());
}

#[test]
fn execute_command_blocks_git_destructive_to_uncommitted() {
    // checkout: discards worktree changes
    assert!(tools::validate_execute_command("git checkout -- src/main.rs").is_err());
    assert!(tools::validate_execute_command("git checkout -- .").is_err());
    assert!(tools::validate_execute_command("git checkout .").is_err());
    assert!(tools::validate_execute_command("git checkout -f main").is_err());
    assert!(tools::validate_execute_command("git checkout --force main").is_err());
    assert!(tools::validate_execute_command("git -C /repo checkout -- x").is_err());
    // checkout: pure branch switch; git protects uncommitted changes, allow
    assert!(tools::validate_execute_command("git checkout main").is_ok());
    assert!(tools::validate_execute_command("git checkout -b feature/x").is_ok());
    // checkout: -B force-resets and switches branches, discarding uncommitted changes
    assert!(tools::validate_execute_command("git checkout -B main").is_err());
    assert!(tools::validate_execute_command("git checkout --force-create main").is_err());
    // checkout: no -- but the path heuristic judges it a file (has an extension)
    assert!(tools::validate_execute_command("git checkout src/main.rs").is_err());
    assert!(tools::validate_execute_command("git checkout README.md").is_err());
    assert!(tools::validate_execute_command("git checkout main.rs").is_err());
    assert!(tools::validate_execute_command("git checkout archive.tar.gz").is_err());
    // checkout: arguments without an extension are not falsely blocked (branch/tag shapes)
    assert!(tools::validate_execute_command("git checkout main").is_ok());
    assert!(tools::validate_execute_command("git checkout v1.2.3").is_ok());
    assert!(tools::validate_execute_command("git checkout feature/x").is_ok());

    // restore: restoring the worktree by default discards uncommitted changes
    assert!(tools::validate_execute_command("git restore src/main.rs").is_err());
    assert!(tools::validate_execute_command("git restore --worktree src/main.rs").is_err());
    assert!(tools::validate_execute_command("git restore --source=HEAD~1 src/main.rs").is_err());
    // restore: only unstages, the worktree is untouched, allow
    assert!(tools::validate_execute_command("git restore --staged src/main.rs").is_ok());
    assert!(
        tools::validate_execute_command("git restore --staged --source=HEAD src/main.rs").is_ok()
    );

    // reset: --hard/--merge/--keep discard uncommitted changes
    assert!(tools::validate_execute_command("git reset --hard").is_err());
    assert!(tools::validate_execute_command("git reset --hard HEAD~1").is_err());
    assert!(tools::validate_execute_command("git reset --merge").is_err());
    assert!(tools::validate_execute_command("git reset --keep").is_err());
    // reset: --soft / default (mixed) keep the worktree, allow
    assert!(tools::validate_execute_command("git reset --soft HEAD~1").is_ok());
    assert!(tools::validate_execute_command("git reset").is_ok());
    assert!(tools::validate_execute_command("git reset HEAD~1").is_ok());

    // clean: -f deletes untracked files, not recoverable
    assert!(tools::validate_execute_command("git clean -f").is_err());
    assert!(tools::validate_execute_command("git clean -fd").is_err());
    assert!(tools::validate_execute_command("git clean --force").is_err());
    // clean: dry-run and friends do not actually delete, allow
    assert!(tools::validate_execute_command("git clean -n").is_ok());

    // switch: force branch switch discards uncommitted changes
    assert!(tools::validate_execute_command("git switch -f main").is_err());
    assert!(tools::validate_execute_command("git switch --force main").is_err());
    assert!(tools::validate_execute_command("git switch --discard-changes main").is_err());
    assert!(tools::validate_execute_command("git switch -C fix").is_err());
    assert!(tools::validate_execute_command("git switch --force-create fix").is_err());
    // switch: pure branch create/switch; git protects uncommitted changes, allow
    assert!(tools::validate_execute_command("git switch main").is_ok());
    assert!(tools::validate_execute_command("git switch -c feature/x").is_ok());

    // Must also be blocked through indirect wrappers (env/xargs)
    assert!(tools::validate_execute_command("env git checkout -- x").is_err());
    assert!(tools::validate_execute_command("xargs git reset --hard").is_err());
}

#[test]
fn execute_command_allows_common_shell_syntax() {
    assert!(tools::validate_execute_command("ls | wc").is_ok());
    assert!(tools::validate_execute_command("ls && pwd").is_ok());
    assert!(tools::validate_execute_command("echo hi > /tmp/a").is_ok());
}

#[test]
fn execute_command_allows_readonly_commands() {
    assert!(tools::validate_execute_command("ls").is_ok());
    assert!(tools::validate_execute_command("pwd").is_ok());
    assert!(tools::validate_execute_command("cat Cargo.toml").is_ok());
    assert!(tools::validate_execute_command("rg main src").is_ok());
}

#[test]
fn execute_command_captures_stdout() {
    let tool_call = ToolCall {
        id: "call_1".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "execute_command".to_string(),
            arguments: r#"{"command":"echo hello"}"#.to_string(),
        },
    };
    let res = tools::execute_tool_call(&tool_call).unwrap();
    assert_eq!(res.content.trim(), "hello");
}
