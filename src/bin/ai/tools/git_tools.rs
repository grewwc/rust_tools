use std::process::Command;

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

fn params_git_status() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "cwd": {
                "type": "string",
                "description": "Working directory containing the git repository (default: \".\")."
            }
        }
    })
}

fn params_git_diff() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "cwd": {
                "type": "string",
                "description": "Working directory containing the git repository (default: \".\")."
            },
            "cached": {
                "type": "boolean",
                "description": "If true, diff staged changes (--cached)."
            },
            "pathspec": {
                "type": "string",
                "description": "Optional pathspec to limit the diff (e.g. \"src\" or \"Cargo.toml\")."
            }
        }
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "git_status",
        description: "Run `git status --porcelain=v1 --branch` in a directory and return the output.",
        parameters: params_git_status,
        execute: execute_git_status,
        groups: &["executor"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "git_diff",
        description: "Run `git diff` (optionally --cached and/or with a pathspec) and return the diff (truncated).",
        parameters: params_git_diff,
        execute: execute_git_diff,
        groups: &["executor"],
    }
});

fn output_to_strings(output: &std::process::Output) -> (String, String) {
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

fn format_exit_code_output(output: &std::process::Output, stdout: &str, stderr: &str) -> String {
    format!(
        "Exit code: {}\n{}\n{}",
        output.status.code().unwrap_or(-1),
        stdout.trim(),
        stderr.trim()
    )
}

pub(crate) fn execute_git_status(args: &Value) -> Result<String, String> {
    let cwd = args["cwd"].as_str().unwrap_or(".");

    let output = Command::new("git")
        .args(["status", "--porcelain=v1", "--branch"])
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("Failed to execute git: {}", e))?;

    let (stdout, stderr) = output_to_strings(&output);
    if output.status.success() {
        Ok(stdout.trim().to_string())
    } else {
        Ok(format_exit_code_output(&output, &stdout, &stderr))
    }
}

pub(crate) fn execute_git_diff(args: &Value) -> Result<String, String> {
    let cwd = args["cwd"].as_str().unwrap_or(".");
    let cached = args["cached"].as_bool().unwrap_or(false);
    let pathspec = args["pathspec"].as_str().unwrap_or("").trim().to_string();

    let mut cmd = Command::new("git");
    cmd.arg("diff");
    if cached {
        cmd.arg("--cached");
    }
    if !pathspec.is_empty() {
        cmd.arg("--").arg(pathspec);
    }
    let output = cmd
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("Failed to execute git: {}", e))?;

    let (stdout, stderr) = output_to_strings(&output);
    let mut out = if output.status.success() {
        stdout
    } else {
        format_exit_code_output(&output, &stdout, &stderr)
    };

    const MAX_CHARS: usize = 16_000;
    if out.len() > MAX_CHARS {
        out.truncate(MAX_CHARS);
        out.push_str("\n... (truncated)");
    }
    Ok(out.trim().to_string())
}
