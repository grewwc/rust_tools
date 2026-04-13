use std::process::Command;

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

fn params_cargo_check() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "cwd": {
                "type": "string",
                "description": "Working directory for the cargo command (default: \".\")."
            },
            "workspace": {
                "type": "boolean",
                "description": "If true (default), run for the whole workspace (--workspace)."
            },
            "all_features": {
                "type": "boolean",
                "description": "If true, enable all features (--all-features)."
            },
            "package": {
                "type": "string",
                "description": "Optional package name to target (-p <name>)."
            }
        }
    })
}

fn params_cargo_test() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "cwd": {
                "type": "string",
                "description": "Working directory for the cargo command (default: \".\")."
            },
            "workspace": {
                "type": "boolean",
                "description": "If true (default), run for the whole workspace (--workspace)."
            },
            "all_features": {
                "type": "boolean",
                "description": "If true, enable all features (--all-features)."
            },
            "package": {
                "type": "string",
                "description": "Optional package name to target (-p <name>)."
            }
        }
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "cargo_check",
        description: "Run `cargo check` with optional workspace/all-features/package flags and return the output.",
        parameters: params_cargo_check,
        execute: execute_cargo_check,
        groups: &["executor"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "cargo_test",
        description: "Run `cargo test` with optional workspace/all-features/package flags and return the output.",
        parameters: params_cargo_test,
        execute: execute_cargo_test,
        groups: &["executor"],
    }
});

fn cargo_common_args(args: &Value) -> (String, bool, bool, Option<String>) {
    let cwd = args["cwd"].as_str().unwrap_or(".").to_string();
    let workspace = args["workspace"].as_bool().unwrap_or(true);
    let all_features = args["all_features"].as_bool().unwrap_or(false);
    let package = args["package"].as_str().map(|s| s.trim().to_string());
    let package = package.filter(|s| !s.is_empty());
    (cwd, workspace, all_features, package)
}

fn execute_cargo_command(subcommand: &str, args: &Value) -> Result<String, String> {
    let (cwd, workspace, all_features, package) = cargo_common_args(args);

    let mut cmd = Command::new("cargo");
    cmd.arg(subcommand);
    if workspace {
        cmd.arg("--workspace");
    }
    if all_features {
        cmd.arg("--all-features");
    }
    if let Some(pkg) = package {
        cmd.args(["-p", &pkg]);
    }
    let output = cmd
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("Failed to execute cargo: {}", e))?;

    let (stdout, stderr) = output_to_strings(&output);
    if output.status.success() {
        Ok(format!("{}\n{}", stdout.trim(), stderr.trim())
            .trim()
            .to_string())
    } else {
        Ok(format_exit_code_output(&output, &stdout, &stderr))
    }
}

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

pub(crate) fn execute_cargo_check(args: &Value) -> Result<String, String> {
    execute_cargo_command("check", args)
}

pub(crate) fn execute_cargo_test(args: &Value) -> Result<String, String> {
    execute_cargo_command("test", args)
}
