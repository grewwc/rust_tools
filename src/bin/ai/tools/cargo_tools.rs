use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

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
                "description": "Working directory for the cargo command. Defaults to \".\" (current directory)."
            },
            "workspace": {
                "type": "boolean",
                "description": "If true, run tests for the whole workspace (--workspace). Defaults to false. When false, only the specified package or current crate is tested."
            },
            "all_features": {
                "type": "boolean",
                "description": "If true, enable all features (--all-features). Defaults to false."
            },
            "package": {
                "type": "string",
                "description": "Optional package name to target (-p <name>). When omitted and workspace is false, tests the current crate."
            },
            "timeout_secs": {
                "type": "integer",
                "description": "Maximum seconds to wait for cargo to finish. The process is killed after this duration. Defaults to 300 (5 minutes)."
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
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["executor"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "cargo_test",
        description: "Run `cargo test` with optional workspace/all-features/package flags and return the output.",
        parameters: params_cargo_test,
        execute: execute_cargo_test,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["executor"],
    }
});

fn cargo_common_args(args: &Value, default_workspace: bool) -> (String, bool, bool, Option<String>) {
    let cwd = args["cwd"].as_str().unwrap_or(".").to_string();
    let workspace = args["workspace"].as_bool().unwrap_or(default_workspace);
    let all_features = args["all_features"].as_bool().unwrap_or(false);
    let package = args["package"].as_str().map(|s| s.trim().to_string());
    let package = package.filter(|s| !s.is_empty());
    (cwd, workspace, all_features, package)
}

fn execute_cargo_command(subcommand: &str, args: &Value, default_timeout_secs: u64, default_workspace: bool) -> Result<String, String> {
    let (cwd, workspace, all_features, package) = cargo_common_args(args, default_workspace);
    let timeout_secs = args["timeout_secs"]
        .as_u64()
        .unwrap_or(default_timeout_secs);
    const MAX_CARGO_OUTPUT_BYTES: usize = 512 * 1024; // 512KB 输出上限

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

    let mut child = cmd
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to execute cargo: {}", e))?;

    // 读取 stdout/stderr 到缓冲区，同时检查超时
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // 在单独线程中读取 stdout/stderr，避免阻塞
    let stdout_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut s) = stdout {
            let _ = s
                .by_ref()
                .take((MAX_CARGO_OUTPUT_BYTES + 1) as u64)
                .read_to_end(&mut buf);
        }
        buf
    });
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut s) = stderr {
            let _ = s
                .by_ref()
                .take((MAX_CARGO_OUTPUT_BYTES + 1) as u64)
                .read_to_end(&mut buf);
        }
        buf
    });

    // 轮询等待子进程，超时则 kill
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "cargo {} timed out after {} seconds",
                        subcommand, timeout_secs
                    ));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(format!("Failed to wait for cargo: {}", e)),
        }
    };

    let stdout_raw = stdout_handle.join().unwrap_or_default();
    let stderr_raw = stderr_handle.join().unwrap_or_default();
    let stdout_truncated = stdout_raw.len() > MAX_CARGO_OUTPUT_BYTES;
    let stderr_truncated = stderr_raw.len() > MAX_CARGO_OUTPUT_BYTES;
    let stdout =
        String::from_utf8_lossy(&stdout_raw[..stdout_raw.len().min(MAX_CARGO_OUTPUT_BYTES)])
            .to_string();
    let stderr =
        String::from_utf8_lossy(&stderr_raw[..stderr_raw.len().min(MAX_CARGO_OUTPUT_BYTES)])
            .to_string();
    let stdout = if stdout_truncated {
        format!(
            "{}\n... (output truncated, {} bytes total)",
            stdout.trim(),
            stdout_raw.len()
        )
    } else {
        stdout
    };
    let stderr = if stderr_truncated {
        format!(
            "{}\n... (output truncated, {} bytes total)",
            stderr.trim(),
            stderr_raw.len()
        )
    } else {
        stderr
    };

    // 构造模拟 Output 对象
    let output = std::process::Output {
        status,
        stdout: stdout.into_bytes(),
        stderr: stderr.into_bytes(),
    };

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
    execute_cargo_command("check", args, 300, true)
}

pub(crate) fn execute_cargo_test(args: &Value) -> Result<String, String> {
    execute_cargo_command("test", args, 300, false)
}
