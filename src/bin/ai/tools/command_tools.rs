use std::time::Duration;

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

fn params_execute_command() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute. Destructive/network/escalation commands are blocked."
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory for the command (default: current directory)."
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (1-300; default: 30)."
            }
        },
        "required": ["command"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "execute_command",
        description: "Run a shell command with an optional working directory and timeout. Destructive/network/escalation commands are blocked; output is truncated and includes an exit code on failure.",
        parameters: params_execute_command,
        execute: execute_command,
        groups: &["builtin"],
    }
});

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out = String::with_capacity(max_chars + 32);
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("\n... (truncated)");
    out
}

pub fn validate_execute_command(command: &str) -> Result<(), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("empty command".to_string());
    }

    let tokens = command.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return Err("empty command".to_string());
    }

    let program = tokens[0].to_lowercase();

    let denied_programs = [
        "fish", "jshell", "rm", "mv", "dd", "chmod", "chown", "chgrp", "kill", "pkill",
        "killall", "sudo", "su", "passwd", "shutdown", "reboot", "launchctl", "systemctl",
        "service", "diskutil", "mount", "umount", "ln", "truncate", "ssh", "scp",
    ];
    if denied_programs.contains(&program.as_str()) {
        return Err(format!("program '{program}' is blocked"));
    }

    let denied_tokens = [
        "-delete", "--remove", "rm", "mv", "chmod", "chown", "sudo", "ssh", "scp", "rsync",
    ];
    for token in tokens.iter().skip(1) {
        let t = token.to_lowercase();
        if denied_tokens.contains(&t.as_str()) {
            return Err(format!("argument '{t}' is blocked"));
        }
    }

    Ok(())
}

pub(crate) fn execute_command(args: &Value) -> Result<String, String> {
    let command = args["command"].as_str().ok_or("Missing command")?;
    let cwd = args["cwd"].as_str().filter(|dir| !dir.trim().is_empty());
    let timeout = args["timeout"].as_u64().unwrap_or(30).clamp(1, 300);

    if let Err(reason) = validate_execute_command(command) {
        return Ok(format!("Command blocked: {reason}"));
    }

    let output = match crate::cmd::run::run_cmd_output_with_timeout(
        command,
        crate::cmd::run::RunCmdOptions { cwd },
        Duration::from_secs(timeout),
    ) {
        Ok(v) => v,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::TimedOut {
                return Ok("Command blocked: timeout".to_string());
            }
            return Err(format!("Failed to execute command: {}", e));
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    let stdout_trimmed = stdout.trim();
    let stderr_trimmed = stderr.trim();

    if output.status.success() {
        let combined = if stdout_trimmed.is_empty() {
            stderr_trimmed.to_string()
        } else if stderr_trimmed.is_empty() {
            stdout_trimmed.to_string()
        } else {
            format!("{stdout_trimmed}\n{stderr_trimmed}")
        };
        Ok(truncate_chars(combined.trim(), 16_000))
    } else {
        Ok(truncate_chars(
            &format!(
                "Exit code: {}\n{}\n{}",
                output.status.code().unwrap_or(-1),
                stdout_trimmed,
                stderr_trimmed
            ),
            16_000,
        ))
    }
}
