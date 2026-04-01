use serde_json::Value;

use crate::ai::tools::storage::command_runner;

fn truncate_chars(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let mut output = String::with_capacity(max_chars + 32);
    for (idx, ch) in content.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        output.push(ch);
    }
    output.push_str("\n... (truncated)");
    output
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
        "fish",
        "jshell",
        "rm",
        "mv",
        "dd",
        "chmod",
        "chown",
        "chgrp",
        "kill",
        "pkill",
        "killall",
        "sudo",
        "su",
        "passwd",
        "shutdown",
        "reboot",
        "launchctl",
        "systemctl",
        "service",
        "diskutil",
        "mount",
        "umount",
        "ln",
        "truncate",
        "ssh",
        "scp",
    ];
    if denied_programs.contains(&program.as_str()) {
        return Err(format!("program '{program}' is blocked"));
    }

    let denied_tokens = [
        "-delete", "--remove", "rm", "mv", "chmod", "chown", "sudo", "ssh", "scp", "rsync",
    ];
    for token in tokens.iter().skip(1) {
        let token = token.to_lowercase();
        if denied_tokens.contains(&token.as_str()) {
            return Err(format!("argument '{token}' is blocked"));
        }
    }

    Ok(())
}

pub(crate) fn execute_command(args: &Value) -> Result<String, String> {
    let command = args["command"].as_str().ok_or("Missing command")?;
    let cwd = args["cwd"].as_str().filter(|dir| !dir.trim().is_empty());
    let timeout = args["timeout"].as_u64().unwrap_or(60).clamp(1, 300);

    if let Err(reason) = validate_execute_command(command) {
        return Ok(format!("Command blocked: {reason}"));
    }

    let output = command_runner::run_command(command, cwd, timeout)?;
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
