use serde_json::Value;
use std::process::Output;

use crate::ai::tools::storage::command_runner;

const MAX_COMMAND_OUTPUT_CHARS: usize = 16_000;

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
    let normalize_path = |path: &std::path::Path| {
        let mut normalized = std::path::PathBuf::new();
        for component in path.components() {
            match component {
                std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
                std::path::Component::RootDir => normalized.push(component.as_os_str()),
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    normalized.pop();
                }
                std::path::Component::Normal(part) => normalized.push(part),
            }
        }
        normalized
    };

    if program == "rm" || program == "mv" {
        let base_dir = std::env::current_dir()
            .map_err(|err| format!("failed to resolve current directory: {err}"))?;
        let base_dir = normalize_path(&base_dir);
        let mut path_args: Vec<String> = Vec::new();
        let mut iter = tokens.iter().skip(1).peekable();
        let mut end_of_options = false;

        while let Some(token) = iter.next() {
            if !end_of_options {
                if *token == "--" {
                    end_of_options = true;
                    continue;
                }

                if token.starts_with('-') {
                    if program == "mv" {
                        let option = token.to_lowercase();
                        if option == "-t" || option == "--target-directory" {
                            let dir = iter
                                .next()
                                .ok_or_else(|| format!("missing target directory for '{token}'"))?;
                            path_args.push((*dir).to_string());
                            continue;
                        }

                        if let Some(dir) = option.strip_prefix("--target-directory=") {
                            if dir.is_empty() {
                                return Err(format!("missing target directory for '{token}'"));
                            }
                            path_args.push(dir.to_string());
                            continue;
                        }

                        if token.starts_with("-t") && token.len() > 2 {
                            path_args.push(token[2..].to_string());
                            continue;
                        }
                    }

                    continue;
                }
            }

            path_args.push((*token).to_string());
        }

        if path_args.is_empty() {
            return Err(format!("program '{program}' requires path arguments"));
        }

        for raw_path in path_args {
            let raw_path = raw_path.trim();
            if raw_path.is_empty() {
                return Err(format!("program '{program}' contains an empty path"));
            }

            let resolved = if std::path::Path::new(raw_path).is_absolute() {
                normalize_path(std::path::Path::new(raw_path))
            } else {
                normalize_path(&base_dir.join(raw_path))
            };

            if !resolved.starts_with(&base_dir) {
                return Err(format!(
                    "path '{raw_path}' is outside the current directory"
                ));
            }
        }

        return Ok(());
    }

    let denied_programs = [
        "fish",
        "jshell",
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

fn format_command_output(output: Output) -> String {
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
        truncate_chars(combined.trim(), MAX_COMMAND_OUTPUT_CHARS)
    } else {
        truncate_chars(
            &format!(
                "Exit code: {}\n{}\n{}",
                output.status.code().unwrap_or(-1),
                stdout_trimmed,
                stderr_trimmed
            ),
            MAX_COMMAND_OUTPUT_CHARS,
        )
    }
}

fn execute_command_inner<F>(args: &Value, on_chunk: F) -> Result<String, String>
where
    F: FnMut(&[u8]),
{
    let command = args["command"].as_str().ok_or("Missing command")?;
    let cwd = args["cwd"].as_str().filter(|dir| !dir.trim().is_empty());
    let timeout = args["timeout"].as_u64().unwrap_or(60).clamp(1, 300);

    if let Err(reason) = validate_execute_command(command) {
        return Ok(format!("Command blocked: {reason}"));
    }

    let output = command_runner::run_command_streaming(command, cwd, timeout, on_chunk)?;
    Ok(format_command_output(output))
}

pub(crate) fn execute_command(args: &Value) -> Result<String, String> {
    execute_command_inner(args, |_| {})
}

pub(crate) fn execute_command_streaming<F>(args: &Value, on_chunk: F) -> Result<String, String>
where
    F: FnMut(&[u8]),
{
    execute_command_inner(args, on_chunk)
}
