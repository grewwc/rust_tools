use crate::{common::utils::expanduser, strw::split::split_space_keep_symbol};

use std::{
    ffi::OsString,
    io,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct RunCmdOptions<'a> {
    pub cwd: Option<&'a str>,
}

fn normalize_command(command: &str) -> io::Result<&str> {
    let command = command.trim();
    if command.is_empty() {
        return Err(io::Error::other("empty command"));
    }
    Ok(command)
}

fn should_use_shell(command: &str) -> bool {
    command.contains('|')
        || command.contains('>')
        || command.contains('<')
        || command.contains("&&")
        || command.contains("||")
        || command.contains(';')
}

fn build_shell_command(command: &str, opts: RunCmdOptions<'_>) -> Command {
    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = Command::new("cmd");
        c.args(["/C", command]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c
    };
    if let Some(dir) = opts.cwd {
        cmd.current_dir(dir);
    }
    cmd
}

fn build_command(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Command> {
    let command = normalize_command(command)?;
    if should_use_shell(command) {
        Ok(build_shell_command(command, opts))
    } else {
        build_no_shell_command(command, opts)
    }
}

fn build_no_shell_command(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Command> {
    let command = normalize_command(command)?;

    let mut iter = split_space_keep_symbol(command, r#"""#);
    let Some(program) = iter.next() else {
        return Err(io::Error::other("empty command"));
    };

    let mut cmd = Command::new(program);
    if let Some(dir) = opts.cwd {
        cmd.current_dir(dir);
    }
    iter.for_each(|arg| {
        let new_arg = expanduser(arg);
        if new_arg == arg {
            cmd.arg(OsString::from(new_arg.as_ref()));
        } else {
            cmd.arg(OsString::from(new_arg.into_owned()));
        }
    });
    Ok(cmd)
}

pub fn run_cmd_output(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Output> {
    build_command(command, opts)?.output()
}

pub fn run_cmd_output_with_timeout(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
) -> io::Result<Output> {
    let mut cmd = build_command(command, opts)?;

    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn()?;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(io::Error::new(io::ErrorKind::TimedOut, "timeout"));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(e),
        }
    }
    child.wait_with_output()
}

pub fn run_cmd(command: &str) -> io::Result<String> {
    if command.trim().is_empty() {
        return Ok("".to_owned());
    }

    let output = run_cmd_output(command, RunCmdOptions::default())?;
    let mut result = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !stderr.is_empty() {
        result.push_str(&stderr);
    }
    Ok(result)
}
