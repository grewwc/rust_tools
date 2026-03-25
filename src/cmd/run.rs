use crate::{common::utils::expanduser, strw::split::split_space_keep_symbol};

use std::{
    io,
    process::{Command, Output},
};

#[derive(Debug, Clone, Copy)]
pub struct RunCmdOptions<'a> {
    pub cwd: Option<&'a str>,
}

impl<'a> Default for RunCmdOptions<'a> {
    fn default() -> Self {
        Self { cwd: None }
    }
}

pub fn run_cmd_output(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Output> {
    if command.is_empty() {
        return Err(io::Error::other("empty command"));
    }

    if command.contains('|') || command.contains('>') || command.contains('<') {
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

        return cmd.output();
    }

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
            cmd.arg(new_arg.as_ref());
        } else {
            cmd.arg(new_arg.into_owned());
        }
    });

    cmd.output()
}

pub fn run_cmd(command: &str) -> io::Result<String> {
    if command.is_empty() {
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
