use std::process::Output;
use std::time::Duration;

pub(crate) fn run_command(
    command: &str,
    cwd: Option<&str>,
    timeout_secs: u64,
) -> Result<Output, String> {
    crate::cmd::run::run_cmd_output_with_timeout(
        command,
        crate::cmd::run::RunCmdOptions { cwd },
        Duration::from_secs(timeout_secs),
    )
    .map_err(map_command_error)
}

fn map_command_error(err: std::io::Error) -> String {
    if err.kind() == std::io::ErrorKind::TimedOut {
        "Command blocked: timeout".to_string()
    } else {
        format!("Failed to execute command: {}", err)
    }
}
