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

pub(crate) fn run_command_streaming<F>(
    command: &str,
    cwd: Option<&str>,
    timeout_secs: u64,
    on_chunk: F,
) -> Result<crate::cmd::run::CommandRunResult, String>
where
    F: FnMut(&[u8]),
{
    // 命令若用 `&` 派生了常驻后台服务（如 `python app.py &`），前台返回后它会成为
    // 孤儿进程。把其进程组 pgid 登记到会话级注册表，会话结束时统一清理。
    let session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    crate::cmd::run::run_cmd_output_streaming_with_timeout_tracked(
        command,
        crate::cmd::run::RunCmdOptions { cwd },
        Duration::from_secs(timeout_secs),
        on_chunk,
        crate::ai::tools::registry::common::is_tool_cancel_requested,
        |pgid| crate::ai::tools::storage::process_registry::register(&session_id, pgid),
    )
    .map_err(map_command_error)
}

fn map_command_error(err: std::io::Error) -> String {
    if err.kind() == std::io::ErrorKind::TimedOut {
        "Command blocked: timeout".to_string()
    } else if err.kind() == std::io::ErrorKind::Interrupted {
        "Command blocked: cancelled".to_string()
    } else {
        format!("Failed to execute command: {}", err)
    }
}
