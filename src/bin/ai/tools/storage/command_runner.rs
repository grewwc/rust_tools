use std::process::Output;
use std::sync::atomic::Ordering;
use std::time::Duration;

pub(crate) fn run_command(
    command: &str,
    cwd: Option<&str>,
    timeout_secs: u64,
) -> Result<Output, String> {
    crate::cmd::run::run_cmd_output_with_timeout_non_interactive(
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
    pseudo_terminal: bool,
    on_chunk: F,
) -> Result<crate::cmd::run::CommandRunResult, String>
where
    F: FnMut(&[u8]),
{
    // 命令若用 `&` 派生了常驻后台服务（如 `python app.py &`），前台返回后它会成为
    // 孤儿进程。把其进程组 pgid 登记到会话级注册表，会话结束时统一清理。
    let session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let on_background_group =
        |pgid| crate::ai::tools::storage::process_registry::register(&session_id, pgid);
    let opts = crate::cmd::run::RunCmdOptions { cwd };
    let result = if pseudo_terminal {
        crate::cmd::run::run_cmd_output_streaming_with_timeout_tracked_pseudo_terminal(
            command,
            opts,
            Duration::from_secs(timeout_secs),
            on_chunk,
            is_command_cancel_requested,
            on_background_group,
        )
    } else {
        crate::cmd::run::run_cmd_output_streaming_with_timeout_tracked_non_interactive(
            command,
            opts,
            Duration::from_secs(timeout_secs),
            on_chunk,
            is_command_cancel_requested,
            on_background_group,
        )
    };
    result.map_err(map_command_error)
}

fn is_command_cancel_requested() -> bool {
    crate::ai::tools::registry::common::is_tool_cancel_requested()
        || crate::ai::driver::runtime_ctx::try_current()
            .is_some_and(|context| context.app_proto.cancel_stream.load(Ordering::Acquire))
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
