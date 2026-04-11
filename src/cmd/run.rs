//! 命令执行实现
//!
//! 提供系统命令的执行功能，包括超时控制和工作目录设置。

use crate::{commonw::utils::expanduser, strw::split::split_space_keep_symbol};

use std::{
    ffi::OsString,
    io::{self, Read},
    process::{Command, Output, Stdio},
    sync::mpsc::{self, RecvTimeoutError, Sender},
    thread,
    time::{Duration, Instant},
};

/// 命令执行选项
///
/// 用于配置命令执行的行为。
///
/// # 字段
///
/// * `cwd` - 可选的工作目录路径
///
/// # 示例
///
/// ```rust
/// use rust_tools::cmd::RunCmdOptions;
///
/// // 在当前目录执行
/// let opts = RunCmdOptions::default();
///
/// // 在指定目录执行
/// let opts = RunCmdOptions {
///     cwd: Some("/tmp"),
/// };
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RunCmdOptions<'a> {
    /// 命令执行的工作目录
    ///
    /// 如果为 `None`，则在当前进程的工作目录执行
    pub cwd: Option<&'a str>,
}

/// 标准化命令字符串
///
/// 去除首尾空白并检查是否为空。
///
/// # 参数
///
/// * `command` - 原始命令字符串
///
/// # 返回值
///
/// - `Ok(&str)` - 标准化后的命令
/// - `Err(io::Error)` - 命令为空
fn normalize_command(command: &str) -> io::Result<&str> {
    let command = command.trim();
    if command.is_empty() {
        return Err(io::Error::other("empty command"));
    }
    Ok(command)
}

/// 判断命令是否需要使用 Shell 执行
///
/// 检测命令中是否包含需要 Shell 解析的特殊字符：
/// - 管道符 `|`
/// - 重定向符 `>`, `<`
/// - 逻辑运算符 `&&`, `||`
/// - 分号 `;`
///
/// # 参数
///
/// * `command` - 命令字符串
///
/// # 返回值
///
/// 如果需要使用 Shell 返回 `true`，否则返回 `false`
fn should_use_shell(command: &str) -> bool {
    command.contains('|')
        || command.contains('>')
        || command.contains('<')
        || command.contains("&&")
        || command.contains("||")
        || command.contains(';')
        || command.contains('\'')
        || command.contains('`')
        || command.contains("$(")
        || command.contains("${")
        || command.contains('\\')
        || command.contains('&')
        || command.contains('*')
        || command.contains('?')
        || command.contains('#')
        || command.contains("<<")
        || command.contains('(')
        || command.contains(')')
}

/// 构建使用 Shell 的命令对象
///
/// 根据操作系统选择合适的 Shell：
/// - Windows: `cmd /C`
/// - Unix-like: `sh -c`
///
/// # 参数
///
/// * `command` - 要执行的命令
/// * `opts` - 执行选项
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

/// 构建命令对象
///
/// 自动判断是否需要使用 Shell，并构建相应的 `Command` 对象。
///
/// # 参数
///
/// * `command` - 要执行的命令
/// * `opts` - 执行选项
///
/// # 返回值
///
/// - `Ok(Command)` - 成功构建命令对象
/// - `Err(io::Error)` - 构建失败
fn build_command(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Command> {
    let command = normalize_command(command)?;
    if should_use_shell(command) {
        Ok(build_shell_command(command, opts))
    } else {
        build_no_shell_command(command, opts)
    }
}

/// 构建不使用 Shell 的命令对象
///
/// 直接解析命令和参数，避免 Shell 注入风险。
///
/// # 参数
///
/// * `command` - 要执行的命令（包含参数）
/// * `opts` - 执行选项
///
/// # 返回值
///
/// - `Ok(Command)` - 成功构建命令对象
/// - `Err(io::Error)` - 构建失败
fn build_no_shell_command(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Command> {
    let command = normalize_command(command)?;

    // 分割命令和参数
    let mut iter = split_space_keep_symbol(command, r#"""#);
    let Some(program) = iter.next() else {
        return Err(io::Error::other("empty command"));
    };

    let mut cmd = Command::new(program);
    if let Some(dir) = opts.cwd {
        cmd.current_dir(dir);
    }
    // 处理参数，展开用户路径
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

/// 执行命令并返回输出
///
/// 执行命令并捕获标准输出和标准错误。
///
/// # 参数
///
/// * `command` - 要执行的命令
/// * `opts` - 执行选项
///
/// # 返回值
///
/// - `Ok(Output)` - 命令执行成功，返回输出
/// - `Err(io::Error)` - 命令执行失败
///
/// # 示例
///
/// ```rust,no_run
/// use rust_tools::cmd::run_cmd_output;
///
/// let output = run_cmd_output("ls -la", Default::default())
///     .expect("命令执行失败");
///
/// println!("状态码：{}", output.status);
/// println!("输出：{}", String::from_utf8_lossy(&output.stdout));
/// ```
pub fn run_cmd_output(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Output> {
    build_command(command, opts)?.output()
}

/// 执行命令并带超时控制
///
/// 执行命令，如果在指定时间内未完成则终止。
///
/// # 参数
///
/// * `command` - 要执行的命令
/// * `opts` - 执行选项
/// * `timeout` - 超时时间
///
/// # 返回值
///
/// - `Ok(Output)` - 命令在超时前完成
/// - `Err(io::Error)` - 命令执行失败或超时
///   - `ErrorKind::TimedOut` - 命令超时
///
/// # 示例
///
/// ```rust,no_run
/// use rust_tools::cmd::run_cmd_output_with_timeout;
/// use std::time::Duration;
///
/// match run_cmd_output_with_timeout(
///     "sleep 5",
///     Default::default(),
///     Duration::from_secs(2),
/// ) {
///     Ok(output) => println!("完成：{}", String::from_utf8_lossy(&output.stdout)),
///     Err(e) if e.kind() == std::io::ErrorKind::TimedOut => println!("超时"),
///     Err(e) => println!("错误：{}", e),
/// }
/// ```
///
/// # 注意事项
///
/// - 超时后会尝试终止子进程
/// - 标准输入被设置为 null
pub fn run_cmd_output_with_timeout(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
) -> io::Result<Output> {
    run_cmd_output_streaming_with_timeout(command, opts, timeout, |_| {}, || false)
}

fn spawn_pipe_reader<R>(mut reader: R, tx: Sender<Vec<u8>>) -> thread::JoinHandle<io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut collected = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(read) => {
                    collected.extend_from_slice(&buf[..read]);
                    if tx.send(buf[..read].to_vec()).is_err() {
                        break;
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err),
            }
        }
        Ok(collected)
    })
}

fn join_pipe_reader(handle: thread::JoinHandle<io::Result<Vec<u8>>>) -> io::Result<Vec<u8>> {
    match handle.join() {
        Ok(result) => result,
        Err(_) => Err(io::Error::other("command output reader thread panicked")),
    }
}

pub fn run_cmd_output_streaming_with_timeout<F, C>(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
    mut on_chunk: F,
    should_cancel: C,
) -> io::Result<Output>
where
    F: FnMut(&[u8]),
    C: Fn() -> bool,
{
    let mut cmd = build_command(command, opts)?;

    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("missing stdout pipe"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("missing stderr pipe"))?;
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let stdout_handle = spawn_pipe_reader(stdout, tx.clone());
    let stderr_handle = spawn_pipe_reader(stderr, tx.clone());
    drop(tx);

    let deadline = Instant::now() + timeout;
    let status = loop {
        while let Ok(chunk) = rx.try_recv() {
            on_chunk(&chunk);
        }

        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if should_cancel() {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = join_pipe_reader(stdout_handle);
                    let _ = join_pipe_reader(stderr_handle);
                    while let Ok(chunk) = rx.try_recv() {
                        on_chunk(&chunk);
                    }
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = join_pipe_reader(stdout_handle);
                    let _ = join_pipe_reader(stderr_handle);
                    while let Ok(chunk) = rx.try_recv() {
                        on_chunk(&chunk);
                    }
                    return Err(io::Error::new(io::ErrorKind::TimedOut, "timeout"));
                }
                match rx.recv_timeout(Duration::from_millis(20)) {
                    Ok(chunk) => on_chunk(&chunk),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {}
                }
            }
            Err(err) => return Err(err),
        }
    };

    let stdout = join_pipe_reader(stdout_handle)?;
    let stderr = join_pipe_reader(stderr_handle)?;
    while let Ok(chunk) = rx.try_recv() {
        on_chunk(&chunk);
    }

    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

/// 执行命令并返回输出文本
///
/// 执行命令并将标准输出和标准错误合并为字符串返回。
///
/// # 参数
///
/// * `command` - 要执行的命令
///
/// # 返回值
///
/// - `Ok(String)` - 命令输出（stdout + stderr）
/// - `Err(io::Error)` - 命令执行失败
///
/// # 示例
///
/// ```rust,no_run
/// use rust_tools::cmd::run_cmd;
///
/// let output = run_cmd("echo Hello").expect("命令执行失败");
/// println!("输出：{}", output);
/// ```
///
/// # 注意事项
///
/// - 空命令会返回空字符串
/// - stdout 和 stderr 会被合并
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

#[cfg(test)]
mod tests {
    use super::{run_cmd, run_cmd_output, run_cmd_output_streaming_with_timeout, RunCmdOptions};
    use std::time::Duration;

    #[test]
    fn test_run_cmd_basic() {
        #[cfg(unix)]
        {
            let output = run_cmd("echo test").unwrap();
            assert!(output.contains("test"));
        }
    }

    #[test]
    fn test_run_cmd_empty() {
        let output = run_cmd("").unwrap();
        assert_eq!(output, "");
    }

    #[test]
    fn test_run_cmd_output_basic() {
        #[cfg(unix)]
        {
            let output = run_cmd_output("echo hello", RunCmdOptions::default()).unwrap();
            assert!(output.status.success());
            assert!(output.stdout.contains(&b'h'));
        }
    }

    #[test]
    fn test_run_cmd_output_streaming_collects_chunks() {
        #[cfg(unix)]
        {
            let mut streamed = Vec::new();
            let output = run_cmd_output_streaming_with_timeout(
                "printf 'hello\\nworld'",
                RunCmdOptions::default(),
                Duration::from_secs(2),
                |chunk| streamed.extend_from_slice(chunk),
                || false,
            )
            .unwrap();
            assert!(output.status.success());
            assert_eq!(String::from_utf8_lossy(&streamed), "hello\nworld");
        }
    }
}
