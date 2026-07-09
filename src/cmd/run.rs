//! 命令执行实现
//!
//! 提供系统命令的执行功能，包括超时控制和工作目录设置。

use crate::{commonw::utils::expanduser, strw::split::split_space_keep_symbol};

use std::{
    ffi::OsString,
    io::{self, Read},
    process::{Child, Command, Output, Stdio},
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender},
    thread,
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

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

fn is_shell_boundary(byte: Option<u8>) -> bool {
    byte.is_none_or(|b| {
        b.is_ascii_whitespace() || matches!(b, b'|' | b'&' | b';' | b'<' | b'>' | b'(' | b')')
    })
}

/// 判断命令是否需要使用 Shell 执行。
///
/// 这里只识别“引号外确实需要 shell 解释”的语法，避免把双引号内的字面量
/// `<` / `>` / `|` 之类误判为必须走 `sh -c`。
///
/// 注意：本函数仍保守地把单引号、反斜杠等视为需要 shell，因为当前
/// `build_no_shell_command` 只实现了双引号分组，不负责完整复刻 shell 的
/// 转义/引用语义。
fn should_use_shell(command: &str) -> bool {
    let bytes = command.as_bytes();
    let mut i = 0usize;
    let mut in_double = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_double {
            match b {
                b'\\' | b'`' => return true,
                b'"' => {
                    in_double = false;
                    i += 1;
                    continue;
                }
                b'$' if i + 1 < bytes.len() && matches!(bytes[i + 1], b'(' | b'{') => {
                    return true;
                }
                _ => {
                    i += 1;
                    continue;
                }
            }
        }

        match b {
            b'"' => {
                in_double = true;
            }
            b'\'' | b'\\' | b'`' => return true,
            b'$' if i + 1 < bytes.len() && matches!(bytes[i + 1], b'(' | b'{') => return true,
            b'|' | b'>' | b'<' | b';' | b'&' | b'*' | b'?' => return true,
            b'#' if is_shell_boundary(i.checked_sub(1).map(|idx| bytes[idx])) => return true,
            // `(` 只有在 token 起始边界上才视为 shell 分组/子 shell 语法；
            // 普通参数里的 `foo(bar)` 不应该把命令强行送进 shell。
            b'(' if is_shell_boundary(i.checked_sub(1).map(|idx| bytes[idx])) => {
                return true;
            }
            // `)` 只有在 token 结束边界上才视为 shell 分组闭合。
            b')' if is_shell_boundary(bytes.get(i + 1).copied()) => {
                return true;
            }
            b'\n' => return true,
            _ => {}
        }
        i += 1;
    }
    false
}

/// 返回命令是否会走 shell 执行路径。
///
/// 供上层安全校验复用，确保验证语义与实际执行语义保持一致。
pub fn command_requires_shell(command: &str) -> bool {
    should_use_shell(command)
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
    // 处理参数，展开用户路径。
    // 非 shell 路径下，双引号只承担“分组”职责，不应作为字面量传给子进程。
    // 这里不尝试复刻完整 shell 语义；更复杂的单引号/反斜杠转义仍由
    // `should_use_shell` 保守地导向 shell 路径。
    iter.for_each(|arg| {
        let normalized_arg = if arg.contains('"') {
            arg.replace('"', "")
        } else {
            arg.to_string()
        };
        let new_arg = expanduser(&normalized_arg);
        if new_arg == normalized_arg {
            cmd.arg(OsString::from(new_arg.as_ref()));
        } else {
            cmd.arg(OsString::from(new_arg.into_owned()));
        }
    });
    Ok(cmd)
}

#[cfg(unix)]
fn configure_child_process_group(cmd: &mut Command) {
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn configure_child_process_group(_cmd: &mut Command) {}

#[cfg(unix)]
fn terminate_child(child: &mut Child) {
    let pgid = child.id() as libc::pid_t;
    unsafe {
        let _ = libc::kill(-pgid, libc::SIGKILL);
    }
    let _ = child.kill();
}

#[cfg(not(unix))]
fn terminate_child(child: &mut Child) {
    let _ = child.kill();
}

/// 前台命令退出后，检测其进程组内是否仍有存活成员（典型：命令用 `&` 派生的
/// 常驻服务，如 `python app.py &`）。`kill(-pgid, 0)` 不发送信号，仅用于探测：
/// 返回 0 表示进程组仍存在成员，`ESRCH` 表示已无成员。
///
/// 因为子进程通过 `setsid` 成为组长，`child.id()` 即该进程组的 pgid；即使前台
/// 组长已被 reap，只要还有后台成员存活，pgid 就不会被回收，此探测是安全的。
#[cfg(unix)]
fn background_group_alive(pgid: u32) -> bool {
    unsafe { libc::kill(-(pgid as libc::pid_t), 0) == 0 }
}

#[cfg(not(unix))]
fn background_group_alive(_pgid: u32) -> bool {
    false
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

#[derive(Clone, Copy)]
enum StreamKind {
    Stdout,
    Stderr,
}

/// 读取线程：把读到的数据按流类型打标签后经 channel 送回主线程累积。
///
/// 关键：不再在线程内累积并通过 `join` 回收。因为命令若在后台派生常驻进程
/// （如 `python app.py &` 启动的 Flask），该子进程会继承同一个 stdout/stderr
/// 管道写端 fd，导致 `read()` 永远等不到 EOF、线程永不退出。主线程一旦 `join`
/// 这样的线程就会死锁。改为纯 channel 汇报后，主线程可在前台命令结束时直接
/// 返回、把读取线程作为 daemon 泄漏（其阻塞在无害的管道读上，进程退出即回收）。
fn spawn_pipe_reader<R>(mut reader: R, kind: StreamKind, tx: Sender<(StreamKind, Vec<u8>)>)
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(read) => {
                    if tx.send((kind, buf[..read].to_vec())).is_err() {
                        break;
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => {
                    let _ = err;
                    break;
                }
            }
        }
    });
}

/// 前台命令退出后，给读取线程一个短暂的宽限期把「已经缓冲在管道里」的尾部
/// 输出排空，然后返回。若命令派生了继承管道的后台进程，channel 永远不会
/// `Disconnected`，因此必须用固定宽限期兜底，绝不能无限等待。
const DRAIN_GRACE: Duration = Duration::from_millis(100);

fn drain_channel<F>(
    rx: &Receiver<(StreamKind, Vec<u8>)>,
    stdout_buf: &mut Vec<u8>,
    stderr_buf: &mut Vec<u8>,
    on_chunk: &mut F,
    grace: Duration,
) where
    F: FnMut(&[u8]),
{
    let deadline = Instant::now() + grace;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            // 仍尽量取走已就绪的数据，但不再等待。
            while let Ok((kind, chunk)) = rx.try_recv() {
                accumulate_chunk(kind, &chunk, stdout_buf, stderr_buf, on_chunk);
            }
            break;
        }
        match rx.recv_timeout(remaining) {
            Ok((kind, chunk)) => accumulate_chunk(kind, &chunk, stdout_buf, stderr_buf, on_chunk),
            Err(RecvTimeoutError::Timeout) => break,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn accumulate_chunk<F>(
    kind: StreamKind,
    chunk: &[u8],
    stdout_buf: &mut Vec<u8>,
    stderr_buf: &mut Vec<u8>,
    on_chunk: &mut F,
) where
    F: FnMut(&[u8]),
{
    match kind {
        StreamKind::Stdout => stdout_buf.extend_from_slice(chunk),
        StreamKind::Stderr => stderr_buf.extend_from_slice(chunk),
    }
    on_chunk(chunk);
}

pub fn run_cmd_output_streaming_with_timeout<F, C>(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
    on_chunk: F,
    should_cancel: C,
) -> io::Result<Output>
where
    F: FnMut(&[u8]),
    C: Fn() -> bool,
{
    run_cmd_output_streaming_with_timeout_tracked(
        command,
        opts,
        timeout,
        on_chunk,
        should_cancel,
        |_| {},
    )
}

/// 与 [`run_cmd_output_streaming_with_timeout`] 相同，但额外接收
/// `on_background_group` 回调：当前台命令正常退出后，如果其进程组内仍有存活
/// 成员（命令用 `&` 派生了常驻服务，如 `python app.py &`），会以该进程组的
/// pgid 调用一次回调。上层可据此把 pgid 登记到会话级注册表，在会话结束时统一
/// 清理，避免遗留孤儿进程。
///
/// 注意：pgid 仅在当前进程存活期间有意义，不应持久化到磁盘——重启后同一数值
/// 可能被无关进程复用，届时 `killpg` 会误杀。
pub fn run_cmd_output_streaming_with_timeout_tracked<F, C, G>(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
    mut on_chunk: F,
    should_cancel: C,
    mut on_background_group: G,
) -> io::Result<Output>
where
    F: FnMut(&[u8]),
    C: Fn() -> bool,
    G: FnMut(u32),
{
    let mut cmd = build_command(command, opts)?;

    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    configure_child_process_group(&mut cmd);

    let mut child = cmd.spawn()?;
    let pgid = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("missing stdout pipe"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("missing stderr pipe"))?;
    let (tx, rx) = mpsc::channel::<(StreamKind, Vec<u8>)>();
    spawn_pipe_reader(stdout, StreamKind::Stdout, tx.clone());
    spawn_pipe_reader(stderr, StreamKind::Stderr, tx.clone());
    drop(tx);

    let mut stdout_buf = Vec::new();
    let mut stderr_buf = Vec::new();

    let deadline = Instant::now() + timeout;
    let status = loop {
        while let Ok((kind, chunk)) = rx.try_recv() {
            accumulate_chunk(
                kind,
                &chunk,
                &mut stdout_buf,
                &mut stderr_buf,
                &mut on_chunk,
            );
        }

        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if should_cancel() {
                    // 取消：杀掉整个进程组（含后台派生进程），管道随之关闭，读取线程退出。
                    terminate_child(&mut child);
                    let _ = child.wait();
                    drain_channel(
                        &rx,
                        &mut stdout_buf,
                        &mut stderr_buf,
                        &mut on_chunk,
                        DRAIN_GRACE,
                    );
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
                }
                if Instant::now() >= deadline {
                    // 超时：同样杀掉整个进程组，避免后台进程继续持有管道。
                    terminate_child(&mut child);
                    let _ = child.wait();
                    drain_channel(
                        &rx,
                        &mut stdout_buf,
                        &mut stderr_buf,
                        &mut on_chunk,
                        DRAIN_GRACE,
                    );
                    return Err(io::Error::new(io::ErrorKind::TimedOut, "timeout"));
                }
                match rx.recv_timeout(Duration::from_millis(20)) {
                    Ok((kind, chunk)) => accumulate_chunk(
                        kind,
                        &chunk,
                        &mut stdout_buf,
                        &mut stderr_buf,
                        &mut on_chunk,
                    ),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {}
                }
            }
            Err(err) => return Err(err),
        }
    };

    // 前台命令已退出。若它派生了继承管道的后台进程（如 `flask &`），channel
    // 不会 Disconnected，这里用固定宽限期排空尾部输出后返回，绝不 join 读取线程。
    drain_channel(
        &rx,
        &mut stdout_buf,
        &mut stderr_buf,
        &mut on_chunk,
        DRAIN_GRACE,
    );

    // 前台组长已退出，但若进程组内仍有后台成员存活，上报 pgid 以便会话级清理。
    if background_group_alive(pgid) {
        on_background_group(pgid);
    }

    Ok(Output {
        status,
        stdout: stdout_buf,
        stderr: stderr_buf,
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
    use super::{
        RunCmdOptions, run_cmd, run_cmd_output, run_cmd_output_streaming_with_timeout,
        should_use_shell,
    };
    use std::time::{Duration, Instant};

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
    fn test_should_use_shell_ignores_double_quoted_literals() {
        assert!(!should_use_shell(r#"printf "%s" "<literal>|foo(bar)#bar""#));
        assert!(!should_use_shell(r#"printf "%s" ">(literal)""#));
    }

    #[test]
    fn test_should_use_shell_detects_real_shell_syntax_outside_quotes() {
        assert!(should_use_shell("cat < input.txt"));
        assert!(should_use_shell("echo hi | wc -c"));
        assert!(should_use_shell("diff <(echo a) <(echo b)"));
        assert!(should_use_shell("echo foo # comment"));
    }

    #[test]
    fn test_should_use_shell_does_not_treat_hash_in_word_as_comment() {
        assert!(!should_use_shell("printf %s foo#bar"));
    }

    #[test]
    fn test_run_cmd_output_preserves_double_quoted_lt_literal_without_shell() {
        #[cfg(unix)]
        {
            let output =
                run_cmd_output(r#"printf "%s" "<literal>""#, RunCmdOptions::default()).unwrap();
            assert!(output.status.success());
            assert_eq!(String::from_utf8_lossy(&output.stdout), "<literal>");
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

    #[test]
    fn test_timeout_kills_shell_descendants_without_waiting_for_them() {
        #[cfg(unix)]
        {
            let started = Instant::now();
            let result = run_cmd_output_streaming_with_timeout(
                "sh -c 'sleep 5'",
                RunCmdOptions::default(),
                Duration::from_millis(200),
                |_| {},
                || false,
            );

            assert!(matches!(
                result.as_ref().map_err(|err| err.kind()),
                Err(std::io::ErrorKind::TimedOut)
            ));
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "timeout waited for shell descendant to exit"
            );
        }
    }

    #[test]
    fn test_returns_promptly_when_command_backgrounds_a_long_lived_process() {
        // 回归：前台命令结束、但后台派生进程继承了 stdout/stderr 管道。
        // 旧实现会 join 永不退出的读取线程而死锁；新实现应在宽限期后返回。
        #[cfg(unix)]
        {
            let started = Instant::now();
            let mut streamed = Vec::new();
            let output = run_cmd_output_streaming_with_timeout(
                "sh -c 'sleep 30 & echo ready'",
                RunCmdOptions::default(),
                Duration::from_secs(10),
                |chunk| streamed.extend_from_slice(chunk),
                || false,
            )
            .expect("should return without hanging");

            assert!(output.status.success());
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("ready"),
                "expected foreground output to be captured"
            );
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "hung waiting for backgrounded descendant to close pipes"
            );
        }
    }

    #[test]
    fn test_reports_pgid_when_background_group_survives() {
        // 派生一个后台常驻进程后前台立即返回：应上报存活进程组的 pgid，
        // 供上层登记会话级清理。上报后测试自行杀掉该组，避免泄漏。
        #[cfg(unix)]
        {
            let mut reported: Vec<u32> = Vec::new();
            let output = super::run_cmd_output_streaming_with_timeout_tracked(
                "sh -c 'sleep 30 & echo up'",
                RunCmdOptions::default(),
                Duration::from_secs(10),
                |_| {},
                || false,
                |pgid| reported.push(pgid),
            )
            .expect("should return without hanging");

            assert!(output.status.success());
            assert_eq!(reported.len(), 1, "expected exactly one surviving group");
            let pgid = reported[0];
            assert!(pgid > 0);
            // 该进程组应确实存活。
            assert_eq!(unsafe { libc::kill(-(pgid as libc::pid_t), 0) }, 0);
            // 清理：杀掉整个组。
            unsafe {
                let _ = libc::kill(-(pgid as libc::pid_t), libc::SIGKILL);
            }
        }
    }

    #[test]
    fn test_does_not_report_pgid_for_clean_foreground_command() {
        // 纯前台命令退出后进程组内无成员，不应上报 pgid。
        #[cfg(unix)]
        {
            let mut reported: Vec<u32> = Vec::new();
            let output = super::run_cmd_output_streaming_with_timeout_tracked(
                "echo hello",
                RunCmdOptions::default(),
                Duration::from_secs(5),
                |_| {},
                || false,
                |pgid| reported.push(pgid),
            )
            .expect("should succeed");

            assert!(output.status.success());
            assert!(
                reported.is_empty(),
                "clean foreground command should not report a surviving group, got {reported:?}"
            );
        }
    }
}
