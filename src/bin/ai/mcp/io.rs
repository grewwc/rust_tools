use std::{
    io::BufRead,
    process::ChildStdout,
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

#[cfg(unix)]
fn wait_fd_readable(fd: i32, timeout_ms: u64) -> Result<(), String> {
    let timeout = if timeout_ms > i32::MAX as u64 {
        i32::MAX
    } else {
        timeout_ms as i32
    };

    loop {
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let rc = unsafe { libc::poll(&mut pfd, 1, timeout) };
        if rc > 0 {
            return Ok(());
        }
        if rc == 0 {
            return Err(format!("MCP response timeout after {} ms", timeout_ms));
        }

        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(format!("Failed waiting for MCP response: {}", err));
    }
}

#[cfg(not(unix))]
fn wait_fd_readable(_fd: i32, _timeout_ms: u64) -> Result<(), String> {
    Ok(())
}

pub(super) fn read_line_with_timeout_process(
    stdout: &mut std::io::BufReader<ChildStdout>,
    timeout_ms: u64,
    response_line: &mut String,
) -> Result<(), String> {
    let line = read_line_with_timeout_buf(stdout, stdout.get_ref().as_raw_fd(), timeout_ms)?;
    response_line.push_str(&line);
    Ok(())
}

pub(super) fn read_line_with_timeout_buf<R: std::io::Read>(
    stdout: &mut std::io::BufReader<R>,
    fd: i32,
    timeout_ms: u64,
) -> Result<String, String> {
    let timeout = Duration::from_millis(timeout_ms);
    let deadline = Instant::now() + timeout;
    let mut buf = Vec::<u8>::new();

    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(format!("MCP response timeout after {} ms", timeout_ms));
        }
        let remaining = deadline.saturating_duration_since(now).as_millis().max(1) as u64;

        wait_fd_readable(fd, remaining)?;

        let available = stdout
            .fill_buf()
            .map_err(|e| format!("Failed to read response: {}", e))?;
        if available.is_empty() {
            return Err("MCP server closed the stream unexpectedly".to_string());
        }

        if let Some(pos) = available.iter().position(|b| *b == b'\n') {
            buf.extend_from_slice(&available[..=pos]);
            stdout.consume(pos + 1);
            let line = String::from_utf8(buf)
                .map_err(|e| format!("MCP response is not valid UTF-8: {}", e))?;
            return Ok(line);
        }

        buf.extend_from_slice(available);
        let consumed = available.len();
        stdout.consume(consumed);
    }
}

#[cfg(unix)]
fn is_fd_readable_now(fd: i32) -> bool {
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN | libc::POLLHUP,
        revents: 0,
    };
    let rc = unsafe { libc::poll(&mut pfd, 1, 0) };
    rc > 0
}

#[cfg(not(unix))]
fn is_fd_readable_now(_fd: i32) -> bool {
    false
}

pub(super) fn read_available_buf<R: std::io::Read>(
    reader: &mut std::io::BufReader<R>,
    fd: i32,
) -> Vec<u8> {
    let mut out = Vec::new();

    loop {
        #[cfg(unix)]
        if !is_fd_readable_now(fd) {
            break;
        }

        let Ok(available) = reader.fill_buf() else {
            break;
        };
        if available.is_empty() {
            break;
        }

        out.extend_from_slice(available);
        let consumed = available.len();
        reader.consume(consumed);
    }

    out
}
