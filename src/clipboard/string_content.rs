//! 文本剪贴板内容处理
//!
//! 提供文本剪贴板的读取、写入和文件操作功能。
//! 支持本地剪贴板和 SSH 会话中的 OSC52 协议。

use std::{
    fmt::Display,
    fs,
    io::{self, Error, Read, Write},
    time::Duration,
};

use crate::common::filename::add_suffix;

/// 非文本文件错误
#[derive(Debug)]
struct NonTextErr(String);

impl NonTextErr {
    fn new(msg: String) -> Self {
        Self(msg)
    }
}

impl Display for NonTextErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for NonTextErr {}

/// 检查当前是否为 SSH 会话
///
/// 通过检查以下环境变量判断：
/// - `SSH_CONNECTION`
/// - `SSH_CLIENT`
/// - `SSH_TTY`
///
/// # 返回值
///
/// 如果在 SSH 会话中返回 `true`，否则返回 `false`
fn is_ssh_session() -> bool {
    std::env::var("SSH_CONNECTION").is_ok()
        || std::env::var("SSH_CLIENT").is_ok()
        || std::env::var("SSH_TTY").is_ok()
}

/// 通过 OSC52 转义序列设置剪贴板内容
///
/// OSC52 是一种终端转义序列，允许应用程序设置终端的剪贴板内容。
/// 在 SSH 会话中特别有用，因为可以在远程服务器上操作本地剪贴板。
///
/// # 参数
///
/// * `content` - 要复制到剪贴板的文本内容
///
/// # 返回值
///
/// - `Ok(())` - 成功设置剪贴板
/// - `Err(...)` - 设置失败
///
/// # 工作原理
///
/// 1. 将内容 Base64 编码
/// 2. 发送 OSC52 转义序列：`\x1b]52;c;<base64>\x07`
/// 3. 终端解码并设置剪贴板
fn set_clipboard_via_osc52(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    use base64::Engine as _;
    use base64::engine::general_purpose;

    let encoded = general_purpose::STANDARD.encode(content);
    let osc52 = format!("\x1b]52;c;{}\x07", encoded);

    let mut stdout = io::stdout();
    stdout.write_all(osc52.as_bytes())?;
    stdout.flush()?;

    Ok(())
}

/// 检查标准输入是否为 TTY
fn stdin_is_tty() -> bool {
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

/// 检查标准输出是否为 TTY
fn stdout_is_tty() -> bool {
    unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 }
}

/// 通过 OSC52 终端查询读取原始字节
///
/// 发送 OSC52 查询序列并等待终端响应，返回剪贴板的原始字节数据。
/// 不要求内容是有效的 UTF-8。
///
/// # 返回值
///
/// - `Some(Vec<u8>)` - 成功读取剪贴板字节
/// - `None` - 读取失败（非 TTY、超时等）
///
/// # 注意事项
///
/// - 需要终端支持 OSC52 查询
/// - 会临时修改终端设置为非规范模式
/// - 超时时间为 1.5 秒
fn read_osc52_bytes() -> Option<Vec<u8>> {
    use std::os::unix::io::AsRawFd;

    if !stdin_is_tty() || !stdout_is_tty() {
        return None;
    }

    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    // 发送查询请求
    stdout.write_all(b"\x1b]52;c;?\x07").ok()?;
    stdout.flush().ok()?;

    let stdin = io::stdin();
    let mut stdin = stdin.lock();

    // 保存原始终端设置
    let fd = stdin.as_raw_fd();
    let mut original_termios: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut original_termios) } != 0 {
        return None;
    }

    // 设置为非规范模式（立即返回）
    let mut new_termios = original_termios;
    new_termios.c_lflag &= !(libc::ICANON | libc::ECHO);
    new_termios.c_cc[libc::VMIN] = 0;
    new_termios.c_cc[libc::VTIME] = 1;

    if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &new_termios) } != 0 {
        return None;
    }

    // 读取响应
    let result = (|| {
        let mut response = Vec::new();
        let mut buf = [0u8; 1024];
        let start = std::time::Instant::now();
        let timeout = Duration::from_millis(1500);

        while start.elapsed() < timeout {
            match stdin.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    response.extend_from_slice(&buf[..n]);
                    // 检查响应结束标记
                    if response.contains(&b'\x07') || response.windows(2).any(|w| w == b"\x1b\\") {
                        break;
                    }
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }

        // 解析响应，提取 Base64 数据
        let response_str = String::from_utf8_lossy(&response);
        if let Some(start_idx) = response_str.find("]52;c;") {
            let data_start = start_idx + 6;
            use base64::Engine as _;
            use base64::engine::general_purpose;
            if let Some(end_idx) = response_str[data_start..].find('\x07') {
                let base64_data = &response_str[data_start..data_start + end_idx];
                return general_purpose::STANDARD.decode(base64_data).ok();
            }
            if let Some(end_idx) = response_str[data_start..].find("\x1b\\") {
                let base64_data = &response_str[data_start..data_start + end_idx];
                return general_purpose::STANDARD.decode(base64_data).ok();
            }
        }
        None
    })();

    // 恢复原始终端设置
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, &original_termios) };

    result
}

/// 通过 OSC52 读取剪贴板文本
fn get_clipboard_via_osc52() -> Option<String> {
    read_osc52_bytes().and_then(|bytes| String::from_utf8(bytes).ok())
}

/// 通过 OSC52 读取剪贴板原始字节（仅 SSH 会话）
///
/// # 返回值
///
/// - `Some(Vec<u8>)` - 在 SSH 会话中成功读取剪贴板字节
/// - `None` - 非 SSH 会话或读取失败
///
/// # 示例
///
/// ```rust,no_run
/// use rust_tools::clipboard::string_content::get_clipboard_raw_bytes_via_osc52;
///
/// if let Some(bytes) = get_clipboard_raw_bytes_via_osc52() {
///     println!("读取到 {} 字节", bytes.len());
/// }
/// ```
pub fn get_clipboard_raw_bytes_via_osc52() -> Option<Vec<u8>> {
    if is_ssh_session() {
        read_osc52_bytes()
    } else {
        None
    }
}

/// 保存剪贴板内容到文件
///
/// 读取当前剪贴板文本内容并保存到指定文件。
/// 如果文件名没有扩展名，会自动添加 `.txt` 后缀。
///
/// # 参数
///
/// * `fname` - 目标文件名
///
/// # 返回值
///
/// - `Ok(())` - 成功保存
/// - `Err(io::Error)` - 保存失败（剪贴板为空或 IO 错误）
///
/// # 示例
///
/// ```rust,no_run
/// use rust_tools::clipboard::string_content::save_to_file;
///
/// save_to_file("clipboard_content.txt").expect("保存失败");
/// ```
pub fn save_to_file(fname: &str) -> io::Result<()> {
    let fname = add_suffix(fname, ".txt", || !fname.contains('.'));
    let text = get_clipboard_content();
    if !text.is_empty() {
        fs::write(fname.as_str(), text)?;
        println!("save to file: {fname}");
        Ok(())
    } else {
        Err(Error::other("no text"))
    }
}

/// 从文件复制内容到剪贴板
///
/// 读取文件内容并设置到剪贴板。
///
/// # 参数
///
/// * `fname` - 源文件名
///
/// # 返回值
///
/// - `Ok(())` - 成功复制
/// - `Err(...)` - 复制失败（文件为空或不是文本文件）
///
/// # 示例
///
/// ```rust,no_run
/// use rust_tools::clipboard::string_content::copy_from_file;
///
/// copy_from_file("document.txt").expect("复制失败");
/// ```
pub fn copy_from_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    let text = match fs::read_to_string(fname) {
        Ok(text) => text,
        Err(_) => "".to_string(),
    };
    if text.is_empty() {
        return Err(Box::new(NonTextErr::new(format!(
            "{} is not text file.",
            fname
        ))));
    }
    set_clipboard_content(&text)?;
    Ok(())
}

/// 获取剪贴板文本内容
///
/// 自动检测环境并使用合适的方式读取剪贴板：
/// - 本地会话：使用 `arboard` crate
/// - SSH 会话：使用 OSC52 协议
///
/// # 返回值
///
/// 返回剪贴板文本内容，如果读取失败则返回空字符串。
///
/// # 示例
///
/// ```rust,no_run
/// use rust_tools::clipboard::string_content::get_clipboard_content;
///
/// let text = get_clipboard_content();
/// println!("剪贴板内容：{}", text);
/// ```
pub fn get_clipboard_content() -> String {
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => clipboard.get_text().unwrap_or_default(),
        Err(_) => {
            if is_ssh_session() {
                get_clipboard_via_osc52().unwrap_or_default()
            } else {
                String::new()
            }
        }
    }
}

/// 设置剪贴板文本内容
///
/// 自动检测环境并使用合适的方式设置剪贴板：
/// - 本地会话：使用 `arboard` crate
/// - SSH 会话：使用 OSC52 协议
///
/// # 参数
///
/// * `content` - 要设置的文本内容
///
/// # 返回值
///
/// - `Ok(())` - 成功设置
/// - `Err(...)` - 设置失败
///
/// # 示例
///
/// ```rust,no_run
/// use rust_tools::clipboard::string_content::set_clipboard_content;
///
/// set_clipboard_content("Hello, World!").expect("设置失败");
/// ```
pub fn set_clipboard_content(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => {
            clipboard.set_text(content.to_string())?;
            Ok(())
        }
        Err(_) => {
            if is_ssh_session() {
                set_clipboard_via_osc52(content)
            } else {
                Err("failed to set clipboard content".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::is_ssh_session;

    #[test]
    fn test_ssh_session_detection() {
        // 这个测试取决于环境变量
        // 只是验证函数不会 panic
        let _ = is_ssh_session();
    }
}
