// side_note_input.rs — 流式输出期间的同终端 side-note 输入监听（Ctrl+G）
//
// 主 agent 在模型流式输出时 stdin 处于空闲（交互式输入框未打开）。本模块在 cbreak
// 模式（关闭 ICANON/ECHO，保留 ISIG/OPOST）下轮询监听 Ctrl+G（0x07）：命中后弹单行
// 输入，Esc / F2 / Alt+Enter 提交为 `from="user"` 的 side-note 写入 foreground 队列，
// 下一轮迭代由 `driver::side_note::poll_and_inject` 注入 LLM 上下文。输入中再次按
// Ctrl+G 放弃当前草稿；回车不发送（与主输入"回车=换行、Esc/F2 提交"的键位语义对齐）。
//
// 设计要点：
// - 保留 ISIG：Ctrl+C 仍走现有 SIGINT 中断路径，不破坏流式中断语义。
// - 保留 OPOST：主 agent 的渲染输出 `\n` → `\r\n` 转换不受影响。
// - 关闭 ECHO 自行回显：backspace 能按字符宽度擦除，且 Ctrl+G 本身无回显不干扰渲染。
// - poll 短轮询 + stop 标志：stream_response 结束（任意 return 路径）时由守卫请求
//   退出并等待终端恢复，避免遗留 cbreak 状态影响后续输入框。
use std::{
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

use tokio::task::JoinHandle;

use crate::ai::{
    driver::{runtime_ctx, side_note::push_side_note},
    theme::{ACCENT_MUTED, RESET},
};

/// Ctrl+G（BEL）
const CTRL_G: u8 = 0x07;
/// poll 轮询间隔（毫秒）：同时约束 stop 标志的响应延迟。
const POLL_MS: i32 = 50;
/// Drop 时等待监听任务收尾（恢复终端）的上限。
const SHUTDOWN_WAIT_MS: u64 = 250;
/// 启动时等待 listener 完成终端接管的上限；超时则不让尚未开始的 task 修改终端。
const STARTUP_WAIT_MS: u64 = 250;
/// 底部 composer 固定占用的物理行数。模型输出在其上方的滚动区域连续滚动。
const FOOTER_ROWS: u16 = 1;
const COMPOSER_PREFIX: &str = "  [side-note] Esc/F2 send · Ctrl+G cancel > ";
const COMPOSER_CURSOR: char = '▌';

/// 以 DECSTBM 保留的终端底部 footer。所有输出仍在主屏，避免 alternate screen
/// 隐藏 transcript；composer 每次重绘会保存/恢复输出光标，因此不影响模型持续输出。
struct FooterReservation {
    cols: u16,
    rows: u16,
}

impl FooterReservation {
    fn enter(stop: &AtomicBool) -> io::Result<Self> {
        if stop.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "side-note stopped",
            ));
        }
        let (cols, rows) = terminal_size()?;
        if rows <= FOOTER_ROWS {
            return Err(io::Error::other(
                "terminal is too short for side-note footer",
            ));
        }
        let footer = Self { cols, rows };
        footer.apply_reservation(stop)?;
        Ok(footer)
    }

    fn output_bottom(&self) -> u16 {
        self.rows - FOOTER_ROWS
    }

    fn apply_reservation(&self, stop: &AtomicBool) -> io::Result<()> {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        if stop.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "side-note stopped",
            ));
        }
        // 无条件把当前可见屏幕完整滚动两行，显式造出“输出区最后一行 + footer”两
        // 个空行。不能假定物理末行在 cursor 之后没有旧内容；只有先滚进 scrollback
        // 才能保证 leave() 只清理本 footer 自己创建的空白行。
        write!(
            out,
            "\x1b[{};1H\n\n\x1b[1;{}r\x1b[{};1H",
            self.rows,
            self.output_bottom(),
            self.output_bottom()
        )?;
        out.flush()
    }

    fn refresh(&mut self) -> io::Result<()> {
        let (cols, rows) = terminal_size()?;
        if cols == self.cols && rows == self.rows {
            return Ok(());
        }
        if rows <= FOOTER_ROWS {
            return Err(io::Error::other(
                "terminal is too short for side-note footer",
            ));
        }
        let old_rows = self.rows;
        self.cols = cols;
        self.rows = rows;
        let stdout = io::stdout();
        let mut out = stdout.lock();
        // 清旧 footer 后再重设滚动区域；保存/恢复保证模型输出光标不被 resize 重绘挪走。
        write!(
            out,
            "\x1b7\x1b[{};1H\x1b[2K\x1b[r\x1b[1;{}r\x1b8",
            old_rows,
            self.output_bottom()
        )?;
        out.flush()
    }

    fn draw(&mut self, input: &[char]) -> io::Result<()> {
        self.refresh()?;
        let (prefix, visible, caret) = composer_line_parts(input, self.cols as usize);
        let stdout = io::stdout();
        let mut out = stdout.lock();
        // stdout 的锁与 stream renderer 共用：整段控制序列不会被模型正文插入。
        // 不把真实 cursor 留在 footer，避免下一个模型/token 输出从输入行开始；尾部用
        // 可见的 block caret 表示编辑位置。
        write!(
            out,
            "\x1b7\x1b[?25l\x1b[{};1H\x1b[2K{ACCENT_MUTED}{prefix}{RESET}{visible}{ACCENT_MUTED}{caret}{RESET}\x1b8\x1b[?25h",
            self.rows
        )?;
        out.flush()
    }

    fn clear(&mut self) -> io::Result<()> {
        self.refresh()?;
        let footer_row = terminal_size()
            .map(|(_, rows)| rows)
            .unwrap_or(self.rows)
            .max(1);
        let stdout = io::stdout();
        let mut out = stdout.lock();
        write!(
            out,
            "\x1b7\x1b[?25l\x1b[{};1H\x1b[2K\x1b[0m\x1b8\x1b[?25h",
            footer_row
        )?;
        out.flush()
    }

    fn leave(&mut self) -> io::Result<()> {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        write!(out, "\x1b7\x1b[{};1H\x1b[2K\x1b[r\x1b8\x1b[0m", self.rows)?;
        out.flush()
    }
}

fn terminal_size() -> io::Result<(u16, u16)> {
    use std::os::unix::io::AsRawFd;

    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::ioctl(io::stdout().as_raw_fd(), libc::TIOCGWINSZ, &mut ws) };
    if rc == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
        Ok((ws.ws_col, ws.ws_row))
    } else {
        Err(io::Error::last_os_error())
    }
}

/// 是否启用同终端 side-note 输入监听：仅前台主 agent + 交互式 stdin + 终端输出开启。
pub(crate) fn side_note_input_enabled() -> bool {
    let term_out = runtime_ctx::terminal_output_enabled();
    let stdin_tty = io::stdin().is_terminal();
    let depth = runtime_ctx::current_subagent_depth();
    let enabled = term_out && stdin_tty && depth == 0;
    // 临时诊断（RUST_TOOLS_SIDE_NOTE_DEBUG=1 时打印各启用条件的真实值），
    // 定位"Ctrl+G 不生效"时监听器为何未启动；定位后移除。
    if std::env::var_os("RUST_TOOLS_SIDE_NOTE_DEBUG").is_some() {
        eprintln!(
            "[side-note-debug] enabled={enabled} term_out={term_out} stdin_tty={stdin_tty} depth={depth}"
        );
    }
    enabled
}

/// RAII 守卫：持有后台监听任务，drop 时请求退出并等待终端恢复。
pub(crate) struct SideNoteInputGuard {
    stop: Arc<AtomicBool>,
    task: Option<JoinHandle<()>>,
}

impl SideNoteInputGuard {
    pub(crate) fn spawn(history_file: PathBuf) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let task_stop = stop.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let task = tokio::task::spawn_blocking(move || {
            // 若 guard 已在 blocking task 获得线程前被 drop，绝不能事后切换 cbreak
            // 或设置滚动区域；否则下一轮 prompt 会继承孤立的终端状态。
            if task_stop.load(Ordering::Relaxed) {
                return;
            }
            let term = match CbreakTerm::enter() {
                Ok(term) => term,
                Err(_) => {
                    let _ = ready_tx.send(false);
                    return;
                }
            };
            if task_stop.load(Ordering::Relaxed) {
                return;
            }
            // 监听器只接管 stdin；未按 Ctrl+G 时绝不设置滚动区域或移动 stdout，
            // 避免普通 turn 的状态/正文被无条件推到终端底部。
            let _ = ready_tx.send(true);
            side_note_input_loop(&history_file, &task_stop, term);
        });
        if !ready_rx
            .recv_timeout(Duration::from_millis(STARTUP_WAIT_MS))
            .unwrap_or(false)
        {
            stop.store(true, Ordering::Relaxed);
        }
        SideNoteInputGuard {
            stop,
            task: Some(task),
        }
    }
}

impl Drop for SideNoteInputGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(task) = self.task.take() {
            let deadline = std::time::Instant::now() + Duration::from_millis(SHUTDOWN_WAIT_MS);
            while std::time::Instant::now() < deadline {
                if task.is_finished() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            // 不 abort：poll 以 POLL_MS(50ms) 粒度检查 stop，监听任务会自行清除
            // footer、重置滚动区域并恢复 CbreakTerm。abort 无法中断阻塞的 poll，反而
            // 会在终端尚未恢复时放行，与后续 prompt_user 的 raw mode 形成竞态。
        }
    }
}

/// cbreak 终端模式：关闭 ICANON（单键即时送达）与 ECHO（自行回显，便于按字符宽度
/// 擦除），保留 ISIG（Ctrl+C 仍触发 SIGINT）与 OPOST（输出换行转换不受影响）。
struct CbreakTerm {
    saved: libc::termios,
}

impl CbreakTerm {
    fn enter() -> io::Result<Self> {
        // SAFETY: 只读写 stdin 的 termios，无跨线程共享的可变访问。
        unsafe {
            let mut t = std::mem::zeroed::<libc::termios>();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut t) != 0 {
                return Err(io::Error::last_os_error());
            }
            let saved = t;
            t.c_lflag &= !(libc::ICANON | libc::ECHO);
            t.c_cc[libc::VMIN] = 1;
            t.c_cc[libc::VTIME] = 0;
            if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(CbreakTerm { saved })
        }
    }
}

impl Drop for CbreakTerm {
    fn drop(&mut self) {
        // SAFETY: 恢复进入前的原始终端状态。
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.saved);
        }
    }
}

/// 字符在终端中占用的列数（East Asian Wide / Fullwidth 计 2 列，其余 1 列）。
/// 用于输入回显与 backspace 擦除的精确列对齐。
fn char_width(c: char) -> usize {
    let cp = c as u32;
    if (0x1100..=0x115F).contains(&cp)
        || (0x2E80..=0xA4CF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp) // CJK 扩展 A
        || (0xAC00..=0xD7A3).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0xFE30..=0xFE4F).contains(&cp)
        || (0xFF00..=0xFF60).contains(&cp)
        || (0xFFE0..=0xFFE6).contains(&cp)
        || (0x1F300..=0x1F9FF).contains(&cp)
        || (0x20000..=0x2FFFD).contains(&cp)
    // CJK 扩展 B+
    {
        2
    } else {
        1
    }
}

fn display_width(chars: impl IntoIterator<Item = char>) -> usize {
    chars.into_iter().map(char_width).sum()
}

fn composer_layout(cols: usize) -> (&'static str, Option<char>) {
    let full_width = display_width(COMPOSER_PREFIX.chars()) + char_width(COMPOSER_CURSOR) + 1;
    if cols >= full_width {
        (COMPOSER_PREFIX, Some(COMPOSER_CURSOR))
    } else if cols >= 4 {
        ("> ", Some(COMPOSER_CURSOR))
    } else {
        ("", None)
    }
}

/// 把长输入裁成单行尾部视图，永远给 composer 的可见 caret 留一列，避免底部行换行。
fn input_viewport_with_layout(
    input: &[char],
    cols: usize,
    prefix: &str,
    caret: Option<char>,
) -> String {
    let fixed_width = display_width(prefix.chars()) + caret.map(char_width).unwrap_or(0) + 1;
    let available = cols.saturating_sub(fixed_width);
    if available == 0 {
        return String::new();
    }

    let mut tail = Vec::new();
    let mut width = 0;
    for &ch in input.iter().rev() {
        let char_width = char_width(ch);
        if width + char_width > available {
            break;
        }
        width += char_width;
        tail.push(ch);
    }
    tail.reverse();
    if tail.len() == input.len() {
        return tail.into_iter().collect();
    }

    while width > available.saturating_sub(1) {
        let Some(ch) = tail.first().copied() else {
            break;
        };
        width -= char_width(ch);
        tail.remove(0);
    }
    let mut visible = String::from('…');
    visible.extend(tail);
    visible
}

fn input_viewport(input: &[char], cols: usize) -> String {
    let (prefix, caret) = composer_layout(cols);
    input_viewport_with_layout(input, cols, prefix, caret)
}

fn composer_line_parts(input: &[char], cols: usize) -> (&'static str, String, String) {
    let (prefix, caret) = composer_layout(cols);
    let visible = input_viewport_with_layout(input, cols, prefix, caret);
    (prefix, visible, caret.map(String::from).unwrap_or_default())
}

fn redraw_input(footer: &mut FooterReservation, input: &[char]) -> io::Result<()> {
    footer.draw(input)
}

fn clear_input(footer: &mut FooterReservation) -> io::Result<()> {
    footer.clear()
}

fn refresh_footer(footer: &mut FooterReservation, input: Option<&[char]>) -> io::Result<()> {
    footer.refresh()?;
    if let Some(input) = input {
        footer.draw(input)?;
    }
    Ok(())
}

/// 发送当前草稿（Esc/F2/Alt+Enter 提交键共用）：内容为空时仅退出输入模式；
/// 非空时写入 foreground 队列。写入失败保留草稿并继续 composer，不静默丢失指令。
fn submit_draft(
    history_file: &Path,
    footer: &mut FooterReservation,
    input: &mut Vec<char>,
    pending: &mut Vec<u8>,
    in_input: &mut bool,
) -> io::Result<()> {
    let content: String = input.drain(..).collect();
    pending.clear();
    let content = content.trim().to_string();
    if content.is_empty() {
        *in_input = false;
        return clear_input(footer);
    }
    // 不在 transcript 中插入确认行：模型可能正流式生成一个 Markdown 段落，
    // 额外换行会改变其语义布局。footer 清除即为发送反馈。
    match push_side_note(history_file, &content, "user", None) {
        Ok(_) => {
            *in_input = false;
            clear_input(footer)
        }
        Err(_) => {
            // 保留草稿并继续显示 composer，避免静默丢失用户指令；可再次按
            // Esc/F2 重试，或 Ctrl+G 放弃。
            input.extend(content.chars());
            redraw_input(footer, input)
        }
    }
}

/// 带超时的单字节读取（仅用于 Esc 后判定是否为提交型功能键序列）。
/// 返回 None 表示超时或 stdin 关闭/出错。
fn read_byte_timeout(timeout_ms: i32) -> Option<u8> {
    loop {
        let mut pfd = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pollfd 为栈上独占可变引用。
        let ret = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if ret < 0 {
            // EINTR（SIGINT/SIGWINCH 等）：与主循环一致重试，而不是把"被信号打断"
            // 误判为"裸 Esc 无跟随字节"而提交草稿。
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return None;
        }
        if ret == 0 {
            return None; // 超时：无跟随字节，即裸 Esc
        }
        if pfd.revents & libc::POLLIN == 0 {
            return None;
        }
        let mut byte = [0u8; 1];
        // SAFETY: 单字节栈缓冲，poll 已确认可读，阻塞 read 立即返回。
        let n = unsafe { libc::read(libc::STDIN_FILENO, byte.as_mut_ptr().cast(), 1) };
        if n == 1 {
            return Some(byte[0]);
        }
        if n < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
            continue; // read 被信号打断，重试
        }
        return None; // EOF 或错误
    }
}

/// 判定 ESC（0x1b）后的按键是否为提交键：裸 Esc、F2（`ESC O Q` / `ESC [ 1 2 ~`）、
/// Alt+Enter（`ESC \r` / `ESC \n`）。终端在单个写突发内发出完整功能键序列，跟随
/// 字节几乎立即可达；无论是否提交，本函数都会消费掉 ESC 之后的全部跟随字节，
/// 避免方向键等其他序列的字节混入草稿。
fn is_submit_escape() -> bool {
    const ESCAPE_FOLLOWUP_MS: i32 = 30;
    match read_byte_timeout(ESCAPE_FOLLOWUP_MS) {
        None => true,                    // 裸 Esc
        Some(0x0d) | Some(0x0a) => true, // Alt+Enter
        Some(0x4f) => {
            // SS3 形式：`ESC O Q` → F2；其余（F1/F3/F4 等）吞掉并忽略。
            read_byte_timeout(ESCAPE_FOLLOWUP_MS) == Some(0x51)
        }
        Some(0x5b) => {
            // CSI 形式：读至 `~` 终止符；`ESC [ 1 2 ~` → F2，其余忽略。
            let mut seq = Vec::new();
            loop {
                match read_byte_timeout(ESCAPE_FOLLOWUP_MS) {
                    Some(0x7e) => return seq == b"12",
                    Some(b) => seq.push(b),
                    None => return false,
                }
            }
        }
        Some(_) => false,
    }
}

fn side_note_input_loop(history_file: &PathBuf, stop: &AtomicBool, _term: CbreakTerm) {
    // 输入模式：UTF-8 字节累积缓冲 + 已解析字符 + 已回显列数
    let mut pending: Vec<u8> = Vec::new();
    let mut input: Vec<char> = Vec::new();
    let mut in_input = false;
    let mut footer: Option<FooterReservation> = None;

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let b = {
            let mut pfd = libc::pollfd {
                fd: libc::STDIN_FILENO,
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: pollfd 为栈上独占可变引用。
            let ret = unsafe { libc::poll(&mut pfd, 1, POLL_MS) };
            if ret < 0 {
                // EINTR：终端尺寸变化（SIGWINCH）等信号会中断 poll，属正常现象，
                // 重新计算 footer 位置；其余错误（fd 失效等）才退出监听。
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    if let Some(active_footer) = footer.as_mut() {
                        if refresh_footer(active_footer, in_input.then_some(input.as_slice()))
                            .is_err()
                        {
                            break;
                        }
                    }
                    continue;
                }
                break;
            }
            if ret == 0 {
                // renderer 中仍有少量历史的整屏擦除序列（CSI 0J）。DECSTBM 会阻止
                // 其滚动进 footer，却不限制擦除范围；输入激活时以 poll 周期重绘，使
                // 草稿即使被异步重画短暂清掉也会在 50ms 内恢复，而无需把所有 renderer
                // 的成熟重写状态机改成另一套光标协议。
                if in_input {
                    let Some(active_footer) = footer.as_mut() else {
                        break;
                    };
                    if redraw_input(active_footer, &input).is_err() {
                        break;
                    }
                }
                continue;
            }
            if pfd.revents & libc::POLLIN == 0 {
                if pfd.revents & (libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0 {
                    break; // stdin 关闭/出错，退出监听
                }
                continue;
            }
            let mut byte = [0u8; 1];
            // SAFETY: 单字节栈缓冲，poll 已确认可读，阻塞 read 立即返回。
            let n = unsafe { libc::read(libc::STDIN_FILENO, byte.as_mut_ptr().cast(), 1) };
            if n == 0 {
                break; // EOF
            }
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::WouldBlock
                    || err.kind() == io::ErrorKind::Interrupted
                {
                    continue;
                }
                break;
            }
            byte[0]
        };

        if !in_input {
            // 监听状态：仅 Ctrl+G 进入输入模式；Ctrl+C 由 SIGINT 路径处理（读到即退出）
            if b == CTRL_G {
                input.clear();
                pending.clear();
                match FooterReservation::enter(stop) {
                    Ok(mut active_footer) => {
                        if redraw_input(&mut active_footer, &input).is_err() {
                            let _ = active_footer.leave();
                            break;
                        }
                        footer = Some(active_footer);
                        in_input = true;
                    }
                    Err(_) => {
                        // 当前尺寸无法安全建立 footer 时忽略本次 Ctrl+G；监听继续，
                        // 用户调整终端尺寸后可以重试。
                    }
                }
            } else if b == 0x03 {
                break; // 流式期间 Ctrl+C：主循环中断处理，监听任务退出
            }
            continue;
        }

        // 输入模式：控制字节直接处理，其余按 UTF-8 累积解码
        match b {
            0x0d | 0x0a => {
                // 回车不再发送：与主输入"回车=换行、Esc/F2 提交"的键位语义对齐。
                // 单行 composer 下回车直接忽略，避免误触提交。
            }
            0x7f | 0x08 => {
                // backspace：重绘单行 viewport，长输入不会在 footer 自动换行。
                if input.pop().is_some() {
                    pending.clear(); // 避免半字符残留与后续字节拼出非法序列
                    let Some(active_footer) = footer.as_mut() else {
                        break;
                    };
                    if redraw_input(active_footer, &input).is_err() {
                        break;
                    }
                }
            }
            0x1b => {
                // Esc / F2 / Alt+Enter：提交草稿。is_submit_escape 会消费掉 F2 等
                // 功能键的完整转义序列，避免其跟随字节混入草稿；方向键等其他序列
                // 被吞掉而不提交。
                if is_submit_escape() {
                    let Some(active_footer) = footer.as_mut() else {
                        break;
                    };
                    if submit_draft(
                        history_file,
                        active_footer,
                        &mut input,
                        &mut pending,
                        &mut in_input,
                    )
                    .is_err()
                    {
                        break;
                    }
                    if !in_input {
                        if let Some(mut finished_footer) = footer.take() {
                            let _ = finished_footer.leave();
                        }
                    }
                }
            }
            0x07 => {
                // 再次 Ctrl+G：放弃当前草稿并退出输入模式（取消）。
                input.clear();
                pending.clear();
                in_input = false;
                if let Some(mut finished_footer) = footer.take() {
                    if clear_input(&mut finished_footer).is_err() {
                        let _ = finished_footer.leave();
                        break;
                    }
                    let _ = finished_footer.leave();
                }
            }
            0x03 => {
                // 输入中 Ctrl+C：SIGINT 已触发主中断，放弃本条输入
                input.clear();
                pending.clear();
                if let Some(active_footer) = footer.as_mut() {
                    let _ = clear_input(active_footer);
                }
                break;
            }
            _ => {
                pending.push(b);
                match std::str::from_utf8(&pending) {
                    Ok(s) => {
                        for ch in s.chars() {
                            if ch.is_control() {
                                continue;
                            }
                            input.push(ch);
                        }
                        let Some(active_footer) = footer.as_mut() else {
                            break;
                        };
                        if redraw_input(active_footer, &input).is_err() {
                            break;
                        }
                        pending.clear();
                    }
                    Err(e) if e.error_len().is_none() => {
                        // 不完整的多字节序列，继续累积
                    }
                    Err(_) => {
                        pending.clear(); // 非法字节，丢弃
                    }
                }
            }
        }
    }
    // 循环退出（stop / EOF / 错误 / Ctrl+C）：由监听线程独占完成终端清理，避免 guard
    // 在阻塞 poll 尚未返回时抢先重置滚动区域。
    if let Some(mut active_footer) = footer {
        if in_input {
            let _ = clear_input(&mut active_footer);
        }
        let _ = active_footer.leave();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_width_ascii_and_wide() {
        assert_eq!(char_width('a'), 1);
        assert_eq!(char_width(' '), 1);
        assert_eq!(char_width('中'), 2);
        assert_eq!(char_width('。'), 2);
        assert_eq!(char_width('🚀'), 2);
    }

    #[test]
    fn enabled_check_does_not_panic() {
        // 非交互 stdin（管道/CI）下绝不应启用，避免监听任务污染重定向输入。
        // 无论本机 stdin 是否为 tty，该查询都不应 panic。
        let _ = side_note_input_enabled();
    }

    #[test]
    fn input_viewport_keeps_the_tail_on_one_line() {
        let input: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        // cols 取能容纳完整 COMPOSER_PREFIX 的宽度，验证完整提示前缀下的单行约束。
        let visible = input_viewport(&input, 60);
        assert!(visible.starts_with('…'));
        assert!(visible.ends_with("vwxyz"));
        let total_width = display_width(COMPOSER_PREFIX.chars())
            + display_width(visible.chars())
            + char_width(COMPOSER_CURSOR)
            + 1;
        assert!(total_width <= 60);
    }

    #[test]
    fn input_viewport_keeps_wide_characters_intact() {
        let input: Vec<char> = "前缀-修复这些问题-🚀".chars().collect();
        // 同上：用能容纳完整前缀的宽度验证宽字符（CJK/emoji）不被截断。
        let visible = input_viewport(&input, 64);
        assert!(visible.contains('🚀'));
        let total_width = display_width(COMPOSER_PREFIX.chars())
            + display_width(visible.chars())
            + char_width(COMPOSER_CURSOR)
            + 1;
        assert!(total_width <= 64);
    }

    #[test]
    fn composer_line_never_wraps_in_narrow_terminals() {
        let input: Vec<char> = "输入很长的 side note 🚀".chars().collect();
        for cols in 1..=12 {
            let (prefix, visible, caret) = composer_line_parts(&input, cols);
            assert!(
                display_width(prefix.chars())
                    + display_width(visible.chars())
                    + display_width(caret.chars())
                    < cols
            );
        }
    }
}
