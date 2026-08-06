//! Request-layer diagnostics that respect terminal ownership.

use std::fmt;
use std::io::IsTerminal;
use std::io::Write;

/// Whether request diagnostics may be written to the live terminal.
pub(in crate::ai) fn request_diagnostics_enabled() -> bool {
    crate::ai::driver::runtime_ctx::terminal_output_enabled()
}

/// Emit a request diagnostic to stderr only when the current task owns the
/// terminal. Background subagents publish progress through task IPC/status lines
/// instead of writing directly to the foreground TTY.
pub(in crate::ai) fn emit_request_diagnostic(args: fmt::Arguments<'_>) -> bool {
    if !request_diagnostics_enabled() {
        return false;
    }
    eprintln!("{args}");
    true
}

/// 单行瞬态状态行：用于 TPM 限流等待、重试等待这类会反复触发的「进度型」信息。
///
/// - TTY + 前台：用 `\r\x1b[2K` 原地刷新，最终 `clear()` 清除不留痕，避免刷屏。
/// - 非 TTY（pipe / 日志）/ 后台 subagent：降级为普通 `eprintln!`，只在文本变化
///   且距离上次超过一定时间时输出一次，控制条数。
///
/// 调用方持一个 `Option<TransientStatusLine>`，第一次更新时创建，结束时调用
/// `clear()` 并丢弃。所有写入都走 stderr，与主输出（stdout）互不干扰。
pub(in crate::ai) struct TransientStatusLine {
    last_text: String,
    last_emitted: std::time::Instant,
    is_tty: bool,
    visible: bool,
}

impl TransientStatusLine {
    /// 只有在允许输出时才创建；否则返回 None，调用方可以全 no-op。
    pub(in crate::ai) fn new() -> Option<Self> {
        if !request_diagnostics_enabled() {
            return None;
        }
        let is_tty = std::io::stderr().is_terminal();
        // 初始 last_emitted 设为 1 小时前，保证第一次更新总能输出。
        let last_emitted = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(3600))
            .unwrap_or(std::time::Instant::now());
        Some(Self {
            last_text: String::new(),
            last_emitted,
            is_tty,
            visible: false,
        })
    }

    /// 更新当前行内容。TTY 下每次都原地重画；非 TTY 下只有文本变化且距离上次
    /// 输出超过 `MIN_NON_TTY_INTERVAL` 才再打一行，防止日志洪水。
    pub(in crate::ai) fn update(&mut self, text: &str) {
        if self.is_tty {
            // 暗色显示，避免喧宾夺主；用 \r 回到行首 + \x1b[2K 清整行。
            // 注意：不换行，行尾保持在当前行末尾。
            let _ = write!(std::io::stderr(), "\r\x1b[2K\x1b[2m{text}\x1b[0m");
            let _ = std::io::stderr().flush();
            self.visible = true;
        } else if text != self.last_text && self.last_emitted.elapsed() >= MIN_NON_TTY_INTERVAL {
            eprintln!("[Info] {text}");
            self.last_emitted = std::time::Instant::now();
        }
        self.last_text = text.to_string();
    }

    /// 清除瞬态行（仅 TTY 下有意义）。保证心跳/等待提示不残留到下一行输出。
    pub(in crate::ai) fn clear(&mut self) {
        if self.is_tty && self.visible {
            let _ = write!(std::io::stderr(), "\r\x1b[2K");
            let _ = std::io::stderr().flush();
            self.visible = false;
        }
    }
}

impl Drop for TransientStatusLine {
    fn drop(&mut self) {
        self.clear();
    }
}

/// 非 TTY 环境下两次相同类别的瞬态消息之间的最小间隔，避免日志被打满。
const MIN_NON_TTY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
