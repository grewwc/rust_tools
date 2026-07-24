// =============================================================================
// 状态行工具 —— 在当前终端行短暂显示一条提示后自动擦除
// =============================================================================
// 特点：
//   - 只走 stderr，不污染 stdout（assistant 输出）
//   - 用 ANSI 转义擦除当前行，不增加 scrollback
//   - 完全自包含，不依赖 driver / prompt / TUI 渲染模块
// =============================================================================

use std::io::{self, Write};
use std::time::Duration;

/// 在终端当前行显示一条暗色提示，等待 `duration` 后自动擦除。
///
/// 实现原理：
/// 1. `\x1b[2K` — 清除当前整行（光标位置不变）
/// 2. `\x1b[2m` — 设置 dimmed 样式（ANSI 标准，终端一般渲染为灰色）
/// 3. 显示文本 + `\x1b[0m` 重置样式
/// 4. 等待指定时长
/// 5. 再次 `\x1b[2K` 清除该行
///
/// 用户看到的效果：灰色提示短暂出现后消失，终端历史里不留任何痕迹。
pub(crate) fn show_status(msg: &str) {
    show_status_with_duration(msg, Duration::from_secs(2));
}

pub(crate) fn show_status_with_duration(msg: &str, duration: Duration) {
    let mut stderr = io::stderr();

    // 清除当前行 → 写入 dimmed 消息 → 刷新
    let _ = write!(stderr, "\x1b[2K\x1b[2m{}\x1b[0m", msg);
    let _ = stderr.flush();

    std::thread::sleep(duration);

    // 再次清除当前行 → 刷新（用户看到提示消失）
    let _ = write!(stderr, "\x1b[2K");
    let _ = stderr.flush();
}

/// 仅显示状态行但不等待擦除（用于调用方自己控制生命周期的场景）。
pub(crate) fn print_status(msg: &str) {
    let mut stderr = io::stderr();
    let _ = write!(stderr, "\x1b[2K\x1b[2m{}\x1b[0m", msg);
    let _ = stderr.flush();
}

/// 擦除状态行（与 `print_status` 配对使用）。
pub(crate) fn clear_status() {
    let mut stderr = io::stderr();
    let _ = write!(stderr, "\x1b[2K");
    let _ = stderr.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_line_module_compiles() {
        // 确认函数签名正确；不在测试中真正 sleep
        let _ = show_status as fn(&str);
        let _ = show_status_with_duration as fn(&str, Duration);
        let _ = print_status as fn(&str);
        let _ = clear_status as fn();
    }
}
