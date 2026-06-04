//! `/usage` 交互命令：查看 LLM token 用量统计。
//!
//! 数据来源是内核 LLM 设备的审计账本，由 agent 落库到独立的 `token_usage` 表
//! （见 [`crate::ai::tools::storage::token_usage_store`]）。本命令只做只读查询展示。

use crate::ai::tools::storage::token_usage_store as store;

fn print_usage_help() {
    println!("Usage commands:");
    println!();
    println!("  /usage                 show token usage (all-time + last 24h, by model)");
    println!("  /usage today           show usage for the last 24 hours");
    println!("  /usage 7d              show usage for the last 7 days");
    println!("  /usage all             show all-time usage");
    println!("  /usage help            show this help");
    println!();
}

/// 把秒数解析成时间窗口；返回 `Some(None)` 表示全部历史，`Some(Some(secs))`
/// 表示最近 secs 秒，`None` 表示无法解析。
fn parse_window(arg: &str) -> Option<Option<u64>> {
    let a = arg.trim().to_ascii_lowercase();
    match a.as_str() {
        "" => None,
        "all" | "total" => Some(None),
        "today" | "day" | "1d" | "24h" => Some(Some(86_400)),
        "week" | "7d" => Some(Some(7 * 86_400)),
        "30d" | "month" => Some(Some(30 * 86_400)),
        _ => {
            // 支持 "<N>d" / "<N>h" 形式。
            if let Some(num) = a.strip_suffix('d').and_then(|s| s.parse::<u64>().ok()) {
                Some(Some(num.saturating_mul(86_400)))
            } else if let Some(num) = a.strip_suffix('h').and_then(|s| s.parse::<u64>().ok()) {
                Some(Some(num.saturating_mul(3_600)))
            } else {
                None
            }
        }
    }
}

fn window_label(window: Option<u64>) -> String {
    match window {
        None => "all-time".to_string(),
        Some(secs) if secs % 86_400 == 0 => format!("last {}d", secs / 86_400),
        Some(secs) if secs % 3_600 == 0 => format!("last {}h", secs / 3_600),
        Some(secs) => format!("last {}s", secs),
    }
}

fn print_window(window: Option<u64>) {
    let label = window_label(window);
    match store::query_totals(window) {
        Some(t) => {
            println!(
                "  [{}] calls={}  input={}  output={}  total={}",
                label, t.calls, t.input, t.output, t.total
            );
            if let Some(rows) = store::query_by_model(window) {
                for r in rows.iter().filter(|r| r.total > 0) {
                    println!(
                        "      {:<28} calls={:<5} in={:<10} out={:<10} total={}",
                        r.model, r.calls, r.input, r.output, r.total
                    );
                }
            }
        }
        None => {
            println!("  [{}] (no usage store available)", label);
        }
    }
}

pub fn try_handle_usage_command(input: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };

    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if !matches!(cmd, "usage" | "tokens" | "token") {
        return Ok(false);
    }

    let arg = normalized[cmd.len()..].trim();
    if matches!(arg, "help" | "h") {
        print_usage_help();
        return Ok(true);
    }

    println!("Token usage  (store: {})", store::store_path().display());
    if !store::is_enabled() {
        println!("  统计已关闭（ai.token_usage.enable=false）。");
        return Ok(true);
    }

    if arg.is_empty() {
        // 默认：全部历史 + 最近 24h 概览，并按模型拆分。
        print_window(None);
        print_window(Some(86_400));
    } else if let Some(window) = parse_window(arg) {
        print_window(window);
    } else {
        println!("  无法识别的时间窗口: '{}'", arg);
        print_usage_help();
    }
    Ok(true)
}
