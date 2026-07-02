//! `/usage` 交互命令：查看 LLM token 用量统计。
//!
//! 数据来源是内核 LLM 设备的审计账本，由 agent 落库到独立的 `token_usage` 表
//! （见 [`crate::ai::tools::storage::token_usage_store`]）。本命令只做只读查询展示。
//!
//! 输出优化：大数字带千位分隔符，超过 1M 的自动缩写为 K/M/B 单位。

use crate::ai::tools::storage::token_usage_store as store;

fn print_usage_help() {
    println!("Usage commands:");
    println!();
    println!("  /usage                 show token usage (all-time + 7d + 24h + daily trend)");
    println!("  /usage today           show usage for the last 24 hours (by model)");
    println!("  /usage 7d              show usage for the last 7 days");
    println!("  /usage 30d             show usage for the last 30 days");
    println!("  /usage all             show all-time usage");
    println!("  /usage daily           show daily breakdown for the last 14 days");
    println!("  /usage help            show this help");
    println!();
}

/// 格式化数字：≥1B 用 B，≥1M 用 M，≥1K 用 K，否则带千位分隔符。
fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        let b = n as f64 / 1_000_000_000.0;
        if b >= 100.0 {
            format!("{:.0}B", b)
        } else if b >= 10.0 {
            format!("{:.1}B", b)
        } else {
            format!("{:.2}B", b)
        }
    } else if n >= 1_000_000 {
        let m = n as f64 / 1_000_000.0;
        if m >= 100.0 {
            format!("{:.0}M", m)
        } else if m >= 10.0 {
            format!("{:.1}M", m)
        } else {
            format!("{:.2}M", m)
        }
    } else if n >= 1_000 {
        let k = n as f64 / 1_000.0;
        if k >= 100.0 {
            format!("{:.0}K", k)
        } else if k >= 10.0 {
            format!("{:.1}K", k)
        } else {
            format!("{:.2}K", k)
        }
    } else {
        // < 1K：带千位分隔符
        let s = n.to_string();
        let mut out = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                out.push(',');
            }
            out.push(c);
        }
        out.chars().rev().collect()
    }
}

/// 把秒数解析成时间窗口；返回 `Some(None)`=全部历史，`Some(Some(secs))`=最近 secs 秒，`None`=无法解析。
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

fn parse_daily_arg(arg: &str) -> Option<u64> {
    let a = arg.trim().to_ascii_lowercase();
    if a.is_empty() || a == "daily" || a == "days" || a == "trend" {
        Some(14)
    } else if let Some(n) = a.strip_suffix('d').and_then(|s| s.parse::<u64>().ok()) {
        Some(n)
    } else {
        a.parse::<u64>().ok().or(Some(14))
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
            // 表头行的标签列宽度与下方模型行的列宽度对齐：
            // 表头 = 2 空格 + 22 宽标签 = 24；模型行 = 6 空格 + 18 宽模型名 = 24。
            // 这样 calls/in/out/total 各列在表头与模型行之间垂直对齐。列宽收窄到
            // 实际内容尺寸（模型名 ≤17 字符、缩写数字 ≤5 字符），整行约 77 列，
            // 避免在 80 列终端里把 total= 的值折到下一行。
            println!(
                "  {:<22} calls={:>6}  in={:>7}  out={:>7}  total={:>7}",
                format!("[{}]", label),
                format_number(t.calls),
                format_number(t.input),
                format_number(t.output),
                format_number(t.total)
            );
            if let Some(rows) = store::query_by_model(window) {
                for r in rows.iter().filter(|r| r.calls > 0) {
                    println!(
                        "      {:<18} calls={:>6}  in={:>7}  out={:>7}  total={:>7}",
                        r.model,
                        format_number(r.calls),
                        format_number(r.input),
                        format_number(r.output),
                        format_number(r.total)
                    );
                }
            }
        }
        None => {
            println!("  [{}] (no usage store available)", label);
        }
    }
}

fn print_daily_breakdown(days: u64) {
    match store::query_daily_breakdown(days) {
        Some(rows) if rows.is_empty() => {
            println!("  [daily last {}d] 无数据", days);
        }
        Some(rows) => {
            println!("  [daily last {}d]", days);
            println!(
                "      {:<12} {:>6}  {:>7}  {:>7}  {:>7}",
                "date", "calls", "in", "out", "total"
            );
            for r in &rows {
                println!(
                    "      {:<12} {:>6}  {:>7}  {:>7}  {:>7}",
                    r.day,
                    format_number(r.calls),
                    format_number(r.input),
                    format_number(r.output),
                    format_number(r.total)
                );
            }
        }
        None => {
            println!("  [daily] (no usage store available)");
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

    println!("Token usage  store: {}", store::store_path().display());
    if !store::is_enabled() {
        println!("  统计已关闭（ai.token_usage.enable=false）。");
        return Ok(true);
    }

    if arg.is_empty() {
        // 默认：全部历史 + 最近 7d + 最近 24h 概览，并按模型拆分 + 近 14 天趋势。
        println!();
        print_window(None);
        print_window(Some(7 * 86_400));
        print_window(Some(86_400));
        println!();
        print_daily_breakdown(3);
    } else if matches!(arg, "daily" | "days" | "trend") || arg.ends_with("d") && arg.len() <= 4 {
        let days = parse_daily_arg(arg).unwrap_or(14);
        print_daily_breakdown(days);
    } else if let Some(window) = parse_window(arg) {
        print_window(window);
    } else {
        println!("  无法识别的时间窗口: '{}'", arg);
        print_usage_help();
    }
    Ok(true)
}
