use std::io::{BufRead, Read, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::json;

fn get_cli_path() -> String {
    std::env::var("ARGOS_CLI_PATH").unwrap_or_else(|_| "argos".to_string())
}

fn get_timeout() -> u64 {
    std::env::var("ARGOS_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120)
}

fn call_argos_tool(
    tool_name: &str,
    input: &serde_json::Value,
    env: &str,
) -> Result<serde_json::Value, String> {
    let cli = get_cli_path();
    let input_str = serde_json::to_string(input).unwrap_or_default();

    let args = vec![
        "tool",
        "log",
        tool_name,
        &input_str,
        "-e",
        env,
        "--json",
    ];

    let mut child = Command::new(&cli)
        .args(&args)
        .env("PATH", format!("{}:{}/.local/bin", std::env::var("PATH").unwrap_or_default(), std::env::var("HOME").unwrap_or_default()))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("启动argos失败: {}", e))?;

    let start = std::time::Instant::now();
    let max = Duration::from_secs(get_timeout());

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child
                    .stdout
                    .take()
                    .map(|mut s| {
                        let mut b = String::new();
                        s.read_to_string(&mut b).ok();
                        b
                    })
                    .unwrap_or_default();
                let err = child
                    .stderr
                    .take()
                    .map(|mut s| {
                        let mut b = String::new();
                        s.read_to_string(&mut b).ok();
                        b
                    })
                    .unwrap_or_default();

                if !status.success() {
                    let err_msg = err.lines()
                        .filter(|l| !l.contains("Background refresh failed") && !l.contains("EPERM") && !l.trim().is_empty())
                        .collect::<Vec<_>>()
                        .join("\n");
                    return Err(if err_msg.is_empty() {
                        format!("argos退出码{:?}", status.code())
                    } else {
                        err_msg
                    });
                }

                for line in out.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) {
                        return Ok(v);
                    }
                }

                let clean_out: Vec<&str> = out.lines()
                    .filter(|l| !l.contains("Background refresh failed") && !l.contains("EPERM") && !l.trim().is_empty())
                    .collect();
                if !clean_out.is_empty() {
                    return Ok(json!({"raw": clean_out.join("\n")}));
                }
                return Ok(json!({"raw": out}));
            }
            Ok(None) => {
                if start.elapsed() > max {
                    let _ = child.kill();
                    return Err(format!("查询超时{}秒", get_timeout()));
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(format!("等待argos进程失败: {}", e)),
        }
    }
}

fn tools_list() -> serde_json::Value {
    json!({
        "tools": [
            {
                "name": "logid_search",
                "description": "Search Argos logs by logid or requestId. Use this tool when: (1) user mentions a logid (format: 2026042311592019214102C61110FEDFF8 or 021742526761243fdbddc0100180041234054b2cb00000360e83e), (2) user wants to check/troubleshoot server logs, (3) user asks about request errors, traces, or service issues, (4) user mentions requestId. This tool queries the Argos log platform via the argos CLI and returns log entries with links.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "log_id": {
                            "type": "string",
                            "description": "The logid or requestId to search (e.g. 2026042311592019214102C61110FEDFF8)"
                        },
                        "region": {
                            "type": "string",
                            "description": "Region: China-North, I18n-TT, etc.",
                            "default": "China-North"
                        },
                        "psm_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "PSM list to filter logs, e.g. [\"my.service.psm\"]"
                        },
                        "snap_id": {
                            "type": "string",
                            "description": "Snapshot ID from previous query result"
                        },
                        "deduplication": {
                            "type": "boolean",
                            "description": "Whether to deduplicate log entries",
                            "default": false
                        }
                    },
                    "required": ["log_id"]
                }
            },
            {
                "name": "logid_prune",
                "description": "Query Argos logs by logid (compatible with Argos Skill tool format). Same as logid_search but uses the logid_prune tool name for compatibility with existing Argos Skill workflows. Use when the agent or skill explicitly calls logid_prune.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "log_id": {
                            "type": "string",
                            "description": "The logid or requestId to search"
                        },
                        "region": {
                            "type": "string",
                            "description": "Region: China-North, I18n-TT, etc.",
                            "default": "China-North"
                        },
                        "psm_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "PSM list to filter logs"
                        },
                        "snap_id": {
                            "type": "string",
                            "description": "Snapshot ID"
                        },
                        "deduplication": {
                            "type": "boolean",
                            "description": "Whether to deduplicate",
                            "default": false
                        }
                    },
                    "required": ["log_id"]
                }
            }
        ]
    })
}

fn handle_tool_call(name: &str, args: &serde_json::Value) -> Result<serde_json::Value, String> {
    match name {
        "logid_search" | "logid_prune" => {
            let log_id = args
                .get("log_id")
                .and_then(|v| v.as_str())
                .ok_or("缺少 log_id 参数")?;

            let region_arg = args
                .get("region")
                .and_then(|v| v.as_str())
                .unwrap_or("China-North");

            let env = match region_arg.to_lowercase().as_str() {
                r if r.starts_with("i18n") || r.starts_with("sg") || r.starts_with("va") => "i18n",
                r if r.starts_with("boe") => "boe",
                _ => "cn",
            };

            let mut input = json!({
                "log_id": log_id,
                "region": region_arg,
            });

            if let Some(psm_list) = args.get("psm_list").and_then(|v| v.as_array()) {
                if !psm_list.is_empty() {
                    input["psm_list"] = json!(psm_list);
                }
            }

            if let Some(snap_id) = args.get("snap_id").and_then(|v| v.as_str()) {
                if !snap_id.is_empty() {
                    input["snap_id"] = json!(snap_id);
                }
            }

            if let Some(dedup) = args.get("deduplication").and_then(|v| v.as_bool()) {
                input["deduplication"] = json!(dedup);
            }

            let result = call_argos_tool("logid_prune", &input, env)?;

            let text = serde_json::to_string_pretty(&result)
                .unwrap_or_else(|_| format!("{:?}", result));

            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        _ => Err(format!("未知工具: {}", name)),
    }
}

fn main() {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        let req: serde_json::Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    "{}",
                    json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": {"code": -32700, "message": format!("解析失败: {}", e)}
                    })
                );
                stdout.flush().ok();
                continue;
            }
        };

        let id = req.get("id").cloned();
        let method = req
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let params = req.get("params").cloned();

        if id.is_none() {
            continue;
        }

        let result: Result<serde_json::Value, String> = match method.as_str() {
            "initialize" => Ok(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mcp_argos", "version": "1.0.0"}
            })),
            "notifications/initialized" => Ok(json!({})),
            "tools/list" => Ok(tools_list()),
            "tools/call" => {
                let p = params.unwrap_or(json!({}));
                let name = p
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let args = p
                    .get("arguments")
                    .cloned()
                    .unwrap_or(json!({}));
                handle_tool_call(name, &args)
            }
            "resources/list" => Ok(json!({"resources": []})),
            "prompts/list" => Ok(json!({"prompts": []})),
            _ => Err(format!("未知方法: {}", method)),
        };

        match result {
            Ok(r) => {
                let _ = writeln!(
                    stdout,
                    "{}",
                    json!({"jsonrpc": "2.0", "id": id, "result": r})
                );
            }
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    "{}",
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32603, "message": e}})
                );
            }
        }
        stdout.flush().ok();
    }
}
