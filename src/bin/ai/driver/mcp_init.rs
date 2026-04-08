use std::error::Error;
use std::fs;
use std::io;
use std::io::BufReader;
use std::os::unix::process::CommandExt;
use std::path::Path;

use crate::ai::mcp::{McpClient, connection::McpServerConnection};
use crate::ai::types::{App, McpServerConfig, McpTool, McpResource, McpPrompt};
use std::process::{Command, Stdio};

const FEISHU_MCP_MIN_REQUEST_TIMEOUT_MS: u64 = 20_000;

#[derive(Debug, Clone)]
pub struct McpInitReport {
    pub config_path: String,
    pub loaded: bool,
    pub server_count: usize,
    pub tool_count: usize,
    pub resource_count: usize,
    pub prompt_count: usize,
    pub failures: Vec<String>,
}

fn has_feishu_app_credentials_in_configw() -> bool {
    let cfg = crate::commonw::configw::get_all_config();
    let app_id = cfg
        .get_opt("feishu.app_id")
        .unwrap_or_default()
        .trim()
        .to_string();
    let app_secret = cfg
        .get_opt("feishu.app_secret")
        .unwrap_or_default()
        .trim()
        .to_string();
    !app_id.is_empty() && !app_secret.is_empty()
}

fn is_feishu_mcp_server(name: &str, command: &str) -> bool {
    if name == "feishu" {
        return true;
    }
    let Some(file_name) = Path::new(command).file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    file_name == "mcp_feishu" || file_name == "mcp_feishu.exe"
}

pub async fn init_mcp(app: &mut App, mcp_client: &mut McpClient) -> McpInitReport {
    let cfg = crate::commonw::configw::get_all_config();
    let mcp_path = if !app.cli.mcp_config.trim().is_empty() {
        app.cli.mcp_config.trim().to_string()
    } else {
        cfg.get_opt("ai.mcp.config")
            .unwrap_or_else(|| "~/.config/mcp.json".to_string())
    };
    let mcp_path = crate::commonw::utils::expanduser(&mcp_path);
    let mcp_path = mcp_path.as_ref();

    let mut report = McpInitReport {
        config_path: mcp_path.to_string(),
        loaded: false,
        server_count: 0,
        tool_count: 0,
        resource_count: 0,
        prompt_count: 0,
        failures: Vec::new(),
    };
    if let Err(err) = fs::metadata(mcp_path) {
        if err.kind() != io::ErrorKind::NotFound {
            eprintln!("[mcp] failed to access config file {}: {}", mcp_path, err);
        }
        return report;
    }

    let servers = match super::super::mcp::load_mcp_config_from_file(mcp_path) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("[mcp] failed to load config from {}: {}", mcp_path, err);
            return report;
        }
    };

    let enable_feishu_mcp = has_feishu_app_credentials_in_configw();
    let mut loaded_servers = servers;
    loaded_servers.retain(|name, server_cfg| {
        if is_feishu_mcp_server(name, &server_cfg.command) && !enable_feishu_mcp {
            eprintln!(
                "[mcp] skip server '{}': missing feishu.app_id / feishu.app_secret in ~/.configW",
                name
            );
            false
        } else {
            true
        }
    });

    for (name, server_cfg) in &mut loaded_servers {
        if is_feishu_mcp_server(name, &server_cfg.command)
            && server_cfg.request_timeout_ms < FEISHU_MCP_MIN_REQUEST_TIMEOUT_MS
        {
            eprintln!(
                "[mcp] raise server '{}' request_timeout_ms from {} to {} for Feishu network calls",
                name, server_cfg.request_timeout_ms, FEISHU_MCP_MIN_REQUEST_TIMEOUT_MS
            );
            server_cfg.request_timeout_ms = FEISHU_MCP_MIN_REQUEST_TIMEOUT_MS;
        }
    }

    // 异步并行连接所有 MCP 服务器
    let server_futures: Vec<_> = loaded_servers
        .iter()
        .map(|(name, cfg)| {
            let name = name.clone();
            let cfg = cfg.clone();
            async move { (name.clone(), connect_single_server_async(name, cfg).await) }
        })
        .collect();

    // 使用 futures_util 而不是 futures
    use futures_util::future::join_all;
    let results = join_all(server_futures).await;

    for (name, result) in results {
        match result {
            Ok(conn) => {
                mcp_client.servers.insert(name.clone(), std::sync::Mutex::new(conn));
            }
            Err(err) => {
                eprintln!("[mcp] failed to connect to server {}: {}", name, err);
                report.failures.push(format!("{}: {}", name, err));
            }
        }
    }

    // 统一重建缓存
    mcp_client.rebuild_metadata_cache();

    if let Some(ctx) = app.agent_context.as_mut() {
        ctx.mcp_servers = loaded_servers;
        ctx.tools.extend(mcp_client.get_all_tools());
    }
    report.loaded = true;
    report.server_count = app
        .agent_context
        .as_ref()
        .map(|c| c.mcp_servers.len())
        .unwrap_or(0);
    report.tool_count = mcp_client.get_all_tools().len();
    report.resource_count = mcp_client.get_all_resources().len();
    report.prompt_count = mcp_client.get_all_prompts().len();
    report
}

/// 异步连接单个 MCP 服务器（使用 spawn_blocking 包装阻塞操作）
async fn connect_single_server_async(
    name: String,
    config: McpServerConfig,
) -> Result<McpServerConnection, String> {
    tokio::task::spawn_blocking(move || connect_single_server(&name, &config))
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

/// 同步连接单个 MCP 服务器（提取自 McpClient::connect_server）
fn connect_single_server(name: &str, config: &McpServerConfig) -> Result<McpServerConnection, String> {
    if config.disabled {
        return Err("Server is disabled".to_string());
    }

    let mut cmd = Command::new(&config.command);
    cmd.args(&config.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    unsafe {
        let _ = cmd.pre_exec(|| {
            libc::signal(libc::SIGINT, libc::SIG_IGN);
            libc::signal(libc::SIGPIPE, libc::SIG_IGN);
            Ok(())
        });
    }

    for (key, value) in &config.env {
        cmd.env(key, value);
    }

    let mut process = cmd
        .spawn()
        .map_err(|e| format!("Failed to start MCP server '{}': {}", name, e))?;

    let stdin = process.stdin.take().ok_or("Failed to get stdin")?;
    let stdout = process.stdout.take().ok_or("Failed to get stdout")?;
    let stderr = process.stderr.take().ok_or("Failed to get stderr")?;

    let mut conn = McpServerConnection {
        config: config.clone(),
        process,
        stdin,
        stdout: BufReader::new(stdout),
        stderr: BufReader::new(stderr),
        request_timeout_ms: config.request_timeout_ms.max(100),
        tools: Vec::new(),
        resources: Vec::new(),
        prompts: Vec::new(),
    };

    // 使用独立的请求 ID 计数器
    let next_id = std::sync::atomic::AtomicU64::new(1);
    
    initialize_server(&mut conn, &next_id)?;
    conn.tools = list_tools(&mut conn, &next_id)?;
    conn.resources = list_resources(&mut conn, &next_id)?;
    conn.prompts = list_prompts(&mut conn, &next_id)?;

    Ok(conn)
}

// 以下函数提取自 McpClient，使用独立的 AtomicU64 计数器
fn initialize_server(conn: &mut McpServerConnection, next_id: &std::sync::atomic::AtomicU64) -> Result<(), String> {
    use serde_json::json;
    use crate::ai::mcp::send_request_to_conn;
    
    let params = json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {},
            "resources": {},
            "prompts": {}
        },
        "clientInfo": {
            "name": "rust-tools-ai",
            "version": "1.0.0"
        }
    });

    let id1 = next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    send_request_to_conn(conn, id1, "initialize", Some(params))?;
    let id2 = next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Err(err) = send_request_to_conn(conn, id2, "notifications/initialized", None) {
        let is_method_not_found = err.contains("-32601")
            && (err.contains("notifications/initialized") || err.contains("initialized"));
        if !is_method_not_found {
            return Err(err);
        }
    }
    Ok(())
}

fn list_tools(conn: &mut McpServerConnection, next_id: &std::sync::atomic::AtomicU64) -> Result<Vec<McpTool>, String> {
    use crate::ai::mcp::send_request_to_conn;
    let id = next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let result = send_request_to_conn(conn, id, "tools/list", None)?;
    let tools = result["tools"]
        .as_array()
        .ok_or("Invalid tools response")?
        .iter()
        .filter_map(|t| serde_json::from_value(t.clone()).ok())
        .collect();
    Ok(tools)
}

fn list_resources(conn: &mut McpServerConnection, next_id: &std::sync::atomic::AtomicU64) -> Result<Vec<McpResource>, String> {
    use crate::ai::mcp::send_request_to_conn;
    let id = next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let result = match send_request_to_conn(conn, id, "resources/list", None) {
        Ok(v) => v,
        Err(err) => {
            let is_method_not_found = err.contains("-32601") && err.contains("resources/list");
            if is_method_not_found {
                return Ok(Vec::new());
            }
            return Err(err);
        }
    };
    let resources = result["resources"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|r| serde_json::from_value(r.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
    Ok(resources)
}

fn list_prompts(conn: &mut McpServerConnection, next_id: &std::sync::atomic::AtomicU64) -> Result<Vec<McpPrompt>, String> {
    use crate::ai::mcp::send_request_to_conn;
    let id = next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let result = match send_request_to_conn(conn, id, "prompts/list", None) {
        Ok(v) => v,
        Err(err) => {
            let is_method_not_found = err.contains("-32601") && err.contains("prompts/list");
            if is_method_not_found {
                return Ok(Vec::new());
            }
            return Err(err);
        }
    };
    let prompts = result["prompts"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|p| serde_json::from_value(p.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
    Ok(prompts)
}

pub async fn drain_response(response: &mut reqwest::Response) -> Result<(), Box<dyn Error>> {
    while response.chunk().await?.is_some() {}
    Ok(())
}
