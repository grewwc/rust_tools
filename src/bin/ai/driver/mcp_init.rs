use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;

use crate::ai::{mcp::McpClient, types::App};

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

pub fn init_mcp(app: &mut App, mcp_client: &mut McpClient) -> McpInitReport {
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

    for (name, server_cfg) in &loaded_servers {
        if let Err(err) = mcp_client.connect_server(name, server_cfg) {
            eprintln!("[mcp] failed to connect to server {}: {}", name, err);
            report.failures.push(format!("{}: {}", name, err));
        }
    }

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

pub async fn drain_response(response: &mut reqwest::Response) -> Result<(), Box<dyn Error>> {
    while response.chunk().await?.is_some() {}
    Ok(())
}
