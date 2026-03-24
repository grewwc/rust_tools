use reqwest::blocking::Response;
use std::error::Error;
use std::fs;
use std::io;

use crate::ai::{mcp::McpClient, types::App};

#[derive(Debug, Clone)]
pub struct McpInitReport {
    pub config_path: String,
    pub loaded: bool,
    pub server_count: usize,
    pub tool_count: usize,
    pub failures: Vec<String>,
}

pub fn init_mcp(app: &mut App, mcp_client: &mut McpClient) -> McpInitReport {
    let cfg = crate::common::configw::get_all_config();
    let mcp_path = if !app.cli.mcp_config.trim().is_empty() {
        app.cli.mcp_config.trim().to_string()
    } else {
        cfg.get_opt("ai.mcp.config")
            .unwrap_or_else(|| "~/.config/mcp.json".to_string())
    };
    let mcp_path = crate::common::utils::expanduser(&mcp_path);
    let mcp_path = mcp_path.as_ref();

    let mut report = McpInitReport {
        config_path: mcp_path.to_string(),
        loaded: false,
        server_count: 0,
        tool_count: 0,
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

    for (name, server_cfg) in &servers {
        if let Err(err) = mcp_client.connect_server(name, server_cfg) {
            eprintln!("[mcp] failed to connect to server {}: {}", name, err);
            report.failures.push(format!("{}: {}", name, err));
        }
    }

    if let Some(ctx) = app.agent_context.as_mut() {
        ctx.mcp_servers = servers;
        ctx.tools.extend(mcp_client.get_all_tools());
    }
    report.loaded = true;
    report.server_count = app
        .agent_context
        .as_ref()
        .map(|c| c.mcp_servers.len())
        .unwrap_or(0);
    report.tool_count = mcp_client.get_all_tools().len();
    report
}

pub fn drain_response(response: &mut Response) -> Result<(), Box<dyn Error>> {
    response.copy_to(&mut io::sink())?;
    Ok(())
}
