use std::error::Error;
use std::fs;
use std::io;

use crate::ai::mcp::{McpClient, connection::McpServerConnection, initialize_and_discover};
use crate::ai::types::{App, McpServerConfig};
use rust_tools::cw::SkipMap;

#[derive(Debug, Clone)]
pub struct McpConfigProbe {
    pub config_path: String,
    pub exists: bool,
    pub server_count: usize,
}

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

pub struct PreparedMcpInit {
    pub report: McpInitReport,
    pub loaded_servers: SkipMap<String, McpServerConfig>,
    pub client: McpClient,
}

fn empty_prepared_mcp_init(report: McpInitReport) -> PreparedMcpInit {
    PreparedMcpInit {
        report,
        loaded_servers: SkipMap::default(),
        client: McpClient::new(),
    }
}

pub fn probe_mcp_config(app: &App) -> McpConfigProbe {
    let cfg = crate::commonw::configw::get_all_config();
    let mcp_path = if !app.cli.mcp_config.trim().is_empty() {
        app.cli.mcp_config.trim().to_string()
    } else {
        cfg.get_opt("ai.mcp.config")
            .unwrap_or_else(|| "~/.config/mcp.json".to_string())
    };
    let mcp_path = crate::commonw::utils::expanduser(&mcp_path);
    let mcp_path = mcp_path.as_ref().to_string();

    let exists = fs::metadata(&mcp_path).is_ok();
    let server_count = if exists {
        super::super::mcp::load_mcp_config_from_file(&mcp_path)
            .map(|servers| servers.len())
            .unwrap_or(0)
    } else {
        0
    };

    McpConfigProbe {
        config_path: mcp_path,
        exists,
        server_count,
    }
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "M",
    "driver::run:init_mcp",
    "[DEBUG] initializing mcp",
    "[DEBUG] initialized mcp",
    {},
    {
        "loaded": __agent_hang_result.loaded,
        "server_count": __agent_hang_result.server_count,
        "tool_count": __agent_hang_result.tool_count,
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
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

    let prepared = prepare_mcp_initialization_from_path(mcp_path.to_string()).await;
    apply_prepared_mcp_init(app, mcp_client, prepared)
}

pub async fn prepare_mcp_initialization_from_path(config_path: String) -> PreparedMcpInit {
    let fallback_config_path = config_path.clone();
    prepare_mcp_initialization_from_path_inner(config_path, false, None)
        .await
        .unwrap_or_else(|| {
            empty_prepared_mcp_init(McpInitReport {
                config_path: fallback_config_path,
                loaded: false,
                server_count: 0,
                tool_count: 0,
                resource_count: 0,
                prompt_count: 0,
                failures: Vec::new(),
            })
        })
}

pub async fn prepare_mcp_initialization_from_path_interruptible(
    config_path: String,
    local_interrupt_futex: Option<aios_kernel::primitives::FutexAddr>,
) -> Option<PreparedMcpInit> {
    prepare_mcp_initialization_from_path_inner(config_path, true, local_interrupt_futex).await
}

async fn prepare_mcp_initialization_from_path_inner(
    config_path: String,
    interruptible: bool,
    local_interrupt_futex: Option<aios_kernel::primitives::FutexAddr>,
) -> Option<PreparedMcpInit> {
    let mut report = McpInitReport {
        config_path: config_path.clone(),
        loaded: false,
        server_count: 0,
        tool_count: 0,
        resource_count: 0,
        prompt_count: 0,
        failures: Vec::new(),
    };

    if let Err(err) = fs::metadata(&config_path) {
        if err.kind() != io::ErrorKind::NotFound {
            eprintln!(
                "[mcp] failed to access config file {}: {}",
                config_path, err
            );
        }
        return Some(empty_prepared_mcp_init(report));
    }

    let servers = match super::super::mcp::load_mcp_config_from_file(&config_path) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("[mcp] failed to load config from {}: {}", config_path, err);
            return Some(empty_prepared_mcp_init(report));
        }
    };

    let loaded_servers = servers;

    let server_entries: Vec<_> = loaded_servers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut client = McpClient::new();
    for (name, cfg) in &server_entries {
        if interruptible
            && crate::ai::driver::signal::interrupt_sources_ready(local_interrupt_futex)
        {
            return None;
        }

        let result = connect_single_server_async(name.clone(), cfg.clone()).await;

        if interruptible
            && crate::ai::driver::signal::interrupt_sources_ready(local_interrupt_futex)
        {
            return None;
        }

        match result {
            Ok(conn) => {
                client.servers.insert(
                    name.clone(),
                    std::sync::Arc::new(std::sync::Mutex::new(conn)),
                );
            }
            Err(err) => {
                eprintln!("[mcp] failed to connect to server {}: {}", name, err);
                report.failures.push(format!("{}: {}", name, err));
            }
        }
    }

    client.rebuild_metadata_cache();
    report.loaded = true;
    report.server_count = loaded_servers.len();
    report.tool_count = client.get_all_tools().len();
    report.resource_count = client.get_all_resources().len();
    report.prompt_count = client.get_all_prompts().len();

    Some(PreparedMcpInit {
        report,
        loaded_servers,
        client,
    })
}

pub fn apply_prepared_mcp_init(
    app: &mut App,
    mcp_client: &mut McpClient,
    prepared: PreparedMcpInit,
) -> McpInitReport {
    let PreparedMcpInit {
        report,
        loaded_servers,
        client,
    } = prepared;

    *mcp_client = client;

    if let Some(ctx) = app.agent_context.as_mut() {
        ctx.mcp_servers = loaded_servers;
        ctx.tools.extend(mcp_client.get_all_tools());
    }
    crate::ai::tools::enable_tools::set_available_mcp_tools(mcp_client.get_all_tools());

    report
}

/// Asynchronously connect a single MCP server (wraps blocking operations with
/// spawn_blocking).
async fn connect_single_server_async(
    name: String,
    config: McpServerConfig,
) -> Result<McpServerConnection, String> {
    tokio::task::spawn_blocking(move || connect_single_server(&name, &config))
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};

    #[tokio::test]
    async fn interruptible_prepare_mcp_init_stops_before_connecting_servers() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let kernel = crate::ai::driver::new_local_kernel();
        init_os_tools_globals(kernel.clone());
        crate::ai::driver::signal::clear_request_interrupt();

        let config_path =
            std::env::temp_dir().join(format!("mcp-interruptible-{}.json", uuid::Uuid::new_v4()));
        std::fs::write(
            &config_path,
            r#"{
                "mcpServers": {
                    "demo": {
                        "command": "/definitely/not/run",
                        "args": ["--demo"]
                    }
                }
            }"#,
        )
        .unwrap();

        crate::ai::driver::signal::signal_request_interrupt();
        let prepared = prepare_mcp_initialization_from_path_interruptible(
            config_path.to_string_lossy().to_string(),
            None,
        )
        .await;

        assert!(prepared.is_none());

        crate::ai::driver::signal::clear_request_interrupt();
        let _ = std::fs::remove_file(config_path);
        if let Ok(mut guard) = GLOBAL_OS.lock() {
            *guard = None;
        }
    }

    #[cfg(unix)]
    #[test]
    fn mcp_handshake_startup_uses_a_fresh_counter_per_server() {
        use crate::ai::mcp::test_support::*;

        let _guard = serial_guard();
        let first = ScriptedServer::new(complete_responses(1, "first"));
        let second = ScriptedServer::new(complete_responses(1, "second"));

        let first_conn = connect_single_server("first", &first.config).unwrap();
        let second_conn = connect_single_server("second", &second.config).unwrap();

        assert_eq!(first.transcript(), complete_transcript(1));
        assert_eq!(second.transcript(), complete_transcript(1));
        assert_eq!(observed_catalogs(&first_conn), catalogs("first"));
        assert_eq!(observed_catalogs(&second_conn), catalogs("second"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_handshake_startup_keeps_successful_servers_after_discovery_failure() {
        use crate::ai::mcp::test_support::*;
        use serde_json::json;

        let _guard = serial_guard();
        let mut failed_responses = complete_responses(1, "failed");
        failed_responses[3] = failure(3, -32000, "resource catalog failed");
        let failed = ScriptedServer::new(failed_responses);
        let working = ScriptedServer::new(complete_responses(1, "working"));
        let config_path = std::env::temp_dir().join(format!(
            "mcp-handshake-config-{}.json",
            uuid::Uuid::new_v4()
        ));
        fs::write(
            &config_path,
            serde_json::to_vec(&json!({
                "mcpServers": {"failed": failed.config, "working": working.config}
            }))
            .unwrap(),
        )
        .unwrap();

        let prepared =
            prepare_mcp_initialization_from_path(config_path.to_string_lossy().to_string()).await;
        fs::remove_file(&config_path).unwrap();

        assert_eq!(failed.transcript(), complete_transcript(1)[..4]);
        assert_eq!(working.transcript(), complete_transcript(1));
        assert!(prepared.report.loaded);
        assert_eq!(prepared.report.server_count, 2);
        assert_eq!(prepared.report.tool_count, 1);
        assert_eq!(prepared.report.resource_count, 1);
        assert_eq!(prepared.report.prompt_count, 1);
        assert_eq!(
            prepared.report.failures,
            ["failed: MCP error -32000: resource catalog failed"]
        );
        assert_eq!(prepared.client.servers.len(), 1);
        assert!(prepared.client.servers.get_str_ref("failed").is_none());
        let conn = prepared
            .client
            .servers
            .get_str_ref("working")
            .unwrap()
            .lock()
            .unwrap();
        assert_eq!(observed_catalogs(&conn), catalogs("working"));
    }
}

/// Synchronously connect one startup server with its own request-ID counter.
fn connect_single_server(
    name: &str,
    config: &McpServerConfig,
) -> Result<McpServerConnection, String> {
    if config.disabled {
        return Err("Server is disabled".to_string());
    }
    let mut conn = McpServerConnection::connect(name, config)?;

    // Use a dedicated request-ID counter.
    let next_id = std::sync::atomic::AtomicU64::new(1);

    initialize_and_discover(&mut conn, &next_id)?;

    Ok(conn)
}

pub async fn drain_response(response: &mut reqwest::Response) -> Result<(), Box<dyn Error>> {
    while response.chunk().await?.is_some() {}
    Ok(())
}
