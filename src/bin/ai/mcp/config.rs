use rust_tools::cw::SkipMap;

use serde_json::Value;

use crate::ai::types::McpServerConfig;

pub(in crate::ai) fn load_mcp_config_from_file(
    path: &str,
) -> Result<SkipMap<String, McpServerConfig>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read MCP config: {}", e))?;

    let config: Value =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse MCP config: {}", e))?;

    let servers = config["mcpServers"]
        .as_object()
        .ok_or("Invalid mcpServers in config")?;

    let mut result = SkipMap::default();
    for (name, value) in servers {
        // A single malformed entry must not disable every configured server:
        // skip it with a warning and keep loading the rest. The caller (driver
        // mcp_init) only logs the overall load error, so an early `?` here
        // would silently kill all MCP servers (browser, excel, ...) on one typo.
        match serde_json::from_value::<McpServerConfig>(value.clone()) {
            Ok(server_config) => {
                result.insert(name.clone(), server_config);
            }
            Err(e) => {
                eprintln!("[mcp] skipping invalid server config '{}': {}", name, e);
            }
        }
    }

    Ok(result)
}
