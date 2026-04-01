use rust_tools::commonw::FastMap;
 
use serde_json::Value;

use crate::ai::types::McpServerConfig;

pub(in crate::ai) fn load_mcp_config_from_file(
    path: &str,
) -> Result<FastMap<String, McpServerConfig>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read MCP config: {}", e))?;

    let config: Value =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse MCP config: {}", e))?;

    let servers = config["mcpServers"]
        .as_object()
        .ok_or("Invalid mcpServers in config")?;

    let mut result = FastMap::default();
    for (name, value) in servers {
        let server_config: McpServerConfig = serde_json::from_value(value.clone())
            .map_err(|e| format!("Invalid server config for '{}': {}", name, e))?;
        result.insert(name.clone(), server_config);
    }

    Ok(result)
}
