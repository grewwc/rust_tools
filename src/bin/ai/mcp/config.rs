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
            Ok(server_config) => match server_config.transport_kind() {
                Ok(_) => {
                    result.insert(name.clone(), server_config);
                }
                Err(e) => {
                    eprintln!("[mcp] skipping invalid server config '{}': {}", name, e);
                }
            },
            Err(e) => {
                eprintln!("[mcp] skipping invalid server config '{}': {}", name, e);
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::McpTransportKind;
    use serde_json::json;

    fn parse(value: serde_json::Value) -> McpServerConfig {
        serde_json::from_value(value).expect("valid MCP server config")
    }

    #[test]
    fn url_defaults_to_streamable_http_and_keeps_headers() {
        let config = parse(json!({
            "url": "http://127.0.0.1:9000/mcp",
            "headers": { "Authorization": "Bearer test" },
        }));
        assert_eq!(config.transport_kind().unwrap(), McpTransportKind::Http);
        assert_eq!(
            config.headers.get(&"Authorization".to_string()).as_deref(),
            Some("Bearer test")
        );
    }

    #[test]
    fn explicit_sse_and_stdio_transports_are_distinguished() {
        let sse = parse(json!({
            "type": "sse",
            "url": "http://127.0.0.1:9000/sse",
        }));
        assert_eq!(sse.transport_kind().unwrap(), McpTransportKind::Sse);

        let stdio = parse(json!({
            "command": "/bin/echo",
            "args": ["hello"],
        }));
        assert_eq!(stdio.transport_kind().unwrap(), McpTransportKind::Stdio);
    }

    #[test]
    fn ambiguous_or_incomplete_transport_is_rejected() {
        let both = parse(json!({
            "command": "/bin/echo",
            "url": "http://127.0.0.1:9000/mcp",
        }));
        assert!(both.transport_kind().is_err());

        let empty = parse(json!({}));
        assert!(empty.transport_kind().is_err());

        let stdio_with_url = parse(json!({
            "type": "stdio",
            "url": "http://127.0.0.1:9000/mcp",
        }));
        assert!(stdio_with_url.transport_kind().is_err());
    }
}
