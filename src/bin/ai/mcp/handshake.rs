use std::sync::atomic::{AtomicU64, Ordering};

use rust_tools::commonw::FastSet;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use super::{
    connection::McpServerConnection, is_method_not_found, send_notification_to_conn,
    send_request_to_conn,
};

/// Initialize and discover through the caller's live connection and ID counter.
/// Catalogs are assigned sequentially so failures retain earlier assignments.
pub(in crate::ai) fn initialize_and_discover(
    conn: &mut McpServerConnection,
    next_id: &AtomicU64,
) -> Result<(), String> {
    let params = json!({
        "protocolVersion": conn.protocol_version(),
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

    let id = next_id.fetch_add(1, Ordering::Relaxed);
    send_request_to_conn(conn, id, "initialize", Some(params))?;
    send_notification_to_conn(conn, "notifications/initialized", None)?;

    let discovered: Vec<DiscoveredTool> = list_catalog(conn, next_id, "tools/list", "tools")?;
    let mut retry_safe_tools = FastSet::default();
    let mut unsafe_tools = FastSet::default();
    conn.tools = discovered
        .into_iter()
        .map(|entry| {
            if entry.annotations["readOnlyHint"].as_bool() == Some(true)
                || entry.annotations["idempotentHint"].as_bool() == Some(true)
            {
                retry_safe_tools.insert(entry.tool.name.clone());
            } else {
                unsafe_tools.insert(entry.tool.name.clone());
            }
            entry.tool
        })
        .collect();
    // Duplicate names with conflicting hints must not escalate replay safety.
    retry_safe_tools.retain(|name| !unsafe_tools.contains(name));
    conn.retry_safe_tools = retry_safe_tools;
    conn.resources = list_catalog(conn, next_id, "resources/list", "resources")?;
    conn.prompts = list_catalog(conn, next_id, "prompts/list", "prompts")?;
    Ok(())
}

#[derive(Deserialize)]
struct DiscoveredTool {
    #[serde(flatten)]
    tool: crate::ai::types::McpTool,
    #[serde(default)]
    annotations: Value,
}

fn list_catalog<T: DeserializeOwned>(
    conn: &mut McpServerConnection,
    next_id: &AtomicU64,
    method: &str,
    key: &str,
) -> Result<Vec<T>, String> {
    let id = next_id.fetch_add(1, Ordering::Relaxed);
    let result = match send_request_to_conn(conn, id, method, None) {
        Ok(value) => value,
        Err(err) if is_method_not_found(&err) => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };

    Ok(result[key]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| serde_json::from_value(item.clone()).ok())
                .collect()
        })
        .unwrap_or_default())
}

#[cfg(all(test, unix))]
pub(in crate::ai) mod test_support {
    use std::{
        fs,
        path::PathBuf,
        sync::MutexGuard,
        time::{Duration, Instant},
    };

    use serde_json::{Value, json};

    use crate::ai::{mcp::connection::McpServerConnection, types::McpServerConfig};

    pub(in crate::ai) fn serial_guard() -> MutexGuard<'static, ()> {
        crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    pub(in crate::ai) struct ScriptedServer {
        pub(in crate::ai) config: McpServerConfig,
        directory: PathBuf,
    }

    impl ScriptedServer {
        pub(in crate::ai) fn new(responses: Vec<Option<Value>>) -> Self {
            Self::with_lifetime(responses, false)
        }

        pub(in crate::ai) fn new_closing(responses: Vec<Option<Value>>) -> Self {
            Self::with_lifetime(responses, true)
        }

        fn with_lifetime(responses: Vec<Option<Value>>, close: bool) -> Self {
            let directory =
                std::env::temp_dir().join(format!("mcp-handshake-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&directory).unwrap();
            let transcript = directory.join("transcript.jsonl");
            let executions = directory.join("executions.jsonl");
            fs::write(&transcript, "").unwrap();
            fs::write(&executions, "").unwrap();

            // Recording a call models its server-side effect before replying.
            let record = concat!(
                "printf '%s\\n' \"$request\" >> \"$1\"\n",
                "case \"$request\" in *'\"method\":\"tools/call\"'*) ",
                "printf '%s\\n' \"$request\" >> \"$2\";; esac\n"
            );
            let mut script = String::new();
            for response in responses {
                script.push_str("IFS= read -r request || exit 1\n");
                script.push_str(record);
                if let Some(response) = response {
                    let quoted = response.to_string().replace('\'', "'\\''");
                    script.push_str(&format!("printf '%s\\n' '{quoted}'\n"));
                }
            }
            if !close {
                // Keep the pipe live and record any unexpected later requests.
                script.push_str("while IFS= read -r request; do\n");
                script.push_str(record);
                script.push_str("done\n");
            }

            let config = serde_json::from_value(json!({
                "command": "/bin/sh",
                "args": ["-c", script, "mcp-handshake-test", transcript, executions],
                "request_timeout_ms": 2000
            }))
            .unwrap();
            Self { config, directory }
        }

        pub(in crate::ai) fn transcript(&self) -> Vec<Value> {
            self.read_records("transcript.jsonl")
        }

        pub(in crate::ai) fn executions(&self) -> Vec<Value> {
            self.read_records("executions.jsonl")
        }

        pub(in crate::ai) fn wait_for_execution(&self) {
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                if fs::read_to_string(self.directory.join("executions.jsonl"))
                    .is_ok_and(|records| records.ends_with('\n'))
                {
                    return;
                }
                assert!(
                    Instant::now() < deadline,
                    "scripted tool was not dispatched"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        fn read_records(&self, name: &str) -> Vec<Value> {
            fs::read_to_string(self.directory.join(name))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        }
    }

    impl Drop for ScriptedServer {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }

    pub(in crate::ai) fn catalogs(name: &str) -> [Value; 3] {
        [
            json!([{
                "name": name,
                "description": "fixture tool",
                "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}
            }]),
            json!([{
                "uri": format!("test://{name}"),
                "name": name,
                "description": null,
                "mimeType": "text/plain"
            }]),
            json!([{
                "name": name,
                "description": null,
                "arguments": [{"name": "topic", "description": null, "required": true}]
            }]),
        ]
    }

    pub(in crate::ai) fn observed_catalogs(conn: &McpServerConnection) -> [Value; 3] {
        [
            serde_json::to_value(&conn.tools).unwrap(),
            serde_json::to_value(&conn.resources).unwrap(),
            serde_json::to_value(&conn.prompts).unwrap(),
        ]
    }

    pub(in crate::ai) fn success(id: u64, result: Value) -> Option<Value> {
        Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    pub(in crate::ai) fn failure(id: u64, code: i64, message: &str) -> Option<Value> {
        Some(json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}))
    }

    pub(in crate::ai) fn complete_responses(first_id: u64, name: &str) -> Vec<Option<Value>> {
        let [tools, resources, prompts] = catalogs(name);
        vec![
            success(first_id, json!({"protocolVersion": "2024-11-05"})),
            None,
            success(first_id + 1, json!({"tools": tools})),
            success(first_id + 2, json!({"resources": resources})),
            success(first_id + 3, json!({"prompts": prompts})),
        ]
    }

    pub(in crate::ai) fn complete_transcript(first_id: u64) -> Vec<Value> {
        vec![
            json!({
                "jsonrpc": "2.0", "id": first_id, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                    "clientInfo": {"name": "rust-tools-ai", "version": "1.0.0"}
                }
            }),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json!({"jsonrpc": "2.0", "id": first_id + 1, "method": "tools/list"}),
            json!({"jsonrpc": "2.0", "id": first_id + 2, "method": "resources/list"}),
            json!({"jsonrpc": "2.0", "id": first_id + 3, "method": "prompts/list"}),
        ]
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::{test_support::*, *};

    #[test]
    fn mcp_handshake_complete_transcript_and_catalogs() {
        let _guard = serial_guard();
        let server = ScriptedServer::new(complete_responses(11, "complete"));
        let mut conn = McpServerConnection::connect("complete", &server.config).unwrap();
        let next_id = AtomicU64::new(11);

        initialize_and_discover(&mut conn, &next_id).unwrap();

        assert_eq!(server.transcript(), complete_transcript(11));
        assert_eq!(observed_catalogs(&conn), catalogs("complete"));
        assert_eq!(next_id.load(Ordering::Relaxed), 15);
    }

    #[test]
    fn mcp_handshake_unsupported_discovery_clears_catalogs() {
        let _guard = serial_guard();
        let server = ScriptedServer::new(vec![
            success(1, json!({})),
            None,
            failure(2, -32601, "Method not found"),
            failure(3, -32601, "Method not found"),
            failure(4, -32601, "Method not found"),
        ]);
        let mut conn = McpServerConnection::connect("unsupported", &server.config).unwrap();
        let [tools, resources, prompts] = catalogs("old");
        conn.tools = serde_json::from_value(tools).unwrap();
        conn.resources = serde_json::from_value(resources).unwrap();
        conn.prompts = serde_json::from_value(prompts).unwrap();
        let next_id = AtomicU64::new(1);

        initialize_and_discover(&mut conn, &next_id).unwrap();

        assert_eq!(server.transcript(), complete_transcript(1));
        assert_eq!(observed_catalogs(&conn), [json!([]), json!([]), json!([])]);
        assert_eq!(next_id.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn mcp_handshake_filters_malformed_items_and_missing_arrays() {
        let _guard = serial_guard();
        let [tools, resources, prompts] = catalogs("valid");
        let cases = [
            (
                [
                    json!({"tools": [null, {"name": 42}, tools[0], {"name": "no-schema"}]}),
                    json!({"resources": [false, {}, resources[0]]}),
                    json!({"prompts": [{"name": 42}, prompts[0], null]}),
                ],
                catalogs("valid"),
            ),
            (
                [
                    json!({}),
                    json!({"resources": {}}),
                    json!({"prompts": null}),
                ],
                [json!([]), json!([]), json!([])],
            ),
        ];
        for ([tools, resources, prompts], expected) in cases {
            let server = ScriptedServer::new(vec![
                success(1, json!({})),
                None,
                success(2, tools),
                success(3, resources),
                success(4, prompts),
            ]);
            let mut conn = McpServerConnection::connect("parsing", &server.config).unwrap();
            let next_id = AtomicU64::new(1);

            initialize_and_discover(&mut conn, &next_id).unwrap();

            assert_eq!(server.transcript(), complete_transcript(1));
            assert_eq!(observed_catalogs(&conn), expected);
            assert_eq!(next_id.load(Ordering::Relaxed), 5);
        }
    }

    #[test]
    fn mcp_handshake_first_error_preserves_partial_assignment_and_lazy_ids() {
        let _guard = serial_guard();
        let [_, old_resources, old_prompts] = catalogs("old");
        let [new_tools, new_resources, _] = catalogs("new");
        let cases = [
            (0, 1, -32601, catalogs("old")),
            (2, 2, -32000, catalogs("old")),
            (
                3,
                3,
                -32000,
                [new_tools.clone(), old_resources, old_prompts.clone()],
            ),
            (4, 4, -32000, [new_tools, new_resources, old_prompts]),
        ];
        for (response_index, failed_id, code, expected) in cases {
            let mut responses = complete_responses(1, "new");
            responses[response_index] = Some(json!({
                "jsonrpc": "2.0", "id": failed_id,
                "error": {"code": code, "message": "scripted failure", "data": {"phase": response_index}}
            }));
            let server = ScriptedServer::new(responses);
            let mut conn = McpServerConnection::connect("failure", &server.config).unwrap();
            let [tools, resources, prompts] = catalogs("old");
            conn.tools = serde_json::from_value(tools).unwrap();
            conn.resources = serde_json::from_value(resources).unwrap();
            conn.prompts = serde_json::from_value(prompts).unwrap();
            let next_id = AtomicU64::new(1);

            let err = initialize_and_discover(&mut conn, &next_id).unwrap_err();

            assert_eq!(
                err,
                format!("MCP error {code}: scripted failure ({{\"phase\":{response_index}}})")
            );
            assert_eq!(
                server.transcript(),
                complete_transcript(1)[..=response_index]
            );
            assert_eq!(observed_catalogs(&conn), expected);
            assert_eq!(next_id.load(Ordering::Relaxed), failed_id + 1);
        }
    }
}
