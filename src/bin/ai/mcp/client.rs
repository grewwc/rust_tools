use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicU64, Ordering},
};

use rust_tools::cw::SkipMap;
use serde_json::{Value, json};

use crate::ai::types::{
    FunctionDefinition, McpPrompt, McpResource, McpServerConfig, ToolDefinition,
};

use super::{connection::McpServerConnection, initialize_and_discover};

/// Sends a JSON-RPC request to an MCP server connection. The transport owns
/// request/response correlation, so stdio and HTTP share this entry point.
pub(in crate::ai) fn send_request_to_conn(
    conn: &mut McpServerConnection,
    id: u64,
    method: &str,
    params: Option<Value>,
) -> Result<Value, String> {
    conn.request(id, method, params)
}

/// Sends a JSON-RPC notification (no id, no response expected).
pub(in crate::ai) fn send_notification_to_conn(
    conn: &mut McpServerConnection,
    method: &str,
    params: Option<Value>,
) -> Result<(), String> {
    conn.notify(method, params)
}

pub(in crate::ai) type SharedMcpClient = Arc<std::sync::Mutex<McpClient>>;

type ServerId = String;

/// Preserve the MCP result envelope until its text-only driver projection.
/// A successful single text block without structured content keeps its legacy
/// bytes; richer results project as the complete JSON envelope.
struct McpToolResultEnvelope(Value);

impl McpToolResultEnvelope {
    fn into_text_result(self) -> Result<String, String> {
        if self.0["isError"].as_bool() == Some(true) {
            return Err(format!("MCP tool returned isError=true: {}", self.0));
        }

        if self.0.get("structuredContent").is_none()
            && self
                .0
                .get("isError")
                .is_none_or(|value| value.as_bool() == Some(false))
            && let Some(content) = self.0["content"].as_array()
        {
            if let [block] = content.as_slice()
                && block
                    .get("type")
                    .is_none_or(|value| value.as_str() == Some("text"))
                && let Some(text) = block["text"].as_str()
            {
                return Ok(text.to_string());
            }
            if content.is_empty()
                && self.0.as_object().is_some_and(|object| {
                    object
                        .keys()
                        .all(|key| matches!(key.as_str(), "content" | "isError"))
                })
            {
                return Ok(String::new());
            }
        }
        Ok(self.0.to_string())
    }
}

pub(in crate::ai) struct McpClient {
    pub(in crate::ai) servers: SkipMap<ServerId, Arc<Mutex<McpServerConnection>>>,
    next_id: AtomicU64,
    // Shared via Arc inside an RwLock: `routing_snapshot` is called by the
    // orchestrator every turn, and deep-cloning the whole tool/resource/prompt
    // cache is expensive (more so as tool schemas grow). Arc::clone is O(1),
    // and the cache is replaced wholesale, never modified in place. The RwLock
    // lets a restart refresh the cache through `&self` (`restart_connection`
    // and `call_tool` only have `&self`), keeping the snapshot consistent with
    // the live tool set without changing caller signatures.
    cached_tool_definitions: RwLock<Arc<Vec<ToolDefinition>>>,
    cached_resources: RwLock<Arc<Vec<(String, McpResource)>>>,
    cached_prompts: RwLock<Arc<Vec<(String, McpPrompt)>>>,
    cached_server_prefixes: RwLock<Arc<Vec<(String, String)>>>,
}

impl McpClient {
    pub(in crate::ai) fn new() -> Self {
        Self {
            servers: SkipMap::default(),
            next_id: AtomicU64::new(1),
            cached_tool_definitions: RwLock::new(Arc::new(Vec::new())),
            cached_resources: RwLock::new(Arc::new(Vec::new())),
            cached_prompts: RwLock::new(Arc::new(Vec::new())),
            cached_server_prefixes: RwLock::new(Arc::new(Vec::new())),
        }
    }

    pub(in crate::ai) fn routing_snapshot(&self) -> Self {
        Self {
            servers: SkipMap::default(),
            next_id: AtomicU64::new(self.next_id.load(Ordering::Relaxed)),
            cached_tool_definitions: RwLock::new(Arc::clone(
                &*self
                    .cached_tool_definitions
                    .read()
                    .unwrap_or_else(|e| e.into_inner()),
            )),
            cached_resources: RwLock::new(Arc::clone(
                &*self
                    .cached_resources
                    .read()
                    .unwrap_or_else(|e| e.into_inner()),
            )),
            cached_prompts: RwLock::new(Arc::clone(
                &*self
                    .cached_prompts
                    .read()
                    .unwrap_or_else(|e| e.into_inner()),
            )),
            cached_server_prefixes: RwLock::new(Arc::clone(
                &*self
                    .cached_server_prefixes
                    .read()
                    .unwrap_or_else(|e| e.into_inner()),
            )),
        }
    }

    pub(in crate::ai) fn connect_server(
        &mut self,
        name: &str,
        config: &McpServerConfig,
    ) -> Result<(), String> {
        if config.disabled {
            return Ok(());
        }

        let mut conn = McpServerConnection::connect(name, config)?;

        initialize_and_discover(&mut conn, &self.next_id)?;

        self.servers
            .insert(name.to_string(), Arc::new(Mutex::new(conn)));
        self.rebuild_metadata_cache();
        Ok(())
    }

    fn is_transport_error(err: &str) -> bool {
        let e = err.to_lowercase();
        e.contains("broken pipe")
            || e.contains("closed the stream unexpectedly")
            || e.contains("mcp response timeout")
            || e.contains("mcp http")
            || e.contains("mcp sse")
            || e.contains("failed waiting for mcp response")
            || e.contains("failed to read response")
            || e.contains("process exited with status")
            || e.contains("failed to get stdin")
            || e.contains("failed to get stdout")
            || e.contains("connection refused")
            || e.contains("error sending request")
    }

    fn is_protocol_desync_error(err: &str) -> bool {
        let e = err.to_lowercase();
        e.contains("mcp response id mismatch")
            || e.contains("failed to parse response")
            || e.contains("invalid json-rpc version")
    }

    fn is_user_interrupt_error(err: &str) -> bool {
        let e = err.to_lowercase();
        e.contains("canceled by user")
            || e.contains("cancelled by user")
            || e.contains("interrupted by user")
    }

    fn restart_connection(&self, conn: &mut McpServerConnection) -> Result<(), String> {
        conn.restart()?;

        initialize_and_discover(conn, &self.next_id)
    }

    pub(in crate::ai) fn reset_server(&self, server_name: &str) -> Result<(), String> {
        let conn_cell = self
            .servers
            .get_str_ref(server_name)
            .ok_or_else(|| format!("Server not found: {}", server_name))?;
        let mut conn = conn_cell
            .lock()
            .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
        self.restart_connection(&mut conn)
    }

    pub(in crate::ai) fn rebuild_metadata_cache(&self) {
        let mut tool_definitions = Vec::new();
        let mut resources = Vec::new();
        let mut prompts = Vec::new();
        let mut server_prefixes = Vec::with_capacity(self.servers.len());

        for (server_name, conn) in &self.servers {
            server_prefixes.push((server_name.clone(), format!("mcp_{server_name}_")));

            let Ok(conn) = conn.lock() else {
                continue;
            };

            tool_definitions.reserve(conn.tools.len());
            for tool in &conn.tools {
                tool_definitions.push(ToolDefinition {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: format!("mcp_{}_{}", server_name, tool.name),
                        description: tool.description.clone().unwrap_or_default(),
                        parameters: tool.input_schema.clone(),
                    },
                });
            }

            resources.reserve(conn.resources.len());
            resources.extend(
                conn.resources
                    .iter()
                    .cloned()
                    .map(|resource| (server_name.clone(), resource)),
            );

            prompts.reserve(conn.prompts.len());
            prompts.extend(
                conn.prompts
                    .iter()
                    .cloned()
                    .map(|prompt| (server_name.clone(), prompt)),
            );
        }

        rust_tools::sortw::stable_sort_by(&mut server_prefixes, |a, b| b.1.len().cmp(&a.1.len()));
        *self
            .cached_tool_definitions
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Arc::new(tool_definitions);
        *self
            .cached_resources
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Arc::new(resources);
        *self
            .cached_prompts
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Arc::new(prompts);
        *self
            .cached_server_prefixes
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Arc::new(server_prefixes);
    }

    fn next_request_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    pub(in crate::ai) fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: Value,
    ) -> Result<String, String> {
        let conn_cell = self
            .servers
            .get_str_ref(server_name)
            .ok_or_else(|| format!("Server not found: {}", server_name))?;
        Self::call_tool_with_recovery(
            conn_cell,
            server_name,
            tool_name,
            arguments,
            || Ok(self.next_request_id()),
            || {
                let mut conn = conn_cell
                    .lock()
                    .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
                let result = self.restart_connection(&mut conn);
                drop(conn);
                if result.is_ok() {
                    self.rebuild_metadata_cache();
                }
                result
            },
        )
    }

    /// Request IO holds only the per-server lock. Shared ID allocation and
    /// restart retain the owner's counter, and cache rebuilding follows the
    /// outer-to-inner lock order after releasing the connection lock.
    pub(in crate::ai) fn call_tool_on_shared(
        shared: &SharedMcpClient,
        server_name: &str,
        tool_name: &str,
        arguments: Value,
    ) -> Result<String, String> {
        let conn_arc = {
            let guard = shared
                .lock()
                .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
            guard
                .servers
                .get_str_ref(server_name)
                .cloned()
                .ok_or_else(|| format!("Server not found: {}", server_name))?
        };
        Self::call_tool_with_recovery(
            &conn_arc,
            server_name,
            tool_name,
            arguments,
            || {
                let guard = shared
                    .lock()
                    .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
                Ok(guard.next_request_id())
            },
            || {
                let result = {
                    let guard = shared
                        .lock()
                        .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
                    let mut conn = conn_arc
                        .lock()
                        .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
                    guard.restart_connection(&mut conn)
                };
                if result.is_ok() {
                    let guard = shared
                        .lock()
                        .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
                    guard.rebuild_metadata_cache();
                }
                result
            },
        )
    }

    fn call_tool_with_recovery(
        conn_arc: &Arc<Mutex<McpServerConnection>>,
        server_name: &str,
        tool_name: &str,
        arguments: Value,
        mut next_id: impl FnMut() -> Result<u64, String>,
        mut restart: impl FnMut() -> Result<(), String>,
    ) -> Result<String, String> {
        let params = json!({ "name": tool_name, "arguments": arguments });
        let id = next_id()?;
        let (first, retry_safe_before) = {
            let mut conn = conn_arc
                .lock()
                .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
            let retry_safe = conn.retry_safe_tools.contains(tool_name);
            (
                conn.request_with_delivery(id, "tools/call", Some(params.clone())),
                retry_safe,
            )
        };
        let failure = match first {
            Ok(result) => return McpToolResultEnvelope(result).into_text_result(),
            Err(failure) => failure,
        };

        if Self::is_user_interrupt_error(&failure.error) {
            let canceled = format!("MCP tool call canceled by user (Ctrl+C): {}", tool_name);
            // Nothing entered the transport, so the current connection remains
            // usable after the caller clears cancellation.
            if !failure.may_have_been_sent {
                return Err(canceled);
            }
            return match restart() {
                Ok(()) => Err(canceled),
                Err(restart_err) => Err(format!("{} | restart failed: {}", canceled, restart_err)),
            };
        }
        if !Self::is_transport_error(&failure.error)
            && !Self::is_protocol_desync_error(&failure.error)
        {
            return Err(failure.error);
        }

        let kind = if Self::is_protocol_desync_error(&failure.error) {
            "Protocol desync"
        } else {
            "Transport error"
        };
        if let Err(restart_err) = restart() {
            return Err(format!(
                "{}: {} | restart failed: {}",
                kind, failure.error, restart_err
            ));
        }

        let not_replayed = || {
            format!(
                "{}: {} | MCP tool call was not replayed: {}. The server may have executed the call; replay requires explicit safety in both connection generations.",
                kind, failure.error, tool_name
            )
        };
        let can_replay = |conn: &McpServerConnection| {
            !failure.may_have_been_sent
                || (retry_safe_before && conn.retry_safe_tools.contains(tool_name))
        };
        {
            let conn = conn_arc
                .lock()
                .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
            if !can_replay(&conn) {
                return Err(not_replayed());
            }
        }

        // Allocate outside the per-server lock: the shared allocator takes the
        // outer lock. Recheck safety under the lock used for the actual retry.
        let id = next_id()?;
        let mut conn = conn_arc
            .lock()
            .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
        if !can_replay(&conn) {
            return Err(not_replayed());
        }
        let result = conn
            .request_with_delivery(id, "tools/call", Some(params))
            .map_err(|failure| failure.error)?;
        McpToolResultEnvelope(result).into_text_result()
    }

    pub(in crate::ai) fn get_all_tools(&self) -> Vec<ToolDefinition> {
        self.cached_tool_definitions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .clone()
    }

    /// Returns an `Arc` snapshot of the cached tool definitions, avoiding a
    /// full `Vec<ToolDefinition>` clone (O(n)) while holding the
    /// `SharedMcpClient` outer lock. This shortens the outer-lock hold time of
    /// `tool_definitions` and eases lock contention with `call_tool`'s blocking
    /// IO.
    pub(in crate::ai) fn cached_tool_definitions_arc(&self) -> Arc<Vec<ToolDefinition>> {
        Arc::clone(
            &self
                .cached_tool_definitions
                .read()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }

    pub(in crate::ai) fn parse_tool_name_for_known_server(
        &self,
        full_name: &str,
    ) -> Option<(String, String)> {
        if !full_name.starts_with("mcp_") {
            return None;
        }

        let prefixes = self
            .cached_server_prefixes
            .read()
            .unwrap_or_else(|e| e.into_inner());
        for (server_name, prefix) in prefixes.iter() {
            if let Some(tool_name) = full_name.strip_prefix(prefix)
                && !tool_name.is_empty()
            {
                return Some((server_name.clone(), tool_name.to_string()));
            }
        }

        None
    }

    pub(in crate::ai) fn get_all_resources(&self) -> Vec<(String, McpResource)> {
        self.cached_resources
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .clone()
    }

    pub(in crate::ai) fn get_all_prompts(&self) -> Vec<(String, McpPrompt)> {
        self.cached_prompts
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .clone()
    }

    pub(in crate::ai) fn disconnect_all(&mut self) {
        for (_, conn) in self.servers.drain() {
            drop(conn);
        }
        *self
            .cached_tool_definitions
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Arc::new(Vec::new());
        *self
            .cached_resources
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Arc::new(Vec::new());
        *self
            .cached_prompts
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Arc::new(Vec::new());
        *self
            .cached_server_prefixes
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Arc::new(Vec::new());
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.disconnect_all();
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::ai::mcp::test_support::*;

    #[derive(Clone, Copy, Debug)]
    enum CallApi {
        Direct,
        Shared,
    }

    const CALL_APIS: [CallApi; 2] = [CallApi::Direct, CallApi::Shared];

    fn call(api: CallApi, shared: &SharedMcpClient) -> Result<String, String> {
        match api {
            CallApi::Direct => shared.lock().unwrap().call_tool(
                "fixture",
                "read_with_effects",
                json!({"value": "run"}),
            ),
            CallApi::Shared => McpClient::call_tool_on_shared(
                shared,
                "fixture",
                "read_with_effects",
                json!({"value": "run"}),
            ),
        }
    }

    fn responses(first_id: u64, annotations: Value) -> Vec<Option<Value>> {
        let mut responses = complete_responses(first_id, "read_with_effects");
        responses[2].as_mut().unwrap()["result"]["tools"][0]["annotations"] = annotations;
        responses
    }

    fn client(first: &ScriptedServer, restarted: Option<&ScriptedServer>) -> SharedMcpClient {
        let mut client = McpClient::new();
        client.connect_server("fixture", &first.config).unwrap();
        if let Some(restarted) = restarted {
            client
                .servers
                .get_str_ref("fixture")
                .unwrap()
                .lock()
                .unwrap()
                .config = restarted.config.clone();
        }
        Arc::new(Mutex::new(client))
    }

    #[test]
    fn mcp_reliability_lost_response_never_replays_unknown_or_unsafe_calls() {
        let _guard = serial_guard();
        for api in CALL_APIS {
            for annotations in [
                json!({}),
                json!({"readOnlyHint": false, "idempotentHint": false}),
                json!({"readOnlyHint": "true", "idempotentHint": 1}),
            ] {
                let mut first_responses = responses(1, annotations.clone());
                first_responses.push(None);
                let first = ScriptedServer::new_closing(first_responses);
                let mut restarted_responses = responses(6, annotations.clone());
                restarted_responses.push(success(
                    10,
                    json!({"content": [{"type": "text", "text": "replayed"}]}),
                ));
                let restarted = ScriptedServer::new(restarted_responses);
                let shared = client(&first, Some(&restarted));
                let old_cache = shared.lock().unwrap().cached_tool_definitions_arc();

                let result = call(api, &shared);

                assert!(result.is_err(), "{api:?}: {annotations}: {result:?}");
                assert!(result.unwrap_err().contains("not replayed"));
                assert_eq!(first.executions().len(), 1);
                assert!(restarted.executions().is_empty());
                assert_eq!(restarted.transcript(), complete_transcript(6));
                let guard = shared.lock().unwrap();
                assert!(!Arc::ptr_eq(
                    &old_cache,
                    &guard.cached_tool_definitions_arc()
                ));
                assert_eq!(guard.get_all_tools().len(), 1);
                assert_eq!(guard.get_all_resources().len(), 1);
                assert_eq!(guard.get_all_prompts().len(), 1);
            }
        }
    }

    #[test]
    fn mcp_reliability_desync_does_not_replay_an_unknown_call() {
        let _guard = serial_guard();
        for api in CALL_APIS {
            let mut first_responses = responses(1, json!({}));
            first_responses.push(success(99, json!({"content": [{"text": "wrong id"}]})));
            let first = ScriptedServer::new(first_responses);
            let mut restarted_responses = responses(6, json!({}));
            restarted_responses.push(success(10, json!({"content": [{"text": "replayed"}]})));
            let restarted = ScriptedServer::new(restarted_responses);
            let shared = client(&first, Some(&restarted));

            let result = call(api, &shared);

            assert!(result.is_err(), "{api:?}: {result:?}");
            assert_eq!(first.executions().len(), 1);
            assert!(restarted.executions().is_empty());
        }
    }

    #[test]
    fn mcp_reliability_dead_connection_retries_before_any_tool_dispatch() {
        let _guard = serial_guard();
        for api in CALL_APIS {
            let first = ScriptedServer::new(responses(1, json!({})));
            let mut restarted_responses = responses(6, json!({}));
            restarted_responses.push(success(
                10,
                json!({"content": [{"type": "text", "text": "once"}]}),
            ));
            let restarted = ScriptedServer::new(restarted_responses);
            let shared = client(&first, Some(&restarted));
            shared
                .lock()
                .unwrap()
                .servers
                .get_str_ref("fixture")
                .unwrap()
                .lock()
                .unwrap()
                .shutdown_and_reap();

            assert_eq!(call(api, &shared).unwrap(), "once");
            assert!(first.executions().is_empty());
            assert_eq!(first.transcript(), complete_transcript(1));
            assert_eq!(restarted.executions().len(), 1);
        }
    }

    #[test]
    fn mcp_reliability_explicit_safe_retry_uses_the_same_result_projection() {
        let _guard = serial_guard();
        let envelope = json!({
            "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
            "structuredContent": {"answer": 42}, "isError": false
        });
        for api in CALL_APIS {
            for annotations in [
                json!({"readOnlyHint": true}),
                json!({"idempotentHint": true}),
            ] {
                let mut first_responses = responses(1, annotations.clone());
                first_responses.push(None);
                let first = ScriptedServer::new_closing(first_responses);
                let mut restarted_responses = responses(6, annotations);
                restarted_responses.push(success(10, envelope.clone()));
                let restarted = ScriptedServer::new(restarted_responses);
                let shared = client(&first, Some(&restarted));

                assert_eq!(call(api, &shared).unwrap(), envelope.to_string());
                assert_eq!(first.executions().len(), 1);
                assert_eq!(restarted.executions().len(), 1);
                assert_eq!(
                    first.executions()[0]["params"],
                    restarted.executions()[0]["params"]
                );
            }
        }
    }

    #[test]
    fn mcp_reliability_retry_requires_safety_in_both_connection_generations() {
        let _guard = serial_guard();
        for api in CALL_APIS {
            for (before, after) in [
                (json!({"readOnlyHint": true}), json!({})),
                (json!({}), json!({"idempotentHint": true})),
            ] {
                let mut first_responses = responses(1, before);
                first_responses.push(None);
                let first = ScriptedServer::new_closing(first_responses);
                let mut restarted_responses = responses(6, after);
                restarted_responses.push(success(10, json!({"content": [{"text": "replayed"}]})));
                let restarted = ScriptedServer::new(restarted_responses);
                let shared = client(&first, Some(&restarted));

                assert!(call(api, &shared).is_err());
                assert_eq!(first.executions().len(), 1);
                assert!(restarted.executions().is_empty());
            }
        }
    }

    #[test]
    fn mcp_reliability_rich_result_projection_keeps_all_payloads() {
        let _guard = serial_guard();
        for api in CALL_APIS {
            for envelope in [
                json!({"content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}),
                json!({"content": [
                    {"type": "image", "data": "aW1hZ2U=", "mimeType": "image/png"},
                    {"type": "audio", "data": "YXVkaW8=", "mimeType": "audio/wav"},
                    {"type": "resource", "resource": {"uri": "test://file", "text": "resource text"}},
                    {"type": "resource_link", "uri": "test://link", "name": "link"}
                ]}),
                json!({"content": [], "structuredContent": {"answer": 42}}),
                json!({"structuredContent": {"answer": 42}}),
                json!({"content": [{"type": "image", "text": "not a text block", "data": "aW1hZ2U="}]}),
                json!({"content": [{"type": "text", "text": "one", "annotations": {"audience": ["user"]}}], "structuredContent": {"answer": 42}, "_meta": {"trace": "fixture"}}),
            ] {
                let mut scripted = responses(1, json!({}));
                scripted.push(success(5, envelope.clone()));
                let first = ScriptedServer::new(scripted);
                let shared = client(&first, None);

                assert_eq!(call(api, &shared).unwrap(), envelope.to_string(), "{api:?}");
                assert_eq!(first.executions().len(), 1);
            }
        }
    }

    #[test]
    fn mcp_reliability_single_text_and_jsonrpc_errors_keep_existing_outputs() {
        let _guard = serial_guard();
        for api in CALL_APIS {
            for envelope in [
                json!({"content": [{"type": "text", "text": "unchanged\\ntext"}]}),
                json!({"content": [{"text": "unchanged\\ntext"}], "isError": false}),
                json!({"content": [{"type": "text", "text": "unchanged\\ntext", "annotations": {"audience": ["user"]}}], "_meta": {"trace": "fixture"}}),
            ] {
                let mut scripted = responses(1, json!({}));
                scripted.push(success(5, envelope));
                let first = ScriptedServer::new(scripted);
                assert_eq!(
                    call(api, &client(&first, None)).unwrap(),
                    "unchanged\\ntext"
                );
            }
            let mut scripted = responses(1, json!({}));
            scripted.push(Some(json!({
                "jsonrpc": "2.0", "id": 5,
                "error": {"code": -32602, "message": "invalid arguments", "data": {"field": "value"}}
            })));
            let first = ScriptedServer::new(scripted);
            assert_eq!(
                call(api, &client(&first, None)).unwrap_err(),
                "MCP error -32602: invalid arguments ({\"field\":\"value\"})"
            );
            assert_eq!(first.executions().len(), 1);
        }
    }

    #[test]
    fn mcp_reliability_safe_retry_is_bounded_and_does_not_swallow_is_error() {
        let _guard = serial_guard();
        for api in CALL_APIS {
            for lost_second_response in [false, true] {
                let annotations = json!({"readOnlyHint": true});
                let mut first_responses = responses(1, annotations.clone());
                first_responses.push(None);
                let first = ScriptedServer::new_closing(first_responses);
                let mut restarted_responses = responses(6, annotations);
                let envelope =
                    json!({"content": [{"type": "text", "text": "tool failed"}], "isError": true});
                restarted_responses.push(if lost_second_response {
                    None
                } else {
                    success(10, envelope.clone())
                });
                let restarted = ScriptedServer::new_closing(restarted_responses);
                let shared = client(&first, Some(&restarted));

                let result = call(api, &shared);

                assert!(result.is_err(), "{api:?}: {result:?}");
                if !lost_second_response {
                    let error = result.unwrap_err();
                    assert!(error.contains("isError"));
                    assert!(error.contains(&envelope.to_string()));
                }
                assert_eq!(first.executions().len(), 1);
                assert_eq!(restarted.executions().len(), 1);
            }
        }
    }

    #[test]
    fn mcp_handshake_client_connections_share_the_current_counter() {
        let _guard = serial_guard();
        let first = ScriptedServer::new(complete_responses(21, "first"));
        let second = ScriptedServer::new(complete_responses(25, "second"));
        let mut client = McpClient::new();
        client.next_id.store(21, Ordering::Relaxed);

        client.connect_server("first", &first.config).unwrap();
        client.connect_server("second", &second.config).unwrap();

        assert_eq!(first.transcript(), complete_transcript(21));
        assert_eq!(second.transcript(), complete_transcript(25));
        assert_eq!(client.next_id.load(Ordering::Relaxed), 29);
        for name in ["first", "second"] {
            let conn = client.servers.get_str_ref(name).unwrap().lock().unwrap();
            assert_eq!(observed_catalogs(&conn), catalogs(name));
        }
        assert_eq!(client.get_all_tools().len(), 2);
        assert_eq!(client.get_all_resources().len(), 2);
        assert_eq!(client.get_all_prompts().len(), 2);
    }

    #[test]
    fn mcp_handshake_failed_shared_reconnect_preserves_partial_state_and_cache() {
        let _guard = serial_guard();
        let mut first_responses = complete_responses(41, "old");
        first_responses.push(failure(45, -32000, "broken pipe"));
        let first = ScriptedServer::new(first_responses);
        let mut restart_responses = complete_responses(46, "new");
        restart_responses[3] = failure(48, -32000, "resource catalog failed");
        let restarted = ScriptedServer::new(restart_responses);
        let mut client = McpClient::new();
        client.next_id.store(41, Ordering::Relaxed);
        client.connect_server("restart", &first.config).unwrap();
        let cached_tools = client.cached_tool_definitions_arc();
        let cached_resources = Arc::clone(&client.cached_resources.read().unwrap());
        let cached_prompts = Arc::clone(&client.cached_prompts.read().unwrap());
        let conn_arc = client.servers.get_str_ref("restart").unwrap().clone();
        // Supply the next scripted process to the caller-owned restart path.
        conn_arc.lock().unwrap().config = restarted.config.clone();
        let shared = Arc::new(Mutex::new(client));

        let err =
            McpClient::call_tool_on_shared(&shared, "restart", "old", json!({"value": "run"}))
                .unwrap_err();

        assert_eq!(
            err,
            "Transport error: MCP error -32000: broken pipe | restart failed: MCP error -32000: resource catalog failed"
        );
        let mut first_transcript = complete_transcript(41);
        first_transcript.push(json!({
            "jsonrpc": "2.0", "id": 45, "method": "tools/call",
            "params": {"name": "old", "arguments": {"value": "run"}}
        }));
        assert_eq!(first.transcript(), first_transcript);
        assert_eq!(restarted.transcript(), complete_transcript(46)[..4]);
        let client = shared.lock().unwrap();
        assert_eq!(client.next_id.load(Ordering::Relaxed), 49);
        assert!(Arc::ptr_eq(
            &cached_tools,
            &client.cached_tool_definitions_arc()
        ));
        assert!(Arc::ptr_eq(
            &cached_resources,
            &client.cached_resources.read().unwrap()
        ));
        assert!(Arc::ptr_eq(
            &cached_prompts,
            &client.cached_prompts.read().unwrap()
        ));
        assert!(Arc::ptr_eq(
            &conn_arc,
            client.servers.get_str_ref("restart").unwrap()
        ));
        let conn = conn_arc.lock().unwrap();
        assert_eq!(
            observed_catalogs(&conn),
            [catalogs("new")[0].clone(), json!([]), json!([])]
        );
    }
}
