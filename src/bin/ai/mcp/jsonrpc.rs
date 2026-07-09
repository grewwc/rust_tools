use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct JsonRpcRequest {
    pub(super) jsonrpc: String,
    pub(super) id: u64,
    pub(super) method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) params: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub(super) struct JsonRpcResponse {
    pub(super) jsonrpc: String,
    #[serde(default)]
    pub(super) id: Option<u64>,
    #[serde(default)]
    pub(super) result: Option<Value>,
    #[serde(default)]
    pub(super) error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub(super) struct JsonRpcError {
    pub(super) code: i64,
    pub(super) message: String,
    #[serde(default)]
    pub(super) data: Option<Value>,
}

#[derive(Debug, PartialEq)]
pub(super) enum InboundJsonRpc {
    Response(JsonRpcResponse),
    Notification {
        method: String,
    },
    Request {
        id: Value,
        method: String,
    },
}

pub(super) fn classify_inbound_jsonrpc(line: &str) -> Result<InboundJsonRpc, String> {
    let raw: Value =
        serde_json::from_str(line).map_err(|e| format!("Failed to parse response: {}", e))?;
    let Some(jsonrpc) = raw.get("jsonrpc").and_then(Value::as_str) else {
        return Err("Missing JSON-RPC version".to_string());
    };
    if jsonrpc != "2.0" {
        return Err(format!("Invalid JSON-RPC version: {}", jsonrpc));
    }

    let method = raw
        .get("method")
        .and_then(Value::as_str)
        .map(str::to_string);
    let id = raw.get("id").cloned();
    let has_result = raw.get("result").is_some();
    let has_error = raw.get("error").is_some();

    match (method, id, has_result, has_error) {
        (Some(method), Some(id), _, _) => Ok(InboundJsonRpc::Request { id, method }),
        (Some(method), None, _, _) => Ok(InboundJsonRpc::Notification { method }),
        (None, _, true, _) | (None, _, _, true) => serde_json::from_value(raw)
            .map(InboundJsonRpc::Response)
            .map_err(|e| format!("Failed to parse response: {}", e)),
        _ => Err("Invalid JSON-RPC message shape".to_string()),
    }
}

pub(super) fn unsupported_server_request_payload(id: Value, method: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": -32601,
            "message": format!("Unsupported MCP server request: {}", method),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        InboundJsonRpc, classify_inbound_jsonrpc, unsupported_server_request_payload,
    };
    use serde_json::json;

    #[test]
    fn classify_notification_without_id() {
        let inbound = classify_inbound_jsonrpc(
            r#"{"jsonrpc":"2.0","method":"notifications/tools/list_changed","params":{"x":1}}"#,
        )
        .unwrap();
        assert_eq!(
            inbound,
            InboundJsonRpc::Notification {
                method: "notifications/tools/list_changed".to_string(),
            }
        );
    }

    #[test]
    fn classify_server_request_with_id() {
        let inbound = classify_inbound_jsonrpc(
            r#"{"jsonrpc":"2.0","id":7,"method":"sampling/createMessage","params":{"x":1}}"#,
        )
        .unwrap();
        assert_eq!(
            inbound,
            InboundJsonRpc::Request {
                id: json!(7),
                method: "sampling/createMessage".to_string(),
            }
        );
    }

    #[test]
    fn unsupported_server_request_payload_uses_method_not_found() {
        let payload = unsupported_server_request_payload(json!(9), "sampling/createMessage");
        assert_eq!(payload["jsonrpc"], json!("2.0"));
        assert_eq!(payload["id"], json!(9));
        assert_eq!(payload["error"]["code"], json!(-32601));
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("sampling/createMessage")
        );
    }

    #[test]
    fn classify_response_with_null_id() {
        // 部分 MCP 服务器（如 mcp_ocr）会对 notifications/initialized 通知
        // 发送冗余确认响应 {"jsonrpc":"2.0","id":null,"result":{}}。
        // 该响应应被分类为 Response 且 id 为 None，以便 send_request_to_conn 跳过。
        let inbound = classify_inbound_jsonrpc(
            r#"{"jsonrpc":"2.0","id":null,"result":{}}"#,
        )
        .unwrap();
        match inbound {
            InboundJsonRpc::Response(resp) => {
                assert!(resp.id.is_none(), "id should be None for null id");
                assert!(resp.result.is_some());
            }
            other => panic!("expected Response, got {:?}", other),
        }
    }
}
