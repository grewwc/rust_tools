use std::io::{self, BufRead, Write};

use serde_json::{Value, json};

pub(super) fn serve<R: BufRead, W: Write>(mut reader: R, mut writer: W) -> io::Result<()> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = match reader.read_line(&mut line) {
            Ok(n) => n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        let params = req.get("params").cloned().unwrap_or(Value::Null);

        let result = match method.as_str() {
            "initialize" => json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "example-mcp",
                    "version": "0.1.0"
                }
            }),
            "notifications/initialized" => json!({}),
            "tools/list" => json!({
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo back the provided text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": { "type": "string" }
                            },
                            "required": ["text"]
                        }
                    },
                    {
                        "name": "add",
                        "description": "Add two integers",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": { "type": "integer" },
                                "b": { "type": "integer" }
                            },
                            "required": ["a", "b"]
                        }
                    }
                ]
            }),
            "tools/call" => handle_tools_call(params),
            "resources/list" => json!({ "resources": [] }),
            "resources/read" => json!({ "contents": [] }),
            "prompts/list" => json!({ "prompts": [] }),
            "prompts/get" => json!({ "messages": [] }),
            _ => {
                let resp = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                });
                let _ = writeln!(
                    writer,
                    "{}",
                    serde_json::to_string(&resp).unwrap_or_default()
                );
                let _ = writer.flush();
                continue;
            }
        };

        let resp = json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        });
        let _ = writeln!(
            writer,
            "{}",
            serde_json::to_string(&resp).unwrap_or_default()
        );
        let _ = writer.flush();
    }

    Ok(())
}

pub(super) fn serve_stdio() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    serve(stdin.lock(), stdout.lock())
}


fn handle_tools_call(params: Value) -> Value {
    let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(Value::Null);

    match name {
        "echo" => {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
            json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            })
        }
        "add" => {
            let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
            let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
            json!({
                "content": [
                    { "type": "text", "text": format!("{}", a + b) }
                ]
            })
        }
        _ => json!({
            "content": [
                { "type": "text", "text": "unknown tool" }
            ]
        }),
    }
}
