use super::*;

#[derive(Debug, Clone)]
pub(super) struct JsonRpcErr {
    pub(super) code: i64,
    pub(super) message: String,
    pub(super) data: Option<Value>,
}

pub(super) fn json_rpc_error(code: i64, message: &str, data: Option<Value>) -> JsonRpcErr {
    JsonRpcErr {
        code,
        message: message.to_string(),
        data,
    }
}

pub(super) fn write_json_rpc_result(out: &mut dyn Write, id: Option<&Value>, result: Value) -> io::Result<()> {
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id.cloned().unwrap_or(Value::Null),
        "result": result
    });
    writeln!(out, "{payload}")?;
    out.flush()
}

pub(super) fn write_json_rpc_error(
    out: &mut dyn Write,
    id: Option<&Value>,
    code: i64,
    message: &str,
    data: Option<Value>,
) -> io::Result<()> {
    let mut err = json!({
        "code": code,
        "message": message
    });
    if let Some(d) = data {
        err["data"] = d;
    }
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id.cloned().unwrap_or(Value::Null),
        "error": err
    });
    writeln!(out, "{payload}")?;
    out.flush()
}
