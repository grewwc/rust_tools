use serde_json::Value;

/// Mirror an agent_hang_debug event into the AIOS kernel trace ring.
///
/// 这是 Phase 0 trace 下沉的接入点：所有 `agent_hang_debug!` / `agent_hang_span`
/// 展开后最终都会走到 `report_agent_hang_debug`，在这里镜像一份到
/// `TraceOps::trace_event`，从而让内核 trace ring 成为所有 span/event 的权威副本。
/// 后续可以逐步删除 HTTP 上报，改由 kernel ring 消费者统一输出。
fn mirror_to_aios_trace(
    run_id: &str,
    hypothesis_id: &str,
    location: &str,
    msg: &str,
    data: &Value,
) {
    use aios_kernel::primitives::TraceLevel;
    use rust_tools::commonw::FastMap;

    let g = match crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let kernel = match g.as_ref() {
        Some(k) => k.clone(),
        None => return,
    };
    drop(g);

    let mut fields: FastMap<String, String> = FastMap::default();
    fields.insert("run_id".to_string(), run_id.to_string());
    fields.insert("hypothesis_id".to_string(), hypothesis_id.to_string());
    fields.insert("location".to_string(), location.to_string());
    if let Value::Object(map) = data {
        for (k, v) in map {
            let s = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            fields.insert(k.clone(), s);
        }
    } else if !data.is_null() {
        fields.insert("data".to_string(), data.to_string());
    }

    if let Ok(mut guard) = kernel.lock() {
        guard.trace_event(
            location.to_string(),
            TraceLevel::Debug,
            None,
            fields,
            Some(msg.to_string()),
        );
    }
}

#[cfg(feature = "agent-hang-debug")]
pub(in crate::ai) fn report_agent_hang_debug(
    run_id: &'static str,
    hypothesis_id: &'static str,
    location: &'static str,
    msg: &'static str,
    data: Value,
) {
    mirror_to_aios_trace(run_id, hypothesis_id, location, msg, &data);
    std::thread::spawn(move || {
        let mut debug_server_url = "http://127.0.0.1:7777/event".to_string();
        let mut debug_session_id = "agent-hang".to_string();
        if let Ok(env_text) = std::fs::read_to_string(".dbg/agent-hang.env") {
            for line in env_text.lines() {
                if let Some(value) = line.strip_prefix("DEBUG_SERVER_URL=") {
                    if !value.trim().is_empty() {
                        debug_server_url = value.trim().to_string();
                    }
                } else if let Some(value) = line.strip_prefix("DEBUG_SESSION_ID=") {
                    if !value.trim().is_empty() {
                        debug_session_id = value.trim().to_string();
                    }
                }
            }
        }
        let payload = serde_json::json!({
            "sessionId": debug_session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
            "ts": chrono::Utc::now().timestamp_millis(),
        });
        if let Ok(client) = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_millis(300))
            .build()
        {
            let _ = client.post(debug_server_url).json(&payload).send();
        }
    });
}

#[cfg(not(feature = "agent-hang-debug"))]
pub(in crate::ai) fn report_agent_hang_debug(
    run_id: &'static str,
    hypothesis_id: &'static str,
    location: &'static str,
    msg: &'static str,
    data: Value,
) {
    mirror_to_aios_trace(run_id, hypothesis_id, location, msg, &data);
}
