use serde_json::Value;

/// Mirror an agent_hang_debug event into the AIOS kernel trace ring.
///
/// This is the integration point for Phase 0 trace sinking: every expansion of
/// `agent_hang_debug!` / `agent_hang_span` eventually reaches
/// `report_agent_hang_debug`, which mirrors a copy here into
/// `TraceOps::trace_event`, making the kernel trace ring the authoritative
/// copy of all spans/events. The HTTP reporting can later be phased out and
/// replaced by unified output from kernel ring consumers.
fn mirror_to_aios_trace(
    run_id: &str,
    hypothesis_id: &str,
    location: &str,
    msg: &str,
    data: &Value,
) {
    use aios_kernel::FastMap;
    use aios_kernel::primitives::TraceLevel;

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
