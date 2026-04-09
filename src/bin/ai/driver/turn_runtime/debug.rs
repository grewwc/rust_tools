use serde_json::Value;

#[cfg(feature = "agent-hang-debug")]
pub(in crate::ai) fn report_agent_hang_debug(
    run_id: &'static str,
    hypothesis_id: &'static str,
    location: &'static str,
    msg: &'static str,
    data: Value,
) {
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
    _run_id: &'static str,
    _hypothesis_id: &'static str,
    _location: &'static str,
    _msg: &'static str,
    _data: Value,
) {
}
