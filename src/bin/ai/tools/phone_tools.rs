//! Phone actions executed by the `a serve` Android shell.
//!
//! The turn child has no Android APIs. It publishes one action request over the
//! serve live FIFO; the web page answers by running the request through the
//! shell's JavaScript bridge and posts one JSON line back to `/confirm`, which
//! the child reads here. The tool therefore blocks for exactly as long as the
//! user takes to grant a permission or pick a notes app, and fails closed when
//! no Android client is attached.

use serde_json::{Value, json};

use crate::ai::tools::registry::common::{ToolRegistration, ToolSpec};

/// Maximum note body accepted from the model. The Android side enforces the
/// same ceiling; keeping it here turns an oversized note into a tool error the
/// model can correct instead of a silent truncation.
// A note travels inside one serve frame (4 KB); a larger body would be
// rejected by the transport after the model already produced it.
const MAX_NOTE_CHARS: usize = 3_000;
const MAX_LABEL_CHARS: usize = 120;
const MAX_TITLE_CHARS: usize = 200;

fn bounded_string(args: &Value, key: &str, max_chars: usize) -> String {
    args.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .chars()
        .take(max_chars)
        .collect()
}

/// Validate and normalize one phone action. Only the fields the Android bridge
/// understands survive, so a model cannot smuggle an arbitrary intent through
/// the action name.
fn build_phone_spec(args: &Value) -> Result<Value, String> {
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    match action {
        "alarm" => {
            let hour = args
                .get("hour")
                .and_then(Value::as_i64)
                .ok_or_else(|| "'hour' must be an integer from 0 to 23".to_string())?;
            let minute = args
                .get("minute")
                .and_then(Value::as_i64)
                .ok_or_else(|| "'minute' must be an integer from 0 to 59".to_string())?;
            if !(0..=23).contains(&hour) {
                return Err(format!("'hour' must be 0-23, got {hour}"));
            }
            if !(0..=59).contains(&minute) {
                return Err(format!("'minute' must be 0-59, got {minute}"));
            }
            Ok(json!({
                "action": "alarm",
                "hour": hour,
                "minute": minute,
                "label": bounded_string(args, "label", MAX_LABEL_CHARS),
            }))
        }
        "note" => {
            let text = args
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            if text.is_empty() {
                return Err("'text' must contain the note body".to_string());
            }
            if text.chars().count() > MAX_NOTE_CHARS {
                return Err(format!(
                    "'text' is longer than {MAX_NOTE_CHARS} characters; summarize it first"
                ));
            }
            Ok(json!({
                "action": "note",
                "title": bounded_string(args, "title", MAX_TITLE_CHARS),
                "text": text,
            }))
        }
        "location" => Ok(json!({ "action": "location" })),
        "" => Err("'action' is required: alarm, note, or location".to_string()),
        other => Err(format!(
            "unknown phone action '{other}'; use alarm, note, or location"
        )),
    }
}

fn execute_phone_action(args: &Value) -> Result<String, String> {
    let spec = build_phone_spec(args)?;
    if !crate::ai::serve_confirm::active() {
        return Err(
            "Phone actions are only available when this turn runs from the a serve \
             Android app; no phone is attached to this client."
                .to_string(),
        );
    }
    let answer = crate::ai::serve_confirm::native(&spec).ok_or_else(|| {
        "The phone did not answer the action request (client disconnected, cancelled, \
         or the turn ended)."
            .to_string()
    })?;
    let result: Value = serde_json::from_str(&answer)
        .map_err(|err| format!("invalid response from the phone client: {err}"))?;
    if result.get("ok").and_then(Value::as_bool) != Some(true) {
        let error = result
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("phone action failed");
        return Err(format!("Phone action failed: {error}"));
    }
    match spec["action"].as_str().unwrap_or("") {
        "alarm" => {
            let hour = spec["hour"].as_i64().unwrap_or(0);
            let minute = spec["minute"].as_i64().unwrap_or(0);
            let label = spec["label"].as_str().unwrap_or("");
            let suffix = if label.is_empty() {
                String::new()
            } else {
                format!(" (label: {label})")
            };
            Ok(format!(
                "Alarm set on the phone for {hour:02}:{minute:02}{suffix}."
            ))
        }
        "note" => Ok(
            "Opened the phone's share sheet with the note text. The user still chooses \
             the notes app and saves it."
                .to_string(),
        ),
        "location" => {
            let data = result.get("data").cloned().unwrap_or(Value::Null);
            let latitude = data
                .get("latitude")
                .and_then(Value::as_f64)
                .ok_or_else(|| "the phone returned no latitude".to_string())?;
            let longitude = data
                .get("longitude")
                .and_then(Value::as_f64)
                .ok_or_else(|| "the phone returned no longitude".to_string())?;
            let accuracy = data.get("accuracy").and_then(Value::as_f64).unwrap_or(0.0);
            Ok(format!(
                "Phone location: {latitude:.5}, {longitude:.5} (accuracy about {accuracy:.0} m)."
            ))
        }
        _ => Ok("Phone action completed.".to_string()),
    }
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "phone_action",
        description: "",
        execute: execute_phone_action,
    }
});

#[cfg(test)]
mod tests {
    use super::build_phone_spec;
    use serde_json::json;

    #[test]
    fn alarm_spec_drops_extra_fields_and_bounds_minutes() {
        let spec = build_phone_spec(&json!({
            "action": "alarm",
            "hour": 6,
            "minute": 0,
            "label": "wake up",
            "intent": "com.example.evil",
        }))
        .expect("valid alarm");
        assert_eq!(spec, json!({
            "action": "alarm",
            "hour": 6,
            "minute": 0,
            "label": "wake up",
        }));
        assert!(build_phone_spec(&json!({"action": "alarm", "hour": 24, "minute": 0})).is_err());
        assert!(build_phone_spec(&json!({"action": "alarm", "hour": 6, "minute": 60})).is_err());
        assert!(build_phone_spec(&json!({"action": "alarm", "hour": "6", "minute": 0})).is_err());
    }

    #[test]
    fn note_and_location_specs_are_validated() {
        assert!(build_phone_spec(&json!({"action": "note", "text": "  "})).is_err());
        let note = build_phone_spec(&json!({"action": "note", "title": "t", "text": "body"}))
            .expect("valid note");
        assert_eq!(note["text"], "body");
        assert_eq!(
            build_phone_spec(&json!({"action": "location"})).expect("valid location"),
            json!({"action": "location"})
        );
        assert!(build_phone_spec(&json!({"action": "launch", "package": "x"})).is_err());
    }
}