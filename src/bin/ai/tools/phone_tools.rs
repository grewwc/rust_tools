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
const MAX_PLACE_CHARS: usize = 200;
const MAX_DESCRIPTION_CHARS: usize = 1_000;
const MAX_SMS_CHARS: usize = 1_000;
const MAX_CONTACT_CHARS: usize = 120;
const MAX_NUMBER_CHARS: usize = 40;

fn bounded_string(args: &Value, key: &str, max_chars: usize) -> String {
    args.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .chars()
        .take(max_chars)
        .collect()
}

/// Accept only the local wall-clock form the Android side parses
/// (`YYYY-MM-DDTHH:MM`). Rejecting anything else here keeps date arithmetic
/// in the model's hands and prevents a silent timezone reinterpretation.
fn parse_local_minute(value: &str) -> Option<(i64, i64, i64, i64, i64)> {
    let bytes = value.as_bytes();
    if bytes.len() != 16 {
        return None;
    }
    let shaped = !bytes.iter().enumerate().any(|(index, byte)| match index {
        4 | 7 => *byte != b'-',
        10 => *byte != b'T',
        13 => *byte != b':',
        _ => !byte.is_ascii_digit(),
    });
    if !shaped {
        return None;
    }
    let field = |start: usize, len: usize| value[start..start + len].parse::<i64>().ok();
    let year = field(0, 4)?;
    let month = field(5, 2)?;
    let day = field(8, 2)?;
    let hour = field(11, 2)?;
    let minute = field(14, 2)?;
    if !(1970..=2100).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
    {
        return None;
    }
    Some((year, month, day, hour, minute))
}

/// A dialable string: digits plus the separators every dialer accepts. The
/// number is never dialled or messaged directly, only pre-filled.
fn is_dialable(value: &str) -> bool {
    !value.is_empty()
        && value.chars().any(|c| c.is_ascii_digit())
        && value
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | ' ' | '(' | ')' | '.'))
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
        "now" => Ok(json!({ "action": "now" })),
        "calendar_read" => {
            let days = args.get("days").and_then(Value::as_i64).unwrap_or(7);
            if !(1..=30).contains(&days) {
                return Err(format!("'days' must be 1-30, got {days}"));
            }
            let max = args.get("max").and_then(Value::as_i64).unwrap_or(10);
            if !(1..=20).contains(&max) {
                return Err(format!("'max' must be 1-20, got {max}"));
            }
            Ok(json!({
                "action": "calendar_read",
                "days": days,
                "max": max,
            }))
        }
        "calendar_add" => {
            let title = args
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .chars()
                .take(MAX_TITLE_CHARS)
                .collect::<String>();
            if title.is_empty() {
                return Err("'title' must contain the event title".to_string());
            }
            let begin = args
                .get("begin_local")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if parse_local_minute(begin).is_none() {
                return Err(
                    "'begin_local' must be local time as YYYY-MM-DDTHH:MM; call \
                     phone_action with action 'now' first to learn the phone's local date"
                        .to_string(),
                );
            }
            let duration = args
                .get("duration_min")
                .and_then(Value::as_i64)
                .unwrap_or(60);
            if !(5..=600).contains(&duration) {
                return Err(format!("'duration_min' must be 5-600, got {duration}"));
            }
            Ok(json!({
                "action": "calendar_add",
                "title": title,
                "begin_local": begin,
                "duration_min": duration,
                "location": bounded_string(args, "location", MAX_PLACE_CHARS).trim().to_string(),
                "description": bounded_string(args, "description", MAX_DESCRIPTION_CHARS)
                    .trim()
                    .to_string(),
            }))
        }
        "sms" => {
            let contact = bounded_string(args, "contact", MAX_CONTACT_CHARS)
                .trim()
                .to_string();
            let number = bounded_string(args, "number", MAX_NUMBER_CHARS)
                .trim()
                .to_string();
            if contact.is_empty() && number.is_empty() {
                return Err("'contact' or 'number' is required for sms".to_string());
            }
            if !number.is_empty() && !is_dialable(&number) {
                return Err("'number' may contain only digits and + - ( ) . spaces".to_string());
            }
            let text = args
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            if text.is_empty() {
                return Err("'text' must contain the message body".to_string());
            }
            if text.chars().count() > MAX_SMS_CHARS {
                return Err(format!(
                    "'text' is longer than {MAX_SMS_CHARS} characters; shorten it first"
                ));
            }
            Ok(json!({
                "action": "sms",
                "contact": contact,
                "number": number,
                "text": text,
            }))
        }
        "" => Err(
            "'action' is required: alarm, note, location, now, calendar_read, \
             calendar_add, or sms"
                .to_string(),
        ),
        other => Err(format!(
            "unknown phone action '{other}'; use alarm, note, location, now, \
             calendar_read, calendar_add, or sms"
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
        "now" => {
            let data = result.get("data").cloned().unwrap_or(Value::Null);
            let local = data.get("local").and_then(Value::as_str).unwrap_or("");
            let timezone = data.get("timezone").and_then(Value::as_str).unwrap_or("");
            let weekday = data.get("weekday").and_then(Value::as_str).unwrap_or("");
            if local.is_empty() {
                return Err("the phone returned no local time".to_string());
            }
            Ok(format!(
                "Phone local time: {local} ({weekday}, timezone {timezone}). \
                 Use YYYY-MM-DDTHH:MM in this timezone for 'begin_local'."
            ))
        }
        "calendar_read" => {
            let data = result.get("data").cloned().unwrap_or(Value::Null);
            let timezone = data.get("timezone").and_then(Value::as_str).unwrap_or("");
            let events = data
                .get("events")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if events.is_empty() {
                return Ok(format!(
                    "No calendar events in the requested window (phone timezone {timezone})."
                ));
            }
            let mut lines = Vec::with_capacity(events.len() + 1);
            lines.push(format!(
                "Calendar events (phone timezone {timezone}), {} found:",
                events.len()
            ));
            for event in events {
                let begin = event.get("begin_local").and_then(Value::as_str).unwrap_or("");
                let end = event.get("end_local").and_then(Value::as_str).unwrap_or("");
                let title = event
                    .get("title")
                    .and_then(Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                    .unwrap_or("(untitled)");
                let place = event.get("location").and_then(Value::as_str).unwrap_or("");
                let calendar = event
                    .get("calendar")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let mut line = if end.is_empty() {
                    format!("- {begin} · {title}")
                } else {
                    format!("- {begin}–{end} · {title}")
                };
                if !place.trim().is_empty() {
                    line.push_str(&format!(" @ {place}"));
                }
                if !calendar.trim().is_empty() {
                    line.push_str(&format!(" [{calendar}]"));
                }
                lines.push(line);
            }
            Ok(lines.join("\n"))
        }
        "calendar_add" => {
            let data = result.get("data").cloned().unwrap_or(Value::Null);
            let begin = data.get("begin_local").and_then(Value::as_str).unwrap_or("");
            let calendar = data
                .get("calendar")
                .and_then(Value::as_str)
                .unwrap_or("");
            let title = spec["title"].as_str().unwrap_or("");
            Ok(format!(
                "Created calendar event \"{title}\" at {begin} on the phone's {calendar} \
                 calendar."
            ))
        }
        "sms" => {
            let data = result.get("data").cloned().unwrap_or(Value::Null);
            let name = data.get("contact").and_then(Value::as_str).unwrap_or("");
            let number = data.get("number").and_then(Value::as_str).unwrap_or("");
            let target = match (name.is_empty(), number.is_empty()) {
                (false, false) => format!("{name} ({number})"),
                (false, true) => name.to_string(),
                (true, false) => number.to_string(),
                (true, true) => "the chosen contact".to_string(),
            };
            Ok(format!(
                "Opened the SMS composer for {target} with the text pre-filled. The user \
                 still has to press send; the message was not sent."
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

    #[test]
    fn calendar_actions_validate_bounds_and_keep_only_known_fields() {
        let read = build_phone_spec(&json!({
            "action": "calendar_read",
            "days": 3,
            "max": 5,
            "uri": "content://contacts",
        }))
        .expect("valid calendar_read");
        assert_eq!(read, json!({"action": "calendar_read", "days": 3, "max": 5}));
        assert!(build_phone_spec(&json!({"action": "calendar_read", "days": 31})).is_err());
        assert!(build_phone_spec(&json!({"action": "calendar_read", "max": 0})).is_err());

        let add = build_phone_spec(&json!({
            "action": "calendar_add",
            "title": "客户会议",
            "begin_local": "2026-10-21T15:00",
            "duration_min": 45,
            "location": "会议室 A",
            "description": "带合同",
            "calendar_id": 7,
        }))
        .expect("valid calendar_add");
        assert_eq!(add, json!({
            "action": "calendar_add",
            "title": "客户会议",
            "begin_local": "2026-10-21T15:00",
            "duration_min": 45,
            "location": "会议室 A",
            "description": "带合同",
        }));
        // Absolute timestamps and near-miss formats must not reach the phone:
        // the Android side would silently reinterpret them in local time.
        for bad in [
            "2026-10-21 15:00",
            "2026-10-21T15:00:00",
            "2026-13-21T15:00",
            "2026-10-21T24:00",
            "1761dummy",
        ] {
            assert!(
                build_phone_spec(&json!({
                    "action": "calendar_add",
                    "title": "x",
                    "begin_local": bad,
                }))
                .is_err(),
                "{bad} should be rejected"
            );
        }
        assert!(build_phone_spec(&json!({
            "action": "calendar_add",
            "begin_local": "2026-10-21T15:00",
        }))
        .is_err());
    }

    #[test]
    fn sms_requires_a_target_and_rejects_malformed_numbers() {
        let by_name = build_phone_spec(&json!({
            "action": "sms",
            "contact": "张三",
            "text": "马上到",
        }))
        .expect("valid sms by name");
        assert_eq!(by_name["contact"], "张三");
        assert_eq!(by_name["number"], "");

        let by_number = build_phone_spec(&json!({
            "action": "sms",
            "number": "+86 138-0000 0000",
            "text": "hi",
        }))
        .expect("valid sms by number");
        assert_eq!(by_number["number"], "+86 138-0000 0000");

        assert!(build_phone_spec(&json!({"action": "sms", "text": "hi"})).is_err());
        assert!(build_phone_spec(&json!({"action": "sms", "number": "1", "text": " "})).is_err());
        // Intent smuggling through the number field is rejected, not escaped.
        assert!(build_phone_spec(&json!({
            "action": "sms",
            "number": "123;tel:999",
            "text": "hi",
        }))
        .is_err());
        assert!(build_phone_spec(&json!({
            "action": "sms",
            "number": "not-a-number",
            "text": "hi",
        }))
        .is_err());
    }
}