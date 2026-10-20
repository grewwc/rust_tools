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

/// Character limits are separate from the serialized native FIFO byte budget.
const MAX_NOTE_CHARS: usize = 3_000;
const MAX_LABEL_CHARS: usize = 120;
const MAX_TITLE_CHARS: usize = 200;
const MAX_PLACE_CHARS: usize = 200;
const MAX_DESCRIPTION_CHARS: usize = 1_000;
const MAX_SMS_CHARS: usize = 1_000;
const MAX_CONTACT_CHARS: usize = 120;
const MAX_NUMBER_CHARS: usize = 40;
// A full page of 20 notes, including JSON escaping, fits without losing bodies.
const MAX_RESULT_BYTES: usize = 512 * 1024;
const MAX_FILE_CHARS: i64 = 1_048_576;
const PHONE_ACTIONS: &str = "now, calendar_read, calendar_add, sms, alarm, timer, note, \
    location, note_create, note_list, note_update, file_import, file_read, file_save, screen_capture, screen_stop";

fn string_arg<'a>(args: &'a Value, key: &str, max: usize) -> Result<Option<&'a str>, String> {
    let Some(value) = args.get(key) else {
        return Ok(None);
    };
    let text = value.as_str().ok_or_else(|| format!("'{key}' must be a string"))?;
    if text.chars().count() > max {
        return Err(format!("'{key}' must be at most {max} characters"));
    }
    Ok(Some(text))
}

fn required_text(args: &Value) -> Result<&str, String> {
    string_arg(args, "text", MAX_NOTE_CHARS)?
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| "'text' must contain nonempty text".to_string())
}

fn integer_arg(args: &Value, key: &str, default: Option<i64>, min: i64, max: i64) -> Result<i64, String> {
    let value = match args.get(key) {
        None => default,
        Some(value) => value.as_i64(),
    };
    value.filter(|n| (min..=max).contains(n))
        .ok_or_else(|| format!("'{key}' must be an integer from {min} to {max}"))
}

fn note_kind(args: &Value) -> Result<Option<&str>, String> {
    match string_arg(args, "kind", 4)? {
        kind @ (None | Some("note" | "todo")) => Ok(kind),
        _ => Err("'kind' must be 'note' or 'todo'".to_string()),
    }
}

fn uuid_arg(args: &Value, key: &str) -> Result<String, String> {
    let text = string_arg(args, key, 36)?.unwrap_or("");
    let id = uuid::Uuid::parse_str(text).ok().filter(|_| {
        text.len() == 36 && [8, 13, 18, 23].iter().all(|&i| text.as_bytes()[i] == b'-')
    }).ok_or_else(|| format!("'{key}' must be a UUID in 8-4-4-4-12 form"))?;
    Ok(id.to_string())
}

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
    let spec = match action {
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
            let mut spec = json!({
                "action": "alarm",
                "hour": hour,
                "minute": minute,
                "label": string_arg(args, "label", MAX_LABEL_CHARS)?.unwrap_or(""),
            });
            if let Some(value) = args.get("weekdays") {
                let days = value.as_array().ok_or("'weekdays' must be an array of unique ISO weekdays 1-7")?;
                let mut seen = [false; 7];
                for day in days {
                    let day = day.as_i64().filter(|n| (1..=7).contains(n))
                        .ok_or("'weekdays' must contain only integers from 1 (Monday) to 7 (Sunday)")?;
                    if std::mem::replace(&mut seen[day as usize - 1], true) {
                        return Err("'weekdays' must not contain duplicates".to_string());
                    }
                }
                spec["weekdays"] = value.clone();
            }
            Ok(spec)
        }
        "timer" => Ok(json!({
            "action": "timer",
            "seconds": integer_arg(args, "seconds", None, 1, 86_400)?,
            "label": string_arg(args, "label", MAX_LABEL_CHARS)?.unwrap_or(""),
        })),
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
        "note_create" => Ok(json!({
            "action": "note_create",
            "kind": note_kind(args)?.unwrap_or("note"),
            "title": string_arg(args, "title", MAX_TITLE_CHARS)?.unwrap_or(""),
            "text": required_text(args)?,
        })),
        "note_list" => {
            let mut spec = json!({
                "action": "note_list",
                "query": string_arg(args, "query", MAX_TITLE_CHARS)?.unwrap_or(""),
                "max": integer_arg(args, "max", Some(10), 1, 20)?,
                "offset": integer_arg(args, "offset", Some(0), 0, 100_000)?,
            });
            if let Some(kind) = note_kind(args)? {
                spec["kind"] = json!(kind);
            }
            Ok(spec)
        }
        "note_update" => {
            let mut spec = json!({"action": "note_update", "entry_id": uuid_arg(args, "entry_id")?});
            if let Some(title) = string_arg(args, "title", MAX_TITLE_CHARS)? {
                spec["title"] = json!(title);
            }
            if args.get("text").is_some() {
                spec["text"] = json!(required_text(args)?);
            }
            if let Some(completed) = args.get("completed") {
                spec["completed"] = json!(completed.as_bool().ok_or("'completed' must be a boolean")?);
            }
            if spec.as_object().map_or(0, |fields| fields.len()) == 2 {
                return Err("note_update requires at least one of title, text, or completed; completed applies only to todos".to_string());
            }
            Ok(spec)
        }
        "file_import" => Ok(json!({"action": "file_import"})),
        "screen_capture" | "screen_stop" => Ok(json!({"action": action})),
        "file_read" => Ok(json!({
            "action": "file_read",
            "file_id": uuid_arg(args, "file_id")?,
            "offset": integer_arg(args, "offset", Some(0), 0, MAX_FILE_CHARS)?,
            "limit": integer_arg(args, "limit", Some(3_000), 1, 3_000)?,
        })),
        "file_save" => {
            let filename = string_arg(args, "filename", MAX_LABEL_CHARS)?.unwrap_or("");
            if filename.trim().is_empty() || matches!(filename, "." | "..")
                || filename.chars().any(|c| c.is_control() || matches!(c, '/' | '\\')) {
                return Err("'filename' must be a nonempty basename, not '.' or '..', without slash, backslash, or control characters".to_string());
            }
            Ok(json!({"action": "file_save", "filename": filename, "text": required_text(args)?}))
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
        "" => Err(format!("'action' is required: {PHONE_ACTIONS}")),
        other => Err(format!(
            "unknown phone action '{other}'; use {PHONE_ACTIONS}"
        )),
    }?;
    validate_native_size(&spec)?;
    Ok(spec)
}

fn validate_native_size(spec: &Value) -> Result<(), String> {
    // Match serve_confirm::native's nested JSON, reserving the longest possible
    // request id. Raw character/byte counts alone miss both escaping layers.
    let prompt = format!("{}{spec}", crate::ai::serve_confirm::NATIVE_PREFIX);
    let payload = json!({"id": u64::MAX, "prompt": prompt}).to_string();
    let max = crate::ai::background::MAX_SERVE_FRAME_BYTES - 64;
    if payload.len() > max {
        return Err(format!(
            "Phone request needs {} serialized bytes; native FIFO limit is {max}. \
             Shorten the text or other fields and retry; nothing was sent or truncated.",
            payload.len()
        ));
    }
    Ok(())
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
    format_phone_result(&spec, &answer)
}

fn decode_screen_image(data: &Value) -> Result<Vec<u8>, String> {
    use base64::Engine;
    if data["mime_type"].as_str() != Some("image/jpeg") {
        return Err("phone screenshot must be a JPEG".to_string());
    }
    let encoded = data["image_base64"].as_str().ok_or("phone screenshot bytes missing")?;
    if encoded.len() > 350_000 {
        return Err("phone screenshot exceeds size limit".to_string());
    }
    let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)
        .map_err(|e| format!("invalid phone screenshot: {e}"))?;
    if bytes.len() > 256 * 1024 || !bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Err("invalid or oversized phone JPEG".to_string());
    }
    let dimensions = image::ImageReader::with_format(std::io::Cursor::new(&bytes), image::ImageFormat::Jpeg)
        .into_dimensions().map_err(|e| format!("invalid phone JPEG dimensions: {e}"))?;
    if dimensions.0 == 0 || dimensions.1 == 0 || dimensions.0 > 2048 || dimensions.1 > 2048 {
        return Err("phone screenshot dimensions exceed limit".to_string());
    }
    Ok(bytes)
}

fn format_phone_result(spec: &Value, answer: &str) -> Result<String, String> {
    if answer.len() > MAX_RESULT_BYTES {
        return Err(format!("Phone response exceeds the {MAX_RESULT_BYTES}-byte full-result limit; no partial result was returned. Use a smaller page for read actions; do not repeat a write whose outcome is unknown."));
    }
    let result: Value = serde_json::from_str(answer)
        .map_err(|err| format!("invalid response from the phone client: {err}"))?;
    if result.get("ok").and_then(Value::as_bool) != Some(true) {
        let error = result
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("phone action failed");
        return Err(format!("Phone action failed: {error}"));
    }
    let data = result.get("data").unwrap_or(&Value::Null);
    let action = spec["action"].as_str().unwrap_or("");
    if action == "screen_capture" {
        let bytes = decode_screen_image(data)?;
        let directory = crate::ai::driver::runtime_ctx::temp_dir().map_err(|e| format!("cannot prepare screenshot directory: {e}"))?;
        let path = directory.join(format!("phone-screen-{}.jpg", uuid::Uuid::new_v4()));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        use std::io::Write;
        let mut file = options.open(&path).map_err(|e| format!("cannot save phone screenshot: {e}"))?;
        file.write_all(&bytes).map_err(|e| format!("cannot save phone screenshot: {e}"))?;
        return Ok(format!("Phone screenshot saved to {}. Use read_file on this path to inspect the image before answering. The user chose the capture moment from the a serve notification. This is a captured frame, not app history; secure windows may be blank. Sharing remains visible until stopped, and can be stopped with screen_stop.", path.display()));
    }
    if matches!(action, "note_create" | "note_list" | "note_update" | "file_import" | "file_read" | "file_save") && !data.is_object() {
        return Err("the phone returned no result data; the action outcome is unknown".to_string());
    }
    let summary: Result<String, String> = match action {
        "screen_stop" => Ok("Phone screen sharing stopped; another session requires system consent.".to_string()),
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
                "Dispatched the alarm request for {hour:02}:{minute:02}{suffix} to the phone's clock app. Creation is not verified; check the clock app. Without weekdays (or with []), this requests only the next occurrence, not a date."
            ))
        }
        "timer" => Ok(format!(
            "Dispatched the {}-second timer request to the phone's clock app. Creation is not verified; check the clock app.",
            spec["seconds"]
        )),
        "note_create" | "note_update" => Ok(
            "Phone-local entry result. A todo is stored text/completion state, not a scheduled task or reminder.".to_string()
        ),
        "note_list" => Ok("Phone-local entries (full returned page, not scheduled reminders):".to_string()),
        "file_import" | "file_read" => Ok(
            "Phone UTF-8 text file page. Offsets count Unicode codepoints; when has_more is true, call file_read with file_id and next_offset.".to_string()
        ),
        "file_save" => Ok("Phone UTF-8 document save result after the system create-document picker:".to_string()),
        "note" => Ok(
            "Opened the phone's share sheet with the note text. The user still chooses \
             the notes app and saves it."
                .to_string(),
        ),
        "location" => {
            let data = result.get("data").cloned().unwrap_or(Value::Null);
            data
                .get("latitude")
                .and_then(Value::as_f64)
                .ok_or_else(|| "the phone returned no latitude".to_string())?;
            data
                .get("longitude")
                .and_then(Value::as_f64)
                .ok_or_else(|| "the phone returned no longitude".to_string())?;
            Ok("Phone location (Android-reported values below): accuracy is in meters and age_ms in milliseconds; provider and precision describe the actual fix/permission. Missing fields are unknown, not zero or precise.".to_string())
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
    };
    // Preserve every returned field and body. A summary is never a substitute
    // for the bounded full JSON data, especially for paginated notes/files.
    Ok(format!("{}\nPhone result data (JSON):\n{data}", summary?))
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
    use super::*;

    const ID: &str = "12345678-1234-4abc-8def-123456789abc";

    #[test]
    fn screen_capture_validates_jpeg_and_exposes_no_silent_consent_override() {
        use base64::Engine;
        assert_eq!(build_phone_spec(&json!({"action":"screen_capture", "silent":true, "path":"/private"})).unwrap(), json!({"action":"screen_capture"}));
        assert_eq!(build_phone_spec(&json!({"action":"screen_stop"})).unwrap(), json!({"action":"screen_stop"}));
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 3).write_to(&mut jpeg, image::ImageFormat::Jpeg).unwrap();
        let encoded = base64::engine::general_purpose::STANDARD.encode(jpeg.get_ref());
        assert_eq!(decode_screen_image(&json!({"mime_type":"image/jpeg", "image_base64":encoded})).unwrap(), *jpeg.get_ref());
        for data in [json!({}), json!({"mime_type":"image/png", "image_base64":encoded}),
            json!({"mime_type":"image/jpeg", "image_base64":"invalid"}),
            json!({"mime_type":"image/jpeg", "image_base64":"a".repeat(350_001)})] {
            assert!(decode_screen_image(&data).is_err());
        }
    }

    #[test]
    fn timer_and_alarm_weekdays_require_strict_types_and_ranges() {
        for seconds in [1, 86_400] {
            assert_eq!(build_phone_spec(&json!({"action": "timer", "seconds": seconds})).unwrap(),
                json!({"action": "timer", "seconds": seconds, "label": ""}));
        }
        for seconds in [json!(0), json!(86_401), json!(-1), json!(1.5), json!("1"), json!(null), json!(true)] {
            assert!(build_phone_spec(&json!({"action": "timer", "seconds": seconds})).is_err());
        }
        assert!(build_phone_spec(&json!({"action": "timer"})).is_err());
        for label in [json!("x".repeat(121)), json!(false), json!(null)] {
            assert!(build_phone_spec(&json!({"action": "timer", "seconds": 1, "label": label})).is_err());
        }
        assert!(build_phone_spec(&json!({"action": "timer", "seconds": 1, "label": "😀".repeat(120)})).is_ok());
        for weekdays in [json!([]), json!([1, 7]), json!([7, 6, 5, 4, 3, 2, 1])] {
            let spec = build_phone_spec(&json!({"action": "alarm", "hour": 9, "minute": 0, "weekdays": weekdays, "date": "2026-10-21"})).unwrap();
            assert_eq!(spec["weekdays"], weekdays);
            assert!(spec.get("date").is_none());
        }
        for weekdays in [json!(null), json!("1,2"), json!(1), json!([0]), json!([8]), json!([1, 1]), json!([1.0]), json!(["1"]), json!([true])] {
            assert!(build_phone_spec(&json!({"action": "alarm", "hour": 9, "minute": 0, "weekdays": weekdays})).is_err());
        }
    }

    #[test]
    fn stored_notes_preserve_bodies_and_validate_updates() {
        let text = "  标题😀\ncontent\n ";
        assert_eq!(build_phone_spec(&json!({"action": "note_create", "text": text})).unwrap(),
            json!({"action": "note_create", "kind": "note", "title": "", "text": text}));
        assert_eq!(build_phone_spec(&json!({"action": "note_create", "kind": "todo", "title": "x", "text": text})).unwrap()["kind"], "todo");
        let update = build_phone_spec(&json!({"action": "note_update", "entry_id": ID.to_uppercase(), "title": "", "text": text, "completed": false, "schedule": "tomorrow"})).unwrap();
        assert_eq!(update, json!({"action": "note_update", "entry_id": ID, "title": "", "text": text, "completed": false}));
        assert!(build_phone_spec(&json!({"action": "note_update", "entry_id": ID, "completed": true})).is_ok());
        assert!(build_phone_spec(&json!({"action": "note_update", "entry_id": ID, "kind": "todo"})).is_err());
        for fields in [json!({}), json!({"text": " "}), json!({"text": 1}), json!({"text": null}), json!({"text": "x".repeat(3001)}), json!({"text": "ok", "title": "x".repeat(201)}), json!({"text": "ok", "title": false}), json!({"text": "ok", "kind": "task"}), json!({"text": "ok", "kind": null})] {
            let mut args = fields;
            args["action"] = json!("note_create");
            assert!(build_phone_spec(&args).is_err(), "{args}");
        }
        for fields in [json!({"text": " "}), json!({"text": null}), json!({"title": null}), json!({"completed": "false"}), json!({"completed": 0}), json!({"completed": null})] {
            let mut args = fields;
            args["action"] = json!("note_update");
            args["entry_id"] = json!(ID);
            assert!(build_phone_spec(&args).is_err(), "{args}");
        }
    }

    #[test]
    fn note_list_preserves_all_filters_and_pagination() {
        assert_eq!(build_phone_spec(&json!({"action": "note_list"})).unwrap(),
            json!({"action": "note_list", "query": "", "max": 10, "offset": 0}));
        assert_eq!(build_phone_spec(&json!({"action": "note_list", "query": "查找😀", "kind": "todo", "max": 20, "offset": 100_000, "completed": true})).unwrap(),
            json!({"action": "note_list", "query": "查找😀", "kind": "todo", "max": 20, "offset": 100_000}));
        for fields in [json!({"query": "x".repeat(201)}), json!({"query": 1}), json!({"query": null}), json!({"max": 0}), json!({"max": 21}), json!({"max": "10"}), json!({"max": null}), json!({"offset": -1}), json!({"offset": 100_001}), json!({"offset": 1.0}), json!({"offset": "0"}), json!({"kind": "reminder"})] {
            let mut args = fields;
            args["action"] = json!("note_list");
            assert!(build_phone_spec(&args).is_err(), "{args}");
        }
    }

    #[test]
    fn files_validate_ids_names_text_and_codepoint_page_bounds() {
        assert_eq!(build_phone_spec(&json!({"action": "file_import", "path": "/private"})).unwrap(), json!({"action": "file_import"}));
        assert_eq!(build_phone_spec(&json!({"action": "file_read", "file_id": ID})).unwrap(),
            json!({"action": "file_read", "file_id": ID, "offset": 0, "limit": 3000}));
        assert!(build_phone_spec(&json!({"action": "file_read", "file_id": ID, "offset": 1_048_576, "limit": 1})).is_ok());
        for fields in [json!({"offset": -1}), json!({"offset": 1_048_577}), json!({"offset": "0"}), json!({"offset": null}), json!({"limit": 0}), json!({"limit": 3001}), json!({"limit": 1.5}), json!({"limit": true})] {
            let mut args = fields;
            args["action"] = json!("file_read");
            args["file_id"] = json!(ID);
            assert!(build_phone_spec(&args).is_err(), "{args}");
        }
        for id in [json!(null), json!(123), json!("../x"), json!("1234567812344abc8def123456789abc"), json!("12345678-1234-4abc-8def-123456789abz"), json!(format!("{{{ID}}}")), json!(format!(" {ID}"))] {
            assert!(build_phone_spec(&json!({"action": "file_read", "file_id": id})).is_err());
            assert!(build_phone_spec(&json!({"action": "note_update", "entry_id": id, "completed": true})).is_err());
        }
        assert!(build_phone_spec(&json!({"action": "file_read"})).is_err());
        assert!(build_phone_spec(&json!({"action": "note_update", "text": "ok"})).is_err());
        for filename in [json!(""), json!(" "), json!("."), json!(".."), json!("a/b"), json!("a\\b"), json!("a\nb"), json!("a\u{007f}b"), json!("a\u{0085}b"), json!("x".repeat(121)), json!(false), json!(null)] {
            assert!(build_phone_spec(&json!({"action": "file_save", "filename": filename, "text": "body"})).is_err(), "{filename}");
        }
        let text = "\n 中文😀 \n";
        assert_eq!(build_phone_spec(&json!({"action": "file_save", "filename": "笔记😀.txt", "text": text})).unwrap(),
            json!({"action": "file_save", "filename": "笔记😀.txt", "text": text}));
        for text in [json!(null), json!(0), json!(" "), json!("x".repeat(3001))] {
            assert!(build_phone_spec(&json!({"action": "file_save", "filename": "a.txt", "text": text})).is_err());
        }
    }

    #[test]
    fn native_request_budget_counts_utf8_and_both_json_escaping_layers() {
        let max = crate::ai::background::MAX_SERVE_FRAME_BYTES - 64;
        let empty = json!({"text": ""});
        let overhead = json!({"id": u64::MAX, "prompt": format!("{}{empty}", crate::ai::serve_confirm::NATIVE_PREFIX)}).to_string().len();
        assert!(validate_native_size(&json!({"text": "x".repeat(max - overhead)})).is_ok());
        assert!(validate_native_size(&json!({"text": "x".repeat(max - overhead + 1)})).is_err());
        assert_eq!(build_phone_spec(&json!({"action": "file_save", "filename": "a", "text": "x".repeat(3000)})).unwrap()["text"].as_str().unwrap().len(), 3000);
        for text in ["中".repeat(1500), "😀".repeat(1000), "\"".repeat(1300), "\u{0001}".repeat(600)] {
            for action in ["note_create", "note_update", "file_save"] {
                let args = json!({"action": action, "filename": "a", "entry_id": ID, "text": text});
                let error = execute_phone_action(&args).unwrap_err();
                assert!(error.contains("native FIFO limit"), "{action}: {error}");
                assert!(error.contains("nothing was sent or truncated"));
            }
        }
    }

    fn result_data(output: &str) -> Value {
        serde_json::from_str(output.split_once("Phone result data (JSON):\n").unwrap().1).unwrap()
    }

    #[test]
    fn phone_results_keep_full_pages_and_actual_location_metadata() {
        let entries: Vec<Value> = (0..20).map(|n| json!({"entry_id": ID, "title": format!("entry {n}"), "text": "\u{0001}".repeat(2999) + "😀", "completed": false})).collect();
        let data = json!({"entries": entries, "offset": 20, "next_offset": 40, "has_more": true});
        let output = format_phone_result(&json!({"action": "note_list"}), &json!({"ok": true, "data": data}).to_string()).unwrap();
        assert_eq!(result_data(&output), data);
        let page = json!({"file_id": ID, "filename": "a.txt", "mime_type": "text/plain", "total_chars": 8, "text": "中😀\n", "next_offset": 3, "has_more": true});
        for action in ["file_import", "file_read"] {
            let output = format_phone_result(&json!({"action": action}), &json!({"ok": true, "data": page}).to_string()).unwrap();
            assert_eq!(result_data(&output), page);
        }
        for data in [json!({"latitude": 1.12345678, "longitude": 2.12345678, "accuracy": 351.2, "age_ms": 400, "provider": "network", "precision": "approximate"}), json!({"latitude": 1, "longitude": 2})] {
            let output = format_phone_result(&json!({"action": "location"}), &json!({"ok": true, "data": data}).to_string()).unwrap();
            assert_eq!(result_data(&output), data);
            assert!(!output.contains("accuracy about 0"));
        }
        assert!(format_phone_result(&json!({"action": "file_read"}), &json!({"ok": true, "data": {"text": "x".repeat(MAX_RESULT_BYTES)}}).to_string()).unwrap_err().contains("no partial result"));
        assert!(format_phone_result(&json!({"action": "file_read"}), r#"{"ok":true}"#).is_err());
        assert!(format_phone_result(&json!({"action": "file_read"}), r#"{"ok":false,"error":"cancelled"}"#).unwrap_err().contains("cancelled"));
    }

    #[test]
    fn clock_share_and_sms_results_do_not_overclaim_side_effects() {
        for action in ["alarm", "timer"] {
            let output = format_phone_result(&json!({"action": action, "seconds": 60, "hour": 9, "minute": 0}), r#"{"ok":true,"data":{"dispatched":true}}"#).unwrap();
            assert!(output.contains("Creation is not verified"));
            assert_eq!(result_data(&output), json!({"dispatched": true}));
        }
        let note = format_phone_result(&json!({"action": "note"}), r#"{"ok":true}"#).unwrap();
        assert!(note.contains("user still chooses"));
        let sms = format_phone_result(&json!({"action": "sms"}), r#"{"ok":true}"#).unwrap();
        assert!(sms.contains("message was not sent"));
    }

    #[test]
    fn phone_description_and_backend_actions_stay_in_sync() {
        let description: Value = serde_json::from_str(include_str!("../tool_descriptions/phone_action.json")).unwrap();
        assert_eq!(description["groups"], json!(["phone"]));
        let fixtures = [
            json!({"action": "now"}),
            json!({"action": "calendar_read"}),
            json!({"action": "calendar_add", "title": "x", "begin_local": "2026-10-21T15:00"}),
            json!({"action": "sms", "number": "123", "text": "hi"}),
            json!({"action": "alarm", "hour": 9, "minute": 0}),
            json!({"action": "timer", "seconds": 1}),
            json!({"action": "note", "text": "body"}),
            json!({"action": "location"}),
            json!({"action": "note_create", "text": "body"}),
            json!({"action": "note_list"}),
            json!({"action": "note_update", "entry_id": ID, "completed": false}),
            json!({"action": "file_import"}),
            json!({"action": "file_read", "file_id": ID}),
            json!({"action": "file_save", "filename": "a.txt", "text": "body"}),
            json!({"action": "screen_capture"}),
            json!({"action": "screen_stop"}),
        ];
        let mut implemented: Vec<&str> = fixtures.iter().map(|args| {
            build_phone_spec(args).expect("advertised action must parse");
            args["action"].as_str().unwrap()
        }).collect();
        let mut advertised: Vec<&str> = description["parameters"]["properties"]["action"]["enum"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        implemented.sort_unstable();
        advertised.sort_unstable();
        assert_eq!(implemented, advertised);
    }

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