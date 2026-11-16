//! Shared parameter and JavaScript contracts for both drivers.

use mcp_stdio::JsonRpcErr;
use serde_json::{Value, json};

fn invalid(message: &str) -> JsonRpcErr {
    JsonRpcErr::new(-32602, message, None)
}

/// Validate against the published schemas before launching or touching a page.
pub(crate) fn parse_call(params: Option<Value>) -> Result<(String, Value), JsonRpcErr> {
    let params = params.ok_or_else(|| invalid("tools/call params must be an object"))?;
    if !params.is_object() {
        return Err(invalid("tools/call params must be an object"));
    }
    let name = require_str(&params, "name")?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    if !args.is_object() {
        return Err(invalid("arguments must be an object"));
    }
    let registry = crate::tools::tools_list_result();
    let tool = registry["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == name)
        .ok_or_else(|| JsonRpcErr::new(-32601, "Unknown tool", Some(json!({"tool": name}))))?;
    let schema = &tool["inputSchema"];
    if let Some(required) = schema["required"].as_array() {
        for key in required.iter().filter_map(Value::as_str) {
            if args.get(key).is_none() {
                return Err(invalid(&format!("missing '{key}'")));
            }
        }
    }
    for (key, value) in args.as_object().unwrap() {
        let spec = schema["properties"]
            .get(key)
            .ok_or_else(|| invalid(&format!("unknown parameter '{key}'")))?;
        let valid = match spec["type"].as_str() {
            Some("string") => value
                .as_str()
                .is_some_and(|text| key == "text" || !text.trim().is_empty()),
            Some("boolean") => value.is_boolean(),
            Some("integer") => value.as_u64().is_some(),
            Some("number") => value.as_f64().is_some_and(|n| n.is_finite()),
            _ => false,
        };
        if !valid {
            return Err(invalid(&format!(
                "'{key}' must be a valid {}{}",
                spec["type"].as_str().unwrap_or("value"),
                if key == "text" {
                    ""
                } else {
                    " (nonempty/nonnegative where applicable)"
                }
            )));
        }
        if let Some(min) = spec["minimum"].as_f64() {
            if value.as_f64().is_some_and(|n| n < min) {
                return Err(invalid(&format!("'{key}' must be >= {min}")));
            }
        }
        if let Some(max) = spec["maximum"].as_f64() {
            if value.as_f64().is_some_and(|n| n > max) {
                return Err(invalid(&format!("'{key}' must be <= {max}")));
            }
        }
        if let Some(choices) = spec["enum"].as_array() {
            if !choices.contains(value) {
                return Err(invalid(&format!(
                    "invalid '{key}'; expected one of {choices:?}"
                )));
            }
        }
    }
    if matches!(name.as_str(), "click" | "type_text") {
        if args.get("selector").is_some() == args.get("ref").is_some() {
            return Err(invalid("supply exactly one of 'selector' or 'ref'"));
        }
    }
    if name == "wait_for" {
        parse_wait_target(&args)?;
    }
    Ok((name, args))
}

pub(crate) fn require_str(args: &Value, key: &str) -> Result<String, JsonRpcErr> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| invalid(&format!("missing, wrong type, or empty '{key}'")))
}

pub(crate) fn opt_str(args: &Value, key: &str) -> Option<String> {
    // parse_call validates every supplied option before this accessor is used.
    args.get(key).and_then(Value::as_str).map(str::to_string)
}

pub(crate) fn evaluation_script(expression: &str) -> String {
    format!(
        "(() => {{ try {{ const value = (0,eval)({}); return JSON.stringify({{ok:true,value: value === undefined ? null : value}}); }} catch(e) {{ return JSON.stringify({{ok:false,error:String(e)}}); }} }})()",
        json!(expression)
    )
}

pub(crate) fn decode_evaluation(raw: &str) -> Result<String, String> {
    let result: Value =
        serde_json::from_str(raw).map_err(|e| format!("invalid JavaScript result: {e}"))?;
    if result["ok"] != true {
        return Err(format!(
            "JavaScript error: {}",
            result["error"].as_str().unwrap_or("unknown exception")
        ));
    }
    Ok(result.get("value").unwrap_or(&Value::Null).to_string())
}

pub(crate) fn wait_script(selector: &str, condition: &str) -> String {
    format!(
        "(() => {{ const el = document.querySelector({}); const condition = {}; if(condition === 'absent') return !el; if(condition === 'present') return !!el; if(!el) return false; const rect = el.getBoundingClientRect(); const style = getComputedStyle(el); return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden' && style.visibility !== 'collapse'; }})()",
        json!(selector),
        json!(condition)
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WaitTarget {
    Selector { selector: String, condition: String },
    Url(String),
}

pub(crate) fn parse_wait_target(args: &Value) -> Result<WaitTarget, JsonRpcErr> {
    if args.get("selector").is_some() == args.get("url").is_some() {
        return Err(invalid(
            "wait_for requires exactly one of 'selector' or 'url'",
        ));
    }
    if args.get("url").is_some() {
        if args.get("condition").is_some() {
            return Err(invalid(
                "'condition' applies only to selector waits; URL waits use exact matching",
            ));
        }
        return Ok(WaitTarget::Url(require_str(args, "url")?));
    }
    Ok(WaitTarget::Selector {
        selector: require_str(args, "selector")?,
        condition: opt_str(args, "condition").unwrap_or_else(|| "present".into()),
    })
}

impl WaitTarget {
    pub(crate) fn script(&self) -> String {
        let predicate = match self {
            Self::Selector {
                selector,
                condition,
            } => wait_script(selector, condition),
            Self::Url(url) => format!("location.href === {}", json!(url)),
        };
        format!(
            "(() => {{ const matched = {predicate}; return {{matched,current_url:location.href}}; }})()"
        )
    }

    pub(crate) fn description(&self) -> String {
        match self {
            Self::Selector {
                selector,
                condition,
            } => format!("condition '{condition}' for '{selector}'"),
            Self::Url(url) => format!("exact URL '{url}'"),
        }
    }
}

/// Read the value before submission can navigate away. Passwords are compared
/// in Rust but their values are redacted from the returned evidence.
pub(crate) const TYPE_OBSERVATION_FUNCTION: &str = r#"function(){
    const root = this.getRootNode();
    return {connected:this.isConnected, editable:['INPUT','TEXTAREA'].includes(this.tagName),
        disabled:!!this.disabled, read_only:!!this.readOnly,
        focused:(root.activeElement || this.ownerDocument.activeElement) === this,
        input_type:this.type || '', value:typeof this.value === 'string' ? this.value : null};
}"#;

pub(crate) fn unknown_action(action: &str, reason: &str) -> Value {
    json!({"action":action,"status":"unknown","reason":reason})
}

pub(crate) fn type_verdict(expected: &str, observed: &Value) -> Value {
    let (status, reason) = if observed["connected"] == false {
        (
            "unknown",
            "target detached before its value could be confirmed",
        )
    } else if observed["editable"] == false
        || observed["disabled"] == true
        || observed["read_only"] == true
    {
        (
            "not_applied",
            "target is not an enabled editable input or textarea",
        )
    } else if let Some(value) = observed["value"].as_str() {
        if value == expected {
            (
                "applied",
                "read-back value exactly matches the requested text",
            )
        } else {
            (
                "not_applied",
                "read-back value differs; the page may have sanitized or changed it",
            )
        }
    } else {
        ("unknown", "input value could not be observed")
    };
    let value = observed["value"].as_str();
    let visible_value = if observed["input_type"] == "password" {
        json!("[redacted]")
    } else {
        value
            .map(|value| {
                let mut text: String = value.chars().take(256).collect();
                if value.chars().count() > 256 {
                    text.push_str(" [truncated]");
                }
                json!(text)
            })
            .unwrap_or(Value::Null)
    };
    json!({"action":"type_text","status":status,"reason":reason,"evidence":{
        "value_matches":value.map(|value| value == expected),
        "observed_value":visible_value,"observed_chars":value.map(|value| value.chars().count()),
        "focused":observed["focused"],"connected":observed["connected"]
    }})
}

pub(crate) const CLICK_OBSERVATION_FUNCTION: &str = r#"function(){
    const root = this.getRootNode();
    return {connected:this.isConnected, disabled:!!this.disabled || this.matches(':disabled') || this.getAttribute('aria-disabled') === 'true',
        focused:(root.activeElement || this.ownerDocument.activeElement) === this,
        checked:typeof this.checked === 'boolean' ? this.checked : null,
        aria_pressed:this.getAttribute('aria-pressed'),aria_expanded:this.getAttribute('aria-expanded'),
        text:(this.textContent || '').slice(0,300)};
}"#;

pub(crate) fn click_verdict(observed: &Value) -> Value {
    let before = &observed["before"];
    let after = &observed["after"];
    let changed = [
        "connected",
        "focused",
        "checked",
        "aria_pressed",
        "aria_expanded",
        "text",
    ]
    .into_iter()
    .any(|key| {
        before
            .get(key)
            .zip(after.get(key))
            .is_some_and(|(a, b)| a != b)
    });
    let url_changed = observed["url_before"]
        .as_str()
        .zip(observed["url_after"].as_str())
        .is_some_and(|(before, after)| before != after);
    let (status, reason) = if before["disabled"] == true {
        ("not_applied", "target is disabled")
    } else if before["connected"] == false {
        ("unknown", "target is detached")
    } else if observed["event_received"] == false {
        (
            "not_applied",
            "no click event reached the target; it may be obscured",
        )
    } else if observed["event_received"] == true && (changed || url_changed) {
        (
            "applied",
            "click reached the target and changed its state, focus, or URL",
        )
    } else if observed["event_received"] == true {
        (
            "unknown",
            "click reached the target but no observable state change was confirmed",
        )
    } else {
        ("unknown", "click target state could not be observed")
    };
    json!({"action":"click","status":status,"reason":reason,"evidence":observed})
}

pub(crate) fn click_script(selector: &str) -> String {
    format!(
        r#"(() => {{
        const el = document.querySelector({selector});
        if(!el) throw Error('click target not found');
        const observe = {observe};
        const before = observe.call(el);
        if(!before.connected || before.disabled) return {{before}};
        const url_before = location.href;
        let event = null;
        const mark = e => {{ event = e; }};
        el.addEventListener('click',mark,{{capture:true,once:true}});
        try {{
            el.scrollIntoView({{block:'center'}});
            el.dispatchEvent(new MouseEvent('mousedown',{{bubbles:true,cancelable:true}}));
            el.dispatchEvent(new MouseEvent('mouseup',{{bubbles:true,cancelable:true}}));
            el.click();
        }} finally {{ el.removeEventListener('click',mark,true); }}
        return {{before,after:observe.call(el),url_before,url_after:location.href,
            event_received:!!event,event_trusted:event ? event.isTrusted : null,
            default_prevented:event ? event.defaultPrevented : null}};
    }})()"#,
        selector = json!(selector),
        observe = CLICK_OBSERVATION_FUNCTION
    )
}

pub(crate) fn submission_summary(submission: &Value) -> Result<String, String> {
    if submission["method"] != "requestSubmit" || submission["submit_event"] != true {
        return Err("requestSubmit did not produce a form submit event".into());
    }
    Ok(format!(
        "form submitted via requestSubmit() (submit event observed; default_prevented={})",
        submission["default_prevented"]
    ))
}

pub(crate) const STATE_SCRIPT: &str = "JSON.stringify({current_url:location.href,readyState:document.readyState,scroll:{x:scrollX,y:scrollY}})";

pub(crate) fn extraction_script(args: &Value, html: bool) -> String {
    let selector = args["selector"]
        .as_str()
        .unwrap_or(if html { "html" } else { "body" });
    let offset = args["offset"].as_u64().unwrap_or(0);
    let max_chars = args["max_chars"].as_u64().unwrap_or(12_000);
    let property = if html { "outerHTML" } else { "innerText" };
    format!(
        "(() => {{ const el = document.querySelector({}); if(!el) throw Error('selector not found'); const chars = Array.from(el.{property} || ''); const offset = {offset}; const end = Math.min(chars.length, offset + {max_chars}); return JSON.stringify({{text:chars.slice(offset,end).join(''),total_chars:chars.length,offset:offset,next_offset:end < chars.length ? end : null}}); }})()",
        json!(selector)
    )
}

pub(crate) fn decode_extraction(raw: &str) -> Result<String, String> {
    let result: Value =
        serde_json::from_str(raw).map_err(|e| format!("invalid extraction result: {e}"))?;
    let text = result["text"].as_str().ok_or("missing extracted text")?;
    Ok(format!(
        "{text}\n[output offset={} total_chars={} next_offset={}]",
        result["offset"], result["total_chars"], result["next_offset"]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: Value) -> Result<(String, Value), JsonRpcErr> {
        parse_call(Some(json!({"name":name,"arguments":args})))
    }

    #[test]
    fn browser_fixes_validation_rejects_invalid_options() {
        for (name, args) in [
            (
                "type_text",
                json!({"selector":"input","text":"x","submit":"false"}),
            ),
            ("get_html", json!({"selector":false})),
            ("scroll", json!({"x":1.5})),
            ("scroll", json!({"y":-1})),
            ("wait_for", json!({"selector":"x","timeout_ms":0})),
            ("wait_for", json!({"selector":"x","condition":"bad"})),
            ("screenshot", json!({"full_page":1})),
            ("get_text", json!({"offset":null})),
            ("get_text", json!({"max_chars":0})),
            ("get_text", json!({"max_chars":20001})),
            ("navigate", json!({"url":" "})),
            ("click", json!({"selector":"a","ref":"x"})),
            ("click", json!({})),
        ] {
            assert_eq!(call(name, args).unwrap_err().code, -32602, "{name}");
        }
        assert!(call("type_text", json!({"selector":"input","text":""})).is_ok());
        assert!(call("click", json!({"ref":"target:12"})).is_ok());
        assert!(call("get_text", json!({"offset":0,"max_chars":100})).is_ok());
        assert!(parse_call(Some(json!({"name":"get_text","arguments":[]}))).is_err());
    }

    #[test]
    fn browser_fixes_json_and_paging_contract() {
        assert_eq!(
            decode_evaluation(r#"{"ok":true,"value":"line\ntext"}"#).unwrap(),
            r#""line\ntext""#
        );
        assert_eq!(
            decode_evaluation(r#"{"ok":true,"value":{"a":[1,true]}}"#).unwrap(),
            r#"{"a":[1,true]}"#
        );
        assert!(
            decode_evaluation(r#"{"ok":false,"error":"Error: boom"}"#)
                .unwrap_err()
                .contains("boom")
        );
        assert!(evaluation_script("'x' // comment\n").contains("\\n"));
        assert_eq!(
            decode_extraction(r#"{"text":"🙂\nhi","offset":0,"total_chars":9,"next_offset":4}"#)
                .unwrap(),
            "🙂\nhi\n[output offset=0 total_chars=9 next_offset=4]"
        );
    }

    #[test]
    fn browser_fixes_url_wait_validation_and_exact_matching() {
        let url = "https://example.com/page?q=中文#section";
        assert!(call("wait_for", json!({"url":url,"timeout_ms":1000})).is_ok());
        let target = parse_wait_target(&json!({"url":url})).ok().unwrap();
        assert_eq!(target, WaitTarget::Url(url.into()));
        assert!(
            target
                .script()
                .contains(&format!("location.href === {}", json!(url)))
        );
        assert!(target.script().contains("current_url:location.href"));
        assert_eq!(
            parse_wait_target(&json!({"selector":"#x","condition":"visible"}))
                .ok()
                .unwrap(),
            WaitTarget::Selector {
                selector: "#x".into(),
                condition: "visible".into()
            }
        );
        for args in [
            json!({}),
            json!({"url":url,"selector":"#x"}),
            json!({"url":url,"ref":"x"}),
            json!({"url":url,"condition":"present"}),
            json!({"url":true}),
            json!({"url":" "}),
        ] {
            assert_eq!(call("wait_for", args).unwrap_err().code, -32602);
        }
    }

    #[test]
    fn browser_fixes_typing_verdict_uses_read_back() {
        let mut observed = json!({"connected":true,"editable":true,"disabled":false,"read_only":false,
            "focused":true,"input_type":"text","value":"中文🙂"});
        assert_eq!(type_verdict("中文🙂", &observed)["status"], "applied");
        assert_eq!(
            type_verdict("中文🙂", &observed)["evidence"]["observed_value"],
            "中文🙂"
        );
        assert_eq!(
            type_verdict("中文🙂extra", &observed)["status"],
            "not_applied"
        );
        observed["value"] = json!("");
        assert_eq!(type_verdict("", &observed)["status"], "applied");
        observed["disabled"] = json!(true);
        assert_eq!(type_verdict("", &observed)["status"], "not_applied");
        observed["disabled"] = json!(false);
        observed["value"] = json!("secret");
        observed["input_type"] = json!("password");
        let verdict = type_verdict("secret", &observed);
        assert_eq!(verdict["status"], "applied");
        assert_eq!(verdict["evidence"]["observed_value"], "[redacted]");
        assert!(!verdict.to_string().contains("secret"));
        observed["connected"] = json!(false);
        assert_eq!(type_verdict("secret", &observed)["status"], "unknown");
        assert_eq!(type_verdict("x", &json!({}))["status"], "unknown");
    }

    #[test]
    fn browser_fixes_click_verdict_requires_observed_effect() {
        let before = json!({"connected":true,"disabled":false,"focused":false,"checked":false});
        let mut observed = json!({"before":before,"after":before,"event_received":true,
            "url_before":"https://example.com/","url_after":"https://example.com/"});
        assert_eq!(click_verdict(&observed)["status"], "unknown");
        observed["after"]["focused"] = json!(true);
        assert_eq!(click_verdict(&observed)["status"], "applied");
        observed["after"] = before.clone();
        observed["after"]["checked"] = json!(true);
        assert_eq!(click_verdict(&observed)["status"], "applied");
        observed["after"] = before;
        observed["url_after"] = json!("https://example.com/next");
        assert_eq!(click_verdict(&observed)["status"], "applied");
        observed["event_received"] = json!(false);
        assert_eq!(click_verdict(&observed)["status"], "not_applied");
        observed["before"]["disabled"] = json!(true);
        assert_eq!(click_verdict(&observed)["status"], "not_applied");
        assert_eq!(click_verdict(&Value::Null)["status"], "unknown");
    }

    #[test]
    fn browser_fixes_submit_summary_requires_submit_event() {
        let summary = submission_summary(
            &json!({"method":"requestSubmit","submit_event":true,"default_prevented":false}),
        )
        .unwrap();
        assert!(summary.contains("form submitted via requestSubmit()"));
        assert!(!summary.contains("pressed Enter"));
        assert!(
            submission_summary(&json!({"method":"requestSubmit","submit_event":false})).is_err()
        );
        assert!(submission_summary(&Value::Null).is_err());
    }
}
