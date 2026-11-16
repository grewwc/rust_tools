//! Native accessibility snapshots and DOM references scoped to the active target.

use crate::contract::{
    CLICK_OBSERVATION_FUNCTION, TYPE_OBSERVATION_FUNCTION, click_verdict, type_verdict,
    unknown_action,
};
use chromiumoxide::cdp::browser_protocol::accessibility::{EnableParams, GetFullAxTreeParams};
use chromiumoxide::cdp::browser_protocol::dom::{GetContentQuadsParams, ResolveNodeParams};
use chromiumoxide::cdp::browser_protocol::input::InsertTextParams;
use chromiumoxide::cdp::js_protocol::runtime::{CallFunctionOnParams, RemoteObjectId};
use chromiumoxide::layout::Point;
use serde_json::{Value, json};

use crate::browser::BrowserSession;

pub(crate) async fn snapshot(
    s: &mut BrowserSession,
    offset: usize,
    limit: usize,
) -> Result<String, String> {
    s.page
        .execute(EnableParams::default())
        .await
        .map_err(|e| format!("accessibility enable failed: {e}"))?;
    let tree = s
        .page
        .execute(GetFullAxTreeParams::default())
        .await
        .map_err(|e| format!("accessibility snapshot failed: {e}"))?
        .result;
    let target = s.page.target_id().as_ref();
    let mut elements = Vec::new();
    for node in tree.nodes.into_iter().filter(|node| !node.ignored) {
        let Some(backend) = node.backend_dom_node_id else {
            continue;
        };
        let id = format!("{target}:{}", backend.inner());
        if !s.snapshot_refs.iter().any(|(known, _)| known == &id) {
            s.snapshot_refs.push((id.clone(), backend));
        }
        elements.push(json!({
            "ref": id,
            "role": node.role.and_then(|v| v.value).unwrap_or(Value::Null),
            "name": node.name.and_then(|v| v.value).unwrap_or(Value::Null),
            "description": node.description.and_then(|v| v.value).unwrap_or(Value::Null),
        }));
    }
    Ok(snapshot_page(target, elements, offset, limit)?.to_string())
}

const SNAPSHOT_PAGE_CHARS: usize = 18_000;
const SNAPSHOT_FIELD_CHARS: usize = 512;

pub(crate) fn validate_payload(payload: &str) -> Result<(), String> {
    if payload.chars().count() > SNAPSHOT_PAGE_CHARS {
        return Err("snapshot payload exceeds the output budget".into());
    }
    Ok(())
}

fn snapshot_page(
    target: &str,
    elements: Vec<Value>,
    offset: usize,
    limit: usize,
) -> Result<Value, String> {
    if limit == 0 {
        return Err("snapshot limit must be positive".into());
    }
    let total = elements.len();
    // Budget the serialized root metadata even when no nodes will be selected.
    // Reserve the longest possible continuation so replacing it cannot grow
    // the final result after elements have been added.
    let mut page = json!({"tab_id":target,"elements":[],"total":total,"offset":offset,
        "next_offset":(offset < total).then_some(usize::MAX)});
    let mut chars = page.to_string().chars().count();
    if chars > SNAPSHOT_PAGE_CHARS {
        return Err("snapshot root identity exceeds the output budget".into());
    }
    let mut selected = Vec::new();
    for mut element in elements.into_iter().skip(offset).take(limit) {
        let mut truncated = Vec::new();
        for key in ["role", "name", "description"] {
            let Some(value) = element.get_mut(key) else {
                continue;
            };
            let raw = value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string());
            if raw.chars().count() > SNAPSHOT_FIELD_CHARS {
                let mut bounded: String = raw.chars().take(SNAPSHOT_FIELD_CHARS).collect();
                bounded.push_str(" [truncated]");
                *value = json!(bounded);
                truncated.push(key);
            }
        }
        if !truncated.is_empty() {
            element["truncated_fields"] = json!(truncated);
        }
        let size = element.to_string().chars().count() + usize::from(!selected.is_empty());
        if chars + size > SNAPSHOT_PAGE_CHARS {
            if selected.is_empty() {
                // Generated DOM identities are small, but a malformed external
                // target id must produce an error instead of stalled pagination.
                return Err("snapshot node identity exceeds the output budget".into());
            }
            break;
        }
        chars += size;
        selected.push(element);
    }
    let next = offset.saturating_add(selected.len());
    page["elements"] = json!(selected);
    page["next_offset"] = json!((next < total).then_some(next));
    validate_payload(&page.to_string())?;
    Ok(page)
}

pub(crate) async fn resolve_target(
    s: &BrowserSession,
    args: &Value,
) -> Result<RemoteObjectId, String> {
    if let Some(selector) = args["selector"].as_str() {
        return s
            .page
            .find_element(selector)
            .await
            .map(|e| e.remote_object_id)
            .map_err(|e| format!("element '{selector}' not found: {e}"));
    }
    let id = args["ref"]
        .as_str()
        .ok_or("missing element selector or ref")?;
    let (target, _) = id.rsplit_once(':').ok_or("invalid snapshot ref")?;
    if target != s.page.target_id().as_ref() {
        return Err("snapshot ref belongs to a different tab; switch_tab first".into());
    }
    let backend = s
        .snapshot_refs
        .iter()
        .find(|(known, _)| known == id)
        .map(|(_, backend)| *backend)
        .ok_or("unknown snapshot ref; take a snapshot first")?;
    s.page
        .execute(
            ResolveNodeParams::builder()
                .backend_node_id(backend)
                .build(),
        )
        .await
        .map_err(|e| format!("stale snapshot ref; take a new snapshot: {e}"))?
        .result
        .object
        .object_id
        .ok_or("snapshot ref has no DOM object".into())
}

pub(crate) async fn call_on(
    s: &BrowserSession,
    object: &RemoteObjectId,
    function: &str,
) -> Result<Value, String> {
    let response = s
        .page
        .execute(
            CallFunctionOnParams::builder()
                .object_id(object.clone())
                .function_declaration(function)
                .return_by_value(true)
                .await_promise(true)
                .build()
                .map_err(|e| e.to_string())?,
        )
        .await
        .map_err(|e| format!("element action failed: {e}"))?
        .result;
    if let Some(error) = response.exception_details {
        return Err(format!("element JavaScript error: {}", error.text));
    }
    Ok(response.result.value.unwrap_or(Value::Null))
}

pub(crate) async fn click(s: &BrowserSession, object: &RemoteObjectId) -> Result<Value, String> {
    let prepare = format!(
        r#"function(){{
        const before = ({observe}).call(this);
        if(!before.connected || before.disabled) return {{before}};
        this.scrollIntoView({{block:'center',inline:'center'}});
        const state = {{before,url_before:this.ownerDocument.location.href,event:null}};
        state.listener = e => {{ state.event = e; }};
        this.addEventListener('click',state.listener,{{capture:true,once:true}});
        this[Symbol.for('mcp_browser.click_observation')] = state;
        return {{before}};
    }}"#,
        observe = CLICK_OBSERVATION_FUNCTION
    );
    let before = call_on(s, object, &prepare).await?;
    if before["before"]["disabled"] == true || before["before"]["connected"] == false {
        return Ok(click_verdict(&before));
    }
    let action = async {
        let quads = s
            .page
            .execute(
                GetContentQuadsParams::builder()
                    .object_id(object.clone())
                    .build(),
            )
            .await
            .map_err(|e| format!("click geometry failed: {e}"))?
            .result
            .quads;
        let quad = quads
            .iter()
            .find(|quad| {
                let points = quad.inner();
                points.len() == 8
                    && (points[2] - points[0]).abs() + (points[3] - points[1]).abs() > 0.5
            })
            .ok_or("element has no visible click target")?
            .inner();
        let x = (quad[0] + quad[2] + quad[4] + quad[6]) / 4.0;
        let y = (quad[1] + quad[3] + quad[5] + quad[7]) / 4.0;
        s.page
            .click(Point { x, y })
            .await
            .map_err(|e| format!("click failed: {e}"))?;
        Ok::<_, String>(())
    }
    .await;
    let observe = format!(
        r#"function(){{
        const key = Symbol.for('mcp_browser.click_observation');
        const state = this[key];
        if(!state) return null;
        this.removeEventListener('click',state.listener,true);
        delete this[key];
        return {{before:state.before,after:({observe}).call(this),url_before:state.url_before,
            url_after:this.ownerDocument.location.href,event_received:!!state.event,
            event_trusted:state.event ? state.event.isTrusted : null,
            default_prevented:state.event ? state.event.defaultPrevented : null}};
    }}"#,
        observe = CLICK_OBSERVATION_FUNCTION
    );
    let observed = call_on(s, object, &observe).await;
    action?;
    Ok(match observed {
        Ok(observed) => click_verdict(&observed),
        Err(e) => unknown_action(
            "click",
            &format!("post-click target inspection failed: {e}"),
        ),
    })
}

fn insertion_command(text: &str) -> Option<InsertTextParams> {
    (!text.is_empty()).then(|| InsertTextParams::new(text))
}

pub(crate) async fn replace_text(
    s: &BrowserSession,
    object: &RemoteObjectId,
    text: &str,
    submit: bool,
) -> Result<Value, String> {
    let before = call_on(s, object, TYPE_OBSERVATION_FUNCTION).await?;
    if before["connected"] == false
        || before["editable"] == false
        || before["disabled"] == true
        || before["read_only"] == true
    {
        return Ok(type_verdict(text, &before));
    }
    let platform = call_on(s, object, "function(){ if(!this.isConnected) throw Error('element detached'); if(!['INPUT','TEXTAREA'].includes(this.tagName) || this.disabled || this.readOnly) throw Error('target must be an editable input or textarea'); this.scrollIntoView({block:'center'}); this.focus(); const root=this.getRootNode(); return {platform:navigator.platform,focused:(root.activeElement || this.ownerDocument.activeElement) === this}; }").await?;
    use chromiumoxide::cdp::browser_protocol::input::{
        DispatchKeyEventParams, DispatchKeyEventType,
    };
    if platform["focused"] != true {
        return Ok(unknown_action(
            "type_text",
            "the intended input did not receive focus; no typing was attempted",
        ));
    }
    let modifier = if platform["platform"]
        .as_str()
        .is_some_and(|p| p.contains("Mac"))
    {
        4
    } else {
        2
    };
    // Select all through Chrome's input dispatcher so frameworks observe native
    // deletion and insertion events, rather than bypassing their value trackers.
    for kind in [
        DispatchKeyEventType::RawKeyDown,
        DispatchKeyEventType::KeyUp,
    ] {
        s.page
            .execute(
                DispatchKeyEventParams::builder()
                    .r#type(kind)
                    .modifiers(modifier)
                    .key("a")
                    .code("KeyA")
                    .windows_virtual_key_code(65)
                    .build()
                    .map_err(|e| e.to_string())?,
            )
            .await
            .map_err(|e| format!("select-all failed: {e}"))?;
    }
    let keyboard = s
        .page
        .find_element("body")
        .await
        .map_err(|e| e.to_string())?;
    keyboard
        .press_key("Backspace")
        .await
        .map_err(|e| format!("clear field failed: {e}"))?;
    if let Some(command) = insertion_command(text) {
        // Input.insertText supports arbitrary Unicode without a keyboard table.
        s.page
            .execute(command)
            .await
            .map_err(|e| format!("Unicode insertion failed: {e}"))?;
    }
    call_on(
        s,
        object,
        "function(){ this.dispatchEvent(new Event('change',{bubbles:true})); }",
    )
    .await?;
    let mut verdict = match call_on(s, object, TYPE_OBSERVATION_FUNCTION).await {
        Ok(observed) => type_verdict(text, &observed),
        Err(e) => unknown_action("type_text", &format!("input read-back failed: {e}")),
    };
    if submit && verdict["status"] == "applied" {
        keyboard
            .press_key("Enter")
            .await
            .map_err(|e| format!("submit failed: {e}"))?;
        verdict["submission"] = json!("Enter dispatched; form completion is not confirmed");
    } else if submit {
        verdict["submission"] = json!("skipped because the requested value was not confirmed");
    }
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_fixes_unicode_insertion_preserves_text() {
        for text in ["ASCII", "中文输入🙂", "line one\n第二行", "é👩‍💻"] {
            let command = insertion_command(text).unwrap();
            assert_eq!(InsertTextParams::IDENTIFIER, "Input.insertText");
            assert_eq!(serde_json::to_value(command).unwrap()["text"], text);
        }
        assert!(insertion_command("").is_none());
    }

    #[test]
    fn browser_fixes_snapshot_oversized_field_advances() {
        let large = json!({"ref":"target:1","role":"button","name":"🙂\n".repeat(20000),"description":"x".repeat(20000)});
        let small = json!({"ref":"target:2","role":"button","name":"next","description":null});
        let page = snapshot_page("target", vec![large, small.clone()], 0, 1).unwrap();
        assert_eq!(page["elements"].as_array().unwrap().len(), 1);
        assert!(page["next_offset"].as_u64().unwrap() > page["offset"].as_u64().unwrap());
        assert!(
            page["elements"][0]["name"]
                .as_str()
                .unwrap()
                .contains("[truncated]")
        );
        assert_eq!(
            page["elements"][0]["truncated_fields"],
            json!(["name", "description"])
        );
        assert!(page.to_string().chars().count() < SNAPSHOT_PAGE_CHARS);
        let second = snapshot_page("target", vec![Value::Null, small], 1, 1).unwrap();
        assert_eq!(second["elements"][0]["ref"], "target:2");
        assert!(second["next_offset"].is_null());
    }

    #[test]
    fn browser_fixes_snapshot_oversized_root_identity() {
        for target in [
            "x".repeat(SNAPSHOT_PAGE_CHARS),
            "\n".repeat(SNAPSHOT_PAGE_CHARS / 2),
        ] {
            for (elements, offset) in [(Vec::new(), 0), (vec![json!({"ref":"target:1"})], 1)] {
                let error = snapshot_page(&target, elements, offset, 1).unwrap_err();
                assert!(error.contains("root identity"));
            }
        }
        let empty = snapshot_page("target", Vec::new(), usize::MAX, 1)
            .unwrap()
            .to_string();
        assert!(validate_payload(&empty).is_ok());
        assert!(validate_payload(&"x".repeat(SNAPSHOT_PAGE_CHARS + 1)).is_err());
        let elements = vec![json!({"ref":"target:1","name":"n".repeat(512)}); 200];
        let page = snapshot_page(&"t".repeat(5_000), elements, 0, 200).unwrap();
        assert!(page.to_string().chars().count() <= SNAPSHOT_PAGE_CHARS);
        assert!(page["next_offset"].as_u64().is_some_and(|next| next > 0));
    }
}
