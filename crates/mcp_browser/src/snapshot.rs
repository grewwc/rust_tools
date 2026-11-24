//! Native accessibility snapshots and DOM references scoped to the active target.

use crate::contract::{
    CLICK_OBSERVATION_FUNCTION, TYPE_OBSERVATION_FUNCTION, click_verdict, type_verdict,
    unknown_action,
};
use chromiumoxide::cdp::browser_protocol::accessibility::{
    AxNode, AxPropertyName, EnableParams, GetFullAxTreeParams,
};
use chromiumoxide::cdp::browser_protocol::dom::{
    GetContentQuadsParams, GetNodeForLocationParams, ResolveNodeParams,
};
use chromiumoxide::cdp::browser_protocol::dom_snapshot::{
    CaptureSnapshotParams, CaptureSnapshotReturns, NodeTreeSnapshot, RareStringData, StringIndex,
};
use chromiumoxide::cdp::browser_protocol::input::{
    DispatchMouseEventParams, DispatchMouseEventType, InsertTextParams, MouseButton,
};
use chromiumoxide::cdp::js_protocol::runtime::{
    CallArgument, CallFunctionOnParams, ReleaseObjectGroupParams, RemoteObjectId,
};
use chromiumoxide::layout::Point;
use serde_json::{Value, json};

use crate::browser::BrowserSession;

pub(crate) async fn snapshot(
    s: &mut BrowserSession,
    offset: usize,
    limit: usize,
    deadline: tokio::time::Instant,
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
    let nodes = tree
        .nodes
        .iter()
        .filter(|node| !node.ignored && node.backend_dom_node_id.is_some());
    // AX values and a later DOM type check cannot safely be joined: an input
    // can change type between commands. Use AX only to select candidates.
    let mut candidates: Vec<_> = nodes
        .clone()
        .skip(offset)
        .take(limit)
        .filter_map(|node| ax_value(node).and(node.backend_dom_node_id))
        .map(|id| *id.inner())
        .collect();
    let budget = (deadline.saturating_duration_since(tokio::time::Instant::now()) / 4)
        .min(std::time::Duration::from_millis(350));
    let value_nodes = if !budget.is_zero() && !candidates.is_empty() {
        candidates.sort_unstable();
        // An empty computed-style list is omitted by the generated CDP client,
        // but Chrome requires the parameter. Do not expose any layout/style data.
        let sample = CaptureSnapshotParams::new(vec!["display".into()]);
        tokio::time::timeout(budget, s.page.execute(sample))
            .await
            .ok()
            .and_then(Result::ok)
            .map(|response| sampled_values(&response.result, &candidates))
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let target = s.page.target_id().as_ref();
    let mut elements = Vec::new();
    for node in nodes {
        let backend = node.backend_dom_node_id.unwrap();
        let id = format!("{target}:{}", backend.inner());
        if !s.snapshot_refs.iter().any(|(known, _)| known == &id) {
            s.snapshot_refs.push((id.clone(), backend));
        }
        let sampled = value_nodes
            .binary_search_by_key(backend.inner(), |(id, _)| *id)
            .ok()
            .map(|index| &value_nodes[index].1);
        elements.push(snapshot_element(node, id, sampled));
    }
    Ok(snapshot_page(target, elements, offset, limit)?.to_string())
}

fn sampled_values(sample: &CaptureSnapshotReturns, candidates: &[i64]) -> Vec<(i64, Value)> {
    let mut values = Vec::new();
    for document in &sample.documents {
        let nodes = &document.nodes;
        for (row, id) in nodes.backend_node_id.iter().flatten().enumerate() {
            if candidates.binary_search(id.inner()).is_ok()
                && let Some(value) = sampled_value(nodes, &sample.strings, row)
            {
                values.push((*id.inner(), value));
            }
        }
    }
    values.sort_unstable_by_key(|(id, _)| *id);
    values
}

fn snapshot_string(strings: &[String], index: StringIndex) -> Option<&str> {
    strings
        .get(usize::try_from(*index.inner()).ok()?)
        .map(String::as_str)
}

fn sampled_attribute<'a>(
    nodes: &NodeTreeSnapshot,
    strings: &'a [String],
    row: usize,
    key: &str,
) -> Option<Option<&'a str>> {
    let pairs = nodes.attributes.as_ref()?.get(row)?.inner().chunks_exact(2);
    if !pairs.remainder().is_empty() {
        return None;
    }
    let mut value = None;
    for pair in pairs {
        if snapshot_string(strings, pair[0])?.eq_ignore_ascii_case(key) {
            if value.is_some() {
                return None;
            }
            value = Some(snapshot_string(strings, pair[1])?);
        }
    }
    Some(value)
}

fn sampled_value(nodes: &NodeTreeSnapshot, strings: &[String], row: usize) -> Option<Value> {
    if *nodes.node_type.as_ref()?.get(row)? != 1 {
        return None;
    }
    let tag = snapshot_string(strings, *nodes.node_name.as_ref()?.get(row)?)?;
    let attribute = |key| sampled_attribute(nodes, strings, row, key);
    let live_value = |data: &RareStringData| {
        let index = data.index.binary_search(&(row as i64)).ok()?;
        snapshot_string(strings, *data.value.get(index)?)
    };
    let value = if tag.eq_ignore_ascii_case("input") {
        let kind = attribute("type")?.unwrap_or("text").to_ascii_lowercase();
        // Refuse unknown types and never read the value entry for a password.
        let numeric = match kind.as_str() {
            "range" | "number" => true,
            "text" | "search" | "email" | "url" | "tel" | "date" | "time" | "datetime-local"
            | "month" | "week" | "color" | "file" | "button" | "submit" | "reset" | "checkbox"
            | "radio" | "image" => false,
            _ => return None,
        };
        let value = live_value(nodes.input_value.as_ref()?)?;
        if numeric && let Ok(number) = value.parse::<serde_json::Number>() {
            return Some(Value::Number(number));
        }
        value
    } else if tag.eq_ignore_ascii_case("textarea") {
        live_value(nodes.text_value.as_ref()?)?
    } else if let Some(text) = attribute("aria-valuetext")?.filter(|text| !text.is_empty()) {
        text
    } else if let Some(number) = attribute("aria-valuenow")? {
        return number.parse::<serde_json::Number>().ok().map(Value::Number);
    } else if tag.eq_ignore_ascii_case("select") {
        return selected_option_label(nodes, strings, row)
            .filter(|text| !text.is_empty())
            .map(Value::String);
    } else {
        return None;
    };
    (!value.is_empty()).then(|| json!(value))
}

fn selected_option_label(
    nodes: &NodeTreeSnapshot,
    strings: &[String],
    select: usize,
) -> Option<String> {
    let parents = nodes.parent_index.as_ref()?;
    let option = nodes
        .option_selected
        .as_ref()?
        .index
        .iter()
        .copied()
        .find_map(|index| {
            let option = usize::try_from(index).ok()?;
            let mut child = option;
            // An option can belong to the select directly or through an optgroup.
            while let Some(&parent) = parents.get(child) {
                let parent = usize::try_from(parent).ok()?;
                if parent >= child || parent < select {
                    return None;
                }
                if parent == select {
                    return Some(option);
                }
                child = parent;
            }
            None
        })?;
    if let Some(label) = sampled_attribute(nodes, strings, option, "label")?
        .map(str::trim)
        .filter(|label| !label.is_empty())
    {
        return Some(label.to_owned());
    }
    let types = nodes.node_type.as_ref()?;
    let names = nodes.node_name.as_ref()?;
    let text = nodes.node_value.as_ref()?;
    let mut label = String::new();
    let mut script_root = None;
    // Chrome collects nested text in tree order, skipping script subtrees.
    // Do not treat other elements such as style or noscript as scripts.
    for (row, &parent) in parents.iter().enumerate().skip(option + 1) {
        if parent < option as i64 {
            break;
        }
        if parent >= row as i64 {
            return None;
        }
        if script_root.is_some_and(|script| parent >= script) {
            continue;
        }
        script_root = if *types.get(row)? == 1
            && snapshot_string(strings, *names.get(row)?)?.eq_ignore_ascii_case("script")
        {
            Some(row as i64)
        } else {
            None
        };
        if matches!(types.get(row), Some(3 | 4)) {
            label.push_str(snapshot_string(strings, *text.get(row)?)?);
        }
    }
    Some(label.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn ax_value(node: &AxNode) -> Option<&Value> {
    node.value
        .as_ref()?
        .value
        .as_ref()
        .filter(|value| match value {
            Value::String(text) => !text.is_empty(),
            Value::Bool(_) | Value::Number(_) => true,
            _ => false,
        })
}

fn snapshot_element(node: &AxNode, id: String, sampled: Option<&Value>) -> Value {
    use AxPropertyName::*;

    let ax_field =
        |field: &Option<chromiumoxide::cdp::browser_protocol::accessibility::AxValue>| {
            field
                .as_ref()
                .and_then(|v| v.value.clone())
                .unwrap_or(Value::Null)
        };
    let role = ax_field(&node.role);
    let name = ax_field(&node.name);
    let value = ax_value(node);
    let property = |key| {
        node.properties
            .as_deref()?
            .iter()
            .find(|p| p.name == key)?
            .value
            .value
            .as_ref()
    };
    let active = |value: Option<&Value>| value.is_some_and(|v| v == true || v == "true");
    let interactive = INTERACTIVE_ROLES.contains(&role.as_str().unwrap_or_default())
        || (role != "RootWebArea"
            && active(property(Focusable))
            && (name.as_str().is_some_and(|s| !s.trim().is_empty())
                || value.is_some()
                || active(property(Editable))
                || matches!(
                    property(Editable).and_then(Value::as_str),
                    Some("plaintext" | "richtext")
                )));
    let mut states = Vec::new();
    for (key, on, off) in [
        (Disabled, "disabled", None),
        (Checked, "checked", Some("unchecked")),
        (Expanded, "expanded", Some("collapsed")),
        (Selected, "selected", None),
        (Focused, "focused", None),
        (Readonly, "readonly", None),
        (Required, "required", None),
        (Invalid, "invalid", None),
    ] {
        let value = property(key);
        if active(value) {
            states.push(on);
        } else if let Some(value) = value {
            if let Some(off) = off.filter(|_| value == false || value == "false") {
                states.push(off);
            } else if on == "checked" && value == "mixed" {
                states.push("mixed");
            } else if on == "invalid"
                && matches!(value.as_str(), Some("grammar" | "spelling" | "other"))
            {
                states.push(on);
            }
        }
    }
    let mut element = json!({"ref":id,"role":role,"name":name,
        "description":ax_field(&node.description),"interactive":interactive});
    if !states.is_empty() {
        element["states"] = json!(states);
    }
    if let Some(value) = sampled {
        element["value"] = value.clone();
    }
    element
}

const INTERACTIVE_ROLES: &[&str] = &[
    "button",
    "link",
    "textbox",
    "searchbox",
    "checkbox",
    "radio",
    "combobox",
    "listbox",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "tab",
    "switch",
    "slider",
    "spinbutton",
    "option",
    "treeitem",
];
const SNAPSHOT_PAGE_CHARS: usize = 18_000;
const SNAPSHOT_FIELD_CHARS: usize = 512;
const SNAPSHOT_VALUE_CHARS: usize = 200;

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
        for (key, max_chars) in [
            ("role", SNAPSHOT_FIELD_CHARS),
            ("name", SNAPSHOT_FIELD_CHARS),
            ("description", SNAPSHOT_FIELD_CHARS),
            ("value", SNAPSHOT_VALUE_CHARS),
        ] {
            let Some(value) = element.get_mut(key) else {
                continue;
            };
            let raw = value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string());
            if raw.chars().count() > max_chars {
                let mut bounded: String = raw.chars().take(max_chars).collect();
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
    call_on_with_arg(s, object, function, None).await
}

async fn call_on_with_arg(
    s: &BrowserSession,
    object: &RemoteObjectId,
    function: &str,
    argument: Option<RemoteObjectId>,
) -> Result<Value, String> {
    let mut command = CallFunctionOnParams::builder()
        .object_id(object.clone())
        .function_declaration(function)
        .return_by_value(true)
        .await_promise(true);
    if let Some(argument) = argument {
        command = command.argument(CallArgument::builder().object_id(argument).build());
    }
    let response = s
        .page
        .execute(command.build()?)
        .await
        .map_err(|e| format!("element action failed: {e}"))?
        .result;
    if let Some(error) = response.exception_details {
        return Err(format!("element JavaScript error: {}", error.text));
    }
    Ok(response.result.value.unwrap_or(Value::Null))
}

const CLICK_TARGET_DETACHED: &str = "click target detached; nothing was clicked";
const CLICK_ZERO_AREA: &str = "click target has zero-area geometry; nothing was clicked";

#[derive(Debug, PartialEq, Eq)]
enum QuadError {
    Unavailable,
    ZeroArea,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClickPoint {
    x: i32,
    y: i32,
}

impl ClickPoint {
    fn round(center: Point) -> Option<Self> {
        let coordinate = |value: f64| {
            let rounded = value.round();
            (f64::from(i32::MIN)..=f64::from(i32::MAX))
                .contains(&rounded)
                .then_some(rounded as i32)
        };
        Some(Self {
            x: coordinate(center.x)?,
            y: coordinate(center.y)?,
        })
    }

    fn native_hit_query(self) -> GetNodeForLocationParams {
        GetNodeForLocationParams::new(self.x, self.y)
    }

    fn mouse_event(self, kind: DispatchMouseEventType) -> DispatchMouseEventParams {
        let button_event = matches!(
            kind,
            DispatchMouseEventType::MousePressed | DispatchMouseEventType::MouseReleased
        );
        let mut event = DispatchMouseEventParams::new(kind, f64::from(self.x), f64::from(self.y));
        if button_event {
            event.button = Some(MouseButton::Left);
            event.click_count = Some(1);
        }
        event
    }
}

fn quad_center(coordinates: &[f64]) -> Result<ClickPoint, QuadError> {
    let &[ax, ay, bx, by, cx, cy, dx, dy] = coordinates else {
        return Err(QuadError::Unavailable);
    };
    if !coordinates.iter().all(|value| value.is_finite()) {
        return Err(QuadError::Unavailable);
    }
    let corners = [(ax, ay), (bx, by), (cx, cy), (dx, dy)];
    let center = Point {
        x: ax / 4.0 + bx / 4.0 + cx / 4.0 + dx / 4.0,
        y: ay / 4.0 + by / 4.0 + cy / 4.0 + dy / 4.0,
    };
    // A quad may be rotated. Validate both the geometric center and the
    // rounded input point against all four edges, not just the bounding box.
    let sides = |px, py| {
        std::array::from_fn::<_, 4, _>(|i| {
            let (x, y) = corners[i];
            let (next_x, next_y) = corners[(i + 1) % 4];
            (next_x - x) * (py - y) - (next_y - y) * (px - x)
        })
    };
    let inside = |sides: &[f64; 4]| {
        sides.iter().all(|side| side.is_finite())
            && (sides.iter().all(|&side| side > 0.0) || sides.iter().all(|&side| side < 0.0))
    };
    let center_sides = sides(center.x, center.y);
    if !inside(&center_sides) {
        return Err(
            if center_sides.contains(&0.0) && center_sides.iter().all(|side| side.is_finite()) {
                QuadError::ZeroArea
            } else {
                QuadError::Unavailable
            },
        );
    }
    let point = ClickPoint::round(center).ok_or(QuadError::Unavailable)?;
    inside(&sides(f64::from(point.x), f64::from(point.y)))
        .then_some(point)
        .ok_or(QuadError::Unavailable)
}

// CDP points use the main-frame viewport. For subframe targets a document-local
// hit comparison would need a transform, but connectivity and size still apply.
const CLICK_HIT_TEST_FUNCTION: &str = r#"function(x, y, nativeHit){
    const target = this;
    // The point belongs to the document/window recorded before the hover.
    // Adoption can keep the node connected while changing that coordinate space.
    const {doc, view} = target[Symbol.for('mcp_browser.click_observation')] || {};
    const documentError = () => {
        if (!target.isConnected) return 'detached';
        const currentDoc = target.ownerDocument;
        const currentView = currentDoc?.defaultView;
        if (target.nodeType !== 1 || !view || !currentView) return 'no_viewport';
        if (currentDoc !== doc || currentView !== view || target.getRootNode({composed:true}) !== doc) {
            return 'changed_document';
        }
    };
    let status = documentError();
    if (status) return {status};
    // A native hit can reveal a closed root even when assignedSlot is hidden
    // from the light-DOM node. Inspect its slots from inside that root.
    const nativeRoot = nativeHit?.getRootNode();
    const closedSlots = nativeRoot?.mode === 'closed' ? [...nativeRoot.querySelectorAll('slot')] : [];
    const parent = node => node.assignedSlot || closedSlots.find(slot => slot.assignedNodes().includes(node)) || node.parentNode || node.host;
    const includes = (ancestor, node) => {
        for (let n = node; n; n = parent(n)) if (n === ancestor) return true;
        return false;
    };
    const compare = hit => {
        if (includes(target, hit) || includes(hit, target)) return {status:'hit', related:true};
        // Keep page-controlled prose out of the host's tool error channel.
        const token = text => (text || '').slice(0, 48).replace(/[^a-z0-9_-]/gi, '_');
        const element = hit.nodeType === 1 ? hit : hit.parentElement;
        const id = token(element?.id);
        const role = token(element?.getAttribute('role'));
        return {status:'hit', related:false, blocker:`<${token(element?.localName) || 'element'}${id ? '#' + id : ''}${role ? ' role=' + role : ''}>`};
    };
    const inspect = () => {
        if (target.disabled === true || target.matches(':disabled') || target.getAttribute('aria-disabled') === 'true') {
            return {status:'disabled'};
        }
        const roots = [];
        // Walk the composed parent chain, not getRootNode(): slotted light-DOM
        // targets can render inside one or more shadow roots.
        for (let node = target; node; node = parent(node)) {
            if (node.nodeType === 1 && node.inert) return {status:'disabled'};
            if (node.nodeType === 11 && node.host) roots.push(node);
        }
        const style = view.getComputedStyle(target);
        if (typeof target.checkVisibility === 'function'
                ? !target.checkVisibility({checkOpacity:true, checkVisibilityCSS:true})
                : style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse') {
            return {status:'hidden'};
        }
        if (view !== view.top) return {status:'unavailable'};
        // An outer overlay must still block a target inside a shadow host.
        let deepest;
        for (const root of [doc, ...roots.reverse()]) {
            deepest = root.elementFromPoint(x, y);
            if (!deepest) return {status:nativeHit ? 'ancestor' : 'unavailable'};
            const result = compare(deepest);
            if (!result.related) return result;
        }
        if (deepest !== target && includes(deepest, target)) {
            // The browser may have retargeted a closed-root hit to an ancestor.
            return nativeHit?.isConnected ? compare(nativeHit) : {status:'ancestor'};
        }
        return {status:'hit', related:true};
    };
    let hit;
    try { hit = inspect(); } catch { hit = {status:nativeHit ? 'ancestor' : 'unavailable'}; }
    // Optional failures cannot bypass checks against the current layout.
    status = documentError();
    if (status) return {status};
    const rects = Array.from(view.Element.prototype.getClientRects.call(target));
    if (!rects.length) return {status:'no_rects'};
    const usable = rects.filter(rect => Number.isFinite(rect.width) && Number.isFinite(rect.height) && rect.width > 0 && rect.height > 0);
    if (!usable.length) return {status:'zero_area'};
    if (view === view.top && (x < 0 || y < 0 || x >= view.innerWidth || y >= view.innerHeight ||
            !usable.some(rect => x > rect.left && x < rect.right && y > rect.top && y < rect.bottom))) {
        return {status:'outside'};
    }
    if (view.getComputedStyle(target).pointerEvents === 'none') return {status:'pointer_events_none'};
    status = documentError();
    if (status) return {status};
    return {...hit, layout_validated:true};
}"#;

const CLICK_HIT_OBJECT_GROUP: &str = "mcp_browser.click_hit";

async fn native_hit_for(s: &BrowserSession, point: ClickPoint) -> Result<RemoteObjectId, String> {
    let hit = s
        .page
        .execute(point.native_hit_query())
        .await
        .map_err(|e| e.to_string())?
        .result;
    s.page
        .execute(
            ResolveNodeParams::builder()
                .backend_node_id(hit.backend_node_id)
                .object_group(CLICK_HIT_OBJECT_GROUP)
                .build(),
        )
        .await
        .map_err(|e| e.to_string())?
        .result
        .object
        .object_id
        .ok_or_else(|| "native click hit has no DOM object".into())
}

fn classify_click_probe(hit: &Value) -> Result<&'static str, String> {
    match hit["status"].as_str() {
        Some("detached") => Err(CLICK_TARGET_DETACHED.into()),
        Some("changed_document") => Err("click target changed document or viewport; nothing was clicked".into()),
        Some("no_viewport") => Err("click target has no usable layout viewport; nothing was clicked".into()),
        Some("no_rects") => Err("click target has no client rects; nothing was clicked".into()),
        Some("zero_area") => Err(CLICK_ZERO_AREA.into()),
        Some("outside") => Err("click point is outside the target's current viewport geometry; nothing was clicked".into()),
        Some("hidden") => Err("click target is not visible; nothing was clicked".into()),
        Some("pointer_events_none") => Err("click target has computed pointer-events: none; nothing was clicked".into()),
        Some("disabled") => Ok("disabled"),
        Some("unavailable") if hit["layout_validated"] == true => Ok("unavailable"),
        Some("hit") if hit["related"] == true && hit["layout_validated"] == true => Ok("clear"),
        Some("hit") if hit["related"] == false => Err(format!(
            "click target is covered by {}; nothing was clicked",
            hit["blocker"].as_str().unwrap_or("an unrelated element")
        )),
        _ => Err("click preflight could not confirm target connectivity and geometry; nothing was clicked".into()),
    }
}

fn cdp_click_verdict(observed: Value) -> Value {
    if observed["after"]["connected"] != false {
        return click_verdict(&observed);
    }
    let mut verdict = unknown_action(
        "click",
        "target was detached at post-click inspection; click delivery could not be confirmed",
    );
    verdict["evidence"] = Value::Null;
    verdict
}

pub(crate) async fn click(
    s: &BrowserSession,
    object: &RemoteObjectId,
    deadline: tokio::time::Instant,
) -> Result<Value, String> {
    let prepare = format!(
        r#"function(){{
        const before = ({observe}).call(this);
        if(!before.connected || before.disabled) return {{before}};
        const doc = this.ownerDocument, view = doc?.defaultView;
        this.scrollIntoView({{block:'center',inline:'center'}});
        if(!this.isConnected) return {{before:{{...before,connected:false}}}};
        const state = {{before,doc,view,url_before:doc?.location?.href,event:null}};
        state.listener = e => {{ state.event = e; }};
        this.addEventListener('click',state.listener,{{capture:true,once:true}});
        this[Symbol.for('mcp_browser.click_observation')] = state;
        return {{before}};
    }}"#,
        observe = CLICK_OBSERVATION_FUNCTION
    );
    let before = call_on(s, object, &prepare).await?;
    if before["before"]["connected"] == false {
        return Err(CLICK_TARGET_DETACHED.into());
    }
    if before["before"]["disabled"] == true {
        return Ok(click_verdict(&before));
    }
    let mut used_native_hit = false;
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
        let mut zero_area = false;
        let point = quads
            .iter()
            .find_map(|quad| match quad_center(quad.inner()) {
                Ok(point) => Some(point),
                Err(QuadError::ZeroArea) => {
                    zero_area = true;
                    None
                }
                Err(QuadError::Unavailable) => None,
            })
            .ok_or(if zero_area {
                CLICK_ZERO_AREA
            } else {
                "element has no visible click target"
            })?;
        // Keep the native hover event, but let its handlers run before the final
        // preflight: they can remove the target or show an overlay at this point.
        s.page
            .execute(point.mouse_event(DispatchMouseEventType::MouseMoved))
            .await
            .map_err(|e| format!("click pointer move failed: {e}"))?;
        let ClickPoint { x, y } = point;
        let events = [DispatchMouseEventType::MousePressed, DispatchMouseEventType::MouseReleased]
            .map(|kind| point.mouse_event(kind));
        let hit_script = format!(
            "function(nativeHit){{return ({CLICK_HIT_TEST_FUNCTION}).call(this,{x},{y},nativeHit);}}"
        );
        let hit_budget = (deadline.saturating_duration_since(tokio::time::Instant::now()) / 4)
            .min(std::time::Duration::from_millis(150));
        let hit = tokio::time::timeout(hit_budget, async {
            let hit = call_on(s, object, &hit_script).await?;
            if hit["status"] == "ancestor" {
                used_native_hit = true;
                let native_hit = native_hit_for(s, point).await?;
                return call_on_with_arg(s, object, &hit_script, Some(native_hit)).await;
            }
            Ok(hit)
        }).await.ok().and_then(Result::ok).unwrap_or(Value::Null);
        let hit_test = classify_click_probe(&hit)?;
        if hit_test != "disabled" {
            // CDP cannot atomically check the DOM and press. Do not insert
            // another browser command here; clear records only the probe time.
            for event in events {
                s.page.execute(event).await.map_err(|e| format!("click failed: {e}"))?;
            }
        }
        Ok::<_, String>(hit_test)
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
    if used_native_hit {
        let _ = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            s.page
                .execute(ReleaseObjectGroupParams::new(CLICK_HIT_OBJECT_GROUP)),
        )
        .await;
    }
    let hit_test = action?;
    let mut verdict = if hit_test == "disabled" {
        let mut disabled = before;
        disabled["before"]["disabled"] = json!(true);
        click_verdict(&disabled)
    } else {
        match observed {
            Ok(observed) => cdp_click_verdict(observed),
            Err(e) => unknown_action(
                "click",
                &format!("post-click target inspection failed: {e}"),
            ),
        }
    };
    verdict["hit_test"] = json!(hit_test);
    Ok(verdict)
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
    fn browser_fixes_snapshot_ax_states_and_interactivity() {
        let control: AxNode = serde_json::from_value(json!({"nodeId":"1","ignored":false,
        "role":{"type":"role","value":"generic"}, "value":{"type":"number","value":0},
        "properties":[
            {"name":"focusable","value":{"type":"booleanOrUndefined","value":true}},
            {"name":"checked","value":{"type":"tristate","value":"mixed"}},
            {"name":"expanded","value":{"type":"booleanOrUndefined","value":false}},
            {"name":"invalid","value":{"type":"token","value":"spelling"}},
            {"name":"selected","value":{"type":"booleanOrUndefined","value":false}}
        ]}))
        .unwrap();
        let element = snapshot_element(&control, "tab:1".into(), Some(&json!(0)));
        assert_eq!(element["interactive"], true);
        assert_eq!(element["value"], 0);
        assert_eq!(element["states"], json!(["mixed", "collapsed", "invalid"]));
        let text: AxNode = serde_json::from_value(json!({"nodeId":"2","ignored":false,
            "role":{"type":"internalRole","value":"StaticText"},
            "name":{"type":"computedString","value":"Keep this text"}}))
        .unwrap();
        let element = snapshot_element(&text, "tab:2".into(), None);
        assert_eq!(element["name"], "Keep this text");
        assert_eq!(element["interactive"], false);
        assert!(element.get("value").is_none());
        assert!(element.get("states").is_none());
    }

    #[test]
    fn browser_fixes_snapshot_values_fail_closed_and_follow_paging_budget() {
        let strings = [
            "INPUT",
            "type",
            "PaSsWoRd",
            "text",
            "unknown",
            "password-shaped-sample",
        ]
        .map(str::to_owned)
        .into_iter()
        .chain(["🙂".repeat(300)])
        .collect::<Vec<_>>();
        let dom: NodeTreeSnapshot = serde_json::from_value(json!({
            "nodeType":[1,1,1,1,1], "nodeName":[0,0,0,0,0],
            "attributes":[[1,2],[1,3],[1,4],[1],[1,3,1,2]],
            "inputValue":{"index":[0,1,2,3,4],"value":[5,6,5,5,5]}
        }))
        .unwrap();
        let field: AxNode = serde_json::from_value(json!({"nodeId":"value","ignored":false,
            "role":{"type":"role","value":"textbox"},
            "value":{"type":"string","value":"stale-AX-password-mask"}}))
        .unwrap();
        let elements = (0..3)
            .map(|row| {
                let value = sampled_value(&dom, &strings, row);
                snapshot_element(&field, format!("tab:{row}"), value.as_ref())
            })
            .collect();
        assert!(sampled_value(&dom, &strings, 3).is_none());
        assert!(sampled_value(&dom, &strings, 4).is_none());
        let page = snapshot_page("tab", elements, 0, 3).unwrap();
        assert_eq!(page["total"], 3);
        assert!(page["elements"][0].get("value").is_none());
        assert_eq!(
            page["elements"][1]["value"],
            format!("{} [truncated]", "🙂".repeat(200))
        );
        assert_eq!(page["elements"][1]["truncated_fields"], json!(["value"]));
        assert!(page["elements"][2].get("value").is_none());
    }

    #[test]
    fn browser_fixes_click_quad_requires_an_interior_point() {
        for quad in [
            [0., 0., 80., 0., 80., 0., 0., 0.],
            [0., 0., 0., 0., 0., 80., 0., 80.],
            [0., 0., 20., 20., 40., 40., 20., 20.],
        ] {
            assert_eq!(quad_center(&quad), Err(QuadError::ZeroArea));
        }
        assert_eq!(quad_center(&[0.; 6]), Err(QuadError::Unavailable));
        assert_eq!(
            quad_center(&[0., 0., f64::NAN, 1., 2., 2., 1., 1.]),
            Err(QuadError::Unavailable)
        );
        assert_eq!(
            quad_center(&[10., 0., 20., 10., 10., 20., 0., 10.]),
            Ok(ClickPoint { x: 10, y: 10 })
        );
    }

    #[test]
    fn browser_fixes_fractional_click_uses_one_integer_point() {
        let point =
            quad_center(&[100.25, 40.75, 180.25, 40.75, 180.25, 80.75, 100.25, 80.75]).unwrap();
        assert_eq!(point, ClickPoint { x: 140, y: 61 });
        let query = point.native_hit_query();
        assert_eq!((query.x, query.y), (140, 61));
        for kind in [
            DispatchMouseEventType::MouseMoved,
            DispatchMouseEventType::MousePressed,
            DispatchMouseEventType::MouseReleased,
        ] {
            let event = point.mouse_event(kind);
            assert_eq!((event.x, event.y), (query.x as f64, query.y as f64));
        }
        // Rounding a positive-area subpixel box can leave the valid quad.
        assert_eq!(
            quad_center(&[0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4]),
            Err(QuadError::Unavailable)
        );
    }

    #[test]
    fn browser_fixes_selected_option_label_includes_nested_text() {
        let strings = [
            "",
            " Prefix ",
            "Nested ",
            "next option",
            "label",
            "  Explicit  ",
            "SELECT",
            "OPTION",
            "#text",
            "SPAN",
            "SCRIPT",
            "NOT LABEL",
            "tail",
        ]
        .map(str::to_owned);
        let mut nodes: NodeTreeSnapshot = serde_json::from_value(json!({
            "parentIndex":[-1,0,1,1,3,1,5,1,0,8], "nodeType":[1,1,3,1,3,1,3,3,1,3],
            "nodeName":[6,7,8,9,8,10,8,8,7,8], "nodeValue":[0,0,1,0,2,0,11,12,0,3],
            "attributes":[[],[4,0],[],[],[],[],[],[],[],[]], "optionSelected":{"index":[1]}
        }))
        .unwrap();
        assert_eq!(
            selected_option_label(&nodes, &strings, 0).as_deref(),
            Some("Prefix Nested tail")
        );
        nodes.attributes.as_mut().unwrap()[1] = serde_json::from_value(json!([4, 5])).unwrap();
        assert_eq!(
            selected_option_label(&nodes, &strings, 0).as_deref(),
            Some("Explicit")
        );
    }

    #[test]
    fn browser_fixes_click_probe_distinguishes_composed_blockers() {
        assert_eq!(
            classify_click_probe(&json!({"status":"hit","related":true,"layout_validated":true})),
            Ok("clear")
        );
        let unrelated = json!({"status":"hit","related":false,"blocker":"<div#slot-cover>"});
        assert!(
            classify_click_probe(&unrelated)
                .unwrap_err()
                .contains("<div#slot-cover>")
        );
        assert!(classify_click_probe(&json!({"status":"hit","related":false})).is_err());
    }

    #[test]
    fn browser_fixes_click_probe_never_treats_detached_as_unavailable() {
        assert_eq!(
            classify_click_probe(&json!({"status":"detached"})),
            Err(CLICK_TARGET_DETACHED.into())
        );
        assert_eq!(
            classify_click_probe(&json!({"status":"zero_area"})),
            Err(CLICK_ZERO_AREA.into())
        );
        assert_eq!(
            classify_click_probe(&json!({"status":"unavailable","layout_validated":true})),
            Ok("unavailable")
        );
        assert!(classify_click_probe(&Value::Null).is_err());
    }

    #[test]
    fn browser_fixes_click_requires_live_viewport_and_rects() {
        for (status, reason) in [
            ("changed_document", "changed document or viewport"),
            ("no_viewport", "layout viewport"),
            ("no_rects", "no client rects"),
        ] {
            assert!(
                classify_click_probe(&json!({"status":status}))
                    .unwrap_err()
                    .contains(reason)
            );
        }
        for observation in [
            json!({"status":"unavailable"}),
            json!({"status":"hit","related":true}),
        ] {
            assert!(classify_click_probe(&observation).is_err());
        }
    }

    #[test]
    fn browser_fixes_click_rejects_pointer_events_none() {
        let result =
            classify_click_probe(&json!({"status":"pointer_events_none","layout_validated":true}));
        assert!(result.unwrap_err().contains("pointer-events: none"));
    }

    #[test]
    fn browser_fixes_cdp_click_detached_after_dispatch_reports_unknown() {
        for received in [false, true] {
            let verdict = cdp_click_verdict(
                json!({"before":{"connected":true},"after":{"connected":false},
                "event_received":received,"event_trusted":received,"default_prevented":false}),
            );
            assert_eq!(verdict["status"], "unknown");
            assert!(verdict["reason"].as_str().unwrap().contains("detached"));
            assert_eq!(verdict.get("evidence"), Some(&Value::Null));
        }
        let connected = json!({"before":{"connected":true,"focused":false},
            "after":{"connected":true,"focused":true},"event_received":true,"event_trusted":true});
        assert_eq!(
            cdp_click_verdict(connected.clone()),
            click_verdict(&connected)
        );
    }

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
