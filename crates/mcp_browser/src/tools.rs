//! Tool schema declarations + `tools/call` dispatch and per-tool logic.
//!
//! All tools return text via `content[0].text` (the host reads only this slot).
//! Every CDP operation is wrapped in `with_timeout`; a timeout returns a clean error free of transport trigger words.

use std::path::PathBuf;

use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::page::{Page, ScreenshotParams};
use serde_json::{Value, json};

use crate::browser::{BrowserSession, ensure_session};
use crate::contract::{opt_str, require_str};
use mcp_stdio::{JsonRpcErr, cap_text, text_content, with_timeout};

/// Per-operation timeout, defined in `browser.rs` (the session layer) so the
/// runtime never depends on the schema layer. Re-exported here because
/// `applescript.rs` imports it via `crate::tools::op_timeout_ms`.
pub(crate) use crate::browser::op_timeout_ms;

/// Result of initialize.
pub fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": { "tools": {}, "resources": {}, "prompts": {} },
        "serverInfo": { "name": "mcp-browser", "version": "0.1.0" }
    })
}

/// Result of tools/list: schemas for browser automation tools.
pub fn tools_list_result() -> Value {
    let mut registry = json!({
        "tools": [
            {
                "name": "navigate",
                "description": "Navigate the browser session to a URL and wait for load (CDP mode: a controlled Chrome; AppleScript mode: a tab in your own Chrome, keeping cookies/login). Keeps the session (cookies/login) alive across calls. If the response contains [USER_ACTION_REQUIRED: <category>], the page needs manual user intervention (captcha/slider/sms_otp/twofa/login_required/payment_verify/identity_verify) - stop further browser automation, hand control back to the user, and resume only after they confirm.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "Absolute URL to open, e.g. https://example.com" },
                        "wait_selector": { "type": "string", "description": "Optional CSS selector to wait for after navigation" }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "click",
                "description": "Click an element by selector or snapshot ref (scrolled into view first). Returns applied/not_applied/unknown based on the observed click event, target state/focus, or URL; disabled targets are not clicked. If the response starts with [HUMAN_ACTION_PENDING: <category>], a human verification is still pending on the page - stop and call wait_for_human (or return control to the user) before continuing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "CSS selector of the element to click" }
                    },
                    "required": ["selector"]
                }
            },
            {
                "name": "type_text",
                "description": "Replace the entire current value of an input/textarea with Unicode text, including Chinese and emoji, on both drivers (empty text clears it). Use selector or a snapshot ref. Returns applied/not_applied/unknown with value read-back evidence. With submit=true, CDP dispatches Enter and AppleScript uses requestSubmit(); submission errors are surfaced. If the response starts with [HUMAN_ACTION_PENDING: <category>], a human verification is still pending - stop and call wait_for_human (or return control to the user) before continuing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "CSS selector of the input/textarea" },
                        "text": { "type": "string", "description": "Text to type" },
                        "submit": { "type": "boolean", "description": "Submit after confirmed replacement: CDP dispatches Enter; AppleScript uses requestSubmit() and reports its submit event (default false)", "default": false }
                    },
                    "required": ["selector", "text"]
                }
            },
            {
                "name": "press_key",
                "description": "Press a keyboard key (e.g. Enter, Tab, Escape, ArrowDown). Optionally focus a selector first. If the response starts with [HUMAN_ACTION_PENDING: <category>], a human verification is still pending - stop and call wait_for_human (or return control to the user) before continuing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "key": { "type": "string", "description": "Key name, e.g. Enter / Tab / Escape / ArrowDown" },
                        "selector": { "type": "string", "description": "Optional CSS selector to focus before pressing" }
                    },
                    "required": ["key"]
                }
            },
            {
                "name": "scroll",
                "description": "Scroll an element into view (if selector given) or scroll the window to x,y.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "CSS selector to scroll into view" },
                        "x": { "type": "integer", "description": "Window scroll X (used when no selector)" },
                        "y": { "type": "integer", "description": "Window scroll Y (used when no selector)" }
                    }
                }
            },
            {
                "name": "wait_for",
                "description": "Wait for exactly one target: selector with present (default), visible, or absent condition; or url with exact-string equality to location.href, including query and fragment. Returns observed current_url. condition applies only to selector waits. Bounded by timeout_ms and the server operation cap.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "CSS selector to wait for" },
                        "timeout_ms": { "type": "integer", "description": "Max wait in ms (default 15000)", "default": 15000 }
                    },
                    "required": ["selector"]
                }
            },
            {
                "name": "evaluate_js",
                "description": "Evaluate JavaScript and return JSON text identically on both drivers: strings are quoted, arrays/objects serialized, undefined becomes null; runtime exceptions are errors.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": { "type": "string", "description": "JS expression, e.g. document.title" }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_text",
                "description": "Extract visible text (innerText). Whole page (body) if no selector, else the matched element. If the response contains [USER_ACTION_REQUIRED: <category>], the page needs manual user intervention - stop automation and hand control back to the user.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "Optional CSS selector; defaults to body" }
                    }
                }
            },
            {
                "name": "get_html",
                "description": "Extract HTML. Full document if no selector, else the matched element's outerHTML. If the response contains [USER_ACTION_REQUIRED: <category>], the page needs manual user intervention - stop automation and hand control back to the user.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "Optional CSS selector; defaults to full document" }
                    }
                }
            },
            {
                "name": "screenshot",
                "description": "Save a PNG screenshot to disk and return its absolute path (image bytes are NOT returned).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Optional output path; defaults to MCP_BROWSER_SCREENSHOT_DIR / temp dir" },
                        "full_page": { "type": "boolean", "description": "Capture the full scrollable page (default false; in AppleScript mode this degrades to a window capture, noted in the result)", "default": false }
                    }
                }
            },
            {
                "name": "wait_for_human",
                "description": "BLOCK and wait for the user to manually complete an action in the visible (headed) browser window - captcha, slider, SMS/OTP code, 2FA, login, payment or identity verification. Call this when a page shows [USER_ACTION_REQUIRED] or otherwise needs a human. It polls the page and returns AS SOON AS the blocking signal is gone (status=resolved). Each call waits within the server operation cap (below the default host deadline); if the user has not finished yet it returns status=still_waiting WITHOUT error - END YOUR TURN and ask the user to complete the action in the browser window; when they reply, call wait_for_human again to verify and continue. In headless mode it returns status=unavailable immediately (no visible window, so the user cannot act).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expect": { "type": "string", "description": "Optional expected category being waited on (captcha/slider/sms_otp/twofa/login_required/payment_verify/identity_verify); used only for messaging" },
                        "message": { "type": "string", "description": "Optional short instruction to relay to the user about what to do in the window" }
                    }
                }
            },
            {
                "name": "list_tabs",
                "description": "List open tabs with readable index, stable id, URL, and current automation target. Pass id to switch_tab/close_tab; indices are informational.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "close_browser",
                "description": "Close the browser session: CDP self-launch quits the controlled Chrome; attached CDP and AppleScript close only owned tabs and never quit your own Chrome.",
                "inputSchema": { "type": "object", "properties": {} }
            }
        ]
    });
    for tool in registry["tools"].as_array_mut().unwrap() {
        let name = tool["name"].as_str().unwrap().to_string();
        let schema = &mut tool["inputSchema"];
        schema["additionalProperties"] = json!(false);
        if matches!(name.as_str(), "click" | "type_text") {
            schema["properties"]["ref"] = json!({"type":"string", "description":"Stable element ref from snapshot (CDP only), instead of selector"});
            schema["required"] = if name == "type_text" {
                json!(["text"])
            } else {
                json!([])
            };
            schema["oneOf"] = json!([{"required":["selector"]},{"required":["ref"]}]);
        }
        if name == "wait_for" {
            schema["properties"]["url"] = json!({"type":"string","description":"Exact current URL to wait for, including query and fragment; mutually exclusive with selector"});
            schema["required"] = json!([]);
            schema["oneOf"] = json!([{"required":["selector"]},{"required":["url"]}]);
            schema["properties"]["condition"] =
                json!({"type":"string","enum":["present","visible","absent"],"default":"present"});
            schema["properties"]["timeout_ms"]["minimum"] = json!(1);
        }
        if matches!(name.as_str(), "get_text" | "get_html") {
            schema["properties"]["max_chars"] = json!({"type":"integer","minimum":1,"maximum":20000,"default":12000,"description":"Maximum Unicode characters to return; result includes next_offset for paging"});
            schema["properties"]["offset"] = json!({"type":"integer","minimum":0,"default":0,"description":"Unicode character offset into the extracted content"});
        }
        for (key, spec) in schema["properties"].as_object_mut().unwrap() {
            if spec["type"] == "string" && key != "text" {
                spec["minLength"] = json!(1);
            }
            if spec["type"] == "integer" && spec.get("minimum").is_none() {
                spec["minimum"] = json!(0);
                spec["maximum"] = json!(9_007_199_254_740_991u64);
            }
        }
    }
    registry["tools"].as_array_mut().unwrap().extend([
        json!({"name":"switch_tab","description":"Select the tab used by subsequent tools by stable id from list_tabs. CDP does not activate a user window; AppleScript changes the automation target without selecting its visible tab.","inputSchema":{"type":"object","properties":{"id":{"type":"string","minLength":1}},"required":["id"],"additionalProperties":false}}),
        json!({"name":"close_tab","description":"Close a tab by stable id from list_tabs. Attached CDP and AppleScript sessions may close only tabs they created. Closing the current tab selects another owned tab or creates a blank replacement in CDP; AppleScript navigate opens a fresh tab if none remains.","inputSchema":{"type":"object","properties":{"id":{"type":"string","minLength":1}},"required":["id"],"additionalProperties":false}}),
        json!({"name":"snapshot","description":"CDP accessibility snapshot: structured roles, accessible names/labels, descriptions, and stable DOM refs accepted by click/type_text. Refs remain valid while their node exists in that tab. Uses offset/limit paging. Not supported in AppleScript mode.","inputSchema":{"type":"object","properties":{"offset":{"type":"integer","minimum":0,"default":0},"limit":{"type":"integer","minimum":1,"maximum":200,"default":100}},"additionalProperties":false}}),
    ]);
    registry
}

/// tools/call dispatch.
pub async fn handle_tools_call(
    session: &mut Option<BrowserSession>,
    params: Option<Value>,
) -> Result<Value, JsonRpcErr> {
    let (name, args) = crate::contract::parse_call(params)?;

    match name.as_str() {
        "navigate" => tool_navigate(session, &args).await,
        "click" => tool_click(session, &args).await,
        "type_text" => tool_type_text(session, &args).await,
        "press_key" => tool_press_key(session, &args).await,
        "scroll" => tool_scroll(session, &args).await,
        "wait_for" => tool_wait_for(session, &args).await,
        "evaluate_js" => tool_evaluate_js(session, &args).await,
        "get_text" => tool_get_text(session, &args).await,
        "get_html" => tool_get_html(session, &args).await,
        "wait_for_human" => tool_wait_for_human(session, &args).await,
        "screenshot" => tool_screenshot(session, &args).await,
        "list_tabs" => tool_list_tabs(session).await,
        "switch_tab" => tool_switch_tab(session, &args).await,
        "close_tab" => tool_close_tab(session, &args).await,
        "snapshot" => tool_snapshot(session, &args).await,
        "close_browser" => tool_close_browser(session).await,
        _ => Err(JsonRpcErr::new(
            -32601,
            "Unknown tool",
            Some(json!({ "tool": name })),
        )),
    }
}

/// Scans the page for actions that require manual completion by the user.
///
/// Covers 7 categories: captcha, slider, sms_otp, twofa, login_required,
/// payment_verify, identity_verify. Returns the category tag on a hit, None otherwise.
/// Inspection failures remain errors; callers must not infer resolution from them.
///
/// Detection is conservative: DOM structure signals (iframe/class/id, dedicated input fields) come first and are most trusted;
/// plain body keyword matching easily false-positives (e.g. an article explaining 2FA/OTP), so text-only signals are ranked after
/// structure signals, and keyword + structure features are required to match together wherever possible.
/// All structure signals are visibility-gated (`anyVisible`): hidden inputs/iframes
/// (display:none / visibility:hidden / zero-size: unopened modals, templates, invisible
/// captchas) never count as blockers.
async fn detect_user_action_required(
    page: &chromiumoxide::page::Page,
) -> Result<Option<String>, String> {
    let result = page
        .evaluate_expression(DETECT_USER_ACTION_JS)
        .await
        .map_err(|e| format!("human-action inspection failed: {e}"))?;
    parse_detection(result.value().unwrap_or(&Value::Null))
}

pub(crate) fn parse_detection(value: &Value) -> Result<Option<String>, String> {
    match value {
        Value::Null => Ok(None),
        Value::String(cat) if !cat.is_empty() => Ok(Some(cat.clone())),
        _ => Err("human-action inspection returned an invalid result".into()),
    }
}

pub(crate) const DETECT_USER_ACTION_JS: &str = r#"(function(){
          // body can be null on early or frameset documents.
          var t = "";
          try { t = ((document.body && document.body.innerText) || "").toLowerCase(); } catch(e) { t = ""; }
          function has(sel){ try { return !!document.querySelector(sel); } catch(e){ return false; } }
          function anyVisible(sel){
            try {
              var els = document.querySelectorAll(sel);
              for (var i = 0; i < els.length; i++) {
                var r = els[i].getBoundingClientRect();
                var st = getComputedStyle(els[i]);
                if (r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none') return true;
              }
            } catch(e){}
            return false;
          }
          // Solved captcha responses live in a hidden host-document textarea.
          // Captcha iframes remain after solving; presence alone is not a blocker.
          // Only an unsolved visible widget requires human action.
          function recaptchaSolved(){
            var ta = document.querySelector('textarea[name="g-recaptcha-response"], textarea[name="h-captcha-response"]');
            return !!(ta && ta.value && ta.value.length > 0);
          }
          // 1) Visible captcha iframe or class, gated on the unsolved state.
          // Hidden badges and dismissed dialogs do not count.
          if (anyVisible('iframe[src*="recaptcha/api2/anchor"]') && !recaptchaSolved()) return 'captcha';
          if (anyVisible('iframe[src*="hcaptcha"]') && !has('iframe[src*="badge"]') && !recaptchaSolved()) return 'captcha';
          if (anyVisible('iframe[src*="captcha"]') && !has('iframe[src*="recaptcha/api2/bframe"]') && !has('iframe[src*="badge"]') && !recaptchaSolved()) return 'captcha';
          if ((has('[class*="captcha"]') || has('[id*="captcha"]')) && anyVisible('[class*="captcha"], [id*="captcha"]') && !recaptchaSolved()) return 'captcha';
          // 2) Sliders require DOM evidence, not just text.
          if (anyVisible('.geetest_slider_button, .geetest_slider, [class*="slider_track"], [class*="slider-btn"]') || (anyVisible('[class*="slider"]') && /滑动|拖动|slide|滑块/.test(t))) return 'slider';
          // 3) OTP requires a dedicated input or visible input plus matching text.
          if (anyVisible('input[autocomplete="one-time-code"]')) return 'sms_otp';
          if (anyVisible('input[type="tel"], input[type="text"], input[type="number"]') && /短信验证码|短信动态码|sms.{0,5}code|one-time.{0,5}code|verification code|动态验证码/.test(t)) return 'sms_otp';
          // 4) Two-factor verification requires text and a visible input.
          if (anyVisible('input[type="tel"], input[type="text"], input[type="number"]') && /two-factor|双因素|二次验证|authenticator/.test(t)) return 'twofa';
          // 5) Login requires a prompt and a visible login/password field.
          if (anyVisible('input[type="password"], input[name*="login"], input[id*="login"]') && /请登录|请先登录|登录后查看|sign in to continue|please sign in|log in to continue/.test(t)) return 'login_required';
          // 6) Payment verification requires a visible input.
          if (anyVisible('input[type="password"], input[type="tel"]') && /支付密码|payment password|银行短信|bank verification/.test(t)) return 'payment_verify';
          // 7) Identity verification requires a visible input or form.
          if (/实名认证|identity verification|verify your identity/.test(t) && (anyVisible('input') || anyVisible('form'))) return 'identity_verify';
          return null;
        })()"#;

/// Builds the "user action required" tag appended to tool output (single source of truth, avoids duplication).
fn user_action_tag(cat: &str) -> String {
    format!(
        "\n[USER_ACTION_REQUIRED: {cat}] 页面需要用户手动完成操作。可调用 wait_for_human 阻塞等待用户在可见浏览器窗口完成，或直接停止自动化并请用户完成后告知继续。"
    )
}

/// Detects human verification and records the result in the session (`pending_human` flag):
/// on a hit, stores the category and returns the tag to append; on a miss, clears the flag and returns an empty string.
/// Reused by navigate / get_text / get_html / click / type_text / press_key,
/// keeping the session flag consistent with the actual page state.
async fn detect_and_mark(s: &mut BrowserSession) -> String {
    match detect_user_action_required(&s.page).await {
        Ok(Some(cat)) => {
            s.pending_human = Some(cat.clone());
            user_action_tag(&cat)
        }
        Ok(None) => {
            s.pending_human = None;
            String::new()
        }
        Err(reason) => format!("\n[HUMAN_ACTION_UNKNOWN] {reason}"),
    }
}

/// Prefix prepended to mutating tool output while a human action is still pending in the session.
fn pending_warning(s: &BrowserSession) -> String {
    match &s.pending_human {
        Some(cat) => format!(
            "[HUMAN_ACTION_PENDING: {cat}] 页面仍需要用户手动完成验证：请先调用 wait_for_human 等待用户在浏览器中完成，或结束本轮把操作交给用户；在用户完成前继续自动化通常无效。\n"
        ),
        None => String::new(),
    }
}

// ---- Tool implementations ----

/// Polls for an element to appear: one `querySelector` every 200ms; errors if it has not appeared by `timeout_ms`.
/// Aligned with the AppleScript driver's `wait_for_selector` (applescript.rs),
/// so `wait_selector` behaves the same under both drivers (real waiting, not a single lookup).
async fn wait_for_selector(page: &Page, selector: &str, timeout_ms: u64) -> Result<(), String> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
    loop {
        if page.find_element(selector).await.is_ok() {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "等待超时：{}ms 内未找到选择器 '{}'",
                timeout_ms, selector
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}

async fn tool_navigate(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let url = require_str(args, "url")?;
    let wait_selector = opt_str(args, "wait_selector");
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    let summary = with_timeout(ms, async {
        s.page
            .goto(url.clone())
            .await
            .map_err(|e| format!("navigation failed: {e}"))?;
        if let Some(sel) = &wait_selector {
            wait_for_selector(&s.page, sel, ms.saturating_sub(1_000))
                .await
                .map_err(|e| {
                    format!("navigate 等待元素超时：{e}（可忽略后继续，用 wait_for 重试）")
                })?;
        }
        let title = s.page.get_title().await.ok().flatten().unwrap_or_default();
        let final_url = s.page.url().await.ok().flatten().unwrap_or(url.clone());
        // New page = new state: clear the stale human-action flag, then re-detect on the new page.
        s.pending_human = None;
        let state = post_action_state(s).await;
        let outcome = if state.contains("\"readyState\":\"complete\"") {
            "complete"
        } else {
            "incomplete"
        };
        let mut summary =
            format!("Navigated to {final_url}\nTitle: {title}\nload_outcome={outcome}\n{state}");
        summary.push_str(&detect_and_mark(s).await);
        Ok(summary)
    })
    .await?;

    Ok(text_content(summary))
}

async fn tool_click(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let target = args["selector"]
        .as_str()
        .or_else(|| args["ref"].as_str())
        .unwrap();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;
    let pending = pending_warning(s);
    let summary = with_timeout(op_timeout_ms(), async {
        let object = crate::snapshot::resolve_target(s, args).await?;
        let verdict = crate::snapshot::click(s, &object).await?;
        let state = post_action_state(s).await;
        let tag = detect_and_mark(s).await;
        Ok(format!(
            "{pending}click target={target}\n{verdict}\n{state}{tag}"
        ))
    })
    .await?;
    Ok(text_content(summary))
}

async fn tool_type_text(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let target = args["selector"]
        .as_str()
        .or_else(|| args["ref"].as_str())
        .unwrap();
    let text = args["text"].as_str().unwrap();
    let submit = args["submit"].as_bool().unwrap_or(false);
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;
    let pending = pending_warning(s);
    let summary = with_timeout(op_timeout_ms(), async {
        let object = crate::snapshot::resolve_target(s, args).await?;
        let verdict = crate::snapshot::replace_text(s, &object, text, submit).await?;
        let state = post_action_state(s).await;
        let tag = detect_and_mark(s).await;
        Ok(format!(
            "{pending}type_text target={target} requested_chars={} (submit={submit})\n{verdict}\n{state}{tag}",
            text.chars().count()
        ))
    })
    .await?;
    Ok(text_content(summary))
}

async fn tool_press_key(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let key = require_str(args, "key")?;
    let selector = opt_str(args, "selector");
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    // While a human action is still pending, prompt the model to stop first (see pending_warning).
    let pending = pending_warning(s);
    let tag = with_timeout(ms, async {
        // Keyboard actions go through element handles: use the given selector if present, otherwise fall back to body.
        let target = selector.clone().unwrap_or_else(|| "body".to_string());
        let el = s
            .page
            .find_element(target.clone())
            .await
            .map_err(|e| format!("focus target '{target}' not found: {e}"))?;
        if selector.is_some() {
            el.focus().await.map_err(|e| format!("focus failed: {e}"))?;
        } else {
            // No explicit target: the trusted CDP key event lands on the page's
            // *currently focused* element (chromiumoxide sends Input.dispatchKeyEvent),
            // so focusing <body> here would steal focus from an input the model just
            // typed into, and a following Enter would no longer submit the form (the
            // AppleScript driver keeps `document.activeElement` instead). Only give
            // body focus when nothing else has it, so page-level keys (Tab/ArrowDown)
            // still have a target.
            s.page
                .evaluate_expression(
                    "(() => { const a = document.activeElement; \
                     if (!a || a === document.body || a === document.documentElement) \
                       (document.body || document.documentElement).focus(); \
                     return true; })()",
                )
                .await
                .map_err(|e| format!("focus prep failed: {e}"))?;
        }
        el.press_key(&key)
            .await
            .map_err(|e| format!("press_key '{key}' failed: {e}"))?;
        // Enter submissions often trigger captchas/login walls: detect and flag immediately after the action so the model knows right away.
        Ok(format!(
            "\n{}{}",
            post_action_state(s).await,
            detect_and_mark(s).await
        ))
    })
    .await?;

    Ok(text_content(format!("{pending}Pressed {key}{tag}")))
}

async fn tool_scroll(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let selector = opt_str(args, "selector");
    let x = args.get("x").and_then(|v| v.as_u64()).unwrap_or(0);
    let y = args.get("y").and_then(|v| v.as_u64()).unwrap_or(0);
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    let summary = with_timeout(ms, async {
        let action = if let Some(sel) = &selector {
            let el = s
                .page
                .find_element(sel.clone())
                .await
                .map_err(|e| format!("element '{sel}' not found: {e}"))?;
            el.scroll_into_view()
                .await
                .map_err(|e| format!("scroll_into_view failed: {e}"))?;
            format!("Scrolled {sel} into view")
        } else {
            s.page
                .evaluate_expression(format!("window.scrollTo({x}, {y})"))
                .await
                .map_err(|e| format!("window scroll failed: {e}"))?;
            format!("Scrolled window to ({x}, {y})")
        };
        Ok(format!(
            "{}{}\n{}{}",
            pending_warning(s),
            action,
            post_action_state(s).await,
            detect_and_mark(s).await
        ))
    })
    .await?;

    Ok(text_content(summary))
}

async fn tool_wait_for(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let target = crate::contract::parse_wait_target(args)?;
    let timeout_ms = args["timeout_ms"]
        .as_u64()
        .unwrap_or(15_000)
        .min(op_timeout_ms().saturating_sub(1_000));
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;
    let summary = with_timeout(op_timeout_ms(), async {
        let start = std::time::Instant::now();
        loop {
            let result = s
                .page
                .evaluate_expression(target.script())
                .await
                .map_err(|e| format!("wait_for inspection failed: {e}"))?;
            let observation = result.value().ok_or("wait_for observation missing")?;
            let url = observation["current_url"]
                .as_str()
                .ok_or("wait_for URL observation missing")?;
            if observation["matched"] == true {
                let tag = detect_and_mark(s).await;
                return Ok(format!(
                    "Wait met: {} after {} ms\ncurrent_url: {url}{tag}",
                    target.description(),
                    start.elapsed().as_millis()
                ));
            }
            if start.elapsed().as_millis() >= timeout_ms as u128 {
                return Err(format!(
                    "{} not met within {timeout_ms} ms; observed current_url: {url}",
                    target.description()
                ));
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    })
    .await?;
    Ok(text_content(summary))
}

async fn tool_evaluate_js(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let expression = require_str(args, "expression")?;
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    let value = with_timeout(ms, async {
        let result = s
            .page
            .evaluate_expression(crate::contract::evaluation_script(&expression))
            .await
            .map_err(|e| format!("evaluate failed: {e}"))?;
        let raw = result
            .value()
            .and_then(Value::as_str)
            .ok_or("missing JavaScript envelope")?;
        crate::contract::decode_evaluation(raw)
    })
    .await?;

    Ok(text_content(cap_text(&value)))
}

async fn tool_get_text(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    tool_extract(session, args, false).await
}

async fn tool_get_html(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    tool_extract(session, args, true).await
}

async fn tool_extract(
    session: &mut Option<BrowserSession>,
    args: &Value,
    html: bool,
) -> Result<Value, JsonRpcErr> {
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;
    let summary = with_timeout(op_timeout_ms(), async {
        let result = s
            .page
            .evaluate_expression(crate::contract::extraction_script(args, html))
            .await
            .map_err(|e| format!("extraction failed: {e}"))?;
        let raw = result
            .value()
            .and_then(Value::as_str)
            .ok_or("missing extraction envelope")?;
        let body = crate::contract::decode_extraction(raw)?;
        let tag = detect_and_mark(s).await;
        Ok(format!("{body}{tag}"))
    })
    .await?;
    Ok(text_content(summary))
}

/// Wait in a bounded segment; only successful clean inspections can resolve it.
async fn tool_wait_for_human(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    if crate::browser::headless_from_env() && std::env::var_os("MCP_BROWSER_WS_URL").is_none() {
        return Ok(text_content(
            "status=unavailable\nBrowser is headless. Relaunch with MCP_BROWSER_HEADLESS=0 for a visible dedicated window.",
        ));
    }
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;
    let budget_ms = crate::browser::human_wait_budget_ms(op_timeout_ms());
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(budget_ms);
    let mut last_seen = s.pending_human.clone();
    let mut clean_polls = 0;
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(
            remaining.min(std::time::Duration::from_secs(5)),
            detect_user_action_required(&s.page),
        )
        .await
        {
            Ok(Ok(Some(cat))) => {
                s.pending_human = Some(cat.clone());
                last_seen = Some(cat);
                clean_polls = 0;
            }
            Ok(Ok(None)) => {
                clean_polls += 1;
                if clean_polls >= 2 {
                    s.pending_human = None;
                    return Ok(text_content(format!(
                        "status=resolved\n{}\nYou may continue browser automation.",
                        post_action_state(s).await
                    )));
                }
            }
            Ok(Err(reason)) => {
                return Ok(text_content(format!(
                    "status=unknown\nHuman action remains pending: inspection failed: {reason}"
                )));
            }
            Err(_) => {
                return Ok(text_content(
                    "status=unknown\nHuman action remains pending: inspection exceeded its polling deadline.",
                ));
            }
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        tokio::time::sleep(remaining.min(std::time::Duration::from_secs(2))).await;
    }
    let category = last_seen
        .or_else(|| opt_str(args, "expect"))
        .unwrap_or_else(|| "unknown".into());
    s.pending_human = Some(category.clone());
    let relay = opt_str(args, "message")
        .map(|m| format!("\nInstruction for the user: {m}"))
        .unwrap_or_default();
    Ok(text_content(format!(
        "status=still_waiting\nWaiting for '{category}' after {budget_ms} ms.{relay}\nEnd your turn and ask the user to complete it; call wait_for_human after they reply."
    )))
}

async fn tool_screenshot(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let full_page = args
        .get("full_page")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let out_path = resolve_screenshot_path(opt_str(args, "path"))?;
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    let saved = out_path.clone();
    let tag = with_timeout(ms, async {
        let params = ScreenshotParams::builder()
            .format(CaptureScreenshotFormat::Png)
            .full_page(full_page)
            .build();
        s.page
            .save_screenshot(params, &saved)
            .await
            .map_err(|e| format!("screenshot failed: {e}"))?;
        Ok(detect_and_mark(s).await)
    })
    .await?;

    Ok(text_content(format!(
        "Saved screenshot to {} (full_page={full_page}){tag}",
        out_path.display()
    )))
}

/// Resolves the screenshot save path: explicit path > MCP_BROWSER_SCREENSHOT_DIR > system temp dir/mcp_browser.
/// Creates the directory if missing; returns an absolute path.
pub(crate) fn resolve_screenshot_path(explicit: Option<String>) -> Result<PathBuf, JsonRpcErr> {
    let path = if let Some(p) = explicit {
        PathBuf::from(p)
    } else {
        let dir = std::env::var("MCP_BROWSER_SCREENSHOT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("mcp_browser"));
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        dir.join(format!("screenshot-{ts}.png"))
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            JsonRpcErr::new(-32000, &format!("cannot create screenshot dir: {e}"), None)
        })?;
    }
    Ok(std::path::absolute(&path).unwrap_or(path))
}

async fn tool_list_tabs(session: &mut Option<BrowserSession>) -> Result<Value, JsonRpcErr> {
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    let listing = with_timeout(ms, async {
        let pages = s
            .browser
            .pages()
            .await
            .map_err(|e| format!("list pages failed: {e}"))?;
        let mut lines = Vec::with_capacity(pages.len());
        for (i, p) in pages.iter().enumerate() {
            let url = p.url().await.ok().flatten().unwrap_or_default();
            let id = p.target_id().as_ref();
            let current = if id == s.page.target_id().as_ref() {
                " [current]"
            } else {
                ""
            };
            let owned = if s.owned_pages.iter().any(|known| known == id) {
                " [owned]"
            } else {
                ""
            };
            lines.push(format!("#{i} id={id}: {url}{current}{owned}"));
        }
        if lines.is_empty() {
            Ok("(no open tabs)".to_string())
        } else {
            Ok(lines.join("\n"))
        }
    })
    .await?;

    Ok(text_content(listing))
}

async fn tool_close_browser(session: &mut Option<BrowserSession>) -> Result<Value, JsonRpcErr> {
    match session.take() {
        Some(s) => {
            s.shutdown().await;
            Ok(text_content("Browser closed"))
        }
        None => Ok(text_content("No active browser session")),
    }
}

async fn post_action_state(s: &BrowserSession) -> String {
    match s
        .page
        .evaluate_expression(crate::contract::STATE_SCRIPT)
        .await
    {
        Ok(result) => result
            .value()
            .and_then(Value::as_str)
            .unwrap_or("post_action_state=unknown")
            .to_string(),
        Err(e) => format!("post_action_state=unknown: {e}"),
    }
}

async fn tool_snapshot(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;
    let output = with_timeout(
        op_timeout_ms(),
        crate::snapshot::snapshot(
            s,
            args["offset"].as_u64().unwrap_or(0) as usize,
            args["limit"].as_u64().unwrap_or(100) as usize,
        ),
    )
    .await?;
    crate::snapshot::validate_payload(&output)
        .map_err(|reason| JsonRpcErr::new(-32000, &reason, None))?;
    Ok(text_content(output))
}

async fn tool_switch_tab(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let id = require_str(args, "id")?;
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;
    let output = with_timeout(op_timeout_ms(), async {
        let page = s
            .browser
            .get_page(id.clone().into())
            .await
            .map_err(|e| format!("unknown tab id '{id}': {e}"))?;
        s.page = page;
        s.pending_human = None;
        Ok(format!(
            "Selected tab {id}\n{}{}",
            post_action_state(s).await,
            detect_and_mark(s).await
        ))
    })
    .await?;
    Ok(text_content(output))
}

async fn tool_close_tab(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let id = require_str(args, "id")?;
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;
    let output = with_timeout(op_timeout_ms(), async {
        if !s.may_close_tab(&id) {
            return Err("cannot close a tab this session did not create".into());
        }
        let current = id == s.page.target_id().as_ref();
        // Create a replacement before closing the last active page, so every
        // subsequent tool still has a valid target without touching foreign tabs.
        if current {
            let replacement = s
                .browser
                .pages()
                .await
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|page| {
                    page.target_id().as_ref() != id && s.may_close_tab(page.target_id().as_ref())
                });
            s.page = match replacement {
                Some(page) => page,
                None => {
                    let page = s
                        .browser
                        .new_page("about:blank")
                        .await
                        .map_err(|e| format!("replacement tab failed: {e}"))?;
                    s.owned_pages.push(page.target_id().as_ref().to_string());
                    page
                }
            };
            s.pending_human = None;
        }
        s.browser
            .execute(
                chromiumoxide::cdp::browser_protocol::target::CloseTargetParams::new(id.clone()),
            )
            .await
            .map_err(|e| format!("close tab failed: {e}"))?;
        s.owned_pages.retain(|known| known != &id);
        s.snapshot_refs
            .retain(|(reference, _)| !reference.starts_with(&format!("{id}:")));
        Ok(format!("Closed tab {id}\n{}", post_action_state(s).await))
    })
    .await?;
    Ok(text_content(output))
}
