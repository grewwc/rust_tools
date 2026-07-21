//! 工具 schema 声明 + `tools/call` 分发与各工具逻辑。
//!
//! 全部 13 个工具都通过 `content[0].text` 回传文本（宿主只读这一处）。
//! 每个 CDP 操作都包在 `with_timeout` 里，超时返回不含 transport 触发词的干净错误。

use std::path::PathBuf;

use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::page::ScreenshotParams;
use serde_json::{Value, json};

use crate::browser::{BrowserSession, ensure_session};
use mcp_stdio::{JsonRpcErr, cap_text, text_content, with_timeout};

/// 每操作超时（毫秒）。默认 90s，短于宿主 request_timeout_ms（建议 120s），
/// 使超时由 server 侧兜底、返回干净错误，而非被宿主 kill 掉整个会话。
fn op_timeout_ms() -> u64 {
    std::env::var("MCP_BROWSER_OP_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(90_000)
}

/// initialize 结果。
pub fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": { "tools": {}, "resources": {}, "prompts": {} },
        "serverInfo": { "name": "mcp-browser", "version": "0.1.0" }
    })
}

/// tools/list 结果：13 个浏览器自动化工具的 schema。
pub fn tools_list_result() -> Value {
    json!({
        "tools": [
            {
                "name": "navigate",
                "description": "Launch/reuse a controlled Chrome and navigate to a URL, waiting for load. Keeps the session (cookies/login) alive across calls. If the response contains [USER_ACTION_REQUIRED: <category>], the page needs manual user intervention (captcha/slider/sms_otp/twofa/login_required/payment_verify/identity_verify) - stop further browser automation, hand control back to the user, and resume only after they confirm.",
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
                "description": "Click the first element matching a CSS selector (scrolled into view first).",
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
                "description": "Focus an input matching the selector and type text. Optionally press Enter to submit.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "CSS selector of the input/textarea" },
                        "text": { "type": "string", "description": "Text to type" },
                        "submit": { "type": "boolean", "description": "Press Enter after typing (default false)", "default": false }
                    },
                    "required": ["selector", "text"]
                }
            },
            {
                "name": "press_key",
                "description": "Press a keyboard key (e.g. Enter, Tab, Escape, ArrowDown). Optionally focus a selector first.",
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
                "description": "Poll until a CSS selector appears in the DOM, up to timeout_ms.",
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
                "description": "Evaluate a JavaScript expression in the page and return its JSON value.",
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
                        "full_page": { "type": "boolean", "description": "Capture the full scrollable page (default false)", "default": false }
                    }
                }
            },
            {
                "name": "wait_for_human",
                "description": "BLOCK and wait for the user to manually complete an action in the visible (headed) browser window - captcha, slider, SMS/OTP code, 2FA, login, payment or identity verification. Call this when a page shows [USER_ACTION_REQUIRED] or otherwise needs a human. It polls the page and returns AS SOON AS the blocking signal is gone (status=resolved). Each call waits up to a bounded budget (default 60s, always kept safely under the host request timeout); if the user has not finished yet it returns status=still_waiting WITHOUT error - simply call it again to keep waiting. Requires headed mode (a visible window) so the user can act.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expect": { "type": "string", "description": "Optional expected category being waited on (captcha/slider/sms_otp/twofa/login_required/payment_verify/identity_verify); used only for messaging" },
                        "budget_ms": { "type": "integer", "description": "Max time to block on THIS call in ms (default 60000). Hard-clamped below the host timeout; on expiry returns still_waiting so you can call again." },
                        "message": { "type": "string", "description": "Optional short instruction to relay to the user about what to do in the window" }
                    }
                }
            },
            {
                "name": "list_tabs",
                "description": "List open tabs (index and URL) in the controlled browser.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "close_browser",
                "description": "Close the controlled Chrome and release the session.",
                "inputSchema": { "type": "object", "properties": {} }
            }
        ]
    })
}

/// tools/call 分发。
pub async fn handle_tools_call(
    session: &mut Option<BrowserSession>,
    params: Option<Value>,
) -> Result<Value, JsonRpcErr> {
    let params = params.unwrap_or_else(|| json!({}));
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

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
        "close_browser" => tool_close_browser(session).await,
        _ => Err(JsonRpcErr::new(
            -32601,
            "Unknown tool",
            Some(json!({ "tool": name })),
        )),
    }
}

// ---- 参数助手 ----

fn require_str(args: &Value, key: &str) -> Result<String, JsonRpcErr> {
    args.get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .ok_or_else(|| JsonRpcErr::new(-32602, &format!("missing or empty '{key}'"), None))
}

fn opt_str(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// 扫描页面是否出现需要用户手动完成的操作。
///
/// 覆盖 7 类：captcha、slider（滑块）、sms_otp、twofa、login_required、
/// payment_verify、identity_verify。命中则返回分类标签，未命中返回 None。
/// 检测失败（JS 执行异常等）静默返回 None，不影响正常流程。
///
/// 检测偏保守：DOM 结构信号（iframe/class/id、专用输入框）优先且最可信；
/// 纯正文关键词匹配容易误报（如一篇讲 2FA/OTP 原理的文章），因此把纯文本类
/// 归到结构信号之后，且尽量要求「关键词 + 结构特征」同时命中。
async fn detect_user_action_required(page: &chromiumoxide::page::Page) -> Option<String> {
    let result = page
        .evaluate_expression(r#"(function(){
          // body 可能为 null（极早期/纯 frameset 页），用 try 兜底避免整表达式 reject。
          var t = "";
          try { t = ((document.body && document.body.innerText) || "").toLowerCase(); } catch(e) { t = ""; }
          function has(sel){ try { return !!document.querySelector(sel); } catch(e){ return false; } }
          // 1) captcha：结构信号最强，直接判定。
          if (has('iframe[src*="recaptcha"]') || has('iframe[src*="hcaptcha"]') || has('[class*="captcha"]') || has('[id*="captcha"]')) return 'captcha';
          // 2) slider：需要滑块 DOM 结构，纯文案不足以判定。
          if (has('.geetest_slider_button') || has('.geetest_slider') || has('[class*="slider_track"]') || has('[class*="slider-btn"]') || (has('[class*="slider"]') && /滑动|拖动|slide|滑块/.test(t))) return 'slider';
          // 3) sms_otp：优先专用输入框；纯文本需同时存在可输入的验证码框，降低误报。
          if (has('input[autocomplete="one-time-code"]')) return 'sms_otp';
          if ((has('input[type="tel"]') || has('input[type="text"]') || has('input[type="number"]')) && /短信验证码|短信动态码|sms.{0,5}code|one-time.{0,5}code|verification code|动态验证码/.test(t)) return 'sms_otp';
          // 4) twofa：要求「关键词 + 可输入框」同时命中，避免误伤科普文。
          if ((has('input[type="tel"]') || has('input[type="text"]') || has('input[type="number"]')) && /two-factor|双因素|二次验证|authenticator/.test(t)) return 'twofa';
          // 5) login_required：登录提示 + 页面存在密码/登录框。
          if ((has('input[type="password"]') || has('input[name*="login"]') || has('input[id*="login"]')) && /请登录|请先登录|登录后查看|sign in to continue|please sign in|log in to continue/.test(t)) return 'login_required';
          // 6) payment_verify：支付/银行验证，要求存在输入框。
          if ((has('input[type="password"]') || has('input[type="tel"]')) && /支付密码|payment password|银行短信|bank verification/.test(t)) return 'payment_verify';
          // 7) identity_verify：实名认证。
          if (/实名认证|identity verification|verify your identity/.test(t) && (has('input') || has('form'))) return 'identity_verify';
          return null;
        })()"#)
        .await
        .ok()?;
    let v = result.value().and_then(|v| v.as_str())?.to_string();
    if v.is_empty() { None } else { Some(v) }
}

/// 生成追加到工具输出末尾的「需人工介入」提示标签（单一事实来源，避免多处重复）。
fn user_action_tag(cat: &str) -> String {
    format!(
        "\n[USER_ACTION_REQUIRED: {cat}] 页面需要用户手动完成操作。可调用 wait_for_human 阻塞等待用户在可见浏览器窗口完成，或直接停止自动化并请用户完成后告知继续。"
    )
}

// ---- 各工具实现 ----

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
        s.page
            .wait_for_navigation()
            .await
            .map_err(|e| format!("wait_for_navigation failed: {e}"))?;
        if let Some(sel) = &wait_selector {
            s.page
                .find_element(sel.clone())
                .await
                .map_err(|e| format!("wait_selector '{sel}' not found: {e}"))?;
        }
        let title = s.page.get_title().await.ok().flatten().unwrap_or_default();
        let final_url = s.page.url().await.ok().flatten().unwrap_or(url.clone());
        let mut summary = format!("Navigated to {final_url}\nTitle: {title}");
        if let Some(cat) = detect_user_action_required(&s.page).await {
            summary.push_str(&user_action_tag(&cat));
        }
        Ok(summary)
    })
    .await?;

    Ok(text_content(summary))
}

async fn tool_click(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let selector = require_str(args, "selector")?;
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    with_timeout(ms, async {
        let el = s
            .page
            .find_element(selector.clone())
            .await
            .map_err(|e| format!("element '{selector}' not found: {e}"))?;
        el.scroll_into_view()
            .await
            .map_err(|e| format!("scroll_into_view failed: {e}"))?;
        el.click().await.map_err(|e| format!("click failed: {e}"))?;
        Ok(())
    })
    .await?;

    Ok(text_content(format!("Clicked {selector}")))
}

async fn tool_type_text(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let selector = require_str(args, "selector")?;
    let text = args
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| JsonRpcErr::new(-32602, "missing 'text'", None))?
        .to_string();
    let submit = args
        .get("submit")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let char_count = text.chars().count();
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    with_timeout(ms, async {
        let el = s
            .page
            .find_element(selector.clone())
            .await
            .map_err(|e| format!("element '{selector}' not found: {e}"))?;
        el.scroll_into_view()
            .await
            .map_err(|e| format!("scroll_into_view failed: {e}"))?;
        el.focus().await.map_err(|e| format!("focus failed: {e}"))?;
        el.type_str(&text)
            .await
            .map_err(|e| format!("type failed: {e}"))?;
        if submit {
            el.press_key("Enter")
                .await
                .map_err(|e| format!("submit (Enter) failed: {e}"))?;
        }
        Ok(())
    })
    .await?;

    let suffix = if submit { " and pressed Enter" } else { "" };
    Ok(text_content(format!(
        "Typed {char_count} chars into {selector}{suffix}"
    )))
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

    with_timeout(ms, async {
        // 键盘动作走元素句柄：给定 selector 则用它，否则退到 body。
        let target = selector.clone().unwrap_or_else(|| "body".to_string());
        let el = s
            .page
            .find_element(target.clone())
            .await
            .map_err(|e| format!("focus target '{target}' not found: {e}"))?;
        el.focus().await.map_err(|e| format!("focus failed: {e}"))?;
        el.press_key(&key)
            .await
            .map_err(|e| format!("press_key '{key}' failed: {e}"))?;
        Ok(())
    })
    .await?;

    Ok(text_content(format!("Pressed {key}")))
}

async fn tool_scroll(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let selector = opt_str(args, "selector");
    let x = args.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
    let y = args.get("y").and_then(|v| v.as_i64()).unwrap_or(0);
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    let summary = with_timeout(ms, async {
        if let Some(sel) = &selector {
            let el = s
                .page
                .find_element(sel.clone())
                .await
                .map_err(|e| format!("element '{sel}' not found: {e}"))?;
            el.scroll_into_view()
                .await
                .map_err(|e| format!("scroll_into_view failed: {e}"))?;
            Ok(format!("Scrolled {sel} into view"))
        } else {
            s.page
                .evaluate_expression(format!("window.scrollTo({x}, {y})"))
                .await
                .map_err(|e| format!("window scroll failed: {e}"))?;
            Ok(format!("Scrolled window to ({x}, {y})"))
        }
    })
    .await?;

    Ok(text_content(summary))
}

async fn tool_wait_for(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let selector = require_str(args, "selector")?;
    let timeout_ms = args
        .get("timeout_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(15_000);
    // 轮询上限取 min(操作 cap, 用户 timeout)，保证 server cap 优先。
    let cap = op_timeout_ms().min(timeout_ms);
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    let summary = with_timeout(cap, async {
        let start = std::time::Instant::now();
        loop {
            if s.page.find_element(selector.clone()).await.is_ok() {
                let ms = start.elapsed().as_millis();
                return Ok(format!("Found {selector} after {ms} ms"));
            }
            if start.elapsed().as_millis() as u64 >= timeout_ms {
                return Err(format!(
                    "selector '{selector}' not found within {timeout_ms} ms"
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
            .evaluate_expression(expression.clone())
            .await
            .map_err(|e| format!("evaluate failed: {e}"))?;
        let json_val = result.value().cloned().unwrap_or(Value::Null);
        Ok(serde_json::to_string_pretty(&json_val).unwrap_or_else(|_| json_val.to_string()))
    })
    .await?;

    Ok(text_content(cap_text(&value)))
}

async fn tool_get_text(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let selector = opt_str(args, "selector").unwrap_or_else(|| "body".to_string());
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    let text = with_timeout(ms, async {
        let el = s
            .page
            .find_element(selector.clone())
            .await
            .map_err(|e| format!("element '{selector}' not found: {e}"))?;
        let inner = el
            .inner_text()
            .await
            .map_err(|e| format!("inner_text failed: {e}"))?
            .unwrap_or_default();
        // 检测在正文之外单独返回，避免标签被 cap_text 截断吞掉（正文可能 ≥ 24K）。
        let cat = detect_user_action_required(&s.page).await;
        Ok((inner, cat))
    })
    .await?;

    let (inner, cat) = text;
    let mut out = cap_text(&inner);
    if let Some(cat) = cat {
        out.push_str(&user_action_tag(&cat));
    }
    Ok(text_content(out))
}

async fn tool_get_html(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let selector = opt_str(args, "selector");
    let ms = op_timeout_ms();
    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    let html = with_timeout(ms, async {
        let content = match &selector {
            Some(sel) => {
                let el = s
                    .page
                    .find_element(sel.clone())
                    .await
                    .map_err(|e| format!("element '{sel}' not found: {e}"))?;
                Ok(el
                    .outer_html()
                    .await
                    .map_err(|e| format!("outer_html failed: {e}"))?
                    .unwrap_or_default())
            }
            None => s
                .page
                .content()
                .await
                .map_err(|e| format!("content failed: {e}")),
        }?;
        // 检测在正文之外单独返回，避免标签被 cap_text 截断吞掉（HTML 极易 ≥ 24K）。
        let cat = detect_user_action_required(&s.page).await;
        Ok((content, cat))
    })
    .await?;

    let (content, cat) = html;
    let mut out = cap_text(&content);
    if let Some(cat) = cat {
        out.push_str(&user_action_tag(&cat));
    }
    Ok(text_content(out))
}

/// 阻塞等待用户在可见窗口手动完成验证类操作（captcha / 滑块 / 短信码 / 2FA / 登录 …）。
///
/// # 为什么是「可续期分段阻塞」而非「一直等到底」
/// 宿主对每个 MCP server 只有一个 `request_timeout_ms`（默认 120s），一旦某次
/// tools/call 超过它，宿主会**kill 并 restart 子进程、销毁整个浏览器会话**
/// （见 AGENTS.md 铁律 #2）。人工完成验证码常需数分钟，若本工具一直阻塞到底，
/// 必然超时→窗口被杀→人做到一半前功尽弃。
///
/// 因此本工具**单次调用只阻塞一个有界预算**（默认 60s，且被硬夹紧到显著小于
/// server op 上限），期间每 2s 轮询一次 `detect_user_action_required`：
/// - 一旦检测不到阻塞信号 → 立即返回 `status=resolved`（人工已完成，可继续自动化）；
/// - 预算耗尽仍未完成 → **正常返回**（非 error）`status=still_waiting`，提示模型
///   「再次调用本工具即可继续等待」。绝不返回 -32001 超时错误，避免模型误判为失败。
///
/// 这样既是**真阻塞**（窗口内人一完成就马上被感知并返回），又永远不会触发宿主的
/// kill+restart，人可以跨多次调用从容完成任意长的手工操作。
async fn tool_wait_for_human(
    session: &mut Option<BrowserSession>,
    args: &Value,
) -> Result<Value, JsonRpcErr> {
    let expect = opt_str(args, "expect");
    let relay = opt_str(args, "message");

    // 单次调用的阻塞预算：默认 60s。硬夹紧到 [2s, op_timeout - 15s]，
    // 给最后一次轮询 + 返回留足余量，确保绝不逼近宿主 request_timeout。
    let op_cap = op_timeout_ms();
    let ceiling = op_cap.saturating_sub(15_000).max(5_000);
    let requested = args
        .get("budget_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(60_000);
    let budget_ms = requested.clamp(2_000, ceiling);

    let s = ensure_session(session)
        .await
        .map_err(|e| JsonRpcErr::new(-32000, &e, None))?;

    // 无头模式下用户根本看不到窗口、无法手动操作——诚实拒绝而非假装等待。
    let headless = std::env::var("MCP_BROWSER_HEADLESS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let poll_interval = std::time::Duration::from_millis(2_000);
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(budget_ms);

    // 段内轮询循环。注意：本循环自身即受 budget_ms 约束，不再套 with_timeout。
    let mut last_seen: Option<String> = None;
    loop {
        match detect_user_action_required(&s.page).await {
            Some(cat) => {
                // 仍存在阻塞信号：记录分类，继续等。
                last_seen = Some(cat);
            }
            None => {
                // 阻塞信号消失 = 人工已完成（或本就没有）。立即返回成功。
                let url = s.page.url().await.ok().flatten().unwrap_or_default();
                let note = match &last_seen {
                    Some(c) => format!("resolved: '{c}' cleared"),
                    None => "resolved: no blocking signal detected".to_string(),
                };
                return Ok(text_content(format!(
                    "status=resolved\n{note}\ncurrent_url: {url}\nYou may continue browser automation now."
                )));
            }
        }

        if std::time::Instant::now() + poll_interval >= deadline {
            break;
        }
        tokio::time::sleep(poll_interval).await;
    }

    // 预算耗尽仍未完成：正常返回 still_waiting（非 error），引导模型续等。
    let cat = last_seen
        .or(expect)
        .unwrap_or_else(|| "unknown".to_string());
    let hint = relay
        .map(|m| format!("\nInstruction for the user: {m}"))
        .unwrap_or_default();
    let mode = if headless {
        "\nWARNING: browser is HEADLESS — the user has no visible window to act in. Relaunch in headed mode (MCP_BROWSER_HEADLESS=0) so the user can complete the action."
    } else {
        ""
    };
    Ok(text_content(format!(
        "status=still_waiting\nStill waiting for the user to complete '{cat}' in the visible browser window after {budget_ms} ms.{hint}{mode}\nCall wait_for_human again to keep waiting; it will return status=resolved as soon as the user finishes."
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
    with_timeout(ms, async {
        let params = ScreenshotParams::builder()
            .format(CaptureScreenshotFormat::Png)
            .full_page(full_page)
            .build();
        s.page
            .save_screenshot(params, &saved)
            .await
            .map_err(|e| format!("screenshot failed: {e}"))?;
        Ok(())
    })
    .await?;

    Ok(text_content(format!(
        "Saved screenshot to {} (full_page={full_page})",
        out_path.display()
    )))
}

/// 解析截图落盘路径：显式 path > MCP_BROWSER_SCREENSHOT_DIR > 系统临时目录/mcp_browser。
/// 目录不存在则创建；返回绝对路径。
fn resolve_screenshot_path(explicit: Option<String>) -> Result<PathBuf, JsonRpcErr> {
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
            lines.push(format!("#{i}: {url}"));
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
