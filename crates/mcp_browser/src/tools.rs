//! 工具 schema 声明 + `tools/call` 分发与各工具逻辑。
//!
//! 全部 12 个工具都通过 `content[0].text` 回传文本（宿主只读这一处）。
//! 每个 CDP 操作都包在 `with_timeout` 里，超时返回不含 transport 触发词的干净错误。

use std::path::PathBuf;

use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::page::ScreenshotParams;
use serde_json::{Value, json};

use crate::browser::{BrowserSession, ensure_session};
use crate::jsonrpc::{JsonRpcErr, cap_text, text_content, with_timeout};

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

/// tools/list 结果：12 个浏览器自动化工具的 schema。
pub fn tools_list_result() -> Value {
    json!({
        "tools": [
            {
                "name": "navigate",
                "description": "Launch/reuse a controlled Chrome and navigate to a URL, waiting for load. Keeps the session (cookies/login) alive across calls.",
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
                "description": "Extract visible text (innerText). Whole page (body) if no selector, else the matched element.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "Optional CSS selector; defaults to body" }
                    }
                }
            },
            {
                "name": "get_html",
                "description": "Extract HTML. Full document if no selector, else the matched element's outerHTML.",
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
        let title = s
            .page
            .get_title()
            .await
            .ok()
            .flatten()
            .unwrap_or_default();
        let final_url = s.page.url().await.ok().flatten().unwrap_or(url.clone());
        Ok(format!("Navigated to {final_url}\nTitle: {title}"))
    })
    .await?;

    Ok(text_content(summary))
}

async fn tool_click(session: &mut Option<BrowserSession>, args: &Value) -> Result<Value, JsonRpcErr> {
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
        el.click()
            .await
            .map_err(|e| format!("click failed: {e}"))?;
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
    let submit = args.get("submit").and_then(|v| v.as_bool()).unwrap_or(false);
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
        el.focus()
            .await
            .map_err(|e| format!("focus failed: {e}"))?;
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
        el.focus()
            .await
            .map_err(|e| format!("focus failed: {e}"))?;
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
                return Err(format!("selector '{selector}' not found within {timeout_ms} ms"));
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
        Ok(inner)
    })
    .await?;

    Ok(text_content(cap_text(&text)))
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
        match &selector {
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
        }
    })
    .await?;

    Ok(text_content(cap_text(&html)))
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
        std::fs::create_dir_all(parent)
            .map_err(|e| JsonRpcErr::new(-32000, &format!("cannot create screenshot dir: {e}"), None))?;
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
