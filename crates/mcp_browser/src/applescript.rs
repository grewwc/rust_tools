//! AppleScript driver mode: drives the **user's already-open Chrome** directly (the default driver).
//!
//! Why it exists: CDP can only attach to a "specially launched" Chrome, and Chrome 136+
//! refuses to enable `--remote-debugging-port` on the default profile, so a browser the
//! user opens in daily use cannot be attached via CDP. This module instead drives the
//! user's running Chrome through macOS Apple Events (`osascript` + Chrome's
//! `execute javascript`): it opens **a new tab in the session window** and reuses the
//! user's real cookies/login state throughout, **never quitting the user's browser**.
//!
//! Prerequisite: Chrome must have "View > Developer > Allow JavaScript from Apple Events"
//! enabled (off by default; enable once. When disabled, an actionable error is returned).
//!
//! Key differences from CDP mode:
//! - Every operation is an independent `osascript` subprocess (the only state kept is
//!   window id + tab id, dispatched by this process via Apple Events to the session
//!   window/tab; it never `activate`s -- activating would steal focus).
//! - All page interaction goes through JS (click/input/key are synthetic events; sites
//!   that rely on real keyboard/`hasFocus` may not work -- an inherent limitation of
//!   this mode).
//! - Screenshots use `screencapture -l <window-id>` (requires screen-recording permission);
//!   `full_page` automatically degrades to a window screenshot and falls back to
//!   rect/fullscreen so the model never fails just from passing the argument.
//!   A non-prompting TCC preflight (`CGPreflightScreenCaptureAccess`) runs first:
//!   without Screen Recording permission, `-l` always fails while the region/
//!   fullscreen fallbacks still "succeed" with a wallpaper-only image (no app
//!   windows), so failing fast with re-authorization guidance beats producing
//!   silently useless screenshots.
//! - If the user manually closes the session tab, later operations return an explicit
//!   "tab lost" error that `navigate` can rebuild from (`screenshot` excepted: when no
//!   window exists it automatically rebuilds via `about:blank`).
use std::process::Stdio;
use std::time::{Duration, Instant};

use mcp_stdio::{JsonRpcErr, cap_text, text_content, with_timeout};
use serde_json::{json, Value};
use tokio::process::Command;

use crate::tools::{op_timeout_ms, resolve_screenshot_path};

/// Non-blocking probe of macOS Screen Recording permission (TCC). Returns false
/// when the permission is absent or revoked (macOS 15+ re-verifies it monthly),
/// and never shows the authorization prompt -- unlike CGRequestScreenCaptureAccess.
#[cfg(target_os = "macos")]
fn screen_capture_allowed() -> bool {
    // Direct framework binding avoids pulling in an FFI bindings crate for one call.
    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGPreflightScreenCaptureAccess() -> bool;
    }
    unsafe { CGPreflightScreenCaptureAccess() }
}

/// Non-macOS builds compile this module too (cdp is their default driver and the
/// AppleScript driver is never selected there); report "allowed" so callers are
/// not blocked in theory-reachable-but-unused paths.
#[cfg(not(target_os = "macos"))]
fn screen_capture_allowed() -> bool {
    true
}

/// One screencapture invocation: Ok(()) on success, Err(stderr trimmed) on failure.
async fn screencapture_capture(args: &[&str]) -> Result<(), String> {
    let child = Command::new("/usr/sbin/screencapture")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("screencapture 启动失败: {}", e))?;
    let out = child
        .wait_with_output()
        .await
        .map_err(|e| format!("screencapture 失败: {}", e))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Every osascript call only sends Apple events and never activates Chrome (activating
/// would steal the user's screen focus). The only exception: when a new tab is created,
/// Chrome activates itself to the foreground and cannot be prevented at the source; see
/// "record frontmost app + restore focus after creation" in open_tab.

/// Read the (bundle id, display name) of the current frontmost app (queried via
/// lsappinfo, no system permissions required). Returns None on failure.
fn frontmost_app() -> Option<(String, String)> {
    let raw = std::process::Command::new("lsappinfo")
        .arg("front")
        .output()
        .ok()?;
    let asn = String::from_utf8_lossy(&raw.stdout).trim().to_string();
    if asn.is_empty() {
        return None;
    }
    // `-only <key>` output looks like "LSDisplayName"="Google Chrome";
    // always take the content of the first quoted pair after the first '='.
    let bid = ls_info(&asn, "bundleid");
    let name = ls_info(&asn, "name");
    match (bid, name) {
        (Some(b), Some(n)) => Some((b, n)),
        (Some(b), None) => Some((b, String::new())),
        (None, Some(n)) => Some((String::new(), n)),
        (None, None) => None,
    }
}

fn ls_info(asn: &str, key: &str) -> Option<String> {
    let out = std::process::Command::new("lsappinfo")
        .args(["info", "-only", key, asn])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let eq = s.find('=')?;
    let rest = &s[eq + 1..];
    let name = rest.split('"').nth(1).unwrap_or("");
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Give focus back to the previously recorded app; no Apple Events authorization is
/// needed (so no per-app automation permission dialog appears). Prefer bundle id
/// (`open -b`; e.g. the app whose Chinese display name is "Feishu" is actually
/// registered as "Lark", so looking it up by name would fail), falling back to the
/// display name via `open -a`.
/// Chrome always activates itself when creating a tab; this function restores focus
/// after creation. Failures are silently ignored (e.g. the app just quit; the next tab
/// creation retries). Not called when the user is already in Chrome.
fn restore_focus(prev: Option<&(String, String)>) {
    let Some((bid, name)) = prev else { return };
    if bid == "com.google.Chrome" || name == "Google Chrome" {
        return;
    }
    if !bid.is_empty() && std::process::Command::new("open").args(["-b", bid]).status().is_ok() {
        return;
    }
    if !name.is_empty() {
        let _ = std::process::Command::new("open").args(["-a", name]).status();
    }
}

/// Categorized "human action required" tag, in the same format as the detect logic in tools.rs.
pub fn user_action_tag(category: &str) -> String {
    format!(
        "\n[USER_ACTION_REQUIRED: {category}] 页面需要用户手动完成操作。可调用 wait_for_human 阻塞等待用户在可见浏览器窗口完成，或直接停止自动化并请用户完成后告知继续。"
    )
}

/// Warning prepended to mutating tools' output while a human action is pending (same as pending_warning in tools.rs).
pub fn pending_warning(session: &ApplescriptSession) -> String {
    match &session.pending_human {
        Some(cat) => format!("[HUMAN_ACTION_PENDING: {}] ", cat),
        None => String::new(),
    }
}

/// Human-verification detection with the same semantics as tools.rs: captcha/slider/
/// sms_otp/twofa/login_required/payment_verify/identity_verify. captcha means "present
/// and unsolved" (a dismissed popup does not count); text-only clues require a
/// keyword + input field to avoid false positives. The JS result is JSON.stringify'd.
pub const DETECT_USER_ACTION_JS: &str = r#"(function(){
  var vis = function(el){ return !!(el && el.offsetWidth && el.offsetHeight && el.getClientRects().length); };
  var out = null;
  try {
    var hash = location.hash;
    var jo = function(p){ if (location.hash.indexOf(p) >= 0) return true; if (document.body && document.body.innerHTML.indexOf(p) >= 0) return true; return false; };
    if (jo('sms_otp') || jo('twofa') || jo('two_fa') || jo('factor') || jo('otp')) out = out || 'sms_otp';
    if (jo('login_required') || jo('login-required') || jo('please login') || jo('请登录') || jo('登录后')) out = out || 'login_required';
    if (jo('payment_verify') || jo('payment-verify') || jo('verify payment') || jo('验证支付')) out = out || 'payment_verify';
    if (jo('identity_verify') || jo('identity-verify') || jo('verify identity') || jo('实名') || jo('身份验证')) out = out || 'identity_verify';
    var ifr = document.querySelectorAll('iframe[src*="recaptcha"], iframe[src*="hcaptcha"], iframe[src*="captcha"]');
    for (var i = 0; i < ifr.length; i++) {
      var f = ifr[i];
      var resp = f.contentDocument && (f.contentDocument.querySelector('textarea[name="g-recaptcha-response"], textarea[name="h-captcha-response"]') ||
                                       f.contentDocument.querySelector('[name="g-recaptcha-response"], [name="h-captcha-response"]'));
      var filled = resp && resp.value && resp.value.length > 0;
      if (!filled) { out = out || 'captcha'; break; }
    }
    if (!out) {
      var caps = document.querySelectorAll('[class*="captcha"], [id*="captcha"], [class*="Captcha"], [id*="Captcha"]');
      for (var j = 0; j < caps.length; j++) {
        var el = caps[j];
        if (!vis(el) || el.tagName === 'SCRIPT') continue;
        var txt = (el.innerText || '').toLowerCase();
        if (txt.indexOf('correct') >= 0 || txt.indexOf('solved') >= 0 || txt.indexOf('success') >= 0) continue;
        out = 'captcha'; break;
      }
    }
    if (!out) {
      var inputs = document.querySelectorAll('input[type="text"], input:not([type]), input[type="password"], textarea');
      var hits = {
        captcha: /captcha|验证码|图形验证|人机验证|点击.?验证/,
        slider: /slider|drag|滑动|滑块|拖动/,
        sms_otp: /sms|otp|verification code|验证码|短信/,
        twofa: /two.?factor|2fa|authenticator|双重验证|二次验证/,
        login_required: /login|sign.?in|登录|登入/,
        payment_verify: /payment|支付|verif/,
        identity_verify: /identity|real.?name|实名|身份/
      };
      var cats = ['captcha','slider','sms_otp','twofa','login_required','payment_verify','identity_verify'];
      for (var k = 0; k < inputs.length; k++) {
        var inp = inputs[k];
        var ph = ((inp.placeholder || '') + ' ' + (inp.getAttribute('aria-label') || '') + ' ' + (inp.name || '')).toLowerCase();
        for (var c = 0; c < cats.length; c++) {
          if (hits[cats[c]].test(ph)) { out = out || cats[c]; }
        }
      }
    }
  } catch (e) {}
  return out ? out : null;
})()"#;

/// Run a block of osascript (passed via stdin); returns the last stdout line (trimmed).
async fn run_osascript(script: &str) -> Result<String, String> {
    let mut child = Command::new("/usr/bin/osascript")
        .arg("-") // Read the script from stdin to avoid command-line escaping hell.
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("osascript spawn failed: {}", e))?;
    {
        use tokio::io::AsyncWriteExt;
        let mut stdin = child.stdin.take().ok_or("osascript stdin missing")?;
        stdin
            .write_all(script.as_bytes())
            .await
            .map_err(|e| format!("osascript stdin write failed: {}", e))?;
        stdin.shutdown().await.ok();
    }
    let out = child.wait_with_output().await.map_err(|e| format!("osascript failed: {}", e))?;
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    if !out.status.success() {
        // Contains "applescript" (supports URL) => "Allow JavaScript from Apple Events" is off
        if stderr.to_lowercase().contains("applescript") {
            return Err(format!(
                "Chrome 未开启 AppleScript JavaScript 执行（execute javascript 被拒绝）。\
                 请先在 Chrome 菜单 View > Developer > Allow JavaScript from Apple Events \
                 （显示 > 开发者 > 允许来自 Apple 事件的 JavaScript）开启后重试。原始错误：{}",
                stderr.trim()
            ));
        }
        return Err(stderr.trim().to_string());
    }
    Ok(stdout.trim().to_string())
}

/// Escape text for embedding in an AppleScript string literal (\\, \", newlines, etc.).
fn esc_applescript(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Session: holds the window id + tab id inside the user's already-open Chrome.
#[derive(Debug, Default)]
pub struct ApplescriptSession {
    pub window_id: Option<String>,
    pub tab_id: Option<String>,
    pub pending_human: Option<String>,
}

impl ApplescriptSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute JS on the current session tab; the function body must return a string (usually JSON.stringify).
    async fn js_on_session_tab(&self, js: &str) -> Result<String, String> {
        let (wid, tid) = match (&self.window_id, &self.tab_id) {
            (Some(w), Some(t)) => (w.clone(), t.clone()),
            _ => return Err("还没有会话标签页：请先调用 navigate 打开一个页面".to_string()),
        };
        let script = format!(
            "tell application \"Google Chrome\"\n  execute (first tab of window id {} whose id is \"{}\") javascript \"{}\"\nend tell",
            wid, tid, esc_applescript(js)
        );
        run_osascript(&script).await.map_err(|e| translate_err(&e))
    }

    /// Execute JS on the session tab with a timeout (both timeouts and errors become readable messages).
    async fn exec_js(&self, js: &str) -> Result<String, String> {
        let js = js.to_string();
        let timeout = op_timeout_ms();
        match with_timeout(timeout, self.js_on_session_tab(&js)).await {
            Ok(inner) => Ok(inner),
            Err(e) => Err(format!(
                "执行 JS 失败(超过 {}ms 上限或 osascript 出错): {}",
                timeout, e.message
            )),
        }
    }

    /// Navigate: create a session tab if none exists (first tab), otherwise reuse the
    /// existing tab and `set URL`. Then wait for the page to finish loading; if
    /// wait_selector is given, also wait for the element to appear.
    pub async fn navigate(&mut self, url: &str, wait_selector: Option<&str>) -> Result<(), String> {
        self.pending_human = None;
        if self.tab_id.is_some() && self.tab_alive().await {
            let (wid, tid) = (
                self.window_id.clone().unwrap(),
                self.tab_id.clone().unwrap(),
            );
            let script = format!(
            "tell application \"Google Chrome\"\n  set URL of (first tab of window id {} whose id is \"{}\") to \"{}\"\nend tell",
            wid, tid, esc_applescript(url)
            );
            run_osascript(&script).await?;
        } else {
            self.open_tab(url).await?;
        }
        // Wait for loading to finish (up to ~20s; keep going past the deadline since pages may load slowly due to ads).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let state = self.exec_js("document.readyState").await.unwrap_or_default();
            if state.trim() == "complete" || std::time::Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }
        if let Some(sel) = wait_selector {
            self.wait_for_selector(sel, 30_000).await.map_err(|e| {
                format!("navigate 等待元素超时：{e}（可忽略后继续，用 wait_for 重试）")
            })?;
        }
        Ok(())
    }

    /// The (url, title) of the current session page.
    pub async fn url_title(&self) -> Result<(String, String), String> {
        let out = self
            .exec_js("JSON.stringify({url: location.href, title: document.title})")
            .await?;
        let v: Value = serde_json::from_str(out.trim())
            .unwrap_or_else(|_| json!({ "url": "", "title": "" }));
        Ok((
            v["url"].as_str().unwrap_or("").to_string(),
            v["title"].as_str().unwrap_or("").to_string(),
        ))
    }

    /// Create a new tab for the session in the user's Chrome (first time or when the tab is lost).
    /// Returns (window_id, tab_id).
    ///
    /// Focus handling: Chrome always activates itself when it receives a "new tab" event
    /// (verified: both `make new tab` and `open -g` steal focus and cannot be prevented at
    /// the source), so we record the frontmost app before creating and immediately hand
    /// focus back with `open -a` afterwards, so the user's screen is not locked to the new
    /// tab. No restore is done if creation fails or the user is already in Chrome.
    async fn open_tab(&mut self, url: &str) -> Result<(String, String), String> {
        let prev = frontmost_app();
        let script = format!(
            r#"tell application "Google Chrome"
  if (count of windows) is 0 then
    make new window
  end if
  set w to front window
  set t to make new tab at end of tabs of w with properties {{URL:"{}"}}
  set active tab index of w to (count of tabs of w)
  return (id of w) & "|" & (id of t)
end tell"#,
            esc_applescript(url)
        );
        let out = run_osascript(&script).await?;
        let mut parts = out.split('|');
        match (parts.next(), parts.next()) {
            (Some(w), Some(t)) if !w.is_empty() && !t.is_empty() => {
                self.window_id = Some(w.to_string());
                self.tab_id = Some(t.to_string());
                restore_focus(prev.as_ref());
                Ok((w.to_string(), t.to_string()))
            }
            _ => Err(format!("新建标签页失败：无法解析返回 ({})", out)),
        }
    }

    /// Whether the tab still exists.
    async fn tab_alive(&self) -> bool {
        let (wid, tid) = match (&self.window_id, &self.tab_id) {
            (Some(w), Some(t)) => (w.clone(), t.clone()),
            _ => return false,
        };
        let script = format!(
            "tell application \"Google Chrome\"\n  exists (first tab of window id {} whose id is \"{}\")\nend tell",
            wid, tid
        );
        run_osascript(&script).await.map(|s| s.trim() == "true").unwrap_or(false)
    }

    /// List the tabs in the session window (#idx: url [active]).
    pub async fn list_tabs(&self) -> Result<String, String> {
        let wid = match &self.window_id {
            Some(w) => w.clone(),
            None => return Ok("(no session window)".to_string()),
        };
        let script = format!(
            r#"tell application "Google Chrome"
  set ws to URL of every tab of window id {}
  set ais to active tab index of window id {}
  set out to ""
  repeat with i from 1 to (count of ws)
    set u to item i of ws
    if i is ais then
      set out to out & (i as text) & ": " & u & " [active]" & linefeed
    else
      set out to out & (i as text) & ": " & u & linefeed
    end if
  end repeat
  return out
end tell"#,
            wid, wid
        );
        let out = run_osascript(&script).await?;
        Ok(out.trim().to_string())
    }

    /// Close the session tab (does not quit Chrome or touch other windows/tabs).
    pub async fn close_session_tab(&mut self) -> Result<(), String> {
        if let (Some(w), Some(t)) = (&self.window_id.clone(), &self.tab_id.clone()) {
            let script = format!(
            "tell application \"Google Chrome\"\n  close (first tab of window id {} whose id is \"{}\")\nend tell",
            w, t
            );
            if run_osascript(&script).await.is_ok() {
                self.window_id = None;
                self.tab_id = None;
                self.pending_human = None;
                return Ok(());
            }
        }
        // The tab is already gone (closed by the user) -- treat it as "closed" and just clear the state.
        self.window_id = None;
        self.tab_id = None;
        self.pending_human = None;
        Ok(())
    }

    /// Extract page text (selector optional, defaults to body); append the detection tag after capping.
    pub async fn get_text_sel(&self, selector: Option<&str>) -> Result<String, String> {
        let js = match selector {
            Some(sel) => format!(
                "JSON.stringify(document.querySelector({}) ? document.querySelector({}).innerText : null)",
                json!(sel),
                json!(sel)
            ),
            None => "JSON.stringify(document.body ? document.body.innerText : '')".to_string(),
        };
        let raw = self.exec_js(&js).await?;
        let inner = unquote(&raw);
        if inner == "null" {
            return Err(format!("未找到选择器 {}", selector.unwrap_or("(body)")));
        }
        Ok(cap_text(&inner))
    }

    /// Extract page HTML (selector optional, defaults to documentElement); append the detection tag after capping.
    pub async fn get_html_sel(&self, selector: Option<&str>) -> Result<String, String> {
        let js = match selector {
            Some(sel) => format!(
                "JSON.stringify(document.querySelector({}) ? document.querySelector({}).outerHTML : null)",
                json!(sel),
                json!(sel)
            ),
            None => "JSON.stringify(document.documentElement.outerHTML)".to_string(),
        };
        let raw = self.exec_js(&js).await?;
        let inner = unquote(&raw);
        if inner == "null" {
            return Err(format!("未找到选择器 {}", selector.unwrap_or("(document)")));
        }
        Ok(cap_text(&inner))
    }

    /// Execute an arbitrary JS expression in the page (result returned after JSON.stringify).
    pub async fn evaluate_js(&self, expr: &str) -> Result<String, String> {
        let js = format!("JSON.stringify((() => {{ try {{ return ({}); }} catch(e) {{ return 'JS_ERROR: ' + e; }} }})())", expr);
        // User expressions may contain newlines (multi-line expressions are legal); join the lines into a single line.
        let js = js.lines().collect::<Vec<_>>().join(" ");
        let raw = self.exec_js(&js).await?;
        Ok(unquote(&raw))
    }

    /// Wait for an element to appear (polling); returns the number of milliseconds waited.
    pub async fn wait_for_selector(&self, selector: &str, timeout_ms: u64) -> Result<u64, String> {
        let sel = selector.to_string();
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            let js = format!("JSON.stringify(!!document.querySelector({}))", json!(&sel));
            let raw = self.exec_js(&js).await?;
            if unquote(&raw) == "true" {
                return Ok(deadline.elapsed().as_millis() as u64);
            }
            if Instant::now() >= deadline {
                return Err(format!("等待超时：{}ms 内未找到选择器 '{}'", timeout_ms, selector));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    /// Scroll: x/y are window coordinates, or scroll a selector into view.
    pub async fn scroll(&self, x: Option<i64>, y: Option<i64>, selector: Option<&str>) -> Result<(), String> {
        let js = if let Some(sel) = selector {
            format!(
                "(() => {{ const el = document.querySelector({}); if (!el) return 'not found'; el.scrollIntoView({{block:'center'}}); return 'ok'; }})()",
                json!(sel)
            )
        } else {
            let (x, y) = (x.unwrap_or(0), y.unwrap_or(0));
            format!("(() => {{ window.scrollTo({}, {}); return 'ok'; }})()", x, y)
        };
        let raw = self.exec_js(&js).await?;
        if unquote(&raw) == "not found" {
            return Err(format!("未找到滚动目标 {}", selector.unwrap_or("")));
        }
        Ok(())
    }

    /// Click (synthetic click event + native setter input); returns the output text.
    pub async fn click(&self, selector: &str) -> Result<String, String> {
        let sel = selector.to_string();
        let js = format!(
            "(() => {{ const el = document.querySelector({}); if (!el) return JSON.stringify({{ok:false,error:'not found'}}); \
              el.scrollIntoView({{block:'center'}}); \
              el.dispatchEvent(new MouseEvent('mousedown', {{bubbles:true, cancelable:true}})); \
              el.dispatchEvent(new MouseEvent('mouseup', {{bubbles:true, cancelable:true}})); \
              el.click(); return JSON.stringify({{ok:true}}); }})()",
            json!(&sel)
        );
        let raw = self.exec_js(&js).await?;
        if unquote(&raw).contains("\"ok\":false") {
            return Err(format!("未找到可点击元素 {}", selector));
        }
        Ok(format!("Clicked {}", selector))
    }

    /// Type text (native setter + input/change events; can submit the form).
    pub async fn type_text(&self, selector: &str, text: &str, submit: bool) -> Result<String, String> {
        let sel = selector.to_string();
        let text_json = json!(text).to_string();
        let submit_js = if submit {
            " if (el.form) { try { el.form.requestSubmit(); } catch(e) {} }"
        } else {
            ""
        };
        let js = format!(
            "(() => {{ const el = document.querySelector({}); if (!el) return JSON.stringify({{ok:false}}); \
              el.scrollIntoView({{block:'center'}}); el.focus(); \
              const proto = (el.tagName === 'TEXTAREA') ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype; \
              const setter = Object.getOwnPropertyDescriptor(proto, 'value').set; \
              setter.call(el, {}); \
              el.dispatchEvent(new Event('input', {{bubbles:true}})); \
              el.dispatchEvent(new Event('change', {{bubbles:true}}));{} \
              return JSON.stringify({{ok:true}}); }})()",
            json!(&sel),
            text_json,
            submit_js
        );
        let raw = self.exec_js(&js).await?;
        if unquote(&raw).contains("\"ok\":false") {
            return Err(format!("未找到输入框 {}", selector));
        }
        Ok(format!(
            "Typed {} chars into {}{}",
            text.chars().count(),
            selector,
            if submit { " and pressed Enter" } else { "" }
        ))
    }

    /// Press a key (synthetic keydown/keyup; Enter auto-submits when inside a form).
    pub async fn press_key(&self, key: &str, selector: Option<&str>) -> Result<String, String> {
        let focus_js = match selector {
            Some(sel) => format!(
                "const el = document.querySelector({}) || document.activeElement; if (el) {{ el.focus(); el.scrollIntoView({{block:'center'}}); }}",
                json!(sel)
            ),
            None => "const el = document.activeElement;".to_string(),
        };
        let js = format!(
            "(() => {{ {} const el2 = document.activeElement; if (!el2) return JSON.stringify({{ok:false}}); \
              const kd = new KeyboardEvent('keydown', {{key:'{}', code:'{}', bubbles:true, cancelable:true}}); \
              el2.dispatchEvent(kd); \
              if ({} && !kd.defaultPrevented && el2.form) {{ try {{ el2.form.requestSubmit(); }} catch(e) {{}} }} \
              el2.dispatchEvent(new KeyboardEvent('keyup', {{key:'{}', code:'{}', bubbles:true}})); \
              return JSON.stringify({{ok:true}}); }})()",
            focus_js,
            esc_js_key(key),
            esc_js_key(key),
            if key.eq_ignore_ascii_case("enter") { "true" } else { "false" },
            esc_js_key(key),
            esc_js_key(key)
        );
        let raw = self.exec_js(&js).await?;
        if unquote(&raw).contains("\"ok\":false") {
            return Err("没有可接收按键的焦点元素".to_string());
        }
        Ok(format!("Pressed {}", key))
    }

    /// Human-verification detection (JS with the same semantics); returns the category or None.
    pub async fn detect_user_action_required(&self) -> Option<String> {
        let js = format!("JSON.stringify({})", DETECT_USER_ACTION_JS);
        let raw = self.exec_js(&js).await.ok()?;
        let v = unquote(&raw);
        if v.is_empty() || v == "null" || v.starts_with("JS_ERROR") {
            None
        } else {
            Some(v)
        }
    }

    /// Wait for the user to handle it manually (same contract: status=resolved /
    /// still_waiting; 60s budget, resolved only after 2 consecutive clean polls).
    pub async fn wait_for_human_tool(&mut self, expect: Option<&str>) -> String {
        let op_cap = op_timeout_ms();
        let ceiling = op_cap.saturating_sub(15_000).max(5_000);
        let budget_ms = 60_000u64.min(ceiling);
        let deadline = Instant::now() + Duration::from_millis(budget_ms);
        let mut clean_streak: u32 = 0;
        let mut note = String::new();
        let cat = expect
            .map(|e| e.trim().to_string())
            .filter(|e| !e.is_empty());
        loop {
            // Each poll has its own timeout: a hung poll counts as "still blocked".
            let poll = self.detect_user_action_required_guarded().await;
            match poll {
                None => {
                    clean_streak += 1;
                    if clean_streak >= 2 {
                        self.pending_human = None;
                        return format!("status=resolved\nYou may now continue browser automation.\n{}", note.trim());
                    }
                }
                Some(c) => {
                    clean_streak = 0;
                    note = format!("current blocker: [USER_ACTION_REQUIRED: {}]", c);
                }
            }
            if Instant::now() >= deadline {
                let hint = cat
                    .clone()
                    .map(|c| format!(" (expect: {})", c))
                    .unwrap_or_default();
                self.pending_human = expect.map(|e| e.trim().to_string()).filter(|e| !e.is_empty());
                return format!(
                    "status=still_waiting\nStill waiting for the user to complete '{}' in the visible browser window after {} ms.{}",
                    cat.unwrap_or_default(),
                    budget_ms,
                    hint
                );
            }
            tokio::time::sleep(Duration::from_millis(2000)).await;
        }
    }

    /// Detection with a 5s poll timeout: a hung poll is treated as "still blocked"
    /// (anything other than None is not judged resolved).
    async fn detect_user_action_required_guarded(&self) -> Option<String> {
        match with_timeout(5000, async {
            Ok::<_, String>(self.detect_user_action_required().await)
        })
        .await
        {
            Ok(Some(c)) => Some(c),
            Ok(None) => None,
            Err(_) => Some("poll_timeout".to_string()),
        }
    }

    /// Query the Chrome window bounds: returns (x, y, w, h), or None on failure (e.g. window closed).
    async fn window_bounds(&self, wid: &str) -> Option<(i32, i32, i32, i32)> {
        let script = format!(
            "tell application \"Google Chrome\"\n  get bounds of window id {}\nend tell",
            wid
        );
        let out = run_osascript(&script).await.ok()?;
        // Looks like "0, 44, 1440, 878" or "{0, 44, 1440, 878}".
        let s = out.trim().trim_matches(|c| c == '{' || c == '}').trim().to_string();
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 4 {
            return None;
        }
        let parse = |p: &str| p.trim().parse::<i32>().ok();
        let x1 = parse(parts[0])?;
        let y1 = parse(parts[1])?;
        let x2 = parse(parts[2])?;
        let y2 = parse(parts[3])?;
        let w = x2 - x1;
        let h = y2 - y1;
        if w <= 0 || h <= 0 {
            return None;
        }
        Some((x1, y1, w, h))
    }

    /// Screenshot: prefer `screencapture -l <wid>`; on failure use `bounds -> -R`, then
    /// fall back to fullscreen. Under AppleScript, `full_page` automatically degrades to
    /// a window screenshot (returns a warning instead of Err to avoid a model retry loop).
    /// Takes `&mut self` so it can auto-`open_tab("about:blank")` when no window exists.
    pub async fn screenshot(&mut self, path: &str, full_page: bool) -> Result<(String, String), String> {
        // Gate every capture attempt behind the Screen Recording preflight. Without
        // it `screencapture -l` fails outright ("could not create image from
        // window") and the later rect/fullscreen fallbacks exit 0 yet emit a
        // wallpaper-only image the model mistakes for real page content.
        if !screen_capture_allowed() {
            return Err(
                "屏幕录制权限未授予或已被系统收回，无法捕获 Chrome 窗口内容。请到 系统设置 > \
                 隐私与安全性 > 屏幕录制 勾选运行本工具的宿主应用（Terminal/iTerm/VS Code/Cursor 等），\
                 然后完全退出并重新打开该应用使授权生效（macOS 15+ 每月会重新确认该授权，\
                 被撤销时截图会突然全部失败或只截到桌面壁纸）。生效后重试；\
                 若想避开系统授权依赖，可改用受控实例驱动：MCP_BROWSER_DRIVER=cdp"
                    .to_string(),
            );
        }
        let full_page_warn = if full_page {
            " [warn: AppleScript 模式不支持 full_page，已按窗口截图]"
        } else {
            ""
        };
        // When there is no window or the tab was lost, auto-create a blank tab (no navigate needed before the first screenshot).
        let needs_open = match (&self.window_id, &self.tab_id) {
            (Some(_), Some(_)) => !self.tab_alive().await,
            _ => true,
        };
        if needs_open {
            if let Err(e) = self.open_tab("about:blank").await {
                return Err(format!(
                    "还没有会话窗口且自动创建失败：{}（请先调用 navigate）",
                    translate_err(&e)
                ));
            }
        }
        let wid = self
            .window_id
            .clone()
            .ok_or("还没有会话窗口：请先调用 navigate")?;
        let parent = std::path::Path::new(path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        std::fs::create_dir_all(&parent).map_err(|e| format!("创建截图目录失败: {}", e))?;
        // 1) Window-level capture (most precise; shadows/rounded corners are handled by the system).
        if screencapture_capture(&["-l", &wid, "-x", path]).await.is_ok() {
            // Double-check the file actually landed and is non-empty (may be 0 bytes when the window is hidden).
            if std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false) {
                return Ok((path.to_string(), full_page_warn.to_string()));
            }
        }
        // 2) rect fallback: convert Chrome bounds to -R x,y,w,h.
        if let Some((x, y, w, h)) = self.window_bounds(&wid).await {
            let rect = format!("{},{},{},{}", x, y, w, h);
            if screencapture_capture(&["-R", &rect, "-x", path]).await.is_ok()
                && std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false)
            {
                let note = format!("{} [fallback: rect {}]", full_page_warn, rect);
                return Ok((path.to_string(), note));
            }
        }
        // 3) fullscreen fallback (still produces an image when the window is invisible/hidden, avoiding total failure).
        if screencapture_capture(&["-x", path]).await.is_ok()
            && std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false)
        {
            let note = format!("{} [fallback: fullscreen - 窗口不可见已退化为全屏]", full_page_warn);
            return Ok((path.to_string(), note));
        }
        Err(format!(
            "screencapture 失败（窗口 {} 无法捕获）。请检查：1) 系统设置>隐私与安全性>屏幕录制，已勾选启动本工具的终端应用（Terminal/iTerm/VS Code/Cursor/Arc 等，授权后需重启该应用）；2) Chrome 窗口保持可见、未最小化且不在别的 Space/Stage Manager 隐藏中。重试或切 CDP 模式：MCP_BROWSER_DRIVER=cdp",
            wid
        ))
    }
}

/// osascript error translation: give actionable hints for lost session tab/window errors.
fn translate_err(e: &str) -> String {
    let le = e.to_lowercase();
    if le.contains("applescript") && (le.contains("javascript") || le.contains("java script")) {
        "Chrome 未开启“Apple 事件中的 JavaScript”：请在 Chrome 菜单 视图(View) > 开发者(Developer) > \
         允许 Apple 事件中的 JavaScript(Allow JavaScript from Apple Events) 打开后重试（一次性设置）。"
            .to_string()
    } else if le.contains("不能获得") || le.contains("can't get") || le.contains("-1728") {
        "会话标签页/窗口已丢失（可能被手动关闭或切到别的 Chrome 窗口/实例）。\
         请重新调用 navigate 打开新标签页，或把浏览器窗口切回会话窗口。"
            .to_string()
    } else {
        e.to_string()
    }
}

/// Strip the outer quotes from a JSON.stringify result (keeping the escaped content verbatim).
fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn esc_js_key(k: &str) -> String {
    k.replace('\\', "\\\\").replace('\'', "\\'").replace('"', "\\\"")
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn require_str(args: &Value, key: &str) -> Result<String, JsonRpcErr> {
    str_arg(args, key)
        .map(str::to_string)
        .ok_or_else(|| JsonRpcErr::new(-32602, &format!("missing '{key}'"), None))
}

fn opt_str(args: &Value, key: &str) -> Option<String> {
    str_arg(args, key).map(str::to_string)
}

impl ApplescriptSession {
    /// Session teardown: only close the session tab, never quit the user's browser.
    pub async fn shutdown(&mut self) {
        let _ = self.close_session_tab().await;
        self.window_id = None;
        self.tab_id = None;
    }
}

/// The 13 tools with the same names and argument keys as in tools.rs; results go only
/// into content[0].text (capped at 24K), using the same `[...]` markers as CDP mode to
/// remind the model.
pub async fn handle_tools_call(
    session: &mut ApplescriptSession,
    params: Option<Value>,
) -> Result<Value, JsonRpcErr> {
    let params = params.unwrap_or_else(|| json!({}));
    let cmd = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| JsonRpcErr::new(-32602, "missing 'name'", None))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let ms = op_timeout_ms();

    match cmd {
        "navigate" => {
            let url = require_str(&args, "url")?;
            let wait_selector = opt_str(&args, "wait_selector");
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                session.navigate(&url, wait_selector.as_deref()).await?;
                let (u, t) = session.url_title().await?;
                let tag = session.detect_user_action_required().await.map_or(String::new(), |c| user_action_tag(&c));
                Ok(format!("{pend}Navigated to {u}\nTitle: {t}{tag}"))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "click" => {
            let sel = require_str(&args, "selector")?;
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                session.click(&sel).await?;
                let tag = session.detect_user_action_required().await.map_or(String::new(), |c| user_action_tag(&c));
                Ok(format!("{pend}Clicked {sel}{tag}"))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "type_text" => {
            let sel = require_str(&args, "selector")?;
            let text = require_str(&args, "text")?;
            let submit = args.get("submit").and_then(Value::as_bool).unwrap_or(false);
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                let n = session.type_text(&sel, &text, submit).await?;
                let tag = session.detect_user_action_required().await.map_or(String::new(), |c| user_action_tag(&c));
                let suffix = if submit { " and pressed Enter" } else { "" };
                Ok(format!("{pend}Typed {n} chars into {sel}{suffix}{tag}"))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "press_key" => {
            let key = require_str(&args, "key")?;
            let sel = opt_str(&args, "selector");
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                session.press_key(&key, sel.as_deref()).await?;
                let tag = session.detect_user_action_required().await.map_or(String::new(), |c| user_action_tag(&c));
                Ok(format!("{pend}Pressed {key}{tag}"))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "scroll" => {
            let sel = opt_str(&args, "selector");
            let x = args.get("x").and_then(Value::as_f64);
            let y = args.get("y").and_then(Value::as_f64);
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                session
                    .scroll(x.map(|v| v as i64), y.map(|v| v as i64), sel.as_deref())
                    .await?;
                let tag = session.detect_user_action_required().await.map_or(String::new(), |c| user_action_tag(&c));
                Ok(match &sel {
                    Some(s) => format!("{pend}Scrolled {s} into view{tag}"),
                    None => format!("{pend}Scrolled window to ({x:?}, {y:?}){tag}"),
                })
            })
            .await?;
            Ok(text_content(&summary))
        }
        "wait_for" => {
            let sel = require_str(&args, "selector")?;
            let timeout_ms = args
                .get("timeout_ms")
                .and_then(Value::as_u64)
                .unwrap_or(10_000);
            let summary = with_timeout(ms, async {
                let waited = session.wait_for_selector(&sel, timeout_ms).await?;
                let tag = session.detect_user_action_required().await.map_or(String::new(), |c| user_action_tag(&c));
                Ok(format!("Found {sel} after {waited} ms{tag}"))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "evaluate_js" => {
            let expr = require_str(&args, "expression")?;
            let out = with_timeout(ms, async { session.evaluate_js(&expr).await }).await?;
            Ok(text_content(&cap_text(&out)))
        }
        "get_text" => {
            let sel = opt_str(&args, "selector");
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                let mut t = session.get_text_sel(sel.as_deref()).await?;
                if let Some(c) = session.detect_user_action_required().await {
                    t.push_str(&user_action_tag(&c));
                }
                Ok(format!("{pend}{t}"))
            })
            .await?;
            Ok(text_content(&cap_text(&summary)))
        }
        "get_html" => {
            let sel = opt_str(&args, "selector");
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                let mut t = session.get_html_sel(sel.as_deref()).await?;
                if let Some(c) = session.detect_user_action_required().await {
                    t.push_str(&user_action_tag(&c));
                }
                Ok(format!("{pend}{t}"))
            })
            .await?;
            Ok(text_content(&cap_text(&summary)))
        }
        "screenshot" => {
            let path = opt_str(&args, "path");
            let out_path = resolve_screenshot_path(path)?;
            let full_page = args.get("full_page").and_then(Value::as_bool).unwrap_or(false);
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                let (p, extra) = session.screenshot(&out_path.to_string_lossy(), full_page).await?;
                let tag = session.detect_user_action_required().await.map_or(String::new(), |c| user_action_tag(&c));
                Ok(format!("{pend}Saved screenshot to {p} (full_page={full_page}){extra}{tag}"))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "wait_for_human" => {
            let expect = opt_str(&args, "expect");
            let summary = with_timeout(ms, async {
                Ok::<_, String>(session.wait_for_human_tool(expect.as_deref()).await)
            })
            .await?;
            Ok(text_content(&summary))
        }
        "list_tabs" => {
            let r = with_timeout(ms, async { session.list_tabs().await }).await?;
            Ok(text_content(&cap_text(&r)))
        }
        "close_browser" => {
            let r = with_timeout(ms, async {
                if session.window_id.is_none() {
                    Ok("No active browser session".to_string())
                } else {
                    session.close_session_tab().await?;
                    session.window_id = None;
                    session.tab_id = None;
                    Ok("Browser closed".to_string())
                }
            })
            .await?;
            Ok(text_content(&r))
        }
        _ => Err(JsonRpcErr::new(-32601, &format!("unknown tool: {cmd}"), None)),
    }
}