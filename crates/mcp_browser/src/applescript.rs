//! AppleScript driver mode: drives the **user's already-open Chrome** directly (explicit opt-in).
//!
//! Why it exists: CDP can only attach to a "specially launched" Chrome, and Chrome 136+
//! refuses to enable `--remote-debugging-port` on the default profile, so a browser the
//! user opens in daily use cannot be attached via CDP. This module instead drives the
//! user's running Chrome through macOS Apple Events (`osascript` + Chrome's
//! `execute javascript`): it opens **a new tab in the session window** and reuses the
//! user's real cookies/login state throughout, **never quitting the user's browser**.
//!
//! Prerequisite: Chrome must have "View > Developer > Allow JavaScript from Apple Events"
//! enabled (off by default). Chrome reads that preference **at startup only**: ticking the
//! menu item in a running browser (or editing the profile on disk) does not change the
//! running process, so the gate stays closed until Chrome is fully quit and restarted.
//! The error returned in that case names the restart step explicitly.
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
use serde_json::{Value, json};
use tokio::process::Command;

use crate::contract::{opt_str, require_str};
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
/// (`open -b`; a display name lookup can fail when the registered name differs
/// from the localized one), falling back to the display name via `open -a`.
/// Chrome always activates itself when creating a tab; this function restores focus
/// after creation. Failures are silently ignored (e.g. the app just quit; the next tab
/// creation retries). Not called when the user is already in Chrome.
fn restore_focus(prev: Option<&(String, String)>) {
    let Some((bid, name)) = prev else { return };
    if bid == "com.google.Chrome" || name == "Google Chrome" {
        return;
    }
    if !bid.is_empty()
        && std::process::Command::new("open")
            .args(["-b", bid])
            .status()
            .is_ok()
    {
        return;
    }
    if !name.is_empty() {
        let _ = std::process::Command::new("open")
            .args(["-a", name])
            .status();
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

/// Run a block of osascript (passed via stdin); returns the last stdout line (trimmed).
async fn run_osascript(script: &str) -> Result<String, String> {
    let mut child = Command::new("/usr/bin/osascript")
        .kill_on_drop(true)
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
    let out = child
        .wait_with_output()
        .await
        .map_err(|e| format!("osascript failed: {}", e))?;
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    if !out.status.success() {
        // Chrome refuses `execute javascript` while "Allow JavaScript from Apple Events"
        // is off; the refusal names AppleScript and JavaScript in every locale.
        if is_js_apple_events_disabled(&stderr) {
            return Err(js_apple_events_message(&stderr));
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
    owned_tabs: Vec<(String, String)>,
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
            wid,
            tid,
            esc_applescript(js)
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
                wid,
                tid,
                esc_applescript(url)
            );
            run_osascript(&script).await?;
        } else {
            self.open_tab(url).await?;
        }
        // Wait for loading to finish (up to ~20s; keep going past the deadline since pages may load slowly due to ads).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match self.exec_js("document.readyState").await {
                Ok(state) => {
                    if state.trim() == "complete" || std::time::Instant::now() >= deadline {
                        break;
                    }
                }
                // The JS gate stays closed until Chrome quits and restarts: retrying
                // for 20s cannot help, so fail fast with the actionable hint.
                Err(e) if is_gate_closed_message(&e) => return Err(e),
                Err(_) => {
                    // Transient page error (mid-navigation, no document yet): keep waiting.
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }
        if let Some(sel) = wait_selector {
            self.wait_for_selector(sel, "present", op_timeout_ms().saturating_sub(1_000))
                .await
                .map_err(|e| {
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
        let v: Value =
            serde_json::from_str(out.trim()).unwrap_or_else(|_| json!({ "url": "", "title": "" }));
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
                self.owned_tabs.push((w.to_string(), t.to_string()));
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
        run_osascript(&script)
            .await
            .map(|s| s.trim() == "true")
            .unwrap_or(false)
    }

    /// Stable window/tab ids survive reordering; the index is only for reading.
    pub async fn list_tabs(&self) -> Result<String, String> {
        if self.window_id.is_none() {
            return Ok("(no session window)".into());
        }
        let script = r##"tell application "Google Chrome"
  set out to ""
  set i to 0
  repeat with w in windows
    repeat with t in tabs of w
      set out to out & "#" & (i as text) & " id=" & (id of w as text) & ":" & (id of t as text) & ": " & (URL of t) & linefeed
      set i to i + 1
    end repeat
  end repeat
  return out
end tell"##;
        let mut out = run_osascript(script).await?;
        if let (Some(w), Some(t)) = (&self.window_id, &self.tab_id) {
            out.push_str(&format!("\ncurrent_id={w}:{t}"));
        }
        Ok(out)
    }

    pub async fn switch_tab(&mut self, id: &str) -> Result<String, String> {
        let (window, tab) = parse_tab_id(id)?;
        let script = format!(
            "tell application \"Google Chrome\"\n exists (first tab of window id {window} whose id is \"{tab}\")\nend tell"
        );
        if run_osascript(&script).await? != "true" {
            return Err(format!("unknown tab id '{id}'"));
        }
        // Change only the automation target; selecting a visible Chrome tab
        // would disrupt the user's current page.
        self.window_id = Some(window.to_string());
        self.tab_id = Some(tab.to_string());
        self.pending_human = None;
        Ok(format!("Selected tab {id}"))
    }

    pub async fn close_tab(&mut self, id: &str) -> Result<String, String> {
        let (window, tab) = parse_tab_id(id)?;
        if !self.owned_tabs.iter().any(|(w, t)| w == window && t == tab) {
            return Err("cannot close a tab this session did not create".into());
        }
        let script = format!(
            "tell application \"Google Chrome\"\n close (first tab of window id {window} whose id is \"{tab}\")\nend tell"
        );
        run_osascript(&script).await?;
        self.owned_tabs.retain(|(w, t)| w != window || t != tab);
        if self.window_id.as_deref() == Some(window) && self.tab_id.as_deref() == Some(tab) {
            let replacement = self.owned_tabs.first().cloned();
            self.window_id = replacement.as_ref().map(|(w, _)| w.clone());
            self.tab_id = replacement.map(|(_, t)| t);
            self.pending_human = None;
        }
        Ok(format!("Closed tab {id}"))
    }

    /// Shutdown closes owned tabs, including after selecting a foreign tab.
    pub async fn close_session_tab(&mut self) -> Result<(), String> {
        let tabs = self.owned_tabs.clone();
        for (w, t) in tabs {
            let _ = self.close_tab(&format!("{w}:{t}")).await;
        }
        self.owned_tabs.clear();
        self.window_id = None;
        self.tab_id = None;
        self.pending_human = None;
        Ok(())
    }

    async fn post_action_state(&self) -> String {
        match self.exec_js(crate::contract::STATE_SCRIPT).await {
            Ok(state) => state,
            Err(e) => format!("post_action_state=unknown: {e}"),
        }
    }

    /// Extract only the requested Unicode character slice in the page.
    pub async fn extract(&self, args: &Value, html: bool) -> Result<String, String> {
        let raw = self
            .exec_js(&crate::contract::extraction_script(args, html))
            .await?;
        crate::contract::decode_extraction(&raw)
    }

    /// Preserve the same JSON serialization and exception contract as CDP.
    pub async fn evaluate_js(&self, expr: &str) -> Result<String, String> {
        let raw = self
            .exec_js(&crate::contract::evaluation_script(expr))
            .await?;
        crate::contract::decode_evaluation(&raw)
    }

    /// Navigation's selector wait shares the exact same observation path.
    pub async fn wait_for_selector(
        &self,
        selector: &str,
        condition: &str,
        timeout_ms: u64,
    ) -> Result<u64, String> {
        let target = crate::contract::WaitTarget::Selector {
            selector: selector.into(),
            condition: condition.into(),
        };
        self.wait_for_target(&target, timeout_ms)
            .await
            .map(|(elapsed, _)| elapsed)
    }

    /// Wait for a selector condition or an exact URL, reporting the observed URL.
    async fn wait_for_target(
        &self,
        target: &crate::contract::WaitTarget,
        timeout_ms: u64,
    ) -> Result<(u64, String), String> {
        let start = Instant::now();
        let timeout_ms = timeout_ms.min(op_timeout_ms().saturating_sub(1_000));
        loop {
            let raw = self.evaluate_js(&target.script()).await?;
            let observation: Value = serde_json::from_str(&raw)
                .map_err(|e| format!("wait_for observation invalid: {e}"))?;
            let url = observation["current_url"]
                .as_str()
                .ok_or("wait_for URL observation missing")?;
            if observation["matched"] == true {
                return Ok((start.elapsed().as_millis() as u64, url.to_string()));
            }
            if start.elapsed().as_millis() >= timeout_ms as u128 {
                return Err(format!(
                    "{} not met within {timeout_ms} ms; observed current_url: {url}",
                    target.description()
                ));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    /// Scroll: x/y are window coordinates, or scroll a selector into view.
    pub async fn scroll(
        &self,
        x: Option<i64>,
        y: Option<i64>,
        selector: Option<&str>,
    ) -> Result<(), String> {
        let js = if let Some(sel) = selector {
            format!(
                "(() => {{ const el = document.querySelector({}); if (!el) return 'not found'; el.scrollIntoView({{block:'center'}}); return 'ok'; }})()",
                json!(sel)
            )
        } else {
            let (x, y) = (x.unwrap_or(0), y.unwrap_or(0));
            format!(
                "(() => {{ window.scrollTo({}, {}); return 'ok'; }})()",
                x, y
            )
        };
        let raw = self.exec_js(&js).await?;
        if unquote(&raw) == "not found" {
            return Err(format!("未找到滚动目标 {}", selector.unwrap_or("")));
        }
        Ok(())
    }

    /// Report a synthetic click's observed event and resulting target state.
    pub async fn click(&self, selector: &str) -> Result<String, String> {
        let raw = self
            .evaluate_js(&crate::contract::click_script(selector))
            .await?;
        let observed: Value =
            serde_json::from_str(&raw).map_err(|e| format!("click observation invalid: {e}"))?;
        Ok(format!(
            "click target={selector}\n{}",
            crate::contract::click_verdict(&observed)
        ))
    }

    /// Replace through the native Unicode setter, then read back before submit.
    pub async fn type_text(
        &self,
        selector: &str,
        text: &str,
        submit: bool,
    ) -> Result<String, String> {
        let script = format!(
            r#"(() => {{
            const el = document.querySelector({selector});
            if(!el) throw Error('input target not found');
            const observe = {observe};
            let observation = observe.call(el);
            if(!observation.connected || !observation.editable || observation.disabled || observation.read_only)
                return {{observation}};
            el.scrollIntoView({{block:'center'}});
            el.focus();
            const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            Object.getOwnPropertyDescriptor(proto,'value').set.call(el,{text});
            el.dispatchEvent(new Event('input',{{bubbles:true}}));
            el.dispatchEvent(new Event('change',{{bubbles:true}}));
            observation = observe.call(el);
            let submission = null;
            if({submit} && observation.connected && observation.editable && !observation.disabled && !observation.read_only && observation.value === {text}) {{
                if(!el.form) throw Error('submit requested but input has no form');
                const form = el.form;
                let event = null;
                const mark = e => {{ event = e; }};
                form.addEventListener('submit',mark,{{once:true}});
                try {{ form.requestSubmit(); }} finally {{ form.removeEventListener('submit',mark); }}
                if(!event) throw Error('requestSubmit did not produce a submit event; form validation may have prevented submission');
                submission = {{method:'requestSubmit',submit_event:true,default_prevented:event.defaultPrevented}};
            }}
            return {{observation,submission}};
        }})()"#,
            selector = json!(selector),
            text = json!(text),
            submit = submit,
            observe = crate::contract::TYPE_OBSERVATION_FUNCTION
        );
        let raw = self.evaluate_js(&script).await?;
        let result: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("type_text observation invalid: {e}"))?;
        let mut verdict = crate::contract::type_verdict(text, &result["observation"]);
        let submission = if submit && verdict["status"] == "applied" {
            crate::contract::submission_summary(&result["submission"])?
        } else if submit {
            "skipped because the requested value was not confirmed".into()
        } else {
            "not requested".into()
        };
        verdict["submission"] = json!(submission);
        Ok(format!(
            "type_text target={selector} requested_chars={}\n{verdict}",
            text.chars().count()
        ))
    }

    /// Press a key (synthetic keydown/keyup; Enter auto-submits when inside a form).
    pub async fn press_key(&self, key: &str, selector: Option<&str>) -> Result<String, String> {
        let focus_js = match selector {
            Some(sel) => format!(
                "const el = document.querySelector({}); if (!el) throw Error('focus selector not found'); if (el) {{ el.focus(); el.scrollIntoView({{block:'center'}}); }}",
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
            if key.eq_ignore_ascii_case("enter") {
                "true"
            } else {
                "false"
            },
            esc_js_key(key),
            esc_js_key(key)
        );
        let raw = self.exec_js(&js).await?;
        if unquote(&raw).contains("\"ok\":false") {
            return Err("没有可接收按键的焦点元素".to_string());
        }
        Ok(format!("Pressed {}", key))
    }

    /// An inspection error is unknown, never proof that a blocker disappeared.
    pub async fn detect_user_action_required(&self) -> Result<Option<String>, String> {
        let raw = self
            .exec_js(&format!(
                "JSON.stringify({})",
                crate::tools::DETECT_USER_ACTION_JS
            ))
            .await?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("human-action inspection failed: {e}"))?;
        crate::tools::parse_detection(&value)
    }

    async fn detect_and_mark(&mut self) -> String {
        match self.detect_user_action_required().await {
            Ok(Some(cat)) => {
                self.pending_human = Some(cat.clone());
                user_action_tag(&cat)
            }
            Ok(None) => {
                self.pending_human = None;
                String::new()
            }
            Err(reason) => format!("\n[HUMAN_ACTION_UNKNOWN] {reason}"),
        }
    }

    pub async fn wait_for_human_tool(
        &mut self,
        expect: Option<&str>,
        message: Option<&str>,
    ) -> String {
        let budget_ms = crate::browser::human_wait_budget_ms(op_timeout_ms());
        let deadline = Instant::now() + Duration::from_millis(budget_ms);
        let mut clean_polls = 0;
        let mut last_cat = self.pending_human.clone();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining.min(Duration::from_secs(5)), self.detect_user_action_required()).await {
                Ok(Ok(Some(cat))) => { self.pending_human = Some(cat.clone()); last_cat = Some(cat); clean_polls = 0; }
                Ok(Ok(None)) => {
                    clean_polls += 1;
                    if clean_polls >= 2 {
                        self.pending_human = None;
                        return format!("status=resolved\n{}\nYou may continue browser automation.", self.post_action_state().await);
                    }
                }
                Ok(Err(reason)) => return format!("status=unknown\nHuman action remains pending: inspection failed: {reason}"),
                Err(_) => return "status=unknown\nHuman action remains pending: inspection exceeded its polling deadline.".into(),
            }
            tokio::time::sleep(
                deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_secs(2)),
            )
            .await;
        }
        let cat = last_cat
            .or_else(|| expect.map(str::to_string))
            .unwrap_or_else(|| "unknown".into());
        self.pending_human = Some(cat.clone());
        let relay = message
            .map(|m| format!("\nInstruction for the user: {m}"))
            .unwrap_or_default();
        format!(
            "status=still_waiting\nWaiting for '{cat}' after {budget_ms} ms.{relay}\nEnd your turn and ask the user to complete it; call wait_for_human after they reply."
        )
    }

    /// Query the Chrome window bounds: returns (x, y, w, h), or None on failure (e.g. window closed).
    async fn window_bounds(&self, wid: &str) -> Option<(i32, i32, i32, i32)> {
        let script = format!(
            "tell application \"Google Chrome\"\n  get bounds of window id {}\nend tell",
            wid
        );
        let out = run_osascript(&script).await.ok()?;
        // Looks like "0, 44, 1440, 878" or "{0, 44, 1440, 878}".
        let s = out
            .trim()
            .trim_matches(|c| c == '{' || c == '}')
            .trim()
            .to_string();
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
    pub async fn screenshot(
        &mut self,
        path: &str,
        full_page: bool,
    ) -> Result<(String, String), String> {
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
        if screencapture_capture(&["-l", &wid, "-x", path])
            .await
            .is_ok()
        {
            // Double-check the file actually landed and is non-empty (may be 0 bytes when the window is hidden).
            if std::fs::metadata(path)
                .map(|m| m.len() > 0)
                .unwrap_or(false)
            {
                return Ok((path.to_string(), full_page_warn.to_string()));
            }
        }
        // 2) rect fallback: convert Chrome bounds to -R x,y,w,h.
        if let Some((x, y, w, h)) = self.window_bounds(&wid).await {
            let rect = format!("{},{},{},{}", x, y, w, h);
            if screencapture_capture(&["-R", &rect, "-x", path])
                .await
                .is_ok()
                && std::fs::metadata(path)
                    .map(|m| m.len() > 0)
                    .unwrap_or(false)
            {
                let note = format!("{} [fallback: rect {}]", full_page_warn, rect);
                return Ok((path.to_string(), note));
            }
        }
        // 3) fullscreen fallback (still produces an image when the window is invisible/hidden, avoiding total failure).
        if screencapture_capture(&["-x", path]).await.is_ok()
            && std::fs::metadata(path)
                .map(|m| m.len() > 0)
                .unwrap_or(false)
        {
            let note = format!(
                "{} [fallback: fullscreen - 窗口不可见已退化为全屏]",
                full_page_warn
            );
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
    translate_known_error(e).unwrap_or_else(|| e.to_string())
}

/// Chrome's "Allow JavaScript from Apple Events" gate is closed.
///
/// Chrome reads that preference **only at startup**, so ticking the menu item in a
/// running browser (or editing the profile on disk) leaves the running process refusing
/// `execute javascript`. Only a full quit + relaunch makes it effective, so every error
/// reporting this gate must spell that out: "enable it and retry" alone sends the user
/// back into the failure they just came from.
const JS_APPLE_EVENTS_HINT: &str = "\
Chrome 未开启“Apple 事件中的 JavaScript”（Allow JavaScript from Apple Events）：\
请在 Chrome 菜单 视图(View) > 开发者(Developer) > 允许 Apple 事件中的 JavaScript 打开，\
然后**完全退出 Chrome（⌘Q）并重新启动**——该开关只在 Chrome 启动时读取，\
运行中勾选或改配置文件都不会生效。";

/// Detect Chrome's "executing JavaScript through AppleScript is turned off" error.
///
/// The wording differs between Chinese and English Chrome builds (the gate is
/// zh-CN, "Executing JavaScript through AppleScript is turned off" in en), but every
/// variant names AppleScript and JavaScript and links
/// `https://support.google.com/chrome/?p=applescript`, so those two lowercase
/// substrings are the locale-independent markers.
fn is_js_apple_events_disabled(err: &str) -> bool {
    let le = err.to_lowercase();
    le.contains("applescript") && (le.contains("javascript") || le.contains("java script"))
}

/// User-facing message for a closed "Allow JavaScript from Apple Events" gate.
fn js_apple_events_message(raw_error: &str) -> String {
    format!("{} 原始错误：{}", JS_APPLE_EVENTS_HINT, raw_error.trim())
}

/// Whether an `exec_js` failure means Chrome's JS gate is closed (permanent
/// until Chrome quits and restarts), rather than a transient page error.
///
/// Polling callers (e.g. `navigate`'s readyState wait) use this to fail fast
/// instead of swallowing the actionable hint and retrying a hopeless gate.
/// The hint marker survives `exec_js`'s timeout wrapping, while page text can
/// never contain this exact string unless it quotes our own message.
fn is_gate_closed_message(msg: &str) -> bool {
    msg.contains(JS_APPLE_EVENTS_HINT)
}

/// Map a known osascript/Chrome failure to an actionable message.
///
/// Returns `None` for errors without a specific hint, so the caller can keep the
/// original text.
fn translate_known_error(e: &str) -> Option<String> {
    // `run_osascript` already wrapped the gate error; keep that wording (and the
    // original stderr it quotes) instead of wrapping it a second time.
    if e.contains(JS_APPLE_EVENTS_HINT) {
        return Some(e.to_string());
    }
    if is_js_apple_events_disabled(e) {
        return Some(js_apple_events_message(e));
    }
    let le = e.to_lowercase();
    if le.contains("不能获得") || le.contains("can't get") || le.contains("-1728") {
        return Some(
            "会话标签页/窗口已丢失（可能被手动关闭或切到别的 Chrome 窗口/实例）。\
             请重新调用 navigate 打开新标签页，或把浏览器窗口切回会话窗口。"
                .to_string(),
        );
    }
    None
}

/// Decode a JSON.stringify result: when it is a quoted string, parse it as JSON
/// (unescapes `\"`, `\n`, `\t`, `\\` etc., so page text/HTML comes back readable
/// instead of JSON-escaped); otherwise return it unchanged (non-string results
/// like `{"ok":false}` are not quoted JSON strings).
fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        serde_json::from_str::<String>(s).unwrap_or_else(|_| s[1..s.len() - 1].to_string())
    } else {
        s.to_string()
    }
}

fn esc_js_key(k: &str) -> String {
    k.replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
}

fn parse_tab_id(id: &str) -> Result<(&str, &str), String> {
    let (window, tab) = id
        .split_once(':')
        .ok_or("tab id must be window_id:tab_id from list_tabs")?;
    for part in [window, tab] {
        if !part.chars().all(|c| c.is_ascii_digit())
            || part
                .parse::<u64>()
                .ok()
                .is_none_or(|n| n == 0 || n > i64::MAX as u64)
        {
            return Err("invalid stable tab id; use an id from list_tabs".into());
        }
    }
    Ok((window, tab))
}

impl ApplescriptSession {
    /// Session teardown: only close the session tab, never quit the user's browser.
    pub async fn shutdown(&mut self) {
        let _ = self.close_session_tab().await;
        self.window_id = None;
        self.tab_id = None;
    }
}

/// The tools with the same names and argument keys as in tools.rs; results go only
/// into content[0].text (capped at 24K), using the same `[...]` markers as CDP mode to
/// remind the model.
pub async fn handle_tools_call(
    session: &mut ApplescriptSession,
    params: Option<Value>,
) -> Result<Value, JsonRpcErr> {
    let (cmd, args) = crate::contract::parse_call(params)?;
    let ms = op_timeout_ms();

    match cmd.as_str() {
        "navigate" => {
            let url = require_str(&args, "url")?;
            let wait_selector = opt_str(&args, "wait_selector");
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                session.navigate(&url, wait_selector.as_deref()).await?;
                let (u, t) = session.url_title().await?;
                let tag = session.detect_and_mark().await;
                let state = session.post_action_state().await;
                let outcome = if state.contains("\"readyState\":\"complete\"") {
                    "complete"
                } else {
                    "incomplete"
                };
                Ok(format!(
                    "{pend}Navigated to {u}\nTitle: {t}\nload_outcome={outcome}\n{state}{tag}"
                ))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "snapshot" => Err(JsonRpcErr::new(
            -32000,
            "snapshot is not supported in AppleScript mode; use the isolated CDP driver",
            None,
        )),
        "click" => {
            if args.get("ref").is_some() {
                return Err(JsonRpcErr::new(
                    -32000,
                    "snapshot refs are not supported in AppleScript mode; use selector",
                    None,
                ));
            }
            let sel = require_str(&args, "selector")?;
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                let message = session.click(&sel).await?;
                let tag = session.detect_and_mark().await;
                Ok(format!(
                    "{pend}{message}\n{}{tag}",
                    session.post_action_state().await
                ))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "type_text" => {
            if args.get("ref").is_some() {
                return Err(JsonRpcErr::new(
                    -32000,
                    "snapshot refs are not supported in AppleScript mode; use selector",
                    None,
                ));
            }
            let sel = require_str(&args, "selector")?;
            let text = args["text"].as_str().unwrap().to_string();
            let submit = args.get("submit").and_then(Value::as_bool).unwrap_or(false);
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                // Keep the value verdict and submit evidence returned by the driver.
                let msg = session.type_text(&sel, &text, submit).await?;
                let tag = session.detect_and_mark().await;
                Ok(format!(
                    "{pend}{msg}\n{}{tag}",
                    session.post_action_state().await
                ))
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
                let tag = session.detect_and_mark().await;
                Ok(format!(
                    "{pend}Pressed {key}\n{}{tag}",
                    session.post_action_state().await
                ))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "scroll" => {
            let sel = opt_str(&args, "selector");
            let x = args.get("x").and_then(Value::as_i64);
            let y = args.get("y").and_then(Value::as_i64);
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                session.scroll(x, y, sel.as_deref()).await?;
                let tag = session.detect_and_mark().await;
                let action = match &sel {
                    Some(s) => format!("{pend}Scrolled {s} into view{tag}"),
                    None => format!(
                        "{pend}Scrolled window to ({}, {}){tag}",
                        x.unwrap_or(0),
                        y.unwrap_or(0)
                    ),
                };
                Ok(format!("{action}\n{}", session.post_action_state().await))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "wait_for" => {
            let target = crate::contract::parse_wait_target(&args)?;
            let timeout_ms = args["timeout_ms"].as_u64().unwrap_or(15_000);
            let summary = with_timeout(ms, async {
                let (elapsed, url) = session.wait_for_target(&target, timeout_ms).await?;
                let tag = session.detect_and_mark().await;
                Ok(format!(
                    "Wait met: {} after {elapsed} ms\ncurrent_url: {url}{tag}",
                    target.description()
                ))
            })
            .await?;
            Ok(text_content(summary))
        }
        "evaluate_js" => {
            let expr = require_str(&args, "expression")?;
            let out = with_timeout(ms, async { session.evaluate_js(&expr).await }).await?;
            Ok(text_content(&cap_text(&out)))
        }
        "get_text" => {
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                let t = session.extract(&args, false).await?;
                let tag = session.detect_and_mark().await;
                Ok((t, tag))
            })
            .await?;
            let (t, tag) = summary;
            // Extraction is bounded in the page; append detection after paging metadata.
            let mut out = t;
            out.push_str(&tag);
            Ok(text_content(format!("{pend}{out}")))
        }
        "get_html" => {
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                let t = session.extract(&args, true).await?;
                let tag = session.detect_and_mark().await;
                Ok((t, tag))
            })
            .await?;
            let (t, tag) = summary;
            let mut out = t;
            out.push_str(&tag);
            Ok(text_content(format!("{pend}{out}")))
        }
        "screenshot" => {
            let path = opt_str(&args, "path");
            let out_path = resolve_screenshot_path(path)?;
            let full_page = args
                .get("full_page")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let pend = pending_warning(session);
            let summary = with_timeout(ms, async {
                let (p, extra) = session
                    .screenshot(&out_path.to_string_lossy(), full_page)
                    .await?;
                let tag = session.detect_and_mark().await;
                Ok(format!(
                    "{pend}Saved screenshot to {p} (full_page={full_page}){extra}{tag}"
                ))
            })
            .await?;
            Ok(text_content(&summary))
        }
        "wait_for_human" => {
            let expect = opt_str(&args, "expect");
            let message = opt_str(&args, "message");
            let summary = with_timeout(ms, async {
                Ok::<_, String>(
                    session
                        .wait_for_human_tool(expect.as_deref(), message.as_deref())
                        .await,
                )
            })
            .await?;
            Ok(text_content(&summary))
        }
        "switch_tab" => {
            let id = require_str(&args, "id")?;
            let r = with_timeout(ms, async { session.switch_tab(&id).await }).await?;
            Ok(text_content(r))
        }
        "close_tab" => {
            let id = require_str(&args, "id")?;
            let r = with_timeout(ms, async { session.close_tab(&id).await }).await?;
            Ok(text_content(r))
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
        _ => Err(JsonRpcErr::new(
            -32601,
            &format!("unknown tool: {cmd}"),
            None,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_fixes_stable_apple_tab_ids() {
        assert_eq!(parse_tab_id("12:34").unwrap(), ("12", "34"));
        for id in [
            "",
            "0:1",
            "1:0",
            "1:2:3",
            "1:bad",
            "-1:2",
            "1:2\"",
            "999999999999999999999:1",
        ] {
            assert!(parse_tab_id(id).is_err(), "{id}");
        }
        assert_eq!(crate::tools::parse_detection(&Value::Null).unwrap(), None);
        assert_eq!(
            crate::tools::parse_detection(&json!("captcha")).unwrap(),
            Some("captcha".into())
        );
        assert!(crate::tools::parse_detection(&json!({"error":"boom"})).is_err());
    }

    #[test]
    fn unquote_decodes_json_stringify_output() {
        // Escapes produced by JSON.stringify must be decoded, not kept verbatim:
        // page text comes back with real newlines and HTML with real quotes.
        assert_eq!(unquote(r#""line1\nline2""#), "line1\nline2");
        assert_eq!(unquote(r#""<div class=\"x\">""#), "<div class=\"x\">");
        assert_eq!(unquote(r#""a\"b\\c""#), "a\"b\\c");
        // Non-string JSON results (objects) and plain text are left unchanged.
        assert_eq!(unquote("{\"ok\":false}"), "{\"ok\":false}");
        assert_eq!(unquote("\"null\""), "null");
        assert_eq!(unquote("plain"), "plain");
        assert_eq!(unquote(""), "");
    }

    #[test]
    fn require_str_rejects_empty() {
        let empty = json!({ "url": "" });
        assert!(require_str(&empty, "url").is_err());
        let ok = json!({ "url": "https://example.com" });
        assert_eq!(
            require_str(&ok, "url").ok(),
            Some("https://example.com".to_string())
        );
    }

    /// Chrome's own wording when the gate is closed (zh-CN build).
    const CHROME_JS_GATE_ERROR: &str = "36:87: execution error: “Google Chrome”遇到一个错误：通过 AppleScript 执行 JavaScript 的功能已关闭。要开启此功能，请在菜单栏中依次转到“查看”>“开发者”>“允许 Apple 事件中的 JavaScript”。如需更多信息，请访问 https://support.google.com/chrome/?p=applescript (12)";

    #[test]
    fn js_gate_error_names_the_restart_step() {
        // Chrome reads the preference at startup only, so a hint that says "enable it
        // and retry" leaves the user in a loop: the restart step must be in the message.
        let msg = translate_err(CHROME_JS_GATE_ERROR);
        assert!(
            msg.contains("完全退出 Chrome"),
            "missing restart step: {}",
            msg
        );
        assert!(
            msg.contains("允许 Apple 事件中的 JavaScript"),
            "missing menu path: {}",
            msg
        );
        assert!(msg.contains("功能已关闭"), "original error lost: {}", msg);
    }

    #[test]
    fn english_js_gate_error_is_detected() {
        let msg = translate_err(
            "Executing JavaScript through AppleScript is turned off. To turn on this feature, \
             go to View > Developer > Allow JavaScript from Apple Events.",
        );
        assert!(
            msg.contains("完全退出 Chrome"),
            "missing restart step: {}",
            msg
        );
    }

    #[test]
    fn js_gate_message_is_not_applied_twice() {
        // `run_osascript` already wraps the raw stderr; translating again must not
        // duplicate the hint or the quoted original error.
        let once = js_apple_events_message(CHROME_JS_GATE_ERROR);
        assert_eq!(once, translate_err(&once));
    }

    #[test]
    fn stale_tab_error_keeps_its_own_hint() {
        let msg = translate_err("execution error: 不能获得“window id 999999”。 (-1728)");
        assert!(
            msg.contains("会话标签页/窗口已丢失"),
            "unexpected message: {}",
            msg
        );
        assert!(!msg.contains("完全退出 Chrome"), "wrong hint: {}", msg);
    }

    #[test]
    fn gate_closed_message_is_recognized_through_exec_js_wrapping() {
        // `navigate`'s readyState wait fails fast only on this marker: it must
        // survive `exec_js`'s timeout wrapping, while transient errors stay false.
        let inner = js_apple_events_message(CHROME_JS_GATE_ERROR);
        assert!(is_gate_closed_message(&inner));
        let wrapped = format!(
            "执行 JS 失败(超过 25000ms 上限或 osascript 出错): {}",
            inner
        );
        assert!(is_gate_closed_message(&wrapped));
        assert!(!is_gate_closed_message(""));
        assert!(!is_gate_closed_message(
            "execution error: 不能获得“window id 999999”。 (-1728)"
        ));
        assert!(!is_gate_closed_message(
            "operation reached the 25000 ms server cap"
        ));
    }
}
