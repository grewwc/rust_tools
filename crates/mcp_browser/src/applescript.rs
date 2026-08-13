//! AppleScript 驱动模式：直接复用**用户已打开的 Chrome**（默认驱动方式）。
//!
//! 为什么需要它：CDP 只能接管"受控新开"的 Chrome，而 Chrome 136+ 拒绝在默认
//! profile 上开启 `--remote-debugging-port`，所以用户日常打开的浏览器无法被 CDP
//! attach。本模块改用 macOS 的 Apple Events（`osascript` + Chrome 的
//! `execute javascript`）驱动用户已开的 Chrome：在**会话窗口开一个新标签页**，
//! 全程复用用户真实 cookie/登录态，**绝不退出用户浏览器**。
//!
//! 前提：Chrome 需开启 "View > Developer > Allow JavaScript from Apple Events"
//! （默认关闭，开启一次即可；未开启时会给出一条可操作的报错）。
//!
//! 与 CDP 模式的关键差异：
//! - 每个操作都是独立的 `osascript` 子进程（有状态只有 窗口id+标签页id，由本进程
//!   AppleEvent 调度到会话窗口/标签页，绝不 `activate`——activate 会把焦点抢走）。
//! - 页面交互全部通过 JS（click/输入/按键为合成事件；对依赖真实键盘/`hasFocus` 的
//!   站点可能无效——这是本模式的固有限制）。
//! - 截图走 `screencapture -l <窗口id>`（需要屏幕录制权限），不支持整页截图。
//! - 会话标签页被用户手动关闭时，后续操作会得到"标签页已丢失"的明确报错，可
//!   navigate 重建。

use std::process::Stdio;
use std::time::{Duration, Instant};

use mcp_stdio::{JsonRpcErr, cap_text, text_content, with_timeout};
use serde_json::{json, Value};
use tokio::process::Command;

use crate::tools::{op_timeout_ms, resolve_screenshot_path};

/// 所有 osascript 调用都只发 Apple 事件、绝不 activate Chrome（activate 会把
/// 用户屏幕焦点抢到 Chrome）。唯一例外：新建标签页时 Chrome 会自行激活到前台，
/// 无法从源头阻止，见 open_tab 里的"记录前台 + 创建后还原焦点"。

/// 读取当前前台应用的 (bundle id, 显示名)（lsappinfo 查询，无需任何系统权限）。
/// 失败返回 None。
fn frontmost_app() -> Option<(String, String)> {
    let raw = std::process::Command::new("lsappinfo")
        .arg("front")
        .output()
        .ok()?;
    let asn = String::from_utf8_lossy(&raw.stdout).trim().to_string();
    if asn.is_empty() {
        return None;
    }
    // -only 单键输出形如 "LSDisplayName"="Google Chrome"；
    // 统一取第一个 '=' 之后第一个引号对里的内容。
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

/// 把前台还给之前记录的应用，无需 Apple Events 授权（也就没有 per-app 的
/// 自动化授权弹窗）。优先按 bundle id（`open -b`，中文显示名如 "飞书" 的
/// 注册名其实是 "Lark"，按名字找会失败），失败再退回显示名 `open -a`。
/// 建标签页时 Chrome 必激活自己，此函数用于创建后的焦点还原；失败静默忽略
/// （应用恰好退出等，下次建标签页会再试）。用户本来就在 Chrome 里时不调用。
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

/// 分类的"需要人工操作"标签：与 tools.rs 的 detect 保持同款格式。
pub fn user_action_tag(category: &str) -> String {
    format!(
        "\n[USER_ACTION_REQUIRED: {category}] 页面需要用户手动完成操作。可调用 wait_for_human 阻塞等待用户在可见浏览器窗口完成，或直接停止自动化并请用户完成后告知继续。"
    )
}

/// 待人工操作时，改动类工具输出前的提醒（与 tools.rs 的 pending_warning 同款）。
pub fn pending_warning(session: &ApplescriptSession) -> String {
    match &session.pending_human {
        Some(cat) => format!("[HUMAN_ACTION_PENDING: {}] ", cat),
        None => String::new(),
    }
}

/// 与 tools.rs 相同语义的人机校验检测：captcha/slider/sms_otp/twofa/login_required/
/// payment_verify/identity_verify；captcha 是"存在且未解决"（关掉的弹层不算），
/// 纯文本线索需"关键词+输入框"才报，避免误报。JS 求值后 JSON.stringify 成字符串。
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

/// 执行一段 osascript（脚本经 stdin 传入），返回 stdout 末尾行（trim）。
async fn run_osascript(script: &str) -> Result<String, String> {
    let mut child = Command::new("/usr/bin/osascript")
        .arg("-") // 从 stdin 读脚本，避免命令行转义地狱
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
        // 包含 "applescript"（支持 URL）=> 未开启"允许 Apple 事件中的 JavaScript"
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

/// 转义用于嵌入 AppleScript 字符串字面量的文本（\\、\"、换行等）。
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

/// 会话：持有"用户已打开的 Chrome"里的窗口 id + 标签页 id。
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

    /// 在当前会话标签页上执行 JS；函数体必须返回字符串（通常 JSON.stringify）。
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

    /// 对会话标签页执行 JS，带超时（超时/错误都会转成可读信息）。
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

    /// 导航：无会话标签页则新建（首个 tab），否则复用现有 tab 并 `set URL`。
    /// 之后等待页面加载完成；有 wait_selector 时再等待元素出现。
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
        // 等待加载完成（最多 ~20s；到达时限也继续，页面可能因广告长载）。
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

    /// 当前会话页的 (url, title)。
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

    /// 在用户 Chrome 中为会话新建一个标签页（首次或标签页丢失时）。
    /// 返回 (window_id, tab_id)。
    ///
    /// 焦点处理：Chrome 收到"开新标签页"事件时必定自己激活到前台（实测
    /// `make new tab` 与 `open -g` 都会抢焦点，无法从源头阻止），所以这里在
    /// 创建前记录前台应用，创建成功后立即用 `open -a` 把焦点还回去，不让
    /// 用户的屏幕被锁到新标签页。创建失败或用户本来就在 Chrome 里则不还原。
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

    /// 标签页是否仍存在。
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

    /// 列出会话窗口里的标签页（#idx: url [active]）。
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

    /// 关闭会话标签页（不退出 Chrome，不动其它窗口/标签页）。
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
        // 标签页已不在（用户自己关的）——视为"已关闭"，清状态即可
        self.window_id = None;
        self.tab_id = None;
        self.pending_human = None;
        Ok(())
    }

    /// 提取页面文本（selector 可选，缺省 body），cap 后追加检测标签。
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

    /// 提取页面 HTML（selector 可选，缺省 documentElement），cap 后追加检测标签。
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

    /// 在页面内执行任意 JS 表达式（结果 JSON.stringify 后返回）。
    pub async fn evaluate_js(&self, expr: &str) -> Result<String, String> {
        let js = format!("JSON.stringify((() => {{ try {{ return ({}); }} catch(e) {{ return 'JS_ERROR: ' + e; }} }})())", expr);
        // 用户表达式可能含换行（多行表达式合法），逐行拼接成单行
        let js = js.lines().collect::<Vec<_>>().join(" ");
        let raw = self.exec_js(&js).await?;
        Ok(unquote(&raw))
    }

    /// 等待元素出现（轮询），返回等待毫秒数。
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

    /// 滚动：x/y 为窗口坐标，或 selector 滚入视野。
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

    /// 点击（合成 click 事件 + 原生 setter 输入），返回输出文本。
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

    /// 输入文本（原生 setter + input/change 事件，可提交表单）。
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

    /// 按键（合成 keydown/keyup；Enter 在表单内会自动提交）。
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

    /// 人机校验检测（同款语义的 JS），返回类别或 None。
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

    /// 等待用户人工处理（同款契约：status=resolved / still_waiting；60s 预算，
    /// 2 次连续干净检测才判定解决）。
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
            // 每次轮询独立超时：挂起的轮询按"仍阻塞"处理
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

    /// 带 5s 轮询超时的检测：轮询挂起视为"仍阻塞"（返回 None 以外不判定）。
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

    /// 截图：screencapture 按窗口 id 抓取（需要屏幕录制权限）。
    pub async fn screenshot(&self, path: &str, full_page: bool) -> Result<String, String> {
        if full_page {
            return Err("full_page 整页截图仅支持 CDP 模式；AppleScript 模式按窗口抓取".to_string());
        }
        let wid = self
            .window_id
            .clone()
            .ok_or("还没有会话窗口：请先调用 navigate")?;
        // 先确保窗口存在
        let alive = self.tab_alive().await;
        if !alive {
            return Err("会话标签页已丢失：请重新 navigate".to_string());
        }
        let parent = std::path::Path::new(path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        std::fs::create_dir_all(&parent).map_err(|e| format!("创建截图目录失败: {}", e))?;
        let child = Command::new("/usr/sbin/screencapture")
            .arg("-l")
            .arg(&wid)
            .arg("-x")
            .arg(path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("screencapture 启动失败: {}", e))?;
        let out = child
            .wait_with_output()
            .await
            .map_err(|e| format!("screencapture 失败: {}", e))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr).to_string();
            return Err(format!(
                "screencapture 失败（可能需要屏幕录制权限: 系统设置>隐私与安全性>屏幕录制，勾选本终端应用）：{}",
                err.trim()
            ));
        }
        Ok(path.to_string())
    }
}

/// osascript 错误转义：会话标签页/窗口丢失的报错给出可操作提示。
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

/// 去除 JSON.stringify 结果的外层引号（保留转义内容原文）。
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
    /// 会话结束收尾：只关掉会话标签页，绝不退出用户浏览器。
    pub async fn shutdown(&mut self) {
        let _ = self.close_session_tab().await;
        self.window_id = None;
        self.tab_id = None;
    }
}

/// 与 tools.rs 同名同参数 key 的 13 个工具；结果只放 content[0].text（cap 24K），
/// 使用与 CDP 模式一致的 `[...]` 标记提醒模型。
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
                let p = session.screenshot(&out_path.to_string_lossy(), full_page).await?;
                let tag = session.detect_user_action_required().await.map_or(String::new(), |c| user_action_tag(&c));
                Ok(format!("{pend}Saved screenshot to {p} (full_page={full_page}){tag}"))
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