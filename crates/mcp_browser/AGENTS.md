# AGENTS.md - mcp_browser

## Scope

Standalone stdio JSON-RPC **MCP server** for browser automation (navigate/click/type/extract/screenshot). Not a library — spawned as MCP subprocess via `mcp_stdio::run()`.

**Two drivers, one server:** `MCP_BROWSER_DRIVER=applescript` (default macOS, drives user's running Chrome via `osascript`) vs `cdp` (`chromiumoxide`/CDP; non-macOS default, or forced by `MCP_BROWSER_WS_URL`). `MCP_BROWSER_WS_URL` may be `ws://` or `http://host:port`/bare `host:port` (fetches `webSocketDebuggerUrl`). Both inherit cookies/login; `close_browser` only closes session tab — never quits user's browser.

## Layout

```text
src/main.rs      # #[tokio::main(multi_thread)]; BrowserServer { mode, session } + impl McpServer; gc_stale_profiles() + mcp_stdio::run()
src/browser.rs   # BrowserSession { browser, page, handler_task, temp_profile_dir, pending_human }; launch()/ensure_session()/shutdown() + Session enum + DriverMode::from_env()
src/applescript.rs # AppleScript driver: ApplescriptSession + osascript runner + 13 tool impls
src/tools.rs     # initialize/tools_list schemas + handle_tools_call dispatch + 13 tool impls
```

Transport (`JsonRpcErr`/`cap_text(24K)`/`with_timeout(90s)`/dispatch loop) lives in `mcp_stdio`.

## Build / Test

```bash
cargo build -p mcp_browser   # ~9 min cold (chromiumoxide)
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | ./target/debug/mcp_browser
```

No focused unit tests (needs real Chrome). Gate = build + smoke test above; add `tools/call navigate` for full round-trip (`MCP_BROWSER_HEADLESS=1` to avoid window). AppleScript smoke: needs Chrome *Allow JavaScript from Apple Events* once, verify one tab opens/closes without touching others.

## Invariants (do not break)

1. **Server cap < host timeout.** Host kills+restarts on `request_timeout_ms` (no per-tool override). Every CDP/osascript op wrapped in `with_timeout(op_timeout_ms())` default **90s**; host must be **120s**.
2. **Error wording avoids host transport triggers.** Never emit `mcp response timeout`/`broken pipe`/`closed the stream`/`process exited`/`failed to read response`/`failed waiting for mcp response` — shared timeout says "operation reached the N ms server cap".
3. **Only `content[0].text` reaches model.** Screenshots save to disk and return path; extracted text/HTML capped **24K** (`cap_text`, host offloads >32K).
4. **Sequential + Handler polling.** `main` owns `Option<BrowserSession>` by `&mut`; host serializes requests so one reused `Page` keeps cookies. CDP `Handler` must be continuously polled (`tokio::spawn while handler.next().await` in `launch()`), else all CDP hangs.
5. **launch vs connect.** `Browser::launch` = new controlled Chrome (throwaway profile). `Browser::connect` (when `MCP_BROWSER_WS_URL` set) = attach existing `--remote-debugging` instance. Default is launch.
6. **Per-pid profile dir + startup GC.** Chromiumoxide default `<temp>/chromiumoxide-runner` is single fixed path → `SingletonLock` collision. `launch()` uses `<temp>/mcp_browser-profile-<pid>` + `purge_singleton_locks()`. Host usually SIGKILLs subprocess → `shutdown()` never runs, so `gc_stale_profiles()` at startup removes dirs whose pid is dead (`kill(pid,0)`). Do not revert to shared dir.
7. **Human-required detection + `wait_for_human`.** `navigate`/`get_text`/`get_html` (and post-action re-check in `click`/`type_text`/`press_key`) detect captcha/slider/sms_otp/twofa/login_required/payment_verify/identity_verify via one `evaluate_expression` (conservative DOM signals; keyword+input required for text-only; JS errors → `None`). Appends `[USER_ACTION_REQUIRED: <category>]` **after** `cap_text` (never truncated). Single tag source: `user_action_tag()`. Captcha = present **and unsolved** (iframe stays in DOM after solve — check `g-recaptcha-response`/`h-captcha-response` + visibility, not bare presence). `wait_for_human`: blocking wait resumable in bounded segments — polls every 2s (each 5s timeout, hung poll = still-blocked), needs 2 consecutive clean polls to resolve; fixed 60s budget clamped to `op_timeout_ms-15s`; on expiry returns `status=still_waiting` (normal result, not error — model ends turn, asks user, calls again). Stays under host timeout so user can take minutes without kill. Headless → `status=unavailable` fast. Session tracks `pending_human: Option<String>` (set on detection, cleared on resolve/empty/navigate); while set, mutating tools prepend `[HUMAN_ACTION_PENDING: <category>]`.

## AppleScript driver specifics

Same 13 tools/args as CDP; `wait_for_human` never `unavailable`. Dedicated invariants:

1. **Never steal focus.** No `activate`; `open_tab` records frontmost app before creating tab and restores via `open -b <bundle id>` (fallback display name; bundle-id first because `open -a "飞书"` fails — registered as "Lark"). Skip restore if already in Chrome. Ops address `window id W` + `whose id is "T"` (int window + quoted tab; other forms -1719/-1728) → stray events fail dry instead of mutating foreign tabs. Chrome self-activates on new tab — unavoidable, hence restore.
2. **Allow JavaScript from Apple Events** (View > Developer). Server detects disabled error (message carries support URL) and returns menu-path instruction.
3. **JS embeds as one AppleScript string literal** — escape `\`/`"`/newlines via `esc_applescript`; whole script via `osascript -` stdin (`run_osascript`), no temp files. Multi-line JS (detect/wait polling) is fine.
4. **One session tab, reused.** `navigate` re-sets `URL of` tab unless vanished (then fresh tab in session window; first session uses front window). Tab becomes active tab *of its window* (for `screencapture -l`) but window stays background.
5. **Synthetic events only.** JS `click()`, native-setter `value`+`input`/`change`, `KeyboardEvent` dispatch (Enter also `form.requestSubmit()`). No OS key chords/drag&drop. `full_page` 降级为窗口截图；`screenshot` 三段回退 `screencapture -l <wid>` → `bounds → -R x,y,w,h` → 全屏，首次调用无窗口时自动 `about:blank` 建会话，需 *Screen Recording* 权限（授权终端应用后重启）。
6. **Every osascript call wrapped in `with_timeout` (90s).** CDP-only env vars (`MCP_BROWSER_USER_DATA_DIR`/`MCP_BROWSER_HEADLESS`/`MCP_BROWSER_CHROME`/`MCP_BROWSER_WS_URL` win) don't apply.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MCP_BROWSER_DRIVER` | auto `applescript`/else `cdp` | driver; `MCP_BROWSER_WS_URL` forces `cdp` |
| `MCP_BROWSER_HEADLESS` | `0` | `1`/`true` = headless (CDP only) |
| `MCP_BROWSER_CHROME` | `/Applications/Google Chrome.app/...` | Chrome exe (CDP) |
| `MCP_BROWSER_WS_URL` | (unset) | attach via `Browser::connect` |
| `MCP_BROWSER_USER_DATA_DIR` | (unset) | explicit profile (reused, never GC'd; don't share) |
| `MCP_BROWSER_OP_TIMEOUT_MS` | `90000` | per-op cap < host timeout |
| `MCP_BROWSER_SCREENSHOT_DIR` | `<temp>/mcp_browser` | screenshot dir |

## chromiumoxide 0.9 notes

Keyboard/typing on **`Element`** (`type_str`/`press_key`/`focus`/`scroll_into_view`/`click`/`inner_text`/`outer_html`); `Page` lacks `press_key` → focus `body` first. Screenshot: `ScreenshotParams::builder().format(CaptureScreenshotFormat::Png).full_page(bool).build()` (`chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat`).

## Host registration

`~/.config/mcp.json` → `mcpServers` (tools `mcp_browser_*`). Keep `request_timeout_ms` 120000 > server cap 90000.
