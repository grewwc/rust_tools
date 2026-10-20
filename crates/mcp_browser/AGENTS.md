# AGENTS.md - mcp_browser

## Scope

Standalone stdio JSON-RPC **MCP server** for browser automation (navigate/click/type/extract/screenshot). Not a library — spawned as MCP subprocess via `mcp_stdio::run()`.

**Two drivers, one server:** `MCP_BROWSER_DRIVER=applescript` (default macOS, drives user's running Chrome via `osascript`) vs `cdp` (`chromiumoxide`/CDP; non-macOS default, or forced by `MCP_BROWSER_WS_URL`). `MCP_BROWSER_WS_URL` may be `ws://` or `http://host:port`/bare `host:port` (fetches `webSocketDebuggerUrl`). Both inherit cookies/login; `close_browser` only closes session tab — never quits user's browser.

## Layout

```text
src/main.rs      # #[tokio::main(multi_thread)]; BrowserServer { mode, session } + impl McpServer; gc_stale_profiles() + mcp_stdio::run()
src/browser.rs   # BrowserSession + launch()/ensure_session()/shutdown() + Session enum + DriverMode::from_env()
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

1. **Server cap < host timeout** (`crates/mcp_stdio/AGENTS.md` invariant 5). No per-tool override — the host kills+restarts the subprocess on `request_timeout_ms`, so every CDP/osascript op stays under the **90s** cap (host **120s**).
2. **Error wording avoids host transport triggers** — the trigger list and the shared timeout wording live in `crates/mcp_stdio/AGENTS.md` invariant 2.
3. **Only `content[0].text` reaches model.** Screenshots save to disk and return the path; extracted text/HTML is capped by `cap_text` (`mcp_stdio`; host offloads >32K).
4. **Sequential + Handler polling.** `main` owns `Option<BrowserSession>` by `&mut`; host serializes requests so one reused `Page` keeps cookies. CDP `Handler` must be continuously polled, else all CDP hangs.
5. **launch vs connect.** `Browser::launch` = new controlled Chrome (throwaway profile). `Browser::connect` (when `MCP_BROWSER_WS_URL` set) = attach existing `--remote-debugging` instance. Default is launch.
6. **Per-pid profile dir + startup GC.** A shared profile path collides (`SingletonLock`), and the host usually SIGKILLs the subprocess so `shutdown()` never runs — `launch()` uses a per-pid dir and `gc_stale_profiles()` at startup removes dirs whose pid is dead. Do not revert to a shared dir.
7. **Human-required detection + `wait_for_human`.**
   - **Detect** captcha/slider/sms_otp/twofa/login_required/payment_verify/
     identity_verify in `navigate`/`get_text`/`get_html` and after
     `click`/`type_text`/`press_key`; one `evaluate_expression` pass per call, and
     a JS error means "no signal".
   - **Signals must be visibility-gated** — hidden inputs/iframes never count, and
     **no raw `innerHTML` substring matching** (SPA script text false-positives);
     text clues come from visibility-safe `body.innerText` plus a visible input.
     Both drivers share the same keyword rules.
   - **Tag** `[USER_ACTION_REQUIRED: <category>]` **after** `cap_text` (never
     truncated), from the single source `user_action_tag()`; captcha counts only
     while present *and* unsolved. While a human action is pending, mutating
     tools prepend `[HUMAN_ACTION_PENDING: <category>]`.
   - **`wait_for_human`** waits in bounded segments under the host timeout (60s
     budget) and returns `status=still_waiting` — a normal result, not an error:
     the model ends the turn and asks the user. Headless reports
     `status=unavailable`. Detector internals live in `src/tools.rs` (CDP) and
     `src/applescript.rs` (AppleScript).

## AppleScript driver specifics

Same 13 tools/args as CDP; `wait_for_human` never `unavailable`. Dedicated invariants:

1. **Never steal focus.** No `activate`; `open_tab` records the frontmost app before creating the tab and restores it afterwards (skip when already in Chrome). Ops address `window id W` + `whose id is "T"` (int window + quoted tab; other forms -1719/-1728) → stray events fail dry instead of mutating foreign tabs. Chrome self-activates on a new tab — hence the restore.
2. **Allow JavaScript from Apple Events** (View > Developer). Server detects disabled error (message carries support URL) and returns menu-path instruction.
3. **JS embeds as one AppleScript string literal** — escape `\`/`"`/newlines via `esc_applescript`; the whole script goes over `osascript -` stdin, no temp files. Multi-line JS (detect/wait polling) is fine.
4. **One session tab, reused.** `navigate` re-sets `URL of` tab unless vanished (then fresh tab in session window; first session uses front window). Tab becomes active tab *of its window* (for `screencapture -l`) but window stays background.
5. **Synthetic events only.** JS `click()`, native-setter `value`+`input`/`change`, `KeyboardEvent` dispatch (Enter also `form.requestSubmit()`). No OS key chords / drag&drop. `full_page` degrades to a window screenshot.
6. **Screenshot TCC preflight.** `screenshot` probes Screen Recording permission first: without it the window capture fails while the region/full-screen fallbacks still exit 0 with a wallpaper-only image that looks like success — so the server must error out with authorization instructions instead of returning a useless frame. The probe and fallback chain live in `src/applescript.rs`; the grant applies only after fully quitting and restarting the host app.
7. **Every osascript call wrapped in `with_timeout` (90s).** CDP-only env vars don't apply.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MCP_BROWSER_DRIVER` | auto `applescript`/else `cdp` | driver; `MCP_BROWSER_WS_URL` forces `cdp` |
| `MCP_BROWSER_HEADLESS` | `0` | `1`/`true` = headless (CDP only) |
| `MCP_BROWSER_CHROME` | auto-detected per OS | Explicit Chrome/Chromium executable (CDP) |
| `MCP_BROWSER_WS_URL` | (unset) | attach via `Browser::connect` |
| `MCP_BROWSER_USER_DATA_DIR` | (unset) | explicit profile (reused, never GC'd; don't share) |
| `MCP_BROWSER_OP_TIMEOUT_MS` | `90000` | per-op cap < host timeout |
| `MCP_BROWSER_SCREENSHOT_DIR` | `<temp>/mcp_browser` | screenshot dir |

## chromiumoxide 0.9 notes

Keyboard/typing must go through **`Element`**; `Page` has no `press_key`, so focus `body` first. The element/screenshot API details live in `src/tools.rs`.

## Host registration

`~/.config/mcp.json` → `mcpServers` (tools `mcp_browser_*`). Keep `request_timeout_ms` 120000 > server cap 90000.
