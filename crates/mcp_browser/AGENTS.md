# AGENTS.md - mcp_browser

## Scope

Standalone stdio JSON-RPC **MCP server** giving the `a` Agent browser automation
(navigate / click / type / extract / screenshot ...). **Two drivers, one server**:

- **AppleScript driver (default, `MCP_BROWSER_DRIVER=applescript`)** — drives the
  **user's already-open Chrome** via `osascript` (`execute javascript`), opens a
  new tab in their browser, inherits their cookies/login, and **never quits
  their browser**. No profile copying, no re-login.
- **CDP driver (`MCP_BROWSER_DRIVER=cdp` or with `MCP_BROWSER_WS_URL` set)** —
  `chromiumoxide` (Chrome DevTools Protocol). Without
  `MCP_BROWSER_WS_URL`: a controlled new Chrome instance (throwaway profile).
  With `MCP_BROWSER_WS_URL`: **attach-only** to an already-running Chrome —
  the value may be a `ws://` URL **or** `http://host:port` / bare `host:port`
  (auto-fetches `webSocketDebuggerUrl` from `/json/version`). This is the
  **Windows/Linux reuse-your-browser path** (start Chrome once with
  `--remote-debugging-port=9222`), inherits cookies, and `close_browser`
  there only closes the session tab - the user's browser is never shut down.
  Non-macOS builds default to `cdp` (no `osascript`).

Not a library - a single binary the Agent spawns as an MCP subprocess.

## Layout

```text
src/main.rs      # #[tokio::main(multi_thread)]; BrowserServer { mode: DriverMode, session: Option<Session> } + impl mcp_stdio::McpServer; per-mode dispatch + shutdown override; startup gc + mcp_stdio::run()
src/browser.rs   # BrowserSession { browser, page, handler_task, temp_profile_dir, pending_human }, launch(), ensure_session(), shutdown(), gc_stale_profiles(); + Session enum (Cdp | AppleScript) + DriverMode::from_env()
src/applescript.rs # AppleScript driver: ApplescriptSession { window_id, tab_id, pending_human } + osascript runner + 13 tool impls (same arg keys as tools.rs)
src/tools.rs     # initialize/tools_list schemas + handle_tools_call() dispatch + 13 tool impls
```

> Transport (JsonRpcErr / write_result / write_err / text_content /
> `cap_text(24K)` / `with_timeout`) and the stdin dispatch loop live in the shared
> `mcp_stdio` lib crate. This crate only implements the `McpServer` trait
> (initialize / tools_list / tools_call + a `shutdown` override closing the CDP
> session) and its tool logic + driver.

## Build / Test

```bash
cargo build -p mcp_browser   # the ONLY command that compiles chromiumoxide (~9 min cold)
```

Smoke test (pipe JSON-RPC lines to the binary):

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | ./target/debug/mcp_browser
```

For a full Chrome round-trip add a `tools/call` with `navigate`. Set
`MCP_BROWSER_HEADLESS=1` to avoid popping a window. No focused unit tests
(runtime behavior needs a real Chrome); the gate is the build + smoke test.

**AppleScript-driver smoke** (drive the running Chrome, opens ONE new tab):
user's Chrome needs the one-time *Allow JavaScript from Apple Events* toggle;
pipe `initialize` + `tools/call navigate` (harmless URL) + `close_browser` -
the tab must open in the session window and close without touching other tabs.

## Invariants (do not break)

1. **Server cap < host timeout.** The host has one `request_timeout_ms` per
   server (no per-tool override); on timeout it **kills + restarts** the
   subprocess, destroying the browser session. Every CDP op is wrapped in
   `with_timeout(op_timeout_ms())` (default **90s**); configure the host larger
   (**120s** recommended).
2. **Error messages must avoid the host's transport trigger words.** Never let a
   returned error contain `mcp response timeout`, `broken pipe`,
   `closed the stream`, `process exited`, `failed to read response`, or
   `failed waiting for mcp response` - those make the host kill the subprocess.
   The shared timeout error (in `mcp_stdio::with_timeout`) uses "operation
   reached the N ms server cap" on purpose - domain-neutral, none of the triggers.
3. **Only `content[0].text` reaches the model.** Screenshots **save to disk and
   return the path** - image bytes are never returned. Extracted text/HTML is
   capped to **24K chars** (`cap_text`) because the host offloads results above
   ~32K to disk.
4. **Sequential processing + Handler polling.** `main` owns one
   `Option<BrowserSession>` by `&mut`; requests are handled **sequentially**
   (the host serializes the round-trip), so no session lock is needed and a
   single **reused Page** keeps login/cookies across calls. The CDP `Handler`
   **must be polled continuously** (`tokio::spawn` a `while handler.next().await`
   loop in `launch()`), or every CDP call hangs.
5. **launch vs connect.** `Browser::launch` starts a **new** controlled Chrome
   with a throwaway profile - it does NOT hijack the user's open windows. To
   attach an existing `--remote-debugging` instance, set `MCP_BROWSER_WS_URL`
   (uses `Browser::connect`); default is launch.
6. **Per-process profile dir + startup GC - never share the fixed default.**
   chromiumoxide's default `user_data_dir` is a **single fixed** path
   (`<temp>/chromiumoxide-runner`); reusing it makes concurrent/previous
   instances collide on Chrome's `SingletonLock`, so `launch` dies with
   `Failed to create ... SingletonLock: File exists (17)`. `launch()` therefore
   assigns each process its own `<temp>/mcp_browser-profile-<pid>` and
   `purge_singleton_locks()` before starting. Because the host usually **SIGKILLs**
   the subprocess on one-shot exit (uncatchable - `shutdown()` can't run), stale
   profiles are reclaimed by `gc_stale_profiles()` at **startup**: it scans
   `mcp_browser-profile-<pid>` dirs and `remove_dir_all`s those whose pid is dead
   (`kill(pid, 0)`). Do not revert to the shared default dir.
7. **Detect "must-do-by-user" pages, then block-and-wait on demand.** When
   `navigate`, `get_text`, or `get_html` finds a page needing manual intervention
   (captcha / slider / sms_otp / twofa / login_required / payment_verify /
   identity_verify) it appends `[USER_ACTION_REQUIRED: <category>]` to the
   returned text; `click` / `type_text` / `press_key` also re-detect **after** the
   action, so a captcha triggered by a submit lands in the next tool output.
   Detection is one `evaluate_expression` call, best-effort and **conservative**:
   it prefers DOM-structure signals and requires "keyword + input element" for
   text-only cases to avoid false positives. JS errors / no-match return `None`
   silently. The tag is appended **after** `cap_text`, so a >24K body can never
   truncate it away. Single source for the tag string: `user_action_tag()`.
   **captcha is a "present AND unsolved" check, not a presence check**: the
   reCAPTCHA/hCaptcha iframe **stays in the DOM after solving** (only the hidden
   `g-recaptcha-response` / `h-captcha-response` textarea gets filled), so a
   presence-only check would make `wait_for_human` never resolve; generic
   `[class*="captcha"]` / `[id*="captcha"]` elements must also be *visible* to
   count (hidden footer badges / closed overlays do not block).
   To pause for the human, the model calls **`wait_for_human`**: a **real
   blocking wait, resumable in bounded segments**. Each call polls the page
   (every 2s, each poll with its own 5s timeout - a hung poll counts as
   still-blocked, never as resolved) and returns `status=resolved` only after 2
   **consecutive** clean polls (guards against transient re-render false
   "resolved"); if its fixed 60s budget (hard-clamped to `op_timeout_ms - 15s`
   for safety) expires first it returns `status=still_waiting` - as a
   **normal result, not an error** - telling the model to **end its turn and ask
   the user**, then call `wait_for_human` again when the user replies. This keeps
   every call safely under the host `request_timeout_ms`, so a user can take
   minutes on a captcha without the host killing the subprocess (invariant #1).
   Headless mode fails fast with `status=unavailable` (no visible window to act
   in) instead of waiting pointlessly.

   The session tracks `pending_human: Option<String>`: set by any detection hit,
   cleared by `wait_for_human`'s resolution, an empty detection, or a fresh
   `navigate`. While set, mutating tools (`click` / `type_text` / `press_key`)
   prepend `[HUMAN_ACTION_PENDING: <category>]` to their output so the model is
   reminded to stop and hand control to the user instead of blindly continuing.

## AppleScript driver (reuse the user's open Chrome)

Default mode (`MCP_BROWSER_DRIVER=applescript`): the server drives the
**already-running** Chrome via `osascript` - opens ONE session tab in the user's
front window, inherits their real cookies/login, waits at `wait_for_human` when
a captcha/login/payment flow needs the human, and **never quits the user's
browser**. `close_browser`/`shutdown` only close the session tab. Login state
therefore survives across agent sessions - no re-login each time.

Same 13 tools (names/args/result shape identical to CDP; `tools_list` comes
from tools.rs so the host needs zero changes). `wait_for_human` here is never
`unavailable` (the window is always visible); everything else follows the CDP
contract above.

Dedicated invariants:
1. **Never steal the user's screen focus.** No script calls `activate` -
   plain Apple events (query / `execute javascript` / `set URL` / close) do
   not bring Chrome forward. The only unavoidable exception: Chrome self-
   activates whenever a new tab is created (both `make new tab` and
   `open -g` trigger it - verified). `open_tab` therefore records the
   frontmost app (lsappinfo, no permissions) *before* creating the tab and
   restores focus *after* via `open -b <bundle id>` (fallback `open -a
   <display name>`; `open -a "飞书"` fails because its registered name is
   "Lark", hence bundle-id-first). Restore is skipped when the user was
   already in Chrome. Session ops always address `window id W` + `whose id
   is "T"` (**integer** window id + **quoted** tab id - `whose id =` /
   string window ids fail with -1719/-1728), so if events land on a
   different instance every op fails dry ("会话标签页/窗口已丢失") instead
   of mutating foreign tabs.
2. **One-time setting: "Allow JavaScript from Apple Events".** Disabled by
   default; the server detects the disabled error (its text carries the
   `applescript` support URL in every locale) and returns the exact
   menu-path instruction (View > Developer > Allow JavaScript from Apple
   Events) - never a raw parse error.
3. **JS embeds into one AppleScript string literal**: escape `\`, `"` and
   newlines (`\n`) via `esc_applescript`; multi-line JS (detect script, wait
   polling) is fine. The whole script goes to `osascript -` via stdin (see
   `run_osascript`), no temp files.
4. **One session tab, reused.** `navigate` re-sets `URL of` the session tab
   unless it vanished (then a fresh tab in the session window; the very first
   session uses the front window). The tab becomes the active tab *of its
   window* (so `screencapture -l` captures the right content and
   `wait_for_human`'s visible window works) but the *window itself* stays in
   the background - focus was restored to the pre-creation app (invariant
   #1).
5. **Synthetic events**: JS `click()`, native-setter `value` + `input`/`change`
   events, `KeyboardEvent` dispatch (Enter also `form.requestSubmit()` unless
   defaultPrevented). Covers forms/links; real OS key chords / drag&drop are
   not reproduced. `full_page` screenshots unsupported (error); window
   screenshots go through `screencapture -l <window id>` and need the host
   process granted *Screen Recording* permission.
6. **Every osascript call is wrapped in `with_timeout`** (same 90s cap) -
   hung Chrome can't outlive the host timeout. CDP-only env vars
   (`MCP_BROWSER_USER_DATA_DIR`, `MCP_BROWSER_HEADLESS`, `MCP_BROWSER_CHROME`,
   `MCP_BROWSER_WS_URL` behavior wins over driver mode) don't apply here.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MCP_BROWSER_DRIVER` | auto: `applescript` on macOS, else `cdp` | `applescript` = drive the user's running Chrome (never quit it, never steal focus); `cdp` = controlled Chrome instance. `MCP_BROWSER_WS_URL` forces `cdp`. |
| `MCP_BROWSER_HEADLESS` | `0` (headed) | `1`/`true` = headless |
| `MCP_BROWSER_CHROME` | `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` | Chrome executable |
| `MCP_BROWSER_WS_URL` | (unset) | If set, attach via `Browser::connect` instead of launching |
| `MCP_BROWSER_USER_DATA_DIR` | (unset) | Explicit Chrome profile dir (persists login/cookies). When set it is reused and **never** GC'd; sharing one across concurrent processes will collide. Unset -> per-pid temp dir, GC'd at startup. |
| `MCP_BROWSER_OP_TIMEOUT_MS` | `90000` | Per-op server-side cap (keep < host timeout) |
| `MCP_BROWSER_SCREENSHOT_DIR` | `<temp>/mcp_browser` | Where screenshots land |

## chromiumoxide 0.9 API notes (non-obvious)

Confirmed against vendored source + a successful build. Keyboard/typing live on
**`Element`** (`type_str`, `press_key`, `focus`, `scroll_into_view`, `click`,
`inner_text`, `outer_html`) - the public `Page` does not expose `press_key`, so
page-level key presses focus `body` first. Screenshot:
`ScreenshotParams::builder().format(CaptureScreenshotFormat::Png).full_page(bool).build()`,
where the format enum is at
`chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat`.

## Host registration

Add to `~/.config/mcp.json` under `mcpServers` (tools appear as
`mcp_browser_navigate`, etc.). Keep `request_timeout_ms` (120000) > server cap
(90000) - see invariant #1.