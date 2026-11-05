# AGENTS.md - mcp_browser

## Scope

Standalone stdio JSON-RPC **MCP server** for browser automation (navigate/click/type/extract/screenshot). Not a library — spawned as MCP subprocess via `mcp_stdio::run()`.

**Two drivers, one tool surface:** AppleScript drives the user's Chrome via
`osascript`; CDP uses `chromiumoxide` to attach or launch. Driver selection and
environment settings are owned by the symbols in **Environment variables**.

## Layout

```text
src/main.rs      # #[tokio::main(multi_thread)]; BrowserServer { mode, session } + impl McpServer; gc_stale_profiles() + mcp_stdio::run()
src/browser.rs   # BrowserSession + launch()/ensure_session()/shutdown() + Session enum + DriverMode::from_env()
src/applescript.rs # AppleScript driver: ApplescriptSession + osascript runner + tools
src/tools.rs     # initialize/tools_list schemas + handle_tools_call dispatch + tools
```

Shared transport, text delivery, error wording, and timeout contracts live in
[the mcp_stdio guide](../mcp_stdio/AGENTS.md).

## Build / Test

Capability-gating and AppleScript error-translation helpers have focused tests
that run without Chrome. Protocol/schema dispatch changes require this binary
smoke:

```bash
cargo build -p mcp_browser
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | ./target/debug/mcp_browser
```

Changes to launch/connect, tab ownership, AppleScript execution, DOM interaction
or extraction, human-action detection, or screenshots additionally require a
round-trip through the affected driver in real Chrome. Add `tools/call navigate`
and the affected operation; CDP may use headless mode where applicable. For
AppleScript, enable JS-from-Apple-Events and verify that only the session tab
opens/closes.

## Invariants (do not break)

1. **Browser timeout policy.** Operations share `op_timeout_ms`
   (`src/browser.rs`), which clamps environment overrides to the ceiling defined
   in code; do not introduce per-tool cap overrides. Shared ordering is defined
   by **Per-op timeout < host timeout** in `crates/mcp_stdio/AGENTS.md`.
2. **Error translation.** `translate_err` (`src/applescript.rs`) and CDP error
   construction obey **Error wording** in `crates/mcp_stdio/AGENTS.md`.
3. **Screenshot results.** Save captures to disk and return the path, never
   inline image bytes; shared delivery is defined by **Text delivery** in
   `crates/mcp_stdio/AGENTS.md`.
4. **Sequential + Handler polling.** `main` owns one `Session` (`Option<Session>`) by `&mut`; host serializes requests so one reused `Page` keeps cookies. CDP `Handler` must be continuously polled, else all CDP hangs.
5. **launch vs connect.** `Browser::launch` = new controlled Chrome (throwaway profile). `Browser::connect` (when `MCP_BROWSER_WS_URL` set) = attach existing `--remote-debugging` instance. Default is launch. `owns_browser` records which one happened: shutdown closes the whole browser only when this process launched it, an attached browser only ever loses its session tab.
6. **Per-pid profile dir + startup GC.** Controlled launches use per-pid profiles;
   `gc_stale_profiles` (`src/browser.rs`) reclaims dead owners' directories, never
   explicit user profiles. Keep the startup deadline and failed-page cleanup in
   `BrowserSession::launch` (`src/browser.rs`), which runs outside the tool wrapper.
7. **Human-required detection + `wait_for_human`.**
   - **Detect** captcha/slider/sms_otp/twofa/login_required/payment_verify/
     identity_verify in `get_text`/`get_html` and after every mutating call
     (`navigate`/`click`/`type_text`/`press_key`/`scroll`/`wait_for`/`screenshot`/
     `wait_for_human`); one `evaluate_expression` pass per call, and a JS error
     means "no signal".
   - **Signals must be visibility-gated** — hidden inputs/iframes never count, and
     **no raw `innerHTML` substring matching** (SPA script text false-positives);
     text clues come from visibility-safe `body.innerText` plus a visible input.
     Both drivers share the same keyword rules.
   - **Tag** `[USER_ACTION_REQUIRED: <category>]` **after** `cap_text` (never
     truncated), from `user_action_tag()` — one copy per driver (`src/tools.rs`,
     `src/applescript.rs`), kept byte-identical; cap the body once
     and append the tag afterwards — a second cap over the summary cuts the tag
     off on long pages. Captcha counts only while present *and* unsolved: the
     solved state is read from the **host** document
     (`textarea[name="g-recaptcha-response"]`), never from the widget iframe
     (cross-origin → `contentDocument` is null). `pending_human` is set on a
     hit and cleared on a miss (`detect_and_mark` in both drivers), so mutating
     tools prepend `[HUMAN_ACTION_PENDING: <category>]` exactly while a blocker
     is still on the page.
   - **`wait_for_human`** waits in bounded segments under the host timeout and returns `status=still_waiting` — a normal result, not an error:
     the model ends the turn and asks the user. Headless reports
     `status=unavailable`. Detector internals live in `src/tools.rs` (CDP) and
     `src/applescript.rs` (AppleScript).

## AppleScript driver specifics

Same tools/args as CDP; `wait_for_human` never `unavailable`. Dedicated invariants:

1. **Never steal focus.** No `activate`; `open_tab` records the frontmost app before creating the tab and restores it afterwards (skip when already in Chrome). Ops address `window id W` + `whose id is "T"` (int window id + quoted tab id; a stale reference fails with the -1728 `can't get` error that the server maps to a rebuild hint) → stray events fail dry instead of mutating foreign tabs. Chrome self-activates on a new tab — hence the restore.
2. **Allow JavaScript from Apple Events** (View > Developer). Chrome reads this preference **at startup only**: enabling it in a running browser (or editing the profile on disk) changes nothing, so the error must name the required full Chrome restart, not just the menu path. The wording lives in one shared constant (`JS_APPLE_EVENTS_HINT` in `src/applescript.rs`) used by both error paths, so they cannot drift.
3. **JS embeds as one AppleScript string literal** — escape `\`/`"`/newlines via `esc_applescript`; the whole script goes over `osascript -` stdin, no temp files. Multi-line JS (detect/wait polling) is fine.
4. **One session tab, reused.** `navigate` re-sets `URL of` tab unless vanished (then fresh tab in session window; first session uses front window). Tab becomes active tab *of its window* (for `screencapture -l`) but window stays background.
5. **Synthetic events only.** JS `click()`, native-setter `value`+`input`/`change`, `KeyboardEvent` dispatch (Enter also `form.requestSubmit()`). No OS key chords / drag&drop. `full_page` degrades to a window screenshot.
6. **Screenshot TCC preflight.** `screenshot` probes Screen Recording permission first: without it the window capture fails while the region/full-screen fallbacks still exit 0 with a wallpaper-only image that looks like success — so the server must error out with authorization instructions instead of returning a useless frame. The probe and fallback chain live in `src/applescript.rs`; the grant applies only after fully quitting and restarting the host app.
7. **AppleScript timeout wrapper.** `handle_tools_call` (`src/applescript.rs`)
   wraps operations with `with_timeout` using the shared browser cap; CDP-only
   environment options do not apply.
8. **Results are decoded, not escaped.** `get_text`/`get_html` parse the `JSON.stringify` envelope, so the model sees real newlines and real HTML instead of `\n` and `\"`.

## Environment variables

Driver selection and CDP launch/attach settings belong to `DriverMode::from_env`
and `BrowserSession::launch` (`src/browser.rs`). Timeout defaults and clamps
belong to `op_timeout_ms` there; screenshot path settings belong to
`resolve_screenshot_path` (`src/tools.rs`). Keep option values/defaults in those
symbols rather than copying them into this guide.

## Keyboard focus

Keyboard/typing must go through **`Element`**; `Page` has no `press_key`. `Element::press_key` sends a page-level `Input.dispatchKeyEvent`, so the key lands on the **focused** element — never focus `body` first, that would move focus off the input the model just typed into. Focus `body` only when nothing else is focused. The element/screenshot API details live in `src/tools.rs`.

## Host registration

Host tool family: `mcp_browser_*`. Shared configuration is described in
**Host registration** in `crates/mcp_stdio/AGENTS.md`.
