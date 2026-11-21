# AGENTS.md - mcp_browser

## Scope

Standalone stdio JSON-RPC MCP server for browser automation. Shared transport,
text delivery, error wording, and timeout contracts live in
[the mcp_stdio guide](../mcp_stdio/AGENTS.md).

**Isolation is the default.** `DriverMode::from_env`, `headless_from_env`, and
`BrowserSession::launch` (`src/browser.rs`) select a self-launched headless CDP
browser with a dedicated temporary profile on every platform. Model activity
must not change the user's ordinary browser tabs, focus, profile, or cookies.
The capability preflight checks the executable without opening a window or tab;
a default headless configuration requires no GUI login session.

## Layout

```text
src/main.rs        # BrowserServer, capability gating, shared call deadline, shutdown
src/browser.rs     # driver/config parsing, preflight, launch, profile GC, owned tabs
src/attach.rs      # AttachGuard: ownership enforcement before CDP reaches an external browser
src/tools.rs       # MCP schemas, CDP dispatch, shared human-action detector
src/contract.rs    # argument validation, JSON encoding, wait/extraction scripts
src/snapshot.rs    # native accessibility tree, stable DOM refs, trusted CDP actions
src/applescript.rs # explicit AppleScript mode, stable tab ids, synthetic actions
```

## Build / Test

Follow the root verification ladder and scope Cargo commands to this crate.
Pure validation, configuration, encoding, ownership, and tab-id tests run without
Chrome. Protocol/schema changes additionally require the binary smoke:

```bash
cargo check -p mcp_browser
cargo test -p mcp_browser browser_fixes_
cargo build -p mcp_browser
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | ./target/debug/mcp_browser
```

Changes to launch/connect, ownership, DOM interaction, human-action inspection,
extraction, or screenshots also require real Chrome round-trips through the
affected driver. Test CDP with an isolated headless instance. An attach smoke
must use a separate debugging browser and verify that existing targets survive
session cleanup. AppleScript verification requires JS-from-Apple-Events; inspect
its capability before running mutations in the user's Chrome. Keep automated
unit tests independent of browser availability. Unicode insertion, URL waits,
action verdicts, and requestSubmit behavior are fully verified only by live
CDP/AppleScript round-trips; pure tests establish their command and verdict
contracts without claiming site behavior.

## Invariants

1. **One call deadline.** `op_timeout_ms` (`src/browser.rs`) owns defaults and
   clamps positive overrides below the host default deadline. `BrowserServer`
   (`src/main.rs`) includes lazy CDP startup and the operation in one deadline.
   Human waits use `human_wait_budget_ms`, bounded polls, and reserved return
   time; shorter host deadlines require an appropriately shorter server cap.
2. **Ownership controls cleanup.** Self-launch shutdown closes the controlled
   browser. Attach shutdown closes only targets explicitly created by the
   session. `AttachGuard` (`src/attach.rs`) also blocks chromiumoxide's automatic
   `Target.closeTarget` on foreign targets after initialization failures and
   rejects whole-browser closure. Keep the guard on every live attach path.
3. **Profile protection.** Startup GC (`gc_stale_profiles`, `src/browser.rs`)
   reclaims temp-dir entries matching the `mcp_browser-profile-<pid>` pattern
   whose owner PID is dead — pattern-based, so an explicit
   `MCP_BROWSER_USER_DATA_DIR` pointing at a matching directory can still be
   deleted. Other explicit profile directories retain their contents across
   sessions; `purge_singleton_locks` (`src/browser.rs`) removes singleton
   locks only for auto-generated profiles and never for a user-set directory.
4. **Sequential session, continuously polled handler.** `BrowserServer` owns a
   mutable `Session`; the CDP `Handler` must keep being polled for calls to make
   progress. `switch_tab` changes the automation target without activating a
   window. Shutdown ownership must be independent of the selected target.
5. **Strict tool arguments.** `parse_call` (`src/contract.rs`) validates the
   published schema before any browser mutation. Wrong optional types, nulls,
   empty selectors, negative coordinates/offsets, and meaningless zero limits
   are invalid parameters. Supply exactly one of `selector` or `ref` for click
   and typing. Empty `text` intentionally clears an editable field.
6. **Evidence after actions.** Navigate reports final URL and observed load
   outcome. Click, typing, keys, and scroll report current URL, readyState, and
   scroll coordinates. Click and typing include applied/not_applied/unknown
   verdicts backed by click target state/focus or exact value read-back. Disabled
   targets and mismatched/sanitized values must not claim application. Inspection
   failures explicitly report unknown evidence; password read-back is redacted.
7. **Bounded results.** Extraction slices Unicode characters in the page with
   `max_chars` and `offset`, then reports `total_chars` and `next_offset`.
   Snapshots page nodes with `offset`/`limit` and preserve valid bounded JSON.
   Oversized fields carry explicit truncation markers. Continuation must advance
   when more nodes remain; an oversized identity errors instead of stalling.
   Append human-action markers after the bounded body. Screenshots return an
   absolute file path rather than image bytes.
8. **Human-action correctness.** Both drivers use `DETECT_USER_ACTION_JS`
   (`src/tools.rs`), with visible DOM evidence and host-document captcha
   response tokens. Hidden inputs/iframes and script contents are not blockers.
   Successful empty inspections clear `pending_human`; failed inspections keep
   it and report unknown. `wait_for_human` resolves only after consecutive
   successful clean polls, returns `still_waiting` for a continuing blocker,
   `unknown` with the inspection reason, and `unavailable` for self-launched
   headless mode. A pending/unknown result asks the model to return control to
   the user.
9. **Error wording.** The shared error wording contract is only partially met:
   CDP construction errors embed the underlying message (`src/browser.rs`) and
   unknown AppleScript errors pass through unchanged via `translate_err`
   (`src/applescript.rs`), so a returned error containing a host restart
   trigger phrase (`is_transport_error` / `is_protocol_desync_error` /
   `is_user_interrupt_error` in `src/bin/ai/mcp/client.rs`) can still restart
   the host connection.

## Tool contracts

Existing names remain available: navigate, click, type_text, press_key, scroll,
wait_for, evaluate_js, get_text, get_html, wait_for_human, screenshot, list_tabs,
close_browser. Additional names are switch_tab, close_tab, and snapshot.

- `type_text` replaces the complete input/textarea value on both drivers,
  including clearing with empty text. Both drivers support Unicode (Chinese,
  emoji, and other text): CDP uses native select-all/deletion and Input.insertText;
  AppleScript uses the native value setter plus input/change events. Read back
  the field before submission and compare it with the requested value. With
  submit=true, CDP dispatches Enter and AppleScript uses requestSubmit() after
  confirming the replacement. AppleScript reports the observed submit event;
  missing forms, validation failures, and submit exceptions produce errors.
- `evaluate_js` returns JSON text on both drivers: strings stay quoted,
  objects/arrays are serialized, and undefined becomes null. Runtime and
  serialization exceptions produce errors. Multi-line code retains its lines.
  The result passes through `cap_text` (`src/tools.rs`): oversized values are
  cut and a truncation marker appended, so only sub-cap results are guaranteed
  valid JSON.
- `list_tabs` reports stable ids plus readable indices and URLs. CDP ids are
  target ids; AppleScript ids combine window and tab ids and survive tab
  reordering. `switch_tab(id)` selects the automation target. `close_tab(id)`
  protects foreign tabs in attached CDP and AppleScript modes. Closing the
  current CDP tab selects another controlled tab or creates a blank replacement;
  AppleScript clears its target when no owned tab remains and navigate rebuilds.
- `wait_for` takes exactly one of selector or url. Selector conditions are
  present (default), visible, and absent. The url parameter waits for exact
  string equality to location.href, including query and fragment; condition
  applies only to selectors, and URL-plus-selector/ref arguments are invalid.
  Its own timeout_ms deadline is subordinate to the call cap; both drivers
  report actual elapsed time and the observed current URL.
- `snapshot(offset, limit)` uses Chrome's Accessibility domain for native
  roles, accessible names/labels, descriptions, and stable refs accepted by
  click/type_text. Refs are scoped to their target and remain usable while the
  DOM node exists; navigation or node replacement may invalidate them. Snapshot
  explicitly returns an unsupported-mode error under AppleScript. Oversized
  fields include [truncated] markers and truncated_fields metadata; paging
  advances even when one source node would exceed the output budget.

## Environment and isolation trade-offs

Configuration parsing belongs to `DriverMode::from_env`, `BrowserSession::launch`,
`headless_from_env`, and `op_timeout_ms` (`src/browser.rs`); keep numeric tuning
values in code. Screenshot paths belong to `resolve_screenshot_path`
(`src/tools.rs`).

- No overrides: isolated CDP self-launch, headless, throwaway profile.
- `MCP_BROWSER_CHROME`: explicit Chrome/Chromium executable; otherwise detect it.
- `MCP_BROWSER_HEADLESS=0`: explicitly open a visible controlled browser. Its
  profile remains separate, but a new window can affect desktop focus.
- `MCP_BROWSER_USER_DATA_DIR`: explicitly persist a dedicated automation profile.
  Do not point it at the user's ordinary profile or share it across running
  processes. Locked profiles fail normally; their locks are never removed.
- `MCP_BROWSER_WS_URL`: explicitly attach an existing debugging browser; it takes
  precedence over driver selection. This shares that browser's profile and
  cookies, and CDP initialization or operations can affect its tabs/settings.
  Use a dedicated debugging instance to preserve isolation. The attach guard
  protects foreign targets from closure, including automatic failure cleanup.
- `MCP_BROWSER_DRIVER=applescript`: explicitly drive the user's Chrome on macOS.
  It shares their cookies/profile and can change the active tab when creating
  a session tab or capturing its window. Focus restoration is best effort;
  this mode cannot provide the default CDP isolation guarantee.
- `MCP_BROWSER_OP_TIMEOUT_MS`: positive override, clamped by the server ceiling;
  invalid/zero values use the default. Configure below a shorter host deadline.
- `MCP_BROWSER_SCREENSHOT_DIR`: default directory for screenshots.

## AppleScript specifics

No `activate` calls or OS keyboard chords. Scripts target stable window/tab ids,
escape JS as one AppleScript string literal, and travel through osascript stdin.
Only tabs created by this session are closed, even after switching to a foreign
one. Synthetic clicks, native setter input/change events, and KeyboardEvents
have site limitations; `full_page` degrades to a window screenshot.

Chrome requires View > Developer > Allow JavaScript from Apple Events and a full
Chrome restart after changing that preference. `JS_APPLE_EVENTS_HINT` and
`translate_err` (`src/applescript.rs`) own actionable error translation. Window
screenshots require Screen Recording permission and its preflight must run before
capture; denied permission produces an error instead of a misleading desktop
image. `switch_tab` does not select the visible Chrome tab.

## Keyboard focus

CDP keyboard input must keep the intended element focused. Element keyboard
methods send page-level input events; fetching a body handle does not mean body
should be focused. `replace_text` (`src/snapshot.rs`) and `tool_press_key`
(`src/tools.rs`) own these focus rules.

## Host registration

Host tool family: `mcp_<mcpServers key>_*` — the host derives the prefix from
the arbitrary `mcpServers` config key (`src/bin/ai/mcp/client.rs`), so
`mcp_browser_*` requires the key `browser`, not `mcp_browser`. Shared
configuration is described in **Host registration** in the mcp_stdio guide.
Tool additions require the client to refresh tools/list; no client-side changes
are part of this crate.
