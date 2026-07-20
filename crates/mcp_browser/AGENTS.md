# AGENTS.md — mcp_browser

## Scope

Standalone stdio JSON-RPC **MCP server** giving the `a` Agent browser automation
(navigate / click / type / extract / screenshot ...) by driving an installed
Chrome via `chromiumoxide` (Chrome DevTools Protocol). Not a library — a single
binary the Agent spawns as an MCP subprocess.

## Layout

```text
src/main.rs      # #[tokio::main(multi_thread)]; stdin line loop; JSON-RPC dispatch; owns Option<BrowserSession>
src/jsonrpc.rs   # JsonRpcErr, write_result/write_err (async), text_content(), cap_text(24K), with_timeout()
src/browser.rs   # BrowserSession { browser, page, handler_task, temp_profile_dir }, launch(), ensure_session(), shutdown(), gc_stale_profiles()
src/tools.rs     # initialize/tools_list schemas + handle_tools_call() dispatch + 12 tool impls
```

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
`MCP_BROWSER_HEADLESS=1` to avoid popping a window during CI-style checks.

There are no focused unit tests yet (runtime behavior needs a real Chrome). The
gate is `cargo build -p mcp_browser` + the smoke test above.

## Invariants (do not break)

1. **Timeout coordination — server cap < host timeout.** The host MCP client has
   one `request_timeout_ms` per server (no per-tool override); on timeout it
   **kills + restarts** the subprocess, destroying the browser session. So every
   CDP op is wrapped in `with_timeout(op_timeout_ms())` (default **90s**), and the
   user must configure the host to a larger value (**120s** recommended).
2. **Error messages must avoid the host's transport trigger words.** Never let a
   returned error contain `mcp response timeout`, `broken pipe`,
   `closed the stream`, `process exited`, `failed to read response`, or
   `failed waiting for mcp response` — those make the host kill the subprocess.
   The timeout error uses "reached the N ms mcp_browser server cap" on purpose.
3. **Only `content[0].text` reaches the model.** The host reads a single text
   field. Screenshots therefore **save to disk and return the path** — image
   bytes are never returned. Extracted text/HTML is capped to **24K chars**
   (`cap_text`) because the host offloads results above ~32K to disk.
4. **Session persistence + sequential processing + Handler polling.** `main` owns
   one `Option<BrowserSession>` passed by `&mut`; requests are handled
   **sequentially** (the Agent calls one tool at a time and the host serializes
   the round-trip), so no session lock is needed and newline framing stays
   intact. A single **reused Page** keeps login/cookies across calls. The CDP
   `Handler` **must be polled continuously** (`tokio::spawn` a
   `while handler.next().await` loop in `launch()`), or every CDP call hangs.
5. **launch vs connect.** `Browser::launch` starts a **new** controlled Chrome
   with a throwaway profile — it does NOT hijack the user's open windows. To
   attach an existing `--remote-debugging` instance, set `MCP_BROWSER_WS_URL`
   (uses `Browser::connect`); default is launch.
6. **Per-process profile dir + startup GC — never share the fixed default.**
   chromiumoxide's default `user_data_dir` is a **single fixed** path
   (`<temp>/chromiumoxide-runner`); reusing it makes concurrent instances and a
   previously-killed instance collide on Chrome's `SingletonLock`, so `launch`
   dies with `Failed to create ... SingletonLock: File exists (17)`. `launch()`
   therefore assigns each process its own `<temp>/mcp_browser-profile-<pid>` and
   `purge_singleton_locks()` before starting. Because the host usually **SIGKILLs**
   the subprocess on one-shot exit (uncatchable — `shutdown()` can't run), stale
   profiles are reclaimed by `gc_stale_profiles()` at **startup**: it scans
   `mcp_browser-profile-<pid>` dirs and `remove_dir_all`s those whose pid is dead
   (`kill(pid, 0)`). Do not revert to the shared default dir.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MCP_BROWSER_HEADLESS` | `0` (headed) | `1`/`true` = headless |
| `MCP_BROWSER_CHROME` | `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` | Chrome executable |
| `MCP_BROWSER_WS_URL` | (unset) | If set, attach via `Browser::connect` instead of launching |
| `MCP_BROWSER_USER_DATA_DIR` | (unset) | Explicit Chrome profile dir (persists login/cookies). When set it is reused and **never** GC'd; sharing one across concurrent processes will collide. Unset → per-pid temp dir, GC'd at startup. |
| `MCP_BROWSER_OP_TIMEOUT_MS` | `90000` | Per-op server-side cap (keep < host timeout) |
| `MCP_BROWSER_SCREENSHOT_DIR` | `<temp>/mcp_browser` | Where screenshots land |

## chromiumoxide 0.9 API notes

Confirmed against the vendored source and by a successful build:
- `Browser::launch(config) -> (Browser, Handler)`, `Browser::connect(url)`,
  `Browser::new_page(impl Into<CreateTargetParams>)` (a `&str` works),
  `Browser::pages()`, `Browser::close()`.
- `BrowserConfig::builder()` → `BrowserConfigBuilder` with `.chrome_executable()`,
  `.with_head()`, `.new_headless_mode()`, `.build() -> Result<_, String>`.
- `Page`: `goto`, `wait_for_navigation`, `url`, `get_title`, `find_element`,
  `content`, `evaluate_expression(impl Into<EvaluateParams>)` →
  `EvaluationResult::value() -> Option<&Value>`, `save_screenshot(params, path)`.
- Keyboard/typing live on **`Element`** (`type_str`, `press_key`, `focus`,
  `scroll_into_view`, `click`, `inner_text`, `outer_html`) — the public `Page`
  does not expose `press_key`, so page-level key presses focus `body` first.
- Screenshot: `ScreenshotParams::builder().format(CaptureScreenshotFormat::Png).full_page(bool).build()`;
  format enum at `chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat`.

## Host registration

Add to `~/.config/mcp.json` under `mcpServers` (tools appear as
`mcp_browser_navigate`, etc.). **Keep `request_timeout_ms` (120000) > server cap
(90000).**
