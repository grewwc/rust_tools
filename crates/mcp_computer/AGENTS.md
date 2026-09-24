# AGENTS.md - mcp_computer

## Scope

Standalone MCP server that gives the `a` agent **generic computer-use** control of
the local desktop: screen capture, display / window / application queries, and
mouse / keyboard / scroll injection. Reuses the `mcp_stdio` skeleton for the
protocol (read loop, dispatch, `with_timeout`, `cap_text`) and adds only the tool
surface plus one backend per platform.

Product framing: this is the *desktop* counterpart of `mcp_browser` (which drives
one browser). It is deliberately app-agnostic — it drives whatever is on screen —
so it is the fallback path for any GUI that has no dedicated MCP server.

## Layout

```text
src/main.rs              # mod backend; mod tools; -> mcp_stdio::run(ComputerServer)
src/tools.rs             # tool schemas + tools/call dispatch + per-op timeout
src/backend.rs           # the platform seam: cfg-selects one backend, nothing else
src/backend/macos/       # CoreGraphics: input.rs (CGEvent), capture.rs (screencapture),
                         #   screen.rs (displays / windows / permissions)
src/backend/linux.rs     # the session's own CLIs: grim|scrot|import, xdotool|ydotool|wtype
src/backend/unsupported.rs # every call reports a clear "no backend on this OS" error
```

## Invariants (do not break)

1. **Adding a platform must not touch `tools.rs`.** A backend exposes exactly these
   signatures; `backend.rs` picks one by `cfg`:
   `screenshot(Option<(f64, f64, f64, f64)>, Option<u32>, Option<u32>, bool) -> Result<String, String>`,
   `click(f64, f64, &str, u32, &[String])`, `drag(f64, f64, f64, f64, &str, &[String], u32)`,
   `move_mouse(f64, f64)`, `scroll(Option<f64>, Option<f64>, f64, f64, &str)`,
   `type_text(&str)`, `press_key(&str)` (all `Result<(), String>`),
   `activate_app(String)`, `list_windows(Option<String>)`, `screen_info()` (all `Result<String, String>`).
   Queries return a model-facing report; actions return `Ok(())` or a model-facing error.
   Coordinates are global screen points, top-left origin, on every platform.
2. **No silent no-ops.** A parameter a platform cannot honour (a window id without
   an X11 capture tool, the `pixel` scroll unit on Linux) is an error naming the
   limitation, never a dropped argument.
3. **Screenshots return a path, not image content.** The host reads only
   `content[0].text`, so bytes in the envelope would be discarded; the path plus
   `a`'s image-aware `read_file` is what shows the frame to a vision model.
4. **Error text must avoid the host's transport trigger words** (`mcp response
   timeout`, `broken pipe`, `closed the stream`, `process exited`, `failed to read
   response`, `failed waiting for mcp response` — `mcp/client.rs`). Helper stderr is
   forwarded to the model, so each backend scans it (`sanitize`) first; forwarding
   one of those phrases makes the host kill and restart this server.
5. **Backend calls run under `with_timeout` on a blocking thread.** Capture and
   injection block (subprocess + sleeps), so `tools.rs` goes through `spawn_blocking`:
   run inline they would hold the server's only executor thread and the per-op cap
   (default 90s, `MCP_COMPUTER_OP_TIMEOUT_MS`) could never fire before the host's
   120s `request_timeout_ms` kills the subprocess. On timeout the helper is left
   running to completion — the cap bounds what the model waits for, not the platform.
6. **Permissions are reported, not assumed.** `screen_info` states whether
   Accessibility (input injection) and Screen Recording (capture) are granted,
   because an empty capture from a missing grant is otherwise indistinguishable
   from an empty screen.

## Build / verify

```bash
cargo check -p mcp_computer                 # macOS backend
cargo build -p mcp_computer --release && cp target/release/mcp_computer bin/mcp_computer
```

Smoke test (no host needed), replacing `<bin>` with the built path:

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"screen_info","arguments":{}}}' \
  | <bin>
```

`src/backend/linux.rs` is `cfg(target_os = "linux")`, so a macOS build never parses
it. It is std-only and free of `use super::*` so it can be type-checked on macOS:

```bash
rustc --edition 2024 --crate-type lib --emit=metadata -o /dev/null \
  crates/mcp_computer/src/backend/linux.rs
```

That check proves it compiles, not that the Linux helpers behave — a real Linux
session (X11 and Wayland) still has to be exercised before claiming Linux support.

## Status / known gaps

- Linux backend is compile-checked only (never run on a Linux host).
- Pixels-per-point must be taken from the display *mode* (`CGDisplayCopyDisplayMode` +
  `CGDisplayMode::pixel_width`); `CGDisplayPixelsWide` returns the *point* size on a HiDPI
  mode, which described this Retina screen as 1.00 px/point while its captures came back
  3024x1964 px. `capture.rs` also re-measures the ratio from the PNG it wrote, so the
  pixel→point mapping the model is given stays right for window captures too.
- `screen_info` reports `cursor: unknown` when CoreGraphics refuses the null mouse
  event used to read the pointer position (`screen.rs::cursor_position`).