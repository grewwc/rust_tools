# AGENTS.md - mcp_computer

## Scope

Standalone MCP server that gives the `a` agent **generic computer-use** control of
the local desktop: screen capture, display / window / application queries, and
mouse / keyboard / scroll injection. Reuses the `mcp_stdio` skeleton for the
protocol and adds only the tool surface plus one backend per platform.

Product framing: this is the *desktop* counterpart of `mcp_browser` (which drives
one browser). It is deliberately app-agnostic — it drives whatever is on screen —
so it is the fallback path for any GUI that has no dedicated MCP server.

Shared transport, text delivery, error wording, and timeout contracts live in
[the mcp_stdio guide](../mcp_stdio/AGENTS.md).

## Layout

```text
src/main.rs              # mod backend; mod tools; -> mcp_stdio::run(ComputerServer)
src/tools.rs             # tool schemas + tools/call dispatch + per-op timeout
src/frame.rs             # capture scope + temporary probe lifecycle + visual frame ids
src/backend.rs           # the platform seam: cfg-selects one backend, nothing else
src/backend/macos/       # CoreGraphics: input.rs (CGEvent), capture.rs (screencapture),
                         #   screen.rs (displays / windows / permissions)
src/backend/linux.rs     # the session's own CLIs: grim|scrot|import, xdotool|ydotool|wtype
src/backend/unsupported.rs # every call reports a clear "no backend on this OS" error
```

## Invariants (do not break)

1. **Adding a platform must not touch `tools.rs`.** A backend exposes exactly these
   signatures; `backend.rs` picks one by `cfg`:
   `screenshot(Option<(f64, f64, f64, f64)>, Option<u32>, Option<u32>, bool, &Path) -> Result<Screenshot, String>`,
   `click(f64, f64, &str, u32, &[String])`, `drag(f64, f64, f64, f64, &str, &[String], u32)`,
   `move_mouse(f64, f64)`, `scroll(Option<f64>, Option<f64>, f64, f64, &str)`,
   `type_text(&str)`, `press_key(&str)` (all `Result<(), String>`),
   `activate_app(String)`, `activate_window(u32)`, `list_windows(Option<String>)`,
   `screen_info()` (all `Result<String, String>`).
   `Screenshot` carries the model-facing report plus an optional image-pixel to
   screen-point mapping. Queries return a report; actions return `Ok(())` or an error.
   Coordinates are global screen points, top-left origin, on every platform.
2. **No silent no-ops.** A parameter a platform cannot honour (the `display`
   parameter or the `pixel` scroll unit on Linux, `include_cursor` with
   ImageMagick's `import`) is an error naming the limitation, never a dropped
   argument. Placeholders models send for unused optionals (`""`, `[]`, an
   all-zero `region`, `window_id`/`display` ≤ 0) are invalid values on every
   backend and mean "not given"; normalize them before backend dispatch.
3. **Screenshots return a path.** Follow shared **Text delivery** in
   `crates/mcp_stdio/AGENTS.md`; `a`'s image-aware `read_file` shows the saved frame
   to a vision model. `frame.rs` binds each `frame_id` to its capture scope; unchanged
   probe files are deleted. A kept HiDPI frame is stored at one pixel per point
   (the report's click formula always describes the stored file); the
   `frame_id` digest and OCR use the native pixels. Kept frames are scratch:
   `a`'s `read_file` copies an image into the session's assets before showing
   it, so the session owns what the model saw and this server only prunes its
   own frames by age — never by count, since every `a` process runs its own
   server over the shared directory. `find_text`/`click_text` reuse
   `rust_tools::pdfw` OCR; `click_text` never acts unless exactly one OCR
   line matches, and for window captures it activates and verifies the exact
   window — before the fresh OCR frame and again immediately before input.
   An activation failure returns without a click; `find_text` never activates.
4. **Helper error boundary.** Each backend's `sanitize` scans helper stderr
   before forwarding it; apply shared **Error wording** in
   `crates/mcp_stdio/AGENTS.md` at this boundary.
5. **Backend calls run under `with_timeout` on a blocking thread.** Capture and
   injection block (subprocess + sleeps), so `tools.rs` goes through `spawn_blocking`:
   run inline they would hold the server's only executor thread and the per-op cap
   from `op_timeout_ms` (`src/tools.rs`) could never fire. Shared timeout ordering
   is defined in `crates/mcp_stdio/AGENTS.md`.
   On timeout the helper is left running to completion — the cap bounds what the
   model waits for, not the platform.
6. **Permissions are reported, not assumed.** `screen_info` states whether
   Accessibility (input injection) and Screen Recording (capture) are granted,
   because an empty capture from a missing grant is otherwise indistinguishable
   from an empty screen.

## Build / verify

Crate type-check target:

```bash
cargo check -p mcp_computer                 # macOS backend
```

Protocol/schema dispatch changes require a built-server smoke (no host needed),
replacing `<bin>` with the built path:

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"screen_info","arguments":{}}}' \
  | <bin>
```

Capture, coordinate mapping, OCR targeting, input injection, window activation,
or permission-reporting changes additionally require the affected tool round-trip
in a real desktop session on the affected OS.

`src/backend/linux.rs` is `cfg(target_os = "linux")`, so a macOS build never parses
it. It is std-only and free of `use super::*` so it can be type-checked on macOS:

```bash
rustc --edition 2024 --crate-type lib --emit=metadata -o /dev/null \
  crates/mcp_computer/src/backend/linux.rs
```

That check proves it compiles, not that the Linux helpers behave.

## Deployment

Build and install the release artifact when deploying:

```bash
cargo build -p mcp_computer --release
cp target/release/mcp_computer bin/mcp_computer
```

## Platform verification notes

- Compile checks do not establish Linux helper behavior; exercise the affected
  X11 or Wayland session before claiming support for it.
- Pixels-per-point must be taken from the display *mode* (`CGDisplayCopyDisplayMode` +
  `CGDisplayMode::pixel_width`); `CGDisplayPixelsWide` returns the *point* size on a
  HiDPI mode. `frame.rs` also measures the ratio from the decoded capture against the
  frame the backend reports, so the pixel→point mapping stays right for window
  captures too.
