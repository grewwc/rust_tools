# AGENTS.md - mcp_excel

## Scope

Standalone stdio JSON-RPC **MCP server** driving the **real Microsoft Excel
app** via AppleScript (`osascript`). A single binary the Agent spawns as an MCP
subprocess - the Excel analogue of `mcp_browser` (osascript plays CDP's role).

## Layout

```text
src/main.rs   # #[tokio::main(multi_thread)]; ExcelServer + impl mcp_stdio::McpServer; no session state.
src/osa.rs    # osascript wrapper + AppleScript templates - every Excel quirk lives here.
src/tools.rs  # tools_list schemas + handle_tools_call() dispatch + 9 tool impls + CSV helpers.
```

> Transport (JSON-RPC framing, `cap_text(24K)`, `with_timeout`, dispatch loop)
> lives in the shared `crates/mcp_stdio` lib; this crate implements only the
> `McpServer` trait + tool logic + osascript driver.

## Build / Test

```bash
cargo build -p mcp_excel   # ~10s; zero heavy deps (tokio + serde_json)
```

`cargo check --bin a` is unaffected (not a dependency of `a`). No focused unit
tests - runtime behavior needs a real Excel install. Gate: build + smoke-test
the canonical round-trip `open_workbook -> read_range -> write_cell -> read_cell
-> export_csv -> close_workbook` against a cold Excel.

## AppleScript golden rules (hard-won; mirrored in `osa.rs`)

Every rule was found by real `-50` / `-10003` failures. **Do not "simplify"
templates back into the broken form.**

1. **One `tell worksheet` block = one op kind.** Mixing `set value` (write) then
   `value of` (read) triggers `-10003`; read/write templates are strictly separated.
2. **Address cells as `range "A1"`, never `cell "A1"`.** Cross-call refs to an
   open workbook make `cell` `-10003`; `range` (incl. single cell) is stable.
3. **`open` is asynchronous.** On cold start `open POSIX file` returns before the
   workbook is registered; immediate access `-50`s. `open_workbook` **polls
   `exists workbook` until ready**. Use `open POSIX file`, never `open workbook
   workbook file name` (uncatchable -50 on cold start).
4. **Prefer bulk property over object-ref iteration.** `name of every worksheet`
   is stable; `repeat with ws in (every worksheet)` -50s after cold open.
5. **Read a block via `value of used range`**, rebuild TSV in Rust - osascript
   flattens the 2D list and loses row structure.
6. **`tab` resolves to the literal string "tab"** in `format!` templates - use
   `(ASCII character 9)` for a real tab.

## Invariants (do not break)

1. **No session / no shutdown hook.** The Excel app owns workbook state and
   shares it across independent `osascript` subprocesses. `main` holds no state;
   each tool call is one-shot `osascript -e`. Idempotent `open_workbook`
   (reuse-if-open) stitches calls together.
2. **Server cap < host timeout.** Host kills the subprocess on timeout. Every
   osascript op is wrapped in `with_timeout(op_timeout_ms())` (default **90s**);
   configure host larger (**120s** recommended).
3. **Error messages avoid the host's transport trigger words.** The shared
   timeout error says "operation reached the N ms server cap" on purpose. Raw
   Excel codes (`-50`/`-10003`) pass through verbatim - diagnostic, not triggers.
4. **Only `content[0].text` reaches the model**, capped to **24K** (`cap_text`).
   Large ranges must be exported to disk, not returned inline.
5. **Persist via `export_csv`, not `save_workbook`.** Sandboxed Excel makes
   AppleScript `save workbook as` return `-50` in non-interactive context - a
   **systemic sandbox limitation**, not a syntax bug; do not re-attempt. Export
   reads the range via Excel then writes the file from Rust. `save_workbook`
   stays EXPERIMENTAL with an honest failure explanation.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MCP_EXCEL_OP_TIMEOUT_MS` | `90000` | Per-op server-side cap (keep < host timeout) |

## Tool set (9 verbs - no raw AppleScript exposed)

`open_workbook` · `list_sheets` · `read_cell` · `read_range` · `write_cell` ·
`write_range` · `export_csv` · `save_workbook` (experimental) · `close_workbook`.

Reads/writes operate in Excel's in-memory workbook; use `export_csv` to persist.

## Host registration

Add to `~/.config/mcp.json` under `mcpServers` (tools appear as
`mcp_excel_open_workbook`, etc.). **Keep `request_timeout_ms` (120000) > server
cap (90000).** macOS-only (Excel app + `osascript`).
