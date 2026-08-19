# AGENTS.md - mcp_excel

## Scope

Standalone stdio JSON-RPC **MCP server** driving real Microsoft Excel via AppleScript (`osascript`). Spawned as MCP subprocess — Excel analogue of `mcp_browser` (osascript plays CDP's role).

## Layout

```text
src/main.rs   # #[tokio::main(multi_thread)]; ExcelServer + impl McpServer; no session state
src/osa.rs    # osascript wrapper + AppleScript templates — every Excel quirk lives here
src/tools.rs  # tools_list + handle_tools_call dispatch + 9 tool impls + CSV helpers
```

Transport (`cap_text(24K)`/`with_timeout`/`JsonRpcErr`/dispatch loop) lives in `crates/mcp_stdio`.

## Build / Test

```bash
cargo build -p mcp_excel   # ~10s; tokio+serde_json only
```

`cargo check --bin a` unaffected (not a dep). No unit tests — needs real Excel. Gate: build + smoke-test cold-Excel round-trip `open_workbook -> read_range -> write_cell -> read_cell -> export_csv -> close_workbook`.

## AppleScript golden rules (mirrored in `osa.rs`; found via real `-50`/`-10003` — do not simplify)

1. One `tell worksheet` block = one op kind — mixing `set value` then `value of` → `-10003`; keep read/write templates strictly separate.
2. `range "A1"` not `cell "A1"` — cross-call `cell` refs → `-10003`; `range` (even single cell) is stable.
3. `open` is async — cold `open POSIX file` returns before workbook registered → immediate `-50`; `open_workbook` polls `exists workbook` until ready; use `open POSIX file` (never `open workbook workbook file name` — uncatchable `-50`).
4. Bulk property over iteration — `name of every worksheet` stable; `repeat with ws in (every worksheet)` → `-50` after cold open.
5. Block read via `value of used range`, rebuild TSV in Rust — osascript flattens 2D list and loses row structure.
6. `tab` is literal `"tab"` in `format!` — use `(ASCII character 9)` for real tab.

## Invariants (do not break)

1. **No session / no shutdown hook.** Excel owns state across one-shot `osascript -e` calls; `main` holds no state; idempotent `open_workbook` (reuse-if-open) stitches calls.
2. **Server cap < host timeout.** Every op `with_timeout(op_timeout_ms())` default 90s; host 120s recommended.
3. **Error wording avoids transport triggers.** Timeout says "operation reached the N ms server cap"; raw `-50`/`-10003` pass verbatim (diagnostic, not triggers).
4. **Only `content[0].text` (cap 24K via `cap_text`).** Large ranges must be exported to disk, not returned inline.
5. **Persist via `export_csv`, not `save_workbook`.** Sandboxed `save workbook as` → systemic `-50` in non-interactive context (not a syntax bug — do not re-attempt); `export_csv` reads range then Rust writes file; `save_workbook` stays EXPERIMENTAL.

## Environment / Tool set / Host

- `MCP_EXCEL_OP_TIMEOUT_MS=90000` per-op cap (keep < host `request_timeout_ms` 120000). macOS-only (Excel + `osascript`).
- 9 verbs (no raw AppleScript): `open_workbook` · `list_sheets` · `read_cell` · `read_range` · `write_cell` · `write_range` · `export_csv` · `save_workbook` (experimental) · `close_workbook` — reads/writes are in-memory; `export_csv` persists.
- `~/.config/mcp.json` → `mcpServers` (tools `mcp_excel_*`).
