# AGENTS.md - mcp_excel

## Scope

Standalone stdio JSON-RPC **MCP server** driving real Microsoft Excel via AppleScript (`osascript`). Spawned as MCP subprocess — Excel analogue of `mcp_browser` (osascript plays CDP's role).

## Layout

```text
src/main.rs   # #[tokio::main(multi_thread)]; ExcelServer + impl McpServer; no session state
src/osa.rs    # osascript wrapper + AppleScript templates — every Excel quirk lives here
src/tools.rs  # tools_list + handle_tools_call dispatch + 9 tool impls + CSV helpers
```

Transport (`cap_text(24K)`/`with_timeout`/`JsonRpcErr`/dispatch loop) lives in `crates/mcp_stdio`, whose `AGENTS.md` owns the shared transport invariants (per-op cap < host timeout, error wording, text cap).

## Build / Test

```bash
cargo build -p mcp_excel   # ~10s; third-party deps tokio+serde_json (+ local mcp_stdio)
```

`cargo check --bin a` unaffected (not a dep). No unit tests — needs real Excel. Gate: build + smoke-test cold-Excel round-trip `open_workbook -> read_range -> write_cell -> read_cell -> export_csv -> close_workbook`.

## AppleScript rules (binding — do not simplify)

The rule set lives with the templates: `src/osa.rs` module docstring,
"已验证的 AppleScript 铁律" (7 rules, each traced to a real `-50`/`-10003`
diagnostic). The three that cost the most when broken:

- One `tell worksheet` block = one op kind — mixing a write then a read → `-10003`.
- `range "A1"`, never `cell "A1"` — cross-call `cell` refs → `-10003`.
- `open` is async — `open_workbook` polls until the workbook is registered, or the next call → `-50`.

## Invariants (do not break)

1. **No session / no shutdown hook.** Excel owns state across one-shot `osascript -e` calls; `main` holds no state; idempotent `open_workbook` (reuse-if-open) stitches calls.
2. **Raw AppleScript errors pass verbatim.** `-50`/`-10003` are diagnostics, not host transport triggers; never wrap them in wording the host kills on (`mcp_stdio`).
3. **Large ranges must be exported to disk, not returned inline** — only `content[0].text` reaches the model.
4. **Persist via `export_csv`, not `save_workbook`.** Sandboxed `save workbook as` → systemic `-50` in non-interactive context (not a syntax bug — do not re-attempt); `export_csv` reads range then Rust writes file; `save_workbook` stays EXPERIMENTAL.

## Environment / Tool set / Host

- `MCP_EXCEL_OP_TIMEOUT_MS=90000` per-op cap (keep < host `request_timeout_ms` 120000). macOS-only (Excel + `osascript`).
- 9 verbs (no raw AppleScript): `open_workbook` · `list_sheets` · `read_cell` · `read_range` · `write_cell` · `write_range` · `export_csv` · `save_workbook` (experimental) · `close_workbook` — reads/writes are in-memory; `export_csv` persists.
- `~/.config/mcp.json` → `mcpServers` (tools `mcp_excel_*`).
