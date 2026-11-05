# AGENTS.md - mcp_excel

## Scope

Standalone stdio JSON-RPC **MCP server** driving real Microsoft Excel via AppleScript (`osascript`). Spawned as MCP subprocess — Excel analogue of `mcp_browser` (osascript plays CDP's role).

## Layout

```text
src/main.rs   # #[tokio::main(flavor = "multi_thread")]; ExcelServer + impl McpServer; no session state
src/osa.rs    # osascript wrapper + AppleScript templates — every Excel quirk lives here
src/tools.rs  # tools_list + handle_tools_call dispatch + tools + CSV helpers
```

Shared transport, text delivery, error wording, and timeout contracts live in
[the mcp_stdio guide](../mcp_stdio/AGENTS.md).

## Build / Test

Crate type-check target: `cargo check -p mcp_excel`. Protocol/schema dispatch
changes require a binary protocol smoke. Changes to AppleScript execution,
workbook lifecycle, reads/writes, or CSV export additionally require
`cargo build -p mcp_excel` and a cold-Excel round-trip on a test workbook:
`open_workbook -> read_range -> write_cell -> read_cell -> export_csv -> close_workbook`.

## AppleScript rules (binding — do not simplify)

The rule set lives with the templates: `src/osa.rs` module docstring,
"已验证的 AppleScript 铁律" (7 rules, traced to real `-50`/`-10003`
diagnostic). The three that cost the most when broken:

- One `tell worksheet` block = one op kind — mixing a write then a read → `-10003`.
- `range "A1"`, never `cell "A1"` — cross-call `cell` refs → `-10003`.
- `open` is async — `open_workbook` polls until the workbook is registered, or the next call → `-50`.

## Invariants (do not break)

1. **No session / no shutdown hook.** Excel owns state across one-shot `osascript -e` calls; `main` holds no state; idempotent `open_workbook` (reuse-if-open) stitches calls.
2. **Raw AppleScript diagnostics.** Preserve `-50`/`-10003` verbatim while
   obeying shared **Error wording** in `crates/mcp_stdio/AGENTS.md`.
3. **Large ranges use CSV export.** Export large ranges to disk under shared
   **Text delivery** in `crates/mcp_stdio/AGENTS.md`.
4. **Persist via `export_csv`, not `save_workbook`.** Sandboxed `save workbook as` → systemic `-50` in non-interactive context (not a syntax bug — do not re-attempt); `export_csv` reads range then Rust writes file; `save_workbook` stays EXPERIMENTAL.

## Environment / Tool surface

- macOS-only (Excel + `osascript`). `op_timeout_ms` (`src/tools.rs`) owns the
  operation-cap environment setting; shared ordering lives in the mcp_stdio guide.
- Tool schemas belong to `tools_list_result` (`src/tools.rs`); expose workbook
  operations rather than raw AppleScript. Reads/writes are in-memory;
  `export_csv` persists and `save_workbook` remains experimental.
- Host tool family: `mcp_excel_*`; shared configuration is described in
  **Host registration** in `crates/mcp_stdio/AGENTS.md`.
