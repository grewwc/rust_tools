# AGENTS.md — mcp_excel

## Scope

Standalone stdio JSON-RPC **MCP server** giving the `a` Agent spreadsheet
automation (open / read / write / export ...) by driving the **real installed
Microsoft Excel app** via AppleScript (`osascript`). Not a library — a single
binary the Agent spawns as an MCP subprocess. This is the Excel analogue of
`mcp_browser`: `osascript` here plays the role CDP plays there.

## Layout

```text
src/main.rs      # #[tokio::main(multi_thread)]; ExcelServer (unit) + impl mcp_stdio::McpServer; mcp_stdio::run(). NO session state.
src/osa.rs       # osascript wrapper + AppleScript templates. THE core — every Excel quirk is captured here.
src/tools.rs     # initialize/tools_list schemas + handle_tools_call() dispatch + 9 tool impls + CSV helpers
```

> The JSON-RPC transport (JsonRpcErr / write_result / write_err / text_content /
> cap_text(24K) / with_timeout) and the stdin dispatch loop live in the shared
> `crates/mcp_stdio` lib crate. This crate only implements the `McpServer` trait
> (initialize / tools_list / tools_call; `shutdown` uses the no-op default since
> there is no session) plus its tool logic + osascript driver.

## Build / Test

```bash
cargo build -p mcp_excel   # fast (~10s); zero heavy deps (only tokio + serde_json)
```

`cargo check --bin a` stays unaffected — this crate is a workspace member but not
a dependency of `a`.

Smoke test (pipe JSON-RPC lines to the binary):

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | ./target/debug/mcp_excel
```

Full round-trip against real Excel needs a workbook file. From a cold Excel
(`osascript -e 'tell application "Microsoft Excel" to quit saving no'`), the
canonical end-to-end sequence is: `open_workbook → read_range → write_cell →
read_cell → export_csv → close_workbook`. There are no focused unit tests
(runtime behavior needs a real Excel install); the gate is
`cargo build -p mcp_excel` + this sequence returning all-OK.

## AppleScript golden rules (hard-won; mirrored in `osa.rs` module docs)

Every one of these was found by real -50 / -10003 failures. **Do not "simplify"
the templates back into the broken form.**

1. **One `tell worksheet` block does one kind of op** — pure read or pure write.
   Writing (`set value`) then reading (`value of`) in the same block triggers
   `-10003 access not allowed`. Read/write templates are strictly separated.
2. **Always address cells as `range "A1"`, never `cell "A1"`.** Across-call
   references to an already-open workbook make the `cell` keyword `-10003`;
   `range` (including a single cell) is stable.
3. **`open` is asynchronous.** On cold start `open POSIX file` returns before the
   workbook is registered in the `workbooks` collection; any immediate access
   then `-50`s. `open_workbook` **polls `exists workbook` until ready**. Also:
   `open workbook workbook file name "..."` throws an *uncatchable* -50 on cold
   start — always use the generic `open POSIX file` verb.
4. **Collection access prefers "bulk property" over "object-ref iteration".**
   `name of every worksheet of workbook X` (one-shot property) is stable;
   `repeat with ws in (every worksheet of X)` iterating worksheet **objects**
   `-50`s after a cold open. Same root cause as reading data via
   `value of used range` rather than per-cell refs.
5. **Read a block via `value of used range`** and rebuild TSV in Rust — osascript
   flattens the 2D list and loses row structure, so the AppleScript keeps the
   inner `repeat` and joins with an explicit separator.
6. **`tab` keyword resolves to the literal string "tab"** in these
   `format!` template contexts — use `(ASCII character 9)` for a real tab.

## Invariants (do not break)

1. **No session / no shutdown hook.** Unlike `mcp_browser`, there is no long-lived
   handle to manage: the Excel *app* owns the open-workbook state and shares it
   across independent `osascript` subprocesses. `main` holds no state; each tool
   call is one or more one-shot `osascript -e` invocations. Idempotent
   `open_workbook` (reuse-if-open) is what stitches calls together.
2. **Timeout coordination — server cap < host timeout.** The host MCP client has
   one `request_timeout_ms` per server; on timeout it **kills the subprocess**.
   Every osascript op is wrapped in `with_timeout(op_timeout_ms())` (default
   **90s**); configure the host larger (**120s** recommended).
3. **Error messages must avoid the host's transport trigger words** (`mcp
   response timeout`, `broken pipe`, `process exited`, ...). The shared timeout
   error (in `mcp_stdio::with_timeout`) says "operation reached the N ms server
   cap" on purpose. Raw Excel error codes (`-50`/`-10003`) are passed through
   verbatim — they are diagnostic, not transport triggers.
4. **Only `content[0].text` reaches the model**, capped to **24K chars**
   (`cap_text`). Large ranges must be exported to disk, not returned inline.
5. **Persist via `export_csv`, not `save_workbook`.** Sandboxed Excel (present
   here: `~/Library/Containers/com.microsoft.Excel` exists) makes AppleScript
   `save workbook as` return `-50` in a non-interactive context — a **systemic
   sandbox limitation**, confirmed by exhaustive local trials + external sources,
   NOT a syntax bug. Do not re-attempt fixing it. `export_csv` reads the used
   range via Excel then **writes the file from the Rust side**, bypassing the
   sandbox. `save_workbook` is kept EXPERIMENTAL and returns an honest sandbox
   explanation on failure.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MCP_EXCEL_OP_TIMEOUT_MS` | `90000` | Per-op server-side cap (keep < host timeout) |

## Tool set (9 structured verbs — no raw AppleScript exposed to the model)

`open_workbook` · `list_sheets` · `read_cell` · `read_range` · `write_cell` ·
`write_range` · `export_csv` · `save_workbook` (experimental) · `close_workbook`.

Reads/writes operate in Excel's in-memory workbook; use `export_csv` to persist.

## Host registration

Add to `~/.config/mcp.json` under `mcpServers` (tools appear as
`mcp_excel_open_workbook`, etc.). **Keep `request_timeout_ms` (120000) > server
cap (90000).** macOS-only (depends on the Excel app + `osascript`).
