# Tools Guide

## Scope

Applies to `src/bin/ai/tools/**`. Layers: `registry/` (schema/metadata), `service/` (execution), `storage/` (helpers/state).

## Key invariants

1. **Layer separation.** Keep schema in `registry/`, logic in `service/`, helpers in `storage/`.
2. **Names & paths.** Use `snake_case` and follow the existing tool family's
   naming convention, including `knowledge_*`; relative paths resolve through
   `runtime_ctx::effective_cwd()`.
3. **Progressive loading.** Default `core` (`DEFAULT_TURN_TOOL_GROUPS` in `driver/skill_runtime.rs`); lazy tools load via `enable_tools` — by tool name or group shortcut (`knowledge`, `skills`, `task`, `agent_team`, ...; `builtin` is only the enable-ability flag, never expands).
    - **Hidden gating (`executor`).** Its members (process / IPC / shm / env primitives in `os_tools.rs`) carry `hidden: true`: for agents not declaring a hidden-gating group they are deferred out of resident schemas, absent from `enable_tools list`, and rejected by name — see `agent_may_enable_tool` / `group_visible_to_agent` in `enable_tools.rs`. A manifest declaring `executor` unlocks them, after which `executor` plus `builtin` are required. No builtin agent declares it today: background work goes through the `task` family, and only an opt-in skill or an active agent manifest declaring the group can unlock it (`declares_hidden_group` in `driver/skill_runtime.rs`).
    - Explicit `tools:` lists pin eager visibility.
4. **Registry metadata is authoritative.** Display/history/replay policies are
   registered, never inferred from names. Auto-discovered
   `tool_descriptions/<tool>.json` owns description, compact schema, group, and
   optional post-first-call guidance; do not duplicate schema or driver policy.
   Groups resolve to the closed `ToolGroup` enum and every registered tool needs
   a present `groups` field (may be `[]` for driver-injected control tools);
   `hidden: true` defers eager loading. `ToolSpec` has no groups.
5. **History policy.** Compression and pruning are independent registered
   dimensions (`ToolLossyCompressPolicy` Allow-by-default / Never,
   `ToolPrunePolicy` Allow / Never / AfterIntegration in `registry/common.rs`).
   `plan` allows lossy compression — the recent-window protects the latest
   plan, so effectively only superseded plans are summarized — but forbids
   model-directed pruning (`plan_tools.rs`). Task pruning fails closed when
   durable-ledger authorization is unavailable. Retention and integration follow
   **History is truth** in `src/bin/ai/AGENTS.md`.
6. **Writable roots.** Temp writes use `runtime_ctx::temp_dir()`, and only `temp=true` writes register the session temp directory; ordinary `write_file` without `temp=true` additionally allows system/shared temp paths (`/tmp`, `/var/tmp` — `validate_write_file_access` in `storage/file_store.rs`). Project deletes use `apply_patch`. Skills and rust_tools config dirs are always writable; `configured_write_roots` in `storage/file_store.rs` is authoritative. Session `--add-dir` roots come from the current App snapshot via `runtime_ctx::additional_write_roots()`, never process-global config. Writable roots and the `AGENT_COMMAND_ENV` origin marker in `src/bin/ai/write_roots.rs` are accidental-write guardrails, not a security boundary. Models with command execution can bypass the marker and write outside the declared roots; user-only grant expansion is not an implemented security guarantee.
7. **Process groups.** Track each command pgid for teardown; never persist it. Kill commands target registered groups only (a shell redirection is not a kill target), and `rm` targets session temp only; external processes remain blocked and project deletes use `apply_patch`. A detaching launcher (`open -n`) is rejected: its app is re-parented to launchd and can never be registered, so launch the app binary with `&` instead.
8. **Truncation.** Truncating tools must distinguish complete/failed/incomplete. The shared helper truncates by char count with a continuation marker (`truncate_tool_output` in `storage/text_utils.rs`); over-long command output keeps head/tail with shown-vs-total counts and a narrow/page hint (`truncate_chars` in `service/command.rs`).
9. **Patch/read/search.** Anchor patches on removed text, normalize confusables, and reject ambiguity; explicit envelopes own deletes and pure inserts use line numbers. `Replace in line` accepts exactly three single-line fields and requires a unique exact/trimmed match. Failures identify the hunk and echo reusable patch text. `patch` and `patch_file` are exclusive (8K/64K); patch files stay in session temp or `effective_cwd`.
10. **Subagent tools are top-level only.** Hide the task family at depth and never re-enable
    it. Scope by session+owner; persistence/integration follows the AI guide's
    **History is truth** contract. `task_evidence_read` defaults to progress; exact
    delivered reads neither collect nor mutate state. The bounded ledger points
    to `task_audit`; plan ID mentions affect detail priority only, and lone-spawn
    hints are one-shot display state.
11. **Wall-clock net.** Both `task_wait` and the driver loop reap past
    `SUBAGENT_WALL_CLOCK_TIMEOUT`; both paths signal `cancel_stream` and request
    the Tokio abort first, then publish the terminal timeout status
    (`reap_timed_out_subagents` in `task_tools/dispatch.rs`; timeout path in
    `task_tools/wait.rs`). Result retention follows the AI guide's
    **History is truth** contract.
12. **Interactive handoff.** `request_user_input` is a driver-owned control tool injected by name for every interactive turn (skill or not; goal mode, background, and subagent turns exclude it — subagents have no attached user). Scope by `TURN_IDENTITY`; never infer continuation from text.
13. **Commands are non-interactive by default.** Non-PTY disables pagers/prompts/color; PTY preserves interactive behavior. Mutating commands auto-load scoped instructions. `git commit`/`git push`/`git stash`/`git checkout`/`git switch` family gates in `service/command.rs`: prompt in TTY (serve child: via `serve_confirm`), fail-closed otherwise. The segment audit (`service/audit/`) also covers second-interpretation wrappers: an `osascript` script must be statically visible via `-e`, and System Events GUI input, `do shell script`, and the one-line `quit` forms (`tell application "X" to quit`, `quit app "X"`) stay blocked; a block-form `quit` inside `tell ... end tell` is normalized past the `toquit`/`quitapp` scan (`service/audit/osascript.rs`).
14. **Mutation log and audit inventory.** Writes/patches append capped, serialized best-effort JSONL, excluding session artifacts. `/audit` and `/audit --deep` both work from frozen previous-turn evidence captured before the audit turn (`capture_previous_turn_scope` in `driver/commands/audit.rs`): the previous turn's canonical tool records bound the scope, since session mutation logs and Git do not carry reliable per-turn ownership. `show_changes` exposes the mutation-log session-first view plus seq/time/turn/task/path windows and `mode=history` per-mutation timelines (seq, timestamp, file version, diff), while the Git fallback stays outside time/seq windows.
15. **Team/Graph reuses task truth.** `manage_team`/`run_agent_graph` checkpoint orchestration only; all work flows through kernel task registry with integration/cancellation/depth guard.
16. **Tool permissions are middleware.** `PermissionMiddleware` enforces
    `ai.tools.permissions`; empty rules allow all. Denied calls become aligned
    per-slot error results. Exact names beat prefixes; config order breaks prefix
    ties. For `ask`, missing input rejects the call only when neither a local TTY
    nor a serve confirmation channel is available (`PermissionExecutor::decide`
    in `permissions.rs`); declined or interrupted answers still deny. Re-executed
    background processes install policy in the driver setup (`driver/mod.rs`),
    while in-process subagents skip it: `App::fork_for_subagent`
    (`src/bin/ai/types.rs`) clears the middleware vectors.
17. **Plan state is durable.** `plan` atomically creates/replaces active state so it survives compression. `plan_update` failures stay non-mutating; only the `apply_update` failure path (e.g., unknown step) reports the full plan and its persisted path (`plan_state/store.rs`), while parameter errors return plain errors. Inspect stored state with `read_file`, never `plan`.
