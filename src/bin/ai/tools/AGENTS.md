# Tools Guide

## Scope

Applies to `src/bin/ai/tools/**`. Layers: `registry/` (schema/metadata), `service/` (execution), `storage/` (helpers/state).

## Key invariants

1. **Layer separation.** Keep schema in `registry/`, logic in `service/`, helpers in `storage/`.
2. **Names & paths.** Verb-first `snake_case`; relative paths via `runtime_ctx::effective_cwd()`.
3. **Progressive loading.** Default `core` (`DEFAULT_TURN_TOOL_GROUPS` in `driver/skill_runtime.rs`); lazy tools load via `enable_tools` — by tool name or group shortcut (`knowledge`, `skills`, `task`, `agent_team`, ...; `builtin` is only the enable-ability flag, never expands).
    - **Hidden gating (`executor`).** Its members (process / IPC / shm / env primitives in `os_tools.rs`) carry `hidden: true`: for agents not declaring a hidden-gating group they are deferred out of resident schemas, absent from `enable_tools list`, and rejected by name — see `agent_may_enable_tool` / `group_visible_to_agent` in `enable_tools.rs`. A manifest declaring `executor` unlocks them, after which `executor` plus `builtin` are required. No builtin agent declares it today (`build`, `audit`, `audit-fast`, `sharp`): background work goes through the `task` family, and only opt-in skills can unlock the group.
    - Explicit `tools:` lists pin eager visibility.
4. **Registry metadata is authoritative.** Display/history/replay policies are
   registered, never inferred from names. Auto-discovered
   `tool_descriptions/<tool>.json` owns description, compact schema, group, and
   optional post-first-call guidance; do not duplicate schema or driver policy.
   Groups resolve to the closed `ToolGroup` enum and every registered tool needs
   non-empty groups; `hidden: true` defers eager loading. `ToolSpec` has no groups.
5. **History policy.** Lossy compression and pruning are independent. Preserve
   truth-bearing reads/commands/tasks via overflow stubs; only superseded plans
   may compress. Task evidence prunes only after the durable ledger proves every
   named task integrated, failing closed when authorization is unavailable.
6. **Writable roots.** Temp writes use `runtime_ctx::temp_dir()` plus the session registry; project deletes use `apply_patch`. Skills and rust_tools config dirs are always writable; `configured_write_roots` in `file_store.rs` is authoritative.
7. **Process groups.** Track each command pgid for teardown; never persist it. Kill commands target registered groups only, and `rm` targets session temp only; external processes remain blocked and project deletes use `apply_patch`.
8. **Truncation.** Truncating tools must distinguish complete/failed/incomplete, include shown-vs-total counts and a narrow/page hint.
9. **Patch/read/search.** Anchor patches on removed text, normalize confusables, and reject ambiguity; explicit envelopes own deletes and pure inserts use line numbers. `Replace in line` accepts exactly three single-line fields and requires a unique exact/trimmed match. Failures identify the hunk and echo reusable patch text. `patch` and `patch_file` are exclusive (8K/64K); patch files stay in session temp or `effective_cwd`.
10. **Subagent tools are top-level only.** Hide `task` at depth and never re-enable it. Scope by session+owner, persist before IPC cleanup, and require integration. `task_evidence_read` defaults to progress; exact delivered reads neither collect nor mutate state. The bounded ledger points to `task_audit`; plan ID mentions affect detail priority only, and lone-spawn hints are one-shot display state.
11. **Wall-clock net.** Both `task_wait` and driver loop reap past `SUBAGENT_WALL_CLOCK_TIMEOUT` (60m); write terminal result, preserve finished results, signal `cancel_stream` before aborting.
12. **Interactive handoff.** `request_user_input` is a driver-owned control tool injected by name for every interactive turn (skill or not; goal mode, background, and subagent turns exclude it — subagents have no attached user). Scope by `TURN_IDENTITY`; never infer continuation from text.
13. **Commands are non-interactive by default.** Non-PTY disables pagers/prompts/color; PTY preserves interactive behavior. Mutating commands auto-load scoped instructions. `git commit`/`git push`/`git stash`/`git checkout`/`git switch` family gates in `service/command.rs`: prompt in TTY (serve child: via `serve_confirm`), fail-closed otherwise.
14. **Mutation log.** Writes/patches append capped, serialized best-effort JSONL, excluding session artifacts; `/audit` reads it via `DRIVER_CTX` and falls back to `git diff` only when empty.
15. **Team/Graph reuses task truth.** `manage_team`/`run_agent_graph` checkpoint orchestration only; all work flows through kernel task registry with integration/cancellation/depth guard.
16. **Tool permissions are middleware.** `PermissionMiddleware` enforces
    `ai.tools.permissions`; empty rules allow all. Denied calls become aligned
    per-slot error results. Exact names beat prefixes; config order breaks prefix
    ties. `ask` fails closed without a TTY, including `-bg`; re-execed background
    processes install policy, while in-process subagents skip it via `App::clone`.
17. **Plan state is durable.** `plan` atomically creates/replaces active state so it survives compression. `plan_update` failures stay non-mutating and report the full plan and path; inspect stored state with `read_file`, never `plan`.
