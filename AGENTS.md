# AGENTS.md - rust_tools Project Guide

Root-level overview and repo-wide invariants only; subsystem details live in
scoped `AGENTS.md` files. Only `AGENTS.md` / `Agent.md` / `CLAUDE.md` are
auto-discovered.

Rust 2024 workspace: utility library + CLI binaries. Primary product is `a`, an
LLM-based AI agent runtime (AIOS) with process scheduling, agent/skill routing,
tool registry, and MCP integration. Workspace members: root crate,
`crates/rust_tools_macros`, `crates/aios_kernel`, `crates/mcp_stdio`,
`crates/mcp_browser`, `crates/mcp_excel`. macOS-first (`objc2` deps); core
library cross-platform.

## Layout

```text
src/lib.rs                  # utility library
src/bin/a.rs                # AI agent entry point
src/bin/ai/                 # AI runtime core
src/bin/ff/                 # file-finder module embedded by `a`
src/bin/*.rs                # one-off/experimental CLI tools (c, j, secret, pdf, ...) - not part of `a`
crates/aios_kernel/         # scheduler / IPC / process state machine
crates/rust_tools_macros/   # proc macros
crates/mcp_stdio/           # shared MCP-over-stdio skeleton (lib): JSON-RPC transport + run<McpServer> loop
crates/mcp_browser/         # standalone MCP server: browser automation; macOS default AppleScript driver reuses the user's running Chrome (new tab, keep cookies, never quits it, never steals focus); Windows/Linux use CDP attach (MCP_BROWSER_WS_URL=http://127.0.0.1:9222 against a --remote-debugging-port Chrome); plain CDP = controlled instance
crates/mcp_excel/           # standalone MCP server: real Microsoft Excel automation via AppleScript (osascript)
tests/                      # integration tests
models.json                 # model registry
```

> `mcp_browser` / `mcp_excel` are standalone binary crates (not deps of `a`),
> so `cargo check --bin a` stays fast. Both reuse the `mcp_stdio` lib crate for
> transport + `run<S: McpServer>` dispatch; new "drive an OS-native app" MCP
> servers should follow the same pattern.

## Build / Test

`cargo check` / `cargo test` are expensive here (heavy deps, slow incremental).
Don't run speculatively, but **do** run them when the ladder below demands it -
reading code cannot confirm compilation. Always scope to the narrowest target.

```bash
cargo check --bin a                  # fast type-check for the main binary
cargo check -p aios_kernel           # type-check one workspace crate
cargo test --bin a test_name         # run one targeted test in `a`
cargo test -p aios_kernel test_name  # run one targeted test in a crate
cargo test --lib --bin a test_name   # only when one named test spans lib + bin
```

Never run bare `cargo test` / `cargo build --release` / workspace-wide commands
for routine verification, and never repeat a `cargo test` without a code change
in between. Prefer an existing focused test before running one.

**Verification ladder:**

1. **No code change / docs-only**: no Cargo command required.
2. **Type-level / compile-risk / mechanical refactor**: run the narrowest `cargo check`.
3. **Runtime behavior changed, focused test exists**: run that named test.
4. **Runtime behavior changed, no focused test**: run the narrowest `cargo check`; say no targeted test was found.
5. **Bug fix with regression/new test**: run that named test; on failure, fix and re-run.

## Global Engineering Rules

1. **Module boundaries**: use `pub(super)` / `pub(crate)`.
2. **Chinese comments** to match surrounding style.
3. **Collections**: prefer `rustc-hash` FxHashMap/FxHashSet via existing re-exports.
4. **Config keys**: add only in `src/bin/ai/config_schema.rs`.
5. **AI tools**: schema/registration in `tools/registry/`, logic in `tools/service/`.
6. **Focused changes**: do not modify unrelated code; avoid opportunistic refactors or formatting churn - if truly necessary, explain first and get confirmation.
7. **Tests**: keep close to the changed module; serial tests use `test_support::ENV_LOCK`.
8. **Extensibility**: prefer data-driven/registration-based design over hardcoded `if`/`else` chains. Additive, optional registration over modifying shared structs.
9. **AGENTS.md maintenance**: after every code change, check whether the nearest scoped `AGENTS.md` (or this root file) needs updating. **Delete or revise** stale content - do not merely append. Outdated rules that contradict current behavior are worse than missing rules.
10. **Git safety**: never `git stash` / `git stash drop` someone else's uncommitted changes. Use a temporary branch, worktree, or stash only your own (and pop back).
11. **Architecture-first**: if a path needs many layers of fallback, the data flow is wrong - refactor so the happy path is clean instead of piling on defensive `if`/`else`.

## High-Value Pitfalls

1. `.agent` files are compiled into `a` via `include_str!` (editing recompiles); builtin `.skill` files under `src/bin/ai/builtin_skills/` are compiled in via `include_str!` too; user `.skill` files load at runtime from the skills dir (no recompile).
2. `src/bin/ff/` is embedded into `a` via `include!`; changes there affect the agent binary.
3. `runtime_ctx::effective_cwd()` is the working-directory authority for tools and sub-agents.
4. `objc2*` dependencies are macOS-only.
