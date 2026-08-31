# AGENTS.md - rust_tools Project Guide

Root-level overview and repo-wide invariants only; subsystem details live in
scoped `AGENTS.md` files. Auto-discovered instruction files are the `AGENTS.md` /
`Agent.md` / `CLAUDE.md` families plus lowercase variants (`agent.md`, `Claude.md`,
`claude.md`) — see `PROJECT_INSTRUCTION_FILENAMES` in `src/bin/ai/agents.rs`.

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
src/bin/ff/                 # file-finder module embedded by `a` via include! in ai/mod.rs
src/bin/*.rs                # one-off CLI tools (c, j, secret, pdf, ...) - not part of `a`
crates/aios_kernel/         # scheduler / IPC / process state machine
crates/rust_tools_macros/   # proc macros
crates/mcp_stdio/           # shared MCP-over-stdio skeleton: JSON-RPC transport + run<McpServer>
crates/mcp_browser/         # standalone MCP server: browser automation (macOS AppleScript / CDP)
crates/mcp_excel/           # standalone MCP server: Excel via AppleScript
tests/                      # integration tests
models/                     # per-model JSON registry (read at runtime)
```

> `mcp_browser` / `mcp_excel` are standalone binaries (not deps of `a`), so
> `cargo check --bin a` stays fast. Both reuse `mcp_stdio`; new OS-app MCP
> servers should follow the same pattern. Standalone exceptions: `src/bin/mcp_feishu.rs`
> and `src/bin/mcp_ocr.rs` are hand-rolled JSON-RPC servers that don't reuse `mcp_stdio`.

## Build / Test

`cargo check` / `cargo test` are expensive (heavy deps, slow incremental).
Don't run speculatively, but do run when the ladder demands it — reading code
cannot confirm compilation. Always scope to the narrowest target.

```bash
cargo check --bin a                  # fast type-check for the main binary
cargo check -p aios_kernel           # type-check one crate
cargo test --bin a test_name         # run one targeted test in `a`
cargo test -p aios_kernel test_name  # run one targeted test in a crate
```

Never run bare `cargo test` / `cargo build --release` / workspace-wide commands
for routine verification, and never repeat `cargo test` without a code change.
Prefer an existing focused test before running one.

**Verification ladder:**
1. **No code change / docs-only**: no Cargo command required.
2. **Type-level / compile-risk / mechanical refactor**: narrowest `cargo check`.
3. **Runtime behavior changed, focused test exists**: run that named test.
4. **Runtime behavior changed, no focused test**: narrowest `cargo check`; note no test found.
5. **Bug fix with regression/new test**: run that named test; on failure fix and re-run.

## Global Engineering Rules

1. **Module boundaries**: `pub(super)` / `pub(crate)`.
2. **English comments**: all code comments — line (`//`), block (`/* */`), and doc
   comments (`///`, `//!`) — must be written in English. This applies to new code and
   to any code you modify; when you touch code containing Chinese comments, translate
   them as part of the change (repo-wide Chinese-comment migration).
3. **Collections**: `rustc-hash` FxHashMap/FxHashSet via re-exports.
4. **Config keys**: add only in `src/bin/ai/config_schema.rs`.
5. **AI tools**: schema in `tools/registry/`, logic in `tools/service/`.
6. **Tests**: keep close to changed module; serial tests use `test_support::ENV_LOCK`.
7. **Extensibility**: data-driven/registration-based over hardcoded `if`/`else`.
8. **AGENTS.md maintenance**: after code changes, revise/delete stale rules nearby — don't just append. Contradictory stale rules are worse than missing ones.
9. **Git safety**: never `stash`/`stash drop` others' uncommitted changes. Use temp branch/worktree or stash only your own.
10. **Architecture-first**: many fallbacks = wrong data flow — refactor the happy path instead.

## High-Value Pitfalls

1. `.agent` / builtin `.skill` files are compiled in via `include_str!` (editing recompiles); user `.skill` files load at runtime.
2. `src/bin/ff/` is embedded into `a` via `include!` in `src/bin/ai/mod.rs` (`a.rs`
   itself is just `mod ai;`); changes affect the agent binary.
3. `runtime_ctx::effective_cwd()` is the working-directory authority for tools and sub-agents.
4. `objc2*` deps are macOS-only.
