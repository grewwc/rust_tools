# AGENTS.md - rust_tools Project Guide

Root-level overview and repo-wide invariants only; the nearest scoped
`AGENTS.md` supplies subsystem rules.

Rust 2024 workspace: utility library + CLI binaries. Primary product is `a`, an
LLM-based AI agent runtime (AIOS) with process scheduling, agent/skill routing,
tool registry, and MCP integration. Workspace members: root crate,
`crates/rust_tools_macros`, `crates/aios_kernel`, `crates/mcp_stdio`,
`crates/mcp_browser`, `crates/mcp_computer`, `crates/mcp_excel`,
`crates/mcp_pdf`, `crates/re`.
macOS-first (`objc2` deps); core library cross-platform.

## Layout

- `src/bin/a.rs` + `src/bin/ai/`: agent entry point and runtime; `src/bin/ff/`
  is embedded into `a` via `include!`.
- `src/bin/*.rs`: standalone utilities, not modules of `a`.
- `crates/aios_kernel/`: scheduler, IPC, and process state machine.
- `crates/mcp_stdio/` + `crates/mcp_{browser,computer,excel,pdf}/`: shared
  stdio transport and standalone MCP servers.
- `crates/rust_tools_macros/`, `crates/re/`, `scripts/`, `tests/`, `models/`:
  macros, memo CLI, helpers, integration tests, and runtime model registry.

The `mcp_*` crates are not dependencies of `a`; new OS-app servers should reuse
`mcp_stdio` when possible. `src/bin/mcp_ocr.rs` is a hand-rolled exception.

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

To deploy changes to the server-capable `a`, build and install all release
features, then restart the managed server:

```bash
make install-all
a --serve-restart
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
2. **English comments**: all line, block, and doc comments must be English; when
   modifying code, translate any Chinese comments you touch.
3. **Collections**: `rustc-hash` FastMap/FastSet via re-exports.
4. **Config keys**: add only in `src/bin/ai/config_schema.rs`.
5. **Tests**: keep close to changed module; serial tests use `test_support::ENV_LOCK`.
6. **Extensibility**: data-driven/registration-based over hardcoded `if`/`else`.
7. **AGENTS.md maintenance**: update only superseded nearby rules, in place; never
   append history. Keep injected instructions concise and delete stale rules.
8. **Git safety**: never `stash`/`stash drop` others' uncommitted changes. Use temp branch/worktree or stash only your own.
9. **Architecture-first**: many fallbacks = wrong data flow — refactor the happy path instead.

## High-Value Pitfalls

1. `.agent` / builtin `.skill` files are compiled via `include_str!`; user `.skill`
   files load at runtime.
2. `objc2*` dependencies are macOS-only.
