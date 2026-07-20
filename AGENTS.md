# AGENTS.md — rust_tools Project Guide

## Scope

Root-level overview and repo-wide invariants only. Subsystem details belong in
scoped `AGENTS.md` files (under important subdirectories) or
`docs/agent-guides/*.md` (long-form, loaded on demand). Only `AGENTS.md` /
`Agent.md` / `CLAUDE.md` are auto-discovered; `docs/agent-guides/` are **not**.

## Overview

Rust 2024 workspace: utility library + CLI binaries. Primary product is `a`,
an LLM-based AI agent runtime (AIOS) with process scheduling, agent/skill
routing, tool registry, and MCP integration.

- **Workspace members**: root crate, `crates/rust_tools_macros`, `crates/aios_kernel`, `crates/mcp_browser`
- **Platform**: macOS-first (`objc2` deps); core library cross-platform

## Top-Level Layout

```text
src/lib.rs                  # utility library
src/bin/a.rs                # AI agent entry point
src/bin/ai/                 # AI runtime core
src/bin/ff/                 # file-finder module embedded by `a`
crates/aios_kernel/         # scheduler / IPC / process state machine
crates/rust_tools_macros/   # proc macros
crates/mcp_browser/         # standalone MCP server: browser automation via chromiumoxide (CDP)
tests/                      # integration tests
models.json                 # model registry
docs/agent-guides/          # long-form on-demand subsystem docs
```

> `crates/mcp_browser` pulls in the heavy `chromiumoxide` dep, but it is a
> **standalone binary crate** — it only compiles under `cargo build -p mcp_browser`,
> so `cargo check --bin a` (i.e. `-p rust_tools`) stays fast and unpolluted.

## Build / Test

```bash
cargo check --bin a                  # fast type-check for the main binary
cargo check -p aios_kernel           # type-check one workspace crate
cargo test --bin a test_name         # run one targeted test in `a`
cargo test -p aios_kernel test_name  # run one targeted test in a crate
cargo test --lib --bin a test_name   # only when one named test spans lib + bin
```

**Scope every verification.** Always use `--bin`, `--lib`, `-p`, or a specific
test name. Never run bare `cargo test`, bare `cargo build --release`, or broad
workspace-wide commands for routine verification.

**Verification ladder:**

1. **No code change / docs-only / comments-only**: no Cargo command required.
   Say that verification was not run because no executable code changed.
2. **Type-level, compile-risk, or mechanical refactor**: run the narrowest
   relevant `cargo check` command.
3. **Runtime behavior changed**: run the narrowest relevant existing test
   (`cargo test ... test_name`) if one clearly covers the changed path.
4. **Runtime behavior changed but no focused test exists**: run the narrowest
   relevant `cargo check`, then explicitly say no targeted test was found. Do
   not run broad tests just to satisfy this rule.
5. **Bug fix with a known regression test or newly added test**: run that named
   test. If it fails, fix the code and re-run the same test after the code
   change.

**Do not run tests speculatively.** Prefer reading the affected code and locating
an existing focused test before choosing a Cargo command.

**Avoid repeated test loops.** Never run the same `cargo test` command repeatedly
without a code change in between. After one successful focused test, stop unless
the user asks for broader verification.

## Global Engineering Rules

1. **Module boundaries**: use `pub(super)` / `pub(crate)`.
2. **Chinese comments** to match surrounding style.
3. **Collections**: prefer `rustc-hash` FxHashMap/FxHashSet via existing re-exports.
4. **Config keys**: add only in `src/bin/ai/config_schema.rs`.
5. **AI tools**: schema/registration in `tools/registry/`, logic in `tools/service/`.
6. **Focused changes**: do not modify unrelated code. Avoid opportunistic refactors
   or formatting churn — if truly necessary, explain first and get confirmation.
7. **Tests**: keep close to the changed module; serial tests use `test_support::ENV_LOCK`.
8. **Extensibility**: prefer data-driven/registration-based design over hardcoded
   `if`/`else` chains. Additive, optional registration over modifying shared structs.
9. **AGENTS.md maintenance**: after every code change, check whether the nearest
   scoped `AGENTS.md` (or this root file for repo-wide invariants) needs updating.
   **Delete or revise** stale content — do not merely append. Outdated rules that
   contradict current behavior are worse than missing rules.
10. **Git safety**: never `git stash` / `git stash drop` someone else's uncommitted
    changes. Use a temporary branch, worktree, or stash only your own (and pop back).
11. **Architecture-first, no excessive fallback logic**: Do not pile on defensive
    `if`/`else` fallbacks to work around a design problem. If a code path needs many
    layers of fallback to function correctly, the abstraction or data flow is wrong —
    refactor the design first so the happy path is clean and straightforward.

## High-Value Pitfalls

1. `.agent` / `.skill` files are compiled with `include_str!`; editing them recompiles `a`.
2. `src/bin/ff/` is embedded into `a` via `include!`; changes there affect the agent binary.
3. `runtime_ctx::effective_cwd()` is the working-directory authority for tools and sub-agents.
4. `objc2*` dependencies are macOS-only.

> Subsystem-specific pitfalls live in their scoped `AGENTS.md`.
