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

- **Workspace members**: root crate, `crates/rust_tools_macros`, `crates/aios_kernel`
- **Platform**: macOS-first (`objc2` deps); core library cross-platform

## Top-Level Layout

```text
src/lib.rs                  # utility library
src/bin/a.rs                # AI agent entry point
src/bin/ai/                 # AI runtime core
src/bin/ff/                 # file-finder module embedded by `a`
crates/aios_kernel/         # scheduler / IPC / process state machine
crates/rust_tools_macros/   # proc macros
tests/                      # integration tests
models.json                 # model registry
docs/agent-guides/          # long-form on-demand subsystem docs
```

## Build / Test

```bash
cargo check --bin a          # fast type-check
cargo test --lib --bin a     # run lib + a's tests
cargo test --bin a test_name # run one test
```

**Only verify what you changed.** Always scope with `--bin`, `--lib`, `-p`, or a
specific test name. Never bare `cargo test` / `cargo build --release`.

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

## High-Value Pitfalls

1. `.agent` / `.skill` files are compiled with `include_str!`; editing them recompiles `a`.
2. `src/bin/ff/` is embedded into `a` via `include!`; changes there affect the agent binary.
3. `runtime_ctx::effective_cwd()` is the working-directory authority for tools and sub-agents.
4. `objc2*` dependencies are macOS-only.

> Subsystem-specific pitfalls live in their scoped `AGENTS.md`.
