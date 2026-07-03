# AGENTS.md — rust_tools Project Guide

## Scope

This root file is the repository-wide overview for the `rust_tools` workspace.
Keep it short. Do not put long subsystem playbooks here.

Detailed subsystem guidance is layered:

- root `AGENTS.md`: repo-wide overview and invariants only
- scoped `AGENTS.md` files under important subdirectories: short local rules
- `docs/agent-guides/*.md`: long-form references that should be loaded on demand

Only `AGENTS.md` / `Agent.md` / `CLAUDE.md` case variants are auto-discovered as
project instruction docs. Files under `docs/agent-guides/` are **not**
auto-injected into every prompt.

## Overview

This Rust workspace provides a utility library plus several CLI binaries. The
primary product is `a`, an LLM-based AI agent runtime (AIOS) with built-in
process scheduling, agent/skill routing, a tool registry, and MCP integration.

- **Edition**: Rust 2024
- **Workspace members**: root crate, `crates/rust_tools_macros`, `crates/aios_kernel`
- **Platform**: macOS-first (`objc2` deps); core library remains cross-platform

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
make all
make install
cargo build --release --bin a
cargo check --bin a
cargo test --lib --bin a
cargo test --bin a test_name
```

**Only verify what you changed — avoid full checks.**
Run focused checks (lint, typecheck, test) for the touched module first, then broaden only when the change crosses shared behavior.
Do not run the full workspace `cargo check`, `cargo test`, or other verification unless necessary.

## Global Engineering Rules

1. Use `pub(super)` / `pub(crate)` to enforce module boundaries.
2. Keep comments consistent with the surrounding Chinese comment style.
3. Prefer `rustc-hash` FxHashMap/FxHashSet via existing re-exports.
4. Add AI config keys only in `src/bin/ai/config_schema.rs`; never scatter raw keys.
5. For AI tools, keep schema/registration in `src/bin/ai/tools/registry/` and logic in `src/bin/ai/tools/service/`.
6. Prefer reuse over reinvention; avoid unrelated refactors and formatting churn.
7. Avoid brittle hardcoded string rules when a typed, structural, or data-driven path exists.
8. Keep tests close to the changed module where practical; serial tests should use `test_support::ENV_LOCK`.
9. Do not modify code unrelated to the current task. Only change files and logic directly tied to the requirement; avoid opportunistic refactors, formatting, or "improvements" to unrelated code. If touching unrelated code is truly necessary, explain the reason first and get confirmation.

## High-Value Pitfalls

1. `.agent` / `.skill` files are compiled with `include_str!`; editing them recompiles `a`.
2. `src/bin/ff/` is embedded into `a` via `include!`; changes there affect the agent binary.
3. MCP servers are stdio subprocesses, not HTTP services.
4. `runtime_ctx::effective_cwd()` is the working-directory authority for tools and sub-agents.
5. `objc2*` dependencies are macOS-only.

## Instruction Layering

When work moves into a scoped subsystem, the closer `AGENTS.md` should carry the
local rules and point to long-form references only when needed.

Current scoped entry points:

- `src/bin/ai/AGENTS.md`
- `src/bin/ai/driver/AGENTS.md`
- `src/bin/ai/tools/AGENTS.md`
- `src/bin/ai/mcp/AGENTS.md`
- `src/bin/ai/provider/AGENTS.md`
- `src/bin/ai/knowledge/AGENTS.md`

Detailed references live under:

- `docs/agent-guides/ai-driver.md`
- `docs/agent-guides/ai-tools.md`
- `docs/agent-guides/ai-mcp.md`
- `docs/agent-guides/ai-provider.md`

## Maintaining Instruction Docs

When code changes would make any statement in these files stale (layout,
invariants, build/test commands, subsystem rules, on-demand guide references):

1. Update the nearest scoped `AGENTS.md` for the touched subsystem.
2. Keep this root file limited to repo-wide overview, layout, and invariants.
3. Put long subsystem explanations in `docs/agent-guides/*.md`, then reference them from the nearest scoped `AGENTS.md`.
4. Keep the top-level layout and build/test commands accurate.

> Principle: root instructions should stay concise, scoped instructions should stay local, and long references should be loaded only when the task actually needs them.
