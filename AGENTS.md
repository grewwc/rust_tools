# AGENTS.md - rust_tools Project Guide

Root-level overview, repo-wide invariants, and verification policy.

Rust 2024 workspace: utility library + CLI binaries. Primary product is `a`, an
LLM-based AI agent runtime (AIOS) with process scheduling, agent/skill routing,
tool registry, and MCP integration. macOS-first (`objc2*` dependencies are
macOS-only); core library cross-platform.

## Instruction hierarchy

- Read this guide and the scoped guides on the path to the target directory,
  root-to-deep. Child guides own local mechanisms and explicitly scoped gates;
  sibling guides apply only through an explicit reference.
- Scoped rules cannot weaken the Build / Test prohibitions, English code-comment
  rule, protection of others' Git changes, AGENTS.md maintenance rules,
  or the runtime-session inspection boundary linked below. Other refinements
  must stay within their scope and any exception stated by the owning rule.
- The root verification ladder selects Cargo checks. Local gates add targeted
  integration evidence only for the behavior changes they name; they do not
  replace required checks or apply to docs-only changes. Command examples are
  scoped targets, not unconditional gates. Release builds and installation are
  deployment steps.

## Layout

Workspace members are the root crate plus the `crates/*` below.

- `src/bin/a.rs` + `src/bin/ai/`: agent entry point and runtime; `src/bin/ff/`
  is embedded into `a` via `include!`.
- `src/bin/*.rs`: standalone utilities, not modules of `a`.
- `crates/aios_kernel/`: scheduler, IPC, and process state machine.
- `crates/mcp_stdio/` + `crates/mcp_{browser,computer,excel,pdf}/`: shared
  stdio transport and standalone MCP servers.
- `crates/rust_tools_macros/`, `crates/re/`, `scripts/`, `tests/`, `models/`:
  macros, memo CLI, helpers, integration tests, and runtime model registry.

The `mcp_*` crates are not dependencies of `a`; their shared server contract lives
in [the mcp_stdio guide](crates/mcp_stdio/AGENTS.md).
`src/bin/mcp_ocr.rs` is a hand-rolled exception.

## Build / Test

`cargo check` / `cargo test` are expensive (heavy deps, slow incremental).
Don't run speculatively, but do run when the ladder demands it — reading code
cannot confirm compilation. Always scope to the narrowest target, and never run a
bare `cargo test` / `cargo build --release` / workspace-wide command for routine
verification. Never repeat `cargo test` without a code change, and prefer an
existing focused test over writing a new one.

```bash
cargo check --bin a                  # fast type-check for the main binary
cargo check -p aios_kernel           # type-check one crate
cargo test --bin a test_name         # run one targeted test in `a`
cargo test -p aios_kernel test_name  # run one targeted test in a crate
```

**Verification ladder:**

1. **No code change / docs-only**: no Cargo command required.
2. **Type-level / compile-risk / mechanical refactor**: narrowest `cargo check`.
3. **Runtime behavior changed, focused test exists**: run that named test.
4. **Runtime behavior changed, no focused test**: narrowest `cargo check`; note no test found.
5. **Bug fix with regression/new test**: run that named test; on failure fix and re-run.

**Formatting scope.** Never run `cargo fmt` (any form): at workspace root it
rewrites every member, and `cargo fmt -- <file>` is not scoped. Format only a
file you touched with
`rustfmt --edition 2024 --config skip_children=true <file>`.

**Flag verification.** A CLI/config flag is wired only when its effect is
observable (effective config, argv, or output); a clean exit does not prove the
key was accepted, because unknown keys may be ignored.

To deploy changes to the server-capable `a`, build and install all release
features, then restart the managed server:

```bash
make install-all
a --serve-restart
```

Read [the Serve guide](src/bin/ai/serve/AGENTS.md) for restart timing and
verification before using this entry point.

## Global Engineering Rules

1. **Module boundaries**: prefer `pub(super)` / `pub(crate)` for implementation
   details; use `pub` for the crate's intentional public API.
2. **English comments**: all line, block, and doc comments must be English; when
   modifying code, translate any Chinese comments you touch.
3. **Collections**: use `rustc-hash` FastMap/FastSet through the owning crate's
   exports; `rust_tools::commonw::{FastMap, FastSet}` is the root-crate example,
   not a dependency requirement for standalone crates.
4. **Tests**: keep close to the changed module; binary-specific test facilities
   belong in that binary's guide.
5. **Extensibility**: data-driven/registration-based over hardcoded `if`/`else`.
6. **AGENTS.md maintenance**: update only superseded nearby rules, in place; never
   append history. Keep injected instructions concise and delete stale rules.
   Reference code by symbol + file, never by line number (line refs rot on any
   edit); keep tuning numbers in code, only structure and invariants in docs.
7. **Git safety**: never `stash`/`stash drop` others' uncommitted changes. Use temp branch/worktree or stash only your own.
8. **Architecture-first**: many fallbacks = wrong data flow — refactor the happy path instead.

## Runtime-session inspection

Inspect runtime sessions only through the `session` tools; the full boundary is
in [the AI guide](src/bin/ai/AGENTS.md#session-inspection--sessionid-debugging).
