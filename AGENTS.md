# AGENTS.md — rust_tools Project Guide

## Overview

Rust workspace producing a core utility library and multiple CLI binaries. The primary product is `a` — an LLM-based AI Agent system (AIOS) with built-in process scheduling, Agent/Skill routing, tool registry, and MCP integration.

- **Edition**: Rust 2024
- **Workspace**: `.` (main), `crates/rust_tools_macros`, `crates/aios_kernel`
- **Platform**: macOS-first (`objc2` deps); core library is cross-platform

## Directory Layout

```
src/
├── lib.rs                    # Library root
├── algow/ clipboardw/ cmd/ commonw/ cw/ jsonw/ pdfw/ sortw/ strw/ terminalw/
└── bin/
    ├── a.rs                  # ★ AI Agent entry point
    ├── ai/                   # ★ AI Agent core (largest module)
    │   ├── driver/           # AIOS driver (run_loop, turn_runtime, thinking, reflection)
    │   ├── tools/            # Tool registry (registry/ service/ storage/ + *_tools.rs)
    │   ├── builtin_agents/   # 5 agents: build, executor, explore, plan, prompt-skill
    │   ├── builtin_skills/   # 3 skills: debugger, code-review, refactor
    │   ├── agents.rs skills.rs models.rs config.rs types.rs cli.rs
    │   ├── history/ knowledge/ mcp/ prompt/ stream/ config/
    │   └── config_schema.rs  # All config key constants (AiConfig)
    ├── ff/ fk/               # File search tools
    └── c.rs j.rs gx.rs ...   # Other CLI utilities
crates/
├── aios_kernel/              # Process scheduler (state machine, IPC, signals, shm)
├── rust_tools_macros/        # Proc macros (measure_time, lru_cache, agent_hang)
└── redis_lock/               # Redis distributed lock (not a workspace member)
tests/                        # Integration tests (Go compat layer + macro tests)
models.json                   # LLM model registry
```

## Build & Test

```bash
make all                          # Full build (all binaries)
make install                      # Incremental install (changed binaries only)
cargo build --release --bin a     # Build AI Agent
cargo check --bin a               # Type-check (fast validation)
cargo test --lib --bin a          # Run tests (531 tests)
cargo test --bin a test_xxx       # Filter tests by name
```

## Architecture: AIOS

1. **aios_kernel** — Process scheduler
   - State machine: Ready → Running → Waiting/Sleeping/Stopped → Terminated
   - Signals: SIGCANCEL / SIGTERM / SIGSTOP / SIGCONT / SIGKILL
   - IPC: mailbox messaging + shared memory (shm_create/read/write/delete)

2. **Driver layer** (`src/bin/ai/driver/`)
   - `run()` → `run_loop()` main event loop
   - Each tick: schedule → background processes → foreground input → turn execution
   - `turn_runtime/`: prepare → iterate (LLM + tools) → finalize
   - `thinking/`: goal decomposition and verification
   - `reflection/`: background reflection and knowledge writeback
   - `commands/`: interactive commands (agent/feishu/session/model/share)
   - `embedding/`: document embedding index
   - Agent/Skill routing: ML models (intent/agent_route/skill_match) + heuristics

3. **Agent/Skill system**
   - `.agent` files: YAML front-matter + Markdown prompt. Priority: project > workspace > user > builtin
   - `.skill` files: YAML front-matter + prompt body. Triggered via tool calls
   - Routing: ML models (intent/agent_route/skill_match) + heuristic rules

4. **Tool system** (`src/bin/ai/tools/`)
   - Progressive loading: core tools enabled by default; extras via `enable_tools`
   - Registry pattern: registry/ defines JSON Schema → service/ implements → storage/ persists
   - MCP integration: stdio JSON-RPC transport
   - Tool groups: core, builtin, executor, etc.
   - `ast_symbols/`: multi-language AST extraction (Rust/Python/Java/Go/TS/JS/C/C++)

5. **Configuration**
   - Keys defined in `config_schema.rs` (`AiConfig` constants) — never use raw string literals
   - Runtime config via `configw::get_all_config()`
   - Model registry: `models.json` (endpoints, quality tiers, VL support)

## Coding Standards

### General

1. **Visibility**: Use `pub(super)` / `pub(crate)` to enforce module boundaries
2. **Comments**: Project uses Chinese comments extensively — keep new code consistent
3. **FastMap/FastSet**: Use `rustc-hash` FxHashMap/FxHashSet (re-exported via `commonw` and `aios_kernel::types`)
4. **SkipMap**: Custom concurrent skip-list for ordered iteration (Agent/Skill loading)
5. **Error handling**: Library code → `Result<T, Box<dyn Error>>`; Agent tools → `Result<String, String>`
6. **Tests**: Inline `#[cfg(test)] mod tests` in modules; cross-crate in `tests/`; use `ENV_LOCK` for serial tests

### AI Module

1. **Agent files** (`.agent`): Required: `name`/`description`. Optional: `mode`/`model`/`tools`/`tool_groups`/`routing_tags`/`model_tier`/`color`
2. **Skill files** (`.skill`): Required: `name`/`description`. Optional: `tools`/`tool_groups`/`triggers`/`priority`/`skip_recall`
3. **Tool naming**: snake_case, verb-first (`read_file`, `execute_command`)
4. **Tool registration**: Define schema in `tools/registry/`, implement in `tools/service/`
5. **Config keys**: Add to `config_schema.rs` `AiConfig` — no scattered string literals

### Testing

- Naming: `test_feature_description` (snake_case)
- Integration tests with `_go_compat` suffix = Go compatibility layer tests
- Full suite: `cargo test --lib --bin a` (currently 531 tests)
- Serial tests: guard with `test_support::ENV_LOCK`

## Pitfalls

1. **`include_str!`**: Agent/Skill files are compiled into the binary — editing `.agent`/`.skill` triggers recompilation
2. **`ff_embed`**: AI Agent embeds `ff` module via `include!` — changes to `src/bin/ff/` affect `a`
3. **Feature flags**: `agent-hang-debug` is debug-only; do not enable in normal development
4. **Process safety**: `GLOBAL_OS` and `App.os` share `Arc<Mutex<Kernel>>`; use `with_os_kernel` or `DRIVER_CTX` task-local on hot paths
5. **macOS-only**: `objc2*` crates compile only under `cfg(target_os = "macos")`

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `tokio` | Async runtime (multi-thread) |
| `reqwest` | HTTP client (LLM API calls) |
| `serde` + `serde_json` | Serialization |
| `rusqlite` | SQLite (history, command storage) |
| `tree-sitter-*` | Multi-language AST analysis |
| `crossterm` + `ratatui` | Terminal UI |
| `inventory` | Compile-time tool registration |

## Project Instruction Injection

When AI Agent runs in this project, it auto-discovers and loads these files as context:
`AGENTS.md`, `Agent.md`, `CLAUDE.md` (case variants).
Max 8,000 chars per file, 16,000 total. Project root detected via `.git`/`Cargo.toml` markers.
