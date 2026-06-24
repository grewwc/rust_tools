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
    │   ├── agents.rs skills.rs persona.rs models.rs config.rs types.rs cli.rs
    │   ├── history/ knowledge/ mcp/ prompt/ stream/ config/ provider/
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
cargo test --lib --bin a          # Run tests (709 tests)
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
   - `commands/`: interactive commands (agent/feishu/session/model/share/persona)
   - `embedding/`: document embedding index
   - Persona system: `persona.rs` persists long-lived personas in `~/.config/rust_tools/personas.json`; switching persona rewrites the active history root and memory path so each persona keeps isolated sessions and long-term memory. `/personas` supports create/select/delete/current/list, and avatar is optional but unique when set.
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
   - Agent teams: `agent_team` launches parent-mediated multi-agent deliberation phases (`start` → `challenge` → `synthesize`) on top of existing `task_spawn` / `task_wait` kernel process, channel, and futex plumbing.
   - Knowledge tools: `knowledge_save`, `knowledge_forget`, `knowledge_search`, `knowledge_list`, `knowledge_consolidate`
   - `knowledge_consolidate` — two-phase AI-driven consolidation:
     `action: "read_all"` → returns all entries for LLM analysis;
     `action: "execute"` → batch-deletes obsolete IDs + batch-saves entries.
     Backed by `MemoryStore::delete_by_ids` / `append_batch`.
   - Memory tools: `memory_save`, `memory_search`, `memory_recent` (agent internal)

5. **Configuration**
   - Keys defined in `config_schema.rs` (`AiConfig` constants) — never use raw string literals
   - Runtime config via `configw::get_all_config()`
   - Model registry: `models.json` (endpoints, quality tiers, VL support)
   - Automatic model selection skips runtime-unhealthy models after request failures; `ai.model.disabled` can still exclude known unavailable or too-expensive model keys/names without editing `models.json`.
   - Embedding (optional, off by default): set `ai.embedding.enable=true` + `aliyun.api_key` (or `ai.embedding.api_key`) to enable semantic recall via Aliyun 百炼 OpenAI-compatible `/embeddings` (`text-embedding-v4`). Any failure degrades to BM25/lexical — see [embedder.rs](src/bin/ai/knowledge/indexing/embedder.rs).

6. **Provider adapter layer** (`src/bin/ai/provider/`)
   - `ApiProvider` enum { Compatible, OpenAi, OpenCode } + `ModelQualityTier` / `ReasoningEffort` live in `provider/mod.rs`
   - `provider/adapter.rs`: `trait ProviderAdapter` (template-method + override) collapses all per-provider behavior differences into one place — request-body fields (`enable_thinking`/`enable_search`/`reasoning_effort`/nested `reasoning`), aux-task thinking disable, default endpoint, API-key candidate chain, stream-chunk parsing, and the waiting-hint UX flag
   - Zero-state static singletons: `CompatibleAdapter` / `OpenAiAdapter` / `OpenRouterAdapter` / `OpenCodeAdapter`; dispatch via `adapter_for(provider, endpoint)` (OpenRouter detected by endpoint containing `openrouter.ai`)
   - Main pipeline (`request.rs` / `models.rs` / `stream/normalize.rs` / `stream/runtime.rs` / `driver/reflection/background.rs`) keeps free-function skeletons and only calls adapter hooks at difference points — wire format stays byte-identical (locked by per-provider `build_request_body` wire-guard tests)

## Coding Standards

### General

1. **Visibility**: Use `pub(super)` / `pub(crate)` to enforce module boundaries
2. **Comments**: Project uses Chinese comments extensively — keep new code consistent
3. **FastMap/FastSet**: Use `rustc-hash` FxHashMap/FxHashSet (re-exported via `commonw` and `aios_kernel::types`)
4. **SkipMap**: Custom concurrent skip-list for ordered iteration (Agent/Skill loading)
5. **Error handling**: Library code → `Result<T, Box<dyn Error>>`; Agent tools → `Result<String, String>`
6. **Tests**: Inline `#[cfg(test)] mod tests` in modules; cross-crate in `tests/`; use `ENV_LOCK` for serial tests
7. **Scope**: Do not reformat or reorder files unrelated to the task. Only modify files that need to change — no incidental formatting.
8. **No unrelated changes**: Do not make any changes that are not required by the current task. This includes refactoring, reformatting, renaming, reorganizing, commenting, or any other modification to files, symbols, or code that is not directly relevant to the feature or fix being implemented.
 
9. **Avoid hardcoded string rules**: Do not resort to hardcoded string-based judgment logic unless absolutely unavoidable. Prefer data-driven approaches — configuration, model-based routing, trait dispatch, or match on well-typed enums — over brittle string matching or regex-based rules.

### AI Module

1. **Agent files** (`.agent`): Required: `name`/`description`. Optional: `mode`/`model`/`tools`/`tool_groups`/`mcp_servers`/`disable_mcp_tools`/`routing_tags`/`model_tier`/`color`
2. **Skill files** (`.skill`): Required: `name`/`description`. Optional: `tools`/`tool_groups`/`triggers`/`priority`/`skip_recall`
3. **Tool naming**: snake_case, verb-first (`read_file`, `execute_command`)
4. **Tool registration**: Define schema in `tools/registry/`, implement in `tools/service/`
5. **Config keys**: Add to `config_schema.rs` `AiConfig` — no scattered string literals
6. **Knowledge consolidation**: `knowledge_consolidate` tool provides two-phase AI-driven consolidation: `read_all` returns all entries for LLM analysis; `execute` batch-deletes + batch-saves. Backed by `MemoryStore::delete_by_ids` / `append_batch`.

### Testing

- Naming: `test_feature_description` (snake_case)
- Integration tests with `_go_compat` suffix = Go compatibility layer tests
- Full suite: `cargo test --lib --bin a` (currently 709 tests)
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

## Maintaining AGENTS.md

**This document MUST be kept in sync whenever new features are added.** Specifically:

1. **New modules/directories** → update the `Directory Layout` tree
2. **New tools** → document in `Tool system` or `AI Module`
3. **New agents/skills** → update counts and names in `builtin_agents/` / `builtin_skills/`
4. **New dependencies** → add to the `Key Dependencies` table
5. **New config keys** → document in `Configuration`
6. **Architecture changes** → update the relevant `Architecture` section
7. **Build/test changes** → update commands or test counts in `Build & Test`
8. **New pitfalls** → add to `Pitfalls` when encountered

> Principle: Keep AGENTS.md consistent with the actual codebase so the AI Agent always loads accurate project context.
