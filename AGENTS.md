# AGENTS.md — rust_tools Project Guide

## Overview

This Rust workspace produces a core utility library and multiple CLI binaries. The primary product is `a`, an LLM-based AI Agent system (AIOS) featuring built-in process scheduling, Agent/Skill routing, a tool registry, and MCP integration.

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
    │   ├── builtin_agents/   # 5 built-in agents: build, executor, explore, plan, prompt-skill
    │   ├── builtin_skills/   # 3 built-in skills: debugger, code-review, refactor
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
cargo test --lib --bin a          # Run tests (956 tests)
cargo test --bin a test_xxx       # Filter tests by name
```

## Architecture: AIOS

1. **aios_kernel** — Process scheduler
   - State machine: Ready → Running → Waiting/Sleeping/Stopped → Terminated
   - Signals: SIGCANCEL / SIGTERM / SIGSTOP / SIGCONT / SIGKILL
   - IPC: mailbox messaging + shared memory (shm_create/read/write/delete)

2. **Driver layer** (`src/bin/ai/driver/`)
   - `run()` entry point invokes the main `run_loop()` event loop
   - Each iteration of the event loop: schedule → background processes → foreground input → turn execution
   - `turn_runtime/`: prepares context, then iterates through LLM calls interleaved with tool calls, then finalizes
   - `thinking/`: decomposes the user's goal into sub-goals and verifies the decomposition against available tools
   - `reflection/`: runs background reflection after each turn and writes knowledge back to long-term memory
   - `commands/`: processes interactive commands (agent/feishu/session/model/share/persona)
   - `embedding/`: builds and queries a document embedding index for semantic search
   - Persona system: `persona.rs` persists long-lived personas in `~/.config/rust_tools/personas.json`. Switching persona rewrites the active history root and memory path, so each persona keeps isolated sessions and long-term memory. `/personas` supports create/select/delete/current/list. An avatar is optional but must be unique when set.
   - Agent/Skill routing: uses ML models (intent/agent_route/skill_match) plus heuristic fallback rules.

3. **Agent/Skill system**
   - `.agent` files: YAML front-matter followed by a Markdown prompt body. Priority order: project > workspace > user > builtin.
   - `.skill` files: YAML front-matter followed by a prompt body. Activated through tool-driven routing and explicit `activate_skill` selection.
   - Skill packages: user skills may also be directories or `.zip` archives containing `SKILL.md` (or a top-level `.skill` manifest plus bundled resources). Zip packages are extracted into the skills cache and exposed to the active skill as a resource root. A directory or zip may also be a **collection** of multiple packages (e.g. `feishu/skills/<pkg>/SKILL.md`): when the root itself is not a single package, loading recurses into nested subdirectories and registers every package found. A directory that *is* a package short-circuits and is never descended into, so its bundled `references/*.skill` resources are not mistaken for separate skills.
   - Discovery sources: skill loading merges built-in skills, the user skills directory, and packaged `SKILL.md` skills discovered from standard `~/.trae-cn` install roots (builtin/global/extension skills). Built-in skill names remain pinned if a lower-precedence source reuses the same name.
   - Routing: uses ML models (intent/agent_route/skill_match) plus heuristic fallback rules.

4. **Tool system** (`src/bin/ai/tools/`)
   - **Progressive loading**: core tools are enabled by default; additional tools are loaded via `enable_tools`.
   - **Registry pattern**: `registry/` defines JSON Schema → `service/` implements the logic → `storage/` persists data.
   - **File tools**: `FileStore::new` resolves relative paths against `runtime_ctx::effective_cwd()`, so sub-agent-scoped working directories apply consistently to read/write/patch flows. `apply_patch` accepts both raw unified-diff hunks and the common single-file `*** Begin Patch` envelope; it also tolerates `path` as a compatibility alias for `file_path`.
   - **MCP integration**: communicates via stdio JSON-RPC transport.
   - **Tool groups**: organized into groups such as core, builtin, executor, etc.
   - **Code analysis**: `ast_symbols/` extracts AST symbols from multiple languages (Rust/Python/Java/Go/TS/JS/C/C++).
   - **Agent teams**: `agent_team` launches parent-mediated multi-agent deliberation phases (`start` → `challenge` → `synthesize`). This sits on top of `task_spawn`/`task_wait` using the existing kernel process, channel, and futex plumbing.
   - **Knowledge tools**: `knowledge_save`, `knowledge_forget`, `knowledge_search`, `knowledge_list`, `knowledge_consolidate`.
     - `knowledge_consolidate` is a two-phase AI-driven consolidation tool:
       - `action: "read_all"` — returns all entries for LLM analysis.
       - `action: "execute"` — batch-deletes obsolete IDs and batch-saves new entries.
       - Backed by `MemoryStore::delete_by_ids` and `append_batch`.
   - **Memory tools**: `memory_save`, `memory_search`, `memory_recent` — these are agent-internal, not user-facing.

5. **Configuration**
   - All config keys are defined as `AiConfig` constants in `config_schema.rs`. Never use raw string literals.
   - Runtime configuration is read through `configw::get_all_config()`.
   - Model registry lives in `models.json` and tracks endpoints, providers, quality tiers, and vision-language (VL) support. The user-facing selector is always lowercase `name-provider` (for example `deepseek-v4-flash-alibaba` or `deepseek-v4-flash-opencode`); `name` is the provider-facing model id, so multiple providers may share the same `name`. `key` should match the selector, while `aliases` may preserve older keys for backward compatibility.
   - Alibaba/DashScope models use provider `alibaba`; API key lookup prefers `alibaba.api_key`, then `aliyun.api_key`, then `compatible.api_key`, then the global `api_key`.
   - Automatic model selection: after a request failure, the runtime skips the unhealthy model for subsequent requests. You can also exclude known-unavailable or too-expensive model keys/names via `ai.model.disabled`, without editing `models.json`.
   - Embedding support (optional, off by default): set `ai.embedding.enable=true` and provide `aliyun.api_key` (or `ai.embedding.api_key`) to enable semantic recall via Aliyun 百炼's OpenAI-compatible `/embeddings` endpoint with `text-embedding-v4`. If embedding fails for any reason, the system gracefully degrades to BM25/lexical search — see [embedder.rs](src/bin/ai/knowledge/indexing/embedder.rs).
   - Decision-log sidecar persistence is optional: set `ai.decision_log.persist.enable=true` to write per-session `*.decision-log.jsonl`. Default is off; the in-memory `DecisionLogStore` still powers live routing/adaptation signals.

6. **Provider adapter layer** (`src/bin/ai/provider/`)
   - The `ApiProvider` enum (`Alibaba`, `Compatible`, `OpenAi`, `OpenCode`) and the types `ModelQualityTier`/`ReasoningEffort` are defined in `provider/mod.rs`.
   - `provider/adapter.rs` defines the `ProviderAdapter` trait using a template-method pattern with override hooks. It consolidates all per-provider differences into one place:
     - Request-body fields: `enable_thinking`, `enable_search`, `reasoning_effort`, nested `reasoning`.
     - Auxiliary task: disabling thinking for non-primary requests.
     - Default endpoint URL and API-key candidate chain.
     - Stream-chunk parsing logic.
     - The waiting-hint UX flag (indicates whether the provider shows a waiting hint in-stream).
   - Zero-state static singletons: `AlibabaAdapter`, `CompatibleAdapter`, `OpenAiAdapter`, `OpenRouterAdapter`, `OpenCodeAdapter`. Dispatch is done via `adapter_for(provider, endpoint)` — OpenRouter is detected when the endpoint contains `openrouter.ai`.
   - The main pipeline (`request.rs`, `models.rs`, `stream/normalize.rs`, `stream/runtime.rs`, `driver/reflection/background.rs`) keeps free-function skeletons and only calls adapter hooks at the points where providers differ. The wire format stays byte-identical across providers (locked by per-provider `build_request_body` wire-guard tests).

## Coding Standards

### General

1. **Visibility**: Use `pub(super)` / `pub(crate)` to enforce module boundaries.
2. **Comments**: Project uses Chinese comments extensively — keep new code consistent.
3. **FastMap/FastSet**: Use `rustc-hash` FxHashMap/FxHashSet (re-exported via `commonw` and `aios_kernel::types`).
4. **SkipMap**: Custom concurrent skip-list for ordered iteration (used during Agent/Skill loading).
5. **Error handling**: Library code uses `Result<T, Box<dyn Error>>`; Agent tools use `Result<String, String>`.
6. **Tests**: Write inline tests with `#[cfg(test)] mod tests` inside modules. Cross-crate tests go in `tests/`. Use `ENV_LOCK` for serial tests.
7. **Scope**: Only modify files that need to change. Do not reformat or reorder unrelated code.
8. **No unrelated changes**: Never refactor, reformat, rename, reorganize, or comment on code that is not directly relevant to the current task.
9. **Avoid hardcoded string rules**: Do not resort to hardcoded string-based judgment logic unless absolutely unavoidable. Prefer data-driven approaches — configuration, model-based routing, trait dispatch, or match on well-typed enums — over brittle string matching or regex-based rules.
10. **Favor reuse over reinvention**: Before introducing any new abstraction, dependency, or utility, first audit the existing codebase for components, traits, or tooling that can fulfill the requirement directly or with minimal extension. Preferring composition over custom implementation reduces maintenance surface area and preserves architectural coherence.

### AI Module

1. **Agent files** (`.agent`): Required fields are `name` and `description`. Optional fields are `mode`, `model`, `tools`, `tool_groups`, `mcp_servers`, `disable_mcp_tools`, `routing_tags`, `model_tier`, and `color`.
2. **Skill files** (`.skill`): Required fields are `name` and `description`. Optional fields are `tools`, `tool_groups`, `priority`, and `skip_recall`.
3. **Tool naming**: Use snake_case with a verb-first convention (e.g., `read_file`, `execute_command`).
4. **Tool registration**: Define the tool schema in `tools/registry/`, and implement the logic in `tools/service/`.
5. **Config keys**: Add new keys to `config_schema.rs` as `AiConfig` constants — never scatter string literals.
6. **Knowledge consolidation**: Use the `knowledge_consolidate` tool (two-phase AI-driven: `read_all` followed by `execute`). Backed by `MemoryStore::delete_by_ids` and `append_batch`.

### Testing

- **Naming**: Use snake_case: `test_feature_description`.
- **Integration tests**: If a test has a `_go_compat` suffix, it belongs to the Go compatibility layer test suite.
- **Full suite**: Run `cargo test --lib --bin a` (currently 956 tests).
- **Serial tests**: Guard them with `test_support::ENV_LOCK`.

## Pitfalls

1. **`include_str!`**: Agent and Skill files are compiled into the binary. This means editing a `.agent` or `.skill` file triggers recompilation.
2. **`ff_embed`**: The AI Agent embeds the `ff` module via `include!`. Changes to `src/bin/ff/` affect `a`.
3. **Feature flags**: `agent-hang-debug` is debug-only. Do not enable it in normal development.
4. **Process safety**: `GLOBAL_OS` and `App.os` share an `Arc<Mutex<Kernel>>`. Use `with_os_kernel` or the `DRIVER_CTX` task-local on hot paths.
5. **macOS-only**: `objc2*` crates compile only under `cfg(target_os = "macos")`.

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
| `zip` | Read user skill package archives |

## Project Instruction Injection

When the AI Agent runs inside this project directory, it auto-discovers and loads these files as project context:
`AGENTS.md`, `Agent.md`, `CLAUDE.md` (and case variants).
Each file is limited to 8,000 characters, with a total limit of 16,000 characters across all files. The project root is detected by the presence of `.git` or `Cargo.toml` markers.

## Maintaining AGENTS.md

**This document must be kept in sync whenever new features are added.** Specifically:

1. **New modules/directories** → update the `Directory Layout` tree.
2. **New tools** → document in the `Tool system` or `AI Module` section.
3. **New agents/skills** → update counts and names in `builtin_agents/` / `builtin_skills/`.
4. **New dependencies** → add to the `Key Dependencies` table.
5. **New config keys** → document in the `Configuration` section.
6. **Architecture changes** → update the relevant `Architecture` section.
7. **Build/test changes** → update commands or test counts in `Build & Test`.
8. **New pitfalls** → add to the `Pitfalls` section when encountered.

> Principle: Keep AGENTS.md consistent with the actual codebase so the AI Agent always loads accurate project context.
