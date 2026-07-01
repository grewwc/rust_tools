# AI Tools Detailed Guide

## Scope

Long-form reference for `src/bin/ai/tools/**`.

## Architecture

- `registry/`: schema, metadata, registration
- `service/`: execution logic
- `storage/`: shared helpers, persistence, command/file/path support
- tool groups and progressive loading determine default exposure

## Registration rules

- Use verb-first snake_case names.
- Define new tool schema/metadata in `registry/` and execution in `service/`.
- Put reusable state/path helpers in `storage/` instead of duplicating logic in
  each tool.
- Keep tests close to the changed tool when practical.

## Path and cwd semantics

- File-related helpers resolve relative paths against
  `runtime_ctx::effective_cwd()`.
- `FileStore::new` and default write roots must respect scoped sub-agent cwd.
- Do not re-implement path expansion when `commonw::utils::expanduser` or an
  existing helper already covers the case.

## Skill and tool availability

- Keep baseline entry-point tools explicit (`discover_skills`, `load_skill`,
  `enable_tools`, etc.) so narrow skill tool lists do not strand the model.
- Distinguish read-only introspection tools from behavior-changing tools
  (`load_skill` vs `activate_skill`).

## Command validation

- `service/command.rs` validates shell execution paths, not plain file contents.
- Prefer quote-aware / heredoc-aware / wrapper-aware parsing over naive
  substring blocking.
- Dangerous text inside literal file-writing payloads should not be blocked;
  actual shell-evaluated substitution or indirect execution should be.

## MCP-related tool exposure

- Real MCP tool names come from configured servers.
- If prompt hints say a tool can be enabled, the runtime should be able to
  enable that exact configured `mcp_*` tool name.

## Useful tests

- `ai::tools::service::command::tests::*`
- focused registry/service tests next to the touched tool
