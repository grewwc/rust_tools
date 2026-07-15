# Driver Guide

## Module layout

- `mod.rs`: main run loop, session management, MCP init
- `background_dispatch.rs`: background process dispatch — select ready processes, decode task goals, build questions, clone app context, spawn tokio tasks with scope chain
- `scheduler.rs`: background process scheduler — epoch management, dispatch scoring, cooldown / circuit-breaker, batch selection
- `session.rs`: startup session choice, suspended session preview/selection, resume logic
- `agent_routing.rs`: skill manifest loading, primary agent activation, auto-routing, hot-reload, runtime manifest init
- `mcp_lifecycle.rs`: MCP preload, initialization, status display
- `process_context.rs`: history path generation, background subagent context resolution, wake-up prompt, turn quota finalization
- `tests.rs`: driver-level integration tests
- `tools/`: tool execution subsystem
  - `mod.rs`: tool routing, execution, retry, parallel batches, tool cache
  - `async_pipe.rs`: async tool pipe system (task_spawn/task_status/task_wait lifecycle, channel/futex, snapshot persistence)
  - `tests.rs`, `barrier.rs`, `oauth.rs`, `sync_task.rs`: specialized submodules
- `commands/goal.rs`: `/goal` slash command — persistent auto-continuation until goal achieved

All extracted modules re-export via `use <module>::*` so call sites stay unqualified.

## Scope

Applies to `src/bin/ai/driver/**`.
Read `docs/agent-guides/ai-driver.md` before changing the run loop, turn preparation, prompt assembly, thinking, reflection, or runtime context.

## Key invariants

1. **System prompt assembly.** `skill_runtime.rs` owns prompt assembly, project-instruction injection, and per-turn tool visibility (two lazy axes: MCP tools via `select_mcp_tools`/`build_hidden_mcp_tool_catalog`, executor primitives via `manifest_tool_definitions`/`build_hidden_execution_primitive_catalog`). See `mcp/AGENTS.md` inv. 6, `tools/AGENTS.md` inv. 4.

2. **Background process tracking + temp dir.** Background processes tracked by `background_dispatch.rs`. `runtime_ctx::temp_dir()` is per-session (`<sessions_root>/<session>.assets/tmp/`, outside project dir), persists across turns, no auto-cleanup — agent deletes via `delete_path`.

3. **Turn preparation.** `push_project_context()` keeps repo-local instruction docs available and must not silently drop scoped instruction docs. `build_project_instruction_prompt()` materializes auto-discovered `AGENTS.md`/`CLAUDE.md`. One-shot knowledge maintenance runs before interactive `run_loop()`.

4. **Subagent spawn depth.** `runtime_ctx::SUBAGENT_DEPTH` (task-local) tracks nesting; `MAX_SUBAGENT_SPAWN_DEPTH` (=2) caps the chain. Subagent at depth ≥2 calling `task`/`task_spawn` gets an error. Depth carried in `OsTaskGoal.spawn_depth`, scoped by both dispatch paths.

5. **Background dispatch notes.** Process/correction notes (truncation-retry hints, cache-hit notes, `discover_skills` followups) are turn-scoped: push to `messages` only, never via `append_message_pair`, so they don't persist into `turn_messages` and can't accumulate across turns.

6. **Session-title generation.** Post-turn background task, not critical path. In foreground turns, `finalize_turn()` schedules in background. Inline only for paths exiting immediately. Guard with per-session in-flight latch.

7. **Truncation retry.** Progressive escalation, dialect-aware: degrades `reasoning_effort` for effort-based models; forces `thinking_disabled_override` for `enable_thinking`-switch models on 1st truncation. Stream-error truncations (`StreamResult::stream_error`) use independent counter without effort degradation. `max_tokens` is clamped per request by remaining window (prompt tokens estimated ~2 chars/token, including tool schema, capped at 2× char estimate for compression turns).

8. **Foreground resume turns.** Wake-up prompt persisted as `internal_note` (not `user`) via `runtime_ctx::IS_RESUME_TURN` task-local. Prevents synthetic prompts from polluting `/history user` or inflating compression's user-turn count.

9. **Coaching notes use `ROLE_INTERNAL_NOTE`.** Never `role: "system"` for runtime-injected notes (loop-breaker, iteration-soft-limit, task-anchor, etc.). `role: "system"` is `Critical/Never` in `classify_message`, so trimming it triggers compression rollback cascade. `ROLE_INTERNAL_NOTE` is `Medium/SafeLossy` and remapped to `"system"` before the API call.

10. **Iteration soft-limit.** Early converge prompt at `128` rounds (default `max_iterations=2048`). Inject `task-anchor` with `[iteration-soft-limit]` note to pull back exploratory runs.

11. **Tool-loop detection.** Normalize low-yield `execute_command` variants (collapse trailing `| head -N`, `2>/dev/null`, flag churn on same target). Escalate persistent repeats to no-tool handoff: reject new tool calls, request stage summary / conclusion / handoff. Preserve search-pattern anchors.

12. **Tool-result overflow.** Newly produced results enter `messages` raw first — ingress must not weaken them before the "recent stay verbatim" protection applies. Non-compressible tools (`read_file`/`code_search`/`text_grep`/`plan`) skip lossy folding. Pre-request budget offload (`prepare_tool_messages_structured`) protects most recent `KEEP_RECENT_TOOL_MESSAGES` results from spilling. Mid-turn compress (`orchestrator.rs`) dedups byte-identical repeated reads via `dedup_repeated_tool_results` — but only fires above its soft threshold, so **long loops lower the effective threshold** to `MID_TURN_COMPRESS_SOFT_FLOOR` at `LONG_LOOP_COMPRESS_ITERATION_THRESHOLD` iterations (`effective_mid_turn_soft_threshold`), curbing O(n²) resend on medium-history/many-iteration turns. Gate and actual `mid_turn_compress` call must share the same threshold (the call no-ops below it).

13. **Progress budget.** Third loop-defense layer (`TurnSupervisor::assess_progress`) after exact-byte and coarse detection. Catches "args change but task never advances" loops. Billed by intent-alignment (`classify_task_intent`: Mutation vs ReadOnly), not action count. Escalation: free-explore → `LowProgressSoft` → `LowProgressLedger` → `LowProgressHard` (no-tool handoff). Thresholds are hardcoded `const`.

14. **Multi-agent delegation nudges.** `ThinkingOrchestrator::on_prepare_rich` injects optional Context Budget (P3, `turn_index >= 12`) and Task Decomposition Hint (P4, `detect_decomposition_signals`) sections. Both advisory, one-shot latched, gated on `task_spawn` availability.

15. **Context-budget rollback.** Condition 2 uses strict `>` (not `>=`): `after_chars > after_lossless_chars` triggers rollback only when lossy is strictly worse. On tie, lossy kept for better structure.

## Goal 模式

`/goal` slash command 启动 goal 模式：agent 自动持续推进目标直到完成。

- **状态存储**: `App::goal_mode` — `None`=未启用；`Some("")`=等待目标；`Some(goal)`=已设定。`last_turn_had_tool_calls` / `last_turn_interrupted` track prior-turn state.
- **自动推进**: 上一轮有工具调用时跳过用户输入，注入 continuation prompt 驱动下一轮。
- **收尾判定**: `should_exit_goal_on_idle` — 自然完成视为目标达成并退出；Ctrl+C 打断则保留 goal 模式、回落到等待输入。

## Related detailed guide

- `docs/agent-guides/ai-driver.md`
