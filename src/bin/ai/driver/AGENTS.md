# Driver Guide

## Module layout

- `mod.rs`: main run loop, session management, background dispatch, MCP init
- `tests.rs`: driver-level integration tests
- `tools/`: tool execution subsystem
  - `mod.rs`: tool routing, execution, retry, parallel batches, tool cache
  - `async_pipe.rs`: async tool pipe system (task_spawn/task_status/task_wait lifecycle, channel/futex, snapshot persistence)
  - `tests.rs`: tool execution tests
  - `barrier.rs`, `oauth.rs`, `sync_task.rs`: specialized submodules
- `commands/goal.rs`: `/goal` slash command — activates goal mode (persistent auto-continuation until goal achieved)

## Scope

Applies to `src/bin/ai/driver/**`.

Read `docs/agent-guides/ai-driver.md` before changing the run loop, turn
preparation, prompt assembly, thinking, reflection, or runtime context.

## Key invariants

1. `skill_runtime.rs` owns system prompt assembly and project-instruction injection.
2. `build_project_instruction_prompt()` materializes auto-discovered
   `AGENTS.md` / `CLAUDE.md` docs from `agents.rs`.
3. `push_project_context()` must keep repo-local instruction docs available for
   repo-scoped work; it should not silently drop the actual scoped instruction
   docs.
4. One-shot knowledge maintenance flows run before the interactive `run_loop()`.
5. `runtime_ctx::temp_dir()` is per-session
   (`<sessions_root>/<session>.assets/tmp/`, co-located with tool-overflow,
   outside the project dir; falls back to `<effective_cwd>/.agent_tmp/<session>/`
   when DRIVER_CTX is unavailable) and persists across turns. There is no
   auto-cleanup; the agent explicitly deletes temp files via `delete_path`
   (which only works on files registered via `write_file(temp=true)`).
6. Process/correction notes (truncation-retry hints, cache-hit notes,
   discover_skills followups) are turn-scoped: push them to `messages` only,
   never via `append_message_pair`, so they are not persisted into
   `turn_messages` and cannot accumulate across turns. This also applies to
   **partial assistant text from truncated responses**: it is a half-finished
   artifact, not a valid conversation record — persisting it pollutes the
   history file and crowds out real dialog under `history_max_chars`.
7. `turn_runtime/persistence.rs` skips persisting turn messages only for
   *ephemeral* one-shot runs (`one_shot_mode && cli.session.is_none()`), i.e.
   the runs `cleanup_one_shot` will delete right after. Background mode
   (`a -bg`) and explicit `--session` one-shot (`a -ss <id> "q"`) keep the
   session, so they MUST persist — otherwise `/sessions` title and `/history`
   come up empty.
8. Truncation retry uses **progressive escalation**, not a flat retry:
   - Escalation is **dialect-aware**: `provider::reasoning_effort_reduces_thinking_for`
     decides whether degrading `reasoning_effort` can actually shrink the thinking
     chain for this model.
   - For **top-level `reasoning_effort` models** (OpenAI-compatible family),
     effort degrades 1st truncation to Low, 2nd to Minimal, 3rd+ to disabled
     (frees output budget for actual content).
   - For **`enable_thinking`-switch models** (e.g. GLM via `EnableThinkingDialect`),
     effort degradation is a **no-op** — the request body carries no `effort`
     field at all. So the ladder is skipped: the **1st** genuine truncation sets
     `thinking_disabled_override` (an `App.cli` flag read by
     `request::resolve_thinking`) to force thinking off entirely, instead of
     wasting two retry rounds on an ineffective effort ladder. Both this flag and
     `reasoning_effort_override` are saved on turn entry and restored at every
     `break 'turn` exit, so the downgrade never leaks into later turns.
   - The shrink note is **replaced** each truncation (not idempotent) and
     carries the consecutive count, so the model gets escalating feedback
     instead of flying blind after the first attempt.
   - **Stream-error truncations** (`StreamResult::stream_error`, caused by
     decode errors / dropped malformed tool calls) are retried **without**
     reasoning_effort degradation or shrink-note injection — the model didn't
     output too much, the stream broke. They use an **independent counter**
     (`consecutive_stream_errors`, capped by `MAX_STREAM_ERROR_RETRIES`) rather
     than `consecutive_truncations`, so persistent disconnects still terminate
     the turn (critical for background tasks whose `max_iterations` is
     `usize::MAX`). Only `truncated_by_length` truncations (genuine output-limit
     hits) get the progressive-escalation treatment.
   - **Root-cause guard (prevention, not just retry):** `max_tokens` is clamped
     per request by remaining window — `min(model.max_output_tokens,
     context_window_tokens - est_prompt_tokens - margin)` in
     `request::clamp_max_tokens_for_prompt` (only emitted when the model
     declares `max_output_tokens`). Prompt tokens are estimated conservatively
     (~2 chars/token) and **include the serialized tool schema** (`tools` JSON),
     since tool/MCP definitions ship with every request and occupy the prompt
     window — omitting them over-estimates the available output budget. When a
     server-reported `known_prompt_tokens` is passed, it
     is capped at `2×` the char estimate: after a compression turn the prompt
     drops sharply but the carried-over `known` is still the pre-compression high
     value — using it verbatim would mis-clamp `remaining` to the floor (1024),
     which an always-thinking model (GLM) burns entirely on reasoning → zero
     visible text → length-truncation retry storm. The cap falls back to this
     turn's char estimate on compression turns. Compression thresholds
     (`mid_turn_compress_soft/hard_threshold`, `pre_request_llm_summary_threshold`)
     are additionally capped by `token_window_char_ceiling(model)` (window * 2 *
     0.6 chars), so a high-occupancy prompt triggers compression before it
     approaches the real token window — fixing the char-vs-token unit mismatch
     that let GLM prompts overflow without ever tripping the char-only threshold.
9. Foreground resume turns (process woke up by mailbox events) must persist
   their wake-up prompt as `internal_note` (not `user`). The
   `runtime_ctx::IS_RESUME_TURN` task-local is scoped by
   `run_foreground_resume`; `prepare_turn` reads it and sets the
   `turn_messages` message role accordingly. The `messages` array sent to the
   API keeps `role: "user"` for provider compatibility. This prevents
   synthetic wake-up prompts from polluting `/history user`, inflating
   compression's user-turn count, or being misread by the model as repeated
   user questions.

10. Tool-result overflow (`turn_runtime/tool_result/`): **newly produced**
   tool results must enter `messages` as raw model content first. Do not let
   the ingress formatter weaken the current round's result into an overflow
   stub / summary before the "recent tool results stay verbatim" protection has
   a chance to apply; otherwise `KEEP_RECENT_TOOL_MESSAGES` only preserves an
   already-weakened placeholder and the model behaves as if the tool result was
   never really returned. The terminal preview may still use overflow files /
   summaries for UX, but the model-side content for the current round stays raw.
   Historical / non-recent tool results still use the existing overflow path:
   single tool results are inline up to `max_tool_result_inline_chars(model)`
   (model-aware: 32K floor, ~12.5% of context window, 64K cap), and older large
   results may spill to a session file stub that includes summary + `key_lines`
   + head/tail preview as recall anchors. `is_non_compressible_tool`
   (read_file/code_search/text_grep/web_search/**plan**/etc.) never goes through
   lossy `line_trim_middle` or whole-group folding — `plan` is the multi-step
   roadmap anchor: small in size but losing it makes the model forget the task
   mid-flight, so it is preserved verbatim like retrieval results.
   Pre-request budget offload (`prepare_tool_messages_structured`) is therefore
   the **second** line of defense: it additionally protects the most recent
   `KEEP_RECENT_TOOL_MESSAGES` tool results from spilling, including
   non-compressible ones. Only precision results *outside* the recent window
   spill to disk.

11. Coarse tool-loop detection must also normalize low-yield
   `execute_command` shell variants, not just JSON paging args. Collapse pure
   result-window / noise differences such as trailing `| head -N`,
   `2>/dev/null`, and `ls -la/-lt` flag churn when the command is still probing
   the same target path, so repeated log-directory listing triggers the
   `[low-yield-repetition]` note. If the same coarse `execute_command` target
   keeps repeating across a longer window, escalate to a **no-tool handoff
   mode** instead of waiting for the generic iteration soft-limit. This mode is
   not a `Ctrl+C`-style abort: the current stream is not cancelled; the next
   request keeps tool schemas visible for grounding, but the execution layer
   rejects any new tool call and asks the model to produce a stage summary /
   current conclusion / remaining work / next-step handoff. Preserve substantive
   anchors such as search patterns / target paths so distinct sub-questions can
   still differ.

12. Single-turn iteration soft-limit is an early converge prompt, not a last-ditch
   warning. With the default `max_iterations=2048`, cap the soft prompt at `128`
   rounds (not several hundred), and inject a `task-anchor` together with the
   `[iteration-soft-limit]` note so long exploratory runs get pulled back to the
   user goal before they spend another hundred tool calls drifting.
13. Session-title generation is a post-turn quality task, not part of the
   foreground interaction critical path. In live interactive foreground turns,
   `finalize_turn()` must schedule it in the background instead of awaiting the
   model call before returning to `run_loop()`. Keep inline generation only for
   paths that are about to exit immediately (ephemeral one-shot / explicit quit),
   and guard with a per-session in-flight latch so repeated turns do not launch
   duplicate title jobs.

## Goal 模式

`/goal` slash command 启动 goal 模式：agent 自动持续推进目标直到完成。

- **状态存储**: `App::goal_mode` — `None`=未启用；`Some("")`=等待用户输入目标；
  `Some(goal)`=目标已设定，自动推进。`App::last_turn_had_tool_calls` 标记上一轮
  是否有工具调用；`App::last_turn_interrupted` 标记上一轮是否被 Ctrl+C 打断
  （每次 `run_turn` 入口清零，仅 `finish_interrupted_turn` 置位）。
- **交互方式**: `/goal` 后可直接跟目标文本，也可只输入 `/goal` 再在下一轮输入目标。
  `/goal off` 退出 goal 模式。
- **自动推进**: `run_loop` 在 goal 模式下，若上一轮调用过工具，则跳过用户输入、
  注入 continuation prompt 驱动下一轮。
- **收尾判定**: 上一轮无工具调用时由 `commands::goal::should_exit_goal_on_idle`
  决策——**自然完成**（未打断）视为目标达成，打印 `Goal achieved` 并退出；
  **被 Ctrl+C 打断**则保留 goal 模式、静默回落到等待用户输入，不误报达成。
  二者都会把 `last_turn_had_tool_calls` 置 false，故必须靠 `last_turn_interrupted`
  区分，不能仅凭前者判定。

## Related detailed guide

- `docs/agent-guides/ai-driver.md`
