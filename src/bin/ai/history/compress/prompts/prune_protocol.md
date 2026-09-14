
## Context Management Protocol
When your context holds outdated tool results, actively reclaim space by marking them.
Each tool result in the history has a stable id (the `call_id` / `tool_call_id` shown on
that tool output). Include a hidden self-note listing the ids to prune:
`<meta:self_note>prune:call_abc,call_xyz</meta:self_note>`
Mark any tool result that is now superseded or no longer needed — including old file
reads and code/search results whose content you have already used, that you have since
re-read, or that describe code you have already edited.
Rules:
- Never mark user messages, system instructions, assistant messages, plans, or the most recent tool results.
- Marking is safe and reversible: pruning is loss-free — the full result is archived to a
session file and the kept stub shows its `file_path`, so you can re-read it anytime if you
turn out to still need it. Marks accumulate across turns and need not be consecutive: a result
offloads only once it reaches its threshold (`marks n/threshold` in the candidate list), except
that very large results offload after a single mark. Recent results and plans are always protected.
- Put the `prune:` directive on its own line; if you also write a normal self_note, keep it in the same hidden note.