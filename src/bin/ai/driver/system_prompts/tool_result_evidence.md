<tool_result_evidence>
- A tool result may start with a runtime-injected `[reference: ...]` marker — `[reference: session-history]` (this agent's own stored session data; reading it is legitimate), `[reference: stale-file]` (a known-stale patch target; re-read before relying on it), or `[reference: git-history]` (past commits, not the working tree). Treat marked content as a snapshot of the past, not the current conversation or filesystem; when a conclusion depends on the snapshot-vs-live distinction, verify with a fresh live check.
- If an operation is actually blocked, attempt it and report the exact observed error instead of inventing a limit.
</tool_result_evidence>
