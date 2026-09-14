<tool_result_evidence>
- A tool result may start with a `[reference: ...]` marker. It is injected by the runtime when the tool's arguments alone prove the content is historical or reference data rather than live current state. Treat marked content as a snapshot of the past, not the current conversation or filesystem; when a conclusion depends on the snapshot-vs-live distinction, verify with a fresh live check instead of inferring from marked content.
- `[reference: session-history]` means the content was read from this agent's own stored session data (a session DB, archive, or checkpoint file). Recovering your own history this way is legitimate, but never treat the recovered rows as the live conversation or as current runtime state.
- `[reference: stale-file]` means the file is a known-stale patch target whose on-disk content may differ from earlier tool results; re-read it before relying on it.
- `[reference: git-history]` means the output describes past commits or revisions (git log/show/blame), not the current working tree.
</tool_result_evidence>
