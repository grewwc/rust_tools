<compressed_context_recovery>
- Compressed-out evidence is not lost: truncated/folded tool output and folded messages are archived verbatim into the session overflow archive, leaving stubs/pointers in history.
- Stub markers (`[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]`, `[context-overflow-truncated]`) point to a plain-text archive file, not project source; the inline preview is a recall anchor, not the whole output. Read the archive only for exact historical output; when a stub names an original file or command, follow its hint and re-read that source for current content. These history pointers are distinct from the runtime-injected `[reference: ...]` snapshot markers.
- `_context_overflow_truncated` inside tool arguments is a placeholder, not real arguments: never resend it as a tool call.
- For consequential absence claims (where "not found" or "does not exist" changes a conclusion), search the session archive with `search_overflow` first — absence claims must cover the archived scope, not just the current context window. Trivial negative observations do not require an archive search.
- `search_overflow` returns verbatim excerpts with absolute file paths and line numbers; default to `scope=all` unless you know where the content lives.
</compressed_context_recovery>
