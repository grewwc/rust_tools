# scripts/ AGENTS.md

Repo shell/Python scripts. Keep each script self-contained and documented in
its own header.

## postprocess_terminal.py

Display-only post-processor for the agent's terminal body text. Converts
Chinese (fullwidth/ideographic) punctuation **inside code or file-location
contexts** to ASCII, plus fullwidth parentheses `（` `）` and fullwidth
colon `：` (as `: `) in plain prose (deliberate exceptions so mixed
technical prose reads consistently); all other Chinese punctuation in
plain prose stays untouched.

- Contexts translated: fenced code blocks, inline code spans (`` `...` ``),
  and file-path/file-reference spans in prose (path indicators, drive
  prefixes, `name.ext[:line[:col]]`, dotted words followed by a separator),
  and fullwidth parentheses / colon in plain prose.
- ANSI escape sequences are preserved verbatim, so it also works as a pipe
  filter directly on rendered terminal output.
- Filter contract: reads stdin, writes transformed text to stdout.
  `--selftest` runs the built-in regression cases.

Integration point in the agent: `ai.output.postprocess_command` config key
(`src/bin/ai/config_schema.rs`), applied in
`src/bin/ai/driver/turn_runtime/finalize.rs` right before the final
`render_markdown_block`. The pipe is best-effort: on any failure the original
text is shown unchanged and canonical history is never modified.
