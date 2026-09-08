# scripts/ AGENTS.md

Repo shell/Python scripts. Keep each script self-contained and documented in
its own header.

## postprocess_terminal.py

Display-only post-processor for the agent's terminal body text. Converts
Chinese (fullwidth/ideographic) punctuation **inside code or file-location
contexts** to ASCII, plus fullwidth parentheses `（` `）`, and fullwidth
colon `：` (as `: `) / period `。` (as `. `) -- each only when it directly
abuts an ASCII letter or follows a closing bracket, except that a period
immediately before an inline-code span starting with an ASCII letter receives
the same prose replacement. Prose paren spacing is omitted next to Markdown
emphasis delimiters (`**（x）**` stays bare) and at word edges. This keeps the
rendered code span separated from the sentence while pure-Chinese prose
retains `：` and `。`; all other Chinese punctuation in plain prose stays
untouched. In the reverse direction, halfwidth English sentence punctuation
(`,` `.` `!` `?` `;` `:`) in plain prose gets a halfwidth space wherever it
directly abuts a CJK ideograph, so mixed CJK/ASCII text never jams a mark
against a Chinese character.

- Contexts translated: fenced code blocks, inline code spans (`` `...` ``),
  and file-path/file-reference spans in prose (path indicators, drive
  prefixes, `name.ext[:line[:col]]`, dotted words followed by a separator),
  and fullwidth parentheses / colon / period in plain prose.
- Reverse direction: halfwidth English sentence punctuation (`,` `.` `!` `?`
  `;` `:`) in plain prose gets a halfwidth space wherever it directly abuts a
  CJK ideograph -- before the mark (`完成,rustc` -> `完成 ,rustc`), after it
  (`rustc,完成` -> `rustc, 完成`), or both (`完成!继续` -> `完成 ! 继续`),
  e.g. `` `mod.rs:11`,未改变 `` becomes `` `mod.rs:11`, 未改变 `` -- so the
  break is not jammed against the Chinese text. Marks inside code/path
  tokens, next to ASCII letters/digits, and at line edges (no trailing
  whitespace) are left alone.
- ANSI escape sequences are preserved verbatim, so it also works as a pipe
  filter directly on rendered terminal output.
- Filter contract: reads stdin, writes transformed text to stdout.
  `--selftest` runs the built-in regression cases.

Integration point in the agent: `ai.output.postprocess_command` config key
(`src/bin/ai/config_schema.rs`), applied in
`src/bin/ai/driver/turn_runtime/finalize.rs` right before the final
`render_markdown_block`, and in the `/history last` replay
(`src/bin/ai/driver/input.rs`) so replays render exactly what the live turn
painted. The pipe is best-effort: on any failure the original text is shown
unchanged and canonical history is never modified.
