# scripts/ AGENTS.md

Repo shell/Python scripts. Keep each script self-contained and documented in
its own header.

## postprocess_terminal.py

Display-only filter for the agent's terminal body text. The script header
(`scripts/postprocess_terminal.py`) is the documentation home for its rules:
which Chinese punctuation is converted in code / file-location contexts and in
prose, the reverse CJK/ASCII spacing pass, ANSI preservation, and the
`--selftest` regression cases.

Integration: `ai.output.postprocess_command` (see `src/bin/ai/AGENTS.md`
invariant 11) applies it at both render sites — the live turn
(`driver/turn_runtime/finalize.rs`) and the `/history last` replay
(`driver/input.rs`, so replays render what the live turn painted).
