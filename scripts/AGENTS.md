# scripts/ AGENTS.md

Repo shell/Python scripts. Keep each script self-contained and documented in
its own header.

## postprocess_terminal.py

Display-only filter for the agent's terminal body text; rules live in the
script header (`scripts/postprocess_terminal.py`).

Integration: `ai.output.postprocess_command` (see `src/bin/ai/AGENTS.md`
invariant 11) applies it at both render sites — the live turn
(`driver/turn_runtime/finalize.rs`) and the `/history last` replay
(`driver/input.rs`, so replays render what the live turn painted).

## prompt_eval.py

Opt-in prompt-behaviour harness for the `a` agent; rules live in the script
header (`scripts/prompt_eval.py`). Bait set:
`scripts/prompt_eval_cases.jsonl`; recorded runs: `scripts/prompt_eval_runs/`.

- `--selftest` / `--score` / `--dry-run` / `--compare` are offline (no model
  calls). `--record` spends one agent turn per case (`--repeat N`: N turns),
  so run it deliberately.
- A run that read the bait set's scoring keys or named its own case is
  **contaminated** (NOT MEASURABLE — never search the bait set from inside a
  recorded run); a run without a final answer is **incomplete** and excluded.

## cleanup_test_files.sh

One-off helper that removes temporary test files (e.g. `test_*.rs`) left in the
repo root; details in the script header.

## test_memory_feature.sh

Quick smoke test for memory / knowledge-base retrieval (compile check plus
`/memory` save/recent/search round-trips via `cargo run --bin a`); details in
the script header.
