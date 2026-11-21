# scripts/ AGENTS.md

Repo shell/Python scripts. Keep each script self-contained and documented in
its own header.

## postprocess_terminal.py

Display-only filter for the agent's terminal body text; rules live in the
script header (`scripts/postprocess_terminal.py`).

Runtime integration belongs to **Output post-processing is display-only** in
`src/bin/ai/AGENTS.md`. Keep the filter's rendered output consistent between live
turns and `/history last` replay.

## prompt_eval.py

Opt-in prompt-behaviour harness for the `a` agent; rules live in the script
header (`scripts/prompt_eval.py`). Bait set:
`scripts/prompt_eval_cases.jsonl`; recorded runs: `scripts/prompt_eval_runs/`.

- `--selftest` / `--score` / `--dry-run` / `--compare` are offline (no model
  calls). `--record` replays each case's prompts (one agent turn per prompt;
  multi-turn cases spend more), and `--repeat N` runs every case N times. The
  current bait set of 41 cases needs 44 agent turns per round, so run it
  deliberately.
- A run that read the bait set's scoring keys or named its own case is
  **contaminated** (NOT MEASURABLE — never search the bait set from inside a
  recorded run); only a nonzero exit code (timeout or crash) marks a run
  **incomplete** and excluded — an empty transcript with exit 0 is still scored.

## cleanup_test_files.sh

One-off historical helper: it removes a hardcoded list of test sources/binaries
and moves `test_memory_feature.sh` into `scripts/`, under a hardcoded
`/Users/bytedance/rust_tools` path. None of those files exist at the repo root
today, so the move fails under `set -e`.

## test_memory_feature.sh

Opt-in smoke script for memory / knowledge-base retrieval: runs `cargo check
--bin a`, then exercises save/recent/search via the `/memory` CLI command
(stale — the runtime command is now `/memo`) and prints retrieval config hints.
It hardcodes `cd /Users/bytedance/rust_tools`, and its header documents no usage.
