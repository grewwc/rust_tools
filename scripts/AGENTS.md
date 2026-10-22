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

## prompt_eval.py

Opt-in prompt-behaviour harness for the `a` agent: it exists to test claims about
anti-hallucination prompt emphasis (for example whether ALL-CAPS keywords change
behaviour) with data instead of intuition. The script header
(`scripts/prompt_eval.py`) is the documentation home for its rules — scoring
signals, controls, replica and failure policy. The bait set lives in
`prompt_eval_cases.jsonl`, recorded runs in `prompt_eval_runs/`.

- Offline modes (no model calls): `--selftest` (scores synthetic transcripts),
  `--score DIR` (exported transcripts), `--dry-run` (the commands `--record` would
  run), and `--compare DIR_A DIR_B` (per-case verdict diff between two record runs).
  `--record` drives one real agent session per case and spends one agent turn per
  case, so run it deliberately; `--repeat N` spends N turns per case to tell an
  intermittent failure from a systematic one.
  `--score --fail-on-violation` exits 2 on any hard violation.
- Run quality is reported beside the verdict: a run that read the bait set's
  scoring keys or named its own case is **contaminated** and reported as NOT
  MEASURABLE (never search `prompt_eval_cases.jsonl` from inside a recorded
  run); a run that ended without a final answer is **incomplete** and excluded
  from the verdict. Signals, dimensions, controls and the attribution model:
  the script header.

## cleanup_test_files.sh

One-off helper that removes temporary test files (e.g. `test_*.rs`) left in the
repo root; details in the script header.

## test_memory_feature.sh

Quick smoke test for memory / knowledge-base retrieval (compile check plus
`/memory` save/recent/search round-trips via `cargo run --bin a`); details in
the script header.
