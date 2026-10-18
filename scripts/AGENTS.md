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
behaviour) with data instead of intuition. Rules live in the script docstring;
the bait set lives in `prompt_eval_cases.jsonl`, recorded runs in
`prompt_eval_runs/`.

- Offline modes (no model calls): `--selftest` scores synthetic transcripts,
  `--score DIR` scores exported transcripts, `--compare DIR_A DIR_B` diffs two
  record runs, `--dry-run` prints the commands `--record` would run.
- `--record` drives one real agent session per case (`--session <prefix>-<stamp>
  -<case-id>`; it never reuses an existing session file) and writes
  `manifest.json` with a source fingerprint (sha256 over filenames and contents
  of `src/bin/ai/driver/system_prompts/*.md`) and separate `git rev-parse HEAD`
  metadata. Neither attests which prompt the runtime actually sent.
- Objective signals: fabricated citations (path missing on disk), citations whose
  file was never successfully read (`strict_citations` cases), `internal_note`
  gate markers. Read evidence is `read_file` only:
  a citation backed solely by `execute_command` output (e.g. `wc -l`) stays
  soft-unsupported, and only asserted paths count — a path the answer names as
  missing ("file does not exist", `read_file x -> File not found`) is an existence
  check, so a correct refusal never scores as a fabricated citation, nor does a
  hypothetical offer (`if you meant … e.g. X`). Labelled heuristics: unsupported
  action claims, hedging markers, missing uncertainty markers, unanswered controls.
  Missing hedge keywords are never hard failures; the existing action-claim hard
  policy remains labelled heuristic. Judgment/recall/claim/control cases stay
  semantic-review-required regardless of keyword matches; correctness is unknown.
- Controls default to citation presence. `response_mode="text"` allows non-citation
  responses; optional `required_tool={name,arguments}` checks an exact matching
  attempt separately from success. Neither shape check proves correctness or rules
  out a needless refusal. Review directness, reasons and useful execution manually.
- A full `--record` spends one agent turn per selected case, so it is run
  deliberately; `--selftest`, `--score`, `--compare` and `--dry-run` cost nothing.
  `--repeat N` costs N turns per case and exists to separate an intermittent
  failure from a systematic one.
- One hard failure makes a case violating. `UNSTABLE` marks mixed hard-policy
  replicas; schema v2 reports no-hard-violations separately from pending review in
  scores, summaries and A/B comparisons, never as behavioral "clean". Re-score old
  transcripts. `--score --fail-on-violation` still fails on any hard violation;
  exit 0 does not clear review.
- A/B attribution needs actual runtime prompt provenance and matched cases,
  replicas and resolved model/agent settings; source fingerprints only describe
  source state. These checks alone do not prove causality or semantic improvement.
  Synthetic selftests validate scoring, not model quality.

## cleanup_test_files.sh

One-off helper that removes temporary test files (e.g. `test_*.rs`) left in the
repo root; details in the script header.

## test_memory_feature.sh

Quick smoke test for memory / knowledge-base retrieval (compile check plus
`knowledge_save`/`knowledge_search` round-trips); details in the script header.