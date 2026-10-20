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
- Objective signals: fabricated citations (path missing on disk), citations whose
  file was never successfully read (`strict_citations` cases), `internal_note` gate
  markers. Read evidence is `read_file` only — a citation backed solely by
  `execute_command` output stays soft-unsupported — and only asserted paths count,
  so a correct refusal (or a hypothetical offer) never scores as a fabricated
  citation. Labelled heuristics (hedge markers, missing uncertainty markers,
  unanswered controls) are never hard failures; the action-claim hard policy is
  retained — it stays labelled heuristic, not objective proof.
- Controls default to citation presence; `response_mode="text"` allows
  non-citation responses, and optional `required_tool={name,arguments}` checks an
  exact matching attempt separately from success. Neither shape check proves
  correctness.
- One hard failure makes a case violating, and `--score --fail-on-violation` still
  fails on any hard violation; `UNSTABLE` marks mixed hard-policy replicas, and
  judgment/recall/claim/control cases stay semantic-review-required — correctness
  is unknown, so review directness, reasons and useful execution manually. Exit 0
  does not clear review; re-score exported transcripts rather than mixing old scores.
- A recorded run's `manifest.json` fingerprint (sha256 over
  `src/bin/ai/driver/system_prompts/*.md`, with `git rev-parse HEAD` recorded
  separately) describes source state, not the prompt the runtime actually sent:
  A/B attribution needs runtime prompt provenance and matched cases, replicas and
  resolved model/agent settings. These checks alone do not prove causality or
  semantic improvement, and synthetic selftests validate scoring, not model quality.

## cleanup_test_files.sh

One-off helper that removes temporary test files (e.g. `test_*.rs`) left in the
repo root; details in the script header.

## test_memory_feature.sh

Quick smoke test for memory / knowledge-base retrieval (compile check plus
`knowledge_save`/`knowledge_search` round-trips); details in the script header.