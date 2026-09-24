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
  `manifest.json` holding a prompt fingerprint (sha256 over
  `src/bin/ai/driver/system_prompts/*.md` plus `git rev-parse HEAD`).
- Objective signals: fabricated citations (path missing on disk), citations whose
  file was never successfully read (`strict_citations` cases), `internal_note`
  gate markers, and claim-vs-tool-call pairing. Read evidence is `read_file` only:
  a citation backed solely by `execute_command` output (e.g. `wc -l`) stays
  soft-unsupported, and only asserted paths count — a path the answer names as
  missing ("file does not exist", `read_file x -> File not found`) is an existence
  check, so a correct refusal never scores as a fabricated citation, nor does a
  hypothetical offer (`if you meant … e.g. X`). Labelled heuristics: unsupported
  action claims, hedging markers, unanswered controls.
- A full `--record` spends one agent turn per case (12 by default), so it is run
  deliberately; `--selftest`, `--score`, `--compare` and `--dry-run` cost nothing.
  `--repeat N` costs N turns per case and exists to separate an intermittent
  failure from a systematic one.
- A case is clean only when every run of it is clean: a bait checkable with one
  tool call failing once is a defect, not noise. `UNSTABLE` marks mixed replicas,
  and `--score --fail-on-violation` exits non-zero on any hard violation.
- An A/B difference is attributable to the prompt only when both manifests differ
  in prompt fingerprint and share model/agent; a change that removes no violation
  but adds none is "not worse", not "better".