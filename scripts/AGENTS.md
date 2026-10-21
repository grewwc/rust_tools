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
- Objective signals: fabricated citations (path missing on disk), citations the
  session never observed at all (`strict_citations` cases), forbidden tool calls
  and forbidden command patterns (`forbid_tools` / `forbid_command_patterns`),
  `internal_note` gate markers. A path counts as observed when it was read
  successfully, appeared in tool output, or was named by a tool call; a path only
  seen in tool output is the softer `unread_citation`, never a hard failure. The
  strong read signal stays `read_file`-only — shell output never upgrades to a
  successful read. Only asserted paths count, so a correct refusal (or a
  hypothetical offer) never scores as a fabricated citation. Labelled heuristics
  (hedge markers, missing uncertainty markers, unanswered controls, `must_contain`
  / `must_match` / `must_not_assert` misses) are never hard failures unless a case
  opts into `must_contain_hard`; the action-claim hard policy is retained — it
  stays labelled heuristic, not objective proof.
- Cases carry a `dimension` (citation, cause, unknown, correction, recall, action,
  comparison, control) and the summary reports one row per dimension, so a weak
  hallucination axis shows up as a weak axis instead of being averaged away.
  A `turns` array replays several prompts inside one session, which is how a
  correction that contradicts what the agent just read gets tested.
- Controls default to citation presence; `response_mode="text"` allows
  non-citation responses, and optional `required_tool={name,arguments}` checks an
  exact matching attempt separately from success. Neither shape check proves
  correctness.
- One hard failure makes a case violating, and `--score --fail-on-violation` still
  fails on any hard violation; `UNSTABLE` marks mixed hard-policy replicas, and
  judgment/recall/claim/control cases stay semantic-review-required — correctness
  is unknown, so review directness, reasons and useful execution manually. Exit 0
  does not clear review; re-score exported transcripts rather than mixing old scores.
- Run quality is reported beside the verdict: a run that read the bait set's
  scoring keys or named its own case is **contaminated** (the bait set lives in
  this repository, so an agent that searches it can read the answers), and a run
  that ended without a final answer is **incomplete**. Neither counts as a pass
  or a failure: contaminated and incomplete runs are excluded from the case
  verdict and reported as NOT MEASURABLE with their reason, and an incomplete
  run's behavioural heuristics are suppressed because a cutoff is not a behaviour.
  A bare file name in a `git status` line is not contamination.
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