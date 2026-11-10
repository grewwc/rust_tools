# Deep audit: only changes owned by the main agent for this request

## When to use

Apply this private procedure only when the user selects `/audit --deep` (optionally followed by `--fast`).

## Procedure

### 1. Build a self-contained audit prompt

**R0 — mechanical inventory, before building the prompt:** collect the union of the session mutation log (`show_changes`), `git status --porcelain`, and `git diff --stat`. Maintain a lead-owned checklist filtered by ownership for this request, including delegated writes. Mark every item `covered` (cited read range), `gap` (not yet read), or `N.A.` (out of scope, with reason); unknown ownership remains an explicit `gap`. This checklist is not a new report field.

`show_changes` falls back to `git status --porcelain` / `git diff HEAD` when the log is empty; use `include_git: false` for a session-only view when needed. A nonempty mutation log never skips the git cross-check: it covers tool-level writes (`write_file` / `apply_patch` and the patch tools), including parallel subagent writes, but not indirect writes from `execute_command`. Compression drops earlier turns, so memory alone under-reports scope.

Create an `audit_prompt` from the changes you made for this request. It must include:

- **Scope (most important):** Start by stating that the audit covers only changes the main agent owns for this request, including its delegated writes. List the changed files. Tell the auditor not to audit or report other workspace changes, concurrent work, or pre-existing issues unless they directly affect the correctness of these changes.
- **Change summary:** For each changed file, state what changed and why. This lets the auditor distinguish your intent from unrelated diffs.
- **Changed contracts and verification focus:** State user-authorized changes, pre/post behavior, newly required premises, preserved guarantees, known consumers, and remaining uncertainties. Distinguish evidence from hypotheses; the auditor independently checks the baseline. For producer/consumer boundaries, name existing paired assertions and any coverage gap. Limit checks to affected acceptance criteria under the shared impact/invariant and task-convergence rules.
- **Verification already run:** State the exact commands and outcomes (build, focused tests, manual checks) or say that none were run. The auditor is told not to repeat the caller's commands, so it needs this to distinguish unverified behavior from behavior you already exercised.
- **Evidence read list (required):** For each changed file, name the ranges the auditor must read (`path:start-end`, taken from your own `read_file` calls) and the invariant each range checks. Include the caller/dependent ranges that prove the change's impact surface: the audit report requires caller/contract evidence, and the auditor starts from this list and completes the baseline itself with bounded `git`/`rg` reads. The auditor may only cite evidence from its own `read_file` calls in its turn, so this list is what lets it verify quickly. Never list a range you did not read; name it as a coverage gap instead. Include the scope's R0 items and gaps; keep one list per audit scope — see the scoped parallel path in §2.

A task subagent cannot see your conversation history. The `/audit` path adds this session's change context, but the `task_spawn` path receives only `audit_prompt`; therefore, `audit_prompt` must stand alone.

### 2. Prepare the scoped audit round

Prepare one self-contained scope for a small change or 2-4 scopes for a large change set or independent subsystems, split by invariant/consumer boundary. All children on one snapshot form one round. The command driver launches the scopes through the foreground audit runner; return its requested `<audit_plan>` rather than spawning children while planning. Preserve default inheritance, the existing `<audit_report>` protocol, and each child's progress protocol. Collect and integrate every delivered result using its full task ID.

### 3. Verify findings and take ownership

After the audit subagent returns, continue the work rather than forwarding its result directly to the user:

- Verify every reported issue against the code and direct evidence (`read_file`, `execute_command`, `git diff`, or equivalent).
- Fix confirmed issues when already authorized by the task; explain why any false positive is not actionable. If a fix exceeds existing authorization (high cost, high impact, irreversible action, or broad side effects), supply evidence, options, impact, reversibility, and estimated cost; the **user** decides fix / narrow / accept / rollback. Confidence after multiple audit rounds is not authorization.
- Resolve `open_questions` under the same task-convergence rule as concerns you found yourself. Check the missing premise, record the closure evidence or specific blocker, and keep any affected acceptance criterion unresolved until established. An audit adds evidence; it does not transfer responsibility or certify completion.
- Re-run appropriate verification after a fix, confirm the work remains within your change scope, and apply the disposition gate below.

Run one main round. A second round requires a recorded trigger:

| Trigger | Evidence required | Narrow follow-up |
| --- | --- | --- |
| T1 | A concrete `coverage_gap` or `open_question` remains unclosed. | Check the named missing range or premise. |
| T2 | A post-audit mutation changes behavior or a contract. | Review the new diff and affected consumer; re-read citations after line drift. Record comment/doc-only edits that leave behavior/contracts unchanged without re-audit. |
| T3 | A mechanical high-impact predicate hits (persistence/schema/shared-contract/auth/security/concurrency) and round one has no P0/P1 findings. | Falsify the high-impact guarantee through another consumer, scope, or tier; use full `audit` when required. |

Allow at most one narrow follow-up per trigger; combine applicable follow-ups into round two, with **at most two rounds total**. The second round must change axis (new scope/diff/consumer, falsification task, or different tier); never re-run the same prompt. After round two, report unresolved items and stop auditing.

**Final disposition gate:** refresh R0 after subsequent writes; maintain a lead-owned table covering every inventory item, `coverage_gap`, `open_question`, and P0/P1 finding, each marked `verified`, `refuted`, `accepted-with-reason`, or `fixed`, with evidence or reason. Exit with a closed audit only when every item has a disposition, no unreviewed source mutation remains since the last round, and `coverage_gaps` are empty or explicitly accepted. Otherwise report the open items and stop. A completed bounded audit report with gaps is not completed task acceptance; keep affected acceptance criteria unresolved until established under the task-convergence rule.

- **Timeout or failure: recover the partial report.** A `/audit` command run cut short by its foreground timeout returns a `SUBAGENT_TIMEOUT_RECOVERY_V1` result naming the preserved child history and, when the auditor followed the progress protocol, its latest `AUDIT_CHECKPOINT:` records. A `task_spawn` child reaped at the wall-clock cap instead reports a timeout terminal state; treat that and any preserved child history/checkpoints as partial evidence. Verify each recovered record yourself, keep what holds, and state which coverage the audit never reached. Resume from the recovered checkpoints instead of repeating the same scope; use a narrower follow-up audit to close a named gap within the two-round limit, otherwise report the gap and stop. Never present an unrecovered checkpoint as a verified conclusion.

## Constraints

- Address only issues related to this request's changes. Do not attribute others' changes or pre-existing issues to your work.
- Treat subagent output as evidence, not authority: a clean audit is not sufficient proof, and a reported issue is not automatically correct.
- Audit confidence does not expand authorization. Proceed with already authorized fixes; for fixes beyond it, give the user evidence, options, impact, reversibility, and estimated cost for their fix / narrow / accept / rollback decision.
- Never use a vague audit prompt such as "review my changes". Include the changed-file list and concrete verification goals so the auditor can distinguish your changes from unrelated work.
- Select review by error consequences, reversibility, affected scope, and decisive evidence gaps, not task type, file extension, diff size, or producer confidence. Ordinary low-risk agent-team deliveries need no extra audit when parent acceptance closes the evidence; `audit-fast` may target an unresolved low-risk review criterion when no full audit is required. Safety, persistence, concurrency, security, critical shared-code contracts, or prompt-contract changes always get a full `audit`, split by scope when useful. Never downgrade an audit the user asked for; existing sufficient checks need not be repeated.


## Command continuation

Follow this private deep-audit procedure for verification and closure. Verify every finding against current files and direct evidence; a clean audit is not proof. Close `open_questions` with evidence or a specific blocker, keeping affected acceptance criteria unresolved until established. Proceed with confirmed fixes already authorized by the task and run their narrowest focused checks. Audit confidence is not authorization: for a fix beyond existing authorization, prepare evidence, options, impact, reversibility, and estimated cost for the user's fix / narrow / accept / rollback decision, then wait before that action while completing authorized preparation and independent work.

Run one main round; a second round requires a recorded trigger: T1, a concrete `coverage_gap` or `open_question` remains unclosed (check the named missing range or premise); T2, a post-audit mutation changes behavior or a contract (review the new diff and affected consumer, re-reading citations after line drift; record comment/doc-only edits with unchanged behavior/contracts without re-audit); T3, a mechanical high-impact predicate hits (persistence/schema/shared-contract/auth/security/concurrency) and round one has no P0/P1 findings (falsify that guarantee through another consumer, scope, or tier). Allow at most one narrow follow-up per trigger and combine them into round two, with at most two rounds total. Round two must change axis: new scope/diff/consumer, a falsification task, or a different tier; use full `audit` when required and never repeat the same prompt. After round two, report unresolved items and stop auditing.

Refresh R0 after writes. Keep a lead-owned disposition table for every inventory item, `coverage_gap`, `open_question`, and P0/P1 finding: `verified`, `refuted`, `accepted-with-reason`, or `fixed`, with evidence or reason. Close the audit only when every item has a disposition, no unreviewed source mutation remains since the last round, and `coverage_gaps` are empty or explicitly accepted. Otherwise report open items and stop; bounded audit completion does not close unresolved task acceptance. Timeout/failure reports and recovered checkpoints are partial evidence: verify recovered records, close named gaps within the two-round limit, and never present unverified recovery as a conclusion.
