# Resolve processing versions

A non-terminal task is identified by `(repo_id, original_mr_id, source_commit_sha, target_commit_sha)`.
The database `active_key` is `v1:` followed by the SHA-256 hex digest of the colon-separated tuple.
Terminal transitions, including failure and cancellation, release this key. The same commits may then start another task.

`handle_trigger` reads `SourceBranch.Commit.Id` and `TargetBranch.Commit.Id` from `GetMergeRequest`
with `Selector.Branch=true`, and uses that current commit pair to check for an existing task and capture conflict files.
Missing branch HEADs are never replaced with historical MR version SHAs.
The MR version number is retained for provenance; its historical source and target SHAs do not define the processing baseline.
In particular, a target branch push can create a new processing version without creating a new MR version.
Codebase returns the live conflict snapshot ID as `source_sha..target_sha`. Before reading file contents, the resolver requires
that ID to match the captured HEADs, including for an empty conflict list. Unknown snapshot ID formats fail closed.
Once the conflict snapshot matches, the input is frozen; later source or target branch movement does not invalidate it.
There is no additional HEAD or MR version readback before enqueueing. A mismatched snapshot causes both HEADs to be recaptured,
up to three attempts; if no matching snapshot is available, the trigger fails without creating a task.
These reads are not an upstream atomic snapshot API.
The accepted task persists its SHAs, MR version, author, and conflict-file snapshot in the same insert that claims the unique key.
Concurrent duplicate inserts recover the existing task ID. Webhook delivery and per-payload comment deduplication are unchanged.

Workers build their resolver branch from the persisted source SHA and parse the persisted conflict files.
A new source or target HEAD can create another task without cancelling, modifying, or replacing the previous task.
Each task retains its own resolver branch, Coco session, result, resolver MR, report, and approvals.
First-pass and retry validation still check the resolver HEAD, commit parents, resolved file contents, tests, and attribution against frozen inputs.
They no longer require the original source and target branches to remain at their captured HEADs.

If the live target has advanced, the old task can publish its review artifacts even when its resolver MR is no longer mergeable.
This case records `RESOLVE_MR_TARGET_ADVANCED`. Reports and approval notifications describe the frozen version;
users must consult the MR's current checks before merging. Approval does not merge the MR automatically.

## Cancellation

- On a task's resolver MR, `/resolve cancel` cancels that task.
- On the original MR, `/resolve cancel <task_id>` selects a specific non-terminal task.
- Without a task ID, the original trigger thread selects its task when unambiguous; otherwise, a single active task can be selected.
- When multiple tasks still match, the reply lists their IDs and commit pairs and asks for an explicit ID. No task is cancelled.
- The task must belong to that repository and MR. Cancellation remains restricted to its `/resolve` trigger author.

Cancellation preserves branches and MRs. It stops subsequent Conciliator progression, and late Coco completion polling skips cancelled tasks.
It does not send a termination request to an already running Coco session; external work already in progress may finish.

## Rollout

Apply `0006_processing_version` (or its generated incremental SQL) before starting the updated application.
Quiesce webhook processing and workers during migration; do not mix old MR-only deduplication code with the new version keys.
The migration converts existing active keys whose two SHAs are already populated and adds nullable `conflict_files_snapshot`.
It does not end existing tasks.

Existing tasks that already persisted their conflict blocks can continue using those blocks.
For older queued tasks without frozen SHAs, the compatibility path freezes them when preparation runs.
A legacy task lacking a conflict snapshot can only capture live conflicts if its frozen SHAs still match the current MR.
If that baseline is already outdated, the original content cannot be reconstructed from the existing task record, so it fails explicitly.
New tasks always capture their inputs before enqueueing.

Downgrade restores one active key per MR and will fail on the unique constraint if multiple versions remain active.
Finish or explicitly cancel extra active versions before downgrading.

## Validation

`tests/unit/test_processing_version.py` exercises concurrent duplicate inserts through the real repository on SQLite,
parallel versions, terminal key release, capture races, immutable queued inputs, retry baseline checks, stale-target publication,
and cancellation isolation. The MySQL repository integration suite remains required before rollout;
SQLite tests and offline MySQL DDL rendering are not substitutes for that suite or a live NextCode/Coco flow.
