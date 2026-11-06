use super::quiet::TaskWaitReturnReason;
use super::*;

pub(super) static TASK_WAIT_STATES: LazyLock<Mutex<FxHashMap<TaskWaitKey, TaskWaitState>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

/// Consecutive `task_wait` calls that found every referenced task already delivered (the
/// "all completed" no-op path). Once results are delivered, waiting is over: PARKED /
/// BUDGET-ELAPSED legitimately ask the model to re-call `task_wait` to keep waiting, and a stuck
/// model can wrongly extend that instruction to this terminal case (observed: an 11-round wait
/// loop on an already-delivered task). After `TASK_WAIT_NOOP_ERROR_THRESHOLD` consecutive no-op
/// calls for the same key, the tool escalates the soft hint into an error so the loop is broken
/// by a distinct signal. Any real wait (pending tasks still running) resets the count for that
/// key.
pub(super) const TASK_WAIT_NOOP_ERROR_THRESHOLD: u32 = 3;

pub(super) static TASK_WAIT_NOOP_COUNTS: LazyLock<Mutex<FxHashMap<TaskWaitKey, u32>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

pub(super) fn bump_task_wait_noop_count(key: &TaskWaitKey) -> u32 {
    let mut counts = lock_unpoisoned(&TASK_WAIT_NOOP_COUNTS);
    let count = counts.entry(key.clone()).or_insert(0);
    *count += 1;
    *count
}

pub(super) fn reset_task_wait_noop_count(key: &TaskWaitKey) {
    lock_unpoisoned(&TASK_WAIT_NOOP_COUNTS).remove(key);
}

pub(super) const OUTSTANDING_SUBAGENT_TASKS_NOTE_PREFIX: &str = "[pending-subagent-tasks]";

/// Task id list produced by the most recent successful `task_spawn`, used to
/// detect the "lone spawn" anti-pattern: spawning a single task and immediately `task_wait`-ing to
/// collect it. That scenario gains no concurrency and should use the synchronous `task` tool
/// (spawn + wait is only slower). The hint is light normative guidance: it fires once, never
/// rejects or blocks, and the model may ignore it (e.g. when it really did interleave parent-side
/// work between spawn and wait).
pub(super) struct LastSpawnBatch {
    task_ids: Vec<String>,
    hinted: bool,
}
pub(super) static LAST_SPAWN_BATCH: LazyLock<Mutex<Option<LastSpawnBatch>>> =
    LazyLock::new(|| Mutex::new(None));

pub(super) fn record_last_spawn_batch(task_ids: Vec<String>) {
    *lock_unpoisoned(&LAST_SPAWN_BATCH) = Some(LastSpawnBatch {
        task_ids,
        hinted: false,
    });
}

/// If this wait's task_ids match "the most recent spawn was a single task" and the hint has not
/// been shown yet, return the normative hint text once (consuming the hinted flag so it fires at
/// most once per whole session turn).
pub(super) fn lone_spawn_hint_note(waited_task_ids: &[String]) -> Option<String> {
    let mut guard = match LAST_SPAWN_BATCH.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let record = guard.as_mut()?;
    if record.hinted || record.task_ids.len() != 1 {
        return None;
    }
    let sole_id = &record.task_ids[0];
    if !waited_task_ids.contains(sole_id) {
        return None;
    }
    record.hinted = true;
    Some(
        "[tool_followup:lone_spawn]\n\
         This task_wait collects the only task spawned by the most recent task_spawn call. \
         Without parent-side work in between, task_spawn + task_wait adds no concurrency over the \
         synchronous `task` tool. Keep the asynchronous form when an orchestration policy requires \
         it or its immediate spawn status matters; otherwise prefer synchronous `task` for \
         one-shot handoffs. \
         (If you intentionally ran parent-side work between spawn and wait, ignore this hint.)"
            .to_string(),
    )
}

#[cfg(test)]
pub(crate) fn reset_last_spawn_batch_for_test() {
    *lock_unpoisoned(&LAST_SPAWN_BATCH) = None;
}

pub(super) fn load_or_create_task_wait_state(
    key: &TaskWaitKey,
    timeout_secs: u64,
    agent_team_mode: bool,
) -> TaskWaitState {
    let now = Instant::now();
    let mut states = lock_unpoisoned(&TASK_WAIT_STATES);
    let mut inserted = false;
    let state = states.entry(key.clone()).or_insert_with(|| {
        inserted = true;
        TaskWaitState {
            deadline: now + Duration::from_secs(timeout_secs),
            timeout_secs,
            expired: false,
            agent_team_mode,
        }
    });
    if state.agent_team_mode != agent_team_mode {
        *state = TaskWaitState {
            deadline: now + Duration::from_secs(timeout_secs),
            timeout_secs,
            expired: false,
            agent_team_mode,
        };
        inserted = true;
    }
    if now >= state.deadline {
        state.expired = true;
    }
    let state = *state;
    drop(states);
    if inserted {
        crate::ai::driver::notify_scheduler_after(Duration::from_secs(timeout_secs));
    }
    state
}

pub(super) fn clear_task_wait_state(key: &TaskWaitKey) {
    let mut states = lock_unpoisoned(&TASK_WAIT_STATES);
    states.remove(key);
}

/// Time remaining until the scheduler must next check task_wait wall-clock deadlines.
///
/// A delayed notify is only an early wake-up signal and cannot be the sole source of truth for
/// the deadline: if the notification task was not registered successfully or a notify race
/// occurs, the scheduler must still wake itself up at the real deadline returned here.
pub(crate) fn next_task_wait_wakeup_delay() -> Option<Duration> {
    let now = Instant::now();
    lock_unpoisoned(&TASK_WAIT_STATES)
        .values()
        .filter(|state| !state.expired)
        .map(|state| state.deadline.saturating_duration_since(now))
        .min()
}

#[cfg(test)]
pub(crate) fn expire_task_wait_states_for_test() {
    let mut states = lock_unpoisoned(&TASK_WAIT_STATES);
    let expired_at = Instant::now() - Duration::from_secs(1);
    for state in states.values_mut() {
        state.deadline = expired_at;
        state.expired = false;
    }
}

#[cfg(test)]
pub(crate) fn task_wait_state_count_for_test() -> usize {
    lock_unpoisoned(&TASK_WAIT_STATES).len()
}

#[cfg(test)]
pub(crate) fn reset_task_wait_noop_counts_for_test() {
    lock_unpoisoned(&TASK_WAIT_NOOP_COUNTS).clear();
}

pub(crate) fn wake_expired_task_waits() {
    let expired = {
        let now = Instant::now();
        let mut states = lock_unpoisoned(&TASK_WAIT_STATES);
        states
            .iter_mut()
            .filter_map(|(key, state)| {
                if state.expired || now < state.deadline {
                    return None;
                }
                state.expired = true;
                Some((key.clone(), *state))
            })
            .collect::<Vec<_>>()
    };
    if expired.is_empty() {
        return;
    }

    // Wake each expired owner one by one; meanwhile record the wait keys whose owner process no
    // longer exists or has terminated. wake_process only returns an owner in the Waiting state to
    // Ready (when not Waiting the owner is already scheduled, so a lost wake-up is harmless), but
    // once the owner is Terminated the wake-up misses entirely and its wait state stays in the
    // global table forever with expired=true — leaking memory and leaving no cleanup path beyond
    // next_task_wait_wakeup_delay. These orphan keys are collected here and then deleted in bulk
    // to self-heal.
    let orphaned = with_os_kernel(|os| {
        let mut woken: SkipSet<u64> = SkipSet::default();
        let mut orphaned: Vec<TaskWaitKey> = Vec::new();
        for (key, state) in expired {
            let owner_alive = os
                .get_process(key.owner_pid)
                .is_some_and(|proc| !matches!(proc.state, ProcessState::Terminated));
            if !owner_alive {
                orphaned.push(key);
                continue;
            }
            if !woken.insert(key.owner_pid) {
                continue;
            }
            let message = if state.agent_team_mode {
                // The deadline is the original wait instant plus its budget. Include any
                // scheduler delay in this wake notice, using only monotonic durations.
                let waited = Duration::from_secs(state.timeout_secs)
                    .saturating_add(Instant::now().saturating_duration_since(state.deadline));
                super::quiet::format_agent_team_task_wait(
                    &key.task_ids,
                    TaskWaitReturnReason::BudgetElapsed,
                    waited,
                )
            } else {
                let task_ids = key.task_ids.join(", ");
                let mut message = format!(
                    "[TASK_WAIT_TIMEOUT]\nWall-clock task_wait budget elapsed after {}s. Re-call `task_wait` with the same task_ids to collect any ready results and receive the budget-elapsed status. task_ids=[{}]",
                    state.timeout_secs, task_ids
                );
                append_task_progress_snapshots(os, &mut message, &key.task_ids);
                append_pending_task_side_note_summaries(&mut message, &key.task_ids);
                message
            };
            let _ = os.wake_process(key.owner_pid, message);
        }
        Ok(orphaned)
    })
    .unwrap_or_default();

    if !orphaned.is_empty() {
        let mut states = lock_unpoisoned(&TASK_WAIT_STATES);
        for key in orphaned {
            states.remove(&key);
        }
    }
}
pub(super) fn parse_task_wait_options(
    args: &Value,
    agent_team_mode: bool,
) -> Result<(u64, WaitPolicy), String> {
    let max_timeout_secs = if agent_team_mode {
        MAX_AGENT_TEAM_TASK_WAIT_TIMEOUT_SECS
    } else {
        MAX_TASK_WAIT_TIMEOUT_SECS
    };
    let timeout_secs = args
        .get("timeout_secs")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_TASK_WAIT_TIMEOUT_SECS)
        .clamp(1, max_timeout_secs);
    let wait_policy = match args.get("wait_policy").and_then(Value::as_str) {
        Some("any") | None => WaitPolicy::Any,
        Some("all") => WaitPolicy::All,
        Some(other) => {
            return Err(format!(
                "Unknown wait_policy: {} (expected 'any' or 'all')",
                other
            ));
        }
    };
    Ok((timeout_secs, wait_policy))
}

pub(crate) fn execute_task_wait(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_wait")?;
    let current_session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let requested_task_ids = args["task_ids"]
        .as_array()
        .ok_or("Missing 'task_ids' array parameter")?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<Vec<_>>();

    if requested_task_ids.is_empty() {
        return Err("task_ids array cannot be empty".to_string());
    }

    // Wait budget for this single task_wait call. See the DEFAULT_TASK_WAIT_TIMEOUT_SECS comment —
    // a timeout only means this call did not get the result; the subagent keeps running and no
    // resources are released.
    let agent_team_mode = is_agent_team_mode();
    let (timeout_secs, wait_policy) = parse_task_wait_options(args, agent_team_mode)?;

    // wait_policy: "any" | "all", default "any", so the foreground is not held up by the slowest
    // task.
    // - all  — return only after every pending task completes (for when results must be
    //          aggregated);
    // - any  — return as soon as any pending task completes, the rest keep running and can be
    //          collected by further task_wait calls (for fan-out where results are gathered as
    //          they arrive).
    let current_owner_pid = current_task_owner_pid()?;
    let registry = lock_task_registry();
    let mut foreign_session_task_ids = Vec::new();
    let mut foreign_owner_task_ids = Vec::new();
    let mut missing_task_ids = Vec::new();
    let mut task_ids_filtered = Vec::new();
    for tid in &requested_task_ids {
        match registry.get_ref(&tid) {
            Some(entry) if task_entry_owned_by(entry, &current_session_id, current_owner_pid) => {
                task_ids_filtered.push(tid.clone());
            }
            Some(entry) if entry.session_id == current_session_id => {
                foreign_owner_task_ids.push(tid.clone());
            }
            Some(_) => foreign_session_task_ids.push(tid.clone()),
            None => missing_task_ids.push(tid.clone()),
        }
    }
    if !foreign_session_task_ids.is_empty() {
        return Err(format!(
            "Refusing to wait on task_id(s) owned by another session: {}",
            foreign_session_task_ids.join(", ")
        ));
    }
    if !foreign_owner_task_ids.is_empty() {
        return Err(format!(
            "Refusing to wait on task_id(s) not owned by current process pid={}: {}",
            current_owner_pid,
            foreign_owner_task_ids.join(", ")
        ));
    }
    // Snapshot the validated entries and release the registry lock before any kernel call. The
    // kernel closure must never run under the registry lock: the foreground status line takes the
    // kernel first, and holding registry -> kernel here would create a lock cycle. Removal happens
    // after the kernel closure returns, so a failure on a later task cannot strand entries that
    // were already collected.
    let entries = task_ids_filtered
        .iter()
        .map(|tid| {
            (
                tid.clone(),
                registry.get_ref(tid).expect("validated task id").clone(),
            )
        })
        .collect::<FxHashMap<_, _>>();
    drop(registry);
    // A registry miss can mean either the task was normally cleaned up after delivery, or the
    // model mistyped an id, referenced a stale cross-process id, or hit a registry anomaly. Only a
    // tombstone actually present in the durable evidence ledger proves the task was delivered.
    let mut already_delivered = Vec::new();
    let mut unknown_task_ids = Vec::new();
    if !missing_task_ids.is_empty() {
        let context = crate::ai::driver::runtime_ctx::try_current()
            .ok_or("task_wait cannot classify missing task_ids outside an active driver turn")?;
        for task_id in missing_task_ids {
            let delivered = crate::ai::history::task_evidence_exists(
                context.app_proto.config.history_file.as_path(),
                &current_session_id,
                &task_id,
            )
            .map_err(|error| {
                format!("failed to inspect durable task evidence for {task_id}: {error}")
            })?;
            if delivered {
                already_delivered.push(task_id);
            } else {
                unknown_task_ids.push(task_id);
            }
        }
    }
    if !unknown_task_ids.is_empty() {
        return Err(format!(
            "Unknown task_id(s): {}. These ids are neither active in the current session/process \
             nor present in its durable delivered-task ledger.",
            unknown_task_ids.join(", ")
        ));
    }
    // Mixing already-delivered ids with still-pending ones is expected input: PARKED /
    // BUDGET-ELAPSED asks the model to keep waiting with the same set of ids. Only ids confirmed
    // delivered by the ledger are dropped here.
    let task_ids = task_ids_filtered;
    if task_ids.is_empty() {
        // Repeatedly waiting on already-delivered ids is a no-op: the results were already surfaced
        // by an earlier task result tool call. PARKED / BUDGET-ELAPSED instruct re-calling
        // task_wait to keep waiting, and a stuck model may wrongly extend that instruction to this
        // terminal case. Count consecutive no-op calls for this key and escalate to an error after
        // TASK_WAIT_NOOP_ERROR_THRESHOLD so the loop is broken with a distinct signal instead of an
        // ever-softer hint.
        let wait_key = task_wait_key(
            &current_session_id,
            current_owner_pid,
            &wait_policy,
            &requested_task_ids,
        );
        let noop_count = bump_task_wait_noop_count(&wait_key);
        if noop_count >= TASK_WAIT_NOOP_ERROR_THRESHOLD {
            return Err(format!(
                "Repeated task_wait on task_id(s) whose results were already delivered \
                 ({} consecutive no-op calls). The wait is OVER — this is NOT a PARKED or \
                 BUDGET-ELAPSED state, so do NOT call task_wait again for these ids; each repeat \
                 produces no new information. Call `task_integrate` with the task_id(s) below to \
                 persist the delivered results into the evidence ledger, or `task_status` for a \
                 non-blocking snapshot.\n\
                 task_ids: {}",
                noop_count,
                already_delivered.join(", ")
            ));
        }
        return Ok(format!(
            "[task_wait] All {} referenced task(s) already completed and \
             their results were delivered by an earlier task result tool call. No tasks remain to \
             wait on; the wait is OVER (unlike PARKED/BUDGET-ELAPSED, do NOT re-call task_wait for \
             these ids — call `task_integrate` or continue reasoning with the results you already \
             collected).",
            already_delivered.len()
        ));
    }
    // lone-spawn normative hint: computed only once (consuming the hinted flag), then uniformly
    // appended at every later return point.
    let lone_spawn_hint = if agent_team_mode {
        None
    } else {
        lone_spawn_hint_note(&task_ids)
    };
    let wait_key = task_wait_key(
        &current_session_id,
        current_owner_pid,
        &wait_policy,
        &requested_task_ids,
    );
    // A real wait (still-pending tasks) proves the model is not stuck polling delivered ids, so
    // reset the no-op escalation counter for this key.
    reset_task_wait_noop_count(&wait_key);
    // Measure this invocation separately from the budget retained across cooperative parks.
    let wait_started_at = Instant::now();
    let wait_state = load_or_create_task_wait_state(&wait_key, timeout_secs, agent_team_mode);
    let wait_budget_elapsed = wait_state.expired;

    let mut ready = Vec::new();
    let mut pending = Vec::new();
    let return_on_ready = agent_team_mode || wait_policy == WaitPolicy::Any;
    // Collect the task_ids finished in this call (success / failure, channel/futex destroyed,
    // needing removal from the registry); the suspended and budget-elapsed early-return paths also
    // use it for cleanup.
    let mut finished: Vec<String> = Vec::new();
    // `write_terminal_subagent_result` only terminates the kernel process; it does not stop the
    // hosting Tokio Future. So first abort every worker that exceeded its total lifetime, then
    // enter the kernel critical section to publish the terminal state.
    for tid in &task_ids {
        let entry = entries.get(tid).expect("validated");
        if entry.started_at.elapsed() > SUBAGENT_WALL_CLOCK_TIMEOUT {
            entry.cancel_stream.store(true, Ordering::Release);
            if let Some(handle) = &entry.abort_handle {
                handle.abort();
            }
        }
    }
    // The closure borrows wait_policy / registry / pending / ready / finished by reference by
    // default; no `move` is added, so code after the closure (e.g. `if !pending.is_empty()`) can
    // still access them.
    let wait_result = with_os_kernel(|os| {
        for tid in &task_ids {
            let entry = entries.get(tid).expect("validated");
            // ⚠️ This block previously marked a task TIMEOUT and destroyed its channel/futex as
            // soon as `entry.started_at.elapsed() >= timeout_secs` — that was a bug:
            // `started_at` is the spawn time, not the start of this task_wait call. If the main
            // agent first calls task_wait long after spawning, every task would be **immediately**
            // reported as TIMEOUT and its result_channel destroyed, permanently losing the real
            // subagent result while the main agent naturally concludes "subagent is stuck".
            //
            // Current behavior: only look at whether a ready payload exists on the channel; if
            // not, uniformly take the pending branch. The per-call task_wait budget is governed by
            // the real wall-clock deadline in TASK_WAIT_STATES; the driver run_loop wakes the
            // owner process when it expires, and only the next task_wait returns BUDGET ELAPSED.
            // Budget exhaustion also **never destroys the channel/futex**, so the main agent can
            // keep calling task_wait to wait longer.
            // Wall-clock total-lifetime check: if a subagent still has no result past
            // SUBAGENT_WALL_CLOCK_TIMEOUT (typically stuck in a single tool execution that never
            // returns), terminate it proactively and write a timeout terminal result, so the
            // immediately following read_task_result reads a result and the main agent does not
            // spin in a "timeout -> wait again -> timeout" loop. Unlike the historical bug that
            // compared started_at against the per-call timeout_secs, this uses an independent
            // total-lifetime cap far larger than a single wait budget, and writes a failure result
            // instead of destroying the channel, so no result is lost.
            if entry.started_at.elapsed() > SUBAGENT_WALL_CLOCK_TIMEOUT {
                write_terminal_subagent_result(
                    os,
                    tid,
                    entry.pid,
                    entry.result_channel_id,
                    entry.completion_futex_addr,
                    TaskTerminalStatus::Timeout.as_str(),
                    &format!(
                        "Subagent exceeded wall-clock lifetime of {}s (likely stuck in a non-returning tool execution)",
                        SUBAGENT_WALL_CLOCK_TIMEOUT.as_secs()
                    ),
                );
            }
            match drain_task_result(os, tid, entry, "subagent result collected")? {
                Some(collected) => {
                    ready.push(collected.rendered);
                    finished.push(tid.clone());
                }
                None => pending.push((tid.clone(), entry.pid)),
            }
        }

        // With `any`, once the first scan already collected a result, we must return immediately
        // instead of being suspended by the remaining pending tasks.
        if !pending.is_empty() && !wait_budget_elapsed && !(return_on_ready && !ready.is_empty()) {
            let pending_ids = pending
                .iter()
                .map(|(tid, _)| tid.clone())
                .collect::<Vec<_>>();
            let wait_sources = task_wait_sources(os, &pending_ids, &entries)?;
            // `task_wait`'s `wait_policy=all` is tool-layer semantics: all task results must be
            // collected before returning. The underlying park cannot use `WaitPolicy::All` to wait
            // on every event source, because the sources also include the cancel futex used to
            // interrupt the current process, which never completes on the normal path. So we wait
            // for "any task event" to wake us, then re-scan all task states; if everything is not
            // collected yet, the model can call task_wait again with the same task_ids.
            let wait = epoll_wait_many(
                os,
                &format!("task_wait:{}", pending_ids.join(",")),
                &wait_sources,
                WaitPolicy::Any,
                None,
            )?;
            // Whether or not epoll_wait_many suspended, always re-scan first to collect results
            // that became ready during the wait. If it suspended and all tasks are now complete,
            // return the results directly (instead of PARKED), so the model is not forced to call
            // task_wait repeatedly under wait_policy=all. PARKED is returned only when tasks are
            // still pending after the re-scan and the wait really did suspend.
            pending.clear();
            for tid in &pending_ids {
                let entry = entries.get(tid).expect("validated after wait");
                match drain_task_result(os, tid, entry, "subagent result collected after wait")? {
                    Some(collected) => {
                        ready.push(collected.rendered);
                        finished.push(tid.clone());
                    }
                    None => pending.push((tid.clone(), entry.pid)),
                }
            }
            // Tasks are still pending after the re-scan and the wait really suspended (a
            // cooperative yield, not budget exhaustion), so return PARKED with the partial results
            // collected so far. Terminal wording like "BUDGET ELAPSED" must **never** be used
            // here: a suspend returns synchronously within milliseconds (it does not really wait
            // the full timeout_secs), otherwise the model would misread "timeout right after
            // starting to wait" as "subtask stuck" and give up early to fall back to manual
            // analysis.
            if !pending.is_empty() && wait.suspended && !(return_on_ready && !ready.is_empty()) {
                let mut parts = Vec::new();
                if !ready.is_empty() {
                    parts.push(ready.join("\n\n---\n\n"));
                }
                let policy_label = match wait_policy {
                    WaitPolicy::Any => "any",
                    WaitPolicy::All => "all",
                };
                let still_pending = pending
                    .iter()
                    .map(|(tid, _)| tid.clone())
                    .collect::<Vec<_>>();
                let parked_message = if agent_team_mode {
                    super::quiet::format_agent_team_task_wait(
                        &still_pending,
                        TaskWaitReturnReason::Parked,
                        Instant::now().saturating_duration_since(wait_started_at),
                    )
                } else {
                    let mut parked_message = format!(
                        "[task_wait PARKED] Yielded CPU so {} pending subagent task(s) can run. \
                    This is normal cooperative scheduling, NOT a timeout and NOT a stall — the wait budget \
                    ({timeout_secs}s, wait_policy={policy_label}) has NOT elapsed. The scheduler will wake this \
                    agent as soon as a result is ready. \
                    Pending task_ids: [{}]. event_ids={}. \
                    Do NOT assume the subagents are stuck and do NOT abandon them to work around this; \
                    when woken, re-call `task_wait` with the same task_ids to collect results, or use \
                    `task_status` for a non-blocking snapshot.",
                        still_pending.len(),
                        still_pending.join(", "),
                        wait.event_ids
                            .iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    append_task_progress_snapshots(os, &mut parked_message, &still_pending);
                    append_pending_task_side_note_summaries(&mut parked_message, &still_pending);
                    parked_message
                };
                parts.push(parked_message);
                return Ok(Some(parts.join("\n\n---\n\n")));
            }
        }
        Ok(None)
    });
    // Remove collected entries before propagating a mid-batch kernel error: their channels and
    // futexes are already destroyed, so keeping them would make every later task_wait/task_status
    // call fail on a missing channel.
    for tid in &finished {
        remove_task_entry(tid);
    }
    let wait_message = wait_result?;
    if let Some(message) = wait_message {
        return Ok(append_lone_spawn_hint(message, lone_spawn_hint.as_deref()));
    }
    if agent_team_mode && !ready.is_empty() {
        // Deliver terminal evidence immediately, including partial `all` waits. The quiet
        // deadline must never postpone a result or an error already collected from the channel.
        let mut parts = vec![ready.join("\n\n---\n\n")];
        if !pending.is_empty() {
            let pending_ids = pending
                .iter()
                .map(|(tid, _)| tid.clone())
                .collect::<Vec<_>>();
            let reason = if wait_budget_elapsed {
                TaskWaitReturnReason::BudgetElapsed
            } else {
                TaskWaitReturnReason::ResultsReady
            };
            parts.push(super::quiet::format_agent_team_task_wait(
                &pending_ids,
                reason,
                Instant::now().saturating_duration_since(wait_started_at),
            ));
        }
        clear_task_wait_state(&wait_key);
        return Ok(parts.join("\n\n---\n\n"));
    }
    // `any` exists to return the first finisher without blocking on the rest. Once the wait
    // budget has already elapsed, the owner was woken for the timeout, so returning ready
    // results alone would hide the still-pending tasks and clear the elapsed wait state.
    // Fall through to the pending status branch, which reports ready output and pending ids.
    if wait_policy == WaitPolicy::Any && !ready.is_empty() && !wait_budget_elapsed {
        clear_task_wait_state(&wait_key);
        return Ok(append_lone_spawn_hint(
            ready.join("\n\n---\n\n"),
            lone_spawn_hint.as_deref(),
        ));
    }
    if !pending.is_empty() {
        // Surface partial progress instead of dropping it on the floor.
        let mut parts = Vec::new();
        if !ready.is_empty() {
            parts.push(ready.join("\n\n---\n\n"));
        }
        let pending_ids = pending
            .iter()
            .map(|(tid, _)| tid.clone())
            .collect::<Vec<_>>();
        let policy_label = match wait_policy {
            WaitPolicy::Any => "any",
            WaitPolicy::All => "all",
        };
        // A ready cancel/completion source can return without suspension or a terminal result.
        let reason = if wait_budget_elapsed {
            TaskWaitReturnReason::BudgetElapsed
        } else if crate::ai::tools::registry::common::is_tool_cancel_requested() {
            TaskWaitReturnReason::Interrupted
        } else {
            TaskWaitReturnReason::Woke
        };
        let pending_message = if agent_team_mode {
            super::quiet::format_agent_team_task_wait(
                &pending_ids,
                reason,
                Instant::now().saturating_duration_since(wait_started_at),
            )
        } else {
            let mut pending_message = if wait_budget_elapsed {
                format!(
                    "[task_wait BUDGET ELAPSED] {} pending subagent task(s) still running in the background. \
            wait_policy={policy_label}, timeout_secs={timeout_secs}. The subagent(s) are NOT stalled and NOT cancelled; \
            their result channels and completion futexes remain alive. \
            Pending task_ids: [{}]. \
            Next steps: call `task_status` for a snapshot, or call `task_wait` again with the same task_ids to keep waiting \
            (consider `wait_policy=\"any\"` if you only need the first finisher).",
                    pending.len(),
                    pending_ids.join(", ")
                )
            } else {
                let (label, cause) = if matches!(reason, TaskWaitReturnReason::Interrupted) {
                    ("INTERRUPTED", "Tool cancellation interrupted this wait")
                } else {
                    ("WOKE", "A wait source became ready")
                };
                format!(
                    "[task_wait {label}] {cause}. {} pending subagent task(s) still running in the background. \
                     wait_policy={policy_label}, timeout_secs={timeout_secs}. The wait budget has not elapsed; \
                     subagent result channels and completion futexes remain alive. Pending task_ids: [{}]. \
                     Call `task_status` for a snapshot, or re-call `task_wait` with the same task_ids to keep waiting.",
                    pending.len(),
                    pending_ids.join(", ")
                )
            };
            with_os_kernel(|os| {
                append_task_progress_snapshots(os, &mut pending_message, &pending_ids);
                Ok(())
            })?;
            append_pending_task_side_note_summaries(&mut pending_message, &pending_ids);
            pending_message
        };
        parts.push(pending_message);
        clear_task_wait_state(&wait_key);
        return Ok(append_lone_spawn_hint(
            parts.join("\n\n---\n\n"),
            lone_spawn_hint.as_deref(),
        ));
    }

    clear_task_wait_state(&wait_key);
    Ok(append_lone_spawn_hint(
        ready.join("\n\n---\n\n"),
        lone_spawn_hint.as_deref(),
    ))
}

pub(super) fn append_lone_spawn_hint(mut text: String, hint: Option<&str>) -> String {
    if let Some(hint) = hint {
        text.push_str("\n\n");
        text.push_str(hint);
    }
    text
}

// Queued backend notes remain visible after a checkpoint revision has been delivered.
fn append_pending_task_side_note_summaries(output: &mut String, task_ids: &[String]) {
    let snapshots = task_ids
        .iter()
        .filter_map(|task_id| {
            pending_task_side_note_progress(task_id)
                .map(|pending| format!("- {task_id}: {pending}"))
        })
        .collect::<Vec<_>>();
    if snapshots.is_empty() {
        return;
    }
    output.push_str("\nPending side-note summaries:\n");
    output.push_str(&snapshots.join("\n"));
}

/// Writes a terminal result to a subagent's result channel and terminates its kernel process.
/// Used for task_cancel (explicit cancellation) and wall-clock total-lifetime timeout. The result
/// uses the same status/output/error format as `publish_background_task_failure`, so the
/// task_wait / task_status collection paths read it normally. This function only releases the
/// producer-side named ownership and stores the futex to wake the waiter; destroying the
/// channel/futex is left to the collector (task_wait's ready path or task_cancel itself) to avoid
/// double release.
pub(super) fn write_terminal_subagent_result(
    os: &mut dyn aios_kernel::kernel::Kernel,
    task_id: &str,
    pid: u64,
    result_channel_id: u64,
    completion_futex_addr: aios_kernel::primitives::FutexAddr,
    status: &str,
    error: &str,
) {
    // The caller must first abort the Tokio task actually running the subagent; the kernel process
    // state alone does not stop the Future inside the hosting process. Only then terminate the
    // kernel process and publish the terminal result.
    let _ = os.kill_process(pid, format!("{}: {}", status, error));
    let backend_session = backend_session::record_session(task_backend_id(task_id), task_id, None);
    // Then write the terminal result as the subagent and release the producer side (the result
    // channel's producer ownership check requires current == pid). Although the process is already
    // terminated, the channel/futex resources are not reclaimed yet (that happens in
    // drop_terminated), so the write is still valid.
    let original = os.current_process_id();
    os.set_current_pid(Some(pid));
    let payload = serde_json::json!({
        "status": status,
        "output": "",
        "error": error,
        "progress": current_task_progress(task_id),
        "backend_session": backend_session,
    })
    .to_string();
    if let Err(error) = publish_task_result(os, Some(task_id), pid, result_channel_id, payload) {
        eprintln!("[Task result delivery] {error}");
    }
    let _ = os.futex_store(completion_futex_addr, 1);
    os.set_current_pid(original);
}
pub(super) fn task_wait_sources(
    os: &mut dyn Kernel,
    task_ids: &[String],
    entries: &FxHashMap<String, AsyncTaskEntry>,
) -> Result<Vec<WaitManySource>, String> {
    let mut sources = Vec::new();
    for tid in task_ids {
        let entry = entries
            .get(tid)
            .ok_or_else(|| format!("Unknown task_id: {}", tid))?;
        sources.extend(wait_sources_for_channel_and_futex(
            os,
            entry.result_channel_id,
            Some(entry.completion_futex_addr),
        )?);
    }
    append_current_process_cancel_source(os, &mut sources)?;
    Ok(sources)
}

pub(super) fn is_task_pending(os: &mut dyn Kernel, pid: u64) -> Result<bool, String> {
    let Some(proc) = os.get_process(pid) else {
        return Ok(false);
    };
    // Stopped is pending, not finished: SIGSTOP pauses the process and SIGCONT
    // resumes it, so a stopped subagent can still consume queued side-notes and
    // must not be collected/cancelled as if it had terminated.
    Ok(matches!(
        proc.state,
        ProcessState::Ready
            | ProcessState::Running
            | ProcessState::Waiting { .. }
            | ProcessState::Sleeping { .. }
            | ProcessState::Stopped
    ))
}

pub(super) fn task_state_string(
    os: &mut dyn Kernel,
    result_channel_id: u64,
    pid: u64,
) -> Result<String, String> {
    if let Some(result) = read_task_result(os, result_channel_id, false)? {
        return Ok(result.status);
    }
    let state = match os.get_process(pid) {
        Some(proc) => match proc.state {
            ProcessState::Ready => "ready",
            ProcessState::Running => "running",
            ProcessState::Waiting { .. } => "waiting",
            ProcessState::Sleeping { .. } => "sleeping",
            ProcessState::Stopped => "stopped",
            ProcessState::Terminated => "terminated",
        },
        None => "unknown",
    };
    Ok(state.to_string())
}

pub(crate) fn format_subagent_model_label(model: &str, reasoning_effort: Option<&str>) -> String {
    match reasoning_effort {
        Some(effort) => format!("{model} (effort: {effort})"),
        None => model.to_string(),
    }
}

pub(super) fn format_task_result(
    entry: &AsyncTaskEntry,
    result: StoredTaskResult,
    reasoning_effort: Option<&str>,
) -> String {
    let duration_secs = entry.started_at.elapsed().as_secs_f64();
    let task_blocked = result.status == TaskTerminalStatus::Blocked.as_str();
    let mut parts = vec![format!(
        "[Task: {} via {} @ {}] {} after {:.1}s",
        entry.description,
        entry.agent_name,
        format_subagent_model_label(&entry.model, reasoning_effort),
        result.status.to_uppercase(),
        duration_secs
    )];
    let backend = task_backend(entry.backend);
    let selection_explanation = if let Some(template) = backend.descriptor().text.model_reason {
        let agent_reason = entry
            .selection_explanation
            .split_once('\n')
            .map(|(agent_reason, _)| agent_reason)
            .unwrap_or(&entry.selection_explanation);
        format!(
            "{agent_reason}\n{}",
            template.replace("{model}", &entry.model)
        )
    } else {
        entry.selection_explanation.clone()
    };
    parts.push(selection_explanation);
    if let Some(error) = result.error
        && !error.trim().is_empty()
    {
        parts.push(format!("Failure reason: {}", error));
    }
    if let Some(progress) = result.progress
        && !progress.trim().is_empty()
    {
        parts.push(format!("Last known progress: {}", progress));
    }
    if !result.output.trim().is_empty() {
        if result.status == "completed" {
            parts.push(result.output.trim().to_string());
        } else {
            parts.push(format!("Partial output:\n{}", result.output.trim()));
        }
    } else {
        parts.push("(subagent did not produce any final assistant text)".to_string());
    }
    if let Some(session) = result.backend_session
        && !session.trim().is_empty()
    {
        parts.push(backend_session_footer_for(backend.as_ref(), &session));
    }
    if task_blocked {
        parts.push(SUBAGENT_BLOCKED_PARENT_GUIDANCE.to_string());
    }
    parts.push(SUBAGENT_PARENT_SUMMARY_REMINDER.to_string());
    parts.join("\n")
}

pub(super) fn format_task_result_with_id(
    task_id: &str,
    entry: &AsyncTaskEntry,
    result: StoredTaskResult,
) -> String {
    let retryable = is_retryable_task_status(&result.status);
    let reasoning_effort = task_reasoning_effort(task_id);
    let mut rendered = format!(
        "[task_id={task_id}]\n{}",
        format_task_result(entry, result, reasoning_effort.as_deref())
    );
    if retryable {
        rendered.push_str(&format!(
            "\nRetry available: call `task_retry` with `task_id=\"{task_id}\"` to rerun the same subagent configuration as a new linked attempt."
        ));
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use aios_kernel::kernel::{KernelInternal, Syscall};
    use aios_kernel::local::LocalOS;
    use aios_kernel::primitives::{FutexOps, IpcOps};

    struct NormalProgressTestGuard {
        session_id: String,
        task_id: String,
    }

    impl Drop for NormalProgressTestGuard {
        fn drop(&mut self) {
            remove_task_entry(&self.task_id);
            lock_unpoisoned(&TASK_PROGRESS_REGISTRY).remove(&self.task_id);
            lock_unpoisoned(&TASK_WAIT_STATES).retain(|key, _| key.session_id != self.session_id);
        }
    }

    #[test]
    fn agent_team_quiet_normal_task_paths_deliver_checkpoints_once_per_revision() {
        use crate::ai::driver::runtime_ctx;

        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        let mut app = super::super::tests::test_app_with_model("test-model".to_string());
        app.session_id = format!("normal-progress-{}", Uuid::new_v4().simple());
        let task_id = format!("task-{}", Uuid::new_v4().simple());
        let _guard = NormalProgressTestGuard {
            session_id: app.session_id.clone(),
            task_id: task_id.clone(),
        };
        let (owner_pid, pid, channel, futex) = {
            let mut os = app.os.lock().unwrap();
            let owner_pid = os.begin_foreground("parent".into(), "goal".into(), 10, 8, None);
            let pid = os
                .spawn(
                    Some(owner_pid),
                    "child".into(),
                    "goal".into(),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();
            let channel = os.channel_create(Some(pid), 1, "normal-result".into());
            let futex = os.futex_create(0, "normal-complete".into());
            os.set_current_pid(Some(owner_pid));
            (owner_pid, pid, channel.raw(), futex)
        };
        insert_task_entry_for_test(
            task_id.clone(),
            AsyncTaskEntry {
                evidence_history_file: None,
                session_id: app.session_id.clone(),
                owner_pid,
                pid,
                result_channel_id: channel,
                completion_futex_addr: futex,
                description: "normal progress test".into(),
                agent_name: "build".into(),
                backend: None,
                model: "test-model".into(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: "test selection".into(),
                inherit: InheritOptions::default(),
                abort_handle: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
                started_at: Instant::now(),
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
            },
        );
        let slot = runtime_ctx::new_subagent_progress_slot();
        lock_unpoisoned(&TASK_PROGRESS_REGISTRY).insert(task_id.clone(), slot.clone());
        let checkpoint_body = "normal checkpoint body";
        runtime_ctx::publish_subagent_checkpoint_to_slot(
            &slot,
            "saved checkpoint",
            checkpoint_body,
        );
        let context = runtime_ctx::DriverContext::new(
            app,
            Arc::new(Mutex::new(crate::ai::mcp::McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        let wait_args = serde_json::json!({"task_ids": [task_id.clone()], "timeout_secs": 60});
        runtime_ctx::TURN_IDENTITY.sync_scope((context.app_proto.session_id.clone(), 0), || {
            runtime_ctx::DRIVER_CTX.sync_scope(context.clone(), || {
                assert!(!is_agent_team_mode());
                let mut outputs = Vec::new();
                for _ in 0..2 {
                    let status = execute_task_status(&serde_json::json!({})).unwrap();
                    assert!(status.starts_with("TaskID"), "{status}");
                    outputs.push(status);
                }
                let parked = execute_task_wait(&wait_args).unwrap();
                assert!(parked.contains("[task_wait PARKED]"), "{parked}");
                outputs.push(parked);

                let key = task_wait_key(
                    &context.app_proto.session_id,
                    owner_pid,
                    &WaitPolicy::Any,
                    &[task_id.clone()],
                );
                lock_unpoisoned(&TASK_WAIT_STATES)
                    .get_mut(&key)
                    .unwrap()
                    .deadline = Instant::now() - Duration::from_secs(1);
                wake_expired_task_waits();
                let timeout = context
                    .app_proto
                    .os
                    .lock()
                    .unwrap()
                    .get_process(owner_pid)
                    .unwrap()
                    .mailbox
                    .iter()
                    .find(|message| message.contains("[TASK_WAIT_TIMEOUT]"))
                    .cloned()
                    .expect("normal wait deadline must wake the parent");
                outputs.push(timeout);
                context
                    .app_proto
                    .os
                    .lock()
                    .unwrap()
                    .set_current_pid(Some(owner_pid));
                let elapsed = execute_task_wait(&wait_args).unwrap();
                assert!(elapsed.contains("[task_wait BUDGET ELAPSED]"), "{elapsed}");
                outputs.push(elapsed);

                for (index, output) in outputs.iter().enumerate() {
                    assert!(output.contains("saved checkpoint"), "{output}");
                    if index == 0 {
                        assert_eq!(output.matches(checkpoint_body).count(), 1, "{output}");
                        assert!(
                            output.contains("partial checkpoint excerpt; unverified"),
                            "{output}"
                        );
                    } else {
                        assert!(!output.contains(checkpoint_body), "{output}");
                        assert!(!output.contains("partial checkpoint excerpt"), "{output}");
                    }
                }

                // Saving identical text creates a new revision that must be delivered again.
                runtime_ctx::publish_subagent_checkpoint_to_slot(
                    &slot,
                    "saved checkpoint",
                    checkpoint_body,
                );
                let revised = execute_task_wait(&wait_args).unwrap();
                assert!(revised.contains("[task_wait PARKED]"), "{revised}");
                assert_eq!(revised.matches(checkpoint_body).count(), 1, "{revised}");
                assert!(
                    revised.contains("partial checkpoint excerpt; unverified"),
                    "{revised}"
                );
                context
                    .app_proto
                    .os
                    .lock()
                    .unwrap()
                    .set_current_pid(Some(owner_pid));
                let repeated_status = execute_task_status(&serde_json::json!({})).unwrap();
                assert!(
                    repeated_status.contains("saved checkpoint"),
                    "{repeated_status}"
                );
                assert!(
                    !repeated_status.contains(checkpoint_body),
                    "{repeated_status}"
                );
                let repeated_wait = execute_task_wait(&wait_args).unwrap();
                assert!(
                    repeated_wait.contains("[task_wait PARKED]"),
                    "{repeated_wait}"
                );
                assert!(!repeated_wait.contains(checkpoint_body), "{repeated_wait}");
                assert!(
                    runtime_ctx::take_subagent_parent_progress_snapshot(&slot, true)
                        .unwrap()
                        .checkpoint_excerpt
                        .is_none()
                );
            });
        });
    }

    #[test]
    fn traecli_terminal_results_surface_session_after_collection() {
        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let session = "cccccccc-1111-2222-3333-444444444444";
        for status in [
            TaskTerminalStatus::Failed,
            TaskTerminalStatus::Cancelled,
            TaskTerminalStatus::Timeout,
        ] {
            for known_session in [true, false] {
                let task_id = format!("task_{}", Uuid::new_v4().simple());
                let slot = super::super::traecli::track_traecli_session(&task_id);
                if known_session {
                    *lock_unpoisoned(&slot) = Some(session.to_string());
                }
                assert!(resolve_continue_target(&task_id).is_err());

                let mut os = LocalOS::new();
                let owner_pid =
                    os.begin_foreground("parent".to_string(), "goal".to_string(), 10, 8, None);
                let pid = os
                    .spawn(
                        Some(owner_pid),
                        "child".to_string(),
                        "goal".to_string(),
                        20,
                        8,
                        None,
                        None,
                    )
                    .unwrap();
                let channel = os.channel_create_tagged_with_holders(
                    Some(owner_pid),
                    1,
                    "task-result".to_string(),
                    ChannelOwnerTag::TaskResult,
                    vec![
                        "task_result.producer".to_string(),
                        "task_result.consumer".to_string(),
                    ],
                );
                let completion_futex = os.futex_create(0, "task-complete".to_string());
                let entry = AsyncTaskEntry {
                    evidence_history_file: None,
                    session_id: String::new(),
                    owner_pid,
                    pid,
                    result_channel_id: channel.raw(),
                    completion_futex_addr: completion_futex,
                    description: "resume terminal run".to_string(),
                    agent_name: "explore".to_string(),
                    backend: Some(SubagentBackend::TraeCli),
                    model: "backend-test-model".to_string(),
                    is_model_auto_selected: false,
                    auto_model_fallback: None,
                    selection_explanation: "explicit override".to_string(),
                    inherit: InheritOptions::default(),
                    abort_handle: None,
                    cancel_stream: Arc::new(AtomicBool::new(false)),
                    started_at: Instant::now(),
                    last_progress_notification_at: None,
                    last_progress_persisted_at: None,
                };
                insert_task_entry_for_test(task_id.clone(), entry.clone());
                let error = "original terminal reason";
                write_terminal_subagent_result(
                    &mut os,
                    &task_id,
                    pid,
                    channel.raw(),
                    completion_futex,
                    status.as_str(),
                    error,
                );
                let collected = drain_task_result(&mut os, &task_id, &entry, "test collection")
                    .unwrap()
                    .expect("collectable terminal result");
                let result = collected.result;
                assert_eq!(result.status, status.as_str());
                assert_eq!(result.error.as_deref(), Some(error));
                assert_eq!(
                    result.backend_session.as_deref(),
                    known_session.then_some(session)
                );
                let output = collected.rendered;
                assert!(output.contains(&format!("Failure reason: {error}")));
                if known_session {
                    assert!(
                        output.contains(&backend_session_footer(session)),
                        "{output}"
                    );
                } else {
                    assert!(!output.contains("Backend session:"), "{output}");
                }

                remove_task_entry(&task_id).expect("collected task entry");
                assert!(os.channel_meta(channel).is_none());
                assert!(os.futex_event_id(completion_futex).is_none());
                assert!(os.get_process(pid).is_none());
                if known_session {
                    assert_eq!(resolve_continue_target(&task_id).unwrap(), session);
                } else {
                    assert!(resolve_continue_target(&task_id).is_err());
                }
            }
        }
    }
}
