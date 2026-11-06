use super::*;

#[test]
fn traecli_reliability_reduces_protocol_errors_after_body_and_recovers_on_completion() {
    let body =
        r#"{"type":"item.completed","item":{"type":"agent_message","text":"  saved evidence\n"}}"#;
    for (events, is_error) in [
        (
            format!("{body}\n{{\"type\":\"error\",\"message\":\"late failure\"}}"),
            true,
        ),
        (
            format!(
                "{body}\n{{\"type\":\"turn.failed\",\"error\":{{\"message\":\"terminal failure\"}}}}"
            ),
            true,
        ),
        (
            format!(
                "{body}\n{{\"type\":\"turn.completed\"}}\n{{\"type\":\"error\",\"error\":{{\"message\":\"late failure\"}}}}"
            ),
            true,
        ),
        (
            format!(
                "{{\"type\":\"error\",\"message\":\"recoverable\"}}\n{body}\n{{\"type\":\"turn.completed\"}}"
            ),
            false,
        ),
        (
            format!(
                "{{\"type\":\"turn.failed\",\"error\":{{\"message\":\"previous turn\"}}}}\n{{\"type\":\"turn.started\"}}\n{body}\n{{\"type\":\"turn.completed\"}}"
            ),
            false,
        ),
    ] {
        let outcome = parse_stream_json_output(&events).unwrap();
        assert_eq!(outcome.result, "  saved evidence\n");
        assert_eq!(outcome.is_error, is_error, "{events}");
    }
    for event in [
        r#"{"type":"turn.failed","error":{"message":"nested failure"}}"#,
        r#"{"type":"error","error":{"message":"nested failure"}}"#,
        r#"{"type":"turn.failed","message":"","error":{"message":"nested failure"}}"#,
    ] {
        let outcome = parse_stream_json_output(event)
            .expect("terminal failures must be recognized without a body");
        assert!(outcome.is_error);
        assert_eq!(outcome.result, "nested failure", "{event}");
    }
}

#[cfg(unix)]
#[test]
fn traecli_reliability_exit_status_and_terminal_error_keep_failure_evidence() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    for (terminal, exit, expected_error) in [
        (r#"{"type":"turn.completed"}"#, 7, "exited with status"),
        (
            r#"{"type":"error","message":"late failure"}"#,
            0,
            "late failure",
        ),
        (
            r#"{"type":"turn.failed","error":{"message":"terminal failure"}}"#,
            0,
            "terminal failure",
        ),
    ] {
        let script = terminal_test_script(&format!(
            r#"/bin/echo '{{"type":"thread.started","thread_id":"failure-session"}}'
/bin/echo '{{"type":"item.completed","item":{{"type":"agent_message","text":"  saved evidence\n"}}}}'
/bin/echo '{terminal}'
printf 'diagnostic tail\n' >&2
exit {exit}"#
        ));
        let evidence = Arc::new(Mutex::new(None));
        let result = run_traecli_blocking_with_evidence(
            &test_spec(),
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_secs(2),
            Vec::new(),
            Some(evidence.clone()),
        );
        std::fs::remove_file(script).unwrap();
        let error = result.expect_err("a body must not hide a terminal failure");
        assert!(error.contains(expected_error), "{error}");
        if exit != 0 {
            assert!(error.contains("diagnostic tail"), "{error}");
        }
        let outcome = lock_unpoisoned(&evidence).clone().unwrap();
        assert_eq!(outcome.result, "  saved evidence\n");
        assert!(outcome.is_error);
        assert_eq!(outcome.session_id.as_deref(), Some("failure-session"));
    }
}

#[cfg(unix)]
#[test]
fn traecli_reliability_intermediate_error_can_finish_successfully() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let script = terminal_test_script(
        r#"/bin/echo '{"type":"error","message":"temporary disconnect"}'
/bin/echo '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}'
/bin/echo '{"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":2}}'"#,
    );
    let result = run_traecli_blocking_with(
        &test_spec(),
        Arc::new(AtomicBool::new(false)),
        script.to_str().unwrap(),
        Duration::from_secs(2),
        Vec::new(),
    );
    std::fs::remove_file(script).unwrap();
    let outcome = result.unwrap();
    assert!(!outcome.is_error);
    assert_eq!(outcome.result, "ok");
    assert_eq!(outcome.num_turns, Some(1));
    assert_eq!(outcome.input_tokens, Some(3));
    assert_eq!(outcome.output_tokens, Some(2));
}

#[cfg(unix)]
#[tokio::test]
async fn traecli_reliability_unknown_tool_results_do_not_retry_even_with_session() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    for item in [
        r#"{"type":"item.started","item":{"id":"cmd","type":"command_execution","command":"write fixture"}}"#,
        r#"{"type":"item.completed","item":{"id":"tool","type":"tool_call"}}"#,
        r#"{"type":"item.failed","item":{"id":"tool","type":"tool_use"}}"#,
        r#"{"type":"item.completed","item":{"type":"file_change","changes":[{"path":"fixture"}]}}"#,
        r#"{"type":"item.started","item":{"type":"mcp_tool_call","tool":"fixture"}}"#,
    ] {
        let script = retry_test_script(&format!(
            r#"/bin/echo '{{"type":"thread.started","thread_id":"effect-session"}}'
/bin/echo '{item}'
/bin/echo '{{"type":"error","message":"connection reset by peer"}}'
exit 1"#
        ));
        let mut spec = test_spec();
        spec.resume_session = Some("explicit-session".to_string());
        let result = run_traecli_subagent_with(
            &spec,
            Arc::new(AtomicBool::new(false)),
            script.to_str().unwrap(),
            Duration::from_secs(2),
        )
        .await;
        let count = std::fs::read_to_string(script.with_extension("count")).unwrap();
        remove_retry_test_script(&script);
        let error = result.unwrap_err();
        assert!(error.contains("connection reset by peer"), "{error}");
        assert_eq!(count, "1\n", "the effect is unknown: {item}");
    }
}

#[cfg(unix)]
#[tokio::test]
async fn traecli_reliability_nonzero_exit_after_tool_does_not_replay() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let script = retry_test_script(
        r#"/bin/echo '{"type":"item.completed","item":{"type":"command_execution","exit_code":0}}'
printf 'transport ended\n' >&2
exit 9"#,
    );
    let result = run_traecli_subagent_with(
        &test_spec(),
        Arc::new(AtomicBool::new(false)),
        script.to_str().unwrap(),
        Duration::from_secs(2),
    )
    .await;
    let count = std::fs::read_to_string(script.with_extension("count")).unwrap();
    remove_retry_test_script(&script);
    assert!(result.unwrap_err().contains("transport ended"));
    assert_eq!(count, "1\n");
}

#[cfg(unix)]
#[tokio::test]
async fn traecli_reliability_late_failure_keeps_body_without_retry_or_cancellation() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let script = retry_test_script(
        r#"/bin/echo '{"type":"thread.started","thread_id":"body-session"}'
/bin/echo '{"type":"item.completed","item":{"type":"agent_message","text":"  saved evidence\n"}}'
/bin/echo '{"type":"turn.failed","error":{"message":"transport failed after body"}}'
exit 1"#,
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let session = Arc::new(Mutex::new(None));
    let evidence = Arc::new(Mutex::new(None));
    let result = run_traecli_subagent_with_session(
        &test_spec(),
        cancel.clone(),
        script.to_str().unwrap(),
        Duration::from_secs(2),
        Some(session.clone()),
        Some(evidence.clone()),
    )
    .await;
    let count = std::fs::read_to_string(script.with_extension("count")).unwrap();
    remove_retry_test_script(&script);
    assert!(result.unwrap_err().contains("transport failed after body"));
    assert_eq!(count, "1\n");
    assert!(
        !cancel.load(Ordering::Acquire),
        "ordinary errors must not cancel the caller"
    );
    assert_eq!(lock_unpoisoned(&session).as_deref(), Some("body-session"));
    let evidence = lock_unpoisoned(&evidence).clone().unwrap();
    assert!(evidence.is_error);
    assert_eq!(evidence.result, "  saved evidence\n");
}

#[cfg(unix)]
#[tokio::test]
async fn traecli_reliability_reasoning_only_failure_can_safely_retry_and_resume() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let script = retry_test_script(
        r#"if [ "$attempt" = 1 ]; then
    /bin/echo '{"type":"thread.started","thread_id":"reasoning-session"}'
    /bin/echo '{"type":"item.completed","item":{"type":"reasoning","text":"thinking"}}'
    /bin/echo '{"type":"error","message":"temporary disconnect"}'
    exit 1
fi
/bin/echo '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}'
/bin/echo '{"type":"turn.completed"}'"#,
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let spec = test_spec();
    let result = run_traecli_subagent_with(
        &spec,
        cancel.clone(),
        script.to_str().unwrap(),
        Duration::from_secs(2),
    )
    .await;
    let count = std::fs::read_to_string(script.with_extension("count")).unwrap();
    let args = std::fs::read_to_string(script.with_extension("args")).unwrap();
    remove_retry_test_script(&script);
    assert_eq!(result.unwrap().result, "ok");
    assert_eq!(count, "2\n");
    assert!(
        !cancel.load(Ordering::Acquire),
        "success must leave cancellation unchanged"
    );
    let mut resumed = spec;
    resumed.resume_session = Some("reasoning-session".to_string());
    assert_eq!(
        args.split("attempt\n")
            .nth(2)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        build_traecli_args(&resumed),
    );
}

#[cfg(unix)]
#[tokio::test]
async fn traecli_reliability_permanent_stderr_is_not_lost_when_tail_is_bounded() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let script = retry_test_script(
        r#"printf 'authentication failed\n' >&2
i=0
while [ "$i" -lt 2048 ]; do
    printf '0123456789abcdef0123456789abcdef' >&2
    i=$((i + 1))
done
printf '\nlast diagnostic\n' >&2
exit 1"#,
    );
    let result = run_traecli_subagent_with(
        &test_spec(),
        Arc::new(AtomicBool::new(false)),
        script.to_str().unwrap(),
        Duration::from_secs(2),
    )
    .await;
    let count = std::fs::read_to_string(script.with_extension("count")).unwrap();
    remove_retry_test_script(&script);
    assert!(result.is_err());
    assert_eq!(
        count, "1\n",
        "bounded stderr must preserve permanent failure classification"
    );
}

#[cfg(unix)]
struct FixtureProcesses {
    script: PathBuf,
}

#[cfg(unix)]
impl FixtureProcesses {
    fn new(stdout: bool) -> Self {
        Self::with_finish(stdout, "wait \"$descendant\"")
    }

    fn with_finish(stdout: bool, finish: &str) -> Self {
        let event = if stdout {
            "/bin/echo '{\"type\":\"thread.started\",\"thread_id\":\"owned-session\"}'"
        } else {
            ""
        };
        let script = terminal_test_script(&format!(
            r#"(
    while :; do
        printf '.' >> "${{0%.sh}}.heartbeat"
        sleep 0.05
    done
) &
descendant=$!
while [ ! -f "${{0%.sh}}.heartbeat" ]; do sleep 0.01; done
printf '%s %s\n' "$$" "$descendant" > "${{0%.sh}}.pids"
printf 'live stderr diagnostic\n' >&2
{event}
{finish}"#
        ));
        Self { script }
    }

    fn pids(&self) -> (u32, u32) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok(text) = std::fs::read_to_string(self.script.with_extension("pids")) {
                let ids = text
                    .split_whitespace()
                    .filter_map(|part| part.parse::<u32>().ok())
                    .collect::<Vec<_>>();
                if ids.len() == 2 {
                    return (ids[0], ids[1]);
                }
            }
            assert!(Instant::now() < deadline, "fixture did not start");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[cfg(unix)]
#[test]
fn traecli_reliability_stderr_is_available_before_inherited_pipe_eof() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let fixture = FixtureProcesses::with_finish(false, "exit 9");
    let started = Instant::now();
    let result = run_traecli_blocking_with(
        &test_spec(),
        Arc::new(AtomicBool::new(false)),
        fixture.script.to_str().unwrap(),
        Duration::from_secs(2),
        Vec::new(),
    );
    let elapsed = started.elapsed();
    drop(fixture);
    let error = result.unwrap_err();
    assert!(error.contains("live stderr diagnostic"), "{error}");
    assert!(elapsed < Duration::from_secs(2));
}

#[cfg(unix)]
#[test]
fn traecli_reliability_stderr_keeps_bounded_recent_diagnostics() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let script = terminal_test_script(
        r#"i=0
while [ "$i" -lt 2048 ]; do
    printf '0123456789abcdef0123456789abcdef' >&2
    i=$((i + 1))
done
printf '\nlatest stderr diagnostic\n' >&2
exit 9"#,
    );
    let result = run_traecli_blocking_with(
        &test_spec(),
        Arc::new(AtomicBool::new(false)),
        script.to_str().unwrap(),
        Duration::from_secs(2),
        Vec::new(),
    );
    std::fs::remove_file(script).unwrap();
    let error = result.unwrap_err();
    assert!(error.contains("latest stderr diagnostic"), "{error}");
    assert!(error.contains("stderr truncated"), "{error}");
    assert!(error.len() < 2_200, "diagnostics must stay bounded");
}

#[cfg(unix)]
impl Drop for FixtureProcesses {
    fn drop(&mut self) {
        if let Ok(text) = std::fs::read_to_string(self.script.with_extension("pids")) {
            for pid in text
                .split_whitespace()
                .skip(1)
                .filter_map(|part| part.parse::<i32>().ok())
            {
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
            }
        }
        let _ = std::fs::remove_file(&self.script);
        let _ = std::fs::remove_file(self.script.with_extension("pids"));
        let _ = std::fs::remove_file(self.script.with_extension("heartbeat"));
    }
}

#[cfg(unix)]
fn wait_for_fixture_stop(fixture: &FixtureProcesses) -> bool {
    let heartbeat = fixture.script.with_extension("heartbeat");
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let before = std::fs::read(&heartbeat).unwrap_or_default();
        std::thread::sleep(Duration::from_millis(300));
        if std::fs::read(&heartbeat).unwrap_or_default() == before {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
    }
}

#[cfg(unix)]
#[test]
fn traecli_reliability_cancellation_cleans_descendants() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let fixture = FixtureProcesses::new(true);
    let binary = fixture.script.to_str().unwrap().to_string();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancel.clone();
    let worker = std::thread::spawn(move || {
        run_traecli_blocking_with(
            &test_spec(),
            worker_cancel,
            &binary,
            Duration::from_secs(2),
            Vec::new(),
        )
    });
    let (parent, _) = fixture.pids();
    let pgid = unsafe { libc::getpgid(parent as libc::pid_t) };
    let cancelled_at = Instant::now();
    cancel.store(true, Ordering::Release);
    let result = worker.join().unwrap();
    let stopped = wait_for_fixture_stop(&fixture);
    drop(fixture);
    assert_eq!(result.unwrap_err(), "traecli subagent cancelled");
    assert!(cancelled_at.elapsed() < Duration::from_secs(3));
    assert!(stopped, "cancel left a descendant running");
    assert_eq!(pgid as u32, parent, "CLI must own a separate process group");
}

#[cfg(unix)]
#[test]
fn traecli_reliability_startup_timeout_cleans_descendants_and_keeps_stderr() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let fixture = FixtureProcesses::new(false);
    let started_at = Instant::now();
    let result = run_traecli_blocking_with(
        &test_spec(),
        Arc::new(AtomicBool::new(false)),
        fixture.script.to_str().unwrap(),
        Duration::from_millis(750),
        Vec::new(),
    );
    fixture.pids();
    let stopped = wait_for_fixture_stop(&fixture);
    drop(fixture);
    let error = result.unwrap_err();
    assert!(error.contains("did not emit a startup event"), "{error}");
    assert!(error.contains("live stderr diagnostic"), "{error}");
    assert!(started_at.elapsed() < Duration::from_secs(3));
    assert!(stopped, "timeout left a descendant running");
}

#[cfg(unix)]
#[test]
fn traecli_reliability_child_guard_drop_cleans_descendants() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let fixture = FixtureProcesses::new(false);
    let mut command = std::process::Command::new(&fixture.script);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let guard = TraeCliChildGuard::spawn(&mut command).unwrap();
    fixture.pids();
    drop(guard);
    let stopped = wait_for_fixture_stop(&fixture);
    drop(fixture);
    assert!(stopped, "Drop left a descendant running");
}

#[cfg(unix)]
#[test]
fn traecli_reliability_reaped_child_guard_never_signals_saved_group() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let fixture = FixtureProcesses::with_finish(false, "exit 0");
    let mut command = std::process::Command::new(&fixture.script);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut guard = TraeCliChildGuard::spawn(&mut command).unwrap();
    fixture.pids();
    assert!(guard.wait().unwrap().success());
    guard.kill().unwrap();
    drop(guard);
    let stopped = wait_for_fixture_stop(&fixture);
    drop(fixture);
    assert!(
        !stopped,
        "an already reaped child must not trigger a group signal"
    );
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn traecli_reliability_dropping_async_waiter_cancels_actual_cli() {
    let _process_guard = lock_unpoisoned(&PROCESS_TEST_LOCK);
    let fixture = FixtureProcesses::new(true);
    let binary = fixture.script.to_str().unwrap().to_string();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancel.clone();
    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let worker_task_id = task_id.clone();
    let waiter = tokio::spawn(async move {
        crate::ai::driver::runtime_ctx::SUBAGENT_TASK_ID
            .scope(
                worker_task_id,
                run_traecli_subagent_with(
                    &test_spec(),
                    worker_cancel,
                    &binary,
                    Duration::from_secs(2),
                ),
            )
            .await
    });
    fixture.pids();
    assert!(backend_session::task_backend_run_is_active(
        BackendId::TRAECLI,
        &task_id
    ));
    waiter.abort();
    assert!(waiter.await.unwrap_err().is_cancelled());
    let stopped = wait_for_fixture_stop(&fixture);
    let automatic_cancel = cancel.load(Ordering::Acquire);
    // Clean up the old implementation too, so a red regression cannot leak its worker.
    cancel.store(true, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(2);
    while backend_session::task_backend_run_is_active(BackendId::TRAECLI, &task_id)
        && Instant::now() < deadline
    {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let active = backend_session::task_backend_run_is_active(BackendId::TRAECLI, &task_id);
    forget_traecli_session(&task_id);
    drop(fixture);
    assert!(
        !active,
        "run guard must release only after the actual worker stops"
    );
    assert!(
        automatic_cancel,
        "dropping the waiter did not signal cancellation"
    );
    assert!(
        stopped,
        "dropping the waiter left the CLI descendants running"
    );
}
