// Temporary end-to-end verification: faithfully reproduce the user-reported
// bytedcli QR-code login hang scenario.
// Verify stall detection and process-group cleanup through the real production
// path (default stall thresholds 10s/20s).
use std::time::{Duration, Instant};

#[test]
#[ignore = "requires bytedcli installed; run explicitly: cargo test --test e2e_pty_stall_check -- --ignored"]
fn real_bytedcli_login_hang_is_stall_killed() {
    let started = Instant::now();
    let result =
        rust_tools::cmd::run::run_cmd_output_streaming_with_timeout_tracked_pseudo_terminal(
            "bytedcli auth login --session 2>&1 | tail -15",
            rust_tools::cmd::run::RunCmdOptions::default(),
            Duration::from_secs(120),
            |_| {},
            || false,
            |_| {},
        )
        .expect("PTY command should run");

    let elapsed = started.elapsed();
    eprintln!("elapsed={elapsed:?} result={result:?}");
    assert!(
        result.stalled,
        "expected stalled (waiting for interactive input), got: {result:?}"
    );
    assert!(
        elapsed < Duration::from_secs(60),
        "must return well before the 60s default timeout, took {elapsed:?}"
    );
    assert!(!result.timed_out && !result.cancelled);
}

#[test]
#[ignore = "requires bytedcli installed; run explicitly: cargo test --test e2e_pty_stall_check -- --ignored"]
fn real_bytedcli_login_without_pipe_captures_qr_then_stalls() {
    // Without a pipe, part of the QR-code output should still get captured
    // (stall detection is based on "silence following output").
    let result =
        rust_tools::cmd::run::run_cmd_output_streaming_with_timeout_tracked_pseudo_terminal(
            "bytedcli auth login --session",
            rust_tools::cmd::run::RunCmdOptions::default(),
            Duration::from_secs(120),
            |_| {},
            || false,
            |_| {},
        )
        .expect("PTY command should run");

    let out = String::from_utf8_lossy(&result.stdout);
    eprintln!("stalled={} out_bytes={}", result.stalled, out.len());
    assert!(result.stalled, "expected stall kill, got: {result:?}");
    assert!(
        out.len() > 0,
        "partial output (QR code) should be captured before termination"
    );
}
