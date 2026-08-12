// 临时端到端验证：真实复现用户报告的 bytedcli 扫码登录挂起场景。
// 通过真实生产路径（默认停滞阈值 10s/20s）验证停滞判定与进程组清理。
use std::time::{Duration, Instant};

#[test]
#[ignore = "requires bytedcli installed; run explicitly: cargo test --test e2e_pty_stall_check -- --ignored"]
fn real_bytedcli_login_hang_is_stall_killed() {
    let started = Instant::now();
    let result = rust_tools::cmd::run::run_cmd_output_streaming_with_timeout_tracked_pseudo_terminal(
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
    // 不带管道时，二维码应被部分输出捕获（停滞判定基于"输出后静默"）。
    let result = rust_tools::cmd::run::run_cmd_output_streaming_with_timeout_tracked_pseudo_terminal(
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
