use super::*;

#[tokio::test]
async fn wait_for_interrupt_observes_request_interrupt_source() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    let waiter = wait_for_interrupt(&app);
    let trigger = async {
        tokio::time::sleep(Duration::from_millis(20)).await;
        crate::ai::driver::signal::signal_request_interrupt();
    };

    tokio::join!(waiter, trigger);
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn wait_for_interrupt_or_timeout_returns_true_on_request_interrupt() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    let waiter = tokio::spawn(async move {
        wait_for_interrupt_or_timeout(&app, Some(Duration::from_secs(5))).await
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    crate::ai::driver::signal::signal_request_interrupt();

    let interrupted = tokio::time::timeout(Duration::from_millis(200), waiter)
        .await
        .expect("stream retry wait should wake on interrupt")
        .expect("waiter should complete");
    assert!(interrupted);

    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}
