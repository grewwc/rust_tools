use super::common::{test_app};
use super::super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

#[tokio::test]
async fn sleep_with_cancel_observes_request_interrupt_source() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    let waiter = tokio::spawn(async move { sleep_with_cancel(&app, Duration::from_secs(5)).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    crate::ai::driver::signal::signal_request_interrupt();

    let cancelled = tokio::time::timeout(Duration::from_millis(200), waiter)
        .await
        .expect("retry wait should wake on interrupt")
        .expect("waiter should complete");
    assert!(cancelled);

    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn clears_stale_interrupt_for_new_request_but_keeps_active_cancel() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    crate::ai::driver::signal::signal_request_interrupt();
    assert!(crate::ai::driver::signal::request_interrupt_ready());
    clear_stale_request_interrupt_before_request(&app);
    assert!(!crate::ai::driver::signal::request_interrupt_ready());

    app.cancel_stream
        .store(true, std::sync::atomic::Ordering::Relaxed);
    crate::ai::driver::signal::signal_request_interrupt();
    clear_stale_request_interrupt_before_request(&app);
    assert!(crate::ai::driver::signal::request_interrupt_ready());

    app.cancel_stream
        .store(false, std::sync::atomic::Ordering::Relaxed);
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn response_body_read_observes_request_interrupt_source() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind local test server");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut socket, _) = listener.accept().await.expect("accept connection");
        let mut request = [0u8; 1024];
        let _ = socket.read(&mut request).await;
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
            .await
            .expect("write headers");
        tokio::time::sleep(Duration::from_secs(5)).await;
    });

    let response = app
        .client
        .get(format!("http://{addr}/"))
        .send()
        .await
        .expect("headers should arrive");
    let read = super::super::transport::response_text_with_cancel(&app, response, "body cancel");
    tokio::pin!(read);

    tokio::select! {
        result = read.as_mut() => panic!("body read completed before interrupt: {result:?}"),
        _ = tokio::time::sleep(Duration::from_millis(20)) => {}
    }
    crate::ai::driver::signal::signal_request_interrupt();

    let err = tokio::time::timeout(Duration::from_millis(200), read.as_mut())
        .await
        .expect("body read should wake on interrupt")
        .expect_err("body read should return cancellation");
    assert!(err.to_string().contains("body cancel"), "{err}");

    server.abort();
    let _ = server.await;
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}
