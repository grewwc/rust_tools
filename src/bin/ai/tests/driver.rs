//! Driver behavior tests: model resolution, OCR fallback, and interrupt-futex signaling.

use std::sync::{Arc, atomic::AtomicBool};

use super::super::*;
use super::{any_model_name, any_vl_model_name, test_app_with_cancel_stream};

#[test]
fn resolve_model_is_unicode_safe() {
    use std::path::PathBuf;

    let cli = cli::ParsedCli::default();
    let config = types::AppConfig {
        api_key: String::new(),
        base_history_file: PathBuf::new(),
        history_file: PathBuf::new(),
        endpoint: String::new(),
        vl_default_model: any_vl_model_name(),
        history_max_chars: 12000,
        history_keep_last: 8,
        history_summary_max_chars: 4000,
        intent_model: None,
    };
    let client = reqwest::Client::builder().build().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let streaming = Arc::new(AtomicBool::new(false));
    let cancel_stream = Arc::new(AtomicBool::new(false));
    let app = types::App {
        cli,
        hooks: Default::default(),
        config,
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        active_persona: persona::default_persona(),
        client,
        current_model: any_model_name(),
        current_agent: "build".to_string(),
        current_agent_manifest: None,
        pending_files: None,
        forced_skills: Vec::new(),
        forced_skill_source: None,
        pending_skill_continuation: None,
        forced_question: None,
        attached_image_files: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        ignore_next_prompt_interrupt: false,
        prompt_editor: None,
        agent_context: None,
        last_skill_bias: None,
        os: crate::ai::driver::new_local_kernel(),
        agent_reload_counter: None,
        observers: vec![Box::new(
            crate::ai::driver::thinking::ThinkingOrchestrator::new(),
        )],
        last_known_prompt_tokens: None,
        last_known_cached_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
        turn_reasoning_items: Default::default(),
        stale_patch_targets: Default::default(),
        tool_middlewares: Vec::new(),
        llm_middlewares: Vec::new(),
    };

    let mut question = "a 什么是rust的一个crate？".to_string();
    let model = driver::resolve_model_for_input(&app, false, &mut question);
    assert_eq!(model, app.current_model);
    assert_eq!(question, "a 什么是rust的一个crate？");
}

#[test]
fn image_files_keep_text_model() {
    let model = driver::attachment_forced_model("qwen3.5-flash", true, "any", false);
    assert_eq!(model, None);
}

#[test]
fn take_stream_cancelled_clears_request_interrupt_futex() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let cancel_stream = Arc::new(AtomicBool::new(true));
    let app = test_app_with_cancel_stream(cancel_stream.clone());
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    crate::ai::driver::signal::signal_request_interrupt();

    let futex = crate::ai::driver::signal::request_interrupt_futex().unwrap();
    {
        let os = app.os.lock().unwrap();
        assert_eq!(os.futex_load(futex), Some(1));
    }

    assert!(types::take_stream_cancelled(&app));
    {
        let os = app.os.lock().unwrap();
        assert_eq!(os.futex_load(futex), Some(0));
    }
}

#[test]
fn request_shutdown_sets_request_interrupt_futex() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    crate::ai::driver::signal::request_shutdown(app.shutdown.as_ref());

    assert!(app.shutdown.load(std::sync::atomic::Ordering::Relaxed));
    let futex = crate::ai::driver::signal::request_interrupt_futex().unwrap();
    let os = app.os.lock().unwrap();
    assert_eq!(os.futex_load(futex), Some(1));
    drop(os);
    crate::ai::driver::signal::clear_request_interrupt();
}

#[tokio::test]
async fn wait_for_interrupt_sources_returns_after_shutdown_request() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    let shutdown = app.shutdown.clone();
    let waiter = tokio::spawn(async move {
        crate::ai::driver::signal::wait_for_interrupt_sources(None, None, None).await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    crate::ai::driver::signal::request_shutdown(shutdown.as_ref());

    tokio::time::timeout(std::time::Duration::from_millis(200), waiter)
        .await
        .expect("shutdown should wake interrupt waiter")
        .expect("waiter should complete cleanly");
    crate::ai::driver::signal::clear_request_interrupt();
}

#[tokio::test]
async fn wait_for_interrupt_sources_returns_after_daemon_cancel() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let local_interrupt =
        crate::ai::driver::signal::alloc_interrupt_futex("background_cancel_test")
            .expect("local interrupt futex");
    let (handle, cancel_token) = {
        let mut os = app.os.lock().unwrap();
        os.daemon_register(
            "background_cancel_test".to_string(),
            aios_kernel::primitives::DaemonKind::Reflection,
            None,
        )
    };

    let waiter = tokio::spawn(async move {
        crate::ai::driver::signal::wait_for_interrupt_sources(
            Some(cancel_token),
            Some(local_interrupt),
            None,
        )
        .await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    {
        let mut os = app.os.lock().unwrap();
        assert!(os.cancel_daemon(handle));
    }

    tokio::time::timeout(std::time::Duration::from_millis(200), waiter)
        .await
        .expect("daemon cancel should wake interrupt waiter")
        .expect("waiter should complete cleanly");

    {
        let os = app.os.lock().unwrap();
        assert_eq!(os.futex_load(local_interrupt), Some(1));
    }
    crate::ai::driver::signal::destroy_interrupt_futex(local_interrupt);
}

#[test]
fn successful_ocr_keeps_text_model_for_images() {
    let vl = any_vl_model_name();
    let model = driver::attachment_forced_model("qwen3.5-flash", true, vl.as_str(), true);
    assert_eq!(model, None);
}

#[test]
fn partial_ocr_success_still_counts_as_usable_for_text_models() {
    let ocr = driver::model::OcrExtraction {
        tool_name: "mcp_ocr_ocr_image".to_string(),
        content: "ok".to_string(),
        images: vec![
            driver::model::OcrImageSummary {
                file_name: "ok.png".to_string(),
                extracted_chars: 2,
                error: None,
            },
            driver::model::OcrImageSummary {
                file_name: "bad.png".to_string(),
                extracted_chars: 0,
                error: Some("failed".to_string()),
            },
        ],
    };
    assert!(ocr.has_usable_text());
}

#[test]
fn all_failed_ocr_does_not_keep_text_model() {
    let ocr = driver::model::OcrExtraction {
        tool_name: "mcp_ocr_ocr_image".to_string(),
        content: String::new(),
        images: vec![driver::model::OcrImageSummary {
            file_name: "bad.png".to_string(),
            extracted_chars: 0,
            error: Some("failed".to_string()),
        }],
    };
    assert!(!ocr.has_usable_text());
}
