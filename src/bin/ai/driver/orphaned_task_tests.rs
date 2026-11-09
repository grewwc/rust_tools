//! Regression scenarios for subagent lifetime after a foreground request fails.

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::io::AsyncReadExt;

use super::*;
use crate::ai::tools::task_tools::{self, BackendId, InheritOptions, PreparedSubagentTask};

struct TraeCliFixture {
    app: App,
    root: PathBuf,
    listener: tokio::net::UnixListener,
    original_binary: Option<std::ffi::OsString>,
}

impl TraeCliFixture {
    fn new() -> Self {
        let root = PathBuf::from("/tmp").join(format!(
            "orphaned-traecli-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let script = root.join("traecli.py");
        let listener = tokio::net::UnixListener::bind(root.join("traecli.py.control")).unwrap();
        std::fs::write(
            &script,
            r#"#!/usr/bin/env python3
import json
import socket
import sys

sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": "11111111-1111-4111-8111-111111111111"}), flush=True)
with socket.socket(socket.AF_UNIX) as control:
    control.connect(__file__ + ".control")
    control.sendall(b"started\n")
    control.recv(1)
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "orphaned CLI result is intact"}}), flush=True)
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}), flush=True)
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let original_binary = std::env::var_os("TRAECLI_BIN");
        unsafe { std::env::set_var("TRAECLI_BIN", script) };
        let mut app = crate::ai::middleware::test_util::test_app();
        app.current_agent = "build".to_string();
        app.current_model.clear();
        app.session_id = format!("orphaned-{}", uuid::Uuid::new_v4().simple());
        app.config.history_file = root.join("history.sqlite");
        app.config.base_history_file = app.config.history_file.clone();
        app.session_history_file = root.join("foreground.sqlite");
        app.session_default_skills = vec!["agent-team".to_string()];
        Self {
            app,
            root,
            listener,
            original_binary,
        }
    }

    fn context(&self) -> Arc<runtime_ctx::DriverContext> {
        let context = runtime_ctx::DriverContext::from_app_snapshot(
            &self.app,
            Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(vec![
                serde_json::from_value(serde_json::json!({
                    "name": "build", "description": "Fixture worker", "mode": "all"
                }))
                .unwrap(),
            ]),
        );
        *context.active_skill_names.write().unwrap() = self.app.session_default_skills.clone();
        context
    }

    fn prepared(&self) -> PreparedSubagentTask {
        PreparedSubagentTask {
            description: "foreground failure fixture".to_string(),
            prompt: "Return the fixture result".to_string(),
            response_schema: None,
            agent_name: "build".to_string(),
            model: String::new(),
            reasoning_effort: None,
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "fixture backend".to_string(),
            inherit: InheritOptions::from_value(&serde_json::json!("none")).unwrap(),
            backend: BackendId::new("traecli"),
            show_backend_launch_tag: true,
            backend_model: Some("fixture-model".to_string()),
            backend_continue_from: None,
            image_files: Vec::new(),
        }
    }

    fn scope_owner<R>(&self, owner: u64, action: impl FnOnce() -> R) -> R {
        self.app.os.lock().unwrap().set_current_pid(Some(owner));
        runtime_ctx::DRIVER_CTX.sync_scope(self.context(), || {
            runtime_ctx::TURN_IDENTITY.sync_scope((self.app.session_id.clone(), 1), || {
                TASK_PID.sync_scope(Some(owner), action)
            })
        })
    }

    fn start(&mut self) -> (u64, task_tools::SpawnedSubagentTask) {
        let owner = self.app.os.lock().unwrap().begin_foreground(
            "foreground".to_string(),
            "fixture".to_string(),
            10,
            usize::MAX,
            None,
        );
        let spawned = self.scope_owner(owner, || {
            task_tools::spawn_subagent_kernel_task(&self.prepared()).unwrap()
        });
        let context = self.context();
        let mut skills = context.skill_manifests.clone();
        let mut agents = context.agent_manifests.clone();
        let mut loaded = true;
        background_dispatch::dispatch_background_batch(
            &mut self.app,
            &context.mcp_client,
            &mut skills,
            &mut agents,
            &mut loaded,
            1,
        );
        (owner, spawned)
    }

    async fn started(&self) -> tokio::net::UnixStream {
        let (mut control, _) =
            tokio::time::timeout(Duration::from_secs(10), self.listener.accept())
                .await
                .expect("fake CLI did not start")
                .unwrap();
        let mut started = [0; 8];
        tokio::time::timeout(Duration::from_secs(10), control.read_exact(&mut started))
            .await
            .expect("fake CLI startup handshake timed out")
            .unwrap();
        assert_eq!(&started, b"started\n");
        control
    }

    fn fail_foreground(&self, owner: u64, child: u64) {
        let result = "Failed: injected foreground request error".to_string();
        let destroyed = task_tools::destroy_pending_tasks_of_dead_foreground(
            &self.app.os,
            &self.app.session_id,
            owner,
            &result,
        );
        assert_eq!(destroyed, 1, "the running subagent must be destroyed");
        let mut os = self.app.os.lock().unwrap();
        terminate_and_cleanup(os.as_mut(), owner, result, true);
        assert!(os.get_process(owner).is_none());
        assert!(os.get_process(child).is_none());
    }

    fn replacement(&self) -> u64 {
        self.app.os.lock().unwrap().begin_foreground(
            "foreground".to_string(),
            "next turn".to_string(),
            10,
            usize::MAX,
            None,
        )
    }

    async fn wait_until(&self, ready: impl Fn() -> bool) {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let notified = SCHEDULER_NOTIFY.notified();
                if ready() {
                    return;
                }
                notified.await;
            }
        })
        .await
        .expect("fixture event did not arrive");
    }
}

impl Drop for TraeCliFixture {
    fn drop(&mut self) {
        runtime_ctx::DRIVER_CTX.sync_scope(self.context(), || {
            task_tools::discard_tasks_for_session(&self.app.session_id);
        });
        unsafe {
            match &self.original_binary {
                Some(binary) => std::env::set_var("TRAECLI_BIN", binary),
                None => std::env::remove_var("TRAECLI_BIN"),
            }
        }
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_failure_destroys_running_subagent_and_its_resources() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut fixture = TraeCliFixture::new();
    let (owner, spawned) = fixture.start();
    let _control = fixture.started().await;
    fixture.fail_foreground(owner, spawned.pid);

    assert!(
        task_tools::with_task_entry(&spawned.task_id, |_| ()).is_none(),
        "the destroyed task must not stay in the registry"
    );
    let os = fixture.app.os.lock().unwrap();
    assert!(os.get_process(spawned.pid).is_none());
    assert!(
        os.channel_meta(aios_kernel::primitives::ChannelId(
            spawned.result_channel_id
        ))
        .is_none()
    );
    assert!(os.futex_load(spawned.completion_futex_addr).is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_failure_prevents_adoption_and_wait() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut fixture = TraeCliFixture::new();
    let (owner, spawned) = fixture.start();
    let _control = fixture.started().await;
    fixture.fail_foreground(owner, spawned.pid);
    let replacement = fixture.replacement();

    let adopted = fixture
        .scope_owner(replacement, || {
            task_tools::recover_current_foreground_tasks(&fixture.app.session_id)
        })
        .unwrap();
    assert!(adopted.is_empty(), "{adopted:?}");

    let wait = fixture.scope_owner(replacement, || {
        task_tools::execute_task_wait(&serde_json::json!({
            "task_ids": [&spawned.task_id], "timeout_secs": 1,
        }))
    });
    let error = wait.expect_err("a destroyed task must not be collectable");
    assert!(error.contains("Unknown task_id"), "{error}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_failure_kills_external_backend_process() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut fixture = TraeCliFixture::new();
    let (owner, spawned) = fixture.start();
    let mut control = fixture.started().await;
    fixture.fail_foreground(owner, spawned.pid);

    let mut bytes = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), control.read_to_end(&mut bytes))
        .await
        .expect("foreground death did not terminate the external CLI")
        .unwrap();
    fixture
        .wait_until(|| BG_SUBAGENT_INFLIGHT.load(Ordering::Acquire) == 0)
        .await;
    fixture
        .wait_until(|| {
            !task_tools::backend_session::task_backend_run_is_active(
                BackendId::TRAECLI,
                &spawned.task_id,
            )
        })
        .await;
}
