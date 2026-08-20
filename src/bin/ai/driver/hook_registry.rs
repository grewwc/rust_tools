// =============================================================================
// Turn hook firing — `App` 上 turn 生命周期钩子的触发点
// =============================================================================
// driver/hooks.rs 现有的是 shell 级钩子（pre_turn / post_turn 脚本）；本模块
// 负责在 turn 起点/终点触发 `App.hooks` 中进程内注册的钩子。注册表本身是
// `pipeline::HookRegistry`（StageKind → callbacks），语义化注册（on_turn_start
// 等）由调用方按 StageKind 映射直接完成——driver 不提供独立门面类型，避免
// 核心层 `types::App` 反向依赖 driver。

use crate::ai::pipeline::hook::HookRegistry;
use crate::ai::pipeline::{PipelineContext, StageKind};
use crate::ai::types::App;
use crate::ai::history::Message;

impl App {
    /// 触发钩子的统一入口：临时取出注册表，避免 `&mut App`（被 PipelineContext 持有）
    /// 与 `fire_* (&self)` 的借用冲突；无论成败都归还，保证已注册钩子不丢失。
    /// `messages` 提供时借入 ctx 并在触发后写回调用方（before_request 用，避免
    /// mem::take 导致请求上下文丢失）；`f` 负责具体 fire_* 调用与 stage 推进。
    /// 钩子失败只记录不中断 turn；空注册表 = 零行为变化。
    fn fire_hooks<F>(
        &mut self,
        label: &'static str,
        turn_index: usize,
        mut messages: Option<&mut Vec<Message>>,
        f: F,
    ) where
        F: FnOnce(&HookRegistry, &mut PipelineContext<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        let hooks = std::mem::take(&mut self.hooks);
        let owned = messages.as_deref_mut().map(|m| std::mem::take(m)).unwrap_or_default();
        let mut ctx = PipelineContext::new(self, owned, turn_index);
        let result = f(&hooks, &mut ctx);
        if let Some(m) = messages.as_deref_mut() {
            *m = std::mem::take(&mut ctx.messages);
        }
        self.hooks = hooks;
        if let Err(err) = result {
            eprintln!("[hook] {label} failed: {err}");
        }
    }

    /// 触发 turn 起点钩子（on_turn_start，映射 Prepare.before）。
    /// turn_index 为 run_turn 预先分配的真实轮次号，避免钩子读到伪造的 0。
    pub(crate) fn fire_turn_start_hooks(&mut self, turn_index: usize) {
        self.fire_hooks("on_turn_start", turn_index, None, |hooks, ctx| {
            hooks.fire_before(ctx, StageKind::Prepare)
        });
    }

    /// 触发 turn 终点钩子（on_turn_end，映射 Finalize.after）。
    pub(crate) fn fire_turn_end_hooks(&mut self, turn_index: usize) {
        self.fire_hooks("on_turn_end", turn_index, None, |hooks, ctx| {
            hooks.fire_after(ctx, StageKind::Finalize)
        });
    }

    /// 触发请求构建前钩子（on_before_request，映射 BuildRequest.before）。
    /// 使用 stage-only 触发，避免重复触发 turn 级全局钩子；turn_index 取当前
    /// TURN_IDENTITY 的真实轮次号（未在 turn 内时回退 0）。messages 是正在构建的
    /// 真实请求消息：钩子可检查/改写，无论钩子成败都写回调用方。
    pub(crate) fn fire_before_request_hooks(&mut self, messages: &mut Vec<Message>) {
        let turn_index = crate::ai::driver::runtime_ctx::current_turn_id_or_zero();
        self.fire_hooks("before_request", turn_index, Some(messages), |hooks, ctx| {
            ctx.advance(StageKind::BuildRequest);
            hooks.fire_stage_before(ctx, StageKind::BuildRequest)
        });
    }

    /// 触发流解析完成后钩子（on_after_stream，映射 ParseStream.after）。
    pub(crate) fn fire_after_stream_hooks(&mut self) {
        let turn_index = crate::ai::driver::runtime_ctx::current_turn_id_or_zero();
        self.fire_hooks("after_stream", turn_index, None, |hooks, ctx| {
            ctx.advance(StageKind::ParseStream);
            hooks.fire_stage_after(ctx, StageKind::ParseStream)
        });
    }

    /// 触发工具执行前钩子（on_before_tools，映射 ExecuteTools.before）。
    pub(crate) fn fire_before_tools_hooks(&mut self) {
        let turn_index = crate::ai::driver::runtime_ctx::current_turn_id_or_zero();
        self.fire_hooks("before_tools", turn_index, None, |hooks, ctx| {
            ctx.advance(StageKind::ExecuteTools);
            hooks.fire_stage_before(ctx, StageKind::ExecuteTools)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn turn_hooks_fire_on_app_start_and_end() {
        let mut app = crate::ai::middleware::test_util::test_app();
        let fired = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let f = std::sync::Arc::clone(&fired);
        app.hooks.register_before(StageKind::Prepare, "start", move |ctx| {
            ctx.tags.push("start".into());
            f.lock().unwrap().push("start".to_string());
            Ok(())
        });
        let f = std::sync::Arc::clone(&fired);
        let stages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let s = std::sync::Arc::clone(&stages);
        app.hooks.register_after(StageKind::Finalize, "end", move |ctx| {
            s.lock().unwrap().push(ctx.stage);
            f.lock().unwrap().push("end".to_string());
            Ok(())
        });

        app.fire_turn_start_hooks(0);
        app.fire_turn_end_hooks(0);

        assert_eq!(*fired.lock().unwrap(), vec!["start".to_string(), "end".to_string()]);
        // end 钩子读到的是 Finalize 阶段，而非 PipelineContext::new 的 Prepare 默认值。
        assert_eq!(*stages.lock().unwrap(), vec![StageKind::Finalize]);
    }

    #[test]
    fn turn_hook_failure_is_logged_not_fatal() {
        let mut app = crate::ai::middleware::test_util::test_app();
        app.hooks.register_before(StageKind::Prepare, "fail", |_ctx| Err("boom".into()));
        // 钩子失败只记录、不中断 turn，也不丢失注册表。
        app.fire_turn_start_hooks(0);
        app.fire_turn_end_hooks(0);
        assert_eq!(app.hooks.len(), 1);
    }

    #[test]
    fn intermediate_hooks_fire_only_stage_hooks() {
        let mut app = crate::ai::middleware::test_util::test_app();
        let fired = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let f = std::sync::Arc::clone(&fired);
        app.hooks.register_before(StageKind::BuildRequest, "before_request", move |_ctx| {
            f.lock().unwrap().push("before_request".to_string());
            Ok(())
        });
        let f = std::sync::Arc::clone(&fired);
        app.hooks.register_after(StageKind::ParseStream, "after_stream", move |_ctx| {
            f.lock().unwrap().push("after_stream".to_string());
            Ok(())
        });
        let f = std::sync::Arc::clone(&fired);
        app.hooks.register_before(StageKind::ExecuteTools, "before_tools", move |_ctx| {
            f.lock().unwrap().push("before_tools".to_string());
            Ok(())
        });
        // 全局钩子不应被 stage-only 触发重复调用
        let f = std::sync::Arc::clone(&fired);
        app.hooks.register_global_before("gb", move |_ctx| {
            f.lock().unwrap().push("gb".to_string());
            Ok(())
        });
        let f = std::sync::Arc::clone(&fired);
        app.hooks.register_global_after("ga", move |_ctx| {
            f.lock().unwrap().push("ga".to_string());
            Ok(())
        });

        app.fire_before_request_hooks(&mut Vec::new());
        app.fire_after_stream_hooks();
        app.fire_before_tools_hooks();

        // 仅三个 stage 钩子触发，全局钩子未触发
        assert_eq!(
            *fired.lock().unwrap(),
            vec!["before_request".to_string(), "after_stream".to_string(), "before_tools".to_string()]
        );
    }

    #[test]
    fn intermediate_hook_failure_is_logged_not_fatal() {
        let mut app = crate::ai::middleware::test_util::test_app();
        app.hooks.register_before(StageKind::BuildRequest, "fail", |_ctx| Err("boom".into()));
        app.fire_before_request_hooks(&mut Vec::new());
        app.fire_after_stream_hooks();
        app.fire_before_tools_hooks();
        // 失败只记录，注册表不丢失
        assert_eq!(app.hooks.len(), 1);
    }

    #[test]
    fn before_request_hook_sees_and_modifies_real_messages() {
        let mut app = crate::ai::middleware::test_util::test_app();
        app.hooks.register_before(StageKind::BuildRequest, "req", |ctx| {
            // 钩子看到的是真实请求消息，而非空 Vec
            assert_eq!(ctx.messages.len(), 1);
            assert_eq!(ctx.messages[0].role, "user");
            // 钩子可改写请求消息
            ctx.messages.push(Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String("hook-injected".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
            Ok(())
        });
        let mut messages = vec![Message {
            role: "user".to_string(),
            content: serde_json::Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        app.fire_before_request_hooks(&mut messages);
        // 钩子的改写写回调用方，且消息不丢失
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[1].content,
            serde_json::Value::String("hook-injected".to_string())
        );
    }

    #[test]
    fn before_request_hook_failure_still_restores_messages() {
        let mut app = crate::ai::middleware::test_util::test_app();
        app.hooks.register_before(StageKind::BuildRequest, "fail", |ctx| {
            ctx.messages.push(Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String("partial".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
            Err("boom".into())
        });
        let mut messages = vec![Message {
            role: "user".to_string(),
            content: serde_json::Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        app.fire_before_request_hooks(&mut messages);
        // 失败只记录；消息仍写回，请求上下文不因 mem::take 丢失
        assert_eq!(messages.len(), 2);
        assert_eq!(app.hooks.len(), 1);
    }
}
